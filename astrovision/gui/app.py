"""The desktop application: a local web app served by the package itself.

`astrovision gui` starts an HTTP server on localhost and opens the browser
on a page that does what the command line does -- analyse an image or a
series, simulate a field, read alerts, hand candidates to the vetting page
-- with a file browser, a progress bar per pipeline stage, the report, the
catalog and cutouts. Nothing here needs a GUI toolkit: the server is the
standard library's, the page is one HTML file, and the browser is whatever
the machine has. That is what lets it run on any PC where Python runs, and
what lets PyInstaller wrap it into a single folder for PCs where Python
does not.

The server binds to 127.0.0.1 and nothing else. It can read any file the
user can, which is the point of a desktop application and the reason it
must never be exposed on a network.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from ..core.backend import capabilities, describe_capabilities
from ..core.config import PRESETS, AstroVisionConfig
from ..core.exceptions import PipelineCancelled
from ..core.logging import get_logger

log = get_logger("gui")

#: Files the browser lists; everything else is hidden to keep the list short.
IMAGE_SUFFIXES = (".fits", ".fit", ".fts", ".fits.fz", ".fits.gz", ".npy", ".npz",
                  ".png", ".jpg", ".jpeg", ".tif", ".tiff")
ALERT_SUFFIXES = (".avro",)
OTHER_SUFFIXES = (".csv", ".json", ".sqlite", ".db", ".yaml", ".yml", ".txt", ".html")

#: The stages the single-field pipeline runs, in order, for the progress list.
STAGES = ["preprocess", "detect", "photometry", "calibration", "multiband",
          "segmentation", "morphology", "classification", "photoz", "crossmatch",
          "embeddings", "anomaly", "lensing", "transient", "moving", "timeseries",
          "clustering", "statistics", "assistant"]

BOUNDARY = ("This software analyses images and ranks candidates. It never declares a "
            "discovery: a human astronomer and observational validation decide.")


# -- jobs ---------------------------------------------------------------------
@dataclass
class Job:
    """One run of something the page asked for, and what came of it."""

    id: str
    kind: str                                   # analyze | series | simulate
    title: str
    params: Dict[str, Any]
    status: str = "queued"                      # queued | running | done | failed
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    log_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=400))
    error: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    analysis: Any = None
    image: Any = None
    warnings: List[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def stage_update(self, result) -> None:
        entry = {"name": result.name, "status": result.status,
                 "seconds": float(getattr(result, "seconds", 0.0) or 0.0),
                 "message": str(getattr(result, "message", "") or "")}
        for existing in self.stages:
            if existing["name"] == result.name:
                existing.update(entry)
                break
        else:
            self.stages.append(entry)

    def to_dict(self, with_result: bool = True) -> Dict[str, Any]:
        out = {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
               "params": self.params, "created": self.created, "started": self.started,
               "finished": self.finished, "stages": list(self.stages),
               "error": self.error, "warnings": list(self.warnings),
               "seconds": (None if self.started is None else
                           (self.finished or time.time()) - self.started)}
        if with_result:
            out["result"] = self.result
            out["log"] = list(self.log_lines)[-60:]
        return out


class _JobLogHandler(logging.Handler):
    """Routes the package's log lines into whichever job is running on this thread."""

    def __init__(self, app: "App") -> None:
        super().__init__(level=logging.INFO)
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        job = self.app.job_for_thread(threading.get_ident())
        if job is None:
            return
        try:
            line = f"{time.strftime('%H:%M:%S')} {record.levelname[:4]} {record.getMessage()}"
        except Exception:                                   # pragma: no cover
            return
        job.log_lines.append(line)


# -- the application ----------------------------------------------------------
class App:
    """State shared by every request: jobs, their threads, the last results."""

    def __init__(self, workdir: Optional[str] = None) -> None:
        self.workdir = os.path.abspath(workdir or os.getcwd())
        self.jobs: Dict[str, Job] = {}
        self._threads: Dict[int, str] = {}
        self._lock = threading.Lock()
        self.vetting_servers: List[Any] = []
        handler = _JobLogHandler(self)
        logging.getLogger("astrovision").addHandler(handler)
        # Probing the optional backends imports them (PyTorch alone is
        # seconds), so it happens once, off the request thread; the status
        # call reports what is known so far.
        self._backends: Optional[Dict[str, bool]] = None
        self._backends_text = ""
        threading.Thread(target=self._probe_backends, daemon=True).start()

    def _probe_backends(self) -> None:
        try:
            self._backends = {k: bool(v) for k, v in capabilities().items()}
            self._backends_text = describe_capabilities()
        except Exception as exc:                                # pragma: no cover
            log.debug("backend probe failed: %s", exc)
            self._backends = {}

    # -- job plumbing --------------------------------------------------------
    def job_for_thread(self, ident: int) -> Optional[Job]:
        job_id = self._threads.get(ident)
        return self.jobs.get(job_id) if job_id else None

    def submit(self, kind: str, title: str, params: Dict[str, Any], work) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title, params=params)
        with self._lock:
            self.jobs[job.id] = job

        def run() -> None:
            self._threads[threading.get_ident()] = job.id
            job.status, job.started = "running", time.time()
            try:
                job.result = work(job) or {}
                job.status = "done"
                for stage in job.stages:                    # stages this run never reached
                    if stage["status"] in ("pending", "running"):
                        stage["status"], stage["message"] = "skipped", "not part of this run"
            except PipelineCancelled as exc:
                job.status = "cancelled"
                job.error = str(exc)
                for stage in job.stages:
                    if stage["status"] in ("pending", "running"):
                        stage["status"], stage["message"] = "cancelled", "not run"
            except Exception as exc:                        # noqa: BLE001 - shown on the page
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log_lines.append(traceback.format_exc().strip().splitlines()[-1])
                log.exception("job %s failed", job.id)
            finally:
                job.finished = time.time()
                self._threads.pop(threading.get_ident(), None)

        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    def cancel(self, job_id: str) -> Dict[str, Any]:
        """Ask a running job to stop after its current stage."""
        job = self.job(job_id)
        if job.status in ("queued", "running"):
            job.cancel_event.set()
            job.log_lines.append(f"{time.strftime('%H:%M:%S')} INFO cancel requested; the "
                                 "current stage finishes first")
        return {"id": job.id, "status": job.status, "cancel_requested": job.cancel_event.is_set()}

    def job_file(self, job_id: str, name: str) -> Tuple[str, bytes]:
        """One of the files a job wrote, by the key the result lists it under."""
        job = self.job(job_id)
        files = dict(job.result.get("files") or {})
        if job.result.get("truth"):
            files["truth"] = job.result["truth"]
        for index, path in enumerate(job.result.get("paths") or []):
            files[f"path{index}"] = path
        if job.result.get("alerts"):
            files["alerts"] = job.result["alerts"]["path"]
        path = files.get(name)
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(f"this job wrote no file called {name!r}")
        with open(path, "rb") as handle:
            return os.path.basename(path), handle.read()

    # -- status ----------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {"version": __version__, "python": sys.version.split()[0],
                "platform": sys.platform, "numpy": np.__version__,
                "backends": self._backends or {}, "backends_ready": self._backends is not None,
                "backends_text": self._backends_text, "presets": list(PRESETS),
                "workdir": self.workdir, "home": os.path.expanduser("~"),
                "boundary": BOUNDARY,
                "jobs": [j.to_dict(with_result=False) for j in
                         sorted(self.jobs.values(), key=lambda j: -j.created)]}

    # -- files -----------------------------------------------------------------
    def browse(self, path: Optional[str]) -> Dict[str, Any]:
        directory = os.path.abspath(os.path.expanduser(path or self.workdir))
        if os.path.isfile(directory):
            directory = os.path.dirname(directory)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"not a directory: {directory}")
        dirs, files = [], []
        try:
            names = sorted(os.listdir(directory), key=str.lower)
        except PermissionError as exc:
            raise PermissionError(f"cannot read {directory}") from exc
        for name in names:
            if name.startswith("."):
                continue
            full = os.path.join(directory, name)
            if os.path.isdir(full):
                dirs.append({"name": name, "path": full})
            else:
                lowered = name.lower()
                kind = ("image" if lowered.endswith(IMAGE_SUFFIXES)
                        else "alerts" if lowered.endswith(ALERT_SUFFIXES)
                        else "other" if lowered.endswith(OTHER_SUFFIXES) else None)
                if kind is None:
                    continue
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                files.append({"name": name, "path": full, "kind": kind, "size": size})
        parent = os.path.dirname(directory) if os.path.dirname(directory) != directory else None
        return {"path": directory, "parent": parent, "dirs": dirs, "files": files}

    def inspect(self, path: str) -> Dict[str, Any]:
        from ..io.image import AstroImage

        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such file: {path}")
        image = AstroImage.load(path)
        stats = image.stats()
        wcs = image.wcs
        out = {"path": path, "name": image.name, "shape": list(image.shape),
               "band": image.band, "mjd": image.mjd, "exposure_time": image.exposure_time,
               "stats": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                         for k, v in stats.items()},
               "wcs": None if wcs is None else wcs.to_dict()}
        if wcs is not None:
            ra, dec = wcs.pixel_to_world(image.shape[1] / 2.0, image.shape[0] / 2.0)
            out["centre"] = [float(ra), float(dec)]
        header = {k: v for k, v in list(image.header.items())[:80]
                  if isinstance(v, (int, float, str, bool))}
        out["header"] = header
        return out

    def preview_png(self, path: str, max_size: int = 900) -> bytes:
        from ..io.image import AstroImage
        from ..vetting.png import encode_png, stretch

        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such file: {path}")
        image = AstroImage.load(path)
        return _preview(image.data, max_size, encode_png, stretch)

    # -- the work ----------------------------------------------------------------
    def _config(self, params: Dict[str, Any]) -> AstroVisionConfig:
        config = AstroVisionConfig()
        preset = params.get("preset")
        if preset:
            config.with_preset(str(preset))
        if params.get("threshold") not in (None, ""):
            config.detection.threshold_sigma = float(params["threshold"])
        formats = params.get("formats") or ["html", "text", "json"]
        config.report.formats = [str(f) for f in formats]
        # The desktop default is every core but one; the library's is one.
        config.n_workers = int(params.get("workers", 0) or 0)
        output = params.get("output_dir") or os.path.join(self.workdir, "astrovision_output")
        config.report.output_dir = os.path.abspath(os.path.expanduser(str(output)))
        return config

    def analyze(self, params: Dict[str, Any]) -> Job:
        path = os.path.abspath(os.path.expanduser(str(params.get("path", ""))))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such file: {path}")

        def work(job: Job) -> Dict[str, Any]:
            from ..engine import Pipeline
            from ..io.image import AstroImage
            from ..preprocess import Preprocessor
            from ..report import generate_reports

            config = self._config(params)
            job.stages = [{"name": s, "status": "pending", "seconds": 0.0, "message": ""}
                          for s in STAGES]
            image = AstroImage.load(path)
            # Preprocess outside the pipeline so the cleaned image with its
            # background model is the one the cutouts are cut from.
            job.stage_update(_Stage("preprocess", "running"))
            clean = Preprocessor(config.preprocess).run(image)
            job.image = clean
            if job.cancel_event.is_set():
                raise PipelineCancelled("run stopped after preprocessing")
            pipeline = Pipeline(config, progress=job.stage_update, cancel=job.cancel_event.is_set)
            redshift = params.get("redshift")
            analysis = pipeline.run(clean, redshift=None if redshift in (None, "")
                                    else float(redshift), preprocess=False)
            job.stage_update(_Stage("preprocess", "ok", pipeline.preprocessor.report))
            job.analysis = analysis
            job.warnings = list(analysis.warnings)
            output_dir = os.path.join(config.report.output_dir,
                                      os.path.splitext(os.path.basename(path))[0])
            written = generate_reports(analysis, output_dir, config.report.formats,
                                       title=config.report.title,
                                       top_candidates=config.report.top_candidates,
                                       image=clean)
            result = {"summary": analysis.summary(), "files": written,
                      "output_dir": output_dir, "n_sources": len(analysis.catalog)}
            if params.get("db"):
                from ..catalog import CatalogDB, ingest_analysis
                db = CatalogDB(os.path.abspath(os.path.expanduser(str(params["db"]))))
                stored = ingest_analysis(db, analysis, clean)
                result["database"] = {"path": db.path, "field_id": stored.field_id,
                                      "n_detections": stored.n_detections,
                                      "n_matched": stored.n_matched,
                                      "n_new_objects": stored.n_new_objects}
            return result

        return self.submit("analyze", os.path.basename(path), dict(params, path=path), work)

    def series(self, params: Dict[str, Any]) -> Job:
        paths = [os.path.abspath(os.path.expanduser(str(p))) for p in params.get("paths") or []]
        missing = [p for p in paths if not os.path.isfile(p)]
        if len(paths) < 2:
            raise ValueError("a series needs at least two epochs")
        if missing:
            raise FileNotFoundError(f"no such file: {missing[0]}")

        def work(job: Job) -> Dict[str, Any]:
            from ..engine import Pipeline
            from ..io.image import ImageSeries
            from ..report import generate_reports

            config = self._config(params)
            job.stages = [{"name": s, "status": "pending", "seconds": 0.0, "message": ""}
                          for s in STAGES]
            series = ImageSeries.from_paths(paths, name=str(params.get("name") or "series"))
            pipeline = Pipeline(config, progress=job.stage_update, cancel=job.cancel_event.is_set)
            redshift = params.get("redshift")
            analysis = pipeline.run_series(series, redshift=None if redshift in (None, "")
                                           else float(redshift))
            job.analysis = analysis
            job.image = series.reference if hasattr(series, "reference") else series[0]
            job.warnings = list(analysis.warnings)
            output_dir = os.path.join(config.report.output_dir, str(params.get("name") or "series"))
            written = generate_reports(analysis, output_dir, config.report.formats,
                                       title=config.report.title,
                                       top_candidates=config.report.top_candidates,
                                       image=job.image)
            result = {"summary": analysis.summary(), "files": written, "output_dir": output_dir,
                      "n_sources": len(analysis.catalog),
                      "transients": [t.to_dict() for t in analysis.transients[:50]]}
            if params.get("alerts"):
                from ..alerts import packets_from_analysis, write_alerts
                packets = packets_from_analysis(analysis, series=series)
                out = os.path.abspath(os.path.expanduser(str(params["alerts"])))
                n = write_alerts(out, packets)
                result["alerts"] = {"path": out, "n_packets": n}
            return result

        title = f"{len(paths)} epochs"
        return self.submit("series", title, dict(params, paths=paths), work)

    def simulate(self, params: Dict[str, Any]) -> Job:
        def work(job: Job) -> Dict[str, Any]:
            from ..simulate import SkyConfig, SkySimulator

            size = int(params.get("size", 256))
            config = SkyConfig(shape=(size, size), seed=int(params.get("seed", 42)),
                               n_stars=int(params.get("stars", 80)),
                               n_galaxies=int(params.get("galaxies", 15)),
                               n_nebulae=int(params.get("nebulae", 1)),
                               n_clusters=int(params.get("clusters", 1)),
                               n_lenses=int(params.get("lenses", 1)),
                               n_anomalies=int(params.get("anomalies", 1)))
            simulator = SkySimulator(config)
            out = os.path.abspath(os.path.expanduser(
                str(params.get("out") or os.path.join(self.workdir, "synthetic_field.fits"))))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            epochs = int(params.get("epochs", 1))
            job.stages = [{"name": "simulate", "status": "running", "seconds": 0.0, "message": ""}]
            started = time.time()
            if epochs > 1:
                series, truth, transients = simulator.generate_series(
                    n_epochs=epochs, n_transients=int(params.get("transients", 2)))
                base, ext = os.path.splitext(out)
                written = []
                for index, image in enumerate(series):
                    path = f"{base}_epoch{index:02d}{ext or '.fits'}"
                    image.write(path)
                    written.append(path)
                truth_path = f"{base}_truth.json"
                with open(truth_path, "w", encoding="utf-8") as handle:
                    json.dump({"static": [t.to_dict() for t in truth], "transients": transients,
                               "epochs": written}, handle, indent=2, default=str)
                result = {"paths": written, "truth": truth_path, "n_objects": len(truth)}
            else:
                image, truth = simulator.generate()
                image.write(out)
                truth_path = os.path.splitext(out)[0] + "_truth.json"
                with open(truth_path, "w", encoding="utf-8") as handle:
                    json.dump([t.to_dict() for t in truth], handle, indent=2, default=str)
                result = {"paths": [out], "truth": truth_path, "n_objects": len(truth)}
            job.stages[0].update({"status": "ok", "seconds": time.time() - started})
            return result

        return self.submit("simulate", "synthetic field", dict(params), work)

    # -- results ----------------------------------------------------------------
    def job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"no job {job_id}")
        return job

    def report_html(self, job_id: str) -> str:
        from ..report.html import render_html
        job = self.job(job_id)
        if job.analysis is None:
            raise ValueError("this job has no analysis")
        return render_html(job.analysis, image=job.image)

    def report_json(self, job_id: str) -> Dict[str, Any]:
        from ..report.schema import build_report
        job = self.job(job_id)
        if job.analysis is None:
            raise ValueError("this job has no analysis")
        return build_report(job.analysis, include_catalog=False)

    def catalog(self, job_id: str, offset: int = 0, limit: int = 200,
                sort: str = "snr", descending: bool = True,
                query: str = "") -> Dict[str, Any]:
        job = self.job(job_id)
        if job.analysis is None:
            raise ValueError("this job has no analysis")
        rows = [_catalog_row(s) for s in job.analysis.catalog]
        if query:
            needle = query.lower()
            rows = [r for r in rows if needle in json.dumps(r).lower()]
        key = sort if rows and sort in rows[0] else "snr"
        rows.sort(key=lambda r: (r.get(key) is None, r.get(key) if r.get(key) is not None else 0),
                  reverse=descending)
        if descending:
            # None sorts last either way.
            rows = [r for r in rows if r.get(key) is not None] + [r for r in rows if r.get(key) is None]
        return {"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit],
                "columns": list(rows[0].keys()) if rows else []}

    def candidates(self, job_id: str, limit: int = 40) -> List[Dict[str, Any]]:
        from ..engine.priority import rank_candidates
        job = self.job(job_id)
        if job.analysis is None:
            return []
        return [item.to_dict() for item in rank_candidates(job.analysis, limit=limit)]

    def cutout_png(self, job_id: str, x: float, y: float, size: int = 64,
                   subtracted: bool = False, factor: int = 4) -> bytes:
        from ..vetting.png import stamp_png
        job = self.job(job_id)
        if job.image is None:
            raise ValueError("this job has no image")
        stamp = job.image.cutout(float(x), float(y), int(size), subtract_background=subtracted)
        return stamp_png(stamp, factor=factor)

    def job_preview_png(self, job_id: str, max_size: int = 900) -> bytes:
        from ..vetting.png import encode_png, stretch
        job = self.job(job_id)
        if job.image is None:
            raise ValueError("this job has no image")
        return _preview(job.image.data, max_size, encode_png, stretch)

    # -- alerts and vetting -----------------------------------------------------
    def alerts(self, path: str, limit: int = 200) -> Dict[str, Any]:
        from ..alerts import read_alerts
        path = os.path.abspath(os.path.expanduser(path))
        schema, packets = read_alerts(path)
        rows = []
        for p in packets[:limit]:
            rows.append({"object_id": p.object_id, "candid": p.candid, "ra": p.ra, "dec": p.dec,
                         "mjd": p.mjd, "band": p.band, "mag": p.mag, "mag_err": p.mag_err,
                         "real_bogus": p.real_bogus, "deep_real_bogus": p.deep_real_bogus,
                         "publisher": p.publisher, "format": p.source_format,
                         "n_history": len(p.history), "has_cutouts": p.cutout_science is not None,
                         "classification": p.classification, "verdict": p.verdict})
        return {"path": path, "schema": schema.get("name") if isinstance(schema, dict) else None,
                "n_packets": len(packets), "rows": rows}

    def vet(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a vetting server for a job's candidates or an alert file."""
        from ..vetting import build_queue, queue_for_alert_file, serve
        db = None
        if params.get("db"):
            from ..catalog import CatalogDB
            db = CatalogDB(os.path.abspath(os.path.expanduser(str(params["db"]))))
        limit = int(params.get("limit", 40))
        if params.get("path"):
            queue = queue_for_alert_file(os.path.abspath(os.path.expanduser(str(params["path"]))),
                                         limit=limit, db=db)
        else:
            job = self.job(str(params.get("job_id", "")))
            if job.analysis is None:
                raise ValueError("this job has no analysis to vet")
            queue = build_queue(job.analysis, job.image, limit=limit,
                                include_sources=bool(params.get("all_sources")), db=db)
        if len(queue) == 0:
            raise ValueError("nothing to vet: no candidates")
        log_path = os.path.abspath(os.path.expanduser(
            str(params.get("log") or os.path.join(self.workdir, "verdicts.json"))))
        server = serve(queue, log_path=log_path, host="127.0.0.1", port=0,
                       open_browser=False, block=False)
        self.vetting_servers.append(server)
        return {"url": server.url, "n_items": len(queue), "log": log_path}

    def shutdown(self) -> None:
        for server in self.vetting_servers:
            try:
                server.stop()
            except Exception:                                   # pragma: no cover
                pass


class _Stage:
    """A minimal stand-in for StageResult, for stages run outside the pipeline."""

    def __init__(self, name: str, status: str, detail: Any = None) -> None:
        self.name, self.status, self.seconds, self.message = name, status, 0.0, ""
        self.detail = detail


def _preview(data: np.ndarray, max_size: int, encode_png, stretch) -> bytes:
    array = np.asarray(data, dtype=float)
    step = max(1, int(np.ceil(max(array.shape) / float(max_size))))
    if step > 1:
        h, w = (array.shape[0] // step) * step, (array.shape[1] // step) * step
        array = array[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    return encode_png(stretch(array)[::-1])          # north up: FITS rows start at the bottom


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    return value


def _catalog_row(source) -> Dict[str, Any]:
    p, m = source.photometry, source.morphology
    return {k: _clean(v) for k, v in {
        "id": source.id, "x": round(float(source.x), 2), "y": round(float(source.y), 2),
        "ra": source.ra, "dec": source.dec, "class": source.object_class.value,
        "confidence": source.class_confidence, "mag": p.magnitude, "mag_err": p.magnitude_err,
        "flux": p.flux, "snr": p.snr, "fwhm": m.fwhm, "ellipticity": m.ellipticity,
        "morphology": m.label.value, "sersic_n": m.sersic_index,
        "anomaly": source.anomaly_score, "lens": source.lens_score,
        "flags": ",".join(sorted(source.flags)),
    }.items()}


# -- HTTP ----------------------------------------------------------------------
def _handler_for(app: App):
    from .page import PAGE

    class Handler(BaseHTTPRequestHandler):
        server_version = f"AstroVisionX/{__version__}"

        def log_message(self, fmt, *args):                     # quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload, default=_json_default).encode("utf-8"))

        def _fail(self, exc: Exception) -> None:
            status = 404 if isinstance(exc, (FileNotFoundError, KeyError)) else 400
            self._json(status, {"error": f"{type(exc).__name__}: {exc}"})

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("the request body must be a JSON object")
            return data

        def do_GET(self) -> None:                               # noqa: N802
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            try:
                if not parts:
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif parts == ["favicon.ico"]:
                    self._send(204, b"", "image/x-icon")
                elif parts[0] != "api":
                    self._json(404, {"error": "not found"})
                elif parts[1] == "status":
                    self._json(200, app.status())
                elif parts[1] == "browse":
                    self._json(200, app.browse(query.get("path")))
                elif parts[1] == "inspect":
                    self._json(200, app.inspect(query["path"]))
                elif parts[1] == "preview.png":
                    self._send(200, app.preview_png(query["path"]), "image/png")
                elif parts[1] == "alerts":
                    self._json(200, app.alerts(query["path"], int(query.get("limit", 200))))
                elif parts[1] == "jobs" and len(parts) == 2:
                    self._json(200, [j.to_dict(with_result=False)
                                     for j in sorted(app.jobs.values(), key=lambda j: -j.created)])
                elif parts[1] == "jobs" and len(parts) == 3:
                    self._json(200, app.job(parts[2]).to_dict())
                elif parts[1] == "jobs" and len(parts) == 4:
                    job_id, what = parts[2], parts[3]
                    if what == "report.html":
                        self._send(200, app.report_html(job_id).encode("utf-8"),
                                   "text/html; charset=utf-8")
                    elif what == "report.json":
                        self._json(200, app.report_json(job_id))
                    elif what == "catalog":
                        self._json(200, app.catalog(
                            job_id, int(query.get("offset", 0)), int(query.get("limit", 200)),
                            query.get("sort", "snr"), query.get("order", "desc") != "asc",
                            query.get("q", "")))
                    elif what == "candidates":
                        self._json(200, app.candidates(job_id, int(query.get("limit", 40))))
                    elif what == "cutout.png":
                        self._send(200, app.cutout_png(
                            job_id, float(query["x"]), float(query["y"]),
                            int(query.get("size", 64)), query.get("kind") == "subtracted"),
                            "image/png")
                    elif what == "preview.png":
                        self._send(200, app.job_preview_png(job_id), "image/png")
                    elif what == "file":
                        filename, body = app.job_file(job_id, query.get("name", ""))
                        self.send_response(200)
                        self.send_header("Content-Type", _content_type(filename))
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Content-Disposition",
                                         f'attachment; filename="{filename}"'
                                         if not filename.endswith((".html", ".txt", ".json"))
                                         else f'inline; filename="{filename}"')
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self._json(404, {"error": "not found"})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:                            # noqa: BLE001
                self._fail(exc)

        def do_POST(self) -> None:                              # noqa: N802
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                body = self._body()
                if parts[:2] == ["api", "analyze"]:
                    self._json(200, app.analyze(body).to_dict(with_result=False))
                elif parts[:2] == ["api", "series"]:
                    self._json(200, app.series(body).to_dict(with_result=False))
                elif parts[:2] == ["api", "simulate"]:
                    self._json(200, app.simulate(body).to_dict(with_result=False))
                elif parts[:2] == ["api", "vet"]:
                    self._json(200, app.vet(body))
                elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                    self._json(200, app.cancel(parts[2]))
                elif parts[:2] == ["api", "shutdown"]:
                    self._json(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:                            # noqa: BLE001
                self._fail(exc)

    return Handler


_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                  ".json": "application/json", ".csv": "text/csv", ".fits": "application/fits",
                  ".png": "image/png", ".avro": "application/octet-stream"}


def _content_type(filename: str) -> str:
    return _CONTENT_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


class AppServer:
    """The HTTP server around an :class:`App`; start, serve, stop."""

    def __init__(self, app: App, host: str = "127.0.0.1", port: int = 8770) -> None:
        self.app = app
        self.httpd = ThreadingHTTPServer((host, int(port)), _handler_for(app))
        self.httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> "AppServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.shutdown()

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
            self.app.shutdown()


def launch(host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True,
           workdir: Optional[str] = None, block: bool = True) -> AppServer:
    """Start the application server and, by default, open the browser on it."""
    app = App(workdir=workdir)
    try:
        server = AppServer(app, host=host, port=port)
    except OSError:
        server = AppServer(app, host=host, port=0)          # the port was taken; any free one
    print(f"AstroVision-X {__version__} desktop: {server.url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(server.url)).start()
    if block:
        server.serve_forever()
        return server
    return server.start()


def self_test() -> int:
    """Run a small simulated analysis end to end; the packaged build's smoke test."""
    import tempfile

    from ..engine import Pipeline
    from ..simulate import quick_field

    started = time.time()
    # Which optional backends the build carries, and why any is missing: a
    # frozen bundle can hold a package and still fail to import it.
    import importlib
    for name in ("scipy", "astropy", "sklearn", "skimage", "matplotlib", "pandas", "torch"):
        try:
            module = importlib.import_module(name)
            print(f"  {name:<11} {getattr(module, '__version__', '?')}")
        except Exception as exc:                            # noqa: BLE001 - reported
            print(f"  {name:<11} not available ({type(exc).__name__}: {str(exc)[:120]})")
    image, truth = quick_field((128, 128), seed=3, n_stars=25, n_galaxies=5)
    analysis = Pipeline().run(image)
    with tempfile.TemporaryDirectory() as tmp:
        from ..report import generate_reports
        written = generate_reports(analysis, tmp, ("html", "json"))
        ok = all(os.path.getsize(p) > 0 for p in written.values())
    print(f"self-test: {len(analysis.catalog)} sources from {len(truth)} injected objects, "
          f"reports {'written' if ok else 'MISSING'}, {time.time() - started:.1f}s")
    return 0 if ok and len(analysis.catalog) > 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="astrovision-gui",
                                     description="AstroVision-X desktop application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--workdir", default=None, help="folder the file browser opens on")
    parser.add_argument("--version", action="version", version=f"AstroVision-X {__version__}")
    parser.add_argument("--self-test", action="store_true",
                        help="analyse a simulated field and exit; used by the packaged builds")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    launch(host=args.host, port=args.port, open_browser=not args.no_browser,
           workdir=args.workdir, block=True)
    return 0


__all__ = ["App", "AppServer", "Job", "launch", "main", "self_test", "STAGES", "BOUNDARY"]
