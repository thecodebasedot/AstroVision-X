"""The end-to-end analysis pipeline.

This is the orchestrator the whole platform exists to support: pixels in,
a scientific report out.  Stages run in a fixed order because each depends
on the last -- detection needs the background and PSF, photometry needs the
segmentation, morphology needs the aperture radii, classification needs the
morphology, and the research assistant needs all of it.

Every stage is individually skippable through configuration, and a stage
that fails is recorded as a warning rather than being allowed to abort the
run: a partial catalog with an honest account of what is missing is far more
useful than no catalog at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..anomaly import AnomalyEngine
from ..astrophysics import annotate_physical, field_statistics, stellar_population_summary
from ..astrophysics.cosmology import Cosmology
from ..calibration.astrometry import apply_solution, solve_plate
from ..calibration.photometry import apply_zero_point, solve_zero_point
from ..classify import Classifier
from ..core.config import AstroVisionConfig
from ..core.exceptions import PipelineError
from ..core.logging import get_logger, timed
from ..core.provenance import build_manifest, catalog_digest
from ..core.types import FieldAnalysis
from ..detect import Detector
from ..io.external import build_service, crossmatch_catalog
from ..io.image import AstroImage, ImageSeries
from ..lensing import LensSearch
from ..ml.clustering import cluster, silhouette_score
from ..ml.features import catalog_embeddings, catalog_features, feature_report
from ..morphology import MorphologyAnalyzer
from ..moving import MovingObjectFinder
from ..photometry import Photometer
from ..photometry.multiband import forced_photometry, measure_colours
from ..photoz import PhotoZLibrary, fit_catalog
from ..preprocess import Preprocessor
from ..preprocess.psf import build_psf
from ..segment import Segmenter
from ..timeseries import LightCurveAnalyzer
from ..transient import TransientDetector
from .assistant import ResearchAssistant

log = get_logger("engine.pipeline")

#: Below this PSF FWHM (pixels) a source's extent cannot be measured, and
#: everything that depends on "resolved" is reported with a warning.
UNDERSAMPLED_FWHM_PX = 2.0

#: Conventional blue-to-red ordering, so that a default colour pair is a
#: colour and not its negative.  Unknown names keep their given order after
#: the ones that are recognised.
BAND_SEQUENCE = ("u", "g", "r", "i", "z", "y", "J", "H", "K")


def _field_cone_of(catalog):
    """The cone covering a catalog, or ``None`` without sky coordinates."""
    from ..io.external import field_cone
    return field_cone(catalog)


def _band_order(band: str) -> int:
    """Sort key putting known filters blue to red and unknown ones last."""
    return BAND_SEQUENCE.index(band) if band in BAND_SEQUENCE else len(BAND_SEQUENCE)


def _order_bands(images: Dict[str, Any]) -> List[str]:
    """Sort band names blue to red, leaving unrecognised ones at the end."""
    known = [b for b in BAND_SEQUENCE if b in images]
    rest = [b for b in images if b not in known]
    return known + rest


@dataclass
class StageResult:
    """Bookkeeping for one pipeline stage."""

    name: str
    status: str = "pending"          # pending | ok | skipped | failed
    seconds: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status,
                "seconds": round(float(self.seconds), 3),
                "detail": _jsonable(self.detail), "message": self.message}


class Pipeline:
    """Runs every analysis stage over one field or one multi-epoch series.

    >>> from astrovision.simulate import quick_field
    >>> image, _ = quick_field((160, 160))
    >>> analysis = Pipeline().run(image)
    >>> analysis.summary()["n_sources"] > 0
    True
    """

    def __init__(self, config: Optional[AstroVisionConfig] = None,
                 progress: Optional[Callable[[StageResult], None]] = None):
        self.config = config or AstroVisionConfig()
        self.stages: List[StageResult] = []
        #: Called with a :class:`StageResult` as each stage starts
        #: (``status == "running"``) and ends; a GUI's progress bar hangs on it.
        self.progress = progress
        self.preprocessor = Preprocessor(self.config.preprocess)
        self.detector = Detector(self.config.detection)
        self.photometer = Photometer(self.config.photometry)
        self.segmenter = Segmenter(self.config.segmentation)
        self.morphology = MorphologyAnalyzer(self.config.morphology)
        self.classifier = Classifier(self.config.classification)
        self.anomaly = AnomalyEngine(self.config.anomaly)
        self.lensing = LensSearch(self.config.lensing)
        self.transient = TransientDetector(self.config.transient)
        self.moving = MovingObjectFinder(self.config.moving)
        self.timeseries = LightCurveAnalyzer(self.config.timeseries)
        self.assistant = ResearchAssistant(self.config.report.top_candidates)
        self.cosmology = Cosmology(self.config.cosmology.H0,
                                   self.config.cosmology.Om0,
                                   self.config.cosmology.Ode0)
        self._service = None

    # -- stage plumbing -----------------------------------------------------
    def _stage(self, name: str, enabled: bool, work: Callable[[], Dict[str, Any]],
               analysis: FieldAnalysis) -> StageResult:
        """Run one stage, recording timing, status and any failure."""
        result = StageResult(name=name)
        if not enabled:
            result.status = "skipped"
            result.message = "disabled in configuration"
            self.stages.append(result)
            self._notify(result)
            log.info("stage '%s' skipped (disabled)", name)
            return result

        self._notify(StageResult(name=name, status="running"))
        start = time.perf_counter()
        try:
            result.detail = work() or {}
            result.status = "ok"
        except Exception as exc:                     # noqa: BLE001 - reported, not raised
            result.status = "failed"
            result.message = f"{type(exc).__name__}: {exc}"
            analysis.warn(f"stage '{name}' failed: {result.message}")
            log.exception("stage '%s' failed", name)
        result.seconds = time.perf_counter() - start
        self.stages.append(result)
        self._notify(result)
        return result

    def _notify(self, result: StageResult) -> None:
        """Tell whoever is watching (a progress bar) where the run is."""
        if self.progress is None:
            return
        try:
            self.progress(result)
        except Exception:                            # noqa: BLE001 - a watcher never stops the run
            log.debug("progress callback failed", exc_info=True)

    def _reference_service(self):
        """The external-catalog backend, built once and shared.

        Calibration and the known-object crossmatch ask the same service the
        same kind of question, so they share one instance -- and, with a cache
        configured, one set of answers.
        """
        if self._service is None:
            cfg = self.config.crossmatch
            self._service = build_service(
                cfg.backend, path=cfg.path, catalog=cfg.catalog,
                timeout=cfg.timeout, cache_dir=cfg.cache_dir,
                cache_max_age_days=cfg.cache_max_age_days)
        return self._service

    # -- single-field analysis ---------------------------------------------
    def run(self, image: AstroImage, series: Optional[ImageSeries] = None,
            redshift: Optional[float] = None,
            preprocess: bool = True,
            bands: Optional[Dict[str, AstroImage]] = None) -> FieldAnalysis:
        """Analyse one field, optionally with a series for time-domain work.

        Set ``preprocess=False`` for an image that has already been
        calibrated and background-subtracted -- a stacked template, for
        instance -- so it is not subtracted a second time.

        ``bands`` supplies the other filters of the same sky, keyed by band
        name.  Detection, segmentation and morphology stay on ``image``;
        the extra bands are measured by forced photometry at the positions
        already found, which is the only way to get colours that mean
        anything.  They are expected to be preprocessed already.
        """
        cfg = self.config
        self.stages = []
        analysis = FieldAnalysis()
        started = time.time()

        with timed(f"pipeline '{cfg.name}' on '{image.name}'", log):
            state: Dict[str, Any] = {"image": image, "segmentation": None,
                                     "embeddings": None}

            def run_preprocess() -> Dict[str, Any]:
                state["image"] = self.preprocessor.run(image)
                return dict(self.preprocessor.report)

            if preprocess:
                self._stage("preprocess", True, run_preprocess, analysis)
            else:
                self.stages.append(StageResult(
                    name="preprocess", status="skipped",
                    message="image is already calibrated and background-subtracted"))
            clean: AstroImage = state["image"]

            def detect() -> Dict[str, Any]:
                catalog, segmentation = self.detector.detect(clean)
                analysis.catalog = catalog
                state["segmentation"] = segmentation
                return {"n_sources": len(catalog),
                        "backend": cfg.detection.backend,
                        "threshold_sigma": cfg.detection.threshold_sigma}

            self._stage("detect", True, detect, analysis)
            if len(analysis.catalog) == 0:
                analysis.warn("no sources detected; later stages have nothing to work on")
                return self._finish(analysis, clean, started, redshift)

            self._stage("photometry", True, lambda: dict(
                self.photometer.run(clean, analysis.catalog,
                                    state["segmentation"]).meta.get("photometry", {})),
                analysis)

            def calibrate() -> Dict[str, Any]:
                service = self._reference_service()
                cone = _field_cone_of(analysis.catalog)
                if cone is None:
                    analysis.warn("calibration needs sky coordinates; skipped")
                    return {"skipped": "no WCS"}
                reference = service.query(*cone)
                if not reference:
                    analysis.warn("no reference standards returned; the WCS and "
                                  "zero point are the header's, not measured")
                    return {"n_reference": 0}
                out: Dict[str, Any] = {"n_reference": len(reference)}
                if cfg.calibration.astrometry and clean.wcs is not None:
                    solution = solve_plate(
                        analysis.catalog, reference, clean.wcs,
                        radius_arcsec=cfg.calibration.match_radius_arcsec,
                        min_matches=cfg.calibration.min_matches,
                        rounds=cfg.calibration.rounds)
                    out["astrometry"] = solution.to_dict()
                    if solution.succeeded:
                        clean.wcs = solution.wcs
                        apply_solution(analysis.catalog, solution)
                    else:
                        analysis.warn(f"plate solution failed: {solution.reason}")
                if cfg.calibration.photometry:
                    solution = solve_zero_point(
                        analysis.catalog, reference, band=clean.band or "",
                        reference_band=cfg.calibration.reference_band,
                        colour_pair=(tuple(cfg.calibration.colour_pair)
                                     if cfg.calibration.colour_pair else None),
                        radius_arcsec=cfg.calibration.standard_radius_arcsec,
                        min_standards=cfg.calibration.min_standards,
                        min_snr=cfg.calibration.min_standard_snr)
                    out["photometry"] = solution.to_dict()
                    if solution.succeeded:
                        apply_zero_point(analysis.catalog, solution, clean.band or "")
                    else:
                        analysis.warn(f"zero point not calibrated: {solution.reason}")
                return out

            self._stage("calibration",
                        cfg.calibration.astrometry or cfg.calibration.photometry,
                        calibrate, analysis)

            def multiband() -> Dict[str, Any]:
                images = dict(bands or {})
                images.setdefault(clean.band or "detection", clean)
                if len(images) < 2:
                    return {"skipped": "only one band supplied"}
                report = forced_photometry(
                    images, analysis.catalog,
                    detection_band=(cfg.multiband.detection_band
                                    or clean.band or "detection"),
                    aperture_arcsec=cfg.multiband.aperture_arcsec,
                    homogenise_psf=cfg.multiband.homogenise_psf,
                    target_fwhm_arcsec=cfg.multiband.target_fwhm_arcsec,
                    use_kron=cfg.multiband.use_kron,
                    segmentation=state["segmentation"],
                    annulus_arcsec=tuple(cfg.multiband.annulus_arcsec))
                pairs = cfg.multiband.colour_pairs
                if not pairs:
                    ordered = _order_bands(images)
                    pairs = list(zip(ordered[:-1], ordered[1:]))
                counts = measure_colours(analysis.catalog, pairs,
                                         cfg.multiband.min_colour_snr)
                for message in report.warnings:
                    analysis.warn(f"multi-band: {message}")
                return {**report.to_dict(), "colours": counts}

            self._stage("multiband", cfg.multiband.enabled and bool(bands),
                        multiband, analysis)

            def segment() -> Dict[str, Any]:
                components = self.segmenter.run(clean, analysis.catalog,
                                                state["segmentation"])
                return {"n_decomposed": len(components)}

            self._stage("segmentation", cfg.segmentation.enabled, segment, analysis)

            def morphology() -> Dict[str, Any]:
                self.morphology.run(clean, analysis.catalog, state["segmentation"])
                measured = sum(1 for s in analysis.catalog
                               if s.morphology.label.value not in ("unknown", "unresolved"))
                return {"n_measured": measured}

            self._stage("morphology", cfg.morphology.enabled, morphology, analysis)

            self._stage("classification", True, lambda: dict(
                self.classifier.run(clean, analysis.catalog) and self.classifier.report),
                analysis)

            def photoz() -> Dict[str, Any]:
                measured = sorted({band for source in analysis.catalog
                                   for band in source.bands}, key=_band_order)
                bands = cfg.photoz.bands or measured
                if len(bands) < 3:
                    analysis.warn(
                        f"photometric redshifts need at least three filters; "
                        f"{len(bands)} available")
                    return {"skipped": f"only {len(bands)} band(s)"}
                library = PhotoZLibrary(bands=bands, z_min=cfg.photoz.z_min,
                                        z_max=cfg.photoz.z_max, n_z=cfg.photoz.n_z)
                report = fit_catalog(analysis.catalog, library,
                                     min_snr=cfg.photoz.min_snr)
                if len(bands) < 5:
                    analysis.warn(
                        f"photometric redshifts from {len(bands)} filters: with "
                        "fewer than five the redshift, spectral type and dust are "
                        "not separable and the outlier rate is high")
                return report

            self._stage("photoz",
                        cfg.photoz.enabled and any(s.bands for s in analysis.catalog),
                        photoz, analysis)

            def crossmatch() -> Dict[str, Any]:
                service = self._reference_service()
                report = crossmatch_catalog(
                    analysis.catalog, service,
                    radius_arcsec=cfg.crossmatch.radius_arcsec,
                    max_field_radius_arcsec=cfg.crossmatch.max_field_radius_arcsec)
                if report.error:
                    analysis.warn(f"external crossmatch: {report.error}")
                if not report.conclusive:
                    analysis.warn(
                        "no external catalog was consulted, so nothing in this "
                        "field has been shown to be previously unknown")
                return report.to_dict()

            self._stage("crossmatch", cfg.crossmatch.backend not in ("none", "", None),
                        crossmatch, analysis)

            def embed() -> Dict[str, Any]:
                # Morphology can change which sources are worth embedding, so
                # this runs after classification rather than with detection.
                embeddings = catalog_embeddings(analysis.catalog, clean,
                                                cfg.segmentation.cutout_size)
                state["embeddings"] = embeddings
                return {"dimension": int(embeddings.shape[1]) if embeddings.size else 0}

            self._stage("embeddings", cfg.classification.use_embeddings, embed, analysis)

            def anomaly() -> Dict[str, Any]:
                analysis.anomalies = self.anomaly.run(analysis.catalog,
                                                      state["embeddings"])
                return dict(self.anomaly.report)

            self._stage("anomaly", cfg.anomaly.enabled, anomaly, analysis)

            def lensing() -> Dict[str, Any]:
                analysis.lenses = self.lensing.run(clean, analysis.catalog)
                return dict(self.lensing.report)

            self._stage("lensing", cfg.lensing.enabled, lensing, analysis)

            if series is not None and len(series) >= 2:
                def transients() -> Dict[str, Any]:
                    analysis.transients = self.transient.run(series, analysis.catalog)
                    return dict(self.transient.report)

                self._stage("transient", cfg.transient.enabled, transients, analysis)

                def movers() -> Dict[str, Any]:
                    per_epoch = getattr(self.transient, "per_epoch", None)
                    if not per_epoch:
                        return {"skipped": "the transient stage produced no detections"}
                    result = self.moving.run(
                        series, per_epoch,
                        getattr(self.transient, "differences", None))
                    analysis.tracklets = result.tracklets
                    if result.tracklets:
                        analysis.warn(
                            f"{len(result.tracklets)} moving-object tracklet(s) found; "
                            "each needs an orbit and a Minor Planet Center check "
                            "before it is an object, let alone a new one")
                    # Movers are demoted inside the transient list rather than
                    # removed, so the evidence behind the interpretation stays
                    # visible to anyone who disagrees with it.
                    analysis.transients = [
                        c for c in analysis.transients
                        if "moving_object" not in c.flags] + [
                        c for c in analysis.transients if "moving_object" in c.flags]
                    return dict(self.moving.report)

                self._stage("moving", cfg.moving.enabled, movers, analysis)

                def light_curves() -> Dict[str, Any]:
                    analysis.light_curves = self.timeseries.run(series, analysis.catalog)
                    return dict(self.timeseries.report)

                self._stage("timeseries", cfg.timeseries.enabled, light_curves, analysis)
            else:
                for name in ("transient", "moving", "timeseries"):
                    self.stages.append(StageResult(
                        name=name, status="skipped",
                        message="needs a multi-epoch series"))

            def clustering() -> Dict[str, Any]:
                embeddings = state["embeddings"]
                if embeddings is None or len(embeddings) < cfg.clustering.n_clusters:
                    return {"skipped": "too few sources"}
                result = cluster(embeddings, cfg.clustering.method,
                                 n_clusters=cfg.clustering.n_clusters,
                                 min_cluster_size=cfg.clustering.min_cluster_size,
                                 eps=cfg.clustering.eps,
                                 random_state=cfg.clustering.random_state)
                labels = result["labels"]
                for source, value in zip(analysis.catalog, labels):
                    source.meta["cluster"] = int(value)
                return {"method": cfg.clustering.method,
                        "n_clusters": int(len(set(labels) - {-1})),
                        "n_noise": int(np.sum(labels == -1)),
                        "silhouette": float(silhouette_score(embeddings, labels))}

            self._stage("clustering", cfg.clustering.enabled, clustering, analysis)

        return self._finish(analysis, clean, started, redshift)

    def run_series(self, series: ImageSeries, redshift: Optional[float] = None
                   ) -> FieldAnalysis:
        """Analyse a multi-epoch series: deep stack for the catalog, epochs for
        the time domain."""
        if len(series) == 0:
            raise PipelineError("cannot analyse an empty image series")
        prepared = ImageSeries([self.preprocessor.run(image) for image in series],
                               name=series.name)
        reference = prepared.stack("median")
        reference.background = np.zeros(reference.shape, dtype=float)
        # Combining N epochs reduces the noise by roughly sqrt(N); without
        # this the detection threshold on the stack would be set from a
        # single epoch's noise and the stack's extra depth would be wasted.
        reference.background_rms = (prepared.reference.rms_map() /
                                    np.sqrt(max(len(prepared), 1)))
        reference.meta = dict(prepared.reference.meta)
        reference.mask = None
        masks = [image.mask for image in prepared if image.mask is not None]
        if masks:
            reference.mask = np.logical_and.reduce(masks)
        # Re-measure the PSF on the stack itself: co-adding epochs of
        # slightly different seeing does not preserve any one of them.
        psf_model = build_psf(reference.data, rms=reference.rms_map())
        reference.meta["psf_model"] = psf_model
        reference.meta["psf"] = psf_model.to_dict()

        analysis = self.run(reference, series=prepared, redshift=redshift,
                            preprocess=False)
        analysis.provenance["series"] = {
            "n_epochs": len(prepared),
            "epochs": [image.name for image in prepared],
            "times": prepared.times.tolist(),
            "bands": prepared.bands(),
        }
        return analysis

    # -- finalisation -------------------------------------------------------
    def _data_quality_warnings(self, analysis: FieldAnalysis, image: AstroImage) -> None:
        """Say plainly when the data cannot support what the later stages do.

        Both came from running real images.  A Spitzer IRAC frame with a
        1.7-pixel PSF classified three-quarters of a Galactic-plane star
        field as galaxies and flagged hundreds of lens candidates: at that
        sampling the resolved/unresolved distinction the classifier, the
        morphology and the lens search all rest on does not exist.  And an
        image in MJy/sr with no zero point in its header was reported in
        magnitudes that meant nothing.
        """
        psf = image.meta.get("psf_model")
        fwhm = float(getattr(psf, "fwhm", np.nan)) if psf is not None else np.nan
        if np.isfinite(fwhm) and fwhm < UNDERSAMPLED_FWHM_PX:
            analysis.warn(
                f"PSF FWHM {fwhm:.2f} px: the image is undersampled, so star/galaxy "
                "separation, morphology and the lens search cannot tell resolved from "
                "unresolved; treat every extended-source class and lens candidate as "
                "unverified")
        load = image.meta.get("survey_load") or {}
        if load and "MAGZP" not in image.header:
            unit = load.get("unit") or "unknown units"
            analysis.warn(
                f"no photometric zero point in the header (pixels in {unit}); "
                f"magnitudes assume the configured zero point "
                f"{self.config.photometry.zero_point:g} and are instrumental")

    def _finish(self, analysis: FieldAnalysis, image: AstroImage, started: float,
                redshift: Optional[float]) -> FieldAnalysis:
        """Attach statistics, physical properties, provenance and narrative."""
        cfg = self.config
        pixel_scale = image.pixel_scale if image.wcs is not None else cfg.photometry.pixel_scale

        def statistics() -> Dict[str, Any]:
            analysis.statistics["field"] = field_statistics(
                analysis.catalog, image.shape, pixel_scale)
            analysis.statistics["stellar"] = stellar_population_summary(analysis.catalog)
            analysis.statistics["photometry"] = dict(self.photometer.report)
            if self.transient.report:
                analysis.statistics["transient"] = dict(self.transient.report)
            if self.timeseries.report:
                analysis.statistics["timeseries"] = dict(self.timeseries.report)
            matrix, names = catalog_features(analysis.catalog)
            analysis.statistics["features"] = feature_report(matrix, names)
            analysis.statistics["physical"] = annotate_physical(
                analysis.catalog, redshift, pixel_scale, image.band, self.cosmology)
            return {"n_sources": len(analysis.catalog)}

        self._stage("statistics", True, statistics, analysis)
        self._data_quality_warnings(analysis, image)

        analysis.provenance = {
            "pipeline": cfg.name,
            "image": {"name": image.name, "shape": list(image.shape),
                      "band": image.band, "mjd": image.mjd,
                      "exposure_time": image.exposure_time,
                      "pixel_scale_arcsec": float(pixel_scale)},
            "wcs": None if image.wcs is None else image.wcs.to_dict(),
            "psf": image.meta.get("psf"),
            "config": cfg.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "elapsed_seconds": round(time.time() - started, 3),
            "capabilities": _capabilities(),
            "version": _version(),
        }
        # What it would take to get this exact catalog again: the
        # configuration, the code revision, the numerical dependencies, the
        # seeds and the input checksum -- and a digest of the catalog itself,
        # so a repeat run can be checked against this one.
        inputs = [p for p in (image.meta.get("source_path"),) if p]
        manifest = build_manifest(cfg, inputs=inputs,
                                  seeds={"random_state": int(cfg.random_state)})
        manifest.outputs["catalog_digest"] = catalog_digest(analysis.catalog)
        analysis.provenance["manifest"] = manifest.to_dict()
        analysis.provenance["reproducibility_key"] = manifest.reproducibility_key()

        self._stage("assistant", True,
                    lambda: {"narrative": self.assistant.report(analysis)}, analysis)
        for stage in self.stages:
            if stage.name == "assistant" and stage.status == "ok":
                analysis.statistics["narrative"] = stage.detail["narrative"]

        failed = [s.name for s in self.stages if s.status == "failed"]
        if failed:
            analysis.warn(f"stages that failed: {', '.join(failed)}")
        log.info("pipeline finished in %.2fs: %s",
                 analysis.provenance["elapsed_seconds"], analysis.summary())
        return analysis


def _capabilities() -> Dict[str, bool]:
    from ..core.backend import capabilities
    return capabilities()


def _version() -> str:
    from .. import __version__
    return __version__


def _jsonable(value: Any) -> Any:
    """Coerce stage details into something JSON can hold."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
