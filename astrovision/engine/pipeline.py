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
from ..classify import Classifier
from ..core.config import AstroVisionConfig
from ..core.exceptions import PipelineError
from ..core.logging import get_logger, timed
from ..core.types import FieldAnalysis
from ..detect import Detector
from ..io.image import AstroImage, ImageSeries
from ..lensing import LensSearch
from ..ml.clustering import cluster, silhouette_score
from ..ml.features import catalog_embeddings, catalog_features, feature_report
from ..morphology import MorphologyAnalyzer
from ..photometry import Photometer
from ..preprocess import Preprocessor
from ..preprocess.psf import build_psf
from ..segment import Segmenter
from ..timeseries import LightCurveAnalyzer
from ..transient import TransientDetector
from .assistant import ResearchAssistant

log = get_logger("engine.pipeline")


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

    def __init__(self, config: Optional[AstroVisionConfig] = None):
        self.config = config or AstroVisionConfig()
        self.stages: List[StageResult] = []
        self.preprocessor = Preprocessor(self.config.preprocess)
        self.detector = Detector(self.config.detection)
        self.photometer = Photometer(self.config.photometry)
        self.segmenter = Segmenter(self.config.segmentation)
        self.morphology = MorphologyAnalyzer(self.config.morphology)
        self.classifier = Classifier(self.config.classification)
        self.anomaly = AnomalyEngine(self.config.anomaly)
        self.lensing = LensSearch(self.config.lensing)
        self.transient = TransientDetector(self.config.transient)
        self.timeseries = LightCurveAnalyzer(self.config.timeseries)
        self.assistant = ResearchAssistant(self.config.report.top_candidates)
        self.cosmology = Cosmology(self.config.cosmology.H0,
                                   self.config.cosmology.Om0,
                                   self.config.cosmology.Ode0)

    # -- stage plumbing -----------------------------------------------------
    def _stage(self, name: str, enabled: bool, work: Callable[[], Dict[str, Any]],
               analysis: FieldAnalysis) -> StageResult:
        """Run one stage, recording timing, status and any failure."""
        result = StageResult(name=name)
        if not enabled:
            result.status = "skipped"
            result.message = "disabled in configuration"
            self.stages.append(result)
            log.info("stage '%s' skipped (disabled)", name)
            return result

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
        return result

    # -- single-field analysis ---------------------------------------------
    def run(self, image: AstroImage, series: Optional[ImageSeries] = None,
            redshift: Optional[float] = None,
            preprocess: bool = True) -> FieldAnalysis:
        """Analyse one field, optionally with a series for time-domain work.

        Set ``preprocess=False`` for an image that has already been
        calibrated and background-subtracted -- a stacked template, for
        instance -- so it is not subtracted a second time.
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

                def light_curves() -> Dict[str, Any]:
                    analysis.light_curves = self.timeseries.run(series, analysis.catalog)
                    return dict(self.timeseries.report)

                self._stage("timeseries", cfg.timeseries.enabled, light_curves, analysis)
            else:
                for name in ("transient", "timeseries"):
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
