"""The morphology stage: measure shape statistics for every resolved source."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.config import MorphologyConfig
from ..core.logging import get_logger
from ..core.types import Morphology, ObjectClass, Source, SourceCatalog
from ..io.image import AstroImage
from .cas import asymmetry, smoothness
from .classify import classify_morphology
from .gini_m20 import bulge_statistic, gini_m20, merger_statistic
from .sersic import fit_sersic
from .uncertainty import annotate_uncertainty, bootstrap_morphology
from .segmentation import petrosian_segmentation
from .spiral import detect_bar, detect_spiral_arms

log = get_logger("morphology.analyzer")


class MorphologyAnalyzer:
    """Runs the full non-parametric and parametric morphology suite.

    Point sources are skipped: CAS, Gini/M20 and Sersic indices are
    meaningless for anything unresolved, and computing them anyway would
    fill the catalog with numbers that look like measurements but are not.
    """

    def __init__(self, config: Optional[MorphologyConfig] = None):
        self.config = config or MorphologyConfig()

    def run(self, image: AstroImage, catalog: SourceCatalog,
            segmentation: Optional[np.ndarray] = None) -> SourceCatalog:
        """Measure and classify morphology for every eligible source."""
        if not self.config.enabled:
            return catalog

        cfg = self.config
        data = image.subtracted()
        psf_model = image.meta.get("psf_model")
        psf_kernel = psf_model.as_kernel() if psf_model is not None else None
        psf_fwhm = float(psf_model.fwhm) if psf_model is not None else 0.0
        n_measured = 0

        for source in catalog:
            if not self._eligible(source, psf_fwhm):
                source.morphology.label = Morphology.UNRESOLVED
                continue

            # Morphology needs the whole object, not the tight detection
            # box: a curve of growth or an M20 truncated at the isophote
            # produces numbers that look measured but are not.
            reach = self._reach(source, psf_fwhm)
            rows, cols = self._window(source, data.shape, reach)
            cut = data[rows, cols]
            if cut.size < 16:
                continue
            centre = (source.x - cols.start, source.y - rows.start)

            detection_footprint = None
            fit_region = None
            if segmentation is not None:
                window = segmentation[rows, cols]
                detection_footprint = window == source.segment_label
                # Replace neighbouring sources with sky.  Left in place they
                # drag the Sersic index up and the concentration down, since
                # both are dominated by the faint outer parts of the profile.
                neighbours = (window > 0) & ~detection_footprint
                if neighbours.any():
                    cut = np.where(neighbours, 0.0, cut)
                    fit_region = ~neighbours
                    source.meta["neighbour_pixels_masked"] = int(neighbours.sum())
            footprint = petrosian_segmentation(
                cut, centre, source.photometry.petrosian_radius,
                fallback=detection_footprint)
            if footprint is None or footprint.sum() < cfg.min_area_for_morphology:
                footprint = detection_footprint
            if footprint is not None and footprint.sum() < cfg.min_area_for_morphology:
                footprint = None
            source.meta["morphology_area"] = int(footprint.sum()) if footprint is not None else 0
            local_noise = float(source.meta.get("local_rms", 0.0) or 0.0)
            extras: Dict[str, Any] = {}

            if cfg.compute_cas:
                # Concentration already comes from the photometry stage's
                # curve of growth, so only A and S are computed here.
                result = asymmetry(cut, centre, footprint)
                source.morphology.asymmetry = result["asymmetry"]
                result_s = smoothness(cut, centre=centre, mask=footprint)
                source.morphology.smoothness = result_s["smoothness"]
                extras.update(result_s)

            if cfg.compute_gini_m20:
                stats = gini_m20(cut, footprint)
                source.morphology.gini = stats.get("gini", float("nan"))
                source.morphology.m20 = stats.get("m20", float("nan"))
                extras["merger_statistic"] = merger_statistic(
                    source.morphology.gini, source.morphology.m20)
                extras["bulge_statistic"] = bulge_statistic(
                    source.morphology.gini, source.morphology.m20)

            if cfg.uncertainty and local_noise > 0:
                # Error bars come from re-measuring on noise realisations, so
                # this genuinely costs `bootstrap_samples` times the shape
                # measurement -- which is why it is off by default rather
                # than quietly making every run an order of magnitude slower.
                errors = bootstrap_morphology(
                    cut, local_noise, centre, footprint,
                    n_samples=cfg.bootstrap_samples, seed=source.id)
                annotate_uncertainty(source, errors)

            axis_ratio = self._axis_ratio(source)
            if cfg.fit_sersic:
                fit = fit_sersic(cut, centre, footprint, axis_ratio,
                                 source.morphology.position_angle,
                                 refine=True, psf=psf_kernel, psf_fwhm=psf_fwhm,
                                 fit_mask=fit_region,
                                 r_half=float(source.meta.get("r50", float("nan"))),
                                 noise=local_noise if local_noise > 0 else float("nan"))
                if fit.success:
                    source.morphology.sersic_index = fit.n
                    source.morphology.effective_radius = fit.r_eff
                    source.meta["sersic"] = fit.to_dict()
                    if abs(fit.worst_correlation[2]) > 0.95:
                        source.add_flag("degenerate_sersic_fit")

            if cfg.detect_spiral_arms and source.morphology.area_pixels >= 40:
                max_radius = min(reach, 0.48 * min(cut.shape))
                arms = detect_spiral_arms(cut, centre, axis_ratio,
                                          source.morphology.position_angle,
                                          max_radius, n_angular=cfg.polar_bins,
                                          noise=local_noise)
                source.morphology.spiral_strength = arms["spiral_strength"]
                source.morphology.arm_count = arms["arm_count"]
                source.meta["spiral"] = arms
                extras.update({"coherence": arms["coherence"],
                               "pitch_angle": arms["pitch_angle"],
                               "arm_significance": arms["arm_significance"],
                               "winding": arms.get("winding", 0.0)})

                bar = detect_bar(cut, centre, axis_ratio,
                                 source.morphology.position_angle,
                                 max_radius, n_angular=cfg.polar_bins)
                source.morphology.bar_strength = bar["bar_strength"]
                source.meta["bar"] = bar
                extras.update({"bar_detected": bar["bar_detected"]})

            label, confidence, scores = classify_morphology(source.morphology, extras)
            source.morphology.label = label
            source.morphology.label_confidence = confidence
            source.meta["morphology_scores"] = scores
            source.meta["morphology_extras"] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in extras.items()}
            n_measured += 1

        log.info("morphology measured for %d of %d sources", n_measured, len(catalog))
        return catalog

    def _eligible(self, source: Source, psf_fwhm: float) -> bool:
        """Only resolved sources get a morphological measurement."""
        if source.object_class == ObjectClass.STAR:
            return False
        if source.morphology.area_pixels < self.config.min_area_for_morphology:
            return False
        # Resolved means measurably broader than the point-spread function.
        if psf_fwhm > 0 and np.isfinite(source.morphology.fwhm):
            if source.morphology.fwhm < 1.15 * psf_fwhm:
                return False
        return True

    @staticmethod
    def _reach(source: Source, psf_fwhm: float) -> float:
        """Radius, in pixels, that comfortably encloses the whole object."""
        candidates = [4.0 * max(source.morphology.semi_major, 1.0)]
        if np.isfinite(source.photometry.petrosian_radius):
            candidates.append(1.8 * source.photometry.petrosian_radius)
        if np.isfinite(source.photometry.kron_radius):
            candidates.append(4.0 * source.photometry.kron_radius)
        if psf_fwhm > 0:
            candidates.append(3.0 * psf_fwhm)
        return float(np.clip(max(candidates), 8.0, 160.0))

    @staticmethod
    def _window(source: Source, shape, reach: float):
        """Square slice of the image centred on the source, clipped to bounds."""
        ny, nx = shape
        half = int(np.ceil(reach)) + 2
        y0 = int(max(0, np.floor(source.y) - half))
        y1 = int(min(ny, np.ceil(source.y) + half + 1))
        x0 = int(max(0, np.floor(source.x) - half))
        x1 = int(min(nx, np.ceil(source.x) + half + 1))
        return slice(y0, max(y0 + 1, y1)), slice(x0, max(x0 + 1, x1))

    @staticmethod
    def _axis_ratio(source: Source) -> float:
        major = source.morphology.semi_major
        minor = source.morphology.semi_minor
        if not (np.isfinite(major) and np.isfinite(minor)) or major <= 0:
            return 1.0
        return float(np.clip(minor / major, 0.1, 1.0))
