"""The photometry stage: measure every detected source properly."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.config import PhotometryConfig
from ..core.logging import get_logger
from ..core.types import SourceCatalog
from ..io.image import AstroImage
from ..preprocess.varying_psf import psf_at
from .aperture import (circular_aperture_weights, elliptical_photometry, multi_aperture,
                       stamp_box)
from .growth import auto_aperture, concentration_index, curve_of_growth, flux_radius
from .magnitudes import flux_to_magnitude, limiting_magnitude, surface_brightness

log = get_logger("photometry.photometer")


class Photometer:
    """Measures flux, magnitude, size and signal-to-noise for a catalog.

    Each source gets a curve of growth, an adaptive (Kron) aperture, a set
    of fixed apertures for colour work, and the derived magnitudes.  The
    stage also records the image's limiting magnitude, without which no
    completeness statement in the report would mean anything.

    >>> from astrovision.simulate import quick_field
    >>> from astrovision.preprocess import Preprocessor
    >>> from astrovision.detect import Detector
    >>> image, _ = quick_field((128, 128))
    >>> clean = Preprocessor().run(image)
    >>> catalog, segmentation = Detector().detect(clean)
    >>> _ = Photometer().run(clean, catalog, segmentation)
    >>> all(s.photometry.flux == s.photometry.flux for s in catalog)   # no NaN
    True
    """

    def __init__(self, config: Optional[PhotometryConfig] = None):
        self.config = config or PhotometryConfig()
        self.report: Dict[str, float] = {}

    @staticmethod
    def aperture_correction(psf_model, radius: float) -> float:
        """Flux fraction a circular aperture of ``radius`` misses on the PSF.

        Real point-spread functions have wings -- a Moffat profile loses
        several percent outside any practical aperture -- so aperture
        photometry is biased faint unless it is corrected by the enclosed
        energy of the PSF itself.  Returns the multiplicative correction.
        """
        if psf_model is None or radius <= 0:
            return 1.0
        stamp = np.clip(psf_model.as_kernel(), 0, None)
        total = float(stamp.sum())
        if total <= 0:
            return 1.0
        centre = ((stamp.shape[1] - 1) / 2.0, (stamp.shape[0] - 1) / 2.0)
        weights = circular_aperture_weights(stamp.shape, centre, float(radius), subpixels=5)
        enclosed = float((stamp * weights).sum()) / total
        # The empirical PSF stamp is itself truncated, so a correction
        # larger than ~1.5 means the model is unreliable: fall back to 1.
        if not 0.3 < enclosed <= 1.0:
            return 1.0
        return float(1.0 / enclosed)

    def run(self, image: AstroImage, catalog: SourceCatalog,
            segmentation: Optional[np.ndarray] = None) -> SourceCatalog:
        """Measure every source in ``catalog`` in place; returns the catalog."""
        cfg = self.config
        data = image.subtracted()
        rms = image.rms_map()
        gain = float(image.header.get("GAIN", cfg.gain) or cfg.gain)
        zero_point = float(image.header.get("MAGZP", cfg.zero_point) or cfg.zero_point)
        pixel_scale = image.pixel_scale if image.wcs is not None else cfg.pixel_scale

        # Neighbours are masked per source below, so a companion cannot leak
        # into another object's aperture.
        if segmentation is not None:
            segmentation = np.asarray(segmentation, dtype=np.int32)

        radii = sorted(set(list(cfg.aperture_radii) + [cfg.primary_aperture]))
        psf_model = image.meta.get("psf_model")
        apply_correction = psf_model is not None
        auto_max_radius = max(12.0, cfg.annulus_outer)
        for source in catalog:
            # The curve of growth must reach past r80 or the concentration
            # index is truncated -- scale it to the object's own size.
            growth_limit = max(cfg.annulus_inner, 3.0 * cfg.primary_aperture,
                               6.0 * max(source.morphology.semi_major, 1.0))
            growth_limit = min(growth_limit, 0.45 * min(data.shape))
            growth_radii = np.linspace(1.0, growth_limit, 28)

            # Everything below is radial and bounded, so it is measured on a
            # stamp that contains the widest of the apertures, annuli and
            # growth curves. On a survey frame this is the difference
            # between a second per source and a millisecond, and the numbers
            # are the same to the last bit.
            reach = 1.1 * max(growth_limit, auto_max_radius, cfg.annulus_outer) + 3.0
            rows, cols, centre = stamp_box(data.shape, (source.x, source.y), reach)
            stamp = data[rows, cols]
            stamp_rms = rms[rows, cols]
            labels = None if segmentation is None else segmentation[rows, cols]

            neighbours = None
            if labels is not None:
                neighbours = (labels > 0) & (labels != source.segment_label)
            if image.mask is not None:
                local_mask = image.mask[rows, cols]
                neighbours = local_mask if neighbours is None else (neighbours | local_mask)

            fixed = multi_aperture(
                stamp, centre, radii, rms=stamp_rms, gain=gain,
                local_background=cfg.local_background,
                annulus=(cfg.annulus_inner, cfg.annulus_outer),
                mask=neighbours)
            source.meta["apertures"] = {
                f"{r:g}": {"flux": result.flux, "flux_err": result.flux_err,
                           "snr": result.snr}
                for r, result in fixed.items()}

            # Neighbours inside the growth annuli inflate the outer flux and
            # bias the concentration index low; replace them with sky.
            growth_data = stamp
            if neighbours is not None and neighbours.any():
                growth_data = np.where(neighbours, 0.0, stamp)
            radii_array, cumulative = curve_of_growth(growth_data, centre, growth_radii)
            source.meta["r50"] = float(flux_radius(radii_array, cumulative, 0.5))
            source.meta["r90"] = float(flux_radius(radii_array, cumulative, 0.9))
            source.morphology.concentration = float(concentration_index(radii_array, cumulative))

            if cfg.auto_aperture:
                footprint = None if labels is None else labels == source.segment_label
                adaptive = auto_aperture(growth_data, centre, footprint, cfg.kron_factor,
                                         min_radius=max(2.0, cfg.primary_aperture * 0.6),
                                         max_radius=auto_max_radius)
                measurement = elliptical_photometry(
                    stamp, centre,
                    max(adaptive["radius"], 2.0),
                    max(adaptive["radius"] * _axis_ratio(source), 1.5),
                    source.morphology.position_angle, rms=stamp_rms, gain=gain,
                    mask=neighbours)
                source.photometry.kron_radius = adaptive["kron_radius"]
                source.photometry.petrosian_radius = adaptive["petrosian_radius"]
                source.photometry.aperture_radius = adaptive["radius"]
            else:
                measurement = fixed[float(cfg.primary_aperture)]
                source.photometry.aperture_radius = float(cfg.primary_aperture)

            # Fall back to the fixed aperture if the adaptive one failed.
            if not np.isfinite(measurement.flux) or measurement.flux <= 0:
                measurement = fixed[float(cfg.primary_aperture)]
                source.add_flag("aperture_fallback")

            correction = 1.0
            if apply_correction and np.isfinite(source.photometry.aperture_radius):
                # The *local* PSF where the field has a spatial model: an
                # aperture correction derived from a field-average PSF is
                # wrong by the amount the PSF varies, and it is wrong in
                # opposite directions at the centre and the corners.
                local = psf_at(image.meta, source.x, source.y) or psf_model
                correction = self.aperture_correction(
                    local, source.photometry.aperture_radius)
                source.meta["aperture_correction"] = correction

            source.photometry.flux = float(measurement.flux * correction)
            source.photometry.flux_err = float(measurement.flux_err * correction)
            source.photometry.snr = float(measurement.snr)
            source.photometry.background = float(measurement.background)
            magnitude, magnitude_err = flux_to_magnitude(
                source.photometry.flux, zero_point, source.photometry.flux_err)
            source.photometry.magnitude = float(magnitude)
            source.photometry.magnitude_err = float(magnitude_err)
            source.photometry.zero_point = float(zero_point)
            source.photometry.surface_brightness = surface_brightness(
                source.photometry.flux, max(source.morphology.area_pixels, 1),
                pixel_scale, zero_point)

            if not np.isfinite(source.photometry.magnitude):
                source.add_flag("negative_flux")
            if np.isfinite(source.photometry.snr) and source.photometry.snr < 3.0:
                source.add_flag("low_snr")

        median_rms = float(np.median(rms))
        self.report = {
            "zero_point": zero_point,
            "gain": gain,
            "pixel_scale": float(pixel_scale),
            "median_rms": median_rms,
            "limiting_magnitude_5sigma": limiting_magnitude(
                median_rms, cfg.primary_aperture, zero_point, 5.0),
            "n_measured": len(catalog),
            "aperture_corrected": bool(apply_correction),
        }
        catalog.meta["photometry"] = dict(self.report)
        log.info("photometry on %d sources: zp=%.2f, 5-sigma limit=%.2f mag",
                 len(catalog), zero_point, self.report["limiting_magnitude_5sigma"])
        return catalog


def _axis_ratio(source) -> float:
    """Minor/major axis ratio, clamped to a sane range for aperture shaping."""
    major = source.morphology.semi_major
    minor = source.morphology.semi_minor
    if not (np.isfinite(major) and np.isfinite(minor)) or major <= 0:
        return 1.0
    return float(np.clip(minor / major, 0.2, 1.0))
