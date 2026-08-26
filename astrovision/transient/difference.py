"""Difference imaging.

Subtracting a reference epoch from a new one removes every constant source
and leaves only what changed.  Getting that subtraction clean is the whole
problem: the two epochs must be aligned to a fraction of a pixel, matched in
point-spread function, and scaled to a common flux system.  Any of those
done badly leaves residuals at every star that look exactly like transients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, nan_to_finite, sigma_clipped_stats
from ..io.image import AstroImage
from ..preprocess.align import Transform, align_image
from ..preprocess.psf import PSFModel, build_psf, match_psf, matching_kernel

log = get_logger("transient.difference")


@dataclass
class DifferenceResult:
    """A difference image plus everything needed to interpret it."""

    difference: np.ndarray
    noise: np.ndarray
    science: AstroImage
    reference: AstroImage
    transform: Optional[Transform] = None
    flux_scale: float = 1.0
    science_psf: Optional[PSFModel] = None
    reference_psf: Optional[PSFModel] = None
    convolved: str = "none"          # which side was blurred to match
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def significance(self) -> np.ndarray:
        """The difference image in units of its own local noise."""
        return self.difference / np.clip(self.noise, 1e-9, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flux_scale": float(self.flux_scale),
            "convolved": self.convolved,
            "transform": None if self.transform is None else self.transform.to_dict(),
            "science_psf": None if self.science_psf is None else self.science_psf.to_dict(),
            "reference_psf": None if self.reference_psf is None else self.reference_psf.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def flux_scale_factor(science: np.ndarray, reference: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """Photometric ratio between two epochs, from their common bright pixels.

    Exposure time, transparency and airmass all differ between epochs.  The
    ratio is fitted on pixels that are bright in *both* frames, which are
    dominated by the constant sources the subtraction must cancel.
    """
    a = nan_to_finite(as_float_image(science), 0.0)
    b = nan_to_finite(as_float_image(reference), 0.0)
    _, _, noise_a = sigma_clipped_stats(a)
    _, _, noise_b = sigma_clipped_stats(b)
    good = (a > 5 * noise_a) & (b > 5 * noise_b)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    if good.sum() < 25:
        return 1.0
    # A robust ratio: the median of per-pixel ratios resists the very
    # variables and transients the subtraction is meant to reveal.
    ratio = a[good] / np.maximum(b[good], 1e-9)
    scale = float(np.median(ratio))
    return float(np.clip(scale, 0.05, 20.0)) if np.isfinite(scale) else 1.0


def subtract(science: AstroImage, reference: AstroImage, align: bool = True,
             psf_match: bool = True, scale_flux: bool = True) -> DifferenceResult:
    """Produce ``science - reference`` after alignment, PSF matching and scaling."""
    science_data = science.subtracted()
    reference_data = reference.subtracted()
    diagnostics: Dict[str, Any] = {}
    transform: Optional[Transform] = None

    if align and science_data.shape == reference_data.shape:
        reference_data, transform = align_image(science_data, reference_data)
        diagnostics["alignment"] = transform.to_dict()
        # Warping leaves NaN outside the valid footprint; fill with sky.
        reference_data = np.where(np.isfinite(reference_data), reference_data, 0.0)
    elif science_data.shape != reference_data.shape:
        raise ValueError(
            f"science {science_data.shape} and reference {reference_data.shape} "
            "must have the same shape; reproject one onto the other first")

    science_psf = build_psf(science_data, rms=science.rms_map())
    reference_psf = build_psf(reference_data, rms=reference.rms_map())
    convolved = "none"

    if psf_match:
        # Always blur the *sharper* image: deconvolving the blurrier one
        # would amplify noise and ring around every bright star.
        if science_psf.fwhm < reference_psf.fwhm:
            science_data = match_psf(science_data, science_psf, reference_psf)
            convolved = "science"
        elif reference_psf.fwhm < science_psf.fwhm:
            reference_data = match_psf(reference_data, reference_psf, science_psf)
            convolved = "reference"
        diagnostics["psf_match"] = {
            "science_fwhm": float(science_psf.fwhm),
            "reference_fwhm": float(reference_psf.fwhm),
            "convolved": convolved,
            "kernel_size": int(matching_kernel(science_psf, reference_psf).shape[0]),
        }

    scale = flux_scale_factor(science_data, reference_data) if scale_flux else 1.0
    diagnostics["flux_scale"] = scale

    difference = science_data - scale * reference_data

    # Noise adds in quadrature; the reference contributes scaled by the same
    # factor its pixels were.
    science_rms = science.rms_map()
    reference_rms = reference.rms_map()
    noise = np.sqrt(np.clip(science_rms, 0, None) ** 2 +
                    (scale * np.clip(reference_rms, 0, None)) ** 2)

    _, residual_median, residual_rms = sigma_clipped_stats(difference)
    diagnostics["residual_median"] = float(residual_median)
    diagnostics["residual_rms"] = float(residual_rms)
    diagnostics["expected_rms"] = float(np.median(noise))
    # A residual RMS far above the propagated noise means the subtraction
    # itself is leaving structure behind, and candidates will be unreliable.
    diagnostics["subtraction_quality"] = float(
        np.clip(float(np.median(noise)) / max(residual_rms, 1e-9), 0.0, 1.0))

    log.info("differenced '%s' - %.3f x '%s': residual rms %.3g vs expected %.3g",
             science.name, scale, reference.name, residual_rms, float(np.median(noise)))
    return DifferenceResult(
        difference=difference - residual_median, noise=noise,
        science=science, reference=reference, transform=transform,
        flux_scale=scale, science_psf=science_psf, reference_psf=reference_psf,
        convolved=convolved, diagnostics=diagnostics)


def build_template(series, method: str = "median", exclude: Optional[int] = None
                   ) -> AstroImage:
    """Combine epochs into a deep reference, optionally holding one out.

    Holding out the epoch being searched is essential: including it would
    put a fraction of any transient into the very template used to subtract
    it, suppressing exactly what the search is looking for.
    """
    images = [im for i, im in enumerate(series) if i != exclude]
    if not images:
        raise ValueError("cannot build a template with no images")
    cube = np.stack([im.subtracted() for im in images])
    if method == "mean":
        combined = np.nanmean(cube, axis=0)
    elif method == "trimmed":
        # Trimmed mean: drops the extremes so a transient in one epoch does
        # not leak into the template, while keeping more depth than a median.
        ordered = np.sort(cube, axis=0)
        trim = 1 if len(images) >= 5 else 0
        combined = np.nanmean(ordered[trim:len(images) - trim or None], axis=0)
    else:
        combined = np.nanmedian(cube, axis=0)

    base = series.reference
    template = base.copy_with(combined, name=f"template_{method}")
    template.background = np.zeros_like(combined)
    # Combining N epochs reduces the noise by roughly sqrt(N).
    template.background_rms = base.rms_map() / np.sqrt(max(len(images), 1))
    template.meta = dict(base.meta)
    masks = [im.mask for im in images if im.mask is not None]
    if masks:
        # A pixel bad in any contributing epoch is suspect in the stack.
        template.mask = np.logical_or.reduce(masks)
    template.meta["template_from"] = [im.name for im in images]
    template.meta["template_method"] = method
    return template
