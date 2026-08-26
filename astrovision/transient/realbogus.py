"""Real/bogus vetting of difference-image candidates.

Any difference image produces far more artefacts than transients: dipole
residuals from imperfect alignment, cosmic rays, saturation spikes, and
edge effects.  The classic solution is a "real/bogus" classifier, and the
features below are the ones such classifiers have always relied on --
computed here so the decision is inspectable rather than opaque.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, logistic, nan_to_finite
from ..preprocess.psf import PSFModel

log = get_logger("transient.realbogus")

#: Feature names produced by :func:`stamp_features`, in a fixed order.
RB_FEATURES = [
    "peak_significance", "psf_correlation", "dipole_ratio", "negative_fraction",
    "sharpness", "elongation", "centroid_offset", "flux_ratio", "n_pixels",
]


def stamp_features(stamp: np.ndarray, noise: float,
                   psf: Optional[PSFModel] = None) -> Dict[str, float]:
    """Descriptors of one difference-image candidate stamp."""
    data = nan_to_finite(as_float_image(stamp), 0.0)
    ny, nx = data.shape
    if data.size == 0 or noise <= 0:
        return {name: float("nan") for name in RB_FEATURES}

    centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    peak = float(np.max(data))
    trough = float(np.min(data))

    # A genuine point source correlates with the PSF; a cosmic ray or a
    # subtraction artefact does not.
    correlation = float("nan")
    if psf is not None:
        kernel = psf.as_kernel()
        size = min(min(data.shape), min(kernel.shape))
        size = max(3, size | 1)
        half = size // 2
        cy, cx = ny // 2, nx // 2
        ky, kx = kernel.shape[0] // 2, kernel.shape[1] // 2
        a = data[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
        b = kernel[ky - half:ky + half + 1, kx - half:kx + half + 1].ravel()
        if a.size == b.size and a.size > 4 and np.std(a) > 0 and np.std(b) > 0:
            correlation = float(np.corrcoef(a, b)[0, 1])

    # Dipoles -- a positive lobe beside a negative one -- are the signature
    # of a sub-pixel misalignment, not of a new source.
    dipole = float(abs(trough) / max(peak, 1e-9)) if peak > 0 else 1.0
    negative_fraction = float(np.mean(data < -2.0 * noise))

    positive = np.clip(data, 0, None)
    total = float(positive.sum())
    if total > 0:
        yy, xx = np.mgrid[0:ny, 0:nx]
        cx_m = float((positive * xx).sum() / total)
        cy_m = float((positive * yy).sum() / total)
        offset = float(np.hypot(cx_m - centre[0], cy_m - centre[1]))
        dx, dy = xx - cx_m, yy - cy_m
        mxx = float((positive * dx * dx).sum() / total)
        myy = float((positive * dy * dy).sum() / total)
        mxy = float((positive * dx * dy).sum() / total)
        common = np.sqrt(max(((mxx - myy) / 2) ** 2 + mxy ** 2, 0.0))
        mean = (mxx + myy) / 2
        major = np.sqrt(max(mean + common, 1e-9))
        minor = np.sqrt(max(mean - common, 1e-9))
        elongation = float(major / minor)
    else:
        offset, elongation = float(nx), 1.0

    # Sharpness: how much of the flux sits in the single peak pixel.  A
    # cosmic ray is far sharper than the PSF allows.
    sharpness = float(peak / max(total, 1e-9)) if total > 0 else 1.0
    ratio = float(peak / max(abs(trough), 1e-9))

    return {
        "peak_significance": float(peak / max(noise, 1e-9)),
        "psf_correlation": correlation,
        "dipole_ratio": dipole,
        "negative_fraction": negative_fraction,
        "sharpness": sharpness,
        "elongation": elongation,
        "centroid_offset": offset,
        "flux_ratio": ratio,
        "n_pixels": float(np.sum(data > 2.0 * noise)),
    }


def real_bogus_score(features: Dict[str, float], psf_fwhm: float = 3.0,
                     dipole_threshold: float = 0.35) -> Tuple[float, Dict[str, float]]:
    """Combine the features into a 0-1 "is this astrophysically real" score.

    Each term is a soft test rather than a hard cut, so a candidate that is
    marginal on one criterion is penalised rather than discarded -- and the
    per-term breakdown is returned so a human can see what drove the answer.
    """
    terms: Dict[str, float] = {}

    significance = features.get("peak_significance", 0.0)
    terms["significance"] = float(logistic(significance, scale=1.2, midpoint=5.0))

    correlation = features.get("psf_correlation", float("nan"))
    if np.isfinite(correlation):
        # A real point source matches the PSF closely; anything below about
        # 0.5 correlation is not a point source at this seeing.
        terms["psf_match"] = float(np.clip((correlation - 0.25) / 0.5, 0.0, 1.0))

    dipole = features.get("dipole_ratio", 1.0)
    terms["not_dipole"] = float(logistic(-dipole, scale=0.12,
                                         midpoint=-float(dipole_threshold)))

    terms["not_negative"] = float(1.0 - np.clip(
        features.get("negative_fraction", 0.0) / 0.25, 0.0, 1.0))

    # Compare the observed sharpness with what the PSF permits: a cosmic ray
    # concentrates far more flux in one pixel than seeing allows.
    expected_sharpness = 1.0 / max(np.pi * (max(psf_fwhm, 1.0) / 2.0) ** 2, 1.0)
    sharpness = features.get("sharpness", 1.0)
    terms["not_cosmic_ray"] = float(np.clip(
        1.0 - (sharpness / max(expected_sharpness, 1e-9) - 1.6) / 2.0, 0.0, 1.0))

    elongation = features.get("elongation", 1.0)
    terms["not_streak"] = float(np.clip(1.0 - (elongation - 1.6) / 2.0, 0.0, 1.0))

    offset = features.get("centroid_offset", 0.0)
    terms["centred"] = float(np.clip(1.0 - offset / max(psf_fwhm, 1.0), 0.0, 1.0))

    weights = {"significance": 1.4, "psf_match": 1.6, "not_dipole": 1.5,
               "not_negative": 1.0, "not_cosmic_ray": 1.2, "not_streak": 0.8,
               "centred": 1.0}
    usable = {k: v for k, v in terms.items() if np.isfinite(v)}
    if not usable:
        return 0.0, terms

    # A *weighted geometric* mean, not an arithmetic one.  Vetting is a
    # veto: an object that fails one test decisively -- a clean dipole, a
    # one-pixel cosmic ray -- is bogus no matter how well it does on the
    # rest, and an arithmetic mean would let the other terms outvote that.
    total = sum(weights.get(k, 1.0) for k in usable)
    log_score = sum(weights.get(k, 1.0) * np.log(max(v, 1e-4))
                    for k, v in usable.items()) / total
    return float(np.clip(np.exp(log_score), 0.0, 1.0)), terms


def classify_artifact(features: Dict[str, float], terms: Dict[str, float]) -> str:
    """Name the dominant failure mode of a rejected candidate."""
    ranked = sorted((v, k) for k, v in terms.items() if np.isfinite(v))
    if not ranked:
        return "unknown"
    worst = ranked[0][1]
    return {
        "not_dipole": "subtraction_dipole",
        "not_cosmic_ray": "cosmic_ray",
        "not_streak": "streak_or_satellite",
        "not_negative": "oversubtraction",
        "psf_match": "not_point_like",
        "centred": "off_centre_residual",
        "significance": "low_significance",
    }.get(worst, "unknown")
