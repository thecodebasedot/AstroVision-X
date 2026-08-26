"""Intensity scaling.

Astronomical dynamic range spans many orders of magnitude, so both display
and neural-network input need a nonlinear stretch.  These transforms are
the standard ones from DS9/astropy visualisation, implemented in NumPy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.numeric import as_float_image, sigma_clipped_stats


def zscale_limits(image: np.ndarray, contrast: float = 0.25,
                  n_samples: int = 6000, krej: float = 2.5,
                  max_iterations: int = 5) -> Tuple[float, float]:
    """IRAF ``zscale`` limits -- the display stretch astronomers expect.

    A sorted sample of pixels is fitted with an iteratively-clipped line;
    its slope sets the contrast so faint structure stays visible while
    bright sources do not wash the image out.
    """
    data = as_float_image(image)
    values = data[np.isfinite(data)]
    if values.size == 0:
        return 0.0, 1.0
    if values.size > n_samples:
        step = max(1, values.size // n_samples)
        values = values[::step]
    values = np.sort(values)
    npix = values.size
    if npix < 5:
        return float(values.min()), float(values.max())

    midpoint = npix // 2
    x = np.arange(npix, dtype=float) - midpoint
    mask = np.ones(npix, dtype=bool)
    slope, intercept = 0.0, float(values[midpoint])
    for _ in range(max_iterations):
        if mask.sum() < 5:
            break
        fit = np.polyfit(x[mask], values[mask], 1)
        slope, intercept = float(fit[0]), float(fit[1])
        residual = values - (slope * x + intercept)
        sigma = float(np.std(residual[mask]))
        if sigma <= 0:
            break
        new_mask = np.abs(residual) < krej * sigma
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    if contrast > 0:
        slope = slope / contrast
    median = float(np.median(values))
    z1 = max(float(values[0]), median + slope * (0 - midpoint))
    z2 = min(float(values[-1]), median + slope * (npix - 1 - midpoint))
    if z2 <= z1:
        return float(values[0]), float(values[-1]) if values[-1] > values[0] else float(values[0]) + 1.0
    return z1, z2


def zscale(image: np.ndarray, contrast: float = 0.25) -> np.ndarray:
    """Rescale to ``[0, 1]`` using zscale limits."""
    data = as_float_image(image)
    z1, z2 = zscale_limits(data, contrast)
    return np.clip((data - z1) / max(z2 - z1, 1e-12), 0.0, 1.0)


def asinh_stretch(image: np.ndarray, softening: Optional[float] = None,
                  percentile: float = 99.5) -> np.ndarray:
    """Inverse-hyperbolic-sine stretch (Lupton et al. 2004).

    Linear near the noise, logarithmic for bright pixels -- the best
    general-purpose stretch for feeding images into a CNN.
    """
    data = as_float_image(image)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    _, median, std = sigma_clipped_stats(finite)
    beta = float(softening) if softening else max(2.0 * std, 1e-9)
    stretched = np.arcsinh((data - median) / beta)
    hi = float(np.percentile(stretched[np.isfinite(stretched)], percentile))
    lo = float(np.percentile(stretched[np.isfinite(stretched)], 100 - percentile))
    return np.clip((stretched - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def percentile_stretch(image: np.ndarray, low: float = 1.0,
                       high: float = 99.5) -> np.ndarray:
    """Linear stretch between two percentiles."""
    data = as_float_image(image)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((data - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def zscore(image: np.ndarray) -> np.ndarray:
    """Standardise to zero median and unit robust sigma (model input)."""
    data = as_float_image(image)
    _, median, std = sigma_clipped_stats(data)
    return (data - median) / max(std, 1e-9)


def log_stretch(image: np.ndarray, a: float = 1000.0) -> np.ndarray:
    """Logarithmic stretch of the ``[0, 1]``-rescaled image."""
    scaled = percentile_stretch(image, 0.5, 99.9)
    return np.log(a * scaled + 1.0) / np.log(a + 1.0)


#: Name -> transform, used by :func:`normalize`.
TRANSFORMS = {
    "zscale": zscale,
    "asinh": asinh_stretch,
    "percentile": percentile_stretch,
    "zscore": zscore,
    "log": log_stretch,
    "none": lambda image: as_float_image(image),
}


def normalize(image: np.ndarray, method: str = "zscale") -> np.ndarray:
    """Apply a named normalisation transform."""
    key = str(method or "none").lower()
    if key not in TRANSFORMS:
        raise ValueError(f"unknown normalization '{method}'; "
                         f"choose from {', '.join(sorted(TRANSFORMS))}")
    return TRANSFORMS[key](image)
