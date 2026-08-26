"""Gini coefficient and M20: light distribution without assuming a centre.

The Gini coefficient measures how unequally light is distributed among an
object's pixels, and M20 measures how far the brightest 20 % of the light
sits from the centre.  Together they identify mergers and double nuclei
that CAS alone can miss, since neither assumes circular symmetry.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.numeric import as_float_image, nan_to_finite


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a set of pixel values, in ``[0, 1]``.

    0 means every pixel is equally bright; 1 means all light is in one
    pixel.  Ellipticals sit around 0.55-0.65, discs lower.
    """
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size < 2:
        return float("nan")
    absolute = np.sort(np.abs(data))
    n = absolute.size
    total = absolute.sum()
    if total <= 0:
        return float("nan")
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * absolute).sum() / (total * (n - 1)))


def m20(cutout: np.ndarray, mask: Optional[np.ndarray] = None,
        centre: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Second-order moment of the brightest 20 % of an object's light.

    Defined as ``log10(sum(M_i) / M_total)`` over the pixels that make up
    the brightest fifth of the flux.  Values near -1.6 are typical of
    single-nucleus galaxies; values above about -1.1 flag double nuclei.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    values = np.where(footprint, np.clip(data, 0, None), 0.0)
    total_flux = float(values.sum())
    if total_flux <= 0 or footprint.sum() < 4:
        return {"m20": float("nan"), "total_moment": float("nan")}

    yy, xx = np.mgrid[0:ny, 0:nx]
    if centre is None:
        cx = float((values * xx).sum() / total_flux)
        cy = float((values * yy).sum() / total_flux)
        # The canonical definition minimises the total moment over centre;
        # a short local search does that cheaply.
        for step in (1.0, 0.5):
            improved = True
            while improved:
                improved = False
                base = float((values * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum())
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    candidate = float((values * ((xx - cx - dx) ** 2 +
                                                 (yy - cy - dy) ** 2)).sum())
                    if candidate < base - 1e-9:
                        cx, cy, base, improved = cx + dx, cy + dy, candidate, True
                        break
    else:
        cx, cy = float(centre[0]), float(centre[1])

    moments = values * ((xx - cx) ** 2 + (yy - cy) ** 2)
    total_moment = float(moments.sum())
    if total_moment <= 0:
        return {"m20": float("nan"), "total_moment": 0.0}

    flat_values = values.ravel()
    order = np.argsort(flat_values)[::-1]
    cumulative = np.cumsum(flat_values[order])
    cutoff = int(np.searchsorted(cumulative, 0.2 * total_flux)) + 1
    brightest = order[:max(cutoff, 1)]
    bright_moment = float(moments.ravel()[brightest].sum())
    if bright_moment <= 0:
        return {"m20": float("nan"), "total_moment": total_moment}
    return {"m20": float(np.log10(bright_moment / total_moment)),
            "total_moment": total_moment,
            "m20_centre_x": cx, "m20_centre_y": cy}


def gini_m20(cutout: np.ndarray, mask: Optional[np.ndarray] = None,
             segment_threshold: Optional[float] = None) -> Dict[str, float]:
    """Both statistics on the same pixel set.

    Gini is computed over the segmentation footprint, which is what makes
    it comparable between objects of different sizes.
    """
    data = as_float_image(cutout)
    if mask is None:
        if segment_threshold is None:
            finite = data[np.isfinite(data)]
            segment_threshold = float(np.percentile(finite, 80)) if finite.size else 0.0
        mask = data > segment_threshold
    footprint = np.asarray(mask, dtype=bool)
    result = {"gini": gini_coefficient(data[footprint])}
    result.update(m20(data, footprint))
    return result


def merger_statistic(gini: float, m20_value: float) -> float:
    """Distance above the Lotz et al. (2008) merger separation line.

    Objects with ``G > -0.14 * M20 + 0.33`` are merger candidates; the
    returned value is how far above that line the object sits.
    """
    if not (np.isfinite(gini) and np.isfinite(m20_value)):
        return float("nan")
    return float(gini - (-0.14 * m20_value + 0.33))


def bulge_statistic(gini: float, m20_value: float) -> float:
    """Distance above the early/late-type separation line (Lotz 2008).

    ``G > 0.14 * M20 + 0.80`` marks bulge-dominated (early-type) systems.
    """
    if not (np.isfinite(gini) and np.isfinite(m20_value)):
        return float("nan")
    return float(gini - (0.14 * m20_value + 0.80))
