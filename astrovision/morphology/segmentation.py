"""Object footprints for morphological measurement.

Non-parametric statistics are only comparable between objects if they are
measured over comparably-defined pixel sets.  Lotz et al. (2004) define
that set by smoothing the image and thresholding at the mean surface
brightness at the Petrosian radius -- a definition that scales with each
galaxy instead of with the depth of the exposure, which is what a raw
isophotal detection footprint does.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, gaussian_filter, nan_to_finite
from ..detect.labeling import label
from ..photometry.growth import petrosian_radius

log = get_logger("morphology.segmentation")


def petrosian_segmentation(cutout: np.ndarray, centre: Tuple[float, float],
                           r_petrosian: Optional[float] = None,
                           smoothing_fraction: float = 0.2,
                           fallback: Optional[np.ndarray] = None
                           ) -> Optional[np.ndarray]:
    """Lotz-style morphology footprint for one object.

    Returns a boolean mask of the connected region containing ``centre``,
    or ``None`` when the object is too small or the threshold degenerate.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if min(ny, nx) < 8:
        return fallback

    r_petro = r_petrosian
    if r_petro is None or not np.isfinite(r_petro) or r_petro <= 1.0:
        r_petro = petrosian_radius(data, centre, 0.2,
                                   max_radius=min(ny, nx) / 2.0 - 1.0)
    if not np.isfinite(r_petro) or r_petro <= 1.0:
        return fallback
    r_petro = float(np.clip(r_petro, 1.5, min(ny, nx) / 2.0 - 1.0))

    smoothed = gaussian_filter(data, max(smoothing_fraction * r_petro, 0.6))

    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(xx - centre[0], yy - centre[1])
    annulus = (radius >= 0.85 * r_petro) & (radius <= 1.15 * r_petro)
    if annulus.sum() < 6:
        return fallback
    threshold = float(np.mean(smoothed[annulus]))
    if not np.isfinite(threshold):
        return fallback

    above = smoothed >= threshold
    if not above.any():
        return fallback

    labels, count = label(above)
    if count == 0:
        return fallback
    # Keep only the component the object actually sits in, so a nearby
    # neighbour above the same threshold does not join the footprint.
    cy = int(np.clip(round(centre[1]), 0, ny - 1))
    cx = int(np.clip(round(centre[0]), 0, nx - 1))
    value = int(labels[cy, cx])
    if value == 0:
        window = labels[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
        nonzero = window[window > 0]
        if nonzero.size == 0:
            return fallback
        value = int(np.bincount(nonzero.ravel()).argmax())
    return labels == value


def dilate_footprint(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Grow a footprint slightly, to capture faint outer light."""
    from ..detect.labeling import binary_dilate
    return binary_dilate(mask, iterations)
