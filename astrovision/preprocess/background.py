"""Sky-background estimation.

Astronomical images sit on a spatially varying sky.  Detection thresholds
are only meaningful relative to that sky and its noise, so the pipeline
builds a smooth background model plus an RMS map before anything else.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    MAD_TO_SIGMA,
    as_float_image,
    bilinear_resize,
    median_filter,
    sigma_clipped_stats,
)

log = get_logger("preprocess.background")


def mode_estimate(values: np.ndarray) -> float:
    """SExtractor-style mode: ``2.5*median - 1.5*mean`` after clipping.

    In crowded fields the mean is pulled up by sources; this estimator is
    much closer to the true sky level, falling back to the median when the
    distribution is not skewed.
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    if std <= 0:
        return float(median)
    if abs(mean - median) / std > 0.3:
        return float(median)
    return float(2.5 * median - 1.5 * mean)


def background_mesh(image: np.ndarray, box_size: int = 64,
                    filter_size: int = 3,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate background and RMS on a coarse mesh.

    Returns full-resolution ``(background, rms)`` maps obtained by
    median-filtering the mesh and interpolating it back up -- the standard
    approach used by SExtractor and photutils.
    """
    data = as_float_image(image)
    ny, nx = data.shape
    box = max(8, int(box_size))
    n_boxes_y = max(1, ny // box)
    n_boxes_x = max(1, nx // box)

    valid = np.isfinite(data)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    mesh_bg = np.zeros((n_boxes_y, n_boxes_x), dtype=float)
    mesh_rms = np.zeros((n_boxes_y, n_boxes_x), dtype=float)
    edges_y = np.linspace(0, ny, n_boxes_y + 1).astype(int)
    edges_x = np.linspace(0, nx, n_boxes_x + 1).astype(int)

    for j in range(n_boxes_y):
        for i in range(n_boxes_x):
            cell = data[edges_y[j]:edges_y[j + 1], edges_x[i]:edges_x[i + 1]]
            cell_valid = valid[edges_y[j]:edges_y[j + 1], edges_x[i]:edges_x[i + 1]]
            values = cell[cell_valid]
            if values.size < 4:
                mesh_bg[j, i] = np.nan
                mesh_rms[j, i] = np.nan
                continue
            mesh_bg[j, i] = mode_estimate(values)
            clipped_median = np.median(values)
            mesh_rms[j, i] = float(
                MAD_TO_SIGMA * np.median(np.abs(values - clipped_median)))

    mesh_bg = _fill_nan(mesh_bg)
    mesh_rms = _fill_nan(mesh_rms)

    if filter_size and filter_size > 1 and min(mesh_bg.shape) >= filter_size:
        mesh_bg = median_filter(mesh_bg, int(filter_size))
        mesh_rms = median_filter(mesh_rms, int(filter_size))

    background = bilinear_resize(mesh_bg, (ny, nx))
    rms = bilinear_resize(mesh_rms, (ny, nx))
    rms = np.clip(rms, 1e-9, None)
    log.debug("background mesh %dx%d: median=%.4g rms=%.4g",
              n_boxes_x, n_boxes_y, float(np.median(background)), float(np.median(rms)))
    return background, rms


def _fill_nan(mesh: np.ndarray) -> np.ndarray:
    """Replace NaN mesh cells with the global median of the valid cells."""
    out = np.asarray(mesh, dtype=float).copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros_like(out)
    if bad.any():
        out[bad] = float(np.median(out[~bad]))
    return out


def global_background(image: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Single-value sky level and noise for small or flat images."""
    data = as_float_image(image)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    values = data[valid]
    if values.size == 0:
        return 0.0, 1.0
    level = mode_estimate(values)
    _, median, std = sigma_clipped_stats(values)
    return float(level), float(max(std, 1e-9))


def estimate_background(image: np.ndarray, box_size: int = 64, filter_size: int = 3,
                        mask: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Background/RMS maps, using a mesh when the image is large enough."""
    data = as_float_image(image)
    if min(data.shape) < 2 * max(8, box_size):
        level, rms = global_background(data, mask)
        return (np.full(data.shape, level, dtype=float),
                np.full(data.shape, rms, dtype=float))
    return background_mesh(data, box_size, filter_size, mask)


def subtract_background(image: np.ndarray, box_size: int = 64, filter_size: int = 3,
                        mask: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(subtracted, background, rms)``."""
    background, rms = estimate_background(image, box_size, filter_size, mask)
    return as_float_image(image) - background, background, rms
