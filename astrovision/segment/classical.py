"""Classical segmentation: watershed splitting and isophotal contours."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.numeric import as_float_image, gaussian_filter, maximum_filter

log = get_logger("segment.classical")


def find_peaks(image: np.ndarray, min_distance: int = 3,
               threshold: Optional[float] = None,
               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Local maxima, used as watershed seeds. Returns an ``(N, 2)`` ``(y, x)`` array."""
    data = as_float_image(image)
    size = max(3, 2 * int(min_distance) + 1)
    peaks = data >= maximum_filter(data, size)
    if threshold is not None:
        peaks &= data > float(threshold)
    if mask is not None:
        peaks &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(peaks)
    return np.column_stack([ys, xs])


def watershed(image: np.ndarray, markers: np.ndarray,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Marker-controlled watershed on the *inverted* intensity surface.

    Bright peaks become basins, so overlapping sources are split along the
    saddle between them.  SciPy/scikit-image are used when present; the
    fallback is a priority-flood implementation with identical semantics.
    """
    data = as_float_image(image)
    seeds = np.asarray(markers, dtype=np.int32)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    skimage_seg = try_import("skimage.segmentation")
    if skimage_seg is not None:
        return skimage_seg.watershed(-data, seeds, mask=footprint).astype(np.int32)
    return _priority_flood(data, seeds, footprint)


def _priority_flood(data: np.ndarray, seeds: np.ndarray,
                    footprint: np.ndarray) -> np.ndarray:
    """Watershed by flooding from the brightest seed pixels outwards."""
    import heapq

    ny, nx = data.shape
    labels = np.where(footprint, seeds, 0).astype(np.int32)
    queue: List[Tuple[float, int, int, int]] = []
    ys, xs = np.nonzero(labels > 0)
    for y, x in zip(ys, xs):
        heapq.heappush(queue, (-float(data[y, x]), int(y), int(x), int(labels[y, x])))

    while queue:
        _, y, x, value = heapq.heappop(queue)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny_, nx_ = y + dy, x + dx
            if not (0 <= ny_ < ny and 0 <= nx_ < nx):
                continue
            if labels[ny_, nx_] != 0 or not footprint[ny_, nx_]:
                continue
            labels[ny_, nx_] = value
            heapq.heappush(queue, (-float(data[ny_, nx_]), ny_, nx_, value))
    return labels


def watershed_split(image: np.ndarray, mask: np.ndarray, smooth: float = 1.0,
                    min_distance: int = 3, min_peak_fraction: float = 0.05
                    ) -> np.ndarray:
    """Split one footprint into basins around its significant peaks."""
    data = as_float_image(image)
    footprint = np.asarray(mask, dtype=bool)
    if not footprint.any():
        return np.zeros(data.shape, dtype=np.int32)

    smoothed = gaussian_filter(data, smooth) if smooth > 0 else data
    peak_floor = float(np.nanmax(smoothed[footprint])) * float(min_peak_fraction)
    peaks = find_peaks(smoothed, min_distance, peak_floor, footprint)
    if len(peaks) <= 1:
        return footprint.astype(np.int32)

    seeds = np.zeros(data.shape, dtype=np.int32)
    for index, (y, x) in enumerate(peaks, start=1):
        seeds[y, x] = index
    return watershed(smoothed, seeds, footprint)


def isophotal_contours(image: np.ndarray, mask: np.ndarray, n_levels: int = 5,
                       base: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Nested isophotes of one object, brightest level first.

    Isophotes are how galaxy structure has been measured since photographic
    plates: the shape and spacing of successive light contours encode the
    bulge-to-disc ratio and any bar or interaction.
    """
    data = as_float_image(image)
    footprint = np.asarray(mask, dtype=bool)
    if not footprint.any():
        return {"levels": np.array([]), "masks": np.empty((0,) + data.shape, dtype=bool)}

    values = data[footprint]
    peak = float(np.nanmax(values))
    floor = float(base) if base is not None else float(np.nanpercentile(values, 5))
    if peak <= floor:
        return {"levels": np.array([floor]), "masks": footprint[None, ...]}

    levels = np.linspace(peak, floor, int(n_levels) + 1)[:-1]
    masks = np.stack([footprint & (data >= level) for level in levels])
    return {"levels": levels, "masks": masks}


def segment_object(image: np.ndarray, mask: np.ndarray, split: bool = True,
                   smooth: float = 1.0) -> np.ndarray:
    """Segment a single object footprint, optionally splitting blends."""
    if not split:
        return np.asarray(mask, dtype=bool).astype(np.int32)
    return watershed_split(image, mask, smooth)
