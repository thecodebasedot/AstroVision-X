"""Multi-threshold deblending.

Two galaxies that overlap on the sky are detected as a single connected
region.  Following SExtractor, the region is re-thresholded at a series of
exponentially spaced levels: branches that split off and carry more than
``contrast`` of the total flux become separate objects.  Remaining pixels
are then reassigned to whichever branch they most plausibly belong to.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image
from .labeling import label

log = get_logger("detect.deblend")


def deblend_segment(data: np.ndarray, mask: np.ndarray, threshold: float,
                    n_levels: int = 32, contrast: float = 0.005,
                    min_area: int = 3) -> np.ndarray:
    """Split one detected region into components.

    ``data`` and ``mask`` are the (background-subtracted) cutout and the
    detection footprint within it.  Returns an integer label image over
    the same cutout, with 1..K marking the deblended children.
    """
    values = as_float_image(data)
    footprint = np.asarray(mask, dtype=bool)
    if not footprint.any():
        return np.zeros(values.shape, dtype=np.int32)

    peak = float(np.nanmax(values[footprint]))
    base = float(threshold)
    if not np.isfinite(peak) or peak <= base:
        return footprint.astype(np.int32)

    total_flux = float(np.nansum(np.clip(values[footprint] - base, 0, None)))
    if total_flux <= 0:
        return footprint.astype(np.int32)

    # Exponentially spaced levels resolve faint companions near the base
    # and still separate bright cores near the peak.
    levels = base * np.power(peak / base, np.linspace(0, 1, max(2, int(n_levels)))) \
        if base > 0 else np.linspace(base, peak, max(2, int(n_levels)))
    levels = levels[:-1]  # the top level contains only the peak pixel

    # Track branches: a branch is a component that persists as the
    # threshold rises and is significant on its own.
    cores: List[np.ndarray] = []
    for level in levels[1:]:
        above = footprint & (values > level)
        if not above.any():
            break
        labels, count = label(above)
        if count < 2:
            continue
        # Each component at this level that splits from its parent and is
        # bright enough becomes a candidate core.
        for value in range(1, count + 1):
            component = labels == value
            if component.sum() < min_area:
                continue
            flux = float(np.nansum(np.clip(values[component] - base, 0, None)))
            if flux < contrast * total_flux:
                continue
            # Keep the deepest (largest) version of each distinct core.
            merged = False
            for i, core in enumerate(cores):
                if (core & component).any():
                    if component.sum() > core.sum() and _same_peak(values, core, component):
                        cores[i] = component
                    merged = True
                    break
            if not merged:
                cores.append(component)

    cores = _prune_nested(cores, values, base, contrast, total_flux)
    if len(cores) < 2:
        return footprint.astype(np.int32)

    out = np.zeros(values.shape, dtype=np.int32)
    for index, core in enumerate(cores, start=1):
        out[core] = index
    remaining = footprint & (out == 0)
    if remaining.any():
        out = _assign_remaining(values, out, remaining, cores)
    log.debug("deblended segment into %d components", len(cores))
    return out


def _same_peak(values: np.ndarray, core: np.ndarray, component: np.ndarray) -> bool:
    """True when two footprints share their brightest pixel."""
    core_peak = np.unravel_index(int(np.argmax(np.where(core, values, -np.inf))), values.shape)
    comp_peak = np.unravel_index(int(np.argmax(np.where(component, values, -np.inf))), values.shape)
    return core_peak == comp_peak


def _prune_nested(cores: List[np.ndarray], values: np.ndarray, base: float,
                  contrast: float, total_flux: float) -> List[np.ndarray]:
    """Drop cores fully contained in another, and cores that are too faint."""
    kept: List[np.ndarray] = []
    for core in sorted(cores, key=lambda c: -float(np.nansum(np.clip(values[c] - base, 0, None)))):
        flux = float(np.nansum(np.clip(values[core] - base, 0, None)))
        if flux < contrast * total_flux:
            continue
        if any(_same_peak(values, other, core) for other in kept):
            continue
        kept.append(core)
    return kept


def _assign_remaining(values: np.ndarray, labels: np.ndarray, remaining: np.ndarray,
                      cores: List[np.ndarray]) -> np.ndarray:
    """Assign leftover pixels to the core with the highest expected flux.

    Each core is approximated by a Gaussian in distance from its centroid,
    scaled by its flux; a pixel joins whichever core dominates there.  This
    mirrors SExtractor's bivariate-Gaussian reassignment.
    """
    out = labels.copy()
    ys, xs = np.nonzero(remaining)
    if ys.size == 0:
        return out

    weights = []
    for core in cores:
        cy, cx = np.nonzero(core)
        flux = float(np.nansum(np.clip(values[core], 0, None)))
        centre_y = float(cy.mean())
        centre_x = float(cx.mean())
        # Effective radius from the core area keeps the kernel scale sane.
        sigma = max(np.sqrt(core.sum() / np.pi), 1.0)
        weights.append((centre_x, centre_y, max(flux, 1e-9), sigma))

    scores = np.empty((len(cores), ys.size), dtype=float)
    for i, (cx, cy, flux, sigma) in enumerate(weights):
        r2 = (xs - cx) ** 2 + (ys - cy) ** 2
        scores[i] = np.log(flux) - 0.5 * r2 / (sigma ** 2)
    out[ys, xs] = np.argmax(scores, axis=0) + 1
    return out


def deblend_all(data: np.ndarray, segmentation: np.ndarray, threshold: np.ndarray,
                n_levels: int = 32, contrast: float = 0.005,
                min_area: int = 3) -> Tuple[np.ndarray, int]:
    """Deblend every segment of a segmentation map.

    Returns a new segmentation image and the total number of objects.
    """
    values = as_float_image(data)
    segments = np.asarray(segmentation, dtype=np.int32)
    n_input = int(segments.max())
    if n_input == 0:
        return segments, 0
    thresholds = (np.asarray(threshold, dtype=float)
                  if np.ndim(threshold) else np.full(values.shape, float(threshold)))

    out = np.zeros_like(segments)
    next_label = 0
    for value in range(1, n_input + 1):
        footprint = segments == value
        if not footprint.any():
            continue
        ys, xs = np.nonzero(footprint)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cut_values = values[y0:y1, x0:x1]
        cut_mask = footprint[y0:y1, x0:x1]
        level = float(np.median(thresholds[y0:y1, x0:x1][cut_mask]))
        children = deblend_segment(cut_values, cut_mask, level, n_levels, contrast, min_area)
        n_children = int(children.max())
        if n_children == 0:
            continue
        region = out[y0:y1, x0:x1]
        region[children > 0] = children[children > 0] + next_label
        out[y0:y1, x0:x1] = region
        next_label += n_children

    log.debug("deblending expanded %d segments into %d objects", n_input, next_label)
    return out, next_label
