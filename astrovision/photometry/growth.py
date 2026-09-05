"""Curve-of-growth analysis: Kron and Petrosian radii.

A fixed aperture under-counts big galaxies and over-counts noise for small
ones.  Kron and Petrosian radii scale the aperture to each object's own
light profile, which is what makes photometry comparable across a survey.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, safe_divide

log = get_logger("photometry.growth")


def curve_of_growth(image: np.ndarray, centre: Tuple[float, float],
                    radii: Sequence[float], background: float = 0.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Enclosed flux versus radius; returns ``(radii, cumulative_flux)``."""
    from .aperture import circular_aperture_block, stamp_box

    data = as_float_image(image)
    reach = float(max(radii)) + 3.0 if len(radii) else 3.0
    rows, cols, local = stamp_box(data.shape, centre, reach)
    block = np.nan_to_num(data[rows, cols], nan=0.0) - float(background)
    fluxes = []
    for radius in radii:
        weights, (y0, y1, x0, x1) = circular_aperture_block(block.shape, local, radius,
                                                            subpixels=4)
        fluxes.append(float((block[y0:y1, x0:x1] * weights).sum()))
    return np.asarray(radii, dtype=float), np.asarray(fluxes, dtype=float)


def kron_radius(image: np.ndarray, centre: Tuple[float, float],
                mask: Optional[np.ndarray] = None, max_radius: float = 40.0,
                background: float = 0.0) -> float:
    """First-moment (Kron) radius ``R_1 = sum(r*I) / sum(I)``.

    An aperture of ``2.5 * R_1`` captures roughly 94 % of the light of a
    typical galaxy regardless of its profile shape -- the basis of
    SExtractor's ``MAG_AUTO``.
    """
    from .aperture import stamp_box

    data = as_float_image(image)
    rows, cols, local = stamp_box(data.shape, centre, float(max_radius) + 1.0)
    block = np.nan_to_num(data[rows, cols], nan=0.0) - float(background)
    yy, xx = np.mgrid[0:block.shape[0], 0:block.shape[1]]
    r = np.hypot(xx - local[0], yy - local[1])
    within = r <= float(max_radius)
    if mask is not None:
        within &= np.asarray(mask, dtype=bool)[rows, cols]
    weights = np.clip(block, 0, None) * within
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float((weights * r).sum() / total)


def petrosian_radius(image: np.ndarray, centre: Tuple[float, float], eta: float = 0.2,
                     max_radius: float = 40.0, n_steps: int = 60,
                     background: float = 0.0) -> float:
    """Radius where local surface brightness falls to ``eta`` of the mean inside it.

    The Petrosian radius is independent of distance and of the depth of
    the image, which is why large surveys use it for galaxy sizes.
    """
    from .aperture import stamp_box

    rows, cols, local = stamp_box(as_float_image(image).shape, centre,
                                  1.1 * float(max_radius) + 1.0)
    data = np.nan_to_num(as_float_image(image)[rows, cols], nan=0.0) - float(background)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    r = np.hypot(xx - local[0], yy - local[1])
    edges = np.linspace(0.5, float(max_radius), int(n_steps))

    ratios = []
    for radius in edges:
        annulus = (r >= radius * 0.9) & (r < radius * 1.1)
        inside = r < radius
        if annulus.sum() < 3 or inside.sum() < 3:
            ratios.append(np.nan)
            continue
        local = float(data[annulus].mean())
        mean_inside = float(data[inside].mean())
        ratios.append(float(safe_divide(local, mean_inside, fill=np.nan)))

    ratios = np.asarray(ratios, dtype=float)
    finite = np.isfinite(ratios)
    if finite.sum() < 3:
        return float("nan")
    below = np.nonzero(finite & (ratios <= float(eta)))[0]
    if below.size == 0:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return float(edges[0])
    # Linear interpolation onto the eta crossing.
    r0, r1 = edges[index - 1], edges[index]
    v0, v1 = ratios[index - 1], ratios[index]
    if not np.isfinite(v0) or abs(v1 - v0) < 1e-12:
        return float(r1)
    return float(r0 + (v0 - eta) / (v0 - v1) * (r1 - r0))


def flux_radius(radii: np.ndarray, cumulative: np.ndarray,
                fraction: float = 0.5) -> float:
    """Radius enclosing ``fraction`` of the total measured flux."""
    radii = np.asarray(radii, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    if radii.size == 0 or cumulative.size == 0:
        return float("nan")
    total = float(cumulative[-1])
    if total <= 0:
        return float("nan")
    target = fraction * total
    index = int(np.searchsorted(cumulative, target))
    if index <= 0:
        return float(radii[0])
    if index >= len(radii):
        return float(radii[-1])
    c0, c1 = cumulative[index - 1], cumulative[index]
    if abs(c1 - c0) < 1e-12:
        return float(radii[index])
    return float(radii[index - 1] + (target - c0) / (c1 - c0) * (radii[index] - radii[index - 1]))


def concentration_index(radii: np.ndarray, cumulative: np.ndarray,
                        inner: float = 0.2, outer: float = 0.8) -> float:
    """``5 log10(r_outer / r_inner)`` -- the standard concentration statistic.

    Around 5 for a de Vaucouleurs bulge, around 2.7 for an exponential
    disc, so it separates early- from late-type galaxies on its own.
    """
    r_inner = flux_radius(radii, cumulative, inner)
    r_outer = flux_radius(radii, cumulative, outer)
    if not (np.isfinite(r_inner) and np.isfinite(r_outer)) or r_inner <= 0:
        return float("nan")
    return float(5.0 * np.log10(r_outer / r_inner))


def auto_aperture(image: np.ndarray, centre: Tuple[float, float],
                  mask: Optional[np.ndarray] = None, kron_factor: float = 2.5,
                  min_radius: float = 2.0, max_radius: float = 40.0,
                  background: float = 0.0) -> Dict[str, float]:
    """Pick an adaptive aperture and report the radii behind the choice."""
    r_kron = kron_radius(image, centre, mask, max_radius, background)
    r_petro = petrosian_radius(image, centre, 0.2, max_radius, background=background)
    if np.isfinite(r_kron):
        radius = float(np.clip(kron_factor * r_kron, min_radius, max_radius))
    elif np.isfinite(r_petro):
        radius = float(np.clip(2.0 * r_petro, min_radius, max_radius))
    else:
        radius = float(min_radius)
    return {"radius": radius, "kron_radius": float(r_kron),
            "petrosian_radius": float(r_petro), "kron_factor": float(kron_factor)}
