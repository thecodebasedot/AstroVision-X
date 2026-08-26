"""The CAS system: Concentration, Asymmetry and Smoothness.

Conselice (2003) showed that these three numbers separate ellipticals,
discs and mergers without any model fitting, which makes them the standard
non-parametric morphology basis for large surveys.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.numeric import as_float_image, gaussian_filter, nan_to_finite
from ..photometry.growth import concentration_index, curve_of_growth, flux_radius


def concentration(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                  max_radius: Optional[float] = None,
                  background: float = 0.0) -> Dict[str, float]:
    """``C = 5 log10(r80 / r20)``: how centrally concentrated the light is.

    Roughly 5 for a de Vaucouleurs bulge, 2.7 for an exponential disc.
    """
    data = as_float_image(cutout)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    if max_radius is None:
        max_radius = min(ny, nx) / 2.0 - 1.0
    radii = np.linspace(0.75, max(float(max_radius), 2.0), 30)
    r, cumulative = curve_of_growth(data, centre, radii, background)
    return {
        "concentration": float(concentration_index(r, cumulative)),
        "r20": float(flux_radius(r, cumulative, 0.2)),
        "r50": float(flux_radius(r, cumulative, 0.5)),
        "r80": float(flux_radius(r, cumulative, 0.8)),
    }


def asymmetry(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
              mask: Optional[np.ndarray] = None, background_asymmetry: float = 0.0,
              refine_centre: bool = True) -> Dict[str, float]:
    """``A = sum|I - I_180| / (2 sum|I|)``: rotational asymmetry.

    Mergers and disturbed systems have high asymmetry; relaxed ellipticals
    are close to zero.  The rotation centre is refined by minimising A,
    because the result is very sensitive to a mis-centred rotation.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    def measure(cx: float, cy: float) -> float:
        rotated = _rotate180(data, (cx, cy))
        residual = np.abs(data - rotated)[footprint]
        total = np.abs(data)[footprint].sum()
        if total <= 0:
            return float("nan")
        return float(residual.sum() / (2.0 * total))

    best_centre = (float(centre[0]), float(centre[1]))
    best = measure(*best_centre)
    if refine_centre:
        # A small local search is enough: A is smooth near its minimum.
        for step in (1.0, 0.5, 0.25):
            improved = True
            while improved:
                improved = False
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    candidate = (best_centre[0] + dx, best_centre[1] + dy)
                    value = measure(*candidate)
                    if np.isfinite(value) and value < best - 1e-6:
                        best, best_centre, improved = value, candidate, True
    corrected = best - float(background_asymmetry)
    return {"asymmetry": float(corrected), "asymmetry_raw": float(best),
            "asymmetry_centre_x": best_centre[0], "asymmetry_centre_y": best_centre[1]}


def _rotate180(data: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """Rotate by 180 degrees about an arbitrary (sub-pixel) centre."""
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    src_x = 2.0 * centre[0] - xx
    src_y = 2.0 * centre[1] - yy
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    wx, wy = src_x - x0, src_y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0c = np.clip(x0, 0, nx - 2)
    y0c = np.clip(y0, 0, ny - 2)
    top = data[y0c, x0c] * (1 - wx) + data[y0c, x0c + 1] * wx
    bottom = data[y0c + 1, x0c] * (1 - wx) + data[y0c + 1, x0c + 1] * wx
    return np.where(inside, top * (1 - wy) + bottom * wy, 0.0)


def smoothness(cutout: np.ndarray, smoothing_radius: Optional[float] = None,
               centre: Optional[Tuple[float, float]] = None,
               mask: Optional[np.ndarray] = None,
               inner_exclusion: float = 0.25) -> Dict[str, float]:
    """``S``: the fraction of light in high-spatial-frequency structure.

    Star-forming clumps and spiral arms raise S; smooth ellipticals sit
    near zero.  The nucleus is excluded because it is intrinsically sharp.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    if smoothing_radius is None:
        smoothing_radius = max(min(ny, nx) * 0.05, 1.0)

    blurred = gaussian_filter(data, float(smoothing_radius))
    residual = data - blurred

    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - centre[0], yy - centre[1])
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    outer = footprint & (r > inner_exclusion * min(ny, nx) / 2.0)
    if outer.sum() < 8:
        outer = footprint

    total = np.abs(data)[outer].sum()
    if total <= 0:
        return {"smoothness": float("nan"), "clumpiness": float("nan")}
    value = float(np.abs(residual)[outer].sum() / total)
    positive = float(np.clip(residual, 0, None)[outer].sum() / total)
    return {"smoothness": value, "clumpiness": positive}


def cas_statistics(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                   mask: Optional[np.ndarray] = None,
                   background: float = 0.0) -> Dict[str, float]:
    """Compute all three CAS statistics for one object."""
    result: Dict[str, float] = {}
    result.update(concentration(cutout, centre, background=background))
    result.update(asymmetry(cutout, centre, mask))
    result.update(smoothness(cutout, centre=centre, mask=mask))
    return result
