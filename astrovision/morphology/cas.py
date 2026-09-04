"""The CAS system: Concentration, Asymmetry and Smoothness.

Conselice (2003) showed that these three numbers separate ellipticals,
discs and mergers without any model fitting, which makes them the standard
non-parametric morphology basis for large surveys.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.numeric import as_float_image, convolve, nan_to_finite
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


def _sky_region(shape: Tuple[int, int], mask: Optional[np.ndarray],
                sky_mask: Optional[np.ndarray]) -> np.ndarray:
    """Pixels that are sky: given, else outside the footprint, else the
    outer border of the cutout (the only place sky can be when there is no
    footprint at all)."""
    if sky_mask is not None:
        return np.asarray(sky_mask, dtype=bool)
    if mask is not None:
        return ~np.asarray(mask, dtype=bool)
    ny, nx = shape
    border = max(2, int(round(0.15 * min(ny, nx))))
    sky = np.zeros(shape, dtype=bool)
    sky[:border, :] = sky[-border:, :] = True
    sky[:, :border] = sky[:, -border:] = True
    return sky


def asymmetry(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
              mask: Optional[np.ndarray] = None, sky_mask: Optional[np.ndarray] = None,
              background_asymmetry: Optional[float] = None,
              refine_centre: bool = True) -> Dict[str, float]:
    """``A = sum|I - I_180| / sum|I| - A_sky``: rotational asymmetry.

    Mergers and disturbed systems have high asymmetry; relaxed ellipticals
    are close to zero. Two things make the number mean that rather than
    "how noisy was this stamp":

    * the rotation centre is refined by minimising A, because A is very
      sensitive to a mis-centred rotation;
    * the asymmetry of the *sky* is subtracted. Noise contributes
      ``|n - n'|`` to every pixel whichever way it goes, so an uncorrected A
      grows as S/N falls and ranks faint smooth galaxies above bright
      disturbed ones. ``A_sky`` is the mean ``|B - B_180|`` over sky pixels
      times the footprint's pixel count, over the same ``sum|I|`` -- the
      Conselice (2003) correction as statmorph applies it. Compared with
      statmorph on the same segments the uncorrected version had a rank
      correlation of -0.8 with theirs; see ``docs/validation.md``.

    The normalisation follows Lotz et al. (2004) and statmorph (no factor
    of two in the denominator), so the numbers are comparable with theirs.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    sky = _sky_region(data.shape, mask, sky_mask) & ~footprint if mask is not None \
        else _sky_region(data.shape, None, sky_mask)

    def measure(cx: float, cy: float) -> Tuple[float, float]:
        rotated = _rotate180(data, (cx, cy))
        residual = np.abs(data - rotated)
        total = np.abs(data)[footprint].sum()
        if total <= 0:
            return float("nan"), 0.0
        raw = float(residual[footprint].sum() / total)
        # Sky pixels whose rotation partner is also sky and inside the stamp.
        partner_is_sky = _rotate180(sky.astype(float), (cx, cy)) > 0.5
        pairs = sky & partner_is_sky
        if pairs.sum() >= 8:
            sky_term = float(residual[pairs].mean() * footprint.sum() / total)
        else:
            sky_term = 0.0
        return raw, sky_term

    best_centre = (float(centre[0]), float(centre[1]))
    best_raw, best_sky = measure(*best_centre)
    best = best_raw - best_sky
    if refine_centre:
        # A small local search is enough: A is smooth near its minimum.
        for step in (1.0, 0.5, 0.25):
            improved = True
            while improved:
                improved = False
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    candidate = (best_centre[0] + dx, best_centre[1] + dy)
                    raw, sky_term = measure(*candidate)
                    value = raw - sky_term
                    if np.isfinite(value) and value < best - 1e-6:
                        best, best_raw, best_sky, best_centre, improved = (
                            value, raw, sky_term, candidate, True)
    if background_asymmetry is not None:
        best = best_raw - float(background_asymmetry)
        best_sky = float(background_asymmetry)
    return {"asymmetry": float(best), "asymmetry_raw": float(best_raw),
            "asymmetry_sky": float(best_sky),
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


def _boxcar(data: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size) | 1)
    kernel = np.ones((size, size), dtype=float) / float(size * size)
    return convolve(data, kernel)


def smoothness(cutout: np.ndarray, smoothing_radius: Optional[float] = None,
               centre: Optional[Tuple[float, float]] = None,
               mask: Optional[np.ndarray] = None, sky_mask: Optional[np.ndarray] = None,
               petrosian_radius: Optional[float] = None,
               inner_exclusion: float = 0.25) -> Dict[str, float]:
    """``S = sum(I - I_S)_+ / sum I - S_sky``: light in small-scale structure.

    Star-forming clumps and spiral arms raise S; smooth ellipticals sit near
    zero. As in Lotz et al. (2004) and statmorph: ``I_S`` is a boxcar
    smoothing of width ``0.25 r_petro``; only positive residuals count
    (clumps, not the troughs between them); the inner ``0.25 r_petro`` is
    excluded because a nucleus is sharp by nature; and the same statistic
    on sky pixels is subtracted, since noise is small-scale structure too.
    Before those, S was the absolute residual over the whole footprint with
    a Gaussian blur of 5 % of the stamp: on the statmorph comparison it sat
    0.5 above theirs and anticorrelated with it.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - centre[0], yy - centre[1])
    if petrosian_radius is None or not np.isfinite(petrosian_radius) or petrosian_radius <= 0:
        # Fall back to the footprint's own size: r_petro is about 0.6 of the
        # radius of an equal-area circle for a typical profile.
        petrosian_radius = 0.6 * float(np.sqrt(footprint.sum() / np.pi)) if footprint.any() \
            else min(ny, nx) / 4.0
    if smoothing_radius is None:
        smoothing_radius = max(0.25 * float(petrosian_radius), 1.0)
    box = max(3, int(round(2.0 * float(smoothing_radius))) | 1)
    smooth = _boxcar(data, box)
    residual = np.clip(data - smooth, 0.0, None)

    inner = float(inner_exclusion) * float(petrosian_radius)
    outer = 1.5 * float(petrosian_radius)
    annulus = footprint & (r > inner) & (r <= outer)
    if annulus.sum() < 8:
        annulus = footprint & (r > inner)
    if annulus.sum() < 8:
        annulus = footprint
    total = float(data[annulus].sum())
    if total <= 0:
        return {"smoothness": float("nan"), "clumpiness": float("nan"),
                "smoothness_raw": float("nan"), "smoothness_sky": float("nan")}
    raw = float(residual[annulus].sum() / total)
    sky = _sky_region(data.shape, mask, sky_mask)
    if mask is not None:
        sky = sky & ~footprint
    sky_term = float(residual[sky].mean() * annulus.sum() / total) if sky.sum() >= 8 else 0.0
    value = raw - sky_term
    return {"smoothness": float(value), "clumpiness": float(value),
            "smoothness_raw": raw, "smoothness_sky": sky_term,
            "smoothing_box_pixels": int(box)}


def cas_statistics(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                   mask: Optional[np.ndarray] = None,
                   background: float = 0.0) -> Dict[str, float]:
    """Compute all three CAS statistics for one object."""
    result: Dict[str, float] = {}
    result.update(concentration(cutout, centre, background=background))
    result.update(asymmetry(cutout, centre, mask))
    result.update(smoothness(cutout, centre=centre, mask=mask))
    return result
