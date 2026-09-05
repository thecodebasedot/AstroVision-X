"""Aperture photometry.

Flux is measured by summing pixels inside an aperture and subtracting a
locally-estimated sky.  Apertures are computed with *fractional* pixel
coverage, because at the few-pixel radii typical of astronomical sources
a binary mask biases the flux by several percent.

Every routine here works on the smallest rectangle that contains the
aperture, never on the whole frame. An aperture of radius five touches a
hundred pixels; the frame it sits in may hold a hundred million, and a
photometry stage that multiplies the whole frame once per source and per
radius is one that takes a day on a survey image. The full-frame weight
maps are still available (:func:`circular_aperture_weights` and its
elliptical twin) for callers that want them, built by pasting the block into
zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, sigma_clipped_stats

log = get_logger("photometry.aperture")

#: ``(y0, y1, x0, x1)`` -- the half-open pixel bounds of a block.
Bounds = Tuple[int, int, int, int]


def stamp_box(shape: Tuple[int, int], centre: Tuple[float, float], reach: float
              ) -> Tuple[slice, slice, Tuple[float, float]]:
    """Slices for the square of half-width ``reach`` around ``centre``, clipped
    to the frame, and the centre expressed in the cut-out's own pixels.

    >>> rows, cols, local = stamp_box((100, 100), (10.0, 50.0), 20)
    >>> (rows.start, rows.stop, cols.start, cols.stop), local
    ((30, 71, 0, 31), (10.0, 20.0))
    """
    ny, nx = int(shape[0]), int(shape[1])
    reach = float(max(reach, 0.0))
    x0 = max(0, int(np.floor(centre[0] - reach)))
    x1 = min(nx, int(np.ceil(centre[0] + reach)) + 1)
    y0 = max(0, int(np.floor(centre[1] - reach)))
    y1 = min(ny, int(np.ceil(centre[1] + reach)) + 1)
    x1, y1 = max(x1, x0), max(y1, y0)
    return slice(y0, y1), slice(x0, x1), (float(centre[0]) - x0, float(centre[1]) - y0)


def circular_aperture_block(shape: Tuple[int, int], centre: Tuple[float, float],
                            radius: float, subpixels: int = 5
                            ) -> Tuple[np.ndarray, Bounds]:
    """Fractional coverage of a circle, on just the pixels it can touch.

    Returns the weight block and its ``(y0, y1, x0, x1)`` bounds in the frame.
    Each partially-covered pixel is subdivided into ``subpixels**2`` samples
    and weighted by the fraction inside the circle -- the scheme photutils
    uses, accurate to well under one percent.
    """
    ny, nx = int(shape[0]), int(shape[1])
    radius = max(float(radius), 1e-6)
    x0 = max(0, int(np.floor(centre[0] - radius - 1)))
    x1 = min(nx, int(np.ceil(centre[0] + radius + 2)))
    y0 = max(0, int(np.floor(centre[1] - radius - 1)))
    y1 = min(ny, int(np.ceil(centre[1] + radius + 2)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=float), (y0, y0, x0, x0)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    inside = distance <= radius - 0.7071          # fully inside
    outside = distance >= radius + 0.7071         # fully outside
    partial = ~inside & ~outside

    block = np.zeros((y1 - y0, x1 - x0), dtype=float)
    block[inside] = 1.0
    if partial.any():
        n = max(2, int(subpixels))
        offsets = (np.arange(n) + 0.5) / n - 0.5
        sub_y, sub_x = np.meshgrid(offsets, offsets, indexing="ij")
        py, px = np.nonzero(partial)
        cy = yy[partial][:, None, None] + sub_y[None, :, :]
        cx = xx[partial][:, None, None] + sub_x[None, :, :]
        covered = (np.hypot(cx - centre[0], cy - centre[1]) <= radius).mean(axis=(1, 2))
        block[py, px] = covered
    return block, (y0, y1, x0, x1)


def elliptical_aperture_block(shape: Tuple[int, int], centre: Tuple[float, float],
                              a: float, b: float, theta_deg: float,
                              subpixels: int = 5) -> Tuple[np.ndarray, Bounds]:
    """Fractional coverage of an ellipse on the pixels it can touch."""
    ny, nx = int(shape[0]), int(shape[1])
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    theta = np.deg2rad(theta_deg)
    reach = a + 2.0
    x0 = max(0, int(np.floor(centre[0] - reach)))
    x1 = min(nx, int(np.ceil(centre[0] + reach + 1)))
    y0 = max(0, int(np.floor(centre[1] - reach)))
    y1 = min(ny, int(np.ceil(centre[1] + reach + 1)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=float), (y0, y0, x0, x0)

    n = max(2, int(subpixels))
    offsets = (np.arange(n) + 0.5) / n - 0.5
    sub_y, sub_x = np.meshgrid(offsets, offsets, indexing="ij")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy = yy[..., None, None] + sub_y
    cx = xx[..., None, None] + sub_x
    dx, dy = cx - centre[0], cy - centre[1]
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    block = (((xr / a) ** 2 + (yr / b) ** 2) <= 1.0).mean(axis=(-2, -1))
    return np.asarray(block, dtype=float), (y0, y1, x0, x1)


def _paste(shape: Tuple[int, int], block: np.ndarray, bounds: Bounds) -> np.ndarray:
    weights = np.zeros((int(shape[0]), int(shape[1])), dtype=float)
    y0, y1, x0, x1 = bounds
    if block.size:
        weights[y0:y1, x0:x1] = block
    return weights


def circular_aperture_weights(shape: Tuple[int, int], centre: Tuple[float, float],
                              radius: float, subpixels: int = 5) -> np.ndarray:
    """Fractional pixel coverage of a circular aperture, as a full-frame map.

    Convenient on a postage stamp; on a survey frame use
    :func:`circular_aperture_block` and slice the data to its bounds.
    """
    block, bounds = circular_aperture_block(shape, centre, radius, subpixels)
    return _paste(shape, block, bounds)


def elliptical_aperture_weights(shape: Tuple[int, int], centre: Tuple[float, float],
                                a: float, b: float, theta_deg: float,
                                subpixels: int = 5) -> np.ndarray:
    """Fractional coverage of an elliptical aperture, as a full-frame map."""
    block, bounds = elliptical_aperture_block(shape, centre, a, b, theta_deg, subpixels)
    return _paste(shape, block, bounds)


def _slice(array: Optional[np.ndarray], bounds: Bounds) -> Optional[np.ndarray]:
    if array is None:
        return None
    y0, y1, x0, x1 = bounds
    return array[y0:y1, x0:x1]


def annulus_background(image: np.ndarray, centre: Tuple[float, float],
                       inner: float, outer: float,
                       mask: Optional[np.ndarray] = None) -> Tuple[float, float, int]:
    """Robust sky level in an annulus; returns ``(median, rms, n_pixels)``.

    Neighbouring sources contaminate the annulus, so the estimate is
    sigma-clipped rather than a plain mean.
    """
    data = as_float_image(image)
    rows, cols, local = stamp_box(data.shape, centre, float(outer) + 1.0)
    block = data[rows, cols]
    if block.size == 0:
        return 0.0, 0.0, 0
    yy, xx = np.mgrid[0:block.shape[0], 0:block.shape[1]]
    distance = np.hypot(xx - local[0], yy - local[1])
    ring = (distance >= float(inner)) & (distance <= float(outer)) & np.isfinite(block)
    if mask is not None:
        ring &= ~np.asarray(mask, dtype=bool)[rows, cols]
    values = block[ring]
    if values.size < 5:
        return 0.0, 0.0, int(values.size)
    _, median, std = sigma_clipped_stats(values, sigma=3.0)
    return float(median), float(std), int(values.size)


@dataclass
class ApertureResult:
    """Outcome of one aperture measurement."""

    flux: float
    flux_err: float
    area: float
    background: float = 0.0
    background_rms: float = 0.0
    radius: float = float("nan")
    snr: float = float("nan")
    n_masked: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {"flux": self.flux, "flux_err": self.flux_err, "area": self.area,
                "background": self.background, "background_rms": self.background_rms,
                "radius": self.radius, "snr": self.snr, "n_masked": int(self.n_masked)}


def _sum_in_aperture(data: np.ndarray, weights: np.ndarray, bounds: Bounds,
                     rms: Optional[np.ndarray], mask: Optional[np.ndarray]
                     ) -> Tuple[float, float, float, int]:
    """``(raw flux, area, background variance, n masked)`` over one block."""
    if mask is not None:
        bad = np.asarray(_slice(mask, bounds), dtype=bool)
        n_masked = int((weights > 0)[bad].sum())
        weights = np.where(bad, 0.0, weights)
    else:
        n_masked = 0
    area = float(weights.sum())
    if area <= 0:
        return float("nan"), 0.0, 0.0, n_masked
    values = np.nan_to_num(_slice(data, bounds), nan=0.0)
    raw_flux = float((values * weights).sum())
    if rms is not None:
        noise = np.clip(np.asarray(_slice(rms, bounds), dtype=float), 0, None)
        variance = float(((noise ** 2) * weights).sum())
    else:
        variance = float("nan")
    return raw_flux, area, variance, n_masked


def aperture_photometry(image: np.ndarray, centre: Tuple[float, float], radius: float,
                        rms: Optional[np.ndarray] = None, gain: float = 1.0,
                        local_background: bool = True,
                        annulus: Tuple[float, float] = (8.0, 14.0),
                        mask: Optional[np.ndarray] = None,
                        subpixels: int = 5) -> ApertureResult:
    """Measure flux inside a circular aperture.

    The uncertainty combines Poisson noise from the source itself with the
    background noise over the aperture area -- the standard CCD equation.
    """
    data = as_float_image(image)
    weights, bounds = circular_aperture_block(data.shape, centre, radius, subpixels)
    raw_flux, area, background_variance, n_masked = _sum_in_aperture(
        data, weights, bounds, rms, mask)
    if area <= 0:
        return ApertureResult(float("nan"), float("nan"), 0.0, radius=float(radius))

    sky, sky_rms, _ = (annulus_background(data, centre, annulus[0], annulus[1], mask)
                       if local_background else (0.0, 0.0, 0))
    flux = raw_flux - sky * area

    if rms is None:
        background_variance = (sky_rms ** 2) * area if sky_rms > 0 else 0.0
    poisson_variance = max(flux, 0.0) / max(float(gain), 1e-9)
    flux_err = float(np.sqrt(max(background_variance + poisson_variance, 0.0)))
    snr = float(flux / flux_err) if flux_err > 0 else float("nan")

    return ApertureResult(flux=flux, flux_err=flux_err, area=area, background=sky,
                          background_rms=sky_rms, radius=float(radius), snr=snr,
                          n_masked=n_masked)


def elliptical_photometry(image: np.ndarray, centre: Tuple[float, float],
                          a: float, b: float, theta_deg: float,
                          rms: Optional[np.ndarray] = None, gain: float = 1.0,
                          background: float = 0.0,
                          mask: Optional[np.ndarray] = None) -> ApertureResult:
    """Flux inside an ellipse, as used for Kron (``FLUX_AUTO``) magnitudes."""
    data = as_float_image(image)
    weights, bounds = elliptical_aperture_block(data.shape, centre, a, b, theta_deg)
    raw_flux, area, variance, n_masked = _sum_in_aperture(data, weights, bounds, rms, mask)
    if area <= 0:
        return ApertureResult(float("nan"), float("nan"), 0.0)

    flux = raw_flux - float(background) * area
    if rms is None:
        variance = 0.0
    variance += max(flux, 0.0) / max(float(gain), 1e-9)
    flux_err = float(np.sqrt(max(variance, 0.0)))
    return ApertureResult(flux=flux, flux_err=flux_err, area=area, background=background,
                          radius=float(a), snr=float(flux / flux_err) if flux_err > 0 else float("nan"),
                          n_masked=n_masked)


def multi_aperture(image: np.ndarray, centre: Tuple[float, float],
                   radii: Sequence[float], **kwargs) -> Dict[float, ApertureResult]:
    """Measure a series of concentric apertures (the curve of growth)."""
    return {float(r): aperture_photometry(image, centre, r, **kwargs) for r in radii}
