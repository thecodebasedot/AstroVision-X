"""Aperture photometry.

Flux is measured by summing pixels inside an aperture and subtracting a
locally-estimated sky.  Apertures are computed with *fractional* pixel
coverage, because at the few-pixel radii typical of astronomical sources
a binary mask biases the flux by several percent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, sigma_clipped_stats

log = get_logger("photometry.aperture")


def circular_aperture_weights(shape: Tuple[int, int], centre: Tuple[float, float],
                              radius: float, subpixels: int = 5) -> np.ndarray:
    """Fractional pixel coverage of a circular aperture.

    Each pixel is subdivided into ``subpixels**2`` samples; the returned
    weight is the fraction of samples inside the circle.  This is the same
    scheme photutils uses and is accurate to well under one percent.
    """
    ny, nx = int(shape[0]), int(shape[1])
    radius = max(float(radius), 1e-6)
    weights = np.zeros((ny, nx), dtype=float)

    x0 = max(0, int(np.floor(centre[0] - radius - 1)))
    x1 = min(nx, int(np.ceil(centre[0] + radius + 2)))
    y0 = max(0, int(np.floor(centre[1] - radius - 1)))
    y1 = min(ny, int(np.ceil(centre[1] + radius + 2)))
    if x1 <= x0 or y1 <= y0:
        return weights

    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    inside = distance <= radius - 0.7071          # fully inside
    outside = distance >= radius + 0.7071         # fully outside
    partial = ~inside & ~outside

    block = weights[y0:y1, x0:x1]
    block[inside] = 1.0

    if partial.any():
        n = max(2, int(subpixels))
        step = 1.0 / n
        offsets = (np.arange(n) + 0.5) * step - 0.5
        sub_y, sub_x = np.meshgrid(offsets, offsets, indexing="ij")
        py, px = np.nonzero(partial)
        cy = yy[partial][:, None, None] + sub_y[None, :, :]
        cx = xx[partial][:, None, None] + sub_x[None, :, :]
        covered = (np.hypot(cx - centre[0], cy - centre[1]) <= radius).mean(axis=(1, 2))
        block[py, px] = covered

    weights[y0:y1, x0:x1] = block
    return weights


def elliptical_aperture_weights(shape: Tuple[int, int], centre: Tuple[float, float],
                                a: float, b: float, theta_deg: float,
                                subpixels: int = 5) -> np.ndarray:
    """Fractional coverage of an elliptical aperture (Kron-style photometry)."""
    ny, nx = int(shape[0]), int(shape[1])
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    theta = np.deg2rad(theta_deg)
    reach = a + 2.0

    x0 = max(0, int(np.floor(centre[0] - reach)))
    x1 = min(nx, int(np.ceil(centre[0] + reach + 1)))
    y0 = max(0, int(np.floor(centre[1] - reach)))
    y1 = min(ny, int(np.ceil(centre[1] + reach + 1)))
    weights = np.zeros((ny, nx), dtype=float)
    if x1 <= x0 or y1 <= y0:
        return weights

    n = max(2, int(subpixels))
    offsets = (np.arange(n) + 0.5) / n - 0.5
    sub_y, sub_x = np.meshgrid(offsets, offsets, indexing="ij")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy = yy[..., None, None] + sub_y
    cx = xx[..., None, None] + sub_x
    dx, dy = cx - centre[0], cy - centre[1]
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    weights[y0:y1, x0:x1] = (((xr / a) ** 2 + (yr / b) ** 2) <= 1.0).mean(axis=(-2, -1))
    return weights


def annulus_background(image: np.ndarray, centre: Tuple[float, float],
                       inner: float, outer: float,
                       mask: Optional[np.ndarray] = None) -> Tuple[float, float, int]:
    """Robust sky level in an annulus; returns ``(median, rms, n_pixels)``.

    Neighbouring sources contaminate the annulus, so the estimate is
    sigma-clipped rather than a plain mean.
    """
    data = as_float_image(image)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    ring = (distance >= float(inner)) & (distance <= float(outer)) & np.isfinite(data)
    if mask is not None:
        ring &= ~np.asarray(mask, dtype=bool)
    values = data[ring]
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
    weights = circular_aperture_weights(data.shape, centre, radius, subpixels)
    if mask is not None:
        bad = np.asarray(mask, dtype=bool)
        n_masked = int((weights > 0)[bad].sum())
        weights = np.where(bad, 0.0, weights)
    else:
        n_masked = 0

    area = float(weights.sum())
    if area <= 0:
        return ApertureResult(float("nan"), float("nan"), 0.0, radius=float(radius))

    values = np.nan_to_num(data, nan=0.0)
    raw_flux = float((values * weights).sum())

    sky, sky_rms, _ = (annulus_background(data, centre, annulus[0], annulus[1], mask)
                       if local_background else (0.0, 0.0, 0))
    flux = raw_flux - sky * area

    if rms is not None:
        noise_map = np.clip(np.asarray(rms, dtype=float), 0, None)
        background_variance = float(((noise_map ** 2) * weights).sum())
    else:
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
    weights = elliptical_aperture_weights(data.shape, centre, a, b, theta_deg)
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), 0.0, weights)
    area = float(weights.sum())
    if area <= 0:
        return ApertureResult(float("nan"), float("nan"), 0.0)

    values = np.nan_to_num(data, nan=0.0)
    flux = float((values * weights).sum()) - float(background) * area
    if rms is not None:
        variance = float(((np.clip(np.asarray(rms, dtype=float), 0, None) ** 2) * weights).sum())
    else:
        variance = 0.0
    variance += max(flux, 0.0) / max(float(gain), 1e-9)
    flux_err = float(np.sqrt(max(variance, 0.0)))
    return ApertureResult(flux=flux, flux_err=flux_err, area=area, background=background,
                          radius=float(a), snr=float(flux / flux_err) if flux_err > 0 else float("nan"))


def multi_aperture(image: np.ndarray, centre: Tuple[float, float],
                   radii: Sequence[float], **kwargs) -> Dict[float, ApertureResult]:
    """Measure a series of concentric apertures (the curve of growth)."""
    return {float(r): aperture_photometry(image, centre, r, **kwargs) for r in radii}
