"""Analytic surface-brightness profiles used by the sky simulator.

These are the same functional forms the morphology stage fits back out of
the data, which makes the simulator a genuine end-to-end test bed.
"""

from __future__ import annotations

import numpy as np

from ..core.numeric import SIGMA_TO_FWHM


def sersic_bn(n: float) -> float:
    """Ciotti & Bertin (1999) approximation to the Sersic ``b_n`` constant."""
    n = max(float(n), 0.05)
    return (2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) +
            46.0 / (25515.0 * n ** 2) + 131.0 / (1148175.0 * n ** 3))


def sersic_profile(r: np.ndarray, amplitude: float, r_eff: float, n: float) -> np.ndarray:
    """Sersic intensity ``I(r) = I_e exp(-b_n [(r/r_e)^(1/n) - 1])``."""
    r_eff = max(float(r_eff), 1e-3)
    n = float(np.clip(n, 0.05, 12.0))
    bn = sersic_bn(n)
    ratio = np.maximum(np.asarray(r, dtype=float), 0.0) / r_eff
    return amplitude * np.exp(-bn * (np.power(ratio, 1.0 / n) - 1.0))


def gaussian_psf(shape, centre, fwhm: float, amplitude: float = 1.0) -> np.ndarray:
    """Circular Gaussian point-spread function rendered on a pixel grid."""
    ny, nx = int(shape[0]), int(shape[1])
    sigma = max(float(fwhm), 0.4) / SIGMA_TO_FWHM
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2
    return amplitude * np.exp(-0.5 * r2 / sigma ** 2)


def moffat_psf(shape, centre, fwhm: float, beta: float = 3.5,
               amplitude: float = 1.0) -> np.ndarray:
    """Moffat PSF -- a more realistic seeing profile with broad wings."""
    ny, nx = int(shape[0]), int(shape[1])
    beta = max(float(beta), 1.05)
    alpha = max(float(fwhm), 0.4) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2
    return amplitude * (1.0 + r2 / alpha ** 2) ** (-beta)


def supersample(shape, centre, render, factor: int = 3) -> np.ndarray:
    """Render a profile on a ``factor``x finer grid, then average down.

    Steeply-cusped profiles (a de Vaucouleurs bulge has I(0)/I(r_e) ~ 2000)
    are wildly over-weighted if the analytic form is point-sampled at pixel
    centres.  Averaging sub-pixel samples approximates the integral over
    each pixel, which is what a detector actually records.
    """
    factor = max(1, int(factor))
    ny, nx = int(shape[0]), int(shape[1])
    if factor == 1:
        return render((ny, nx), centre)
    fine_centre = ((centre[0] + 0.5) * factor - 0.5, (centre[1] + 0.5) * factor - 0.5)
    fine = render((ny * factor, nx * factor), fine_centre)
    return fine.reshape(ny, factor, nx, factor).mean(axis=(1, 3))


def elliptical_radius(shape, centre, axis_ratio: float, pa_deg: float) -> np.ndarray:
    """Elliptical radius map used to render inclined galaxies."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    theta = np.deg2rad(pa_deg)
    dx = xx - centre[0]
    dy = yy - centre[1]
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    q = float(np.clip(axis_ratio, 0.05, 1.0))
    return np.sqrt(xr ** 2 + (yr / q) ** 2)


def spiral_pattern(shape, centre, r_eff: float, n_arms: int = 2,
                   pitch_deg: float = 20.0, strength: float = 0.6,
                   axis_ratio: float = 1.0, pa_deg: float = 0.0,
                   phase: float = 0.0) -> np.ndarray:
    """Multiplicative logarithmic-spiral modulation in ``[1-s, 1+s]``."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    theta_pa = np.deg2rad(pa_deg)
    dx = xx - centre[0]
    dy = yy - centre[1]
    xr = dx * np.cos(theta_pa) + dy * np.sin(theta_pa)
    yr = (-dx * np.sin(theta_pa) + dy * np.cos(theta_pa)) / max(axis_ratio, 0.05)
    r = np.hypot(xr, yr)
    phi = np.arctan2(yr, xr)
    pitch = np.tan(np.deg2rad(np.clip(pitch_deg, 3.0, 70.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(np.maximum(r, 1e-3) / max(r_eff, 1e-3))
    wave = np.cos(int(n_arms) * (phi - log_r / max(pitch, 1e-3)) + phase)
    # Arms fade in the core and at large radii, as in real disc galaxies.
    envelope = np.clip(r / max(0.35 * r_eff, 1e-3), 0.0, 1.0) * np.exp(-r / (2.6 * max(r_eff, 1e-3)))
    return 1.0 + float(strength) * wave * envelope


def bar_pattern(shape, centre, length: float, width: float, pa_deg: float,
                strength: float = 0.5) -> np.ndarray:
    """Additive stellar-bar component (a smooth elongated ridge)."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    theta = np.deg2rad(pa_deg)
    dx = xx - centre[0]
    dy = yy - centre[1]
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    a = max(float(length), 1.0)
    b = max(float(width), 0.5)
    return float(strength) * np.exp(-0.5 * ((xr / a) ** 4 + (yr / b) ** 2))


def einstein_arc(shape, centre, radius: float, width: float,
                 span_deg: float = 120.0, pa_deg: float = 0.0,
                 amplitude: float = 1.0) -> np.ndarray:
    """A tangentially stretched lensed arc at the Einstein radius."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - centre[0]
    dy = yy - centre[1]
    r = np.hypot(dx, dy)
    phi = np.degrees(np.arctan2(dy, dx))
    delta = (phi - pa_deg + 180.0) % 360.0 - 180.0
    half_span = max(float(span_deg), 1.0) / 2.0
    radial = np.exp(-0.5 * ((r - float(radius)) / max(float(width), 0.4)) ** 2)
    # Smooth taper at the arc ends instead of a hard azimuthal cut.
    azimuthal = np.exp(-0.5 * (np.clip(np.abs(delta) - half_span, 0, None) /
                               max(half_span * 0.25, 3.0)) ** 2)
    return float(amplitude) * radial * azimuthal
