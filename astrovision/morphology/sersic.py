"""Parametric profile fitting.

The Sersic index ``n`` is the single most informative parametric
descriptor of a galaxy: ``n ~ 1`` is an exponential disc, ``n ~ 4`` the de
Vaucouleurs profile of a classical elliptical.  Fitting the azimuthally
averaged profile is far more robust than a full 2-D fit at survey depth,
so that is what this module does, with an optional 2-D refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.numeric import as_float_image, convolve, nan_to_finite, radial_profile
from ..simulate.profiles import sersic_bn, sersic_profile

log = get_logger("morphology.sersic")


@dataclass
class SersicFit:
    """Result of fitting a Sersic profile to one object."""

    amplitude: float = float("nan")     # intensity at r_eff
    r_eff: float = float("nan")
    n: float = float("nan")
    axis_ratio: float = 1.0
    position_angle: float = 0.0
    background: float = 0.0
    chi2: float = float("nan")
    reduced_chi2: float = float("nan")
    success: bool = False
    method: str = "none"
    n_points: int = 0

    @property
    def total_flux(self) -> float:
        """Analytic total flux of the fitted profile."""
        if not (np.isfinite(self.amplitude) and np.isfinite(self.r_eff) and np.isfinite(self.n)):
            return float("nan")
        n = float(np.clip(self.n, 0.05, 12.0))
        bn = sersic_bn(n)
        from math import gamma
        return float(self.amplitude * self.r_eff ** 2 * 2 * np.pi * n *
                     np.exp(bn) / bn ** (2 * n) * gamma(2 * n) * self.axis_ratio)

    def evaluate(self, r) -> np.ndarray:
        """The fitted intensity at radius ``r``."""
        return sersic_profile(np.asarray(r, dtype=float), self.amplitude, self.r_eff, self.n)

    def to_dict(self) -> Dict[str, Any]:
        return {"amplitude": float(self.amplitude), "r_eff": float(self.r_eff),
                "n": float(self.n), "axis_ratio": float(self.axis_ratio),
                "position_angle": float(self.position_angle),
                "background": float(self.background), "chi2": float(self.chi2),
                "reduced_chi2": float(self.reduced_chi2), "success": bool(self.success),
                "method": self.method, "total_flux": self.total_flux}


def fit_sersic_1d(radii: np.ndarray, intensity: np.ndarray,
                  errors: Optional[np.ndarray] = None,
                  n_grid: Optional[np.ndarray] = None) -> SersicFit:
    """Fit ``I(r)`` by scanning ``n`` and solving the rest in closed form.

    For a fixed ``n``, taking logs makes the profile linear in
    ``log(amplitude)`` and ``(r/r_e)^(1/n)``, so each trial ``n`` costs one
    least-squares solve.  Scanning is more reliable than a gradient
    optimiser here because the chi-squared surface in ``n`` is shallow and
    riddled with local minima at low signal-to-noise.
    """
    r = np.asarray(radii, dtype=float).ravel()
    values = np.asarray(intensity, dtype=float).ravel()
    good = np.isfinite(r) & np.isfinite(values) & (values > 0) & (r > 0)
    if good.sum() < 4:
        return SersicFit(n_points=int(good.sum()), method="insufficient_points")

    r, values = r[good], values[good]
    weights = np.ones_like(values)
    if errors is not None:
        errs = np.asarray(errors, dtype=float).ravel()[good]
        weights = np.where(np.isfinite(errs) & (errs > 0), 1.0 / np.maximum(errs, 1e-9), 1.0)

    if n_grid is None:
        n_grid = np.concatenate([np.linspace(0.3, 2.0, 35), np.linspace(2.1, 8.0, 40)])

    log_values = np.log(values)
    best: Optional[Tuple[float, float, float, float]] = None
    for n in n_grid:
        # log I = log I_e - bn * ((r/r_e)^(1/n) - 1); with u = r^(1/n) this
        # is linear: log I = a + b*u, and r_e follows from b.
        u = np.power(r, 1.0 / n)
        design = np.column_stack([np.ones_like(u), u]) * weights[:, None]
        target = log_values * weights
        try:
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
            continue
        intercept, slope = float(solution[0]), float(solution[1])
        if slope >= 0:
            continue                     # profile must decline outwards
        bn = sersic_bn(n)
        r_eff = float((-bn / slope) ** n)
        if not np.isfinite(r_eff) or r_eff <= 0 or r_eff > 10 * float(r.max()):
            continue
        amplitude = float(np.exp(intercept + bn))
        model = sersic_profile(r, amplitude, r_eff, n)
        residual = (values - model) * weights
        chi2 = float(np.sum(residual ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, amplitude, r_eff, float(n))

    if best is None:
        return SersicFit(n_points=int(r.size), method="no_valid_solution")
    chi2, amplitude, r_eff, n = best
    dof = max(r.size - 3, 1)
    return SersicFit(amplitude=amplitude, r_eff=r_eff, n=n, chi2=chi2,
                     reduced_chi2=chi2 / dof, success=True, method="profile_scan",
                     n_points=int(r.size))


def fit_sersic_2d(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                  axis_ratio: float = 1.0, position_angle: float = 0.0,
                  mask: Optional[np.ndarray] = None,
                  initial: Optional[SersicFit] = None,
                  psf: Optional[np.ndarray] = None,
                  max_nfev: int = 120,
                  fit_mask: Optional[np.ndarray] = None,
                  r_half: float = float("nan"),
                  noise: float = float("nan")) -> SersicFit:
    """Refine a Sersic fit on the 2-D image with SciPy, if it is installed.

    The 1-D scan supplies the starting point, so the optimiser only has to
    polish rather than search, which keeps it from wandering off.  When a
    ``psf`` kernel is supplied the model is convolved with it before
    comparison -- without that, seeing flattens the central cusp and every
    concentrated galaxy is fitted with an index that is far too low.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    start = initial
    if start is None or not start.success:
        radii, profile = radial_profile(np.where(footprint, data, np.nan), centre, nbins=24)
        start = fit_sersic_1d(radii, profile)
    if not start.success:
        return start

    optimize = try_import("scipy.optimize")
    if optimize is None:
        start.axis_ratio = float(axis_ratio)
        start.position_angle = float(position_angle)
        return start

    yy, xx = np.mgrid[0:ny, 0:nx]
    kernel = _compact_kernel(psf)

    # The centroid is already measured to a fraction of a pixel by the
    # detection stage, so it is held fixed: five free parameters converge
    # several times faster than seven, and the model must be re-convolved
    # on every function evaluation.
    cx, cy = float(centre[0]), float(centre[1])
    dx0, dy0 = xx - cx, yy - cy

    # The fit region has to be scaled to the object.  Restricted to the
    # bright isophote it loses the outer profile that constrains the index;
    # extended over the whole cutout, the optimiser trades a larger radius
    # against a steeper index to absorb pure sky noise.  A few effective
    # radii is the compromise every profile-fitting code makes.
    scale_radius = float(r_half) if np.isfinite(r_half) and r_half > 0 else start.r_eff
    fit_radius = float(np.clip(3.5 * scale_radius, 4.0, min(ny, nx) / 2.0))
    yy_r, xx_r = np.mgrid[0:ny, 0:nx]
    fit_region = np.hypot(xx_r - centre[0], yy_r - centre[1]) <= fit_radius
    fit_region |= footprint
    if fit_mask is not None:
        fit_region &= np.asarray(fit_mask, dtype=bool)
    if fit_region.sum() < 20:
        fit_region = footprint

    def model(params: np.ndarray) -> np.ndarray:
        amplitude, r_eff, n, q, pa, sky = params
        theta = np.deg2rad(pa)
        xr = dx0 * np.cos(theta) + dy0 * np.sin(theta)
        yr = -dx0 * np.sin(theta) + dy0 * np.cos(theta)
        r = np.sqrt(xr ** 2 + (yr / max(q, 0.05)) ** 2)
        rendered = sersic_profile(r, amplitude, max(r_eff, 0.3), float(np.clip(n, 0.2, 10.0)))
        if kernel is not None:
            rendered = convolve(rendered, kernel)
        return rendered + sky

    def residual(params: np.ndarray) -> np.ndarray:
        return (model(params) - data)[fit_region]

    # A free sky pedestal matters: an over- or under-subtracted background
    # of even a fraction of the noise tilts the outer profile and drags the
    # Sersic index with it.
    guess = np.array([start.amplitude, start.r_eff, start.n,
                      float(axis_ratio), float(position_angle), 0.0], dtype=float)
    # The sky pedestal must be bounded by the *noise*, not by the spread of
    # the data: a bright galaxy inflates the latter enormously, and a sky
    # term free to roam that far simply eats the object's outer flux.
    scale = float(noise) if np.isfinite(noise) and noise > 0 else _robust_noise(data, footprint)
    # ``n`` and ``r_eff`` are strongly degenerate: a too-large radius can
    # always be compensated by a steeper index.  The half-light radius is
    # measured directly from the curve of growth and is, by definition,
    # exactly what ``r_eff`` means -- so it pins down the degenerate
    # direction without constraining the shape.
    if np.isfinite(r_half) and r_half > 0:
        r_lo, r_hi = 0.4 * float(r_half), 2.5 * float(r_half)
    else:
        r_lo, r_hi = 0.3, max(ny, nx) * 2.0
    r_lo = max(r_lo, 0.3)
    r_hi = max(r_hi, r_lo * 1.5)
    bounds = (
        np.array([1e-6, r_lo, 0.2, 0.05, -360.0, -3.0 * scale - 1e-6]),
        np.array([np.inf, r_hi, 8.0, 1.0, 360.0, 3.0 * scale + 1e-6]),
    )
    guess = np.clip(guess, bounds[0] + 1e-9, bounds[1] - 1e-9)
    try:
        result = optimize.least_squares(
            residual, guess, bounds=bounds, method="trf",
            max_nfev=max_nfev, ftol=1e-3, xtol=1e-3, gtol=1e-3,
            x_scale=np.array([max(abs(guess[0]), 1e-3), max(guess[1], 1.0), 1.0, 0.3, 45.0,
                              max(scale, 1e-3)]))
    except Exception as exc:  # pragma: no cover - optimiser edge cases
        log.debug("2-D Sersic fit failed (%s); keeping the 1-D solution", exc)
        return start

    amplitude, r_eff, n, q, pa, sky = result.x
    dof = max(int(fit_region.sum()) - 6, 1)
    chi2 = float(np.sum(result.fun ** 2))
    return SersicFit(amplitude=float(amplitude), r_eff=float(r_eff), n=float(n),
                     axis_ratio=float(q), position_angle=float(pa % 180.0),
                     background=float(sky), chi2=chi2, reduced_chi2=chi2 / dof,
                     success=bool(result.success), method="least_squares_2d",
                     n_points=int(fit_region.sum()))


def _robust_noise(data: np.ndarray, footprint: Optional[np.ndarray] = None) -> float:
    """Median-absolute-deviation noise from the pixels outside the object."""
    values = np.asarray(data, dtype=float)
    if footprint is not None and (~np.asarray(footprint, dtype=bool)).sum() > 25:
        values = values[~np.asarray(footprint, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size < 10:
        return 1.0
    return float(max(1.4826 * np.median(np.abs(values - np.median(values))), 1e-9))


def _compact_kernel(psf: Optional[np.ndarray], max_size: int = 11) -> Optional[np.ndarray]:
    """Trim and renormalise a PSF kernel for use inside the fitting loop.

    A 25x25 empirical stamp costs several times more per convolution than
    the 11x11 core that carries essentially all of its weight, and the
    model is convolved on every optimiser evaluation.
    """
    if psf is None:
        return None
    kernel = np.asarray(psf, dtype=float)
    if kernel.ndim != 2 or kernel.size == 0:
        return None
    size = min(int(max_size) | 1, min(kernel.shape) | 1)
    if size < min(kernel.shape):
        cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
        half = size // 2
        kernel = kernel[cy - half:cy + half + 1, cx - half:cx + half + 1]
    total = float(kernel.sum())
    return kernel / total if total > 0 else None


def fit_sersic(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
               mask: Optional[np.ndarray] = None, axis_ratio: float = 1.0,
               position_angle: float = 0.0, refine: bool = True,
               psf: Optional[np.ndarray] = None, psf_fwhm: float = 0.0,
               max_pixels: int = 20000,
               fit_mask: Optional[np.ndarray] = None,
               r_half: float = float("nan"),
               noise: float = float("nan")) -> SersicFit:
    """Fit a Sersic profile: 1-D scan, then optional PSF-aware 2-D refinement.

    Pass ``psf`` (a kernel) or ``psf_fwhm`` so the model is compared to the
    data through the same seeing the data was taken in.
    """
    data = as_float_image(cutout)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    footprint = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    # The 1-D scan cannot deconvolve, so exclude the seeing-dominated core
    # where the profile shape carries no information about n.
    profile_limit = min(data.shape) / 2.0
    if np.isfinite(r_half) and r_half > 0:
        profile_limit = min(profile_limit, 6.0 * float(r_half))
    radii, profile = radial_profile(data, centre, nbins=28)
    usable = radii <= profile_limit
    if psf_fwhm and psf_fwhm > 0:
        # Inside about one FWHM the profile is pure seeing and says nothing
        # about the intrinsic shape.
        usable &= radii > 0.9 * float(psf_fwhm)
    sigma = float(noise) if np.isfinite(noise) and noise > 0 else _robust_noise(data, footprint)
    if sigma > 0:
        # Beyond the radius where the profile sinks into the noise, the
        # log-linear scan is fitting sky, and it drags the index with it.
        usable &= profile > 1.5 * sigma
        if usable.sum() >= 5:
            last = int(np.nonzero(usable)[0][-1])
            usable[last + 1:] = False
    if usable.sum() < 5:
        usable = np.isfinite(profile) & (profile > 0)
    fit = fit_sersic_1d(radii[usable], profile[usable])
    # Refining on a very large cutout is dominated by pixels far outside
    # the object, which cost time without constraining the profile.
    region_size = int(footprint.sum() if fit_mask is None else np.asarray(fit_mask).sum())
    if refine and fit.success and region_size > max_pixels:
        refine = False
        fit.method = "profile_scan_large"
    if refine and fit.success:
        refined = fit_sersic_2d(data, centre, axis_ratio, position_angle, footprint,
                                fit, psf, fit_mask=fit_mask, r_half=r_half, noise=sigma)
        if refined.success and np.isfinite(refined.n):
            return refined
    fit.axis_ratio = float(axis_ratio)
    fit.position_angle = float(position_angle)
    return fit
