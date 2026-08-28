"""Measuring emission and absorption lines, and what the measurement costs.

A line flux is a small number of things: an amplitude, a centre, a width, and
a continuum to measure them against. Getting each one wrong has a signature:

* **The continuum.** An equivalent width is a ratio to the continuum, so a
  continuum estimated 5 % high makes every equivalent width 5 % small. Here the
  continuum is fitted locally, in windows on either side of the line and not
  under it, because a window that includes the line is a window the line has
  raised.
* **Blends.** H-alpha sits 15 Angstroms from [N II] 6584 and 21 from [N II]
  6548. At a typical resolution those are one feature, and fitting a single
  Gaussian to it gives an H-alpha flux 30-60 % too high -- which propagates
  straight into every diagnostic ratio that uses it. Lines closer than a few
  resolution elements are therefore fitted *together*, sharing a redshift and
  a width, which is also what the physics says: the same gas emits them.
* **Non-detections.** A line that is not there still returns a fitted
  amplitude, and its sign is decided by noise. Reporting that number as a flux
  is how a diagnostic diagram acquires points in impossible places. Below a
  set significance the fit returns an upper limit instead of a value, and the
  code that consumes it can tell the difference.

The fitter is linear least squares over amplitudes with a small search over
the shared width and offset, which is enough here and avoids depending on an
optimiser for something so small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import mad_std
from .calibrate import fit_continuum
from .templates import LINES, Spectrum1D, velocity_sigma

log = get_logger("spectra.lines")

#: Lines closer together than this many times the *widest width the fit is
#: allowed to try* are fitted as one group with a shared width and velocity.
#:
#: Measured against the resolution instead -- the obvious choice -- this
#: threshold put [N II] 6584 just outside H-alpha's group, 20.7 Angstroms
#: apart against a 20 Angstrom threshold. Fitted alone, that line widened to
#: the top of the velocity search and absorbed H-alpha's flux, and the
#: [N II]/H-alpha ratio came out inverted: it *fell* as the simulated
#: ionisation rose. The threshold has to describe what the fit can confuse,
#: which is set by the widest line it may fit, not by the instrument.
BLEND_SEPARATION = 4.0

#: Amplitude significance below which a line is reported as an upper limit.
DETECTION_SIGMA = 3.0

#: Balmer lines, which sit in the stellar absorption troughs of their own
#: series.  Every one of them is fitted with a broad absorption component
#: underneath the narrow emission.
BALMER = ("H alpha", "H beta", "H gamma", "H delta")

#: Width, km/s, of that stellar absorption component.  It is the velocity
#: dispersion of the stars, not of the gas, so it is far broader than the
#: emission line sitting in it -- which is exactly what makes the two
#: separable in a single linear fit.
STELLAR_ABSORPTION_KM_S = 350.0


@dataclass
class LineMeasurement:
    """One measured line."""

    name: str
    rest_wavelength: float
    observed_wavelength: float = float("nan")
    flux: float = float("nan")             # integrated, continuum-subtracted
    flux_error: float = float("nan")
    equivalent_width: float = float("nan")  # positive for emission
    continuum: float = float("nan")
    sigma: float = float("nan")            # Gaussian sigma, Angstroms
    velocity_width_km_s: float = float("nan")
    significance: float = float("nan")
    detected: bool = False
    upper_limit: float = float("nan")
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "rest_wavelength": self.rest_wavelength,
                "flux": self.flux, "flux_error": self.flux_error,
                "equivalent_width": self.equivalent_width,
                "significance": self.significance, "detected": self.detected,
                "upper_limit": self.upper_limit,
                "velocity_width_km_s": self.velocity_width_km_s,
                "flags": list(self.flags)}


def group_lines(names: Sequence[str], redshift: float, resolution: float,
                max_velocity_km_s: float = 600.0) -> List[List[str]]:
    """Split lines into blended groups.

    Two lines belong together when the widest profile the fit may give them
    would overlap. H-alpha and [N II] 6584 always do; H-alpha and H-beta
    never do.

    >>> groups = group_lines(["H beta", "H alpha", "[N II] 6584"], 0.0, 5.0)
    >>> [len(g) for g in groups]
    [1, 2]
    """
    ordered = sorted(names, key=lambda n: LINES[n])
    instrumental = float(resolution) / 2.3548
    groups: List[List[str]] = []
    for name in ordered:
        centre = LINES[name] * (1.0 + redshift)
        if groups:
            last = LINES[groups[-1][-1]] * (1.0 + redshift)
            widest = math.hypot(instrumental,
                                velocity_sigma(centre, max_velocity_km_s))
            if abs(centre - last) < BLEND_SEPARATION * widest:
                groups[-1].append(name)
                continue
        groups.append([name])
    return groups


def _local_continuum(spectrum: Spectrum1D, centre: float, half_width: float,
                     gap: float) -> Tuple[float, float]:
    """Continuum level and its noise, from windows either side of a line."""
    offset = np.abs(spectrum.wavelength - centre)
    band = (offset > gap) & (offset < gap + half_width) & spectrum.good
    if band.sum() < 5:
        band = (offset < gap + half_width) & spectrum.good
    if band.sum() < 3:
        return float("nan"), float("nan")
    values = spectrum.flux[band]
    return float(np.median(values)), float(mad_std(values))


def fit_lines(spectrum: Spectrum1D, redshift: float,
              names: Sequence[str] = tuple(LINES),
              resolution: float = 5.0,
              velocity_range: Tuple[float, float] = (30.0, 600.0),
              continuum: Optional[np.ndarray] = None) -> Dict[str, LineMeasurement]:
    """Fit every named line at a known redshift.

    The redshift is an input, not a free parameter: a line fit that solves for
    its own redshift will find one, and on a noisy spectrum it finds the noise.
    The velocity width *is* free, over a range, because it is a real property
    of the gas and because the instrument's own resolution is folded into it.
    """
    if continuum is None:
        continuum = fit_continuum(spectrum)
    continuum = np.asarray(continuum, dtype=float)
    residual = spectrum.flux - continuum
    if spectrum.error is not None:
        errors = np.where(np.isfinite(spectrum.error) & (spectrum.error > 0),
                          spectrum.error, np.nan)
        noise = float(np.nanmedian(errors))
    else:
        noise = float(mad_std(residual[np.isfinite(residual)]))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0

    usable = [n for n in names if n in LINES]
    results: Dict[str, LineMeasurement] = {}
    dispersion = spectrum.dispersion()

    for group in group_lines(usable, redshift, resolution,
                             max_velocity_km_s=velocity_range[1]):
        centres = np.array([LINES[n] * (1.0 + redshift) for n in group])
        span = float(centres.max() - centres.min())
        half = max(4.0 * resolution + span, 6.0 * dispersion)
        window = (np.abs(spectrum.wavelength - centres.mean()) < half) & spectrum.good
        if window.sum() < len(group) + 3:
            for name in group:
                results[name] = LineMeasurement(name, LINES[name],
                                                flags=["outside_the_spectrum"])
            continue

        grid = spectrum.wavelength[window]
        values = residual[window]
        level, level_noise = _local_continuum(spectrum, float(centres.mean()),
                                              6.0 * resolution, half)
        pixel_noise = noise
        if np.isfinite(level_noise) and level_noise > 0:
            pixel_noise = max(level_noise, noise * 0.5)

        # A Balmer emission line sits inside the absorption trough of the same
        # transition in the galaxy's own stars, and the trough is far wider
        # than the line. Measuring the emission against a smooth continuum
        # therefore measures emission *minus* absorption, and the error does
        # not cancel in a ratio: measured here, [O III]/H-beta came out 19 %
        # high at every ionisation, which is enough to move a galaxy across a
        # diagnostic boundary. Adding a broad component under each Balmer line
        # -- free in amplitude, so the fit can also decide there is none --
        # measures the emission above the trough instead.
        # Only for a Balmer line that is *alone* in its group. The trough is
        # only about 1.5 times wider than the emission line sitting in it, so
        # when another emission line is 20 Angstroms away -- [N II] beside
        # H-alpha -- the trough and that neighbour trade off freely: adding
        # the component moved the measured [N II]/H-alpha from 0.213, against
        # a true 0.215, to 0.31. H-beta, H-gamma and H-delta have no such
        # neighbour, and that is where the correction is both needed and
        # identifiable.
        absorption = ([0] if len(group) == 1 and group[0] in BALMER else [])

        best = None
        instrumental = resolution / 2.3548
        for velocity in np.linspace(velocity_range[0], velocity_range[1], 28):
            sigmas = np.hypot(instrumental,
                              np.array([velocity_sigma(c, velocity) for c in centres]))
            columns = [np.exp(-0.5 * ((grid - c) / s) ** 2)
                       for c, s in zip(centres, sigmas)]
            for i in absorption:
                wide = math.hypot(instrumental,
                                  velocity_sigma(centres[i], STELLAR_ABSORPTION_KM_S))
                columns.append(np.exp(-0.5 * ((grid - centres[i]) / wide) ** 2))
            design = np.stack(columns, axis=1)
            solution, *_ = np.linalg.lstsq(design, values, rcond=None)
            model = design @ solution
            chi2 = float(np.sum((values - model) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, solution, sigmas, design)

        chi2, amplitudes, sigmas, design = best
        # Amplitude errors from the design matrix: the diagonal of
        # (A^T A)^-1 scaled by the pixel noise.  For a blend this correctly
        # inflates the error of each component, because they trade off.
        try:
            covariance = np.linalg.inv(design.T @ design) * pixel_noise ** 2
            amplitude_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        except np.linalg.LinAlgError:                    # pragma: no cover
            amplitude_errors = np.full(len(group), np.nan)

        for i, name in enumerate(group):
            sigma = float(sigmas[i])
            area = math.sqrt(2.0 * math.pi) * sigma
            flux = float(amplitudes[i]) * area
            flux_error = float(amplitude_errors[i]) * area
            significance = (abs(flux) / flux_error
                            if flux_error > 0 else float("nan"))
            measurement = LineMeasurement(
                name=name, rest_wavelength=LINES[name],
                observed_wavelength=float(centres[i]),
                flux=flux, flux_error=flux_error, continuum=level,
                sigma=sigma, significance=significance,
                velocity_width_km_s=float(299792.458 * sigma / centres[i]))
            measurement.detected = bool(np.isfinite(significance)
                                        and significance >= DETECTION_SIGMA
                                        and flux > 0)
            fitted_velocity = measurement.velocity_width_km_s
            if fitted_velocity >= 0.98 * velocity_range[1]:
                # A width pinned to the top of the search is not a measured
                # width; it is the fit trying to leave the range it was given,
                # usually because it is absorbing a neighbour.
                measurement.flags.append("width_at_search_limit")
                measurement.detected = False
            if np.isfinite(level) and level > 0:
                measurement.equivalent_width = flux / level
            if not measurement.detected:
                measurement.upper_limit = float(DETECTION_SIGMA * flux_error)
                measurement.flags.append("not_detected")
            if len(group) > 1:
                measurement.flags.append("blended")
            if i in absorption:
                trough = float(amplitudes[len(group) + absorption.index(i)])
                if trough < 0:
                    measurement.flags.append("stellar_absorption_corrected")
            results[name] = measurement

    return results


def line_ratio(lines: Dict[str, LineMeasurement], numerator: str,
               denominator: str) -> Tuple[float, float, str]:
    """A ratio of two line fluxes, with its error and its status.

    Returns ``(value, error, status)`` where status is ``measured``,
    ``upper_limit``, ``lower_limit`` or ``unavailable``. The distinction
    matters: a ratio built from a non-detection is a limit, and a diagnostic
    diagram that plots limits as points draws conclusions the data do not
    support.
    """
    top = lines.get(numerator)
    bottom = lines.get(denominator)
    if top is None or bottom is None:
        return float("nan"), float("nan"), "unavailable"
    if bottom.detected and top.detected:
        value = top.flux / bottom.flux
        error = abs(value) * math.hypot(
            top.flux_error / abs(top.flux) if top.flux else np.inf,
            bottom.flux_error / abs(bottom.flux) if bottom.flux else np.inf)
        return float(value), float(error), "measured"
    if bottom.detected and not top.detected:
        return float(top.upper_limit / bottom.flux), float("nan"), "upper_limit"
    if top.detected and not bottom.detected:
        return float(top.flux / bottom.upper_limit), float("nan"), "lower_limit"
    return float("nan"), float("nan"), "unavailable"


def measure_velocity_dispersion(spectrum: Spectrum1D, redshift: float,
                                template: Spectrum1D,
                                resolution: float = 5.0) -> Dict[str, Any]:
    """Stellar velocity dispersion, by broadening a template until it matches.

    The template is convolved with a Gaussian of increasing width and compared
    with the observed absorption lines; the best width, with the instrumental
    resolution removed in quadrature, is the dispersion.

    Two honest limits are reported rather than hidden. The measurement cannot
    resolve a dispersion below the instrumental resolution -- at 5 Angstroms
    that is about 130 km/s at 5000 Angstroms -- and it is degenerate with
    template mismatch: an old template fitted to a younger galaxy has
    intrinsically narrower lines and the fit makes up the difference in
    broadening.
    """
    from .redshift import log_grid, prepare

    ok = spectrum.good
    if ok.sum() < 100:
        return {"sigma_km_s": float("nan"), "reliable": False,
                "reason": "too few usable pixels"}
    low = float(np.nanmin(spectrum.wavelength[ok]))
    high = float(np.nanmax(spectrum.wavelength[ok]))
    grid = log_grid(low, high, 25.0)
    observed = prepare(spectrum, grid)
    rest = template.redshifted(redshift)
    step_km_s = 25.0

    best = None
    trials = np.arange(0.0, 500.0, 20.0)
    for width in trials:
        if width <= 0:
            broadened = prepare(rest, grid)
        else:
            pixels = width / step_km_s
            size = int(max(4 * pixels, 3)) | 1
            offsets = np.arange(size) - size // 2
            kernel = np.exp(-0.5 * (offsets / max(pixels, 1e-6)) ** 2)
            kernel = kernel / kernel.sum()
            broadened = np.convolve(prepare(rest, grid), kernel, mode="same")
        spread = float(np.std(broadened))
        if spread <= 0:
            continue
        scale = float(np.dot(observed, broadened) / np.dot(broadened, broadened))
        chi2 = float(np.sum((observed - scale * broadened) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, float(width))

    if best is None:
        return {"sigma_km_s": float("nan"), "reliable": False,
                "reason": "no trial width fitted"}
    width = best[1]
    instrumental = 299792.458 * (resolution / 2.3548) / 5000.0
    resolved = width > instrumental
    intrinsic = math.sqrt(max(width ** 2 - instrumental ** 2, 0.0))
    return {"sigma_km_s": intrinsic if resolved else float("nan"),
            "fitted_width_km_s": width,
            "instrumental_km_s": float(instrumental),
            "reliable": bool(resolved),
            "reason": ("measured by template broadening" if resolved else
                       f"the fitted width {width:.0f} km/s is at or below the "
                       f"instrumental {instrumental:.0f} km/s; the dispersion "
                       "is unresolved and only an upper limit")}
