"""Redshifts from a spectrum: cross-correlation, lines, and when to refuse.

A spectroscopic redshift is the most reliable distance measurement in
extragalactic astronomy, and the reason is worth stating: it is a *pattern*
match, not a single measurement. Twenty absorption lines all shifted by the
same fraction is an enormously stronger statement than any one of them, and
the cross-correlation is the arithmetic that adds them up.

The implementation follows Tonry & Davis (1979). Both spectra are put on a
grid uniform in ``log lambda``, where a redshift is a pure *shift* -- since
``log(lambda(1+z)) = log(lambda) + log(1+z)`` -- rather than a stretch, so one
correlation covers every redshift at once. The peak of the correlation gives
the shift; its height relative to the noise in the rest of the correlation
gives the ``R`` statistic, which is what separates a measurement from a
coincidence.

Three things this module refuses to do:

**It does not report the peak without R.** A cross-correlation always has a
maximum. On a pure-noise spectrum it is as tall as noise allows and sits at an
arbitrary redshift, and reporting it as a redshift is how surveys accumulate
catastrophic outliers. Measured here, the R statistic separates the two
populations cleanly, and the threshold is set from that measurement.

**It does not average incompatible answers.** When the absorption-line
cross-correlation and the emission lines disagree by more than their errors,
that is information -- usually a misidentified single line, sometimes two
objects on the slit -- and the disagreement is reported rather than split.

**It does not silently prefer a galaxy template for a star.** Stars are in the
template set and are only searched near rest, so a spectrum best matched by a
stellar template is reported as a star rather than given a redshift.

One thing it does **not** claim: that the winning template classifies the
object. It does not. A quasar in these tests is matched by the starburst
template -- its narrow [O III] lines correlate better than its broad Balmer
lines do, at every continuum window tried -- and the redshift comes out right
regardless, to 0.0003. The correlation locates features; naming the object is
what the diagnostics in :mod:`~astrovision.spectra.diagnostics` are for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import mad_std
from .calibrate import normalise
from .templates import LINES, Spectrum1D, standard_templates

log = get_logger("spectra.redshift")

#: Below this Tonry-Davis R the correlation peak is not a redshift.
#:
#: Measured rather than adopted. Forty pure-noise spectra gave R between 3.6
#: and 6.6; correct redshifts on real spectra start at 5.6 and have a median
#: of 13. A threshold of 7 rejects every one of the noise spectra while
#: keeping 21 of 24 correct measurements at signal-to-noise 3 to 8.
MIN_R = 7.0

#: The correlation peak must beat the best peak at a *different* redshift by
#: this factor in R.
#:
#: R alone does not do this job, and the measurement says so plainly: at low
#: signal-to-noise, wrong redshifts reached R = 24 -- as strong a correlation
#: as the right answers -- because a catastrophic failure is not a weak match,
#: it is a confident match to the wrong feature. What distinguishes them is
#: whether a *rival* explanation exists. Requiring the winner to lead the best
#: alternative by 30 % in R removes 63 % of the wrong answers and 21 % of the
#: right ones.
MIN_PEAK_RATIO = 1.3

#: Velocity difference, km/s, above which two redshift estimates for the same
#: object are called inconsistent rather than combined.
CONSISTENCY_KM_S = 600.0

#: Stellar templates are only searched within this velocity of rest.  A star
#: has a radial velocity of a few hundred km/s, not a redshift, and a stellar
#: template allowed to roam the full range will happily match a galaxy's
#: absorption lines at some other redshift -- measured here, a G-star template
#: won the fit for a galaxy at z = 0.39, reporting the right redshift for
#: entirely the wrong reason.
STAR_VELOCITY_LIMIT_KM_S = 3000.0

C_KM_S = 299792.458


@dataclass
class RedshiftResult:
    """One redshift measurement, with everything needed to distrust it."""

    z: float = float("nan")
    z_error: float = float("nan")
    r_statistic: float = float("nan")
    template: str = ""
    method: str = ""
    z_emission: float = float("nan")
    n_emission_lines: int = 0
    second_z: float = float("nan")         # next-best correlation peak
    second_r: float = float("nan")
    peak_ratio: float = float("nan")       # R of the winner over R of the rival
    reliable: bool = False
    is_star: bool = False
    flags: List[str] = field(default_factory=list)
    reason: str = ""

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    @property
    def velocity_error_km_s(self) -> float:
        if not np.isfinite(self.z_error):
            return float("nan")
        return float(C_KM_S * self.z_error / (1.0 + max(self.z, 0.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {"z": self.z, "z_error": self.z_error,
                "r_statistic": self.r_statistic, "template": self.template,
                "method": self.method, "z_emission": self.z_emission,
                "n_emission_lines": self.n_emission_lines,
                "second_z": self.second_z, "peak_ratio": self.peak_ratio,
                "reliable": self.reliable,
                "is_star": self.is_star, "flags": list(self.flags),
                "reason": self.reason}


def log_grid(low: float, high: float, velocity_step_km_s: float = 50.0
             ) -> np.ndarray:
    """A wavelength grid uniform in log wavelength.

    The step is quoted as a velocity because that is what it is: a fixed
    fractional wavelength step is a fixed velocity step, which is what makes
    the correlation shift-invariant in redshift.

    >>> grid = log_grid(4000.0, 4100.0, 100.0)
    >>> float(np.round(299792.458 * (grid[1] / grid[0] - 1.0), 1))
    100.0
    """
    ratio = 1.0 + float(velocity_step_km_s) / C_KM_S
    n = int(math.ceil(math.log(float(high) / float(low)) / math.log(ratio))) + 1
    return float(low) * ratio ** np.arange(max(n, 2))


def prepare(spectrum: Spectrum1D, grid: np.ndarray,
            continuum_window: int = 151) -> np.ndarray:
    """Continuum-subtract, resample to ``grid``, and apodise.

    The apodisation -- a cosine taper over the outer tenth of the spectrum --
    is not cosmetic. A correlation of two abruptly truncated arrays is
    dominated by the discontinuity at the ends, which produces a spurious
    peak at zero shift: the classic way to measure a redshift of exactly zero
    for everything.
    """
    flattened = normalise(spectrum, window=continuum_window)
    # Interpolating across a masked gap would bridge it with a straight line,
    # which is a feature the template does not have.  Masked pixels are set to
    # the continuum instead, so they contribute nothing to the correlation.
    filled = Spectrum1D(flattened.wavelength,
                        np.where(flattened.good, flattened.flux, 1.0),
                        flattened.error, None, flattened.meta)
    resampled = filled.resample(grid)
    values = np.where(np.isfinite(resampled.flux), resampled.flux, 1.0) - 1.0

    n = values.size
    taper = np.ones(n)
    edge = max(int(0.1 * n), 1)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]
    values = values * taper
    spread = float(np.std(values))
    return values / spread if spread > 0 else values


def cross_correlate(observed: np.ndarray, template: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation and the lag of each element.

    Computed through the FFT, which for arrays of this length is not about
    speed but about doing every lag at once -- and the lags are the redshifts,
    so a partial correlation is a partial redshift search.
    """
    n = observed.size
    size = int(2 ** math.ceil(math.log2(max(2 * n, 2))))
    fa = np.fft.rfft(observed, size)
    fb = np.fft.rfft(template, size)
    correlation = np.fft.irfft(fa * np.conj(fb), size)
    correlation = np.concatenate([correlation[-(n - 1):], correlation[:n]])
    lags = np.arange(-(n - 1), n)
    norm = math.sqrt(float(np.sum(observed ** 2) * np.sum(template ** 2)))
    return (correlation / norm if norm > 0 else correlation), lags


def tonry_davis_r(correlation: np.ndarray, peak_index: int,
                  exclude: int = 30) -> float:
    """The height of the correlation peak in units of its **antisymmetric** noise.

    This is the construction from the original paper, and the antisymmetric
    part is the whole point. A true match produces a symmetric peak: the
    correlation falls away at the same rate on both sides, because it is the
    autocorrelation of the template's own features. Noise produces a peak that
    is lopsided, because there is no feature under it. So the noise is
    measured as

    .. math::

        \sigma_a^2 = \frac{1}{2N} \sum_\delta
                      [c(p + \delta) - c(p - \delta)]^2

    and ``R = h / (sqrt(2) sigma_a)``.

    A simpler statistic -- peak height over the scatter of the correlation
    away from the peak -- was tried first and does not work. On pure noise it
    reached R = 18.7 against a broad-lined template, because the correlation
    of noise with a smooth template is itself smooth, so its scatter is small
    and the ratio is large. The asymmetry test does not have that failure
    mode: it asks about the shape of the peak, not the roughness of the
    background.
    """
    values = np.asarray(correlation, dtype=float)
    n = values.size
    if n < 5 or not 0 <= peak_index < n:
        return float("nan")
    # The antisymmetric noise is measured over the whole correlation, not a
    # window around the peak.  Within a narrow window even a noise peak looks
    # symmetric -- the correlation of noise with a smooth template is smooth
    # on the scale of the template's features -- and R comes out large for
    # everything.  Measured with a 30-lag window, ten pure-noise spectra all
    # passed; measured over the full overlap, none does.
    carries_signal = np.flatnonzero(np.abs(values) > 1e-12)
    if carries_signal.size < 2 * max(exclude, 5):
        return float("nan")
    low, high = int(carries_signal[0]), int(carries_signal[-1])
    if not low <= peak_index <= high:
        return float("nan")
    reach = min(peak_index - low, high - peak_index)
    if reach < max(exclude, 5):
        return float("nan")
    offsets = np.arange(1, reach + 1)
    antisymmetric = 0.5 * (values[peak_index + offsets] - values[peak_index - offsets])
    sigma_a = float(np.sqrt(np.mean(antisymmetric ** 2)))
    if not np.isfinite(sigma_a) or sigma_a <= 0:
        return float("nan")
    height = float(values[peak_index])
    return float(height / (math.sqrt(2.0) * sigma_a))


def _peak_shift(correlation: np.ndarray, index: int) -> float:
    """Sub-sample peak position by a parabola through three points."""
    if index <= 0 or index >= correlation.size - 1:
        return float(index)
    left, centre, right = correlation[index - 1:index + 2]
    denominator = left - 2.0 * centre + right
    if denominator == 0:
        return float(index)
    return float(index) + float(np.clip(0.5 * (left - right) / denominator, -1, 1))


def measure_emission_redshift(spectrum: Spectrum1D,
                              min_significance: float = 5.0,
                              lines: Sequence[str] = ("[O II]", "H beta",
                                                      "[O III] 5007", "H alpha",
                                                      "[N II] 6584")
                              ) -> Tuple[float, int, List[str]]:
    """Redshift from emission lines, by asking which one they *all* agree on.

    A single emission line gives as many redshifts as there are lines it could
    be, and picking the strongest identification is how [O II] at z = 0.9
    becomes H-alpha at z = 0.25. So the redshift is not read off one line: a
    grid of trial redshifts is scored by how much line flux lands on the
    expected positions of *several* lines at once, which is only large when
    the identification is right.
    """
    flattened = normalise(spectrum, window=101)
    excess = flattened.flux - 1.0
    noise = float(mad_std(excess[np.isfinite(excess)]))
    if not np.isfinite(noise) or noise <= 0:
        return float("nan"), 0, []

    rest = np.array([LINES[name] for name in lines])
    dispersion = spectrum.dispersion()
    lowest = max(spectrum.wavelength.min() / rest.max() - 1.0, 0.0)
    highest = spectrum.wavelength.max() / rest.min() - 1.0
    if not np.isfinite(highest) or highest <= lowest:
        return float("nan"), 0, []
    trials = np.arange(lowest, highest, 0.2 * dispersion / float(np.median(rest)))

    best_score, best_z = 0.0, float("nan")
    for z in trials:
        observed = rest * (1.0 + z)
        inside = ((observed > spectrum.wavelength.min())
                  & (observed < spectrum.wavelength.max()))
        if inside.sum() < 2:
            continue
        # Sum of the flux excess at every expected line position.  Requiring
        # two lines is what makes a single strong feature insufficient.
        values = np.interp(observed[inside], spectrum.wavelength, excess)
        score = float(np.sum(np.clip(values, 0, None))) / noise
        if score > best_score:
            best_score, best_z = score, float(z)

    if not np.isfinite(best_z):
        return float("nan"), 0, []

    found: List[str] = []
    for name in lines:
        centre = LINES[name] * (1.0 + best_z)
        if not (spectrum.wavelength.min() < centre < spectrum.wavelength.max()):
            continue
        near = np.abs(spectrum.wavelength - centre) < max(3.0 * dispersion, 6.0)
        if near.sum() and float(np.max(excess[near])) > min_significance * noise:
            found.append(name)
    if len(found) < 2:
        return float("nan"), len(found), found
    return best_z, len(found), found


def measure_redshift(spectrum: Spectrum1D,
                     templates: Optional[Sequence[Spectrum1D]] = None,
                     z_min: float = -0.005, z_max: float = 1.2,
                     velocity_step_km_s: float = 50.0) -> RedshiftResult:
    """Cross-correlate against every template and report the best match.

    Every template is tried at every redshift; the winner is the one whose
    correlation peak has the highest R. Templates that are stars are included
    deliberately, so that a star produces a *star*, not a redshift.
    """
    result = RedshiftResult()
    library = list(templates) if templates is not None else standard_templates()
    if not library:
        result.reason = "no templates supplied"
        return result

    ok = spectrum.good
    if ok.sum() < 100:
        result.reason = f"only {int(ok.sum())} usable pixels"
        result.add_flag("too_few_pixels")
        return result

    low = float(np.nanmin(spectrum.wavelength[ok]))
    high = float(np.nanmax(spectrum.wavelength[ok]))
    # The grid has to reach blueward of the observed range by (1 + z_max), or
    # the rest-frame features of a high-redshift object fall off the end of
    # the template and the correlation never sees them.
    grid = log_grid(low / (1.0 + max(z_max, 0.0)), high, velocity_step_km_s)
    observed = prepare(spectrum, grid)

    step = math.log(grid[1] / grid[0])
    best: Optional[Tuple[float, float, float, str, int]] = None
    peaks: List[Tuple[float, float, str]] = []
    for template in library:
        prepared = prepare(template, grid)
        correlation, lags = cross_correlate(observed, prepared)
        redshifts = np.expm1(lags * step)
        inside = (redshifts >= z_min) & (redshifts <= z_max)
        if str(template.meta.get("kind", "")) == "star":
            limit = STAR_VELOCITY_LIMIT_KM_S / C_KM_S
            inside = inside & (np.abs(redshifts) <= limit)
        if not inside.any():
            continue
        candidates = np.flatnonzero(inside)
        index = int(candidates[int(np.argmax(correlation[candidates]))])
        # The runner-up peak of *this* template, at a genuinely different
        # redshift.  A second template's peak is not the right comparison:
        # templates resemble each other, so their peaks sit on top of one
        # another and the comparison is between two names for one answer.
        peak_z = float(np.expm1((index - (len(observed) - 1)) * step))
        elsewhere = candidates[np.abs(redshifts[candidates] - peak_z) > 0.02]
        runner = float("nan")
        if elsewhere.size:
            second = int(elsewhere[int(np.argmax(correlation[elsewhere]))])
            runner = tonry_davis_r(correlation, second)
        # R is measured on the full correlation, at the peak found inside the
        # searched range: the noise it needs lives in the lags outside that
        # range, so handing it only the searched slice leaves it nothing to
        # measure the peak against.
        r = tonry_davis_r(correlation, index)
        position = _peak_shift(correlation, index)
        z = float(np.expm1((position - (len(observed) - 1)) * step))
        name = str(template.meta.get("name", template.meta.get("kind", "?")))
        kind = str(template.meta.get("kind", ""))
        peaks.append((z, r, name))
        if best is None or (np.isfinite(r) and r > best[1]):
            best = (z, r, runner, name, kind)

    if best is None:
        result.reason = "no template produced a correlation peak in range"
        return result

    z, r, runner_r, name, kind = best
    result.z = z
    result.r_statistic = r
    result.template = name
    result.method = "cross-correlation"
    # From the template's kind, never from its name.  A name test spelt
    # ``startswith("star")`` matched the *starburst* galaxy template and
    # reported a star-forming galaxy at z = 0.12 as a star.
    result.is_star = bool(kind == "star")

    # The error from R, following the usual empirical form: the width of the
    # correlation peak divided by (1 + R).  It is not a likelihood width, and
    # it is quoted because it tracks the measured scatter, which is the only
    # property an error needs.
    if np.isfinite(r) and r > 0:
        result.z_error = float((1.0 + z) * (velocity_step_km_s / C_KM_S)
                               * 3.0 / (1.0 + r))

    result.second_r = runner_r
    if np.isfinite(r) and np.isfinite(runner_r) and runner_r > 0:
        result.peak_ratio = float(r / runner_r)
    others = sorted((p for p in peaks if abs(p[0] - z) > 0.01),
                    key=lambda item: -(item[1] if np.isfinite(item[1]) else -1))
    if others:
        result.second_z = float(others[0][0])

    z_emission, n_lines, found = measure_emission_redshift(spectrum)
    result.z_emission = z_emission
    result.n_emission_lines = n_lines

    if not np.isfinite(r) or r < MIN_R:
        result.add_flag("low_correlation")
        result.reason = (f"the best correlation reaches R = {r:.1f}, below the "
                         f"{MIN_R:.0f} that separates real matches from noise "
                         "peaks; no redshift is reported as measured")
        # An emission-line redshift can still stand on its own: it does not
        # need the continuum the correlation failed on.
        if np.isfinite(z_emission) and n_lines >= 2:
            result.z = z_emission
            result.method = "emission lines"
            result.reliable = True
            result.reason = (f"correlation failed (R = {r:.1f}) but {n_lines} "
                             f"emission lines agree on z = {z_emission:.4f}: "
                             + ", ".join(found))
            result.z_error = float(0.3 * spectrum.dispersion()
                                   / float(np.mean([LINES[f] for f in found])))
        return result

    if (np.isfinite(result.peak_ratio)
            and result.peak_ratio < MIN_PEAK_RATIO):
        result.add_flag("rival_peak")
        result.reason = (f"the correlation reaches R = {r:.1f} at z = {z:.4f}, but a "
                         f"peak at z = {result.second_z:.4f} is nearly as strong "
                         f"(ratio {result.peak_ratio:.2f}); the spectrum does not "
                         "choose between them")
        if np.isfinite(z_emission) and n_lines >= 2:
            velocity = abs(z_emission - z) * C_KM_S / (1.0 + z)
            if velocity < CONSISTENCY_KM_S:
                # The emission lines break the tie: they are an independent
                # identification, not another peak in the same correlation.
                result.reliable = True
                result.reason += (f", but {n_lines} emission lines agree with the "
                                  "correlation peak, which settles it")
                return result
        return result

    result.reliable = True
    result.reason = f"cross-correlation with the {name} template, R = {r:.1f}"

    if np.isfinite(z_emission) and n_lines >= 2:
        velocity = abs(z_emission - z) * C_KM_S / (1.0 + z)
        if velocity > CONSISTENCY_KM_S:
            result.add_flag("emission_absorption_disagree")
            result.reliable = False
            result.reason = (f"the correlation gives z = {z:.4f} and the emission "
                             f"lines give z = {z_emission:.4f}, {velocity:.0f} km/s "
                             "apart; one of the two identifications is wrong")
    if result.is_star and abs(z) < 0.002:
        result.reason = (f"best matched by the {name} template at rest; this is "
                         "a star, not a redshift")
    return result
