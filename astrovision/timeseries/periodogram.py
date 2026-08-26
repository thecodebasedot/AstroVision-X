"""Period searching with the Lomb-Scargle periodogram.

Astronomical time series are almost never evenly sampled -- weather, day and
night, and telescope scheduling see to that -- so an FFT is not applicable.
Lomb-Scargle fits a sinusoid at each trial frequency instead, which handles
arbitrary sampling.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.types import LightCurve

log = get_logger("timeseries.periodogram")


def lomb_scargle(times: np.ndarray, values: np.ndarray, frequencies: np.ndarray,
                 errors: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalised Lomb-Scargle power at each trial frequency.

    Uses Astropy's implementation when available; otherwise evaluates the
    classical generalised formulation directly.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if errors is not None:
        errors = np.asarray(errors, dtype=float)[good]
    if t.size < 4 or f.size == 0:
        return np.zeros_like(f)

    astropy_ts = try_import("astropy.timeseries")
    if astropy_ts is not None:
        try:
            model = astropy_ts.LombScargle(t, y, dy=errors)
            return np.asarray(model.power(f), dtype=float)
        except Exception as exc:  # pragma: no cover - degenerate inputs
            log.debug("astropy Lomb-Scargle failed (%s); using the built-in version", exc)

    weights = np.ones_like(y)
    if errors is not None and np.all(np.isfinite(errors)) and np.all(errors > 0):
        weights = 1.0 / errors ** 2
    weights = weights / weights.sum()
    mean = float(np.sum(weights * y))
    centred = y - mean
    variance = float(np.sum(weights * centred ** 2))
    if variance <= 0:
        return np.zeros_like(f)

    power = np.empty_like(f)
    for i, frequency in enumerate(f):
        omega = 2.0 * np.pi * frequency
        cos = np.cos(omega * t)
        sin = np.sin(omega * t)
        # The tau offset makes the sine and cosine terms orthogonal, which
        # is what distinguishes Lomb-Scargle from a naive periodogram.
        sin2 = float(np.sum(weights * 2 * sin * cos))
        cos2 = float(np.sum(weights * (cos ** 2 - sin ** 2)))
        tau = 0.5 * np.arctan2(sin2, cos2) / omega
        ct = np.cos(omega * (t - tau))
        st = np.sin(omega * (t - tau))
        cc = float(np.sum(weights * ct ** 2))
        ss = float(np.sum(weights * st ** 2))
        yc = float(np.sum(weights * centred * ct))
        ys = float(np.sum(weights * centred * st))
        term = 0.0
        if cc > 1e-12:
            term += yc ** 2 / cc
        if ss > 1e-12:
            term += ys ** 2 / ss
        power[i] = 0.5 * term / variance * 2.0
    return np.clip(power, 0.0, 1.0)


def frequency_grid(baseline: float, min_period: float = 0.02,
                   max_period: float = 100.0, n_frequencies: int = 2000,
                   oversample: float = 5.0) -> np.ndarray:
    """Trial frequencies spanning the periods a baseline can actually resolve."""
    baseline = max(float(baseline), 1e-6)
    f_min = max(1.0 / max(float(max_period), 1e-6), 1.0 / (oversample * baseline))
    f_max = 1.0 / max(float(min_period), 1e-6)
    if f_max <= f_min:
        f_max = f_min * 10.0
    return np.linspace(f_min, f_max, max(16, int(n_frequencies)))


def false_alarm_probability(power: float, n_samples: int, n_frequencies: int) -> float:
    """Probability that pure noise would produce this peak somewhere.

    Uses the standard analytic expression for normalised Lomb-Scargle power
    with an independent-frequency correction -- adequate for triage, though
    a bootstrap is preferable before publishing anything.
    """
    if not np.isfinite(power) or n_samples < 4:
        return float("nan")
    power = float(np.clip(power, 0.0, 1.0 - 1e-12))
    single = (1.0 - power) ** ((n_samples - 3) / 2.0)
    independent = max(1, int(n_frequencies))
    return float(np.clip(1.0 - (1.0 - single) ** independent, 0.0, 1.0))


def find_period(curve: LightCurve, min_period: float = 0.02, max_period: float = 100.0,
                n_frequencies: int = 2000) -> Dict[str, float]:
    """Search for a dominant period; returns the peak and its significance."""
    clean = curve.clean()
    empty = {"period": float("nan"), "frequency": float("nan"), "power": 0.0,
             "false_alarm_probability": float("nan"), "n_epochs": float(len(clean))}
    if len(clean) < 5 or clean.baseline <= 0:
        return empty

    frequencies = frequency_grid(clean.baseline, min_period,
                                 min(max_period, clean.baseline * 2.0), n_frequencies)
    power = lomb_scargle(clean.times, clean.fluxes, frequencies, clean.errors)
    if not np.isfinite(power).any():
        return empty
    peak = int(np.argmax(power))
    best_frequency = float(frequencies[peak])
    if best_frequency <= 0:
        return empty

    # Refine with a parabola through the three points around the peak.
    if 0 < peak < len(power) - 1:
        y0, y1, y2 = power[peak - 1], power[peak], power[peak + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-12:
            offset = 0.5 * (y0 - y2) / denominator
            step = frequencies[1] - frequencies[0]
            best_frequency = float(frequencies[peak] + np.clip(offset, -1, 1) * step)

    return {
        "period": float(1.0 / best_frequency),
        "frequency": best_frequency,
        "power": float(power[peak]),
        "false_alarm_probability": false_alarm_probability(
            float(power[peak]), len(clean), len(frequencies)),
        "n_epochs": float(len(clean)),
    }


def phase_fold(curve: LightCurve, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a light curve on ``period``; returns ``(phase, flux)`` sorted by phase."""
    clean = curve.clean()
    if period <= 0 or len(clean) == 0:
        return np.array([]), np.array([])
    phase = ((clean.times - clean.times[0]) / period) % 1.0
    order = np.argsort(phase)
    return phase[order], clean.fluxes[order]
