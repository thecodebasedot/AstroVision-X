"""Variability statistics for light curves.

The question these answer is: "did this object actually change, or is the
scatter just noise?"  Each statistic attacks it differently -- against the
quoted errors, against the curve's own scatter, or against correlations
between consecutive epochs -- and together they are far harder to fool than
any one of them.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.numeric import MAD_TO_SIGMA
from ..core.types import LightCurve


def reduced_chi2(curve: LightCurve) -> float:
    """Chi-squared per degree of freedom against a constant-brightness model.

    Values near 1 mean the scatter is consistent with the errors; much
    larger means genuine variability (or underestimated errors).
    """
    clean = curve.clean()
    if len(clean) < 2 or clean.errors is None:
        return float("nan")
    errors = np.where(clean.errors > 0, clean.errors, np.nan)
    weights = 1.0 / errors ** 2
    good = np.isfinite(weights)
    if good.sum() < 2:
        return float("nan")
    weighted_mean = float(np.sum(clean.fluxes[good] * weights[good]) / np.sum(weights[good]))
    chi2 = float(np.sum(((clean.fluxes[good] - weighted_mean) ** 2) * weights[good]))
    return chi2 / max(good.sum() - 1, 1)


def stetson_j(curve: LightCurve) -> float:
    """Stetson's J index from consecutive-epoch correlations.

    A real variable brightens and fades coherently, so neighbouring epochs
    deviate from the mean in the *same* direction; noise does not.  That
    makes J robust to a few bad measurements in a way plain scatter is not.
    """
    clean = curve.clean()
    n = len(clean)
    if n < 4:
        return float("nan")
    errors = clean.errors
    if errors is None or not np.isfinite(errors).any() or np.all(errors <= 0):
        errors = np.full(n, max(MAD_TO_SIGMA * np.median(
            np.abs(clean.fluxes - np.median(clean.fluxes))), 1e-9))
    mean = float(np.mean(clean.fluxes))
    delta = np.sqrt(n / max(n - 1, 1)) * (clean.fluxes - mean) / np.maximum(errors, 1e-9)
    pairs = delta[:-1] * delta[1:]
    return float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / max(len(pairs), 1))


def median_absolute_deviation(curve: LightCurve) -> float:
    clean = curve.clean()
    if len(clean) < 2:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(clean.fluxes - np.median(clean.fluxes))))


def fractional_variability(curve: LightCurve) -> float:
    """Excess variance beyond the measurement errors, as a fraction of the mean."""
    clean = curve.clean()
    if len(clean) < 3:
        return float("nan")
    mean = float(np.mean(clean.fluxes))
    if abs(mean) < 1e-12:
        return float("nan")
    variance = float(np.var(clean.fluxes, ddof=1))
    noise = (float(np.mean(clean.errors ** 2)) if clean.errors is not None
             and np.isfinite(clean.errors).all() else 0.0)
    excess = variance - noise
    if excess <= 0:
        return 0.0
    return float(np.sqrt(excess) / abs(mean))


def amplitude(curve: LightCurve, percentile: float = 5.0) -> float:
    """Robust peak-to-peak amplitude, in flux units."""
    clean = curve.clean()
    if len(clean) < 3:
        return float("nan")
    low, high = np.percentile(clean.fluxes, [percentile, 100 - percentile])
    return float(high - low)


def skewness(curve: LightCurve) -> float:
    """Third moment: eruptive variables rise sharply and decay slowly."""
    clean = curve.clean()
    if len(clean) < 3:
        return float("nan")
    values = clean.fluxes
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return 0.0
    return float(np.mean(((values - values.mean()) / sigma) ** 3))


def kurtosis(curve: LightCurve) -> float:
    """Fourth moment (excess): distinguishes flares from smooth modulation."""
    clean = curve.clean()
    if len(clean) < 4:
        return float("nan")
    values = clean.fluxes
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return 0.0
    return float(np.mean(((values - values.mean()) / sigma) ** 4) - 3.0)


def beyond_1std(curve: LightCurve) -> float:
    """Fraction of epochs more than one standard deviation from the mean."""
    clean = curve.clean()
    if len(clean) < 3:
        return float("nan")
    sigma = float(np.std(clean.fluxes))
    if sigma <= 1e-12:
        return 0.0
    return float(np.mean(np.abs(clean.fluxes - clean.fluxes.mean()) > sigma))


def von_neumann_eta(curve: LightCurve) -> float:
    """Von Neumann ratio: below 2 means consecutive epochs are correlated."""
    clean = curve.clean()
    if len(clean) < 3:
        return float("nan")
    variance = float(np.var(clean.fluxes, ddof=1))
    if variance <= 1e-12:
        return float("nan")
    return float(np.mean(np.diff(clean.fluxes) ** 2) / variance)


def linear_trend(curve: LightCurve) -> float:
    """Slope of a straight-line fit, in flux per unit time."""
    clean = curve.clean()
    if len(clean) < 3 or clean.baseline <= 0:
        return float("nan")
    return float(np.polyfit(clean.times, clean.fluxes, 1)[0])


def variability_features(curve: LightCurve) -> Dict[str, float]:
    """All variability statistics for one light curve."""
    return {
        "n_epochs": float(len(curve)),
        "baseline": float(curve.baseline),
        "mean_flux": float(np.mean(curve.fluxes)) if len(curve) else float("nan"),
        "median_flux": float(np.median(curve.fluxes)) if len(curve) else float("nan"),
        "std_flux": float(np.std(curve.fluxes)) if len(curve) else float("nan"),
        "mad": median_absolute_deviation(curve),
        "reduced_chi2": reduced_chi2(curve),
        "stetson_j": stetson_j(curve),
        "fractional_variability": fractional_variability(curve),
        "amplitude": amplitude(curve),
        "skewness": skewness(curve),
        "kurtosis": kurtosis(curve),
        "beyond_1std": beyond_1std(curve),
        "von_neumann_eta": von_neumann_eta(curve),
        "linear_trend": linear_trend(curve),
    }


def variability_score(curve: LightCurve, threshold: float = 3.0) -> float:
    """A single 0-1 score for "how likely is this object genuinely variable".

    Each available statistic votes; the score is the fraction of votes that
    clear their own significance threshold, so an object measured with no
    error bars is judged only on the statistics that do not need them.
    """
    features = variability_features(curve)
    votes = []
    chi2 = features["reduced_chi2"]
    if np.isfinite(chi2):
        votes.append(min(chi2 / max(threshold, 1e-6), 1.0))
    stetson = features["stetson_j"]
    if np.isfinite(stetson):
        votes.append(min(max(stetson, 0.0) / max(threshold, 1e-6), 1.0))
    fractional = features["fractional_variability"]
    if np.isfinite(fractional):
        votes.append(min(fractional / 0.05, 1.0))
    eta = features["von_neumann_eta"]
    if np.isfinite(eta):
        # Smooth modulation drives eta well below 2; pure noise sits at 2.
        votes.append(float(np.clip((2.0 - eta) / 1.2, 0.0, 1.0)))
    if not votes:
        return 0.0
    return float(np.clip(np.mean(votes), 0.0, 1.0))
