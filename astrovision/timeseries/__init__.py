"""Time-series analysis: light curves, variability and period searching."""

from .features import (
    amplitude,
    beyond_1std,
    fractional_variability,
    kurtosis,
    linear_trend,
    median_absolute_deviation,
    reduced_chi2,
    skewness,
    stetson_j,
    variability_features,
    variability_score,
    von_neumann_eta,
)
from .lightcurve import LightCurveAnalyzer, classify_variable, extract_light_curve
from .models import (
    VARIABLE_CLASSES,
    NearestCentroidVariabilityClassifier,
    SequenceClassifier,
    encode_curve,
)
from .periodogram import (
    false_alarm_probability,
    find_period,
    frequency_grid,
    lomb_scargle,
    phase_fold,
)

__all__ = [
    "LightCurveAnalyzer", "extract_light_curve", "classify_variable",
    "variability_features", "variability_score", "reduced_chi2", "stetson_j",
    "median_absolute_deviation", "fractional_variability", "amplitude",
    "skewness", "kurtosis", "beyond_1std", "von_neumann_eta", "linear_trend",
    "find_period", "lomb_scargle", "frequency_grid", "phase_fold",
    "false_alarm_probability",
    "SequenceClassifier", "NearestCentroidVariabilityClassifier",
    "VARIABLE_CLASSES", "encode_curve",
]
