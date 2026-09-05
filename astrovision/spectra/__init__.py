"""Spectroscopy: extraction, calibration, redshifts, lines and classification."""

from .templates import (
    BPT_LINES,
    LINES,
    REST_GRID,
    Spectrum1D,
    galaxy_spectrum,
    quasar_spectrum,
    standard_templates,
    star_spectrum,
    supernova_spectrum,
    supernova_templates,
    velocity_sigma,
)
from .extract import (
    Trace,
    boxcar_extract,
    estimate_sky,
    extract_spectrum,
    find_trace,
    optimal_extract,
)
from .analyse import SpectrumAnalysis, analyse_frame, analyse_spectrum
from .lines import (
    DETECTION_SIGMA,
    LineMeasurement,
    fit_lines,
    group_lines,
    line_ratio,
    measure_velocity_dispersion,
)
from .diagnostics import (
    BPTClassification,
    SupernovaMatch,
    balmer_decrement,
    classify_bpt,
    classify_supernova,
    kauffmann_line,
    kewley_line,
    schawinski_line,
)
from .redshift import (
    MIN_R,
    RedshiftResult,
    cross_correlate,
    log_grid,
    measure_emission_redshift,
    measure_redshift,
    tonry_davis_r,
)
from .calibrate import (
    WavelengthSolution,
    apply_solution,
    check_against_sky_lines,
    find_peaks,
    fit_continuum,
    fit_wavelength_solution,
    match_lines,
    normalise,
    vote_for_linear_solution,
)

__all__ = [
    "Spectrum1D", "REST_GRID", "LINES", "BPT_LINES", "velocity_sigma",
    "galaxy_spectrum", "star_spectrum", "quasar_spectrum",
    "supernova_spectrum", "standard_templates", "supernova_templates",
    "Trace", "find_trace", "estimate_sky", "boxcar_extract",
    "optimal_extract", "extract_spectrum",
    "WavelengthSolution", "find_peaks", "match_lines",
    "fit_wavelength_solution", "apply_solution", "check_against_sky_lines",
    "fit_continuum", "normalise", "vote_for_linear_solution",
    "RedshiftResult", "measure_redshift", "measure_emission_redshift",
    "cross_correlate", "tonry_davis_r", "log_grid", "MIN_R",
    "LineMeasurement", "fit_lines", "line_ratio", "group_lines",
    "measure_velocity_dispersion", "DETECTION_SIGMA",
    "BPTClassification", "classify_bpt", "kauffmann_line", "kewley_line",
    "schawinski_line", "SupernovaMatch", "classify_supernova",
    "balmer_decrement",
    "SpectrumAnalysis", "analyse_spectrum", "analyse_frame",
]
