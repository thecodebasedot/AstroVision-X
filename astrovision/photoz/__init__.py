"""Photometric redshifts: distance from a handful of broad-band colours.

The inference works because a galaxy spectrum has features -- the 4000
Angstrom break of an old stellar population, the emission lines of a young
one -- and those features slide through the filters as the galaxy recedes.
Fitting a library of redshifted spectra to the observed colours recovers the
redshift, with an accuracy set almost entirely by how many filters there are.

Three things this module refuses to hide: the posterior is often bimodal and
both peaks are reported; the posterior width understates the error, because
the template library does not contain the real galaxy, so a floor is added;
and with fewer than five filters the problem is underdetermined and says so.
"""

from .fit import (
    SECOND_PEAK_FRACTION,
    TEMPLATE_FLOOR,
    PhotoZLibrary,
    PhotoZResult,
    fit_catalog,
    fit_photoz,
)
from .templates import (
    EMISSION_LINES,
    FILTER_BANDS,
    WAVELENGTH,
    SEDTemplate,
    build_template,
    colour_grid,
    describe_break_crossings,
    draw_template,
    filter_curve,
    standard_library,
)

__all__ = [
    "PhotoZLibrary", "PhotoZResult", "fit_photoz", "fit_catalog",
    "TEMPLATE_FLOOR", "SECOND_PEAK_FRACTION",
    "SEDTemplate", "build_template", "draw_template", "standard_library",
    "colour_grid", "filter_curve", "describe_break_crossings",
    "FILTER_BANDS", "EMISSION_LINES", "WAVELENGTH",
]
