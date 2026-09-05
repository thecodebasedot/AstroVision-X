"""Strong gravitational-lens candidate searching, and modelling what is found.

Detecting arcs says a lens is probably there.  Fitting a mass model says how
much mass is inside the Einstein radius and how it is shaped -- the difference
between a candidate and a measurement.  Both live here, and the second only
runs when the first found enough to constrain it.
"""

from .arcs import (
    Arc,
    detect_arcs,
    einstein_radius,
    ring_completeness,
    subtract_smooth_light,
)
from .lens import LensSearch, deflector_plausibility, velocity_dispersion
from .model import (
    MAX_PLAUSIBLE_SHEAR,
    SHEAR_EVIDENCE_FACTOR,
    MIN_SHEAR_SPAN_DEG,
    LensFit,
    LensModel,
    arc_sample_points,
    azimuthal_span,
    einstein_mass,
    fit_lens_model,
    image_plane_residual,
    ray_trace,
    shear_deflection,
    sie_deflection,
    sis_deflection,
    source_plane_scatter,
)

__all__ = [
    "LensSearch", "deflector_plausibility", "velocity_dispersion",
    "Arc", "detect_arcs", "ring_completeness", "einstein_radius",
    "subtract_smooth_light",
    "LensModel", "LensFit", "fit_lens_model", "ray_trace", "einstein_mass",
    "sie_deflection", "sis_deflection", "shear_deflection",
    "source_plane_scatter", "image_plane_residual", "arc_sample_points",
    "azimuthal_span", "MIN_SHEAR_SPAN_DEG", "MAX_PLAUSIBLE_SHEAR", "SHEAR_EVIDENCE_FACTOR",
]
