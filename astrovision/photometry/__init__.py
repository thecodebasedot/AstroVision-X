"""Photometry: apertures, curves of growth, magnitudes and colours."""

from .aperture import (
    ApertureResult,
    annulus_background,
    aperture_photometry,
    circular_aperture_weights,
    elliptical_aperture_weights,
    elliptical_photometry,
    multi_aperture,
)
from .growth import (
    auto_aperture,
    concentration_index,
    curve_of_growth,
    flux_radius,
    kron_radius,
    petrosian_radius,
)
from .magnitudes import (
    FILTER_WAVELENGTHS,
    SOLAR_ABSOLUTE_MAGNITUDE,
    absolute_magnitude,
    add_magnitudes,
    colour_index,
    colour_temperature,
    distance_modulus,
    flux_to_magnitude,
    limiting_magnitude,
    luminosity_solar,
    magnitude_to_flux,
    surface_brightness,
    zero_point_from_standards,
)
from .photometer import Photometer

__all__ = [
    "Photometer",
    "aperture_photometry", "elliptical_photometry", "multi_aperture",
    "circular_aperture_weights", "elliptical_aperture_weights",
    "annulus_background", "ApertureResult",
    "curve_of_growth", "kron_radius", "petrosian_radius", "flux_radius",
    "concentration_index", "auto_aperture",
    "flux_to_magnitude", "magnitude_to_flux", "limiting_magnitude",
    "surface_brightness", "colour_index", "colour_temperature",
    "absolute_magnitude", "distance_modulus", "luminosity_solar",
    "add_magnitudes", "zero_point_from_standards",
    "FILTER_WAVELENGTHS", "SOLAR_ABSOLUTE_MAGNITUDE",
]
