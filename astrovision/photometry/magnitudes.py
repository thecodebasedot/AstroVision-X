"""Magnitudes, colours and surface brightness.

Astronomy reports brightness on a logarithmic magnitude scale where
*smaller is brighter*, anchored by an instrumental zero point.  These
helpers convert between the linear fluxes the pipeline measures and the
magnitudes a scientific report must quote.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Approximate effective wavelengths (Angstroms) of common survey filters.
FILTER_WAVELENGTHS = {
    "u": 3543.0, "g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0,
    "y": 10_200.0, "J": 12_350.0, "H": 16_620.0, "K": 21_590.0,
    "B": 4450.0, "V": 5510.0, "R": 6580.0, "I": 8060.0, "clear": 6000.0,
}

#: Solar absolute magnitudes per band, for luminosity conversions.
SOLAR_ABSOLUTE_MAGNITUDE = {
    "u": 6.39, "g": 5.11, "r": 4.65, "i": 4.53, "z": 4.50,
    "B": 5.44, "V": 4.81, "R": 4.43, "I": 4.10, "clear": 4.74,
}


def flux_to_magnitude(flux, zero_point: float = 25.0,
                      flux_err=None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flux to magnitude; non-positive fluxes yield NaN.

    Returns ``(magnitude, magnitude_error)``; the error uses the standard
    ``2.5 / ln(10) * dF / F`` propagation.
    """
    values = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.where(values > 0, zero_point - 2.5 * np.log10(np.maximum(values, 1e-30)),
                             np.nan)
    if flux_err is None:
        return magnitude, np.full_like(magnitude, np.nan)
    errors = np.asarray(flux_err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_err = np.where(values > 0, 2.5 / np.log(10) * errors / np.maximum(values, 1e-30),
                                 np.nan)
    return magnitude, magnitude_err


def magnitude_to_flux(magnitude, zero_point: float = 25.0) -> np.ndarray:
    """Inverse of :func:`flux_to_magnitude`."""
    return np.power(10.0, (float(zero_point) - np.asarray(magnitude, dtype=float)) / 2.5)


def limiting_magnitude(rms: float, aperture_radius: float, zero_point: float = 25.0,
                       sigma: float = 5.0) -> float:
    """Faintest magnitude detectable at ``sigma`` in one aperture.

    This is the number that defines how deep an exposure is, and every
    completeness statement in the report is relative to it.
    """
    area = np.pi * max(float(aperture_radius), 1e-6) ** 2
    flux_limit = float(sigma) * float(rms) * np.sqrt(area)
    if flux_limit <= 0:
        return float("nan")
    return float(zero_point - 2.5 * np.log10(flux_limit))


def surface_brightness(flux: float, area_pixels: float, pixel_scale: float = 1.0,
                       zero_point: float = 25.0) -> float:
    """Mean surface brightness in magnitudes per square arcsecond."""
    area_arcsec = float(area_pixels) * float(pixel_scale) ** 2
    if flux <= 0 or area_arcsec <= 0:
        return float("nan")
    return float(zero_point - 2.5 * np.log10(flux / area_arcsec))


def colour_index(magnitude_blue: float, magnitude_red: float) -> float:
    """A colour index such as ``g - r``; larger means redder."""
    if not (np.isfinite(magnitude_blue) and np.isfinite(magnitude_red)):
        return float("nan")
    return float(magnitude_blue - magnitude_red)


def colour_temperature(colour: float, band_pair: Tuple[str, str] = ("g", "r")) -> float:
    """Rough effective temperature from a colour index, in kelvin.

    Uses the Ballesteros (2012) blackbody relation calibrated on ``B-V``;
    other colour pairs are first converted to an approximate ``B-V``.  It
    is an order-of-magnitude guide for triage, not a spectroscopic fit.
    """
    if not np.isfinite(colour):
        return float("nan")
    pair = (band_pair[0], band_pair[1])
    if pair == ("B", "V"):
        bv = float(colour)
    elif pair == ("g", "r"):
        bv = 0.98 * float(colour) + 0.22       # Jester et al. (2005), approximate
    elif pair == ("V", "R"):
        bv = 1.8 * float(colour)
    else:
        bv = float(colour)
    bv = float(np.clip(bv, -0.4, 2.0))
    return float(4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)))


def absolute_magnitude(apparent_magnitude: float, distance_pc: float,
                       extinction: float = 0.0) -> float:
    """Apparent magnitude corrected to 10 parsecs (the distance modulus)."""
    if not np.isfinite(apparent_magnitude) or distance_pc <= 0:
        return float("nan")
    return float(apparent_magnitude - 5.0 * np.log10(distance_pc / 10.0) - extinction)


def distance_modulus(distance_pc: float) -> float:
    """``m - M`` for a given distance in parsecs."""
    if distance_pc <= 0:
        return float("nan")
    return float(5.0 * np.log10(distance_pc / 10.0))


def luminosity_solar(absolute_mag: float, band: str = "clear") -> float:
    """Luminosity in solar units from an absolute magnitude."""
    solar = SOLAR_ABSOLUTE_MAGNITUDE.get(band, SOLAR_ABSOLUTE_MAGNITUDE["clear"])
    if not np.isfinite(absolute_mag):
        return float("nan")
    return float(10.0 ** (-0.4 * (absolute_mag - solar)))


def add_magnitudes(magnitudes: Sequence[float]) -> float:
    """Combine several magnitudes into the magnitude of their total flux."""
    values = np.asarray([m for m in magnitudes if np.isfinite(m)], dtype=float)
    if values.size == 0:
        return float("nan")
    return float(-2.5 * np.log10(np.sum(np.power(10.0, -0.4 * values))))


def magnitude_error_to_snr(magnitude_err: float) -> float:
    """Signal-to-noise implied by a magnitude uncertainty."""
    if not np.isfinite(magnitude_err) or magnitude_err <= 0:
        return float("nan")
    return float(2.5 / np.log(10) / magnitude_err)


def zero_point_from_standards(instrumental_mags: Sequence[float],
                              catalog_mags: Sequence[float]) -> Tuple[float, float]:
    """Photometric zero point from standard stars: ``(zero_point, scatter)``."""
    instrumental = np.asarray(instrumental_mags, dtype=float)
    catalog = np.asarray(catalog_mags, dtype=float)
    good = np.isfinite(instrumental) & np.isfinite(catalog)
    if good.sum() < 2:
        return float("nan"), float("nan")
    offsets = catalog[good] - instrumental[good]
    return float(np.median(offsets)), float(np.std(offsets))
