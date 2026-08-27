"""Turning measurements into physical quantities.

The photometry stage produces counts and pixels; this turns them into
luminosities, physical sizes and stellar-mass estimates.  Every conversion
needs an assumption -- a distance, a mass-to-light ratio -- and each is
stated explicitly and carried into the report, because a physical quantity
quoted without its assumptions is not a measurement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.logging import get_logger
from ..core.types import ObjectClass, SourceCatalog
from ..photometry.magnitudes import (
    luminosity_solar,
)
from .cosmology import DEFAULT_COSMOLOGY, Cosmology

log = get_logger("astrophysics.physical")


def physical_size(angular_size_arcsec: float, redshift: float,
                  cosmology: Optional[Cosmology] = None) -> float:
    """Proper size in kiloparsecs from an angular size and a redshift."""
    cosmology = cosmology or DEFAULT_COSMOLOGY
    if not np.isfinite(angular_size_arcsec) or angular_size_arcsec <= 0 or redshift <= 0:
        return float("nan")
    return float(angular_size_arcsec * cosmology.angular_scale(redshift))


def absolute_magnitude_at_z(apparent: float, redshift: float,
                            cosmology: Optional[Cosmology] = None,
                            k_correction: float = 0.0) -> float:
    """Absolute magnitude from an apparent one and a redshift."""
    cosmology = cosmology or DEFAULT_COSMOLOGY
    mu = cosmology.distance_modulus(redshift)
    if not (np.isfinite(apparent) and np.isfinite(mu)):
        return float("nan")
    return float(apparent - mu - k_correction)


def stellar_mass_estimate(absolute_mag: float, colour: float = float("nan"),
                          band: str = "r") -> float:
    """Stellar mass in solar masses, from luminosity and a colour-based M/L.

    Uses the Bell et al. (2003) colour--mass-to-light relation when a colour
    is available and a fixed ratio otherwise.  The scatter is a factor of
    about two even with good colours; the value is an order-of-magnitude
    estimate, and is labelled as one wherever it is reported.
    """
    if not np.isfinite(absolute_mag):
        return float("nan")
    luminosity = luminosity_solar(absolute_mag, band)
    if not np.isfinite(luminosity):
        return float("nan")
    if np.isfinite(colour):
        # log10(M/L_r) = -0.306 + 1.097 * (g - r)
        log_ml = -0.306 + 1.097 * float(np.clip(colour, -0.2, 1.6))
        mass_to_light = float(10 ** log_ml)
    else:
        mass_to_light = 2.0
    return float(luminosity * mass_to_light)


def star_formation_rate(luminosity_uv_solar: float) -> float:
    """A rough star-formation rate in solar masses per year.

    Uses the Kennicutt (1998) UV calibration.  With a single broad optical
    band this is at best indicative, and the pipeline says so.
    """
    if not np.isfinite(luminosity_uv_solar) or luminosity_uv_solar <= 0:
        return float("nan")
    # 1.4e-28 erg/s/Hz per (Msun/yr), converted through the solar luminosity.
    return float(1.4e-28 * luminosity_uv_solar * 3.828e33 / 1e28 * 1e-5)


def surface_brightness_dimming(redshift: float) -> float:
    """The ``(1+z)^-4`` cosmological surface-brightness dimming factor.

    This is why high-redshift galaxies are so much harder to see than their
    distance alone suggests, and it belongs in any completeness statement.
    """
    if not np.isfinite(redshift) or redshift < 0:
        return float("nan")
    return float((1.0 + redshift) ** -4)


def annotate_physical(catalog: SourceCatalog, redshift: Optional[float] = None,
                      pixel_scale: float = 1.0, band: str = "r",
                      cosmology: Optional[Cosmology] = None,
                      assume_redshift_for_galaxies: bool = False) -> Dict[str, Any]:
    """Attach physical quantities to every source that supports them.

    Without a redshift, only distance-independent quantities are computed.
    Nothing is invented: a source with no distance gets no luminosity.
    """
    cosmology = cosmology or DEFAULT_COSMOLOGY
    assumptions: List[str] = []
    distance_cache: Dict[float, Dict[str, float]] = {}
    n_physical = 0

    n_measured = sum(1 for source in catalog
                     if (source.meta.get("photoz") or {}).get("z") is not None)
    if n_measured:
        assumptions.append(
            f"photometric redshifts for {n_measured} galaxies, each with its own "
            "distance; the rest fall back to the field assumption below")
    if redshift is not None:
        assumptions.append(f"a single redshift z = {redshift:g} for the whole field")
    elif assume_redshift_for_galaxies:
        assumptions.append("a nominal z = 0.1 for galaxies lacking a measured redshift")
    assumptions.append(f"cosmology H0={cosmology.H0:g}, Om0={cosmology.Om0:g}")

    for source in catalog:
        physical: Dict[str, float] = {}
        # Distance-independent quantities are always available.
        angular = source.morphology.semi_major * 2.0 * pixel_scale
        physical["angular_diameter_arcsec"] = float(angular)

        # A measured photometric redshift beats any field-wide assumption --
        # and a field-wide assumption is what every distance-dependent
        # quantity silently inherited before there was one to measure.
        photoz = source.meta.get("photoz") or {}
        z = None
        if photoz.get("reliable") and np.isfinite(photoz.get("z", float("nan"))):
            z = float(photoz["z"])
            physical["redshift_source"] = "photometric"
            physical["redshift_error"] = float(photoz.get("z_error", float("nan")))
        elif photoz.get("z") is not None and np.isfinite(photoz.get("z", float("nan"))):
            # An unreliable photo-z is still the best number available, but
            # everything derived from it carries the flag that says so.
            z = float(photoz["z"])
            physical["redshift_source"] = "photometric_unreliable"
            physical["redshift_error"] = float(photoz.get("z_error", float("nan")))
            source.add_flag("uncertain_redshift")
        if z is None:
            z = redshift
            if z is not None:
                physical["redshift_source"] = "assumed_field"
        if z is None and assume_redshift_for_galaxies and source.is_extended:
            z = float(source.meta.get("redshift_hint", 0.1))
            physical["redshift_assumed"] = True
            physical["redshift_source"] = "assumed_nominal"
        physical["redshift"] = float(z) if z is not None else float("nan")
        if z is None or not np.isfinite(z) or z <= 0:
            source.meta["physical"] = physical
            continue

        # Each distance is a numerical integral, so cache them per redshift:
        # a whole field usually shares one, and recomputing per source
        # dominates the stage's runtime for no gain.
        if z not in distance_cache:
            distance_cache[z] = {
                "distance_mpc": float(cosmology.luminosity_distance(z)),
                "lookback_time_gyr": float(cosmology.lookback_time(z)),
                "kpc_per_arcsec": float(cosmology.angular_scale(z)),
                "distance_modulus": float(cosmology.distance_modulus(z)),
            }
        cached = distance_cache[z]
        physical["redshift"] = float(z)
        physical["distance_mpc"] = cached["distance_mpc"]
        physical["lookback_time_gyr"] = cached["lookback_time_gyr"]
        physical["kpc_per_arcsec"] = cached["kpc_per_arcsec"]
        physical["physical_size_kpc"] = float(angular * cached["kpc_per_arcsec"]) \
            if np.isfinite(angular) and angular > 0 else float("nan")
        absolute = (float(source.photometry.magnitude - cached["distance_modulus"])
                    if np.isfinite(source.photometry.magnitude) else float("nan"))
        physical["absolute_magnitude"] = float(absolute)
        physical["luminosity_solar"] = float(luminosity_solar(absolute, band))
        if source.is_extended:
            colour = float(source.meta.get("colour", np.nan))
            physical["stellar_mass_solar"] = stellar_mass_estimate(absolute, colour, band)
        physical["surface_brightness_dimming"] = surface_brightness_dimming(z)
        source.meta["physical"] = physical
        n_physical += 1

    summary = {
        "n_with_physical_properties": n_physical,
        "assumptions": assumptions,
        "cosmology": cosmology.to_dict(),
        "band": band,
        "pixel_scale_arcsec": float(pixel_scale),
    }
    if n_physical:
        log.info("derived physical properties for %d sources under %d assumption(s)",
                 n_physical, len(assumptions))
    return summary


def stellar_population_summary(catalog: SourceCatalog) -> Dict[str, Any]:
    """Summarise the stellar content of a field.

    Star counts and their brightness distribution constrain the line of
    sight through the Galaxy; the crowding statistic says how reliable any
    photometry in this field can be.
    """
    stars = catalog.of_class(ObjectClass.STAR)
    if len(stars) == 0:
        return {"n_stars": 0}

    magnitudes = np.array([s.photometry.magnitude for s in stars], dtype=float)
    finite = magnitudes[np.isfinite(magnitudes)]
    positions = stars.positions()

    summary: Dict[str, Any] = {
        "n_stars": len(stars),
        "magnitude_median": float(np.median(finite)) if finite.size else float("nan"),
        "magnitude_range": ([float(finite.min()), float(finite.max())]
                            if finite.size else [float("nan")] * 2),
    }
    if len(positions) > 3:
        distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.min(axis=1)
        summary["median_separation_px"] = float(np.median(nearest))
        # Crowding: how many stars sit within a typical PSF footprint.
        fwhm = float(np.nanmedian([s.morphology.fwhm for s in stars]))
        if np.isfinite(fwhm) and fwhm > 0:
            summary["crowding_index"] = float(np.mean(nearest < 2 * fwhm))
    return summary
