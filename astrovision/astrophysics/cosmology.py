"""Cosmological distances for a flat Lambda-CDM universe.

Turning an observed angular size or flux into a physical size or luminosity
requires a cosmology.  The integrals are evaluated numerically, which is
exact enough for any use here and avoids a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

#: Speed of light, km/s.
C_KM_S = 299_792.458
#: Megaparsec in metres, for unit conversions.
MPC_M = 3.0856775814913673e22


@dataclass
class Cosmology:
    """Flat Lambda-CDM with matter and dark energy.

    Distances agree with Astropy's ``FlatLambdaCDM`` to better than 0.01 %.

    >>> cosmo = Cosmology()
    >>> round(cosmo.comoving_distance(1.0))          # Mpc
    3304
    >>> round(cosmo.distance_modulus(1.0), 2)
    44.1
    """

    H0: float = 70.0        # km/s/Mpc
    Om0: float = 0.3
    Ode0: float = 0.7
    n_steps: int = 512

    @property
    def hubble_distance(self) -> float:
        """``c / H0`` in Mpc -- the natural length scale of the universe."""
        return C_KM_S / self.H0

    @property
    def hubble_time(self) -> float:
        """``1 / H0`` in Gyr."""
        return 977.79 / self.H0

    @property
    def Ok0(self) -> float:
        """Curvature density; zero for the flat default."""
        return 1.0 - self.Om0 - self.Ode0

    def efunc(self, z: Union[float, np.ndarray]) -> np.ndarray:
        """``E(z) = H(z) / H0``."""
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.Om0 * (1 + z) ** 3 + self.Ok0 * (1 + z) ** 2 + self.Ode0)

    def hubble_parameter(self, z: Union[float, np.ndarray]) -> np.ndarray:
        """``H(z)`` in km/s/Mpc."""
        return self.H0 * self.efunc(z)

    def comoving_distance(self, z: float, z_min: float = 0.0) -> float:
        """Line-of-sight comoving distance in Mpc."""
        if z <= z_min:
            return 0.0
        # Simpson's rule needs an even number of intervals.
        n = max(8, int(self.n_steps) // 2 * 2)
        grid = np.linspace(z_min, z, n + 1)
        integrand = 1.0 / self.efunc(grid)
        weights = np.ones(n + 1)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        step = (z - z_min) / n
        return float(self.hubble_distance * step / 3.0 * np.sum(weights * integrand))

    def transverse_comoving_distance(self, z: float) -> float:
        """Comoving distance transverse to the line of sight (flat: identical)."""
        distance = self.comoving_distance(z)
        if abs(self.Ok0) < 1e-8:
            return distance
        root = np.sqrt(abs(self.Ok0))
        scaled = root * distance / self.hubble_distance
        factor = np.sinh(scaled) if self.Ok0 > 0 else np.sin(scaled)
        return float(self.hubble_distance / root * factor)

    def angular_diameter_distance(self, z: float) -> float:
        """Physical size per unit angle, in Mpc."""
        return self.transverse_comoving_distance(z) / (1.0 + z)

    def angular_diameter_distance_between(self, z1: float, z2: float) -> float:
        """Angular-diameter distance between two redshifts (flat universe)."""
        if z2 <= z1:
            return 0.0
        d1 = self.transverse_comoving_distance(z1)
        d2 = self.transverse_comoving_distance(z2)
        if abs(self.Ok0) < 1e-8:
            return float((d2 - d1) / (1.0 + z2))
        # The general expression is only valid for a non-flat universe.
        root = np.sqrt(abs(self.Ok0)) / self.hubble_distance
        if self.Ok0 > 0:
            term = d2 * np.sqrt(1 + (root * d1) ** 2) - d1 * np.sqrt(1 + (root * d2) ** 2)
        else:
            term = d2 * np.sqrt(1 - (root * d1) ** 2) - d1 * np.sqrt(1 - (root * d2) ** 2)
        return float(term / (1.0 + z2))

    def luminosity_distance(self, z: float) -> float:
        """Distance implied by an object's observed flux, in Mpc."""
        return self.transverse_comoving_distance(z) * (1.0 + z)

    def distance_modulus(self, z: float) -> float:
        """``m - M`` for an object at redshift ``z``."""
        distance = self.luminosity_distance(z)
        if distance <= 0:
            return float("nan")
        return float(5.0 * np.log10(distance * 1e6 / 10.0))

    def angular_scale(self, z: float) -> float:
        """Proper kiloparsecs per arcsecond at redshift ``z``."""
        d_a = self.angular_diameter_distance(z)
        return float(d_a * 1000.0 * np.pi / 180.0 / 3600.0)

    def lookback_time(self, z: float) -> float:
        """Time since light left an object at redshift ``z``, in Gyr."""
        if z <= 0:
            return 0.0
        n = max(8, int(self.n_steps) // 2 * 2)
        grid = np.linspace(0.0, z, n + 1)
        integrand = 1.0 / ((1 + grid) * self.efunc(grid))
        weights = np.ones(n + 1)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        step = z / n
        return float(self.hubble_time * step / 3.0 * np.sum(weights * integrand))

    def comoving_volume(self, z: float, area_sq_deg: Optional[float] = None) -> float:
        """Comoving volume out to ``z``, in Mpc^3 (full sky, or a survey area)."""
        distance = self.transverse_comoving_distance(z)
        volume = 4.0 / 3.0 * np.pi * distance ** 3
        if area_sq_deg is not None:
            volume *= float(area_sq_deg) / 41252.96
        return float(volume)

    def redshift_from_distance_modulus(self, mu: float, z_max: float = 10.0) -> float:
        """Invert the distance modulus numerically."""
        if not np.isfinite(mu):
            return float("nan")
        low, high = 1e-4, float(z_max)
        for _ in range(80):
            mid = 0.5 * (low + high)
            if self.distance_modulus(mid) < mu:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))

    def to_dict(self):
        return {"H0": self.H0, "Om0": self.Om0, "Ode0": self.Ode0}


#: A standard concordance cosmology, used when none is supplied.
DEFAULT_COSMOLOGY = Cosmology()


def photometric_redshift_hint(colour: float, magnitude: float,
                              band_pair: str = "g-r") -> float:
    """A crude redshift guess from one colour and one magnitude.

    A real photometric redshift needs several bands and a template library;
    this exists only so a single-band pipeline can put an order-of-magnitude
    distance on an object, and it is labelled as a hint everywhere it is used.
    """
    if not (np.isfinite(colour) and np.isfinite(magnitude)):
        return float("nan")
    # Galaxies redden and fade with redshift; this is a linearised version
    # of that trend anchored on typical low-redshift galaxy colours.
    base = 0.06 * (magnitude - 17.0)
    reddening = 0.35 * max(colour - 0.6, 0.0)
    return float(np.clip(base + reddening, 0.005, 4.0))
