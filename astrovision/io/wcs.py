"""Minimal world-coordinate handling (gnomonic / TAN projection).

Astropy is used when installed.  The fallback implements the TAN
projection directly so pixel <-> sky conversion still works in a bare
NumPy environment -- enough for catalog coordinates and cross-matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger

log = get_logger("io.wcs")


@dataclass
class SimpleWCS:
    """A TAN-projection world coordinate system.

    Attributes mirror the FITS keywords: ``crpix`` is 1-based per the FITS
    convention, ``crval`` is the sky position of that reference pixel in
    degrees, and ``cd`` is the 2x2 pixel-to-degree matrix.
    """

    crpix: Tuple[float, float] = (1.0, 1.0)
    crval: Tuple[float, float] = (0.0, 0.0)
    cd: Optional[np.ndarray] = None
    ctype: Tuple[str, str] = ("RA---TAN", "DEC--TAN")

    def __post_init__(self) -> None:
        if self.cd is None:
            self.cd = np.array([[-1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]])
        self.cd = np.asarray(self.cd, dtype=float).reshape(2, 2)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_header(cls, header: Dict[str, Any]) -> Optional["SimpleWCS"]:
        """Build from FITS header keywords, or return ``None`` if absent."""
        if "CRVAL1" not in header or "CRVAL2" not in header:
            return None
        crpix = (float(header.get("CRPIX1", 1.0)), float(header.get("CRPIX2", 1.0)))
        crval = (float(header["CRVAL1"]), float(header["CRVAL2"]))
        if "CD1_1" in header:
            cd = np.array([
                [float(header.get("CD1_1", 0.0)), float(header.get("CD1_2", 0.0))],
                [float(header.get("CD2_1", 0.0)), float(header.get("CD2_2", 0.0))],
            ])
        else:
            cdelt1 = float(header.get("CDELT1", -1.0 / 3600.0))
            cdelt2 = float(header.get("CDELT2", 1.0 / 3600.0))
            rot = math.radians(float(header.get("CROTA2", 0.0)))
            cd = np.array([
                [cdelt1 * math.cos(rot), -cdelt2 * math.sin(rot)],
                [cdelt1 * math.sin(rot), cdelt2 * math.cos(rot)],
            ])
        ctype = (str(header.get("CTYPE1", "RA---TAN")), str(header.get("CTYPE2", "DEC--TAN")))
        return cls(crpix, crval, cd, ctype)

    @classmethod
    def tangent(cls, ra: float, dec: float, shape: Tuple[int, int],
                pixel_scale_arcsec: float = 1.0, rotation_deg: float = 0.0) -> "SimpleWCS":
        """Convenience constructor centring the field on ``(ra, dec)``."""
        scale = float(pixel_scale_arcsec) / 3600.0
        rot = math.radians(rotation_deg)
        cd = np.array([
            [-scale * math.cos(rot), scale * math.sin(rot)],
            [scale * math.sin(rot), scale * math.cos(rot)],
        ])
        crpix = ((shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0)
        return cls(crpix, (float(ra), float(dec)), cd)

    # -- properties --------------------------------------------------------
    @property
    def pixel_scale(self) -> float:
        """Mean pixel scale in arcsec/pixel."""
        scales = np.sqrt((self.cd ** 2).sum(axis=0)) * 3600.0
        return float(np.mean(scales))

    @property
    def orientation(self) -> float:
        """Position angle of the +y axis, degrees east of north."""
        return float(np.degrees(np.arctan2(self.cd[0, 1], self.cd[1, 1])))

    # -- transforms --------------------------------------------------------
    def pixel_to_world(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 0-based pixel coordinates to ``(ra, dec)`` in degrees."""
        dx = np.asarray(x, dtype=float) + 1.0 - self.crpix[0]
        dy = np.asarray(y, dtype=float) + 1.0 - self.crpix[1]
        xi = np.radians(self.cd[0, 0] * dx + self.cd[0, 1] * dy)
        eta = np.radians(self.cd[1, 0] * dx + self.cd[1, 1] * dy)
        ra0 = math.radians(self.crval[0])
        dec0 = math.radians(self.crval[1])
        denom = np.cos(dec0) - eta * np.sin(dec0)
        ra = ra0 + np.arctan2(xi, denom)
        dec = np.arctan2((np.sin(dec0) + eta * np.cos(dec0)),
                         np.sqrt(xi ** 2 + denom ** 2))
        return np.degrees(ra) % 360.0, np.degrees(dec)

    def world_to_pixel(self, ra, dec) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ``(ra, dec)`` in degrees to 0-based pixel coordinates."""
        ra_r = np.radians(np.asarray(ra, dtype=float))
        dec_r = np.radians(np.asarray(dec, dtype=float))
        ra0 = math.radians(self.crval[0])
        dec0 = math.radians(self.crval[1])
        cos_c = (np.sin(dec0) * np.sin(dec_r) +
                 np.cos(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0))
        cos_c = np.where(np.abs(cos_c) < 1e-12, 1e-12, cos_c)
        xi = np.degrees(np.cos(dec_r) * np.sin(ra_r - ra0) / cos_c)
        eta = np.degrees((np.cos(dec0) * np.sin(dec_r) -
                          np.sin(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0)) / cos_c)
        inv = np.linalg.inv(self.cd)
        dx = inv[0, 0] * xi + inv[0, 1] * eta
        dy = inv[1, 0] * xi + inv[1, 1] * eta
        return dx + self.crpix[0] - 1.0, dy + self.crpix[1] - 1.0

    def separation_arcsec(self, x1, y1, x2, y2) -> np.ndarray:
        """Angular separation between two pixel positions, in arcsec."""
        ra1, dec1 = self.pixel_to_world(x1, y1)
        ra2, dec2 = self.pixel_to_world(x2, y2)
        return angular_separation(ra1, dec1, ra2, dec2) * 3600.0

    def to_header(self) -> Dict[str, Any]:
        return {
            "CTYPE1": self.ctype[0], "CTYPE2": self.ctype[1],
            "CRPIX1": self.crpix[0], "CRPIX2": self.crpix[1],
            "CRVAL1": self.crval[0], "CRVAL2": self.crval[1],
            "CD1_1": self.cd[0, 0], "CD1_2": self.cd[0, 1],
            "CD2_1": self.cd[1, 0], "CD2_2": self.cd[1, 1],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crpix": list(self.crpix), "crval": list(self.crval),
            "cd": self.cd.tolist(), "ctype": list(self.ctype),
            "pixel_scale_arcsec": self.pixel_scale,
        }


def angular_separation(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in degrees (Vincenty formula, numerically safe)."""
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    dra = r2 - r1
    num = np.sqrt((np.cos(d2) * np.sin(dra)) ** 2 +
                  (np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(dra)) ** 2)
    den = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(dra)
    return np.degrees(np.arctan2(num, den))


def wcs_from_header(header: Dict[str, Any]) -> Optional[SimpleWCS]:
    """Prefer Astropy's WCS validation when available, else parse directly."""
    astropy_wcs = try_import("astropy.wcs")
    if astropy_wcs is not None:
        try:
            aw = astropy_wcs.WCS(dict(header))
            if aw.has_celestial:
                cd = aw.pixel_scale_matrix
                crpix = tuple(float(v) for v in aw.wcs.crpix[:2])
                crval = tuple(float(v) for v in aw.wcs.crval[:2])
                return SimpleWCS(crpix, crval, np.asarray(cd, dtype=float))
        except Exception as exc:  # pragma: no cover - malformed headers
            log.debug("astropy WCS parse failed (%s); using built-in TAN parser", exc)
    return SimpleWCS.from_header(header)
