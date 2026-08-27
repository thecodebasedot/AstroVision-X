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
    #: SIP forward distortion, ``a[p, q]`` multiplying ``u**p * v**q``.
    sip_a: Optional[np.ndarray] = None
    sip_b: Optional[np.ndarray] = None
    #: SIP reverse distortion.  Optional: when absent the inverse is found
    #: by iteration, which is exact to well below a milli-pixel and avoids
    #: depending on coefficients many real headers simply do not carry.
    sip_ap: Optional[np.ndarray] = None
    sip_bp: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.cd is None:
            self.cd = np.array([[-1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]])
        self.cd = np.asarray(self.cd, dtype=float).reshape(2, 2)
        for name in ("sip_a", "sip_b", "sip_ap", "sip_bp"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=float))

    # -- distortion --------------------------------------------------------
    @property
    def has_distortion(self) -> bool:
        return self.sip_a is not None and self.sip_b is not None

    def _sip_shift(self, u: np.ndarray, v: np.ndarray,
                   a: Optional[np.ndarray], b: Optional[np.ndarray]
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Polynomial offsets ``(f, g)`` for pixel offsets from the reference.

        The SIP convention (Shupe et al. 2005) writes the distortion as a
        polynomial in the *offset from CRPIX*, added to that offset before
        the linear CD matrix is applied.  Getting that order wrong is the
        classic SIP bug: applying the polynomial to absolute pixel
        coordinates leaves an error that grows across the detector and looks
        exactly like a bad plate scale.
        """
        du = np.zeros_like(u, dtype=float)
        dv = np.zeros_like(v, dtype=float)
        for coefficients, out in ((a, du), (b, dv)):
            if coefficients is None:
                continue
            for p in range(coefficients.shape[0]):
                for q in range(coefficients.shape[1]):
                    value = coefficients[p, q]
                    if value:
                        out += value * (u ** p) * (v ** q)
        return du, dv

    def apply_distortion(self, u, v) -> Tuple[np.ndarray, np.ndarray]:
        """Add the forward SIP distortion to offsets from the reference pixel."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if not self.has_distortion:
            return u, v
        du, dv = self._sip_shift(u, v, self.sip_a, self.sip_b)
        return u + du, v + dv

    def remove_distortion(self, u, v, iterations: int = 12,
                          tolerance: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
        """Invert the forward distortion.

        Uses the reverse coefficients when the header supplied them, and
        otherwise iterates ``u_undistorted = u - f(u_undistorted)`` to a
        fixed point.  The iteration converges because SIP distortions are
        small perturbations by construction -- a few pixels over a detector
        thousands across -- so the map is a contraction.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if not self.has_distortion:
            return u, v
        if self.sip_ap is not None and self.sip_bp is not None:
            du, dv = self._sip_shift(u, v, self.sip_ap, self.sip_bp)
            return u + du, v + dv
        guess_u, guess_v = u.copy(), v.copy()
        for _ in range(int(iterations)):
            du, dv = self._sip_shift(guess_u, guess_v, self.sip_a, self.sip_b)
            next_u, next_v = u - du, v - dv
            shift = max(float(np.max(np.abs(next_u - guess_u))) if next_u.size else 0.0,
                        float(np.max(np.abs(next_v - guess_v))) if next_v.size else 0.0)
            guess_u, guess_v = next_u, next_v
            if shift < tolerance:
                break
        return guess_u, guess_v

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
        sip = {name: _sip_from_header(header, letter)
               for name, letter in (("sip_a", "A"), ("sip_b", "B"),
                                    ("sip_ap", "AP"), ("sip_bp", "BP"))}
        return cls(crpix, crval, cd, ctype, **sip)

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
        dx, dy = self.apply_distortion(dx, dy)
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
        dx, dy = self.remove_distortion(dx, dy)
        return dx + self.crpix[0] - 1.0, dy + self.crpix[1] - 1.0

    def separation_arcsec(self, x1, y1, x2, y2) -> np.ndarray:
        """Angular separation between two pixel positions, in arcsec."""
        ra1, dec1 = self.pixel_to_world(x1, y1)
        ra2, dec2 = self.pixel_to_world(x2, y2)
        return angular_separation(ra1, dec1, ra2, dec2) * 3600.0

    def to_header(self) -> Dict[str, Any]:
        header: Dict[str, Any] = {
            "CTYPE1": self.ctype[0], "CTYPE2": self.ctype[1],
            "CRPIX1": self.crpix[0], "CRPIX2": self.crpix[1],
            "CRVAL1": self.crval[0], "CRVAL2": self.crval[1],
            "CD1_1": self.cd[0, 0], "CD1_2": self.cd[0, 1],
            "CD2_1": self.cd[1, 0], "CD2_2": self.cd[1, 1],
        }
        if self.has_distortion:
            # The -SIP suffix on CTYPE is what tells any other reader that
            # the coefficients are there and must be applied; without it a
            # conforming reader is right to ignore them.
            header["CTYPE1"] = _with_sip_suffix(self.ctype[0])
            header["CTYPE2"] = _with_sip_suffix(self.ctype[1])
            for letter, coefficients in (("A", self.sip_a), ("B", self.sip_b),
                                         ("AP", self.sip_ap), ("BP", self.sip_bp)):
                header.update(_sip_to_header(coefficients, letter))
        return header

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crpix": list(self.crpix), "crval": list(self.crval),
            "cd": self.cd.tolist(), "ctype": list(self.ctype),
            "pixel_scale_arcsec": self.pixel_scale,
        }


def _with_sip_suffix(ctype: str) -> str:
    return ctype if ctype.endswith("-SIP") else ctype + "-SIP"


def _sip_from_header(header: Dict[str, Any], letter: str) -> Optional[np.ndarray]:
    """Read ``A_p_q`` style coefficients into a dense array."""
    order = header.get(f"{letter}_ORDER")
    keys = [k for k in header if k.startswith(f"{letter}_") and k != f"{letter}_ORDER"]
    terms: Dict[Tuple[int, int], float] = {}
    for key in keys:
        parts = key.split("_")
        if len(parts) != 3:
            continue
        try:
            p, q = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        terms[(p, q)] = float(header[key])
    if not terms:
        return None
    size = int(order) + 1 if order is not None else max(max(p, q) for p, q in terms) + 1
    coefficients = np.zeros((size, size), dtype=float)
    for (p, q), value in terms.items():
        if p < size and q < size:
            coefficients[p, q] = value
    return coefficients


def _sip_to_header(coefficients: Optional[np.ndarray], letter: str) -> Dict[str, Any]:
    """Write a coefficient array back as ``A_ORDER`` plus ``A_p_q`` keys."""
    if coefficients is None:
        return {}
    array = np.asarray(coefficients, dtype=float)
    order = array.shape[0] - 1
    header: Dict[str, Any] = {f"{letter}_ORDER": order}
    for p in range(array.shape[0]):
        for q in range(array.shape[1]):
            if array[p, q]:
                header[f"{letter}_{p}_{q}"] = float(array[p, q])
    return header


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
