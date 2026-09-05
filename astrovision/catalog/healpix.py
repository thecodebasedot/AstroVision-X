"""A HEALPix sky index in NumPy alone.

HEALPix divides the sphere into 12 base faces and each face into
``nside x nside`` equal-area pixels. In the *nested* scheme a pixel's index
interleaves the bits of its position within the face, so the four children
of a pixel at one resolution are four consecutive indices at the next: a
coarse pixel is a prefix of every fine pixel inside it, and a cone on the
sky is a short list of pixels at whatever resolution suits the cone. That is
the property a catalog wants for a sky index, and the reason the nested
scheme, not the ring scheme, is implemented here.

The algorithms are those of the HEALPix reference code (Górski et al.
2005, ApJ 622, 759), written with array operations so a million positions
index in a few tens of milliseconds. The test suite checks the round trip
``pix2ang(ang2pix(p)) == p`` for every pixel at several resolutions, that
uniform random positions fall in every pixel equally, that neighbouring
positions share coarse pixels, and -- when healpy happens to be installed --
that every index agrees with it exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Row of each base face (in units of nside) and its column offset, from
#: the reference implementation.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def npix(nside: int) -> int:
    """Number of pixels at ``nside``: ``12 * nside**2``."""
    return 12 * int(nside) ** 2


def order(nside: int) -> int:
    """``log2(nside)``; nside must be a power of two."""
    nside = int(nside)
    if nside < 1 or nside & (nside - 1):
        raise ValueError(f"nside must be a power of two, got {nside}")
    return nside.bit_length() - 1


def pixel_resolution_deg(nside: int) -> float:
    """Approximate side of one pixel in degrees (sqrt of its solid angle)."""
    return float(np.degrees(np.sqrt(4.0 * np.pi / npix(nside))))


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Insert a zero between every bit of ``v`` (for 32-bit inputs)."""
    v = v.astype(np.int64) & 0xFFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def _compress_bits(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_spread_bits`: keep every other bit."""
    v = v.astype(np.int64) & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    v = (v | (v >> 16)) & 0x00000000FFFFFFFF
    return v


def _xyf2nest(ix: np.ndarray, iy: np.ndarray, face: np.ndarray, nside: int) -> np.ndarray:
    return (face.astype(np.int64) << (2 * order(nside))) + _spread_bits(ix) + (_spread_bits(iy) << 1)


def _nest2xyf(pix: np.ndarray, nside: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pix = np.asarray(pix, dtype=np.int64)
    face = pix >> (2 * order(nside))
    within = pix & (nside * nside - 1)
    return _compress_bits(within), _compress_bits(within >> 1), face


def ang2pix(nside: int, ra_deg, dec_deg) -> np.ndarray:
    """Nested pixel index of each ``(ra, dec)`` in degrees."""
    nside = int(nside)
    order(nside)
    ra = np.radians(np.atleast_1d(np.asarray(ra_deg, dtype=float)))
    dec = np.radians(np.atleast_1d(np.asarray(dec_deg, dtype=float)))
    z = np.sin(dec)
    za = np.abs(z)
    tt = np.mod(ra, 2.0 * np.pi) * (2.0 / np.pi)      # in [0, 4)

    ix = np.empty(z.shape, dtype=np.int64)
    iy = np.empty(z.shape, dtype=np.int64)
    face = np.empty(z.shape, dtype=np.int64)

    equatorial = za <= 2.0 / 3.0
    if equatorial.any():
        te, ze = tt[equatorial], z[equatorial]
        temp1 = nside * (0.5 + te)
        temp2 = nside * ze * 0.75
        jp = np.floor(temp1 - temp2).astype(np.int64)
        jm = np.floor(temp1 + temp2).astype(np.int64)
        ifp = jp >> order(nside)
        ifm = jm >> order(nside)
        f = np.where(ifp == ifm, (ifp & 3) + 4, np.where(ifp < ifm, ifp & 3, (ifm & 3) + 8))
        face[equatorial] = f
        ix[equatorial] = jm & (nside - 1)
        iy[equatorial] = nside - (jp & (nside - 1)) - 1

    polar = ~equatorial
    if polar.any():
        tp_all, zp, zap = tt[polar], z[polar], za[polar]
        ntt = np.minimum(np.floor(tp_all).astype(np.int64), 3)
        tp = tp_all - ntt
        tmp = nside * np.sqrt(3.0 * (1.0 - zap))
        jp = np.minimum(np.floor(tp * tmp).astype(np.int64), nside - 1)
        jm = np.minimum(np.floor((1.0 - tp) * tmp).astype(np.int64), nside - 1)
        north = zp >= 0
        face[polar] = np.where(north, ntt, ntt + 8)
        ix[polar] = np.where(north, nside - jm - 1, jp)
        iy[polar] = np.where(north, nside - jp - 1, jm)

    return _xyf2nest(ix, iy, face, nside)


def pix2ang(nside: int, pix) -> Tuple[np.ndarray, np.ndarray]:
    """Centre ``(ra, dec)`` in degrees of each nested pixel."""
    nside = int(nside)
    ix, iy, face = _nest2xyf(np.atleast_1d(pix), nside)
    jr = _JRLL[face] * nside - ix - iy - 1
    fact2 = 4.0 / npix(nside)
    fact1 = 2.0 / (3.0 * nside)

    z = np.empty(jr.shape, dtype=float)
    nr = np.empty(jr.shape, dtype=np.int64)
    kshift = np.zeros(jr.shape, dtype=np.int64)

    north_cap = jr < nside
    south_cap = jr > 3 * nside
    band = ~(north_cap | south_cap)
    nr[north_cap] = jr[north_cap]
    z[north_cap] = 1.0 - nr[north_cap] ** 2 * fact2
    nr[south_cap] = 4 * nside - jr[south_cap]
    z[south_cap] = nr[south_cap] ** 2 * fact2 - 1.0
    nr[band] = nside
    z[band] = (2 * nside - jr[band]) * fact1
    kshift[band] = (jr[band] - nside) & 1

    jp = (_JPLL[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp)
    jp = np.where(jp < 1, jp + 4 * nside, jp)
    phi = (jp - (kshift + 1) * 0.5) * (np.pi / 2.0 / nr)
    return np.degrees(phi), np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))


def angular_separation(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in degrees (Vincenty, stable at all angles)."""
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(v, dtype=float)) for v in (ra1, dec1, ra2, dec2))
    d_ra = ra2 - ra1
    sin1, cos1, sin2, cos2 = np.sin(dec1), np.cos(dec1), np.sin(dec2), np.cos(dec2)
    num = np.hypot(cos2 * np.sin(d_ra), cos1 * sin2 - sin1 * cos2 * np.cos(d_ra))
    den = sin1 * sin2 + cos1 * cos2 * np.cos(d_ra)
    return np.degrees(np.arctan2(num, den))


class SkyIndex:
    """Pixel lookups and cone queries at one fixed ``nside``.

    Pixel centres are cached so a cone query is one vectorised distance
    calculation over ``12 nside^2`` centres: at nside 128 that is 200k
    centres and a few milliseconds, and the resulting pixel list is what the
    database's ``WHERE healpix IN (...)`` clause consumes.
    """

    def __init__(self, nside: int = 128):
        self.nside = int(nside)
        order(self.nside)
        self._centres: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._centres is None:
            self._centres = pix2ang(self.nside, np.arange(npix(self.nside)))
        return self._centres

    @property
    def pixel_radius_deg(self) -> float:
        """Largest centre-to-corner distance of any pixel, with margin."""
        # The reference max pixel radius is about 1.36 x the mean pixel side
        # near the poles; 1.5 is a safe bound for every nside.
        return 1.5 * pixel_resolution_deg(self.nside)

    def pixel(self, ra_deg, dec_deg) -> np.ndarray:
        return ang2pix(self.nside, ra_deg, dec_deg)

    def cone(self, ra_deg: float, dec_deg: float, radius_deg: float,
             nside_out: Optional[int] = None) -> np.ndarray:
        """Every pixel at ``nside_out`` (default: this index's nside) that can
        contain a point within ``radius_deg`` of the position.

        The search starts at this index's cached resolution and refines: a
        touched pixel is replaced by those of its four children whose centre
        lies within the radius plus the child's own reach. Only pixels along
        the cone's edge are ever subdivided, so a 5-arcsecond cone at nside
        8192 costs a few dozen centre calculations, not the 800 million a
        flat scan of that resolution would.
        """
        nside_out = self.nside if nside_out is None else int(nside_out)
        if nside_out < self.nside:
            raise ValueError("nside_out must be at least the index's nside")
        order(nside_out)
        ra_c, dec_c = self.centres
        reach = float(radius_deg) + self.pixel_radius_deg
        band = np.abs(dec_c - float(dec_deg)) <= reach
        pixels = np.flatnonzero(band)
        if pixels.size:
            sep = angular_separation(ra_deg, dec_deg, ra_c[pixels], dec_c[pixels])
            pixels = pixels[sep <= reach]
        nside = self.nside
        while nside < nside_out and pixels.size:
            nside *= 2
            children = (pixels[:, None] * 4 + np.arange(4)[None, :]).ravel()
            c_ra, c_dec = pix2ang(nside, children)
            reach = float(radius_deg) + 1.5 * pixel_resolution_deg(nside)
            sep = angular_separation(ra_deg, dec_deg, c_ra, c_dec)
            pixels = children[sep <= reach]
        return pixels
