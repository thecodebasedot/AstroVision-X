"""Postage stamps as PNG, with nothing but the standard library.

The vetting page needs an image of each candidate and the package promises
to run on NumPy alone, so this writes PNG by hand: a signature, an IHDR
chunk, one zlib-compressed IDAT of unfiltered rows, IEND. Thirty lines,
and no Pillow or Matplotlib on the path that an astronomer's browser reads.

The stretch is asinh about the stamp's own sky, scaled by its own noise:
the core of a bright star and the wings of a faint galaxy are both visible,
and two stamps of different depth look alike. It is the same stretch the
HTML report uses.
"""

from __future__ import annotations

import struct
import zlib
from typing import Optional, Tuple

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png(pixels: np.ndarray) -> bytes:
    """Encode an 8-bit greyscale ``(h, w)`` or RGB ``(h, w, 3)`` array."""
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        colour_type, rows = 0, array
    elif array.ndim == 3 and array.shape[2] == 3:
        colour_type, rows = 2, array
    else:
        raise ValueError("expected an (h, w) or (h, w, 3) array")
    height, width = rows.shape[0], rows.shape[1]
    raw = b"".join(b"\x00" + np.ascontiguousarray(rows[y]).tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (_SIGNATURE + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))


def decode_png_size(data: bytes) -> Tuple[int, int]:
    """``(width, height)`` from a PNG header, for tests and sanity checks."""
    if not data.startswith(_SIGNATURE):
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def stretch(stamp: np.ndarray, sky: Optional[float] = None, noise: Optional[float] = None,
            softening: float = 3.0, ceiling: float = 300.0) -> np.ndarray:
    """Asinh stretch of a stamp to 8-bit grey.

    ``sky`` and ``noise`` default to the stamp's median and a robust scatter
    (1.4826 MAD); ``softening`` is the number of noise units where the
    stretch turns from linear to logarithmic; ``ceiling`` is the number of
    noise units mapped to white.
    """
    data = np.nan_to_num(np.asarray(stamp, dtype=float), nan=0.0)
    if sky is None:
        sky = float(np.median(data))
    if noise is None:
        mad = float(np.median(np.abs(data - sky)))
        noise = max(1.4826 * mad, 1e-9)
    scaled = (data - sky) / (float(noise) * float(softening))
    top = np.arcsinh(float(ceiling) / float(softening))
    value = (np.arcsinh(scaled) + np.arcsinh(1.0)) / (top + np.arcsinh(1.0))
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


def upscale(pixels: np.ndarray, factor: int = 4) -> np.ndarray:
    """Nearest-neighbour enlargement so a 64-pixel stamp is legible."""
    factor = max(1, int(factor))
    return np.repeat(np.repeat(pixels, factor, axis=0), factor, axis=1)


def stamp_png(stamp: np.ndarray, factor: int = 4, **stretch_kwargs) -> bytes:
    """One call from pixels to bytes a browser will draw."""
    return encode_png(upscale(stretch(stamp, **stretch_kwargs), factor))
