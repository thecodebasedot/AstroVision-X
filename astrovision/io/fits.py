"""FITS reading and writing.

Astropy is used when available.  Otherwise a self-contained reader/writer
handles the common single-image case (2880-byte blocks, integer and IEEE
float ``BITPIX``, ``BSCALE``/``BZERO`` scaling), so AstroVision-X can open
real telescope data with NumPy alone.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.exceptions import DataError
from ..core.logging import get_logger

log = get_logger("io.fits")

BLOCK = 2880
CARD = 80

#: FITS ``BITPIX`` -> NumPy big-endian dtype.
_BITPIX_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
_DTYPE_BITPIX = {"uint8": 8, "int16": 16, "int32": 32, "int64": 64,
                 "float32": -32, "float64": -64}

FITS_EXTENSIONS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fits.fz")


def is_fits(path: str) -> bool:
    """True when ``path`` looks like a FITS file by extension."""
    lowered = str(path).lower()
    return lowered.endswith(FITS_EXTENSIONS)


# --------------------------------------------------------------------------
# header parsing
# --------------------------------------------------------------------------
def _parse_value(raw: str) -> Any:
    """Convert a FITS card value field into a Python object."""
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("'"):
        end = raw.rfind("'")
        return raw[1:end].strip() if end > 0 else raw[1:].strip()
    token = raw.split("/")[0].strip()
    if token in ("T", "F"):
        return token == "T"
    if not token:
        return None
    try:
        if any(c in token for c in ".eEdD"):
            return float(token.replace("D", "E").replace("d", "e"))
        return int(token)
    except ValueError:
        return token


def parse_header_block(text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse 80-character cards; returns ``(header, saw_end)``."""
    header: Dict[str, Any] = {}
    for i in range(0, len(text), CARD):
        card = text[i:i + CARD]
        if len(card) < 8:
            break
        key = card[:8].strip()
        if key == "END":
            return header, True
        if not key or key in ("COMMENT", "HISTORY"):
            continue
        if card[8:10] == "= ":
            header[key] = _parse_value(card[10:])
    return header, False


def _format_card(key: str, value: Any) -> str:
    """Render one 80-character header card."""
    key = str(key).upper()[:8].ljust(8)
    if isinstance(value, bool):
        rendered = "T" if value else "F"
        field = rendered.rjust(20)
    elif isinstance(value, (int, np.integer)):
        field = str(int(value)).rjust(20)
    elif isinstance(value, (float, np.floating)):
        field = f"{float(value):.10G}".rjust(20)
    elif value is None:
        field = " " * 20
    else:
        text = str(value).replace("'", "''")[:66]
        field = f"'{text}'".ljust(20)
    return f"{key}= {field}".ljust(CARD)[:CARD]


def _pad_block(payload: bytes) -> bytes:
    remainder = len(payload) % BLOCK
    return payload if remainder == 0 else payload + b" " * (BLOCK - remainder)


# --------------------------------------------------------------------------
# pure-NumPy reader / writer
# --------------------------------------------------------------------------
def _open_maybe_gzip(path: str):
    if str(path).lower().endswith(".gz"):
        import gzip
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_fits_numpy(path: str, hdu: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an image HDU without Astropy."""
    with _open_maybe_gzip(path) as handle:
        index = 0
        while True:
            header: Dict[str, Any] = {}
            saw_end = False
            while not saw_end:
                block = handle.read(BLOCK)
                if not block or len(block) < BLOCK:
                    raise DataError(f"truncated FITS header in {path}")
                part, saw_end = parse_header_block(block.decode("ascii", "replace"))
                header.update(part)

            naxis = int(header.get("NAXIS", 0))
            dims = [int(header.get(f"NAXIS{i}", 0)) for i in range(1, naxis + 1)]
            count = int(np.prod(dims)) if dims else 0
            bitpix = int(header.get("BITPIX", -32))
            dtype = _BITPIX_DTYPE.get(bitpix)
            if dtype is None:
                raise DataError(f"unsupported BITPIX={bitpix} in {path}")
            nbytes = count * abs(bitpix) // 8
            padded = ((nbytes + BLOCK - 1) // BLOCK) * BLOCK

            if index == hdu:
                if count == 0:
                    raise DataError(f"HDU {hdu} of {path} contains no image data")
                raw = handle.read(nbytes)
                if len(raw) < nbytes:
                    raise DataError(f"truncated FITS data in {path}")
                data = np.frombuffer(raw, dtype=dtype).reshape(dims[::-1]).astype(float)
                bscale = float(header.get("BSCALE", 1.0) or 1.0)
                bzero = float(header.get("BZERO", 0.0) or 0.0)
                if bscale != 1.0 or bzero != 0.0:
                    data = data * bscale + bzero
                blank = header.get("BLANK")
                if blank is not None and bitpix > 0:
                    data[data == float(blank) * bscale + bzero] = np.nan
                return np.ascontiguousarray(data), header

            handle.seek(padded, os.SEEK_CUR)
            index += 1
            if index > 64:
                raise DataError(f"HDU {hdu} not found in {path}")


def _write_fits_numpy(path: str, data: np.ndarray,
                      header: Optional[Dict[str, Any]] = None,
                      overwrite: bool = True) -> str:
    """Write a single-image FITS file without Astropy."""
    if os.path.exists(path) and not overwrite:
        raise DataError(f"{path} exists and overwrite=False")
    array = np.asarray(data)
    if array.dtype.kind not in "fiu":
        array = array.astype(np.float32)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    bitpix = _DTYPE_BITPIX.get(array.dtype.name, -32)
    big_endian = array.astype(_BITPIX_DTYPE[bitpix])

    cards: List[str] = [
        _format_card("SIMPLE", True),
        _format_card("BITPIX", bitpix),
        _format_card("NAXIS", array.ndim),
    ]
    for i, size in enumerate(array.shape[::-1], start=1):
        cards.append(_format_card(f"NAXIS{i}", int(size)))
    reserved = {"SIMPLE", "BITPIX", "NAXIS", "END", "EXTEND", "BSCALE", "BZERO"}
    for key, value in (header or {}).items():
        key_up = str(key).upper()
        if key_up in reserved or key_up.startswith("NAXIS"):
            continue
        cards.append(_format_card(key, value))
    cards.append("END".ljust(CARD))

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(_pad_block("".join(cards).encode("ascii", "replace")))
        handle.write(_pad_block(big_endian.tobytes()))
    return path


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def read_fits(path: str, hdu: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read image data and header from ``path``.

    When ``hdu`` is ``None`` the first HDU that actually holds image data
    is used, which is what MEF files from most instruments need.
    """
    if not os.path.exists(path):
        raise DataError(f"file not found: {path}")
    astropy_io = try_import("astropy.io.fits")
    if astropy_io is not None:
        with astropy_io.open(path, memmap=False) as hdul:
            index = hdu
            if index is None:
                index = next(
                    (i for i, h in enumerate(hdul)
                     if getattr(h, "data", None) is not None and np.ndim(h.data) >= 2),
                    0,
                )
            item = hdul[index]
            if item.data is None:
                raise DataError(f"HDU {index} of {path} has no data")
            data = np.array(item.data, dtype=float)
            header = {k: item.header[k] for k in item.header if k not in ("COMMENT", "HISTORY")}
            return data, header
    log.debug("astropy unavailable; using built-in FITS reader for %s", path)
    return _read_fits_numpy(path, 0 if hdu is None else int(hdu))


def write_fits(path: str, data: np.ndarray, header: Optional[Dict[str, Any]] = None,
               overwrite: bool = True) -> str:
    """Write ``data`` to a FITS file, returning the path written."""
    astropy_io = try_import("astropy.io.fits")
    if astropy_io is not None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        hdu = astropy_io.PrimaryHDU(np.asarray(data, dtype=np.float32))
        for key, value in (header or {}).items():
            try:
                hdu.header[str(key)[:8].upper()] = value
            except Exception:  # pragma: no cover - unrepresentable values
                continue
        hdu.writeto(path, overwrite=overwrite)
        return path
    return _write_fits_numpy(path, data, header, overwrite)


def list_hdus(path: str) -> List[Dict[str, Any]]:
    """Summarise the HDUs in a FITS file (index, name, shape)."""
    astropy_io = try_import("astropy.io.fits")
    if astropy_io is not None:
        with astropy_io.open(path, memmap=False) as hdul:
            return [
                {"index": i, "name": h.name,
                 "shape": None if h.data is None else tuple(np.shape(h.data))}
                for i, h in enumerate(hdul)
            ]
    data, header = _read_fits_numpy(path, 0)
    return [{"index": 0, "name": header.get("EXTNAME", "PRIMARY"), "shape": data.shape}]
