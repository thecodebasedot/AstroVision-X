"""Loading survey data products the way they actually arrive.

A calibrated image from a real survey is not one array in one HDU. It is a
multi-extension file carrying the science pixels, a bad-pixel or data-quality
plane, and a weight or variance plane, with the numbers that turn counts into
photons -- gain, read noise, saturation, the flux unit -- scattered through the
header under whichever keywords that pipeline's authors chose. Everything a
measurement needs is there; none of it is where a single-HDU reader looks.

This module finds it. The rules it applies are worth stating because each one
is a place where getting it silently wrong produces plausible numbers:

* **Extensions are found by name, then by shape.** ``SCI``, ``IMAGE``,
  ``MASK``, ``DQ``, ``WHT``, ``VAR``, ``ERR`` and their common spellings are
  recognised; failing that, the first two-dimensional HDU is the image and
  nothing is assumed about the rest.
* **A data-quality plane is a bitmask, not a boolean.** Any non-zero value
  means *something* was flagged, but which bits matter is a per-pipeline
  convention; by default every set bit masks the pixel, and the caller can
  narrow that to specific bits when they know the convention.
* **Weight is inverse variance, and zero weight means no data.** A weight map
  is converted to a per-pixel uncertainty with ``sigma = 1 / sqrt(w)``; a pixel
  of zero weight is masked rather than given infinite error, because it is
  not a noisy measurement, it is the absence of one.
* **Saturated pixels are masked from the header's own limit.** A saturated
  star has a flat top and a wrong flux, and downstream nothing can tell unless
  it is flagged here.
* **The flux unit decides whether gain applies.** Pixels in electrons have
  already had the gain applied; multiplying by it again doubles the Poisson
  noise estimate. ``BUNIT`` is read and the effective gain set accordingly.
* **Large files are memory-mapped.** A 16k x 16k frame is a gigabyte per plane
  in float32; the tiled processor reads the region it needs and no more.

No real archive file was reachable when this was written. The loader is
exercised against files written in these conventions, and the conventions
themselves are the ones the major pipelines document -- but a keyword this
does not know about is a keyword it will not find, and the report it returns
says what it looked for and what it used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.exceptions import DataError
from ..core.logging import get_logger
from .image import AstroImage, _first_key, _float_or_none, _header_time
from .image import _BAND_KEYS, _EXPTIME_KEYS
from .wcs import wcs_from_header

log = get_logger("io.survey")

#: EXTNAME values recognised for each plane, most common first.
SCIENCE_NAMES: Sequence[str] = ("SCI", "IMAGE", "SCIENCE", "DATA", "PRIMARY", "IMG")
MASK_NAMES: Sequence[str] = ("MASK", "MSK", "DQ", "BPM", "FLAGS", "FLAG", "BADPIX")
WEIGHT_NAMES: Sequence[str] = ("WHT", "WEIGHT", "WGT", "IVAR", "INVVAR")
VARIANCE_NAMES: Sequence[str] = ("VAR", "VARIANCE")
ERROR_NAMES: Sequence[str] = ("ERR", "ERROR", "SIGMA", "UNC", "NOISE", "RMS")

#: Header keywords, in order of preference, for each physical quantity.
GAIN_KEYS: Sequence[str] = ("GAIN", "EGAIN", "GAINEFF", "CCDGAIN")
READNOISE_KEYS: Sequence[str] = ("RDNOISE", "RON", "READNOIS", "READNOISE", "RDNOIS")
SATURATE_KEYS: Sequence[str] = ("SATURATE", "SATLEVEL", "SATURATION", "DATAMAX", "SATLEV")
ZERO_POINT_KEYS: Sequence[str] = ("MAGZP", "MAGZERO", "ZP", "ZEROPT", "PHOTZP", "MAGZPT")
UNIT_KEYS: Sequence[str] = ("BUNIT", "UNITS")

#: Flux units that mean the gain has already been applied.
ELECTRON_UNITS = {"electron", "electrons", "e-", "e", "e/s", "electron/s",
                  "electrons/s", "e-/s", "el", "el/s"}

#: Above this many pixels a file is memory-mapped rather than read whole.
MEMMAP_ABOVE = 16_000_000


@dataclass
class SurveyLoadReport:
    """What the loader found, what it used, and what it could not find."""

    path: str = ""
    science_hdu: Optional[int] = None
    mask_hdu: Optional[int] = None
    weight_hdu: Optional[int] = None
    variance_hdu: Optional[int] = None
    error_hdu: Optional[int] = None
    gain: float = float("nan")
    gain_source: str = ""
    read_noise: float = float("nan")
    saturation: float = float("nan")
    unit: str = ""
    zero_point: float = float("nan")
    pixel_scale: float = float("nan")
    n_masked: int = 0
    n_saturated: int = 0
    n_zero_weight: int = 0
    memmapped: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in self.__dict__.items()}


def _hdu_name(hdu) -> str:
    return str(getattr(hdu, "name", "") or "").strip().upper()


def _is_image(hdu) -> bool:
    data = getattr(hdu, "data", None)
    return data is not None and np.ndim(data) == 2


def find_planes(hdul) -> Dict[str, Optional[int]]:
    """Locate the science, mask, weight, variance and error HDUs.

    Names first, because a pipeline that labels its planes has told us what
    they are. Shape second, because one that does not has still put the image
    first more often than not.
    """
    found: Dict[str, Optional[int]] = {"science": None, "mask": None,
                                       "weight": None, "variance": None,
                                       "error": None}
    for index, hdu in enumerate(hdul):
        if not _is_image(hdu):
            continue
        name = _hdu_name(hdu)
        for role, names in (("mask", MASK_NAMES), ("weight", WEIGHT_NAMES),
                            ("variance", VARIANCE_NAMES), ("error", ERROR_NAMES),
                            ("science", SCIENCE_NAMES)):
            if name in names and found[role] is None:
                found[role] = index
                break
    if found["science"] is None:
        # No labelled science plane: the first 2-D HDU that is not already
        # claimed as an auxiliary plane.
        claimed = {v for v in found.values() if v is not None}
        for index, hdu in enumerate(hdul):
            if _is_image(hdu) and index not in claimed:
                found["science"] = index
                break
    return found


def _header_to_dict(header) -> Dict[str, Any]:
    return {k: header[k] for k in header
            if k and k not in ("COMMENT", "HISTORY", "")}


def _read_plane(hdu, memmap: bool) -> np.ndarray:
    data = hdu.data
    if memmap:
        return data                        # leave it lazy; callers slice it
    return np.array(data, dtype=float)


def load_survey_image(path: str, mask_bits: Optional[int] = None,
                      memmap: Optional[bool] = None,
                      saturation: Optional[float] = None,
                      name: Optional[str] = None
                      ) -> Tuple[AstroImage, SurveyLoadReport]:
    """Load a multi-extension survey product into an :class:`AstroImage`.

    ``mask_bits`` narrows a data-quality bitmask to the bits that should mask
    a pixel; by default any set bit does. ``saturation`` overrides the
    header's limit. ``memmap`` defaults to on for files above
    :data:`MEMMAP_ABOVE` pixels.

    Returns the image and a report of what was found, so a caller can check
    that the loader used the planes it expected rather than trusting that it
    did.
    """
    fits = try_import("astropy.io.fits")
    if fits is None:                                    # pragma: no cover
        raise DataError("astropy is required to read multi-extension survey files")
    if not os.path.exists(path):
        raise DataError(f"file not found: {path}")

    report = SurveyLoadReport(path=os.path.abspath(path))
    hdul = fits.open(path, memmap=True)
    try:
        planes = find_planes(hdul)
        if planes["science"] is None:
            raise DataError(f"{path} contains no two-dimensional image HDU")
        science_hdu = hdul[planes["science"]]
        shape = tuple(int(v) for v in np.shape(science_hdu.data))
        use_memmap = (shape[0] * shape[1] > MEMMAP_ABOVE) if memmap is None else bool(memmap)
        report.memmapped = use_memmap
        report.science_hdu = planes["science"]

        # Headers: the primary carries the observation, the extension carries
        # the plane.  Real files split the keywords between them arbitrarily,
        # so both are merged, the extension winning where they conflict.
        header: Dict[str, Any] = {}
        header.update(_header_to_dict(hdul[0].header))
        header.update(_header_to_dict(science_hdu.header))

        data = _read_plane(science_hdu, use_memmap)
        if not use_memmap:
            data = np.asarray(data, dtype=float)

        mask = np.zeros(shape, dtype=bool)
        if planes["mask"] is not None:
            report.mask_hdu = planes["mask"]
            raw = np.asarray(hdul[planes["mask"]].data)
            if raw.shape != shape:
                report.notes.append(f"mask plane shape {raw.shape} does not match "
                                    f"the image {shape}; ignored")
            else:
                if mask_bits is None:
                    flagged = raw != 0
                else:
                    flagged = (raw.astype(np.int64) & int(mask_bits)) != 0
                mask |= flagged
                report.n_masked = int(flagged.sum())

        uncertainty: Optional[np.ndarray] = None
        if planes["error"] is not None:
            report.error_hdu = planes["error"]
            uncertainty = np.asarray(hdul[planes["error"]].data, dtype=float)
        elif planes["variance"] is not None:
            report.variance_hdu = planes["variance"]
            variance = np.asarray(hdul[planes["variance"]].data, dtype=float)
            uncertainty = np.sqrt(np.clip(variance, 0.0, None))
        elif planes["weight"] is not None:
            report.weight_hdu = planes["weight"]
            weight = np.asarray(hdul[planes["weight"]].data, dtype=float)
            # Zero weight is not infinite noise, it is the absence of data.
            empty = ~(weight > 0) | ~np.isfinite(weight)
            report.n_zero_weight = int(empty.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                uncertainty = np.where(empty, np.nan, 1.0 / np.sqrt(weight))
            mask |= empty
        if uncertainty is not None and uncertainty.shape != shape:
            report.notes.append("noise plane shape does not match the image; ignored")
            uncertainty = None

        unit = str(_first_key(header, UNIT_KEYS, "") or "").strip()
        report.unit = unit
        gain = _float_or_none(_first_key(header, GAIN_KEYS, None))
        if unit.lower() in ELECTRON_UNITS:
            # Already in electrons: applying the gain again would double the
            # Poisson noise estimate.  The header gain is kept in the report
            # so nothing is lost, but the effective gain is one.
            report.gain, report.gain_source = 1.0, f"unit {unit!r} implies electrons"
            if gain is not None:
                report.notes.append(f"header gain {gain} not applied: pixels are "
                                    "already in electrons")
        elif gain is not None and gain > 0:
            report.gain, report.gain_source = float(gain), "header"
        else:
            report.gain, report.gain_source = 1.0, "assumed"
            report.notes.append("no gain keyword; Poisson noise assumes 1 e-/count")
        header["GAIN"] = report.gain

        read_noise = _float_or_none(_first_key(header, READNOISE_KEYS, None))
        report.read_noise = float(read_noise) if read_noise is not None else float("nan")

        limit = saturation if saturation is not None \
            else _float_or_none(_first_key(header, SATURATE_KEYS, None))
        if limit is not None and np.isfinite(limit):
            report.saturation = float(limit)
            if not use_memmap:
                saturated = np.asarray(data) >= float(limit)
                report.n_saturated = int(saturated.sum())
                mask |= saturated
            else:
                report.notes.append("saturation is applied per tile on a "
                                    "memory-mapped image")
        else:
            report.notes.append("no saturation keyword; saturated stars are not flagged")

        zero_point = _float_or_none(_first_key(header, ZERO_POINT_KEYS, None))
        if zero_point is not None:
            report.zero_point = float(zero_point)
            header["MAGZP"] = float(zero_point)

        image = AstroImage(
            data=data if not use_memmap else np.asarray(data, dtype=np.float32),
            header=header, wcs=wcs_from_header(header),
            mask=mask if mask.any() else None,
            uncertainty=uncertainty,
            name=name or os.path.basename(path),
            band=str(_first_key(header, _BAND_KEYS, "clear")),
            exposure_time=_float_or_none(_first_key(header, _EXPTIME_KEYS, None)),
            mjd=_header_time(header))
        image.meta.update({"source_path": os.path.abspath(path),
                           "survey_load": report.to_dict()})
        report.pixel_scale = image.pixel_scale if image.wcs is not None else float("nan")
        if image.wcs is None:
            report.notes.append("no WCS in the header; positions are pixels only")
    finally:
        hdul.close()

    log.info("loaded %s: %dx%d, planes science=%s mask=%s noise=%s, gain %.2f (%s)",
             os.path.basename(path), shape[1], shape[0], report.science_hdu,
             report.mask_hdu,
             next((h for h in (report.error_hdu, report.variance_hdu,
                               report.weight_hdu) if h is not None), None),
             report.gain, report.gain_source)
    return image, report


def write_survey_image(path: str, data: np.ndarray,
                       header: Optional[Dict[str, Any]] = None,
                       mask: Optional[np.ndarray] = None,
                       weight: Optional[np.ndarray] = None,
                       variance: Optional[np.ndarray] = None,
                       overwrite: bool = True) -> str:
    """Write a multi-extension product in the conventions the loader reads.

    Exists so the loading path can be tested against files rather than
    arrays, and so this package's own outputs are readable by anything that
    understands the same conventions.
    """
    fits = try_import("astropy.io.fits")
    if fits is None:                                    # pragma: no cover
        raise DataError("astropy is required to write multi-extension files")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    primary = fits.PrimaryHDU()
    for key, value in (header or {}).items():
        try:
            primary.header[str(key)[:8].upper()] = value
        except Exception:                               # pragma: no cover
            continue
    science = fits.ImageHDU(np.asarray(data, dtype=np.float32), name="SCI")
    for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2",
                "CD2_1", "CD2_2", "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1",
                "PC2_2", "CTYPE1", "CTYPE2"):
        if header and key in header:
            science.header[key] = header[key]
    hdus = [primary, science]
    if mask is not None:
        hdus.append(fits.ImageHDU(np.asarray(mask, dtype=np.int16), name="MASK"))
    if weight is not None:
        hdus.append(fits.ImageHDU(np.asarray(weight, dtype=np.float32), name="WHT"))
    if variance is not None:
        hdus.append(fits.ImageHDU(np.asarray(variance, dtype=np.float32), name="VAR"))
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path
