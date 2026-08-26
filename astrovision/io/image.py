"""The :class:`AstroImage` container: pixels plus everything about them."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.exceptions import DataError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, sigma_clipped_stats
from .fits import is_fits, read_fits, write_fits
from .wcs import SimpleWCS, wcs_from_header

log = get_logger("io.image")

#: Header keywords consulted when filling in observation metadata.
_TIME_KEYS = ("MJD-OBS", "MJD", "JD", "DATE-OBS", "EXPSTART")
_BAND_KEYS = ("FILTER", "FILTER1", "BAND", "FILTNAM")
_EXPTIME_KEYS = ("EXPTIME", "EXPOSURE", "ITIME", "TELAPSE")


@dataclass
class AstroImage:
    """A calibrated (or raw) 2-D astronomical image and its metadata.

    The container keeps the pixel array alongside the header, an optional
    WCS, a bad-pixel mask, a per-pixel uncertainty map, and the background
    model produced by the preprocessing stage.
    """

    data: np.ndarray
    header: Dict[str, Any] = field(default_factory=dict)
    wcs: Optional[SimpleWCS] = None
    mask: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    background_rms: Optional[np.ndarray] = None
    name: str = "image"
    band: str = "clear"
    mjd: Optional[float] = None
    exposure_time: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = as_float_image(self.data)
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.data.shape:
                raise DataError("mask shape does not match image shape")
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=float)
            if self.uncertainty.shape != self.data.shape:
                raise DataError("uncertainty shape does not match image shape")

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_fits(cls, path: str, hdu: Optional[int] = None, **kwargs) -> "AstroImage":
        """Load a FITS file, deriving WCS/band/time from the header."""
        data, header = read_fits(path, hdu)
        image = cls(data=data, header=header,
                    name=kwargs.pop("name", os.path.basename(path)), **kwargs)
        image.wcs = image.wcs or wcs_from_header(header)
        image.band = _first_key(header, _BAND_KEYS, image.band)
        image.exposure_time = _float_or_none(_first_key(header, _EXPTIME_KEYS, None))
        image.mjd = _header_time(header)
        image.meta.setdefault("source_path", os.path.abspath(path))
        return image

    @classmethod
    def from_array(cls, data: np.ndarray, **kwargs) -> "AstroImage":
        """Wrap a plain NumPy array."""
        return cls(data=data, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> "AstroImage":
        """Load FITS, ``.npy``/``.npz``, or any format Pillow/Matplotlib reads."""
        lowered = str(path).lower()
        if is_fits(lowered):
            return cls.from_fits(path, **kwargs)
        if lowered.endswith(".npy"):
            return cls(np.load(path), name=os.path.basename(path), **kwargs)
        if lowered.endswith(".npz"):
            with np.load(path) as bundle:
                key = "data" if "data" in bundle else list(bundle.keys())[0]
                return cls(bundle[key], name=os.path.basename(path), **kwargs)
        pil = try_import("PIL.Image")
        if pil is not None:
            with pil.open(path) as handle:
                return cls(np.asarray(handle, dtype=float),
                           name=os.path.basename(path), **kwargs)
        plt_image = try_import("matplotlib.image")
        if plt_image is not None:
            return cls(np.asarray(plt_image.imread(path), dtype=float),
                       name=os.path.basename(path), **kwargs)
        raise DataError(f"cannot read image format: {path}")

    # -- basic properties --------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def pixel_scale(self) -> float:
        """Arcsec per pixel from the WCS, defaulting to 1.0."""
        return self.wcs.pixel_scale if self.wcs is not None else 1.0

    @property
    def valid_mask(self) -> np.ndarray:
        """``True`` where a pixel is finite and not flagged bad."""
        good = np.isfinite(self.data)
        if self.mask is not None:
            good &= ~self.mask
        return good

    def stats(self) -> Dict[str, float]:
        """Robust image statistics used across the pipeline and reports."""
        values = self.data[self.valid_mask]
        if values.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0,
                    "max": 0.0, "n_valid": 0}
        mean, median, std = sigma_clipped_stats(values)
        return {
            "mean": float(mean), "median": float(median), "std": float(std),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "n_valid": int(values.size),
        }

    # -- derived views -----------------------------------------------------
    def subtracted(self) -> np.ndarray:
        """Background-subtracted pixels (the array itself if no model exists)."""
        if self.background is None:
            return self.data
        return self.data - self.background

    def rms_map(self) -> np.ndarray:
        """Per-pixel noise, from the background RMS, uncertainty, or stats."""
        if self.background_rms is not None:
            return self.background_rms
        if self.uncertainty is not None:
            return self.uncertainty
        return np.full(self.shape, max(self.stats()["std"], 1e-9), dtype=float)

    def cutout(self, x: float, y: float, size: int = 64,
               subtract_background: bool = False) -> np.ndarray:
        """Square postage stamp centred on ``(x, y)``, zero-padded at edges."""
        source = self.subtracted() if subtract_background else self.data
        half = int(size) // 2
        ny, nx = self.shape
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        out = np.zeros((int(size), int(size)), dtype=float)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(nx, x0 + int(size)), min(ny, y0 + int(size))
        if sx1 <= sx0 or sy1 <= sy0:
            return out
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = source[sy0:sy1, sx0:sx1]
        return out

    def copy_with(self, data: np.ndarray, **overrides) -> "AstroImage":
        """A shallow copy carrying new pixels but the same metadata."""
        fields = {
            "header": dict(self.header), "wcs": self.wcs, "mask": self.mask,
            "uncertainty": self.uncertainty, "background": self.background,
            "background_rms": self.background_rms, "name": self.name,
            "band": self.band, "mjd": self.mjd,
            "exposure_time": self.exposure_time, "meta": dict(self.meta),
        }
        fields.update(overrides)
        return AstroImage(data=data, **fields)

    def to_world(self, x, y):
        """Pixel -> ``(ra, dec)``; returns ``(None, None)`` without a WCS."""
        if self.wcs is None:
            return None, None
        return self.wcs.pixel_to_world(x, y)

    def write(self, path: str, overwrite: bool = True) -> str:
        """Save to FITS, merging the WCS back into the header."""
        header = dict(self.header)
        if self.wcs is not None:
            header.update(self.wcs.to_header())
        header.setdefault("OBJECT", self.name)
        if self.mjd is not None:
            header.setdefault("MJD-OBS", float(self.mjd))
        if self.band:
            header.setdefault("FILTER", self.band)
        return write_fits(path, self.data, header, overwrite=overwrite)

    def describe(self) -> str:
        """One-paragraph human summary, used by the CLI ``inspect`` command."""
        stats = self.stats()
        lines = [
            f"AstroImage '{self.name}'  {self.shape[1]}x{self.shape[0]} px",
            f"  band={self.band}  mjd={self.mjd}  exptime={self.exposure_time}",
            f"  median={stats['median']:.4g}  rms={stats['std']:.4g}  "
            f"min={stats['min']:.4g}  max={stats['max']:.4g}",
        ]
        if self.wcs is not None:
            ra, dec = self.wcs.pixel_to_world(self.shape[1] / 2, self.shape[0] / 2)
            lines.append(f"  centre=({float(ra):.5f}, {float(dec):.5f}) deg  "
                         f"scale={self.pixel_scale:.3f} arcsec/px")
        else:
            lines.append("  no WCS (positions reported in pixels)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<AstroImage {self.name} {self.shape} band={self.band}>"


class ImageSeries:
    """Multiple epochs of the *same* sky region, ordered in time.

    This is the input to difference imaging and light-curve extraction.
    """

    def __init__(self, images: Sequence[AstroImage], name: str = "series"):
        if not images:
            raise DataError("an ImageSeries needs at least one image")
        self.images: List[AstroImage] = sorted(
            images, key=lambda im: (im.mjd if im.mjd is not None else 0.0)
        )
        self.name = name

    @classmethod
    def from_paths(cls, paths: Sequence[str], name: str = "series",
                   **kwargs) -> "ImageSeries":
        """Load a list of files, assigning sequential epochs when time is absent."""
        images = [AstroImage.load(p, **kwargs) for p in paths]
        for i, image in enumerate(images):
            if image.mjd is None:
                image.mjd = float(i)
                image.meta["synthetic_epoch"] = True
        return cls(images, name=name)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, index) -> AstroImage:
        return self.images[index]

    @property
    def reference(self) -> AstroImage:
        """The template epoch: the deepest exposure, else the first."""
        with_exp = [im for im in self.images if im.exposure_time]
        if with_exp:
            return max(with_exp, key=lambda im: im.exposure_time or 0.0)
        return self.images[0]

    @property
    def times(self) -> np.ndarray:
        return np.array([im.mjd if im.mjd is not None else i
                         for i, im in enumerate(self.images)], dtype=float)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.images[0].shape

    def bands(self) -> List[str]:
        return sorted({im.band for im in self.images})

    def check_alignment(self) -> List[str]:
        """Report shape/band inconsistencies before differencing."""
        problems: List[str] = []
        shape = self.shape
        for image in self.images[1:]:
            if image.shape != shape:
                problems.append(f"{image.name}: shape {image.shape} != reference {shape}")
        if len(self.bands()) > 1:
            problems.append(f"mixed bands in series: {', '.join(self.bands())}")
        return problems

    def stack(self, method: str = "median") -> AstroImage:
        """Combine epochs into a deeper template image."""
        cube = np.stack([im.data for im in self.images])
        if method == "mean":
            combined = np.nanmean(cube, axis=0)
        elif method == "sum":
            combined = np.nansum(cube, axis=0)
        else:
            combined = np.nanmedian(cube, axis=0)
        template = self.reference.copy_with(combined, name=f"{self.name}_{method}_stack")
        template.meta["stacked_from"] = [im.name for im in self.images]
        return template

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ImageSeries {self.name} n={len(self)} bands={self.bands()}>"


# -- header helpers --------------------------------------------------------
def _first_key(header: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _header_time(header: Dict[str, Any]) -> Optional[float]:
    """Extract an MJD from whichever time keyword the instrument wrote."""
    for key in ("MJD-OBS", "MJD", "EXPSTART"):
        value = _float_or_none(header.get(key))
        if value is not None:
            return value
    jd = _float_or_none(header.get("JD"))
    if jd is not None:
        return jd - 2_400_000.5
    date = header.get("DATE-OBS")
    if isinstance(date, str) and len(date) >= 10:
        try:
            import datetime as _dt
            stamp = _dt.datetime.fromisoformat(date.replace("Z", "+00:00").replace("T", "T"))
            epoch = _dt.datetime(1858, 11, 17, tzinfo=stamp.tzinfo)
            return (stamp - epoch).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return None
    return None
