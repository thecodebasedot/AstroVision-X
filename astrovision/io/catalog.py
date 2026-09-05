"""Catalog serialisation: CSV, JSON, and (with Astropy) FITS/VOTable tables."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.backend import try_import
from ..core.exceptions import DataError
from ..core.logging import get_logger
from ..core.types import (
    BoundingBox,
    Morphology,
    MorphologyMetrics,
    ObjectClass,
    Photometry,
    Source,
    SourceCatalog,
)

log = get_logger("io.catalog")

#: Flat column order used for CSV / FITS table export.
COLUMNS: List[str] = [
    "id", "x", "y", "ra", "dec", "class", "class_confidence",
    "flux", "flux_err", "mag", "mag_err", "snr", "peak", "background",
    "kron_radius", "petrosian_radius", "surface_brightness",
    "semi_major", "semi_minor", "ellipticity", "position_angle", "fwhm",
    "area_pixels", "concentration", "asymmetry", "smoothness",
    "gini", "m20", "sersic_index", "effective_radius",
    "spiral_strength", "bar_strength", "arm_count", "morphology",
    "anomaly_score", "lens_score", "variability_score", "flags",
]


def source_to_row(source: Source) -> Dict[str, Any]:
    """Flatten one :class:`Source` into the export schema."""
    p, m = source.photometry, source.morphology
    return {
        "id": source.id, "x": source.x, "y": source.y,
        "ra": source.ra, "dec": source.dec,
        "class": source.object_class.value,
        "class_confidence": source.class_confidence,
        "flux": p.flux, "flux_err": p.flux_err,
        "mag": p.magnitude, "mag_err": p.magnitude_err,
        "snr": p.snr, "peak": p.peak, "background": p.background,
        "kron_radius": p.kron_radius, "petrosian_radius": p.petrosian_radius,
        "surface_brightness": p.surface_brightness,
        "semi_major": m.semi_major, "semi_minor": m.semi_minor,
        "ellipticity": m.ellipticity, "position_angle": m.position_angle,
        "fwhm": m.fwhm, "area_pixels": m.area_pixels,
        "concentration": m.concentration, "asymmetry": m.asymmetry,
        "smoothness": m.smoothness, "gini": m.gini, "m20": m.m20,
        "sersic_index": m.sersic_index, "effective_radius": m.effective_radius,
        "spiral_strength": m.spiral_strength, "bar_strength": m.bar_strength,
        "arm_count": m.arm_count, "morphology": m.label.value,
        "anomaly_score": source.anomaly_score, "lens_score": source.lens_score,
        "variability_score": source.variability_score,
        "flags": "|".join(source.flags),
    }


def row_to_source(row: Dict[str, Any]) -> Source:
    """Rebuild a :class:`Source` from an exported row (inverse of above)."""
    def num(key: str, default: float = float("nan")) -> float:
        value = row.get(key, default)
        if value in (None, "", "None"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    x, y = num("x", 0.0), num("y", 0.0)
    half = max(num("semi_major", 3.0), 1.0)
    bbox = BoundingBox(x - half, y - half, x + half, y + half)
    photometry = Photometry(
        flux=num("flux"), flux_err=num("flux_err"), magnitude=num("mag"),
        magnitude_err=num("mag_err"), peak=num("peak"), background=num("background"),
        snr=num("snr"), kron_radius=num("kron_radius"),
        petrosian_radius=num("petrosian_radius"),
        surface_brightness=num("surface_brightness"),
    )
    morphology = MorphologyMetrics(
        semi_major=num("semi_major"), semi_minor=num("semi_minor"),
        position_angle=num("position_angle"), ellipticity=num("ellipticity"),
        fwhm=num("fwhm"), area_pixels=int(num("area_pixels", 0) or 0),
        concentration=num("concentration"), asymmetry=num("asymmetry"),
        smoothness=num("smoothness"), gini=num("gini"), m20=num("m20"),
        sersic_index=num("sersic_index"), effective_radius=num("effective_radius"),
        spiral_strength=num("spiral_strength"), bar_strength=num("bar_strength"),
        arm_count=int(num("arm_count", 0) or 0),
        label=_enum(Morphology, row.get("morphology"), Morphology.UNKNOWN),
    )
    ra, dec = row.get("ra"), row.get("dec")
    flags = str(row.get("flags") or "")
    return Source(
        id=int(num("id", 0)), x=x, y=y, bbox=bbox,
        ra=None if ra in (None, "", "None") else float(ra),
        dec=None if dec in (None, "", "None") else float(dec),
        object_class=_enum(ObjectClass, row.get("class"), ObjectClass.UNKNOWN),
        class_confidence=num("class_confidence", 0.0),
        photometry=photometry, morphology=morphology,
        anomaly_score=num("anomaly_score", 0.0),
        lens_score=num("lens_score", 0.0),
        variability_score=num("variability_score", 0.0),
        flags=[f for f in flags.split("|") if f],
    )


def _enum(enum_cls, value: Any, default):
    try:
        return enum_cls(str(value))
    except (ValueError, TypeError):
        return default


def catalog_to_rows(catalog: SourceCatalog) -> List[Dict[str, Any]]:
    return [source_to_row(s) for s in catalog]


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def write_csv(catalog: SourceCatalog, path: str,
              columns: Optional[Sequence[str]] = None) -> str:
    """Write the catalog as CSV -- the universally readable export."""
    columns = list(columns or COLUMNS)
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in catalog_to_rows(catalog):
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    log.info("wrote %d sources to %s", len(catalog), path)
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        # Ten significant digits, not six: at six a right ascension of
        # 149.999 deg is only good to 3.6 arcseconds, and a catalog written
        # to CSV and read back could no longer be matched to itself.
        return f"{value:.10g}"
    return value


def write_json(catalog: SourceCatalog, path: str,
               include_embeddings: bool = False) -> str:
    """Write the catalog as JSON, preserving the nested structure."""
    _ensure_dir(path)
    payload = catalog.to_dict(include_embedding=include_embeddings)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    log.info("wrote %d sources to %s", len(catalog), path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (ObjectClass, Morphology)):
        return value.value
    return str(value)


def write_fits_table(catalog: SourceCatalog, path: str) -> str:
    """Write a binary FITS table (requires Astropy)."""
    astropy_io = try_import("astropy.io.fits")
    if astropy_io is None:
        raise DataError("astropy is required to write FITS tables; use write_csv instead")
    rows = catalog_to_rows(catalog)
    if not rows:
        raise DataError("cannot write an empty catalog to a FITS table")
    cols = []
    for name in COLUMNS:
        values = [row.get(name) for row in rows]
        if name in ("class", "morphology", "flags"):
            data = np.array([str(v or "") for v in values])
            cols.append(astropy_io.Column(name=name, format=f"{max(1, data.dtype.itemsize // 4)}A",
                                          array=data))
        elif name in ("id", "area_pixels", "arm_count"):
            cols.append(astropy_io.Column(
                name=name, format="J",
                array=np.array([int(v or 0) for v in values], dtype=np.int32)))
        else:
            cols.append(astropy_io.Column(
                name=name, format="E",
                array=np.array([np.nan if v is None else float(v) for v in values],
                               dtype=np.float32)))
    _ensure_dir(path)
    hdu = astropy_io.BinTableHDU.from_columns(cols)
    hdu.header["NSOURCE"] = len(rows)
    hdu.writeto(path, overwrite=True)
    log.info("wrote %d sources to FITS table %s", len(rows), path)
    return path


def write_catalog(catalog: SourceCatalog, path: str, fmt: Optional[str] = None) -> str:
    """Dispatch to the writer implied by ``fmt`` or the file extension."""
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "csv").lower()
    if fmt in ("csv", "txt", "tsv"):
        return write_csv(catalog, path)
    if fmt == "json":
        return write_json(catalog, path)
    if fmt in ("fits", "fit"):
        return write_fits_table(catalog, path)
    raise DataError(f"unsupported catalog format '{fmt}'")


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def read_csv(path: str) -> SourceCatalog:
    """Read a catalog previously written by :func:`write_csv`."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return SourceCatalog([row_to_source(row) for row in rows],
                         meta={"source_path": os.path.abspath(path)})


def read_json(path: str) -> SourceCatalog:
    """Read a catalog previously written by :func:`write_json`."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sources = payload.get("sources", payload if isinstance(payload, list) else [])
    catalog = SourceCatalog(meta=payload.get("meta", {}) if isinstance(payload, dict) else {})
    for entry in sources:
        flat = {
            **{k: v for k, v in entry.items() if not isinstance(v, dict)},
            **{k: v for k, v in entry.get("photometry", {}).items()},
            **{k: v for k, v in entry.get("morphology", {}).items()},
        }
        flat["class"] = entry.get("object_class", "unknown")
        flat["mag"] = entry.get("photometry", {}).get("magnitude")
        flat["mag_err"] = entry.get("photometry", {}).get("magnitude_err")
        flat["morphology"] = entry.get("morphology", {}).get("label", "unknown")
        flat["flags"] = "|".join(entry.get("flags", []))
        catalog.append(row_to_source(flat))
    return catalog


def read_catalog(path: str) -> SourceCatalog:
    """Read a catalog from CSV or JSON, inferred from the extension."""
    if str(path).lower().endswith(".json"):
        return read_json(path)
    return read_csv(path)


# --------------------------------------------------------------------------
# cross-matching
# --------------------------------------------------------------------------
def crossmatch(catalog_a: SourceCatalog, catalog_b: SourceCatalog,
               radius: float = 2.0, use_world: bool = False) -> List[Dict[str, Any]]:
    """Nearest-neighbour match between two catalogs.

    ``radius`` is in pixels, or arcseconds when ``use_world`` is set and both
    catalogs carry sky coordinates.  Returns one record per matched pair.
    """
    if len(catalog_a) == 0 or len(catalog_b) == 0:
        return []
    if use_world:
        a = np.array([[s.ra, s.dec] for s in catalog_a if s.ra is not None], dtype=float)
        b = np.array([[s.ra, s.dec] for s in catalog_b if s.ra is not None], dtype=float)
        if a.size == 0 or b.size == 0:
            raise DataError("world cross-match requires ra/dec on both catalogs")
        ids_a = [s.id for s in catalog_a if s.ra is not None]
        ids_b = [s.id for s in catalog_b if s.ra is not None]
        from .wcs import angular_separation
        separation = np.array([
            angular_separation(a[i, 0], a[i, 1], b[:, 0], b[:, 1]) * 3600.0
            for i in range(len(a))
        ])
    else:
        a, b = catalog_a.positions(), catalog_b.positions()
        ids_a = [s.id for s in catalog_a]
        ids_b = [s.id for s in catalog_b]
        separation = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])

    matches: List[Dict[str, Any]] = []
    used_b: set = set()
    for i in np.argsort(separation.min(axis=1)):
        order = np.argsort(separation[i])
        for j in order:
            if j in used_b:
                continue
            if separation[i, j] <= radius:
                used_b.add(int(j))
                matches.append({"id_a": ids_a[i], "id_b": ids_b[int(j)],
                                "separation": float(separation[i, j])})
            break
    return matches


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
