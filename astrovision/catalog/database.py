"""A catalog that remembers.

A CSV per field answers "what was in this image". A survey asks different
questions: what is at *this position*, in every image ever taken of it; how
has *this object* behaved over the last year; which fields overlap here. Those
need one store across fields and epochs, indexed by sky position, with a
notion of an *object* that persists between detections.

This is that store, on SQLite -- part of the standard library, so it keeps
the NumPy-only promise -- with three tables:

* **fields**: one row per ingested image, carrying the provenance manifest so
  every detection can be traced to the code and configuration that made it;
* **detections**: one row per source per field, the flat export schema plus
  the field's epoch and band, and a nested HEALPix index at nside 8192
  (26-arcsecond pixels);
* **objects**: one row per distinct thing on the sky. A detection is
  associated with an existing object when one lies within the match radius
  (1.5 arcseconds by default); otherwise it founds a new object. An object
  therefore has a history -- its detections across every field and epoch --
  which is what a light curve is.

The nested index is what makes the queries fast. A cone search asks the
index for the coarse pixels the cone touches and turns each into one
``BETWEEN`` range on the fine index, because in the nested scheme every fine
pixel inside a coarse one is a contiguous run of integers. Association on
ingest is vectorised: candidate pairs come from a sorted-pixel join in NumPy,
and only the survivors are checked with an exact separation.

What this is not: it is not a replacement for a survey's own database, and
the object identity it maintains is positional only. Two sources 1 arcsecond
apart in different epochs are one object to it, whether or not they are; an
object whose measured position wanders by more than the match radius
(a fast mover, or a bad astrometric solution) becomes several. The moving
object stage links tracklets on its own terms and should be used for that.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import SourceCatalog
from ..io.catalog import row_to_source, source_to_row
from .healpix import SkyIndex, ang2pix, angular_separation, order, pixel_resolution_deg

log = get_logger("catalog.database")

SCHEMA_VERSION = 1

#: The fine index stored on every row: nside 8192 is a 26" pixel, far
#: larger than any association radius and fine enough that a cone of a few
#: arcminutes touches a handful of ranges.
FINE_NSIDE = 8192
#: The coarse resolution cone queries are planned at.
COARSE_NSIDE = 128

_COLUMNS = [
    ("source_id", "INTEGER"), ("x", "REAL"), ("y", "REAL"), ("ra", "REAL"), ("dec", "REAL"),
    ("class", "TEXT"), ("class_confidence", "REAL"),
    ("flux", "REAL"), ("flux_err", "REAL"), ("mag", "REAL"), ("mag_err", "REAL"),
    ("snr", "REAL"), ("peak", "REAL"), ("background", "REAL"),
    ("kron_radius", "REAL"), ("petrosian_radius", "REAL"), ("surface_brightness", "REAL"),
    ("semi_major", "REAL"), ("semi_minor", "REAL"), ("ellipticity", "REAL"),
    ("position_angle", "REAL"), ("fwhm", "REAL"), ("area_pixels", "INTEGER"),
    ("concentration", "REAL"), ("asymmetry", "REAL"), ("smoothness", "REAL"),
    ("gini", "REAL"), ("m20", "REAL"), ("sersic_index", "REAL"), ("effective_radius", "REAL"),
    ("spiral_strength", "REAL"), ("bar_strength", "REAL"), ("arm_count", "INTEGER"),
    ("morphology", "TEXT"), ("anomaly_score", "REAL"), ("lens_score", "REAL"),
    ("variability_score", "REAL"), ("flags", "TEXT"),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS fields (
    id INTEGER PRIMARY KEY,
    name TEXT, path TEXT, band TEXT, mjd REAL, exposure_time REAL,
    ra_centre REAL, dec_centre REAL, width INTEGER, height INTEGER,
    pixel_scale REAL, n_sources INTEGER, n_with_sky INTEGER,
    reproducibility_key TEXT, config_hash TEXT, manifest TEXT, ingested TEXT
);
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    ra REAL, dec REAL, healpix INTEGER,
    n_detections INTEGER, first_mjd REAL, last_mjd REAL, bands TEXT,
    founded_by INTEGER
);
CREATE INDEX IF NOT EXISTS objects_healpix ON objects (healpix);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    field_id INTEGER NOT NULL REFERENCES fields (id),
    object_id INTEGER REFERENCES objects (id),
    healpix INTEGER, band TEXT, mjd REAL,
    %s,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS detections_healpix ON detections (healpix);
CREATE INDEX IF NOT EXISTS detections_object ON detections (object_id);
CREATE INDEX IF NOT EXISTS detections_field ON detections (field_id);
""" % ",\n    ".join(f"{name} {kind}" for name, kind in _COLUMNS)


def _clean(value: Any) -> Any:
    """SQLite has no NaN: store it as NULL, and enums as their value."""
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, (int, float, str)):
        return value.value
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return value


@dataclass
class IngestReport:
    """What one ingest did."""

    field_id: int
    n_detections: int = 0
    n_with_sky: int = 0
    n_matched: int = 0
    n_new_objects: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class CatalogDB:
    """The store. Open with a path, or ``":memory:"`` for a scratch one.

    >>> db = CatalogDB(":memory:")
    >>> db.counts()["fields"]
    0
    """

    def __init__(self, path: str = ":memory:", match_radius_arcsec: float = 1.5):
        self.path = path
        self.match_radius_deg = float(match_radius_arcsec) / 3600.0
        if path != ":memory:":
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.executescript(_SCHEMA)
        self._coarse = SkyIndex(COARSE_NSIDE)
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        self._set_meta("fine_nside", str(FINE_NSIDE))
        self._set_meta("match_radius_arcsec", str(match_radius_arcsec))

    # -- housekeeping ------------------------------------------------------
    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                                (key, value))

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "CatalogDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def counts(self) -> Dict[str, int]:
        return {table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("fields", "detections", "objects")}

    def fields(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT id, name, path, band, mjd, n_sources, n_with_sky, reproducibility_key, "
            "ingested FROM fields ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    # -- ingest ------------------------------------------------------------
    def ingest(self, catalog: SourceCatalog, name: str, band: Optional[str] = None,
               mjd: Optional[float] = None, path: Optional[str] = None,
               image: Any = None, provenance: Optional[Dict[str, Any]] = None,
               associate: bool = True) -> IngestReport:
        """Store one field's catalog and link its detections to objects.

        ``image`` (an :class:`AstroImage`) supplies the band, epoch, size and
        centre when given; ``provenance`` is the pipeline's provenance dict,
        from which the manifest and reproducibility key are kept.
        """
        started = time.time()
        if image is not None:
            band = band or getattr(image, "band", None)
            mjd = mjd if mjd is not None else getattr(image, "mjd", None)
            path = path or (getattr(image, "meta", {}) or {}).get("source_path")
        provenance = provenance or {}
        manifest = provenance.get("manifest")
        centre = self._centre(image)
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO fields (name, path, band, mjd, exposure_time, ra_centre, dec_centre, "
            "width, height, pixel_scale, n_sources, n_with_sky, reproducibility_key, "
            "config_hash, manifest, ingested) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, path, band, _clean(mjd),
             _clean(getattr(image, "exposure_time", None)),
             centre[0], centre[1],
             None if image is None else int(image.shape[1]),
             None if image is None else int(image.shape[0]),
             _clean(self._pixel_scale(image)),
             len(catalog), 0, provenance.get("reproducibility_key"),
             (manifest or {}).get("config_hash"),
             None if manifest is None else json.dumps(manifest, default=str),
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        field_id = int(cursor.lastrowid)
        report = IngestReport(field_id=field_id, n_detections=len(catalog))
        if len(catalog) == 0:
            self.connection.commit()
            report.seconds = time.time() - started
            return report

        rows = [source_to_row(source) for source in catalog]
        ra = np.array([np.nan if r["ra"] is None else float(r["ra"]) for r in rows])
        dec = np.array([np.nan if r["dec"] is None else float(r["dec"]) for r in rows])
        has_sky = np.isfinite(ra) & np.isfinite(dec)
        report.n_with_sky = int(has_sky.sum())
        healpix = np.full(len(rows), -1, dtype=np.int64)
        if has_sky.any():
            healpix[has_sky] = ang2pix(FINE_NSIDE, ra[has_sky], dec[has_sky])
        else:
            report.notes.append("no sky coordinates: detections stored without an object link")

        object_ids: List[Optional[int]] = [None] * len(rows)
        if associate and has_sky.any():
            matched, founded = self._associate(cursor, ra, dec, healpix, has_sky,
                                               band, mjd, field_id)
            object_ids = matched
            report.n_matched = int(sum(1 for m in matched if m is not None)) - founded
            report.n_new_objects = founded

        columns = [name for name, _ in _COLUMNS]
        placeholders = ",".join("?" for _ in range(len(columns) + 6))
        extras = []
        for source in catalog:
            extra = {}
            if source.bands:
                extra["bands"] = {k: v.to_dict() for k, v in source.bands.items()}
            for key in ("r50", "r90", "aperture_correction", "tile"):
                if key in source.meta:
                    extra[key] = source.meta[key]
            extras.append(json.dumps(extra, default=str) if extra else None)
        cursor.executemany(
            f"INSERT INTO detections (field_id, object_id, healpix, band, mjd, "
            f"{','.join(columns)}, extra) VALUES ({placeholders})",
            [(field_id, object_ids[i], int(healpix[i]) if healpix[i] >= 0 else None, band,
              _clean(mjd), *[_clean(rows[i].get("id") if c == "source_id" else rows[i].get(c))
                             for c in columns], extras[i])
             for i in range(len(rows))])
        cursor.execute("UPDATE fields SET n_with_sky = ? WHERE id = ?",
                       (report.n_with_sky, field_id))
        self.connection.commit()
        report.seconds = time.time() - started
        log.info("ingested %s: %d detections, %d matched to known objects, %d new, %.2fs",
                 name, report.n_detections, report.n_matched, report.n_new_objects,
                 report.seconds)
        return report

    def _associate(self, cursor, ra, dec, healpix, has_sky, band, mjd, field_id
                   ) -> Tuple[List[Optional[int]], int]:
        """Link each detection with sky coordinates to an object.

        Candidates are objects in the detection's fine pixel or in the pixels
        of sixteen points around it at twice the match radius; a pixel is 26
        arcseconds and the radius a few, so that ring covers every pixel a
        match could sit in. The join is a sorted-pixel merge in NumPy and the
        exact separation is computed only for candidate pairs.
        """
        idx = np.flatnonzero(has_sky)
        n = idx.size
        d_ra, d_dec = ra[idx], dec[idx]
        radius = self.match_radius_deg
        angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        cos_dec = np.maximum(np.cos(np.radians(d_dec)), 1e-6)
        ring_ra = d_ra[:, None] + 2 * radius * np.cos(angles)[None, :] / cos_dec[:, None]
        ring_dec = np.clip(d_dec[:, None] + 2 * radius * np.sin(angles)[None, :], -90.0, 90.0)
        pixels = np.concatenate([healpix[idx][:, None],
                                 ang2pix(FINE_NSIDE, ring_ra.ravel(), ring_dec.ravel())
                                 .reshape(n, -1)], axis=1)
        wanted = np.unique(pixels)

        obj_id, obj_ra, obj_dec, obj_pix = self._objects_in_pixels(wanted)
        matched: List[Optional[int]] = [None] * len(ra)
        if obj_id.size:
            sort = np.argsort(obj_pix, kind="stable")
            obj_id, obj_ra, obj_dec, obj_pix = (obj_id[sort], obj_ra[sort], obj_dec[sort],
                                                obj_pix[sort])
            # Every (detection, candidate pixel) pair -> the run of objects
            # with that pixel, expanded into explicit pairs.
            det_of_pair = np.repeat(np.arange(n), pixels.shape[1])
            flat = pixels.ravel()
            lo = np.searchsorted(obj_pix, flat, side="left")
            hi = np.searchsorted(obj_pix, flat, side="right")
            lengths = hi - lo
            keep = lengths > 0
            if keep.any():
                det_rep = np.repeat(det_of_pair[keep], lengths[keep])
                starts = np.repeat(lo[keep], lengths[keep])
                offsets = np.arange(lengths[keep].sum()) - np.repeat(
                    np.cumsum(lengths[keep]) - lengths[keep], lengths[keep])
                obj_rep = starts + offsets
                pairs = np.unique(np.stack([det_rep, obj_rep], axis=1), axis=0)
                sep = angular_separation(d_ra[pairs[:, 0]], d_dec[pairs[:, 0]],
                                         obj_ra[pairs[:, 1]], obj_dec[pairs[:, 1]])
                close = sep <= radius
                if close.any():
                    pairs, sep = pairs[close], sep[close]
                    by_sep = np.argsort(sep, kind="stable")
                    pairs, sep = pairs[by_sep], sep[by_sep]
                    _, first = np.unique(pairs[:, 0], return_index=True)
                    for det_local, obj_local in pairs[first]:
                        matched[idx[det_local]] = int(obj_id[obj_local])

        known = [m for m in matched if m is not None]

        # Unmatched detections found new objects.
        founded = 0
        new_rows = []
        for local, global_index in enumerate(idx):
            if matched[global_index] is None:
                new_rows.append((float(d_ra[local]), float(d_dec[local]),
                                 int(healpix[global_index]), 1, _clean(mjd), _clean(mjd),
                                 band or "", field_id))
        if new_rows:
            first_id = int(cursor.execute("SELECT COALESCE(MAX(id), 0) FROM objects")
                           .fetchone()[0]) + 1
            cursor.executemany(
                "INSERT INTO objects (ra, dec, healpix, n_detections, first_mjd, last_mjd, "
                "bands, founded_by) VALUES (?,?,?,?,?,?,?,?)", new_rows)
            new_id = first_id
            for global_index in idx:
                if matched[global_index] is None:
                    matched[global_index] = new_id
                    new_id += 1
                    founded += 1

        # Known objects gain a detection, an epoch and possibly a band.
        if known:
            counts: Dict[int, int] = {}
            for value in known:
                counts[value] = counts.get(value, 0) + 1
            cursor.executemany(
                "UPDATE objects SET n_detections = n_detections + ?, "
                "first_mjd = CASE WHEN first_mjd IS NULL OR ? < first_mjd THEN ? ELSE first_mjd END, "
                "last_mjd = CASE WHEN last_mjd IS NULL OR ? > last_mjd THEN ? ELSE last_mjd END, "
                "bands = CASE WHEN instr(',' || bands || ',', ',' || ? || ',') > 0 THEN bands "
                "ELSE CASE WHEN bands = '' THEN ? ELSE bands || ',' || ? END END WHERE id = ?",
                [(k, _clean(mjd), _clean(mjd), _clean(mjd), _clean(mjd),
                  band or "", band or "", band or "", oid) for oid, k in counts.items()])
        return matched, founded

    def _objects_in_pixels(self, pixels: np.ndarray):
        """Objects whose fine pixel is in ``pixels`` (any number of them)."""
        if pixels.size == 0:
            return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
        cursor = self.connection.cursor()
        cursor.execute("CREATE TEMP TABLE IF NOT EXISTS wanted_pixels (pixel INTEGER PRIMARY KEY)")
        cursor.execute("DELETE FROM wanted_pixels")
        cursor.executemany("INSERT OR IGNORE INTO wanted_pixels (pixel) VALUES (?)",
                           [(int(p),) for p in pixels])
        rows = cursor.execute(
            "SELECT o.id, o.ra, o.dec, o.healpix FROM objects o "
            "JOIN wanted_pixels w ON o.healpix = w.pixel").fetchall()
        if not rows:
            return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
        arr = np.array([(r[0], r[1], r[2], r[3]) for r in rows], dtype=float)
        return (arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3].astype(np.int64))

    # -- queries -----------------------------------------------------------
    @staticmethod
    def _planning_nside(radius_deg: float) -> int:
        """The coarsest resolution whose pixels are smaller than the cone.

        A cone should be covered by a few pixels of comparable size, not by
        one pixel a hundred times larger: at 0.46 degrees the coarse pixel
        holds tens of thousands of rows a 5-arcsecond query then has to
        discard. Refined down to at most the stored resolution.
        """
        nside = COARSE_NSIDE
        while nside < FINE_NSIDE and pixel_resolution_deg(nside) > max(radius_deg, 1e-9) / 2.0:
            nside *= 2
        return nside

    def _ranges(self, ra_deg: float, dec_deg: float, radius_deg: float) -> List[Tuple[int, int]]:
        """Fine-index ranges covering a cone, one per planning pixel it touches."""
        nside = self._planning_nside(radius_deg)
        shift = 2 * (order(FINE_NSIDE) - order(nside))
        touched = self._coarse.cone(ra_deg, dec_deg, radius_deg, nside_out=nside)
        ranges = [(int(p) << shift, ((int(p) + 1) << shift) - 1) for p in touched]
        # Merge adjacent runs so the query stays short in a dense cone.
        merged: List[Tuple[int, int]] = []
        for lo, hi in sorted(ranges):
            if merged and lo == merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        return merged

    def cone_search(self, ra_deg: float, dec_deg: float, radius_arcsec: float,
                    table: str = "detections", limit: Optional[int] = None
                    ) -> List[Dict[str, Any]]:
        """Rows of ``table`` within ``radius_arcsec`` of a position, nearest first.

        Each row carries ``separation_arcsec``. ``table`` is ``"detections"``
        (every measurement, across fields and epochs) or ``"objects"``.
        """
        if table not in ("detections", "objects"):
            raise ValueError("table must be 'detections' or 'objects'")
        radius_deg = float(radius_arcsec) / 3600.0
        ranges = self._ranges(ra_deg, dec_deg, radius_deg)
        if not ranges:
            return []
        clause = " OR ".join("healpix BETWEEN ? AND ?" for _ in ranges)
        params: List[Any] = [v for pair in ranges for v in pair]
        if table == "detections":
            sql = ("SELECT d.*, f.name AS field_name FROM detections d "
                   "JOIN fields f ON f.id = d.field_id WHERE " + clause)
        else:
            sql = "SELECT * FROM objects WHERE " + clause
        cursor = self.connection.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            return []
        # Exact separation on the raw rows first; only survivors become dicts.
        names = [d[0] for d in cursor.description]
        i_ra, i_dec = names.index("ra"), names.index("dec")
        ra = np.array([r[i_ra] for r in rows], dtype=float)
        dec = np.array([r[i_dec] for r in rows], dtype=float)
        sep = angular_separation(ra_deg, dec_deg, ra, dec) * 3600.0
        keep = np.flatnonzero(sep <= float(radius_arcsec))
        keep = keep[np.argsort(sep[keep], kind="stable")]
        if limit is not None:
            keep = keep[:int(limit)]
        out = []
        for k in keep:
            row = dict(zip(names, rows[int(k)]))
            row["separation_arcsec"] = float(sep[k])
            out.append(row)
        return out

    def history(self, object_id: int) -> List[Dict[str, Any]]:
        """Every detection of one object, in time order, with its field."""
        rows = self.connection.execute(
            "SELECT d.*, f.name AS field_name, f.path AS field_path FROM detections d "
            "JOIN fields f ON f.id = d.field_id WHERE d.object_id = ? "
            "ORDER BY d.mjd, d.id", (int(object_id),)).fetchall()
        return [dict(r) for r in rows]

    def light_curve(self, object_id: int, band: Optional[str] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mjd, flux, flux_err)`` arrays for one object, one band or all."""
        rows = [r for r in self.history(object_id)
                if band is None or r.get("band") == band]
        rows = [r for r in rows if r.get("mjd") is not None and r.get("flux") is not None]
        if not rows:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        mjd = np.array([r["mjd"] for r in rows], dtype=float)
        flux = np.array([r["flux"] for r in rows], dtype=float)
        err = np.array([np.nan if r["flux_err"] is None else r["flux_err"] for r in rows],
                       dtype=float)
        return mjd, flux, err

    def object(self, object_id: int) -> Optional[Dict[str, Any]]:
        row = self.connection.execute("SELECT * FROM objects WHERE id = ?",
                                      (int(object_id),)).fetchone()
        return None if row is None else dict(row)

    def objects_with_history(self, min_detections: int = 2,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Objects seen at least ``min_detections`` times, most seen first."""
        sql = ("SELECT * FROM objects WHERE n_detections >= ? "
               "ORDER BY n_detections DESC, id")
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.connection.execute(sql, (int(min_detections),))]

    def field_catalog(self, field_id: int) -> SourceCatalog:
        """The catalog of one field, rebuilt as :class:`SourceCatalog`."""
        rows = self.connection.execute("SELECT * FROM detections WHERE field_id = ? ORDER BY id",
                                       (int(field_id),)).fetchall()
        catalog = SourceCatalog()
        for row in rows:
            data = dict(row)
            data["id"] = data.pop("source_id")
            source = row_to_source(data)
            source.meta["object_id"] = data.get("object_id")
            source.meta["detection_id"] = data.get("id")
            catalog.append(source)
        return catalog

    def detections_of_field(self, field_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.connection.execute(
            "SELECT * FROM detections WHERE field_id = ? ORDER BY id", (int(field_id),))]

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _centre(image: Any) -> Tuple[Optional[float], Optional[float]]:
        if image is None or getattr(image, "wcs", None) is None:
            return None, None
        try:
            ra, dec = image.wcs.pixel_to_world(image.shape[1] / 2.0, image.shape[0] / 2.0)
            return float(np.asarray(ra).ravel()[0]), float(np.asarray(dec).ravel()[0])
        except Exception:                                    # pragma: no cover
            return None, None

    @staticmethod
    def _pixel_scale(image: Any) -> Optional[float]:
        if image is None or getattr(image, "wcs", None) is None:
            return None
        try:
            return float(image.pixel_scale)
        except Exception:                                    # pragma: no cover
            return None


def ingest_analysis(db: CatalogDB, analysis: Any, image: Any, name: Optional[str] = None
                    ) -> IngestReport:
    """Store a :class:`FieldAnalysis` with its provenance."""
    return db.ingest(analysis.catalog, name=name or getattr(image, "name", "field"),
                     image=image, provenance=getattr(analysis, "provenance", None))


__all__ = ["CatalogDB", "IngestReport", "ingest_analysis", "FINE_NSIDE", "COARSE_NSIDE"]
