"""Cone searches against external catalogs, and what they mean for a candidate.

Everything this package calls a *candidate* rests on one unstated claim: that
the object is not already known.  Without checking, an anomaly ranking is a
list of the field's oddest objects, which is not the same thing and is
mostly populated by catalogued variable stars, asteroids on their published
ephemerides, and galaxies someone measured in 1991.

So the check belongs in the pipeline, not in the reader's head.  A source
that matches a known object keeps its measurements and loses its claim to
novelty, and the report says which catalog it matched and how far away.

**Offline first.**  The backend is a protocol with several implementations,
and the default does nothing.  A local reference file works with no network
at all; the HTTP backends are opt-in.  This ordering is deliberate -- a
science pipeline that silently fails, or silently stalls, when a remote
service is unreachable is worse than one that never called it.

**One query per field, not per source.**  A cone that covers the whole image
is fetched once and matched locally.  A thousand-source catalog is a thousand
HTTP requests the other way round, which is slow for the caller and rude to
a service that is free to use.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import DataError
from ..core.logging import get_logger
from ..core.types import SourceCatalog
from .wcs import angular_separation

log = get_logger("io.external")

#: Object types that mean "this is a known, catalogued object of that kind".
#: Used to phrase the report, not to make decisions.
TYPE_LABELS: Dict[str, str] = {
    "*": "star", "**": "double star", "V*": "variable star",
    "QSO": "quasar", "AGN": "active galaxy", "G": "galaxy",
    "GinCl": "galaxy in a cluster", "PN": "planetary nebula",
    "SN": "supernova", "SNR": "supernova remnant", "Cl*": "star cluster",
    "Astr": "minor planet",
}


@dataclass
class ReferenceObject:
    """One row from an external catalog."""

    ra: float
    dec: float
    name: str = ""
    catalog: str = ""
    object_type: str = ""
    magnitudes: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def described_type(self) -> str:
        return TYPE_LABELS.get(self.object_type, self.object_type or "object")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ra": float(self.ra), "dec": float(self.dec), "name": self.name,
            "catalog": self.catalog, "object_type": self.object_type,
            "magnitudes": {k: float(v) for k, v in self.magnitudes.items()},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceObject":
        return cls(ra=float(data["ra"]), dec=float(data["dec"]),
                   name=str(data.get("name", "")),
                   catalog=str(data.get("catalog", "")),
                   object_type=str(data.get("object_type", "")),
                   magnitudes={k: float(v) for k, v in (data.get("magnitudes") or {}).items()},
                   meta=dict(data.get("meta") or {}))


class ConeSearch:
    """Interface every reference-catalog backend implements."""

    name = "none"

    def query(self, ra: float, dec: float,
              radius_arcsec: float) -> List[ReferenceObject]:      # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name}


class NullCone(ConeSearch):
    """The default: no external catalog, and honest about it.

    Returning nothing is not the same as finding nothing, and the
    distinction is recorded in the crossmatch report so a run without a
    reference catalog can never be mistaken for one that checked and came
    back clean.
    """

    name = "none"

    def query(self, ra: float, dec: float, radius_arcsec: float) -> List[ReferenceObject]:
        return []


class LocalCone(ConeSearch):
    """A reference catalog held in memory or read from a file.

    Takes a list of :class:`ReferenceObject`, or a path to a JSON list or a
    CSV with ``ra``/``dec`` columns.  This is the backend to use for a survey
    with its own reference catalog, for a downloaded Gaia extract, and for
    every test -- it is exactly as good as a remote service for matching, and
    it cannot fail halfway through.

    >>> service = LocalCone([ReferenceObject(150.0, 2.2, "HD 1", "TEST", "*")])
    >>> [o.name for o in service.query(150.0, 2.2, 5.0)]
    ['HD 1']
    >>> service.query(151.0, 2.2, 5.0)
    []
    """

    name = "local"

    def __init__(self, objects: Iterable[ReferenceObject] | str,
                 catalog_name: str = "local"):
        if isinstance(objects, str):
            self.objects = list(read_reference_file(objects))
            self.source = objects
        else:
            self.objects = list(objects)
            self.source = "<memory>"
        self.catalog_name = catalog_name
        if self.objects:
            self._ra = np.array([o.ra for o in self.objects], dtype=float)
            self._dec = np.array([o.dec for o in self.objects], dtype=float)
        else:
            self._ra = np.zeros(0)
            self._dec = np.zeros(0)

    def query(self, ra: float, dec: float, radius_arcsec: float) -> List[ReferenceObject]:
        if not self.objects:
            return []
        separation = angular_separation(float(ra), float(dec), self._ra, self._dec) * 3600.0
        inside = np.nonzero(separation <= float(radius_arcsec))[0]
        return [self.objects[int(i)] for i in inside]

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "source": self.source,
                "n_objects": len(self.objects), "catalog": self.catalog_name}


class CachedCone(ConeSearch):
    """Wraps another backend with an on-disk cache.

    Cone searches are re-run constantly -- every re-analysis of a field asks
    the same question -- and a cached answer is both faster and kinder to a
    public service.  Entries are keyed on the rounded cone and expire, since
    reference catalogs are updated and a stale "not known" is the one wrong
    answer this whole module exists to prevent.
    """

    name = "cached"

    def __init__(self, inner: ConeSearch, directory: str,
                 max_age_days: float = 30.0):
        self.inner = inner
        self.directory = directory
        self.max_age_seconds = float(max_age_days) * 86400.0
        os.makedirs(directory, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, ra: float, dec: float, radius: float) -> str:
        key = f"{self.inner.name}_{ra:.5f}_{dec:.5f}_{radius:.1f}".replace("-", "m")
        return os.path.join(self.directory, key + ".json")

    def query(self, ra: float, dec: float, radius_arcsec: float) -> List[ReferenceObject]:
        path = self._path(ra, dec, radius_arcsec)
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age <= self.max_age_seconds:
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    self.hits += 1
                    return [ReferenceObject.from_dict(row) for row in payload]
                except (OSError, ValueError, KeyError):
                    log.warning("discarding unreadable cache entry %s", path)
        results = self.inner.query(ra, dec, radius_arcsec)
        self.misses += 1
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([o.to_dict() for o in results], handle)
        except OSError as error:                                   # pragma: no cover
            log.warning("could not write cache entry: %s", error)
        return results

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "inner": self.inner.describe(),
                "hits": self.hits, "misses": self.misses,
                "directory": self.directory}


# --------------------------------------------------------------------------
# HTTP backends
# --------------------------------------------------------------------------
class HttpCone(ConeSearch):
    """Shared plumbing for the remote services: fetch text, never hang.

    A timeout is mandatory and failures are returned as an empty result with
    a recorded error rather than an exception, because an unreachable catalog
    must degrade the pipeline's *claims*, not stop its run.  Callers see the
    difference through :attr:`last_error`, which the crossmatch report
    propagates.
    """

    name = "http"
    url_template = ""

    def __init__(self, timeout: float = 20.0, max_rows: int = 5000):
        self.timeout = float(timeout)
        self.max_rows = int(max_rows)
        self.last_error: Optional[str] = None

    def _fetch(self, url: str) -> Optional[str]:
        import urllib.error
        import urllib.request
        self.last_error = None
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "AstroVision-X cone search"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as error:
            self.last_error = f"{type(error).__name__}: {error}"
            log.warning("%s cone search failed: %s", self.name, self.last_error)
            return None

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "timeout": self.timeout,
                "last_error": self.last_error}


class VizieRCone(HttpCone):
    """Cone search against a VizieR table, over the tab-separated ASU service.

    The default table is Gaia DR3, which is the right first check for point
    sources: it is all-sky, deep enough to contain essentially every star a
    small-telescope image resolves, and carries the proper motions that say
    whether a "new" source is a star that simply moved.
    """

    name = "vizier"
    #: ``{catalog}`` and the cone are substituted in.
    url_template = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
                    "?-source={catalog}&-c={ra:.6f}%20{dec}&-c.rs={radius:.2f}"
                    "&-out={columns}&-out.max={max_rows}")

    def __init__(self, catalog: str = "I/355/gaiadr3",
                 columns: Sequence[str] = ("RA_ICRS", "DE_ICRS", "Source", "Gmag",
                                           "BP-RP", "pmRA", "pmDE"),
                 timeout: float = 20.0, max_rows: int = 5000):
        super().__init__(timeout=timeout, max_rows=max_rows)
        self.catalog = catalog
        self.columns = list(columns)

    def build_url(self, ra: float, dec: float, radius_arcsec: float) -> str:
        # The declination's leading "+" must be percent-encoded.  A bare "+"
        # in a query string decodes to a space, so a northern field would be
        # sent with its sign silently stripped and the service would answer
        # about a different piece of sky.
        declination = f"{float(dec):+.6f}".replace("+", "%2B")
        return self.url_template.format(
            catalog=self.catalog, ra=float(ra), dec=declination,
            radius=float(radius_arcsec) / 60.0,     # the service wants arcminutes
            columns=",".join(self.columns), max_rows=self.max_rows)

    def query(self, ra: float, dec: float, radius_arcsec: float) -> List[ReferenceObject]:
        text = self._fetch(self.build_url(ra, dec, radius_arcsec))
        if text is None:
            return []
        return list(parse_vizier_tsv(text, catalog=self.catalog))

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"catalog": self.catalog, "columns": list(self.columns)})
        return data


class SimbadCone(HttpCone):
    """Cone search against SIMBAD, which is where *identifications* live.

    Gaia says a point source exists; SIMBAD says it is a known cataclysmic
    variable with forty published papers.  For deciding whether a candidate
    is a discovery, the second is the question actually being asked.
    """

    name = "simbad"
    url_template = ("https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
                    "?request=doQuery&lang=adql&format=tsv&query={query}")

    def build_url(self, ra: float, dec: float, radius_arcsec: float) -> str:
        import urllib.parse
        query = (
            "SELECT TOP {max_rows} main_id, ra, dec, otype "
            "FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), "
            "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.6f})) = 1"
        ).format(max_rows=self.max_rows, ra=float(ra), dec=float(dec),
                 radius=float(radius_arcsec) / 3600.0)
        return self.url_template.format(query=urllib.parse.quote(query))

    def query(self, ra: float, dec: float, radius_arcsec: float) -> List[ReferenceObject]:
        text = self._fetch(self.build_url(ra, dec, radius_arcsec))
        if text is None:
            return []
        return list(parse_simbad_tsv(text))


# --------------------------------------------------------------------------
# parsers -- kept separate from transport so they can be tested without a network
# --------------------------------------------------------------------------
def parse_vizier_tsv(text: str, catalog: str = "") -> List[ReferenceObject]:
    """Parse VizieR's tab-separated output.

    The format interleaves ``#`` comment blocks, a header row, a row of unit
    strings, a row of dashes, and then the data -- so rows are identified by
    content rather than by position, which survives the service adding or
    removing preamble.

    >>> text = "#comment\\nRA_ICRS\\tDE_ICRS\\tGmag\\ndeg\\tdeg\\tmag\\n---\\t---\\t---\\n150.0\\t2.2\\t15.4\\n"
    >>> objects = parse_vizier_tsv(text, "I/355/gaiadr3")
    >>> objects[0].ra, objects[0].magnitudes["G"]
    (150.0, 15.4)
    """
    header: Optional[List[str]] = None
    results: List[ReferenceObject] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [value.strip() for value in line.split("\t")]
        if header is None:
            if any(name in fields for name in ("RA_ICRS", "_RAJ2000", "RAJ2000", "ra")):
                header = fields
            continue
        if all(set(value) <= {"-"} for value in fields if value):
            continue                                    # the dashed separator row
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        ra = _first_float(row, ("RA_ICRS", "_RAJ2000", "RAJ2000", "ra"))
        dec = _first_float(row, ("DE_ICRS", "_DEJ2000", "DEJ2000", "dec"))
        if ra is None or dec is None:
            continue                                    # a units row, or blank
        magnitudes = {}
        for key, value in row.items():
            if key.endswith("mag") and len(key) > 3:
                parsed = _as_float(value)
                if parsed is not None:
                    magnitudes[key[:-3]] = parsed
        name = row.get("Source") or row.get("Name") or ""
        results.append(ReferenceObject(
            ra=ra, dec=dec, name=str(name), catalog=catalog,
            object_type="*" if magnitudes else "",
            magnitudes=magnitudes,
            meta={k: v for k, v in row.items() if k in ("pmRA", "pmDE", "Plx")}))
    return results


def parse_simbad_tsv(text: str) -> List[ReferenceObject]:
    """Parse SIMBAD's TAP tab-separated response.

    >>> text = 'main_id\\tra\\tdec\\totype\\n"V* AB Aur"\\t150.0\\t2.2\\t"V*"\\n'
    >>> obj = parse_simbad_tsv(text)[0]
    >>> obj.name, obj.described_type
    ('V* AB Aur', 'variable star')
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [value.strip().strip('"').lower() for value in lines[0].split("\t")]
    results: List[ReferenceObject] = []
    for line in lines[1:]:
        fields = [value.strip().strip('"') for value in line.split("\t")]
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        ra, dec = _as_float(row.get("ra")), _as_float(row.get("dec"))
        if ra is None or dec is None:
            continue
        results.append(ReferenceObject(
            ra=ra, dec=dec, name=row.get("main_id", ""), catalog="SIMBAD",
            object_type=row.get("otype", "")))
    return results


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _first_float(row: Dict[str, str], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            parsed = _as_float(row[key])
            if parsed is not None:
                return parsed
    return None


def read_reference_file(path: str) -> List[ReferenceObject]:
    """Read reference objects from a JSON list or a CSV with ra/dec columns."""
    if not os.path.exists(path):
        raise DataError(f"reference catalog not found: {path}")
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["objects"] if isinstance(payload, dict) else payload
        return [ReferenceObject.from_dict(row) for row in rows]

    import csv
    results: List[ReferenceObject] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lowered = {(k or "").strip().lower(): (v or "").strip()
                       for k, v in row.items()}
            ra = _first_float(lowered, ("ra", "ra_icrs", "raj2000", "_raj2000"))
            dec = _first_float(lowered, ("dec", "de_icrs", "dej2000", "_dej2000"))
            if ra is None or dec is None:
                continue
            magnitudes = {k[:-3]: _as_float(v) for k, v in lowered.items()
                          if k.endswith("mag") and _as_float(v) is not None}
            results.append(ReferenceObject(
                ra=ra, dec=dec, name=lowered.get("name", ""),
                catalog=lowered.get("catalog", os.path.basename(path)),
                object_type=lowered.get("type", "") or lowered.get("otype", ""),
                magnitudes={k: v for k, v in magnitudes.items() if v is not None}))
    return results


def write_reference_file(objects: Sequence[ReferenceObject], path: str) -> str:
    """Write reference objects as JSON, the format :func:`read_reference_file` prefers."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"objects": [o.to_dict() for o in objects]}, handle, indent=2)
    return path


# --------------------------------------------------------------------------
# the crossmatch itself
# --------------------------------------------------------------------------
@dataclass
class CrossmatchReport:
    """What the crossmatch established -- including that it ran at all."""

    performed: bool
    backend: Dict[str, Any]
    n_sources: int = 0
    n_matched: int = 0
    n_reference: int = 0
    radius_arcsec: float = 0.0
    field_radius_arcsec: float = 0.0
    error: Optional[str] = None

    @property
    def conclusive(self) -> bool:
        """Whether "no match" from this run may be read as "not known"."""
        return bool(self.performed and self.error is None and self.n_reference > 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "performed": bool(self.performed),
            "backend": dict(self.backend),
            "n_sources": int(self.n_sources),
            "n_matched": int(self.n_matched),
            "n_reference": int(self.n_reference),
            "radius_arcsec": float(self.radius_arcsec),
            "field_radius_arcsec": float(self.field_radius_arcsec),
            "conclusive": self.conclusive,
            "error": self.error,
        }


def field_cone(catalog: SourceCatalog, margin: float = 1.15
               ) -> Optional[Tuple[float, float, float]]:
    """The smallest cone covering every source, as ``(ra, dec, radius_arcsec)``.

    Returns ``None`` when the catalog has no sky coordinates, which is the
    case for any image without a WCS -- and is exactly when an external
    crossmatch is impossible rather than merely empty.
    """
    coordinates = [(s.ra, s.dec) for s in catalog
                   if s.ra is not None and s.dec is not None
                   and np.isfinite(s.ra) and np.isfinite(s.dec)]
    if not coordinates:
        return None
    ra = np.array([c[0] for c in coordinates], dtype=float)
    dec = np.array([c[1] for c in coordinates], dtype=float)
    # Averaging right ascension directly is wrong across the 0/360 wrap, so
    # the centre is taken as a unit-vector mean.
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    vector = np.array([np.mean(np.cos(dec_rad) * np.cos(ra_rad)),
                       np.mean(np.cos(dec_rad) * np.sin(ra_rad)),
                       np.mean(np.sin(dec_rad))])
    norm = float(np.linalg.norm(vector))
    if norm <= 0:                                              # pragma: no cover
        return None
    vector /= norm
    centre_ra = float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0)
    centre_dec = float(np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
    radius = float(np.max(angular_separation(centre_ra, centre_dec, ra, dec)) * 3600.0)
    return centre_ra, centre_dec, max(radius * float(margin), 1.0)


def crossmatch_catalog(catalog: SourceCatalog, service: Optional[ConeSearch] = None,
                       radius_arcsec: float = 2.0,
                       max_field_radius_arcsec: float = 3600.0) -> CrossmatchReport:
    """Annotate every source that matches a known object, in place.

    Matched sources gain ``meta["known_object"]`` and the ``known`` flag.  The
    report records whether the check actually ran, which is what lets the
    rest of the pipeline distinguish "checked and new" from "never checked".
    """
    service = service or NullCone()
    backend = service.describe()
    if isinstance(service, NullCone) or len(catalog) == 0:
        return CrossmatchReport(performed=False, backend=backend,
                                n_sources=len(catalog), radius_arcsec=radius_arcsec)

    cone = field_cone(catalog)
    if cone is None:
        return CrossmatchReport(performed=False, backend=backend,
                                n_sources=len(catalog), radius_arcsec=radius_arcsec,
                                error="catalog has no sky coordinates (no WCS)")
    centre_ra, centre_dec, field_radius = cone
    if field_radius > max_field_radius_arcsec:
        return CrossmatchReport(
            performed=False, backend=backend, n_sources=len(catalog),
            radius_arcsec=radius_arcsec, field_radius_arcsec=field_radius,
            error=f"field spans {field_radius / 60:.1f}' , beyond the "
                  f"{max_field_radius_arcsec / 60:.1f}' query limit")

    reference = service.query(centre_ra, centre_dec, field_radius)
    error = getattr(service, "last_error", None)
    if error is None and isinstance(service, CachedCone):
        error = getattr(service.inner, "last_error", None)

    report = CrossmatchReport(
        performed=True, backend=backend, n_sources=len(catalog),
        n_reference=len(reference), radius_arcsec=float(radius_arcsec),
        field_radius_arcsec=field_radius, error=error)
    if not reference:
        log.info("external crossmatch: no reference objects returned%s",
                 f" ({error})" if error else "")
        catalog.meta["crossmatch"] = report.to_dict()
        return report

    ref_ra = np.array([o.ra for o in reference], dtype=float)
    ref_dec = np.array([o.dec for o in reference], dtype=float)
    for source in catalog:
        if source.ra is None or source.dec is None:
            continue
        separation = angular_separation(float(source.ra), float(source.dec),
                                        ref_ra, ref_dec) * 3600.0
        best = int(np.argmin(separation))
        if separation[best] > float(radius_arcsec):
            continue
        match = reference[best]
        source.meta["known_object"] = {
            **match.to_dict(),
            "separation_arcsec": float(separation[best]),
            "described_type": match.described_type,
        }
        source.add_flag("known")
        report.n_matched += 1

    catalog.meta["crossmatch"] = report.to_dict()
    log.info("external crossmatch (%s): %d of %d sources match a known object "
             "within %.1f\" (%d reference objects in a %.1f' cone)",
             backend.get("backend", "?"), report.n_matched, len(catalog),
             radius_arcsec, len(reference), field_radius / 60.0)
    return report


def build_service(backend: str = "none", **options: Any) -> ConeSearch:
    """Construct a backend by name, wrapping it in a cache when asked.

    ``cache_dir`` applies to any backend; ``path`` is the local catalog file;
    ``catalog`` selects the VizieR table.
    """
    cache_dir = options.pop("cache_dir", None)
    max_age_days = float(options.pop("cache_max_age_days", 30.0))
    name = (backend or "none").lower()
    if name in ("none", "off", ""):
        return NullCone()
    if name == "local":
        path = options.get("path")
        if not path:
            raise DataError("the local crossmatch backend needs a 'path'")
        service: ConeSearch = LocalCone(path)
    elif name in ("vizier", "gaia"):
        service = VizieRCone(**{k: v for k, v in options.items()
                                if k in ("catalog", "columns", "timeout", "max_rows")})
    elif name == "simbad":
        service = SimbadCone(**{k: v for k, v in options.items()
                                if k in ("timeout", "max_rows")})
    else:
        raise DataError(f"unknown crossmatch backend {backend!r}")
    if cache_dir:
        return CachedCone(service, cache_dir, max_age_days)
    return service
