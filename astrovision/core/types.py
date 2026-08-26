"""Canonical data structures exchanged between AstroVision-X subsystems.

Every stage of the platform speaks these types: the detector emits
:class:`Source` objects, photometry and morphology enrich them in place,
the transient stage produces :class:`TransientCandidate`, and the research
engine consumes the lot as a :class:`FieldAnalysis`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


class ObjectClass(str, Enum):
    """Top-level taxonomy assigned by the classification stage."""

    STAR = "star"
    GALAXY = "galaxy"
    NEBULA = "nebula"
    STAR_CLUSTER = "star_cluster"
    ASTEROID = "asteroid"
    COMET = "comet"
    SUPERNOVA_CANDIDATE = "supernova_candidate"
    LENS_CANDIDATE = "lens_candidate"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


class Morphology(str, Enum):
    """Galaxy morphological types (a coarse Hubble-style scheme)."""

    SPIRAL = "spiral"
    BARRED_SPIRAL = "barred_spiral"
    ELLIPTICAL = "elliptical"
    LENTICULAR = "lenticular"
    IRREGULAR = "irregular"
    MERGER = "merger"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    """How much human follow-up a candidate warrants.

    The platform never claims a discovery; it ranks things for astronomers.
    """

    NOT_INTERESTING = "not_interesting"
    WORTH_A_LOOK = "worth_a_look"
    FOLLOW_UP_RECOMMENDED = "follow_up_recommended"
    HIGH_PRIORITY = "high_priority"


@dataclass
class BoundingBox:
    """Axis-aligned pixel box, ``x``/``y`` in NumPy column/row order."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    def clip(self, shape: Tuple[int, int]) -> "BoundingBox":
        """Clamp the box to an image of shape ``(ny, nx)``."""
        ny, nx = shape
        return BoundingBox(
            float(np.clip(self.x_min, 0, nx)), float(np.clip(self.y_min, 0, ny)),
            float(np.clip(self.x_max, 0, nx)), float(np.clip(self.y_max, 0, ny)),
        )

    def slices(self, shape: Tuple[int, int], pad: int = 0) -> Tuple[slice, slice]:
        """Return ``(rows, cols)`` slices for cutting this box out of an array."""
        ny, nx = shape
        y0 = int(max(0, math.floor(self.y_min) - pad))
        y1 = int(min(ny, math.ceil(self.y_max) + pad))
        x0 = int(max(0, math.floor(self.x_min) - pad))
        x1 = int(min(nx, math.ceil(self.x_max) + pad))
        return slice(y0, max(y0 + 1, y1)), slice(x0, max(x0 + 1, x1))

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union with another box."""
        ix = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        iy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter = ix * iy
        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Photometry:
    """Flux/magnitude measurements for one source."""

    flux: float = float("nan")
    flux_err: float = float("nan")
    magnitude: float = float("nan")
    magnitude_err: float = float("nan")
    peak: float = float("nan")
    background: float = float("nan")
    snr: float = float("nan")
    aperture_radius: float = float("nan")
    kron_radius: float = float("nan")
    petrosian_radius: float = float("nan")
    surface_brightness: float = float("nan")
    saturated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MorphologyMetrics:
    """Non-parametric and parametric shape measurements."""

    semi_major: float = float("nan")
    semi_minor: float = float("nan")
    position_angle: float = float("nan")   # degrees, CCW from +x
    ellipticity: float = float("nan")
    elongation: float = float("nan")
    fwhm: float = float("nan")
    area_pixels: int = 0
    concentration: float = float("nan")    # CAS: C
    asymmetry: float = float("nan")        # CAS: A
    smoothness: float = float("nan")       # CAS: S
    gini: float = float("nan")
    m20: float = float("nan")
    sersic_index: float = float("nan")
    effective_radius: float = float("nan")
    spiral_strength: float = float("nan")
    bar_strength: float = float("nan")
    arm_count: int = 0
    label: Morphology = Morphology.UNKNOWN
    label_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["label"] = self.label.value
        return data


@dataclass
class Source:
    """A single detected astronomical object and everything measured about it."""

    id: int
    x: float
    y: float
    bbox: BoundingBox
    ra: Optional[float] = None
    dec: Optional[float] = None
    object_class: ObjectClass = ObjectClass.UNKNOWN
    class_confidence: float = 0.0
    class_scores: Dict[str, float] = field(default_factory=dict)
    photometry: Photometry = field(default_factory=Photometry)
    morphology: MorphologyMetrics = field(default_factory=MorphologyMetrics)
    anomaly_score: float = 0.0
    lens_score: float = 0.0
    variability_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    segment_label: int = 0
    flags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def is_extended(self) -> bool:
        """Extended sources are resolved beyond the point-spread function."""
        return self.object_class in (
            ObjectClass.GALAXY, ObjectClass.NEBULA, ObjectClass.STAR_CLUSTER,
        )

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def distance_to(self, other: "Source") -> float:
        return float(math.hypot(self.x - other.x, self.y - other.y))

    def cutout(self, image: np.ndarray, pad: int = 4) -> np.ndarray:
        """Extract this source's postage stamp from ``image``."""
        rows, cols = self.bbox.slices(image.shape[:2], pad=pad)
        return image[rows, cols]

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "ra": self.ra,
            "dec": self.dec,
            "bbox": self.bbox.to_dict(),
            "object_class": self.object_class.value,
            "class_confidence": float(self.class_confidence),
            "class_scores": {k: float(v) for k, v in self.class_scores.items()},
            "photometry": self.photometry.to_dict(),
            "morphology": self.morphology.to_dict(),
            "anomaly_score": float(self.anomaly_score),
            "lens_score": float(self.lens_score),
            "variability_score": float(self.variability_score),
            "segment_label": int(self.segment_label),
            "flags": list(self.flags),
            "meta": dict(self.meta),
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = np.asarray(self.embedding, dtype=float).tolist()
        return data


class SourceCatalog:
    """An ordered, queryable collection of :class:`Source` objects."""

    def __init__(self, sources: Optional[Iterable[Source]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.sources: List[Source] = list(sources or [])
        self.meta: Dict[str, Any] = dict(meta or {})

    # -- container protocol ----------------------------------------------
    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> Iterator[Source]:
        return iter(self.sources)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SourceCatalog(self.sources[index], self.meta)
        return self.sources[index]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<SourceCatalog n={len(self)}>"

    # -- mutation ---------------------------------------------------------
    def append(self, source: Source) -> None:
        self.sources.append(source)

    def extend(self, sources: Iterable[Source]) -> None:
        self.sources.extend(sources)

    def renumber(self, start: int = 1) -> "SourceCatalog":
        """Reassign sequential ids (after filtering or merging)."""
        for i, source in enumerate(self.sources, start=start):
            source.id = i
        return self

    # -- queries ----------------------------------------------------------
    def by_id(self, source_id: int) -> Optional[Source]:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def filter(self, predicate) -> "SourceCatalog":
        return SourceCatalog([s for s in self.sources if predicate(s)], self.meta)

    def of_class(self, object_class: ObjectClass) -> "SourceCatalog":
        return self.filter(lambda s: s.object_class == object_class)

    def sorted_by(self, key, descending: bool = True) -> "SourceCatalog":
        return SourceCatalog(sorted(self.sources, key=key, reverse=descending), self.meta)

    def brightest(self, n: int = 10) -> List[Source]:
        finite = [s for s in self.sources if np.isfinite(s.photometry.flux)]
        return sorted(finite, key=lambda s: s.photometry.flux, reverse=True)[:n]

    def positions(self) -> np.ndarray:
        """``(N, 2)`` array of ``(x, y)`` centroids."""
        if not self.sources:
            return np.empty((0, 2), dtype=float)
        return np.array([[s.x, s.y] for s in self.sources], dtype=float)

    def embeddings(self) -> Optional[np.ndarray]:
        """``(N, D)`` embedding matrix, or ``None`` if any source lacks one."""
        if not self.sources or any(s.embedding is None for s in self.sources):
            return None
        return np.vstack([np.asarray(s.embedding, dtype=float) for s in self.sources])

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for source in self.sources:
            counts[source.object_class.value] = counts.get(source.object_class.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def match(self, x: float, y: float, radius: float = 3.0) -> Optional[Source]:
        """Nearest source within ``radius`` pixels of ``(x, y)``."""
        best, best_d = None, radius
        for source in self.sources:
            d = math.hypot(source.x - x, source.y - y)
            if d <= best_d:
                best, best_d = source, d
        return best

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        return {
            "count": len(self),
            "meta": dict(self.meta),
            "sources": [s.to_dict(include_embedding) for s in self.sources],
        }


@dataclass
class LightCurve:
    """Time-ordered brightness measurements for one object."""

    times: np.ndarray
    fluxes: np.ndarray
    errors: Optional[np.ndarray] = None
    band: str = "clear"
    source_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float).ravel()
        self.fluxes = np.asarray(self.fluxes, dtype=float).ravel()
        if self.times.size != self.fluxes.size:
            raise ValueError(
                f"times ({self.times.size}) and fluxes ({self.fluxes.size}) must match"
            )
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float).ravel()
            if self.errors.size != self.times.size:
                raise ValueError("errors must have the same length as times")
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.fluxes = self.fluxes[order]
        if self.errors is not None:
            self.errors = self.errors[order]

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def baseline(self) -> float:
        """Total time span covered by the light curve."""
        return float(self.times[-1] - self.times[0]) if len(self) > 1 else 0.0

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.times) & np.isfinite(self.fluxes)

    def clean(self) -> "LightCurve":
        """Drop non-finite epochs."""
        mask = self.valid
        return LightCurve(
            self.times[mask], self.fluxes[mask],
            None if self.errors is None else self.errors[mask],
            self.band, self.source_id, dict(self.meta),
        )

    def normalized(self) -> np.ndarray:
        """Fluxes divided by their median (unitless, median ~= 1)."""
        median = float(np.nanmedian(self.fluxes)) if len(self) else 0.0
        if not np.isfinite(median) or abs(median) < 1e-12:
            return np.zeros_like(self.fluxes)
        return self.fluxes / median

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band": self.band,
            "source_id": self.source_id,
            "n_epochs": len(self),
            "baseline": self.baseline,
            "times": self.times.tolist(),
            "fluxes": self.fluxes.tolist(),
            "errors": None if self.errors is None else self.errors.tolist(),
            "meta": dict(self.meta),
        }


@dataclass
class TransientCandidate:
    """A brightness change found by comparing epochs of the same sky region."""

    id: int
    x: float
    y: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    significance: float = 0.0        # sigma of the residual detection
    delta_flux: float = 0.0
    delta_magnitude: float = float("nan")
    real_bogus: float = 0.0          # 1 = astrophysically real, 0 = artifact
    classification: str = "unknown"  # supernova / variable_star / mover / ...
    confidence: float = 0.0
    epoch_index: int = 0
    host_source_id: Optional[int] = None
    host_offset: float = float("nan")
    light_curve: Optional[LightCurve] = None
    verdict: Verdict = Verdict.WORTH_A_LOOK
    flags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "ra": self.ra,
            "dec": self.dec,
            "significance": float(self.significance),
            "delta_flux": float(self.delta_flux),
            "delta_magnitude": float(self.delta_magnitude),
            "real_bogus": float(self.real_bogus),
            "classification": self.classification,
            "confidence": float(self.confidence),
            "epoch_index": int(self.epoch_index),
            "host_source_id": self.host_source_id,
            "host_offset": float(self.host_offset),
            "verdict": self.verdict.value,
            "flags": list(self.flags),
            "meta": dict(self.meta),
            "light_curve": None if self.light_curve is None else self.light_curve.to_dict(),
        }


@dataclass
class AnomalyRecord:
    """An object that does not resemble anything the models were trained on."""

    source_id: int
    score: float
    rank: int = 0
    novelty_type: str = "unclassified"
    contributions: Dict[str, float] = field(default_factory=dict)
    nearest_neighbours: List[int] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LensCandidate:
    """A source showing possible strong-gravitational-lensing morphology."""

    source_id: int
    score: float
    arc_count: int = 0
    max_arc_length: float = 0.0
    arc_curvature: float = 0.0
    ring_completeness: float = 0.0
    einstein_radius_px: float = float("nan")
    einstein_radius_arcsec: float = float("nan")
    colour_contrast: float = float("nan")
    verdict: Verdict = Verdict.WORTH_A_LOOK
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


@dataclass
class FieldAnalysis:
    """Everything AstroVision-X derived from one field, ready for reporting."""

    catalog: SourceCatalog = field(default_factory=SourceCatalog)
    transients: List[TransientCandidate] = field(default_factory=list)
    anomalies: List[AnomalyRecord] = field(default_factory=list)
    lenses: List[LensCandidate] = field(default_factory=list)
    light_curves: Dict[int, LightCurve] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def summary(self) -> Dict[str, Any]:
        counts = self.catalog.class_counts()
        return {
            "n_sources": len(self.catalog),
            "class_counts": counts,
            "n_transients": len(self.transients),
            "n_anomalies": len(self.anomalies),
            "n_lens_candidates": len(self.lenses),
            "n_light_curves": len(self.light_curves),
        }

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "catalog": self.catalog.to_dict(include_embedding),
            "transients": [t.to_dict() for t in self.transients],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "lens_candidates": [l.to_dict() for l in self.lenses],
            "statistics": self.statistics,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
        }
