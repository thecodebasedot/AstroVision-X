"""What the astronomer is shown, and in what order.

The queue is the ranked follow-up list the pipeline already produces --
transients, lens candidates, anomalies, ordered by the priority stage --
with, for each entry, everything a person needs to decide in a few seconds:
the cutout, the background-subtracted cutout, the numbers the pipeline used,
the reasons it gave, the caveats it attached, and the object's history
across epochs when a catalog database is available.

The queue does not decide anything. It carries the model's label and its
confidence so the verdict can later be compared with them, which is how the
active-learning log measures whether the model is calibrated on the
decisions people actually make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.logging import get_logger
from ..engine.priority import rank_candidates

log = get_logger("vetting.queue")

#: Verdict keys a reviewer can give, and what they mean.
LABELS: Dict[str, str] = {
    "real": "an astrophysical object or event worth following up",
    "bogus": "an artifact, a cosmic ray, a subtraction residual, a mistake",
    "unsure": "cannot tell from this; keep it, but do not train on it",
}


@dataclass
class VettingItem:
    """One thing to look at."""

    item_id: int
    kind: str                                  # transient | lens | anomaly | source
    source_id: Optional[int]
    candidate_id: Optional[int]
    rank: int
    score: float
    model_verdict: str                         # the pipeline's recommendation
    model_label: str                           # what the model called it
    model_confidence: float
    x: float
    y: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    measurements: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    stamp: Optional[np.ndarray] = None
    stamp_subtracted: Optional[np.ndarray] = None

    @property
    def verdict_key(self) -> int:
        """The id a verdict is recorded against: the source when there is one,
        otherwise a candidate-derived id that cannot collide with a source."""
        if self.source_id is not None:
            return int(self.source_id)
        return -int(self.candidate_id or self.item_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id, "kind": self.kind, "source_id": self.source_id,
            "candidate_id": self.candidate_id, "rank": self.rank,
            "score": float(self.score), "model_verdict": self.model_verdict,
            "model_label": self.model_label,
            "model_confidence": (None if not np.isfinite(self.model_confidence)
                                 else float(self.model_confidence)),
            "x": float(self.x), "y": float(self.y), "ra": self.ra, "dec": self.dec,
            "reasons": list(self.reasons), "caveats": list(self.caveats),
            "evidence": {k: (None if not np.isfinite(v) else float(v))
                         for k, v in self.evidence.items()},
            "measurements": self.measurements, "history": list(self.history),
            "verdict_key": self.verdict_key,
            "has_subtracted": self.stamp_subtracted is not None,
        }


class VettingQueue:
    """An ordered list of items with lookup by id."""

    def __init__(self, items: Sequence[VettingItem], field_name: str = "",
                 image_shape: Optional[tuple] = None):
        self.items: List[VettingItem] = list(items)
        self.field_name = field_name
        self.image_shape = image_shape
        self._by_id = {item.item_id: item for item in self.items}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def get(self, item_id: int) -> Optional[VettingItem]:
        return self._by_id.get(int(item_id))

    def summary(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for item in self.items:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        return {"n_items": len(self.items), "kinds": kinds, "field": self.field_name}


def _measurements(source) -> Dict[str, Any]:
    if source is None:
        return {}
    p, m = source.photometry, source.morphology
    values = {
        "class": source.object_class.value, "class_confidence": source.class_confidence,
        "flux": p.flux, "flux_err": p.flux_err, "mag": p.magnitude, "mag_err": p.magnitude_err,
        "snr": p.snr, "fwhm": m.fwhm, "semi_major": m.semi_major, "ellipticity": m.ellipticity,
        "concentration": m.concentration, "morphology": m.label.value,
        "anomaly_score": source.anomaly_score, "lens_score": source.lens_score,
        "flags": list(source.flags),
    }
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in values.items()}


def build_queue(analysis, image, limit: int = 40, stamp_size: int = 64,
                include_sources: bool = False, db=None,
                history_radius_arcsec: float = 2.0) -> VettingQueue:
    """The pipeline's ranked candidates, with cutouts, as a queue.

    ``include_sources`` adds every catalog source after the candidates, for a
    field where the whole catalog is to be checked. ``db`` is a
    :class:`~astrovision.catalog.CatalogDB`; when given, each item with sky
    coordinates carries its detection history from the store.
    """
    ranked = rank_candidates(analysis, limit=limit)
    catalog = analysis.catalog
    by_id = {source.id: source for source in catalog}
    items: List[VettingItem] = []

    def make(kind, source_id, candidate_id, rank, score, verdict, label, confidence,
             x, y, reasons, caveats, evidence) -> VettingItem:
        source = by_id.get(source_id) if source_id is not None else None
        ra = dec = None
        if source is not None and source.ra is not None:
            ra, dec = float(source.ra), float(source.dec)
        elif getattr(image, "wcs", None) is not None:
            try:
                world = image.wcs.pixel_to_world(x, y)
                ra, dec = float(np.asarray(world[0]).ravel()[0]), float(np.asarray(world[1]).ravel()[0])
            except Exception:                              # pragma: no cover
                ra = dec = None
        item = VettingItem(
            item_id=len(items) + 1, kind=kind, source_id=source_id,
            candidate_id=candidate_id, rank=rank, score=float(score),
            model_verdict=str(verdict), model_label=str(label),
            model_confidence=float(confidence), x=float(x), y=float(y), ra=ra, dec=dec,
            reasons=list(reasons), caveats=list(caveats),
            evidence={k: float(v) for k, v in dict(evidence).items()},
            measurements=_measurements(source))
        if image is not None:
            item.stamp = image.cutout(x, y, stamp_size)
            if getattr(image, "background", None) is not None:
                item.stamp_subtracted = image.cutout(x, y, stamp_size, subtract_background=True)
        if db is not None and ra is not None:
            try:
                rows = db.cone_search(ra, dec, history_radius_arcsec)
                item.history = [{"mjd": r.get("mjd"), "band": r.get("band"),
                                 "flux": r.get("flux"), "flux_err": r.get("flux_err"),
                                 "mag": r.get("mag"), "field": r.get("field_name"),
                                 "separation_arcsec": r.get("separation_arcsec")}
                                for r in rows]
            except Exception as error:                     # pragma: no cover
                log.warning("history lookup failed: %s", error)
        return item

    for entry in ranked:
        x, y = entry.position
        source = by_id.get(entry.source_id) if entry.source_id is not None else None
        if entry.kind == "transient":
            label, confidence = "transient", entry.evidence.get("real_bogus", entry.score)
        elif entry.kind == "lens":
            label, confidence = "lens", entry.score
        elif entry.kind == "anomaly":
            label, confidence = "anomaly", entry.score
        elif source is not None:
            label, confidence = source.object_class.value, source.class_confidence
        else:
            label, confidence = entry.kind, entry.score
        items.append(make(entry.kind, entry.source_id, entry.candidate_id, entry.rank,
                          entry.score, entry.verdict.value, label, confidence, x, y,
                          entry.reasons, entry.caveats, entry.evidence))

    if include_sources:
        seen = {item.source_id for item in items if item.source_id is not None}
        ordered = sorted(catalog, key=lambda s: -(s.photometry.snr
                                                  if np.isfinite(s.photometry.snr) else 0.0))
        for source in ordered:
            if source.id in seen:
                continue
            items.append(make("source", source.id, None, len(items) + 1,
                              source.class_confidence, "not_interesting",
                              source.object_class.value, source.class_confidence,
                              source.x, source.y, [], [], {}))

    queue = VettingQueue(items, field_name=getattr(image, "name", "") or "",
                         image_shape=None if image is None else tuple(image.shape))
    log.info("vetting queue: %d items (%s)", len(queue),
             ", ".join(f"{k} {v}" for k, v in queue.summary()["kinds"].items()) or "empty")
    return queue
