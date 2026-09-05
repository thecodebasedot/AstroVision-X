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
    history_source: str = "catalog database"   # where ``history`` came from
    stamp: Optional[np.ndarray] = None
    stamp_subtracted: Optional[np.ndarray] = None
    stamp_label: str = "image, asinh stretch"
    stamp_subtracted_label: str = "background subtracted"

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
            "history_source": self.history_source,
            "stamp_label": self.stamp_label,
            "stamp_subtracted_label": self.stamp_subtracted_label,
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


# -- alerts as a queue -----------------------------------------------------------
def _finite(value) -> Optional[float]:
    return None if value is None or not np.isfinite(float(value)) else float(value)


def _history_rows(packet) -> List[Dict[str, Any]]:
    rows = []
    for d in packet.history:
        rows.append({"mjd": _finite(d.mjd), "band": d.band, "flux": _finite(d.flux),
                     "flux_err": _finite(d.flux_err), "mag": _finite(d.mag),
                     "mag_err": _finite(d.mag_err), "limiting_mag": _finite(d.limiting_mag),
                     "forced": bool(d.forced), "field": packet.object_id,
                     "separation_arcsec": None})
    rows.append({"mjd": _finite(packet.mjd), "band": packet.band, "flux": _finite(packet.flux),
                 "flux_err": _finite(packet.flux_err), "mag": _finite(packet.mag),
                 "mag_err": _finite(packet.mag_err), "limiting_mag": _finite(packet.limiting_mag),
                 "forced": False, "field": packet.object_id, "separation_arcsec": 0.0})
    rows.sort(key=lambda r: (r["mjd"] is None, r["mjd"] or 0.0))
    return rows


def queue_from_alerts(packets, limit: Optional[int] = None, db=None,
                      history_radius_arcsec: float = 2.0,
                      field_name: str = "alerts") -> VettingQueue:
    """Alert packets -- this package's, ZTF's or Rubin's -- as a vetting queue.

    The science cutout is the stamp and the difference cutout the
    "subtracted" one; the packet's own history (previous detections, upper
    limits, forced photometry) is the light curve; the broker's real-bogus
    score is the rank.  Nothing is re-measured: an alert carries what its
    pipeline saw, and the page shows exactly that.  With ``db`` given, this
    package's own detections within ``history_radius_arcsec`` of the alert
    are added to the history under their field names.
    """
    def score_of(p) -> float:
        for value in (p.real_bogus, p.deep_real_bogus):
            if value is not None and np.isfinite(float(value)):
                return float(value)
        return 0.5

    ordered = sorted(packets, key=lambda p: -score_of(p))
    if limit is not None:
        ordered = ordered[:int(limit)]
    items: List[VettingItem] = []
    for rank, p in enumerate(ordered, 1):
        stamp = p.cutout_science
        subtracted = p.cutout_difference
        if stamp is None and p.cutout_template is not None:
            stamp = p.cutout_template
        height, width = stamp.shape if stamp is not None else (0, 0)
        x, y = (width - 1) / 2.0, (height - 1) / 2.0
        n_detections = sum(1 for d in p.history if d.is_detection and not d.forced)
        n_limits = sum(1 for d in p.history if not d.is_detection and not d.forced)
        n_forced = sum(1 for d in p.history if d.forced)
        epoch = f"MJD {p.mjd:.3f}" if p.mjd is not None and np.isfinite(p.mjd) else "no epoch"
        mag = (f"{p.band} {p.mag:.2f}" + (f" ± {p.mag_err:.2f}" if p.mag_err else "")
               if p.mag is not None else f"{p.band} (no magnitude)")
        reasons = [f"{p.publisher or p.source_format} alert {p.object_id}, candid {p.candid}",
                   f"{mag} at {epoch}",
                   f"{n_detections} earlier detection(s), {n_limits} upper limit(s)"
                   + (f", {n_forced} forced-photometry point(s)" if n_forced else "")
                   + " in the packet"]
        if p.classification:
            reasons.append(f"classified by its pipeline as {p.classification}")
        if p.human_verdict:
            reasons.append(f"a reviewer already said: {p.human_verdict}")
        caveats = ["the score and the cutouts are the alert pipeline's own; nothing here "
                   "was re-measured by this package"]
        if stamp is None:
            caveats.append("the packet carries no cutout")
        if p.host_distance_arcsec is not None:
            caveats.append(f"nearest reference source {p.host_distance_arcsec:.1f} arcsec away"
                           + (f" (star score {p.host_star_score:.2f})"
                              if p.host_star_score is not None else ""))
        evidence = {k: v for k, v in {
            "real_bogus": _finite(p.real_bogus), "deep_real_bogus": _finite(p.deep_real_bogus),
            "fwhm": _finite(p.fwhm), "host_distance_arcsec": _finite(p.host_distance_arcsec),
            "host_mag": _finite(p.host_mag), "host_star_score": _finite(p.host_star_score),
        }.items() if v is not None}
        measurements = {"object_id": p.object_id, "candid": p.candid, "publisher": p.publisher,
                        "schema": p.schema_version, "mjd": _finite(p.mjd), "band": p.band,
                        "mag": _finite(p.mag), "mag_err": _finite(p.mag_err),
                        "flux": _finite(p.flux), "flux_err": _finite(p.flux_err),
                        "limiting_mag": _finite(p.limiting_mag), "is_positive": p.is_positive,
                        "classification": p.classification}
        item = VettingItem(
            item_id=len(items) + 1, kind="transient", source_id=None,
            candidate_id=int(p.candid), rank=rank, score=score_of(p),
            model_verdict=str(p.verdict or "unranked"),
            model_label=str(p.classification or "transient"),
            model_confidence=score_of(p), x=x, y=y,
            ra=_finite(p.ra), dec=_finite(p.dec), reasons=reasons, caveats=caveats,
            evidence=evidence, measurements=measurements, history=_history_rows(p),
            history_source="alert packet", stamp=stamp, stamp_subtracted=subtracted,
            stamp_label=("science cutout from the alert" if p.cutout_science is not None
                         else "template cutout from the alert"),
            stamp_subtracted_label="difference cutout from the alert")
        if db is not None and item.ra is not None and item.dec is not None:
            try:
                for r in db.cone_search(item.ra, item.dec, history_radius_arcsec):
                    item.history.append({"mjd": r.get("mjd"), "band": r.get("band"),
                                         "flux": r.get("flux"), "flux_err": r.get("flux_err"),
                                         "mag": r.get("mag"), "mag_err": None,
                                         "limiting_mag": None, "forced": False,
                                         "field": r.get("field_name"),
                                         "separation_arcsec": r.get("separation_arcsec")})
                item.history_source = "alert packet and catalog database"
            except Exception as error:                     # pragma: no cover
                log.warning("history lookup failed: %s", error)
        items.append(item)
    queue = VettingQueue(items, field_name=field_name, image_shape=None)
    log.info("vetting queue from %d alert(s): %d items", len(list(packets)), len(queue))
    return queue


def is_alert_file(path: str) -> bool:
    """True for an Avro object container (by its magic bytes, not its name)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"Obj\x01"
    except OSError:
        return False


def queue_for_alert_file(path: str, limit: Optional[int] = None, db=None) -> VettingQueue:
    """Read an alert file and queue its packets; see :func:`queue_from_alerts`."""
    from ..alerts import read_alerts

    _, packets = read_alerts(path)
    import os
    return queue_from_alerts(packets, limit=limit, db=db, field_name=os.path.basename(path))
