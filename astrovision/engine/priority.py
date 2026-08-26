"""Ranking what a human should look at first.

A run over a wide field produces thousands of objects and, if it is a good
night, a handful worth a person's time.  This module turns the per-stage
scores into one ordered list, with the reason for each ranking attached --
because a priority nobody can interrogate is not useful to a scientist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.logging import get_logger
from ..core.types import (
    FieldAnalysis,
    TransientCandidate,
    Verdict,
)

log = get_logger("engine.priority")

#: Relative importance of each kind of evidence in the final ranking.
PRIORITY_WEIGHTS = {
    "transient": 1.00,
    "lens": 0.85,
    "anomaly": 0.70,
    "variability": 0.55,
    "morphology": 0.30,
}


@dataclass
class PriorityItem:
    """One entry in the ranked follow-up list."""

    rank: int
    kind: str                       # transient | lens | anomaly | variable | source
    source_id: Optional[int]
    candidate_id: Optional[int]
    score: float
    verdict: Verdict
    position: tuple
    sky_position: Optional[tuple] = None
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank, "kind": self.kind, "source_id": self.source_id,
            "candidate_id": self.candidate_id, "score": float(self.score),
            "verdict": self.verdict.value,
            "position": [float(v) for v in self.position],
            "sky_position": (None if self.sky_position is None
                             else [float(v) for v in self.sky_position]),
            "reasons": list(self.reasons), "caveats": list(self.caveats),
            "evidence": {k: float(v) for k, v in self.evidence.items()},
        }


def rank_candidates(analysis: FieldAnalysis, limit: int = 20) -> List[PriorityItem]:
    """Merge every stage's findings into one ranked follow-up list."""
    items: List[PriorityItem] = []
    catalog = analysis.catalog

    for candidate in analysis.transients:
        if "bogus" in candidate.flags:
            continue
        score = PRIORITY_WEIGHTS["transient"] * _transient_score(candidate)
        reasons = [f"{candidate.significance:.0f} sigma residual in the difference image",
                   f"real/bogus {candidate.real_bogus:.2f}",
                   f"best match: {candidate.classification.replace('_', ' ')}"]
        caveats = ["Photometric candidate only: confirmation needs a second "
                   "independent epoch, and a spectrum for any supernova claim."]
        if "nuclear" in candidate.flags:
            caveats.append("Sits on the host nucleus -- nuclear variability is "
                           "at least as likely as a supernova.")
        if candidate.meta.get("n_detections", 1) < 2:
            caveats.append("Detected in a single epoch only.")
        items.append(PriorityItem(
            rank=0, kind="transient", source_id=candidate.host_source_id,
            candidate_id=candidate.id, score=score, verdict=candidate.verdict,
            position=(candidate.x, candidate.y),
            sky_position=(None if candidate.ra is None else (candidate.ra, candidate.dec)),
            reasons=reasons, caveats=caveats,
            evidence={"real_bogus": candidate.real_bogus,
                      "significance": candidate.significance,
                      "confidence": candidate.confidence,
                      "n_epochs": float(candidate.meta.get("n_detections", 1))}))

    for lens in analysis.lenses:
        source = catalog.by_id(lens.source_id)
        score = PRIORITY_WEIGHTS["lens"] * float(lens.score)
        items.append(PriorityItem(
            rank=0, kind="lens", source_id=lens.source_id, candidate_id=None,
            score=score, verdict=lens.verdict,
            position=(source.x, source.y) if source else (float("nan"),) * 2,
            sky_position=((source.ra, source.dec)
                          if source is not None and source.ra is not None else None),
            reasons=[f"{lens.arc_count} tangential arc(s), longest "
                     f"{lens.max_arc_length:.1f} px",
                     f"Einstein radius {lens.einstein_radius_arcsec:.2f} arcsec",
                     f"ring {100 * lens.ring_completeness:.0f}% complete"],
            caveats=list(lens.notes[-1:]) or
                    ["Lens candidate only: needs colour and spectroscopic confirmation."],
            evidence={"lens_score": lens.score,
                      "ring_completeness": lens.ring_completeness,
                      "arc_count": float(lens.arc_count)}))

    for record in analysis.anomalies:
        source = catalog.by_id(record.source_id)
        if source is None:
            continue
        score = PRIORITY_WEIGHTS["anomaly"] * float(record.score)
        items.append(PriorityItem(
            rank=0, kind="anomaly", source_id=record.source_id, candidate_id=None,
            score=score, verdict=_anomaly_verdict(record.score),
            position=(source.x, source.y),
            sky_position=(source.ra, source.dec) if source.ra is not None else None,
            reasons=[record.explanation.split(". ")[0],
                     f"novelty type: {record.novelty_type.replace('_', ' ')}"],
            caveats=["An outlier score measures dissimilarity from this field, "
                     "not physical novelty; instrumental artefacts score highly too."],
            evidence={"anomaly_score": record.score, **record.contributions}))

    for source in catalog:
        if source.variability_score >= 0.6 and "variable" in source.flags:
            score = PRIORITY_WEIGHTS["variability"] * float(source.variability_score)
            reasons = [f"variability score {source.variability_score:.2f}"]
            period = source.meta.get("period", {})
            if period.get("period") and np.isfinite(period.get("period", np.nan)):
                reasons.append(f"period {period['period']:.3f} d "
                               f"(false-alarm probability "
                               f"{period.get('false_alarm_probability', float('nan')):.1e})")
            items.append(PriorityItem(
                rank=0, kind="variable", source_id=source.id, candidate_id=None,
                score=score, verdict=Verdict.WORTH_A_LOOK,
                position=(source.x, source.y),
                sky_position=(source.ra, source.dec) if source.ra is not None else None,
                reasons=reasons,
                caveats=["Variability measured over a short baseline; a longer "
                         "series is needed to establish a period reliably."],
                evidence={"variability": source.variability_score}))

    # Several stages can flag the same object; keep its strongest case and
    # record that the evidence was corroborated, which matters for ranking.
    merged: Dict[tuple, PriorityItem] = {}
    for item in sorted(items, key=lambda i: -i.score):
        key = (round(item.position[0], 1), round(item.position[1], 1))
        if key in merged:
            existing = merged[key]
            existing.reasons.append(f"also flagged as a {item.kind} candidate")
            existing.evidence.update({f"{item.kind}_score": item.score})
            existing.score = min(1.0, existing.score + 0.15 * item.score)
            continue
        merged[key] = item

    ranked = sorted(merged.values(), key=lambda i: -i.score)[:max(1, limit)]
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    log.info("ranked %d follow-up targets from %d flagged objects",
             len(ranked), len(items))
    return ranked


def _transient_score(candidate: TransientCandidate) -> float:
    """Combine a transient's evidence into a 0-1 priority contribution."""
    score = 0.45 * float(np.clip(candidate.real_bogus, 0, 1))
    score += 0.25 * float(np.clip(candidate.significance / 20.0, 0, 1))
    score += 0.15 * float(np.clip(candidate.confidence, 0, 1))
    if "multi_epoch" in candidate.flags:
        score += 0.15
    if "dipole" in candidate.flags:
        score -= 0.20
    return float(np.clip(score, 0.0, 1.0))


def _anomaly_verdict(score: float) -> Verdict:
    if score >= 0.98:
        return Verdict.FOLLOW_UP_RECOMMENDED
    if score >= 0.9:
        return Verdict.WORTH_A_LOOK
    return Verdict.NOT_INTERESTING
