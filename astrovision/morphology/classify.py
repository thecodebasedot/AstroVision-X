"""Morphological classification of resolved objects.

The rules encode published separations in the non-parametric morphology
plane -- the Lotz Gini/M20 lines, the Conselice CAS boundaries, and the
Sersic index -- rather than arbitrary cuts.  Each is expressed as a soft
score so that objects near a boundary get an honest, low confidence
instead of a false certainty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import softmax
from ..core.types import Morphology, MorphologyMetrics
from .gini_m20 import bulge_statistic, merger_statistic

log = get_logger("morphology.classify")

#: Evidence weights per morphological class; tuned on the reference
#: measurements in the literature rather than fitted to one dataset.
RULE_WEIGHTS: Dict[str, float] = {
    "sersic": 0.8,
    "concentration": 1.6,
    "gini_m20": 1.1,
    "asymmetry": 1.0,
    "smoothness": 0.7,
    "arms": 2.2,
    "bar": 1.6,
    "ellipticity": 0.5,
    "merger": 1.15,
}


def _bell(value: float, centre: float, width: float) -> float:
    """Soft membership in ``[0, 1]``, peaking at ``centre``."""
    if not np.isfinite(value):
        return 0.0
    return float(np.exp(-0.5 * ((value - centre) / max(width, 1e-6)) ** 2))


def _step(value: float, threshold: float, width: float, above: bool = True) -> float:
    """Soft threshold: 1 well past ``threshold``, 0 well before it."""
    if not np.isfinite(value):
        return 0.0
    z = (value - threshold) / max(width, 1e-6)
    if not above:
        z = -z
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))))


def classify_morphology(metrics: MorphologyMetrics,
                        extras: Optional[Dict[str, Any]] = None
                        ) -> Tuple[Morphology, float, Dict[str, float]]:
    """Assign a morphological type from the measured statistics.

    The decision is hierarchical, following the physics rather than a flat
    vote.  First: is the light bulge-dominated or disc-dominated?  The
    Sersic index, the concentration index and the Gini/M20 bulge line all
    answer that, and they agree well.  Then, within each branch, the
    property that actually defines the sub-type decides:

    * bulge-dominated splits on *flattening* -- an elliptical is a
      three-dimensional spheroid and looks round, whereas a lenticular is
      a bulge inside a thin disc and looks flattened;
    * disc-dominated splits on *order* -- an ordered disc shows an arm
      pattern or a bar, a disturbed one does not;
    * a strongly asymmetric object with a high M20 is a merger candidate
      regardless of which branch it started in.

    Returns ``(label, confidence, scores)``; ``scores`` carries the
    normalised evidence for every class so a report can show the runner-up.
    """
    extras = extras or {}

    n = metrics.sersic_index
    concentration = metrics.concentration
    asymmetry = metrics.asymmetry
    smooth = metrics.smoothness
    gini = metrics.gini
    m20_value = metrics.m20
    ellipticity = metrics.ellipticity
    arm_strength = metrics.spiral_strength
    arm_count = metrics.arm_count
    bar_detected = bool(extras.get("bar_detected", False))
    coherent = float(extras.get("coherence", 0.0))

    # A tiny footprint cannot support any of this.
    if metrics.area_pixels and metrics.area_pixels < 12:
        return Morphology.UNRESOLVED, 0.2, {Morphology.UNRESOLVED.value: 1.0}

    w = RULE_WEIGHTS

    # ------------------------------------------------------------------
    # 1. Bulge-dominated or disc-dominated?
    # ------------------------------------------------------------------
    early_votes: List[Tuple[float, float]] = []      # (evidence, weight)
    if np.isfinite(n):
        # A pure spheroid fits n ~ 4; a disc fits n ~ 1.  A disc with a
        # bulge lands in between, which is why this vote is not decisive.
        early_votes.append((_step(n, 2.4, 0.8), w["sersic"]))
    if np.isfinite(concentration):
        # C = 5.2 for a de Vaucouleurs profile, 2.7 for an exponential disc.
        early_votes.append((_step(concentration, 3.05, 0.28), w["concentration"]))
    if np.isfinite(gini) and np.isfinite(m20_value):
        early_votes.append((_step(bulge_statistic(gini, m20_value), 0.0, 0.035),
                            w["gini_m20"]))
    if np.isfinite(smooth):
        early_votes.append((_step(smooth, 0.30, 0.10, above=False), w["smoothness"]))

    if not early_votes:
        return Morphology.UNKNOWN, 0.0, {}
    total_weight = sum(weight for _, weight in early_votes)
    early = float(sum(value * weight for value, weight in early_votes) / total_weight)
    late = 1.0 - early

    # ------------------------------------------------------------------
    # 2. Within the bulge-dominated branch: round or flattened?
    # ------------------------------------------------------------------
    # An elliptical is a spheroid seen from any angle, so it is rarely very
    # flattened; a lenticular's light is dominated by a thin disc.
    flattened = _step(ellipticity, 0.32, 0.09) if np.isfinite(ellipticity) else 0.35

    # ------------------------------------------------------------------
    # 3. Within the disc-dominated branch: ordered or disturbed?
    # ------------------------------------------------------------------
    # Grade the arm evidence rather than treating detection as a switch:
    # at survey depth a genuine pattern often sits just below the formal
    # detection threshold, and that partial signal is still informative.
    arm_evidence = 0.0
    significance = float(extras.get("arm_significance", 0.0) or 0.0)
    if np.isfinite(significance) and significance > 1.0:
        arm_evidence = float(np.clip((significance - 1.2) / 2.5, 0.0, 1.0))
    if np.isfinite(arm_strength) and arm_count in (2, 3, 4):
        confirmed = _step(arm_strength, 0.02, 0.012) * (0.45 + 0.55 * min(coherent / 0.35, 1.0))
        arm_evidence = max(arm_evidence, confirmed)
    if bar_detected:
        arm_evidence = max(arm_evidence, 0.65)

    # What actually separates an irregular from a disc is that its light is
    # neither centrally concentrated nor unequally distributed: low
    # concentration, low Gini and a high M20.  Asymmetry adds to that, but
    # clumpiness on its own does not -- spiral arms are clumpy too.
    disturbance_votes: List[Tuple[float, float]] = []
    if np.isfinite(concentration):
        disturbance_votes.append((_step(concentration, 2.44, 0.13, above=False), 1.3))
    if np.isfinite(gini):
        disturbance_votes.append((_step(gini, 0.462, 0.020, above=False), 1.2))
    if np.isfinite(m20_value):
        disturbance_votes.append((_step(m20_value, -1.59, 0.06), 1.0))
    if np.isfinite(asymmetry):
        disturbance_votes.append((_step(asymmetry, 0.20, 0.06), 1.0))
    if np.isfinite(ellipticity):
        # A flattened outline means an inclined disc, and a disc is an
        # ordered system; irregulars have no preferred plane and look round
        # on average.
        disturbance_votes.append((_step(ellipticity, 0.26, 0.09, above=False), 0.9))
    if disturbance_votes:
        weight_sum = sum(weight for _, weight in disturbance_votes)
        disturbance = float(sum(v * weight for v, weight in disturbance_votes) / weight_sum)
    else:
        disturbance = 0.0

    # Failing to detect arms is weak evidence at survey depth -- roughly
    # half of genuine spirals show no significant Fourier signal -- so an
    # object is called irregular only when it looks actively disturbed.
    ordered = float(np.clip(0.62 + 0.38 * arm_evidence - 1.15 * disturbance, 0.05, 1.0))

    # ------------------------------------------------------------------
    # 4. Merger: strongly asymmetric, or a second nucleus raising M20.
    # ------------------------------------------------------------------
    merger = 0.0
    if np.isfinite(asymmetry):
        # A > 0.35 is Conselice's merger criterion.
        merger = max(merger, _step(asymmetry, 0.35, 0.06))
    if np.isfinite(gini) and np.isfinite(m20_value):
        merger = max(merger, _step(merger_statistic(gini, m20_value), 0.0, 0.04) *
                     _step(m20_value, -1.30, 0.10))

    # ------------------------------------------------------------------
    # 5. Assemble.
    # ------------------------------------------------------------------
    bar_share = 0.0
    if bar_detected:
        bar_share = 0.75
    elif np.isfinite(metrics.bar_strength) and metrics.bar_strength > 0.25:
        bar_share = 0.35

    evidence: Dict[str, float] = {
        Morphology.ELLIPTICAL.value: early * (1.0 - flattened),
        Morphology.LENTICULAR.value: early * flattened,
        Morphology.SPIRAL.value: late * ordered * (1.0 - bar_share),
        Morphology.BARRED_SPIRAL.value: late * ordered * bar_share,
        Morphology.IRREGULAR.value: late * (1.0 - ordered),
        Morphology.MERGER.value: merger * w["merger"],
    }

    names = list(evidence)
    values = np.array([evidence[k] for k in names], dtype=float)
    if not np.isfinite(values).any() or float(values.max()) <= 0:
        return Morphology.UNKNOWN, 0.0, {}

    probabilities = values / values.sum()
    order = np.argsort(probabilities)[::-1]
    best = names[int(order[0])]
    # Confidence reflects the *margin* over the runner-up, not the raw
    # share: a near-tie must report as a near-tie.
    confidence = float(probabilities[order[0]])
    if len(order) > 1:
        margin = float(probabilities[order[0]] - probabilities[order[1]])
        confidence = float(np.clip(confidence * (0.35 + 0.65 * min(margin / 0.25, 1.0)),
                                   0.0, 0.99))

    scores = {names[i]: float(probabilities[i]) for i in order}
    return Morphology(best), confidence, scores


def morphology_summary(metrics: MorphologyMetrics) -> str:
    """A one-line human description of an object's measured morphology."""
    parts = [metrics.label.value.replace("_", " ")]
    if np.isfinite(metrics.sersic_index):
        parts.append(f"n={metrics.sersic_index:.1f}")
    if np.isfinite(metrics.concentration):
        parts.append(f"C={metrics.concentration:.2f}")
    if np.isfinite(metrics.asymmetry):
        parts.append(f"A={metrics.asymmetry:.3f}")
    if metrics.arm_count:
        parts.append(f"{metrics.arm_count} arms")
    if np.isfinite(metrics.ellipticity):
        parts.append(f"e={metrics.ellipticity:.2f}")
    return ", ".join(parts)
