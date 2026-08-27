"""Linking detections across epochs into tracklets.

Difference imaging finds an asteroid as easily as a supernova: both are
sources with no counterpart in the template.  What separates them is that the
supernova appears in the same place every time and the asteroid does not --
so a mover arrives at the transient stage as *N separate single-epoch
candidates*, each looking like a marginal, unconfirmed transient, and the
position-based merging that consolidates a real transient scatters it.

Linking is what puts it back together.  The method is the standard one:

1. Take every pair of detections from two different epochs and read off the
   velocity that pair implies.
2. Reject pairs whose implied rate is outside the range being searched.
3. Predict where that velocity would put the object at every other epoch, and
   collect the detections that are there.
4. Keep the sets with enough points, fit a track, and cut on the residual.

The whole difficulty is step 4's threshold, because unrelated detections do
line up by chance.  Rather than pick a number, this module *estimates the
chance rate* from the data's own density and reports it per tracklet, so a
marginal link is marginal on the record instead of silently accepted.

**What this cannot do.** Motion is treated as linear, which is right over a
night and wrong over weeks: real motion is a great circle with parallax and
acceleration.  The linker refuses arcs longer than
:data:`MAX_LINEAR_ARC_DAYS` rather than fitting a straight line through
curvature and reporting a small residual, which would be the worst of both.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from .tracklet import Detection, Tracklet, build_tracklet

log = get_logger("moving.linking")

#: Beyond this arc, constant-velocity motion stops being a fair model.
MAX_LINEAR_ARC_DAYS = 1.5


@dataclass
class LinkingReport:
    """What the linker did, and how much of it could be chance."""

    n_detections: int
    n_epochs: int
    n_pairs_tried: int
    n_tracklets: int
    arc_days: float
    field_area_pixels: float
    tolerance: float
    expected_chance_tracklets: float
    refused: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_detections": int(self.n_detections),
            "n_epochs": int(self.n_epochs),
            "n_pairs_tried": int(self.n_pairs_tried),
            "n_tracklets": int(self.n_tracklets),
            "arc_days": float(self.arc_days),
            "tolerance_pixels": float(self.tolerance),
            "expected_chance_tracklets": float(self.expected_chance_tracklets),
            "refused": self.refused,
        }


def chance_alignment_rate(n_per_epoch: Sequence[int], tolerance: float,
                          field_area: float, min_points: int) -> float:
    """Expected number of tracklets built from unrelated detections.

    A pair of detections defines a velocity; the chance that an unrelated
    detection in a third epoch sits within ``tolerance`` of the predicted
    position is the ratio of the tolerance disc to the field area, times the
    number of detections in that epoch.  Multiplying that through for the
    epochs a tracklet needs, and over all pairs, gives the expected count.

    It is an estimate, not a bound -- it assumes detections are uniformly
    scattered, and real ones cluster around bright galaxies and bad columns,
    which makes the true rate higher.  Reported so that a tracklet found in a
    field where the estimate is 5 is read differently from one where it is
    0.01.

    >>> round(chance_alignment_rate([20, 20, 20], 3.0, 250000.0, 3), 3)
    2.714

    Two and a half expected false tracklets from twenty detections an epoch
    over three epochs is not a bug in the estimate -- it is what three epochs
    and a three-pixel tolerance actually buy.  More epochs, or a tighter
    tolerance, is the only cure.
    """
    counts = [int(c) for c in n_per_epoch]
    if len(counts) < 2 or field_area <= 0:
        return 0.0
    disc = math.pi * float(tolerance) ** 2
    probability = min(disc / float(field_area), 1.0)
    total = 0.0
    for pair in itertools.combinations(range(len(counts)), 2):
        n_pairs = counts[pair[0]] * counts[pair[1]]
        others = [counts[i] for i in range(len(counts)) if i not in pair]
        needed = max(0, int(min_points) - 2)
        if needed == 0:
            total += n_pairs
            continue
        if len(others) < needed:
            continue
        # Expected number of *ways* to complete the tracklet from the
        # remaining epochs: sum over which epochs supply the extra points.
        for chosen in itertools.combinations(range(len(others)), needed):
            ways = 1.0
            for index in chosen:
                ways *= others[index] * probability
            total += n_pairs * ways
    return float(total)


def _detections_by_epoch(detections: Sequence[Detection]
                         ) -> Tuple[List[List[Detection]], List[float]]:
    """Group detections by epoch, ordered by time."""
    grouped: Dict[int, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(int(detection.epoch), []).append(detection)
    epochs = sorted(grouped)
    times = [float(np.median([d.time for d in grouped[e]])) for e in epochs]
    order = np.argsort(times)
    return ([grouped[epochs[int(i)]] for i in order],
            [times[int(i)] for i in order])


def link_tracklets(detections: Sequence[Detection],
                   min_rate: float = 0.0,
                   max_rate: float = float("inf"),
                   pixel_scale: float = float("nan"),
                   tolerance: float = 3.0,
                   min_points: int = 3,
                   max_rms: float = 2.0,
                   wcs=None,
                   field_shape: Optional[Tuple[int, int]] = None,
                   max_arc_days: float = MAX_LINEAR_ARC_DAYS,
                   ) -> Tuple[List[Tracklet], LinkingReport]:
    """Find sets of detections on a common linear track.

    ``min_rate`` and ``max_rate`` are in **arcseconds per hour** when a pixel
    scale or WCS is available, and in pixels per day otherwise.  A rate range
    is not an optimisation: without an upper bound every pair of detections in
    a crowded field defines a "velocity", and the search returns noise.

    Returns the tracklets and a report that includes how many of them the
    field's own detection density would produce by chance.
    """
    detections = [d for d in detections if np.isfinite(d.x) and np.isfinite(d.y)]
    per_epoch, times = _detections_by_epoch(detections)
    counts = [len(group) for group in per_epoch]
    arc = float(max(times) - min(times)) if len(times) >= 2 else 0.0
    area = float(field_shape[0] * field_shape[1]) if field_shape else float("nan")

    scale = float(pixel_scale)
    if wcs is not None:
        try:
            scale = float(wcs.pixel_scale)
        except (AttributeError, TypeError):                    # pragma: no cover
            pass
    #: Rate limits are converted into the pixels-per-day the search works in.
    if np.isfinite(scale) and scale > 0:
        low = float(min_rate) * 24.0 / scale
        high = float(max_rate) * 24.0 / scale if np.isfinite(max_rate) else float("inf")
    else:
        low, high = float(min_rate), float(max_rate)

    report = LinkingReport(
        n_detections=len(detections), n_epochs=len(per_epoch), n_pairs_tried=0,
        n_tracklets=0, arc_days=arc, field_area_pixels=area, tolerance=float(tolerance),
        expected_chance_tracklets=0.0)

    if len(per_epoch) < max(2, int(min_points)):
        report.refused = (f"linking needs at least {max(2, int(min_points))} epochs; "
                          f"got {len(per_epoch)}")
        log.info("moving-object linking: %s", report.refused)
        return [], report
    if arc > float(max_arc_days):
        report.refused = (
            f"arc of {arc:.2f} days exceeds the {max_arc_days:.2f}-day limit for a "
            "constant-velocity fit; real motion curves over longer baselines")
        log.warning("moving-object linking: %s", report.refused)
        return [], report

    candidates: List[List[Detection]] = []
    for first, second in itertools.combinations(range(len(per_epoch)), 2):
        gap = times[second] - times[first]
        if gap <= 0:
            continue
        for a in per_epoch[first]:
            for b in per_epoch[second]:
                report.n_pairs_tried += 1
                vx = (b.x - a.x) / gap
                vy = (b.y - a.y) / gap
                speed = math.hypot(vx, vy)
                if speed < low or speed > high:
                    continue
                # A pair alone is not a tracklet: collect whatever else lies
                # on the same track, taking the closest detection per epoch.
                members = [a, b]
                for index, group in enumerate(per_epoch):
                    if index in (first, second):
                        continue
                    dt = times[index] - times[first]
                    px, py = a.x + vx * dt, a.y + vy * dt
                    best, best_distance = None, float(tolerance)
                    for other in group:
                        distance = math.hypot(other.x - px, other.y - py)
                        if distance <= best_distance:
                            best, best_distance = other, distance
                    if best is not None:
                        members.append(best)
                if len(members) >= int(min_points):
                    candidates.append(members)

    tracklets: List[Tracklet] = []
    for members in candidates:
        tracklet = build_tracklet(members, wcs=wcs, pixel_scale=scale)
        residual = tracklet.reduced_rms
        if not np.isfinite(residual):
            residual = tracklet.rms
        if not np.isfinite(residual) or residual > float(max_rms):
            continue
        tracklets.append(tracklet)

    tracklets = _deduplicate(tracklets)
    chance = chance_alignment_rate(counts, float(tolerance),
                                   area if np.isfinite(area) else 1e9, int(min_points))
    report.expected_chance_tracklets = chance
    for tracklet in tracklets:
        tracklet.chance_probability = _per_tracklet_chance(
            tracklet, counts, float(tolerance), area)
        tracklet.score = _score(tracklet, len(per_epoch), float(max_rms))
        if tracklet.chance_probability > 0.1:
            tracklet.add_flag("chance_alignment_plausible")
    tracklets.sort(key=lambda t: -t.score)
    report.n_tracklets = len(tracklets)

    log.info("moving-object linking: %d tracklet(s) from %d detections over %d epochs "
             "(%d pairs tried, ~%.2f expected by chance)",
             len(tracklets), len(detections), len(per_epoch), report.n_pairs_tried, chance)
    return tracklets, report


def _per_tracklet_chance(tracklet: Tracklet, counts: Sequence[int],
                         tolerance: float, area: float) -> float:
    """Probability that *this* tracklet's extra points are coincidence.

    The two detections that defined the velocity are given; what needs
    explaining is the others landing on the prediction.
    """
    if not np.isfinite(area) or area <= 0 or tracklet.n_points < 3:
        return float("nan")
    disc = math.pi * tolerance ** 2
    probability = min(disc / area, 1.0)
    typical = float(np.mean(counts)) if len(counts) else 0.0
    extra = tracklet.n_points - 2
    return float(min(1.0, (typical * probability) ** extra))


def _score(tracklet: Tracklet, n_epochs: int, max_rms: float) -> float:
    """0-1 confidence, from how many epochs agree and how well they fit."""
    completeness = tracklet.n_points / max(n_epochs, 1)
    # The *reduced* residual, so a three-point link cannot look tighter than
    # a five-point one simply by having fewer points to disagree.
    residual = tracklet.reduced_rms
    if not np.isfinite(residual):
        residual = tracklet.rms
    tightness = 1.0 - float(np.clip(residual / max(max_rms, 1e-6), 0.0, 1.0))
    score = 0.55 * completeness + 0.30 * tightness
    if np.isfinite(tracklet.trail_agreement):
        # A trail pointing along the track is independent evidence: it comes
        # from within a single exposure, not from the linking at all.
        score += 0.15 * float(np.clip(tracklet.trail_agreement, 0.0, 1.0))
    else:
        score += 0.075          # neither confirmed nor contradicted
    return float(np.clip(score, 0.0, 1.0))


def _deduplicate(tracklets: Sequence[Tracklet]) -> List[Tracklet]:
    """Keep one tracklet per set of detections.

    Every pair of points on a real track proposes the same tracklet, so a
    four-point mover is found six times.  Preference goes to more points,
    then to a tighter fit; overlapping tracklets that share most of their
    detections are treated as the same object.
    """
    ordered = sorted(tracklets, key=lambda t: (-t.n_points, t.rms))
    kept: List[Tracklet] = []
    used: List[set] = []
    for tracklet in ordered:
        key = {(round(d.x, 3), round(d.y, 3), round(d.time, 8)) for d in tracklet.detections}
        if any(len(key & seen) >= max(2, len(key) - 1) for seen in used):
            continue
        kept.append(tracklet)
        used.append(key)
    return kept


def detections_from_candidates(candidates: Sequence[Any], times: Sequence[float]
                               ) -> List[Detection]:
    """Convert per-epoch transient candidates into linker input.

    ``times`` is indexed by epoch.  Candidates already carrying a time in
    their metadata use it; the rest fall back to the epoch's time, which is
    correct whenever every source in an epoch shares one exposure.
    """
    detections: List[Detection] = []
    for candidate in candidates:
        epoch = int(getattr(candidate, "epoch_index", 0))
        time = candidate.meta.get("time") if hasattr(candidate, "meta") else None
        if time is None:
            time = times[epoch] if 0 <= epoch < len(times) else float(epoch)
        detections.append(Detection(
            x=float(candidate.x), y=float(candidate.y), time=float(time), epoch=epoch,
            flux=float(getattr(candidate, "delta_flux", float("nan"))),
            snr=float(getattr(candidate, "significance", float("nan"))),
            source_id=getattr(candidate, "id", None),
            meta={"candidate": candidate}))
    return detections
