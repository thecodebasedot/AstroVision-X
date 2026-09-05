"""This package's morphology against statmorph, on the same segments.

statmorph (Rodriguez-Gomez et al. 2019) is the reference implementation of
the non-parametric morphology statistics -- Gini, M20, concentration,
asymmetry, smoothness -- and of single-Sérsic fits, used across the survey
literature. It is run here on the same background-subtracted pixels, the
same segmentation map and the same PSF the morphology stage saw, and each
statistic is compared source by source. The comparison is of definitions
as much as implementations: statmorph measures within its own Petrosian
ellipse and Gini segmentation, this package within a Petrosian circle on
the detection footprint, and where those choices differ the numbers do,
so the report gives the median offset, the scatter and the rank
correlation rather than one verdict.

Only Sérsic index has a truth to compare both against: the simulator
draws one per galaxy, and the field's report scores each code's error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger

log = get_logger("validation.morphology_benchmark")

#: (our attribute, statmorph attribute)
METRICS = [
    ("gini", "gini"), ("m20", "m20"), ("concentration", "concentration"),
    ("asymmetry", "asymmetry"), ("smoothness", "smoothness"),
    ("sersic_index", "sersic_n"), ("effective_radius", "sersic_rhalf"),
]


def statmorph_available() -> bool:
    return try_import("statmorph") is not None


def run_statmorph(image, segmentation: np.ndarray, catalog, gain: float = 1.0,
                  psf: Optional[np.ndarray] = None, cutout_extent: float = 2.5,
                  min_area: int = 20) -> Dict[int, Dict[str, Any]]:
    """statmorph's measurements keyed by this package's source id.

    ``image`` is the preprocessed :class:`AstroImage` (background-subtracted
    pixels and a noise map are taken from it); ``segmentation`` is the
    detection stage's label map, whose labels match ``source.segment_label``.
    """
    statmorph = try_import("statmorph")
    if statmorph is None:
        raise ImportError("statmorph is not installed; pip install statmorph")
    data = np.asarray(image.subtracted(), dtype=float)
    weight = np.asarray(image.rms_map(), dtype=float)
    labels = np.asarray(segmentation, dtype=np.int32)
    started = time.time()
    results = statmorph.source_morphology(
        data, labels, weightmap=weight, gain=float(gain), psf=psf,
        cutout_extent=float(cutout_extent), min_cutout_size=48, verbose=False)
    by_label = {int(r.label): r for r in results}
    out: Dict[int, Dict[str, Any]] = {}
    for source in catalog:
        r = by_label.get(int(source.segment_label))
        if r is None:
            continue
        out[int(source.id)] = {
            theirs: float(getattr(r, theirs, np.nan)) for _, theirs in METRICS
        }
        out[int(source.id)].update({"flag": int(r.flag), "flag_sersic": int(r.flag_sersic),
                                    "rpetro_circ": float(r.rpetro_circ),
                                    "rhalf_circ": float(r.rhalf_circ)})
    log.info("statmorph measured %d of %d sources in %.1fs", len(out), len(catalog),
             time.time() - started)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


@dataclass
class MetricComparison:
    metric: str
    n: int = 0
    median_difference: float = float("nan")      # ours - theirs
    scatter: float = float("nan")                # 1.4826 MAD of the difference
    rank_correlation: float = float("nan")
    ours_vs_truth: Optional[float] = None        # median |ours - truth| where a truth exists
    theirs_vs_truth: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class MorphologyBenchmark:
    n_sources: int = 0
    n_compared: int = 0
    metrics: List[MetricComparison] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_sources": self.n_sources, "n_compared": self.n_compared,
                "metrics": [m.to_dict() for m in self.metrics], "notes": list(self.notes)}

    def summary(self) -> str:
        lines = [f"statmorph comparison on {self.n_compared} of {self.n_sources} sources"]
        for m in self.metrics:
            line = (f"  {m.metric:<18} n {m.n:>3}  ours-theirs median {m.median_difference:+.3f} "
                    f"scatter {m.scatter:.3f}  rank corr {m.rank_correlation:.2f}")
            if m.ours_vs_truth is not None:
                line += f"  |err| vs truth ours {m.ours_vs_truth:.3f} theirs {m.theirs_vs_truth:.3f}"
            lines.append(line)
        return "\n".join(lines)


def compare_morphology(catalog, theirs: Dict[int, Dict[str, Any]],
                       truth: Optional[Sequence[Any]] = None, match_radius: float = 2.0,
                       good_flags_only: bool = True) -> MorphologyBenchmark:
    """Source-by-source comparison of every metric both codes measured."""
    report = MorphologyBenchmark(n_sources=len(catalog))
    truth_index = None
    if truth is not None:
        galaxies = [o for o in truth if getattr(o, "kind", "") == "galaxy"]
        if galaxies:
            truth_index = (np.array([o.x for o in galaxies]), np.array([o.y for o in galaxies]),
                           galaxies)
    compared = 0
    for ours_name, theirs_name in METRICS:
        pairs, ours_err, theirs_err = [], [], []
        for source in catalog:
            row = theirs.get(int(source.id))
            if row is None:
                continue
            if good_flags_only and row.get("flag", 0) > 1:
                continue
            a = float(getattr(source.morphology, ours_name, np.nan))
            b = float(row.get(theirs_name, np.nan))
            if ours_name in ("sersic_index", "effective_radius") and row.get("flag_sersic", 0) > 0:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            pairs.append((a, b))
            if truth_index is not None and ours_name == "sersic_index":
                tx, ty, galaxies = truth_index
                d = np.hypot(tx - source.x, ty - source.y)
                k = int(np.argmin(d))
                if d[k] <= match_radius and getattr(galaxies[k], "sersic_n", 0) > 0:
                    ours_err.append(abs(a - galaxies[k].sersic_n))
                    theirs_err.append(abs(b - galaxies[k].sersic_n))
        comparison = MetricComparison(metric=ours_name, n=len(pairs))
        if pairs:
            arr = np.array(pairs)
            diff = arr[:, 0] - arr[:, 1]
            comparison.median_difference = float(np.median(diff))
            comparison.scatter = float(1.4826 * np.median(np.abs(diff - np.median(diff))))
            comparison.rank_correlation = _spearman(arr[:, 0], arr[:, 1])
            compared = max(compared, len(pairs))
        if ours_err:
            comparison.ours_vs_truth = float(np.median(ours_err))
            comparison.theirs_vs_truth = float(np.median(theirs_err))
        report.metrics.append(comparison)
    report.n_compared = compared
    if good_flags_only:
        flagged = sum(1 for row in theirs.values() if row.get("flag", 0) > 1)
        if flagged:
            report.notes.append(f"{flagged} sources statmorph flagged as unreliable were left out")
    return report


def benchmark_morphology(image, catalog, segmentation, truth=None, gain: float = 1.0
                         ) -> MorphologyBenchmark:
    """Run statmorph on the same inputs and compare."""
    psf_model = image.meta.get("psf_model")
    psf = None if psf_model is None else np.asarray(psf_model.as_kernel(), dtype=float)
    theirs = run_statmorph(image, segmentation, catalog, gain=gain, psf=psf)
    report = compare_morphology(catalog, theirs, truth=truth)
    log.info(report.summary())
    return report


__all__ = ["statmorph_available", "run_statmorph", "compare_morphology",
           "benchmark_morphology", "MorphologyBenchmark", "MetricComparison", "METRICS"]
