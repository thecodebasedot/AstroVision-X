"""The transient stage: search a multi-epoch series for what changed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.config import TransientConfig
from ..core.logging import get_logger
from ..core.types import SourceCatalog, TransientCandidate
from ..io.image import AstroImage, ImageSeries
from .candidates import (
    build_candidate_light_curves,
    extract_candidates,
    merge_epoch_candidates,
)
from .difference import DifferenceResult, build_template, subtract
from .supernova import assign_verdict, classify_transient, describe

log = get_logger("transient.detector")


class TransientDetector:
    """Difference-image search across every epoch of a series.

    Each epoch is compared against a template built from the *other*
    epochs, so a transient never contaminates the reference used to find
    it.  Candidates are vetted, associated with hosts, given light curves
    across the full series, and finally ranked for follow-up.

    >>> from astrovision.simulate import SkySimulator, SkyConfig
    >>> sim = SkySimulator(SkyConfig(shape=(128, 128), n_stars=15, n_galaxies=3,
    ...                              n_nebulae=0, n_clusters=0, n_lenses=0,
    ...                              n_anomalies=0, seed=2))
    >>> series, _, injected = sim.generate_series(n_epochs=4, n_transients=1)
    >>> len(injected)
    1
    """

    def __init__(self, config: Optional[TransientConfig] = None):
        self.config = config or TransientConfig()
        self.differences: List[DifferenceResult] = []
        self.report: Dict[str, Any] = {}

    def run(self, series: ImageSeries, catalog: Optional[SourceCatalog] = None,
            template: Optional[AstroImage] = None) -> List[TransientCandidate]:
        """Search the whole series; returns ranked, vetted candidates."""
        cfg = self.config
        if not cfg.enabled:
            return []
        if len(series) < 2:
            log.info("transient search needs at least two epochs; got %d", len(series))
            return []

        problems = series.check_alignment()
        for problem in problems:
            log.warning("series consistency: %s", problem)

        self.differences = []
        per_epoch: List[List[TransientCandidate]] = []

        for index, science in enumerate(series):
            # Hold this epoch out of its own template.
            reference = template if template is not None else build_template(
                series, method="trimmed" if len(series) >= 5 else "median", exclude=index)
            try:
                result = subtract(science, reference, cfg.align, cfg.psf_match)
            except Exception as exc:
                log.warning("subtraction failed for epoch %d (%s); skipping", index, exc)
                continue
            self.differences.append(result)
            per_epoch.append(extract_candidates(result, cfg, catalog, epoch_index=index))

        if not per_epoch:
            return []

        candidates = merge_epoch_candidates(per_epoch, radius=max(2.0, cfg.min_area))
        real = [c for c in candidates if "bogus" not in c.flags]

        if real:
            build_candidate_light_curves(real, series, radius=4.0)
        for candidate in candidates:
            classification, confidence, scores = classify_transient(candidate, catalog)
            candidate.classification = classification
            candidate.confidence = confidence
            candidate.meta["class_scores"] = scores
            candidate.verdict = assign_verdict(candidate, cfg.real_bogus_threshold)

        candidates.sort(key=lambda c: (-_verdict_rank(c), -c.real_bogus, -c.significance))
        for index, candidate in enumerate(candidates, start=1):
            candidate.id = index

        quality = [d.diagnostics.get("subtraction_quality", float("nan"))
                   for d in self.differences]
        self.report = {
            "n_epochs": len(series),
            "n_differences": len(self.differences),
            "n_raw_candidates": sum(len(c) for c in per_epoch),
            "n_candidates": len(candidates),
            "n_vetted": len(real),
            "n_multi_epoch": sum(1 for c in candidates if "multi_epoch" in c.flags),
            "median_subtraction_quality": float(np.nanmedian(quality)) if quality else float("nan"),
        }
        log.info("transient search: %d vetted candidates from %d residuals across %d epochs",
                 len(real), self.report["n_raw_candidates"], len(series))
        return candidates

    def run_pair(self, science: AstroImage, reference: AstroImage,
                 catalog: Optional[SourceCatalog] = None) -> List[TransientCandidate]:
        """Search a single science/reference pair."""
        cfg = self.config
        result = subtract(science, reference, cfg.align, cfg.psf_match)
        self.differences = [result]
        candidates = extract_candidates(result, cfg, catalog, epoch_index=0)
        for candidate in candidates:
            classification, confidence, scores = classify_transient(candidate, catalog)
            candidate.classification = classification
            candidate.confidence = confidence
            candidate.meta["class_scores"] = scores
            candidate.verdict = assign_verdict(candidate, cfg.real_bogus_threshold)
        self.report = {
            "n_epochs": 2, "n_differences": 1,
            "n_raw_candidates": len(candidates),
            "n_candidates": len(candidates),
            "n_vetted": sum(1 for c in candidates if "bogus" not in c.flags),
            "median_subtraction_quality": result.diagnostics.get("subtraction_quality"),
        }
        return candidates

    def summary(self, candidates: Sequence[TransientCandidate],
                catalog: Optional[SourceCatalog] = None, limit: int = 5) -> str:
        """Human-readable summary of the top candidates."""
        real = [c for c in candidates if "bogus" not in c.flags]
        if not real:
            return "No transient candidates survived vetting."
        lines = [f"{len(real)} candidate(s) passed vetting:"]
        for candidate in real[:limit]:
            lines.append("  " + describe(candidate, catalog))
        return "\n".join(lines)


_VERDICT_ORDER = {"high_priority": 3, "follow_up_recommended": 2,
                  "worth_a_look": 1, "not_interesting": 0}


def _verdict_rank(candidate: TransientCandidate) -> int:
    return _VERDICT_ORDER.get(candidate.verdict.value, 0)
