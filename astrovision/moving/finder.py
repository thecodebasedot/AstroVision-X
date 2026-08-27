"""The moving-object stage: link difference-image detections into tracklets.

This runs after the transient search and consumes its per-epoch candidates,
because those are exactly the right input: a difference image contains what
changed, and a solar-system object changes position in every frame.

The stage's second job is to *take movers back out* of the transient list.
Left in, one asteroid crossing five epochs is reported as five marginal
one-epoch transient candidates -- five entries in a follow-up queue, none of
them real, each demanding an astronomer's attention.  That is the single
most common way an untended transient pipeline wastes time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.config import MovingObjectConfig
from ..core.logging import get_logger
from ..core.types import TransientCandidate, Verdict
from ..io.image import AstroImage, ImageSeries
from .linking import LinkingReport, detections_from_candidates, link_tracklets
from .tracklet import Tracklet
from .trail import direction_agreement, expected_trail_length, measure_trail

log = get_logger("moving.finder")


@dataclass
class MovingObjectResult:
    """Tracklets found, and the candidates they explain."""

    tracklets: List[Tracklet] = field(default_factory=list)
    linking: Optional[LinkingReport] = None
    claimed_candidate_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_tracklets": len(self.tracklets),
            "tracklets": [t.to_dict() for t in self.tracklets],
            "linking": self.linking.to_dict() if self.linking else None,
            "n_candidates_explained": len(self.claimed_candidate_ids),
        }


class MovingObjectFinder:
    """Finds solar-system objects in a multi-epoch series.

    >>> from astrovision.simulate import SkyConfig, SkySimulator
    >>> from astrovision.preprocess import Preprocessor
    >>> from astrovision.io.image import ImageSeries
    >>> simulator = SkySimulator(SkyConfig(shape=(160, 160), n_stars=20,
    ...     n_galaxies=3, n_nebulae=0, n_clusters=0, n_lenses=0,
    ...     n_anomalies=0, seed=4))
    >>> series, _, truth = simulator.generate_series(n_epochs=4, cadence=0.01,
    ...                                              n_transients=0, n_movers=1)
    >>> finder = MovingObjectFinder()
    >>> isinstance(finder.config.max_rate_arcsec_per_hour, float)
    True
    """

    def __init__(self, config: Optional[MovingObjectConfig] = None):
        self.config = config or MovingObjectConfig()
        self.report: Dict[str, Any] = {}

    def run(self, series: ImageSeries,
            per_epoch_candidates: Sequence[Sequence[TransientCandidate]],
            differences: Optional[Sequence[Any]] = None) -> MovingObjectResult:
        """Link candidates into tracklets; returns the result and fills ``report``."""
        cfg = self.config
        result = MovingObjectResult()
        if not cfg.enabled or len(series) < cfg.min_points:
            self.report = {"skipped": f"needs at least {cfg.min_points} epochs"}
            return result

        times = [float(image.mjd) if image.mjd is not None else float(index)
                 for index, image in enumerate(series)]
        flat = [candidate for group in per_epoch_candidates for candidate in group]
        detections = detections_from_candidates(flat, times)
        reference = series.reference
        tracklets, linking = link_tracklets(
            detections,
            min_rate=cfg.min_rate_arcsec_per_hour,
            max_rate=cfg.max_rate_arcsec_per_hour,
            tolerance=cfg.tolerance_pixels,
            min_points=cfg.min_points,
            max_rms=cfg.max_rms_pixels,
            wcs=reference.wcs,
            pixel_scale=cfg.pixel_scale,
            field_shape=reference.shape,
            max_arc_days=cfg.max_arc_days)
        result.linking = linking

        if cfg.measure_trails and differences is not None:
            self._measure_trails(tracklets, series, differences)

        # Re-score after trails, since a trail is evidence the linker did not
        # have, then apply the acceptance cut.
        from .linking import _score

        kept: List[Tracklet] = []
        for tracklet in tracklets:
            tracklet.score = _score(tracklet, len(series), cfg.max_rms_pixels)
            if tracklet.score >= cfg.min_score:
                kept.append(tracklet)
            else:
                tracklet.add_flag("below_score_threshold")
        result.tracklets = kept

        claimed = self._claim_candidates(kept)
        result.claimed_candidate_ids = claimed
        self.report = {
            **result.to_dict(),
            "n_epochs": len(series),
            "rate_range_arcsec_per_hour": [cfg.min_rate_arcsec_per_hour,
                                           cfg.max_rate_arcsec_per_hour],
        }
        # A tracklet list is not a discovery list.  Anything here needs an
        # orbit before it is an object, and a check against the Minor Planet
        # Center before it is a *new* one.
        log.info("moving objects: %d tracklet(s) explaining %d transient candidates; "
                 "all require an orbit and an MPC check before they are objects",
                 len(kept), len(claimed))
        return result

    def _measure_trails(self, tracklets: Sequence[Tracklet], series: ImageSeries,
                        differences: Sequence[Any]) -> None:
        """Compare each detection's elongation with the tracklet's direction."""
        cfg = self.config
        for tracklet in tracklets:
            agreements, lengths = [], []
            for detection in tracklet.detections:
                image = self._epoch_image(series, differences, detection.epoch)
                if image is None:
                    continue
                psf = image.meta.get("psf_model")
                psf_fwhm = float(psf.fwhm) if psf is not None else float("nan")
                if not np.isfinite(psf_fwhm):
                    continue
                half = int(max(8, 3 * psf_fwhm))
                cutout = image.cutout(detection.x, detection.y, size=2 * half + 1)
                if cutout.size == 0:
                    continue
                noise = float(np.nanmedian(image.rms_map()))
                trail = measure_trail(cutout, psf_fwhm, noise=noise,
                                      min_excess=cfg.min_trail_excess_pixels)
                detection.meta["trail"] = trail.to_dict()
                if trail.trailed:
                    agreements.append(direction_agreement(trail.angle,
                                                          tracklet.heading_deg))
                    lengths.append(trail.excess)
            if agreements:
                tracklet.trail_agreement = float(np.median(agreements))
                tracklet.meta["trail_excess_pixels"] = float(np.median(lengths))
                tracklet.meta["n_trailed"] = len(agreements)
                expected = expected_trail_length(
                    tracklet.rate_arcsec_per_hour,
                    float(series.reference.exposure_time or 0.0),
                    float(series.reference.wcs.pixel_scale)
                    if series.reference.wcs is not None else float("nan"))
                tracklet.meta["expected_trail_pixels"] = float(expected)
                # The trail length the rate predicts and the one measured are
                # independent numbers; agreeing is a real consistency check.
                if np.isfinite(expected) and expected > 0:
                    ratio = float(np.median(lengths)) / expected
                    tracklet.meta["trail_length_ratio"] = ratio
                    if 0.5 <= ratio <= 2.0:
                        tracklet.add_flag("trail_consistent_with_rate")

    @staticmethod
    def _epoch_image(series: ImageSeries, differences: Sequence[Any],
                     epoch: int) -> Optional[AstroImage]:
        """The difference image for an epoch, falling back to the science frame.

        The difference is preferred: a mover on top of a galaxy has its trail
        measured against a clean background there, where on the science frame
        the host's light dominates the second moments entirely.
        """
        for candidate in differences or []:
            if int(getattr(candidate, "epoch_index", -1)) == int(epoch):
                image = getattr(candidate, "difference", None)
                if isinstance(image, AstroImage):
                    return image
        if 0 <= epoch < len(series):
            return series[epoch]
        return None                                            # pragma: no cover

    @staticmethod
    def _claim_candidates(tracklets: Sequence[Tracklet]) -> List[int]:
        """Mark the transient candidates a tracklet accounts for.

        The candidates are flagged and given the ``mover`` classification
        rather than deleted.  Deleting them would hide the evidence: what the
        tracklet asserts is an *interpretation* of those detections, and an
        astronomer disagreeing with it needs to see what was interpreted.
        """
        claimed: List[int] = []
        seen: set = set()
        for tracklet in tracklets:
            for detection in tracklet.detections:
                candidate = detection.meta.get("candidate")
                if candidate is None:
                    continue
                candidate.classification = "moving_object"
                candidate.confidence = float(tracklet.score)
                candidate.add_flag("moving_object")
                candidate.verdict = Verdict.NOT_INTERESTING
                candidate.meta["tracklet"] = {
                    "rate_arcsec_per_hour": tracklet.rate_arcsec_per_hour,
                    "heading_deg": tracklet.heading_deg,
                    "n_points": tracklet.n_points,
                    "rate_class": tracklet.describe_rate(),
                }
                # Two tracklets can legitimately share a detection -- a
                # crossing, or one weak link overlapping a strong one -- so
                # the ids are deduplicated rather than counted twice.
                if candidate.id is not None and int(candidate.id) not in seen:
                    seen.add(int(candidate.id))
                    claimed.append(int(candidate.id))
        return claimed


def summarise_tracklet(tracklet: Tracklet) -> str:
    """One human-readable line, with the caveat that belongs on it."""
    rate = tracklet.rate_arcsec_per_hour
    rate_text = (f"{rate:.1f} arcsec/hour" if np.isfinite(rate)
                 else f"{np.hypot(tracklet.vx, tracklet.vy):.1f} px/day")
    parts = [f"{tracklet.n_points} detections on one track over "
             f"{tracklet.arc_days * 24:.1f} hours",
             f"moving at {rate_text} ({tracklet.describe_rate()})",
             f"track residual {tracklet.rms:.2f} px"]
    if np.isfinite(tracklet.trail_agreement):
        parts.append(f"trail direction agrees {tracklet.trail_agreement:.2f}")
    if np.isfinite(tracklet.chance_probability):
        parts.append(f"chance alignment probability {tracklet.chance_probability:.1e}")
    return "; ".join(parts)
