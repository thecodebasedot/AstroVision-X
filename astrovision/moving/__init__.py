"""Solar-system objects: linking detections that move between exposures.

A difference image finds an asteroid as readily as a supernova.  What tells
them apart is that one of them is in a different place every time -- which
means an asteroid arrives at the transient stage disguised as several
unrelated single-epoch candidates, and stays that way unless something links
them.

The two pieces of evidence here are deliberately independent.  Linking works
*across* exposures: several detections on one straight track.  A trail works
*within* one exposure: the object moved while the shutter was open, so it left
a streak rather than a point.  Either alone can be a coincidence; together
they rarely are.
"""

from .finder import MovingObjectFinder, MovingObjectResult, summarise_tracklet
from .linking import (
    LinkingReport,
    chance_alignment_rate,
    detections_from_candidates,
    link_tracklets,
)
from .trail import (
    TrailMeasurement,
    direction_agreement,
    expected_trail_length,
    field_psf_elongation,
    measure_trail,
    second_moments,
)
from .tracklet import Detection, Tracklet, build_tracklet, fit_linear_motion

__all__ = [
    "MovingObjectFinder", "MovingObjectResult", "summarise_tracklet",
    "Detection", "Tracklet", "build_tracklet", "fit_linear_motion",
    "link_tracklets", "chance_alignment_rate", "detections_from_candidates",
    "LinkingReport",
    "measure_trail", "TrailMeasurement", "second_moments",
    "expected_trail_length", "direction_agreement", "field_psf_elongation",
]
