"""Transient detection: difference imaging, vetting and candidate assessment."""

from .candidates import (
    associate_hosts,
    build_candidate_light_curves,
    extract_candidates,
    merge_epoch_candidates,
)
from .detector import TransientDetector
from .difference import DifferenceResult, build_template, flux_scale_factor, subtract
from .realbogus import (
    RB_FEATURES,
    classify_artifact,
    real_bogus_score,
    stamp_features,
)
from .supernova import (
    TRANSIENT_CLASSES,
    assign_verdict,
    classify_transient,
    describe,
    light_curve_shape,
)

__all__ = [
    "TransientDetector",
    "subtract", "DifferenceResult", "build_template", "flux_scale_factor",
    "extract_candidates", "associate_hosts", "merge_epoch_candidates",
    "build_candidate_light_curves",
    "stamp_features", "real_bogus_score", "classify_artifact", "RB_FEATURES",
    "classify_transient", "assign_verdict", "describe", "light_curve_shape",
    "TRANSIENT_CLASSES",
]
