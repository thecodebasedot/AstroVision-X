"""Segmentation: footprint refinement, semantic labels, galaxy decomposition."""

from .classical import (
    find_peaks,
    isophotal_contours,
    segment_object,
    watershed,
    watershed_split,
)
from .galaxy_parts import (
    COMPONENTS,
    GalaxyComponents,
    annulus_masks,
    component_profile,
    decompose,
    ellipse_from_moments,
)
from .segmenter import Segmenter
from .unet import SEGMENT_CLASSES, UNetSegmenter, labels_from_segmentation

__all__ = [
    "Segmenter",
    "watershed", "watershed_split", "find_peaks", "isophotal_contours", "segment_object",
    "decompose", "GalaxyComponents", "COMPONENTS", "component_profile",
    "ellipse_from_moments", "annulus_masks",
    "UNetSegmenter", "SEGMENT_CLASSES", "labels_from_segmentation",
]
