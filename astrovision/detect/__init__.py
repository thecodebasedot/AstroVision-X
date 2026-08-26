"""Object detection: thresholding, deblending, and deep detectors."""

from .deblend import deblend_all, deblend_segment
from .detector import DETECTORS, BaseDetector, Detector, DNNDetector, ThresholdDetector
from .dnn import DETECTOR_CLASSES, DeepDetector, decode_heatmap, gaussian_target
from .labeling import binary_dilate, find_objects, label, label_sizes, remove_small
from .nms import (
    array_to_boxes,
    average_precision,
    boxes_to_array,
    match_detections,
    nms,
    soft_nms,
)
from .sources import build_segmentation, detection_threshold, extract_sources, measure_segment

__all__ = [
    "Detector", "BaseDetector", "ThresholdDetector", "DNNDetector", "DETECTORS",
    "extract_sources", "build_segmentation", "detection_threshold", "measure_segment",
    "deblend_segment", "deblend_all",
    "label", "find_objects", "label_sizes", "remove_small", "binary_dilate",
    "nms", "soft_nms", "match_detections", "average_precision",
    "boxes_to_array", "array_to_boxes",
    "DeepDetector", "DETECTOR_CLASSES", "decode_heatmap", "gaussian_target",
]
