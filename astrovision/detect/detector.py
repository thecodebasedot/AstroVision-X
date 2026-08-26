"""The unified detection interface.

Everything downstream consumes a :class:`~astrovision.core.types.SourceCatalog`
and a segmentation map, so classical thresholding and the deep detector are
interchangeable behind :class:`Detector`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.config import DetectionConfig
from ..core.exceptions import ConfigError
from ..core.logging import get_logger
from ..core.registry import Registry
from ..core.types import SourceCatalog
from ..io.image import AstroImage
from .sources import build_segmentation, extract_sources

log = get_logger("detect.detector")

#: Detection backends, addressable by name from the configuration.
DETECTORS: Registry = Registry("detector")


class BaseDetector:
    """Interface every detection backend implements."""

    name = "base"

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    def detect(self, image: AstroImage) -> Tuple[SourceCatalog, np.ndarray]:
        raise NotImplementedError


@DETECTORS.register("threshold")
class ThresholdDetector(BaseDetector):
    """Classical matched-filter thresholding with multi-level deblending.

    This is the workhorse: it is deterministic, needs no training data,
    and produces the segmentation map that photometry and morphology
    both depend on.
    """

    name = "threshold"

    def detect(self, image: AstroImage) -> Tuple[SourceCatalog, np.ndarray]:
        return extract_sources(image, self.config)


@DETECTORS.register("dnn")
class DNNDetector(BaseDetector):
    """Deep anchor-free detector, with a classical segmentation fallback.

    A neural detector gives boxes but no pixel footprints, so the
    segmentation map is still built classically and the two are merged:
    deep detections supply the class prior, thresholding supplies the
    footprints that photometry needs.
    """

    name = "dnn"

    def __init__(self, config: Optional[DetectionConfig] = None):
        super().__init__(config)
        from .dnn import DeepDetector
        self.model = DeepDetector(score_threshold=self.config.dnn_score_threshold,
                                  nms_iou=self.config.dnn_nms_iou)
        if self.config.model_path:
            self.model.load(self.config.model_path)
        elif not self.model.available:
            raise ConfigError(
                "detection.backend='dnn' needs PyTorch; install "
                "'astrovision-x[deep]' or use backend='threshold'")

    def detect(self, image: AstroImage) -> Tuple[SourceCatalog, np.ndarray]:
        if self.model.model is None:
            log.warning("no detector weights loaded; using classical detection only")
            return extract_sources(image, self.config)

        catalog, segmentation = extract_sources(image, self.config)
        deep = self.model.detect(image)
        merged = 0
        for prediction in deep:
            match = catalog.match(prediction.x, prediction.y, radius=4.0)
            if match is not None:
                match.object_class = prediction.object_class
                match.class_confidence = prediction.class_confidence
                match.meta["dnn_score"] = prediction.class_confidence
                merged += 1
        catalog.meta["dnn_detections"] = len(deep)
        catalog.meta["dnn_merged"] = merged
        log.info("merged %d/%d deep detections into the classical catalog",
                 merged, len(deep))
        return catalog, segmentation


class Detector:
    """Front door for detection; dispatches on ``config.backend``.

    >>> from astrovision.simulate import quick_field
    >>> from astrovision.preprocess import Preprocessor
    >>> image, _ = quick_field((128, 128))
    >>> catalog, segmentation = Detector().detect(Preprocessor().run(image))
    >>> len(catalog) > 0
    True
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.backend: BaseDetector = DETECTORS.create(self.config.backend, self.config)

    def detect(self, image: AstroImage) -> Tuple[SourceCatalog, np.ndarray]:
        """Detect sources; returns ``(catalog, segmentation_map)``."""
        return self.backend.detect(image)

    def segmentation_only(self, image: AstroImage) -> np.ndarray:
        """Build just the segmentation map (used by transient vetting)."""
        segmentation, _, _ = build_segmentation(
            image.subtracted(), self.config, image.rms_map(), None, image.mask)
        return segmentation
