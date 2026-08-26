"""The segmentation stage: refine footprints and decompose extended objects."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.config import SegmentationConfig
from ..core.logging import get_logger
from ..core.types import ObjectClass, Source, SourceCatalog
from ..io.image import AstroImage
from .classical import watershed_split
from .galaxy_parts import GalaxyComponents, decompose

log = get_logger("segment.segmenter")


class Segmenter:
    """Refines the detection segmentation and decomposes extended sources.

    Detection gives *where* objects are; segmentation decides which pixels
    belong to which structure inside them.  For anything resolved, the
    galaxy decomposition attaches nucleus/bulge/disc measurements that the
    morphology, classification and lensing stages all consume.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self._unet = None

    # -- optional deep backend --------------------------------------------
    def _load_unet(self):
        if self._unet is not None:
            return self._unet
        from .unet import UNetSegmenter
        segmenter = UNetSegmenter()
        if self.config.model_path:
            segmenter.load(self.config.model_path)
        elif not segmenter.available:
            log.warning("segmentation.backend='unet' needs PyTorch; "
                        "falling back to classical segmentation")
            return None
        self._unet = segmenter
        return segmenter

    # -- main entry point --------------------------------------------------
    def run(self, image: AstroImage, catalog: SourceCatalog,
            segmentation: np.ndarray) -> Dict[int, GalaxyComponents]:
        """Attach per-source component structure; returns it keyed by source id."""
        if not self.config.enabled:
            return {}

        data = image.subtracted()
        semantic: Optional[np.ndarray] = None
        if self.config.backend == "unet":
            unet = self._load_unet()
            if unet is not None and unet.model is not None:
                semantic = unet.predict(data)
                image.meta["semantic_segmentation"] = semantic

        components: Dict[int, GalaxyComponents] = {}
        n_split = 0
        for source in catalog:
            footprint = segmentation == source.segment_label
            if not footprint.any():
                continue
            rows, cols = source.bbox.slices(data.shape, pad=4)
            cut = data[rows, cols]
            cut_mask = footprint[rows, cols]

            if self.config.watershed and cut_mask.sum() >= 20:
                split = watershed_split(cut, cut_mask, smooth=1.0)
                if int(split.max()) > 1:
                    source.add_flag("blended")
                    n_split += 1
                    source.meta["watershed_components"] = int(split.max())

            if self.config.decompose_galaxies and _is_extended(source):
                parts = decompose(cut, cut_mask, self.config.component_levels)
                components[source.id] = parts
                source.meta["components"] = parts.to_dict()
                source.morphology.effective_radius = float(
                    parts.radii.get("disc", source.morphology.semi_major))

            if semantic is not None:
                source.meta["semantic_classes"] = _semantic_summary(
                    semantic[rows, cols], cut_mask)

        log.info("segmentation: %d decomposed, %d flagged as blended",
                 len(components), n_split)
        return components

    def cutouts(self, image: AstroImage, catalog: SourceCatalog,
                size: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Uniform postage stamps for every source, for model input."""
        size = int(size or self.config.cutout_size)
        return {source.id: image.cutout(source.x, source.y, size, subtract_background=True)
                for source in catalog}


def _is_extended(source: Source) -> bool:
    """Whether a source is resolved enough for component decomposition."""
    if source.object_class in (ObjectClass.GALAXY, ObjectClass.NEBULA,
                               ObjectClass.STAR_CLUSTER, ObjectClass.LENS_CANDIDATE):
        return True
    if source.object_class == ObjectClass.STAR:
        return False
    # Before classification runs, fall back to size: anything comfortably
    # larger than a point source is worth decomposing.
    return source.morphology.area_pixels >= 25 and source.morphology.semi_major > 2.0


def _semantic_summary(labels: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Fraction of a footprint assigned to each semantic class."""
    from .unet import SEGMENT_CLASSES
    values = labels[mask]
    if values.size == 0:
        return {}
    counts = np.bincount(values.ravel(), minlength=len(SEGMENT_CLASSES))
    total = float(counts.sum())
    return {SEGMENT_CLASSES[i]: float(counts[i] / total)
            for i in range(min(len(counts), len(SEGMENT_CLASSES))) if counts[i] > 0}
