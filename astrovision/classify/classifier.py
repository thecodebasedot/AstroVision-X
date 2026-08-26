"""The classification stage: assign an object class to every source."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.config import ClassificationConfig
from ..core.exceptions import ConfigError
from ..core.logging import get_logger
from ..core.types import ObjectClass, SourceCatalog
from ..io.image import AstroImage
from .colours import annotate_catalog
from .rules import classify_source, field_reference

log = get_logger("classify.classifier")


class Classifier:
    """Assigns object classes, by rules, by a trained CNN, or by both.

    The ``hybrid`` backend is the default and the one to trust: the rules
    give a physically-grounded answer everywhere, and where a trained model
    is confident it overrides them.  A model alone silently fails on data
    unlike its training set; rules alone cannot use pixel-level structure.

    >>> from astrovision.simulate import quick_field
    >>> from astrovision.preprocess import Preprocessor
    >>> from astrovision.detect import Detector
    >>> from astrovision.photometry import Photometer
    >>> image, _ = quick_field((160, 160))
    >>> clean = Preprocessor().run(image)
    >>> catalog, segmentation = Detector().detect(clean)
    >>> _ = Photometer().run(clean, catalog, segmentation)
    >>> _ = Classifier().run(clean, catalog)
    >>> set(catalog.class_counts()) <= {c.value for c in ObjectClass}
    True
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.report: Dict[str, Any] = {}
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        from ..ml.cnn import StampClassifier

        model = StampClassifier()
        if self.config.model_path:
            model.load(self.config.model_path)
            self._model = model
            return model
        if self.config.backend == "cnn":
            raise ConfigError(
                "classification.backend='cnn' needs classification.model_path "
                "to point at trained weights")
        return None

    def run(self, image: AstroImage, catalog: SourceCatalog) -> SourceCatalog:
        """Classify every source in place; returns the catalog."""
        cfg = self.config
        if len(catalog) == 0:
            return catalog

        psf = image.meta.get("psf_model")
        psf_fwhm = float(psf.fwhm) if psf is not None else float(
            np.nanmedian([s.morphology.fwhm for s in catalog]) or 3.0)
        psf_r90 = psf.encircled_radius(0.9) if psf is not None else None
        pixel_scale = image.pixel_scale if image.wcs is not None else 1.0

        reference = field_reference(catalog, psf_fwhm)

        def apply_rules(colour_weight: float) -> int:
            count = 0
            for source in catalog:
                object_class, confidence, scores = classify_source(
                    source, psf_fwhm, psf_r90, cfg.star_galaxy_threshold, pixel_scale,
                    reference, colour_weight)
                source.object_class = object_class
                source.class_confidence = confidence
                source.class_scores = scores
                if confidence < cfg.min_confidence:
                    source.add_flag("uncertain_class")
                count += 1
            return count

        n_rules = 0
        locus = None
        if cfg.backend in ("rules", "hybrid"):
            # Pass one is morphology alone.  It is not wasted work: its
            # `shape_stellarity` is what tells the locus fit which sources are
            # point-like, and a locus seeded by a colour-informed answer would
            # be confirming its own conclusion.
            n_rules = apply_rules(0.0)
            if cfg.use_colours and any(source.bands for source in catalog):
                locus = annotate_catalog(catalog)
                if locus is not None and locus.information_weight > 0:
                    apply_rules(float(cfg.colour_weight) * locus.information_weight)

        n_model = 0
        if cfg.backend in ("cnn", "ml", "hybrid"):
            model = self._load_model()
            if model is not None and model.model is not None:
                before = {s.id: (s.object_class, s.class_confidence) for s in catalog}
                model.annotate(catalog, image, min_confidence=0.0,
                               store_embedding=cfg.use_embeddings)
                for source in catalog:
                    rule_class, rule_confidence = before.get(
                        source.id, (ObjectClass.UNKNOWN, 0.0))
                    model_confidence = source.class_confidence
                    source.meta["rule_class"] = rule_class.value
                    # Keep the rule answer unless the model is clearly more
                    # confident; a marginally better score is not evidence.
                    if cfg.backend == "hybrid" and model_confidence < rule_confidence + 0.15:
                        source.object_class = rule_class
                        source.class_confidence = rule_confidence
                    else:
                        n_model += 1
            elif cfg.backend != "hybrid":
                log.warning("no trained classifier available; using rules only")

        counts = catalog.class_counts()
        self.report = {
            "backend": cfg.backend,
            "psf_fwhm": psf_fwhm,
            "field_reference": reference,
            "n_rule_classified": n_rules,
            "stellar_locus": locus.to_dict() if locus is not None else None,
            "n_model_overrides": n_model,
            "class_counts": counts,
            "n_uncertain": sum(1 for s in catalog if "uncertain_class" in s.flags),
        }
        log.info("classified %d sources (%s): %s", len(catalog), cfg.backend,
                 ", ".join(f"{k}={v}" for k, v in counts.items()))
        return catalog
