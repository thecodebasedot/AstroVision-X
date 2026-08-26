"""Novelty discovery.

The point of this stage is *not* to find objects the classifier is unsure
about -- it is to find objects that do not resemble anything in the field
at all.  Several independent detectors are combined, because each has a
different blind spot: an isolation forest works on raw feature geometry, an
autoencoder on whether a low-dimensional model can reproduce the object,
and a k-nearest-neighbour distance on whether the object has any analogues.

Nothing here declares a discovery.  A high score means "a human should look
at this", and the explanation says why.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.config import AnomalyConfig
from ..core.logging import get_logger
from ..core.numeric import logistic, rescale
from ..core.types import AnomalyRecord, ObjectClass, Source, SourceCatalog
from ..ml.autoencoder import DeepAutoencoder, LinearAutoencoder
from ..ml.features import catalog_features, combine_features
from ..ml.isolation_forest import IsolationForest
from ..ml.metric import SimilaritySearch
from ..ml.scaler import RobustScaler

log = get_logger("anomaly.engine")


class AnomalyEngine:
    """Ensemble novelty detector over a source catalog.

    >>> from astrovision.simulate import quick_field
    >>> from astrovision.preprocess import Preprocessor
    >>> from astrovision.detect import Detector
    >>> image, _ = quick_field((160, 160))
    >>> clean = Preprocessor().run(image)
    >>> catalog, _ = Detector().detect(clean)
    >>> records = AnomalyEngine().run(catalog)
    >>> all(0.0 <= r.score <= 1.0 for r in records)
    True
    """

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.models_: Dict[str, Any] = {}
        self.feature_names_: List[str] = []
        self.report: Dict[str, Any] = {}

    def run(self, catalog: SourceCatalog, embeddings: Optional[np.ndarray] = None
            ) -> List[AnomalyRecord]:
        """Score every source and return the ranked anomaly records."""
        cfg = self.config
        if not cfg.enabled or len(catalog) < 8:
            if len(catalog) and len(catalog) < 8:
                log.info("only %d sources: too few for a meaningful novelty search",
                         len(catalog))
            return []

        tabular, names = catalog_features(catalog)
        matrix = combine_features(tabular, embeddings)
        self.feature_names_ = list(names)

        scaler = RobustScaler().fit(matrix)
        Z = scaler.transform(matrix)
        self.models_["scaler"] = scaler

        contributions: Dict[str, np.ndarray] = {}
        for method in cfg.methods:
            try:
                scores = self._score_method(method, Z, matrix, catalog)
            except Exception as exc:      # one detector failing must not sink the stage
                log.warning("anomaly method '%s' failed (%s); skipping", method, exc)
                continue
            if scores is not None and np.isfinite(scores).any():
                contributions[method] = _normalise(scores)

        if not contributions:
            log.warning("no anomaly detector produced usable scores")
            return []

        combined = np.mean(np.vstack(list(contributions.values())), axis=0)
        # Rank-normalise so the final score reads as "more unusual than this
        # fraction of the field", which is what a reader wants to know.
        ranks = np.argsort(np.argsort(combined))
        final = ranks / max(len(combined) - 1, 1)

        records: List[AnomalyRecord] = []
        order = np.argsort(final)[::-1]
        search = None
        if embeddings is not None and len(embeddings) == len(catalog):
            search = SimilaritySearch().fit(embeddings, [s.id for s in catalog])

        for rank, index in enumerate(order[:max(1, cfg.top_k)], start=1):
            source = catalog[int(index)]
            per_method = {m: float(v[index]) for m, v in contributions.items()}
            neighbours: List[int] = []
            if search is not None:
                ids, _ = search.query(embeddings[index], k=4, exclude_self=True)
                neighbours = [i for i in ids if i != source.id][:3]
            records.append(AnomalyRecord(
                source_id=source.id,
                score=float(final[index]),
                rank=rank,
                novelty_type=_novelty_type(source, per_method),
                contributions=per_method,
                nearest_neighbours=neighbours,
                explanation=explain(source, per_method, tabular[index], names),
            ))

        for index, source in enumerate(catalog):
            source.anomaly_score = float(final[index])

        self.report = {
            "n_scored": len(catalog),
            "methods": list(contributions),
            "n_flagged": int((final >= 1.0 - cfg.contamination).sum()),
            "feature_dimension": int(matrix.shape[1]),
        }
        log.info("novelty search over %d sources using %s: %d in the top %d%%",
                 len(catalog), ", ".join(contributions), self.report["n_flagged"],
                 int(100 * cfg.contamination))
        return records

    def _score_method(self, method: str, Z: np.ndarray, raw: np.ndarray,
                      catalog: SourceCatalog) -> Optional[np.ndarray]:
        cfg = self.config
        method = str(method).lower()
        if method == "isolation_forest":
            model = IsolationForest(cfg.n_estimators, random_state=cfg.random_state,
                                    contamination=cfg.contamination)
            self.models_[method] = model
            return model.fit_score(Z)
        if method == "autoencoder":
            latent = max(2, min(cfg.autoencoder_latent, Z.shape[1] - 1, len(Z) - 1))
            deep = DeepAutoencoder(latent_dim=latent, epochs=cfg.autoencoder_epochs,
                                   random_state=cfg.random_state)
            if deep.available and len(Z) >= 32:
                self.models_[method] = deep
                return deep.fit(Z).score(Z)
            model = LinearAutoencoder(latent_dim=latent, scale=False)
            self.models_[method] = model
            return model.fit(Z).score(Z)
        if method == "knn":
            search = SimilaritySearch(metric="euclidean").fit(Z)
            self.models_[method] = search
            return search.knn_distance(min(cfg.knn_neighbours, max(len(Z) - 1, 1)))
        if method == "mahalanobis":
            # Distance from the population centre in units of its own scatter.
            centre = np.median(Z, axis=0)
            covariance = np.cov(Z, rowvar=False) + np.eye(Z.shape[1]) * 1e-6
            inverse = np.linalg.pinv(covariance)
            delta = Z - centre
            return np.sqrt(np.abs(np.einsum("ij,jk,ik->i", delta, inverse, delta)))
        log.warning("unknown anomaly method '%s'", method)
        return None


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Map raw detector scores onto a comparable ``[0, 1]`` scale."""
    values = np.asarray(scores, dtype=float)
    values = np.nan_to_num(values, nan=float(np.nanmedian(values[np.isfinite(values)]))
                           if np.isfinite(values).any() else 0.0)
    median = float(np.median(values))
    spread = float(np.median(np.abs(values - median))) * 1.4826
    if spread <= 1e-12:
        return rescale(values, 0.0, 1.0)
    # A logistic on the robust z-score keeps a single extreme outlier from
    # compressing everything else into the bottom of the range.
    return np.asarray(logistic(values, scale=spread, midpoint=median), dtype=float)


def _novelty_type(source: Source, contributions: Dict[str, float]) -> str:
    """A short label for *why* an object stands out."""
    morphology = source.morphology
    if source.lens_score >= 0.5:
        return "lens_like"
    if np.isfinite(morphology.m20) and morphology.m20 > -1.1:
        return "multiple_nuclei"
    if np.isfinite(morphology.asymmetry) and morphology.asymmetry > 0.35:
        return "highly_asymmetric"
    if np.isfinite(morphology.sersic_index) and morphology.sersic_index > 6.0:
        return "extreme_profile"
    if np.isfinite(morphology.ellipticity) and morphology.ellipticity > 0.8:
        return "extremely_elongated"
    if source.object_class == ObjectClass.UNKNOWN:
        return "unclassified"
    if contributions.get("knn", 0.0) > 0.8:
        return "isolated_in_feature_space"
    return "statistical_outlier"


def explain(source: Source, contributions: Dict[str, float],
            features: np.ndarray, names: Sequence[str]) -> str:
    """One sentence saying what is unusual about this object."""
    reasons: List[str] = []
    morphology = source.morphology
    if np.isfinite(morphology.asymmetry) and morphology.asymmetry > 0.3:
        reasons.append(f"high asymmetry (A={morphology.asymmetry:.2f})")
    if np.isfinite(morphology.m20) and morphology.m20 > -1.2:
        reasons.append(f"light spread away from one centre (M20={morphology.m20:.2f})")
    if np.isfinite(morphology.sersic_index) and morphology.sersic_index > 6:
        reasons.append(f"very steep profile (n={morphology.sersic_index:.1f})")
    if np.isfinite(morphology.ellipticity) and morphology.ellipticity > 0.75:
        reasons.append(f"extreme elongation (e={morphology.ellipticity:.2f})")
    if np.isfinite(morphology.gini) and morphology.gini > 0.75:
        reasons.append(f"light concentrated in few pixels (Gini={morphology.gini:.2f})")
    if contributions.get("knn", 0.0) > 0.75:
        reasons.append("no close analogue elsewhere in the field")
    if source.flags:
        reasons.append("flagged " + ", ".join(source.flags))

    leader = max(contributions, key=contributions.get) if contributions else "ensemble"
    if not reasons:
        reasons.append(f"flagged mainly by the {leader.replace('_', ' ')} detector")
    return ("Unusual because " + "; ".join(reasons) +
            ". This is a candidate for human inspection, not a detection.")
