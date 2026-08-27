"""Machine-learning components: features, embeddings, models and clustering."""

from .autoencoder import DeepAutoencoder, LinearAutoencoder, build_autoencoder
from .calibration import (
    Calibrator,
    brier_score,
    calibration_report,
    calibrate_catalog,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
)
from .clustering import DBSCAN, HDBSCANLite, KMeans, cluster, silhouette_score
from .cnn import STAMP_CLASSES, StampClassifier, build_cnn, build_vit
from .features import (
    FEATURE_NAMES,
    catalog_embeddings,
    catalog_features,
    combine_features,
    cutout_descriptor,
    feature_report,
    source_features,
)
from .gbdt import GradientBoostedClassifier
from .isolation_forest import IsolationForest
from .metric import SimilaritySearch, find_similar
from .scaler import PCA, RobustScaler

__all__ = [
    "Calibrator", "fit_calibrator", "calibrate_catalog", "reliability_curve",
    "expected_calibration_error", "brier_score", "calibration_report",
    "RobustScaler", "PCA",
    "IsolationForest", "LinearAutoencoder", "DeepAutoencoder", "build_autoencoder",
    "KMeans", "DBSCAN", "HDBSCANLite", "cluster", "silhouette_score",
    "StampClassifier", "STAMP_CLASSES", "build_cnn", "build_vit",
    "GradientBoostedClassifier",
    "SimilaritySearch", "find_similar",
    "FEATURE_NAMES", "source_features", "catalog_features", "catalog_embeddings",
    "combine_features", "cutout_descriptor", "feature_report",
]
