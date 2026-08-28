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
from .datasets import (
    MAX_BAD_FRACTION,
    MIN_VOTE_AGREEMENT,
    SIMULATED_CLASSES,
    StampSet,
    class_balance_report,
    clean_stamp,
    load_alert_stamps,
    load_fits_cutouts,
    read_label_table,
    split_dataset,
    stamps_from_fields,
    write_fits_cutouts,
)
from .transfer import (
    DomainStudy,
    FineTuneResult,
    domain_study,
    evaluate,
    fine_tune,
    freeze_backbone,
    replace_head,
    unfreeze,
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
    "StampSet", "clean_stamp", "read_label_table", "load_fits_cutouts",
    "load_alert_stamps", "write_fits_cutouts", "stamps_from_fields",
    "split_dataset", "class_balance_report", "SIMULATED_CLASSES",
    "MAX_BAD_FRACTION", "MIN_VOTE_AGREEMENT",
    "freeze_backbone", "unfreeze", "replace_head", "fine_tune", "evaluate",
    "FineTuneResult", "DomainStudy", "domain_study",
    "FEATURE_NAMES", "source_features", "catalog_features", "catalog_embeddings",
    "combine_features", "cutout_descriptor", "feature_report",
]
