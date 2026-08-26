"""Core abstractions: configuration, typed records, numerics and plumbing."""

from .backend import capabilities, describe_capabilities, has, require, try_import
from .config import (
    PRESETS,
    AstroVisionConfig,
    AnomalyConfig,
    ClassificationConfig,
    ClusteringConfig,
    CosmologyConfig,
    DetectionConfig,
    LensingConfig,
    MorphologyConfig,
    PhotometryConfig,
    PreprocessConfig,
    ReportConfig,
    SegmentationConfig,
    TimeSeriesConfig,
    TransientConfig,
    default_config,
)
from .exceptions import (
    AstroVisionError,
    ConfigError,
    DataError,
    DimensionError,
    MissingDependencyError,
    ModelError,
    NotFittedError,
    PipelineError,
    RegistryError,
)
from .logging import configure, get_logger, set_level, timed
from .registry import Registry
from .types import (
    AnomalyRecord,
    BoundingBox,
    FieldAnalysis,
    LensCandidate,
    LightCurve,
    Morphology,
    MorphologyMetrics,
    ObjectClass,
    Photometry,
    Source,
    SourceCatalog,
    TransientCandidate,
    Verdict,
)

__all__ = [
    "AstroVisionConfig", "AnomalyConfig", "ClassificationConfig", "ClusteringConfig",
    "CosmologyConfig", "DetectionConfig", "LensingConfig", "MorphologyConfig",
    "PhotometryConfig", "PreprocessConfig", "ReportConfig", "SegmentationConfig",
    "TimeSeriesConfig", "TransientConfig", "PRESETS", "default_config",
    "AstroVisionError", "ConfigError", "DataError", "DimensionError",
    "MissingDependencyError", "ModelError", "NotFittedError", "PipelineError",
    "RegistryError",
    "configure", "get_logger", "set_level", "timed", "Registry",
    "capabilities", "describe_capabilities", "has", "require", "try_import",
    "AnomalyRecord", "BoundingBox", "FieldAnalysis", "LensCandidate", "LightCurve",
    "Morphology", "MorphologyMetrics", "ObjectClass", "Photometry", "Source",
    "SourceCatalog", "TransientCandidate", "Verdict",
]
