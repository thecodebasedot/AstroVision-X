"""Exception hierarchy for AstroVision-X."""

from __future__ import annotations


class AstroVisionError(Exception):
    """Base class for every error raised by AstroVision-X."""


class ConfigError(AstroVisionError):
    """Raised when a configuration value is missing or invalid."""


class DataError(AstroVisionError):
    """Raised when input data cannot be read or is structurally invalid."""


class DimensionError(DataError):
    """Raised when an array has an unexpected shape or dimensionality."""


class MissingDependencyError(AstroVisionError):
    """Raised when an optional third-party dependency is required but absent."""

    def __init__(self, package: str, feature: str = "", extra: str = ""):
        self.package = package
        message = f"optional dependency '{package}' is required"
        if feature:
            message += f" for {feature}"
        if extra:
            message += f"; install it with: pip install 'astrovision-x[{extra}]'"
        else:
            message += f"; install it with: pip install {package}"
        super().__init__(message)


class ModelError(AstroVisionError):
    """Raised when a model is used incorrectly (e.g. predicting before fitting)."""


class NotFittedError(ModelError):
    """Raised when an estimator is used before :meth:`fit` has been called."""


class PipelineError(AstroVisionError):
    """Raised when a pipeline stage fails irrecoverably."""


class RegistryError(AstroVisionError):
    """Raised for duplicate or unknown registry keys."""
