"""Object classification: star/galaxy separation and class assignment."""

from .classifier import Classifier
from .rules import classify_source, stellarity

__all__ = ["Classifier", "classify_source", "stellarity"]
