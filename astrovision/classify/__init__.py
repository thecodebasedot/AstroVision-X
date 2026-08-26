"""Object classification: star/galaxy separation and class assignment."""

from .classifier import Classifier
from .colours import (
    StellarLocus,
    annotate_catalog,
    colour_stellarity,
    fit_stellar_locus,
)
from .rules import classify_source, combine_stellarity, stellarity

__all__ = [
    "Classifier", "classify_source", "stellarity", "combine_stellarity",
    "StellarLocus", "fit_stellar_locus", "colour_stellarity", "annotate_catalog",
]
