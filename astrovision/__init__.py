"""AstroVision-X: computer vision and machine learning for astronomy.

A pipeline that takes telescope imagery and produces a scientific report:
sources are detected and deblended, measured photometrically and
morphologically, classified, searched for novelty and for gravitational
lensing, and -- given several epochs -- differenced for transients and
characterised in the time domain.

The platform reports *candidates* with the evidence behind them.  It does
not announce discoveries: that requires independent observations and human
astronomers, and the wording throughout is written to keep that boundary
visible.

>>> from astrovision import Pipeline, quick_field
>>> image, truth = quick_field((160, 160))
>>> analysis = Pipeline().run(image)
>>> analysis.summary()["n_sources"] > 0
True
"""

from .version import __version__

from .core import (
    AstroVisionConfig,
    FieldAnalysis,
    LightCurve,
    Morphology,
    ObjectClass,
    Source,
    SourceCatalog,
    TransientCandidate,
    Verdict,
    configure,
    default_config,
    describe_capabilities,
    get_logger,
)
from .io import AstroImage, ImageSeries, read_catalog, write_catalog
from .simulate import SkyConfig, SkySimulator, quick_field

__all__ = [
    "__version__",
    "Pipeline", "AstroVisionConfig", "default_config", "configure", "get_logger",
    "describe_capabilities",
    "AstroImage", "ImageSeries", "read_catalog", "write_catalog",
    "Source", "SourceCatalog", "FieldAnalysis", "LightCurve",
    "TransientCandidate", "ObjectClass", "Morphology", "Verdict",
    "SkySimulator", "SkyConfig", "quick_field",
    "analyze", "analyze_series",
]


def __getattr__(name):
    """Import the heavy engine lazily, so ``import astrovision`` stays fast."""
    if name == "Pipeline":
        from .engine import Pipeline as _Pipeline
        return _Pipeline
    raise AttributeError(f"module 'astrovision' has no attribute '{name}'")


def analyze(image, config=None, **kwargs):
    """Analyse a single image and return its :class:`FieldAnalysis`.

    ``image`` may be an :class:`AstroImage` or a path to a FITS file.
    """
    from .engine import Pipeline as _Pipeline

    if isinstance(image, str):
        image = AstroImage.load(image)
    return _Pipeline(config).run(image, **kwargs)


def analyze_series(images, config=None, **kwargs):
    """Analyse a multi-epoch series given images or file paths."""
    from .engine import Pipeline as _Pipeline

    if isinstance(images, ImageSeries):
        series = images
    elif images and isinstance(images[0], str):
        series = ImageSeries.from_paths(list(images))
    else:
        series = ImageSeries(list(images))
    return _Pipeline(config).run_series(series, **kwargs)
