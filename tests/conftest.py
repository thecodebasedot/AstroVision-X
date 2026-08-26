"""Shared fixtures.

Fields are generated once per session and reused: the simulator is
deterministic given a seed, so this costs nothing in coverage and keeps
the suite fast enough to run on every commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.config import AstroVisionConfig
from astrovision.detect import Detector
from astrovision.io.image import ImageSeries
from astrovision.photometry import Photometer
from astrovision.preprocess import Preprocessor
from astrovision.simulate import SkyConfig, SkySimulator


@pytest.fixture(scope="session")
def sky_config() -> SkyConfig:
    return SkyConfig(shape=(192, 192), n_stars=45, n_galaxies=10, n_nebulae=1,
                     n_clusters=1, n_lenses=1, n_anomalies=1, seed=1234)


@pytest.fixture(scope="session")
def synthetic_field(sky_config):
    """A synthetic image and its truth table."""
    return SkySimulator(sky_config).generate()


@pytest.fixture(scope="session")
def clean_image(synthetic_field):
    """The synthetic field after preprocessing."""
    image, _ = synthetic_field
    return Preprocessor().run(image)


@pytest.fixture(scope="session")
def detected(clean_image):
    """``(catalog, segmentation)`` from the detection stage."""
    return Detector().detect(clean_image)


@pytest.fixture()
def measured(clean_image, detected):
    """A photometered copy of the catalog (function-scoped: stages mutate it)."""
    from astrovision.core.types import SourceCatalog

    catalog, segmentation = detected
    fresh = SourceCatalog([_copy_source(s) for s in catalog], dict(catalog.meta))
    Photometer().run(clean_image, fresh, segmentation)
    return fresh, segmentation


@pytest.fixture(scope="session")
def synthetic_series():
    """A short multi-epoch series with injected transients."""
    simulator = SkySimulator(SkyConfig(
        shape=(160, 160), n_stars=25, n_galaxies=6, n_nebulae=0, n_clusters=0,
        n_lenses=0, n_anomalies=0, seed=99))
    series, truth, transients = simulator.generate_series(
        n_epochs=5, cadence=2.0, n_transients=2)
    prepared = ImageSeries([Preprocessor().run(image) for image in series],
                           name="test_series")
    return prepared, truth, transients


@pytest.fixture()
def config() -> AstroVisionConfig:
    return AstroVisionConfig()


def _copy_source(source):
    """Deep-enough copy of a Source so a test cannot leak state into another."""
    import copy
    return copy.deepcopy(source)
