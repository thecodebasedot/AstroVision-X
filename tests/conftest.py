"""Shared fixtures.

Fields are generated once per session and reused: the simulator is
deterministic given a seed, so this costs nothing in coverage and keeps
the suite fast enough to run on every commit.
"""

from __future__ import annotations

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


@pytest.fixture(scope="session")
def multiband_field():
    """A three-band field of the same sky, with per-band seeing.

    Session-scoped because rendering three bands is the most expensive
    fixture in the suite; every test that mutates the catalog builds its own
    copy rather than sharing one.
    """
    simulator = SkySimulator(SkyConfig(
        shape=(200, 200), n_stars=70, n_galaxies=20, n_nebulae=1, n_clusters=0,
        n_lenses=1, n_anomalies=1, seed=77))
    images, truth = simulator.generate_multiband(
        ("g", "r", "i"), seeing={"g": 3.7, "r": 3.2, "i": 3.4})
    preprocessor = Preprocessor()
    return {band: preprocessor.run(image) for band, image in images.items()}, truth


@pytest.fixture()
def multiband_measured(multiband_field):
    """``(bands, truth, catalog, segmentation)`` detected and measured in r."""
    from astrovision.core.types import SourceCatalog

    bands, truth = multiband_field
    catalog, segmentation = Detector().detect(bands["r"])
    fresh = SourceCatalog([_copy_source(s) for s in catalog], dict(catalog.meta))
    Photometer().run(bands["r"], fresh, segmentation)
    return bands, truth, fresh, segmentation


@pytest.fixture()
def reference_objects(multiband_field):
    """Reference catalog entries at the true positions of the brighter stars."""
    from astrovision.io.external import ReferenceObject

    bands, truth = multiband_field
    wcs = bands["r"].wcs
    objects = []
    for item in truth:
        if item.kind != "star" or item.flux < 2000:
            continue
        ra, dec = wcs.pixel_to_world(item.x, item.y)
        objects.append(ReferenceObject(
            ra=float(ra), dec=float(dec), name=f"REF-{item.id}", catalog="TESTREF",
            object_type="*",
            magnitudes={"r": float(25.0 - 2.5 * __import__("math").log10(item.flux))}))
    return objects
