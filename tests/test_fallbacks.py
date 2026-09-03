"""The NumPy fallbacks must give SciPy's answer, edges included.

Every optional backend is a feature, never a requirement, and that promise
is only worth something if the fallback computes the same numbers. It did
not: NumPy's ``reflect`` padding is SciPy's ``mirror``, so every fallback
differed along the edges, and on a 6x6 background mesh the edges are most
of the array. The NumPy-only detection lost six of 48 sources on a test
field and a higher threshold found *more* sources than a lower one.
"""

from __future__ import annotations

import numpy as np
import pytest

import astrovision.core.backend as backend
from astrovision.core import numeric
from astrovision.core.backend import has
from astrovision.detect import labeling

pytestmark = pytest.mark.skipif(not has("scipy.ndimage"),
                                reason="comparing the fallbacks needs SciPy")


@pytest.fixture()
def without_scipy(monkeypatch):
    """Make every module that asks for SciPy believe it is absent."""
    real = backend.try_import

    def denied(name, *args, **kwargs):
        return None if name.startswith("scipy") else real(name, *args, **kwargs)

    for module in (numeric, labeling):
        monkeypatch.setattr(module, "try_import", denied)
    return denied


@pytest.fixture()
def image():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 52))
    data[10:14, 30:35] += 8.0
    data[0, 0] = 5.0          # corners are where padding conventions bite
    data[-1, -1] = -5.0
    return data


def _both(fn, without_scipy, monkeypatch, *args, **kwargs):
    real = backend.try_import
    for module in (numeric, labeling):
        monkeypatch.setattr(module, "try_import", real)
    with_scipy = fn(*args, **kwargs)
    for module in (numeric, labeling):
        monkeypatch.setattr(module, "try_import", without_scipy)
    fallback = fn(*args, **kwargs)
    if isinstance(with_scipy, (tuple, list)):
        return with_scipy, fallback
    return np.asarray(with_scipy), np.asarray(fallback)


class TestFilters:
    @pytest.mark.parametrize("mode", ["reflect", "nearest", "mirror", "wrap", "constant"])
    def test_convolve_matches_in_every_edge_mode(self, image, without_scipy, monkeypatch, mode):
        kernel = numeric.gaussian_kernel(1.3)
        a, b = _both(numeric.convolve, without_scipy, monkeypatch, image, kernel, mode=mode)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_gaussian_filter_matches(self, image, without_scipy, monkeypatch):
        a, b = _both(numeric.gaussian_filter, without_scipy, monkeypatch, image, 1.3)
        np.testing.assert_allclose(a, b, atol=1e-6)

    @pytest.mark.parametrize("size", [3, 5, 7])
    def test_median_filter_matches(self, image, without_scipy, monkeypatch, size):
        a, b = _both(numeric.median_filter, without_scipy, monkeypatch, image, size)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("size", [3, 5])
    def test_maximum_filter_matches(self, image, without_scipy, monkeypatch, size):
        a, b = _both(numeric.maximum_filter, without_scipy, monkeypatch, image, size)
        np.testing.assert_array_equal(a, b)

    def test_a_small_mesh_is_all_edges(self, without_scipy, monkeypatch):
        """The background mesh of a 192-pixel field with 64-pixel boxes is
        3x3: there is no interior at all, so a padding mistake changes
        every value the background model is built from."""
        mesh = np.arange(9, dtype=float).reshape(3, 3) ** 1.5
        a, b = _both(numeric.median_filter, without_scipy, monkeypatch, mesh, 3)
        np.testing.assert_array_equal(a, b)


class TestLabelling:
    def test_label_and_boxes_match(self, image, without_scipy, monkeypatch):
        mask = image > 1.5
        (la, na), (lb, nb) = _both(labeling.label, without_scipy, monkeypatch, mask)
        assert na == nb
        np.testing.assert_array_equal(la, lb)
        boxes_a, boxes_b = _both(labeling.find_objects, without_scipy, monkeypatch, la, na)
        assert list(boxes_a) == list(boxes_b)

    def test_dilation_matches(self, image, without_scipy, monkeypatch):
        mask = image > 2.0
        a, b = _both(labeling.binary_dilate, without_scipy, monkeypatch, mask, 2)
        np.testing.assert_array_equal(a, b)


class TestEndToEnd:
    def test_the_preprocessed_field_is_identical(self, synthetic_field, without_scipy,
                                                 monkeypatch):
        """Same background, same noise map, same PSF, with and without
        SciPy -- the whole detection stage downstream then agrees."""
        from astrovision.preprocess import Preprocessor

        image, _ = synthetic_field
        real = backend.try_import
        for module in (numeric, labeling):
            monkeypatch.setattr(module, "try_import", real)
        with_scipy = Preprocessor().run(image)
        for module in (numeric, labeling):
            monkeypatch.setattr(module, "try_import", without_scipy)
        fallback = Preprocessor().run(image)
        np.testing.assert_allclose(with_scipy.subtracted(), fallback.subtracted(), atol=1e-8)
        np.testing.assert_allclose(with_scipy.rms_map(), fallback.rms_map(), atol=1e-8)
        assert with_scipy.meta["psf_model"].fwhm == pytest.approx(
            fallback.meta["psf_model"].fwhm, rel=1e-9)
