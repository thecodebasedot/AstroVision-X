"""Core types, configuration and numerics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from astrovision.core import numeric
from astrovision.core.config import PRESETS, AstroVisionConfig
from astrovision.core.exceptions import ConfigError, RegistryError
from astrovision.core.registry import Registry
from astrovision.core.types import (
    BoundingBox,
    FieldAnalysis,
    LightCurve,
    ObjectClass,
    Photometry,
    Source,
    SourceCatalog,
)


class TestBoundingBox:
    def test_geometry(self):
        box = BoundingBox(2.0, 4.0, 10.0, 12.0)
        assert box.width == 8.0
        assert box.height == 8.0
        assert box.area == 64.0
        assert box.center == (6.0, 8.0)

    def test_iou_identical_is_one(self):
        box = BoundingBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_disjoint_is_zero(self):
        assert BoundingBox(0, 0, 1, 1).iou(BoundingBox(5, 5, 6, 6)) == 0.0

    def test_clip_to_image(self):
        clipped = BoundingBox(-5, -5, 100, 100).clip((20, 30))
        assert (clipped.x_min, clipped.y_min) == (0.0, 0.0)
        assert (clipped.x_max, clipped.y_max) == (30.0, 20.0)

    def test_slices_stay_in_bounds(self):
        rows, cols = BoundingBox(-3, -3, 5, 5).slices((10, 10), pad=2)
        assert rows.start >= 0 and cols.start >= 0
        assert rows.stop <= 10 and cols.stop <= 10


class TestSourceCatalog:
    def _catalog(self, n=5):
        return SourceCatalog([
            Source(i, float(i * 10), float(i * 5),
                   BoundingBox(i * 10 - 2, i * 5 - 2, i * 10 + 2, i * 5 + 2),
                   object_class=ObjectClass.STAR if i % 2 else ObjectClass.GALAXY,
                   photometry=Photometry(flux=100.0 * i))
            for i in range(1, n + 1)])

    def test_length_and_iteration(self):
        catalog = self._catalog()
        assert len(catalog) == 5
        assert len(list(catalog)) == 5

    def test_class_counts(self):
        counts = self._catalog().class_counts()
        assert counts["star"] + counts["galaxy"] == 5

    def test_filter_and_slice_return_catalogs(self):
        catalog = self._catalog()
        assert isinstance(catalog[1:3], SourceCatalog)
        assert len(catalog.of_class(ObjectClass.STAR)) == 3

    def test_match_by_position(self):
        catalog = self._catalog()
        assert catalog.match(30.0, 15.0, radius=1.0).id == 3
        assert catalog.match(999.0, 999.0, radius=1.0) is None

    def test_brightest_is_ordered(self):
        brightest = self._catalog().brightest(3)
        assert [s.id for s in brightest] == [5, 4, 3]

    def test_renumber(self):
        catalog = self._catalog().filter(lambda s: s.id > 2).renumber()
        assert [s.id for s in catalog] == [1, 2, 3]

    def test_round_trips_through_dict(self):
        payload = self._catalog().to_dict()
        assert payload["count"] == 5
        assert json.dumps(payload)          # must be JSON-serialisable


class TestLightCurve:
    def test_sorts_by_time(self):
        curve = LightCurve([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
        assert list(curve.times) == [1.0, 2.0, 3.0]
        assert list(curve.fluxes) == [10.0, 20.0, 30.0]

    def test_baseline(self):
        assert LightCurve([0.0, 5.0], [1.0, 1.0]).baseline == 5.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            LightCurve([1.0, 2.0], [1.0])

    def test_clean_drops_non_finite(self):
        curve = LightCurve([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
        assert len(curve.clean()) == 2

    def test_normalized_has_unit_median(self):
        curve = LightCurve([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert np.median(curve.normalized()) == pytest.approx(1.0)


class TestConfig:
    def test_defaults_are_complete(self, config):
        assert config.detection.threshold_sigma > 0
        assert config.cosmology.H0 > 0

    def test_dotted_override(self, config):
        config.update("detection.threshold_sigma", 4.5)
        assert config.detection.threshold_sigma == 4.5

    def test_unknown_option_raises(self, config):
        with pytest.raises(ConfigError):
            config.update("detection.no_such_option", 1)

    def test_cli_style_overrides(self, config):
        config.apply_overrides(["detection.min_area=9", "segmentation.enabled=false"])
        assert config.detection.min_area == 9
        assert config.segmentation.enabled is False

    def test_round_trip_through_dict_preserves_types(self, config):
        restored = AstroVisionConfig.from_dict(config.to_dict())
        assert restored.detection.threshold_sigma == config.detection.threshold_sigma
        assert isinstance(restored.detection.aperture_radii if hasattr(
            restored.detection, "aperture_radii") else restored.photometry.aperture_radii,
            list)

    @pytest.mark.parametrize("preset", sorted(PRESETS))
    def test_every_preset_applies(self, config, preset):
        config.with_preset(preset)

    def test_save_and_load_json(self, tmp_path, config):
        path = str(tmp_path / "cfg.json")
        config.save(path)
        assert AstroVisionConfig.load(path).name == config.name

    def test_save_and_load_yaml(self, tmp_path, config):
        pytest.importorskip("yaml", reason="YAML configuration needs PyYAML")
        path = str(tmp_path / "cfg.yaml")
        config.save(path)
        assert AstroVisionConfig.load(path).name == config.name

    def test_describe_is_flat(self, config):
        lines = config.describe()
        assert any(line.startswith("detection.threshold_sigma") for line in lines)


class TestRegistry:
    def test_register_and_create(self):
        registry = Registry("thing")

        @registry.register("widget")
        def _build():
            return "built"

        assert "widget" in registry
        assert registry.create("widget") == "built"

    def test_duplicate_raises(self):
        registry = Registry("thing")
        registry.register("a", lambda: 1)
        with pytest.raises(RegistryError):
            registry.register("a", lambda: 2)

    def test_unknown_raises(self):
        with pytest.raises(RegistryError):
            Registry("thing").get("missing")


class TestNumeric:
    def test_sigma_clipped_stats_resists_outliers(self):
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 5.0, 2000)
        data[:20] = 1e6
        _, median, std = numeric.sigma_clipped_stats(data)
        assert median == pytest.approx(100.0, abs=1.0)
        assert std == pytest.approx(5.0, rel=0.2)

    def test_mad_std_matches_gaussian_sigma(self):
        data = np.random.default_rng(1).normal(0.0, 3.0, 5000)
        assert float(numeric.mad_std(data)) == pytest.approx(3.0, rel=0.1)

    def test_gaussian_kernel_is_normalised_and_odd(self):
        kernel = numeric.gaussian_kernel(2.0)
        assert kernel.sum() == pytest.approx(1.0)
        assert kernel.shape[0] % 2 == 1

    def test_convolution_preserves_flux(self):
        image = np.zeros((32, 32))
        image[16, 16] = 100.0
        convolved = numeric.convolve(image, numeric.gaussian_kernel(1.5))
        assert convolved.sum() == pytest.approx(100.0, rel=1e-6)

    def test_resize_round_trip_shape(self):
        image = np.random.default_rng(2).normal(size=(40, 30))
        assert numeric.bilinear_resize(image, (20, 15)).shape == (20, 15)

    def test_pad_or_crop_exact_shape(self):
        image = np.ones((10, 10))
        assert numeric.pad_or_crop(image, (20, 5)).shape == (20, 5)

    def test_weighted_centroid_finds_the_peak(self):
        image = np.zeros((21, 21))
        image[7, 13] = 1.0
        x, y = numeric.weighted_centroid(image)
        assert (x, y) == pytest.approx((13.0, 7.0))

    def test_softmax_sums_to_one(self):
        assert numeric.softmax([1.0, 2.0, 3.0]).sum() == pytest.approx(1.0)

    def test_safe_divide_handles_zero(self):
        out = numeric.safe_divide(np.array([1.0, 2.0]), np.array([0.0, 2.0]), fill=-1.0)
        assert out[0] == -1.0 and out[1] == 1.0

    def test_as_float_image_rejects_3d(self):
        with pytest.raises(ValueError):
            numeric.as_float_image(np.zeros((4, 4, 5)))

    def test_as_float_image_converts_rgb(self):
        assert numeric.as_float_image(np.zeros((4, 4, 3))).shape == (4, 4)


class TestFieldAnalysis:
    def test_empty_summary(self):
        summary = FieldAnalysis().summary()
        assert summary["n_sources"] == 0

    def test_warnings_are_deduplicated(self):
        analysis = FieldAnalysis()
        analysis.warn("same")
        analysis.warn("same")
        assert analysis.warnings == ["same"]

    def test_to_dict_is_json_serialisable(self):
        assert json.dumps(FieldAnalysis().to_dict())
