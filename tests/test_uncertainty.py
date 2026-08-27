"""Error bars, parameter covariance, and probability calibration.

The tests here check that the uncertainties are *right*, not merely present:
an error bar that does not grow with the noise is decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.config import MorphologyConfig
from astrovision.core.exceptions import DataError
from astrovision.core.types import BoundingBox, Source, SourceCatalog
from astrovision.io.wcs import SimpleWCS
from astrovision.ml.calibration import (
    Calibrator,
    brier_score,
    calibrate_catalog,
    calibration_report,
    expected_calibration_error,
    fit_calibrator,
    fit_isotonic,
    fit_platt,
    reliability_curve,
    to_log_odds,
)
from astrovision.morphology import MorphologyAnalyzer
from astrovision.morphology.uncertainty import (
    annotate_uncertainty,
    bootstrap_morphology,
    covariance_errors,
)


def _galaxy(size=61, scale=7.0, amplitude=400.0):
    grid = np.mgrid[0:size, 0:size]
    radius = np.hypot(grid[0] - (size - 1) / 2.0, grid[1] - (size - 1) / 2.0)
    return amplitude * np.exp(-radius / scale)


class TestBootstrapErrors:
    def test_errors_grow_with_the_noise(self):
        truth = _galaxy()
        centre = ((truth.shape[1] - 1) / 2.0, (truth.shape[0] - 1) / 2.0)
        quiet = bootstrap_morphology(
            truth + np.random.default_rng(1).normal(0, 1.0, truth.shape),
            1.0, centre=centre, n_samples=16, seed=2)
        loud = bootstrap_morphology(
            truth + np.random.default_rng(1).normal(0, 12.0, truth.shape),
            12.0, centre=centre, n_samples=16, seed=2)
        for name in ("concentration", "asymmetry", "gini", "m20"):
            assert loud.error(name) > quiet.error(name)

    def test_reports_the_upward_noise_bias_of_asymmetry(self):
        """Asymmetry is built from absolute differences, so noise pushes it up
        whichever way the noise happens to go."""
        truth = _galaxy()
        noisy = truth + np.random.default_rng(3).normal(0, 10.0, truth.shape)
        centre = ((truth.shape[1] - 1) / 2.0, (truth.shape[0] - 1) / 2.0)
        errors = bootstrap_morphology(noisy, 10.0, centre=centre, n_samples=16, seed=5)
        # Adding more noise to an already noisy image raises asymmetry again.
        from astrovision.morphology.cas import asymmetry

        measured = float(asymmetry(noisy, centre)["asymmetry"])
        assert errors.bias("asymmetry", measured) > 0

    def test_refuses_without_a_noise_estimate(self):
        errors = bootstrap_morphology(_galaxy(), float("nan"))
        assert errors.n_samples == 0
        assert "noise" in errors.reason

    def test_an_empty_cutout_is_handled(self):
        assert bootstrap_morphology(np.zeros((0, 0)), 1.0).n_samples == 0

    def test_unmeasured_statistics_report_nan_not_zero(self):
        errors = bootstrap_morphology(_galaxy(), 2.0, n_samples=4,
                                      statistics=("gini",))
        assert np.isnan(errors.error("asymmetry"))

    def test_annotating_a_source_records_the_errors(self):
        source = Source(id=1, x=30.0, y=30.0, bbox=BoundingBox(0, 0, 60, 60))
        errors = bootstrap_morphology(_galaxy(), 4.0, centre=(30.0, 30.0),
                                      n_samples=8, seed=1)
        annotate_uncertainty(source, errors)
        record = source.meta["morphology_errors"]
        assert record["bootstrap"]["n_samples"] == 8
        assert "gini" in record["bootstrap"]["errors"]


class TestCovarianceErrors:
    def test_recovers_the_error_on_a_straight_line_fit(self):
        """A case with a known analytic answer: the standard error on a mean
        is sigma / sqrt(n)."""
        rng = np.random.default_rng(0)
        n, sigma = 400, 2.0
        data = rng.normal(5.0, sigma, n)

        def residual(theta):
            return (theta[0] - data) / sigma

        errors = covariance_errors(residual, [float(np.mean(data))], ["mean"], n)
        assert errors.of("mean") == pytest.approx(sigma / np.sqrt(n), rel=0.15)

    def test_finds_a_degenerate_pair(self):
        """Two parameters that only ever appear as a sum are perfectly
        correlated, and the fit constrains only their combination."""
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 1.0, 200)

        def residual(theta):
            return (theta[0] + theta[1]) - data

        errors = covariance_errors(residual, [0.0, 0.0], ["a", "b"], 200)
        first, second, value = errors.worst_correlation()
        assert {first, second} == {"a", "b"}
        assert abs(value) > 0.99

    def test_a_non_finite_residual_is_reported_not_raised(self):
        errors = covariance_errors(lambda theta: np.array([np.nan]), [1.0], ["x"], 1)
        assert np.isnan(errors.of("x"))
        assert "not finite" in errors.reason


class TestSersicErrors:
    def test_a_fit_carries_error_bars_and_a_correlation(self, measured, clean_image):
        catalog, segmentation = measured
        MorphologyAnalyzer(MorphologyConfig()).run(clean_image, catalog, segmentation)
        fitted = [s for s in catalog if "sersic" in s.meta]
        assert fitted, "some source should have been fitted"
        with_errors = [s for s in fitted if s.meta["sersic"].get("errors")]
        assert with_errors
        record = with_errors[0].meta["sersic"]
        assert set(record["errors"]) <= {
            "amplitude", "r_eff", "n", "axis_ratio", "position_angle", "background"}
        assert -1.0 <= record["worst_correlation"]["value"] <= 1.0

    def test_a_degenerate_fit_is_flagged(self, measured, clean_image):
        """Sersic index against effective radius is famously degenerate; the
        flag is what stops a bare `n` being read as well determined."""
        catalog, segmentation = measured
        MorphologyAnalyzer(MorphologyConfig()).run(clean_image, catalog, segmentation)
        degenerate = [s for s in catalog if "degenerate_sersic_fit" in s.flags]
        for source in degenerate:
            assert abs(source.meta["sersic"]["worst_correlation"]["value"]) > 0.95

    def test_the_bootstrap_stage_is_off_by_default(self, measured, clean_image):
        catalog, segmentation = measured
        MorphologyAnalyzer(MorphologyConfig()).run(clean_image, catalog, segmentation)
        assert not any("morphology_errors" in s.meta for s in catalog)

    @pytest.mark.slow
    def test_enabling_it_annotates_every_measured_source(self, measured, clean_image):
        catalog, segmentation = measured
        MorphologyAnalyzer(MorphologyConfig(uncertainty=True,
                                            bootstrap_samples=6)).run(
            clean_image, catalog, segmentation)
        assert any("morphology_errors" in s.meta for s in catalog)


class TestCalibration:
    @staticmethod
    def _overconfident(n, seed, sharpness=2.2):
        rng = np.random.default_rng(seed)
        true_p = rng.uniform(0.05, 0.95, n)
        labels = (rng.random(n) < true_p).astype(int)
        reported = true_p ** sharpness / (true_p ** sharpness +
                                          (1 - true_p) ** sharpness)
        return np.clip(reported, 1e-3, 1 - 1e-3), labels

    def test_log_odds_is_the_inverse_of_the_logistic(self):
        values = np.array([0.1, 0.5, 0.9])
        back = 1.0 / (1.0 + np.exp(-to_log_odds(values)))
        assert np.allclose(back, values, atol=1e-6)

    def test_platt_recovers_a_pure_overconfidence_slope(self):
        """An overconfident model multiplies the true log-odds by a constant,
        so the calibrating slope is that constant's reciprocal."""
        scores, labels = self._overconfident(4000, 7, sharpness=2.0)
        calibrator = fit_platt(scores, labels)
        assert calibrator.slope == pytest.approx(0.5, abs=0.12)

    def test_platt_does_not_diverge_on_separable_data(self):
        """Newton's method without a line search runs the slope to millions
        here, producing a calibrator that only ever returns 0 or 1."""
        scores = np.concatenate([np.full(30, 0.2), np.full(30, 0.8)])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        calibrator = fit_platt(scores, labels)
        assert abs(calibrator.slope) < 1e3
        assert np.all(np.isfinite(calibrator.transform(scores)))

    def test_isotonic_is_monotone(self):
        scores, labels = self._overconfident(600, 11)
        calibrator = fit_isotonic(scores, labels)
        grid = np.linspace(0.0, 1.0, 50)
        mapped = calibrator.transform(grid)
        assert np.all(np.diff(mapped) >= -1e-9)

    def test_calibration_reduces_the_calibration_error(self):
        train_scores, train_labels = self._overconfident(1500, 3)
        test_scores, test_labels = self._overconfident(4000, 99)
        before = expected_calibration_error(test_scores, test_labels, 12)
        calibrator = fit_calibrator(train_scores, train_labels)
        after = expected_calibration_error(
            calibrator.transform(test_scores), test_labels, 12)
        assert after < before

    def test_method_is_chosen_by_how_much_data_there_is(self):
        small_scores, small_labels = self._overconfident(50, 1)
        large_scores, large_labels = self._overconfident(600, 2)
        assert fit_calibrator(small_scores, small_labels).method == "platt"
        assert fit_calibrator(large_scores, large_labels).method == "isotonic"

    def test_too_little_data_is_left_uncalibrated(self):
        calibrator = fit_calibrator([0.1, 0.9], [0, 1])
        assert calibrator.method == "identity"
        assert "uncalibrated" in calibrator.reason

    def test_a_single_class_is_left_uncalibrated(self):
        calibrator = fit_calibrator(np.linspace(0.1, 0.9, 40), np.ones(40))
        assert calibrator.method == "identity"
        assert "one class" in calibrator.reason

    def test_unknown_method_is_an_error(self):
        with pytest.raises(DataError):
            fit_calibrator(np.linspace(0.1, 0.9, 40),
                           (np.arange(40) % 2), method="magic")

    def test_identity_calibrator_passes_values_through(self):
        calibrator = Calibrator()
        assert calibrator.transform([0.3])[0] == pytest.approx(0.3)

    def test_isotonic_does_not_extrapolate(self):
        scores, labels = self._overconfident(400, 21)
        calibrator = fit_isotonic(scores, labels)
        low = calibrator.transform([-5.0])[0]
        high = calibrator.transform([5.0])[0]
        assert low == pytest.approx(float(calibrator.knots_y[0]))
        assert high == pytest.approx(float(calibrator.knots_y[-1]))

    def test_a_perfectly_calibrated_model_scores_zero(self):
        scores = [0.1] * 200 + [0.9] * 200
        labels = [0] * 180 + [1] * 20 + [0] * 20 + [1] * 180
        assert expected_calibration_error(scores, labels) == pytest.approx(0.0, abs=1e-9)

    def test_brier_penalises_a_useless_but_calibrated_model(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(500) < 0.5).astype(int)
        base_rate = np.full(500, 0.5)
        informed = np.where(labels > 0, 0.9, 0.1)
        assert brier_score(informed, labels) < brier_score(base_rate, labels)

    def test_reliability_curve_bins_are_populated(self):
        scores, labels = self._overconfident(500, 13)
        curve = reliability_curve(scores, labels, n_bins=8)
        assert curve["counts"].sum() == 500
        assert np.all(curve["counts"] > 0)

    def test_report_says_whether_the_numbers_are_probabilities(self):
        scores, labels = self._overconfident(2000, 17, sharpness=6.0)
        report = calibration_report(scores, labels)
        assert not report["usable_as_probability"]
        calibrator = fit_calibrator(scores, labels)
        better = calibration_report(calibrator.transform(scores), labels)
        assert better["expected_calibration_error"] < \
            report["expected_calibration_error"]

    def test_applying_to_a_catalog_keeps_the_raw_value(self):
        sources = []
        for index in range(20):
            source = Source(id=index, x=0.0, y=0.0, bbox=BoundingBox(0, 0, 1, 1))
            source.class_confidence = 0.9
            sources.append(source)
        catalog = SourceCatalog(sources)
        calibrator = Calibrator(method="platt", slope=0.5, intercept=0.0)
        assert calibrate_catalog(catalog, calibrator) == 20
        first = list(catalog)[0]
        assert first.meta["raw_confidence"] == pytest.approx(0.9)
        assert first.class_confidence < 0.9      # an overconfident 0.9 pulled back
        assert catalog.meta["calibration"]["method"] == "platt"


class TestWcsDistortion:
    @staticmethod
    def _distorted():
        wcs = SimpleWCS.tangent(150.0, 2.2, (2048, 2048), 0.4)
        a = np.zeros((3, 3))
        a[2, 0], a[1, 1], a[0, 2] = 1.2e-6, -4e-7, 8e-7
        b = np.zeros((3, 3))
        b[2, 0], b[1, 1], b[0, 2] = -6e-7, 9e-7, 1.5e-6
        wcs.sip_a, wcs.sip_b = a, b
        return wcs

    def test_the_inverse_is_exact(self):
        wcs = self._distorted()
        x = np.array([10.0, 500.0, 1000.0, 1800.0, 2040.0])
        y = np.array([2040.0, 1500.0, 1000.0, 300.0, 10.0])
        ra, dec = wcs.pixel_to_world(x, y)
        back_x, back_y = wcs.world_to_pixel(ra, dec)
        assert np.allclose(back_x, x, atol=1e-6)
        assert np.allclose(back_y, y, atol=1e-6)

    def test_distortion_actually_moves_the_corners(self):
        from astrovision.io.wcs import angular_separation

        distorted, plain = self._distorted(), SimpleWCS.tangent(
            150.0, 2.2, (2048, 2048), 0.4)
        ra1, dec1 = distorted.pixel_to_world(10.0, 2040.0)
        ra2, dec2 = plain.pixel_to_world(10.0, 2040.0)
        assert angular_separation(ra1, dec1, ra2, dec2) * 3600.0 > 0.3

    def test_header_round_trip_keeps_the_sip_suffix(self):
        wcs = self._distorted()
        header = wcs.to_header()
        assert header["CTYPE1"].endswith("-SIP")
        assert header["A_ORDER"] == 2
        restored = SimpleWCS.from_header(header)
        assert restored.has_distortion
        ra1, dec1 = wcs.pixel_to_world(1500.0, 400.0)
        ra2, dec2 = restored.pixel_to_world(1500.0, 400.0)
        assert ra1 == pytest.approx(ra2)
        assert dec1 == pytest.approx(dec2)

    def test_a_wcs_without_sip_is_unchanged(self):
        wcs = SimpleWCS.tangent(150.0, 2.2, (512, 512), 0.4)
        assert not wcs.has_distortion
        u, v = wcs.apply_distortion(np.array([3.0]), np.array([4.0]))
        assert u[0] == 3.0 and v[0] == 4.0
        assert "A_ORDER" not in wcs.to_header()
