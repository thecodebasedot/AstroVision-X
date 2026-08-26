"""Difference imaging, vetting, transients, variability and lensing."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.config import TransientConfig
from astrovision.core.types import LightCurve
from astrovision.detect import Detector
from astrovision.lensing.arcs import detect_arcs, einstein_radius, subtract_smooth_light
from astrovision.lensing.lens import LensSearch, velocity_dispersion
from astrovision.photometry import Photometer
from astrovision.preprocess.psf import PSFModel
from astrovision.core.numeric import SIGMA_TO_FWHM, gaussian_kernel
from astrovision.simulate.profiles import einstein_arc, gaussian_psf
from astrovision.timeseries.features import (
    reduced_chi2,
    stetson_j,
    variability_features,
    variability_score,
    von_neumann_eta,
)
from astrovision.timeseries.lightcurve import LightCurveAnalyzer, classify_variable
from astrovision.timeseries.periodogram import find_period, phase_fold
from astrovision.transient import TransientDetector
from astrovision.transient.candidates import extract_candidates
from astrovision.transient.difference import build_template, flux_scale_factor, subtract
from astrovision.transient.realbogus import (
    classify_artifact,
    real_bogus_score,
    stamp_features,
)


@pytest.fixture()
def psf():
    return PSFModel(gaussian_kernel(3.0 / SIGMA_TO_FWHM, 21), 3.0)


class TestRealBogus:
    def _score(self, stamp, psf, noise=5.0):
        features = stamp_features(stamp, noise, psf)
        return real_bogus_score(features, psf.fwhm)[0]

    def test_a_real_point_source_scores_high(self, psf):
        stamp = (gaussian_psf((21, 21), (10, 10), 3.0, 200.0) +
                 np.random.default_rng(0).normal(0.0, 5.0, (21, 21)))
        assert self._score(stamp, psf) > 0.7

    def test_a_dipole_is_rejected(self, psf):
        stamp = (gaussian_psf((21, 21), (10.7, 10), 3.0, 200.0) -
                 gaussian_psf((21, 21), (9.3, 10), 3.0, 190.0))
        assert self._score(stamp, psf) < 0.3

    def test_a_cosmic_ray_is_rejected(self, psf):
        stamp = np.zeros((21, 21))
        stamp[10, 10] = 400.0
        stamp[10, 11] = 150.0
        assert self._score(stamp, psf) < 0.3

    def test_a_streak_is_rejected(self, psf):
        stamp = np.zeros((21, 21))
        stamp[10, 4:17] = 120.0
        assert self._score(stamp, psf) < 0.5

    def test_pure_noise_is_rejected(self, psf):
        stamp = np.random.default_rng(1).normal(0.0, 5.0, (21, 21))
        assert self._score(stamp, psf) < 0.3

    def test_artifact_type_is_named(self, psf):
        stamp = np.zeros((21, 21))
        stamp[10, 10] = 400.0
        features = stamp_features(stamp, 5.0, psf)
        _, terms = real_bogus_score(features, psf.fwhm)
        assert classify_artifact(features, terms) in (
            "cosmic_ray", "streak_or_satellite", "not_point_like")


class TestDifferenceImaging:
    def test_flux_scale_of_identical_images_is_one(self, synthetic_series):
        series, _, _ = synthetic_series
        reference = series[0].subtracted()
        assert flux_scale_factor(reference, reference) == pytest.approx(1.0, abs=0.05)

    def test_flux_scale_detects_a_known_ratio(self, synthetic_series):
        series, _, _ = synthetic_series
        reference = series[0].subtracted()
        assert flux_scale_factor(1.5 * reference, reference) == pytest.approx(1.5, rel=0.1)

    def test_self_subtraction_leaves_noise(self, synthetic_series):
        series, _, _ = synthetic_series
        result = subtract(series[0], series[0], align=False, psf_match=False)
        assert abs(float(np.median(result.difference))) < 1e-6

    def test_template_excludes_the_held_out_epoch(self, synthetic_series):
        series, _, _ = synthetic_series
        template = build_template(series, "median", exclude=0)
        assert series[0].name not in template.meta["template_from"]
        assert len(template.meta["template_from"]) == len(series) - 1

    def test_difference_records_diagnostics(self, synthetic_series):
        series, _, _ = synthetic_series
        result = subtract(series[1], build_template(series, "median", exclude=1))
        assert "residual_rms" in result.diagnostics
        assert "subtraction_quality" in result.diagnostics


class TestTransientSearch:
    def test_recovers_injected_transients(self, synthetic_series):
        series, _, injected = synthetic_series
        catalog, segmentation = Detector().detect(series.reference)
        Photometer().run(series.reference, catalog, segmentation)
        candidates = TransientDetector().run(series, catalog)
        vetted = [c for c in candidates if "bogus" not in c.flags]

        recovered = sum(
            1 for entry in injected
            if any(np.hypot(c.x - entry["x"], c.y - entry["y"]) < 4.0 for c in vetted))
        assert recovered >= 1, "at least one injected transient must be recovered"

    def test_injected_transients_are_called_supernovae(self, synthetic_series):
        series, _, injected = synthetic_series
        catalog, segmentation = Detector().detect(series.reference)
        Photometer().run(series.reference, catalog, segmentation)
        candidates = TransientDetector().run(series, catalog)
        matched = [c for c in candidates
                   if any(np.hypot(c.x - e["x"], c.y - e["y"]) < 4.0 for e in injected)]
        assert matched
        assert any(c.classification == "supernova_candidate" for c in matched)

    def test_no_candidates_when_nothing_changes(self, synthetic_series):
        series, _, _ = synthetic_series
        epoch = series[0]
        candidates = extract_candidates(
            subtract(epoch, epoch, align=False, psf_match=False),
            TransientConfig(detection_sigma=5.0))
        assert not [c for c in candidates if "bogus" not in c.flags]

    def test_a_single_epoch_yields_nothing(self, synthetic_series):
        from astrovision.io.image import ImageSeries
        series, _, _ = synthetic_series
        assert TransientDetector().run(ImageSeries([series[0]])) == []


class TestVariability:
    def _curve(self, kind, n=40, seed=0):
        rng = np.random.default_rng(seed)
        times = np.sort(rng.uniform(0.0, 50.0, n))
        errors = np.full(n, 2.0)
        if kind == "constant":
            flux = 100.0 + rng.normal(0.0, 2.0, n)
        elif kind == "sinusoid":
            flux = 100.0 + 15.0 * np.sin(2 * np.pi * times / 3.7) + rng.normal(0, 2.0, n)
        else:
            flux = 100.0 + 40.0 * np.exp(-((times - 30.0) ** 2) / 8.0) + rng.normal(0, 2, n)
        return LightCurve(times, flux, errors)

    def test_constant_curve_has_chi2_near_one(self):
        assert reduced_chi2(self._curve("constant")) == pytest.approx(1.0, abs=0.5)

    def test_variable_curve_has_high_chi2(self):
        assert reduced_chi2(self._curve("sinusoid")) > 5.0

    def test_variability_score_separates_them(self):
        assert (variability_score(self._curve("sinusoid")) >
                variability_score(self._curve("constant")) + 0.4)

    def test_stetson_j_is_higher_for_correlated_variation(self):
        assert stetson_j(self._curve("sinusoid")) > stetson_j(self._curve("constant"))

    def test_von_neumann_eta_near_two_for_noise(self):
        assert von_neumann_eta(self._curve("constant")) == pytest.approx(2.0, abs=0.5)

    def test_period_is_recovered(self):
        result = find_period(self._curve("sinusoid"), 0.5, 30.0)
        assert result["period"] == pytest.approx(3.7, rel=0.05)
        assert result["false_alarm_probability"] < 0.01

    def test_no_period_for_a_constant_curve(self):
        result = find_period(self._curve("constant"), 0.5, 30.0)
        assert result["false_alarm_probability"] > 0.05

    def test_phase_fold_is_ordered(self):
        curve = self._curve("sinusoid")
        phase, _ = phase_fold(curve, 3.7)
        assert np.all(np.diff(phase) >= 0)

    def test_classifies_a_periodic_variable(self):
        curve = self._curve("sinusoid")
        label, _ = classify_variable(curve, find_period(curve, 0.5, 30.0))
        assert label in ("periodic_pulsator", "eclipsing")

    def test_classifies_a_constant_source(self):
        curve = self._curve("constant")
        assert classify_variable(curve, find_period(curve, 0.5, 30.0))[0] == "non_variable"

    def test_features_are_all_present(self):
        features = variability_features(self._curve("eruptive"))
        for key in ("reduced_chi2", "stetson_j", "amplitude", "skewness"):
            assert key in features

    def test_stage_builds_light_curves(self, synthetic_series):
        series, _, _ = synthetic_series
        catalog, segmentation = Detector().detect(series.reference)
        Photometer().run(series.reference, catalog, segmentation)
        curves = LightCurveAnalyzer().run(series, catalog)
        assert curves
        assert all(len(c) == len(series) for c in curves.values())


class TestLensing:
    def _system(self, theta_e=14.0, n_arcs=3, size=90, noise=3.0):
        from astrovision.core.numeric import convolve
        from astrovision.simulate.profiles import (
            elliptical_radius, sersic_profile, supersample)

        centre = (size / 2 - 0.5, size / 2 - 0.5)
        deflector = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 0.85, 20.0), 1.0,
                                        theta_e * 0.55 * (s[0] / size), 4.0), 3)
        deflector = deflector / deflector.sum() * 6e4
        arcs = np.zeros((size, size))
        for k in range(n_arcs):
            # 60-degree spans leave real gaps between images; much wider
            # arcs merge into a ring, which detect_ring handles instead.
            arcs += einstein_arc((size, size), centre, theta_e, 1.6, 60.0,
                                 360.0 * k / n_arcs, 1.0)
        arcs = arcs / max(arcs.sum(), 1e-9) * 2.4e4
        image = convolve(deflector + arcs, gaussian_kernel(1.2))
        return image + np.random.default_rng(0).normal(0.0, noise, (size, size)), centre

    def test_arcs_are_detected(self):
        image, centre = self._system()
        arcs = detect_arcs(image, centre, noise=3.0, threshold_sigma=2.5,
                           min_axis_ratio=2.0, max_width=7.0, min_radius=6.0)
        assert len(arcs) >= 2

    def test_arcs_share_an_einstein_radius(self):
        image, centre = self._system(theta_e=14.0)
        arcs = detect_arcs(image, centre, noise=3.0, min_axis_ratio=2.0,
                           max_width=7.0, min_radius=6.0)
        radius, scatter = einstein_radius(arcs)
        assert radius == pytest.approx(14.0, rel=0.25)
        assert scatter < 0.4 * radius

    def test_a_plain_galaxy_shows_no_arcs(self):
        from astrovision.core.numeric import convolve
        from astrovision.simulate.profiles import (
            elliptical_radius, sersic_profile, supersample)

        size, centre = 90, (44.5, 44.5)
        galaxy = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 0.85, 20.0), 1.0,
                                        8.0 * (s[0] / size), 4.0), 3)
        galaxy = convolve(galaxy / galaxy.sum() * 6e4, gaussian_kernel(1.2))
        galaxy = galaxy + np.random.default_rng(1).normal(0.0, 3.0, (size, size))
        arcs = detect_arcs(galaxy, centre, noise=3.0, min_axis_ratio=2.0,
                           max_width=7.0, min_radius=6.0)
        assert len([a for a in arcs if a.length > 6.0]) == 0

    def test_complete_ring_is_found_by_the_radial_scan(self):
        from astrovision.lensing.arcs import detect_ring
        image, centre = self._system(theta_e=14.0, n_arcs=6)
        assert detect_ring(image, centre, noise=3.0)["ring_detected"]

    def test_smooth_light_subtraction_preserves_partial_rings(self):
        image, centre = self._system(n_arcs=3)
        residual = subtract_smooth_light(image, centre)
        assert float(residual.max()) > 3.0 * float(np.std(residual))

    def test_velocity_dispersion_is_galaxy_scale(self):
        sigma = velocity_dispersion(1.2, z_lens=0.5, z_source=2.0)
        assert 150.0 < sigma < 400.0

    def test_search_returns_a_list(self, clean_image, measured):
        catalog, _ = measured
        assert isinstance(LensSearch().run(clean_image, catalog), list)
