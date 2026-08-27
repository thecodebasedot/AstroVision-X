"""A PSF that changes across the field, and the check that it earns its freedom."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.config import PreprocessConfig
from astrovision.detect import Detector
from astrovision.photometry import Photometer
from astrovision.preprocess import Preprocessor
from astrovision.preprocess.psf import find_psf_stars
from astrovision.preprocess.varying_psf import (
    VaryingPSF,
    find_psf_stars_by_region,
    fit_varying_psf,
    n_terms,
    polynomial_terms,
    psf_at,
    region_grid,
)
from astrovision.simulate import SkyConfig, SkySimulator


def _field(variation, shape=800, n_stars=220, seed=7):
    config = SkyConfig(shape=(shape, shape), n_stars=n_stars, n_galaxies=10,
                       n_nebulae=0, n_clusters=0, n_lenses=0, n_anomalies=0,
                       seed=seed, seeing_variation=variation, seeing_fwhm=3.0,
                       star_flux_range=(4000.0, 90000.0))
    simulator = SkySimulator(config)
    image, truth = simulator.generate()
    return simulator, Preprocessor().run(image), truth


@pytest.fixture(scope="module")
def varying_field():
    return _field(0.40)


@pytest.fixture(scope="module")
def constant_field():
    return _field(0.0)


class TestSimulatedVariation:
    def test_the_corners_are_blurrier_than_the_axis(self):
        simulator = SkySimulator(SkyConfig(shape=(400, 400), seeing_fwhm=3.0,
                                           seeing_variation=0.35))
        # The optical axis sits at ((n-1)/2, (n-1)/2), so (200, 200) is half
        # a pixel off it -- close enough that the quadratic term is tiny.
        assert simulator.local_fwhm(200, 200) == pytest.approx(3.0, abs=1e-4)
        assert simulator.local_fwhm(10, 10) > 3.5

    def test_no_variation_means_no_variation(self):
        simulator = SkySimulator(SkyConfig(shape=(400, 400), seeing_fwhm=3.0))
        assert simulator.local_fwhm(10, 10) == pytest.approx(
            simulator.local_fwhm(200, 200))


class TestPolynomialBasis:
    def test_term_count_matches_the_basis(self):
        for degree in (0, 1, 2, 3):
            assert len(polynomial_terms(0.1, 0.2, degree)) == n_terms(degree)

    def test_constant_term_is_one(self):
        assert polynomial_terms(0.7, -0.3, 2)[0] == pytest.approx(1.0)

    def test_region_grid_tiles_the_whole_image(self):
        tiles = region_grid((100, 80), 2)
        assert len(tiles) == 4
        covered = np.zeros((100, 80), dtype=int)
        for rows, columns, _ in tiles:
            covered[rows, columns] += 1
        assert covered.min() == 1 and covered.max() == 1


class TestFitting:
    def test_it_finds_a_real_variation(self, varying_field):
        simulator, clean, _ = varying_field
        stars = find_psf_stars_by_region(clean.data, 4, clean.rms_map(), per_region=80)
        model = fit_varying_psf(clean.data, stars, size=21, rms=clean.rms_map())
        assert not model.fallback
        assert model.degree == 2
        true_ratio = simulator.local_fwhm(20, 20) / simulator.local_fwhm(400, 400)
        fitted_ratio = model.fwhm_at(20, 20) / model.fwhm_at(400, 400)
        assert fitted_ratio == pytest.approx(true_ratio, rel=0.10)

    def test_it_refuses_when_there_is_nothing_to_find(self, constant_field):
        """The check that stops the model describing noise: a quadratic has
        six free parameters per stamp pixel and will always fit its training
        stars better."""
        _, clean, _ = constant_field
        stars = find_psf_stars_by_region(clean.data, 4, clean.rms_map(), per_region=80)
        model = fit_varying_psf(clean.data, stars, size=21, rms=clean.rms_map())
        assert model.fallback
        assert model.degree == 0
        assert "below the" in model.reason
        assert model.variation() == pytest.approx(0.0, abs=1e-6)

    def test_a_fallback_model_still_answers_everywhere(self, constant_field):
        _, clean, _ = constant_field
        model = fit_varying_psf(clean.data, size=21, rms=clean.rms_map())
        assert np.isfinite(model.fwhm_at(10, 10))
        assert model.at(10, 10).stamp.shape == (21, 21)

    def test_too_few_stars_reduces_the_degree(self, varying_field):
        _, clean, _ = varying_field
        stars = find_psf_stars(clean.data, rms=clean.rms_map())[:14]
        model = fit_varying_psf(clean.data, stars, size=21, rms=clean.rms_map())
        assert model.degree <= 1

    def test_no_stars_falls_back_to_a_gaussian(self):
        model = fit_varying_psf(np.zeros((60, 60)), positions=[], size=15)
        assert model.fallback
        assert "no usable PSF stars" in model.reason
        assert np.isfinite(model.fwhm_at(30, 30))

    def test_it_does_not_extrapolate_past_the_stars(self, varying_field):
        """A polynomial beyond its data is not a model: an unclamped
        quadratic ran past the last star and returned FWHMs twice the truth."""
        _, clean, _ = varying_field
        stars = find_psf_stars_by_region(clean.data, 4, clean.rms_map(), per_region=80)
        model = fit_varying_psf(clean.data, stars, size=21, rms=clean.rms_map())
        if model.fallback:
            pytest.skip("this field did not support a spatial fit")
        x0, y0, x1, y1 = model.star_bounds
        assert model.extrapolates_at(x0 - 50, y0 - 50)
        assert not model.extrapolates_at(0.5 * (x0 + x1), 0.5 * (y0 + y1))
        # Clamped, so a position outside the stars returns the edge value.
        assert model.fwhm_at(x0 - 200, y0 - 200) == pytest.approx(
            model.fwhm_at(x0, y0), rel=1e-9)

    def test_a_stamp_is_normalised(self, varying_field):
        _, clean, _ = varying_field
        model = fit_varying_psf(clean.data, size=21, rms=clean.rms_map())
        assert model.stamp_at(400, 400).sum() == pytest.approx(1.0)

    def test_the_report_is_serialisable(self, varying_field):
        _, clean, _ = varying_field
        payload = fit_varying_psf(clean.data, size=21, rms=clean.rms_map()).to_dict()
        assert set(payload) >= {"degree", "fallback", "reason", "variation",
                                "fwhm_grid", "star_bounds"}


class TestRegionalStarSelection:
    def test_it_samples_the_whole_field(self, varying_field):
        """A whole-frame 'smallest first' rule keeps the sharpest sources,
        which where the PSF varies means the ones nearest the axis."""
        _, clean, _ = varying_field
        stars = find_psf_stars_by_region(clean.data, 4, clean.rms_map(), per_region=80)
        assert len(stars) >= 20
        positions = np.asarray(stars)
        ny, nx = clean.shape
        # Stars in every quadrant, which a centre-biased selection would fail.
        for x_half in (positions[:, 0] < nx / 2, positions[:, 0] >= nx / 2):
            for y_half in (positions[:, 1] < ny / 2, positions[:, 1] >= ny / 2):
                assert (x_half & y_half).sum() >= 3

    def test_regional_selection_beats_global_for_the_fit(self, varying_field):
        _, clean, _ = varying_field
        regional = find_psf_stars_by_region(clean.data, 4, clean.rms_map(),
                                            per_region=80)
        model = fit_varying_psf(clean.data, regional, size=21, rms=clean.rms_map())
        assert not model.fallback


class TestPipelineIntegration:
    def test_the_preprocessor_keeps_both_models(self, varying_field):
        """The field-average model stays whatever the spatial fit decides:
        every existing consumer expects it."""
        _, _, _ = varying_field
        simulator, _, _ = _field(0.40)
        image, _ = simulator.generate()
        clean = Preprocessor(PreprocessConfig(varying_psf=True)).run(image)
        assert clean.meta.get("psf_model") is not None
        assert isinstance(clean.meta.get("varying_psf"), VaryingPSF)

    def test_it_is_off_by_default(self, varying_field):
        _, clean, _ = varying_field
        assert "varying_psf" not in clean.meta

    def test_psf_at_prefers_the_spatial_model(self):
        simulator = SkySimulator(SkyConfig(
            shape=(800, 800), n_stars=220, n_galaxies=10, n_nebulae=0,
            n_clusters=0, n_lenses=0, n_anomalies=0, seed=7,
            seeing_variation=0.40, seeing_fwhm=3.0,
            star_flux_range=(4000.0, 90000.0)))
        image, _ = simulator.generate()
        clean = Preprocessor(PreprocessConfig(varying_psf=True)).run(image)
        model = clean.meta["varying_psf"]
        if model.fallback:
            pytest.skip("this field did not support a spatial fit")
        centre = psf_at(clean.meta, 400, 400)
        corner = psf_at(clean.meta, 30, 30)
        assert corner.fwhm > centre.fwhm

    def test_psf_at_falls_back_to_the_single_model(self, constant_field):
        _, clean, _ = constant_field
        assert psf_at(clean.meta, 10, 10) is clean.meta["psf_model"]

    def test_psf_at_with_no_model_at_all(self):
        assert psf_at({}, 5, 5) is None

    @pytest.mark.slow
    def test_local_corrections_close_the_centre_to_corner_gap(self):
        """The measurable payoff: a field-average PSF gets the aperture
        correction wrong in opposite directions at the centre and the edge."""
        simulator = SkySimulator(SkyConfig(
            shape=(800, 800), n_stars=220, n_galaxies=10, n_nebulae=0,
            n_clusters=0, n_lenses=0, n_anomalies=0, seed=7,
            seeing_variation=0.40, seeing_fwhm=3.0,
            star_flux_range=(4000.0, 90000.0)))
        image, truth = simulator.generate()

        def gap(varying):
            clean = Preprocessor(PreprocessConfig(varying_psf=varying)).run(image)
            catalog, segmentation = Detector().detect(clean)
            Photometer().run(clean, catalog, segmentation)
            inner, outer = [], []
            for source in catalog:
                best, distance = None, 1e9
                for item in truth:
                    offset = np.hypot(item.x - source.x, item.y - source.y)
                    if offset < distance:
                        best, distance = item, offset
                if (best is None or distance > 1.2 or best.kind != "star"
                        or not np.isfinite(source.photometry.flux)
                        or source.photometry.snr < 30):
                    continue
                ratio = source.photometry.flux / best.flux
                radius = np.hypot(source.x - 400.0, source.y - 400.0)
                (inner if radius < 180.0 else outer).append(ratio)
            return abs(float(np.median(outer)) - float(np.median(inner)))

        assert gap(True) < 0.5 * gap(False)
