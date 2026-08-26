"""Preprocessing, detection, photometry, segmentation, morphology, classification.

These tests check the stages against the simulator's truth table rather
than against fixed numbers, so they assert *scientific* behaviour: the
recovered flux matches what was injected, the PSF matches the seeing, and
the recovered morphological statistics fall where the literature says they
should for each galaxy type.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.classify import Classifier
from astrovision.core.config import DetectionConfig
from astrovision.core.types import Morphology, ObjectClass
from astrovision.detect import Detector, extract_sources
from astrovision.detect.deblend import deblend_segment
from astrovision.detect.labeling import label
from astrovision.morphology import MorphologyAnalyzer
from astrovision.morphology.cas import asymmetry, concentration
from astrovision.morphology.gini_m20 import gini_coefficient, gini_m20
from astrovision.morphology.sersic import fit_sersic
from astrovision.morphology.spiral import detect_bar, detect_spiral_arms
from astrovision.photometry import Photometer
from astrovision.photometry.aperture import aperture_photometry, circular_aperture_weights
from astrovision.photometry.growth import concentration_index, curve_of_growth
from astrovision.preprocess import Preprocessor
from astrovision.preprocess.align import Transform, cross_correlation_shift, warp
from astrovision.preprocess.background import estimate_background
from astrovision.preprocess.calibrate import detect_cosmic_rays
from astrovision.preprocess.psf import measure_fwhm
from astrovision.segment import Segmenter, decompose, watershed_split
from astrovision.simulate import SkyConfig, SkySimulator
from astrovision.simulate.profiles import (
    elliptical_radius,
    gaussian_psf,
    sersic_profile,
    supersample,
)


class TestPreprocess:
    def test_background_is_recovered(self):
        image, _ = SkySimulator(SkyConfig(
            shape=(256, 256), background=137.0, n_stars=30, n_galaxies=5,
            n_nebulae=0, n_clusters=0, n_lenses=0, n_anomalies=0,
            seed=3)).generate()
        background, rms = estimate_background(image.data, 64, 3)
        assert float(np.median(background)) == pytest.approx(137.0, rel=0.05)
        assert float(np.median(rms)) > 0

    def test_subtraction_leaves_a_near_zero_sky(self, synthetic_field):
        image, _ = synthetic_field
        clean = Preprocessor().run(image)
        # Faint undetected sources leave a small positive pedestal in any
        # real field; it must stay well inside the noise.
        assert abs(float(np.median(clean.data))) < 0.3 * float(np.median(clean.rms_map()))

    def test_cosmic_rays_are_found_and_stars_are_not(self):
        simulator = SkySimulator(SkyConfig(
            shape=(256, 256), n_stars=50, n_galaxies=5, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, cosmic_ray_rate=1e-4, seed=9))
        image, truth = simulator.generate()
        mask = detect_cosmic_rays(image.data, sigma=6.0, contrast=2.0)
        injected = image.meta["artifacts"]["cosmic_rays"]
        found = sum(1 for cr in injected
                    if mask[max(0, cr["y"] - 2):cr["y"] + 3,
                            max(0, cr["x"] - 2):cr["x"] + 3].any())
        assert found >= 0.8 * len(injected)

        bright = [t for t in truth if t.kind == "star" and t.flux > 2e4]
        hit = sum(1 for s in bright
                  if mask[max(0, int(s.y) - 1):int(s.y) + 2,
                          max(0, int(s.x) - 1):int(s.x) + 2].any())
        assert hit == 0, "cosmic-ray rejection must not remove real stars"

    @pytest.mark.parametrize("fwhm", [2.5, 4.0])
    def test_psf_fwhm_matches_the_seeing(self, fwhm):
        image, _ = SkySimulator(SkyConfig(
            shape=(320, 320), seeing_fwhm=fwhm, n_stars=100, n_galaxies=0,
            n_nebulae=0, n_clusters=0, n_lenses=0, n_anomalies=0,
            seed=4)).generate()
        clean = Preprocessor().run(image, estimate_psf=False)
        measured = measure_fwhm(clean.subtracted())
        assert measured == pytest.approx(fwhm, rel=0.15)

    def test_psf_rejects_galaxies(self):
        """A galaxy-rich field must not inflate the measured PSF."""
        image, _ = SkySimulator(SkyConfig(
            shape=(320, 320), seeing_fwhm=2.8, n_stars=40, n_galaxies=30,
            galaxy_flux_range=(2e4, 2e5), n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, seed=55)).generate()
        clean = Preprocessor().run(image, estimate_psf=False)
        assert measure_fwhm(clean.subtracted()) == pytest.approx(2.8, rel=0.2)

    @pytest.mark.parametrize("shift", [(3.0, -2.0), (0.5, 1.25)])
    def test_registration_recovers_a_known_shift(self, shift):
        image, _ = SkySimulator(SkyConfig(
            shape=(256, 256), n_stars=70, n_galaxies=5, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, cosmic_ray_rate=0, bad_column_count=0,
            seed=17)).generate()
        fill = float(np.median(image.data))
        moved = warp(image.data, Transform(dx=shift[0], dy=shift[1]), fill=fill)
        dx, dy = cross_correlation_shift(image.data, moved, upsample=20)
        # The returned shift is the correction, so it undoes the applied one.
        assert dx == pytest.approx(-shift[0], abs=0.3)
        assert dy == pytest.approx(-shift[1], abs=0.3)


class TestDetection:
    def test_labels_connected_regions(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True
        mask[6:9, 6:9] = True
        _, count = label(mask)
        assert count == 2

    def test_deblends_two_overlapping_sources(self):
        image = (gaussian_psf((60, 60), (22, 30), 7.0, 1000.0) +
                 gaussian_psf((60, 60), (38, 30), 7.0, 900.0))
        labels = deblend_segment(image, image > 20, 20.0)
        assert int(labels.max()) == 2

    def test_leaves_a_single_source_intact(self):
        image = gaussian_psf((60, 60), (30, 30), 7.0, 1000.0)
        assert int(deblend_segment(image, image > 20, 20.0).max()) == 1

    def test_recovers_bright_isolated_stars(self):
        simulator = SkySimulator(SkyConfig(
            shape=(384, 384), n_stars=120, n_galaxies=0, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, cosmic_ray_rate=0, bad_column_count=0,
            seed=31))
        image, truth = simulator.generate()
        clean = Preprocessor().run(image)
        catalog, _ = Detector().detect(clean)
        positions = catalog.positions()

        rms = float(np.median(clean.rms_map()))
        fwhm = clean.meta["psf"]["fwhm"]
        area = np.pi * (fwhm / 2.355) ** 2 * 4
        bright = [t for t in truth if t.flux / (rms * np.sqrt(area)) > 10]
        found = sum(1 for t in bright
                    if np.hypot(positions[:, 0] - t.x, positions[:, 1] - t.y).min() < 3.0)
        assert found >= 0.85 * len(bright)

    def test_spurious_rate_is_low(self):
        simulator = SkySimulator(SkyConfig(
            shape=(384, 384), n_stars=100, n_galaxies=0, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, cosmic_ray_rate=0, bad_column_count=0,
            seed=32))
        image, truth = simulator.generate()
        catalog, _ = Detector(DetectionConfig(threshold_sigma=4.0)).detect(
            Preprocessor().run(image))
        truth_positions = np.array([[t.x, t.y] for t in truth])
        spurious = sum(
            1 for source in catalog
            if np.hypot(truth_positions[:, 0] - source.x,
                        truth_positions[:, 1] - source.y).min() >= 3.0)
        assert spurious <= 0.1 * max(len(catalog), 1)

    def test_higher_threshold_finds_fewer_sources(self, clean_image):
        low, _ = extract_sources(clean_image, DetectionConfig(threshold_sigma=3.0))
        high, _ = extract_sources(clean_image, DetectionConfig(threshold_sigma=6.0))
        assert len(high) <= len(low)

    def test_positions_are_sub_pixel(self, clean_image, detected):
        catalog, _ = detected
        assert any(source.x != round(source.x) for source in catalog)


class TestPhotometry:
    @pytest.mark.parametrize("radius", [2.0, 3.5, 5.0, 8.0])
    def test_aperture_area_matches_geometry(self, radius):
        weights = circular_aperture_weights((40, 40), (20.0, 20.0), radius, 7)
        assert float(weights.sum()) == pytest.approx(np.pi * radius ** 2, rel=0.01)

    def test_flux_of_a_known_star(self):
        total = 10_000.0
        star = gaussian_psf((60, 60), (30.0, 30.0), 3.0, 1.0)
        star = star / star.sum() * total
        image = star + 100.0 + np.random.default_rng(0).normal(0.0, 3.0, (60, 60))
        result = aperture_photometry(image, (30.0, 30.0), 8.0, gain=2.0,
                                     annulus=(16.0, 24.0))
        assert result.flux == pytest.approx(total, rel=0.03)
        assert result.snr > 50

    def test_curve_of_growth_is_monotonic(self):
        star = gaussian_psf((60, 60), (30.0, 30.0), 4.0, 100.0)
        _, cumulative = curve_of_growth(star, (30.0, 30.0), np.arange(1, 20, 1.0))
        assert np.all(np.diff(cumulative) >= -1e-6)

    @pytest.mark.parametrize("n,expected", [(1.0, 2.7), (4.0, 4.9)])
    def test_concentration_matches_the_literature(self, n, expected):
        from astrovision.core.numeric import convolve, gaussian_kernel

        size = 160
        centre = (size / 2 - 0.5, size / 2 - 0.5)
        galaxy = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 1.0, 0.0), 100.0,
                                        10.0 * (s[0] / size), n), 5)
        galaxy = convolve(galaxy, gaussian_kernel(1.2))
        radii, cumulative = curve_of_growth(galaxy, centre, np.arange(1, 70, 1.0))
        assert concentration_index(radii, cumulative) == pytest.approx(expected, abs=0.5)

    @pytest.mark.slow
    def test_recovers_injected_star_fluxes(self):
        simulator = SkySimulator(SkyConfig(
            shape=(256, 256), n_stars=60, n_galaxies=8, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, seed=88))
        image, truth = simulator.generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)
        positions = catalog.positions()

        ratios = []
        for star in truth:
            if star.kind != "star" or star.flux < 2000:
                continue
            distance = np.hypot(positions[:, 0] - star.x, positions[:, 1] - star.y)
            if distance.min() >= 2.0:
                continue
            source = catalog[int(distance.argmin())]
            if "edge" in source.flags:
                continue
            neighbours = min(np.hypot(o.x - star.x, o.y - star.y)
                             for o in truth if o is not star)
            if neighbours < 12:
                continue
            ratios.append(source.photometry.flux / star.flux)

        assert len(ratios) >= 5
        assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.08)

    def test_limiting_magnitude_is_reported(self, clean_image, measured):
        catalog, segmentation = measured
        photometer = Photometer()
        photometer.run(clean_image, catalog, segmentation)
        assert np.isfinite(photometer.report["limiting_magnitude_5sigma"])


class TestSegmentation:
    def test_watershed_splits_a_blend(self):
        image = (gaussian_psf((60, 60), (22, 30), 7.0, 1000.0) +
                 gaussian_psf((60, 60), (38, 30), 7.0, 900.0))
        assert int(watershed_split(image, image > 15).max()) == 2

    def test_bulge_fraction_orders_by_sersic_index(self):
        radius = elliptical_radius((100, 100), (50, 50), 0.8, 30.0)
        bulgey = decompose(sersic_profile(radius, 100.0, 10.0, 4.0),
                           sersic_profile(radius, 100.0, 10.0, 4.0) > 1.0)
        discy = decompose(sersic_profile(radius, 100.0, 10.0, 1.0),
                          sersic_profile(radius, 100.0, 10.0, 1.0) > 1.0)
        assert bulgey.bulge_to_total > discy.bulge_to_total

    def test_stage_records_components(self, clean_image, measured):
        catalog, segmentation = measured
        components = Segmenter().run(clean_image, catalog, segmentation)
        assert isinstance(components, dict)


class TestMorphology:
    def _render(self, n=1.0, spiral=False, size=140, r_eff=14.0):
        from astrovision.core.numeric import convolve, gaussian_kernel
        from astrovision.simulate.profiles import spiral_pattern

        centre = (size / 2 - 0.5, size / 2 - 0.5)
        galaxy = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 1.0, 0.0), 100.0,
                                        r_eff * (s[0] / size), n), 3)
        if spiral:
            galaxy = galaxy * spiral_pattern((size, size), centre, r_eff, 2, 20.0, 0.7)
        return convolve(galaxy, gaussian_kernel(1.2)), centre

    def test_gini_of_a_uniform_image_is_zero(self):
        assert gini_coefficient(np.ones(200)) == pytest.approx(0.0, abs=1e-9)

    def test_elliptical_is_more_concentrated_than_a_disc(self):
        early, centre = self._render(4.0)
        late, _ = self._render(1.0)
        assert (concentration(early, centre)["concentration"] >
                concentration(late, centre)["concentration"])

    def test_gini_separates_early_from_late_types(self):
        early, _ = self._render(4.0)
        late, _ = self._render(1.0)
        early_stats = gini_m20(early, early > np.percentile(early, 85))
        late_stats = gini_m20(late, late > np.percentile(late, 85))
        assert early_stats["gini"] > late_stats["gini"]

    def test_symmetric_galaxy_has_low_asymmetry(self):
        galaxy, centre = self._render(4.0)
        assert asymmetry(galaxy, centre)["asymmetry"] < 0.05

    def test_merger_has_high_asymmetry(self):
        galaxy, centre = self._render(1.0)
        merger = galaxy + 0.6 * np.roll(np.roll(galaxy, 14, axis=0), 10, axis=1)
        assert (asymmetry(merger, centre)["asymmetry"] >
                asymmetry(galaxy, centre)["asymmetry"])

    @pytest.mark.parametrize("n_true", [1.0, 2.5, 4.0])
    def test_sersic_index_is_recovered(self, n_true):
        from astrovision.core.numeric import SIGMA_TO_FWHM, convolve, gaussian_kernel
        from astrovision.photometry.growth import flux_radius

        fwhm = 2.8
        kernel = gaussian_kernel(fwhm / SIGMA_TO_FWHM, 15)
        size = 140
        centre = (size / 2 - 0.5, size / 2 - 0.5)
        galaxy = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 1.0, 0.0), 1.0,
                                        8.0 * (s[0] / size), n_true), 3)
        galaxy = convolve(galaxy / galaxy.sum() * 3e4, kernel)
        noisy = galaxy + np.random.default_rng(0).normal(0.0, 8.0, galaxy.shape)
        radii, cumulative = curve_of_growth(noisy, centre, np.linspace(1, 60, 28))
        fit = fit_sersic(noisy, centre, noisy > 24.0, psf=kernel, psf_fwhm=fwhm,
                         r_half=flux_radius(radii, cumulative, 0.5))
        assert fit.success
        assert fit.n == pytest.approx(n_true, rel=0.35)

    def test_finds_two_spiral_arms(self):
        galaxy, centre = self._render(1.0, spiral=True)
        result = detect_spiral_arms(galaxy, centre, max_radius=50.0, noise=0.0)
        assert result["arm_count"] == 2
        assert result["spiral_strength"] > 0.0

    def test_smooth_galaxy_shows_no_arms(self):
        galaxy, centre = self._render(4.0)
        assert detect_spiral_arms(galaxy, centre, max_radius=50.0)["arm_count"] == 0

    def test_bar_is_distinguished_from_arms(self):
        from astrovision.core.numeric import convolve, gaussian_kernel
        from astrovision.simulate.profiles import bar_pattern, spiral_pattern

        size, r_eff = 140, 14.0
        centre = (size / 2 - 0.5, size / 2 - 0.5)
        disc = supersample(
            (size, size), centre,
            lambda s, c: sersic_profile(elliptical_radius(s, c, 1.0, 0.0), 100.0,
                                        r_eff * (s[0] / size), 1.0), 3)
        barred = convolve(disc + disc.max() * 0.5 * bar_pattern(
            (size, size), centre, r_eff * 1.1, r_eff * 0.25, 30.0, 1.0),
            gaussian_kernel(1.2))
        armed = convolve(disc * spiral_pattern((size, size), centre, r_eff, 2, 20.0, 0.7),
                         gaussian_kernel(1.2))
        assert detect_bar(barred, centre, max_radius=50.0)["bar_detected"]
        assert not detect_bar(armed, centre, max_radius=50.0)["bar_detected"]

    @pytest.mark.slow
    def test_stage_runs_on_a_real_field(self, clean_image, measured):
        catalog, segmentation = measured
        MorphologyAnalyzer().run(clean_image, catalog, segmentation)
        labelled = [s for s in catalog
                    if s.morphology.label not in (Morphology.UNKNOWN,
                                                  Morphology.UNRESOLVED)]
        assert labelled, "at least one source should get a morphological type"


class TestClassification:
    @pytest.mark.slow
    def test_separates_stars_from_galaxies(self):
        simulator = SkySimulator(SkyConfig(
            shape=(384, 384), n_stars=80, n_galaxies=20, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, seed=7))
        image, truth = simulator.generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)
        Segmenter().run(clean, catalog, segmentation)
        MorphologyAnalyzer().run(clean, catalog, segmentation)
        Classifier().run(clean, catalog)

        positions = catalog.positions()
        correct = total = 0
        for entry in truth:
            if entry.kind not in ("star", "galaxy"):
                continue
            distance = np.hypot(positions[:, 0] - entry.x, positions[:, 1] - entry.y)
            if distance.min() >= 3.0:
                continue
            source = catalog[int(distance.argmin())]
            total += 1
            correct += source.object_class.value == entry.kind
        assert total > 20
        assert correct / total >= 0.8

    def test_every_source_gets_a_class(self, clean_image, measured):
        catalog, _ = measured
        Classifier().run(clean_image, catalog)
        assert all(s.object_class in set(ObjectClass) for s in catalog)
