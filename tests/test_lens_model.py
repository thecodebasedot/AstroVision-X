"""The lens mass model: deflections, ray tracing, the fit, and the mass."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.detect.labeling import label
from astrovision.lensing import (
    MAX_PLAUSIBLE_SHEAR,
    MIN_SHEAR_SPAN_DEG,
    LensModel,
    arc_sample_points,
    azimuthal_span,
    einstein_mass,
    fit_lens_model,
    image_plane_residual,
    ray_trace,
    shear_deflection,
    sie_deflection,
    sis_deflection,
    source_plane_scatter,
)


class TestDeflections:
    def test_the_sphere_deflects_by_a_constant(self):
        """The defining property of an isothermal sphere: every ray is bent by
        the same angle, whatever its impact parameter."""
        dx = np.array([1.0, 5.0, 20.0, -13.0])
        dy = np.array([0.0, 12.0, -3.0, 7.0])
        ax, ay = sis_deflection(dx, dy, 9.0)
        assert np.allclose(np.hypot(ax, ay), 9.0)

    def test_the_ellipsoid_reduces_to_the_sphere_when_round(self):
        """Both SIE components carry a 1/sqrt(1-q^2) that diverges as q -> 1,
        so the round case has to be handled rather than approached."""
        dx, dy = np.array([3.0, 7.0, -5.0]), np.array([4.0, -2.0, 9.0])
        for q in (1.0, 1.0 - 1e-9):
            ax, ay = sie_deflection(dx, dy, 10.0, q, 30.0)
            sx, sy = sis_deflection(dx, dy, 10.0)
            assert np.allclose(ax, sx) and np.allclose(ay, sy)

    def test_a_round_lens_maps_its_einstein_ring_to_the_centre(self):
        """The definition of the Einstein radius."""
        model = LensModel(x0=50.0, y0=50.0, theta_e=12.0, axis_ratio=1.0)
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        bx, by = model.source_plane(50 + 12 * np.cos(angles), 50 + 12 * np.sin(angles))
        assert np.allclose(bx, 50.0, atol=1e-9)
        assert np.allclose(by, 50.0, atol=1e-9)

    def test_the_position_angle_rotates_the_deflection(self):
        dx, dy = np.array([10.0]), np.array([0.0])
        flat = sie_deflection(dx, dy, 10.0, 0.6, 0.0)
        turned = sie_deflection(dx, dy, 10.0, 0.6, 90.0)
        assert not np.allclose(flat, turned)

    def test_shear_is_traceless(self):
        """Shear stretches without adding mass: its deflection has zero
        divergence, so it contributes no convergence."""
        step = 1e-4
        for g1, g2 in ((0.05, 0.0), (0.0, 0.05), (0.03, -0.04)):
            dx1, _ = shear_deflection(np.array([step]), np.array([0.0]), g1, g2)
            dx0, _ = shear_deflection(np.array([-step]), np.array([0.0]), g1, g2)
            _, dy1 = shear_deflection(np.array([0.0]), np.array([step]), g1, g2)
            _, dy0 = shear_deflection(np.array([0.0]), np.array([-step]), g1, g2)
            divergence = float((dx1[0] - dx0[0]) / (2 * step)
                               + (dy1[0] - dy0[0]) / (2 * step))
            assert abs(divergence) < 1e-9

    def test_magnification_diverges_on_the_critical_curve(self):
        model = LensModel(x0=50.0, y0=50.0, theta_e=12.0, axis_ratio=1.0)
        on_ring = abs(float(model.magnification(62.0, 50.0)))
        far = abs(float(model.magnification(90.0, 50.0)))
        assert on_ring > 5.0 * far

    def test_the_critical_curve_sits_near_the_einstein_radius(self):
        model = LensModel(x0=50.0, y0=50.0, theta_e=12.0, axis_ratio=0.8,
                          position_angle=20.0)
        mask = model.critical_curve((101, 101))
        yy, xx = np.nonzero(mask)
        radii = np.hypot(xx - 50.0, yy - 50.0)
        assert 8.0 < float(np.median(radii)) < 16.0


class TestRayTracing:
    def test_flux_is_conserved(self):
        model = LensModel(x0=50.0, y0=50.0, theta_e=12.0, axis_ratio=0.8)
        image = ray_trace((101, 101), model, 51.0, 50.5, 1.5, 1000.0)
        assert image.sum() == pytest.approx(1000.0, rel=1e-6)

    def test_a_source_on_the_axis_makes_a_ring(self):
        """The classic configuration, produced by the mass rather than drawn."""
        model = LensModel(x0=50.0, y0=50.0, theta_e=13.0, axis_ratio=1.0)
        image = ray_trace((101, 101), model, 50.0, 50.0, 1.2, 1000.0)
        bright = image > 0.15 * image.max()
        yy, xx = np.nonzero(bright)
        radii = np.hypot(xx - 50.0, yy - 50.0)
        assert float(radii.min()) > 8.0        # hollow in the middle
        assert 11.0 < float(np.median(radii)) < 15.0

    def test_an_offset_source_makes_multiple_images(self):
        model = LensModel(x0=60.0, y0=60.0, theta_e=13.0, axis_ratio=0.7,
                          position_angle=25.0)
        image = ray_trace((121, 121), model, 64.0, 62.0, 1.6, 1000.0)
        _, count = label(image > 0.15 * image.max())
        assert count >= 2

    def test_a_source_far_outside_the_caustic_is_not_multiply_imaged(self):
        model = LensModel(x0=60.0, y0=60.0, theta_e=13.0, axis_ratio=0.7)
        image = ray_trace((121, 121), model, 85.0, 75.0, 1.6, 1000.0)
        _, count = label(image > 0.15 * image.max())
        assert count == 1


class TestAzimuthalSpan:
    def test_a_full_ring_spans_the_sky(self):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        points = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        assert azimuthal_span(points, (0.0, 0.0)) > 330.0

    def test_one_short_arc_does_not(self):
        angles = np.linspace(0.0, 0.4, 8)
        points = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        assert azimuthal_span(points, (0.0, 0.0)) < 40.0

    def test_a_single_point_spans_nothing(self):
        assert azimuthal_span(np.array([[1.0, 1.0]]), (0.0, 0.0)) == 0.0


class TestFitting:
    @staticmethod
    def _exact_images(truth: LensModel, source, tolerance=0.06, step=0.1):
        grid = np.arange(20.0, 101.0, step)
        gx, gy = np.meshgrid(grid, grid)
        bx, by = truth.source_plane(gx, gy)
        inside = np.hypot(bx - source[0], by - source[1]) < tolerance
        return np.column_stack([gx[inside], gy[inside]])

    @pytest.fixture(scope="class")
    def exact(self):
        truth = LensModel(x0=60.0, y0=60.0, theta_e=14.0, axis_ratio=0.70,
                          position_angle=35.0, shear1=0.03, shear2=-0.02)
        return truth, self._exact_images(truth, (61.2, 60.6))

    def test_it_finds_the_true_model_from_exact_positions(self, exact):
        """With constraints whose exact solution is known, the fit must reach
        it -- otherwise nothing measured on real arcs means anything."""
        truth, points = exact
        assert len(points) >= 8
        fit = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0, bootstrap=0)
        assert fit.succeeded
        assert fit.model.theta_e == pytest.approx(truth.theta_e, rel=0.02)
        assert fit.model.axis_ratio == pytest.approx(truth.axis_ratio, abs=0.05)
        assert fit.model.shear_magnitude == pytest.approx(truth.shear_magnitude,
                                                          abs=0.02)

    def test_the_fit_reaches_the_scatter_of_the_true_model(self, exact):
        truth, points = exact
        fit = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0, bootstrap=0)
        assert fit.source_rms <= 1.5 * source_plane_scatter(points, truth)

    def test_omitting_shear_flattens_the_mass(self, exact):
        """The reason external shear is in the model: without it the fit has
        to explain the tidal stretch with the ellipsoid."""
        truth, points = exact
        with_shear = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0,
                                    bootstrap=0)
        without = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0,
                                 fit_shear=False, bootstrap=0)
        assert abs(without.model.axis_ratio - truth.axis_ratio) > \
            abs(with_shear.model.axis_ratio - truth.axis_ratio)

    def test_shear_is_held_at_zero_without_azimuthal_coverage(self):
        """Shear and ellipticity both stretch images; telling them apart needs
        to see the stretch from more than one direction."""
        angles = np.linspace(0.0, 0.5, 10)
        points = np.column_stack([60 + 14 * np.cos(angles), 60 + 14 * np.sin(angles)])
        fit = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0, bootstrap=0)
        assert azimuthal_span(points, (60.0, 60.0)) < MIN_SHEAR_SPAN_DEG
        assert "shear_fixed_to_zero" in fit.flags
        assert fit.model.shear_magnitude == 0.0

    def test_a_shear_the_data_do_not_need_is_dropped(self):
        """A large fitted shear is either a real tidal field or the lens's own
        flattening counted twice.  What separates them is what the shear buys:
        remove it, refit, and see whether the images still line up."""
        truth = LensModel(x0=60.0, y0=60.0, theta_e=14.0, axis_ratio=0.70,
                          position_angle=35.0, shear1=0.03, shear2=0.02)
        # A source far enough out that its images sit on one side of the lens:
        # the geometry where ellipticity and shear are least separable.
        points = self._exact_images(truth, (63.0, 61.5), tolerance=0.2, step=0.2)
        assert len(points) >= 8
        loose = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0,
                               bootstrap=0)
        # Either the coverage gate stopped it or the evidence test dropped it;
        # what must not happen is a nearly round lens in a violent shear.
        assert loose.model.shear_magnitude <= MAX_PLAUSIBLE_SHEAR
        assert loose.model.axis_ratio > 0.4

    def test_a_shear_the_data_do_insist_on_is_kept(self):
        """The other half of the same rule.  A lens genuinely sitting in a
        strong tidal field must keep it -- dropping it would replace a good
        model with a bad one."""
        truth = LensModel(x0=60.0, y0=60.0, theta_e=14.0, axis_ratio=0.85,
                          position_angle=20.0, shear1=0.45, shear2=0.0)
        points = self._exact_images(truth, (60.6, 60.4))
        assert len(points) >= 6
        fit = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0, bootstrap=0)
        assert fit.model.shear_magnitude > MAX_PLAUSIBLE_SHEAR
        assert "implausible_shear" in fit.flags
        assert fit.model.theta_e == pytest.approx(truth.theta_e, rel=0.06)

    def test_it_refuses_with_fewer_constraints_than_parameters(self):
        points = np.array([[70.0, 60.0], [50.0, 60.0]])
        fit = fit_lens_model(points, (60.0, 60.0), bootstrap=0)
        assert not fit.succeeded
        assert "refusing to fit" in fit.reason
        assert fit.model is None

    def test_the_bootstrap_gives_an_error(self, exact):
        _, points = exact
        fit = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0, bootstrap=12)
        assert np.isfinite(fit.theta_e_error)
        assert fit.theta_e_error > 0.0

    def test_the_image_plane_residual_is_small_for_a_good_model(self, exact):
        truth, points = exact
        bx, by = truth.source_plane(points[:, 0], points[:, 1])
        residual = image_plane_residual(points, truth, (float(bx.mean()),
                                                        float(by.mean())))
        assert residual < 1.0

    def test_the_fit_serialises(self, exact):
        _, points = exact
        payload = fit_lens_model(points, (60.0, 60.0), theta_e_guess=14.0,
                                 bootstrap=0).to_dict()
        assert set(payload) >= {"succeeded", "source_rms", "image_rms", "model",
                                "azimuthal_span"}
        assert set(payload["model"]) >= {"theta_e", "axis_ratio", "shear_magnitude"}


class TestArcSampling:
    def test_the_real_ridge_is_preferred(self):
        """Points rebuilt from a radius and an angle lie on a perfect circle,
        which a round lens reproduces exactly -- a fit given those can only
        measure a radius and calls every lens circular."""
        class FakeArc:
            radius, angle, length = 12.0, 30.0, 20.0
            points = np.array([[70.0, 61.0], [69.0, 64.0], [67.0, 67.0]])

        sampled = arc_sample_points([FakeArc()], (60.0, 60.0))
        assert len(sampled) == 3
        assert np.allclose(sampled, FakeArc.points)

    def test_it_falls_back_to_the_nominal_circle(self):
        class RingScan:
            radius, angle, length = 12.0, 0.0, 75.0
            points = np.zeros((0, 2))

        sampled = arc_sample_points([RingScan()], (60.0, 60.0), per_arc=5)
        assert len(sampled) == 5
        assert np.allclose(np.hypot(sampled[:, 0] - 60.0, sampled[:, 1] - 60.0), 12.0)


class TestEinsteinMass:
    def test_a_typical_galaxy_lens_lands_in_the_right_decade(self):
        """A one-arcsecond lens at z=0.5 behind a source at z=2 is the
        best-observed configuration there is, and it weighs a few times
        10^11 solar masses."""
        mass = einstein_mass(1.0, 0.5, 2.0)
        assert 11.0 < mass["log_mass_solar"] < 11.8
        assert 180.0 < mass["velocity_dispersion_km_s"] < 300.0
        assert 3.0 < mass["einstein_radius_kpc"] < 10.0

    def test_mass_scales_as_the_square_of_the_radius(self):
        one = einstein_mass(1.0, 0.5, 2.0)["mass_solar"]
        two = einstein_mass(2.0, 0.5, 2.0)["mass_solar"]
        assert two / one == pytest.approx(4.0, rel=1e-6)

    def test_it_refuses_an_impossible_geometry(self):
        """A source in front of the lens does not get lensed by it."""
        mass = einstein_mass(1.0, 2.0, 0.5)
        assert not np.isfinite(mass["mass_solar"])
        assert "z_lens < z_source" in mass["reason"]

    def test_it_refuses_a_nonsense_radius(self):
        assert not np.isfinite(einstein_mass(-1.0, 0.5, 2.0)["mass_solar"])


class TestSimulatorAndSearch:
    @pytest.fixture(scope="class")
    def searched(self):
        from astrovision.classify import Classifier
        from astrovision.detect import Detector
        from astrovision.lensing import LensSearch
        from astrovision.morphology import MorphologyAnalyzer
        from astrovision.photometry import Photometer
        from astrovision.preprocess import Preprocessor
        from astrovision.simulate import SkyConfig, SkySimulator

        config = SkyConfig(shape=(260, 260), n_stars=45, n_galaxies=14, n_nebulae=0,
                           n_clusters=0, n_lenses=2, n_anomalies=0, seed=11,
                           pixel_scale=0.4)
        image, truth = SkySimulator(config).generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)
        MorphologyAnalyzer().run(clean, catalog, segmentation)
        Classifier().run(clean, catalog)
        return LensSearch().run(clean, catalog), catalog, truth

    def test_the_simulator_ray_traces_its_lenses(self):
        from astrovision.simulate import SkyConfig, SkySimulator

        simulator = SkySimulator(SkyConfig(shape=(160, 160), seeing_fwhm=3.0))
        canvas = np.zeros((160, 160))
        truth = simulator.add_lens_system(canvas, 80.0, 80.0, 60000.0)
        assert truth.meta["ray_traced"] is True
        assert canvas.sum() == pytest.approx(60000.0, rel=0.05)

    def test_the_painted_path_is_still_available(self):
        from astrovision.simulate import SkyConfig, SkySimulator

        simulator = SkySimulator(SkyConfig(shape=(160, 160), seeing_fwhm=3.0))
        canvas = np.zeros((160, 160))
        truth = simulator.add_lens_system(canvas, 80.0, 80.0, 60000.0,
                                          ray_traced=False)
        assert truth.meta["ray_traced"] is False
        assert truth.meta["n_arcs"] >= 2

    def test_candidates_carry_a_model_and_a_mass(self, searched):
        candidates, _, _ = searched
        assert candidates
        modelled = [c for c in candidates if c.model.get("model")]
        assert modelled
        for candidate in modelled:
            assert np.isfinite(candidate.model_theta_e_arcsec)
            assert np.isfinite(candidate.mass.get("log_mass_solar", float("nan")))

    def test_the_notes_say_which_redshifts_were_assumed(self, searched):
        candidates, _, _ = searched
        modelled = [c for c in candidates if c.mass.get("mass_solar")]
        assert modelled
        text = " ".join(modelled[0].notes)
        assert "assumed" in text
        assert modelled[0].mass["z_source_source"] == "assumed"

    def test_an_unmodellable_candidate_is_still_a_candidate(self, searched):
        """A candidate that cannot be modelled keeps its detection; the
        failure is recorded rather than removing it."""
        candidates, _, _ = searched
        for candidate in candidates:
            assert candidate.score > 0
            if not candidate.model.get("model"):
                assert any("No mass model" in note for note in candidate.notes)
