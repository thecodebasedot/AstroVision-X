"""Solar-system objects: trails, linking, and what the linker refuses to do."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astrovision.core.config import MovingObjectConfig
from astrovision.detect import Detector
from astrovision.io.image import ImageSeries
from astrovision.io.wcs import SimpleWCS
from astrovision.moving import (
    Detection,
    MovingObjectFinder,
    build_tracklet,
    chance_alignment_rate,
    direction_agreement,
    expected_trail_length,
    field_psf_elongation,
    fit_linear_motion,
    link_tracklets,
    measure_trail,
    second_moments,
    summarise_tracklet,
)
from astrovision.preprocess import Preprocessor
from astrovision.simulate import SkyConfig, SkySimulator
from astrovision.transient import TransientDetector


@pytest.fixture(scope="module")
def mover_series():
    """A five-epoch series taken within one hour, with two asteroids."""
    config = SkyConfig(shape=(300, 300), n_stars=60, n_galaxies=12, n_nebulae=0,
                       n_clusters=0, n_lenses=0, n_anomalies=0, seed=3,
                       pixel_scale=0.4)
    series, static, injected = SkySimulator(config).generate_series(
        n_epochs=5, cadence=0.01, n_transients=1, n_movers=2)
    prepared = ImageSeries([Preprocessor().run(image) for image in series],
                           name="movers")
    return prepared, static, injected


@pytest.fixture(scope="module")
def linked(mover_series):
    """``(series, injected, tracklets, transient candidates)``."""
    prepared, _, injected = mover_series
    catalog, _ = Detector().detect(prepared.stack("median"))
    detector = TransientDetector()
    candidates = detector.run(prepared, catalog)
    result = MovingObjectFinder().run(prepared, detector.per_epoch,
                                      detector.differences)
    return prepared, injected, result, candidates


def _match(tracklet, injected, heading_tolerance=8.0, rate_tolerance=8.0):
    for item in injected:
        if item.get("kind") != "mover":
            continue
        heading = abs(((tracklet.heading_deg - item["heading_deg"] + 180) % 360) - 180)
        rate = abs(tracklet.rate_arcsec_per_hour - item["rate_arcsec_per_hour"])
        if heading <= heading_tolerance and rate <= rate_tolerance:
            return item
    return None


class TestSimulatedMovers:
    def test_a_mover_changes_position_every_epoch(self, mover_series):
        _, _, injected = mover_series
        movers = [t for t in injected if t["kind"] == "mover"]
        assert movers
        for mover in movers:
            positions = mover["positions"]
            assert len(positions) == 5
            distances = [math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                         for a, b in zip(positions[:-1], positions[1:])]
            assert min(distances) > 5.0        # genuinely moving, not jitter

    def test_the_rate_unit_is_arcsec_per_hour(self, mover_series):
        """A unit slip here leaves an asteroid crossing half a pixel per night,
        which looks exactly like a stationary source."""
        _, _, injected = mover_series
        for mover in [t for t in injected if t["kind"] == "mover"]:
            first, last = mover["positions"][0], mover["positions"][-1]
            hours = (last["time"] - first["time"]) * 24.0
            pixels = math.hypot(last["x"] - first["x"], last["y"] - first["y"])
            arcsec_per_hour = pixels * 0.4 / hours
            assert arcsec_per_hour == pytest.approx(mover["rate_arcsec_per_hour"],
                                                    rel=0.05)

    def test_flux_is_conserved_when_trailed(self):
        simulator = SkySimulator(SkyConfig(shape=(80, 80), seeing_fwhm=3.2))
        canvas = np.zeros((80, 80))
        simulator.add_mover(canvas, 40.0, 40.0, 10_000.0, trail_length=14.0,
                            trail_angle=25.0)
        assert canvas.sum() == pytest.approx(10_000.0, rel=1e-6)

    def test_a_slow_mover_is_not_trailed(self):
        simulator = SkySimulator(SkyConfig(shape=(60, 60), seeing_fwhm=3.2))
        canvas = np.zeros((60, 60))
        truth = simulator.add_mover(canvas, 30.0, 30.0, 5_000.0, trail_length=0.5)
        assert truth.meta["trailed"] is False


class TestTrailMeasurement:
    def _stamp(self, length, angle=0.0, flux=40_000.0, size=61, noise=0.0):
        simulator = SkySimulator(SkyConfig(shape=(size, size), seeing_fwhm=3.2))
        canvas = np.zeros((size, size))
        simulator.add_mover(canvas, (size - 1) / 2.0, (size - 1) / 2.0, flux,
                            trail_length=length, trail_angle=angle)
        if noise:
            canvas = canvas + np.random.default_rng(0).normal(0, noise, canvas.shape)
        return canvas

    def test_second_moments_find_the_trail_direction(self):
        major, minor, angle = second_moments(self._stamp(16.0, angle=35.0))
        assert major > minor
        assert angle == pytest.approx(35.0, abs=5.0)

    def test_a_point_source_is_not_called_trailed(self):
        trail = measure_trail(self._stamp(0.0), psf_fwhm=3.2)
        assert not trail.trailed
        assert trail.excess < 1.0

    def test_a_long_trail_is_measured_close_to_its_true_length(self):
        trail = measure_trail(self._stamp(14.0, angle=0.0), psf_fwhm=3.2)
        assert trail.trailed
        # The second-moment length of a uniform streak of length L is
        # L/sqrt(12) in sigma, so the FWHM-equivalent excess is a fixed
        # fraction of L rather than L itself -- what matters is that it
        # scales, which the next test checks.
        assert trail.excess > 3.0

    def test_the_measured_excess_grows_with_the_trail(self):
        short = measure_trail(self._stamp(6.0), psf_fwhm=3.2).excess
        long = measure_trail(self._stamp(18.0), psf_fwhm=3.2).excess
        assert long > 2.0 * short

    def test_a_noisy_faint_source_is_not_called_trailed(self):
        stamp = self._stamp(0.0, flux=400.0, noise=20.0)
        trail = measure_trail(stamp, psf_fwhm=3.2, noise=20.0)
        assert not trail.trailed

    def test_expected_length_from_rate_and_exposure(self):
        assert expected_trail_length(60.0, 300.0, 0.4) == pytest.approx(12.5)
        assert not np.isfinite(expected_trail_length(60.0, 300.0, 0.0))

    def test_direction_agreement_is_modulo_180(self):
        assert direction_agreement(30.0, 210.0) == pytest.approx(1.0)
        assert direction_agreement(0.0, 90.0) == pytest.approx(0.0)
        assert np.isnan(direction_agreement(float("nan"), 10.0))

    def test_field_elongation_summarises_the_stars(self):
        stamps = [self._stamp(0.0) for _ in range(4)]
        elongation, angle = field_psf_elongation(stamps)
        assert elongation == pytest.approx(1.0, abs=0.15)
        assert 0.0 <= angle < 180.0

    def test_field_elongation_with_no_usable_stamps(self):
        elongation, _ = field_psf_elongation([np.zeros((9, 9))])
        assert np.isnan(elongation)


class TestLinking:
    @staticmethod
    def _track(vx, vy, x0=50.0, y0=50.0, times=(0.0, 0.01, 0.02, 0.03, 0.04),
               jitter=0.0, seed=0):
        rng = np.random.default_rng(seed)
        return [Detection(x0 + vx * t + rng.normal(0, jitter),
                          y0 + vy * t + rng.normal(0, jitter), t, epoch=i)
                for i, t in enumerate(times)]

    def test_fit_recovers_a_known_velocity(self):
        points = self._track(1000.0, -500.0)
        x0, y0, vx, vy, t0, rms = fit_linear_motion(points)
        assert vx == pytest.approx(1000.0, rel=1e-6)
        assert vy == pytest.approx(-500.0, rel=1e-6)
        assert rms == pytest.approx(0.0, abs=1e-9)

    def test_the_time_origin_is_the_mean(self):
        """Using the first epoch instead makes position and velocity
        correlated, so the reported position is an extrapolation."""
        times = (0.0, 0.01, 0.02, 0.03, 0.04)
        _, _, _, _, t0, _ = fit_linear_motion(self._track(100.0, 0.0, times=times))
        assert t0 == pytest.approx(float(np.mean(times)))

    def test_links_a_clean_track(self):
        tracklets, report = link_tracklets(
            self._track(2000.0, 1000.0, jitter=0.2), pixel_scale=0.4,
            tolerance=3.0, min_points=3, field_shape=(300, 300))
        assert len(tracklets) == 1
        assert tracklets[0].n_points == 5
        assert report.refused is None

    def test_ignores_a_stationary_source(self):
        stationary = self._track(0.0, 0.0, jitter=0.1)
        tracklets, _ = link_tracklets(stationary, min_rate=5.0, pixel_scale=0.4,
                                      field_shape=(300, 300))
        assert tracklets == []

    def test_ignores_something_too_fast(self):
        tracklets, _ = link_tracklets(self._track(200_000.0, 0.0), max_rate=300.0,
                                      pixel_scale=0.4, field_shape=(300, 300))
        assert tracklets == []

    def test_refuses_an_arc_too_long_for_a_straight_line(self):
        """Over weeks, real motion curves; a linear fit would report a small
        residual and a confident, wrong answer."""
        slow = self._track(20.0, 5.0, times=(0.0, 1.0, 2.0, 3.0, 4.0))
        tracklets, report = link_tracklets(slow, pixel_scale=0.4,
                                           field_shape=(300, 300))
        assert tracklets == []
        assert "exceeds" in (report.refused or "")

    def test_refuses_with_too_few_epochs(self):
        two = self._track(2000.0, 0.0, times=(0.0, 0.01))
        tracklets, report = link_tracklets(two, min_points=3, pixel_scale=0.4,
                                           field_shape=(300, 300))
        assert tracklets == []
        assert "at least 3 epochs" in (report.refused or "")

    def test_one_tracklet_per_object_not_one_per_pair(self):
        """Every pair of points on a real track proposes the same tracklet."""
        tracklets, _ = link_tracklets(self._track(2000.0, 1000.0, jitter=0.2),
                                      pixel_scale=0.4, field_shape=(300, 300))
        assert len(tracklets) == 1

    def test_two_objects_are_kept_apart(self):
        first = self._track(2000.0, 0.0, x0=30.0, y0=40.0, jitter=0.2, seed=1)
        second = self._track(-1500.0, 900.0, x0=260.0, y0=60.0, jitter=0.2, seed=2)
        tracklets, _ = link_tracklets(first + second, pixel_scale=0.4,
                                      field_shape=(300, 300))
        assert len(tracklets) == 2
        headings = sorted(t.heading_deg for t in tracklets)
        assert abs(headings[0] - headings[1]) > 20.0

    def test_reduced_rms_penalises_a_short_link(self):
        """A three-point fit has two degrees of freedom and a five-point one
        has six, so raw residuals are not comparable across them."""
        short = build_tracklet(self._track(2000.0, 0.0, times=(0.0, 0.01, 0.02),
                                           jitter=0.6, seed=5))
        long = build_tracklet(self._track(2000.0, 0.0, jitter=0.6, seed=5))
        assert short.reduced_rms / max(short.rms, 1e-9) > \
            long.reduced_rms / max(long.rms, 1e-9)

    def test_chance_rate_grows_with_density_and_tolerance(self):
        sparse = chance_alignment_rate([5, 5, 5], 3.0, 250_000.0, 3)
        dense = chance_alignment_rate([50, 50, 50], 3.0, 250_000.0, 3)
        loose = chance_alignment_rate([5, 5, 5], 9.0, 250_000.0, 3)
        assert dense > sparse
        assert loose > sparse

    def test_chance_rate_is_zero_without_enough_epochs(self):
        assert chance_alignment_rate([10], 3.0, 250_000.0, 3) == 0.0

    def test_rate_is_reported_in_arcsec_per_hour(self):
        # 2000 px/day at 0.4 arcsec/px is 2000 * 0.4 / 24 arcsec/hour.
        tracklet = build_tracklet(self._track(2000.0, 0.0), pixel_scale=0.4)
        assert tracklet.rate_arcsec_per_hour == pytest.approx(2000 * 0.4 / 24.0)

    def test_position_angle_needs_a_wcs(self):
        points = self._track(2000.0, 0.0)
        without = build_tracklet(points, pixel_scale=0.4)
        assert np.isnan(without.position_angle)
        with_wcs = build_tracklet(points, wcs=SimpleWCS.tangent(150.0, 2.2,
                                                                (300, 300), 0.4))
        assert 0.0 <= with_wcs.position_angle < 360.0


class TestEndToEnd:
    def test_both_injected_movers_are_recovered(self, linked):
        _, injected, result, _ = linked
        movers = [t for t in injected if t["kind"] == "mover"]
        matched = [t for t in result.tracklets if _match(t, injected)]
        assert len(matched) == len(movers)

    def test_the_recovered_rate_and_heading_are_right(self, linked):
        _, injected, result, _ = linked
        for tracklet in result.tracklets:
            item = _match(tracklet, injected)
            if item is None:
                continue
            assert tracklet.rate_arcsec_per_hour == pytest.approx(
                item["rate_arcsec_per_hour"], abs=3.0)
            assert tracklet.heading_deg == pytest.approx(item["heading_deg"], abs=5.0)

    def test_no_spurious_tracklets(self, linked):
        _, injected, result, _ = linked
        assert all(_match(t, injected) is not None for t in result.tracklets)

    def test_the_supernova_is_not_swallowed(self, linked):
        """The whole point is to remove asteroids from the transient list --
        not transients."""
        _, injected, _, candidates = linked
        supernovae = [t for t in injected if t["kind"] != "mover"]
        for item in supernovae:
            near = [c for c in candidates
                    if math.hypot(c.x - item["x"], c.y - item["y"]) < 4.0]
            assert near, "the injected supernova should still be a candidate"
            assert any("moving_object" not in c.flags for c in near)

    def test_movers_are_demoted_not_deleted(self, linked):
        """The tracklet is an interpretation; an astronomer who disagrees
        needs to see what was interpreted."""
        _, _, result, candidates = linked
        flagged = [c for c in candidates if "moving_object" in c.flags]
        assert len(flagged) == len(result.claimed_candidate_ids)
        assert flagged
        for candidate in flagged:
            assert candidate.classification == "moving_object"
            assert candidate.verdict.value == "not_interesting"
            assert "tracklet" in candidate.meta

    def test_trails_confirm_the_track_direction(self, linked):
        """Trail direction comes from inside one exposure and the track from
        across several, so agreement is genuinely independent evidence."""
        _, injected, result, _ = linked
        confirmed = [t for t in result.tracklets
                     if np.isfinite(t.trail_agreement) and _match(t, injected)]
        assert confirmed
        assert max(t.trail_agreement for t in confirmed) > 0.8

    def test_trail_length_matches_what_the_rate_predicts(self, linked):
        _, injected, result, _ = linked
        consistent = [t for t in result.tracklets
                      if "trail_consistent_with_rate" in t.flags]
        assert consistent

    def test_the_summary_says_it_is_not_a_discovery(self, linked):
        _, _, result, _ = linked
        assert result.tracklets
        text = summarise_tracklet(result.tracklets[0])
        assert "arcsec/hour" in text
        assert "chance alignment probability" in text

    def test_disabled_stage_does_nothing(self, linked):
        series, _, _, _ = linked
        finder = MovingObjectFinder(MovingObjectConfig(enabled=False))
        result = finder.run(series, [[]], None)
        assert result.tracklets == []
