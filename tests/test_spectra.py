"""Spectroscopy: extraction, calibration, redshifts, lines and classification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astrovision.simulate.spectrograph import (
    ARC_LINES,
    SKY_LINES,
    SpectrographConfig,
    SpectrographSimulator,
)
from astrovision.spectra import (
    MIN_R,
    Spectrum1D,
    analyse_frame,
    analyse_spectrum,
    apply_solution,
    balmer_decrement,
    boxcar_extract,
    check_against_sky_lines,
    classify_bpt,
    classify_supernova,
    cross_correlate,
    estimate_sky,
    find_peaks,
    find_trace,
    fit_continuum,
    fit_lines,
    fit_wavelength_solution,
    galaxy_spectrum,
    group_lines,
    kauffmann_line,
    kewley_line,
    line_ratio,
    log_grid,
    measure_redshift,
    normalise,
    optimal_extract,
    quasar_spectrum,
    star_spectrum,
    supernova_spectrum,
    vote_for_linear_solution,
)
from astrovision.spectra.templates import LINES


@pytest.fixture(scope="module")
def instrument():
    return SpectrographConfig(seed=5)


@pytest.fixture(scope="module")
def simulator(instrument):
    return SpectrographSimulator(instrument)


@pytest.fixture(scope="module")
def arc_solution(simulator, instrument):
    arc = simulator.arc_frame()
    flux = np.median(arc.image, axis=0)
    return fit_wavelength_solution(flux, [w for w, _ in ARC_LINES],
                                   resolution=instrument.resolution), arc


class TestTemplates:
    def test_an_old_population_has_deeper_metal_lines(self):
        def depth(template, name):
            centre = LINES[name]
            near = np.abs(template.wavelength - centre) < 8.0
            side = ((np.abs(template.wavelength - centre) > 40)
                    & (np.abs(template.wavelength - centre) < 90))
            return 1.0 - template.flux[near].min() / np.median(template.flux[side])

        assert depth(galaxy_spectrum(11.0), "Ca K") > depth(galaxy_spectrum(0.3), "Ca K")

    def test_emission_ratios_follow_the_ionisation_parameter(self):
        """The simulator has to put galaxies on a physical sequence, or a
        diagnostic test measures nothing."""
        def ratio(u):
            spectrum = galaxy_spectrum(2.0, emission=1.0, ionisation=u)
            lines = fit_lines(spectrum, 0.0, resolution=2.0)
            return lines["[O III] 5007"].flux / lines["H beta"].flux

        assert ratio(0.1) < ratio(0.5) < ratio(0.9)

    def test_lines_are_broadened_by_velocity_not_by_wavelength(self):
        from astrovision.spectra import velocity_sigma

        assert velocity_sigma(6563.0, 300.0) > velocity_sigma(4861.0, 300.0)

    def test_a_supernova_has_the_feature_that_defines_its_type(self):
        """Type Ia is Si II 6355; if the simulator does not draw it, the
        classifier below is matching something else."""
        ia = supernova_spectrum("Ia", 0.0)
        ic = supernova_spectrum("Ic", 0.0)

        def depth(spectrum, centre=6100.0):
            near = np.abs(spectrum.wavelength - centre) < 60
            side = np.abs(spectrum.wavelength - 6800.0) < 60
            return 1.0 - spectrum.flux[near].min() / np.median(spectrum.flux[side])

        assert depth(ia) > depth(ic)

    def test_supernova_features_move_with_phase(self):
        early = supernova_spectrum("Ia", -10.0)
        late = supernova_spectrum("Ia", 30.0)
        window = (early.wavelength > 5900) & (early.wavelength < 6400)
        assert (float(early.wavelength[window][np.argmin(early.flux[window])])
                < float(late.wavelength[window][np.argmin(late.flux[window])]))

    def test_a_spectrum_keeps_its_errors_and_reports_its_signal_to_noise(self):
        grid = np.linspace(4000, 5000, 100)
        spectrum = Spectrum1D(grid, np.full(100, 10.0), np.full(100, 2.0))
        assert spectrum.snr() == pytest.approx(5.0)
        assert spectrum.dispersion() == pytest.approx(grid[1] - grid[0])

    def test_a_mismatched_error_array_is_rejected(self):
        with pytest.raises(ValueError):
            Spectrum1D(np.arange(10.0), np.arange(10.0), np.arange(5.0))


class TestExtraction:
    def test_the_trace_follows_the_curvature(self, simulator):
        frame = simulator.object_frame(galaxy_spectrum(6.0, emission=1.0),
                                       redshift=0.1, total_counts=3e5)
        trace = find_trace(frame.image)
        error = np.abs(trace.centres - frame.true_trace())
        assert float(error.max()) < 0.5
        assert not trace.flags

    def test_a_straight_trace_would_not_do(self, simulator):
        """The curvature is real: a fixed row is wrong by more than a pixel at
        the ends, which is where a fixed aperture starts losing flux."""
        frame = simulator.object_frame(galaxy_spectrum(6.0), total_counts=3e5)
        truth = frame.true_trace()
        assert float(np.abs(truth - truth.mean()).max()) > 1.0

    def test_sky_subtraction_leaves_the_sky_behind(self, simulator):
        frame = simulator.object_frame(galaxy_spectrum(6.0, emission=1.0),
                                       redshift=0.1, total_counts=3e5)
        trace = find_trace(frame.image)
        sky = estimate_sky(frame.image, trace)
        truth = frame.truth["sky"]
        assert abs(float(np.median(sky[0] - truth))) < 0.05 * float(np.median(truth))

    def test_errors_match_the_scatter_of_repeated_exposures(self, instrument):
        """An error bar that does not describe the noise makes every
        significance downstream a fiction."""
        template = galaxy_spectrum(6.0, emission=1.0)
        stack, errors = [], []
        reference = SpectrographSimulator(SpectrographConfig(seed=0))
        first = reference.object_frame(template, redshift=0.1,
                                       total_counts=3e5, cosmic_rays=0)
        trace = find_trace(first.image)
        for seed in range(8):
            simulator = SpectrographSimulator(SpectrographConfig(seed=200 + seed))
            frame = simulator.object_frame(template, redshift=0.1,
                                           total_counts=3e5, cosmic_rays=0)
            sky = estimate_sky(frame.image, trace)
            spectrum = optimal_extract(frame.image, trace, frame.variance, sky=sky)
            stack.append(spectrum.flux)
            errors.append(spectrum.error)
        empirical = np.std(np.asarray(stack), axis=0)
        reported = np.median(np.asarray(errors), axis=0)
        good = np.isfinite(empirical) & (empirical > 0) & np.isfinite(reported)
        assert float(np.median(reported[good] / empirical[good])) == pytest.approx(
            1.0, abs=0.15)

    def test_optimal_extraction_rejects_cosmic_rays(self, simulator):
        """The measured advantage of profile weighting here is not
        signal-to-noise -- it is that a hit does not become a spectral line."""
        template = galaxy_spectrum(6.0, emission=1.0)
        frame = simulator.object_frame(template, redshift=0.1,
                                       total_counts=3e5, cosmic_rays=25)
        trace = find_trace(frame.image)
        sky = estimate_sky(frame.image, trace)
        truth = frame.truth["object_counts"]
        cleaned = optimal_extract(frame.image, trace, frame.variance, sky=sky)
        raw = optimal_extract(frame.image, trace, frame.variance, sky=sky,
                              reject_cosmic_rays=False)

        def spikes(spectrum):
            deviation = (spectrum.flux - truth) / np.where(spectrum.error > 0,
                                                           spectrum.error, np.inf)
            return int(np.sum(deviation > 10))

        assert spikes(cleaned) < spikes(raw)
        assert spikes(cleaned) == 0

    def test_a_boxcar_still_works_when_the_profile_cannot_be_modelled(self, simulator):
        frame = simulator.object_frame(galaxy_spectrum(6.0), total_counts=3e5)
        trace = find_trace(frame.image)
        sky = estimate_sky(frame.image, trace)
        spectrum = boxcar_extract(frame.image, trace, frame.variance,
                                  half_width=4.0, sky=sky)
        assert np.isfinite(spectrum.flux).all()
        assert spectrum.snr() > 1.0

    def test_an_empty_frame_does_not_pretend_to_have_a_trace(self):
        rng = np.random.default_rng(0)
        blank = rng.normal(0.0, 1.0, (40, 300))
        trace = find_trace(blank)
        assert "too_few_centroids" in trace.flags


class TestWavelengthCalibration:
    def test_every_arc_line_is_found(self, arc_solution):
        solution, arc = arc_solution
        peaks = find_peaks(np.median(arc.image, axis=0))
        assert len(peaks) == len(ARC_LINES)

    def test_the_solution_is_right_where_it_was_fitted(self, arc_solution, instrument):
        solution, _ = arc_solution
        assert solution.succeeded
        columns = np.arange(instrument.n_columns)
        error = solution(columns) - instrument.wavelength_at(columns)
        inside = ~solution.extrapolates(columns)
        assert float(np.sqrt(np.mean(error[inside] ** 2))) < 0.3

    def test_a_linear_solution_is_refused(self, arc_solution, instrument):
        """The dispersion is not linear, and a fit that pretends otherwise
        leaves a residual larger than the redshift errors downstream."""
        _, arc = arc_solution
        linear = fit_wavelength_solution(np.median(arc.image, axis=0),
                                         [w for w, _ in ARC_LINES], order=1,
                                         resolution=instrument.resolution)
        assert not linear.succeeded
        assert "residual" in linear.reason

    def test_the_pairwise_vote_identifies_lines_without_a_guess(self, arc_solution,
                                                                instrument):
        """The vote only has to find the right *anchor*. It cannot place every
        line, because no linear solution fits a cubic dispersion -- 16 of 26
        here -- and that is enough for the polynomial refinement to pick up
        the rest."""
        _, arc = arc_solution
        peaks = find_peaks(np.median(arc.image, axis=0))
        coefficients, votes = vote_for_linear_solution(
            peaks, [w for w, _ in ARC_LINES], tolerance=7.5)
        assert votes >= 10
        assert float(coefficients[0]) == pytest.approx(instrument.dispersion,
                                                       rel=0.15)

    def test_the_solution_refuses_to_extrapolate(self, arc_solution, instrument):
        solution, _ = arc_solution
        assert bool(solution.extrapolates(np.array([-50.0]))[0])
        assert not bool(solution.extrapolates(
            np.array([np.mean(solution.column_range)]))[0])

    def test_applying_a_failed_solution_is_an_error(self):
        from astrovision.spectra import WavelengthSolution

        with pytest.raises(ValueError):
            apply_solution(Spectrum1D(np.arange(10.0), np.zeros(10)),
                           WavelengthSolution())

    def test_sky_lines_measure_the_zero_point(self, arc_solution, simulator,
                                              instrument):
        solution, _ = arc_solution
        frame = simulator.object_frame(galaxy_spectrum(6.0), total_counts=3e5)
        trace = find_trace(frame.image)
        sky_model = estimate_sky(frame.image, trace)
        row = int(round(float(trace.centres[len(trace.centres) // 2])))
        columns = np.arange(instrument.n_columns)
        sky = Spectrum1D(solution(columns), sky_model[row])
        check = check_against_sky_lines(sky, [w for w, _ in SKY_LINES])
        assert check["reliable"]
        assert abs(check["offset"]) < 0.5

    def test_a_flexure_shift_is_recovered(self, arc_solution, simulator,
                                          instrument):
        solution, _ = arc_solution
        frame = simulator.object_frame(galaxy_spectrum(6.0), total_counts=3e5)
        trace = find_trace(frame.image)
        sky_model = estimate_sky(frame.image, trace)
        row = int(round(float(trace.centres[len(trace.centres) // 2])))
        columns = np.arange(instrument.n_columns)
        shifted = Spectrum1D(solution(columns) + 3.0, sky_model[row])
        check = check_against_sky_lines(shifted, [w for w, _ in SKY_LINES])
        assert check["offset"] == pytest.approx(3.0, abs=0.3)


class TestContinuum:
    def test_the_continuum_ignores_emission_lines(self):
        grid = np.linspace(4000.0, 7000.0, 3000)
        flux = np.ones_like(grid)
        flux += 40.0 * np.exp(-0.5 * ((grid - 6563.0) / 3.0) ** 2)
        continuum = fit_continuum(Spectrum1D(grid, flux))
        near = np.abs(grid - 6563.0) < 20
        assert float(np.max(continuum[near])) < 1.3

    def test_normalising_masks_pixels_with_no_continuum(self):
        """Dividing noise by a continuum near zero produces a loud spectrum,
        not a faint one."""
        grid = np.linspace(4000.0, 7000.0, 2000)
        rng = np.random.default_rng(2)
        flux = np.where(grid < 5000.0, 0.0, 1.0) + rng.normal(0, 0.02, grid.size)
        flattened = normalise(Spectrum1D(grid, flux, np.full(grid.size, 0.02)))
        assert flattened.meta["n_no_continuum"] > 100
        assert float(np.max(np.abs(flattened.flux))) < 5.0


class TestRedshift:
    def test_a_log_grid_is_uniform_in_velocity(self):
        grid = log_grid(4000.0, 8000.0, 60.0)
        steps = 299792.458 * (grid[1:] / grid[:-1] - 1.0)
        assert float(np.std(steps)) < 1e-6

    def test_the_correlation_peaks_at_the_true_shift(self):
        template = galaxy_spectrum(8.0, emission=0.4)
        grid = log_grid(3500.0, 9000.0, 50.0)
        from astrovision.spectra.redshift import prepare

        observed = prepare(template.redshifted(0.25), grid)
        correlation, lags = cross_correlate(observed, prepare(template, grid))
        step = math.log(grid[1] / grid[0])
        redshifts = np.expm1(lags * step)
        assert float(redshifts[int(np.argmax(correlation))]) == pytest.approx(
            0.25, abs=0.002)

    @pytest.mark.parametrize("z_true", [0.05, 0.22, 0.47])
    def test_redshifts_are_recovered(self, simulator, z_true):
        galaxy = galaxy_spectrum(5.0, emission=0.8, ionisation=0.25)
        result = measure_redshift(simulator.extracted(galaxy, redshift=z_true,
                                                      snr=15.0))
        assert result.reliable
        assert abs(result.z - z_true) / (1 + z_true) < 0.002

    def test_pure_noise_is_refused(self):
        """A cross-correlation always has a maximum; the question is whether
        it means anything."""
        rng = np.random.default_rng(9)
        grid = np.linspace(3700.0, 9700.0, 1400)
        refused = 0
        for _ in range(6):
            spectrum = Spectrum1D(grid, rng.normal(1.0, 0.1, grid.size),
                                  np.full(grid.size, 0.1))
            refused += not measure_redshift(spectrum).reliable
        assert refused == 6

    def test_a_star_is_reported_as_a_star(self, simulator):
        result = measure_redshift(simulator.extracted(star_spectrum("A"), snr=25.0))
        assert result.is_star
        assert abs(result.z) < 0.005

    def test_a_starburst_is_not_a_star(self, simulator):
        """The template's *kind* decides this, not whether its name begins
        with the letters s-t-a-r."""
        galaxy = galaxy_spectrum(0.6, emission=2.0, ionisation=0.2)
        result = measure_redshift(simulator.extracted(galaxy, redshift=0.12,
                                                      snr=20.0))
        assert not result.is_star

    def test_a_rival_peak_costs_the_reliability_flag(self, simulator):
        """At low signal-to-noise a wrong answer is not a weak correlation, it
        is a confident match to the wrong feature -- so a rival peak of nearly
        the same strength removes the claim, unless the emission lines settle
        it independently."""
        rng = np.random.default_rng(3)
        refused = 0
        for _ in range(12):
            z = float(rng.uniform(0.1, 0.5))
            galaxy = galaxy_spectrum(float(rng.uniform(3, 9)), emission=0.3)
            result = measure_redshift(simulator.extracted(galaxy, redshift=z,
                                                          snr=3.0))
            refused += not result.reliable
            if "rival_peak" in result.flags and result.reliable:
                assert result.n_emission_lines >= 2
        assert refused >= 1

    def test_emission_lines_can_carry_a_redshift_alone(self, simulator):
        """A spectrum with no continuum to correlate still has lines."""
        from astrovision.spectra import measure_emission_redshift

        galaxy = galaxy_spectrum(0.4, emission=2.5, ionisation=0.3)
        observed = simulator.extracted(galaxy, redshift=0.18, snr=20.0)
        z, n_lines, found = measure_emission_redshift(observed)
        assert n_lines >= 2
        assert abs(z - 0.18) < 0.005

    def test_the_result_serialises(self, simulator):
        payload = measure_redshift(
            simulator.extracted(galaxy_spectrum(6.0), redshift=0.1,
                                snr=20.0)).to_dict()
        assert set(payload) >= {"z", "r_statistic", "reliable", "template",
                                "peak_ratio", "flags"}

    def test_the_threshold_is_the_documented_one(self):
        assert MIN_R == pytest.approx(7.0)


class TestLines:
    @pytest.fixture(scope="class")
    def measured(self):
        simulator = SpectrographSimulator(SpectrographConfig(seed=9))
        galaxy = galaxy_spectrum(2.0, emission=1.2, ionisation=0.2,
                                 velocity_dispersion=150.0)
        observed = simulator.extracted(galaxy, redshift=0.05, snr=30.0)
        return fit_lines(observed, 0.05, resolution=5.0)

    def test_halpha_and_nitrogen_are_fitted_together(self):
        groups = group_lines(["H alpha", "[N II] 6584", "[N II] 6548", "H beta"],
                             0.0, 5.0)
        blend = [g for g in groups if "H alpha" in g][0]
        assert set(blend) == {"H alpha", "[N II] 6584", "[N II] 6548"}
        assert ["H beta"] in groups

    def test_the_nitrogen_ratio_is_right(self, measured):
        """The simulator draws this ratio as 0.20 + 1.30 u^1.5, which is 0.316
        at the fixture's ionisation. Fitted one line at a time rather than as
        a blend, the same spectrum returned its reciprocal."""
        expected = 0.20 + 1.30 * 0.2 ** 1.5
        ratio, _, status = line_ratio(measured, "[N II] 6584", "H alpha")
        assert status == "measured"
        assert ratio == pytest.approx(expected, rel=0.10)

    def test_blended_lines_share_a_width(self, measured):
        widths = [measured[name].velocity_width_km_s
                  for name in ("H alpha", "[N II] 6584", "[N II] 6548")]
        assert max(widths) - min(widths) < 1.0

    def test_a_line_that_is_not_there_becomes_an_upper_limit(self, simulator):
        quiescent = galaxy_spectrum(11.0, emission=0.0)
        lines = fit_lines(simulator.extracted(quiescent, redshift=0.05, snr=30.0),
                          0.05, names=["[O III] 5007"], resolution=5.0)
        measurement = lines["[O III] 5007"]
        assert not measurement.detected
        assert np.isfinite(measurement.upper_limit)
        assert "not_detected" in measurement.flags

    def test_balmer_absorption_is_fitted_under_the_emission(self, measured):
        assert "stellar_absorption_corrected" in measured["H beta"].flags

    def test_a_ratio_from_a_non_detection_is_labelled_a_limit(self, simulator):
        quiescent = galaxy_spectrum(11.0, emission=0.0)
        lines = fit_lines(simulator.extracted(quiescent, redshift=0.05, snr=30.0),
                          0.05, resolution=5.0)
        _, _, status = line_ratio(lines, "[O III] 5007", "H beta")
        assert status != "measured"

    def test_the_balmer_decrement_refuses_negative_extinction(self):
        from astrovision.spectra import LineMeasurement

        lines = {
            "H alpha": LineMeasurement("H alpha", 6562.8, flux=2.0,
                                       flux_error=0.05, detected=True),
            "H beta": LineMeasurement("H beta", 4861.3, flux=1.0,
                                      flux_error=0.05, detected=True),
        }
        result = balmer_decrement(lines)
        assert not result["reliable"]
        assert "dust cannot do" in result["reason"]

    def test_the_decrement_measures_reddening_when_it_can(self, measured):
        result = balmer_decrement(measured)
        assert result["reliable"]
        assert result["e_bv"] >= 0.0


class TestDiagnostics:
    @staticmethod
    def _classify(ionisation, snr=30.0):
        simulator = SpectrographSimulator(SpectrographConfig(seed=9))
        galaxy = galaxy_spectrum(2.0, emission=1.2, ionisation=ionisation,
                                 velocity_dispersion=150.0)
        observed = simulator.extracted(galaxy, redshift=0.05, snr=snr)
        return classify_bpt(fit_lines(observed, 0.05, resolution=5.0))

    def test_a_star_forming_galaxy_lands_on_the_star_forming_locus(self):
        result = self._classify(0.05)
        assert result.classification == "star-forming"
        assert result.confident

    def test_a_seyfert_lands_above_the_maximum_starburst_line(self):
        result = self._classify(0.6)
        assert result.classification == "Seyfert"

    def test_the_composite_region_is_named_not_rounded(self):
        result = self._classify(0.3)
        assert result.classification == "composite"
        assert "both contribute" in result.reason

    def test_beyond_the_asymptote_is_the_agn_side(self):
        """Past log [N II]/H-alpha = 0.05 the Kauffmann curve does not exist,
        and treating it as infinitely high calls the hardest-ionised objects
        star-forming."""
        assert float(kauffmann_line(np.array([0.3]))[0]) == -np.inf
        assert float(kewley_line(np.array([0.6]))[0]) == -np.inf

    def test_a_galaxy_without_lines_is_not_classified(self, simulator):
        quiescent = galaxy_spectrum(11.0, emission=0.0)
        lines = fit_lines(simulator.extracted(quiescent, redshift=0.05, snr=30.0),
                          0.05, resolution=5.0)
        result = classify_bpt(lines)
        assert result.classification == "unclassified"
        assert result.missing

    def test_missing_lines_are_named(self):
        result = classify_bpt({})
        assert set(result.missing) == {"H beta", "[O III] 5007", "H alpha",
                                       "[N II] 6584"}


class TestSupernovaTyping:
    @pytest.mark.parametrize("sn_type", ["Ia", "Ib", "II"])
    def test_the_type_is_recovered(self, simulator, sn_type):
        observed = simulator.extracted(supernova_spectrum(sn_type, 5.0),
                                       redshift=0.02, snr=20.0)
        match = classify_supernova(observed, redshift=0.02)
        assert match.sn_type == sn_type
        assert match.confident

    def test_a_type_ic_is_often_refused_rather_than_guessed(self, simulator):
        """Type Ic is defined by what it lacks -- no hydrogen, no helium, no
        strong silicon -- so it is the hardest to match, and the classifier
        says so instead of picking the nearest neighbour."""
        observed = simulator.extracted(supernova_spectrum("Ic", -5.0),
                                       redshift=0.02, snr=30.0)
        match = classify_supernova(observed, redshift=0.02)
        assert match.sn_type in ("", "Ic")
        if not match.sn_type:
            assert "no type is claimed" in match.reason

    def test_a_galaxy_is_not_typed_as_a_supernova(self, simulator):
        observed = simulator.extracted(galaxy_spectrum(8.0, emission=0.2),
                                       redshift=0.05, snr=20.0)
        match = classify_supernova(observed, redshift=0.05)
        assert not match.confident or match.margin < 3.0

    def test_a_confident_type_carries_the_caveat(self, simulator):
        observed = simulator.extracted(supernova_spectrum("II", 5.0),
                                       redshift=0.02, snr=20.0)
        match = classify_supernova(observed, redshift=0.02)
        assert match.confident
        assert "candidate classification" in match.caveat
        assert "confirmed supernova" in match.caveat

    def test_the_phase_is_searched_not_assumed(self, simulator):
        late = simulator.extracted(supernova_spectrum("Ia", 25.0),
                                   redshift=0.02, snr=20.0)
        early = simulator.extracted(supernova_spectrum("Ia", -7.0),
                                    redshift=0.02, snr=20.0)
        assert (classify_supernova(late, redshift=0.02).phase_days
                > classify_supernova(early, redshift=0.02).phase_days)


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def analysed(self):
        config = SpectrographConfig(seed=21)
        simulator = SpectrographSimulator(config)
        arc = simulator.arc_frame()
        galaxy = galaxy_spectrum(3.0, emission=1.2, ionisation=0.6,
                                 velocity_dispersion=180.0)
        frame = simulator.object_frame(galaxy, redshift=0.12, total_counts=2.0e6)
        return analyse_frame(frame.image, frame.variance, arc=arc.image,
                             line_list=[w for w, _ in ARC_LINES],
                             sky_lines=[w for w, _ in SKY_LINES],
                             resolution=config.resolution)

    def test_it_measures_the_redshift_from_a_raw_frame(self, analysed):
        assert analysed.redshift.reliable
        assert analysed.redshift.z == pytest.approx(0.12, abs=0.002)

    def test_it_classifies_the_ionisation(self, analysed):
        assert analysed.bpt.classification == "Seyfert"

    def test_it_measures_a_velocity_dispersion(self, analysed):
        assert analysed.dispersion["reliable"]
        assert 100.0 < analysed.dispersion["sigma_km_s"] < 400.0

    def test_without_an_arc_it_stops_and_says_so(self):
        simulator = SpectrographSimulator(SpectrographConfig(seed=21))
        frame = simulator.object_frame(galaxy_spectrum(6.0), total_counts=1e6)
        analysis = analyse_frame(frame.image, frame.variance)
        assert analysis.stopped_at == "no wavelength calibration"
        assert analysis.redshift is None

    def test_an_unreliable_redshift_stops_the_line_measurements(self):
        rng = np.random.default_rng(11)
        grid = np.linspace(3700.0, 9700.0, 1400)
        spectrum = Spectrum1D(grid, rng.normal(1.0, 0.1, grid.size),
                              np.full(grid.size, 0.1))
        analysis = analyse_spectrum(spectrum)
        assert analysis.stopped_at == "redshift not reliable"
        assert not analysis.lines

    def test_a_supernova_is_typed_even_when_the_galaxy_fit_fails(self, simulator):
        """A supernova is not a galaxy, so the galaxy cross-correlation
        failing on one is the expected outcome, not a reason to refuse."""
        observed = simulator.extracted(supernova_spectrum("II", 8.0),
                                       redshift=0.03, snr=15.0)
        analysis = analyse_spectrum(observed, classify_transient=True,
                                    redshift=0.03)
        assert analysis.supernova is not None
        assert analysis.supernova.sn_type == "II"

    def test_the_record_serialises(self, analysed):
        payload = analysed.to_dict()
        assert set(payload) >= {"spectrum", "wavelength_solution", "redshift",
                                "lines", "bpt", "stopped_at", "notes"}
        assert payload["redshift"]["reliable"]

    def test_the_summary_reads_as_a_sentence(self, analysed):
        summary = analysed.summary()
        assert "z = 0.12" in summary
        assert "Seyfert" in summary

    def test_a_quasar_gets_the_right_redshift_from_the_wrong_template(self,
                                                                       simulator):
        """The correlation measures a redshift, not a classification. A
        quasar's narrow [O III] lines match the starburst template's narrow
        lines well enough to win, at every continuum window tried, and the
        redshift is right anyway. The template name is a by-product and this
        package does not claim otherwise."""
        result = measure_redshift(simulator.extracted(quasar_spectrum(),
                                                      redshift=0.3, snr=20.0))
        assert result.z == pytest.approx(0.3, abs=0.01)
        assert result.reliable
        assert not result.is_star
