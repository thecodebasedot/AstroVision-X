"""Photometric redshifts: the spectra, the fit, and what it admits it cannot do."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.types import ObjectClass
from astrovision.photoz import (
    PhotoZLibrary,
    build_template,
    describe_break_crossings,
    draw_template,
    filter_curve,
    fit_catalog,
    fit_photoz,
    standard_library,
)
from astrovision.photoz.templates import FILTER_BANDS, WAVELENGTH


class TestSpectra:
    def test_an_old_population_is_redder_than_a_young_one(self):
        old = build_template(age_gyr=10.0, dust=0.0)
        young = build_template(age_gyr=0.2, dust=0.0)
        assert old.colours(0.0, ("g", "r"))[0] > young.colours(0.0, ("g", "r"))[0]

    def test_dust_reddens(self):
        clean = build_template(age_gyr=2.0, dust=0.0)
        dusty = build_template(age_gyr=2.0, dust=0.8)
        assert dusty.colours(0.0, ("g", "r"))[0] > clean.colours(0.0, ("g", "r"))[0]

    def test_the_4000_angstrom_break_deepens_with_age(self):
        def contrast(template):
            blue = template.flux[np.argmin(np.abs(WAVELENGTH - 3800.0))]
            red = template.flux[np.argmin(np.abs(WAVELENGTH - 4200.0))]
            return red / blue

        assert contrast(build_template(10.0, 0.0)) > contrast(build_template(0.2, 0.0))

    def test_emission_lines_are_added_where_they_belong(self):
        quiet = build_template(0.5, 0.0, emission=0.0)
        loud = build_template(0.5, 0.0, emission=2.0)
        halpha = int(np.argmin(np.abs(WAVELENGTH - 6563.0)))
        continuum = int(np.argmin(np.abs(WAVELENGTH - 6000.0)))
        assert loud.flux[halpha] / loud.flux[continuum] > \
            quiet.flux[halpha] / quiet.flux[continuum]

    def test_the_colour_tracks_the_break_moving_through_the_filters(self):
        """The whole mechanism in one assertion: g-r peaks while the break is
        in g, and r-i peaks later, once it has moved into r."""
        template = build_template(10.0, 0.1)
        grid = np.linspace(0.0, 1.2, 40)
        gr = np.array([template.colours(z, ("g", "r", "i"))[0] for z in grid])
        ri = np.array([template.colours(z, ("g", "r", "i"))[1] for z in grid])
        assert grid[int(np.argmax(gr))] < grid[int(np.argmax(ri))]

    def test_filters_are_normalised_and_bounded(self):
        for band in FILTER_BANDS:
            curve = filter_curve(band)
            assert curve.min() >= 0.0
            assert curve.max() == pytest.approx(1.0, abs=0.05)

    def test_an_unknown_filter_is_an_error(self):
        with pytest.raises(KeyError):
            filter_curve("q")

    def test_break_crossings_are_ordered_by_filter(self):
        ranges = describe_break_crossings(["g", "r", "i"])
        assert ranges["g"][1] < ranges["r"][1] < ranges["i"][1]

    def test_a_drawn_template_is_not_in_the_library(self):
        """The point of the continuous family: if the fit library contained
        the simulated galaxy, the measured scatter would be meaningless."""
        rng = np.random.default_rng(3)
        library = standard_library()
        drawn = [draw_template(rng) for _ in range(20)]
        for template in drawn:
            ages = [abs(t.age_gyr - template.age_gyr) for t in library]
            dusts = [abs(t.dust - template.dust) for t in library]
            assert min(a + d for a, d in zip(ages, dusts)) > 1e-6


class TestFitting:
    @pytest.fixture(scope="class")
    def library(self):
        return PhotoZLibrary(bands=("u", "g", "r", "i", "z"), z_max=1.2, n_z=100)

    def test_it_recovers_a_library_template_exactly(self, library):
        """A galaxy that *is* in the library should be nailed -- if this
        fails, nothing downstream can be trusted."""
        template = library.templates[0]
        for z_true in (0.15, 0.45, 0.80):
            colours = template.colours(z_true, library.bands)
            result = fit_photoz(colours, [0.01] * len(colours), library)
            assert result.z == pytest.approx(z_true, abs=0.05)

    def test_a_drawn_galaxy_is_recovered_within_the_error(self, library):
        rng = np.random.default_rng(21)
        inside = 0
        trials = 40
        for _ in range(trials):
            z_true = float(rng.uniform(0.1, 1.0))
            colours = draw_template(rng).colours(z_true, library.bands)
            noisy = colours + rng.normal(0, 0.03, size=len(colours))
            result = fit_photoz(noisy, [0.03] * len(colours), library)
            inside += abs(result.z - z_true) <= 2.5 * result.z_error
        assert inside >= 0.7 * trials

    def test_more_filters_give_a_better_redshift(self, library):
        """The headline limitation, asserted rather than asserted about."""
        rng = np.random.default_rng(7)
        three = PhotoZLibrary(bands=("g", "r", "i"), z_max=1.2, n_z=100)

        def scatter(target):
            errors = []
            for _ in range(60):
                z_true = float(rng.uniform(0.1, 1.0))
                sed = draw_template(rng)
                colours = sed.colours(z_true, target.bands)
                noisy = colours + rng.normal(0, 0.03, size=len(colours))
                result = fit_photoz(noisy, [0.03] * len(colours), target)
                errors.append((result.z - z_true) / (1 + z_true))
            return float(np.mean(np.abs(np.asarray(errors)) > 0.15))

        assert scatter(library) < scatter(three)

    def test_no_usable_colours_returns_nothing(self, library):
        result = fit_photoz([np.nan] * 4, [np.nan] * 4, library)
        assert not np.isfinite(result.z)
        assert "no_usable_colours" in result.flags

    def test_missing_colours_are_dropped_not_filled(self, library):
        template = library.templates[2]
        colours = list(template.colours(0.4, library.bands))
        colours[0] = float("nan")
        result = fit_photoz(colours, [0.02] * len(colours), library)
        assert result.n_colours == len(colours) - 1
        assert np.isfinite(result.z)

    def test_too_few_colours_is_flagged(self):
        library = PhotoZLibrary(bands=("g", "r", "i"), z_max=1.0, n_z=60)
        template = library.templates[0]
        result = fit_photoz(template.colours(0.3, library.bands),
                            [0.02, 0.02], library)
        assert "underdetermined" in result.flags

    def test_the_error_carries_a_template_floor(self, library):
        """The posterior width is the error *given* the library, and the
        library is wrong."""
        from astrovision.photoz import TEMPLATE_FLOOR

        template = library.templates[0]
        result = fit_photoz(template.colours(0.5, library.bands),
                            [0.001] * 4, library)
        assert result.z_error >= TEMPLATE_FLOOR * (1 + result.z) * 0.99

    def test_a_bimodal_posterior_is_reported_as_ambiguous(self, library):
        """A red galaxy at low redshift and a blue one higher up can give the
        same colours; reporting only the peak turns that into a confident
        wrong answer."""
        rng = np.random.default_rng(5)
        found = False
        for _ in range(60):
            z_true = float(rng.uniform(0.1, 1.0))
            colours = draw_template(rng).colours(z_true, ("g", "r", "i"))
            three = PhotoZLibrary(bands=("g", "r", "i"), z_max=1.2, n_z=100)
            result = fit_photoz(colours + rng.normal(0, 0.08, 2), [0.08] * 2, three)
            if "ambiguous" in result.flags:
                found = True
                assert np.isfinite(result.second_z)
                assert result.second_weight >= 0.25
                assert not result.reliable
                break
        assert found, "three noisy bands should produce some ambiguous fits"

    def test_a_hopeless_fit_is_flagged(self, library):
        result = fit_photoz([5.0, -5.0, 5.0, -5.0], [0.01] * 4, library)
        assert "poor_fit" in result.flags
        assert not result.reliable

    def test_the_posterior_is_normalised(self, library):
        template = library.templates[1]
        result = fit_photoz(template.colours(0.3, library.bands),
                            [0.03] * 4, library)
        assert result.posterior.sum() == pytest.approx(1.0)
        assert result.z_lower < result.z < result.z_upper

    def test_odds_separates_confident_from_diffuse(self, library):
        template = library.templates[0]
        tight = fit_photoz(template.colours(0.5, library.bands), [0.02] * 4, library)
        loose = fit_photoz(template.colours(0.5, library.bands), [0.5] * 4, library)
        assert tight.odds > loose.odds

    def test_the_result_serialises(self, library):
        template = library.templates[0]
        payload = fit_photoz(template.colours(0.4, library.bands),
                             [0.03] * 4, library).to_dict()
        assert set(payload) >= {"z", "z_error", "odds", "reliable", "flags",
                                "second_z", "template"}


class TestCatalogIntegration:
    @pytest.fixture(scope="class")
    def fitted(self):
        from astrovision.detect import Detector
        from astrovision.photometry import Photometer
        from astrovision.photometry.multiband import forced_photometry, measure_colours
        from astrovision.preprocess import Preprocessor
        from astrovision.simulate import SkyConfig, SkySimulator

        config = SkyConfig(shape=(300, 300), n_stars=70, n_galaxies=30, n_nebulae=0,
                           n_clusters=0, n_lenses=0, n_anomalies=0, seed=13,
                           galaxy_flux_range=(8000.0, 90000.0))
        bands = ("u", "g", "r", "i", "z")
        images, truth = SkySimulator(config).generate_multiband(
            bands, redshift_range=(0.05, 0.9))
        preprocessor = Preprocessor()
        clean = {name: preprocessor.run(image) for name, image in images.items()}
        catalog, segmentation = Detector().detect(clean["r"])
        Photometer().run(clean["r"], catalog, segmentation)
        forced_photometry(clean, catalog, detection_band="r", segmentation=segmentation)
        measure_colours(catalog, list(zip(bands[:-1], bands[1:])), min_snr=5.0)
        library = PhotoZLibrary(bands=bands, z_max=1.2, n_z=100)
        report = fit_catalog(catalog, library)
        return catalog, truth, report

    def test_galaxies_get_redshifts(self, fitted):
        catalog, _, report = fitted
        assert report["n_fitted"] > 10
        assert sum(1 for s in catalog if "photoz" in s.meta) == report["n_fitted"]

    def test_stars_do_not(self, fitted):
        """A star has no redshift, and fitting one produces a number that
        will be used as though it did."""
        catalog, _, _ = fitted
        for source in catalog:
            if source.object_class is ObjectClass.STAR:
                assert "photoz" not in source.meta

    def test_the_measured_redshifts_are_right(self, fitted):
        import math

        catalog, truth, _ = fitted
        errors = []
        for source in catalog:
            best, distance = None, 1e9
            for item in truth:
                offset = math.hypot(item.x - source.x, item.y - source.y)
                if offset < distance:
                    best, distance = item, offset
            photoz = source.meta.get("photoz")
            if (best is None or distance > 1.5 or best.kind != "galaxy"
                    or "redshift" not in best.meta or not photoz):
                continue
            errors.append((photoz["z"] - best.meta["redshift"])
                          / (1 + best.meta["redshift"]))
        assert len(errors) >= 10
        values = np.asarray(errors)
        assert abs(float(np.median(values))) < 0.05
        core = values[np.abs(values) < 0.15]
        scatter = 1.4826 * float(np.median(np.abs(core - np.median(core))))
        assert scatter < 0.06

    def test_the_report_names_where_the_break_is_constrained(self, fitted):
        _, _, report = fitted
        low, high = report["well_constrained_range"]
        assert high > low >= 0.0
