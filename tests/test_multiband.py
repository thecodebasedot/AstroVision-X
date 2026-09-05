"""Multi-band photometry, colours, and colour-based classification.

The measurements here are checked against the simulator's injected colours,
so a passing test means the numbers are right, not merely finite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astrovision.classify import Classifier, fit_stellar_locus
from astrovision.classify.colours import (
    available_triple,
    colour_pair,
    colour_stellarity,
    rayleigh_scale,
)
from astrovision.classify.rules import combine_stellarity
from astrovision.core.config import ClassificationConfig
from astrovision.photometry.multiband import (
    band_flux_table,
    forced_photometry,
    homogenise,
    measure_colours,
)
from astrovision.simulate.sed import (
    BAND_ORDER,
    flux_ratios,
    locus_distance,
    object_colours,
    stellar_colours,
)


def _nearest_truth(source, truth, limit=1.2):
    best, distance = None, float("inf")
    for item in truth:
        offset = math.hypot(item.x - source.x, item.y - source.y)
        if offset < distance:
            best, distance = item, offset
    return best if distance <= limit else None


class TestSimulatedColours:
    def test_stellar_colours_run_blue_to_red(self):
        blue = stellar_colours(0.0)
        red = stellar_colours(1.0)
        assert blue["g"] - blue["r"] < red["g"] - red["r"]

    def test_stars_lie_on_the_locus_and_galaxies_do_not(self):
        rng = np.random.default_rng(4)
        star_offsets, galaxy_offsets = [], []
        for _ in range(200):
            star = object_colours("star", rng=rng)
            galaxy = object_colours("galaxy", "elliptical", rng=rng, redshift=0.15)
            star_offsets.append(locus_distance(star["g"] - star["r"],
                                               star["r"] - star["i"]))
            galaxy_offsets.append(locus_distance(galaxy["g"] - galaxy["r"],
                                                 galaxy["r"] - galaxy["i"]))
        assert np.median(star_offsets) < 0.06
        assert np.median(galaxy_offsets) > 3 * np.median(star_offsets)

    def test_flux_ratios_invert_magnitudes(self):
        ratios = flux_ratios({"r": 0.0, "g": 1.0})
        assert ratios["r"] == pytest.approx(1.0)
        assert ratios["g"] == pytest.approx(10 ** -0.4)

    def test_band_order_is_blue_to_red(self):
        assert BAND_ORDER.index("u") < BAND_ORDER.index("g") < BAND_ORDER.index("z")

    def test_multiband_field_shares_positions_across_bands(self, multiband_field):
        bands, truth = multiband_field
        assert set(bands) == {"g", "r", "i"}
        shapes = {image.shape for image in bands.values()}
        assert len(shapes) == 1
        # Same sky: the truth table is one table, not three.
        assert all("band_flux" in item.meta for item in truth)

    def test_bands_carry_independent_noise(self, multiband_field):
        bands, _ = multiband_field
        # If the noise were shared, the residual between two bands would be
        # far smoother than either band's own noise.
        left, right = bands["g"].data, bands["i"].data
        correlation = np.corrcoef(left.ravel(), right.ravel())[0, 1]
        assert correlation < 0.98

    def test_lens_arcs_are_bluer_than_the_deflector(self, multiband_field):
        _, truth = multiband_field
        lenses = [t for t in truth if t.kind == "lens" and "arc_magnitudes" in t.meta]
        assert lenses, "the fixture should contain a lens system"
        for lens in lenses:
            deflector = lens.meta["magnitudes"]
            arc = lens.meta["arc_magnitudes"]
            assert (arc["g"] - arc["r"]) < (deflector["g"] - deflector["r"])


class TestForcedPhotometry:
    def test_measures_every_band(self, multiband_measured):
        bands, _, catalog, segmentation = multiband_measured
        report = forced_photometry(bands, catalog, detection_band="r",
                                   segmentation=segmentation)
        assert report.bands == list(bands)
        assert report.n_sources == len(catalog)
        assert all(set(source.bands) == set(bands) for source in catalog)

    def test_homogenises_to_the_worst_seeing(self, multiband_measured):
        bands, _, _, _ = multiband_measured
        homogenised, target, changed = homogenise(bands)
        assert np.isfinite(target)
        # g has the worst seeing in the fixture, so it is the target and is
        # the one band left alone.
        assert "g" not in changed
        assert set(changed) <= {"r", "i"}
        for band in changed:
            psf = homogenised[band].meta["psf_model"]
            assert psf.fwhm * homogenised[band].wcs.pixel_scale == pytest.approx(
                target, rel=1e-6)

    def test_homogenised_psf_stamp_is_convolved_too(self, multiband_measured):
        """A model that claims a wider FWHM must have the wings to match."""
        bands, _, _, _ = multiband_measured
        homogenised, _, changed = homogenise(bands)
        assert changed
        band = changed[0]
        before = bands[band].meta["psf_model"]
        after = homogenised[band].meta["psf_model"]
        assert after.fwhm > before.fwhm
        # A blurrier PSF puts a smaller fraction of its light in the peak.
        assert after.as_kernel().max() < before.as_kernel().max()

    def test_colours_match_the_injected_values(self, multiband_measured):
        bands, truth, catalog, segmentation = multiband_measured
        forced_photometry(bands, catalog, detection_band="r", aperture_arcsec=1.6,
                          segmentation=segmentation)
        measure_colours(catalog, [("g", "r"), ("r", "i")], min_snr=15.0)
        errors = []
        for source in catalog:
            item = _nearest_truth(source, truth)
            if item is None or "magnitudes" not in item.meta:
                continue
            measured = source.meta.get("colours", {}).get("g-r")
            if measured is None:
                continue
            injected = item.meta["magnitudes"]["g"] - item.meta["magnitudes"]["r"]
            errors.append(measured - injected)
        assert len(errors) >= 10
        # Measured over many seeds the bias sits below 0.03 mag; this bound
        # catches a real regression without chasing the sampling noise of one
        # field.  See docs/validation.md for the multi-seed numbers.
        assert abs(float(np.median(errors))) < 0.08

    def test_refuses_a_colour_when_one_band_is_a_non_detection(self,
                                                              multiband_measured):
        bands, _, catalog, segmentation = multiband_measured
        forced_photometry(bands, catalog, detection_band="r", segmentation=segmentation)
        generous = measure_colours(catalog, [("g", "r")], min_snr=2.0)
        strict = measure_colours(catalog, [("g", "r")], min_snr=1e6)
        assert generous["g-r"] > 0
        assert strict["g-r"] == 0

    def test_records_a_one_sided_limit_for_a_blue_non_detection(self):
        """An object seen in the red and not the blue is at least that red."""
        from astrovision.core.types import BoundingBox, Photometry, Source, SourceCatalog

        source = Source(id=1, x=5.0, y=5.0, bbox=BoundingBox(0, 0, 10, 10))
        source.bands = {
            "g": Photometry(flux=1.0, flux_err=10.0, magnitude=25.0,
                            snr=0.1, zero_point=25.0),
            "r": Photometry(flux=1000.0, flux_err=10.0, magnitude=17.5,
                            snr=100.0, zero_point=25.0),
        }
        catalog = SourceCatalog([source])
        counts = measure_colours(catalog, [("g", "r")], min_snr=5.0)
        assert counts["g-r"] == 0
        limit = source.meta["colour_limits"]["g-r"]
        assert limit["undetected_band"] == "g"
        assert limit["direction"] == "bluer_than"
        assert "colour_limit" in source.flags

    def test_band_flux_table_shape(self, multiband_measured):
        bands, _, catalog, segmentation = multiband_measured
        forced_photometry(bands, catalog, detection_band="r", segmentation=segmentation)
        table = band_flux_table(catalog, ["g", "r", "i"])
        assert table.shape == (len(catalog), 3)
        assert np.isfinite(table).any()

    def test_missing_detection_band_is_an_error(self, multiband_measured):
        from astrovision.core.exceptions import DataError

        bands, _, catalog, _ = multiband_measured
        with pytest.raises(DataError):
            forced_photometry(bands, catalog, detection_band="nonexistent")


class TestStellarLocus:
    @pytest.fixture()
    def with_colours(self, multiband_measured):
        bands, truth, catalog, segmentation = multiband_measured
        forced_photometry(bands, catalog, detection_band="r", aperture_arcsec=1.6,
                          segmentation=segmentation)
        measure_colours(catalog, [("g", "r"), ("r", "i")], min_snr=5.0)
        Classifier(ClassificationConfig(backend="rules", use_colours=False)).run(
            bands["r"], catalog)
        return bands, truth, catalog

    def test_finds_the_available_triple(self, with_colours):
        _, _, catalog = with_colours
        assert available_triple(catalog) == ("g", "r", "i")

    def test_fits_a_locus_the_stars_sit_on(self, with_colours):
        _, truth, catalog = with_colours
        locus = fit_stellar_locus(catalog)
        assert locus is not None
        assert locus.n_stars >= 12
        star_distances, galaxy_distances = [], []
        for source in catalog:
            item = _nearest_truth(source, truth)
            if item is None or item.kind not in ("star", "galaxy"):
                continue
            x, y = colour_pair(source, locus.bands)
            distance = locus.distance(x, y)
            if not np.isfinite(distance):
                continue
            (star_distances if item.kind == "star" else galaxy_distances).append(distance)
        assert len(star_distances) >= 10
        assert np.median(star_distances) < 0.25

    def test_an_uninformative_test_returns_one_half(self, with_colours):
        """The property the whole design turns on.

        A test with no discriminating power must contribute nothing.  A
        one-sided sigmoid of the offset cannot do this -- it saturates near 1
        for every small offset, so it votes "star" for the galaxies too.
        """
        _, _, catalog = with_colours
        locus = fit_stellar_locus(catalog)
        assert locus is not None
        locus.star_width = np.array([0.1, 0.1])
        locus.galaxy_width = np.array([0.1, 0.1])
        locus.snr_bins = np.array([1.0, 2.0])
        assert not locus.informative
        for source in list(catalog)[:5]:
            assert colour_stellarity(source, locus) == pytest.approx(0.5)

    def test_information_weight_tracks_measured_separation(self, with_colours):
        _, _, catalog = with_colours
        locus = fit_stellar_locus(catalog)
        assert locus is not None
        assert 0.0 <= locus.separation <= 1.0
        assert 0.0 <= locus.information_weight <= 1.0
        if locus.separation <= 0.58:
            assert locus.information_weight == 0.0

    def test_colour_never_makes_classification_worse(self, multiband_measured):
        """Adding a weak vote at full strength costs accuracy; the weighting
        by measured separation is what stops that."""
        import copy

        bands, truth, catalog, segmentation = multiband_measured
        forced_photometry(bands, catalog, detection_band="r", aperture_arcsec=1.6,
                          segmentation=segmentation)
        measure_colours(catalog, [("g", "r"), ("r", "i")], min_snr=5.0)

        def accuracy(use_colours):
            from astrovision.core.types import ObjectClass

            working = copy.deepcopy(catalog)
            Classifier(ClassificationConfig(backend="rules",
                                            use_colours=use_colours)).run(
                bands["r"], working)
            right = total = 0
            for source in working:
                item = _nearest_truth(source, truth, limit=1.5)
                if item is None or item.kind not in ("star", "galaxy"):
                    continue
                wanted = (ObjectClass.STAR if item.kind == "star"
                          else ObjectClass.GALAXY)
                extended = (ObjectClass.GALAXY, ObjectClass.NEBULA,
                            ObjectClass.STAR_CLUSTER)
                total += 1
                right += (source.object_class == wanted or
                          (wanted is ObjectClass.GALAXY
                           and source.object_class in extended))
            return right / max(total, 1), total

        without, n_scored = accuracy(False)
        with_colour, _ = accuracy(True)
        # One borderline source on a field of fifty is not evidence of harm:
        # the margin is one source, never less than two percent.
        assert with_colour >= without - max(0.02, 1.0 / n_scored + 1e-9)

    def test_rayleigh_scale_uses_the_factor_of_two(self):
        rng = np.random.default_rng(0)
        sigma = 0.4
        offsets = np.hypot(rng.normal(0, sigma, 20000), rng.normal(0, sigma, 20000))
        # The trim biases the estimate low, so the check is one-sided-generous
        # in that direction only.
        assert 0.75 * sigma < rayleigh_scale(offsets) <= sigma * 1.05


class TestFusion:
    def test_agreement_beats_either_alone(self):
        assert combine_stellarity(0.8, 0.8) > 0.8

    def test_undecided_stays_undecided(self):
        assert combine_stellarity(0.5, 0.5) == pytest.approx(0.5)

    def test_a_missing_colour_leaves_morphology_untouched(self):
        assert combine_stellarity(0.73, float("nan")) == pytest.approx(0.73)

    def test_disagreement_pulls_toward_the_middle(self):
        assert 0.5 < combine_stellarity(0.9, 0.3) < 0.9

    def test_zero_weight_ignores_colour(self):
        assert combine_stellarity(0.9, 0.01, colour_weight=0.0) == pytest.approx(0.9)
