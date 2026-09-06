"""The aperture correction from the field's own stars, and its use as a check."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.config import PhotometryConfig
from astrovision.core.types import BoundingBox, Source, SourceCatalog
from astrovision.photometry.growthcurve import (GrowthCurve, _missing_beyond, build_growth_curve,
                                               select_growth_stars)


@pytest.fixture(scope="module")
def field():
    from astrovision.detect import Detector
    from astrovision.preprocess import Preprocessor
    from astrovision.simulate import SkyConfig, SkySimulator

    image, truth = SkySimulator(SkyConfig(shape=(512, 512), n_stars=120, n_galaxies=60,
                                          n_nebulae=0, n_clusters=0, n_lenses=0,
                                          n_anomalies=0, seed=3)).generate()
    clean = Preprocessor().run(image)
    catalog, segmentation = Detector().detect(clean)
    return clean, catalog, segmentation, truth


def _source(ident, x, y, flux, snr=100.0, fwhm=3.0, flags=()):
    source = Source(id=ident, x=x, y=y, bbox=BoundingBox(0, 0, 1, 1))
    source.photometry.flux = flux
    source.photometry.snr = snr
    source.morphology.fwhm = fwhm
    for flag in flags:
        source.add_flag(flag)
    return source


class TestSelection:
    def test_bright_isolated_point_sources_qualify_and_others_do_not(self):
        catalog = SourceCatalog([
            _source(1, 100, 100, 1e5),                                   # good
            _source(2, 300, 100, 1e5, flags=("saturated",)),             # saturated
            _source(3, 100, 300, 1e5, snr=10),                           # faint
            _source(4, 300, 300, 1e5, fwhm=9.0),                         # resolved
            _source(5, 200, 200, 1e5),                                   # has a bright neighbour
            _source(6, 210, 205, 2e4),                                   # ... this one
            _source(7, 400, 400, 1e5),                                   # faint neighbour only
            _source(8, 405, 405, 1e2),
            _source(9, 5, 5, 1e5),                                       # at the edge
        ])
        chosen = select_growth_stars(catalog, psf_fwhm=3.0, far_radius=15.0,
                                     shape=(512, 512))
        assert [s.id for s in chosen] == [1, 7]

    def test_the_brightest_come_first_and_are_capped(self):
        catalog = SourceCatalog([_source(i, 60 * i, 60 * (i % 3 + 1), 1e4 * i, snr=50.0 * i)
                                 for i in range(1, 8)])
        chosen = select_growth_stars(catalog, 3.0, 12.0, max_stars=3, shape=(512, 512))
        assert [s.id for s in chosen] == [7, 6, 5]


class TestWingExtrapolation:
    def test_a_power_law_deficit_is_read_off_the_outer_curve(self):
        radii = np.linspace(1.0, 20.0, 40)
        deficit = 0.5 * radii ** -1.2                  # 1 - E(r) for r^-1.2 wings
        enclosed = 1.0 - deficit
        missing = _missing_beyond(radii, enclosed)
        assert missing == pytest.approx(0.5 * 20.0 ** -1.2, rel=0.05)

    def test_a_flat_or_noisy_curve_claims_nothing(self):
        radii = np.linspace(1.0, 20.0, 40)
        assert _missing_beyond(radii, np.ones_like(radii)) == 0.0
        assert _missing_beyond(radii, 1.0 - 0.01 * np.ones_like(radii)) == 0.0   # no slope


class TestTheCurve:
    def test_it_matches_the_psf_stamp_and_the_truth_on_the_simulator(self, field):
        from astrovision.photometry.photometer import Photometer
        clean, catalog, _, truth = field
        psf = clean.meta["psf_model"]
        curve = build_growth_curve(clean.subtracted(), catalog, psf.fwhm)
        assert curve is not None and curve.n_stars >= 5
        assert curve.far_radius > 3 * psf.fwhm
        # Half the light inside about half a FWHM, nearly all inside the far radius.
        assert 0.35 * psf.fwhm < np.interp(0.5, curve.enclosed, curve.radii) < 0.8 * psf.fwhm
        assert curve.enclosed[-1] >= 0.98
        # Against the stamp's own enclosed energy at the primary aperture.
        stamp = Photometer.aperture_correction(psf, 5.0)
        assert curve.correction(5.0) == pytest.approx(stamp, rel=0.02)
        assert curve.uncertainty(5.0) < 0.05
        # Against the injected fluxes of bright stars measured in 5 px.
        from astrovision.photometry.growth import curve_of_growth
        stars = [t for t in truth if t.kind == "star"]
        stars.sort(key=lambda t: -t.flux)
        ratios = []
        for star in stars[5:25]:
            _, cumulative = curve_of_growth(clean.subtracted(), (star.x, star.y), [5.0])
            ratios.append(cumulative[0] * curve.correction(5.0) / star.flux)
        assert np.median(ratios) == pytest.approx(1.0, abs=0.02)

    def test_corrections_are_one_outside_what_it_knows(self):
        curve = GrowthCurve(radii=np.linspace(1, 10, 10), enclosed=np.linspace(0.5, 1.0, 10),
                            scatter=np.zeros(10), far_radius=10.0, n_stars=6)
        assert curve.correction(float("nan")) == 1.0 and curve.correction(-1) == 1.0
        assert curve.correction(10.0) == 1.0 and curve.correction(1.0) == pytest.approx(2.0)
        assert curve.uncertainty(5.0) == 0.0
        assert GrowthCurve(radii=np.array([1.0]), enclosed=np.array([1.0]),
                           scatter=np.array([0.0]), far_radius=1.0, n_stars=0).correction(0.5) == 1.0


class TestInThePhotometer:
    def test_auto_keeps_the_stamp_and_records_the_check(self, field):
        import pickle

        from astrovision.photometry import Photometer
        clean, catalog, segmentation, _ = field
        working = pickle.loads(pickle.dumps(catalog))      # the fixture's sources stay clean
        photometer = Photometer(PhotometryConfig(aperture_correction="auto"))
        photometer.run(clean, working, segmentation)
        report = photometer.report
        assert report["aperture_correction_source"] == "psf_model"
        check = report["growth_curve"]
        assert check["n_stars"] >= 5 and not check["applied"]
        assert check["correction_at_primary"] == pytest.approx(
            check["psf_stamp_correction_at_primary"], rel=0.03)

    def test_stars_mode_applies_the_curve(self, field):
        import pickle

        from astrovision.photometry import Photometer
        clean, catalog, segmentation, _ = field
        working = pickle.loads(pickle.dumps(catalog))
        photometer = Photometer(PhotometryConfig(aperture_correction="stars"))
        photometer.run(clean, working, segmentation)
        assert photometer.report["aperture_correction_source"] == "field_stars"
        assert photometer.report["growth_curve"]["applied"]
        curve = clean.meta["growth_curve"]
        corrected = [s for s in working if "aperture_correction" in s.meta]
        assert corrected and all(
            s.meta["aperture_correction"] == pytest.approx(
                curve.correction(s.photometry.aperture_radius))
            for s in corrected)
        assert all("aperture_correction_err" in s.meta for s in corrected)

    def test_none_mode_applies_nothing(self, field):
        import pickle

        from astrovision.photometry import Photometer
        clean, catalog, segmentation, _ = field
        working = pickle.loads(pickle.dumps(catalog))
        photometer = Photometer(PhotometryConfig(aperture_correction="none"))
        photometer.run(clean, working, segmentation)
        assert photometer.report["aperture_correction_source"] == "none"
        assert not photometer.report["aperture_corrected"]
        assert all("aperture_correction" not in s.meta for s in working)


class TestTheWarning:
    def test_a_field_whose_stars_disagree_is_said_to_be_uncertain(self):
        from astrovision.core.types import FieldAnalysis
        from astrovision.engine.pipeline import Pipeline
        from astrovision.io import AstroImage

        pipeline = Pipeline()
        image = AstroImage.from_array(np.zeros((8, 8)))
        pipeline.photometer.report = {"growth_curve": {
            "n_stars": 10, "correction_at_primary": 1.25, "uncertainty_at_primary": 0.15,
            "psf_stamp_correction_at_primary": 1.08}}
        analysis = FieldAnalysis()
        pipeline._data_quality_warnings(analysis, image)
        assert any("disagree on the aperture correction by 15 %" in w for w in analysis.warnings)

        pipeline.photometer.report = {"growth_curve": {
            "n_stars": 10, "correction_at_primary": 1.20, "uncertainty_at_primary": 0.01,
            "psf_stamp_correction_at_primary": 1.08}}
        analysis = FieldAnalysis()
        pipeline._data_quality_warnings(analysis, image)
        assert any("11 % apart" in w for w in analysis.warnings)

        pipeline.photometer.report = {"growth_curve": {
            "n_stars": 10, "correction_at_primary": 1.085, "uncertainty_at_primary": 0.01,
            "psf_stamp_correction_at_primary": 1.08}}
        analysis = FieldAnalysis()
        pipeline._data_quality_warnings(analysis, image)
        assert analysis.warnings == []
