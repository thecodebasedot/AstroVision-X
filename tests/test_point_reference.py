"""A star is the PSF: the star/galaxy votes measured against the PSF stamp."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.classify.rules import point_source_reference, stellarity
from astrovision.core.types import BoundingBox, Source
from astrovision.preprocess.psf import PSFModel


def _gaussian_psf(fwhm: float, size: int = 31) -> PSFModel:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2.0
    stamp = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return PSFModel(stamp=stamp / stamp.sum(), fwhm=fwhm, size=size)


def _winged_psf(fwhm: float, size: int = 41) -> PSFModel:
    """A Gaussian core with a broad halo, as diffraction and plates have."""
    core = _gaussian_psf(fwhm, size).stamp
    halo = _gaussian_psf(4 * fwhm, size).stamp
    stamp = 0.6 * core + 0.4 * halo
    return PSFModel(stamp=stamp / stamp.sum(), fwhm=fwhm, size=size)


class TestPointSourceReference:
    def test_a_gaussian_psf_reproduces_the_analytic_sizes(self):
        ref = point_source_reference(_gaussian_psf(4.0), size_threshold=0.0)
        assert ref["source"] == "psf_stamp"
        # A Gaussian's half-light radius is 0.5 FWHM, its r90 is 0.91 FWHM.
        assert ref["r50"] == pytest.approx(2.0, abs=0.12)
        assert ref["r90"] == pytest.approx(0.91 * 4.0, abs=0.2)
        assert ref["fwhm"] == pytest.approx(4.0, abs=0.15)
        assert ref["peak_fraction"] == pytest.approx(1.0 / (1.13 * 16.0), rel=0.05)

    def test_wings_make_a_star_larger_than_its_core_says(self):
        core = point_source_reference(_gaussian_psf(3.0), size_threshold=0.0)
        winged = point_source_reference(_winged_psf(3.0), size_threshold=0.0)
        assert winged["r50"] > 1.3 * core["r50"]
        assert winged["r90"] > 2 * core["r90"]
        assert winged["peak_fraction"] < 0.7 * core["peak_fraction"]

    def test_the_isophote_narrows_the_width_reference(self):
        psf = _winged_psf(3.0)
        whole = point_source_reference(psf, size_threshold=0.0)["fwhm"]
        core = point_source_reference(psf, size_threshold=0.2)["fwhm"]
        assert core < whole

    def test_an_empty_stamp_falls_back_to_the_analytic_form(self):
        ref = point_source_reference(PSFModel(stamp=np.zeros((5, 5)), fwhm=2.0))
        assert ref["source"] == "analytic" and ref["r50"] == pytest.approx(1.0)


def _source(r50, fwhm, peak_fraction, flux=1000.0):
    source = Source(id=1, x=10.0, y=10.0, bbox=BoundingBox(0, 0, 20, 20))
    source.meta["r50"] = r50
    source.morphology.fwhm = fwhm
    source.morphology.area_pixels = 40
    source.photometry.flux = flux
    source.photometry.peak = peak_fraction * flux
    source.photometry.snr = 50.0
    return source


class TestStellarityAgainstTheStamp:
    def test_a_star_seen_through_a_winged_psf_is_still_a_star(self):
        """Measured against the analytic half-FWHM a winged PSF makes every
        star look resolved; measured against the stamp it does not."""
        psf = _winged_psf(2.0)
        ref = point_source_reference(psf)
        star = _source(r50=ref["r50"], fwhm=ref["fwhm"], peak_fraction=ref["peak_fraction"])
        against_stamp = stellarity(star, psf.fwhm, ref["r90"], point_reference=ref)
        against_analytic = stellarity(star, psf.fwhm, ref["r90"])
        assert against_stamp > 0.8
        assert against_stamp > against_analytic + 0.2

    def test_a_galaxy_is_still_a_galaxy(self):
        psf = _gaussian_psf(3.0)
        ref = point_source_reference(psf)
        galaxy = _source(r50=2.5 * ref["r50"], fwhm=2.3 * ref["fwhm"],
                         peak_fraction=0.3 * ref["peak_fraction"])
        assert stellarity(galaxy, psf.fwhm, ref["r90"], point_reference=ref) < 0.2
        star = _source(r50=1.0 * ref["r50"], fwhm=1.1 * ref["fwhm"],
                       peak_fraction=ref["peak_fraction"])
        assert stellarity(star, psf.fwhm, ref["r90"], point_reference=ref) > 0.8

    def test_the_pipeline_records_the_reference_and_gates_the_lens_search(self):
        from astrovision.classify import Classifier
        from astrovision.core.config import LensingConfig
        from astrovision.detect import Detector
        from astrovision.lensing import LensSearch
        from astrovision.photometry import Photometer
        from astrovision.preprocess import Preprocessor
        from astrovision.simulate import SkyConfig, SkySimulator

        image, truth = SkySimulator(SkyConfig(shape=(256, 256), n_stars=60, n_galaxies=0,
                                              n_nebulae=0, n_clusters=0, n_lenses=0,
                                              n_anomalies=0, seed=5)).generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)
        Classifier().run(clean, catalog)
        assert clean.meta["point_source_reference"]["source"] == "psf_stamp"
        stars = sum(1 for s in catalog if s.object_class.value == "star")
        assert stars >= 0.9 * len(catalog)
        search = LensSearch(LensingConfig(fit_model=False))
        search.run(clean, catalog)
        # A field of stars has nothing a lens search should examine.
        assert search.report["n_examined"] <= 0.1 * len(catalog)
        assert search.report["min_deflector_r50_px"] > 0
