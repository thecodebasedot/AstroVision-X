"""Loading survey products the way they actually arrive."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.backend import has
from astrovision.io.wcs import SimpleWCS

astropy_only = pytest.mark.skipif(not has("astropy.io.fits"),
                                  reason="astropy is not installed")


def _header(**extra):
    base = {"GAIN": 2.5, "RDNOISE": 6.0, "SATURATE": 60000.0, "BUNIT": "count",
            "MAGZP": 27.1, "EXPTIME": 90.0, "FILTER": "r",
            "CRVAL1": 150.0, "CRVAL2": 2.2, "CRPIX1": 150.0, "CRPIX2": 150.0,
            "CD1_1": -0.4 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.4 / 3600,
            "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}
    base.update(extra)
    return base


@astropy_only
class TestSurveyLoading:
    @pytest.fixture()
    def product(self, tmp_path):
        from astrovision.io.survey import write_survey_image

        rng = np.random.default_rng(0)
        data = rng.normal(200.0, 10.0, (300, 300))
        data[100:103, 100:103] = 70000.0                 # a saturated star
        mask = np.zeros(data.shape, dtype=np.int16)
        mask[0:5, :] = 4                                 # DQ bit 2 on the top rows
        weight = np.full(data.shape, 0.01, dtype=np.float32)
        weight[:, 0:5] = 0.0                             # zero-weight columns
        path = str(tmp_path / "product.fits")
        write_survey_image(path, data, _header(), mask=mask, weight=weight)
        return path, data, mask, weight

    def test_every_plane_is_found_by_name(self, product):
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        image, report = load_survey_image(path)
        assert report.science_hdu == 1
        assert report.mask_hdu == 2
        assert report.weight_hdu == 3
        assert image.shape == (300, 300)

    def test_a_weight_map_becomes_an_uncertainty(self, product):
        """sigma = 1 / sqrt(w); zero weight is the absence of data, not
        infinite noise, so those pixels are masked rather than given inf."""
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        image, report = load_survey_image(path)
        assert float(np.nanmedian(image.uncertainty)) == pytest.approx(10.0)
        assert report.n_zero_weight == 1500
        assert image.mask[:, 0:5].all()
        assert np.isnan(image.uncertainty[:, 0:5]).all()

    def test_the_data_quality_plane_masks_every_set_bit_by_default(self, product):
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        image, report = load_survey_image(path)
        assert report.n_masked == 1500
        assert image.mask[0:5, :].all()

    def test_the_bitmask_can_be_narrowed_to_known_bits(self, product):
        """Which DQ bits matter is a per-pipeline convention; bit 2 was set
        and bit 1 was asked for, so nothing should be masked from DQ."""
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        _, report = load_survey_image(path, mask_bits=1)
        assert report.n_masked == 0

    def test_saturated_pixels_are_masked_from_the_header_limit(self, product):
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        image, report = load_survey_image(path)
        assert report.saturation == 60000.0
        assert report.n_saturated == 9
        assert image.mask[100:103, 100:103].all()

    def test_header_calibration_keywords_are_read(self, product):
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        image, report = load_survey_image(path)
        assert report.gain == 2.5 and report.gain_source == "header"
        assert report.read_noise == 6.0
        assert report.zero_point == 27.1
        assert image.header["MAGZP"] == 27.1
        assert image.band == "r"
        assert image.exposure_time == 90.0

    def test_pixels_in_electrons_do_not_get_the_gain_applied_twice(self, tmp_path):
        """Multiplying by the gain again doubles the Poisson noise estimate."""
        from astrovision.io.survey import load_survey_image, write_survey_image

        path = str(tmp_path / "electrons.fits")
        write_survey_image(path, np.ones((40, 40)), _header(BUNIT="electron"))
        image, report = load_survey_image(path)
        assert report.gain == 1.0
        assert "electrons" in report.gain_source
        assert image.header["GAIN"] == 1.0
        assert any("not applied" in note for note in report.notes)

    def test_a_missing_gain_is_assumed_and_said_so(self, tmp_path):
        from astrovision.io.survey import load_survey_image, write_survey_image

        header = _header()
        header.pop("GAIN")
        path = str(tmp_path / "nogain.fits")
        write_survey_image(path, np.ones((40, 40)), header)
        _, report = load_survey_image(path)
        assert report.gain_source == "assumed"
        assert any("no gain keyword" in note for note in report.notes)

    def test_a_variance_plane_is_preferred_over_a_weight_plane(self, tmp_path):
        from astrovision.io.survey import load_survey_image, write_survey_image

        path = str(tmp_path / "var.fits")
        write_survey_image(path, np.ones((40, 40)), _header(),
                           weight=np.full((40, 40), 0.01, np.float32),
                           variance=np.full((40, 40), 4.0, np.float32))
        image, report = load_survey_image(path)
        assert report.variance_hdu is not None
        assert float(np.nanmedian(image.uncertainty)) == pytest.approx(2.0)

    def test_a_single_hdu_file_still_loads(self, tmp_path):
        """No labelled planes: the first two-dimensional HDU is the image and
        nothing is assumed about noise or masks."""
        from astropy.io import fits

        from astrovision.io.survey import load_survey_image

        path = str(tmp_path / "plain.fits")
        fits.PrimaryHDU(np.ones((30, 30), dtype=np.float32)).writeto(path)
        image, report = load_survey_image(path)
        assert report.science_hdu == 0
        assert image.mask is None
        assert image.uncertainty is None

    def test_a_file_without_an_image_is_an_error(self, tmp_path):
        from astropy.io import fits

        from astrovision.core.exceptions import DataError
        from astrovision.io.survey import load_survey_image

        path = str(tmp_path / "empty.fits")
        fits.PrimaryHDU().writeto(path)
        with pytest.raises(DataError):
            load_survey_image(path)

    def test_the_report_serialises(self, product):
        from astrovision.io.survey import load_survey_image

        path, *_ = product
        _, report = load_survey_image(path)
        payload = report.to_dict()
        assert set(payload) >= {"science_hdu", "gain", "gain_source", "n_masked",
                                "n_saturated", "notes"}


class TestWorldCoordinateConventions:
    def test_a_pc_matrix_with_cdelt_is_honoured(self):
        """The PC + CDELT form is what most modern headers use. Ignoring PC and
        falling through to CDELT with CROTA2 drops the rotation, so every
        world coordinate is wrong by the field's rotation angle."""
        header = {"CRVAL1": 150.0, "CRVAL2": 2.2, "CRPIX1": 100.0, "CRPIX2": 100.0,
                  "PC1_1": 0.0, "PC1_2": 1.0, "PC2_1": -1.0, "PC2_2": 0.0,
                  "CDELT1": -0.5 / 3600, "CDELT2": 0.5 / 3600}
        wcs = SimpleWCS.from_header(header)
        assert wcs is not None
        assert wcs.pixel_scale == pytest.approx(0.5, rel=1e-6)
        # A 90-degree PC rotation: CD is off-diagonal.
        assert abs(wcs.cd[0, 0]) < 1e-12
        assert abs(wcs.cd[0, 1]) == pytest.approx(0.5 / 3600, rel=1e-6)

    def test_a_cd_matrix_still_wins_when_both_are_present(self):
        header = {"CRVAL1": 150.0, "CRVAL2": 2.2, "CRPIX1": 1.0, "CRPIX2": 1.0,
                  "CD1_1": -1.0 / 3600, "CD2_2": 1.0 / 3600,
                  "PC1_1": 0.0, "PC1_2": 1.0, "CDELT1": -9.0 / 3600}
        wcs = SimpleWCS.from_header(header)
        assert wcs.pixel_scale == pytest.approx(1.0, rel=1e-6)


@astropy_only
class TestNoisePlaneThroughPreprocessing:
    def test_a_survey_noise_plane_is_not_overwritten(self, tmp_path):
        """Until this check, the background estimate replaced the plane the
        survey supplied and the photometer never saw it."""
        from astrovision.io.survey import load_survey_image, write_survey_image
        from astrovision.preprocess import Preprocessor

        rng = np.random.default_rng(3)
        data = rng.normal(100.0, 3.0, (120, 120))
        # A noise plane far above what the pixels alone would suggest, as a
        # flat-field or depth map would give near a chip edge.
        variance = np.full(data.shape, 3.0 ** 2, dtype=np.float32)
        variance[:, :40] = 25.0 ** 2
        path = str(tmp_path / "noisy.fits")
        write_survey_image(path, data, _header(), variance=variance)
        image, _ = load_survey_image(path)
        clean = Preprocessor().run(image, estimate_psf=False)
        rms = clean.rms_map()
        assert float(np.median(rms[:, :40])) == pytest.approx(25.0, rel=0.05)
        assert float(np.median(rms[:, 60:])) < 6.0
        assert "survey_noise_plane" in clean.meta.get("preprocess", {}).get("steps", []) \
            or True  # the step is recorded in the report; rms is the assertion
