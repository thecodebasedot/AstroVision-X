"""What running real images taught the loaders, and the fixes that came of it.

Each test here reproduces something a real file did that the simulator never
does: a Digitized Sky Survey plate with a multi-line comment and a plate
solution instead of a WCS, a Spitzer mosaic in Galactic coordinates with a
channel number instead of a filter name, a pre-2000 date, an alert carrying
ZTF's forced photometry.  The files themselves are not in the repository;
the headers below carry just the cards that mattered.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.backend import try_import
from astrovision.io.image import _header_band, _header_time
from astrovision.io.wcs import SimpleWCS, angular_separation, wcs_from_header

astropy = try_import("astropy.wcs")
needs_astropy = pytest.mark.skipif(astropy is None, reason="astropy not installed")


def _galactic_header(comment_value: str = "") -> dict:
    header = {"NAXIS": 2, "NAXIS1": 300, "NAXIS2": 200,
              "CTYPE1": "GLON-CAR", "CTYPE2": "GLAT-CAR",
              "CRPIX1": 150.0, "CRPIX2": 100.0, "CRVAL1": 18.0, "CRVAL2": 0.0,
              "CDELT1": -1.2 / 3600.0, "CDELT2": 1.2 / 3600.0}
    if comment_value:
        header["OBJECT"] = comment_value
    return header


class TestFramesAndProjectionsBecomeIcrs:
    @needs_astropy
    def test_a_galactic_frame_is_refit_to_icrs(self):
        """The Spitzer mosaic: GLON/GLAT axes.  Reported as-is they would have
        been written into the catalog's ``ra``/``dec`` columns."""
        from astropy.coordinates import SkyCoord
        header = _galactic_header()
        wcs = wcs_from_header(header)
        assert wcs is not None and "ICRS" in wcs.derived_from
        gx, gy = np.meshgrid(np.linspace(0, 299, 7), np.linspace(0, 199, 7))
        aw = astropy.WCS(header)
        truth = aw.pixel_to_world(gx.ravel(), gy.ravel()).icrs
        ra, dec = wcs.pixel_to_world(gx.ravel(), gy.ravel())
        error_px = angular_separation(ra, dec, truth.ra.deg, truth.dec.deg) * 3600 / wcs.pixel_scale
        assert error_px.max() < 0.02
        centre = SkyCoord(l=18.0, b=0.0, unit="deg", frame="galactic").icrs
        ra0, dec0 = wcs.pixel_to_world(149.0, 99.0)
        assert angular_separation(ra0, dec0, centre.ra.deg, centre.dec.deg) * 3600 < 0.05
        assert wcs.pixel_scale == pytest.approx(1.2, abs=1e-3)

    @needs_astropy
    def test_a_header_astropy_would_refuse_still_yields_the_wcs(self):
        """The DSS plate's header holds a multi-line string; astropy raises
        on it and the whole plate solution used to be lost."""
        bad = "National Geographic Society -\nPalomar Observatory Sky Survey"
        with_comment = wcs_from_header(_galactic_header(bad))
        without = wcs_from_header(_galactic_header())
        assert with_comment is not None
        assert np.allclose(with_comment.crval, without.crval)

    @needs_astropy
    def test_a_polynomial_distortion_is_refit_with_sip_terms(self):
        """A TPV header: wcslib reads the polynomial, the linear parser
        cannot.  The refit must carry it as SIP and reproduce astropy to
        well under a hundredth of a pixel, where the linear reading is off
        by a tenth."""
        header = {"NAXIS": 2, "NAXIS1": 400, "NAXIS2": 400,
                  "CTYPE1": "RA---TPV", "CTYPE2": "DEC--TPV",
                  "CRPIX1": 200.0, "CRPIX2": 200.0, "CRVAL1": 132.8, "CRVAL2": 11.8,
                  "CD1_1": -1.7 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1.7 / 3600,
                  "PV1_0": 0.0, "PV1_1": 1.0, "PV1_7": 0.05,
                  "PV2_0": 0.0, "PV2_1": 1.0, "PV2_7": -0.05}
        wcs = wcs_from_header(header)
        assert wcs is not None and wcs.has_distortion and "SIP" in wcs.derived_from
        aw = astropy.WCS(header)
        gx, gy = np.meshgrid(np.linspace(0, 399, 9), np.linspace(0, 399, 9))
        truth = aw.pixel_to_world(gx.ravel(), gy.ravel()).icrs
        ra, dec = wcs.pixel_to_world(gx.ravel(), gy.ravel())
        error_px = angular_separation(ra, dec, truth.ra.deg, truth.dec.deg) * 3600 / wcs.pixel_scale
        assert error_px.max() < 0.01
        px, py = wcs.world_to_pixel(truth.ra.deg, truth.dec.deg)
        assert np.hypot(px - gx.ravel(), py - gy.ravel()).max() < 0.01
        linear = SimpleWCS.from_header(header)
        ra, dec = linear.pixel_to_world(gx.ravel(), gy.ravel())
        assert (angular_separation(ra, dec, truth.ra.deg, truth.dec.deg) * 3600
                / wcs.pixel_scale).max() > 0.05

    @needs_astropy
    def test_a_plain_equatorial_header_keeps_its_sip_terms(self):
        """With astropy installed the SIP terms used to be dropped on the
        way through; the direct parser keeps them."""
        header = SimpleWCS.tangent(10.0, 20.0, (256, 256), 0.5).to_header()
        header.update({"A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1e-5, "B_0_2": 1e-5,
                       "NAXIS1": 256, "NAXIS2": 256})
        header["CTYPE1"], header["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
        wcs = wcs_from_header(header)
        assert wcs.has_distortion and not wcs.derived_from

    def test_without_astropy_a_galactic_header_is_at_least_not_mislabelled(self):
        """The fallback parser cannot convert frames; it must not pretend."""
        wcs = SimpleWCS.from_header(_galactic_header())
        assert wcs.ctype == ("GLON-CAR", "GLAT-CAR")


class TestHeaderConventions:
    def test_the_pre_2000_date_form_is_read(self):
        """DSS plates: ``DATE-OBS = '29/11/51'`` is 1951 November 29."""
        assert _header_time({"DATE-OBS": "29/11/51"}) == pytest.approx(33979.0)
        assert _header_time({"DATE-OBS": "2006-12-02T01:06:10"}) == pytest.approx(54071.0459, abs=1e-3)
        assert _header_time({"DATE-OBS": "31/02/51"}) is None
        assert _header_time({"DATE-OBS": "not a date"}) is None

    def test_an_instrument_channel_names_the_band(self):
        """Spitzer IRAC writes ``CHNLNUM`` and no filter keyword."""
        assert _header_band({"INSTRUME": "IRAC", "CHNLNUM": 2}) == "IRAC2"
        assert _header_band({"FILTER": "r", "INSTRUME": "IRAC", "CHNLNUM": 2}) == "r"
        assert _header_band({"CHNLNUM": 2}) == "clear"
        assert _header_band({}, "unknown") == "unknown"


class TestForcedPhotometryInAlerts:
    def test_ztf_forced_photometry_joins_the_history(self):
        from astrovision.alerts import AlertPacket
        record = {"schemavsn": "4.02", "publisher": "ZTF", "objectId": "ZTF18a", "candid": 5,
                  "candidate": {"jd": 2459000.5, "fid": 2, "pid": 1, "candid": 5,
                                "isdiffpos": "t", "ra": 10.0, "dec": -5.0,
                                "magpsf": 17.2, "sigmapsf": 0.03},
                  "prv_candidates": None,
                  "fp_hists": [
                      {"jd": 2458990.5, "fid": 2, "pid": 2, "rfid": 1, "programid": 1,
                       "ranr": 10.0, "decnr": -5.0, "forcediffimflux": 500.0,
                       "forcediffimfluxunc": 20.0, "magzpsci": 26.0, "diffmaglim": 20.5},
                      {"jd": 2458980.5, "fid": 1, "pid": 3, "rfid": 1, "programid": 1,
                       "ranr": 10.0, "decnr": -5.0, "forcediffimflux": 10.0,
                       "forcediffimfluxunc": 20.0, "magzpsci": 26.0, "diffmaglim": 20.7},
                      {"jd": 2458970.5, "fid": 1, "pid": 4, "rfid": 1, "programid": 1,
                       "ranr": 10.0, "decnr": -5.0, "forcediffimflux": None,
                       "forcediffimfluxunc": None}],
                  "cutoutScience": None, "cutoutTemplate": None, "cutoutDifference": None}
        p = AlertPacket.from_record(record)
        assert len(p.history) == 2 and all(d.forced for d in p.history)
        strong, weak = p.history
        assert strong.band == "r" and strong.mag == pytest.approx(26.0 - 2.5 * np.log10(500.0))
        assert strong.is_detection and strong.flux == 500.0
        assert weak.mag is None and not weak.is_detection and weak.limiting_mag == 20.7
        assert len(p.detections()) == 1

    @pytest.mark.skipif(try_import("fastavro") is None, reason="fastavro not installed")
    def test_a_file_in_ztfs_own_nested_schema_is_read(self, tmp_path):
        """ZTF's schema is four named records in the ``ztf.alert`` namespace,
        referenced by name from the top level, with the forced-photometry
        array added in 4.x.  fastavro writes it, this package's reader must
        decode it identically."""
        import fastavro
        from astrovision.alerts import read_alerts
        from astrovision.alerts.avro import read_container
        candidate = {"type": "record", "name": "candidate", "namespace": "ztf.alert", "fields": [
            {"name": "jd", "type": "double"}, {"name": "fid", "type": "int"},
            {"name": "pid", "type": "long"}, {"name": "candid", "type": "long"},
            {"name": "isdiffpos", "type": "string"}, {"name": "ra", "type": "double"},
            {"name": "dec", "type": "double"}, {"name": "magpsf", "type": "float"},
            {"name": "sigmapsf", "type": "float"},
            {"name": "diffmaglim", "type": ["null", "float"], "default": None},
            {"name": "rb", "type": ["null", "float"], "default": None},
            {"name": "drb", "type": ["null", "float"], "default": None},
            {"name": "rbversion", "type": "string"}]}
        prv = {"type": "record", "name": "prv_candidate", "namespace": "ztf.alert", "fields": [
            {"name": "jd", "type": "double"}, {"name": "fid", "type": "int"},
            {"name": "candid", "type": ["null", "long"], "default": None},
            {"name": "magpsf", "type": ["null", "float"], "default": None},
            {"name": "sigmapsf", "type": ["null", "float"], "default": None},
            {"name": "diffmaglim", "type": ["null", "float"], "default": None}]}
        fp = {"type": "record", "name": "fp_hist", "namespace": "ztf.alert", "fields": [
            {"name": "jd", "type": "double"}, {"name": "fid", "type": "int"},
            {"name": "ranr", "type": "double"}, {"name": "decnr", "type": "double"},
            {"name": "forcediffimflux", "type": ["null", "float"], "default": None},
            {"name": "forcediffimfluxunc", "type": ["null", "float"], "default": None},
            {"name": "magzpsci", "type": ["null", "float"], "default": None}]}
        cutout = {"type": "record", "name": "cutout", "namespace": "ztf.alert", "fields": [
            {"name": "fileName", "type": "string"}, {"name": "stampData", "type": "bytes"}]}
        alert = {"type": "record", "name": "alert", "namespace": "ztf", "fields": [
            {"name": "schemavsn", "type": "string"}, {"name": "publisher", "type": "string"},
            {"name": "objectId", "type": "string"}, {"name": "candid", "type": "long"},
            {"name": "candidate", "type": candidate},
            {"name": "prv_candidates", "type": ["null", {"type": "array", "items": prv}],
             "default": None},
            {"name": "fp_hists", "type": ["null", {"type": "array", "items": fp}],
             "default": None},
            {"name": "cutoutScience", "type": ["null", cutout], "default": None},
            {"name": "cutoutTemplate", "type": ["null", "ztf.alert.cutout"], "default": None},
            {"name": "cutoutDifference", "type": ["null", "ztf.alert.cutout"], "default": None}]}
        record = {"schemavsn": "4.02", "publisher": "ZTF", "objectId": "ZTF18aaaaaaa",
                  "candid": 472263571115095000,
                  "candidate": {"jd": 2458200.5, "fid": 2, "pid": 1, "candid": 472263571115095000,
                                "isdiffpos": "t", "ra": 150.1, "dec": 2.2, "magpsf": 18.5,
                                "sigmapsf": 0.05, "diffmaglim": 20.3, "rb": 0.9, "drb": 0.95,
                                "rbversion": "t17"},
                  "prv_candidates": [{"jd": 2458198.5, "fid": 1, "candid": 1, "magpsf": 18.9,
                                      "sigmapsf": 0.1, "diffmaglim": None},
                                     {"jd": 2458190.5, "fid": 2, "candid": None, "magpsf": None,
                                      "sigmapsf": None, "diffmaglim": 20.5}],
                  "fp_hists": [{"jd": 2458150.5, "fid": 2, "ranr": 150.1, "decnr": 2.2,
                                "forcediffimflux": 120.0, "forcediffimfluxunc": 15.0,
                                "magzpsci": 26.2}],
                  "cutoutScience": {"fileName": "sci", "stampData": b"\x00\x01\x02"},
                  "cutoutTemplate": None, "cutoutDifference": None}
        path = tmp_path / "ztf.avro"
        with open(path, "wb") as fh:
            fastavro.writer(fh, fastavro.parse_schema(alert), [record])
        with open(path, "rb") as fh:
            schema, records = read_container(fh)
            ours = list(records)
        with open(path, "rb") as fh:
            theirs = list(fastavro.reader(fh))
        assert ours == theirs and schema["name"] in ("alert", "ztf.alert")
        _, packets = read_alerts(str(path))
        p = packets[0]
        assert p.object_id == "ZTF18aaaaaaa" and p.candid == 472263571115095000
        assert p.band == "r" and p.mag == pytest.approx(18.5, abs=1e-5)
        assert p.deep_real_bogus == pytest.approx(0.95, abs=1e-5)
        assert [d.forced for d in p.history] == [False, False, True]
        assert p.history[2].mag == pytest.approx(26.2 - 2.5 * np.log10(120.0), abs=1e-4)
        assert p.cutout_science is None          # three bytes are not a stamp


class TestSersicConvolution:
    def test_padded_fft_convolution_is_the_exact_linear_convolution(self):
        from astrovision.morphology.sersic import _PaddedConvolver
        rng = np.random.default_rng(3)
        kernel = rng.random((7, 9))
        kernel /= kernel.sum()
        conv = _PaddedConvolver(kernel, (20, 30))
        pad = conv.pad
        padded = rng.normal(size=(20 + 2 * pad, 30 + 2 * pad))
        # Direct sum for every pixel of the cutout, over the padded model.
        ky, kx = kernel.shape
        direct = np.zeros((20, 30))
        for j in range(20):
            for i in range(30):
                cy, cx = j + pad, i + pad
                block = padded[cy - ky // 2:cy + ky // 2 + 1, cx - kx // 2:cx + kx // 2 + 1]
                direct[j, i] = (block * kernel[::-1, ::-1]).sum()
        assert np.allclose(conv(padded), direct, atol=1e-12)

    def test_the_fit_still_recovers_the_index_through_a_psf(self):
        from astrovision.morphology.sersic import fit_sersic
        from astrovision.simulate.profiles import sersic_profile
        size = 61
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.hypot(xx - 30, yy - 30)
        truth = sersic_profile(r, 100.0, 6.0, 2.5)
        kernel = np.exp(-0.5 * (np.hypot(*np.mgrid[-5:6, -5:6]) / 1.3) ** 2)
        kernel /= kernel.sum()
        scipy_signal = try_import("scipy.signal")
        if scipy_signal is None:
            pytest.skip("SciPy needed to make the blurred truth")
        blurred = scipy_signal.fftconvolve(truth, kernel, mode="same")
        image = blurred + np.random.default_rng(0).normal(scale=0.5, size=truth.shape)
        fit = fit_sersic(image, centre=(30.0, 30.0), psf=kernel, r_half=6.0, noise=0.5)
        assert fit.success and abs(fit.n - 2.5) < 0.5 and abs(fit.r_eff - 6.0) < 1.5


class TestUndersampledImagesAreSaidToBe:
    def test_the_pipeline_warns_when_the_psf_is_undersampled(self):
        from astrovision.engine.pipeline import Pipeline, UNDERSAMPLED_FWHM_PX
        from astrovision.core.types import FieldAnalysis
        from astrovision.io import AstroImage

        class Psf:
            fwhm = UNDERSAMPLED_FWHM_PX - 0.3

        image = AstroImage.from_array(np.zeros((8, 8)))
        image.meta["psf_model"] = Psf()
        image.meta["survey_load"] = {"unit": "MJy/sr", "zero_point": float("nan")}
        analysis = FieldAnalysis()
        Pipeline()._data_quality_warnings(analysis, image)
        assert any("undersampled" in w for w in analysis.warnings)
        assert any("MJy/sr" in w and "zero point" in w for w in analysis.warnings)
        image.header["MAGZP"] = 25.0
        Psf.fwhm = 3.0
        clean = FieldAnalysis()
        Pipeline()._data_quality_warnings(clean, image)
        assert clean.warnings == []
