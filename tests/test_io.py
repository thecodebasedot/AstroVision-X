"""FITS, image containers, WCS and catalog serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.exceptions import DataError
from astrovision.io import catalog as catalog_io
from astrovision.io.fits import _read_fits_numpy, _write_fits_numpy, is_fits, read_fits, write_fits
from astrovision.io.image import AstroImage, ImageSeries
from astrovision.io.wcs import SimpleWCS, angular_separation


@pytest.fixture()
def array():
    return np.random.default_rng(0).normal(100.0, 5.0, (48, 36)).astype(np.float32)


class TestFits:
    def test_round_trip(self, tmp_path, array):
        path = str(tmp_path / "a.fits")
        write_fits(path, array, {"OBJECT": "M51", "EXPTIME": 300.0})
        data, header = read_fits(path)
        assert data.shape == array.shape
        assert np.allclose(data, array, atol=1e-4)
        assert header["OBJECT"] == "M51"

    def test_builtin_writer_is_readable_by_astropy(self, tmp_path, array):
        path = str(tmp_path / "b.fits")
        _write_fits_numpy(path, array, {"OBJECT": "NGC1234"})
        data, header = read_fits(path)
        assert np.allclose(data, array, atol=1e-4)
        assert header["OBJECT"] == "NGC1234"

    def test_builtin_reader_handles_astropy_files(self, tmp_path, array):
        path = str(tmp_path / "c.fits")
        write_fits(path, array, {"GAIN": 2.5})
        data, header = _read_fits_numpy(path)
        assert np.allclose(data, array, atol=1e-4)
        assert header["GAIN"] == 2.5

    def test_missing_file_raises(self):
        with pytest.raises(DataError):
            read_fits("/nonexistent/file.fits")

    @pytest.mark.parametrize("name,expected", [
        ("a.fits", True), ("a.fit.gz", True), ("a.png", False)])
    def test_is_fits(self, name, expected):
        assert is_fits(name) is expected


class TestWCS:
    def test_pixel_world_round_trip(self):
        wcs = SimpleWCS.tangent(150.0, 2.0, (512, 512), 0.4)
        ra, dec = wcs.pixel_to_world([10.0, 250.0], [20.0, 300.0])
        x, y = wcs.world_to_pixel(ra, dec)
        assert np.allclose(x, [10.0, 250.0], atol=1e-6)
        assert np.allclose(y, [20.0, 300.0], atol=1e-6)

    def test_pixel_scale(self):
        assert SimpleWCS.tangent(0, 0, (100, 100), 0.25).pixel_scale == pytest.approx(0.25)

    def test_header_round_trip(self):
        wcs = SimpleWCS.tangent(30.0, -10.0, (200, 200), 0.6)
        restored = SimpleWCS.from_header(wcs.to_header())
        assert np.allclose(restored.cd, wcs.cd)
        assert restored.crval == pytest.approx(wcs.crval)

    def test_separation_matches_pixel_scale(self):
        wcs = SimpleWCS.tangent(0.0, 0.0, (100, 100), 1.0)
        assert float(wcs.separation_arcsec(0, 0, 10, 0)) == pytest.approx(10.0, rel=1e-3)

    def test_angular_separation_is_symmetric(self):
        a = angular_separation(10.0, 20.0, 11.0, 21.0)
        b = angular_separation(11.0, 21.0, 10.0, 20.0)
        assert float(a) == pytest.approx(float(b))


class TestAstroImage:
    def test_from_array_shape(self, array):
        assert AstroImage.from_array(array).shape == (48, 36)

    def test_stats_are_robust(self, array):
        stats = AstroImage.from_array(array).stats()
        assert stats["median"] == pytest.approx(100.0, abs=1.0)
        assert stats["n_valid"] == array.size

    def test_cutout_is_square_and_padded(self, array):
        image = AstroImage.from_array(array)
        assert image.cutout(2.0, 2.0, 16).shape == (16, 16)
        assert image.cutout(1000.0, 1000.0, 16).sum() == 0.0

    def test_write_and_reload_preserves_metadata(self, tmp_path, array):
        image = AstroImage.from_array(
            array, name="field", band="r", mjd=59000.0,
            wcs=SimpleWCS.tangent(10.0, -5.0, (48, 36), 0.5))
        path = str(tmp_path / "img.fits")
        image.write(path)
        reloaded = AstroImage.from_fits(path)
        assert reloaded.band == "r"
        assert reloaded.mjd == pytest.approx(59000.0)
        assert reloaded.pixel_scale == pytest.approx(0.5, rel=1e-3)

    def test_mask_shape_is_validated(self, array):
        with pytest.raises(DataError):
            AstroImage(data=array, mask=np.zeros((3, 3), dtype=bool))

    def test_npy_round_trip(self, tmp_path, array):
        path = str(tmp_path / "a.npy")
        np.save(path, array)
        assert AstroImage.load(path).shape == array.shape


class TestImageSeries:
    def _series(self, array):
        return ImageSeries([
            AstroImage.from_array(array + i, name=f"e{i}", mjd=59000.0 + i)
            for i in range(4)])

    def test_sorted_by_time(self, array):
        series = ImageSeries([
            AstroImage.from_array(array, name="late", mjd=5.0),
            AstroImage.from_array(array, name="early", mjd=1.0)])
        assert [im.name for im in series] == ["early", "late"]

    def test_times_and_shape(self, array):
        series = self._series(array)
        assert len(series) == 4
        assert series.shape == array.shape
        assert list(series.times) == [59000.0, 59001.0, 59002.0, 59003.0]

    def test_stack_reduces_to_one_image(self, array):
        stacked = self._series(array).stack("median")
        assert stacked.shape == array.shape
        assert np.median(stacked.data) == pytest.approx(np.median(array) + 1.5, abs=0.5)

    def test_alignment_check_reports_shape_mismatch(self, array):
        series = ImageSeries([
            AstroImage.from_array(array, mjd=1.0),
            AstroImage.from_array(array[:20, :20], mjd=2.0)])
        assert series.check_alignment()

    def test_empty_series_raises(self):
        with pytest.raises(DataError):
            ImageSeries([])


class TestCatalogIO:
    def _catalog(self):
        from astrovision.core.types import (
            BoundingBox, Morphology, MorphologyMetrics, ObjectClass, Photometry,
            Source, SourceCatalog)
        return SourceCatalog([
            Source(i, i * 10.0, i * 5.0,
                   BoundingBox(i * 10 - 3, i * 5 - 3, i * 10 + 3, i * 5 + 3),
                   ra=10.0 + i * 0.001, dec=-5.0 + i * 0.001,
                   object_class=ObjectClass.GALAXY if i % 2 else ObjectClass.STAR,
                   class_confidence=0.9,
                   photometry=Photometry(flux=100.0 * i, magnitude=20.0 - i * 0.1,
                                         snr=25.0),
                   morphology=MorphologyMetrics(semi_major=4.0, ellipticity=0.3,
                                                label=Morphology.SPIRAL),
                   flags=["edge"] if i == 2 else [])
            for i in range(1, 6)])

    @pytest.mark.parametrize("extension", ["csv", "json"])
    def test_round_trip(self, tmp_path, extension):
        catalog = self._catalog()
        path = str(tmp_path / f"cat.{extension}")
        catalog_io.write_catalog(catalog, path)
        restored = catalog_io.read_catalog(path)
        assert len(restored) == len(catalog)
        assert restored[0].object_class == catalog[0].object_class
        assert restored[0].photometry.flux == pytest.approx(catalog[0].photometry.flux)

    def test_flags_survive_csv(self, tmp_path):
        path = str(tmp_path / "cat.csv")
        catalog_io.write_csv(self._catalog(), path)
        restored = catalog_io.read_csv(path)
        assert restored[1].flags == ["edge"]

    def test_fits_table(self, tmp_path):
        pytest.importorskip("astropy")
        path = str(tmp_path / "cat.fits")
        catalog_io.write_fits_table(self._catalog(), path)
        from astropy.io import fits
        with fits.open(path) as hdul:
            assert hdul[1].header["NSOURCE"] == 5

    def test_crossmatch_pixel(self):
        catalog = self._catalog()
        matches = catalog_io.crossmatch(catalog, catalog, radius=0.5)
        assert len(matches) == len(catalog)
        assert all(m["separation"] == pytest.approx(0.0) for m in matches)

    def test_crossmatch_world(self):
        catalog = self._catalog()
        matches = catalog_io.crossmatch(catalog, catalog, radius=1.0, use_world=True)
        assert len(matches) == len(catalog)

    def test_unsupported_format_raises(self, tmp_path):
        with pytest.raises(DataError):
            catalog_io.write_catalog(self._catalog(), str(tmp_path / "c.xyz"))
