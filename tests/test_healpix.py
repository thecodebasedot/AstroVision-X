"""The HEALPix index must be HEALPix, not something HEALPix-shaped."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.catalog.healpix import (SkyIndex, ang2pix, angular_separation, npix, order,
                                         pix2ang, pixel_resolution_deg)
from astrovision.core.backend import try_import

healpy = try_import("healpy")


class TestPixelisation:
    @pytest.mark.parametrize("nside", [1, 2, 4, 8, 32])
    def test_every_pixel_round_trips_through_its_centre(self, nside):
        pixels = np.arange(npix(nside))
        ra, dec = pix2ang(nside, pixels)
        back = ang2pix(nside, ra, dec)
        np.testing.assert_array_equal(back, pixels)

    def test_nside_must_be_a_power_of_two(self):
        with pytest.raises(ValueError):
            order(3)
        assert order(1024) == 10

    def test_uniform_positions_fill_every_pixel_equally(self):
        """HEALPix pixels are equal-area, so uniform sky positions land in
        every pixel with the same frequency; a wrong face assignment or a
        polar/equatorial boundary error shows up as empty or crowded pixels."""
        rng = np.random.default_rng(1)
        n, nside = 400_000, 4
        ra = rng.uniform(0.0, 360.0, n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
        counts = np.bincount(ang2pix(nside, ra, dec), minlength=npix(nside))
        assert counts.min() > 0
        expected = n / npix(nside)
        assert np.abs(counts - expected).max() < 5.0 * np.sqrt(expected)

    def test_a_position_lies_within_its_pixels_radius_of_the_centre(self):
        rng = np.random.default_rng(2)
        nside = 64
        ra = rng.uniform(0.0, 360.0, 20_000)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 20_000)))
        pix = ang2pix(nside, ra, dec)
        ra_c, dec_c = pix2ang(nside, pix)
        sep = angular_separation(ra, dec, ra_c, dec_c)
        assert sep.max() <= 1.5 * pixel_resolution_deg(nside)

    def test_the_nested_scheme_is_hierarchical(self):
        """A pixel at nside 2n is a child of pixel ``p // 4`` at nside n."""
        rng = np.random.default_rng(3)
        ra = rng.uniform(0.0, 360.0, 5000)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 5000)))
        fine = ang2pix(64, ra, dec)
        coarse = ang2pix(32, ra, dec)
        np.testing.assert_array_equal(fine >> 2, coarse)

    def test_the_poles_and_the_seam_are_handled(self):
        for ra, dec in [(0.0, 90.0), (0.0, -90.0), (359.9999, 0.0), (360.0, 12.0),
                        (-10.0, -41.8), (720.5, 41.8)]:
            pix = ang2pix(16, ra, dec)
            assert 0 <= int(pix[0]) < npix(16)

    @pytest.mark.skipif(healpy is None, reason="healpy is not installed")
    def test_agrees_with_healpy_exactly(self):
        rng = np.random.default_rng(4)
        ra = rng.uniform(0.0, 360.0, 50_000)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 50_000)))
        for nside in (1, 8, 256):
            ours = ang2pix(nside, ra, dec)
            theirs = healpy.ang2pix(nside, ra, dec, nest=True, lonlat=True)
            np.testing.assert_array_equal(ours, theirs)
            ra_c, dec_c = pix2ang(nside, np.arange(npix(nside)))
            t_ra, t_dec = healpy.pix2ang(nside, np.arange(npix(nside)), nest=True, lonlat=True)
            np.testing.assert_allclose(dec_c, t_dec, atol=1e-9)
            np.testing.assert_allclose(np.mod(ra_c, 360.0), np.mod(t_ra, 360.0), atol=1e-9)


class TestSeparation:
    def test_known_separations(self):
        assert angular_separation(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)
        assert angular_separation(0.0, 0.0, 0.0, 90.0) == pytest.approx(90.0)
        assert angular_separation(10.0, 20.0, 190.0, -20.0) == pytest.approx(180.0)
        assert angular_separation(10.0, 20.0, 10.0, 20.0) == 0.0

    def test_small_angles_are_stable(self):
        one_arcsec = 1.0 / 3600.0
        assert angular_separation(100.0, 30.0, 100.0, 30.0 + one_arcsec) == pytest.approx(
            one_arcsec, rel=1e-9)


class TestCone:
    def test_a_cone_covers_every_point_inside_it(self):
        """The property a sky index must have: no point within the radius is
        in a pixel the cone query left out. Missing one is a silent hole in
        every search that follows."""
        rng = np.random.default_rng(5)
        index = SkyIndex(nside=64)
        for ra0, dec0, radius in [(10.0, 5.0, 0.5), (0.2, 89.0, 2.0), (180.0, -60.0, 0.1),
                                  (359.9, -3.0, 1.0)]:
            n = 5000
            ra = rng.uniform(ra0 - 3 * radius, ra0 + 3 * radius, n)
            dec = np.clip(rng.uniform(dec0 - 3 * radius, dec0 + 3 * radius, n), -90.0, 90.0)
            inside = angular_separation(ra0, dec0, ra, dec) <= radius
            pixels = set(index.cone(ra0, dec0, radius).tolist())
            assert all(int(p) in pixels for p in index.pixel(ra[inside], dec[inside]))

    def test_a_refined_cone_still_covers_every_point(self):
        """Refinement replaces a touched pixel by the children that can
        reach the cone; dropping a child that touches only by a corner would
        leave a hole no coarse-level test would see."""
        rng = np.random.default_rng(6)
        index = SkyIndex(nside=128)
        for ra0, dec0, radius in [(150.1, 2.2, 5.0 / 3600), (0.05, 88.5, 0.02),
                                  (200.0, -30.0, 0.15)]:
            n = 4000
            ra = ra0 + rng.uniform(-1.5 * radius, 1.5 * radius, n) / max(np.cos(np.radians(dec0)), 0.05)
            dec = np.clip(dec0 + rng.uniform(-1.5 * radius, 1.5 * radius, n), -90.0, 90.0)
            inside = angular_separation(ra0, dec0, ra, dec) <= radius
            fine = set(index.cone(ra0, dec0, radius, nside_out=8192).tolist())
            assert fine
            assert all(int(p) in fine for p in ang2pix(8192, ra[inside], dec[inside]))
            # and it is small: a 5" cone is a handful of 26" pixels, not thousands
            if radius < 0.01:
                assert len(fine) <= 16

    def test_a_cone_is_not_much_bigger_than_it_needs_to_be(self):
        index = SkyIndex(nside=128)
        area = np.pi * 1.0 ** 2                                   # a 1 degree cone
        n = len(index.cone(45.0, 10.0, 1.0))
        pixel_area = 4 * np.pi * (180 / np.pi) ** 2 / npix(128)   # square degrees
        assert n * pixel_area < 4.0 * area
