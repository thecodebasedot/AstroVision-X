"""The catalog database: one store across fields and epochs."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.catalog import CatalogDB, ingest_analysis
from astrovision.core.types import BoundingBox, Source, SourceCatalog


def _catalog(ra, dec, flux=None, mjd_hint=None) -> SourceCatalog:
    catalog = SourceCatalog()
    flux = np.full(len(ra), 1000.0) if flux is None else np.asarray(flux, dtype=float)
    for k, (r, d) in enumerate(zip(ra, dec)):
        source = Source(id=k + 1, x=float(k), y=float(k), bbox=BoundingBox(0, 0, 1, 1),
                        ra=float(r), dec=float(d))
        source.photometry.flux = float(flux[k])
        source.photometry.flux_err = 10.0
        source.photometry.magnitude = 25.0 - 2.5 * np.log10(max(float(flux[k]), 1e-3))
        source.photometry.snr = float(flux[k]) / 10.0
        catalog.append(source)
    return catalog


@pytest.fixture()
def sky():
    rng = np.random.default_rng(7)
    ra = 150.0 + rng.uniform(-0.2, 0.2, 60)
    dec = 2.0 + rng.uniform(-0.2, 0.2, 60)
    return ra, dec


class TestIngest:
    def test_a_field_becomes_detections_and_objects(self, sky):
        ra, dec = sky
        db = CatalogDB()
        report = db.ingest(_catalog(ra, dec), name="epoch-1", band="r", mjd=60000.0)
        assert report.n_detections == 60 and report.n_with_sky == 60
        assert report.n_new_objects == 60 and report.n_matched == 0
        assert db.counts() == {"fields": 1, "detections": 60, "objects": 60}

    def test_a_second_epoch_links_to_the_same_objects(self, sky):
        """The property the object table exists for: the same sky position
        in a later image is the same object, not a new one."""
        ra, dec = sky
        db = CatalogDB()
        db.ingest(_catalog(ra, dec), name="epoch-1", band="r", mjd=60000.0)
        jitter = np.random.default_rng(1).normal(0.0, 0.3 / 3600.0, (2, 60))
        second = db.ingest(_catalog(ra + jitter[0], dec + jitter[1], flux=np.full(60, 900.0)),
                           name="epoch-2", band="g", mjd=60003.0)
        assert second.n_matched == 60 and second.n_new_objects == 0
        assert db.counts()["objects"] == 60
        obj = db.object(1)
        assert obj["n_detections"] == 2
        assert obj["first_mjd"] == 60000.0 and obj["last_mjd"] == 60003.0
        assert set(obj["bands"].split(",")) == {"r", "g"}

    def test_a_source_beyond_the_match_radius_founds_a_new_object(self, sky):
        ra, dec = sky
        db = CatalogDB(match_radius_arcsec=1.5)
        db.ingest(_catalog(ra, dec), name="epoch-1", band="r", mjd=60000.0)
        moved = db.ingest(_catalog(ra[:1] + 5.0 / 3600.0, dec[:1]), name="epoch-2",
                          band="r", mjd=60001.0)
        assert moved.n_new_objects == 1 and moved.n_matched == 0
        assert db.counts()["objects"] == 61

    def test_the_nearest_object_wins_when_two_are_close(self):
        db = CatalogDB(match_radius_arcsec=3.0)
        db.ingest(_catalog([10.0, 10.0 + 2.0 / 3600.0], [0.0, 0.0]), name="a", mjd=1.0)
        report = db.ingest(_catalog([10.0 + 1.6 / 3600.0], [0.0]), name="b", mjd=2.0)
        assert report.n_matched == 1
        row = db.history(2)
        assert len(row) == 2                       # object 2 is the nearer one

    def test_a_catalog_without_sky_coordinates_is_stored_but_not_linked(self):
        catalog = SourceCatalog([Source(id=1, x=5.0, y=5.0, bbox=BoundingBox(0, 0, 1, 1))])
        db = CatalogDB()
        report = db.ingest(catalog, name="no-wcs")
        assert report.n_with_sky == 0 and report.n_new_objects == 0
        assert any("no sky coordinates" in note for note in report.notes)
        assert db.counts()["detections"] == 1 and db.counts()["objects"] == 0

    def test_nan_values_become_null_not_garbage(self, sky):
        ra, dec = sky
        db = CatalogDB()
        db.ingest(_catalog(ra[:2], dec[:2]), name="f")
        row = db.detections_of_field(1)[0]
        assert row["kron_radius"] is None            # NaN in the source
        assert row["flux"] == 1000.0

    def test_the_manifest_travels_with_the_field(self):
        from astrovision.engine.pipeline import Pipeline
        from astrovision.simulate import quick_field

        image, _ = quick_field((128, 128), seed=3)
        analysis = Pipeline().run(image)
        db = CatalogDB()
        report = ingest_analysis(db, analysis, image)
        assert report.n_detections == len(analysis.catalog)
        assert report.n_with_sky == len(analysis.catalog)     # the simulator writes a WCS
        stored = db.fields()[0]
        assert stored["reproducibility_key"] == analysis.provenance["reproducibility_key"]
        assert stored["band"] == image.band


class TestQueries:
    @pytest.fixture()
    def populated(self, sky):
        ra, dec = sky
        db = CatalogDB()
        rng = np.random.default_rng(2)
        for epoch in range(4):
            jitter = rng.normal(0.0, 0.2 / 3600.0, (2, 60))
            flux = 1000.0 * (1.0 + 0.1 * epoch) * np.ones(60)
            db.ingest(_catalog(ra + jitter[0], dec + jitter[1], flux=flux),
                      name=f"epoch-{epoch}", band="r" if epoch % 2 == 0 else "g",
                      mjd=60000.0 + 2.0 * epoch)
        return db, ra, dec

    def test_cone_search_finds_every_detection_and_nothing_else(self, populated):
        db, ra, dec = populated
        rows = db.cone_search(ra[0], dec[0], radius_arcsec=2.0)
        assert len(rows) == 4                                   # four epochs
        assert all(r["separation_arcsec"] <= 2.0 for r in rows)
        assert rows == sorted(rows, key=lambda r: r["separation_arcsec"])
        assert "field_name" in rows[0]
        objects = db.cone_search(ra[0], dec[0], radius_arcsec=2.0, table="objects")
        assert len(objects) == 1 and objects[0]["n_detections"] == 4

    def test_cone_search_agrees_with_brute_force(self, populated):
        """The index must not lose anything: a brute-force separation over
        every row is the reference the indexed query is checked against."""
        from astrovision.catalog import angular_separation

        db, ra, dec = populated
        all_rows = db.connection.execute("SELECT ra, dec FROM detections").fetchall()
        all_ra = np.array([r[0] for r in all_rows]); all_dec = np.array([r[1] for r in all_rows])
        for radius in (30.0, 300.0, 1200.0):
            expected = int((angular_separation(150.05, 2.05, all_ra, all_dec) * 3600 <= radius).sum())
            assert len(db.cone_search(150.05, 2.05, radius)) == expected

    def test_an_empty_region_is_empty(self, populated):
        db, *_ = populated
        assert db.cone_search(30.0, -40.0, 60.0) == []

    def test_history_is_a_light_curve(self, populated):
        db, *_ = populated
        history = db.history(5)
        assert [h["mjd"] for h in history] == [60000.0, 60002.0, 60004.0, 60006.0]
        mjd, flux, err = db.light_curve(5)
        np.testing.assert_allclose(flux, [1000.0, 1100.0, 1200.0, 1300.0])
        assert err.shape == (4,)
        mjd_r, flux_r, _ = db.light_curve(5, band="r")
        assert list(mjd_r) == [60000.0, 60004.0]

    def test_objects_with_history_are_ranked_by_detections(self, populated):
        db, ra, dec = populated
        db.ingest(_catalog([ra[0]], [dec[0]]), name="extra", band="i", mjd=60010.0)
        top = db.objects_with_history(min_detections=2, limit=3)
        assert top[0]["n_detections"] == 5
        assert all(o["n_detections"] >= 2 for o in top)

    def test_a_field_catalog_comes_back_as_sources(self, populated):
        db, ra, dec = populated
        catalog = db.field_catalog(1)
        assert len(catalog) == 60
        assert catalog[0].ra == pytest.approx(ra[0], abs=1e-3)
        assert catalog[0].photometry.flux == 1000.0
        assert catalog[0].meta["object_id"] == 1

    def test_the_store_persists_on_disk(self, tmp_path, sky):
        ra, dec = sky
        path = str(tmp_path / "catalog.sqlite")
        with CatalogDB(path) as db:
            db.ingest(_catalog(ra, dec), name="f", mjd=1.0)
        with CatalogDB(path) as again:
            assert again.counts()["detections"] == 60
            assert len(again.cone_search(ra[3], dec[3], 1.0)) == 1
