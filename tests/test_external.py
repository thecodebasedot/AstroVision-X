"""External catalog crossmatch, and the calibrations built on it.

The HTTP backends are exercised through their URL builders and their
parsers, not against the live services: a test suite that needs the network
to pass is a test suite that fails for reasons unrelated to the code.  The
matching logic itself is tested end to end against a local catalog, which
runs the same code path a remote one would.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.calibration import solve_plate, solve_zero_point
from astrovision.calibration.astrometry import apply_solution, match_to_reference
from astrovision.calibration.photometry import apply_zero_point
from astrovision.core.exceptions import DataError
from astrovision.core.types import BoundingBox, Photometry, Source, SourceCatalog
from astrovision.io.external import (
    CachedCone,
    LocalCone,
    NullCone,
    ReferenceObject,
    SimbadCone,
    VizieRCone,
    build_service,
    crossmatch_catalog,
    field_cone,
    parse_simbad_tsv,
    parse_vizier_tsv,
    read_reference_file,
    write_reference_file,
)
from astrovision.io.wcs import SimpleWCS, angular_separation


class TestReferenceObject:
    def test_round_trips_through_a_dict(self):
        obj = ReferenceObject(150.0, 2.2, "HD 1", "GAIA", "V*", {"G": 15.2})
        assert ReferenceObject.from_dict(obj.to_dict()).magnitudes["G"] == 15.2

    def test_describes_its_type_in_words(self):
        assert ReferenceObject(0, 0, object_type="V*").described_type == "variable star"
        assert ReferenceObject(0, 0, object_type="Zz9").described_type == "Zz9"


class TestLocalCone:
    @pytest.fixture()
    def service(self):
        return LocalCone([
            ReferenceObject(150.0000, 2.2000, "A", "T", "*"),
            ReferenceObject(150.0010, 2.2000, "B", "T", "*"),
            ReferenceObject(151.0000, 2.2000, "C", "T", "*"),
        ])

    def test_returns_only_what_is_inside_the_cone(self, service):
        names = {o.name for o in service.query(150.0, 2.2, 10.0)}
        assert names == {"A", "B"}

    def test_an_empty_catalog_returns_nothing(self):
        assert LocalCone([]).query(150.0, 2.2, 100.0) == []

    def test_reads_json_and_csv(self, service, tmp_path):
        path = write_reference_file(service.objects, str(tmp_path / "ref.json"))
        assert len(read_reference_file(path)) == 3

        csv_path = tmp_path / "ref.csv"
        csv_path.write_text("ra,dec,name,rmag\n150.0,2.2,A,15.5\n", encoding="utf-8")
        objects = read_reference_file(str(csv_path))
        assert objects[0].magnitudes["r"] == pytest.approx(15.5)

    def test_a_missing_file_is_an_error(self):
        with pytest.raises(DataError):
            read_reference_file("/nonexistent/reference.json")


class TestCachedCone:
    class _Counting(NullCone):
        name = "counting"

        def __init__(self):
            self.calls = 0

        def query(self, ra, dec, radius_arcsec):
            self.calls += 1
            return [ReferenceObject(ra, dec, "X", "T", "*")]

    def test_second_identical_query_is_served_from_disk(self, tmp_path):
        inner = self._Counting()
        service = CachedCone(inner, str(tmp_path))
        first = service.query(150.0, 2.2, 60.0)
        second = service.query(150.0, 2.2, 60.0)
        assert inner.calls == 1
        assert service.hits == 1
        assert [o.name for o in first] == [o.name for o in second]

    def test_an_expired_entry_is_refetched(self, tmp_path):
        inner = self._Counting()
        service = CachedCone(inner, str(tmp_path), max_age_days=0.0)
        service.query(150.0, 2.2, 60.0)
        service.query(150.0, 2.2, 60.0)
        assert inner.calls == 2

    def test_a_corrupt_entry_is_discarded_not_raised(self, tmp_path):
        inner = self._Counting()
        service = CachedCone(inner, str(tmp_path))
        service.query(150.0, 2.2, 60.0)
        path = service._path(150.0, 2.2, 60.0)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        assert service.query(150.0, 2.2, 60.0)
        assert inner.calls == 2


class TestHttpBackends:
    def test_vizier_url_carries_the_cone_in_arcminutes(self):
        url = VizieRCone().build_url(150.0, 2.2, 120.0)
        assert "-c=150.000000%20%2B2.200000" in url
        assert "-c.rs=2.00" in url
        assert "I/355/gaiadr3" in url

    def test_simbad_url_is_an_adql_cone(self):
        url = SimbadCone().build_url(150.0, -2.2, 3600.0)
        assert "CIRCLE" in url and "1.000000" in url

    def test_vizier_parser_skips_the_preamble_rows(self):
        text = (
            "#comment line\n"
            "\n"
            "RA_ICRS\tDE_ICRS\tSource\tGmag\n"
            "deg\tdeg\t\tmag\n"
            "------\t------\t------\t------\n"
            "150.0\t2.2\t12345\t15.4\n"
            "150.1\t2.3\t12346\t\n"
        )
        objects = parse_vizier_tsv(text, "I/355/gaiadr3")
        assert len(objects) == 2
        assert objects[0].name == "12345"
        assert objects[0].magnitudes["G"] == pytest.approx(15.4)
        assert "G" not in objects[1].magnitudes      # blank cell, not zero

    def test_simbad_parser_strips_quotes(self):
        text = 'main_id\tra\tdec\totype\n"V* AB Aur"\t150.0\t2.2\t"V*"\n'
        obj = parse_simbad_tsv(text)[0]
        assert obj.name == "V* AB Aur"
        assert obj.described_type == "variable star"

    def test_a_failed_fetch_returns_nothing_and_records_why(self):
        service = VizieRCone(timeout=0.001)
        service.url_template = "http://127.0.0.1:9/{catalog}{ra}{dec}{radius}{columns}{max_rows}"
        assert service.query(150.0, 2.2, 60.0) == []
        assert service.last_error


class TestBuildService:
    def test_default_is_the_null_backend(self):
        assert isinstance(build_service("none"), NullCone)

    def test_local_needs_a_path(self):
        with pytest.raises(DataError):
            build_service("local")

    def test_unknown_backend_is_an_error(self):
        with pytest.raises(DataError):
            build_service("nonsense")

    def test_cache_dir_wraps_the_backend(self, tmp_path):
        service = build_service("vizier", cache_dir=str(tmp_path))
        assert isinstance(service, CachedCone)
        assert isinstance(service.inner, VizieRCone)


class TestCrossmatch:
    def _catalog(self, wcs, positions):
        sources = []
        for index, (x, y) in enumerate(positions, start=1):
            source = Source(id=index, x=float(x), y=float(y),
                            bbox=BoundingBox(int(x) - 3, int(y) - 3,
                                             int(x) + 3, int(y) + 3))
            ra, dec = wcs.pixel_to_world(x, y)
            source.ra, source.dec = float(ra), float(dec)
            sources.append(source)
        return SourceCatalog(sources)

    @pytest.fixture()
    def field(self):
        wcs = SimpleWCS.tangent(150.0, 2.2, (200, 200), 0.4)
        catalog = self._catalog(wcs, [(50, 50), (100, 100), (150, 150)])
        return wcs, catalog

    def test_null_backend_is_recorded_as_not_performed(self, field):
        _, catalog = field
        report = crossmatch_catalog(catalog, NullCone())
        assert not report.performed
        assert not report.conclusive

    def test_matches_are_flagged_and_described(self, field):
        wcs, catalog = field
        ra, dec = wcs.pixel_to_world(100.0, 100.0)
        service = LocalCone([ReferenceObject(float(ra), float(dec), "V* Test",
                                             "SIMBAD", "V*")])
        report = crossmatch_catalog(catalog, service, radius_arcsec=2.0)
        assert report.performed and report.conclusive
        assert report.n_matched == 1
        matched = [s for s in catalog if "known" in s.flags]
        assert len(matched) == 1
        assert matched[0].meta["known_object"]["described_type"] == "variable star"

    def test_a_nearby_but_not_coincident_reference_does_not_match(self, field):
        """Inside the field, outside the match radius: checked, and not known."""
        wcs, catalog = field
        ra, dec = wcs.pixel_to_world(100.0, 118.0)      # 18 px = 7.2 arcsec away
        service = LocalCone([ReferenceObject(float(ra), float(dec), "near", "T", "*")])
        report = crossmatch_catalog(catalog, service, radius_arcsec=2.0)
        assert report.n_reference == 1
        assert report.n_matched == 0
        assert report.conclusive     # it did check; there was simply no match

    def test_an_empty_cone_is_not_a_conclusive_answer(self, field):
        """Zero references anywhere in the field means the check established
        nothing -- not that the field contains nothing known."""
        _, catalog = field
        service = LocalCone([ReferenceObject(150.05, 2.25, "far away", "T", "*")])
        report = crossmatch_catalog(catalog, service, radius_arcsec=2.0)
        assert report.performed
        assert report.n_reference == 0
        assert not report.conclusive

    def test_without_sky_coordinates_it_refuses(self):
        source = Source(id=1, x=5.0, y=5.0, bbox=BoundingBox(0, 0, 10, 10))
        report = crossmatch_catalog(SourceCatalog([source]),
                                    LocalCone([ReferenceObject(0, 0)]))
        assert not report.performed
        assert "no sky coordinates" in (report.error or "")

    def test_field_cone_covers_every_source(self, field):
        wcs, catalog = field
        centre_ra, centre_dec, radius = field_cone(catalog)
        for source in catalog:
            separation = angular_separation(centre_ra, centre_dec,
                                            source.ra, source.dec) * 3600.0
            assert separation <= radius

    def test_field_cone_handles_the_right_ascension_wrap(self):
        wcs = SimpleWCS.tangent(0.0, 0.0, (200, 200), 1.0)
        catalog = self._catalog(wcs, [(20, 100), (180, 100)])
        centre_ra, _, radius = field_cone(catalog)
        # Averaging RA directly would put the centre at 180 degrees.
        assert min(centre_ra, 360.0 - centre_ra) < 1.0
        assert radius < 200.0


class TestKnownObjectDemotion:
    def test_a_catalogued_anomaly_ranks_below_an_unknown_one(self):
        from astrovision.core.types import AnomalyRecord, FieldAnalysis
        from astrovision.engine.priority import rank_candidates

        sources = []
        for index, (x, y) in enumerate([(10.0, 10.0), (50.0, 50.0)], start=1):
            sources.append(Source(id=index, x=x, y=y,
                                  bbox=BoundingBox(0, 0, 20, 20)))
        sources[0].meta["known_object"] = {
            "name": "V* Known", "described_type": "variable star",
            "separation_arcsec": 0.4, "catalog": "SIMBAD"}
        analysis = FieldAnalysis(catalog=SourceCatalog(sources))
        analysis.anomalies = [
            AnomalyRecord(source_id=1, score=0.99, novelty_type="statistical outlier",
                          explanation="Unusual. Very."),
            AnomalyRecord(source_id=2, score=0.95, novelty_type="statistical outlier",
                          explanation="Unusual. Also."),
        ]
        ranked = rank_candidates(analysis, limit=5)
        by_id = {item.source_id: item for item in ranked}
        # The known object scored higher before the demotion and must not now.
        assert by_id[2].score > by_id[1].score
        assert any("V* Known" in reason for reason in by_id[1].reasons)
        assert any("already catalogued" in caveat for caveat in by_id[1].caveats)


class TestAstrometricCalibration:
    @pytest.fixture()
    def plate(self):
        truth = SimpleWCS.tangent(150.0, 2.2, (400, 400), 0.4)
        rng = np.random.default_rng(5)
        positions = rng.uniform(20, 380, size=(60, 2))
        reference = []
        for index, (x, y) in enumerate(positions, start=1):
            ra, dec = truth.pixel_to_world(x, y)
            reference.append(ReferenceObject(float(ra), float(dec), f"R{index}",
                                             "TRUTH", "*",
                                             magnitudes={"r": 16.0 + 0.02 * index}))
        sources = []
        for index, (x, y) in enumerate(positions, start=1):
            source = Source(id=index, x=float(x), y=float(y),
                            bbox=BoundingBox(int(x) - 4, int(y) - 4,
                                             int(x) + 4, int(y) + 4))
            source.photometry = Photometry(
                flux=10 ** (-0.4 * (16.0 + 0.02 * index - 25.0)),
                flux_err=1.0, snr=100.0)
            sources.append(source)
        return truth, SourceCatalog(sources), reference

    def _wrong_wcs(self, truth, shift=(5.0, -3.0), rotation=0.4, scale=1.003):
        angle = np.radians(rotation)
        turn = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
        return SimpleWCS(crpix=(truth.crpix[0] + shift[0], truth.crpix[1] + shift[1]),
                         crval=truth.crval, cd=truth.cd @ turn * scale)

    def test_recovers_a_wrong_pointing(self, plate):
        truth, catalog, reference = plate
        solution = solve_plate(catalog, reference, self._wrong_wcs(truth),
                               radius_arcsec=12.0)
        assert solution.succeeded
        assert solution.rms_arcsec < 0.05
        for x, y in ((20.0, 20.0), (380.0, 380.0), (200.0, 200.0)):
            ra1, dec1 = solution.wcs.pixel_to_world(x, y)
            ra2, dec2 = truth.pixel_to_world(x, y)
            assert angular_separation(ra1, dec1, ra2, dec2) * 3600.0 < 0.05

    def test_reports_what_it_changed(self, plate):
        truth, catalog, reference = plate
        solution = solve_plate(catalog, reference, self._wrong_wcs(truth),
                               radius_arcsec=12.0)
        assert solution.shift_arcsec > 1.0
        assert abs(solution.rotation_deg + 0.4) < 0.05
        assert solution.scale_ratio == pytest.approx(1.0 / 1.003, rel=0.01)

    def test_refuses_when_there_are_too_few_matches(self, plate):
        truth, catalog, reference = plate
        solution = solve_plate(catalog, reference[:3], truth, min_matches=8)
        assert not solution.succeeded
        assert solution.wcs is None
        assert "need 8" in solution.reason

    def test_matching_is_mutual(self, plate):
        truth, catalog, reference = plate
        # Two references on top of one detection: only one pair may survive.
        crowded = list(reference) + [reference[0]]
        pixels, _, _ = match_to_reference(catalog, crowded, truth, 2.0)
        assert len(pixels) == len({tuple(row) for row in pixels})

    def test_applying_a_solution_rewrites_sky_positions(self, plate):
        truth, catalog, reference = plate
        solution = solve_plate(catalog, reference, self._wrong_wcs(truth),
                               radius_arcsec=12.0)
        assert apply_solution(catalog, solution) == len(catalog)
        assert all(s.ra is not None for s in catalog)
        assert catalog.meta["astrometry"]["succeeded"]


class TestPhotometricCalibration:
    def _standards(self, zero_point=27.35, n=40, scatter=0.01, seed=1):
        rng = np.random.default_rng(seed)
        wcs = SimpleWCS.tangent(150.0, 2.2, (400, 400), 0.4)
        sources, reference = [], []
        for index in range(1, n + 1):
            x, y = rng.uniform(20, 380), rng.uniform(20, 380)
            magnitude = float(rng.uniform(14.0, 19.0))
            flux = 10 ** (-0.4 * (magnitude - zero_point))
            source = Source(id=index, x=x, y=y,
                            bbox=BoundingBox(int(x) - 4, int(y) - 4,
                                             int(x) + 4, int(y) + 4))
            ra, dec = wcs.pixel_to_world(x, y)
            source.ra, source.dec = float(ra), float(dec)
            source.photometry = Photometry(flux=flux * (1 + rng.normal(0, scatter)),
                                           flux_err=flux * 0.01, snr=100.0)
            sources.append(source)
            reference.append(ReferenceObject(float(ra), float(dec), f"S{index}",
                                             "TRUTH", "*", {"r": magnitude}))
        return SourceCatalog(sources), reference

    def test_recovers_the_zero_point(self):
        catalog, reference = self._standards(zero_point=27.35)
        solution = solve_zero_point(catalog, reference, band="r", reference_band="r")
        assert solution.succeeded
        assert solution.zero_point == pytest.approx(27.35, abs=0.02)
        assert solution.n_standards >= 30

    def test_rejects_a_blended_outlier(self):
        catalog, reference = self._standards()
        list(catalog)[0].photometry.flux *= 4.0        # a blend, 1.5 mag too bright
        solution = solve_zero_point(catalog, reference, band="r", reference_band="r")
        assert solution.zero_point == pytest.approx(27.35, abs=0.03)
        assert solution.n_rejected >= 1

    def test_refuses_with_too_few_standards(self):
        catalog, reference = self._standards(n=3)
        solution = solve_zero_point(catalog, reference, band="r", reference_band="r",
                                    min_standards=5)
        assert not solution.succeeded
        assert "need 5" in solution.reason

    def test_skips_saturated_sources(self):
        catalog, reference = self._standards()
        for source in list(catalog)[:10]:
            source.add_flag("saturated")
        solution = solve_zero_point(catalog, reference, band="r", reference_band="r")
        assert solution.n_standards <= len(catalog) - 10

    def test_applying_it_rewrites_magnitudes(self):
        catalog, reference = self._standards(zero_point=27.35)
        solution = solve_zero_point(catalog, reference, band="r", reference_band="r")
        assert apply_zero_point(catalog, solution, "r") == len(catalog)
        first = list(catalog)[0]
        assert first.photometry.zero_point == pytest.approx(27.35, abs=0.02)
        assert np.isfinite(first.photometry.magnitude)
