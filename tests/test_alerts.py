"""Alerts: the codec against fastavro, the packet in three vocabularies,
and a TNS draft that a person still has to send."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from astrovision.alerts import (AlertPacket, Detection, draft_tns_report,
                                packets_from_analysis, read_alerts, write_alerts, write_tns_draft)
from astrovision.alerts.avro import (decode, encode, read_avro, read_container, schema_of,
                                     write_avro, write_container)
from astrovision.core.backend import try_import

fastavro = try_import("fastavro")

EVERYTHING = {
    "type": "record", "name": "everything", "fields": [
        {"name": "b", "type": "boolean"}, {"name": "i", "type": "int"},
        {"name": "l", "type": "long"}, {"name": "f", "type": "float"},
        {"name": "d", "type": "double"}, {"name": "s", "type": "string"},
        {"name": "y", "type": "bytes"}, {"name": "n", "type": ["null", "string"]},
        {"name": "a", "type": {"type": "array", "items": "long"}},
        {"name": "m", "type": {"type": "map", "values": "double"}},
        {"name": "e", "type": {"type": "enum", "name": "colour", "symbols": ["red", "blue"]}},
        {"name": "x", "type": {"type": "fixed", "name": "four", "size": 4}},
        {"name": "r", "type": {"type": "record", "name": "inner",
                               "fields": [{"name": "v", "type": "int"}]}},
        {"name": "again", "type": ["null", "inner"], "default": None},
    ],
}
SAMPLE = {"b": True, "i": -12345, "l": 1 << 40, "f": 1.5, "d": -2.25e-7, "s": "héllo",
          "y": b"\x00\xff", "n": None, "a": [1, -2, 3], "m": {"k": 1.0, "z": -0.5},
          "e": "blue", "x": b"abcd", "r": {"v": 7}, "again": {"v": 9}}


class TestCodec:
    def test_every_type_round_trips(self):
        registry = {}
        from astrovision.alerts.avro import _named_types
        _named_types(EVERYTHING, registry)
        out = io.BytesIO()
        encode(SAMPLE, EVERYTHING, registry, out)
        back, pos = decode(out.getvalue(), 0, EVERYTHING, registry)
        assert pos == len(out.getvalue())
        assert back == SAMPLE

    def test_zigzag_longs_at_the_edges(self):
        from astrovision.alerts.avro import _read_long, _write_long
        for value in (0, -1, 1, 63, 64, -64, -65, 2 ** 31 - 1, -2 ** 31, 2 ** 62, -2 ** 63):
            out = io.BytesIO()
            _write_long(out, value)
            assert _read_long(out.getvalue(), 0)[0] == value

    @pytest.mark.parametrize("codec", ["null", "deflate"])
    def test_a_container_round_trips_in_blocks(self, codec):
        records = [dict(SAMPLE, i=k) for k in range(150)]
        buf = io.BytesIO()
        assert write_container(buf, EVERYTHING, records, codec=codec, block_size=64) == 150
        buf.seek(0)
        schema, back = read_container(buf)
        assert schema["name"] == "everything"
        assert list(back) == records

    def test_a_corrupt_sync_marker_is_an_error(self):
        buf = io.BytesIO()
        write_container(buf, EVERYTHING, [SAMPLE], codec="null")
        data = bytearray(buf.getvalue())
        data[-1] ^= 0xFF
        with pytest.raises(ValueError):
            list(read_container(io.BytesIO(bytes(data)))[1])

    def test_a_missing_required_field_is_refused(self):
        registry = {}
        with pytest.raises(ValueError):
            encode({"b": True}, EVERYTHING, registry, io.BytesIO())

    @pytest.mark.skipif(fastavro is None, reason="fastavro not installed")
    def test_fastavro_reads_what_we_write_and_we_read_what_it_writes(self):
        records = [dict(SAMPLE, i=k) for k in range(5)]
        ours = io.BytesIO()
        write_container(ours, EVERYTHING, records, codec="deflate")
        ours.seek(0)
        assert [dict(r) for r in fastavro.reader(ours)] == records
        theirs = io.BytesIO()
        fastavro.writer(theirs, fastavro.parse_schema(EVERYTHING), records, codec="deflate")
        theirs.seek(0)
        assert list(read_container(theirs)[1]) == records

    def test_the_file_api_works_with_and_without_fastavro(self, tmp_path):
        path = str(tmp_path / "x.avro")
        assert write_avro(path, EVERYTHING, [SAMPLE], prefer_fastavro=False) == 1
        assert schema_of(path)["name"] == "everything"
        for prefer in (False, True):
            schema, records = read_avro(path, prefer_fastavro=prefer)
            assert records == [SAMPLE]


@pytest.fixture()
def packet():
    rng = np.random.default_rng(0)
    return AlertPacket(
        object_id="AVX000042", candid=6000050000042, ra=150.123456, dec=2.234567,
        mjd=60000.5, band="r", mag=18.5, mag_err=0.05, flux=1200.0, flux_err=50.0,
        limiting_mag=20.5, real_bogus=0.93, is_positive=True, classification="supernova",
        verdict="follow_up_recommended", human_verdict="real by ana",
        history=[Detection(mjd=59996.5, band="r", limiting_mag=20.4),
                 Detection(mjd=59998.5, band="g", limiting_mag=20.9),
                 Detection(mjd=59999.5, band="g", mag=19.0, mag_err=0.1, flux=800.0,
                           flux_err=60.0)],
        cutout_science=rng.normal(size=(63, 63)), cutout_template=rng.normal(size=(63, 63)),
        provenance={"reproducibility_key": "sha256:abc"})


class TestPacket:
    def test_round_trip_through_the_ztf_vocabulary(self, packet, tmp_path):
        path = str(tmp_path / "alerts.avro")
        assert write_alerts(path, [packet]) == 1
        schema, packets = read_alerts(path)
        assert str(schema["name"]).endswith("alert")     # fastavro qualifies names
        back = packets[0]
        assert back.object_id == packet.object_id and back.candid == packet.candid
        assert back.ra == pytest.approx(packet.ra) and back.mjd == pytest.approx(packet.mjd)
        assert back.mag == pytest.approx(18.5, abs=1e-6)           # stored as a float32
        assert back.real_bogus == pytest.approx(0.93, abs=1e-6)
        assert back.band == "r" and back.is_positive is True
        assert back.classification == "supernova" and back.human_verdict == "real by ana"
        assert back.provenance == {"reproducibility_key": "sha256:abc"}
        assert len(back.history) == 3 and back.history[0].limiting_mag == pytest.approx(20.4, abs=1e-6)
        np.testing.assert_allclose(back.cutout_science, packet.cutout_science.astype(np.float32))
        assert back.cutout_difference is None

    def test_a_real_ztf_shaped_record_is_understood(self):
        record = {"schemavsn": "4.02", "publisher": "ZTF", "objectId": "ZTF18abcdefg",
                  "candid": 1234567890123456789,
                  "candidate": {"jd": 2459000.5, "fid": 2, "pid": 1, "candid": 1234567890123456789,
                                "isdiffpos": "t", "ra": 10.0, "dec": -5.0, "magpsf": 17.2,
                                "sigmapsf": 0.03, "diffmaglim": 20.1, "rb": 0.8, "drb": 0.99,
                                "distnr": 0.4, "magnr": 18.0, "classtar": 0.1, "fwhm": 2.1},
                  "prv_candidates": [{"jd": 2458998.5, "fid": 1, "pid": 0, "candid": None,
                                      "isdiffpos": None, "ra": None, "dec": None,
                                      "magpsf": None, "sigmapsf": None, "diffmaglim": 20.3},
                                     None],
                  "cutoutScience": None, "cutoutTemplate": None, "cutoutDifference": None}
        p = AlertPacket.from_record(record)
        assert p.source_format == "ztf" and p.band == "r" and p.mjd == pytest.approx(59000.0)
        assert p.deep_real_bogus == 0.99 and p.host_distance_arcsec == 0.4
        assert len(p.history) == 1 and p.history[0].band == "g"
        assert not p.history[0].is_detection
        assert p.last_non_detection_before().limiting_mag == 20.3

    def test_a_rubin_shaped_record_is_understood(self):
        record = {"alertId": 99, "diaSource": {"diaSourceId": 501, "diaObjectId": 7,
                                               "midpointMjdTai": 61000.25, "ra": 20.0, "dec": 30.0,
                                               "psfFlux": 3631.0, "psfFluxErr": 100.0,
                                               "band": "i", "snr": 36.3, "reliability": 0.7},
                  "prvDiaSources": [{"diaSourceId": 500, "midpointMjdTai": 60998.25, "ra": 20.0,
                                     "dec": 30.0, "psfFlux": 1000.0, "psfFluxErr": 100.0,
                                     "band": "i"}],
                  "prvDiaForcedSources": [{"midpointMjdTai": 60990.0, "psfFlux": 10.0,
                                           "psfFluxErr": 100.0, "band": "i"}]}
        p = AlertPacket.from_record(record)
        assert p.source_format == "rubin" and p.object_id == "7" and p.candid == 501
        assert p.mag == pytest.approx(22.5, abs=1e-3)              # 3631 nJy is AB 22.5
        assert p.real_bogus == 0.7 and p.band == "i"
        assert len(p.history) == 2 and p.history[0].is_detection
        assert not p.history[1].is_detection                       # 0.1 sigma: a limit

    def test_packets_come_from_the_pipelines_transients(self):
        from astrovision.engine.pipeline import Pipeline
        from astrovision.ml.active import HumanVerdict, VerdictLog
        from astrovision.simulate import SkyConfig, SkySimulator

        simulator = SkySimulator(SkyConfig(shape=(160, 160), n_stars=25, n_galaxies=6,
                                           n_nebulae=0, n_clusters=0, n_lenses=0,
                                           n_anomalies=0, seed=21))
        series, _, injected = simulator.generate_series(n_epochs=4, cadence=2.0, n_transients=2)
        analysis = Pipeline().run_series(series)
        log = VerdictLog()
        packets = packets_from_analysis(analysis, series=series)
        assert len(packets) == len([c for c in analysis.transients if "bogus" not in c.flags])
        if packets:
            first = packets[0]
            assert first.object_id.startswith("AVX") and np.isfinite(first.ra)
            assert first.cutout_science is not None and first.cutout_template is not None
            assert first.provenance.get("reproducibility_key", "").startswith("sha256:")
            candidate = [c for c in analysis.transients if "bogus" not in c.flags][0]
            log.add(HumanVerdict(source_id=-int(candidate.id), label="real", reviewer="ana",
                                 kind="transient", candidate_id=candidate.id))
            vetted = packets_from_analysis(analysis, series=series, verdict_log=log)
            assert vetted[0].human_verdict == "real by ana"


class TestTns:
    def test_a_draft_has_the_tns_layout_and_is_marked_unsent(self, packet, tmp_path):
        report = draft_tns_report(packet, reporter="A. Astronomer", reporting_group_id=5,
                                  data_source_id=5, instrument_id=9, at_type="supernova")
        entry = report["at_report"]["0"]
        assert entry["ra"]["value"] == pytest.approx(150.123456)
        assert entry["reporter"] == "A. Astronomer" and entry["at_type"] == 3
        assert entry["discovery_datetime"] == "2023-02-24 12:00:00"   # MJD 59999.5
        assert len(entry["photometry"]["photometry_group"]) == 2       # history + discovery
        assert entry["non_detection"]["limiting_flux"] == pytest.approx(20.9)
        assert "vetted: real by ana" in entry["remarks"]
        assert report["_draft"] is True and "not been sent" in report["_not_submitted"]
        assert report["_todo"] == []
        path = write_tns_draft(report, str(tmp_path / "tns" / "draft.json"))
        assert json.load(open(path))["_draft"] is True

    def test_missing_pieces_are_listed_not_hidden(self, packet):
        packet.human_verdict = None
        packet.history = []
        report = draft_tns_report(packet, reporter="A. Astronomer")
        todo = " ".join(report["_todo"])
        assert "reporting_group_id" in todo and "non-detection" in todo and "vet it" in todo
        assert report["at_report"]["0"]["non_detection"]["archiveid"] == "0"

    def test_an_anonymous_draft_is_refused(self, packet):
        with pytest.raises(ValueError):
            draft_tns_report(packet, reporter="")

    def test_nothing_in_the_module_can_send(self):
        import inspect

        from astrovision.alerts import tns

        imported = {name.split(".")[0] for line in inspect.getsource(tns).splitlines()
                    for name in ([line.split()[1]] if line.startswith(("import ", "from "))
                                 else [])}
        assert not imported & {"urllib", "http", "requests", "socket", "httpx"}
