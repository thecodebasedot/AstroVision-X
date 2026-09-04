"""The vetting page: what it shows, what it records, what it refuses."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from astrovision.ml.active import HumanVerdict, VerdictLog
from astrovision.vetting import (VettingServer, VettingSession, build_queue, encode_png,
                                 stamp_png, stretch)
from astrovision.vetting.png import decode_png_size


class TestPng:
    def test_a_greyscale_png_is_well_formed(self):
        data = encode_png(np.arange(64, dtype=np.uint8).reshape(8, 8))
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert decode_png_size(data) == (8, 8)
        # The IDAT payload inflates to (1 filter byte + width) * height.
        idat = data.index(b"IDAT") + 4
        length = int.from_bytes(data[idat - 8:idat - 4], "big")
        assert len(zlib.decompress(data[idat:idat + length])) == 9 * 8

    def test_rgb_and_bad_shapes(self):
        data = encode_png(np.zeros((4, 6, 3), dtype=np.uint8))
        assert decode_png_size(data) == (6, 4)
        with pytest.raises(ValueError):
            encode_png(np.zeros((4, 6, 2)))

    def test_the_stretch_shows_faint_and_bright_alike(self):
        rng = np.random.default_rng(0)
        stamp = rng.normal(100.0, 5.0, (32, 32))
        stamp[16, 16] += 5000.0                       # a bright star
        stamp[8, 8] += 30.0                           # a 6-sigma faint source
        grey = stretch(stamp)
        assert grey.dtype == np.uint8
        assert grey[16, 16] == 255
        assert grey[8, 8] > np.median(grey) + 30
        assert 20 < np.median(grey) < 120             # the sky is dark grey, not black

    def test_stamp_png_is_upscaled(self):
        assert decode_png_size(stamp_png(np.zeros((16, 16)), factor=4)) == (64, 64)


@pytest.fixture(scope="module")
def analysed():
    from astrovision.engine.pipeline import Pipeline
    from astrovision.simulate import quick_field

    image, truth = quick_field((160, 160), seed=11, n_stars=30, n_galaxies=8,
                               n_lenses=1, n_anomalies=1)
    analysis = Pipeline().run(image)
    return image, analysis


class TestQueue:
    def test_the_queue_is_the_ranked_list_with_cutouts(self, analysed):
        image, analysis = analysed
        queue = build_queue(analysis, image, limit=10)
        assert 0 < len(queue) <= 10
        first = queue.items[0]
        assert first.rank == 1
        assert first.stamp is not None and first.stamp.shape == (64, 64)
        assert first.model_verdict
        assert first.ra is not None                   # the simulator writes a WCS
        assert [item.item_id for item in queue] == list(range(1, len(queue) + 1))
        payload = first.to_dict()
        assert set(payload) >= {"kind", "score", "model_label", "reasons", "caveats",
                                "evidence", "measurements", "verdict_key"}

    def test_every_catalog_source_can_be_appended(self, analysed):
        image, analysis = analysed
        queue = build_queue(analysis, image, limit=5, include_sources=True)
        assert len(queue) >= len(analysis.catalog)
        kinds = queue.summary()["kinds"]
        assert kinds.get("source", 0) > 0

    def test_history_comes_from_the_database(self, analysed):
        from astrovision.catalog import CatalogDB, ingest_analysis

        image, analysis = analysed
        db = CatalogDB()
        ingest_analysis(db, analysis, image)
        queue = build_queue(analysis, image, limit=5, include_sources=True, db=db)
        with_source = [i for i in queue if i.source_id is not None]
        assert with_source and all(i.history for i in with_source)
        assert with_source[0].history[0]["flux"] is not None


class TestSession:
    def test_next_skips_what_this_reviewer_decided(self, analysed):
        image, analysis = analysed
        session = VettingSession(build_queue(analysis, image, limit=5, include_sources=True))
        first = session.next_item(reviewer="ana")
        session.record(first.item_id, "real", "ana")
        assert session.next_item(reviewer="ana").item_id != first.item_id
        # Another reviewer still sees it: verdicts are per person.
        assert session.next_item(reviewer="ben").item_id == first.item_id
        assert session.next_item(after=first.item_id, direction="prev").item_id == first.item_id

    def test_a_verdict_without_a_reviewer_is_refused(self, analysed):
        image, analysis = analysed
        session = VettingSession(build_queue(analysis, image, limit=3))
        with pytest.raises(ValueError):
            session.record(1, "real", "   ")
        with pytest.raises(ValueError):
            session.record(1, "maybe", "ana")

    def test_verdicts_are_appended_never_overwritten(self, analysed, tmp_path):
        image, analysis = analysed
        path = str(tmp_path / "verdicts.json")
        session = VettingSession(build_queue(analysis, image, limit=3), log_path=path)
        session.record(1, "real", "ana", note="clear point source")
        session.record(1, "bogus", "ana")
        session.record(1, "real", "ben")
        saved = VerdictLog.load(path)
        assert len(saved) == 3
        assert saved.records[0].note == "clear point source"
        assert saved.records[0].model_label == session.queue.get(1).model_label
        assert saved.records[0].kind == session.queue.get(1).kind
        progress = session.progress()
        assert progress["n_done"] == 1 and progress["disagreements"] == 1

    def test_an_old_log_without_the_new_fields_still_loads(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps([{"source_id": 3, "label": "real", "reviewer": "ana",
                                     "confident": True, "model_label": "star",
                                     "model_confidence": 0.9, "model_verdict": "",
                                     "note": "", "timestamp": 1.0}]))
        loaded = VerdictLog.load(str(path))
        assert loaded.records[0].kind == "" and loaded.records[0].candidate_id is None
        assert isinstance(loaded.records[0], HumanVerdict)


class TestHttp:
    @pytest.fixture()
    def server(self, analysed, tmp_path):
        image, analysis = analysed
        session = VettingSession(build_queue(analysis, image, limit=4, include_sources=True),
                                 log_path=str(tmp_path / "log.json"))
        server = VettingServer(session, host="127.0.0.1", port=0).start()
        yield server, session
        server.stop()

    @staticmethod
    def _get(url):
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()

    @staticmethod
    def _post(url, payload):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_the_page_and_the_api_are_served(self, server):
        srv, session = server
        status, kind, body = self._get(srv.url)
        assert status == 200 and "text/html" in kind and b"AstroVision-X vetting" in body
        status, kind, body = self._get(srv.url + "api/next?reviewer=ana")
        item = json.loads(body)
        assert status == 200 and item["item_id"] == 1 and item["n_items"] == len(session.queue)
        status, kind, body = self._get(srv.url + "api/cutout/1.png")
        assert status == 200 and kind == "image/png" and decode_png_size(body) == (256, 256)
        status, kind, body = self._get(srv.url + "api/cutout/1.png?kind=subtracted")
        assert status == 200
        status, _, body = self._get(srv.url + "api/queue")
        assert json.loads(body)["n_items"] == len(session.queue)

    def test_a_verdict_round_trips_to_the_log(self, server):
        srv, session = server
        status, body = self._post(srv.url + "api/verdict",
                                  {"item_id": 1, "label": "bogus", "reviewer": "ana",
                                   "note": "diffraction spike"})
        assert status == 200
        recorded = json.loads(body)["recorded"]
        assert recorded["label"] == "bogus" and recorded["reviewer"] == "ana"
        assert len(VerdictLog.load(session.log_path)) == 1
        status, _, body = self._get(srv.url + "api/next?after=1&reviewer=ana")
        assert json.loads(body)["item_id"] == 2
        status, _, body = self._get(srv.url + "api/progress")
        progress = json.loads(body)
        assert progress["n_done"] == 1 and progress["counts"]["bogus"] == 1

    def test_the_server_refuses_an_anonymous_verdict(self, server):
        srv, session = server
        status, body = self._post(srv.url + "api/verdict",
                                  {"item_id": 1, "label": "real", "reviewer": ""})
        assert status == 400 and b"reviewer" in body
        assert len(session.log) == 0
        status, body = self._post(srv.url + "api/verdict",
                                  {"item_id": 999, "label": "real", "reviewer": "ana"})
        assert status == 400

    def test_when_everything_is_decided_next_is_empty(self, server):
        srv, session = server
        for item in session.queue:
            session.record(item.item_id, "real", "ana")
        status, _, body = self._get(srv.url + "api/next?reviewer=ana")
        assert json.loads(body) == {}
