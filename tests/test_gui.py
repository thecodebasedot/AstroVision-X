"""The desktop application, driven through its HTTP API the way the page drives it."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest

from astrovision.gui.app import App, AppServer


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read()
        return response.status, response.headers.get("Content-Type", ""), body


def _json(url):
    status, _, body = _get(url)
    return status, json.loads(body)


def _post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _wait(base, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job = _json(f"{base}api/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("gui")
    app = App(workdir=str(workdir))
    srv = AppServer(app, host="127.0.0.1", port=0).start()
    yield srv, str(workdir)
    srv.stop()


class TestStatusAndFiles:
    def test_the_page_and_the_status_are_served(self, server):
        srv, workdir = server
        status, ctype, body = _get(srv.url)
        assert status == 200 and "text/html" in ctype and b"AstroVision-X" in body
        _, info = _json(srv.url + "api/status")
        assert info["workdir"] == workdir and "quicklook" in info["presets"]
        assert "never declares a discovery" in info["boundary"]
        assert isinstance(info["backends"], dict)

    def test_browse_lists_only_what_the_page_can_open(self, server):
        srv, workdir = server
        for name in ("a.fits", "b.avro", "c.csv", "ignored.xyz", ".hidden.fits"):
            open(os.path.join(workdir, name), "wb").close()
        os.makedirs(os.path.join(workdir, "sub"), exist_ok=True)
        _, listing = _json(srv.url + "api/browse?path=" + workdir)
        names = {f["name"]: f["kind"] for f in listing["files"]}
        assert names == {"a.fits": "image", "b.avro": "alerts", "c.csv": "other"}
        assert [d["name"] for d in listing["dirs"]] == ["sub"]
        assert listing["parent"] == os.path.dirname(workdir)
        try:
            _get(srv.url + "api/browse?path=" + os.path.join(workdir, "missing"))
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("expected 404")

    def test_a_missing_file_is_a_404_not_a_crash(self, server):
        srv, workdir = server
        status, payload = _post(srv.url + "api/analyze", {"path": os.path.join(workdir, "nope.fits")})
        assert status == 404 and "no such file" in payload["error"]
        try:
            _get(srv.url + "api/inspect?path=" + os.path.join(workdir, "nope.fits"))
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("expected 404")


class TestSimulateThenAnalyse:
    @pytest.fixture(scope="class")
    def simulated(self, server):
        srv, workdir = server
        out = os.path.join(workdir, "field.fits")
        status, job = _post(srv.url + "api/simulate", {"size": 128, "stars": 25, "galaxies": 5,
                                                       "nebulae": 0, "clusters": 0, "lenses": 1,
                                                       "anomalies": 1, "seed": 7, "out": out})
        assert status == 200
        job = _wait(srv.url, job["id"])
        assert job["status"] == "done", job["error"]
        assert job["result"]["paths"] == [out] and os.path.exists(job["result"]["truth"])
        return out

    @pytest.fixture(scope="class")
    def analysed(self, server, simulated):
        srv, workdir = server
        status, job = _post(srv.url + "api/analyze", {"path": simulated, "preset": "quicklook",
                                                      "formats": ["html", "json"],
                                                      "output_dir": os.path.join(workdir, "out"),
                                                      "db": os.path.join(workdir, "cat.sqlite")})
        assert status == 200 and job["status"] in ("queued", "running")
        job = _wait(srv.url, job["id"])
        assert job["status"] == "done", job["error"]
        return job

    def test_inspect_and_preview(self, server, simulated):
        srv, _ = server
        _, info = _json(srv.url + "api/inspect?path=" + simulated)
        assert info["shape"] == [128, 128] and info["wcs"] is not None and "centre" in info
        status, ctype, body = _get(srv.url + "api/preview.png?path=" + simulated)
        assert status == 200 and ctype == "image/png" and body[:8] == b"\x89PNG\r\n\x1a\n"

    def test_stages_were_reported_as_they_ran(self, analysed):
        names = [s["name"] for s in analysed["stages"]]
        assert names[:3] == ["preprocess", "detect", "photometry"]
        by_name = {s["name"]: s for s in analysed["stages"]}
        assert by_name["detect"]["status"] == "ok" and by_name["detect"]["seconds"] > 0
        assert by_name["preprocess"]["status"] == "ok"
        assert all(s["status"] in ("ok", "skipped", "failed") for s in analysed["stages"])
        assert analysed["result"]["n_sources"] > 5
        assert analysed["result"]["database"]["n_detections"] == analysed["result"]["n_sources"]
        assert any("detected" in line for line in analysed["log"])

    def test_the_results_are_all_reachable(self, server, analysed):
        srv, _ = server
        base = f"{srv.url}api/jobs/{analysed['id']}/"
        status, ctype, body = _get(base + "report.html")
        assert status == 200 and "text/html" in ctype and b"OVERVIEW" in body.upper()
        _, report = _json(base + "report.json")
        assert report["summary"]["n_sources"] == analysed["result"]["n_sources"]
        _, catalog = _json(base + "catalog?sort=mag&order=asc&limit=5")
        assert catalog["total"] == analysed["result"]["n_sources"] and len(catalog["rows"]) == 5
        mags = [r["mag"] for r in catalog["rows"] if r["mag"] is not None]
        assert mags == sorted(mags)
        _, filtered = _json(base + "catalog?q=star")
        assert 0 < filtered["total"] <= catalog["total"]
        _, candidates = _json(base + "candidates?limit=5")
        assert isinstance(candidates, list) and all("position" in c for c in candidates)
        row = catalog["rows"][0]
        status, ctype, body = _get(base + f"cutout.png?x={row['x']}&y={row['y']}&size=32")
        assert status == 200 and ctype == "image/png"
        status, ctype, _ = _get(base + "preview.png")
        assert status == 200 and ctype == "image/png"
        assert os.path.exists(analysed["result"]["files"]["html"])

    def test_vetting_hands_off_to_its_own_server(self, server, analysed):
        srv, workdir = server
        status, v = _post(srv.url + "api/vet", {"job_id": analysed["id"], "all_sources": True,
                                                "log": os.path.join(workdir, "verdicts.json")})
        assert status == 200 and v["n_items"] > 0 and v["url"].startswith("http://127.0.0.1:")
        status, _, body = _get(v["url"] + "api/queue")
        assert status == 200

    def test_alerts_are_listed_and_vettable(self, server, tmp_path):
        import numpy as np
        from astrovision.alerts import AlertPacket, Detection, write_alerts
        srv, _ = server
        packet = AlertPacket(object_id="AVX1", candid=11, ra=1.0, dec=2.0, mjd=60000.0, band="r",
                             mag=19.0, mag_err=0.1, real_bogus=0.8,
                             history=[Detection(mjd=59998.0, band="r", mag=19.5, mag_err=0.2)],
                             cutout_science=np.random.default_rng(0).normal(size=(31, 31)))
        path = str(tmp_path / "a.avro")
        write_alerts(path, [packet])
        _, listing = _json(srv.url + "api/alerts?path=" + path)
        assert listing["n_packets"] == 1 and listing["rows"][0]["object_id"] == "AVX1"
        assert listing["rows"][0]["has_cutouts"] and listing["rows"][0]["n_history"] == 1
        status, v = _post(srv.url + "api/vet", {"path": path, "log": str(tmp_path / "v.json")})
        assert status == 200 and v["n_items"] == 1

    def test_runs_are_listed(self, server, analysed):
        srv, _ = server
        _, jobs = _json(srv.url + "api/jobs")
        assert any(j["id"] == analysed["id"] and j["status"] == "done" for j in jobs)
        assert all("result" not in j for j in jobs)          # the list is light


class TestCancelAndFiles:
    def test_a_running_analysis_stops_after_its_current_stage(self, server):
        srv, workdir = server
        out = os.path.join(workdir, "big.fits")
        _, job = _post(srv.url + "api/simulate", {"size": 384, "stars": 150, "galaxies": 40,
                                                  "nebulae": 1, "clusters": 1, "lenses": 1,
                                                  "anomalies": 2, "seed": 5, "out": out})
        assert _wait(srv.url, job["id"])["status"] == "done"
        status, job = _post(srv.url + "api/analyze", {"path": out, "formats": ["json"],
                                                      "output_dir": os.path.join(workdir, "out2")})
        assert status == 200
        # Let it get going, then ask it to stop.
        deadline = time.time() + 60
        while time.time() < deadline:
            _, j = _json(f"{srv.url}api/jobs/{job['id']}")
            if any(s["status"] == "ok" for s in j["stages"]):
                break
            time.sleep(0.3)
        status, reply = _post(f"{srv.url}api/jobs/{job['id']}/cancel", {})
        assert status == 200 and reply["cancel_requested"]
        j = _wait(srv.url, job["id"], timeout=240)
        assert j["status"] == "cancelled", j["error"]
        assert "stopped before stage" in j["error"]
        done = [s["name"] for s in j["stages"] if s["status"] == "ok"]
        not_run = [s for s in j["stages"] if s["status"] == "cancelled"]
        assert done and not_run                        # something ran, something did not
        assert any("cancel requested" in line for line in j["log"])
        # Cancelling a finished job changes nothing.
        status, reply = _post(f"{srv.url}api/jobs/{job['id']}/cancel", {})
        assert status == 200 and reply["status"] == "cancelled"

    def test_written_files_are_served_by_their_key(self, server, tmp_path):
        srv, workdir = server
        out = os.path.join(workdir, "small.fits")
        _, job = _post(srv.url + "api/simulate", {"size": 96, "stars": 12, "galaxies": 2,
                                                  "nebulae": 0, "clusters": 0, "lenses": 0,
                                                  "anomalies": 0, "seed": 9, "out": out})
        sim = _wait(srv.url, job["id"])
        status, ctype, body = _get(f"{srv.url}api/jobs/{sim['id']}/file?name=truth")
        assert status == 200 and "json" in ctype and json.loads(body)
        _, job = _post(srv.url + "api/analyze", {"path": out, "preset": "quicklook",
                                                 "formats": ["html", "json"],
                                                 "output_dir": os.path.join(workdir, "out3")})
        done = _wait(srv.url, job["id"])
        assert done["status"] == "done", done["error"]
        status, ctype, body = _get(f"{srv.url}api/jobs/{done['id']}/file?name=html")
        assert status == 200 and "text/html" in ctype and b"<html" in body.lower()
        status, ctype, body = _get(f"{srv.url}api/jobs/{done['id']}/file?name=catalog_csv")
        assert status == 200 and "csv" in ctype and body.startswith(b"id")
        try:
            _get(f"{srv.url}api/jobs/{done['id']}/file?name=../../etc/passwd")
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("only the job's own files may be served")


def test_the_self_test_passes():
    from astrovision.gui.app import self_test
    assert self_test() == 0
