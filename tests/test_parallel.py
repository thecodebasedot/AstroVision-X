"""The process pool behind the per-source stages."""

from __future__ import annotations

import math

import pytest

from astrovision.core import parallel
from astrovision.core.parallel import map_work, worker_count


class TestWorkerCount:
    def test_zero_means_all_cores_but_one_capped(self, monkeypatch):
        monkeypatch.setattr(parallel.os, "cpu_count", lambda: 4)
        assert worker_count(0) == 3
        monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
        assert worker_count(0) == parallel.MAX_AUTO_WORKERS
        monkeypatch.setattr(parallel.os, "cpu_count", lambda: 1)
        assert worker_count(0) == 1

    def test_explicit_counts_are_kept_and_never_below_one(self):
        assert worker_count(3) == 3 and worker_count(-2) == 1 and worker_count(None) >= 1


class TestMapWork:
    def test_one_worker_or_few_items_runs_in_this_process(self, monkeypatch):
        def no_pool(size):
            raise AssertionError("the pool must not be started")
        monkeypatch.setattr(parallel, "_get_pool", no_pool)
        assert map_work(math.sqrt, range(20), n_workers=1) == [math.sqrt(i) for i in range(20)]
        assert map_work(math.sqrt, range(3), n_workers=4) == [math.sqrt(i) for i in range(3)]

    def test_the_pool_gives_the_same_answer_in_order(self):
        items = list(range(40))
        try:
            got = map_work(math.sqrt, items, n_workers=2)
        finally:
            parallel.close_pool()
        assert got == [math.sqrt(i) for i in items]

    def test_a_pool_that_cannot_start_falls_back_quietly(self, monkeypatch):
        def broken(size):
            raise RuntimeError("worker processes did not start")
        monkeypatch.setattr(parallel, "_get_pool", broken)
        assert map_work(abs, range(-10, 10), n_workers=3) == [abs(i) for i in range(-10, 10)]

    def test_workers_are_left_out_of_the_reproducibility_hash(self):
        from astrovision.core.config import AstroVisionConfig
        from astrovision.core.provenance import config_hash
        a, b = AstroVisionConfig(), AstroVisionConfig()
        b.n_workers = 0
        assert config_hash(a) == config_hash(b)
        b.detection.threshold_sigma = 4.0
        assert config_hash(a) != config_hash(b)


@pytest.mark.slow
class TestStagesAreTheSameInParallel:
    def test_morphology_and_lens_search_match_the_serial_run(self):
        import pickle

        from astrovision.classify import Classifier
        from astrovision.detect import Detector
        from astrovision.lensing import LensSearch
        from astrovision.morphology import MorphologyAnalyzer
        from astrovision.photometry import Photometer
        from astrovision.preprocess import Preprocessor
        from astrovision.simulate import SkyConfig, SkySimulator

        image, _ = SkySimulator(SkyConfig(shape=(256, 256), n_stars=40, n_galaxies=25,
                                          n_lenses=1, n_nebulae=0, n_clusters=0,
                                          n_anomalies=0, seed=21)).generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)

        def run(workers):
            copy = pickle.loads(pickle.dumps(catalog))
            MorphologyAnalyzer(n_workers=workers).run(clean, copy, segmentation)
            Classifier().run(clean, copy)
            found = LensSearch(n_workers=workers).run(clean, copy)
            return ([(s.id, s.morphology.to_dict(), s.lens_score, sorted(s.flags))
                     for s in copy], [(c.source_id, c.score) for c in found])

        try:
            serial, parallel_run = run(1), run(2)
        finally:
            parallel.close_pool()
        assert len(serial[0]) >= 8                       # enough to have used the pool
        for (i, a, la, fa), (j, b, lb, fb) in zip(serial[0], parallel_run[0]):
            assert i == j and la == lb and fa == fb
            for key in a:
                x, y = a[key], b[key]
                if isinstance(x, float) and isinstance(y, float):
                    assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-12)
                else:
                    assert x == y
        assert serial[1] == parallel_run[1]
