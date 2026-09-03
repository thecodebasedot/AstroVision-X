"""A manifest that matches must mean a catalog that matches."""

from __future__ import annotations

import json

import numpy as np
import pytest

from astrovision.core.config import AstroVisionConfig
from astrovision.core.provenance import (Manifest, build_manifest, catalog_digest, config_hash,
                                         file_checksum, same_result)
from astrovision.core.types import BoundingBox, Source, SourceCatalog


def _catalog(offset=0.0):
    catalog = SourceCatalog()
    for index in range(5):
        source = Source(id=index + 1, x=10.0 * index + offset, y=3.0 * index,
                        bbox=BoundingBox(0, 0, 1, 1))
        source.photometry.flux = 100.0 * (index + 1)
        catalog.append(source)
    return catalog


class TestHashes:
    def test_config_hash_ignores_key_order_and_sees_values(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})
        cfg = AstroVisionConfig()
        other = AstroVisionConfig()
        other.detection.threshold_sigma += 0.5
        assert config_hash(cfg) != config_hash(other)

    def test_file_checksum_streams_and_changes_with_content(self, tmp_path):
        path = tmp_path / "frame.bin"
        path.write_bytes(b"\x00" * (3 << 20))
        first = file_checksum(str(path))
        path.write_bytes(b"\x00" * (3 << 20 - 1) + b"\x01")
        assert first.startswith("sha256:") and first != file_checksum(str(path))

    def test_catalog_digest_is_order_free_and_rounds_last_bit_noise(self):
        a, b = _catalog(), _catalog()
        assert catalog_digest(a) == catalog_digest(b)
        b[2].x += 1e-9
        assert same_result(a, b)
        b[2].x += 1e-3
        assert not same_result(a, b)
        assert catalog_digest(_catalog()) != catalog_digest(_catalog(offset=0.5))
        shuffled = SourceCatalog(list(reversed(list(_catalog()))))
        assert same_result(_catalog(), shuffled)


class TestManifest:
    def test_records_everything_that_determines_a_result(self, tmp_path):
        frame = tmp_path / "frame.fits"
        frame.write_bytes(b"SIMPLE  = T" + b" " * 2880)
        manifest = build_manifest(AstroVisionConfig(), inputs=[str(frame)],
                                  seeds={"random_state": 42}, notes=["test"])
        assert manifest.package_version
        assert manifest.config_hash.startswith("sha256:")
        assert manifest.dependencies["numpy"] == np.__version__
        assert manifest.seeds == {"random_state": 42}
        assert manifest.inputs == {"frame.fits": file_checksum(str(frame))}
        assert manifest.python and manifest.platform and manifest.created
        assert "revision" in manifest.git

    def test_a_missing_input_is_noted_not_fatal(self):
        manifest = build_manifest({}, inputs=["/no/such/frame.fits"])
        assert any("not found" in note for note in manifest.notes)

    def test_round_trips_through_json(self, tmp_path):
        manifest = build_manifest(AstroVisionConfig(), seeds={"random_state": 7})
        path = manifest.save(str(tmp_path / "run" / "manifest.json"))
        loaded = Manifest.load(path)
        assert loaded.reproducibility_key() == manifest.reproducibility_key()
        # Tuples become lists in JSON; compare after the same round trip.
        assert loaded.to_dict() == json.loads(json.dumps(manifest.to_dict(), default=str))

    def test_differences_name_the_reason(self):
        first = build_manifest(AstroVisionConfig(), seeds={"random_state": 1})
        second = build_manifest(AstroVisionConfig(), seeds={"random_state": 2})
        assert first.differences(second) and "random seeds differ" in first.differences(second)
        other = AstroVisionConfig()
        other.photometry.primary_aperture += 1.0
        third = build_manifest(other, seeds={"random_state": 1})
        assert "configuration differs" in first.differences(third)
        assert first.reproducibility_key() != third.reproducibility_key()

    def test_the_seed_in_the_config_is_picked_up(self):
        cfg = AstroVisionConfig()
        cfg.random_state = 99
        assert build_manifest(cfg).seeds["random_state"] == 99


class TestReproducibility:
    def test_same_manifest_means_same_catalog(self):
        """The property the whole module exists to guarantee: two runs with
        matching reproducibility keys produce byte-identical measurements."""
        from astrovision.engine.tiles import standard_stage
        from astrovision.simulate import quick_field

        cfg = AstroVisionConfig()
        image, _ = quick_field((160, 160), seed=3)
        first, _ = standard_stage(cfg)(image)
        second, _ = standard_stage(cfg)(image)
        a = build_manifest(cfg, seeds={"random_state": cfg.random_state})
        b = build_manifest(cfg, seeds={"random_state": cfg.random_state})
        assert a.reproducibility_key() == b.reproducibility_key()
        assert same_result(first, second)

    def test_the_pipeline_report_carries_the_manifest(self):
        from astrovision.engine.pipeline import Pipeline
        from astrovision.report.schema import build_report
        from astrovision.simulate import quick_field

        image, _ = quick_field((128, 128), seed=2)
        analysis = Pipeline().run(image)
        manifest = analysis.provenance["manifest"]
        assert manifest["config_hash"] == config_hash(AstroVisionConfig())
        assert manifest["outputs"]["catalog_digest"] == catalog_digest(analysis.catalog)
        assert analysis.provenance["reproducibility_key"].startswith("sha256:")
        report = build_report(analysis)
        assert report["manifest"]["config_hash"] == manifest["config_hash"]
        assert report["reproducibility_key"] == analysis.provenance["reproducibility_key"]
