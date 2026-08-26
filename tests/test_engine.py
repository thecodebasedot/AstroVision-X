"""The pipeline, the research assistant, reports and the CLI."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from astrovision.core.config import AstroVisionConfig
from astrovision.core.types import FieldAnalysis, Verdict
from astrovision.engine import Pipeline, ResearchAssistant, rank_candidates
from astrovision.engine.assistant import DISCOVERY_DISCLAIMER
from astrovision.report import generate_reports, render_html, render_json, render_text


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def analysis(request):
    """One pipeline run, reused by the report and assistant tests."""
    from astrovision.simulate import SkyConfig, SkySimulator

    image, _ = SkySimulator(SkyConfig(
        shape=(192, 192), n_stars=40, n_galaxies=10, n_nebulae=1, n_clusters=1,
        n_lenses=1, n_anomalies=1, seed=2024)).generate()
    result = Pipeline().run(image, redshift=0.1)
    request.node.session._astrovision_image = image
    return result


class TestPipeline:
    def test_produces_a_catalog(self, analysis):
        assert analysis.summary()["n_sources"] > 0

    def test_every_stage_is_accounted_for(self, analysis):
        names = {s["name"] for s in analysis.provenance["stages"]}
        for expected in ("preprocess", "detect", "photometry", "morphology",
                         "classification", "anomaly", "lensing", "statistics"):
            assert expected in names

    def test_no_stage_failed(self, analysis):
        failed = [s for s in analysis.provenance["stages"] if s["status"] == "failed"]
        assert not failed, f"stages failed: {[s['name'] for s in failed]}"

    def test_provenance_records_the_configuration(self, analysis):
        assert analysis.provenance["config"]["detection"]["threshold_sigma"] > 0
        assert analysis.provenance["version"]

    def test_sources_carry_measurements(self, analysis):
        source = analysis.catalog.brightest(1)[0]
        assert np.isfinite(source.photometry.flux)
        assert np.isfinite(source.photometry.magnitude)
        assert source.object_class is not None

    def test_disabled_stages_are_marked_skipped(self):
        from astrovision.simulate import quick_field

        config = AstroVisionConfig().with_preset("quicklook")
        image, _ = quick_field((128, 128), seed=5)
        result = Pipeline(config).run(image)
        skipped = {s["name"] for s in result.provenance["stages"]
                   if s["status"] == "skipped"}
        assert "lensing" in skipped and "anomaly" in skipped

    def test_time_domain_stages_run_on_a_series(self, synthetic_series):
        series, _, _ = synthetic_series
        result = Pipeline().run_series(series)
        statuses = {s["name"]: s["status"] for s in result.provenance["stages"]}
        assert statuses["transient"] == "ok"
        assert statuses["timeseries"] == "ok"
        assert result.provenance["series"]["n_epochs"] == len(series)

    def test_a_blank_image_degrades_gracefully(self):
        from astrovision.io.image import AstroImage

        blank = AstroImage.from_array(
            np.random.default_rng(0).normal(0.0, 1.0, (64, 64)), name="blank")
        result = Pipeline().run(blank)
        assert result.summary()["n_sources"] >= 0
        assert isinstance(result.warnings, list)


class TestAssistant:
    def test_summarises_an_empty_field(self):
        assert "no sources" in ResearchAssistant().summarise(FieldAnalysis()).lower()

    def test_summary_mentions_the_source_count(self, analysis):
        summary = ResearchAssistant().summarise(analysis)
        assert str(len(analysis.catalog)) in summary

    def test_recommendations_are_actionable(self, analysis):
        report = ResearchAssistant().report(analysis)
        assert report["recommendations"]
        assert all(isinstance(action, str) and action for action in report["recommendations"])

    def test_disclaimer_is_always_present(self, analysis):
        assert ResearchAssistant().report(analysis)["disclaimer"] == DISCOVERY_DISCLAIMER

    def test_never_claims_a_discovery(self, analysis):
        """The narrative must speak of candidates, never of detections.

        The disclaimer is excluded from the search because it legitimately
        contains negated forms of these phrases ("nothing here is a
        confirmed detection"); what must not appear is an *assertion*.
        """
        report = ResearchAssistant().report(analysis)
        text = " ".join([report["summary"], *report["recommendations"],
                         *report["priority_text"]]).lower()
        for phrase in ("we have discovered", "confirmed detection",
                       "new supernova found", "is a supernova",
                       "definitive", "proves"):
            assert phrase not in text, f"the narrative must not assert '{phrase}'"

    def test_findings_are_labelled_as_candidates(self, analysis):
        report = ResearchAssistant().report(analysis)
        text = " ".join([report["summary"], report["disclaimer"],
                         *report["priority_text"]]).lower()
        assert "candidate" in text
        assert "confirmed" in report["disclaimer"].lower()

    def test_ranking_is_ordered_and_bounded(self, analysis):
        ranked = rank_candidates(analysis, limit=5)
        assert len(ranked) <= 5
        scores = [item.score for item in ranked]
        assert scores == sorted(scores, reverse=True)
        assert all(item.rank == i for i, item in enumerate(ranked, start=1))

    def test_every_ranked_item_explains_itself(self, analysis):
        for item in rank_candidates(analysis, limit=5):
            assert item.reasons, "a ranked candidate must say why it is ranked"
            assert isinstance(item.verdict, Verdict)


class TestReports:
    def test_text_has_all_the_sections(self, analysis):
        text = render_text(analysis, title="Test Field")
        for heading in ("OVERVIEW", "DATA QUALITY", "FIELD STATISTICS",
                        "RANKED FOLLOW-UP CANDIDATES", "RECOMMENDED NEXT STEPS",
                        "PROVENANCE"):
            assert heading in text
        assert "TEST FIELD" in text

    def test_json_is_valid_and_complete(self, analysis):
        payload = json.loads(render_json(analysis))
        for key in ("summary", "priority", "recommendations", "disclaimer",
                    "stages", "config"):
            assert key in payload
        assert payload["catalog_size"] == len(analysis.catalog)

    def test_json_has_no_nan(self, analysis):
        """NaN is not valid JSON; the writer must convert it to null."""
        raw = render_json(analysis)
        assert "NaN" not in raw and "Infinity" not in raw

    def test_html_is_self_contained(self, analysis, request):
        image = getattr(request.node.session, "_astrovision_image", None)
        document = render_html(analysis, image=image)
        assert document.startswith("<!doctype html>")
        assert "</html>" in document
        # No external resources: everything must be inlined.
        assert "src='http" not in document and 'src="http' not in document

    def test_html_escapes_content(self, analysis):
        document = render_html(analysis, title="<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in document
        assert "&lt;script&gt;" in document

    def test_generate_reports_writes_every_format(self, analysis, tmp_path):
        written = generate_reports(analysis, str(tmp_path), ("text", "json", "html"))
        for key in ("text", "json", "html", "catalog_csv"):
            assert key in written
            assert os.path.getsize(written[key]) > 0

    def test_unknown_format_is_skipped_not_fatal(self, analysis, tmp_path):
        written = generate_reports(analysis, str(tmp_path), ("text", "nonsense"))
        assert "text" in written and "nonsense" not in written


class TestCLI:
    def test_help_exits_cleanly(self, capsys):
        from astrovision.cli.main import main

        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0

    def test_info_lists_backends(self, capsys):
        from astrovision.cli.main import main

        assert main(["info"]) == 0
        assert "AstroVision-X" in capsys.readouterr().out

    def test_config_writes_a_file(self, tmp_path):
        from astrovision.cli.main import main

        path = str(tmp_path / "cfg.json")
        assert main(["config", "--out", path]) == 0
        assert AstroVisionConfig.load(path).name

    def test_simulate_then_analyze(self, tmp_path, capsys):
        from astrovision.cli.main import main

        field = str(tmp_path / "field.fits")
        assert main(["simulate", "--out", field, "--size", "128", "--stars", "25",
                     "--galaxies", "5", "--nebulae", "0", "--clusters", "0",
                     "--lenses", "0", "--anomalies", "0", "--seed", "3"]) == 0
        assert os.path.exists(field)
        assert os.path.exists(str(tmp_path / "field_truth.json"))

        output = str(tmp_path / "out")
        assert main(["--log-level", "error", "analyze", field, "-o", output,
                     "--report", "text,json"]) == 0
        assert os.path.exists(os.path.join(output, "report.txt"))
        assert os.path.exists(os.path.join(output, "catalog.csv"))

    def test_analyze_reports_a_missing_file(self, tmp_path):
        from astrovision.cli.main import main

        assert main(["--log-level", "critical", "analyze",
                     str(tmp_path / "nope.fits")]) == 1

    def test_series_needs_two_epochs(self, tmp_path):
        from astrovision.cli.main import main

        field = str(tmp_path / "one.fits")
        main(["simulate", "--out", field, "--size", "64", "--stars", "5",
              "--galaxies", "0", "--nebulae", "0", "--clusters", "0",
              "--lenses", "0", "--anomalies", "0"])
        assert main(["--log-level", "critical", "series", field]) == 1

    def test_inspect_runs(self, tmp_path, capsys):
        from astrovision.cli.main import main

        field = str(tmp_path / "f.fits")
        main(["simulate", "--out", field, "--size", "64", "--stars", "5",
              "--galaxies", "0", "--nebulae", "0", "--clusters", "0",
              "--lenses", "0", "--anomalies", "0"])
        assert main(["inspect", field]) == 0
        assert "AstroImage" in capsys.readouterr().out


class TestTopLevelApi:
    def test_analyze_helper(self, tmp_path):
        import astrovision
        from astrovision.simulate import quick_field

        image, _ = quick_field((128, 128), seed=11)
        path = str(tmp_path / "f.fits")
        image.write(path)
        assert astrovision.analyze(path).summary()["n_sources"] >= 0

    def test_package_exports(self):
        import astrovision

        assert astrovision.__version__
        assert astrovision.Pipeline is not None
        assert callable(astrovision.quick_field)
