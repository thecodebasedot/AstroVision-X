"""Agreement with photutils and SEP, on the same pixels."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.validation.benchmark import (BenchmarkResult, ToolCatalog, _match,
                                              available_tools, benchmark_field, compare)

TOOLS = available_tools()
needs_photutils = pytest.mark.skipif(not TOOLS["photutils"], reason="photutils not installed")
needs_sep = pytest.mark.skipif(not TOOLS["sep"], reason="sep not installed")


class TestMatching:
    def test_mutual_nearest_neighbours_only(self):
        x1, y1 = np.array([10.0, 50.0, 90.0]), np.array([10.0, 50.0, 90.0])
        x2, y2 = np.array([10.5, 50.2, 30.0]), np.array([10.0, 49.9, 30.0])
        pairs = _match(x1, y1, x2, y2, 2.0)
        assert pairs == [(0, 0), (1, 1)]

    def test_compare_reports_agreement_and_disagreement(self):
        theirs = ToolCatalog("tool", np.array([10.0, 50.0, 90.0]), np.array([10.0, 50.0, 90.0]),
                             np.array([100.0, 200.0, 300.0]), seconds=0.1, n=3)
        result = compare(np.array([10.1, 50.0, 90.0, 80.0]), np.array([10.0, 50.1, 89.9, 80.0]),
                         np.array([102.0, 196.0, 303.0, 5.0]), theirs)
        assert isinstance(result, BenchmarkResult)
        assert result.n_matched == 3 and result.only_ours == 1 and result.only_theirs == 0
        assert result.flux_ratio_median == pytest.approx(1.0, abs=0.03)
        assert result.position_offset_median == pytest.approx(0.1, abs=0.01)
        assert "matched_fraction" in result.to_dict()
        assert "tool" in result.summary()


@pytest.fixture(scope="module")
def measured_field():
    from astrovision.detect import Detector
    from astrovision.photometry import Photometer
    from astrovision.preprocess import Preprocessor
    from astrovision.simulate import quick_field

    image, truth = quick_field((384, 384), seed=1, n_stars=90, n_galaxies=12,
                               n_nebulae=1, n_clusters=1)
    clean = Preprocessor().run(image, estimate_psf=False)
    catalog, segmentation = Detector().detect(clean)
    Photometer().run(clean, catalog, segmentation)
    return clean, catalog, truth


@pytest.mark.parametrize("tool", [pytest.param("photutils", marks=needs_photutils),
                                  pytest.param("sep", marks=needs_sep)])
def test_this_package_agrees_with_the_standard_tools(measured_field, tool):
    """Where both codes detect an object they must find it in the same place
    and measure the same flux through the same aperture; the measured
    agreement (0.06 px, 0.2% in flux) is recorded in docs/validation.md."""
    clean, catalog, truth = measured_field
    # Recall is scored on objects bright enough for every code to see; the
    # faint end is where the codes differ on purpose, see docs/validation.md.
    bright = [o for o in truth if o.flux > 1500]
    results = benchmark_field(clean, catalog, truth=bright, tools=(tool,))
    assert len(results) == 1
    result = results[0]
    assert result.matched_fraction > 0.9
    assert result.position_offset_median < 0.2
    assert abs(result.flux_ratio_median - 1.0) < 0.02
    assert result.flux_ratio_scatter < 0.05
    ours, theirs = result.against_truth["ours"], result.against_truth[tool]
    assert ours["recall"] > 0.8 and theirs["recall"] > 0.8
    assert abs(ours["flux_ratio_median"] - theirs["flux_ratio_median"]) < 0.02
