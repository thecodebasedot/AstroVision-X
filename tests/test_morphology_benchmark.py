"""Morphology against statmorph: the comparison, and what it found."""

from __future__ import annotations

import pytest

from astrovision.core.types import BoundingBox, Source, SourceCatalog
from astrovision.validation.morphology_benchmark import (compare_morphology, statmorph_available)

needs_statmorph = pytest.mark.skipif(not statmorph_available(), reason="statmorph not installed")


def _source(ident, gini, asym, n=None):
    source = Source(id=ident, x=10.0 * ident, y=5.0, bbox=BoundingBox(0, 0, 1, 1))
    source.morphology.gini = gini
    source.morphology.asymmetry = asym
    if n is not None:
        source.morphology.sersic_index = n
    return source


class TestComparison:
    def test_offset_scatter_and_rank_correlation(self):
        catalog = SourceCatalog([_source(k, 0.5 + 0.01 * k, 0.1 * k) for k in range(1, 9)])
        theirs = {k: {"gini": 0.5 + 0.01 * k - 0.02, "asymmetry": 0.8 - 0.1 * k, "flag": 0}
                  for k in range(1, 9)}
        report = compare_morphology(catalog, theirs)
        by_name = {m.metric: m for m in report.metrics}
        assert by_name["gini"].n == 8
        assert by_name["gini"].median_difference == pytest.approx(0.02)
        assert by_name["gini"].rank_correlation == pytest.approx(1.0)
        assert by_name["asymmetry"].rank_correlation == pytest.approx(-1.0)
        assert by_name["m20"].n == 0                          # neither side measured it
        assert "statmorph comparison" in report.summary()

    def test_flagged_and_missing_sources_are_left_out(self):
        catalog = SourceCatalog([_source(1, 0.5, 0.1), _source(2, 0.6, 0.2), _source(3, 0.7, 0.3)])
        theirs = {1: {"gini": 0.5, "flag": 0}, 2: {"gini": 0.9, "flag": 2}}
        report = compare_morphology(catalog, theirs)
        assert report.metrics[0].n == 1
        assert any("flagged" in note for note in report.notes)

    def test_sersic_index_is_scored_against_truth(self):
        class Truth:
            def __init__(self, x, y, n):
                self.x, self.y, self.kind, self.sersic_n = x, y, "galaxy", n

        catalog = SourceCatalog([_source(1, 0.5, 0.1, n=1.5), _source(2, 0.6, 0.2, n=4.5)])
        theirs = {1: {"sersic_n": 1.0, "flag": 0, "flag_sersic": 0},
                  2: {"sersic_n": 4.0, "flag": 0, "flag_sersic": 0}}
        truth = [Truth(10.0, 5.0, 1.0), Truth(20.0, 5.0, 4.0)]
        report = compare_morphology(catalog, theirs, truth=truth)
        sersic = [m for m in report.metrics if m.metric == "sersic_index"][0]
        assert sersic.ours_vs_truth == pytest.approx(0.5)
        assert sersic.theirs_vs_truth == pytest.approx(0.0)


@needs_statmorph
class TestAgainstStatmorph:
    @pytest.fixture(scope="class")
    def report(self):
        from astrovision.core.config import MorphologyConfig
        from astrovision.detect import Detector
        from astrovision.morphology import MorphologyAnalyzer
        from astrovision.photometry import Photometer
        from astrovision.preprocess import Preprocessor
        from astrovision.simulate import SkyConfig, SkySimulator
        from astrovision.validation.morphology_benchmark import benchmark_morphology

        image, truth = SkySimulator(SkyConfig(
            shape=(320, 320), n_stars=20, n_galaxies=30, n_nebulae=0, n_clusters=0,
            n_lenses=0, n_anomalies=0, seed=4, cosmic_ray_rate=0.0,
            bad_column_count=0)).generate()
        clean = Preprocessor().run(image)
        catalog, segmentation = Detector().detect(clean)
        Photometer().run(clean, catalog, segmentation)
        MorphologyAnalyzer(MorphologyConfig(fit_sersic=False)).run(clean, catalog, segmentation)
        return benchmark_morphology(clean, catalog, segmentation, truth=truth)

    def test_gini_and_m20_agree(self, report):
        """Same definitions on the same pixels: the scatter is a few percent."""
        by_name = {m.metric: m for m in report.metrics}
        assert by_name["gini"].n >= 10
        assert abs(by_name["gini"].median_difference) < 0.03
        assert by_name["gini"].scatter < 0.05
        assert by_name["m20"].scatter < 0.1
        assert by_name["m20"].rank_correlation > 0.7

    def test_asymmetry_ranks_the_same_way(self, report):
        """Before the sky correction the rank correlation was -0.8: the
        statistic measured noise, not shape. It must stay positive."""
        asym = {m.metric: m for m in report.metrics}["asymmetry"]
        assert asym.n >= 10
        assert asym.rank_correlation > 0.3
        assert abs(asym.median_difference) < 0.15
