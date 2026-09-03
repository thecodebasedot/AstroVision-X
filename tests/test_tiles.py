"""Tiled processing must give the whole-image answer, minus the memory."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.types import BoundingBox, Source, SourceCatalog
from astrovision.engine.tiles import (Tile, cut, merge_catalogs, plan_tiles, process_tiled,
                                      standard_stage)
from astrovision.io.image import AstroImage
from astrovision.simulate import quick_field


def _source(x, y, tile: Tile, ident=1) -> Source:
    source = Source(id=ident, x=float(x), y=float(y),
                    bbox=BoundingBox(x - 2, y - 2, x + 2, y + 2))
    source.meta["tile"] = tile.index
    source.meta["tile_edge_distance"] = tile.distance_to_edge(x, y)
    return source


class TestPlanning:
    def test_cores_partition_the_frame_exactly(self):
        for shape, tile, overlap in [((1024, 1024), 384, 96), ((700, 500), 300, 50),
                                     ((16000, 16000), 2048, 128), ((300, 300), 2048, 128)]:
            tiles = plan_tiles(shape, tile=tile, overlap=overlap)
            covered = np.zeros(shape, dtype=np.int16) if shape[0] <= 1024 else None
            area = 0
            for piece in tiles:
                area += ((piece.core_row1 - piece.core_row0)
                         * (piece.core_col1 - piece.core_col0))
                if covered is not None:
                    covered[piece.core_row0:piece.core_row1,
                            piece.core_col0:piece.core_col1] += 1
            assert area == shape[0] * shape[1]
            if covered is not None:
                assert covered.min() == 1 and covered.max() == 1

    def test_tiles_are_equal_and_never_thin(self):
        """A 160-pixel remainder strip measured fluxes 6% off the rest of
        the frame; the planner stretches tiles evenly instead."""
        tiles = plan_tiles((1024, 1024), tile=384, overlap=96)
        shapes = {piece.shape for piece in tiles}
        assert shapes == {(328, 328)}
        assert len(tiles) == 16

    def test_neighbours_overlap_by_at_least_the_request(self):
        tiles = plan_tiles((1000, 1000), tile=512, overlap=128)
        by_row = {}
        for piece in tiles:
            by_row.setdefault(piece.row0, []).append(piece)
        for pieces in by_row.values():
            pieces.sort(key=lambda p: p.col0)
            for left, right in zip(pieces, pieces[1:]):
                assert left.col1 - right.col0 >= 128

    def test_a_core_lies_half_an_overlap_inside_its_tile(self):
        tiles = plan_tiles((1024, 1024), tile=384, overlap=96)
        inner = [p for p in tiles if 0 < p.row0 and p.row1 < 1024 and 0 < p.col0 and p.col1 < 1024]
        assert inner
        for piece in inner:
            assert piece.core_row0 - piece.row0 >= 48
            assert piece.row1 - piece.core_row1 >= 48

    def test_a_small_frame_is_one_tile(self):
        tiles = plan_tiles((300, 300), tile=2048, overlap=128)
        assert len(tiles) == 1 and tiles[0].shape == (300, 300)
        assert tiles[0].contains_core(299.9, 0.0)


class TestCutting:
    def test_every_plane_is_cut_and_the_origin_recorded(self):
        rng = np.random.default_rng(0)
        image = AstroImage(rng.normal(size=(200, 300)), mask=np.zeros((200, 300), bool),
                           uncertainty=np.ones((200, 300)), name="frame")
        image.mask[150, 250] = True
        piece = plan_tiles(image.shape, tile=160, overlap=40)[-1]
        sub = cut(image, piece)
        assert sub.shape == piece.shape
        assert sub.mask[150 - piece.row0, 250 - piece.col0]
        assert sub.uncertainty.shape == sub.shape
        assert sub.meta["tile_origin"] == (piece.col0, piece.row0)
        assert np.array_equal(sub.data, image.data[piece.row0:piece.row1,
                                                    piece.col0:piece.col1])


class TestMerging:
    def test_a_source_outside_its_tiles_core_is_dropped_unmatched(self):
        """A truncated fragment at a tile edge has its centroid pulled a few
        pixels toward the edge, farther than any matching radius; it is
        dropped because it is outside the core, not because it matched."""
        tiles = plan_tiles((400, 400), tile=250, overlap=100)
        left, right = tiles[0], tiles[1]
        whole = _source(200.0, 100.0, right, 1)                  # in right's core
        fragment = _source(left.col1 - 3.0, 104.0, left, 2)      # 5 px away, at left's edge
        merged, removed = merge_catalogs([(left, SourceCatalog([fragment])),
                                          (right, SourceCatalog([whole]))])
        assert len(merged) == 1 and removed == 1
        assert merged[0].meta["tile"] == right.index

    def test_a_source_on_a_core_boundary_keeps_the_copy_farther_from_an_edge(self):
        tiles = plan_tiles((400, 400), tile=250, overlap=100)
        left, right = tiles[0], tiles[1]
        boundary = left.core_col1
        a = _source(boundary - 0.4, 100.0, left, 1)
        b = _source(boundary + 0.4, 100.0, right, 2)
        merged, removed = merge_catalogs([(left, SourceCatalog([a])),
                                          (right, SourceCatalog([b]))], match_radius=2.0)
        assert len(merged) == 1 and removed == 1
        kept = merged[0]
        expected = a if a.meta["tile_edge_distance"] >= b.meta["tile_edge_distance"] else b
        assert kept.meta["tile"] == expected.meta["tile"]

    def test_two_real_neighbours_in_one_tile_both_survive(self):
        tiles = plan_tiles((400, 400), tile=400, overlap=0)
        only = tiles[0]
        merged, removed = merge_catalogs([(only, SourceCatalog(
            [_source(100.0, 100.0, only, 1), _source(101.0, 100.5, only, 2)]))])
        assert len(merged) == 2 and removed == 0

    def test_ids_are_renumbered_in_frame_order(self):
        tiles = plan_tiles((400, 400), tile=400, overlap=0)
        only = tiles[0]
        merged, _ = merge_catalogs([(only, SourceCatalog(
            [_source(300.0, 300.0, only, 9), _source(10.0, 10.0, only, 7)]))])
        assert [s.id for s in merged] == [1, 2]
        assert merged[0].y < merged[1].y


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def runs(self):
        image, truth = quick_field((512, 512), seed=5, n_stars=120, n_galaxies=15,
                                   n_nebulae=1, n_clusters=1)
        whole, _ = standard_stage()(image)
        tiled = process_tiled(image, standard_stage(), tile=256, overlap=96)
        return image, truth, whole, tiled

    @staticmethod
    def _pairs(a, b, radius=2.0):
        ax = np.array([s.x for s in a]); ay = np.array([s.y for s in a])
        bx = np.array([s.x for s in b]); by = np.array([s.y for s in b])
        d = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
        f, g = d.argmin(axis=1), d.argmin(axis=0)
        return [(i, j) for i, j in enumerate(f) if g[j] == i and d[i, j] <= radius]

    def test_positions_come_back_in_frame_coordinates(self, runs):
        image, truth, whole, tiled = runs
        bright = [o for o in truth if o.flux > 3000]
        pairs = self._pairs(bright, tiled.catalog)
        assert len(pairs) / len(bright) > 0.85
        for source in tiled.catalog:
            assert 0 <= source.x < image.shape[1] and 0 <= source.y < image.shape[0]
            assert "tile" in source.meta

    def test_tiled_agrees_with_whole_image(self, runs):
        """Same detections, same positions, fluxes within the per-tile
        background difference. The measured numbers are in docs/validation.md."""
        _, _, whole, tiled = runs
        pairs = self._pairs(whole, tiled.catalog)
        assert len(pairs) >= 0.95 * min(len(whole), len(tiled.catalog))
        ratio = np.array([tiled.catalog[j].photometry.flux / whole[i].photometry.flux
                          for i, j in pairs if whole[i].photometry.snr > 20])
        assert abs(float(np.median(ratio)) - 1.0) < 0.02
        assert float(np.median(np.abs(ratio - 1.0))) < 0.03

    def test_accounting_is_recorded(self, runs):
        _, _, _, tiled = runs
        payload = tiled.to_dict()
        assert payload["n_tiles"] == len(tiled.tiles) == 9
        assert payload["n_before_merge"] > payload["n_sources"]
        assert payload["peak_tile_pixels"] < 512 * 512
        assert len(payload["per_tile"]) == 9

    def test_the_shared_psf_is_used_everywhere(self, runs):
        image, *_ = runs
        stage = standard_stage(psf="shared")
        result = process_tiled(image, stage, tile=256, overlap=96)
        assert stage.shared_psf is not None
        from astrovision.photometry import Photometer

        # Every source's correction is the one the shared PSF gives at its
        # radius, whichever tile measured it.
        checked = 0
        for source in result.catalog:
            if "aperture_correction" not in source.meta:
                continue
            expected = Photometer.aperture_correction(stage.shared_psf,
                                                      source.photometry.aperture_radius)
            assert source.meta["aperture_correction"] == pytest.approx(expected, rel=1e-9)
            checked += 1
        assert checked > 50
        assert len({s.meta["tile"] for s in result.catalog}) == 9
