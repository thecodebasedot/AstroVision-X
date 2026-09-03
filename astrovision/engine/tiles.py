"""Processing an image too large to process.

A survey frame is sixteen thousand pixels on a side. The detection and
photometry stages here were written for a field that fits in memory several
times over -- a filtered copy, a segmentation map, a background model, a
noise map -- and on a gigapixel they do not. So the frame is cut into tiles
that do, each is processed as though it were a field of its own, and the
catalogs are merged.

The merge is where the mistakes live, and each one is addressed by name:

* **Tiles overlap**, by more than the largest object expected, so no source
  is ever cut in half at a boundary. A source near a tile edge is measured
  whole in *some* tile.
* **Every source in an overlap is found twice**, and the merge keeps the copy
  that was measured farther from its tile's edge -- the one whose aperture
  and neighbours were fully inside the tile -- and drops the other. Keeping
  the first-seen copy instead would keep the truncated one half the time.
* **Background is estimated per tile**, which is the point: a 16k frame has
  sky structure no single global fit follows. What it costs is a small step
  between tiles, and the overlap is what lets the merge ignore it.
* **Positions are returned in the frame's own pixel coordinates**, with the
  tile's origin added back, and every source records which tile measured it.

The honest measurement is the whole-image catalog against the tiled one on a
frame small enough to do both: the numbers are in `docs/validation.md`, and
the residual differences are the per-tile backgrounds, not the merge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import Source, SourceCatalog
from ..io.image import AstroImage

log = get_logger("engine.tiles")


@dataclass
class Tile:
    """One rectangular piece of a larger frame, with its overlap."""

    index: int
    row0: int
    row1: int
    col0: int
    col1: int
    #: The part of this tile that is *not* overlap with a neighbour -- the
    #: region this tile is responsible for in the merge.
    core_row0: int = 0
    core_row1: int = 0
    core_col0: int = 0
    core_col1: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row1 - self.row0, self.col1 - self.col0)

    def distance_to_edge(self, x: float, y: float) -> float:
        """Pixels from a frame position to the nearest edge of this tile."""
        return float(min(x - self.col0, self.col1 - 1 - x,
                         y - self.row0, self.row1 - 1 - y))

    def contains_core(self, x: float, y: float) -> bool:
        return (self.core_col0 <= x < self.core_col1
                and self.core_row0 <= y < self.core_row1)


def _starts(length: int, tile: int, overlap: int) -> Tuple[List[int], int]:
    """Equal-length tiles covering ``length`` with at least ``overlap``.

    Never a thin remainder: a 160-pixel strip left over from a 384-pixel
    tiling has a background mesh and a PSF star count unlike every other
    tile's, and the fluxes measured in it were 6% off the rest of the frame.
    Instead the tile count is fixed by the requested size and the tiles are
    stretched evenly to fit.
    """
    if length <= tile:
        return [0], length
    step = tile - overlap
    count = int(np.ceil((length - overlap) / step))
    size = int(np.ceil((length + (count - 1) * overlap) / count))
    starts = [int(round(index * (length - size) / (count - 1))) for index in range(count)]
    return starts, size


def plan_tiles(shape: Tuple[int, int], tile: int = 2048, overlap: int = 128
               ) -> List[Tile]:
    """Cut a frame into overlapping tiles of equal size.

    ``overlap`` should exceed the largest object expected plus the largest
    photometry aperture; 128 pixels covers a galaxy of a hundred pixels with
    a 12-pixel annulus around it. The tiles are as close to ``tile`` as the
    frame allows while staying equal, so no tile is a thin strip.

    >>> tiles = plan_tiles((500, 500), tile=300, overlap=50)
    >>> len(tiles), tiles[0].shape, tiles[-1].shape
    (4, (275, 275), (275, 275))
    >>> tiles[1].col0, tiles[0].col1     # they overlap by 50
    (225, 275)
    """
    height, width = int(shape[0]), int(shape[1])
    overlap = max(int(overlap), 0)
    tile = max(int(tile), 2 * overlap + 1)
    rows, tile_height = _starts(height, tile, overlap)
    cols, tile_width = _starts(width, tile, overlap)
    # Core boundaries sit halfway through each actual overlap, so adjacent
    # cores tile the frame exactly once with no gap and no double coverage.
    row_edges = [0] + [(rows[i + 1] + rows[i] + tile_height) // 2
                       for i in range(len(rows) - 1)] + [height]
    col_edges = [0] + [(cols[i + 1] + cols[i] + tile_width) // 2
                       for i in range(len(cols) - 1)] + [width]
    tiles: List[Tile] = []
    for r_index, row0 in enumerate(rows):
        for c_index, col0 in enumerate(cols):
            row1, col1 = min(row0 + tile_height, height), min(col0 + tile_width, width)
            tiles.append(Tile(index=len(tiles), row0=row0, row1=row1, col0=col0, col1=col1,
                              core_row0=row_edges[r_index], core_row1=row_edges[r_index + 1],
                              core_col0=col_edges[c_index], core_col1=col_edges[c_index + 1]))
    return tiles


def cut(image: AstroImage, tile: Tile) -> AstroImage:
    """The sub-image for one tile, carrying its share of every plane."""
    rows, cols = slice(tile.row0, tile.row1), slice(tile.col0, tile.col1)
    piece = AstroImage(
        data=np.array(image.data[rows, cols], dtype=float),
        header=dict(image.header), wcs=image.wcs,
        mask=None if image.mask is None else image.mask[rows, cols].copy(),
        uncertainty=(None if image.uncertainty is None
                     else np.array(image.uncertainty[rows, cols], dtype=float)),
        name=f"{image.name}[tile {tile.index}]", band=image.band,
        mjd=image.mjd, exposure_time=image.exposure_time)
    piece.meta = {**image.meta, "tile": tile.index,
                  "tile_origin": (tile.col0, tile.row0)}
    return piece


@dataclass
class TiledResult:
    """A merged catalog and the accounting behind it."""

    catalog: SourceCatalog
    tiles: List[Tile]
    n_before_merge: int = 0
    n_duplicates_removed: int = 0
    seconds: float = float("nan")
    peak_tile_pixels: int = 0
    per_tile: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_tiles": len(self.tiles), "n_sources": len(self.catalog),
                "n_before_merge": self.n_before_merge,
                "n_duplicates_removed": self.n_duplicates_removed,
                "seconds": self.seconds, "peak_tile_pixels": self.peak_tile_pixels,
                "per_tile": list(self.per_tile)}


def _offset_source(source: Source, tile: Tile) -> Source:
    """Move a source from tile pixels to frame pixels, in place."""
    source.x += tile.col0
    source.y += tile.row0
    box = getattr(source, "bbox", None)
    if box is not None:
        box.x_min += tile.col0
        box.x_max += tile.col0
        box.y_min += tile.row0
        box.y_max += tile.row0
    source.meta["tile"] = tile.index
    source.meta["tile_edge_distance"] = tile.distance_to_edge(source.x, source.y)
    return source


def merge_catalogs(per_tile: Sequence[Tuple[Tile, SourceCatalog]],
                   match_radius: float = 2.0) -> Tuple[SourceCatalog, int]:
    """Combine tile catalogs into one frame catalog.

    A source is kept from the tile whose *core* contains it. The cores
    partition the frame, so every position belongs to exactly one tile, and
    a core lies half an overlap inside its tile's edge, so the kept copy was
    measured with its whole aperture and annulus inside the tile. The copy a
    neighbouring tile made of the same object -- often a truncated fragment
    whose centroid is pulled a few pixels toward the tile edge, which is why
    a nearest-neighbour merge cannot be trusted to catch it -- falls outside
    that tile's core and is dropped without being matched at all.

    What remains for ``match_radius`` is the object sitting on a core
    boundary itself, whose two centroids land on either side of it; those
    are matched and the copy farther from its tile's edge is kept.
    """
    entries: List[Tuple[Source, float]] = []
    removed = 0
    for tile, catalog in per_tile:
        for source in catalog:
            if tile.contains_core(source.x, source.y):
                entries.append((source, float(source.meta.get("tile_edge_distance", 0.0))))
            else:
                removed += 1
    if not entries:
        return SourceCatalog(), removed

    # Best copies first, so a source is kept before any worse copy of it is
    # examined and dropped.
    entries.sort(key=lambda item: -item[1])
    positions = np.array([[s.x, s.y] for s, _ in entries], dtype=float)
    kept: List[Source] = []
    kept_positions: List[np.ndarray] = []
    radius2 = float(match_radius) ** 2
    for (source, _), position in zip(entries, positions):
        duplicate = False
        if kept_positions:
            stack = np.asarray(kept_positions)
            close = np.flatnonzero(((stack - position) ** 2).sum(axis=1) <= radius2)
            for index in close:
                if kept[int(index)].meta.get("tile") != source.meta.get("tile"):
                    duplicate = True
                    break
        if duplicate:
            removed += 1
            continue
        kept.append(source)
        kept_positions.append(position)

    merged = SourceCatalog()
    for new_id, source in enumerate(sorted(kept, key=lambda s: (s.y, s.x)), start=1):
        source.id = new_id
        merged.append(source)
    return merged, removed


def process_tiled(image: AstroImage,
                  stage: Callable[[AstroImage], Tuple[SourceCatalog, Any]],
                  tile: int = 2048, overlap: int = 128,
                  match_radius: float = 2.0,
                  progress: Optional[Callable[[int, int], None]] = None
                  ) -> TiledResult:
    """Run a detect-and-measure stage tile by tile and merge the results.

    ``stage`` takes a sub-image and returns ``(catalog, segmentation)``; the
    segmentation is discarded, since a frame-sized label map is exactly the
    array this exists to avoid holding. Positions come back in frame
    coordinates.
    """
    started = time.time()
    tiles = plan_tiles(image.shape, tile=tile, overlap=overlap)
    share = getattr(stage, "share_psf_from", None)
    if share is not None and getattr(stage, "psf_mode", None) is not None:
        share(cut(image, tiles[len(tiles) // 2]))
    per_tile: List[Tuple[Tile, SourceCatalog]] = []
    accounting: List[Dict[str, Any]] = []
    peak = 0
    for piece in tiles:
        sub = cut(image, piece)
        peak = max(peak, sub.size)
        catalog, _ = stage(sub)
        for source in catalog:
            _offset_source(source, piece)
        per_tile.append((piece, catalog))
        accounting.append({"tile": piece.index, "shape": piece.shape,
                           "n_sources": len(catalog)})
        if progress is not None:
            progress(piece.index + 1, len(tiles))
    before = sum(len(c) for _, c in per_tile)
    merged, removed = merge_catalogs(per_tile, match_radius=match_radius)
    result = TiledResult(catalog=merged, tiles=tiles, n_before_merge=before,
                         n_duplicates_removed=removed,
                         seconds=time.time() - started, peak_tile_pixels=peak,
                         per_tile=accounting)
    log.info("tiled %s: %d tiles, %d sources after removing %d duplicates in %.1fs",
             image.name, len(tiles), len(merged), removed, result.seconds)
    return result


#: Fewer PSF stars than this and a tile's own PSF model is not trusted.
MIN_PSF_STARS = 8


class StandardStage:
    """The detect-and-measure stage the pipeline runs, one tile at a time.

    Preprocessing is done per tile -- the background is the reason, see the
    module docstring -- then detection and photometry exactly as the
    single-image pipeline would run them.

    The PSF is the exception to "per tile". The aperture correction is
    ``1 / (PSF flux inside the aperture)``, and a PSF built from the four
    stars a small tile happens to hold differs from the next tile's by
    several percent, so fluxes would step at every tile boundary. With
    ``psf="shared"`` (the default) one PSF is built from a central tile and
    used everywhere; with ``psf="per-tile"`` each tile fits its own and falls
    back to the shared one when it has fewer than :data:`MIN_PSF_STARS`.
    A PSF that genuinely varies across a mosaic is not modelled in tiled
    mode; the spatially varying fit runs on whole frames only.
    """

    def __init__(self, config: Any = None, preprocess: bool = True,
                 psf: Optional[str] = "shared"):
        from ..core.config import AstroVisionConfig
        from ..detect import Detector
        from ..photometry import Photometer
        from ..preprocess import Preprocessor

        if psf not in ("shared", "per-tile", None):
            raise ValueError("psf must be 'shared', 'per-tile' or None")
        self.config = config or AstroVisionConfig()
        self.preprocess = preprocess
        self.psf_mode = psf
        self.preprocessor = Preprocessor(self.config.preprocess)
        self.detector = Detector(self.config.detection)
        self.photometer = Photometer(self.config.photometry)
        self.shared_psf: Any = None

    def share_psf_from(self, sub: AstroImage) -> Any:
        """Build the shared PSF model from one representative sub-image."""
        if not self.preprocess:
            self.shared_psf = sub.meta.get("psf_model")
        else:
            clean = self.preprocessor.run(sub, estimate_psf=True)
            self.shared_psf = clean.meta.get("psf_model")
        return self.shared_psf

    def __call__(self, sub: AstroImage) -> Tuple[SourceCatalog, Any]:
        # A tile fits its own PSF in per-tile mode, and in shared mode only
        # when nothing has been shared yet -- which is the case when the stage
        # is called directly on a whole frame rather than through the tiler.
        own = self.psf_mode == "per-tile" or (self.psf_mode == "shared"
                                              and self.shared_psf is None)
        clean = (self.preprocessor.run(sub, estimate_psf=own)
                 if self.preprocess else sub)
        model = clean.meta.get("psf_model") if own else None
        if self.psf_mode is None:
            clean.meta.pop("psf_model", None)
        elif model is None or getattr(model, "n_stars", 0) < MIN_PSF_STARS:
            if self.shared_psf is not None:
                clean.meta["psf_model"] = self.shared_psf
                clean.meta["psf"] = self.shared_psf.to_dict()
                clean.meta["psf_source"] = "shared"
        catalog, segmentation = self.detector.detect(clean)
        if len(catalog):
            self.photometer.run(clean, catalog, segmentation)
        return catalog, segmentation


def standard_stage(config: Any = None, preprocess: bool = True,
                   psf: Optional[str] = "shared") -> StandardStage:
    """A :class:`StandardStage`; see there for the PSF choice."""
    return StandardStage(config, preprocess=preprocess, psf=psf)


def iter_tiles(image: AstroImage, tile: int = 2048, overlap: int = 128
               ) -> Iterator[Tuple[Tile, AstroImage]]:
    """Yield ``(tile, sub-image)`` pairs, for callers that want the loop."""
    for piece in plan_tiles(image.shape, tile=tile, overlap=overlap):
        yield piece, cut(image, piece)
