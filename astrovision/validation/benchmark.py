"""Agreement with the tools astronomers already use.

Every number in this package's validation was measured against a simulator's
truth. That proves the code recovers what it claims to measure; it does not
prove it agrees with what the field would have measured on the same pixels.
Those are different questions, and a working astronomer asks the second one
first: *does this give the same catalog as SExtractor would?*

This module answers it by running two field-standard tools on the same image
and comparing, object by object:

* **photutils** -- the Astropy-affiliated package, used here for background
  estimation, segmentation-based detection and aperture photometry;
* **SEP** -- the Source Extractor algorithms as a library, which is the
  closest available thing to running SExtractor itself.

Both are optional (``pip install 'astrovision-x[benchmark]'``), and the
comparison is deliberately asymmetric in its reading. When the tools agree
with each other and this package disagrees with both, that is a bug here
until proven otherwise. When this package agrees with the truth and the tools
do not, that is worth stating -- but it is not the default assumption, and
the report carries all three so the reader can decide.

What is compared, and what is not: positions and aperture fluxes in a fixed
radius, because those are defined identically everywhere. Kron or "auto"
fluxes are not, since each tool draws its own aperture, and comparing them
measures the aperture conventions rather than the photometry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.numeric import as_float_image

log = get_logger("validation.benchmark")


def available_tools() -> Dict[str, bool]:
    """Which external tools can be run in this environment."""
    return {"photutils": try_import("photutils") is not None,
            "sep": try_import("sep") is not None}


@dataclass
class ToolCatalog:
    """A detection list from one tool, in a common shape."""

    tool: str
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    seconds: float = float("nan")
    n: int = 0
    notes: List[str] = field(default_factory=list)


def run_photutils(image: np.ndarray, threshold_sigma: float = 3.5,
                  npixels: int = 5, aperture_radius: float = 5.0,
                  filter_fwhm: float = 3.0, mask: Optional[np.ndarray] = None
                  ) -> ToolCatalog:
    """Detect and measure with photutils, the way its documentation does it.

    Background2D for the sky, a matched-filter convolution, segmentation with
    deblending, then circular aperture photometry at the segment centroids.
    The parameters mirror this package's defaults so the comparison is of the
    implementations, not the settings.
    """
    photutils = try_import("photutils")
    if photutils is None:
        raise ImportError("photutils is not installed; pip install 'astrovision-x[benchmark]'")
    from astropy.convolution import Gaussian2DKernel, convolve
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import deblend_sources, detect_sources

    data = as_float_image(image)
    started = time.time()
    background = Background2D(data, (64, 64), filter_size=(3, 3),
                              bkg_estimator=MedianBackground(), mask=mask)
    subtracted = data - background.background
    threshold = float(threshold_sigma) * background.background_rms
    sigma = float(filter_fwhm) / 2.3548
    kernel = Gaussian2DKernel(sigma, x_size=int(2 * round(3 * sigma) + 1),
                              y_size=int(2 * round(3 * sigma) + 1))
    convolved = convolve(subtracted, kernel, normalize_kernel=True)
    # photutils 3.0 renamed npixels/nlevels/xcentroid; support both.
    try:
        segments = detect_sources(convolved, threshold, n_pixels=int(npixels), mask=mask)
    except TypeError:
        segments = detect_sources(convolved, threshold, npixels=int(npixels), mask=mask)
    notes: List[str] = []
    if segments is None:
        return ToolCatalog("photutils", np.zeros(0), np.zeros(0), np.zeros(0),
                           time.time() - started, 0, ["no sources detected"])
    try:
        try:
            segments = deblend_sources(convolved, segments, n_pixels=int(npixels),
                                       n_levels=32, contrast=0.005, progress_bar=False)
        except TypeError:
            segments = deblend_sources(convolved, segments, npixels=int(npixels),
                                       nlevels=32, contrast=0.005, progress_bar=False)
    except Exception as error:                          # pragma: no cover
        notes.append(f"deblending failed: {error}")

    from photutils.segmentation import SourceCatalog
    catalog = SourceCatalog(subtracted, segments, convolved_data=convolved)
    x = np.asarray(getattr(catalog, "x_centroid", None)
                   if hasattr(catalog, "x_centroid") else catalog.xcentroid, dtype=float)
    y = np.asarray(getattr(catalog, "y_centroid", None)
                   if hasattr(catalog, "y_centroid") else catalog.ycentroid, dtype=float)
    apertures = CircularAperture(np.column_stack([x, y]), r=float(aperture_radius))
    table = aperture_photometry(subtracted, apertures, mask=mask)
    flux = np.asarray(table["aperture_sum"], dtype=float)
    return ToolCatalog("photutils", x, y, flux, time.time() - started, len(x), notes)


def run_sep(image: np.ndarray, threshold_sigma: float = 3.5, minarea: int = 5,
            aperture_radius: float = 5.0, filter_fwhm: float = 3.0,
            mask: Optional[np.ndarray] = None) -> ToolCatalog:
    """Detect and measure with SEP, which is Source Extractor as a library."""
    sep = try_import("sep")
    if sep is None:
        raise ImportError("sep is not installed; pip install 'astrovision-x[benchmark]'")
    data = np.ascontiguousarray(as_float_image(image), dtype=np.float32)
    started = time.time()
    background = sep.Background(data, mask=mask, bw=64, bh=64, fw=3, fh=3)
    subtracted = data - background
    sigma = float(filter_fwhm) / 2.3548
    half = int(round(3 * sigma))
    grid = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (grid[:, None] ** 2 + grid[None, :] ** 2) / sigma ** 2)
    objects = sep.extract(subtracted, float(threshold_sigma),
                          err=background.globalrms, minarea=int(minarea),
                          filter_kernel=kernel.astype(np.float32),
                          deblend_nthresh=32, deblend_cont=0.005, mask=mask)
    x = np.asarray(objects["x"], dtype=float)
    y = np.asarray(objects["y"], dtype=float)
    flux, _, _ = sep.sum_circle(subtracted, x, y, float(aperture_radius),
                                err=background.globalrms, mask=mask)
    return ToolCatalog("sep", x, y, np.asarray(flux, dtype=float),
                       time.time() - started, len(x))


def _match(x1, y1, x2, y2, radius: float) -> List[Tuple[int, int]]:
    """Mutual nearest-neighbour pairs within ``radius`` pixels."""
    if len(x1) == 0 or len(x2) == 0:
        return []
    a = np.column_stack([x1, y1])
    b = np.column_stack([x2, y2])
    distance = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    forward = np.argmin(distance, axis=1)
    backward = np.argmin(distance, axis=0)
    pairs = []
    for i, j in enumerate(forward):
        if backward[j] == i and distance[i, j] <= radius:
            pairs.append((int(i), int(j)))
    return pairs


@dataclass
class BenchmarkResult:
    """This package against one external tool on one image."""

    tool: str = ""
    n_ours: int = 0
    n_theirs: int = 0
    n_matched: int = 0
    match_radius: float = 2.0
    position_offset_median: float = float("nan")    # pixels
    flux_ratio_median: float = float("nan")         # ours / theirs
    flux_ratio_scatter: float = float("nan")        # robust, dex-free
    only_ours: int = 0
    only_theirs: int = 0
    seconds_ours: float = float("nan")
    seconds_theirs: float = float("nan")
    against_truth: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def matched_fraction(self) -> float:
        base = max(min(self.n_ours, self.n_theirs), 1)
        return float(self.n_matched / base)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["matched_fraction"] = self.matched_fraction
        return payload

    def summary(self) -> str:
        return (f"{self.tool}: {self.n_matched} matched of {self.n_ours} ours / "
                f"{self.n_theirs} theirs ({100 * self.matched_fraction:.0f}%), "
                f"positions agree to {self.position_offset_median:.2f} px, "
                f"flux ratio {self.flux_ratio_median:.3f} "
                f"+/- {self.flux_ratio_scatter:.3f}")


def compare(ours_x, ours_y, ours_flux, theirs: ToolCatalog,
            radius: float = 2.0, min_flux: float = 0.0) -> BenchmarkResult:
    """Match two catalogs and summarise where they agree and where they do not."""
    result = BenchmarkResult(tool=theirs.tool, n_ours=len(ours_x),
                             n_theirs=theirs.n, match_radius=float(radius),
                             seconds_theirs=theirs.seconds, notes=list(theirs.notes))
    pairs = _match(np.asarray(ours_x), np.asarray(ours_y), theirs.x, theirs.y, radius)
    result.n_matched = len(pairs)
    result.only_ours = len(ours_x) - len(pairs)
    result.only_theirs = theirs.n - len(pairs)
    if not pairs:
        result.notes.append("no matches within the radius")
        return result
    ours_index = np.array([i for i, _ in pairs])
    theirs_index = np.array([j for _, j in pairs])
    offsets = np.hypot(np.asarray(ours_x)[ours_index] - theirs.x[theirs_index],
                       np.asarray(ours_y)[ours_index] - theirs.y[theirs_index])
    result.position_offset_median = float(np.median(offsets))

    our_flux = np.asarray(ours_flux, dtype=float)[ours_index]
    their_flux = theirs.flux[theirs_index]
    usable = np.isfinite(our_flux) & np.isfinite(their_flux) \
        & (their_flux > float(min_flux)) & (our_flux > 0)
    if usable.sum() >= 3:
        ratio = our_flux[usable] / their_flux[usable]
        result.flux_ratio_median = float(np.median(ratio))
        result.flux_ratio_scatter = float(
            1.4826 * np.median(np.abs(ratio - np.median(ratio))))
    else:
        result.notes.append("too few matched fluxes to compare")
    return result


def benchmark_field(image, catalog, truth: Optional[Sequence[Any]] = None,
                    tools: Sequence[str] = ("photutils", "sep"),
                    aperture_radius: float = 5.0, threshold_sigma: float = 3.5,
                    match_radius: float = 2.0,
                    seconds_ours: float = float("nan")) -> List[BenchmarkResult]:
    """Compare this package's catalog on ``image`` with each available tool.

    ``catalog`` is this package's :class:`SourceCatalog` already measured on
    the image; the aperture flux compared is the one at ``aperture_radius``,
    which must be among the radii the photometer measured. ``truth`` is the
    simulator's table when there is one, in which case every tool -- this
    package included -- is also scored against it, so the report can say not
    just whether two codes agree but which one is right when they do not.
    """
    data = as_float_image(getattr(image, "data", image))
    mask = getattr(image, "mask", None)
    key = f"{aperture_radius:g}"
    ours_x = np.array([s.x for s in catalog], dtype=float)
    ours_y = np.array([s.y for s in catalog], dtype=float)
    ours_flux = np.array([
        float(s.meta.get("apertures", {}).get(key, {}).get("flux", np.nan))
        for s in catalog], dtype=float)
    if not np.isfinite(ours_flux).any():
        ours_flux = np.array([float(getattr(getattr(s, "photometry", None),
                                            "flux", np.nan)) for s in catalog])

    results: List[BenchmarkResult] = []
    runners = {"photutils": run_photutils, "sep": run_sep}
    for name in tools:
        runner = runners.get(name)
        if runner is None or not available_tools().get(name, False):
            log.warning("benchmark tool %s is not available", name)
            continue
        theirs = runner(data, threshold_sigma=threshold_sigma,
                        aperture_radius=aperture_radius, mask=mask)
        result = compare(ours_x, ours_y, ours_flux, theirs, radius=match_radius)
        result.seconds_ours = seconds_ours
        if truth is not None:
            result.against_truth = {
                "ours": _score_against_truth(ours_x, ours_y, ours_flux, truth,
                                             match_radius),
                theirs.tool: _score_against_truth(theirs.x, theirs.y, theirs.flux,
                                                  truth, match_radius)}
        results.append(result)
        log.info(result.summary())
    return results


def _score_against_truth(x, y, flux, truth: Sequence[Any], radius: float
                         ) -> Dict[str, Any]:
    """Recall and flux ratio against a simulator's truth table."""
    tx = np.array([float(getattr(t, "x", np.nan)) for t in truth])
    ty = np.array([float(getattr(t, "y", np.nan)) for t in truth])
    tf = np.array([float(getattr(t, "flux", np.nan)) for t in truth])
    keep = np.isfinite(tx) & np.isfinite(ty)
    tx, ty, tf = tx[keep], ty[keep], tf[keep]
    pairs = _match(np.asarray(x), np.asarray(y), tx, ty, radius)
    if not pairs or len(tx) == 0:
        return {"recall": 0.0, "n_truth": int(len(tx)), "n_matched": 0}
    ours = np.array([i for i, _ in pairs])
    theirs = np.array([j for _, j in pairs])
    ratio = np.asarray(flux, dtype=float)[ours] / tf[theirs]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return {"recall": float(len(pairs) / len(tx)), "n_truth": int(len(tx)),
            "n_matched": int(len(pairs)),
            "spurious": int(len(x) - len(pairs)),
            "flux_ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
            "flux_ratio_scatter": float(1.4826 * np.median(np.abs(ratio - np.median(ratio))))
            if ratio.size else float("nan")}
