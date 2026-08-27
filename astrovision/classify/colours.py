"""Colour-based star/galaxy separation: fitting the field's own stellar locus.

Morphology answers "is this resolved?", and it answers it well when the
object is bright.  It stops working exactly where survey catalogs get
interesting -- near the detection limit, where a small galaxy and a star are
both a few pixels of noise-dominated light and every size statistic collapses
onto the same value.

Colour does not degrade the same way, because it is not a size measurement.
Stars are, to first order, blackbodies behind the same filters, so their
colours fall on a one-parameter curve -- the **stellar locus**.  Galaxies are
integrated, redshifted stellar populations and sit off it.  A faint object
whose morphology is uninformative can still be half a magnitude off the
locus, and that is a real measurement.

The locus is fitted *from the field itself* rather than taken from a table.
That makes the test independent of the zero point, of the filter's exact
throughput, and of any reddening common to the field -- all of which move
the locus bodily without changing the fact that stars lie on it.

The *widths* are measured from the field too, and for a harder reason.  The
test compares how far a source sits off the locus against how far it could
sit off it by chance, so it needs to know that second number.  Formal
photometric errors are not it: measured against simulated truth they come
out around 2.5 times too small, because they count photon and read noise but
not sky estimation, blending, or the residual of matching one band's PSF to
another.  Calibrating both widths -- the stars' and the galaxies' -- from
the field's own populations sidesteps all of that, and has the property that
matters most: when the two widths come out equal, the test says 0.5 and
contributes nothing, instead of confidently reporting noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import Source, SourceCatalog

log = get_logger("classify.colours")

#: Colour pairs to fit, in preference order.  Each entry is the three bands
#: making the two colours ``(b0-b1, b1-b2)``.  The first triple that is fully
#: measured in the catalog is used.
#: Fraction of sources whose colours are assumed contaminated -- blends,
#: neighbours in the aperture, PSF-matching residuals.  It is what gives the
#: likelihood its tails; see :func:`colour_stellarity`.
OUTLIER_FRACTION = 0.05

#: Ceiling on how confident the colour test alone is allowed to be.
CONFIDENCE_CAP = 0.92

#: Separation (ROC area) below which colour is given no weight at all.
SEPARATION_FLOOR = 0.58

PREFERRED_TRIPLES: Sequence[Tuple[str, str, str]] = (
    ("g", "r", "i"),
    ("u", "g", "r"),
    ("r", "i", "z"),
    ("g", "r", "z"),
)


@dataclass
class StellarLocus:
    """A curve fitted through the point-source colours of one field.

    Alongside the curve it carries the two *offset widths* the classification
    test needs: how far point sources scatter from it, and how far resolved
    ones do, both as a function of signal-to-noise.
    """

    bands: Tuple[str, str, str]
    coefficients: np.ndarray                # polynomial y(x), highest power first
    x_range: Tuple[float, float]
    scatter: float                          # rms radial residual of the fit stars
    n_stars: int
    snr_bins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_width: np.ndarray = field(default_factory=lambda: np.zeros(0))
    galaxy_width: np.ndarray = field(default_factory=lambda: np.zeros(0))
    separation: float = 0.5                 # measured AUC of the colour test
    n_galaxies: int = 0
    warnings: List[str] = field(default_factory=list)

    def predict(self, x) -> np.ndarray:
        return np.polyval(self.coefficients, np.asarray(x, dtype=float))

    def distance(self, x: float, y: float, samples: int = 128) -> float:
        """Perpendicular distance from ``(x, y)`` to the locus, in magnitudes.

        Perpendicular rather than vertical: the locus is steep at the red end,
        where a vertical residual would report a large distance for a point
        sitting right on the curve.
        """
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("nan")
        # Sampling a little beyond the fitted range keeps objects just off
        # the end from being assigned to the endpoint by construction.
        span = self.x_range[1] - self.x_range[0]
        low = self.x_range[0] - 0.15 * span
        high = self.x_range[1] + 0.15 * span
        grid = np.linspace(low, high, samples)
        curve = self.predict(grid)
        return float(np.min(np.hypot(grid - float(x), curve - float(y))))

    @property
    def informative(self) -> bool:
        """Whether the two populations are separated enough to be worth using."""
        if self.star_width.size == 0 or self.galaxy_width.size == 0:
            return False
        return bool(np.any(self.galaxy_width > 1.15 * self.star_width))

    def widths(self, snr: float) -> Tuple[float, float]:
        """``(star, galaxy)`` offset widths at this signal-to-noise.

        Interpolated in log signal-to-noise, and ``(nan, nan)`` below the
        calibrated range.  Holding the width flat at the faint end instead
        looks harmless and is not: a source fainter than anything the
        calibration saw gets judged against a width measured on brighter
        objects, so its ordinary colour error reads as a large offset and the
        test confidently calls it resolved.  Measured against truth, faint
        sources mislabelled this way were the only harm the colour test did.

        Brighter than the calibrated range is safe to hold flat: the widths
        are still shrinking there, so the faintest-bin value is conservative.
        """
        if self.star_width.size == 0 or self.galaxy_width.size == 0:
            return float("nan"), float("nan")
        if not np.isfinite(snr) or snr <= 0:
            return float("nan"), float("nan")
        key = float(np.log10(snr))
        # A quarter of a decade of slack: bin centres sit inside their bins,
        # so a source a little fainter than the lowest centre is still within
        # the population that was actually measured.
        if key < float(self.snr_bins[0]) - 0.25:
            return float("nan"), float("nan")
        star = float(np.interp(key, self.snr_bins, self.star_width))
        galaxy = float(np.interp(key, self.snr_bins, self.galaxy_width))
        return star, max(galaxy, star)

    @property
    def information_weight(self) -> float:
        """How much the fusion should trust this field's colour test, in [0, 1].

        Derived from :attr:`separation`, the measured area under the ROC
        curve of the colour test against the morphological labels.  A test
        that separates the populations no better than chance (0.5) gets zero
        weight; one that separates them cleanly (0.85 and above) gets full
        weight.  Fixing this as a constant instead is what turns a capability
        into a liability: a weak, noisy vote added at full strength to a
        strong one costs accuracy in every field where the colours happen to
        be poor, and no single constant can be right for both a shallow
        two-band field and a deep five-band one.
        """
        if not self.informative:
            return 0.0
        # The floor sits above 0.5, not at it: an AUC a little over chance is
        # what a test with no power produces on a finite field about half the
        # time, and acting on it costs accuracy without ever repaying it.
        span = (float(self.separation) - SEPARATION_FLOOR) / 0.30
        return float(np.clip(span, 0.0, 1.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "bands": list(self.bands),
            "colours": [f"{self.bands[0]}-{self.bands[1]}",
                        f"{self.bands[1]}-{self.bands[2]}"],
            "coefficients": [float(c) for c in self.coefficients],
            "x_range": [float(self.x_range[0]), float(self.x_range[1])],
            "scatter": float(self.scatter),
            "n_stars": int(self.n_stars),
            "n_galaxies": int(self.n_galaxies),
            "snr_bins": [float(v) for v in self.snr_bins],
            "star_width": [float(v) for v in self.star_width],
            "galaxy_width": [float(v) for v in self.galaxy_width],
            "informative": self.informative,
            "separation": float(self.separation),
            "information_weight": float(self.information_weight),
            "warnings": list(self.warnings),
        }


def available_triple(catalog: SourceCatalog,
                     triples: Sequence[Tuple[str, str, str]] = PREFERRED_TRIPLES
                     ) -> Optional[Tuple[str, str, str]]:
    """The first colour triple this catalog actually has photometry for."""
    measured: Dict[str, int] = {}
    for source in catalog:
        for band, photometry in source.bands.items():
            if np.isfinite(photometry.magnitude):
                measured[band] = measured.get(band, 0) + 1
    for triple in triples:
        if all(measured.get(band, 0) >= 8 for band in triple):
            return triple
    return None


def colour_pair(source: Source, triple: Tuple[str, str, str]) -> Tuple[float, float]:
    """The two colours ``(b0-b1, b1-b2)`` for one source, using stored values.

    Reads ``meta["colours"]`` when present so that the signal-to-noise cut
    applied by :func:`~astrovision.photometry.multiband.measure_colours`
    is respected -- a colour rejected there must not reappear here.
    """
    colours = source.meta.get("colours") or {}
    first = colours.get(f"{triple[0]}-{triple[1]}")
    second = colours.get(f"{triple[1]}-{triple[2]}")
    if first is None:
        first = source.colour(triple[0], triple[1])
    if second is None:
        second = source.colour(triple[1], triple[2])
    return float(first), float(second)


def rayleigh_scale(distances: Sequence[float], trim: float = 90.0) -> float:
    """Per-axis width of a two-dimensional offset distribution.

    For an isotropic Gaussian offset the radial distance is Rayleigh
    distributed with ``E[d^2] = 2 sigma^2``, so the width is
    ``sqrt(mean(d^2) / 2)``.  The factor of two is not a detail: dropping it
    inflates every width by 40%, and the test's whole output is a ratio of
    two of them.

    The mean is taken over a trimmed sample, because one blended source with
    a wild colour would otherwise set the width for a whole population.

    >>> values = [0.1, 0.12, 0.09, 0.11, 5.0]
    >>> round(rayleigh_scale(values), 3)
    0.075
    """
    d = np.asarray([v for v in distances if np.isfinite(v)], dtype=float)
    if d.size == 0:
        return float("nan")
    if d.size >= 5:
        d = d[d <= np.percentile(d, float(trim))]
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(float(np.mean(d ** 2)) / 2.0))


def _population_widths(catalog: SourceCatalog, locus: StellarLocus,
                       n_bins: int = 3, min_per_bin: int = 6) -> None:
    """Calibrate the star and galaxy offset widths against signal-to-noise.

    Both populations scatter further off the locus as they get fainter, but
    only the galaxies keep a floor when the noise vanishes -- that floor is
    the physical difference the test is trying to detect.  Binning in
    signal-to-noise is what lets a bright source be judged against bright
    scatter instead of the field average.
    """
    stars: List[Tuple[float, float]] = []
    galaxies: List[Tuple[float, float]] = []
    for source in catalog:
        stellarity = source.class_scores.get("shape_stellarity", float("nan"))
        if not np.isfinite(stellarity):
            continue
        x, y = colour_pair(source, locus.bands)
        distance = locus.distance(x, y)
        snr = source.photometry.snr
        if not (np.isfinite(distance) and np.isfinite(snr) and snr > 0):
            continue
        entry = (float(np.log10(snr)), distance)
        if stellarity >= 0.6:
            stars.append(entry)
        elif stellarity <= 0.3:
            galaxies.append(entry)

    locus.n_galaxies = len(galaxies)
    if len(stars) < min_per_bin or len(galaxies) < min_per_bin:
        locus.warnings.append(
            "too few sources to calibrate colour widths; colour will not be used")
        return

    # Bin edges come from the stars, which are the more numerous population
    # and the one whose width varies fastest with brightness.
    star_snr = np.array([s[0] for s in stars])
    edges = np.percentile(star_snr, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6
    centres, star_widths, galaxy_widths = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_star = [d for v, d in stars if low < v <= high]
        in_galaxy = [d for v, d in galaxies if low < v <= high]
        if len(in_star) < min_per_bin:
            continue
        star_width = rayleigh_scale(in_star)
        # A bin with too few galaxies borrows the field-wide galaxy width
        # rather than being dropped: dropping it would silently narrow the
        # signal-to-noise range over which colour can be used at all.
        galaxy_width = (rayleigh_scale(in_galaxy) if len(in_galaxy) >= min_per_bin
                        else rayleigh_scale([d for _, d in galaxies]))
        if not (np.isfinite(star_width) and np.isfinite(galaxy_width)):
            continue
        centres.append(0.5 * (low + high))
        star_widths.append(star_width)
        galaxy_widths.append(max(galaxy_width, star_width))

    if not centres:
        locus.warnings.append("colour widths could not be calibrated")
        return
    order = np.argsort(centres)
    locus.snr_bins = np.asarray(centres, dtype=float)[order]
    locus.star_width = np.asarray(star_widths, dtype=float)[order]
    locus.galaxy_width = np.asarray(galaxy_widths, dtype=float)[order]
    if not locus.informative:
        locus.warnings.append(
            "galaxies scatter no further off the locus than stars do; "
            "colour carries no star/galaxy information in this field")


def _measure_separation(catalog: SourceCatalog, locus: StellarLocus) -> None:
    """Measure how well the colour test separates the morphological classes.

    The area under the ROC curve, computed the rank way (equivalent to the
    Mann-Whitney statistic) so it needs no threshold.  The morphological
    labels are imperfect, which biases the result *downward* -- the test is
    credited only for agreement with a noisy reference.  That is the safe
    direction: it under-trusts colour rather than over-trusting it.
    """
    stars, galaxies = [], []
    for source in catalog:
        stellarity = source.class_scores.get("shape_stellarity", float("nan"))
        if not np.isfinite(stellarity):
            continue
        value = colour_stellarity(source, locus)
        if not np.isfinite(value):
            continue
        if stellarity >= 0.6:
            stars.append(value)
        elif stellarity <= 0.3:
            galaxies.append(value)
    if len(stars) < 5 or len(galaxies) < 5:
        locus.separation = 0.5
        return
    values = np.concatenate([np.asarray(stars), np.asarray(galaxies)])
    ranks = _average_ranks(values)
    star_rank_sum = float(ranks[:len(stars)].sum())
    n_star, n_galaxy = len(stars), len(galaxies)
    auc = (star_rank_sum - n_star * (n_star + 1) / 2.0) / (n_star * n_galaxy)
    locus.separation = float(np.clip(auc, 0.0, 1.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values``, ties sharing the average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def fit_stellar_locus(catalog: SourceCatalog,
                      triple: Optional[Tuple[str, str, str]] = None,
                      degree: int = 3,
                      min_stars: int = 12,
                      clip_sigma: float = 2.5,
                      iterations: int = 4,
                      seed_key: str = "shape_stellarity") -> Optional[StellarLocus]:
    """Fit the stellar locus using the field's own compact sources.

    The *morphological* stellarity already on each source seeds the fit --
    ``shape_stellarity``, never the fused ``stellarity``, or the locus would
    be seeded by its own output and the test would confirm itself.  The fit
    is then iteratively sigma-clipped, which removes the galaxies the
    morphological cut let through.  Where no seed exists the fit falls back
    to every source with both colours, relying on stars outnumbering
    galaxies -- true in most fields, and recorded as a warning when used so
    the assumption is visible rather than buried.
    """
    triple = triple or available_triple(catalog)
    if triple is None:
        return None
    warnings: List[str] = []

    points: List[Tuple[float, float]] = []
    seeded = 0
    for source in catalog:
        x, y = colour_pair(source, triple)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        stellarity = source.class_scores.get(seed_key, float("nan"))
        if np.isfinite(stellarity):
            seeded += 1
            if stellarity < 0.6:
                continue
        points.append((x, y))

    if seeded == 0:
        warnings.append("no morphological seed; assuming stars outnumber galaxies")
    if len(points) < min_stars:
        log.warning("stellar locus: only %d usable colours, need %d",
                    len(points), min_stars)
        return None

    data = np.asarray(points, dtype=float)
    x, y = data[:, 0], data[:, 1]
    keep = np.ones(len(x), dtype=bool)
    order = min(int(degree), max(1, len(x) // 6))
    coefficients = np.polyfit(x, y, order)
    for _ in range(int(iterations)):
        residual = y - np.polyval(coefficients, x)
        # A median absolute deviation, not a standard deviation: the outliers
        # being clipped are exactly what would inflate the latter.
        spread = 1.4826 * float(np.median(np.abs(residual[keep] -
                                                 np.median(residual[keep]))))
        if not np.isfinite(spread) or spread <= 0:
            break
        updated = np.abs(residual) <= clip_sigma * spread
        if updated.sum() < min_stars or np.array_equal(updated, keep):
            keep = updated if updated.sum() >= min_stars else keep
            break
        keep = updated
        coefficients = np.polyfit(x[keep], y[keep], order)

    locus = StellarLocus(bands=tuple(triple), coefficients=coefficients,
                         x_range=(float(np.min(x[keep])), float(np.max(x[keep]))),
                         scatter=0.0, n_stars=int(keep.sum()), warnings=warnings)
    distances = np.array([locus.distance(px, py) for px, py in zip(x[keep], y[keep])])
    locus.scatter = float(np.sqrt(np.mean(distances ** 2))) if distances.size else 0.0
    _population_widths(catalog, locus)
    _measure_separation(catalog, locus)
    log.info("stellar locus in %s-%s vs %s-%s: %d stars, %.3f mag scatter; "
             "colour separation %.2f -> weight %.2f",
             triple[0], triple[1], triple[1], triple[2], locus.n_stars, locus.scatter,
             locus.separation, locus.information_weight)
    for message in locus.warnings:
        log.warning("stellar locus: %s", message)
    return locus


def colour_stellarity(source: Source, locus: StellarLocus) -> float:
    """Probability the source is a star given only its distance from the locus.

    A likelihood ratio between the two calibrated populations, not a
    one-sided sigmoid of the offset.  The difference is decisive: a sigmoid
    saturates near 1 for every small offset, so a test with no discriminating
    power votes "star" for everything -- including the galaxies, which is
    precisely the failure it is supposed to catch.  A ratio of two hypotheses
    returns 0.5 on its own when the hypotheses predict the same thing.

    The offset is two-dimensional, so the distance is Rayleigh distributed
    under both hypotheses and the common factor of distance cancels, leaving
    only the two widths.
    """
    if not locus.informative:
        return 0.5
    x, y = colour_pair(source, locus.bands)
    distance = locus.distance(x, y)
    if not np.isfinite(distance):
        return float("nan")
    star_scale, galaxy_scale = locus.widths(source.photometry.snr)
    if not (np.isfinite(star_scale) and np.isfinite(galaxy_scale)) or star_scale <= 0:
        return float("nan")
    if galaxy_scale <= star_scale * 1.001:
        return 0.5

    def rayleigh(d: float, scale: float) -> float:
        return float(d / scale ** 2 * np.exp(-0.5 * (d / scale) ** 2))

    # Both hypotheses are mixed with a broad outlier component.  Without it
    # the Rayleigh tail is far too thin: a blended star, or one whose colour
    # picked up a neighbour, lands five sigma out and the likelihood ratio
    # declares it a galaxy with certainty.  Measured against simulated truth
    # that single failure mode accounted for nearly all the damage colour did
    # to an otherwise good morphological classification.  A few percent of
    # any real catalog has contaminated photometry, and saying so in the
    # likelihood is what keeps one bad colour from overruling four good size
    # measurements.
    broad = max(galaxy_scale, star_scale) * 4.0
    star = ((1.0 - OUTLIER_FRACTION) * rayleigh(distance, star_scale)
            + OUTLIER_FRACTION * rayleigh(distance, broad))
    galaxy = ((1.0 - OUTLIER_FRACTION) * rayleigh(distance, galaxy_scale)
              + OUTLIER_FRACTION * rayleigh(distance, broad))
    if star + galaxy <= 0:
        return 0.5
    value = star / (star + galaxy)
    # Colour is one measurement among several, so it is not allowed to be
    # decisive on its own; the classifier fuses it with morphology.
    return float(np.clip(value, 1.0 - CONFIDENCE_CAP, CONFIDENCE_CAP))


def annotate_catalog(catalog: SourceCatalog,
                     locus: Optional[StellarLocus] = None,
                     triple: Optional[Tuple[str, str, str]] = None
                     ) -> Optional[StellarLocus]:
    """Store each source's locus distance and colour stellarity.

    Returns the locus used, or ``None`` when the field has no usable
    colours.  Nothing is overwritten on the source's class -- this is
    evidence for the classifier, not a verdict.
    """
    locus = locus or fit_stellar_locus(catalog, triple)
    if locus is None:
        return None
    counted = 0
    for source in catalog:
        x, y = colour_pair(source, locus.bands)
        distance = locus.distance(x, y)
        if not np.isfinite(distance):
            continue
        star_scale, galaxy_scale = locus.widths(source.photometry.snr)
        source.meta["locus_distance"] = float(distance)
        source.meta["locus_sigma"] = float(star_scale)
        value = colour_stellarity(source, locus)
        if np.isfinite(value):
            source.meta["colour_stellarity"] = float(value)
            counted += 1
    catalog.meta["stellar_locus"] = locus.to_dict()
    log.info("colour stellarity assigned to %d of %d sources", counted, len(catalog))
    return locus
