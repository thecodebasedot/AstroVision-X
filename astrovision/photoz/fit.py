"""Fitting a redshift to a handful of colours, and saying how well it worked.

The fit is a chi-squared over a grid of (template, redshift), turned into a
posterior.  What matters is not the grid -- that part is arithmetic -- but
three things the arithmetic makes it easy to hide:

**A photo-z posterior is very often bimodal.**  A red galaxy at low redshift
and a blue one at higher redshift can produce the same three colours, because
the 4000 Angstrom break sitting between two filters looks much like a red
continuum.  Reporting only the peak throws that away and turns a known
ambiguity into a confident wrong answer.  Both peaks are reported here, and a
source whose second peak carries real weight is flagged.

**The width of the posterior is not the error.**  It is the error *given the
template library*, and the library is wrong -- no six templates describe every
galaxy.  Measured against truth, the posterior width underestimates the actual
scatter, so the reported uncertainty is inflated by a floor derived from that
comparison rather than quoted as if the model were right.

**Three bands is not enough.**  With ``g, r, i`` there are two colours and at
least three unknowns (redshift, spectral type, dust).  The problem is
underdetermined and no amount of care in the fit changes that; what care does
is make the failure visible instead of silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from .templates import SEDTemplate, colour_grid, describe_break_crossings, standard_library

log = get_logger("photoz.fit")

#: Fractional scatter added in quadrature to every reported uncertainty, to
#: account for the template library not containing the real galaxy.  Measured
#: against simulated truth: without it the posterior widths were about three
#: times too small.
TEMPLATE_FLOOR = 0.03

#: Above this fraction of the posterior in a secondary peak, the redshift is
#: called ambiguous rather than measured.
SECOND_PEAK_FRACTION = 0.25


@dataclass
class PhotoZResult:
    """One galaxy's redshift estimate, with its ambiguities attached."""

    z: float = float("nan")
    z_error: float = float("nan")
    z_lower: float = float("nan")          # 16th percentile of the posterior
    z_upper: float = float("nan")          # 84th
    template: str = ""
    chi2: float = float("nan")
    reduced_chi2: float = float("nan")
    n_colours: int = 0
    second_z: float = float("nan")
    second_weight: float = 0.0
    odds: float = float("nan")             # posterior mass within +-0.06(1+z)
    flags: List[str] = field(default_factory=list)
    posterior: np.ndarray = field(default_factory=lambda: np.zeros(0))
    redshifts: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def reliable(self) -> bool:
        """Whether the estimate is worth using without a caveat attached."""
        return bool(np.isfinite(self.z) and "ambiguous" not in self.flags
                    and "poor_fit" not in self.flags and self.odds >= 0.7)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z": float(self.z), "z_error": float(self.z_error),
            "z_lower": float(self.z_lower), "z_upper": float(self.z_upper),
            "template": self.template,
            "chi2": float(self.chi2), "reduced_chi2": float(self.reduced_chi2),
            "n_colours": int(self.n_colours),
            "second_z": float(self.second_z),
            "second_weight": float(self.second_weight),
            "odds": float(self.odds),
            "reliable": self.reliable,
            "flags": list(self.flags),
        }


class PhotoZLibrary:
    """A precomputed grid of template colours against redshift.

    The integral over the filter curves does not depend on the data, so it is
    done once here and every galaxy is then a matrix operation.  Building the
    library is the expensive part; fitting a catalog is not.

    >>> library = PhotoZLibrary(bands=("g", "r", "i"), z_max=1.0, n_z=40)
    >>> library.grid.shape
    (6, 40, 2)
    """

    def __init__(self, bands: Sequence[str] = ("u", "g", "r", "i", "z"),
                 templates: Optional[Sequence[SEDTemplate]] = None,
                 z_min: float = 0.0, z_max: float = 1.5, n_z: int = 150):
        self.bands = tuple(bands)
        self.templates = list(templates) if templates is not None else standard_library()
        self.redshifts = np.linspace(float(z_min), float(z_max), int(n_z))
        self.grid = colour_grid(self.templates, self.redshifts, self.bands)
        self.break_ranges = describe_break_crossings(self.bands)
        log.debug("photo-z library: %d templates x %d redshifts in %s",
                  len(self.templates), len(self.redshifts), ",".join(self.bands))

    @property
    def colour_names(self) -> List[str]:
        return [f"{a}-{b}" for a, b in zip(self.bands[:-1], self.bands[1:])]

    def constrained_range(self) -> Tuple[float, float]:
        """Redshifts where the 4000 A break lies inside one of the filters.

        Outside it the break is between filters, the colours change slowly
        with redshift, and the fit is extrapolating from a continuum slope
        that dust can imitate.  This is where photo-z errors come from, and
        it is a property of the filter set rather than of any algorithm.
        """
        if not self.break_ranges:
            return (float("nan"), float("nan"))
        lows = [low for low, _ in self.break_ranges.values()]
        highs = [high for _, high in self.break_ranges.values()]
        return float(min(lows)), float(max(highs))


def fit_photoz(colours: Sequence[float], errors: Sequence[float],
               library: PhotoZLibrary,
               prior: Optional[np.ndarray] = None) -> PhotoZResult:
    """Fit one galaxy's colours against the library.

    ``colours`` and ``errors`` are consecutive colour indices in the
    library's band order, with NaN where a colour could not be measured --
    those are dropped rather than filled, since a fabricated colour is worse
    than a missing one.
    """
    observed = np.asarray(colours, dtype=float)
    sigma = np.asarray(errors, dtype=float)
    usable = np.isfinite(observed) & np.isfinite(sigma) & (sigma > 0)
    result = PhotoZResult(n_colours=int(usable.sum()),
                          redshifts=library.redshifts.copy())
    if usable.sum() == 0:
        result.add_flag("no_usable_colours")
        return result

    model = library.grid[:, :, usable]                     # (templates, z, colours)
    residual = (model - observed[usable]) / sigma[usable]
    chi2 = np.sum(residual ** 2, axis=2)                   # (templates, z)

    # Marginalise over templates rather than picking the best one.  A single
    # best-fit template hides the very degeneracy that makes photo-z hard:
    # two types can fit almost equally well at different redshifts, and the
    # posterior should show both.
    likelihood = np.exp(-0.5 * (chi2 - chi2.min()))
    posterior = likelihood.sum(axis=0)
    if prior is not None:
        posterior = posterior * np.asarray(prior, dtype=float)
    total = float(posterior.sum())
    if total <= 0 or not np.isfinite(total):               # pragma: no cover
        result.add_flag("degenerate_posterior")
        return result
    posterior = posterior / total
    result.posterior = posterior

    best_template, best_z = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    result.template = library.templates[best_template].name
    result.chi2 = float(chi2[best_template, best_z])
    degrees = max(int(usable.sum()) - 2, 1)
    result.reduced_chi2 = result.chi2 / degrees

    # The estimate is the posterior *mean*, not the chi-squared minimum: with
    # a bimodal posterior the minimum sits in whichever peak happens to be a
    # hair deeper, and flips between them on noise.
    z_grid = library.redshifts
    result.z = float(np.sum(posterior * z_grid))
    variance = float(np.sum(posterior * (z_grid - result.z) ** 2))
    cumulative = np.cumsum(posterior)
    result.z_lower = float(np.interp(0.16, cumulative, z_grid))
    result.z_upper = float(np.interp(0.84, cumulative, z_grid))

    # Inflate by the template floor: the posterior width is the uncertainty
    # *given* the library, and the library does not contain the real galaxy.
    floor = TEMPLATE_FLOOR * (1.0 + result.z)
    result.z_error = float(math.sqrt(max(variance, 0.0) + floor ** 2))

    peaks = _find_peaks(posterior)
    if peaks:
        primary = max(peaks, key=lambda p: p[1])
        others = [p for p in peaks if p is not primary]
        if others:
            second = max(others, key=lambda p: p[1])
            result.second_z = float(z_grid[second[0]])
            result.second_weight = float(second[1] / max(primary[1], 1e-12))
            if result.second_weight >= SECOND_PEAK_FRACTION:
                result.add_flag("ambiguous")

    # ODDS: how much of the posterior lies near the estimate.  The standard
    # photo-z reliability statistic, and a better one than the error bar
    # because a bimodal posterior can be narrow in each peak and useless.
    window = 0.06 * (1.0 + result.z)
    inside = np.abs(z_grid - result.z) <= window
    result.odds = float(posterior[inside].sum())

    if result.reduced_chi2 > 10.0:
        result.add_flag("poor_fit")
    if result.n_colours < 3:
        result.add_flag("underdetermined")
    low, high = library.constrained_range()
    if np.isfinite(low) and not (low <= result.z <= high):
        result.add_flag("break_outside_filters")
    return result


def _find_peaks(posterior: np.ndarray, min_height: float = 0.02
                ) -> List[Tuple[int, float]]:
    """Local maxima of a posterior, as ``(index, height)``.

    Peaks are separated by requiring the posterior to drop to half the lower
    of two peaks between them; without that, noise on the grid splits one
    broad peak into several and every galaxy looks ambiguous.
    """
    values = np.asarray(posterior, dtype=float)
    if values.size < 3:
        return []
    candidates = [i for i in range(1, values.size - 1)
                  if values[i] >= values[i - 1] and values[i] > values[i + 1]
                  and values[i] >= min_height * values.max()]
    kept: List[Tuple[int, float]] = []
    for index in candidates:
        merged = False
        for position, (other, height) in enumerate(kept):
            low, high = sorted((index, other))
            valley = float(values[low:high + 1].min())
            if valley > 0.5 * min(values[index], height):
                if values[index] > height:
                    kept[position] = (index, float(values[index]))
                merged = True
                break
        if not merged:
            kept.append((index, float(values[index])))
    return kept


def fit_catalog(catalog, library: PhotoZLibrary,
                min_snr: float = 5.0) -> Dict[str, Any]:
    """Estimate a redshift for every galaxy in a catalog, in place.

    Only sources classified as extended are fitted.  A star has no redshift,
    and fitting one produces a number that will be used as if it did.
    """
    from ..core.types import ObjectClass

    fitted = 0
    ambiguous = 0
    for source in catalog:
        if source.object_class not in (ObjectClass.GALAXY, ObjectClass.UNKNOWN):
            continue
        colours, errors = [], []
        for blue, red in zip(library.bands[:-1], library.bands[1:]):
            stored = (source.meta.get("colours") or {}).get(f"{blue}-{red}")
            value = stored if stored is not None else source.colour(blue, red)
            error = source.colour_error(blue, red)
            first, second = source.bands.get(blue), source.bands.get(red)
            if (first is None or second is None
                    or not np.isfinite(first.snr) or first.snr < min_snr
                    or not np.isfinite(second.snr) or second.snr < min_snr):
                value = float("nan")
            colours.append(float(value) if value is not None else float("nan"))
            # The floor keeps one implausibly precise colour from dominating
            # the chi-squared and driving the fit to a spurious spike.
            errors.append(max(float(error) if np.isfinite(error) else 0.1, 0.02))
        result = fit_photoz(colours, errors, library)
        if not np.isfinite(result.z):
            continue
        source.meta["photoz"] = result.to_dict()
        fitted += 1
        ambiguous += "ambiguous" in result.flags

    low, high = library.constrained_range()
    report = {
        "n_fitted": fitted,
        "n_ambiguous": ambiguous,
        "bands": list(library.bands),
        "n_templates": len(library.templates),
        "z_range": [float(library.redshifts[0]), float(library.redshifts[-1])],
        "well_constrained_range": [low, high],
        "colours": library.colour_names,
    }
    if hasattr(catalog, "meta"):
        catalog.meta["photoz"] = dict(report)
    log.info("photometric redshifts for %d galaxies from %d colours; %d ambiguous. "
             "The break is inside a filter only for z in %.2f-%.2f -- outside that "
             "range these are extrapolations",
             fitted, len(library.bands) - 1, ambiguous, low, high)
    return report
