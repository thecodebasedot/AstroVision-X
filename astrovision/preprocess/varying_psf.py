"""A point-spread function that changes across the field.

One PSF for a whole frame is a convenient fiction.  Optics degrade off-axis,
focal planes are not flat, and a wide-field camera can easily be twenty
percent blurrier in the corners than on the axis.  Everything downstream
inherits that error: aperture corrections are wrong where the model is wrong,
profile fits are deconvolved by the wrong kernel, and -- worst of all --
difference imaging matches every epoch to a single width, so the corners
never subtract cleanly and fill the candidate list with residuals that look
exactly like transients.

The model is the standard one: each pixel of the PSF stamp is a low-order
polynomial in the field coordinates, fitted across the frame's own stars.
Because every stamp pixel shares one design matrix, the whole thing is a
single least-squares solve rather than one per pixel.

**It validates itself.**  A varying model has more freedom than a constant
one and will always fit the stars it was given better.  Whether it is
actually *better* is a different question, answered by holding stars out of
the fit and comparing predictions on them.  When the varying model does not
win on held-out stars, :func:`fit_varying_psf` returns the constant one and
says why -- because a spatial model fitted to too few stars is an
overconfident description of noise, and the failure is silent otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, sigma_clipped_stats
from .psf import SIGMA_TO_FWHM, PSFModel, find_psf_stars, gaussian_kernel

log = get_logger("preprocess.varying_psf")

#: Stars needed per polynomial term before the fit is allowed to use it.  A
#: quadratic in two dimensions has six terms, so a quadratic needs about
#: thirty stars -- fewer and the model is describing noise.
STARS_PER_TERM = 5


def polynomial_terms(x, y, degree: int) -> np.ndarray:
    """Monomials up to ``degree`` in normalised coordinates.

    ``(x, y)`` are expected in ``[-1, 1]``.  Normalising matters: raw pixel
    coordinates in the thousands make the design matrix wildly
    ill-conditioned, and the fit then returns coefficients whose sum is
    meaningful and whose individual values are nonsense.

    >>> [round(float(v), 3) for v in polynomial_terms(0.5, -0.5, 1)]
    [1.0, -0.5, 0.5]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    columns = []
    for total in range(int(degree) + 1):
        for power_x in range(total + 1):
            columns.append((x ** power_x) * (y ** (total - power_x)))
    return np.stack(columns, axis=-1)


def n_terms(degree: int) -> int:
    """Number of monomials up to ``degree`` in two dimensions."""
    return (int(degree) + 1) * (int(degree) + 2) // 2


@dataclass
class VaryingPSF:
    """A PSF whose stamp is a polynomial function of field position."""

    coefficients: np.ndarray            # (n_terms, size, size)
    degree: int
    shape: Tuple[int, int]              # of the image the model describes
    size: int
    n_stars: int
    #: Bounding box of the stars the fit actually saw, in pixels.  Positions
    #: outside it are clamped to its edge rather than extrapolated: a
    #: polynomial beyond its data is not a model, and a quadratic run past
    #: the last star produced FWHMs twice the true value in testing.
    star_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: Residual of the held-out validation, for this model and for the
    #: constant one it was compared against.  Lower is better.
    validation_rms: float = float("nan")
    constant_rms: float = float("nan")
    fallback: bool = False
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def _normalise(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x0, y0, x1, y1 = self.star_bounds
        if x1 > x0 and y1 > y0:
            x = np.clip(x, x0, x1)
            y = np.clip(y, y0, y1)
        return (2.0 * x / max(nx - 1, 1) - 1.0,
                2.0 * y / max(ny - 1, 1) - 1.0)

    def extrapolates_at(self, x: float, y: float) -> bool:
        """Whether a position lies outside the stars the fit was built from."""
        x0, y0, x1, y1 = self.star_bounds
        if not (x1 > x0 and y1 > y0):
            return False
        return not (x0 <= float(x) <= x1 and y0 <= float(y) <= y1)

    def stamp_at(self, x: float, y: float) -> np.ndarray:
        """The normalised PSF stamp at one field position."""
        u, v = self._normalise(x, y)
        basis = polynomial_terms(u, v, self.degree)
        stamp = np.tensordot(basis, self.coefficients, axes=(0, 0))
        stamp = np.clip(stamp, 0.0, None)
        total = float(stamp.sum())
        return stamp / total if total > 0 else stamp

    def at(self, x: float, y: float) -> PSFModel:
        """A :class:`PSFModel` valid near ``(x, y)``.

        Returned as the same type the constant path produces, so every
        consumer works unchanged whether or not the field varies.
        """
        stamp = self.stamp_at(x, y)
        from .psf import _fwhm_from_stamp, _stamp_shape

        fwhm = _fwhm_from_stamp(stamp)
        ellipticity, angle = _stamp_shape(stamp)
        return PSFModel(stamp, float(fwhm), float(ellipticity), float(angle),
                        n_stars=self.n_stars, size=self.size)

    def fwhm_at(self, x: float, y: float) -> float:
        from .psf import _fwhm_from_stamp
        return float(_fwhm_from_stamp(self.stamp_at(x, y)))

    def fwhm_map(self, samples: int = 5) -> np.ndarray:
        """FWHM on a coarse grid over the field -- the diagnostic to look at."""
        ny, nx = self.shape
        xs = np.linspace(0, nx - 1, int(samples))
        ys = np.linspace(0, ny - 1, int(samples))
        return np.array([[self.fwhm_at(x, y) for x in xs] for y in ys])

    def variation(self) -> float:
        """Fractional spread of the FWHM across the field.

        The number that decides whether spatial variation matters at all for
        a given frame: below a few percent, one PSF is a fair description and
        the extra machinery buys nothing.
        """
        grid = self.fwhm_map()
        finite = grid[np.isfinite(grid)]
        if finite.size == 0 or float(np.median(finite)) <= 0:
            return float("nan")
        return float((finite.max() - finite.min()) / np.median(finite))

    def to_dict(self) -> Dict[str, Any]:
        grid = self.fwhm_map()
        return {
            "degree": int(self.degree), "size": int(self.size),
            "n_stars": int(self.n_stars),
            "fallback": bool(self.fallback), "reason": self.reason,
            "validation_rms": float(self.validation_rms),
            "constant_rms": float(self.constant_rms),
            "variation": float(self.variation()),
            "star_bounds": [float(v) for v in self.star_bounds],
            "fwhm_centre": float(self.fwhm_at(self.shape[1] / 2, self.shape[0] / 2)),
            "fwhm_grid": [[float(v) for v in row] for row in grid],
        }


def region_grid(shape: Tuple[int, int], n_regions: int
                ) -> List[Tuple[slice, slice, Tuple[float, float]]]:
    """Split an image into ``n_regions x n_regions`` tiles with their centres.

    Used where a spatially varying kernel has to be *applied* rather than
    evaluated -- convolution is not a per-pixel operation, so the practical
    approach is a piecewise-constant kernel over tiles small enough that the
    PSF is nearly constant within one.

    >>> tiles = region_grid((100, 100), 2)
    >>> len(tiles), tuple(float(v) for v in tiles[0][2])
    (4, (25.0, 25.0))
    """
    ny, nx = int(shape[0]), int(shape[1])
    n = max(1, int(n_regions))
    y_edges = np.linspace(0, ny, n + 1).astype(int)
    x_edges = np.linspace(0, nx, n + 1).astype(int)
    tiles = []
    for row in range(n):
        for column in range(n):
            rows = slice(y_edges[row], y_edges[row + 1])
            columns = slice(x_edges[column], x_edges[column + 1])
            centre = (float(0.5 * (x_edges[column] + x_edges[column + 1])),
                      float(0.5 * (y_edges[row] + y_edges[row + 1])))
            tiles.append((rows, columns, centre))
    return tiles


def find_psf_stars_by_region(image: np.ndarray, n_regions: int = 3,
                             rms: Optional[np.ndarray] = None,
                             per_region: int = 30,
                             **kwargs) -> List[Tuple[float, float]]:
    """Select PSF stars tile by tile rather than over the whole frame.

    The global selection keeps the sharpest sources in the field, which is
    the right way to reject small galaxies when the PSF is constant -- and
    exactly the wrong way when it is not.  A star in a blurry corner is
    *broader* than a galaxy at the sharp centre, so a whole-frame "smallest
    first" rule throws away precisely the stars that carry the information
    about how the PSF varies.  Measured on a field with 32% corner-to-centre
    variation, the global selector returned 32 stars clustered toward the
    axis and the fit could not see the variation at all.

    Selecting within tiles keeps the galaxy rejection -- it is still
    comparing like with like -- while sampling the whole field.
    """
    data = as_float_image(image)
    stars: List[Tuple[float, float]] = []
    for rows, columns, _ in region_grid(data.shape, n_regions):
        tile = data[rows, columns]
        tile_rms = rms[rows, columns] if rms is not None else None
        if min(tile.shape) < 32:
            continue                                       # pragma: no cover
        found = find_psf_stars(tile, max_stars=per_region, rms=tile_rms, **kwargs)
        stars.extend((float(x) + columns.start, float(y) + rows.start)
                     for x, y in found)
    log.debug("regional PSF-star selection: %d stars over %d tiles",
              len(stars), n_regions ** 2)
    return stars


def _star_stamps(image: np.ndarray, stars: Sequence[Tuple[float, float]],
                 size: int, background: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised stamps and positions for the usable stars."""
    half = size // 2
    stamps, positions = [], []
    for x, y in stars:
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        if x0 < 0 or y0 < 0 or y0 + size > image.shape[0] or x0 + size > image.shape[1]:
            continue
        stamp = image[y0:y0 + size, x0:x0 + size] - background
        total = float(stamp.sum())
        if total <= 0 or not np.isfinite(total):
            continue
        stamps.append(stamp / total)
        positions.append((float(x), float(y)))
    if not stamps:
        return np.zeros((0, size, size)), np.zeros((0, 2))
    return np.stack(stamps), np.asarray(positions, dtype=float)


def _solve(basis: np.ndarray, stamps: np.ndarray, iterations: int = 3,
           clip_sigma: float = 3.0) -> np.ndarray:
    """Robust least-squares coefficients: ``(n_terms, size, size)``.

    Every stamp pixel is fitted against the same design matrix, so the solve
    is done once for all of them by flattening the stamps into columns.

    The fit is iteratively clipped, and that is not a refinement: the
    constant model it is measured against is a *median*, which shrugs off a
    contaminated stamp, while a plain least-squares fit does not.  Comparing
    the two without clipping compares structure-plus-fragility against
    no-structure-plus-robustness, and on simulated fields with a genuine 40%
    variation the varying model still lost -- not because the variation was
    absent, but because a handful of blended stamps cost it more than the
    variation gained.
    """
    flat = stamps.reshape(len(stamps), -1)
    keep = np.ones(len(stamps), dtype=bool)
    size = stamps.shape[1]
    solution, *_ = np.linalg.lstsq(basis, flat, rcond=None)
    minimum = basis.shape[1] + 3
    for _ in range(max(1, int(iterations))):
        residual = np.sqrt(np.mean((basis @ solution - flat) ** 2, axis=1))
        centre = float(np.median(residual[keep]))
        spread = 1.4826 * float(np.median(np.abs(residual[keep] - centre)))
        if not np.isfinite(spread) or spread <= 0:
            break
        updated = residual <= centre + clip_sigma * spread
        if updated.sum() < minimum or np.array_equal(updated, keep):
            break
        keep = updated
        solution, *_ = np.linalg.lstsq(basis[keep], flat[keep], rcond=None)
    return solution.reshape(basis.shape[1], size, size)


def _predict_rms(coefficients: np.ndarray, degree: int, shape: Tuple[int, int],
                 positions: np.ndarray, stamps: np.ndarray,
                 weight: Optional[np.ndarray] = None) -> float:
    """Weighted root-mean-square stamp residual of a model on given stars.

    ``weight`` is the mean PSF profile, and supplying it is what makes the
    comparison mean anything.  A 21-pixel stamp of a 3-pixel PSF is mostly
    empty sky: an unweighted residual over all 441 pixels is dominated by
    wing noise, and two models that differ substantially in the core come out
    indistinguishable.  Measured on a field with a real 34% variation across
    it, the unweighted metric could not tell the varying model from the
    constant one at all.
    """
    if len(positions) == 0:
        return float("nan")
    ny, nx = shape
    u = 2.0 * positions[:, 0] / max(nx - 1, 1) - 1.0
    v = 2.0 * positions[:, 1] / max(ny - 1, 1) - 1.0
    basis = polynomial_terms(u, v, degree)
    predicted = np.tensordot(basis, coefficients, axes=(1, 0))
    squared = (predicted - stamps) ** 2
    if weight is None:
        return float(np.sqrt(np.mean(squared)))
    mask = np.clip(np.asarray(weight, dtype=float), 0.0, None)
    total = float(mask.sum())
    if total <= 0:
        return float(np.sqrt(np.mean(squared)))            # pragma: no cover
    return float(np.sqrt(np.sum(squared * mask[None, :, :]) / (len(positions) * total)))


def fit_varying_psf(image: np.ndarray,
                    positions: Optional[Sequence[Tuple[float, float]]] = None,
                    size: int = 25, degree: int = 2,
                    rms: Optional[np.ndarray] = None,
                    min_gain: float = 0.03,
                    validation_fraction: float = 0.3,
                    n_folds: int = 7,
                    seed: int = 0) -> VaryingPSF:
    """Fit a position-dependent PSF, and check that it earns its freedom.

    The degree is reduced until there are :data:`STARS_PER_TERM` stars for
    every polynomial term, and the result is compared against a constant PSF
    on stars *held out of both fits*.  A varying model must beat the constant
    one by ``min_gain`` in fractional residual to be returned; otherwise the
    constant model comes back with ``fallback`` set and the reason recorded.

    That comparison is the whole safeguard.  A quadratic in two dimensions
    has six free parameters per stamp pixel, so with twenty stars it can
    reproduce its training set almost exactly and describe the field's noise
    rather than its optics.
    """
    data = as_float_image(image)
    if size <= 0:
        size = 25                                              # pragma: no cover
    size = max(9, int(size) | 1)
    stars = list(positions) if positions is not None else find_psf_stars(data, rms=rms)
    _, background, _ = sigma_clipped_stats(data)
    stamps, coordinates = _star_stamps(data, stars, size, float(background))

    if len(stamps) == 0:
        fwhm = 3.0
        log.warning("no PSF stars found; falling back to a %.1f px Gaussian", fwhm)
        kernel = gaussian_kernel(fwhm / SIGMA_TO_FWHM, size)
        return VaryingPSF(coefficients=kernel[None, :, :], degree=0, shape=data.shape,
                          size=size, n_stars=0, fallback=True,
                          reason="no usable PSF stars")

    allowed = int(degree)
    while allowed > 0 and len(stamps) < STARS_PER_TERM * n_terms(allowed):
        allowed -= 1
    reason = ""
    if allowed < int(degree):
        reason = (f"degree reduced from {degree} to {allowed}: {len(stamps)} stars "
                  f"support {STARS_PER_TERM} per term")

    ny, nx = data.shape
    u = 2.0 * coordinates[:, 0] / max(nx - 1, 1) - 1.0
    v = 2.0 * coordinates[:, 1] / max(ny - 1, 1) - 1.0

    def constant_model(sample: np.ndarray) -> np.ndarray:
        return np.median(sample, axis=0)[None, :, :]

    if allowed == 0 or len(stamps) < 2 * STARS_PER_TERM:
        coefficients = constant_model(stamps)
        return VaryingPSF(coefficients=coefficients, degree=0, shape=data.shape,
                          size=size, n_stars=len(stamps), fallback=True,
                          reason=reason or "too few stars for a spatial fit")

    # Cross-validated comparison, repeated over several splits.  A single
    # hold-out is not enough: with a dozen held-out stars a six-term
    # polynomial soaks up position-correlated noise and posts a double-digit
    # "gain" on a field whose PSF does not vary at all.  Measured on a
    # constant-PSF simulation, one split claimed an 11.8% improvement and
    # invented a 100% variation across the frame.
    varying_scores, constant_scores = [], []
    for repeat in range(n_folds):
        rng = np.random.default_rng(int(seed) + repeat)
        order = rng.permutation(len(stamps))
        n_hold = max(3, int(round(float(validation_fraction) * len(stamps))))
        hold, train = order[:n_hold], order[n_hold:]
        # The fold's training set only has to *determine* the model, not
        # constrain it as well as the final fit does -- that fit uses every
        # star.  Requiring the full five-per-term here made the loop break
        # immediately on fields with just enough stars, so the comparison
        # silently returned "no gain" and a genuinely varying PSF was thrown
        # away.
        if len(train) < n_terms(allowed) + 3:
            break
        fitted = _solve(polynomial_terms(u[train], v[train], allowed), stamps[train])
        profile = np.median(stamps[train], axis=0)
        varying_scores.append(_predict_rms(fitted, allowed, data.shape,
                                           coordinates[hold], stamps[hold], profile))
        constant_scores.append(_predict_rms(constant_model(stamps[train]), 0,
                                            data.shape, coordinates[hold],
                                            stamps[hold], profile))

    varying_rms = float(np.median(varying_scores)) if varying_scores else float("nan")
    constant_rms = float(np.median(constant_scores)) if constant_scores else float("nan")
    # Every fold must agree, not just the average: a model that wins on
    # average by winning hugely on one split and losing on the others is
    # describing that split.
    gains = [(c - v) / c for v, c in zip(varying_scores, constant_scores)
             if np.isfinite(c) and c > 0]
    gain = float(np.min(gains)) if gains else 0.0
    if not np.isfinite(varying_rms) or gain < float(min_gain):
        model = VaryingPSF(coefficients=constant_model(stamps), degree=0,
                           shape=data.shape, size=size, n_stars=len(stamps),
                           validation_rms=constant_rms, constant_rms=constant_rms,
                           fallback=True,
                           reason=(reason + "; " if reason else "") +
                           f"a spatial fit improved held-out stars by only "
                           f"{100 * gain:.1f}% in its worst of {n_folds} splits, "
                           f"below the {100 * min_gain:.0f}% needed to justify "
                           "the extra freedom")
        log.info("PSF: using one model for the field (%s)", model.reason)
        return model

    # Refit on everything now that the shape of the model is settled.
    bounds = (float(coordinates[:, 0].min()), float(coordinates[:, 1].min()),
              float(coordinates[:, 0].max()), float(coordinates[:, 1].max()))
    coefficients = _solve(polynomial_terms(u, v, allowed), stamps)
    model = VaryingPSF(coefficients=coefficients, degree=allowed, shape=data.shape,
                       size=size, n_stars=len(stamps), star_bounds=bounds,
                       validation_rms=varying_rms, constant_rms=constant_rms,
                       reason=reason or f"degree {allowed} beat a constant PSF by "
                                        f"{100 * gain:.1f}% on held-out stars in "
                                        f"every one of {n_folds} splits")
    log.info("PSF varies across the field: degree %d from %d stars, %.1f%% better "
             "than one model on held-out stars; FWHM spans %.1f%%",
             allowed, len(stamps), 100 * gain, 100 * model.variation())
    return model


def psf_at(image_meta: Dict[str, Any], x: float, y: float) -> Optional[PSFModel]:
    """The PSF to use at one position, from an image's metadata.

    Prefers a fitted spatial model and falls back to the single model, so a
    caller never has to know which one the frame carries.
    """
    varying = image_meta.get("varying_psf")
    if isinstance(varying, VaryingPSF) and not varying.fallback:
        return varying.at(x, y)
    model = image_meta.get("psf_model")
    return model if isinstance(model, PSFModel) else None
