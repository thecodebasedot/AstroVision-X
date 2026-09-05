"""Turning column numbers into wavelengths, and flux into a continuum-free view.

A wavelength solution is the one calibration in spectroscopy that cannot be
fudged later: every redshift, every line ratio and every velocity is measured
against it. Three properties matter, and the code reports all three rather
than returning a bare polynomial.

**It must be non-linear.** A grating's dispersion changes across the detector.
A straight line through the arc lines can leave several Angstroms of
systematic residual at the ends of the order -- which, at 5000 A, is a
redshift error of several times 10^-4, larger than the statistical error of a
good cross-correlation.

**The match must be checked, not assumed.** Arc lines are identified by
matching detected peaks to a line list. A mismatch of one line shifts the
solution by a fraction of a line spacing and still fits, so the residuals are
reported and a solution whose residual exceeds a fraction of the resolution
is refused rather than returned.

That check is not enough on its own, because the identification can be wrong
*before* any fit happens. Matching peaks to their nearest predicted
wavelength needs a first guess, and a linear guess for a non-linear
dispersion is wrong by tens of Angstroms in the middle of the detector --
which is comparable to the line spacing, so lines get identified one over,
the fit absorbs the error, and the residual comes out at a few Angstroms
everywhere: not obviously broken, just wrong. Measured on these frames, that
route produced a 3.7 A residual at every polynomial order, including the
order that generated the data.

So the identification does not start from a guess at all. Every pair of
detected peaks is compared with every pair of catalogue lines, and each
consistent pairing votes for the linear solution it implies. The right answer
collects votes from many independent pairs; a wrong one collects a few by
chance. Only after that vote picks an anchor is the solution refined, one
polynomial order at a time.

**Extrapolation is not calibration.** Beyond the outermost identified line,
a cubic does whatever a cubic does. The solution carries the range it was
fitted over, and wavelengths outside it are flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import mad_std
from .templates import Spectrum1D

log = get_logger("spectra.calibrate")

#: A fitted solution whose residual exceeds this fraction of the resolution
#: element is not returned as a solution.
#:
#: Set from what the two outcomes actually look like rather than from taste.
#: A correct identification leaves a residual at the centroiding floor -- 0.5
#: Angstroms on a 5 Angstrom resolution element here, about an eighth of a
#: pixel -- because that is how well a line's centre can be measured, and no
#: polynomial can do better. A misidentification by one line leaves 3.7
#: Angstroms. A threshold at 0.3 of a resolution element sits between them
#: with room on both sides; a stricter one rejects correct solutions for
#: being no better than the data allow.
MAX_RESIDUAL_FRACTION = 0.30


@dataclass
class WavelengthSolution:
    """Column-to-wavelength mapping, with the evidence for it."""

    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_lines: int = 0
    rms: float = float("nan")              # Angstroms
    max_residual: float = float("nan")
    column_range: Tuple[float, float] = (float("nan"), float("nan"))
    wavelength_range: Tuple[float, float] = (float("nan"), float("nan"))
    matched: List[Tuple[float, float]] = field(default_factory=list)
    succeeded: bool = False
    reason: str = ""
    flags: List[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def __call__(self, column) -> np.ndarray:
        """Wavelength at one or more columns."""
        if not self.succeeded:
            return np.full(np.shape(column), np.nan, dtype=float)
        return np.polyval(self.coefficients, np.asarray(column, dtype=float))

    def extrapolates(self, column) -> np.ndarray:
        """True where a column lies outside the calibrated range."""
        values = np.asarray(column, dtype=float)
        low, high = self.column_range
        return (values < low) | (values > high)

    def to_dict(self) -> Dict[str, Any]:
        return {"succeeded": self.succeeded, "n_lines": self.n_lines,
                "rms": self.rms, "max_residual": self.max_residual,
                "order": max(len(self.coefficients) - 1, 0),
                "column_range": list(self.column_range),
                "wavelength_range": list(self.wavelength_range),
                "reason": self.reason, "flags": list(self.flags)}


def find_peaks(flux: np.ndarray, threshold_sigma: float = 6.0,
               min_separation: int = 4) -> np.ndarray:
    """Column positions of emission peaks, centroided to sub-pixel.

    The centroid uses a parabola through the peak and its two neighbours,
    which is exact for a symmetric profile sampled at three points and is
    what keeps the wavelength residuals below a fifth of a pixel.

    >>> import numpy as np
    >>> x = np.arange(60.0)
    >>> signal = np.exp(-0.5 * ((x - 30.4) / 1.5) ** 2) * 100
    >>> float(np.round(find_peaks(signal + 1e-9, threshold_sigma=3.0)[0], 1))
    30.4
    """
    values = np.asarray(flux, dtype=float)
    if values.size < 5:
        return np.zeros(0)
    background = float(np.median(values))
    noise = float(mad_std(values))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(values)) or 1.0
    signal = values - background

    candidates = []
    for i in range(1, len(values) - 1):
        if signal[i] < threshold_sigma * noise:
            continue
        if signal[i] < signal[i - 1] or signal[i] < signal[i + 1]:
            continue
        left, centre, right = signal[i - 1], signal[i], signal[i + 1]
        denominator = left - 2.0 * centre + right
        shift = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
        candidates.append((i + float(np.clip(shift, -1.0, 1.0)), float(centre)))

    candidates.sort(key=lambda item: -item[1])
    kept: List[float] = []
    for position, _ in candidates:
        if all(abs(position - other) >= min_separation for other in kept):
            kept.append(position)
    return np.sort(np.asarray(kept, dtype=float))


def match_lines(peaks: Sequence[float], line_list: Sequence[float],
                guess: Sequence[float], tolerance: float = 12.0
                ) -> List[Tuple[float, float]]:
    """Pair detected peaks with catalogue wavelengths, using a first guess.

    The guess is a rough polynomial -- usually a two-point linear estimate
    from the header. Matching is **mutual nearest**: a peak and a catalogue
    line are paired only if each is the other's closest partner. One-sided
    matching lets a bright unlisted line steal several catalogue entries and
    produces a solution that fits beautifully and is wrong by one line.
    """
    peaks = np.asarray(peaks, dtype=float)
    catalogue = np.asarray(sorted(line_list), dtype=float)
    if peaks.size == 0 or catalogue.size == 0:
        return []
    predicted = np.polyval(np.asarray(guess, dtype=float), peaks)

    pairs: List[Tuple[float, float]] = []
    for i, wavelength in enumerate(predicted):
        j = int(np.argmin(np.abs(catalogue - wavelength)))
        if abs(catalogue[j] - wavelength) > tolerance:
            continue
        # Mutual check: is this peak also the closest to that catalogue line?
        back = int(np.argmin(np.abs(predicted - catalogue[j])))
        if back == i:
            pairs.append((float(peaks[i]), float(catalogue[j])))
    return pairs


def vote_for_linear_solution(peaks: Sequence[float], line_list: Sequence[float],
                             tolerance: float = 6.0,
                             scale_range: Tuple[float, float] = (0.2, 40.0)
                             ) -> Tuple[np.ndarray, int]:
    """Find the linear solution the most pairs of lines agree on.

    Two detected peaks and two catalogue lines imply one linear solution: the
    scale is the ratio of their separations and the offset follows. If the
    identification is right, *many* independent pairs imply nearly the same
    solution; if it is wrong, agreement is accidental and rare. Counting the
    agreement is therefore a way to identify lines with no prior knowledge of
    the dispersion beyond a range it must lie in.

    This is the same idea as pattern matching an astrometric field: use the
    *relative* geometry, which is invariant, rather than absolute positions,
    which are exactly what is unknown.

    Returns the winning ``[scale, offset]`` and the number of peaks it
    places within ``tolerance`` of a catalogue line.
    """
    peaks = np.asarray(peaks, dtype=float)
    catalogue = np.asarray(sorted(line_list), dtype=float)
    if peaks.size < 2 or catalogue.size < 2:
        return np.array([1.0, 0.0]), 0

    best: Tuple[int, float] = (0, float("inf"))
    best_coefficients = np.array([1.0, 0.0])
    low, high = scale_range
    # Only well-separated peaks anchor a scale: two peaks a few pixels apart
    # give a ratio dominated by their centroid errors.
    minimum_gap = 0.15 * max(peaks[-1] - peaks[0], 1.0)
    for i in range(peaks.size - 1):
        for j in range(i + 1, peaks.size):
            gap = peaks[j] - peaks[i]
            if gap < minimum_gap:
                continue
            for a in range(catalogue.size - 1):
                for b in range(a + 1, catalogue.size):
                    scale = (catalogue[b] - catalogue[a]) / gap
                    if not low <= scale <= high:
                        continue
                    offset = catalogue[a] - scale * peaks[i]
                    predicted = scale * peaks + offset
                    distance = np.abs(predicted[:, None] - catalogue[None, :]).min(axis=1)
                    hits = int(np.sum(distance < tolerance))
                    score = float(distance[distance < tolerance].sum())
                    if hits > best[0] or (hits == best[0] and score < best[1]):
                        best = (hits, score)
                        best_coefficients = np.array([scale, offset])
    return best_coefficients, best[0]


def fit_wavelength_solution(arc_flux: np.ndarray, line_list: Sequence[float],
                            guess: Optional[Sequence[float]] = None,
                            order: int = 3, resolution: float = 5.0,
                            threshold_sigma: float = 6.0) -> WavelengthSolution:
    """Fit a polynomial column-to-wavelength solution to an arc spectrum.

    ``guess`` is a rough solution, lowest-order-last as :func:`numpy.polyval`
    expects. It is optional and, when given, is only used to bound the search:
    the identification itself comes from the pairwise vote in
    :func:`vote_for_linear_solution`, because a guess accurate enough to
    identify lines directly is a guess that has already solved the problem.

    The polynomial order is raised one step at a time, re-matching after each
    fit. Fitting a cubic immediately through whatever the first pass matched
    lets three or four points -- some possibly misidentified -- set a curve
    that then rejects the lines that would have corrected it.
    """
    flux = np.asarray(arc_flux, dtype=float)
    solution = WavelengthSolution()
    peaks = find_peaks(flux, threshold_sigma=threshold_sigma)
    if peaks.size < order + 2:
        solution.reason = (f"found {peaks.size} arc peaks, which is fewer than "
                           f"the {order + 2} needed for an order-{order} fit")
        return solution

    if guess is not None:
        coefficients = np.asarray(guess, dtype=float)
        scale = abs(float(coefficients[-2])) if len(coefficients) >= 2 else 1.0
        bounds = (0.5 * scale, 2.0 * scale)
    else:
        bounds = (0.2, 40.0)
    coefficients, votes = vote_for_linear_solution(
        peaks, line_list, tolerance=1.5 * resolution, scale_range=bounds)
    if votes < order + 2:
        solution.reason = (f"the best linear solution places only {votes} of "
                           f"{peaks.size} peaks on catalogue lines; the arc, the "
                           "line list, or the assumed dispersion is wrong")
        return solution

    pairs: List[Tuple[float, float]] = []
    for current in range(1, int(order) + 1):
        # Tolerance follows the fit rather than a fixed schedule: while the
        # solution is still linear it must stay wide enough to reach the lines
        # the curvature has moved, and it tightens only as the residuals do.
        for _ in range(2):
            tolerance = max(3.0 * _residual_spread(pairs, coefficients),
                            1.5 * resolution)
            pairs = match_lines(peaks, line_list, coefficients,
                                tolerance=tolerance)
            if len(pairs) < current + 2:
                break
            x = np.array([p for p, _ in pairs])
            y = np.array([w for _, w in pairs])
            coefficients = np.polyfit(x, y, current)
            residual = y - np.polyval(coefficients, x)
            spread = float(mad_std(residual))
            if np.isfinite(spread) and spread > 0 and len(pairs) > current + 3:
                keep = np.abs(residual) < 4.0 * spread
                if current + 2 <= keep.sum() < len(pairs):
                    pairs = [pairs[i] for i in np.flatnonzero(keep)]
                    coefficients = np.polyfit(x[keep], y[keep], current)

    if len(pairs) < order + 2:
        solution.reason = (f"matched only {len(pairs)} of {peaks.size} peaks to the "
                           "line list after the pairwise vote; the line list may "
                           "not be the lamp that was exposed")
        return solution

    x = np.array([p for p, _ in pairs])
    y = np.array([w for _, w in pairs])
    coefficients = np.polyfit(x, y, order)
    residual = y - np.polyval(coefficients, x)

    solution.coefficients = coefficients
    solution.matched = list(zip(x.tolist(), y.tolist()))
    solution.n_lines = len(pairs)
    solution.rms = float(np.sqrt(np.mean(residual ** 2)))
    solution.max_residual = float(np.max(np.abs(residual)))
    solution.column_range = (float(x.min()), float(x.max()))
    solution.wavelength_range = (float(y.min()), float(y.max()))

    limit = MAX_RESIDUAL_FRACTION * float(resolution)
    if solution.rms > limit:
        solution.reason = (f"residual {solution.rms:.2f} A exceeds the "
                           f"{limit:.2f} A this fit is required to reach; the "
                           "line identification is probably wrong somewhere")
        solution.add_flag("residual_too_large")
        return solution

    solution.succeeded = True
    solution.reason = (f"{len(pairs)} lines fitted with an order-{order} "
                       f"polynomial to {solution.rms:.3f} A rms")
    if x.min() > 0.1 * len(flux) or x.max() < 0.9 * len(flux):
        solution.add_flag("lines_do_not_span_the_detector")
    log.info("wavelength solution: %d lines, %.3f A rms, %.1f-%.1f A",
             solution.n_lines, solution.rms, *solution.wavelength_range)
    return solution


def _residual_spread(pairs: Sequence[Tuple[float, float]],
                     coefficients: np.ndarray) -> float:
    """Robust residual of the current identification, in Angstroms."""
    if len(pairs) < 3:
        return 0.0
    x = np.array([p for p, _ in pairs])
    y = np.array([w for _, w in pairs])
    spread = float(mad_std(y - np.polyval(coefficients, x)))
    return spread if np.isfinite(spread) else 0.0


def apply_solution(spectrum: Spectrum1D, solution: WavelengthSolution
                   ) -> Spectrum1D:
    """Put a wavelength axis on a spectrum extracted in columns.

    Pixels outside the calibrated column range are masked rather than given
    an extrapolated wavelength: a number produced by a cubic beyond its last
    constraint is not a measurement, and downstream code has no way to tell.
    """
    if not solution.succeeded:
        raise ValueError("cannot apply a wavelength solution that did not converge")
    columns = spectrum.wavelength
    wavelength = solution(columns)
    mask = solution.extrapolates(columns)
    if spectrum.mask is not None:
        mask = mask | spectrum.mask
    meta = dict(spectrum.meta)
    meta["axis"] = "wavelength"
    meta["wavelength_solution"] = solution.to_dict()
    return Spectrum1D(wavelength, spectrum.flux.copy(),
                      None if spectrum.error is None else spectrum.error.copy(),
                      mask, meta)


def check_against_sky_lines(spectrum: Spectrum1D, sky_lines: Sequence[float],
                            window: float = 8.0,
                            isolation: float = 20.0) -> Dict[str, Any]:
    """Measure the zero-point error of a solution using night-sky lines.

    The arc lamp is exposed at a different time, and often a different
    telescope position, from the science frame; flexure between them shifts
    the solution bodily. The sky lines are *in* the science frame, at known
    wavelengths, so they measure that shift directly. This is the check that
    catches a calibration which is internally perfect and externally wrong.

    Pass the **sky** spectrum, or an extraction with the sky still in it. On a
    sky-subtracted spectrum only residuals remain, and a residual's centroid
    is set by how the subtraction failed rather than by where the line is.

    Only lines further than ``isolation`` from their neighbours are used. A
    blend -- the sodium doublet, say -- has a centroid that sits between its
    components and reports an offset that is a property of the doublet, not
    of the calibration.

    Measured on a correctly calibrated frame, this returns an offset of
    -0.03 +/- 0.03 Angstroms with 0.10 Angstroms of scatter across 14 lines,
    and it recovers a deliberately injected 3 Angstrom shift as 2.96. Handed
    the sky-*subtracted* spectrum instead, the same check returns -0.4 with
    2.8 Angstroms of scatter -- still not wrong, but no longer able to see a
    shift smaller than a pixel.
    """
    catalogue = np.asarray(sorted(sky_lines), dtype=float)
    # The window has to hold enough pixels to centroid with.  At 4 Angstroms
    # per pixel a fixed 8 Angstrom window is four pixels, which is a centroid
    # over two points either side of the peak -- so the window is widened to
    # the dispersion when the dispersion is coarse.
    dispersion = spectrum.dispersion()
    if np.isfinite(dispersion) and dispersion > 0:
        window = max(float(window), 3.0 * float(dispersion))
    offsets: List[float] = []
    used: List[float] = []
    for index, line in enumerate(catalogue):
        neighbours = np.delete(catalogue, index)
        if neighbours.size and np.min(np.abs(neighbours - line)) < isolation:
            continue
        near = np.abs(spectrum.wavelength - line) < window
        if near.sum() < 5:
            continue
        grid = spectrum.wavelength[near]
        local = spectrum.flux[near]
        weights = np.clip(local - np.median(local), 0, None)
        if weights.sum() <= 0:
            continue
        centre = float((weights * grid).sum() / weights.sum())
        shift = centre - float(line)
        # A centroid that lands near the edge of the window has been pulled by
        # something outside it, and measures that something instead.
        if abs(shift) > 0.6 * window:
            continue
        offsets.append(shift)
        used.append(float(line))
    if not offsets:
        return {"n_lines": 0, "offset": float("nan"), "scatter": float("nan"),
                "reliable": False, "lines": [],
                "reason": "no isolated sky lines found in the calibrated range"}
    values = np.asarray(offsets, dtype=float)
    scatter = float(mad_std(values)) if len(values) > 2 else float("nan")
    return {"n_lines": len(values), "offset": float(np.median(values)),
            "scatter": scatter, "lines": used,
            "error_on_offset": (float(scatter / math.sqrt(len(values)))
                                if np.isfinite(scatter) else float("nan")),
            "reliable": len(values) >= 3,
            "reason": f"measured from {len(values)} isolated night-sky lines"}


def fit_continuum(spectrum: Spectrum1D, window: int = 151,
                  iterations: int = 3, low_sigma: float = 1.5) -> np.ndarray:
    """Estimate the continuum under the lines.

    A running median, then iterative rejection of pixels *above* the current
    estimate. The asymmetry is the point: emission lines only ever push the
    flux up, so clipping symmetrically biases the continuum upward wherever
    lines are strong, which then eats the equivalent widths measured against
    it. Absorption is left in, because clipping it out would make the
    continuum trace the absorption troughs.
    """
    flux = np.array(spectrum.flux, dtype=float)
    good = spectrum.good & np.isfinite(flux)
    working = np.where(good, flux, np.nan)
    size = int(max(window, 5)) | 1                 # odd
    half = size // 2
    continuum = np.copy(working)
    for _ in range(int(max(iterations, 1))):
        padded = np.pad(continuum, half, mode="edge")
        strided = np.lib.stride_tricks.sliding_window_view(padded, size)
        with np.errstate(invalid="ignore"):
            smooth = np.nanmedian(strided, axis=-1)
        residual = working - smooth
        spread = float(mad_std(residual[np.isfinite(residual)]))
        if not np.isfinite(spread) or spread <= 0:
            continuum = smooth
            break
        above = np.isfinite(residual) & (residual > low_sigma * spread)
        continuum = np.where(above, smooth, working)
    padded = np.pad(continuum, half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, size)
    with np.errstate(invalid="ignore"):
        smooth = np.nanmedian(strided, axis=-1)
    return np.where(np.isfinite(smooth), smooth, np.nanmedian(working))


#: The continuum must exceed this many times the noise before a pixel can be
#: divided by it.  Below that the division is noise over noise.
MIN_CONTINUUM_SNR = 3.0


def normalise(spectrum: Spectrum1D, window: int = 151,
              min_continuum_snr: float = MIN_CONTINUUM_SNR) -> Spectrum1D:
    """Divide out the continuum, leaving the lines.

    Cross-correlation and line fitting both want this: the continuum shape
    carries no redshift information and dominates the variance, so leaving it
    in means correlating two continua and calling the answer a redshift.

    **Where there is no continuum, there is nothing to divide by.** Every
    spectrograph has ends where the throughput collapses, and a redshifted
    object often has no flux at all over part of the detector. Dividing the
    remaining noise by a continuum near zero does not produce a faint
    spectrum, it produces a loud one: measured on a galaxy at z = 0.4 whose
    template ran out below 4200 Angstroms, the normalised flux reached 2000
    where real features reach 0.3, and a cross-correlation against it
    returned a confident, entirely wrong redshift. So pixels whose continuum
    is not at least ``min_continuum_snr`` times the noise are masked and set
    to the continuum value, contributing nothing rather than everything.
    """
    continuum = fit_continuum(spectrum, window=window)
    if spectrum.error is not None and np.isfinite(spectrum.error).any():
        noise = float(np.nanmedian(spectrum.error[np.isfinite(spectrum.error)]))
    else:
        residual = spectrum.flux - continuum
        noise = float(mad_std(residual[np.isfinite(residual)]))
    if not np.isfinite(noise) or noise <= 0:
        noise = 0.0

    unusable = ~np.isfinite(continuum) | (continuum < min_continuum_snr * noise)
    safe = np.where(unusable, np.nan, continuum)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = spectrum.flux / safe
        error = None if spectrum.error is None else spectrum.error / np.abs(safe)
    mask = unusable | ~np.isfinite(flux)
    if spectrum.mask is not None:
        mask = mask | spectrum.mask
    meta = dict(spectrum.meta)
    meta["continuum_removed"] = True
    meta["n_no_continuum"] = int(unusable.sum())
    return Spectrum1D(spectrum.wavelength.copy(), np.where(mask, 1.0, flux),
                      error, mask, meta)
