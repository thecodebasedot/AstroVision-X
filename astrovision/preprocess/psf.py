"""Point-spread-function estimation and matching.

The PSF sets the resolution limit of every measurement: it decides which
sources are point-like, how big an aperture should be, and -- crucially --
how two epochs must be convolved before they can be subtracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    MAD_TO_SIGMA,
    SIGMA_TO_FWHM,
    as_float_image,
    convolve,
    gaussian_kernel,
    maximum_filter,
    radial_profile,
    sigma_clipped_stats,
)

log = get_logger("preprocess.psf")


@dataclass
class PSFModel:
    """An empirical PSF: a normalised stamp plus its summary statistics."""

    stamp: np.ndarray
    fwhm: float
    ellipticity: float = 0.0
    position_angle: float = 0.0
    n_stars: int = 0
    size: int = 25

    @property
    def sigma(self) -> float:
        return float(self.fwhm) / SIGMA_TO_FWHM

    def as_kernel(self) -> np.ndarray:
        """The stamp normalised to unit sum, ready for convolution."""
        total = float(self.stamp.sum())
        return self.stamp / total if total > 0 else self.stamp

    def encircled_radius(self, fraction: float = 0.9) -> float:
        """Radius containing ``fraction`` of the PSF flux, in pixels."""
        stamp = np.clip(self.as_kernel(), 0, None)
        centre = ((stamp.shape[1] - 1) / 2.0, (stamp.shape[0] - 1) / 2.0)
        yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
        r = np.hypot(xx - centre[0], yy - centre[1]).ravel()
        order = np.argsort(r)
        cumulative = np.cumsum(stamp.ravel()[order])
        total = cumulative[-1]
        if total <= 0:
            return float(self.fwhm)
        index = int(np.searchsorted(cumulative, fraction * total))
        return float(r[order][min(index, len(r) - 1)])

    def to_dict(self):
        return {"fwhm": float(self.fwhm), "ellipticity": float(self.ellipticity),
                "position_angle": float(self.position_angle),
                "n_stars": int(self.n_stars), "size": int(self.size),
                "r90": self.encircled_radius(0.9)}


def find_psf_stars(image: np.ndarray, threshold_sigma: float = 12.0,
                   max_stars: int = 60, edge: int = 16,
                   rms: Optional[np.ndarray] = None,
                   reject_extended: bool = True) -> List[Tuple[float, float]]:
    """Locate isolated, unsaturated point sources suitable for PSF fitting.

    Galaxies are the hazard here.  A field with bright extended sources
    will happily supply "stars" that are really small galaxies, and the
    resulting PSF is too broad -- which then biases every profile fit and
    every star/galaxy separation that depends on it.  Candidates are
    therefore cut against the *stellar locus*: the narrow minimum of the
    size distribution that unresolved sources occupy.
    """
    data = as_float_image(image)
    if rms is None:
        _, median, noise = sigma_clipped_stats(data)
    else:
        _, median, _ = sigma_clipped_stats(data)
        noise = float(np.median(rms))
    threshold = median + threshold_sigma * max(noise, 1e-9)

    peaks = (data >= maximum_filter(data, 5)) & (data > threshold)
    ny, nx = data.shape
    peaks[:edge, :] = peaks[-edge:, :] = False
    peaks[:, :edge] = peaks[:, -edge:] = False
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return []

    order = np.argsort(data[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]

    # Discard the very brightest few (likely saturated) and any neighbours.
    saturation = float(np.percentile(data[ys, xs], 97)) if ys.size > 10 else np.inf
    candidates: List[Tuple[float, float]] = []
    for y, x in zip(ys, xs):
        if data[y, x] >= saturation and len(candidates) > 3:
            continue
        if any(abs(x - cx) < edge and abs(y - cy) < edge for cx, cy in candidates):
            continue
        # Reject crowded stars: no other peak within the stamp.
        window = peaks[max(0, y - edge):y + edge + 1, max(0, x - edge):x + edge + 1]
        if int(window.sum()) > 1:
            continue
        candidates.append((float(x), float(y)))
        if len(candidates) >= max_stars * 3:
            break

    if not candidates:
        return []
    if not reject_extended or len(candidates) < 6:
        return candidates[:max_stars]

    # Measure sizes twice: the second-moment box has to be wide enough for
    # the actual seeing, or in poor seeing every source saturates the box
    # and stars stop being distinguishable from small galaxies.
    subtracted = data - median
    sizes = np.array([_second_moment_size(subtracted, cx, cy) for cx, cy in candidates])
    finite = np.isfinite(sizes)
    if finite.sum() < 6:
        return candidates[:max_stars]
    seed_size = float(np.median(np.sort(sizes[finite])[:max(4, int(0.4 * finite.sum()))]))
    box = int(max(9, min(31, 2 * np.ceil(3.5 * seed_size) + 1)))
    if box > 9:
        sizes = np.array([_second_moment_size(subtracted, cx, cy, box) for cx, cy in candidates])
        finite = np.isfinite(sizes)
        if finite.sum() < 6:
            return candidates[:max_stars]

    # The stellar locus is the lower envelope of the size distribution: all
    # point sources share one PSF, so their sizes cluster tightly, while
    # galaxies spread upward from it.  Seeding on the smallest quarter and
    # then iterating downward converges onto that locus even when galaxies
    # outnumber stars among the detections.
    valid = np.sort(sizes[finite])
    selected = valid[:max(3, int(0.10 * valid.size))]
    stellar = float(np.median(selected))
    limit = max(1.22 * stellar, stellar * 1.05)
    for _ in range(5):
        stellar = float(np.median(selected))
        spread = float(MAD_TO_SIGMA * np.median(np.abs(selected - stellar)))
        new_limit = max(1.22 * stellar, stellar + 4.0 * spread)
        refreshed = valid[valid <= new_limit]
        if refreshed.size < 3:
            limit = new_limit
            break
        converged = abs(new_limit - limit) < 1e-3 * max(stellar, 1e-6)
        limit, selected = new_limit, refreshed
        if converged:
            break
    keep = [c for c, s, ok in zip(candidates, sizes, finite) if ok and s <= limit]
    if len(keep) < 5:
        # Too few point sources to define an empirical PSF.  Say so rather
        # than silently averaging galaxies into the model: every profile
        # fit and star/galaxy separation downstream depends on this.
        log.warning("only %d point-source candidates survive the stellar-locus cut "
                    "(of %d detections); the PSF model will be poorly constrained",
                    len(keep), len(candidates))
    log.debug("PSF stars: %d candidates, stellar size %.2f px, kept %d",
              len(candidates), stellar, len(keep))
    return (keep or candidates)[:max_stars]


def _second_moment_size(data: np.ndarray, cx: float, cy: float,
                        box: int = 9) -> float:
    """Flux-weighted rms radius of a source, in pixels."""
    ny, nx = data.shape
    half = int(box) // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or y0 + box > ny or x0 + box > nx:
        return float("nan")
    stamp = np.clip(data[y0:y0 + box, x0:x0 + box], 0, None)
    total = float(stamp.sum())
    if total <= 0:
        return float("nan")
    yy, xx = np.mgrid[0:box, 0:box]
    mx = float((stamp * xx).sum() / total)
    my = float((stamp * yy).sum() / total)
    variance = float((stamp * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / total) / 2.0
    return float(np.sqrt(max(variance, 1e-6)))


def measure_fwhm(image: np.ndarray, positions: Optional[Sequence[Tuple[float, float]]] = None,
                 size: int = 21) -> float:
    """Median FWHM of the point sources in an image, in pixels."""
    data = as_float_image(image)
    stars = list(positions) if positions is not None else find_psf_stars(data)
    if not stars:
        return float("nan")
    _, median, _ = sigma_clipped_stats(data)
    values: List[float] = []
    half = int(size) // 2
    for x, y in stars:
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        if x0 < 0 or y0 < 0 or y0 + size > data.shape[0] or x0 + size > data.shape[1]:
            continue
        stamp = data[y0:y0 + size, x0:x0 + size] - median
        peak = float(stamp.max())
        if peak <= 0:
            continue
        radii, profile = radial_profile(stamp, ((x - x0), (y - y0)), nbins=size)
        finite = np.isfinite(profile)
        if finite.sum() < 3:
            continue
        # Interpolate where the azimuthal profile crosses half the peak.
        half_level = 0.5 * float(profile[finite][0])
        below = np.nonzero(finite & (profile < half_level))[0]
        if below.size == 0:
            continue
        i = int(below[0])
        if i == 0:
            continue
        r1, r0 = radii[i], radii[i - 1]
        p1, p0 = profile[i], profile[i - 1]
        frac = 0.0 if p0 == p1 else (p0 - half_level) / (p0 - p1)
        values.append(2.0 * float(r0 + frac * (r1 - r0)))
    if not values:
        return float("nan")
    return float(np.median(values))


def build_psf(image: np.ndarray, positions: Optional[Sequence[Tuple[float, float]]] = None,
              size: int = 25, rms: Optional[np.ndarray] = None) -> PSFModel:
    """Stack isolated stars into an empirical PSF model."""
    data = as_float_image(image)
    size = max(9, int(size) | 1)
    half = size // 2
    stars = list(positions) if positions is not None else find_psf_stars(data, rms=rms)
    _, median, noise = sigma_clipped_stats(data)

    stamps: List[np.ndarray] = []
    for x, y in stars:
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        if x0 < 0 or y0 < 0 or y0 + size > data.shape[0] or x0 + size > data.shape[1]:
            continue
        stamp = data[y0:y0 + size, x0:x0 + size] - median
        total = float(stamp.sum())
        if total <= 0:
            continue
        stamps.append(stamp / total)

    if not stamps:
        # No usable stars: fall back to a Gaussian at the nominal seeing.
        fwhm = 3.0
        log.warning("no PSF stars found; falling back to a %.1f px Gaussian", fwhm)
        return PSFModel(gaussian_kernel(fwhm / SIGMA_TO_FWHM, size), fwhm, n_stars=0, size=size)

    stack = np.median(np.stack(stamps), axis=0)
    stack = np.clip(stack, 0, None)
    total = stack.sum()
    if total > 0:
        stack /= total

    fwhm = measure_fwhm(data, stars, size=size)
    if not np.isfinite(fwhm):
        fwhm = _fwhm_from_stamp(stack)
    ellipticity, angle = _stamp_shape(stack)
    log.debug("PSF from %d stars: fwhm=%.2f px ellipticity=%.3f", len(stamps), fwhm, ellipticity)
    return PSFModel(stack, float(fwhm), float(ellipticity), float(angle),
                    n_stars=len(stamps), size=size)


def _fwhm_from_stamp(stamp: np.ndarray) -> float:
    """Second-moment FWHM of a normalised PSF stamp."""
    weights = np.clip(stamp, 0, None)
    total = weights.sum()
    if total <= 0:
        return 3.0
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    var = float((weights * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / total) / 2.0
    return float(SIGMA_TO_FWHM * np.sqrt(max(var, 1e-6)))


def _stamp_shape(stamp: np.ndarray) -> Tuple[float, float]:
    """Ellipticity and position angle from the stamp's second moments."""
    weights = np.clip(stamp, 0, None)
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    dx, dy = xx - cx, yy - cy
    mxx = float((weights * dx * dx).sum() / total)
    myy = float((weights * dy * dy).sum() / total)
    mxy = float((weights * dx * dy).sum() / total)
    common = np.sqrt(max(((mxx - myy) / 2.0) ** 2 + mxy ** 2, 0.0))
    mean = (mxx + myy) / 2.0
    major = np.sqrt(max(mean + common, 1e-9))
    minor = np.sqrt(max(mean - common, 1e-9))
    ellipticity = 1.0 - minor / major if major > 0 else 0.0
    angle = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))
    return float(ellipticity), float(angle)


def matching_kernel(source_psf: PSFModel, target_psf: PSFModel,
                    size: Optional[int] = None) -> np.ndarray:
    """Kernel that convolves ``source_psf`` into ``target_psf``.

    For difference imaging the sharper epoch must be degraded to match the
    blurrier one.  When both PSFs are approximately Gaussian this reduces
    to a Gaussian of the quadrature-difference width, which is stable and
    avoids the ringing that naive Fourier deconvolution produces.
    """
    target_sigma = max(target_psf.sigma, 1e-3)
    source_sigma = max(source_psf.sigma, 1e-3)
    if target_sigma <= source_sigma:
        # Source is already as blurry (or blurrier): no convolution needed.
        kernel = np.zeros((3, 3), dtype=float)
        kernel[1, 1] = 1.0
        return kernel
    sigma = float(np.sqrt(target_sigma ** 2 - source_sigma ** 2))
    return gaussian_kernel(sigma, size)


def match_psf(image: np.ndarray, source_psf: PSFModel,
              target_psf: PSFModel) -> np.ndarray:
    """Convolve ``image`` so its PSF matches ``target_psf``."""
    kernel = matching_kernel(source_psf, target_psf)
    if kernel.shape == (3, 3) and kernel[1, 1] == 1.0 and kernel.sum() == 1.0:
        return as_float_image(image)
    return convolve(image, kernel)
