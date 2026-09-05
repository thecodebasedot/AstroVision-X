"""The aperture correction from the field's own stars.

An aperture misses the light in the PSF's wings, and the correction for
that is the enclosed-energy curve of a point source. The PSF *model* gives
one, but it is a 25-pixel stamp stacked from a few stars: its wings are
truncated by its edge, noisy where few stars went in, and on a
photographic plate different for bright and faint stars. The field itself
holds a better estimate. Bright, isolated, unsaturated stars measured out
to a radius well beyond any aperture give the enclosed fraction directly,
and the median over tens of them is what survey pipelines call the
growth-curve correction. This module builds that curve and answers
``correction(radius)`` from it; the photometer uses it when the field has
enough stars and the PSF model otherwise, and says which in its report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.logging import get_logger
from .aperture import annulus_background
from .growth import curve_of_growth

log = get_logger("photometry.growthcurve")


@dataclass
class GrowthCurve:
    """Median enclosed-flux fraction of the field's stars versus radius."""

    radii: np.ndarray
    enclosed: np.ndarray                       # median fraction of the flux inside far_radius
    scatter: np.ndarray                        # 1.4826 MAD across stars, per radius
    far_radius: float
    n_stars: int
    star_ids: List[int] = field(default_factory=list)
    n_candidates: int = 0
    source: str = "field_stars"
    #: Fraction of a star's light estimated to lie beyond ``far_radius``.
    missing_beyond: float = 0.0

    def correction(self, radius: float) -> float:
        """Multiplicative correction for an aperture of ``radius`` pixels."""
        if not np.isfinite(radius) or radius <= 0 or self.n_stars == 0:
            return 1.0
        if radius >= self.far_radius:
            return float(1.0 / max(1.0 - self.missing_beyond, 0.3))
        fraction = float(np.interp(radius, self.radii, self.enclosed))
        if not 0.3 < fraction <= 1.0:
            return 1.0
        return float(1.0 / fraction)

    def uncertainty(self, radius: float) -> float:
        """Relative uncertainty of the correction at ``radius``: the stars' scatter."""
        if self.n_stars < 2 or not np.isfinite(radius) or radius >= self.far_radius:
            return 0.0
        fraction = float(np.interp(radius, self.radii, self.enclosed))
        spread = float(np.interp(radius, self.radii, self.scatter))
        return float(spread / max(fraction, 1e-6))

    def to_dict(self) -> Dict[str, Any]:
        return {"far_radius": float(self.far_radius), "n_stars": int(self.n_stars),
                "missing_beyond": float(self.missing_beyond),
                "n_candidates": int(self.n_candidates), "source": self.source,
                "radii": [float(r) for r in self.radii],
                "enclosed": [float(e) for e in self.enclosed],
                "scatter": [float(s) for s in self.scatter]}


def _missing_beyond(radii: np.ndarray, enclosed: np.ndarray, max_missing: float = 0.08) -> float:
    """Fraction of the light beyond the last radius, from the outer curve's slope.

    Fits ``log(1 - E) = log a - p log r`` over the outer third of the
    radii.  A slope ``p`` under 0.3 means the curve has not turned over and
    nothing can be said; the answer is then 0, and so is it when the fit
    would claim more than ``max_missing``, which is a neighbour, not a wing.
    """
    n = len(radii)
    outer = slice(max(2 * n // 3, 2), n)
    r = np.asarray(radii[outer], dtype=float)
    deficit = 1.0 - np.asarray(enclosed[outer], dtype=float)
    good = (deficit > 1e-4) & (r > 0)
    if good.sum() < 4:
        return 0.0
    slope, intercept = np.polyfit(np.log(r[good]), np.log(deficit[good]), 1)
    p = -float(slope)
    if not np.isfinite(p) or p < 0.3:
        return 0.0
    # Missing fraction beyond R for deficit a r^-p measured from the last
    # sample: F_beyond / F_total = deficit(R) itself, so use the fitted
    # curve at R rather than the noisy last point.
    at_far = float(np.exp(intercept) * r[-1] ** (-p))
    return float(np.clip(at_far, 0.0, max_missing))


def select_growth_stars(catalog, psf_fwhm: float, far_radius: float,
                        min_snr: float = 40.0, max_stars: int = 60,
                        shape: Optional[Sequence[int]] = None,
                        neighbour_fraction: float = 0.05) -> List[Any]:
    """Bright, point-like, isolated, unsaturated sources, brightest first.

    Uses what the detector measured -- isophotal flux, peak, width -- so it
    runs before photometry.  Isolation means no neighbour that matters: a
    detection within the far radius plus a margin whose flux is more than
    ``neighbour_fraction`` of the star's.  A crowded real field has faint
    detections everywhere (on a photographic plate, its grain), and a
    thousandth of the star's light in its wings is not what limits the
    curve; a companion at a twentieth is.
    """
    xs = np.array([s.x for s in catalog], dtype=float)
    ys = np.array([s.y for s in catalog], dtype=float)
    fluxes = np.array([s.photometry.flux for s in catalog], dtype=float)
    fluxes = np.where(np.isfinite(fluxes), fluxes, 0.0)
    chosen = []
    exclusion = far_radius + 4.0
    for index, source in enumerate(catalog):
        flags = source.flags
        if any(f in flags for f in ("saturated", "blended", "edge", "masked_pixels",
                                    "cosmic_ray", "artifact")):
            continue
        snr = float(source.photometry.snr)
        if not np.isfinite(snr) or snr < min_snr:
            continue
        width = float(source.morphology.fwhm)
        if psf_fwhm > 0 and np.isfinite(width) and width > 1.6 * psf_fwhm:
            continue                                     # resolved, or a blend
        if shape is not None:
            if (source.x < exclusion or source.y < exclusion
                    or source.x > shape[1] - 1 - exclusion
                    or source.y > shape[0] - 1 - exclusion):
                continue
        distance = np.hypot(xs - source.x, ys - source.y)
        distance[index] = np.inf
        own = float(source.photometry.flux) if np.isfinite(source.photometry.flux) else 0.0
        close = distance < exclusion
        if own <= 0 or np.any(fluxes[close] > neighbour_fraction * own):
            continue
        chosen.append((snr, source))
    chosen.sort(key=lambda item: -item[0])
    return [source for _, source in chosen[:max_stars]]


def build_growth_curve(data: np.ndarray, catalog, psf_fwhm: float,
                       far_radius: Optional[float] = None, min_stars: int = 5,
                       min_snr: float = 40.0, max_stars: int = 12,
                       n_radii: int = 40, far_factor: float = 5.0,
                       sky_annulus: Optional[Sequence[float]] = (2.0, 8.0)
                       ) -> Optional[GrowthCurve]:
    """The field's growth curve, or None when too few stars qualify.

    ``data`` is the background-subtracted image.  Each star's own sky is
    re-estimated in an annulus outside the far radius, because a residual
    of even a fraction of the noise per pixel, summed over a large
    aperture, is a bias of percent in the wings.  Only the brightest
    ``max_stars`` qualifying stars go in: the wings of a star at
    signal-to-noise 50 are below the noise, and a curve normalised by such
    a star's outer flux is biased high by the amount the noise took.
    """
    data = np.asarray(data, dtype=float)
    if far_radius is None:
        far_radius = max(far_factor * max(psf_fwhm, 1.0), 12.0)
    far_radius = float(min(far_radius, 0.2 * min(data.shape)))
    stars = select_growth_stars(catalog, psf_fwhm, far_radius, min_snr=min_snr,
                                max_stars=max_stars, shape=data.shape)
    n_candidates = len(stars)
    if len(stars) < min_stars:
        log.info("growth curve: %d qualifying star(s), fewer than %d; using the PSF model",
                 len(stars), min_stars)
        return None
    radii = np.linspace(1.0, far_radius, n_radii)
    curves, ids = [], []
    for star in stars:
        centre = (float(star.x), float(star.y))
        sky = 0.0
        if sky_annulus is not None:
            sky, _, n_sky = annulus_background(data, centre, far_radius + float(sky_annulus[0]),
                                               far_radius + float(sky_annulus[1]))
            if n_sky < 20 or not np.isfinite(sky):
                continue
        _, cumulative = curve_of_growth(data, centre, radii, background=sky)
        total = cumulative[-1]
        if not np.isfinite(total) or total <= 0:
            continue
        fraction = cumulative / total
        # A curve that turns down has a neighbour or a bad pixel in it.
        if np.any(np.diff(fraction) < -0.02) or fraction[0] <= 0:
            continue
        curves.append(fraction)
        ids.append(int(star.id))
    if len(curves) < min_stars:
        log.info("growth curve: %d clean star curve(s), fewer than %d; using the PSF model",
                 len(curves), min_stars)
        return None
    stack = np.array(curves)
    enclosed = np.median(stack, axis=0)
    scatter = 1.4826 * np.median(np.abs(stack - enclosed[None, :]), axis=0)
    enclosed = np.maximum.accumulate(np.clip(enclosed, 0.0, 1.0))
    # The wings do not stop at the far radius.  The light still missing
    # there follows from the shape of the outer curve: for any profile
    # with power-law wings the missing fraction 1 - E(r) falls as r^-p, and
    # a fit to the outer third of the curve gives what lies beyond.
    missing = _missing_beyond(radii, enclosed)
    enclosed = enclosed * (1.0 - missing)
    curve = GrowthCurve(radii=radii, enclosed=enclosed, scatter=scatter * (1.0 - missing),
                        far_radius=far_radius, n_stars=len(curves), star_ids=ids,
                        n_candidates=n_candidates, missing_beyond=float(missing))
    log.info("growth curve from %d stars to %.1f px: 50%% of the light inside %.2f px, "
             "90%% inside %.2f px", curve.n_stars, far_radius,
             float(np.interp(0.5, enclosed, radii)), float(np.interp(0.9, enclosed, radii)))
    return curve


__all__ = ["GrowthCurve", "build_growth_curve", "select_growth_stars"]
