"""Arc and ring detection for strong gravitational lensing.

A galaxy that lenses a background source produces images of it stretched
*tangentially* around the deflector -- arcs at a characteristic radius, or a
complete Einstein ring.  The signature is geometric: elongated features
whose long axis is perpendicular to the radius vector, all at about the same
distance from the centre.  That is what this module measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    as_float_image,
    gaussian_filter,
    nan_to_finite,
    sigma_clipped_stats,
)
from ..detect.labeling import label, remove_small
from ..morphology.spiral import polar_transform

log = get_logger("lensing.arcs")


@dataclass
class Arc:
    """One tangentially-elongated feature around a candidate deflector."""

    radius: float                  # distance from the deflector centre, px
    angle: float                   # position angle of the arc centre, degrees
    length: float                  # arc length along the tangential direction
    width: float                   # radial thickness
    axis_ratio: float              # length / width
    tangential_alignment: float    # 1 = perfectly tangential, 0 = radial
    peak_significance: float
    flux: float
    area: int
    #: Flux-weighted ridge positions along the arc, in cutout pixels.  Kept
    #: because a mass model needs the arc's actual *shape*: points rebuilt
    #: from ``radius`` and ``angle`` lie on a perfect circle, which a round
    #: lens reproduces exactly, so a fit given those can only ever measure a
    #: radius and reports every lens as circular.
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def to_dict(self) -> Dict[str, Any]:
        return {"radius": float(self.radius), "angle": float(self.angle),
                "length": float(self.length), "width": float(self.width),
                "axis_ratio": float(self.axis_ratio),
                "tangential_alignment": float(self.tangential_alignment),
                "peak_significance": float(self.peak_significance),
                "flux": float(self.flux), "area": int(self.area)}


def subtract_smooth_light(cutout: np.ndarray, centre: Tuple[float, float],
                          n_bins: int = 40, percentile: float = 25.0) -> np.ndarray:
    """Remove the deflector's own smooth light with an azimuthal baseline.

    The lensing galaxy is far brighter than the arcs it produces, and it is
    smooth in azimuth while the arcs are not -- so a per-radius baseline
    removes the galaxy and leaves the arcs.  The baseline is a *low
    percentile*, not the median: several arcs can fill most of the azimuth
    at the Einstein radius, and a median baseline would then subtract the
    very signal being searched for.  The default of 15 % tolerates arcs
    covering up to about five-sixths of the circle; a genuinely complete
    Einstein ring is instead picked up by :func:`ring_completeness`, which
    measures how uniformly filled that radius is.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(xx - centre[0], yy - centre[1])
    rmax = float(radius.max())
    if rmax <= 0:
        return data

    edges = np.linspace(0, rmax, max(4, int(n_bins) + 1))
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, len(edges) - 2)
    values = data.ravel()
    model = np.zeros_like(values)
    for b in range(len(edges) - 1):
        member = index == b
        if member.sum() >= 4:
            model[member] = float(np.percentile(values[member], percentile))
    return (values - model).reshape(data.shape)


def _ridge_points(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray,
                  centre: Tuple[float, float], n_bins: int = 9) -> np.ndarray:
    """Flux-weighted centre of the arc at several points along its length.

    Binned by azimuth about the deflector, because that is the direction an
    arc runs: within one bin the arc is a short radial profile whose weighted
    mean is its ridge.  The result traces the arc's real curvature, which is
    what tells an elliptical mass distribution from a round one.
    """
    angles = np.degrees(np.arctan2(ys - centre[1], xs - centre[0]))
    # Unwrap about the arc's own mean so a segment crossing +-180 stays whole.
    reference = float(np.degrees(np.arctan2(
        float((weights * np.sin(np.radians(angles))).sum()),
        float((weights * np.cos(np.radians(angles))).sum()))))
    relative = (angles - reference + 180.0) % 360.0 - 180.0
    edges = np.linspace(relative.min(), relative.max(), int(n_bins) + 1)
    points = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (relative >= low) & (relative <= high)
        total = float(weights[inside].sum())
        if total <= 0 or inside.sum() < 2:
            continue
        points.append((float((weights[inside] * xs[inside]).sum() / total),
                       float((weights[inside] * ys[inside]).sum() / total)))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def detect_arcs(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                noise: Optional[float] = None, threshold_sigma: float = 2.5,
                min_area: int = 8, min_axis_ratio: float = 2.5,
                max_width: float = 6.0, min_radius: float = 3.0) -> List[Arc]:
    """Find tangentially-elongated residuals around ``centre``.

    ``min_radius`` excludes the deflector's own core, where an imperfect
    smooth-light model always leaves residuals; set it from the deflector's
    size rather than leaving the default when one is known.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)

    residual = subtract_smooth_light(data, centre)
    if noise is None:
        _, _, noise = sigma_clipped_stats(residual)
    noise = max(float(noise), 1e-9)

    smoothed = gaussian_filter(residual, 1.0)
    above = smoothed > threshold_sigma * noise
    # Ignore the very centre, where residuals from an imperfect smooth-light
    # model are common and no lensed image can be resolved anyway.
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius_map = np.hypot(xx - centre[0], yy - centre[1])
    above &= radius_map > max(float(min_radius), 2.0)
    if not above.any():
        return []

    segmentation, count = label(above)
    segmentation, count = remove_small(segmentation, min_area, count)
    if count == 0:
        return []

    arcs: List[Arc] = []
    for value in range(1, count + 1):
        footprint = segmentation == value
        area = int(footprint.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(footprint)
        weights = np.clip(residual[footprint], 0, None)
        total = float(weights.sum())
        if total <= 0:
            continue
        cx = float((weights * xs).sum() / total)
        cy = float((weights * ys).sum() / total)

        dx, dy = cx - centre[0], cy - centre[1]
        arc_radius = float(np.hypot(dx, dy))
        if arc_radius < max(float(min_radius), 2.0):
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Shape from second moments in the arc's own frame.
        ddx, ddy = xs - cx, ys - cy
        mxx = float((weights * ddx * ddx).sum() / total)
        myy = float((weights * ddy * ddy).sum() / total)
        mxy = float((weights * ddx * ddy).sum() / total)
        common = np.sqrt(max(((mxx - myy) / 2) ** 2 + mxy ** 2, 0.0))
        mean = (mxx + myy) / 2
        major = float(np.sqrt(max(mean + common, 1e-9)))
        minor = float(np.sqrt(max(mean - common, 1e-9)))
        axis_ratio = float(major / max(minor, 1e-9))
        if axis_ratio < min_axis_ratio or minor > max_width:
            continue
        # An arc cannot be longer than the circle it lies on; a segment that
        # is has merged with something else and is not a lensed image.
        if 4.0 * major > 1.1 * 2.0 * np.pi * arc_radius:
            continue

        # The decisive test: is the elongation tangential?  A radial streak
        # of the same shape is a diffraction spike or a merging companion.
        major_angle = 0.5 * np.arctan2(2 * mxy, mxx - myy)
        radial_angle = np.arctan2(dy, dx)
        difference = abs((np.degrees(major_angle - radial_angle) + 90) % 180 - 90)
        alignment = float(difference / 90.0)
        if alignment < 0.5:
            continue

        arcs.append(Arc(
            points=_ridge_points(xs, ys, weights, centre),
            radius=arc_radius, angle=angle,
            length=float(4.0 * major), width=float(2.0 * minor),
            axis_ratio=axis_ratio, tangential_alignment=alignment,
            peak_significance=float(residual[footprint].max() / noise),
            flux=total, area=area))

    arcs.sort(key=lambda a: -a.length)
    log.debug("found %d tangential arc candidates", len(arcs))
    return arcs


def ring_completeness(cutout: np.ndarray, centre: Tuple[float, float],
                      radius: float, width: float = 2.5,
                      n_bins: int = 72, noise: Optional[float] = None,
                      percentile: float = 25.0) -> Dict[str, float]:
    """How much of a ring at ``radius`` is actually filled with light.

    A complete Einstein ring has flux at every position angle; a pair of
    arcs covers perhaps a third of the circle.  The fraction is one of the
    strongest discriminators between a lens and a chance alignment.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    residual = subtract_smooth_light(data, centre, percentile=percentile)
    if noise is None:
        _, _, noise = sigma_clipped_stats(residual)
    noise = max(float(noise), 1e-9)

    transform = polar_transform(residual, centre, n_radial=24, n_angular=int(n_bins),
                                max_radius=max(radius * 1.6, radius + 2 * width),
                                log_radial=False)
    radii = transform["radii"]
    band = np.abs(radii - radius) <= max(width, 1.0)
    if band.sum() == 0:
        return {"completeness": 0.0, "mean_significance": 0.0, "uniformity": 0.0}

    profile = transform["polar"][band].mean(axis=0)
    filled = profile > 1.5 * noise
    completeness = float(filled.mean())
    positive = profile[profile > 0]
    uniformity = float(1.0 - min(np.std(positive) / max(np.mean(positive), 1e-9), 1.0)) \
        if positive.size > 2 else 0.0
    return {
        "completeness": completeness,
        "mean_significance": float(profile.mean() / noise),
        "uniformity": uniformity,
        "n_filled_bins": float(filled.sum()),
    }


def einstein_radius(arcs: List[Arc]) -> Tuple[float, float]:
    """Estimate the Einstein radius and its scatter from the arcs' radii.

    Multiple images of the same source all form near the Einstein radius,
    so several arcs agreeing on one radius is itself lensing evidence.
    """
    if not arcs:
        return float("nan"), float("nan")
    radii = np.array([a.radius for a in arcs], dtype=float)
    weights = np.array([a.flux for a in arcs], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(radii)
    mean = float(np.average(radii, weights=weights))
    scatter = float(np.sqrt(np.average((radii - mean) ** 2, weights=weights)))
    return mean, scatter


def detect_ring(cutout: np.ndarray, centre: Tuple[float, float],
                noise: Optional[float] = None, n_bins: int = 48,
                n_angular: int = 72, min_radius: float = 3.0
                ) -> Dict[str, float]:
    """Look for a complete Einstein ring in the radial profile.

    An azimuthal baseline cannot find a ring that fills its own radius --
    the ring becomes the baseline.  A ring does, however, show up as a
    *bump* in the azimuthally-averaged profile of a galaxy whose light
    otherwise falls monotonically, and that bump is uniformly filled in
    azimuth.  Both together are what this measures.
    """
    from ..core.numeric import median_filter

    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if noise is None:
        _, _, noise = sigma_clipped_stats(data)
    noise = max(float(noise), 1e-9)

    empty = {"ring_detected": False, "radius": float("nan"), "excess": 0.0,
             "significance": 0.0, "uniformity": 0.0}
    max_radius = min(nx, ny) / 2.0 - 2.0
    if max_radius <= min_radius + 2:
        return empty

    transform = polar_transform(data, centre, n_bins, n_angular, max_radius,
                                log_radial=False)
    polar = transform["polar"]
    radii = transform["radii"]
    profile = polar.mean(axis=1)

    # A galaxy's own profile falls monotonically; median-filtering over a
    # wide window follows that fall but not a narrow ring.
    baseline = median_filter(profile.reshape(1, -1), 9).ravel()
    excess = profile - baseline
    usable = radii > min_radius
    if usable.sum() < 5:
        return empty

    # The uncertainty on an azimuthal mean of n_angular samples.
    error = noise / np.sqrt(max(n_angular, 1))
    index = int(np.argmax(np.where(usable, excess, -np.inf)))
    significance = float(excess[index] / max(error, 1e-9))
    if significance < 4.0:
        return {**empty, "radius": float(radii[index]),
                "excess": float(excess[index]), "significance": significance}

    ring_profile = polar[index]
    filled = ring_profile > baseline[index] + 1.0 * noise
    uniformity = float(filled.mean())
    return {
        "ring_detected": bool(uniformity > 0.7),
        "radius": float(radii[index]),
        "excess": float(excess[index]),
        "significance": significance,
        "uniformity": uniformity,
    }
