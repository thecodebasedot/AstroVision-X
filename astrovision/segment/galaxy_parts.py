"""Galaxy component decomposition.

A galaxy is not one thing.  The pipeline separates a detected galaxy into
its nucleus, bulge, disc/arm region and outskirts using the isophotal
structure, then measures each part.  These per-component numbers feed the
morphological classifier and the lensing search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    as_float_image,
    elliptical_mask,
    gaussian_filter,
    radial_profile,
    safe_divide,
)

log = get_logger("segment.galaxy_parts")

#: Component names in radial order, from the centre outwards.
COMPONENTS = ("nucleus", "bulge", "disc", "outskirts")


@dataclass
class GalaxyComponents:
    """Per-component measurements for one galaxy."""

    labels: np.ndarray                              # 0=background, 1..4 = COMPONENTS
    fluxes: Dict[str, float] = field(default_factory=dict)
    areas: Dict[str, int] = field(default_factory=dict)
    radii: Dict[str, float] = field(default_factory=dict)
    ellipticities: Dict[str, float] = field(default_factory=dict)
    position_angles: Dict[str, float] = field(default_factory=dict)
    bulge_to_total: float = float("nan")
    nucleus_contrast: float = float("nan")
    isophote_twist: float = float("nan")
    arm_region_fraction: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fluxes": {k: float(v) for k, v in self.fluxes.items()},
            "areas": {k: int(v) for k, v in self.areas.items()},
            "radii": {k: float(v) for k, v in self.radii.items()},
            "ellipticities": {k: float(v) for k, v in self.ellipticities.items()},
            "position_angles": {k: float(v) for k, v in self.position_angles.items()},
            "bulge_to_total": float(self.bulge_to_total),
            "nucleus_contrast": float(self.nucleus_contrast),
            "isophote_twist": float(self.isophote_twist),
            "arm_region_fraction": float(self.arm_region_fraction),
        }


def ellipse_from_moments(data: np.ndarray, mask: np.ndarray
                         ) -> Tuple[float, float, float, float, float]:
    """Flux-weighted ``(cx, cy, semi_major, axis_ratio, pa_deg)`` of a footprint."""
    weights = np.where(mask, np.clip(np.nan_to_num(data), 0, None), 0.0)
    total = float(weights.sum())
    ny, nx = data.shape
    if total <= 0:
        return (nx - 1) / 2.0, (ny - 1) / 2.0, 1.0, 1.0, 0.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    dx, dy = xx - cx, yy - cy
    mxx = float((weights * dx * dx).sum() / total)
    myy = float((weights * dy * dy).sum() / total)
    mxy = float((weights * dx * dy).sum() / total)
    common = float(np.sqrt(max(((mxx - myy) / 2) ** 2 + mxy ** 2, 0.0)))
    mean = (mxx + myy) / 2
    major = float(np.sqrt(max(mean + common, 1e-9)))
    minor = float(np.sqrt(max(mean - common, 1e-9)))
    angle = float(0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy)))
    return cx, cy, major, float(minor / major) if major > 0 else 1.0, angle


def decompose(cutout: np.ndarray, mask: Optional[np.ndarray] = None,
              n_levels: int = 4, smooth: float = 0.8) -> GalaxyComponents:
    """Split a galaxy cutout into nucleus / bulge / disc / outskirts.

    The split is made on the *curve of growth*: component boundaries fall
    at fixed enclosed-flux fractions, which is robust to the wide range of
    galaxy sizes and surface brightnesses in a survey.
    """
    data = as_float_image(cutout)
    footprint = (np.isfinite(data) if mask is None else np.asarray(mask, dtype=bool))
    labels = np.zeros(data.shape, dtype=np.int32)
    result = GalaxyComponents(labels=labels)
    if not footprint.any():
        return result

    smoothed = gaussian_filter(data, smooth) if smooth > 0 else data
    cx, cy, semi_major, axis_ratio, pa = ellipse_from_moments(smoothed, footprint)

    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    theta = np.deg2rad(pa)
    xr = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    yr = (-(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)) / max(axis_ratio, 0.05)
    elliptical_r = np.hypot(xr, yr)

    positive = np.where(footprint, np.clip(np.nan_to_num(data), 0, None), 0.0)
    total_flux = float(positive.sum())
    if total_flux <= 0:
        return result

    # Curve of growth in elliptical radius -> radii enclosing 20/50/80 %.
    order = np.argsort(elliptical_r.ravel())
    cumulative = np.cumsum(positive.ravel()[order]) / total_flux
    sorted_r = elliptical_r.ravel()[order]
    boundaries = []
    for fraction in (0.2, 0.5, 0.8):
        index = int(np.searchsorted(cumulative, fraction))
        boundaries.append(float(sorted_r[min(index, len(sorted_r) - 1)]))
    r20, r50, r80 = boundaries

    limits = [max(r20 * 0.45, 1.0), max(r20, 1.5), max(r50, 2.5), max(r80, 3.5)]
    limits = np.maximum.accumulate(limits)
    for index, limit in enumerate(limits, start=1):
        region = footprint & (elliptical_r <= limit) & (labels == 0)
        labels[region] = index

    result.labels = labels
    for index, name in enumerate(COMPONENTS, start=1):
        component = labels == index
        area = int(component.sum())
        result.areas[name] = area
        result.fluxes[name] = float(positive[component].sum()) if area else 0.0
        if area >= 6:
            _, _, major, ratio, angle = ellipse_from_moments(smoothed, component)
            result.radii[name] = float(major)
            result.ellipticities[name] = float(1.0 - ratio)
            result.position_angles[name] = float(angle)
        else:
            result.radii[name] = float(limits[index - 1])
            result.ellipticities[name] = float("nan")
            result.position_angles[name] = float("nan")

    inner = result.fluxes.get("nucleus", 0.0) + result.fluxes.get("bulge", 0.0)
    result.bulge_to_total = float(inner / total_flux) if total_flux > 0 else float("nan")

    # Nucleus contrast: how much brighter the centre is than the mean disc.
    nucleus_area = max(result.areas.get("nucleus", 0), 1)
    disc_area = max(result.areas.get("disc", 0), 1)
    nucleus_sb = result.fluxes.get("nucleus", 0.0) / nucleus_area
    disc_sb = result.fluxes.get("disc", 0.0) / disc_area
    result.nucleus_contrast = float(safe_divide(nucleus_sb, max(disc_sb, 1e-12), fill=np.nan))

    # Isophote twist: a changing position angle with radius signals a bar
    # or an interaction, and is a classic morphological discriminator.
    angles = [result.position_angles.get(name) for name in COMPONENTS]
    angles = [a for a in angles if a is not None and np.isfinite(a)]
    if len(angles) >= 2:
        differences = [abs((angles[i + 1] - angles[i] + 90) % 180 - 90)
                       for i in range(len(angles) - 1)]
        result.isophote_twist = float(np.max(differences))

    disc_flux = result.fluxes.get("disc", 0.0) + result.fluxes.get("outskirts", 0.0)
    result.arm_region_fraction = float(disc_flux / total_flux) if total_flux > 0 else float("nan")
    return result


def component_profile(cutout: np.ndarray, mask: Optional[np.ndarray] = None,
                      nbins: int = 24) -> Dict[str, np.ndarray]:
    """Azimuthally averaged surface-brightness profile of one object."""
    data = as_float_image(cutout)
    footprint = np.isfinite(data) if mask is None else np.asarray(mask, dtype=bool)
    cx, cy, _, _, _ = ellipse_from_moments(data, footprint)
    radii, profile = radial_profile(np.where(footprint, data, np.nan), (cx, cy), nbins)
    return {"radius": radii, "surface_brightness": profile}


def annulus_masks(shape: Tuple[int, int], centre: Tuple[float, float],
                  radii: List[float], axis_ratio: float = 1.0,
                  pa_deg: float = 0.0) -> List[np.ndarray]:
    """Nested elliptical annuli, used for isophotal and lensing analysis."""
    masks: List[np.ndarray] = []
    previous = np.zeros(shape, dtype=bool)
    for radius in radii:
        current = elliptical_mask(shape, centre, radius,
                                  radius * max(axis_ratio, 0.05), pa_deg)
        masks.append(current & ~previous)
        previous = current
    return masks
