"""Spiral-arm and bar detection.

Transforming a galaxy into polar coordinates turns logarithmic spiral arms
into straight, periodic ridges.  A Fourier decomposition in azimuth then
measures how many arms there are and how strong they are, and the same
decomposition detects bars from the ``m = 2`` amplitude at small radii.
This is the classical Fourier-mode analysis used in galaxy morphology.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image, nan_to_finite

log = get_logger("morphology.spiral")


def polar_transform(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                    n_radial: int = 64, n_angular: int = 128,
                    max_radius: Optional[float] = None,
                    axis_ratio: float = 1.0, position_angle: float = 0.0,
                    log_radial: bool = True) -> Dict[str, np.ndarray]:
    """Resample an object onto a ``(radius, azimuth)`` grid.

    Deprojecting by ``axis_ratio``/``position_angle`` first is essential:
    an inclined disc otherwise produces a spurious ``m = 2`` signal that
    mimics a bar.  A logarithmic radial axis makes logarithmic spirals map
    to straight lines, which is what the mode analysis assumes.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    ny, nx = data.shape
    if centre is None:
        centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    if max_radius is None:
        max_radius = max(min(ny, nx) / 2.0 - 1.0, 4.0)

    r_min = 1.0
    if log_radial:
        radii = np.geomspace(r_min, max(float(max_radius), r_min * 2), int(n_radial))
    else:
        radii = np.linspace(r_min, float(max_radius), int(n_radial))
    angles = np.linspace(0, 2 * np.pi, int(n_angular), endpoint=False)

    rr, tt = np.meshgrid(radii, angles, indexing="ij")
    theta = np.deg2rad(position_angle)
    q = float(np.clip(axis_ratio, 0.05, 1.0))
    # Sample in the deprojected frame, then rotate back into image pixels.
    xr = rr * np.cos(tt)
    yr = rr * np.sin(tt) * q
    src_x = centre[0] + xr * np.cos(theta) - yr * np.sin(theta)
    src_y = centre[1] + xr * np.sin(theta) + yr * np.cos(theta)

    polar = _bilinear(data, src_x, src_y)
    return {"polar": polar, "radii": radii, "angles": angles}


def _bilinear(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sampling with zero outside the array."""
    ny, nx = data.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0c = np.clip(x0, 0, nx - 2)
    y0c = np.clip(y0, 0, ny - 2)
    top = data[y0c, x0c] * (1 - wx) + data[y0c, x0c + 1] * wx
    bottom = data[y0c + 1, x0c] * (1 - wx) + data[y0c + 1, x0c + 1] * wx
    return np.where(inside, top * (1 - wy) + bottom * wy, 0.0)


def fourier_modes(polar: np.ndarray, max_mode: int = 6) -> Dict[str, np.ndarray]:
    """Azimuthal Fourier amplitudes ``A_m(r) / A_0(r)`` for each radius.

    ``A_2`` dominated by a wide radial range indicates a two-armed spiral
    or a bar; ``A_3``/``A_4`` pick out three- and four-armed patterns.
    """
    data = np.asarray(polar, dtype=float)
    spectrum = np.fft.rfft(data, axis=1)
    amplitudes = np.abs(spectrum)
    mean = amplitudes[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(mean[:, None] > 1e-12,
                              amplitudes / np.maximum(mean[:, None], 1e-12), 0.0)
    n_modes = min(int(max_mode) + 1, normalised.shape[1])
    phases = np.angle(spectrum[:, :n_modes])
    return {"amplitude": normalised[:, :n_modes], "phase": phases, "mean": mean}


def detect_spiral_arms(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                       axis_ratio: float = 1.0, position_angle: float = 0.0,
                       max_radius: Optional[float] = None,
                       n_radial: int = 64, n_angular: int = 128,
                       inner_fraction: float = 0.2,
                       noise: float = 0.0,
                       min_winding: float = 1.2,
                       min_significance: float = 2.0) -> Dict[str, Any]:
    """Measure arm multiplicity, strength and pitch angle.

    Amplitude alone does not identify arms.  A smooth but flattened galaxy
    leaves an ``m = 2`` residual of a few hundredths whenever the assumed
    axis ratio is slightly wrong, and a bar produces a large ``m = 2`` as
    well.  What makes a spiral a spiral is that the mode's phase *winds*
    with radius: on a logarithmic radial axis a logarithmic spiral has a
    constant ``dphi/dln r = m / tan(pitch)``, whereas an ellipse or a bar
    holds its phase fixed.  Arms are therefore confirmed only when the
    amplitude clears the noise floor -- estimated from the high-order
    modes, which carry no two- to four-armed signal -- *and* the phase
    winds coherently with radius.
    """
    data = as_float_image(cutout)
    transform = polar_transform(data, centre, n_radial, n_angular, max_radius,
                                axis_ratio, position_angle, log_radial=True)
    polar = transform["polar"]
    radii = transform["radii"]
    modes = fourier_modes(polar, max_mode=8)
    amplitude = modes["amplitude"]
    mean_profile = modes["mean"] / max(polar.shape[1], 1)

    empty: Dict[str, Any] = {
        "arm_count": 0, "spiral_strength": 0.0, "pitch_angle": float("nan"),
        "coherence": 0.0, "mode_strengths": {}, "lopsidedness": 0.0,
        "noise_floor": float("nan"), "arm_significance": 0.0,
        "winding": 0.0, "winding_r2": 0.0, "n_annuli": 0,
    }

    # Restrict to the disc: the nucleus is smooth, the outskirts are noise.
    keep = np.zeros(len(radii), dtype=bool)
    keep[int(len(radii) * inner_fraction):int(len(radii) * 0.92)] = True
    if noise > 0:
        # Below about two sigma in the azimuthal mean the mode amplitudes
        # are pure noise, and including them only dilutes the measurement.
        significant = mean_profile > 2.0 * float(noise)
        if (keep & significant).sum() >= 6:
            keep &= significant
    if keep.sum() < 6:
        return empty

    strengths = {m: float(np.nanmean(amplitude[keep, m]))
                 for m in range(1, amplitude.shape[1])}
    high = [strengths[m] for m in strengths if m >= 5 and np.isfinite(strengths[m])]
    floor = float(np.median(high)) if high else 0.0

    arm_modes = {m: s for m, s in strengths.items() if 2 <= m <= 4}
    if not arm_modes:
        return empty

    best_mode, best_score = 0, 0.0
    best_metrics: Dict[str, float] = {}
    for mode, raw in arm_modes.items():
        significance = float(raw / floor) if floor > 1e-9 else (99.0 if raw > 0 else 0.0)
        winding, r2 = phase_winding(modes["phase"][keep, mode], radii[keep])
        if significance < min_significance or abs(winding) < min_winding or r2 < 0.45:
            continue
        # Rank candidate modes by how far each clears its own noise floor,
        # weighted by how cleanly its phase winds.
        score = (significance - min_significance) * r2
        if score > best_score:
            best_mode, best_score = mode, score
            best_metrics = {"significance": significance, "winding": winding,
                            "r2": r2, "raw": raw}

    diagnostics = {
        "mode_strengths": {f"m{m}": float(s) for m, s in strengths.items()},
        "noise_floor": float(floor),
        "lopsidedness": float(max(strengths.get(1, 0.0) - floor, 0.0)),
        "n_annuli": int(keep.sum()),
    }

    if best_mode == 0:
        # No mode both clears the floor and winds: keep the diagnostics but
        # assert that there is no arm pattern.
        strongest = max(arm_modes, key=arm_modes.get)
        raw = arm_modes[strongest]
        return {**empty, **diagnostics,
                "arm_significance": float(raw / floor) if floor > 1e-9 else 0.0}

    winding = best_metrics["winding"]
    pitch = (float(np.degrees(np.arctan(abs(best_mode / winding))))
             if abs(winding) > 1e-9 else 90.0)

    column = amplitude[keep, best_mode]
    finite = column[np.isfinite(column)]
    coherence = 0.0
    if finite.size > 3 and finite.mean() > 0:
        coherence = float(np.clip(1.0 - finite.std() / finite.mean(), 0.0, 1.0))
        coherence *= float(best_metrics["r2"])

    return {
        **diagnostics,
        "arm_count": int(best_mode),
        "spiral_strength": float(max(best_metrics["raw"] - floor, 0.0)),
        "pitch_angle": pitch,
        "coherence": coherence,
        "arm_significance": float(min(best_metrics["significance"], 99.0)),
        "winding": float(winding),
        "winding_r2": float(best_metrics["r2"]),
    }


def phase_winding(phase: np.ndarray, radii: np.ndarray) -> Tuple[float, float]:
    """Fit ``phi = a + b ln r``; returns ``(b, R^2)``.

    ``b`` is the winding rate: zero for an ellipse or a bar, and
    ``m / tan(pitch)`` for a logarithmic spiral.  ``R^2`` reports how
    spiral-like that winding is.
    """
    values = np.asarray(phase, dtype=float)
    log_r = np.log(np.asarray(radii, dtype=float))
    if values.size < 5 or not np.all(np.isfinite(values)) or np.ptp(log_r) <= 0:
        return 0.0, 0.0
    unwrapped = np.unwrap(values)
    slope, intercept = np.polyfit(log_r, unwrapped, 1)
    model = slope * log_r + intercept
    residual = float(np.sum((unwrapped - model) ** 2))
    total = float(np.sum((unwrapped - unwrapped.mean()) ** 2))
    r2 = float(1.0 - residual / total) if total > 1e-12 else 0.0
    return float(slope), float(np.clip(r2, 0.0, 1.0))


def detect_bar(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
               axis_ratio: float = 1.0, position_angle: float = 0.0,
               max_radius: Optional[float] = None,
               n_radial: int = 64, n_angular: int = 128) -> Dict[str, Any]:
    """Detect a stellar bar from the inner ``m = 2`` Fourier mode.

    A bar shows a strong ``A_2`` whose phase stays nearly *constant* with
    radius -- unlike a spiral, where the phase winds steadily.  That phase
    behaviour is what separates the two, and it is measured here.
    """
    transform = polar_transform(cutout, centre, n_radial, n_angular, max_radius,
                                axis_ratio, position_angle, log_radial=False)
    modes = fourier_modes(transform["polar"], max_mode=4)
    amplitude = modes["amplitude"]
    if amplitude.shape[1] <= 2:
        return {"bar_strength": 0.0, "bar_length": 0.0, "bar_angle": float("nan"),
                "bar_detected": False}

    radii = transform["radii"]
    a2 = amplitude[:, 2]
    phase2 = np.unwrap(modes["phase"][:, 2]) / 2.0

    inner = slice(0, max(int(len(radii) * 0.55), 4))
    a2_inner = a2[inner]
    if a2_inner.size == 0 or not np.isfinite(a2_inner).any():
        return {"bar_strength": 0.0, "bar_length": 0.0, "bar_angle": float("nan"),
                "bar_detected": False}

    peak_index = int(np.nanargmax(a2_inner))
    peak_strength = float(a2_inner[peak_index])

    # Bar length: where A_2 drops to half its peak beyond the peak radius.
    half = 0.5 * peak_strength
    length = float(radii[peak_index])
    for index in range(peak_index, len(a2)):
        if a2[index] < half:
            length = float(radii[index])
            break

    # Phase constancy over the bar region is the discriminating test.
    region = slice(max(peak_index - 3, 0), min(peak_index + 6, len(phase2)))
    phase_spread = float(np.degrees(np.std(phase2[region]))) if region.stop > region.start else 180.0
    constant_phase = phase_spread < 15.0

    bar_angle = float(np.degrees(np.median(phase2[region])) + position_angle) % 180.0
    detected = bool(peak_strength > 0.2 and constant_phase and length > 2.0)
    return {
        "bar_strength": peak_strength if detected else float(peak_strength * 0.5),
        "bar_length": length,
        "bar_angle": bar_angle,
        "bar_phase_spread": phase_spread,
        "bar_detected": detected,
    }
