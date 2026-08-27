"""Trails: what a moving object looks like within a single exposure.

An object that moves while the shutter is open does not leave a point.  It
leaves the PSF smeared along the track it took -- a streak with the PSF's
profile across its width and, to first order, a flat top along its length.

That gives a second, independent handle on a mover, and the independence is
the point.  Linking says "these detections lie on a line across epochs".  A
trail says "this one detection was moving *during* its own exposure", using
no information from any other frame.  When the two agree on a direction, a
coincidental alignment of unrelated sources is a far worse explanation than
it was before.

The measurement is a comparison, not an absolute: an elongation is only
evidence if it exceeds what the *field's stars* show, since seeing is often
slightly elliptical and every source shares that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import as_float_image

log = get_logger("moving.trail")


@dataclass
class TrailMeasurement:
    """Second-moment shape of one source, relative to the field's PSF."""

    length: float = float("nan")          # FWHM-equivalent along the major axis
    width: float = float("nan")           # and across it
    angle: float = float("nan")           # degrees CCW from +x, 0-180
    elongation: float = float("nan")      # major / minor
    excess: float = float("nan")          # length in excess of the PSF, pixels
    significance: float = float("nan")    # excess relative to its own noise
    trailed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": float(self.length), "width": float(self.width),
            "angle": float(self.angle), "elongation": float(self.elongation),
            "excess": float(self.excess), "significance": float(self.significance),
            "trailed": bool(self.trailed),
        }


def expected_trail_length(rate_arcsec_per_hour: float, exposure_seconds: float,
                          pixel_scale: float) -> float:
    """How long a trail an object at this rate leaves, in pixels.

    >>> round(expected_trail_length(60.0, 300.0, 0.4), 3)
    12.5
    """
    if not all(np.isfinite([rate_arcsec_per_hour, exposure_seconds, pixel_scale])):
        return float("nan")
    if pixel_scale <= 0:
        return float("nan")
    arcsec = float(rate_arcsec_per_hour) * float(exposure_seconds) / 3600.0
    return float(arcsec / float(pixel_scale))


def second_moments(cutout: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                   background: float = 0.0,
                   window_radius: float = float("nan")
                   ) -> Tuple[float, float, float]:
    """Flux-weighted ``(major, minor, angle)`` of a stamp.

    Sizes are returned as FWHM-equivalents -- ``2.355 * sigma`` -- but note
    that for a Moffat profile that number is well above the profile's actual
    FWHM, because the wings carry real weight in a second moment.  Compare
    these sizes with *each other*, never with a fitted FWHM.

    ``window_radius`` bounds the measurement.  Without it a second moment is
    taken over the whole stamp, and on a faint source that means the *stamp*
    is being measured rather than the source: clipping the negative half of
    the noise leaves a positive floor spread over every pixel, whose second
    moment is the stamp's own size.  Measured against truth, an unbounded
    moment reported a 41-pixel "trail" on a round point source.

    Negative pixels are clipped away first: a difference-image stamp has
    negative noise, and weighting a moment by a negative flux moves the
    centroid outward and inflates the width.
    """
    data = np.clip(as_float_image(cutout) - float(background), 0.0, None)
    total = float(data.sum())
    if total <= 0 or data.size == 0:
        return float("nan"), float("nan"), float("nan")
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if centre is None:
        cx = float((data * xx).sum() / total)
        cy = float((data * yy).sum() / total)
    else:
        cx, cy = float(centre[0]), float(centre[1])
    if np.isfinite(window_radius) and window_radius > 0:
        inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(window_radius) ** 2
        data = np.where(inside, data, 0.0)
        total = float(data.sum())
        if total <= 0:
            return float("nan"), float("nan"), float("nan")
        # Re-centre inside the window, since the first pass was pulled by
        # whatever lay outside it.
        cx = float((data * xx).sum() / total)
        cy = float((data * yy).sum() / total)
    dx, dy = xx - cx, yy - cy
    mxx = float((data * dx * dx).sum() / total)
    myy = float((data * dy * dy).sum() / total)
    mxy = float((data * dx * dy).sum() / total)

    common = 0.5 * (mxx + myy)
    spread = math.sqrt(max(0.25 * (mxx - myy) ** 2 + mxy ** 2, 0.0))
    major_variance = max(common + spread, 0.0)
    minor_variance = max(common - spread, 0.0)
    angle = 0.5 * math.degrees(math.atan2(2.0 * mxy, mxx - myy)) % 180.0
    factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    return (factor * math.sqrt(major_variance), factor * math.sqrt(minor_variance),
            angle)


def measure_trail(cutout: np.ndarray, psf_fwhm: float,
                  centre: Optional[Tuple[float, float]] = None,
                  noise: float = float("nan"),
                  background: float = 0.0,
                  min_excess: float = 1.0,
                  min_significance: float = 3.0,
                  min_elongation: float = 1.25,
                  field_elongation: float = 1.0,
                  window_factor: float = 3.0) -> TrailMeasurement:
    """Decide whether a stamp is trailed, and by how much.

    The trail is measured as the source's own **major-against-minor** excess,
    not as its size against a fitted PSF FWHM.  That choice is forced: a
    second moment of a Moffat is far larger than the Moffat's FWHM, because
    the wings carry weight a FWHM ignores, so subtracting one from the other
    reports a multi-pixel trail on a perfectly round star.  Comparing an
    object's two axes against each other cancels the profile shape entirely,
    and cancels the seeing with it.

    ``field_elongation`` is the median axis ratio of the field's own point
    sources.  Tracking errors and elliptical seeing stretch *every* source in
    a frame; only elongation beyond that shared amount is evidence of motion.

    Three conditions, and all of them are needed.  The **elongation** must
    exceed ``min_elongation``, because a trail is by definition an elongation
    and nothing else is: a faint, noise-dominated source has two large but
    nearly equal axes, whose quadrature difference is still several pixels
    even though the source is round to 2%.  The absolute **excess** must
    clear ``min_excess``, because a 0.3-pixel trail on a bright source is
    highly significant and still is not motion.  And the excess must be
    **significant**, or a faint source is called trailed on its own noise.
    """
    window = (float(window_factor) * float(psf_fwhm)
              if np.isfinite(psf_fwhm) and psf_fwhm > 0 else float("nan"))
    major, minor, angle = second_moments(cutout, centre, background, window)
    result = TrailMeasurement(length=major, width=minor, angle=angle)
    if not np.isfinite(major) or not np.isfinite(minor) or minor <= 0:
        return result
    result.elongation = float(major / minor)

    # The width the source would have had if it were as round as the field's
    # stars; anything longer than that is the trail.
    baseline = minor * max(float(field_elongation), 1.0)
    excess_squared = major ** 2 - baseline ** 2
    result.excess = float(math.sqrt(excess_squared)) if excess_squared > 0 else 0.0

    # The uncertainty on a second-moment size goes roughly as the size divided
    # by the signal-to-noise, which is the standard first-order result and is
    # enough to keep a faint, noisy source from being called trailed on the
    # strength of its own noise.
    data = np.clip(as_float_image(cutout) - float(background), 0.0, None)
    flux = float(data.sum())
    if np.isfinite(noise) and noise > 0 and flux > 0:
        n_effective = math.pi * window ** 2 if np.isfinite(window) else data.size
        snr = flux / (noise * math.sqrt(max(n_effective, 1.0)))
        sigma_size = float(major) / max(snr, 1e-6)
        result.significance = float(result.excess / max(sigma_size, 1e-9))
    else:
        result.significance = float("nan")

    result.trailed = bool(
        result.elongation >= float(min_elongation) * max(float(field_elongation), 1.0)
        and result.excess >= float(min_excess)
        and (not np.isfinite(result.significance)
             or result.significance >= float(min_significance)))
    return result


def field_psf_elongation(stamps: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Median elongation and position angle of the field's point sources.

    Seeing is rarely perfectly round, and tracking errors elongate *every*
    source in a frame in the same direction.  An asteroid trail is only
    evidence of motion if it exceeds that shared elongation, so this is what a
    trail measurement should be compared against rather than a circle.
    """
    elongations, angles = [], []
    for stamp in stamps:
        major, minor, angle = second_moments(stamp)
        if not np.isfinite(major) or minor <= 0:
            continue
        elongations.append(major / minor)
        angles.append(math.radians(2.0 * angle))
    if not elongations:
        return float("nan"), float("nan")
    # Angles are averaged as unit vectors at twice the angle, because a
    # position angle is defined modulo 180 degrees and 179 and 1 are close.
    mean_angle = 0.5 * math.degrees(math.atan2(
        float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))) % 180.0
    return float(np.median(elongations)), float(mean_angle)


def direction_agreement(trail_angle: float, track_heading: float) -> float:
    """How well a trail's direction matches a tracklet's, in ``[0, 1]``.

    A trail has no sense of direction -- it is a streak, not an arrow -- so
    the comparison is modulo 180 degrees.  1 is perfect alignment, 0 is
    perpendicular.

    >>> direction_agreement(30.0, 210.0)
    1.0
    >>> direction_agreement(0.0, 90.0)
    0.0
    """
    if not (np.isfinite(trail_angle) and np.isfinite(track_heading)):
        return float("nan")
    difference = abs((float(trail_angle) - float(track_heading)) % 180.0)
    difference = min(difference, 180.0 - difference)
    return float(1.0 - difference / 90.0)
