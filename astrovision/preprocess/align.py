"""Image registration.

Difference imaging is unforgiving: a sub-pixel misalignment turns every
star into a dipole residual that looks exactly like a transient.  This
module aligns epochs either by cross-correlation (fast, translation only)
or by matching star patterns (robust to rotation and scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.numeric import as_float_image, sigma_clipped_stats
from .psf import find_psf_stars

log = get_logger("preprocess.align")


@dataclass
class Transform:
    """A 2-D similarity transform mapping source pixels onto the reference."""

    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0     # degrees
    scale: float = 1.0
    n_matches: int = 0
    rms: float = float("nan")
    method: str = "identity"

    @property
    def is_identity(self) -> bool:
        return (abs(self.dx) < 1e-3 and abs(self.dy) < 1e-3 and
                abs(self.rotation) < 1e-4 and abs(self.scale - 1.0) < 1e-6)

    def matrix(self) -> np.ndarray:
        """The 2x3 affine matrix form of this transform."""
        theta = np.deg2rad(self.rotation)
        c, s = np.cos(theta) * self.scale, np.sin(theta) * self.scale
        return np.array([[c, -s, self.dx], [s, c, self.dy]], dtype=float)

    def apply(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Map coordinates from the source frame into the reference frame."""
        m = self.matrix()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]

    def to_dict(self):
        return {"dx": self.dx, "dy": self.dy, "rotation": self.rotation,
                "scale": self.scale, "n_matches": self.n_matches,
                "rms": self.rms, "method": self.method}


def cross_correlation_shift(reference: np.ndarray, moving: np.ndarray,
                            upsample: int = 10,
                            max_shift: Optional[float] = None) -> Tuple[float, float]:
    """Translation between two images from their phase correlation.

    Returns ``(dx, dy)``: the shift that must be applied to ``moving`` to
    line it up with ``reference``.  Sub-pixel accuracy comes from a
    parabolic fit to the correlation peak.
    """
    ref = _prepare_for_correlation(reference)
    mov = _prepare_for_correlation(moving)
    if ref.shape != mov.shape:
        raise ValueError("cross-correlation requires images of identical shape")

    fft_ref = np.fft.fft2(ref)
    fft_mov = np.fft.fft2(mov)
    cross = fft_ref * np.conj(fft_mov)
    magnitude = np.abs(cross)
    cross = np.where(magnitude > 1e-12, cross / np.maximum(magnitude, 1e-12), 0)
    correlation = np.real(np.fft.ifft2(cross))
    correlation = np.fft.fftshift(correlation)

    ny, nx = correlation.shape
    if max_shift is not None:
        window = np.zeros_like(correlation, dtype=bool)
        cy, cx = ny // 2, nx // 2
        radius = int(max_shift)
        window[max(0, cy - radius):cy + radius + 1,
               max(0, cx - radius):cx + radius + 1] = True
        correlation = np.where(window, correlation, -np.inf)

    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    dy = float(peak[0] - ny // 2)
    dx = float(peak[1] - nx // 2)

    if upsample and upsample > 1:
        # Refine with an upsampled DFT evaluated in a small neighbourhood of
        # the peak (Guizar-Sicairos et al. 2008) -- far more accurate than a
        # parabolic fit to the phase-correlation surface.
        sub_dy, sub_dx = _upsampled_peak(fft_ref, fft_mov, dy, dx, int(upsample))
        return float(sub_dx), float(sub_dy)
    return dx, dy


def _upsampled_peak(fft_ref: np.ndarray, fft_mov: np.ndarray, dy: float, dx: float,
                    upsample: int) -> Tuple[float, float]:
    """Locate the correlation peak on a 1/``upsample``-pixel grid.

    Only a 1.5-pixel window around the integer peak is evaluated, using a
    matrix-multiply DFT, so the cost stays negligible.
    """
    ny, nx = fft_ref.shape
    cross = fft_ref * np.conj(fft_mov)
    magnitude = np.abs(cross)
    cross = np.where(magnitude > 1e-12, cross / np.maximum(magnitude, 1e-12), 0)

    span = 1.5
    steps = int(np.ceil(2 * span * upsample)) + 1
    offsets_y = dy + np.linspace(-span, span, steps)
    offsets_x = dx + np.linspace(-span, span, steps)

    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    # exp(+2i pi f . shift) evaluated only at the offsets we care about.
    kernel_y = np.exp(2j * np.pi * np.outer(offsets_y, fy))
    kernel_x = np.exp(2j * np.pi * np.outer(offsets_x, fx))
    surface = np.real(kernel_y @ cross @ kernel_x.T)

    index = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return float(offsets_y[index[0]]), float(offsets_x[index[1]])


def _prepare_for_correlation(image: np.ndarray) -> np.ndarray:
    """Background-subtract and apodise so edges do not dominate the FFT."""
    data = as_float_image(image)
    _, median, _ = sigma_clipped_stats(data)
    centred = np.nan_to_num(data - median, nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = centred.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    return centred * window


def _parabolic_offset(surface: np.ndarray, peak: Tuple[int, int], axis: int) -> float:
    """Sub-pixel peak offset from a 3-point parabola along one axis."""
    index = peak[axis]
    size = surface.shape[axis]
    if index <= 0 or index >= size - 1:
        return 0.0
    if axis == 0:
        left, centre, right = surface[index - 1, peak[1]], surface[index, peak[1]], surface[index + 1, peak[1]]
    else:
        left, centre, right = surface[peak[0], index - 1], surface[peak[0], index], surface[peak[0], index + 1]
    denominator = left - 2 * centre + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def match_star_patterns(ref_stars: Sequence[Tuple[float, float]],
                        moving_stars: Sequence[Tuple[float, float]],
                        tolerance: float = 2.0,
                        max_stars: int = 40) -> Transform:
    """Estimate a similarity transform by voting on star pair geometry.

    Every pair of stars defines a separation and an orientation.  Pairs
    with compatible separations vote for a rotation and scale; the winning
    vote fixes the linear part, a histogram of residual offsets fixes the
    translation, and the inliers are then refined by least squares.  Unlike
    cross-correlation this needs no prior alignment and tolerates rotation.
    """
    ref = np.asarray(ref_stars, dtype=float)[:max_stars]
    mov = np.asarray(moving_stars, dtype=float)[:max_stars]
    if len(ref) < 3 or len(mov) < 3:
        return Transform(method="insufficient_stars")

    def pair_features(points: np.ndarray):
        i, j = np.triu_indices(len(points), k=1)
        delta = points[j] - points[i]
        return i, j, np.hypot(delta[:, 0], delta[:, 1]), np.degrees(
            np.arctan2(delta[:, 1], delta[:, 0]))

    ri, rj, rd, ra = pair_features(ref)
    mi, mj, md, ma = pair_features(mov)
    if rd.size == 0 or md.size == 0:
        return Transform(method="insufficient_pairs")

    # Vote for the rotation using pairs of similar separation.
    rotations: List[float] = []
    scales: List[float] = []
    for k in range(len(md)):
        if md[k] < 8.0:
            continue
        close = np.nonzero(np.abs(rd - md[k]) < tolerance)[0]
        for c in close:
            rotations.append(float(ra[c] - ma[k]))
            scales.append(float(rd[c] / max(md[k], 1e-6)))
    if len(rotations) < 3:
        return Transform(method="no_pair_matches")

    angles = (np.asarray(rotations) + 180.0) % 360.0 - 180.0
    hist, edges = np.histogram(angles, bins=180, range=(-180, 180))
    best = int(np.argmax(hist))
    window = (angles >= edges[best] - 2.0) & (angles <= edges[best + 1] + 2.0)
    if window.sum() < 3:
        return Transform(method="weak_consensus")
    rotation = float(np.median(angles[window]))
    scale = float(np.median(np.asarray(scales)[window]))
    if not 0.8 < scale < 1.25:
        scale = 1.0

    # With the linear part known, the translation is the mode of the
    # offsets between every reference star and every rotated moving star.
    theta = np.deg2rad(rotation)
    rotated = np.column_stack([
        scale * (mov[:, 0] * np.cos(theta) - mov[:, 1] * np.sin(theta)),
        scale * (mov[:, 0] * np.sin(theta) + mov[:, 1] * np.cos(theta)),
    ])
    offsets = (ref[:, None, :] - rotated[None, :, :]).reshape(-1, 2)
    translation = _offset_mode(offsets, tolerance)
    if translation is None:
        return Transform(rotation=rotation, scale=scale, method="pattern_partial")

    # Refine: nearest-neighbour matching under the current transform.
    for _ in range(3):
        predicted = rotated + translation
        distance = np.hypot(ref[:, None, 0] - predicted[None, :, 0],
                            ref[:, None, 1] - predicted[None, :, 1])
        nearest = np.argmin(distance, axis=1)
        residual = distance[np.arange(len(ref)), nearest]
        inliers = residual < max(tolerance, 2.0)
        if inliers.sum() < 3:
            break
        translation = np.median(ref[inliers] - rotated[nearest[inliers]], axis=0)

    predicted = rotated + translation
    distance = np.hypot(ref[:, None, 0] - predicted[None, :, 0],
                        ref[:, None, 1] - predicted[None, :, 1])
    nearest = np.argmin(distance, axis=1)
    residual = distance[np.arange(len(ref)), nearest]
    inliers = residual < max(tolerance, 2.0)
    n_matches = int(inliers.sum())
    if n_matches < 3:
        return Transform(float(translation[0]), float(translation[1]), rotation, scale,
                         n_matches=n_matches, rms=float("nan"), method="pattern_weak")
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    return Transform(float(translation[0]), float(translation[1]), rotation, scale,
                     n_matches=n_matches, rms=rms, method="pattern")


def _offset_mode(offsets: np.ndarray, tolerance: float) -> Optional[np.ndarray]:
    """Densest cluster of candidate translations, found on a coarse grid."""
    if offsets.size == 0:
        return None
    bin_size = max(2.0 * tolerance, 2.0)
    keys = np.floor(offsets / bin_size).astype(int)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    best = unique[int(np.argmax(counts))]
    if counts.max() < 3:
        return None
    member = np.all(keys == best, axis=1)
    return np.median(offsets[member], axis=0)


def warp(image: np.ndarray, transform: Transform,
         output_shape: Optional[Tuple[int, int]] = None,
         fill: float = np.nan) -> np.ndarray:
    """Resample ``image`` through ``transform`` with bilinear interpolation."""
    data = as_float_image(image)
    ny, nx = output_shape or data.shape
    if transform.is_identity and (ny, nx) == data.shape:
        return data.copy()

    scipy_ndimage = try_import("scipy.ndimage")
    theta = np.deg2rad(transform.rotation)
    c, s = np.cos(theta) * transform.scale, np.sin(theta) * transform.scale
    forward = np.array([[c, -s], [s, c]], dtype=float)
    inverse = np.linalg.inv(forward)

    if scipy_ndimage is not None:
        # affine_transform works in (row, col); swap the axes of the matrix.
        matrix = np.array([[inverse[1, 1], inverse[1, 0]],
                           [inverse[0, 1], inverse[0, 0]]], dtype=float)
        offset_xy = -inverse @ np.array([transform.dx, transform.dy])
        return scipy_ndimage.affine_transform(
            data, matrix, offset=(offset_xy[1], offset_xy[0]),
            output_shape=(ny, nx), order=1, mode="constant", cval=fill)

    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - transform.dx
    dy = yy - transform.dy
    src_x = inverse[0, 0] * dx + inverse[0, 1] * dy
    src_y = inverse[1, 0] * dx + inverse[1, 1] * dy
    return _bilinear_sample(data, src_x, src_y, fill)


def _bilinear_sample(data: np.ndarray, src_x: np.ndarray, src_y: np.ndarray,
                     fill: float) -> np.ndarray:
    ny, nx = data.shape
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    wx = src_x - x0
    wy = src_y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0c = np.clip(x0, 0, nx - 2)
    y0c = np.clip(y0, 0, ny - 2)
    top = data[y0c, x0c] * (1 - wx) + data[y0c, x0c + 1] * wx
    bottom = data[y0c + 1, x0c] * (1 - wx) + data[y0c + 1, x0c + 1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.where(inside, out, fill)


def align_image(reference: np.ndarray, moving: np.ndarray, method: str = "auto",
                max_shift: Optional[float] = None) -> Tuple[np.ndarray, Transform]:
    """Align ``moving`` onto ``reference``; returns the warped image and transform.

    ``method`` may be ``"correlation"``, ``"pattern"`` or ``"auto"``, which
    tries star-pattern matching first and falls back to cross-correlation.
    """
    ref = as_float_image(reference)
    mov = as_float_image(moving)

    if method in ("auto", "pattern"):
        ref_stars = find_psf_stars(ref, threshold_sigma=8.0, max_stars=40, edge=10)
        mov_stars = find_psf_stars(mov, threshold_sigma=8.0, max_stars=40, edge=10)
        transform = match_star_patterns(ref_stars, mov_stars)
        if transform.n_matches >= 3 and np.isfinite(transform.rms) and transform.rms < 3.0:
            log.debug("aligned by star pattern: %s", transform.to_dict())
            return warp(mov, transform, ref.shape, fill=np.nan), transform
        if method == "pattern":
            log.warning("star-pattern alignment failed (%s); returning unshifted image",
                        transform.method)
            return mov.copy(), transform

    dx, dy = cross_correlation_shift(ref, mov, max_shift=max_shift)
    transform = Transform(dx=dx, dy=dy, method="correlation", n_matches=0)
    log.debug("aligned by cross-correlation: dx=%.3f dy=%.3f", dx, dy)
    return warp(mov, transform, ref.shape, fill=np.nan), transform
