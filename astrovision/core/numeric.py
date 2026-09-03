"""Array utilities shared across the platform.

Everything here is pure NumPy so the scientific core runs without SciPy.
When SciPy *is* installed a few routines transparently use its faster or
more accurate implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .backend import try_import

#: FWHM = SIGMA_TO_FWHM * sigma for a Gaussian.
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
#: Median absolute deviation -> Gaussian sigma.
MAD_TO_SIGMA = 1.4826

# Trapezoidal integration, under whichever name the installed NumPy uses.
#
# NumPy 2.0 renamed ``trapz`` to ``trapezoid`` *and removed the old name*.
# Neither name works across the range this package supports (``numpy>=1.21``):
# ``np.trapezoid`` raises on 1.x and ``np.trapz`` raises on 2.x.  Binding it
# once here is the only fix that covers both -- switching to either name
# directly just trades one broken environment for the other.
if hasattr(np, "trapezoid"):          # NumPy >= 2.0
    trapezoid = np.trapezoid
else:                                 # NumPy < 2.0
    trapezoid = np.trapz              # noqa: NPY201


def as_float_image(array: np.ndarray, copy: bool = False) -> np.ndarray:
    """Coerce input to a 2-D float64 array, collapsing trivial extra axes."""
    data = np.asarray(array)
    if data.ndim > 2:
        squeezed = np.squeeze(data)
        if squeezed.ndim == 2:
            data = squeezed
        elif squeezed.ndim == 3 and squeezed.shape[-1] in (3, 4):
            # RGB(A) input: use luminance so colour images still work.
            weights = np.array([0.2126, 0.7152, 0.0722])
            data = squeezed[..., :3].astype(float) @ weights
        else:
            raise ValueError(f"expected a 2-D image, got shape {data.shape}")
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {data.ndim}-D")
    return np.array(data, dtype=float, copy=True) if copy else data.astype(float, copy=False)


def nan_to_finite(array: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/inf with ``fill`` (or the array median when ``fill`` is NaN)."""
    data = np.asarray(array, dtype=float)
    bad = ~np.isfinite(data)
    if not bad.any():
        return data
    out = data.copy()
    value = fill
    if not np.isfinite(value):
        good = data[~bad]
        value = float(np.median(good)) if good.size else 0.0
    out[bad] = value
    return out


def mad_std(array: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Robust standard deviation from the median absolute deviation."""
    data = np.asarray(array, dtype=float)
    med = np.nanmedian(data, axis=axis, keepdims=axis is not None)
    mad = np.nanmedian(np.abs(data - med), axis=axis)
    return MAD_TO_SIGMA * mad


def sigma_clip(array: np.ndarray, sigma: float = 3.0, maxiters: int = 5,
               return_mask: bool = False):
    """Iteratively clip outliers using a robust centre and scale.

    Returns the clipped values, or ``(values, mask)`` when ``return_mask``
    is set -- ``mask`` is ``True`` for *kept* elements in the flat input.
    """
    data = np.asarray(array, dtype=float).ravel()
    mask = np.isfinite(data)
    for _ in range(max(1, int(maxiters))):
        if not mask.any():
            break
        subset = data[mask]
        centre = float(np.median(subset))
        scale = float(MAD_TO_SIGMA * np.median(np.abs(subset - centre)))
        if scale <= 0:
            scale = float(np.std(subset))
        if scale <= 0:
            break
        new_mask = mask & (np.abs(data - centre) <= sigma * scale)
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    if return_mask:
        return data[mask], mask
    return data[mask]


def sigma_clipped_stats(array: np.ndarray, sigma: float = 3.0,
                        maxiters: int = 5) -> Tuple[float, float, float]:
    """Return ``(mean, median, std)`` computed after sigma clipping."""
    kept = sigma_clip(array, sigma=sigma, maxiters=maxiters)
    if kept.size == 0:
        return 0.0, 0.0, 0.0
    std = float(np.std(kept))
    robust = float(MAD_TO_SIGMA * np.median(np.abs(kept - np.median(kept))))
    return float(np.mean(kept)), float(np.median(kept)), max(std, robust, 1e-12)


def gaussian_kernel(sigma: float, size: Optional[int] = None) -> np.ndarray:
    """Normalised 2-D Gaussian kernel."""
    sigma = max(float(sigma), 1e-3)
    if size is None:
        size = int(2 * np.ceil(3.0 * sigma) + 1)
    size = max(3, int(size) | 1)  # force odd
    half = size // 2
    axis = np.arange(-half, half + 1, dtype=float)
    g1 = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(g1, g1)
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def tophat_kernel(radius: float) -> np.ndarray:
    """Normalised circular top-hat kernel of the given pixel radius."""
    radius = max(float(radius), 0.5)
    half = int(np.ceil(radius))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = (xx ** 2 + yy ** 2 <= radius ** 2).astype(float)
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def convolve(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """2-D convolution with edge handling; uses SciPy when available."""
    data = as_float_image(image)
    kern = np.asarray(kernel, dtype=float)
    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        return scipy_ndimage.convolve(data, kern, mode=mode)
    # NumPy fallback: pad then accumulate shifted copies (kernels are small).
    ky, kx = kern.shape
    py, px = ky // 2, kx // 2
    padded = np.pad(data, ((py, py), (px, px)), mode=_np_pad_mode(mode))
    out = np.zeros_like(data)
    flipped = kern[::-1, ::-1]
    for j in range(ky):
        for i in range(kx):
            weight = flipped[j, i]
            if weight != 0.0:
                out += weight * padded[j:j + data.shape[0], i:i + data.shape[1]]
    return out


def _np_pad_mode(mode: str) -> str:
    """SciPy's edge modes in NumPy's vocabulary.

    The two libraries use the word "reflect" for different things. SciPy's
    ``reflect`` repeats the edge sample (``d c b a | a b c d``), which NumPy
    calls ``symmetric``; SciPy's ``mirror`` does not repeat it
    (``d c b | a b c d``), which NumPy calls ``reflect``. Mapping the words
    to themselves made every fallback differ from SciPy along the edges --
    and on a 6x6 background mesh the edges are most of the array, so the
    NumPy-only detection lost six of 48 sources on a test field.
    """
    return {"reflect": "symmetric", "nearest": "edge", "constant": "constant",
            "wrap": "wrap", "mirror": "reflect"}.get(mode, "symmetric")


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing (separable when SciPy is present)."""
    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        return scipy_ndimage.gaussian_filter(as_float_image(image), sigma=float(sigma))
    # SciPy truncates the kernel at int(4 sigma + 0.5); match it so both
    # paths agree to floating-point precision.
    radius = int(4.0 * max(float(sigma), 1e-3) + 0.5)
    return convolve(image, gaussian_kernel(sigma, size=2 * radius + 1))


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter; falls back to a strided NumPy implementation."""
    data = as_float_image(image)
    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        return scipy_ndimage.median_filter(data, size=int(size))
    size = max(3, int(size) | 1)
    half = size // 2
    padded = np.pad(data, half, mode=_np_pad_mode("reflect"))   # SciPy's default
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-2, -1))


def maximum_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Local maximum filter, used for peak finding."""
    data = as_float_image(image)
    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        return scipy_ndimage.maximum_filter(data, size=int(size))
    size = max(3, int(size) | 1)
    half = size // 2
    padded = np.pad(data, half, mode=_np_pad_mode("reflect"))   # SciPy's default
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.max(axis=(-2, -1))


def block_reduce(image: np.ndarray, factor: int, func=np.mean) -> np.ndarray:
    """Downsample by integer ``factor`` applying ``func`` per block."""
    data = as_float_image(image)
    factor = max(1, int(factor))
    if factor == 1:
        return data.copy()
    ny = (data.shape[0] // factor) * factor
    nx = (data.shape[1] // factor) * factor
    trimmed = data[:ny, :nx]
    reshaped = trimmed.reshape(ny // factor, factor, nx // factor, factor)
    return func(reshaped, axis=(1, 3))


def bilinear_resize(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to ``shape`` with bilinear interpolation."""
    data = as_float_image(image)
    ny, nx = int(shape[0]), int(shape[1])
    if data.shape == (ny, nx):
        return data.copy()
    if data.size == 0 or ny <= 0 or nx <= 0:
        return np.zeros((max(ny, 1), max(nx, 1)), dtype=float)
    src_y = np.linspace(0, data.shape[0] - 1, ny) if ny > 1 else np.array([0.0])
    src_x = np.linspace(0, data.shape[1] - 1, nx) if nx > 1 else np.array([0.0])
    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    y1 = np.clip(y0 + 1, 0, data.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, data.shape[1] - 1)
    wy = (src_y - y0)[:, None]
    wx = (src_x - x0)[None, :]
    top = data[np.ix_(y0, x0)] * (1 - wx) + data[np.ix_(y0, x1)] * wx
    bottom = data[np.ix_(y1, x0)] * (1 - wx) + data[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


def pad_or_crop(image: np.ndarray, shape: Tuple[int, int], fill: float = 0.0) -> np.ndarray:
    """Centre-pad or centre-crop an array to exactly ``shape``."""
    data = as_float_image(image)
    ny, nx = int(shape[0]), int(shape[1])
    out = np.full((ny, nx), float(fill), dtype=float)
    sy = min(ny, data.shape[0])
    sx = min(nx, data.shape[1])
    src_y = (data.shape[0] - sy) // 2
    src_x = (data.shape[1] - sx) // 2
    dst_y = (ny - sy) // 2
    dst_x = (nx - sx) // 2
    out[dst_y:dst_y + sy, dst_x:dst_x + sx] = data[src_y:src_y + sy, src_x:src_x + sx]
    return out


def radial_profile(image: np.ndarray, centre: Optional[Tuple[float, float]] = None,
                   nbins: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged profile; returns ``(radii, mean_value)``."""
    data = as_float_image(image)
    ny, nx = data.shape
    cx, cy = centre if centre is not None else ((nx - 1) / 2.0, (ny - 1) / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - cx, yy - cy)
    rmax = float(r.max())
    if rmax <= 0:
        return np.zeros(1), np.array([float(data.mean())])
    edges = np.linspace(0, rmax, max(2, int(nbins) + 1))
    index = np.clip(np.digitize(r.ravel(), edges) - 1, 0, len(edges) - 2)
    values = data.ravel()
    finite = np.isfinite(values)
    sums = np.bincount(index[finite], weights=values[finite], minlength=len(edges) - 1)
    counts = np.bincount(index[finite], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, profile


def circular_mask(shape: Tuple[int, int], centre: Tuple[float, float],
                  radius: float) -> np.ndarray:
    """Boolean mask of pixels whose centres lie within ``radius``."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= float(radius) ** 2


def elliptical_mask(shape: Tuple[int, int], centre: Tuple[float, float],
                    a: float, b: float, theta_deg: float) -> np.ndarray:
    """Boolean mask for an ellipse with semi-axes ``a``/``b`` rotated by theta."""
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = np.deg2rad(theta_deg)
    dx = xx - centre[0]
    dy = yy - centre[1]
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def percentile_clip(image: np.ndarray, low: float = 1.0,
                    high: float = 99.0) -> np.ndarray:
    """Clip an image between two percentiles (display/preprocessing helper)."""
    data = as_float_image(image)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data
    lo, hi = np.percentile(finite, [low, high])
    return np.clip(data, lo, hi)


def safe_divide(numerator, denominator, fill: float = 0.0) -> np.ndarray:
    """Element-wise division that yields ``fill`` where the denominator is ~0."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, float(fill), dtype=float)
    good = np.isfinite(den) & (np.abs(den) > 1e-12) & np.isfinite(num)
    np.divide(num, den, out=out, where=good)
    return out


def weighted_centroid(image: np.ndarray, mask: Optional[np.ndarray] = None,
                      offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Flux-weighted centroid ``(x, y)`` in absolute pixel coordinates."""
    data = as_float_image(image)
    weights = np.clip(nan_to_finite(data, 0.0), 0.0, None)
    if mask is not None:
        weights = weights * np.asarray(mask, dtype=float)
    total = float(weights.sum())
    ny, nx = data.shape
    if total <= 0:
        return (offset[0] + (nx - 1) / 2.0, offset[1] + (ny - 1) / 2.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    return (offset[0] + cx, offset[1] + cy)


def normalise_unit(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise along ``axis``, leaving zero vectors untouched."""
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return np.where(norm > 1e-12, arr / np.maximum(norm, 1e-12), arr)


def softmax(scores: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax over a 1-D score vector."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total > 0 else np.full_like(arr, 1.0 / arr.size)


def logistic(x, scale: float = 1.0, midpoint: float = 0.0) -> np.ndarray:
    """Logistic squashing used to map raw scores into ``[0, 1]``."""
    z = (np.asarray(x, dtype=float) - midpoint) / max(float(scale), 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def rescale(array: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Linearly rescale finite values into ``[low, high]``."""
    data = np.asarray(array, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.full_like(data, low)
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-12:
        return np.full_like(data, low)
    return low + (high - low) * (data - vmin) / (vmax - vmin)
