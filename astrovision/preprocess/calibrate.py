"""Instrumental calibration and artefact rejection."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    as_float_image,
    convolve,
    gaussian_kernel,
    median_filter,
    sigma_clipped_stats,
)

log = get_logger("preprocess.calibrate")

#: 3x3 Laplacian used by the cosmic-ray detector (van Dokkum 2001).
LAPLACIAN = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])


def apply_calibration(image: np.ndarray, bias: Optional[np.ndarray] = None,
                      dark: Optional[np.ndarray] = None,
                      flat: Optional[np.ndarray] = None,
                      exposure_time: Optional[float] = None) -> np.ndarray:
    """Apply the standard CCD reduction ``(raw - bias - dark) / flat``.

    ``dark`` is scaled by ``exposure_time`` when the dark frame carries a
    per-second rate (the usual convention for master darks).
    """
    data = as_float_image(image, copy=True)
    if bias is not None:
        data -= as_float_image(bias)
    if dark is not None:
        scale = float(exposure_time) if exposure_time else 1.0
        data -= as_float_image(dark) * scale
    if flat is not None:
        flat_data = as_float_image(flat)
        normalised = flat_data / max(float(np.nanmedian(flat_data)), 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.where(np.abs(normalised) > 1e-3, data / normalised, np.nan)
    return data


def _subsample2(data: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour upsample (the LACosmic ``blkrep`` step)."""
    return np.repeat(np.repeat(data, 2, axis=0), 2, axis=1)


def _block_average2(data: np.ndarray) -> np.ndarray:
    """2x block average back to the original sampling."""
    ny, nx = data.shape[0] // 2 * 2, data.shape[1] // 2 * 2
    trimmed = data[:ny, :nx]
    return trimmed.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))


def detect_cosmic_rays(image: np.ndarray, sigma: float = 6.0, contrast: float = 2.0,
                       rms: Optional[np.ndarray] = None,
                       max_iterations: int = 2,
                       grow: bool = True) -> np.ndarray:
    """Flag cosmic-ray hits with the LACosmic algorithm (van Dokkum 2001).

    Cosmic rays are sharper than the point-spread function.  The image is
    subsampled 2x, Laplacian-filtered and re-binned, which gives a strong
    response to single-pixel spikes and a weak one to seeing-limited
    stars.  Dividing that response by a "fine structure" image built from
    median filters then separates the two populations; ``contrast`` is the
    cut on that ratio.  Returns a boolean mask of affected pixels.
    """
    data = as_float_image(image)
    if rms is None:
        _, _, noise = sigma_clipped_stats(data)
        rms_map = np.full(data.shape, max(noise, 1e-9))
    else:
        rms_map = np.clip(np.asarray(rms, dtype=float), 1e-9, None)

    working = data.copy()
    mask = np.zeros(data.shape, dtype=bool)

    for _ in range(max(1, int(max_iterations))):
        # Laplacian of the 2x-subsampled image, rebinned to native pixels.
        laplacian = np.clip(convolve(_subsample2(working), LAPLACIAN), 0, None)
        response = _block_average2(laplacian)
        if response.shape != data.shape:
            response = np.pad(response,
                              ((0, data.shape[0] - response.shape[0]),
                               (0, data.shape[1] - response.shape[1])), mode="edge")
        # The factor 2 accounts for the noise correlated by subsampling.
        significance = response / (2.0 * rms_map)
        # Remove large-scale structure so extended sources are not flagged.
        significance = significance - median_filter(significance, 5)

        # Fine-structure image: stars survive median filtering, CRs do not.
        coarse = median_filter(working, 3)
        fine = np.clip(coarse - median_filter(coarse, 7), 1e-9, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpness = response / fine

        new_hits = (significance > sigma) & (sharpness > contrast) & ~mask
        if not new_hits.any():
            break
        mask |= new_hits
        working = repair_pixels(working, mask, size=5)

    if grow and mask.any():
        # Extend onto immediate neighbours that are themselves significant.
        neighbours = convolve(mask.astype(float), np.ones((3, 3))) > 0
        laplacian = np.clip(convolve(_subsample2(data), LAPLACIAN), 0, None)
        response = _block_average2(laplacian)
        if response.shape != data.shape:
            response = np.pad(response,
                              ((0, data.shape[0] - response.shape[0]),
                               (0, data.shape[1] - response.shape[1])), mode="edge")
        mask |= neighbours & (response / (2.0 * rms_map) > sigma * 0.5)

    log.debug("cosmic-ray mask flags %d pixels", int(mask.sum()))
    return mask


def repair_pixels(image: np.ndarray, mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Replace masked pixels with the local median of their neighbourhood."""
    data = as_float_image(image, copy=True)
    bad = np.asarray(mask, dtype=bool)
    if not bad.any():
        return data
    filled = data.copy()
    filled[bad] = np.nan
    # Median-filter an interpolated copy so neighbours are themselves clean.
    interpolated = np.where(bad, np.nanmedian(data[~bad]) if (~bad).any() else 0.0, data)
    replacement = median_filter(interpolated, size)
    data[bad] = replacement[bad]
    return data


def detect_saturated(image: np.ndarray, level: Optional[float] = None,
                     header: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
    """Mask saturated pixels; the level comes from the header when present."""
    data = as_float_image(image)
    if level is None or not np.isfinite(level):
        for key in ("SATURATE", "SATLEVEL", "DATAMAX"):
            if header and key in header:
                try:
                    level = float(header[key])
                    break
                except (TypeError, ValueError):
                    continue
    if level is None or not np.isfinite(level):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros(data.shape, dtype=bool), float("inf")
        # Without a header value, treat a hard ceiling in the histogram as
        # saturation only if many pixels pile up at the maximum.
        peak = float(finite.max())
        at_peak = int((finite >= peak * 0.9999).sum())
        if at_peak < 4:
            return np.zeros(data.shape, dtype=bool), float("inf")
        level = peak
    mask = data >= float(level) * 0.999
    return mask, float(level)


def detect_bad_columns(image: np.ndarray, sigma: float = 6.0,
                       min_fraction: float = 0.55) -> np.ndarray:
    """Flag dead or hot columns and rows.

    A defective column is bad along its *whole* length; a bright source is
    not.  Two conditions are therefore required together: the column's
    median must deviate from its neighbours by more than the median's own
    uncertainty allows, and a majority of the column's pixels must deviate
    in the same direction.  Testing the median alone flags any column that
    happens to contain a bright object -- including, in a difference-image
    search, the transient being looked for.
    """
    data = as_float_image(image)
    mask = np.zeros(data.shape, dtype=bool)
    _, image_median, pixel_noise = sigma_clipped_stats(data)
    if pixel_noise <= 0:
        return mask

    for axis in (0, 1):
        length = data.shape[axis]
        if length < 8:
            continue
        profile = np.nanmedian(data, axis=axis)
        smooth = median_filter(profile.reshape(1, -1), 5).ravel()
        residual = profile - smooth

        # The uncertainty on a median of `length` pixels; without this floor
        # a perfectly flat profile makes the noise estimate collapse and
        # essentially every column is flagged.
        expected = 1.2533 * pixel_noise / np.sqrt(max(length, 1))
        _, _, measured = sigma_clipped_stats(residual)
        noise = max(float(measured), float(expected))

        candidates = np.nonzero(np.abs(residual) > sigma * noise)[0]
        for index in candidates:
            line = data[:, index] if axis == 0 else data[index, :]
            offset = float(residual[index])
            # Is the whole line offset the same way, or just part of it?
            deviating = ((line - image_median) * np.sign(offset) >
                         0.5 * abs(offset))
            if float(np.mean(deviating)) < min_fraction:
                continue
            if axis == 0:
                mask[:, index] = True
            else:
                mask[index, :] = True

    log.debug("bad-column mask flags %d pixels", int(mask.sum()))
    return mask


def smooth_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Optional Gaussian pre-smoothing (helps very low signal-to-noise data)."""
    if not sigma or sigma <= 0:
        return as_float_image(image)
    return convolve(image, gaussian_kernel(float(sigma)))
