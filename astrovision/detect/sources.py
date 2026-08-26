"""Source extraction: turn an image into a catalog of detected objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import DetectionConfig
from ..core.logging import get_logger
from ..core.numeric import (
    SIGMA_TO_FWHM,
    as_float_image,
    convolve,
    gaussian_kernel,
    sigma_clipped_stats,
)
from ..core.types import BoundingBox, MorphologyMetrics, Photometry, Source, SourceCatalog
from ..io.image import AstroImage
from .deblend import deblend_all
from .labeling import find_objects, label, remove_small

log = get_logger("detect.sources")


def detection_threshold(image: np.ndarray, sigma: float = 3.5,
                        rms: Optional[np.ndarray] = None,
                        background: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel detection threshold ``background + sigma * rms``."""
    data = as_float_image(image)
    if rms is None:
        _, median, noise = sigma_clipped_stats(data)
        rms_map = np.full(data.shape, max(noise, 1e-9), dtype=float)
        base = np.full(data.shape, median, dtype=float) if background is None else background
    else:
        rms_map = np.clip(np.asarray(rms, dtype=float), 1e-9, None)
        base = np.zeros(data.shape, dtype=float) if background is None else background
    return base + float(sigma) * rms_map


def build_segmentation(image: np.ndarray, config: DetectionConfig,
                       rms: Optional[np.ndarray] = None,
                       background: Optional[np.ndarray] = None,
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, np.ndarray]:
    """Threshold, label and deblend; returns ``(segmentation, n, threshold)``.

    Detection runs on a PSF-matched-filtered copy of the image -- convolving
    with a kernel that matches the source profile maximises signal-to-noise
    for point sources, which is why faint-object detection always does it.
    """
    data = as_float_image(image)
    threshold = detection_threshold(data, config.threshold_sigma, rms, background)

    if config.filter_fwhm and config.filter_fwhm > 0:
        kernel = gaussian_kernel(config.filter_fwhm / SIGMA_TO_FWHM)
        filtered = convolve(data, kernel)
        # The matched filter reduces the noise by the kernel's RMS, so the
        # threshold must be scaled the same way to stay at N sigma.
        scale = float(np.sqrt((kernel ** 2).sum()))
        detect_threshold = threshold * scale
    else:
        filtered = data
        detect_threshold = threshold

    above = filtered > detect_threshold
    if mask is not None:
        above &= ~np.asarray(mask, dtype=bool)
    if config.border_margin > 0:
        m = int(config.border_margin)
        above[:m, :] = above[-m:, :] = False
        above[:, :m] = above[:, -m:] = False

    segmentation, count = label(above)
    if count == 0:
        return segmentation, 0, threshold
    segmentation, count = remove_small(segmentation, config.min_area, count)
    if count == 0:
        return segmentation, 0, threshold

    if config.deblend and count > 0:
        segmentation, count = deblend_all(
            data, segmentation, threshold, config.deblend_levels,
            config.deblend_contrast, max(3, config.min_area // 2))
        segmentation, count = remove_small(segmentation, config.min_area, count)

    log.debug("segmentation: %d objects above %.2f sigma", count, config.threshold_sigma)
    return segmentation, count, threshold


def measure_segment(data: np.ndarray, footprint: np.ndarray,
                    offset: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
    """Basic moments of one segment: centroid, second moments, peak, area.

    ``offset`` is the ``(x0, y0)`` origin of the cutout in the full image.
    """
    values = np.where(footprint, np.nan_to_num(data, nan=0.0), 0.0)
    weights = np.clip(values, 0.0, None)
    total = float(weights.sum())
    ny, nx = values.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    if total <= 0:
        ys, xs = np.nonzero(footprint)
        cx = float(xs.mean()) if xs.size else 0.0
        cy = float(ys.mean()) if ys.size else 0.0
        mxx = myy = mxy = 0.0
    else:
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        dx, dy = xx - cx, yy - cy
        mxx = float((weights * dx * dx).sum() / total)
        myy = float((weights * dy * dy).sum() / total)
        mxy = float((weights * dx * dy).sum() / total)

    # Diagonalise the second-moment tensor to get the ellipse parameters.
    common = float(np.sqrt(max(((mxx - myy) / 2.0) ** 2 + mxy ** 2, 0.0)))
    mean = (mxx + myy) / 2.0
    semi_major = float(np.sqrt(max(mean + common, 0.0)))
    semi_minor = float(np.sqrt(max(mean - common, 0.0)))
    angle = float(0.5 * np.degrees(np.arctan2(2.0 * mxy, mxx - myy)))

    ys, xs = np.nonzero(footprint)
    peak_index = int(np.argmax(np.where(footprint, data, -np.inf)))
    peak_y, peak_x = np.unravel_index(peak_index, values.shape)

    return {
        "x": cx + offset[0], "y": cy + offset[1],
        "flux_iso": total,
        "peak": float(data[peak_y, peak_x]),
        "peak_x": float(peak_x + offset[0]), "peak_y": float(peak_y + offset[1]),
        "area": int(footprint.sum()),
        "semi_major": semi_major, "semi_minor": semi_minor,
        "position_angle": angle,
        "ellipticity": float(1.0 - semi_minor / semi_major) if semi_major > 0 else 0.0,
        "elongation": float(semi_major / semi_minor) if semi_minor > 0 else float("inf"),
        "x_min": float(xs.min() + offset[0]), "x_max": float(xs.max() + 1 + offset[0]),
        "y_min": float(ys.min() + offset[1]), "y_max": float(ys.max() + 1 + offset[1]),
        "mxx": mxx, "myy": myy, "mxy": mxy,
    }


def extract_sources(image: AstroImage, config: Optional[DetectionConfig] = None,
                    segmentation: Optional[np.ndarray] = None) -> Tuple[SourceCatalog, np.ndarray]:
    """Detect and measure every source in ``image``.

    Returns the catalog and the segmentation map, which downstream stages
    reuse for photometry and morphology rather than re-thresholding.
    """
    config = config or DetectionConfig()
    data = image.subtracted()
    rms = image.rms_map()

    if segmentation is None:
        segmentation, count, threshold = build_segmentation(
            data, config, rms, None, image.mask)
    else:
        segmentation = np.asarray(segmentation, dtype=np.int32)
        count = int(segmentation.max())
        threshold = detection_threshold(data, config.threshold_sigma, rms)

    catalog = SourceCatalog(meta={
        "image": image.name, "band": image.band, "mjd": image.mjd,
        "threshold_sigma": config.threshold_sigma,
        "detection_backend": config.backend,
    })
    if count == 0:
        log.info("no sources detected in '%s'", image.name)
        return catalog, segmentation

    boxes = find_objects(segmentation, count)
    ny, nx = data.shape
    pad = float(config.bbox_pad)
    kept = 0
    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        rows, cols = box
        cut = data[rows, cols]
        footprint = segmentation[rows, cols] == index
        if not footprint.any():
            continue
        area = int(footprint.sum())
        if area < config.min_area or area > config.max_area:
            continue

        measurement = measure_segment(cut, footprint, (cols.start, rows.start))
        half_x = max(measurement["semi_major"] * pad, 2.0)
        half_y = max(measurement["semi_major"] * pad, 2.0)
        bbox = BoundingBox(
            max(measurement["x_min"] - 1, measurement["x"] - half_x),
            max(measurement["y_min"] - 1, measurement["y"] - half_y),
            min(measurement["x_max"] + 1, measurement["x"] + half_x),
            min(measurement["y_max"] + 1, measurement["y"] + half_y),
        ).clip((ny, nx))

        local_rms = float(np.median(rms[rows, cols][footprint]))
        snr = measurement["flux_iso"] / max(local_rms * np.sqrt(area), 1e-9)

        kept += 1
        source = Source(
            id=kept, x=measurement["x"], y=measurement["y"], bbox=bbox,
            segment_label=index,
            photometry=Photometry(
                flux=measurement["flux_iso"], peak=measurement["peak"],
                snr=float(snr), background=0.0,
                flux_err=float(local_rms * np.sqrt(area)),
            ),
            morphology=MorphologyMetrics(
                semi_major=measurement["semi_major"],
                semi_minor=measurement["semi_minor"],
                position_angle=measurement["position_angle"],
                ellipticity=measurement["ellipticity"],
                elongation=measurement["elongation"],
                fwhm=float(SIGMA_TO_FWHM * measurement["semi_major"]),
                area_pixels=area,
            ),
            meta={"peak_x": measurement["peak_x"], "peak_y": measurement["peak_y"],
                  "local_rms": local_rms},
        )
        if image.wcs is not None:
            ra, dec = image.wcs.pixel_to_world(source.x, source.y)
            source.ra, source.dec = float(ra), float(dec)
        _flag_source(source, image, data.shape)
        catalog.append(source)
        if kept >= config.max_sources:
            log.warning("hit max_sources=%d; stopping extraction", config.max_sources)
            break

    log.info("detected %d sources in '%s' at %.1f sigma",
             len(catalog), image.name, config.threshold_sigma)
    return catalog, segmentation


def _flag_source(source: Source, image: AstroImage, shape: Tuple[int, int]) -> None:
    """Attach quality flags that downstream stages and reports rely on."""
    ny, nx = shape
    margin = 8
    if (source.x < margin or source.y < margin or
            source.x > nx - margin or source.y > ny - margin):
        source.add_flag("edge")
    if image.mask is not None:
        rows, cols = source.bbox.slices(shape)
        if image.mask[rows, cols].any():
            source.add_flag("masked_pixels")
    saturation = image.header.get("SATURATE")
    if saturation is not None:
        try:
            if source.photometry.peak >= float(saturation) * 0.95:
                source.add_flag("saturated")
                source.photometry.saturated = True
        except (TypeError, ValueError):
            pass
    if source.morphology.area_pixels <= 4:
        source.add_flag("marginal")
