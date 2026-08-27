"""The preprocessing stage: raw pixels in, analysis-ready image out."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.config import PreprocessConfig
from ..core.logging import get_logger
from ..io.image import AstroImage
from .background import estimate_background
from .calibrate import (
    apply_calibration,
    detect_bad_columns,
    detect_cosmic_rays,
    detect_saturated,
    repair_pixels,
    smooth_image,
)
from .psf import PSFModel, build_psf
from .varying_psf import (
    VaryingPSF,
    find_psf_stars_by_region,
    fit_varying_psf,
)

log = get_logger("preprocess.pipeline")


class Preprocessor:
    """Runs calibration, artefact rejection, background and PSF estimation.

    The result is an :class:`~astrovision.io.image.AstroImage` carrying a
    background model, an RMS map and a bad-pixel mask -- everything the
    detection stage needs to set a statistically meaningful threshold.

    >>> from astrovision.simulate import quick_field
    >>> image, _ = quick_field((128, 128))
    >>> clean = Preprocessor().run(image)
    >>> clean.background is not None
    True
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.psf: Optional[PSFModel] = None
        self.varying_psf: Optional[VaryingPSF] = None
        self.report: Dict[str, Any] = {}

    def run(self, image: AstroImage, bias: Optional[np.ndarray] = None,
            dark: Optional[np.ndarray] = None, flat: Optional[np.ndarray] = None,
            estimate_psf: bool = True) -> AstroImage:
        """Return a cleaned copy of ``image`` with all products attached."""
        cfg = self.config
        report: Dict[str, Any] = {"steps": []}
        data = np.array(image.data, dtype=float, copy=True)

        if bias is not None or dark is not None or flat is not None:
            data = apply_calibration(data, bias, dark, flat, image.exposure_time)
            report["steps"].append("calibration")

        mask = np.zeros(data.shape, dtype=bool)
        if image.mask is not None:
            mask |= image.mask

        if cfg.mask_saturated:
            saturated, level = detect_saturated(data, cfg.saturation_level, image.header)
            if saturated.any():
                mask |= saturated
                report["saturated_pixels"] = int(saturated.sum())
                report["saturation_level"] = float(level)
                report["steps"].append("saturation_mask")

        # A first background pass gives the RMS map that cosmic-ray
        # detection and thresholding both need.
        background, rms = estimate_background(data, cfg.background_box,
                                              cfg.background_filter, mask)

        if cfg.mask_bad_columns:
            # Dead or hot columns survive background subtraction and turn
            # into strings of spurious residuals in every difference image,
            # so they must be masked before detection ever sees them.
            columns = detect_bad_columns(data, cfg.bad_column_sigma)
            if columns.any():
                # Interpolate over them as well as flagging them: leaving the
                # values in place would put a string of spurious residuals
                # down every difference image.
                data = repair_pixels(data, columns, size=5)
                mask |= columns
                report["bad_column_pixels"] = int(columns.sum())
                report["steps"].append("bad_column_repair")

        if cfg.reject_cosmic_rays:
            cosmic = detect_cosmic_rays(data, cfg.cosmic_ray_sigma,
                                        cfg.cosmic_ray_contrast, rms)
            if cosmic.any():
                data = repair_pixels(data, cosmic)
                mask |= cosmic
                report["cosmic_ray_pixels"] = int(cosmic.sum())
                report["steps"].append("cosmic_ray_rejection")

        if cfg.smooth_sigma and cfg.smooth_sigma > 0:
            data = smooth_image(data, cfg.smooth_sigma)
            report["steps"].append(f"smoothing(sigma={cfg.smooth_sigma})")

        # Re-estimate the background on the repaired image.
        background, rms = estimate_background(data, cfg.background_box,
                                              cfg.background_filter, mask)
        report["background_median"] = float(np.median(background))
        report["background_rms_median"] = float(np.median(rms))

        if cfg.subtract_background:
            data = data - background
            report["steps"].append("background_subtraction")
            # `background` is zeroed on the result so `subtracted()` is a no-op
            # downstream, but the model itself is kept: checking it is the
            # first thing to do when a field looks over- or under-subtracted.
            background_out = np.zeros_like(background)
        else:
            background_out = background

        result = image.copy_with(
            data, mask=mask, background=background_out, background_rms=rms,
            name=image.name,
        )
        result.meta = dict(image.meta)

        if estimate_psf:
            self.psf = build_psf(data, rms=rms)
            report["psf"] = self.psf.to_dict()
            result.meta["psf"] = self.psf.to_dict()
            result.meta["psf_model"] = self.psf
            if cfg.varying_psf:
                # The single model above is kept regardless: it is the field
                # average, every existing consumer expects it, and the
                # spatial fit may decide the field does not vary at all.
                # The per-tile cap is generous on purpose: the selector's own
                # isolation and stellar-locus cuts are what limit the count,
                # and a tight cap here silently starves the spatial fit -- a
                # field with a real 40% variation fell back to one PSF simply
                # because each tile was allowed only 26 stars.
                stars = find_psf_stars_by_region(
                    data, cfg.varying_psf_regions, rms, per_region=80)
                # The stamp is sized to the *measured* seeing rather than
                # left at a fixed 25 pixels.  A 25-pixel stamp around a
                # 3-pixel PSF is mostly empty sky, and the held-out
                # comparison that decides whether the field varies is then
                # diluted by wing noise: on a field with a real 40%
                # variation the fixed size scored 2.9% against a 3%
                # threshold and the variation was thrown away.  2.5 FWHM,
                # not more: the fit needs the core, where the PSF genuinely
                # differs across the field, and every extra ring of wing
                # pixels adds noise without adding signal.
                stamp = int(2 * np.ceil(2.5 * max(self.psf.fwhm, 1.5)) + 1)
                self.varying_psf = fit_varying_psf(
                    data, stars, size=min(max(stamp, 11), 41), rms=rms,
                    degree=cfg.varying_psf_degree)
                result.meta["varying_psf"] = self.varying_psf
                report["varying_psf"] = self.varying_psf.to_dict()

        result.meta["background_model"] = background
        result.meta["preprocess"] = report
        self.report = report
        log.info("preprocessed '%s': %s", image.name, ", ".join(report["steps"]) or "no-op")
        return result

    def run_series(self, images, **kwargs):
        """Preprocess every epoch of a series with the same settings."""
        return [self.run(image, **kwargs) for image in images]
