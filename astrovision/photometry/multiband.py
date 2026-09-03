"""Forced photometry across filters, and the colours that come out of it.

A colour is a *difference of magnitudes measured the same way*.  Almost
everything that goes wrong with colours goes wrong because that condition
was quietly broken:

* **Different apertures.**  Detecting independently in each band gives each
  one its own centroid and its own Kron radius, so the two apertures sample
  different parts of the same galaxy.  The fix is forced photometry -- one
  aperture, defined once in the detection band, applied at the same *sky*
  position everywhere.
* **Different seeing.**  A fixed aperture catches a larger fraction of a
  point source in good seeing than in bad, so a star observed in 1.0" and
  1.6" seeing acquires a colour it does not have.  The fix is to convolve
  every band to the worst PSF in the set before measuring.
* **Different pixel grids.**  Two cameras rarely share a pixel scale, so
  "radius 5 pixels" is not one aperture but several.  Apertures here are
  specified in arcseconds and converted per band.

The result is written to ``Source.bands``, keyed by filter, leaving
``Source.photometry`` -- the detection-band measurement the rest of the
pipeline uses -- untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import DataError
from ..core.logging import get_logger
from ..core.types import Photometry, Source, SourceCatalog
from ..io.image import AstroImage
from ..preprocess.psf import PSFModel, match_psf
from .aperture import (annulus_background, circular_aperture_weights, elliptical_photometry,
                       stamp_box)
from .magnitudes import flux_to_magnitude

log = get_logger("photometry.multiband")


@dataclass
class MultiBandReport:
    """What the forced-photometry pass did, for the provenance record."""

    bands: List[str]
    detection_band: str
    aperture_arcsec: float
    target_fwhm_arcsec: float
    homogenised: List[str]
    n_sources: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "bands": list(self.bands),
            "detection_band": self.detection_band,
            "aperture_arcsec": float(self.aperture_arcsec),
            "target_fwhm_arcsec": float(self.target_fwhm_arcsec),
            "homogenised": list(self.homogenised),
            "n_sources": int(self.n_sources),
            "warnings": list(self.warnings),
        }


def _pixel_scale(image: AstroImage, default: float = 1.0) -> float:
    if image.wcs is not None:
        scale = float(image.wcs.pixel_scale)
        if np.isfinite(scale) and scale > 0:
            return scale
    return float(default)


def _psf_of(image: AstroImage) -> Optional[PSFModel]:
    model = image.meta.get("psf_model")
    return model if isinstance(model, PSFModel) else None


def homogenise(images: Mapping[str, AstroImage],
               target_fwhm: Optional[float] = None
               ) -> Tuple[Dict[str, AstroImage], float, List[str]]:
    """Convolve every band to a common PSF width.

    The target defaults to the *worst* seeing in the set, because blurring
    is stable and sharpening is not: a matching kernel that has to remove
    width amplifies noise without bound, and the deconvolution ringing lands
    exactly where the photometry is measured.

    Widths are compared in arcseconds, not pixels, so bands on different
    cameras are handled correctly.  Returns the convolved images, the target
    width in arcsec, and the names of the bands that were actually changed.
    """
    widths: Dict[str, float] = {}
    for band, image in images.items():
        psf = _psf_of(image)
        if psf is None or not np.isfinite(psf.fwhm) or psf.fwhm <= 0:
            continue
        widths[band] = float(psf.fwhm) * _pixel_scale(image)

    if not widths:
        return dict(images), float("nan"), []

    target = float(target_fwhm) if target_fwhm else max(widths.values())
    out: Dict[str, AstroImage] = {}
    changed: List[str] = []
    for band, image in images.items():
        psf = _psf_of(image)
        if psf is None or band not in widths or widths[band] >= target - 1e-6:
            out[band] = image
            continue
        scale = _pixel_scale(image)
        # `match_psf` works in pixels, so the arcsecond target is converted
        # back through *this* band's pixel scale.
        goal = PSFModel(stamp=psf.stamp, fwhm=target / scale, n_stars=psf.n_stars)
        convolved = match_psf(image.subtracted(), psf, goal)
        # The model's *stamp* has to be convolved too.  Leaving the original
        # stamp on a model that claims the widened FWHM makes the model lie
        # about its own wings, and every enclosed-energy correction computed
        # from it is then wrong -- which shows up as a colour bias in exactly
        # the band that was left unconvolved.
        goal = PSFModel(stamp=match_psf(psf.stamp, psf, goal), fwhm=target / scale,
                        ellipticity=psf.ellipticity, position_angle=psf.position_angle,
                        n_stars=psf.n_stars, size=psf.size)
        new_image = image.copy_with(convolved)
        new_image.background = np.zeros_like(np.asarray(convolved))
        # Convolution correlates neighbouring pixels, so the per-pixel noise
        # drops by the kernel's rms -- ignoring that would overstate every
        # signal-to-noise ratio downstream.
        kernel_rms = _kernel_rms(psf, goal)
        if new_image.background_rms is not None:
            new_image.background_rms = np.asarray(
                new_image.background_rms, dtype=float) * kernel_rms
        new_image.meta = dict(image.meta)
        new_image.meta["psf_model"] = goal
        new_image.meta["psf_homogenised_to_arcsec"] = target
        new_image.meta["noise_correlation_factor"] = kernel_rms
        out[band] = new_image
        changed.append(band)
    return out, target, changed


def _kernel_rms(source: PSFModel, target: PSFModel) -> float:
    """Factor by which independent pixel noise shrinks under PSF matching."""
    from ..preprocess.psf import matching_kernel
    kernel = matching_kernel(source, target)
    total = float(kernel.sum())
    if total <= 0:
        return 1.0
    normalised = kernel / total
    return float(np.sqrt((normalised ** 2).sum()))


def _positions_in(image: AstroImage, catalog: SourceCatalog,
                  detection: AstroImage) -> Tuple[np.ndarray, bool]:
    """Where each catalog source falls on ``image``.

    Sky coordinates are used whenever both images carry a WCS, which is the
    only way to be right when the bands are not on a shared pixel grid.  If
    either lacks one the pixel positions are used unchanged and the caller
    is told, so the assumption ends up in the report rather than staying
    silent.
    """
    pixels = catalog.positions()
    if image.wcs is None or detection.wcs is None:
        return pixels, False
    ra, dec = detection.wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
    x, y = image.wcs.world_to_pixel(ra, dec)
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]), True


def forced_photometry(images: Mapping[str, AstroImage],
                      catalog: SourceCatalog,
                      detection_band: Optional[str] = None,
                      aperture_arcsec: float = 2.0,
                      homogenise_psf: bool = True,
                      target_fwhm_arcsec: Optional[float] = None,
                      use_kron: bool = False,
                      segmentation: Optional[np.ndarray] = None,
                      annulus_arcsec: Tuple[float, float] = (5.0, 9.0),
                      ) -> MultiBandReport:
    """Measure every source of ``catalog`` in every band, in place.

    ``images`` maps band name to image; the catalog's pixel coordinates are
    assumed to belong to ``detection_band`` (the first band by default).
    Each source gains a ``bands[name]`` entry.

    ``use_kron`` swaps the fixed circular aperture for the detection band's
    elliptical Kron aperture, scaled into each band's pixels.  That captures
    more of a large galaxy, at the cost of a noisier colour -- which is why
    surveys quote *aperture* colours and *Kron* total magnitudes.
    """
    if not images:
        raise DataError("forced photometry needs at least one band")
    band_names = list(images)
    detection_band = detection_band or band_names[0]
    if detection_band not in images:
        raise DataError(f"detection band {detection_band!r} is not among {band_names}")
    warnings: List[str] = []

    working: Dict[str, AstroImage] = dict(images)
    target = float("nan")
    changed: List[str] = []
    if homogenise_psf:
        working, target, changed = homogenise(working, target_fwhm_arcsec)
        if not changed and not np.isfinite(target):
            warnings.append("no PSF models available; colours are not seeing-corrected")

    detection = working[detection_band]
    for band, image in working.items():
        positions, used_wcs = _positions_in(image, catalog, detection)
        if not used_wcs and image is not detection:
            warnings.append(f"band {band}: no WCS, assuming a shared pixel grid")
        scale = _pixel_scale(image)
        radius = float(aperture_arcsec) / scale
        inner, outer = (annulus_arcsec[0] / scale, annulus_arcsec[1] / scale)
        data = image.subtracted()
        rms = image.rms_map()
        gain = float(image.header.get("GAIN", 1.0) or 1.0)
        zero_point = float(image.header.get("MAGZP", 25.0) or 25.0)
        detection_scale = _pixel_scale(detection)
        # Matching to a common width never gets the wings exactly right --
        # a Gaussian kernel applied to a Moffat leaves a Moffat-ish, not a
        # Gaussian -- so a fixed aperture still catches slightly different
        # fractions per band.  Correcting each band by the enclosed energy of
        # its *own post-matching* PSF removes what is left.
        aperture_correction = _enclosed_energy_correction(_psf_of(image), radius)

        measured = 0
        for source, (x, y) in zip(catalog, positions):
            if not (0 <= x < data.shape[1] and 0 <= y < data.shape[0]):
                source.bands[band] = Photometry()
                continue
            kron = use_kron and np.isfinite(source.photometry.aperture_radius)
            # The Kron aperture was measured in detection-band pixels;
            # convert through both scales so it covers the same sky.
            semi_major = (source.photometry.aperture_radius * detection_scale / scale
                          if kron else float("nan"))
            reach = max(outer, radius, semi_major if kron else 0.0) + 3.0
            rows, cols, centre = stamp_box(data.shape, (float(x), float(y)), reach)
            stamp, stamp_rms = data[rows, cols], rms[rows, cols]
            neighbours = _neighbour_mask(image, segmentation, source, rows, cols)
            if kron:
                axis = _axis_ratio(source)
                result = elliptical_photometry(
                    stamp, centre, max(semi_major, 1.5),
                    max(semi_major * axis, 1.0), source.morphology.position_angle,
                    rms=stamp_rms, gain=gain, mask=neighbours)
                flux, flux_err = result.flux, result.flux_err
                background = result.background
                used_radius = semi_major
            else:
                flux, flux_err, background = _circular_flux(
                    stamp, stamp_rms, centre, radius, gain, (inner, outer), neighbours)
                used_radius = radius

            flux = float(flux) * aperture_correction
            flux_err = float(flux_err) * aperture_correction
            magnitude, magnitude_err = flux_to_magnitude(flux, zero_point, flux_err)
            snr = float(flux / flux_err) if flux_err > 0 else float("nan")
            source.bands[band] = Photometry(
                flux=float(flux), flux_err=float(flux_err),
                magnitude=float(magnitude), magnitude_err=float(magnitude_err),
                background=float(background), snr=snr,
                zero_point=float(zero_point),
                aperture_radius=float(used_radius))
            measured += 1
        log.debug("band %s: measured %d/%d sources", band, measured, len(catalog))

    report = MultiBandReport(
        bands=band_names, detection_band=detection_band,
        aperture_arcsec=float(aperture_arcsec), target_fwhm_arcsec=float(target),
        homogenised=changed, n_sources=len(catalog), warnings=warnings)
    catalog.meta["multiband"] = report.to_dict()
    log.info("forced photometry in %d bands (%s) on %d sources; %s",
             len(band_names), ", ".join(band_names), len(catalog),
             f"homogenised {', '.join(changed)} to {target:.2f}\"" if changed
             else "no PSF homogenisation")
    for message in warnings:
        log.warning("%s", message)
    return report


def _enclosed_energy_correction(psf: Optional[PSFModel], radius: float) -> float:
    """Multiplicative correction for the PSF flux outside ``radius``.

    Returns 1.0 when there is no usable PSF, and when the implied correction
    is extreme -- an empirical stamp is itself truncated, so a correction
    above ~1.6 means the stamp, not the aperture, is the problem.
    """
    if psf is None or not np.isfinite(radius) or radius <= 0:
        return 1.0
    stamp = np.clip(psf.as_kernel(), 0, None)
    total = float(stamp.sum())
    if total <= 0:
        return 1.0
    centre = ((stamp.shape[1] - 1) / 2.0, (stamp.shape[0] - 1) / 2.0)
    weights = circular_aperture_weights(stamp.shape, centre, float(radius), subpixels=5)
    enclosed = float((stamp * weights).sum()) / total
    if not 0.3 < enclosed <= 1.0:
        return 1.0
    return float(1.0 / enclosed)


def _axis_ratio(source: Source) -> float:
    morphology = source.morphology
    if np.isfinite(morphology.semi_major) and morphology.semi_major > 0 \
            and np.isfinite(morphology.semi_minor):
        return float(np.clip(morphology.semi_minor / morphology.semi_major, 0.15, 1.0))
    return 1.0


def _neighbour_mask(image: AstroImage, segmentation: Optional[np.ndarray],
                    source: Source, rows: slice = slice(None), cols: slice = slice(None)
                    ) -> Optional[np.ndarray]:
    """Other objects' pixels plus the image mask, on the ``rows, cols`` stamp."""
    mask = None
    if segmentation is not None and segmentation.shape == image.shape:
        labels = segmentation[rows, cols]
        mask = (labels > 0) & (labels != source.segment_label)
    if image.mask is not None:
        local = image.mask[rows, cols]
        mask = local if mask is None else (mask | local)
    return mask


def _circular_flux(data: np.ndarray, rms: np.ndarray, centre: Tuple[float, float],
                   radius: float, gain: float, annulus: Tuple[float, float],
                   mask: Optional[np.ndarray]) -> Tuple[float, float, float]:
    """Aperture flux with a local sky and a variance that includes the source."""
    weights = circular_aperture_weights(data.shape, centre, max(radius, 1.0), subpixels=5)
    if mask is not None:
        weights = weights * (~mask)
    area = float(weights.sum())
    if area <= 0:
        return float("nan"), float("nan"), 0.0
    level, _, _ = annulus_background(data, centre, annulus[0], annulus[1], mask=mask)
    level = float(level)
    flux = float((data * weights).sum() - level * area)
    variance = float(((rms ** 2) * weights).sum())
    if gain > 0:
        variance += max(flux, 0.0) / gain
    return flux, float(np.sqrt(max(variance, 0.0))), level


def measure_colours(catalog: SourceCatalog,
                    pairs: Sequence[Tuple[str, str]],
                    min_snr: float = 5.0) -> Dict[str, int]:
    """Store colour indices on each source; returns how many were measurable.

    Colours live in ``meta["colours"]`` rather than as attributes because
    which pairs are useful depends entirely on which filters were observed.

    A colour is recorded only when *both* bands clear ``min_snr``.  This is
    not tidiness: a source detected at 40 sigma in the red can easily sit at
    1 sigma in the blue, and the difference of those two magnitudes is not a
    faint blue measurement but noise.  Left in, such values dominate the
    scatter of any colour-based cut and quietly destroy it.  Sources that
    fail carry a ``colour_limit`` entry instead, which *is* real information:
    an object undetected in the blue is at least that red.
    """
    counts = {f"{blue}-{red}": 0 for blue, red in pairs}
    for source in catalog:
        colours = source.meta.setdefault("colours", {})
        limits = source.meta.setdefault("colour_limits", {})
        for blue, red in pairs:
            key = f"{blue}-{red}"
            value = source.colour(blue, red)
            first, second = source.bands.get(blue), source.bands.get(red)
            if first is None or second is None or not np.isfinite(value):
                continue
            snr = (first.snr, second.snr)
            if all(np.isfinite(v) and v >= min_snr for v in snr):
                colours[key] = float(value)
                counts[key] += 1
                continue
            # One band is a non-detection: record the one-sided limit that
            # the *detected* band and the other's noise together imply.
            detected_blue = np.isfinite(snr[0]) and snr[0] >= min_snr
            detected_red = np.isfinite(snr[1]) and snr[1] >= min_snr
            if detected_blue == detected_red:
                continue
            faint = second if detected_blue else first
            limit_flux = max(float(min_snr) * float(faint.flux_err), 1e-9)
            zero_point = faint.zero_point if np.isfinite(faint.zero_point) else 25.0
            limit_mag, _ = flux_to_magnitude(limit_flux, zero_point)
            bright = first if detected_blue else second
            bound = float(bright.magnitude - limit_mag)
            limits[key] = {"bound": bound if detected_blue else -bound,
                           "direction": "redder_than" if detected_blue else "bluer_than",
                           "undetected_band": red if detected_blue else blue}
            source.add_flag("colour_limit")
    return counts


def band_flux_table(catalog: SourceCatalog, bands: Sequence[str]) -> np.ndarray:
    """``(n_sources, n_bands)`` magnitudes, with NaN where unmeasured."""
    table = np.full((len(catalog), len(bands)), np.nan, dtype=float)
    for row, source in enumerate(catalog):
        for column, band in enumerate(bands):
            photometry = source.bands.get(band)
            if photometry is not None:
                table[row, column] = photometry.magnitude
    return table
