"""Deriving a real photometric zero point from standards in the frame.

An instrumental magnitude ``-2.5 log10(flux)`` is a number about a detector.
The zero point is what turns it into a number about the sky::

    m_catalog = m_instrumental + zp + k * colour

The colour term is not optional refinement.  A filter is never exactly the
reference survey's filter, so the offset between the two systems depends on
the *shape* of the source's spectrum -- which colour measures.  Fitting only
a constant leaves that dependence in the residuals, where it masquerades as
scatter and puts a systematic, colour-dependent error into every magnitude:
blue stars too bright, red stars too faint, or the reverse.

The fit is robust by construction.  Standards get mismatched, blended, and
saturated, and a single such outlier moves an ordinary least-squares zero
point by more than the precision anyone wants from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import SourceCatalog
from ..io.external import ReferenceObject
from ..io.wcs import angular_separation

log = get_logger("calibration.photometry")


@dataclass
class PhotometricSolution:
    """A fitted zero point with the evidence behind it."""

    zero_point: float = float("nan")
    zero_point_err: float = float("nan")
    colour_term: float = 0.0
    colour_term_err: float = float("nan")
    colour_pair: Optional[Tuple[str, str]] = None
    band: str = ""
    reference_band: str = ""
    n_standards: int = 0
    n_rejected: int = 0
    rms: float = float("nan")
    succeeded: bool = False
    reason: str = ""
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "succeeded": bool(self.succeeded),
            "zero_point": float(self.zero_point),
            "zero_point_err": float(self.zero_point_err),
            "colour_term": float(self.colour_term),
            "colour_term_err": float(self.colour_term_err),
            "colour_pair": (None if self.colour_pair is None
                            else list(self.colour_pair)),
            "band": self.band, "reference_band": self.reference_band,
            "n_standards": int(self.n_standards),
            "n_rejected": int(self.n_rejected),
            "rms": float(self.rms), "reason": self.reason,
        }


def _instrumental(flux: float) -> float:
    """Instrumental magnitude, with no zero point in it."""
    return float(-2.5 * np.log10(flux)) if flux > 0 else float("nan")


def collect_standards(catalog: SourceCatalog, reference: Sequence[ReferenceObject],
                      band: str, reference_band: Optional[str] = None,
                      radius_arcsec: float = 2.0,
                      min_snr: float = 20.0,
                      colour_pair: Optional[Tuple[str, str]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Gather ``(instrumental, catalogued, colour)`` for usable standards.

    Only high signal-to-noise, unsaturated, unflagged sources qualify.  This
    is not fussiness: the zero point is a single number derived from these
    stars, so one saturated one -- whose measured flux is capped and whose
    instrumental magnitude is therefore too faint -- biases every magnitude
    in the catalog.
    """
    if not reference:
        return np.zeros(0), np.zeros(0), np.zeros(0), ""
    ref_ra = np.array([o.ra for o in reference], dtype=float)
    ref_dec = np.array([o.dec for o in reference], dtype=float)

    available: Dict[str, int] = {}
    for obj in reference:
        for name in obj.magnitudes:
            available[name] = available.get(name, 0) + 1
    if reference_band is None:
        reference_band = band if band in available else (
            max(available, key=available.get) if available else "")
    if not reference_band:
        return np.zeros(0), np.zeros(0), np.zeros(0), ""

    instrumental: List[float] = []
    catalogued: List[float] = []
    colours: List[float] = []
    for source in catalog:
        if source.ra is None or source.dec is None:
            continue
        if "saturated" in source.flags or "blended" in source.flags:
            continue
        photometry = source.bands.get(band, source.photometry)
        if photometry is None or not np.isfinite(photometry.flux) or photometry.flux <= 0:
            continue
        if not np.isfinite(photometry.snr) or photometry.snr < float(min_snr):
            continue
        separation = angular_separation(float(source.ra), float(source.dec),
                                        ref_ra, ref_dec) * 3600.0
        best = int(np.argmin(separation))
        if separation[best] > float(radius_arcsec):
            continue
        value = reference[best].magnitudes.get(reference_band)
        if value is None or not np.isfinite(value):
            continue
        instrumental.append(_instrumental(photometry.flux))
        catalogued.append(float(value))
        colours.append(source.colour(*colour_pair) if colour_pair else float("nan"))

    return (np.asarray(instrumental, dtype=float),
            np.asarray(catalogued, dtype=float),
            np.asarray(colours, dtype=float),
            reference_band)


def solve_zero_point(catalog: SourceCatalog, reference: Sequence[ReferenceObject],
                     band: str = "", reference_band: Optional[str] = None,
                     colour_pair: Optional[Tuple[str, str]] = None,
                     radius_arcsec: float = 2.0, min_standards: int = 5,
                     min_snr: float = 20.0, clip_sigma: float = 3.0,
                     iterations: int = 4) -> PhotometricSolution:
    """Fit ``zp`` (and a colour term, when colours are available).

    The colour term is fitted only when enough standards carry a finite
    colour *and* span a wide enough range of it -- fitting a slope to points
    all at the same colour returns a large, meaningless coefficient with a
    small formal error, which is the worst possible combination.
    """
    instrumental, catalogued, colours, used_band = collect_standards(
        catalog, reference, band, reference_band, radius_arcsec, min_snr, colour_pair)
    if instrumental.size < min_standards:
        return PhotometricSolution(
            band=band, reference_band=used_band, n_standards=int(instrumental.size),
            reason=f"only {instrumental.size} usable standards, need {min_standards}")

    offset = catalogued - instrumental
    usable_colour = np.isfinite(colours)
    colour_span = (float(np.ptp(colours[usable_colour]))
                   if usable_colour.sum() >= min_standards else 0.0)
    fit_colour = colour_pair is not None and colour_span >= 0.3

    keep = np.ones(instrumental.size, dtype=bool)
    if fit_colour:
        keep &= usable_colour
    zero_point, colour_term = float(np.median(offset[keep])), 0.0
    rejected = 0

    for _ in range(max(1, int(iterations))):
        if fit_colour and keep.sum() >= min_standards:
            design = np.column_stack([np.ones(keep.sum()), colours[keep]])
            coefficients, *_ = np.linalg.lstsq(design, offset[keep], rcond=None)
            zero_point, colour_term = float(coefficients[0]), float(coefficients[1])
            model = zero_point + colour_term * colours
        else:
            zero_point = float(np.median(offset[keep]))
            colour_term = 0.0
            model = np.full(offset.shape, zero_point)
        residual = offset - model
        spread = 1.4826 * float(np.median(np.abs(residual[keep] -
                                                 np.median(residual[keep]))))
        if not np.isfinite(spread) or spread <= 0:
            break
        updated = keep & (np.abs(residual - np.median(residual[keep])) <=
                          clip_sigma * spread)
        if updated.sum() < min_standards or np.array_equal(updated, keep):
            break
        rejected += int(keep.sum() - updated.sum())
        keep = updated

    residual = offset - (zero_point + colour_term * np.where(usable_colour, colours, 0.0))
    rms = float(np.std(residual[keep])) if keep.sum() > 1 else float("nan")
    count = int(keep.sum())
    # The error on the mean, not the scatter of one star: what is wanted is
    # how well the *zero point* is known, and that improves with the number
    # of standards even though the per-star scatter does not.
    zero_point_err = float(rms / np.sqrt(count)) if count > 1 else float("nan")
    colour_err = float("nan")
    if fit_colour and count > 2:
        spread_colour = float(np.std(colours[keep]))
        if spread_colour > 0:
            colour_err = float(rms / (np.sqrt(count) * spread_colour))

    solution = PhotometricSolution(
        zero_point=zero_point, zero_point_err=zero_point_err,
        colour_term=colour_term, colour_term_err=colour_err,
        colour_pair=colour_pair if fit_colour else None,
        band=band, reference_band=used_band, n_standards=count,
        n_rejected=rejected, rms=rms, succeeded=True,
        residuals=residual[keep],
        reason=("fitted with a colour term" if fit_colour else
                ("no colours available" if colour_pair is None else
                 f"colour range {colour_span:.2f} mag is too narrow for a colour term")))
    log.info("zero point in %s vs catalogue %s: %.4f +- %.4f from %d standards "
             "(%d rejected), colour term %.4f, scatter %.4f mag",
             band or "detection", used_band, zero_point, zero_point_err,
             count, rejected, colour_term, rms)
    return solution


def apply_zero_point(catalog: SourceCatalog, solution: PhotometricSolution,
                     band: str = "") -> int:
    """Recompute magnitudes from a fitted solution; returns how many changed.

    The colour term is applied per source, so two stars with the same flux
    and different colours correctly come out with different magnitudes.
    Sources without a measured colour get the constant part only and are
    flagged, because their magnitudes carry a systematic error the others
    do not.
    """
    if not solution.succeeded:
        return 0
    band = band or solution.band
    updated = 0
    for source in catalog:
        photometry = source.bands.get(band) if band else None
        if photometry is None:
            photometry = source.photometry
        if not np.isfinite(photometry.flux) or photometry.flux <= 0:
            continue
        colour = (source.colour(*solution.colour_pair)
                  if solution.colour_pair else float("nan"))
        correction = solution.zero_point
        if solution.colour_pair is not None:
            if np.isfinite(colour):
                correction += solution.colour_term * colour
            else:
                source.add_flag("no_colour_term")
        photometry.magnitude = float(_instrumental(photometry.flux) + correction)
        photometry.zero_point = float(correction)
        if np.isfinite(photometry.flux_err) and photometry.flux_err > 0:
            photometry.magnitude_err = float(
                np.hypot(1.0857 * photometry.flux_err / photometry.flux,
                         solution.zero_point_err))
        updated += 1
    catalog.meta.setdefault("photometric_calibration", {})[band or "detection"] = \
        solution.to_dict()
    return updated
