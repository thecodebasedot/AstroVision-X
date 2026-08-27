"""Refitting the world coordinate system against a reference catalog.

A plate solution is a least-squares fit of the map from pixels to sky.  The
awkward part is not the fit -- it is linear once the projection is fixed --
but deciding *which* detected source corresponds to which reference star,
when the starting WCS may be several arcseconds out and the field contains
more detections than references, or the other way round.

The approach here is the practical one: match with a generous radius against
the initial guess, fit, then re-match with a radius set by the residual of
that fit and refit.  Two or three rounds converge, and each round throws out
the pairs the previous solution proved wrong.  It relies on the initial
pointing being roughly right -- within a few times the matching radius --
which is true of any modern telescope and is checked rather than assumed:
if too few pairs survive, the routine refuses to return a solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import SourceCatalog
from ..io.external import ReferenceObject
from ..io.wcs import SimpleWCS, angular_separation

log = get_logger("calibration.astrometry")


@dataclass
class AstrometricSolution:
    """A refitted WCS with the evidence for it."""

    wcs: Optional[SimpleWCS]
    n_matched: int
    rms_arcsec: float
    max_arcsec: float
    shift_arcsec: float                     # how far the solution moved the field
    rotation_deg: float                     # and how much it turned it
    scale_ratio: float                      # and how much it rescaled it
    order: int = 1
    succeeded: bool = False
    reason: str = ""
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "succeeded": bool(self.succeeded),
            "n_matched": int(self.n_matched),
            "rms_arcsec": float(self.rms_arcsec),
            "max_arcsec": float(self.max_arcsec),
            "shift_arcsec": float(self.shift_arcsec),
            "rotation_deg": float(self.rotation_deg),
            "scale_ratio": float(self.scale_ratio),
            "order": int(self.order),
            "reason": self.reason,
            "wcs": self.wcs.to_dict() if self.wcs is not None else None,
        }


def match_to_reference(catalog: SourceCatalog, reference: Sequence[ReferenceObject],
                       wcs: SimpleWCS, radius_arcsec: float = 5.0,
                       brightest: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair detections with reference objects under a trial ``wcs``.

    Returns ``(pixel_xy, reference_radec, separation_arcsec)``.  Matching is
    mutual-nearest: a detection and a reference must each be the other's
    closest partner.  One-sided nearest matching quietly assigns several
    detections to one bright reference star in a crowded field, and those
    duplicated pairs pull the fit toward that one star.

    ``brightest`` restricts the fit to the *n* highest signal-to-noise
    detections, which is usually what you want -- faint centroids are noisy
    and contribute little but scatter.
    """
    sources = [s for s in catalog if np.isfinite(s.x) and np.isfinite(s.y)]
    if brightest:
        sources = sorted(
            sources,
            key=lambda s: -(s.photometry.snr if np.isfinite(s.photometry.snr) else 0.0)
        )[:int(brightest)]
    if not sources or not reference:
        empty = np.zeros((0, 2))
        return empty, empty, np.zeros(0)

    pixels = np.array([[s.x, s.y] for s in sources], dtype=float)
    predicted_ra, predicted_dec = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
    ref_ra = np.array([o.ra for o in reference], dtype=float)
    ref_dec = np.array([o.dec for o in reference], dtype=float)

    separation = np.array([
        angular_separation(predicted_ra[i], predicted_dec[i], ref_ra, ref_dec) * 3600.0
        for i in range(len(sources))
    ])
    nearest_reference = np.argmin(separation, axis=1)
    nearest_source = np.argmin(separation, axis=0)

    rows, refs, distances = [], [], []
    for i, j in enumerate(nearest_reference):
        if nearest_source[j] != i:
            continue                                  # not mutual
        if separation[i, j] > float(radius_arcsec):
            continue
        rows.append(pixels[i])
        refs.append([ref_ra[j], ref_dec[j]])
        distances.append(separation[i, j])
    return (np.asarray(rows, dtype=float).reshape(-1, 2),
            np.asarray(refs, dtype=float).reshape(-1, 2),
            np.asarray(distances, dtype=float))


def _fit_linear(pixels: np.ndarray, intermediate: np.ndarray,
                crpix: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Least-squares CD matrix and reference-pixel offset.

    Fits ``xi = cd @ (pixel - crpix) + offset`` in the tangent plane, in
    degrees.  Solving in the *intermediate* coordinates rather than on the
    sphere is what makes this linear: the projection has already absorbed
    the spherical geometry, so what is left is an affine map.
    """
    du = pixels[:, 0] + 1.0 - crpix[0]
    dv = pixels[:, 1] + 1.0 - crpix[1]
    design = np.column_stack([du, dv, np.ones_like(du)])
    solution, *_ = np.linalg.lstsq(design, intermediate, rcond=None)
    cd = np.array([[solution[0, 0], solution[1, 0]],
                   [solution[0, 1], solution[1, 1]]], dtype=float)
    offset = (float(solution[2, 0]), float(solution[2, 1]))
    return cd, offset


def _to_intermediate(ra: np.ndarray, dec: np.ndarray,
                     crval: Tuple[float, float]) -> np.ndarray:
    """Gnomonic projection of sky coordinates about ``crval``, in degrees."""
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(crval[0]), np.radians(crval[1])
    cos_c = (np.sin(dec0) * np.sin(dec_r) +
             np.cos(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0))
    cos_c = np.where(np.abs(cos_c) < 1e-12, 1e-12, cos_c)
    xi = np.degrees(np.cos(dec_r) * np.sin(ra_r - ra0) / cos_c)
    eta = np.degrees((np.cos(dec0) * np.sin(dec_r) -
                      np.sin(dec0) * np.cos(dec_r) * np.cos(ra_r - ra0)) / cos_c)
    return np.column_stack([xi, eta])


def solve_plate(catalog: SourceCatalog, reference: Sequence[ReferenceObject],
                initial: SimpleWCS, radius_arcsec: float = 5.0,
                min_matches: int = 8, rounds: int = 3,
                clip_sigma: float = 3.0,
                brightest: Optional[int] = 300) -> AstrometricSolution:
    """Refit the WCS so detections land on their reference positions.

    Each round re-matches under the current solution with a radius drawn
    from the last round's residual, so the pairing tightens as the fit
    improves.  Sigma clipping between rounds removes blends and proper-motion
    outliers, which are the two things that reliably ruin an astrometric
    solution and are indistinguishable from a bad match until you have a fit
    good enough to see them.

    Returns a solution whose ``succeeded`` is ``False``, with the WCS left
    untouched, when there is not enough evidence -- a plate solution fitted
    to five stars is worse than the header you started with, because it looks
    authoritative.
    """
    if len(catalog) == 0 or not reference:
        return AstrometricSolution(
            wcs=None, n_matched=0, rms_arcsec=float("nan"), max_arcsec=float("nan"),
            shift_arcsec=0.0, rotation_deg=0.0, scale_ratio=1.0,
            reason="nothing to match")

    current = SimpleWCS(crpix=initial.crpix, crval=initial.crval,
                        cd=np.array(initial.cd, dtype=float), ctype=initial.ctype,
                        sip_a=initial.sip_a, sip_b=initial.sip_b,
                        sip_ap=initial.sip_ap, sip_bp=initial.sip_bp)
    search = float(radius_arcsec)
    best: Optional[AstrometricSolution] = None

    for round_index in range(max(1, int(rounds))):
        pixels, sky, separation = match_to_reference(
            catalog, reference, current, search, brightest=brightest)
        if len(pixels) < min_matches:
            reason = (f"only {len(pixels)} mutual matches within {search:.1f}\", "
                      f"need {min_matches}")
            return best or AstrometricSolution(
                wcs=None, n_matched=len(pixels), rms_arcsec=float("nan"),
                max_arcsec=float("nan"), shift_arcsec=0.0, rotation_deg=0.0,
                scale_ratio=1.0, reason=reason)

        if round_index > 0 and len(separation) > min_matches:
            spread = 1.4826 * float(np.median(np.abs(separation - np.median(separation))))
            if spread > 0:
                keep = separation <= np.median(separation) + clip_sigma * spread
                if keep.sum() >= min_matches:
                    pixels, sky, separation = pixels[keep], sky[keep], separation[keep]

        intermediate = _to_intermediate(sky[:, 0], sky[:, 1], current.crval)
        cd, offset = _fit_linear(pixels, intermediate, current.crpix)
        # The constant term says the reference pixel is not looking where the
        # header claims.  Rather than moving CRVAL -- which would change the
        # projection centre and invalidate the linearisation -- it is folded
        # back into CRPIX, where it belongs.
        try:
            shift = np.linalg.solve(cd, np.array(offset, dtype=float))
        except np.linalg.LinAlgError:                       # pragma: no cover
            return AstrometricSolution(
                wcs=None, n_matched=len(pixels), rms_arcsec=float("nan"),
                max_arcsec=float("nan"), shift_arcsec=0.0, rotation_deg=0.0,
                scale_ratio=1.0, reason="degenerate plate solution")
        current = SimpleWCS(
            crpix=(current.crpix[0] - float(shift[0]), current.crpix[1] - float(shift[1])),
            crval=current.crval, cd=cd, ctype=current.ctype,
            sip_a=current.sip_a, sip_b=current.sip_b,
            sip_ap=current.sip_ap, sip_bp=current.sip_bp)

        fitted_ra, fitted_dec = current.pixel_to_world(pixels[:, 0], pixels[:, 1])
        residual = angular_separation(fitted_ra, fitted_dec, sky[:, 0], sky[:, 1]) * 3600.0
        rms = float(np.sqrt(np.mean(residual ** 2)))
        best = AstrometricSolution(
            wcs=current, n_matched=len(pixels), rms_arcsec=rms,
            max_arcsec=float(np.max(residual)),
            shift_arcsec=_shift_between(initial, current),
            rotation_deg=float(current.orientation - initial.orientation),
            scale_ratio=float(current.pixel_scale / max(initial.pixel_scale, 1e-12)),
            succeeded=True, residuals=residual,
            reason=f"converged after {round_index + 1} round(s)")
        # Next round searches at a few times the residual, floored so a
        # very good fit does not immediately starve itself of matches.
        search = float(np.clip(4.0 * rms, 0.5, radius_arcsec))

    if best is not None:
        log.info("plate solution: %d stars, %.3f\" rms (max %.3f\"), field moved "
                 "%.2f\", rotated %.3f deg, scale x%.5f",
                 best.n_matched, best.rms_arcsec, best.max_arcsec,
                 best.shift_arcsec, best.rotation_deg, best.scale_ratio)
    return best or AstrometricSolution(
        wcs=None, n_matched=0, rms_arcsec=float("nan"), max_arcsec=float("nan"),
        shift_arcsec=0.0, rotation_deg=0.0, scale_ratio=1.0, reason="no solution")


def _shift_between(before: SimpleWCS, after: SimpleWCS) -> float:
    """How far the field centre moved between two solutions, in arcsec."""
    x = float(before.crpix[0]) - 1.0
    y = float(before.crpix[1]) - 1.0
    ra1, dec1 = before.pixel_to_world(x, y)
    ra2, dec2 = after.pixel_to_world(x, y)
    return float(angular_separation(ra1, dec1, ra2, dec2) * 3600.0)


def apply_solution(catalog: SourceCatalog, solution: AstrometricSolution) -> int:
    """Rewrite every source's sky position from a successful solution."""
    if not solution.succeeded or solution.wcs is None:
        return 0
    updated = 0
    for source in catalog:
        ra, dec = solution.wcs.pixel_to_world(source.x, source.y)
        source.ra = float(np.atleast_1d(ra)[0])
        source.dec = float(np.atleast_1d(dec)[0])
        updated += 1
    catalog.meta["astrometry"] = solution.to_dict()
    return updated
