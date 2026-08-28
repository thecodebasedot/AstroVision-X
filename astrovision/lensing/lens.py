"""Strong-lens candidate scoring.

A lens candidate is a galaxy that is (a) massive and early-type enough to
lens, and (b) surrounded by tangential arcs at a consistent radius.  Both
conditions are required; either alone produces mostly false positives, and
the literature is full of them.  Confirmed lensing needs colour information
and, ultimately, spectroscopic redshifts for both the deflector and the
source -- so nothing here is more than a ranked candidate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import LensingConfig
from ..core.logging import get_logger
from ..core.numeric import logistic, sigma_clipped_stats
from ..core.types import (
    LensCandidate,
    Morphology,
    ObjectClass,
    Source,
    SourceCatalog,
    Verdict,
)
from ..io.image import AstroImage
from .arcs import Arc, detect_arcs, detect_ring, einstein_radius, ring_completeness
from .model import arc_sample_points, einstein_mass, fit_lens_model

log = get_logger("lensing.lens")

#: Speed of light in km/s, for the velocity-dispersion estimate.
C_KM_S = 299_792.458


def deflector_plausibility(source: Source) -> Tuple[float, Dict[str, float]]:
    """How plausible is this object as a lensing galaxy?

    Strong lenses are overwhelmingly massive early-type galaxies: they have
    the concentrated mass needed for a large Einstein radius.  A faint
    irregular is not going to lens anything detectable.
    """
    terms: Dict[str, float] = {}
    morphology = source.morphology

    if source.object_class == ObjectClass.STAR:
        return 0.0, {"is_star": 0.0}

    if morphology.label in (Morphology.ELLIPTICAL, Morphology.LENTICULAR):
        terms["early_type"] = 1.0
    elif morphology.label in (Morphology.SPIRAL, Morphology.BARRED_SPIRAL):
        terms["early_type"] = 0.35
    elif morphology.label == Morphology.UNKNOWN:
        terms["early_type"] = 0.5
    else:
        terms["early_type"] = 0.2

    if np.isfinite(morphology.concentration):
        terms["concentrated"] = float(logistic(morphology.concentration,
                                               scale=0.4, midpoint=3.2))
    if np.isfinite(morphology.sersic_index):
        terms["steep_profile"] = float(logistic(morphology.sersic_index,
                                                scale=0.8, midpoint=2.5))
    # Resolved enough that arcs could be separated from the galaxy at all.
    terms["resolved"] = float(np.clip(morphology.area_pixels / 60.0, 0.0, 1.0))
    if np.isfinite(source.photometry.snr):
        terms["bright"] = float(logistic(source.photometry.snr, scale=8.0, midpoint=20.0))

    usable = [v for v in terms.values() if np.isfinite(v)]
    return (float(np.mean(usable)) if usable else 0.0), terms


def velocity_dispersion(theta_e_arcsec: float, z_lens: float = 0.5,
                        z_source: float = 2.0) -> float:
    """Singular-isothermal-sphere velocity dispersion implied by an arc radius.

    For an SIS lens, ``theta_E = 4 pi (sigma/c)^2 D_ls/D_s``.  Without
    redshifts the distance ratio must be assumed, so this is an
    order-of-magnitude sanity check -- a candidate implying 600 km/s is
    almost certainly not a galaxy-scale lens -- and nothing more.
    """
    if not np.isfinite(theta_e_arcsec) or theta_e_arcsec <= 0:
        return float("nan")
    from ..astrophysics.cosmology import Cosmology

    cosmology = Cosmology()
    d_s = cosmology.angular_diameter_distance(z_source)
    d_ls = cosmology.angular_diameter_distance_between(z_lens, z_source)
    if d_s <= 0 or d_ls <= 0:
        return float("nan")
    theta_rad = np.deg2rad(theta_e_arcsec / 3600.0)
    ratio = theta_rad / (4.0 * np.pi) * (d_s / d_ls)
    return float(C_KM_S * np.sqrt(max(ratio, 0.0)))


class LensSearch:
    """Search a catalog for strong-lensing morphology.

    >>> from astrovision.simulate import quick_field
    >>> from astrovision.preprocess import Preprocessor
    >>> from astrovision.detect import Detector
    >>> image, _ = quick_field((192, 192), n_lenses=1)
    >>> clean = Preprocessor().run(image)
    >>> catalog, _ = Detector().detect(clean)
    >>> isinstance(LensSearch().run(clean, catalog), list)
    True
    """

    def __init__(self, config: Optional[LensingConfig] = None):
        self.config = config or LensingConfig()
        self.report: Dict[str, Any] = {}

    def run(self, image: AstroImage, catalog: SourceCatalog) -> List[LensCandidate]:
        """Score every plausible deflector; returns the candidates found."""
        cfg = self.config
        if not cfg.enabled or len(catalog) == 0:
            return []

        data = image.subtracted()
        pixel_scale = image.pixel_scale if image.wcs is not None else 1.0
        psf = image.meta.get("psf_model")
        psf_fwhm = float(psf.fwhm) if psf is not None else 3.0
        _, _, global_noise = sigma_clipped_stats(data)
        candidates: List[LensCandidate] = []
        n_examined = 0

        for source in catalog:
            plausibility, terms = deflector_plausibility(source)
            if plausibility < 0.35:
                continue
            n_examined += 1

            # A compact deflector still lenses at a radius set by its mass,
            # not by its own light, so the search field must not shrink to
            # the galaxy's isophote.
            reach = max(cfg.search_radius_factor * max(source.morphology.semi_major, 2.0),
                        6.0 * psf_fwhm, 20.0)
            size = int(2 * np.ceil(reach) + 1)
            cutout = image.cutout(source.x, source.y, size, subtract_background=True)
            centre = ((cutout.shape[1] - 1) / 2.0, (cutout.shape[0] - 1) / 2.0)
            local_noise = float(source.meta.get("local_rms", global_noise) or global_noise)

            arcs = detect_arcs(cutout, centre, local_noise,
                               threshold_sigma=2.5, min_area=8,
                               min_axis_ratio=cfg.min_axis_ratio,
                               max_width=cfg.max_arc_width,
                               min_radius=max(3.0,
                                              0.9 * source.morphology.semi_major))
            arcs = [a for a in arcs if a.length >= cfg.min_arc_length]

            # A complete Einstein ring fills its own radius, so the
            # azimuthal baseline the arc finder relies on cannot see it.
            # The radial profile can, and that is an independent path in.
            ring_scan = detect_ring(cutout, centre, local_noise)
            if not arcs and not ring_scan["ring_detected"]:
                source.lens_score = 0.0
                continue
            if not arcs:
                arcs = [Arc(radius=ring_scan["radius"], angle=0.0,
                            length=2 * np.pi * ring_scan["radius"],
                            width=max(cfg.max_arc_width * 0.5, 1.5),
                            axis_ratio=float(cfg.min_axis_ratio),
                            tangential_alignment=1.0,
                            peak_significance=ring_scan["significance"],
                            flux=max(ring_scan["excess"], 1e-6),
                            area=int(2 * np.pi * ring_scan["radius"]))]
                source.add_flag("einstein_ring_candidate")

            # Multiple images of one source share a radius.  Discarding
            # features far from the median radius removes unrelated
            # neighbours that happen to fall in the search box, which would
            # otherwise inflate the Einstein-radius scatter.
            arcs = _consistent_radii(arcs)
            theta_e, scatter = einstein_radius(arcs)
            ring = ring_completeness(cutout, centre, theta_e,
                                     width=max(cfg.max_arc_width * 0.6, 2.0),
                                     n_bins=cfg.ring_bins, noise=local_noise)

            score, breakdown = self._score(arcs, theta_e, scatter, ring,
                                           plausibility, source)
            source.lens_score = float(score)
            source.meta["lensing"] = {
                "arcs": [a.to_dict() for a in arcs[:6]],
                "einstein_radius_px": float(theta_e),
                "radius_scatter": float(scatter),
                "ring": ring,
                "ring_scan": ring_scan,
                "deflector_terms": terms,
                "score_breakdown": breakdown,
            }

            if score < cfg.score_threshold:
                continue

            candidate = LensCandidate(
                source_id=source.id, score=float(score),
                arc_count=len(arcs),
                max_arc_length=float(max(a.length for a in arcs)),
                arc_curvature=float(np.mean([a.tangential_alignment for a in arcs])),
                ring_completeness=float(ring["completeness"]),
                einstein_radius_px=float(theta_e),
                einstein_radius_arcsec=float(theta_e * pixel_scale),
                verdict=_verdict(score),
                notes=_notes(arcs, ring, theta_e * pixel_scale, breakdown),
            )
            if cfg.fit_model:
                self._fit_model(candidate, source, arcs, centre, theta_e,
                                pixel_scale)
            candidates.append(candidate)
            source.add_flag("lens_candidate")
            if source.object_class in (ObjectClass.GALAXY, ObjectClass.UNKNOWN):
                source.meta["original_class"] = source.object_class.value

        candidates.sort(key=lambda c: -c.score)
        self.report = {
            "n_examined": n_examined,
            "n_candidates": len(candidates),
            "score_threshold": cfg.score_threshold,
            "pixel_scale": float(pixel_scale),
        }
        log.info("lens search: %d candidates from %d plausible deflectors",
                 len(candidates), n_examined)
        return candidates

    def _fit_model(self, candidate, source, arcs, centre, theta_e_px: float,
                   pixel_scale: float) -> None:
        """Fit a mass model to the arcs and attach the mass it implies.

        The fit is attempted, not assumed: with one short arc there are fewer
        constraints than parameters and the routine refuses, which is
        recorded on the candidate rather than passed over.  A candidate that
        cannot be modelled is still a candidate.
        """
        cfg = self.config
        points = arc_sample_points(arcs, centre, per_arc=cfg.points_per_arc)
        fit = fit_lens_model(points, centre, theta_e_guess=theta_e_px,
                             fit_shear=cfg.fit_shear,
                             bootstrap=cfg.model_bootstrap)
        candidate.model = fit.to_dict()
        if not fit.succeeded or fit.model is None:
            candidate.notes.append(f"No mass model: {fit.reason}")
            return

        candidate.model_theta_e_arcsec = float(fit.model.theta_e * pixel_scale)
        candidate.model_axis_ratio = float(fit.model.axis_ratio)
        candidate.model_shear = float(fit.model.shear_magnitude)
        candidate.model_image_rms_px = float(fit.image_rms)

        # A mass needs both redshifts.  The lens's own may have been measured
        # photometrically; the source's essentially never has been from
        # imaging alone, so it is assumed and labelled as assumed.
        photoz = source.meta.get("photoz") or {}
        z_lens = float(photoz.get("z", float("nan")))
        z_lens_source = "photometric"
        if not np.isfinite(z_lens) or z_lens <= 0:
            z_lens, z_lens_source = cfg.assumed_lens_redshift, "assumed"
        z_source = cfg.assumed_source_redshift
        mass = einstein_mass(candidate.model_theta_e_arcsec, z_lens, z_source)
        mass["z_lens_source"] = z_lens_source
        mass["z_source_source"] = "assumed"
        candidate.mass = mass
        if np.isfinite(mass.get("log_mass_solar", float("nan"))):
            candidate.notes.append(
                f"Mass model: Einstein radius {candidate.model_theta_e_arcsec:.2f} "
                f"arcsec, axis ratio {fit.model.axis_ratio:.2f}, external shear "
                f"{fit.model.shear_magnitude:.3f}; implied projected mass "
                f"{mass['mass_solar']:.2e} solar masses inside it, for a "
                f"{z_lens_source} lens redshift {z_lens:.2f} and an assumed "
                f"source redshift {z_source:.1f}. The mass scales as the square "
                "of the Einstein radius and with the distance ratio, so an "
                "assumed redshift is where its error lives.")
        for flag in fit.flags:
            candidate.notes.append(f"Model note: {flag.replace('_', ' ')}.")

    def _score(self, arcs: List[Arc], theta_e: float, scatter: float,
               ring: Dict[str, float], plausibility: float,
               source: Source) -> Tuple[float, Dict[str, float]]:
        """Combine the geometric evidence into a single 0-1 score."""
        cfg = self.config
        breakdown: Dict[str, float] = {"deflector": float(plausibility)}

        # More arcs is better: multiple images are the defining prediction.
        breakdown["multiplicity"] = float(np.clip((len(arcs) - 1) / 3.0, 0.0, 1.0))
        breakdown["elongation"] = float(np.clip(
            (np.mean([a.axis_ratio for a in arcs]) - cfg.min_axis_ratio) / 3.0, 0.0, 1.0))
        breakdown["tangential"] = float(np.clip(
            (np.mean([a.tangential_alignment for a in arcs]) - 0.5) / 0.45, 0.0, 1.0))

        # Arcs at a *consistent* radius is the strongest single signal: a
        # chance alignment of neighbours has no reason to share one radius.
        if len(arcs) > 1 and np.isfinite(scatter) and theta_e > 0:
            breakdown["radius_consistency"] = float(np.clip(
                1.0 - scatter / (0.35 * theta_e), 0.0, 1.0))
        else:
            breakdown["radius_consistency"] = 0.3

        breakdown["ring"] = float(np.clip(ring.get("completeness", 0.0) / 0.6, 0.0, 1.0))

        # The Einstein radius must be resolvable but not absurdly large for
        # a galaxy-scale deflector.
        size = max(source.morphology.semi_major, 1.0)
        breakdown["scale"] = float(np.exp(-0.5 * ((theta_e / size - 1.8) / 1.6) ** 2)) \
            if np.isfinite(theta_e) else 0.0

        weights = {"deflector": 1.4, "multiplicity": 1.2, "elongation": 1.0,
                   "tangential": 1.5, "radius_consistency": 1.5, "ring": 1.0,
                   "scale": 0.8}
        total = sum(weights[k] for k in breakdown)
        score = sum(v * weights[k] for k, v in breakdown.items()) / total
        return float(np.clip(score, 0.0, 1.0)), breakdown


def _consistent_radii(arcs: List[Arc], tolerance: float = 0.45) -> List[Arc]:
    """Keep only the arcs whose radii agree, weighted by flux."""
    if len(arcs) < 2:
        return arcs
    radii = np.array([a.radius for a in arcs], dtype=float)
    weights = np.array([max(a.flux, 1e-9) for a in arcs], dtype=float)
    centre = float(np.average(radii, weights=weights))
    keep = [a for a, r in zip(arcs, radii) if abs(r - centre) <= tolerance * centre]
    return keep or [max(arcs, key=lambda a: a.flux)]


def _verdict(score: float) -> Verdict:
    if score >= 0.75:
        return Verdict.HIGH_PRIORITY
    if score >= 0.6:
        return Verdict.FOLLOW_UP_RECOMMENDED
    if score >= 0.45:
        return Verdict.WORTH_A_LOOK
    return Verdict.NOT_INTERESTING


def _notes(arcs: List[Arc], ring: Dict[str, float], theta_e_arcsec: float,
           breakdown: Dict[str, float]) -> List[str]:
    """Human-readable reasons behind a lens candidate's score."""
    notes = [f"{len(arcs)} tangential arc(s) detected, "
             f"longest {max(a.length for a in arcs):.1f} px"]
    if ring.get("completeness", 0.0) > 0.55:
        notes.append(f"ring is {100 * ring['completeness']:.0f}% complete "
                     "-- possible Einstein ring")
    if breakdown.get("radius_consistency", 0.0) > 0.7:
        notes.append("arcs share a common radius, as multiple images of one source would")
    if np.isfinite(theta_e_arcsec) and theta_e_arcsec > 0:
        sigma = velocity_dispersion(theta_e_arcsec)
        notes.append(f"Einstein radius {theta_e_arcsec:.2f} arcsec, implying "
                     f"sigma ~ {sigma:.0f} km/s for an SIS lens at z=0.5 "
                     "(assumed redshifts)")
        if np.isfinite(sigma) and sigma > 450:
            notes.append("implied dispersion is cluster-scale, not galaxy-scale "
                         "-- treat with caution")
    notes.append("Candidate only: confirmation needs colour information and "
                 "spectroscopic redshifts for both deflector and source.")
    return notes
