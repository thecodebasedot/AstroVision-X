"""Star/galaxy separation and object-class assignment.

Telling a star from a galaxy is the oldest classification problem in
survey astronomy, and it has a clean physical basis: a star is a point
source, so its light profile *is* the point-spread function.  Anything
measurably broader is resolved.  The other classes -- nebulae, clusters,
artefacts -- are then separated by size, surface brightness and structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import logistic, softmax
from ..core.types import Morphology, ObjectClass, Source, SourceCatalog

log = get_logger("classify.rules")


def stellarity(source: Source, psf_fwhm: float,
               psf_r90: Optional[float] = None) -> float:
    """Probability that a source is unresolved, in ``[0, 1]``.

    Every test is expressed as a *ratio to the measured PSF*, so the same
    thresholds hold whatever the seeing was.  Several independent size
    measures are combined because at low signal-to-noise noise alone can
    broaden or narrow any single one.

    The thresholds sit between the stellar and galaxy loci measured on
    simulated fields: half-light radius separates the two populations most
    cleanly, followed by peak-to-total flux, then the isophotal width.
    """
    if psf_fwhm <= 0:
        return 0.5
    votes: List[Tuple[float, float]] = []      # (value, weight)
    morphology = source.morphology

    r50 = source.meta.get("r50", float("nan"))
    if np.isfinite(r50) and r50 > 0:
        # A point source has r50 close to half the PSF FWHM; a resolved
        # galaxy is well above it.  This is the cleanest single separator.
        votes.append((float(logistic(-(r50 / (0.5 * psf_fwhm)), scale=0.18,
                                     midpoint=-1.55)), 1.6))

    if np.isfinite(source.photometry.peak) and np.isfinite(source.photometry.flux) \
            and source.photometry.flux > 0:
        # Peak-to-total flux is the classic compactness statistic: a point
        # source concentrates the maximum possible fraction into one pixel.
        expected = 1.0 / max(1.13 * psf_fwhm ** 2, 1e-9)
        ratio = (source.photometry.peak / source.photometry.flux) / expected
        votes.append((float(logistic(ratio, scale=0.12, midpoint=0.50)), 1.4))

    if np.isfinite(morphology.fwhm) and morphology.fwhm > 0:
        # Isophotal second-moment width, which sits slightly above the true
        # PSF width even for stars because of the detection threshold.
        votes.append((float(logistic(-(morphology.fwhm / psf_fwhm), scale=0.13,
                                     midpoint=-1.40)), 1.2))

    r90 = source.meta.get("r90", float("nan"))
    if psf_r90 is not None and psf_r90 > 0 and np.isfinite(r90) and r90 > 0:
        # Noisier than r50 -- the outer profile is faint -- so weighted less.
        votes.append((float(logistic(-(r90 / psf_r90), scale=0.35, midpoint=-1.50)), 0.6))

    if not votes:
        return 0.5
    total = sum(weight for _, weight in votes)
    value = float(sum(v * weight for v, weight in votes) / total)

    # A very low signal-to-noise measurement cannot support a confident
    # answer, so pull it back toward the undecided middle.
    snr = source.photometry.snr
    if np.isfinite(snr) and snr < 10:
        weight = float(np.clip(snr / 10.0, 0.15, 1.0))
        value = 0.5 + weight * (value - 0.5)
    return float(np.clip(value, 0.0, 1.0))


def field_reference(catalog: SourceCatalog, psf_fwhm: float) -> Dict[str, float]:
    """Median properties of the field's resolved sources.

    Nebulae and star clusters are defined relative to the ordinary galaxy
    population -- more diffuse, or brighter and granular -- so the
    thresholds are set from the field itself.  That keeps them independent
    of the zero point, the exposure depth and the pixel scale.
    """
    resolved = [s for s in catalog
                if s.morphology.area_pixels > 20
                and np.isfinite(s.meta.get("r50", np.nan))
                and s.meta["r50"] > 0.8 * psf_fwhm]
    if len(resolved) < 5:
        resolved = [s for s in catalog if s.morphology.area_pixels > 20] or list(catalog)
    if not resolved:
        return {"area": 100.0, "r50": 2.0 * psf_fwhm, "surface_brightness": float("nan")}

    def median(values):
        finite = [v for v in values if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    return {
        "area": median([s.morphology.area_pixels for s in resolved]),
        "r50": median([s.meta.get("r50", np.nan) for s in resolved]),
        "surface_brightness": median([s.photometry.surface_brightness for s in resolved]),
        "n_resolved": float(len(resolved)),
    }


def classify_source(source: Source, psf_fwhm: float, psf_r90: Optional[float] = None,
                    threshold: float = 0.5, pixel_scale: float = 1.0,
                    reference: Optional[Dict[str, float]] = None
                    ) -> Tuple[ObjectClass, float, Dict[str, float]]:
    """Assign an object class with a confidence and per-class scores."""
    morphology = source.morphology
    point_like = stellarity(source, psf_fwhm, psf_r90)
    area = morphology.area_pixels
    surface_brightness = source.photometry.surface_brightness

    evidence: Dict[str, float] = {
        ObjectClass.STAR.value: 0.0,
        ObjectClass.GALAXY.value: 0.0,
        ObjectClass.NEBULA.value: 0.0,
        ObjectClass.STAR_CLUSTER.value: 0.0,
        ObjectClass.ARTIFACT.value: 0.0,
    }

    evidence[ObjectClass.STAR.value] = 3.0 * point_like
    resolved = 1.0 - point_like
    evidence[ObjectClass.GALAXY.value] = 3.0 * resolved

    reference = reference or {}
    reference_sb = reference.get("surface_brightness", float("nan"))
    reference_r50 = reference.get("r50", 2.0 * psf_fwhm)
    reference_area = reference.get("area", 100.0)
    r50 = source.meta.get("r50", float("nan"))
    components = int(source.meta.get("watershed_components", 1) or 1)

    # A nebula is diffuse: spread over a larger radius than the field's
    # galaxies at a *fainter* surface brightness (magnitudes: larger is
    # fainter), and without a galaxy's central concentration.
    if np.isfinite(reference_sb) and np.isfinite(surface_brightness) and np.isfinite(r50):
        faint = float(logistic(surface_brightness - reference_sb, scale=0.30, midpoint=0.55))
        spread = float(logistic(r50 / max(reference_r50, 1e-6), scale=0.25, midpoint=1.45))
        diffuse = float(np.sqrt(faint * spread))
        evidence[ObjectClass.NEBULA.value] = 3.4 * diffuse * resolved
        evidence[ObjectClass.GALAXY.value] *= (1.0 - 0.55 * diffuse)

    # A star cluster is granular rather than smooth: several peaks inside
    # one footprint, over a large area, at a brighter surface brightness
    # than the field's galaxies.
    if components >= 2 and area > 1.8 * reference_area:
        granular = float(np.clip((components - 1) / 3.0, 0.0, 1.0))
        big = float(np.clip((area / max(reference_area, 1.0) - 1.8) / 2.0, 0.0, 1.0))
        bright = 0.5
        if np.isfinite(reference_sb) and np.isfinite(surface_brightness):
            bright = float(logistic(reference_sb - surface_brightness,
                                    scale=0.30, midpoint=0.35))
        cluster = float((granular * big * bright) ** (1.0 / 3.0))
        evidence[ObjectClass.STAR_CLUSTER.value] = 3.4 * cluster * resolved
        evidence[ObjectClass.GALAXY.value] *= (1.0 - 0.5 * cluster)

    # Artefacts: implausibly elongated, or on masked/saturated pixels.
    artifact = 0.0
    if np.isfinite(morphology.elongation) and morphology.elongation > 5.0:
        artifact += float(np.clip((morphology.elongation - 5.0) / 5.0, 0.0, 1.0))
    if "saturated" in source.flags:
        artifact += 0.4
    if "masked_pixels" in source.flags and area < 12:
        artifact += 0.4
    if area <= 3:
        artifact += 0.5
    evidence[ObjectClass.ARTIFACT.value] = 2.5 * min(artifact, 1.0)

    # A confident morphological type is direct evidence of a galaxy.
    if morphology.label not in (Morphology.UNKNOWN, Morphology.UNRESOLVED) \
            and morphology.label_confidence > 0.4:
        evidence[ObjectClass.GALAXY.value] += 1.5 * morphology.label_confidence

    names = list(evidence)
    values = np.array([evidence[k] for k in names], dtype=float)
    if float(values.max()) <= 0:
        return ObjectClass.UNKNOWN, 0.0, {}
    probabilities = softmax(values)
    order = np.argsort(probabilities)[::-1]
    best = names[int(order[0])]
    confidence = float(probabilities[order[0]])
    if len(order) > 1:
        margin = float(probabilities[order[0]] - probabilities[order[1]])
        confidence = float(np.clip(confidence * (0.4 + 0.6 * min(margin / 0.3, 1.0)),
                                   0.0, 0.99))
    scores = {names[i]: float(probabilities[i]) for i in order}
    scores["stellarity"] = float(point_like)
    return ObjectClass(best), confidence, scores
