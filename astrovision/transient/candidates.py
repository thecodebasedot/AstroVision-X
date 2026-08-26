"""Candidate extraction and host association from a difference image."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.config import TransientConfig
from ..core.logging import get_logger
from ..core.numeric import as_float_image, maximum_filter, nan_to_finite
from ..core.types import (
    LightCurve,
    SourceCatalog,
    TransientCandidate,
    Verdict,
)
from ..detect.labeling import label, remove_small
from ..photometry.magnitudes import flux_to_magnitude
from .difference import DifferenceResult
from .realbogus import classify_artifact, real_bogus_score, stamp_features

log = get_logger("transient.candidates")


def extract_candidates(result: DifferenceResult, config: Optional[TransientConfig] = None,
                       catalog: Optional[SourceCatalog] = None,
                       epoch_index: int = 0) -> List[TransientCandidate]:
    """Find, vet and describe every significant residual in a difference image.

    Only *positive* residuals are searched: a new or brightened source adds
    flux.  Negative residuals are recorded as a diagnostic, because an
    excess of them means the subtraction is over-subtracting rather than
    that objects are disappearing.
    """
    cfg = config or TransientConfig()
    difference = nan_to_finite(as_float_image(result.difference), 0.0)
    significance = difference / np.clip(result.noise, 1e-9, None)

    above = significance > cfg.detection_sigma
    # A pixel flagged bad in *either* epoch cannot yield a trustworthy
    # difference, so drop it before anything is labelled.
    for image in (result.science, result.reference):
        if image.mask is not None and image.mask.shape == above.shape:
            above &= ~image.mask
    margin = 6
    above[:margin, :] = above[-margin:, :] = False
    above[:, :margin] = above[:, -margin:] = False

    segmentation, count = label(above)
    if count == 0:
        log.info("no residuals above %.1f sigma in the difference image",
                 cfg.detection_sigma)
        return []
    segmentation, count = remove_small(segmentation, cfg.min_area, count)
    if count == 0:
        return []

    psf = result.science_psf
    psf_fwhm = float(psf.fwhm) if psf is not None else 3.0
    stamp_size = max(11, int(2 * np.ceil(3 * psf_fwhm) + 1))
    ny, nx = difference.shape
    wcs = result.science.wcs
    zero_point = float(result.science.header.get("MAGZP", 25.0) or 25.0)

    candidates: List[TransientCandidate] = []
    peaks = difference >= maximum_filter(difference, 3)

    for value in range(1, count + 1):
        footprint = segmentation == value
        if not footprint.any():
            continue
        ys, xs = np.nonzero(footprint & peaks)
        if ys.size == 0:
            ys, xs = np.nonzero(footprint)
        best = int(np.argmax(difference[ys, xs]))
        py, px = int(ys[best]), int(xs[best])

        # Flux-weighted centroid within the footprint gives a sub-pixel position.
        weights = np.clip(difference * footprint, 0, None)
        total = float(weights.sum())
        if total <= 0:
            continue
        yy, xx = np.mgrid[0:ny, 0:nx]
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)

        half = stamp_size // 2
        y0, x0 = max(0, py - half), max(0, px - half)
        stamp = difference[y0:y0 + stamp_size, x0:x0 + stamp_size]
        local_noise = float(np.median(result.noise[footprint]))

        features = stamp_features(stamp, local_noise, psf)
        score, terms = real_bogus_score(features, psf_fwhm, cfg.dipole_threshold)

        magnitude, _ = flux_to_magnitude(total, zero_point)
        candidate = TransientCandidate(
            id=len(candidates) + 1, x=cx, y=cy,
            significance=float(difference[py, px] / max(local_noise, 1e-9)),
            delta_flux=total,
            delta_magnitude=float(magnitude),
            real_bogus=float(score),
            epoch_index=int(epoch_index),
            meta={"features": {k: float(v) for k, v in features.items()},
                  "vetting_terms": {k: float(v) for k, v in terms.items()},
                  "area_pixels": int(footprint.sum()),
                  "local_noise": local_noise,
                  "peak": float(difference[py, px])},
        )
        if wcs is not None:
            ra, dec = wcs.pixel_to_world(cx, cy)
            candidate.ra, candidate.dec = float(ra), float(dec)

        if score < cfg.real_bogus_threshold:
            candidate.classification = classify_artifact(features, terms)
            candidate.add_flag("bogus")
            candidate.verdict = Verdict.NOT_INTERESTING
        if cfg.reject_dipoles and features.get("dipole_ratio", 0.0) > cfg.dipole_threshold:
            candidate.add_flag("dipole")

        candidates.append(candidate)

    if catalog is not None:
        associate_hosts(candidates, catalog, cfg.host_search_radius)

    negative = int((significance < -cfg.detection_sigma).sum())
    real = [c for c in candidates if "bogus" not in c.flags]
    log.info("difference image: %d residuals, %d pass vetting (%d negative pixels)",
             len(candidates), len(real), negative)

    candidates.sort(key=lambda c: (-c.real_bogus, -c.significance))
    for index, candidate in enumerate(candidates[:cfg.max_candidates], start=1):
        candidate.id = index
    return candidates[:cfg.max_candidates]


def associate_hosts(candidates: Sequence[TransientCandidate], catalog: SourceCatalog,
                    radius: float = 25.0) -> None:
    """Attach the most plausible host galaxy to each candidate.

    A supernova sits in or beside a galaxy, so the host is strong evidence
    that a residual is astrophysical, and the offset from the host nucleus
    is one of the first things a follow-up proposal must quote.
    """
    galaxies = [s for s in catalog if s.is_extended or s.morphology.area_pixels > 25]
    if not galaxies:
        galaxies = list(catalog)
    if not galaxies:
        return
    positions = np.array([[s.x, s.y] for s in galaxies], dtype=float)

    for candidate in candidates:
        distance = np.hypot(positions[:, 0] - candidate.x, positions[:, 1] - candidate.y)
        # Prefer a nearby *large* galaxy over a slightly closer point source:
        # normalising by each object's size is what picks the real host.
        sizes = np.array([max(s.morphology.semi_major, 1.0) for s in galaxies])
        normalised = distance / np.maximum(sizes, 1.0)
        order = np.argsort(normalised)
        best = int(order[0])
        if distance[best] > radius:
            candidate.add_flag("hostless")
            continue
        host = galaxies[best]
        candidate.host_source_id = host.id
        candidate.host_offset = float(distance[best])
        candidate.meta["host_offset_over_size"] = float(normalised[best])
        candidate.meta["host_class"] = host.object_class.value
        if distance[best] < 1.5:
            # On the nucleus: could be an active galactic nucleus rather
            # than a supernova, and that distinction matters for follow-up.
            candidate.add_flag("nuclear")


def build_candidate_light_curves(candidates: Sequence[TransientCandidate], series,
                                 radius: float = 4.0) -> Dict[int, LightCurve]:
    """Measure each candidate's position in every epoch of the series."""
    from ..timeseries.lightcurve import extract_light_curve

    curves: Dict[int, LightCurve] = {}
    for candidate in candidates:
        curve = extract_light_curve(series, candidate.x, candidate.y, radius,
                                    source_id=candidate.id)
        candidate.light_curve = curve
        curves[candidate.id] = curve
    return curves


def merge_epoch_candidates(per_epoch: Sequence[Sequence[TransientCandidate]],
                           radius: float = 3.0) -> List[TransientCandidate]:
    """Merge candidates found in several epochs into unique sky positions.

    A transient visible in three epochs should be reported once, with the
    epochs recorded -- and being seen more than once is itself powerful
    evidence that it is real rather than an artefact.
    """
    merged: List[TransientCandidate] = []
    for epoch, candidates in enumerate(per_epoch):
        for candidate in candidates:
            match = None
            for existing in merged:
                if np.hypot(existing.x - candidate.x, existing.y - candidate.y) <= radius:
                    match = existing
                    break
            if match is None:
                candidate.meta.setdefault("epochs_detected", [])
                candidate.meta["epochs_detected"].append(int(candidate.epoch_index))
                merged.append(candidate)
                continue
            match.meta.setdefault("epochs_detected", [])
            match.meta["epochs_detected"].append(int(candidate.epoch_index))
            # Keep the epoch where the candidate looked most convincing.
            if candidate.real_bogus > match.real_bogus:
                match.real_bogus = candidate.real_bogus
                match.significance = max(match.significance, candidate.significance)
                match.epoch_index = candidate.epoch_index
                match.meta["features"] = candidate.meta.get("features", {})

    for candidate in merged:
        epochs = sorted(set(candidate.meta.get("epochs_detected", [])))
        candidate.meta["epochs_detected"] = epochs
        candidate.meta["n_detections"] = len(epochs)
        if len(epochs) >= 2:
            candidate.add_flag("multi_epoch")
    merged.sort(key=lambda c: (-len(c.meta.get("epochs_detected", [])), -c.real_bogus))
    for index, candidate in enumerate(merged, start=1):
        candidate.id = index
    return merged
