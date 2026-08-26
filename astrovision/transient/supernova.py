"""Assessing what a vetted transient candidate might be.

A residual that survives vetting is real; what it *is* depends on where it
sits and how it evolves.  A supernova appears offset from a galaxy nucleus,
rises over days and declines over weeks.  A variable star repeats.  A
moving object is in a different place each epoch.  This module scores those
hypotheses -- and never asserts a discovery, which requires spectroscopy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import logistic
from ..core.types import LightCurve, SourceCatalog, TransientCandidate, Verdict
from ..timeseries.features import variability_features
from ..timeseries.periodogram import find_period

log = get_logger("transient.supernova")

#: The hypotheses the classifier weighs.
TRANSIENT_CLASSES = [
    "supernova_candidate", "variable_star", "agn_variability",
    "moving_object", "unknown_transient",
]


def light_curve_shape(curve: Optional[LightCurve]) -> Dict[str, float]:
    """Rise and decline behaviour, the discriminating shape statistics."""
    empty = {"rise_rate": float("nan"), "decline_rate": float("nan"),
             "peak_time": float("nan"), "peak_fraction": float("nan"),
             "asymmetry": float("nan"), "n_epochs": 0.0}
    if curve is None:
        return empty
    clean = curve.clean()
    if len(clean) < 3:
        return empty

    fluxes = clean.fluxes
    times = clean.times
    peak = int(np.argmax(fluxes))
    peak_flux = float(fluxes[peak])
    baseline = float(np.median(fluxes))
    if peak_flux <= 0:
        return empty

    rise = decline = float("nan")
    if peak > 0:
        dt = times[peak] - times[0]
        if dt > 0:
            rise = float((fluxes[peak] - fluxes[0]) / dt)
    if peak < len(fluxes) - 1:
        dt = times[-1] - times[peak]
        if dt > 0:
            decline = float((fluxes[peak] - fluxes[-1]) / dt)

    # Supernovae rise faster than they fade; the ratio captures that.
    asymmetry = float("nan")
    if np.isfinite(rise) and np.isfinite(decline) and abs(decline) > 1e-9:
        asymmetry = float(rise / abs(decline))

    return {
        "rise_rate": rise, "decline_rate": decline,
        "peak_time": float(times[peak]),
        "peak_fraction": float((peak_flux - baseline) / max(abs(peak_flux), 1e-9)),
        "asymmetry": asymmetry,
        "n_epochs": float(len(clean)),
    }


def classify_transient(candidate: TransientCandidate,
                       catalog: Optional[SourceCatalog] = None
                       ) -> Tuple[str, float, Dict[str, float]]:
    """Weigh the hypotheses for one candidate.

    Returns ``(classification, confidence, scores)``.
    """
    scores = {name: 0.0 for name in TRANSIENT_CLASSES}
    shape = light_curve_shape(candidate.light_curve)
    candidate.meta["light_curve_shape"] = shape

    host_id = candidate.host_source_id
    host = catalog.by_id(host_id) if (catalog is not None and host_id is not None) else None
    offset = candidate.host_offset
    nuclear = "nuclear" in candidate.flags
    hostless = "hostless" in candidate.flags

    # -- supernova: offset from an extended host, single smooth outburst --
    supernova = 0.0
    if host is not None and np.isfinite(offset):
        if host.is_extended or host.morphology.area_pixels > 25:
            supernova += 0.45
        size = max(host.morphology.semi_major, 1.0)
        # Offsets of a fraction to a few host radii are the classic regime.
        ratio = offset / size
        supernova += 0.35 * float(np.exp(-0.5 * ((ratio - 1.2) / 1.4) ** 2))
        if nuclear:
            supernova -= 0.25
    if np.isfinite(shape["asymmetry"]) and shape["asymmetry"] > 1.2:
        supernova += 0.25
    if np.isfinite(shape["n_epochs"]) and shape["n_epochs"] >= 4:
        supernova += 0.1
    scores["supernova_candidate"] = max(supernova, 0.0)

    # -- variable star: coincides with an existing point source, repeats ---
    variable = 0.0
    if host is not None and not host.is_extended and np.isfinite(offset) and offset < 2.0:
        variable += 0.6
    if candidate.light_curve is not None and len(candidate.light_curve) >= 6:
        period = find_period(candidate.light_curve, 0.02, 100.0)
        candidate.meta["period"] = period
        if (np.isfinite(period.get("false_alarm_probability", np.nan)) and
                period["false_alarm_probability"] < 0.01):
            variable += 0.5
            scores["supernova_candidate"] *= 0.4     # periodicity rules out a SN
    features = (variability_features(candidate.light_curve)
                if candidate.light_curve is not None else {})
    if features.get("von_neumann_eta", np.nan) < 0.8:
        variable += 0.1
    scores["variable_star"] = max(variable, 0.0)

    # -- AGN: on the nucleus of a galaxy, stochastic ------------------------
    agn = 0.0
    if nuclear and host is not None and host.is_extended:
        agn += 0.75
    if features and not np.isfinite(features.get("skewness", np.nan)):
        agn += 0.05
    elif features.get("skewness", 0.0) is not None and abs(features.get("skewness", 0.0)) < 0.6:
        agn += 0.15
    scores["agn_variability"] = max(agn, 0.0)

    # -- moving object: appears in one epoch only, no host -----------------
    mover = 0.0
    n_detections = int(candidate.meta.get("n_detections", 1))
    if n_detections <= 1 and hostless:
        mover += 0.5
    if candidate.meta.get("features", {}).get("elongation", 1.0) > 1.8:
        mover += 0.3      # trailed during the exposure
    if "multi_epoch" in candidate.flags:
        mover -= 0.3
    scores["moving_object"] = max(mover, 0.0)

    scores["unknown_transient"] = 0.25

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = float(scores[best] / total) if total > 0 else 0.0
    # A candidate seen in several epochs is more trustworthy whatever it is.
    if "multi_epoch" in candidate.flags:
        confidence = float(min(confidence * 1.15, 0.99))
    return best, confidence, scores


def assign_verdict(candidate: TransientCandidate, threshold: float = 0.5) -> Verdict:
    """How much follow-up this candidate deserves.

    The scale deliberately tops out at "high priority": the platform ranks
    candidates for astronomers and never declares a discovery, which needs
    independent imaging and, for a supernova, a spectrum.
    """
    if candidate.real_bogus < threshold or "bogus" in candidate.flags:
        return Verdict.NOT_INTERESTING
    score = 0.0
    score += 0.35 * float(np.clip(candidate.real_bogus, 0, 1))
    score += 0.25 * float(logistic(candidate.significance, scale=2.0, midpoint=7.0))
    score += 0.20 * float(np.clip(candidate.confidence, 0, 1))
    if "multi_epoch" in candidate.flags:
        score += 0.15
    if candidate.host_source_id is not None and not np.isnan(candidate.host_offset):
        score += 0.10
    if "dipole" in candidate.flags:
        score -= 0.25
    if "nuclear" in candidate.flags:
        score -= 0.05

    if score >= 0.75:
        return Verdict.HIGH_PRIORITY
    if score >= 0.55:
        return Verdict.FOLLOW_UP_RECOMMENDED
    if score >= 0.35:
        return Verdict.WORTH_A_LOOK
    return Verdict.NOT_INTERESTING


def describe(candidate: TransientCandidate, catalog: Optional[SourceCatalog] = None) -> str:
    """A short human-readable summary of one candidate."""
    parts = [f"Candidate #{candidate.id} at ({candidate.x:.1f}, {candidate.y:.1f})"]
    if candidate.ra is not None:
        parts[0] += f" = ({candidate.ra:.5f}, {candidate.dec:+.5f}) deg"
    parts.append(f"{candidate.significance:.1f} sigma residual")
    parts.append(f"real/bogus {candidate.real_bogus:.2f}")
    if candidate.host_source_id is not None:
        host = catalog.by_id(candidate.host_source_id) if catalog is not None else None
        host_type = host.object_class.value if host is not None else "source"
        parts.append(f"{candidate.host_offset:.1f} px from {host_type} "
                     f"#{candidate.host_source_id}")
    elif "hostless" in candidate.flags:
        parts.append("no host within the search radius")
    epochs = candidate.meta.get("n_detections")
    if epochs:
        parts.append(f"detected in {epochs} epoch(s)")
    parts.append(f"best match: {candidate.classification.replace('_', ' ')} "
                 f"(confidence {candidate.confidence:.2f})")
    parts.append(f"verdict: {candidate.verdict.value.replace('_', ' ')}")
    return "; ".join(parts) + "."
