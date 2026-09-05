"""What the lines mean: ionisation diagnostics and supernova typing.

Two classifications live here, and they fail in opposite ways.

**The BPT diagram** (Baldwin, Phillips & Terlevich 1981) separates gas ionised
by young stars from gas ionised by an accreting black hole, using two ratios of
*adjacent* lines -- [O III]/H-beta and [N II]/H-alpha. Adjacency is the whole
trick: the pairs are close enough in wavelength that reddening and flux
calibration cancel, so the diagram works on data that is not calibrated at all.
Its failure mode is over-confidence. There is a real region between the
star-forming locus and the AGN region where both contribute, and reporting a
composite as one or the other is not a rounding error, it is a different
physical claim.

**Supernova typing** matches a spectrum against templates of each type across a
range of phases. Its failure mode is the opposite: the classification is
genuinely uncertain at low signal-to-noise or far from maximum light, and a
type reported without that caveat is a claim nobody checked. So a type comes
with a quality statistic and a margin over the runner-up, and below either
threshold the answer is "no confident type" rather than the best guess.

Nothing here declares a discovery. A supernova type from one spectrum is a
classification of that spectrum; whether the object is a supernova at all is
settled by a light curve, a host galaxy, and a human being.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from .lines import LineMeasurement, line_ratio
from .templates import BPT_LINES, Spectrum1D, supernova_templates

log = get_logger("spectra.diagnostics")


def kauffmann_line(log_n2_ha: np.ndarray) -> np.ndarray:
    """The empirical star-forming boundary (Kauffmann et al. 2003).

    Below this line, essentially every galaxy in a large survey is
    star-forming. It is drawn from where the galaxies actually are, not from
    a model, which is why it sits below the theoretical limit.
    """
    x = np.asarray(log_n2_ha, dtype=float)
    # Past the asymptote the curve does not exist, and the convention is that
    # everything there is on the AGN side -- no star-forming galaxy reaches
    # that much [N II] relative to H-alpha. Returning +inf instead would put
    # every one of them *below* the line and call the hardest-ionised objects
    # in the sample star-forming.
    return np.where(x < 0.05, 0.61 / (x - 0.05) + 1.30, -np.inf)


def kewley_line(log_n2_ha: np.ndarray) -> np.ndarray:
    """The theoretical maximum-starburst boundary (Kewley et al. 2001).

    Above this line no combination of stellar population, metallicity and
    ionisation parameter can produce the observed ratios: something harder
    than stars is doing the ionising.
    """
    x = np.asarray(log_n2_ha, dtype=float)
    return np.where(x < 0.47, 0.61 / (x - 0.47) + 1.19, -np.inf)


def schawinski_line(log_n2_ha: np.ndarray) -> np.ndarray:
    """The Seyfert/LINER division (Schawinski et al. 2007)."""
    return 1.05 * np.asarray(log_n2_ha, dtype=float) + 0.45


@dataclass
class BPTClassification:
    """Where a spectrum sits on the diagnostic diagram, and how surely."""

    log_o3_hb: float = float("nan")
    log_n2_ha: float = float("nan")
    log_o3_hb_error: float = float("nan")
    log_n2_ha_error: float = float("nan")
    classification: str = "unclassified"
    confident: bool = False
    reason: str = ""
    missing: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"log_o3_hb": self.log_o3_hb, "log_n2_ha": self.log_n2_ha,
                "classification": self.classification,
                "confident": self.confident, "reason": self.reason,
                "missing": list(self.missing), "limits": list(self.limits)}


def classify_bpt(lines: Dict[str, LineMeasurement]) -> BPTClassification:
    """Place a spectrum on the [N II] BPT diagram.

    A classification is only returned when all four lines are detected. With
    a limit on one of them the position is a limit too, and the region is only
    named when the limit alone settles it -- which it sometimes does, and
    saying so is more useful than refusing.
    """
    result = BPTClassification()
    missing = [name for name in BPT_LINES if name not in lines]
    if missing:
        result.missing = missing
        result.reason = ("the diagram needs " + ", ".join(BPT_LINES)
                         + "; missing " + ", ".join(missing))
        return result

    o3_hb, o3_error, o3_status = line_ratio(lines, "[O III] 5007", "H beta")
    n2_ha, n2_error, n2_status = line_ratio(lines, "[N II] 6584", "H alpha")
    for label, status in (("[O III]/H beta", o3_status), (("[N II]/H alpha"), n2_status)):
        if status != "measured":
            result.limits.append(f"{label} is an {status.replace('_', ' ')}")

    if not (np.isfinite(o3_hb) and np.isfinite(n2_ha) and o3_hb > 0 and n2_ha > 0):
        undetected = [name for name in BPT_LINES if not lines[name].detected]
        result.reason = ("no usable ratio: " + ", ".join(undetected)
                         + " not detected") if undetected else "ratios are not positive"
        result.missing = undetected
        return result

    result.log_o3_hb = float(math.log10(o3_hb))
    result.log_n2_ha = float(math.log10(n2_ha))
    if np.isfinite(o3_error) and o3_hb > 0:
        result.log_o3_hb_error = float(o3_error / (o3_hb * math.log(10)))
    if np.isfinite(n2_error) and n2_ha > 0:
        result.log_n2_ha_error = float(n2_error / (n2_ha * math.log(10)))

    x, y = result.log_n2_ha, result.log_o3_hb
    below_kauffmann = y < float(kauffmann_line(np.array([x]))[0])
    below_kewley = y < float(kewley_line(np.array([x]))[0])
    if below_kauffmann:
        result.classification = "star-forming"
        result.reason = "below the Kauffmann star-forming boundary"
    elif below_kewley:
        result.classification = "composite"
        result.reason = ("between the empirical star-forming boundary and the "
                         "maximum-starburst line: star formation and an active "
                         "nucleus both contribute, and the data do not separate them")
    elif y > float(schawinski_line(np.array([x]))[0]):
        result.classification = "Seyfert"
        result.reason = "above the maximum-starburst line and the Seyfert/LINER division"
    else:
        result.classification = "LINER"
        result.reason = ("above the maximum-starburst line but below the "
                         "Seyfert division; LINER-like, which can also be "
                         "produced by old stars rather than an active nucleus")

    # Confidence: the errors have to be small enough that the point does not
    # straddle a boundary.  A point 0.02 dex from the Kauffmann line with 0.3
    # dex errors has not been classified, however definite the label looks.
    spread = math.hypot(result.log_o3_hb_error if np.isfinite(result.log_o3_hb_error) else 0.0,
                        result.log_n2_ha_error if np.isfinite(result.log_n2_ha_error) else 0.0)
    distance = min(abs(y - float(kauffmann_line(np.array([x]))[0])),
                   abs(y - float(kewley_line(np.array([x]))[0])))
    result.confident = bool(not result.limits and np.isfinite(distance)
                            and distance > max(2.0 * spread, 0.05))
    if not result.confident and not result.limits:
        result.reason += (f"; the point sits {distance:.2f} dex from a boundary "
                          f"with {spread:.2f} dex of error, so the label is not "
                          "secure")
    return result


@dataclass
class SupernovaMatch:
    """The result of matching a spectrum against supernova templates."""

    sn_type: str = ""
    phase_days: float = float("nan")
    redshift: float = float("nan")
    quality: float = float("nan")          # correlation R of the best match
    margin: float = float("nan")           # R of best minus best of other types
    runner_up: str = ""
    confident: bool = False
    ranking: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""
    caveat: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.sn_type, "phase_days": self.phase_days,
                "redshift": self.redshift, "quality": self.quality,
                "margin": self.margin, "runner_up": self.runner_up,
                "confident": self.confident, "ranking": list(self.ranking),
                "reason": self.reason, "caveat": self.caveat}


#: Minimum correlation quality for a supernova type to be reported at all.
MIN_SN_QUALITY = 5.0

#: The best type must beat the best *other* type by this much in R. Without
#: a margin the classifier reports whichever of four similar matches won by a
#: hair, which is a coin toss wearing a label.
MIN_SN_MARGIN = 2.0


def classify_supernova(spectrum: Spectrum1D, redshift: float = 0.0,
                       templates: Optional[Sequence[Spectrum1D]] = None,
                       search_redshift: bool = False) -> SupernovaMatch:
    """Match a spectrum against supernova templates of each type and phase.

    Both axes are searched because both matter: a Type Ia a month after
    maximum looks nothing like one a week before it, and matching against the
    wrong phase is matching against lines at the wrong wavelength.

    ``redshift`` is normally known from the host galaxy. With
    ``search_redshift`` the correlation itself provides it, which is what a
    real classifier does for a hostless transient -- and is measurably worse,
    because type and redshift trade off against each other.
    """
    from .redshift import cross_correlate, log_grid, prepare, tonry_davis_r

    library = list(templates) if templates is not None else supernova_templates()
    match = SupernovaMatch(redshift=float(redshift))
    ok = spectrum.good
    if ok.sum() < 100:
        match.reason = "too few usable pixels to classify"
        return match

    low = float(np.nanmin(spectrum.wavelength[ok]))
    high = float(np.nanmax(spectrum.wavelength[ok]))
    grid = log_grid(low / (1.0 + max(float(redshift), 0.0) + 0.1), high, 100.0)
    observed = prepare(spectrum, grid, continuum_window=301)
    step = math.log(grid[1] / grid[0])

    scores: List[Tuple[float, str, float, float]] = []
    for template in library:
        rest = template.redshifted(float(redshift)) if redshift else template
        prepared = prepare(rest, grid, continuum_window=301)
        correlation, lags = cross_correlate(observed, prepared)
        shifts = np.expm1(lags * step)
        if search_redshift:
            window = (shifts > -0.02) & (shifts < 0.35)
        else:
            # Without a redshift search the template is already at the host's
            # redshift, so only a small residual shift is allowed -- enough
            # for the supernova's own expansion velocity, not enough to let
            # the fit slide onto a different feature.
            window = np.abs(shifts) < 0.02
        if not window.any():
            continue
        candidates = np.flatnonzero(window)
        index = int(candidates[int(np.argmax(correlation[candidates]))])
        quality = tonry_davis_r(correlation, index)
        if not np.isfinite(quality):
            continue
        name = str(template.meta.get("name", "?"))
        sn_type = str(template.meta.get("type", "?"))
        phase = float(template.meta.get("phase_days", float("nan")))
        extra = float(np.expm1((index - (len(observed) - 1)) * step))
        scores.append((quality, sn_type, phase, extra))

    if not scores:
        match.reason = "no template produced a usable correlation"
        return match

    scores.sort(key=lambda item: -item[0])
    quality, sn_type, phase, extra = scores[0]
    by_type: Dict[str, float] = {}
    for value, name, _, _ in scores:
        by_type[name] = max(by_type.get(name, -np.inf), value)
    ranking = sorted(by_type.items(), key=lambda item: -item[1])

    match.sn_type = sn_type
    match.phase_days = phase
    match.quality = quality
    match.ranking = [(name, float(value)) for name, value in ranking]
    match.redshift = float(redshift) + (extra if search_redshift else 0.0)
    others = [value for name, value in ranking if name != sn_type]
    match.margin = float(quality - max(others)) if others else float("inf")
    match.runner_up = ranking[1][0] if len(ranking) > 1 else ""

    if quality < MIN_SN_QUALITY:
        match.reason = (f"the best match, {sn_type}, reaches only R = {quality:.1f}; "
                        f"below {MIN_SN_QUALITY:.0f} the correlation is not "
                        "distinguishable from noise and no type is claimed")
        match.sn_type = ""
        return match
    if match.margin < MIN_SN_MARGIN:
        match.reason = (f"{sn_type} leads {match.runner_up} by only "
                        f"{match.margin:.1f} in R; the spectrum does not "
                        "separate them and no type is claimed")
        match.sn_type = ""
        return match

    match.confident = True
    match.reason = (f"best match Type {sn_type} at {phase:+.0f} days, R = "
                    f"{quality:.1f}, ahead of Type {match.runner_up} by "
                    f"{match.margin:.1f}")
    match.caveat = ("A spectral match is a classification of this spectrum, not "
                    "a confirmed supernova. Confirmation needs a light curve, a "
                    "host association and human review; the type here is a "
                    "candidate classification for follow-up.")
    return match


def balmer_decrement(lines: Dict[str, LineMeasurement],
                     intrinsic: float = 2.86) -> Dict[str, Any]:
    """Dust extinction from the H-alpha / H-beta ratio.

    Recombination fixes the intrinsic ratio at about 2.86 for case B at
    10,000 K, so anything above that is dust. The measurement is only as good
    as that assumption: in an AGN's broad-line region the intrinsic ratio is
    higher, and applying the star-forming value there overestimates the
    reddening.
    """
    ratio, error, status = line_ratio(lines, "H alpha", "H beta")
    if status != "measured" or not np.isfinite(ratio) or ratio <= 0:
        return {"ratio": ratio, "e_bv": float("nan"), "reliable": False,
                "status": status,
                "reason": "both Balmer lines must be detected to measure reddening"}
    if ratio < intrinsic:
        return {"ratio": float(ratio), "e_bv": 0.0, "reliable": False,
                "status": status,
                "reason": (f"the measured ratio {ratio:.2f} is below the intrinsic "
                           f"{intrinsic:.2f}, which dust cannot do; this is noise "
                           "or a bad H-beta measurement, not negative extinction")}
    # E(B-V) = 2.5 / (k_Hb - k_Ha) * log10(observed / intrinsic), with the
    # Calzetti curve giving k_Hb = 3.61, k_Ha = 2.53.
    e_bv = 2.5 / (3.61 - 2.53) * math.log10(ratio / intrinsic)
    return {"ratio": float(ratio), "ratio_error": float(error),
            "e_bv": float(e_bv), "reliable": True, "status": status,
            "reason": (f"Balmer decrement {ratio:.2f} against an intrinsic "
                       f"{intrinsic:.2f}, Calzetti curve")}
