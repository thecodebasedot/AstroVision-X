"""One spectrum, start to finish: frame in, physical statements out.

The order is forced by what each step needs from the one before, and every
step can stop the chain. That is the design: a wavelength solution that did
not converge must not be followed by a redshift, and a redshift that is not
reliable must not be followed by line ratios quoted at that redshift, because
each would be a number with no meaning attached to it.

What comes out is a record of *statements with their evidence*, not a row of
numbers. A redshift carries the correlation quality it was measured at; a
classification carries the lines it used and whether they were detected or
limits; a supernova type carries the reminder that a spectral match is a
candidate classification and not a confirmed object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.logging import get_logger
from .calibrate import (WavelengthSolution, apply_solution,
                        check_against_sky_lines, fit_wavelength_solution)
from .diagnostics import balmer_decrement, classify_bpt, classify_supernova
from .extract import extract_spectrum
from .lines import fit_lines, measure_velocity_dispersion
from .redshift import measure_redshift
from .templates import LINES, Spectrum1D, standard_templates

log = get_logger("spectra.analyse")


@dataclass
class SpectrumAnalysis:
    """Everything measured from one spectrum, and what stopped where."""

    spectrum: Optional[Spectrum1D] = None
    solution: Optional[WavelengthSolution] = None
    redshift: Optional[Any] = None
    lines: Dict[str, Any] = field(default_factory=dict)
    bpt: Optional[Any] = None
    reddening: Dict[str, Any] = field(default_factory=dict)
    dispersion: Dict[str, Any] = field(default_factory=dict)
    supernova: Optional[Any] = None
    sky_check: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    stopped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectrum": self.spectrum.to_dict() if self.spectrum else {},
            "wavelength_solution": self.solution.to_dict() if self.solution else {},
            "redshift": self.redshift.to_dict() if self.redshift else {},
            "lines": {name: line.to_dict() for name, line in self.lines.items()},
            "bpt": self.bpt.to_dict() if self.bpt else {},
            "reddening": dict(self.reddening),
            "velocity_dispersion": dict(self.dispersion),
            "supernova": self.supernova.to_dict() if self.supernova else {},
            "sky_check": dict(self.sky_check),
            "stopped_at": self.stopped_at,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """A few lines a person can read."""
        parts: List[str] = []
        if self.redshift is not None and np.isfinite(self.redshift.z):
            state = "measured" if self.redshift.reliable else "not reliable"
            parts.append(f"z = {self.redshift.z:.4f} ({state}, "
                         f"{self.redshift.method}, R = {self.redshift.r_statistic:.1f})")
        if self.bpt is not None and self.bpt.classification != "unclassified":
            secure = "" if self.bpt.confident else ", not secure"
            parts.append(f"ionisation: {self.bpt.classification}{secure}")
        if self.supernova is not None and self.supernova.sn_type:
            parts.append(f"supernova candidate: Type {self.supernova.sn_type} "
                         f"at {self.supernova.phase_days:+.0f} days")
        if self.stopped_at:
            parts.append(f"stopped at: {self.stopped_at}")
        return "; ".join(parts) if parts else "nothing measurable"


def analyse_spectrum(spectrum: Spectrum1D,
                     templates: Optional[Sequence[Spectrum1D]] = None,
                     resolution: float = 5.0,
                     classify_transient: bool = False,
                     redshift: Optional[float] = None,
                     lines: Sequence[str] = tuple(LINES)) -> SpectrumAnalysis:
    """Redshift a calibrated spectrum, then measure what the redshift allows.

    No line measurement runs unless the redshift is reliable. A line fitted at
    the wrong redshift is fitted at the wrong wavelength, and it still returns
    a flux -- so the guard is not politeness, it is the difference between a
    measurement and a number.

    ``redshift`` supplies a known one, normally the host galaxy's. Supernova
    typing is deliberately *not* gated on the galaxy cross-correlation: a
    supernova spectrum is not a galaxy, so the correlation against galaxy
    templates failing is the expected outcome, not a reason to refuse. With no
    host redshift the typing searches for its own and says so.
    """
    analysis = SpectrumAnalysis(spectrum=spectrum)
    result = measure_redshift(spectrum, templates=templates)
    analysis.redshift = result
    analysis.notes.append(result.reason)

    known = float(redshift) if redshift is not None and np.isfinite(redshift) else None
    if classify_transient:
        if known is not None:
            analysis.supernova = classify_supernova(spectrum, redshift=known)
        elif result.reliable and not result.is_star:
            analysis.supernova = classify_supernova(spectrum, redshift=result.z)
        else:
            analysis.supernova = classify_supernova(spectrum, redshift=0.0,
                                                    search_redshift=True)
            analysis.notes.append(
                "No host redshift was available, so the supernova match searched "
                "redshift as well as type and phase; the two trade off against "
                "each other and the type is correspondingly less secure.")
        analysis.notes.append(analysis.supernova.reason)
        if analysis.supernova.caveat:
            analysis.notes.append(analysis.supernova.caveat)

    if result.is_star:
        analysis.stopped_at = "identified as a star"
        analysis.notes.append("A star has no redshift and no emission-line "
                              "diagnostics; nothing further applies.")
        return analysis

    if known is None and not result.reliable:
        analysis.stopped_at = "redshift not reliable"
        analysis.notes.append("Line measurements need a redshift to fit at, and "
                              "an unreliable one would put every line at the "
                              "wrong wavelength; they are not attempted.")
        return analysis
    if known is not None:
        analysis.notes.append(f"line measurements use the supplied redshift "
                              f"{known:.4f}")

    working = known if known is not None else result.z
    analysis.lines = fit_lines(spectrum, working, names=lines,
                               resolution=resolution)
    analysis.bpt = classify_bpt(analysis.lines)
    analysis.notes.append(analysis.bpt.reason)
    analysis.reddening = balmer_decrement(analysis.lines)

    library = list(templates) if templates is not None else standard_templates()
    early = next((t for t in library
                  if str(t.meta.get("name", "")) == "early_type"), None)
    if early is not None:
        analysis.dispersion = measure_velocity_dispersion(
            spectrum, working, early, resolution=resolution)
    return analysis


def analyse_frame(image: np.ndarray, variance: np.ndarray,
                  arc: Optional[np.ndarray] = None,
                  line_list: Optional[Sequence[float]] = None,
                  sky_lines: Optional[Sequence[float]] = None,
                  resolution: float = 5.0,
                  order: int = 3,
                  templates: Optional[Sequence[Spectrum1D]] = None,
                  classify_transient: bool = False) -> SpectrumAnalysis:
    """Extract, calibrate and analyse a long-slit frame.

    Without an arc frame this stops after extraction and says so: a spectrum
    in detector columns has no wavelengths, and everything past this point is
    a statement about wavelengths.
    """
    spectrum, trace = extract_spectrum(image, variance)
    analysis = SpectrumAnalysis(spectrum=spectrum)
    analysis.notes.append(
        f"traced over {len(spectrum)} columns with {trace.scatter:.2f} px scatter")

    if arc is None or line_list is None:
        analysis.stopped_at = "no wavelength calibration"
        analysis.notes.append("Without an arc exposure and a line list the axis "
                              "is detector columns, not wavelength.")
        return analysis

    arc_flux = np.median(np.asarray(arc, dtype=float), axis=0)
    solution = fit_wavelength_solution(arc_flux, line_list, order=order,
                                       resolution=resolution)
    analysis.solution = solution
    analysis.notes.append(solution.reason)
    if not solution.succeeded:
        analysis.stopped_at = "wavelength solution failed"
        return analysis

    calibrated = apply_solution(spectrum, solution)
    analysis.spectrum = calibrated
    if sky_lines:
        analysis.sky_check = check_against_sky_lines(calibrated, sky_lines)
        if analysis.sky_check.get("reliable"):
            analysis.notes.append(
                f"sky lines put the zero point {analysis.sky_check['offset']:+.2f} A "
                "from the arc solution")

    downstream = analyse_spectrum(calibrated, templates=templates,
                                   resolution=resolution,
                                   classify_transient=classify_transient)
    downstream.spectrum = calibrated
    downstream.solution = solution
    downstream.sky_check = analysis.sky_check
    downstream.notes = analysis.notes + downstream.notes
    return downstream
