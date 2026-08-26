"""The AstroAI research assistant.

Turns the numerical output of the pipeline into prose an astronomer can
read: what is in the field, what stands out, and what to do about it.

The assistant states candidates and evidence.  It never announces a
discovery.  A transient is a candidate until a second epoch confirms it and,
for a supernova, a spectrum classifies it; a lens is a candidate until
colours and redshifts support it; an outlier is unusual, which is not the
same as being new.  That boundary is enforced in the wording here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from ..core.logging import get_logger
from ..core.types import FieldAnalysis, Source, Verdict
from .priority import PriorityItem, rank_candidates

log = get_logger("engine.assistant")

#: The standing caveat attached to every set of findings.
DISCOVERY_DISCLAIMER = (
    "AstroVision-X reports candidates and the evidence behind them. Nothing "
    "here is a confirmed detection: transient candidates need an independent "
    "epoch and, for supernovae, spectroscopic classification; lens candidates "
    "need colour information and spectroscopic redshifts; anomalies are "
    "objects unlike the rest of this field, which is not the same as objects "
    "unlike anything known. Every candidate requires review by an astronomer."
)


class ResearchAssistant:
    """Writes the narrative layer of a scientific report.

    >>> from astrovision.core.types import FieldAnalysis
    >>> assistant = ResearchAssistant()
    >>> summary = assistant.summarise(FieldAnalysis())
    >>> "no sources" in summary.lower()
    True
    """

    def __init__(self, top_candidates: int = 10):
        self.top_candidates = int(top_candidates)

    # -- narrative sections -------------------------------------------------
    def summarise(self, analysis: FieldAnalysis) -> str:
        """One paragraph describing what the field contains."""
        catalog = analysis.catalog
        if len(catalog) == 0:
            return ("The analysis found no sources above the detection threshold. "
                    "Either the field is empty, the exposure is too shallow, or the "
                    "detection threshold is set too high for these data.")

        counts = catalog.class_counts()
        parts = [f"{len(catalog)} sources were detected"]
        statistics = analysis.statistics or {}
        limit = statistics.get("photometry", {}).get("limiting_magnitude_5sigma")
        if limit and np.isfinite(limit):
            parts[0] += f" to a 5-sigma limiting magnitude of {limit:.1f}"
        parts[0] += "."

        described = ", ".join(f"{n} {_plural(k, n)}" for k, n in counts.items())
        parts.append(f"The field breaks down as {described}.")

        morphologies = statistics.get("field", {}).get("morphology_counts")
        if morphologies:
            top = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in
                            list(morphologies.items())[:3])
            parts.append(f"Among the resolved galaxies the commonest "
                         f"morphologies are {top}.")

        clustering = statistics.get("field", {}).get("clustering", {})
        clark_evans = clustering.get("clark_evans")
        if clark_evans and np.isfinite(clark_evans):
            if clark_evans < 0.85:
                parts.append(f"Sources are more clustered than a random "
                             f"distribution (Clark-Evans {clark_evans:.2f}), which "
                             "suggests real structure in the field.")
            elif clark_evans > 1.15:
                parts.append(f"Sources are unusually evenly spread "
                             f"(Clark-Evans {clark_evans:.2f}); in a real image that "
                             "usually means blending has merged close pairs.")

        findings = []
        if analysis.transients:
            vetted = [t for t in analysis.transients if "bogus" not in t.flags]
            if vetted:
                findings.append(f"{len(vetted)} transient candidate(s)")
        if analysis.lenses:
            findings.append(f"{len(analysis.lenses)} possible gravitational lens(es)")
        strong_anomalies = [a for a in analysis.anomalies if a.score >= 0.95]
        if strong_anomalies:
            findings.append(f"{len(strong_anomalies)} strong outlier(s)")
        variables = [s for s in catalog if "variable" in s.flags]
        if variables:
            findings.append(f"{len(variables)} variable source(s)")

        if findings:
            parts.append("Flagged for attention: " + ", ".join(findings) + ".")
        else:
            parts.append("Nothing in this field met the thresholds for transient, "
                         "lensing or novelty follow-up.")
        return " ".join(parts)

    def describe_priority(self, item: PriorityItem,
                          analysis: FieldAnalysis) -> str:
        """A short paragraph explaining one ranked candidate."""
        source = (analysis.catalog.by_id(item.source_id)
                  if item.source_id is not None else None)
        header = f"#{item.rank}. {item.kind.replace('_', ' ').title()}"
        if item.candidate_id is not None:
            header += f" candidate {item.candidate_id}"
        elif item.source_id is not None:
            header += f" at source {item.source_id}"

        where = f"pixel ({item.position[0]:.1f}, {item.position[1]:.1f})"
        if item.sky_position is not None:
            where += (f", RA {item.sky_position[0]:.5f} deg, "
                      f"Dec {item.sky_position[1]:+.5f} deg")

        lines = [f"{header} -- {where}",
                 f"    Priority score {item.score:.2f}; "
                 f"verdict: {item.verdict.value.replace('_', ' ')}."]
        if item.reasons:
            lines.append("    Why: " + "; ".join(item.reasons) + ".")
        if source is not None:
            lines.append("    Host/source: " + self._describe_source(source))
        if item.caveats:
            lines.append("    Caveats: " + " ".join(item.caveats))
        return "\n".join(lines)

    def _describe_source(self, source: Source) -> str:
        """One line of measured properties for a source."""
        parts = [source.object_class.value.replace("_", " ")]
        if source.class_confidence:
            parts[0] += f" (confidence {source.class_confidence:.2f})"
        photometry = source.photometry
        if np.isfinite(photometry.magnitude):
            parts.append(f"magnitude {photometry.magnitude:.2f} "
                         f"+/- {photometry.magnitude_err:.2f}")
        if np.isfinite(photometry.snr):
            parts.append(f"S/N {photometry.snr:.0f}")
        morphology = source.morphology
        if morphology.label.value not in ("unknown", "unresolved"):
            descriptor = morphology.label.value.replace("_", " ")
            parts.append(f"morphology {descriptor} "
                         f"(confidence {morphology.label_confidence:.2f})")
        if np.isfinite(morphology.sersic_index):
            parts.append(f"Sersic n = {morphology.sersic_index:.2f}")
        physical = source.meta.get("physical", {})
        if np.isfinite(physical.get("physical_size_kpc", np.nan)):
            parts.append(f"physical size ~{physical['physical_size_kpc']:.1f} kpc "
                         "(assumed redshift)")
        return ", ".join(parts)

    def recommendations(self, analysis: FieldAnalysis,
                        ranked: Sequence[PriorityItem]) -> List[str]:
        """Concrete next steps, in the order they should be taken."""
        actions: List[str] = []
        urgent = [i for i in ranked
                  if i.verdict in (Verdict.HIGH_PRIORITY, Verdict.FOLLOW_UP_RECOMMENDED)]
        transients = [i for i in urgent if i.kind == "transient"]
        lenses = [i for i in urgent if i.kind == "lens"]
        anomalies = [i for i in ranked if i.kind == "anomaly" and i.score >= 0.65]

        if transients:
            actions.append(
                f"Re-image the {len(transients)} highest-priority transient "
                "position(s) within a few days: a second independent epoch is "
                "what separates a real transient from a subtraction artefact.")
            supernovae = [i for i in transients
                          if "supernova" in " ".join(i.reasons)]
            if supernovae:
                actions.append(
                    "For the supernova candidates, request a classification "
                    "spectrum while they are near maximum light -- the type "
                    "cannot be established from imaging alone.")
        if lenses:
            actions.append(
                f"Obtain multi-band imaging of the {len(lenses)} lens "
                "candidate(s): lensed sources are typically bluer than their "
                "deflector, and colour is the cheapest discriminator against "
                "chance alignments and ring galaxies.")
        if anomalies:
            actions.append(
                f"Visually inspect the top {min(len(anomalies), 5)} outlier(s) "
                "before anything else -- most high novelty scores in practice "
                "turn out to be instrumental, and eliminating those first is "
                "much cheaper than following them up.")

        statistics = analysis.statistics or {}
        quality = statistics.get("transient", {}).get("median_subtraction_quality")
        if quality is not None and np.isfinite(quality) and quality < 0.7:
            actions.append(
                f"Subtraction quality is poor ({quality:.2f} of the expected "
                "noise level). Improve the alignment or the PSF match before "
                "trusting any transient candidate from this run.")

        field = statistics.get("field", {})
        slope = field.get("counts_slope")
        if slope is not None and np.isfinite(slope) and slope < 0.25:
            actions.append(
                f"The number-count slope is {slope:.2f}, well below the "
                "Euclidean 0.6, so the catalog is already incomplete at the "
                "bright end of its range. Treat population statistics from "
                "this field as lower limits.")

        if not actions:
            actions.append(
                "Nothing in this field requires immediate follow-up. The "
                "catalog is suitable for population statistics.")
        actions.append(
            "Before publishing anything from this run, verify the astrometric "
            "solution and the photometric zero point against an external "
            "reference catalog.")
        return actions

    def report(self, analysis: FieldAnalysis) -> Dict[str, Any]:
        """The complete narrative layer, ready for any output format."""
        ranked = rank_candidates(analysis, self.top_candidates)
        return {
            "summary": self.summarise(analysis),
            "priority": [item.to_dict() for item in ranked],
            "priority_text": [self.describe_priority(item, analysis) for item in ranked],
            "recommendations": self.recommendations(analysis, ranked),
            "disclaimer": DISCOVERY_DISCLAIMER,
            "warnings": list(analysis.warnings),
        }


def _plural(word: str, count: int) -> str:
    """Pluralise a class name for prose."""
    name = word.replace("_", " ")
    if count == 1:
        return name
    if name.endswith("y") and not name.endswith(("ay", "ey", "oy", "uy")):
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"
