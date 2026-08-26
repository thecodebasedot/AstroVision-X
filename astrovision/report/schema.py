"""The report data model.

Every output format renders the same structure, so a text summary, a JSON
payload and an HTML page can never disagree about what the run found.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict

import numpy as np

from ..core.types import FieldAnalysis
from ..engine.assistant import DISCOVERY_DISCLAIMER


def build_report(analysis: FieldAnalysis, title: str = "AstroVision-X Field Analysis",
                 observer: str = "", include_catalog: bool = True,
                 top_candidates: int = 10,
                 include_embeddings: bool = False) -> Dict[str, Any]:
    """Assemble the canonical report structure from a completed analysis."""
    narrative = analysis.statistics.get("narrative", {})
    statistics = analysis.statistics or {}
    provenance = analysis.provenance or {}

    report: Dict[str, Any] = {
        "title": title,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "observer": observer,
        "version": provenance.get("version", "unknown"),
        "summary": {
            **analysis.summary(),
            "narrative": narrative.get("summary", ""),
        },
        "image": provenance.get("image", {}),
        "psf": provenance.get("psf"),
        "wcs": provenance.get("wcs"),
        "photometry": statistics.get("photometry", {}),
        "field": statistics.get("field", {}),
        "stellar": statistics.get("stellar", {}),
        "physical": statistics.get("physical", {}),
        "priority": narrative.get("priority", [])[:top_candidates],
        "priority_text": narrative.get("priority_text", [])[:top_candidates],
        "recommendations": narrative.get("recommendations", []),
        "disclaimer": narrative.get("disclaimer", DISCOVERY_DISCLAIMER),
        "warnings": list(analysis.warnings),
        "stages": provenance.get("stages", []),
        "capabilities": provenance.get("capabilities", {}),
        "config": provenance.get("config", {}),
        "elapsed_seconds": provenance.get("elapsed_seconds"),
    }

    if statistics.get("transient"):
        report["transient_summary"] = statistics["transient"]
    if statistics.get("timeseries"):
        report["timeseries_summary"] = statistics["timeseries"]
    if provenance.get("series"):
        report["series"] = provenance["series"]

    report["transients"] = [c.to_dict() for c in analysis.transients
                            if "bogus" not in c.flags][:top_candidates * 3]
    report["rejected_transients"] = sum(1 for c in analysis.transients
                                        if "bogus" in c.flags)
    report["anomalies"] = [a.to_dict() for a in analysis.anomalies][:top_candidates * 2]
    report["lens_candidates"] = [l.to_dict() for l in analysis.lenses]

    if include_catalog:
        report["catalog"] = analysis.catalog.to_dict(include_embedding=include_embeddings)
    report["catalog_size"] = len(analysis.catalog)
    return report


def flatten_for_display(value: Any, precision: int = 4) -> Any:
    """Round floats and drop non-finite values so output stays readable."""
    if isinstance(value, dict):
        return {k: flatten_for_display(v, precision) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [flatten_for_display(v, precision) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), precision)
    if isinstance(value, np.ndarray):
        return flatten_for_display(value.tolist(), precision)
    return value
