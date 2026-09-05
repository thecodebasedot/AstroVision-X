"""Plain-text scientific report.

Designed to be read in a terminal or pasted into an observing log, and to
be the format a scientist actually skims first.
"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np

from ..core.types import FieldAnalysis
from .schema import build_report

WIDTH = 78


def _rule(char: str = "=") -> str:
    return char * WIDTH


def _heading(text: str, char: str = "=") -> List[str]:
    return [_rule(char), text.upper(), _rule(char)]


def _wrap(text: str, indent: int = 0, width: int = WIDTH) -> List[str]:
    import textwrap
    prefix = " " * indent
    return textwrap.wrap(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix) or [prefix.rstrip()]


def _value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def render_text(analysis: FieldAnalysis, title: str = "AstroVision-X Field Analysis",
                observer: str = "", top_candidates: int = 10) -> str:
    """Render the full report as plain text."""
    report = build_report(analysis, title, observer, include_catalog=False,
                          top_candidates=top_candidates)
    lines: List[str] = []

    lines += _heading(report["title"])
    lines.append(f"Generated: {report['generated']}   "
                 f"AstroVision-X {report['version']}")
    if observer:
        lines.append(f"Observer:  {observer}")
    image = report.get("image", {})
    if image:
        lines.append(f"Image:     {image.get('name')}  "
                     f"{image.get('shape', ['?', '?'])[1]}x{image.get('shape', ['?', '?'])[0]} px  "
                     f"band={image.get('band')}  "
                     f"scale={_value(image.get('pixel_scale_arcsec'), 3)} arcsec/px")
    if report.get("elapsed_seconds") is not None:
        lines.append(f"Runtime:   {report['elapsed_seconds']:.1f} s")
    lines.append("")

    # -- overview ---------------------------------------------------------
    lines += _heading("1. Overview", "-")
    lines += _wrap(report["summary"].get("narrative", "No summary available."))
    lines.append("")
    summary = report["summary"]
    lines.append(f"  Sources detected      : {summary.get('n_sources', 0)}")
    for name, count in (summary.get("class_counts") or {}).items():
        lines.append(f"      {name:<18}: {count}")
    lines.append(f"  Transient candidates  : {summary.get('n_transients', 0)}"
                 + (f" ({report['rejected_transients']} rejected by vetting)"
                    if report.get("rejected_transients") else ""))
    lines.append(f"  Lens candidates       : {summary.get('n_lens_candidates', 0)}")
    lines.append(f"  Anomalies ranked      : {summary.get('n_anomalies', 0)}")
    lines.append(f"  Light curves          : {summary.get('n_light_curves', 0)}")
    lines.append("")

    # -- observation quality ----------------------------------------------
    lines += _heading("2. Data quality", "-")
    psf = report.get("psf") or {}
    photometry = report.get("photometry", {})
    lines.append(f"  PSF FWHM              : {_value(psf.get('fwhm'), 2)} px "
                 f"(from {psf.get('n_stars', 0)} stars)")
    lines.append(f"  PSF ellipticity       : {_value(psf.get('ellipticity'), 3)}")
    lines.append(f"  Background RMS        : {_value(photometry.get('median_rms'), 3)}")
    lines.append(f"  Zero point            : {_value(photometry.get('zero_point'), 2)}")
    lines.append(f"  5-sigma limit         : "
                 f"{_value(photometry.get('limiting_magnitude_5sigma'), 2)} mag")
    lines.append(f"  Aperture corrected    : {photometry.get('aperture_corrected', False)}")
    transient = report.get("transient_summary")
    if transient:
        lines.append(f"  Subtraction quality   : "
                     f"{_value(transient.get('median_subtraction_quality'), 2)} "
                     "(1.0 = noise-limited)")
    lines.append("")

    # -- field statistics --------------------------------------------------
    field = report.get("field", {})
    if field:
        lines += _heading("3. Field statistics", "-")
        lines.append(f"  Area                  : "
                     f"{_value(field.get('area_sq_arcmin'), 2)} sq arcmin")
        lines.append(f"  Source density        : "
                     f"{_value(field.get('source_density_per_sq_arcmin'), 1)} per sq arcmin")
        magnitude_range = field.get("magnitude_range") or []
        if len(magnitude_range) == 2:
            lines.append(f"  Magnitude range       : "
                         f"{_value(magnitude_range[0], 2)} to {_value(magnitude_range[1], 2)}")
        lines.append(f"  Number-count slope    : {_value(field.get('counts_slope'), 3)} "
                     "(Euclidean expectation 0.6)")
        lines.append(f"  Completeness turnover : "
                     f"{_value(field.get('completeness_limit'), 2)} mag")
        clustering = field.get("clustering", {})
        if clustering:
            lines.append(f"  Clark-Evans index     : "
                         f"{_value(clustering.get('clark_evans'), 3)} "
                         "(<1 clustered, >1 regular)")
        morphologies = field.get("morphology_counts")
        if morphologies:
            rendered = ", ".join(f"{k.replace('_', ' ')}: {v}"
                                 for k, v in morphologies.items())
            lines += _wrap(f"Galaxy morphologies   : {rendered}", indent=2)
        stellar = report.get("stellar", {})
        if stellar.get("n_stars"):
            lines.append(f"  Stars                 : {stellar['n_stars']}, median "
                         f"magnitude {_value(stellar.get('magnitude_median'), 2)}")
            if stellar.get("crowding_index") is not None:
                lines.append(f"  Crowding index        : "
                             f"{_value(stellar.get('crowding_index'), 3)} "
                             "(fraction of stars within 2 FWHM of a neighbour)")
        lines.append("")

    # -- ranked candidates -------------------------------------------------
    lines += _heading("4. Ranked follow-up candidates", "-")
    if not report["priority_text"]:
        lines += _wrap("No object in this field met the thresholds for follow-up.",
                       indent=2)
    for text in report["priority_text"]:
        lines.extend(text.split("\n"))
        lines.append("")

    # -- transients --------------------------------------------------------
    if report.get("transients"):
        lines += _heading("5. Transient candidates", "-")
        lines.append(f"  {'ID':<4}{'x':>8}{'y':>8}{'sigma':>8}{'R/B':>6}"
                     f"{'epochs':>8}  {'classification':<22}{'verdict'}")
        for candidate in report["transients"]:
            lines.append(
                f"  {candidate['id']:<4}{candidate['x']:>8.1f}{candidate['y']:>8.1f}"
                f"{candidate['significance']:>8.1f}{candidate['real_bogus']:>6.2f}"
                f"{candidate['meta'].get('n_detections', 1):>8}  "
                f"{candidate['classification'].replace('_', ' '):<22}"
                f"{candidate['verdict'].replace('_', ' ')}")
        lines.append("")

    # -- lenses ------------------------------------------------------------
    if report.get("lens_candidates"):
        lines += _heading("6. Gravitational-lens candidates", "-")
        for lens in report["lens_candidates"]:
            lines.append(f"  Source #{lens['source_id']}: score {lens['score']:.2f}, "
                         f"{lens['arc_count']} arc(s), Einstein radius "
                         f"{_value(lens['einstein_radius_arcsec'], 2)} arcsec, "
                         f"ring {100 * lens['ring_completeness']:.0f}% complete")
            for note in lens.get("notes", []):
                lines += _wrap(note, indent=6)
        lines.append("")

    # -- anomalies ---------------------------------------------------------
    if report.get("anomalies"):
        lines += _heading("7. Novelty candidates", "-")
        for record in report["anomalies"][:top_candidates]:
            lines.append(f"  #{record['rank']} source {record['source_id']}: "
                         f"score {record['score']:.3f} "
                         f"({record['novelty_type'].replace('_', ' ')})")
            lines += _wrap(record["explanation"], indent=6)
        lines.append("")

    # -- recommendations ---------------------------------------------------
    lines += _heading("8. Recommended next steps", "-")
    for index, action in enumerate(report["recommendations"], start=1):
        lines += _wrap(f"{index}. {action}", indent=2)
        lines.append("")

    # -- provenance --------------------------------------------------------
    lines += _heading("9. Provenance", "-")
    for stage in report.get("stages", []):
        detail = stage.get("message") or ""
        lines.append(f"  {stage['name']:<16}{stage['status']:<9}"
                     f"{stage['seconds']:>7.2f}s  {detail}")
    capabilities = report.get("capabilities", {})
    if capabilities:
        available = ", ".join(k for k, v in sorted(capabilities.items()) if v) or "none"
        missing = ", ".join(k for k, v in sorted(capabilities.items()) if not v) or "none"
        lines += _wrap(f"Optional backends available: {available}", indent=2)
        lines += _wrap(f"Optional backends missing:   {missing}", indent=2)
    manifest = report.get("manifest") or {}
    if manifest:
        git = manifest.get("git") or {}
        lines.append("")
        lines.append("  Reproducing this run:")
        lines.append(f"    reproducibility key : {report.get('reproducibility_key')}")
        lines.append(f"    configuration hash  : {manifest.get('config_hash')}")
        lines.append(f"    code revision       : {git.get('revision') or 'not a git checkout'}"
                     f"{'  (uncommitted changes)' if git.get('dirty') else ''}")
        lines.append(f"    package / python    : {manifest.get('package_version')} / "
                     f"{manifest.get('python')}")
        deps = ", ".join(f"{k} {v}" for k, v in sorted(manifest.get("dependencies", {}).items()) if v)
        lines += _wrap(f"dependencies        : {deps or 'none recorded'}", indent=4)
        seeds = ", ".join(f"{k}={v}" for k, v in manifest.get("seeds", {}).items())
        lines.append(f"    seeds               : {seeds or 'none'}")
        for name, checksum in sorted((manifest.get("inputs") or {}).items()):
            lines.append(f"    input {name:<14}: {checksum}")
        digest = (manifest.get("outputs") or {}).get("catalog_digest")
        if digest:
            lines.append(f"    catalog digest      : {digest}")
        for note in manifest.get("notes", []):
            lines += _wrap(f"note: {note}", indent=4)
    physical = report.get("physical", {})
    for assumption in physical.get("assumptions", []):
        lines += _wrap(f"Assumption: {assumption}", indent=2)
    if report["warnings"]:
        lines.append("")
        lines.append("  WARNINGS:")
        for warning in report["warnings"]:
            lines += _wrap(f"- {warning}", indent=4)
    lines.append("")

    lines += _heading("Important", "-")
    lines += _wrap(report["disclaimer"], indent=2)
    lines.append(_rule())
    return "\n".join(lines)


def write_text(analysis: FieldAnalysis, path: str, **kwargs) -> str:
    """Write the text report to ``path``."""
    text = render_text(analysis, **kwargs)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
