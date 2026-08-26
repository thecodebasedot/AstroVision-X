"""Self-contained HTML report.

Everything is inlined -- styles, and any figures as data URIs -- so a single
file can be attached to an email or archived with the data it describes.
"""

from __future__ import annotations

import base64
import html
import io
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.backend import try_import
from ..core.types import FieldAnalysis
from ..preprocess.normalize import asinh_stretch, zscale
from .schema import build_report

STYLE = """
:root{--bg:#ffffff;--fg:#1a1d23;--muted:#5b6472;--line:#dfe3ea;--accent:#2b5fa8;
--warn:#a8642b;--ok:#2b8a55;--card:#f7f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e8eaee;--muted:#9aa3b2;
--line:#2b313a;--accent:#7aa8e8;--warn:#e0a468;--ok:#6ecf9a;--card:#1b1f26}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem}
h2{font-size:1.15rem;margin:2.2rem 0 .6rem;padding-bottom:.3rem;
border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.2rem 0 .4rem}
.sub{color:var(--muted);margin:0 0 1.5rem;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem}
.card .n{font-size:1.5rem;font-weight:600;color:var(--accent)}
.card .l{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.6rem 0}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.cand{border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:6px;padding:.7rem 1rem;margin:.6rem 0;background:var(--card)}
.cand h3{margin-top:0}
.tag{display:inline-block;padding:.1rem .5rem;border-radius:999px;font-size:.75rem;
background:var(--line);color:var(--fg);margin-left:.4rem}
.tag.hi{background:var(--warn);color:#fff}
.tag.ok{background:var(--ok);color:#fff}
.muted{color:var(--muted);font-size:.87rem}
.note{border-left:3px solid var(--warn);background:var(--card);padding:.7rem 1rem;
border-radius:0 6px 6px 0;margin:1rem 0}
ol,ul{padding-left:1.3rem}
figure{margin:1rem 0}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--muted);font-size:.83rem;margin-top:.4rem}
code{background:var(--card);padding:.1rem .3rem;border-radius:3px;font-size:.85em}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if not np.isfinite(number):
        return "&mdash;"
    return f"{number:.{digits}f}"


def _verdict_tag(verdict: str) -> str:
    label = _escape(verdict.replace("_", " "))
    if verdict == "high_priority":
        return f'<span class="tag hi">{label}</span>'
    if verdict in ("follow_up_recommended", "worth_a_look"):
        return f'<span class="tag ok">{label}</span>'
    return f'<span class="tag">{label}</span>'


def _figure(image, catalog=None, max_sources: int = 300) -> Optional[str]:
    """Render the field as an inline PNG with detections marked."""
    matplotlib = try_import("matplotlib")
    if matplotlib is None or image is None:
        return None
    matplotlib.use("Agg")
    plt = try_import("matplotlib.pyplot")
    if plt is None:
        return None

    data = image.subtracted() if hasattr(image, "subtracted") else np.asarray(image)
    figure, axes = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    axes.imshow(asinh_stretch(data), origin="lower", cmap="gray",
                interpolation="nearest")
    if catalog is not None:
        shown = 0
        for source in catalog:
            if shown >= max_sources:
                break
            colour = {"star": "#7aa8e8", "galaxy": "#e0a468"}.get(
                source.object_class.value, "#6ecf9a")
            radius = max(source.morphology.semi_major * 2.0, 3.0)
            axes.add_patch(plt.Circle((source.x, source.y), radius, fill=False,
                                      edgecolor=colour, linewidth=0.6, alpha=0.8))
            shown += 1
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(getattr(image, "name", "field"), fontsize=9)
    figure.tight_layout(pad=0.2)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(analysis: FieldAnalysis, title: str = "AstroVision-X Field Analysis",
                observer: str = "", top_candidates: int = 10,
                image=None) -> str:
    """Render the report as a single self-contained HTML document."""
    report = build_report(analysis, title, observer, include_catalog=False,
                          top_candidates=top_candidates)
    summary = report["summary"]
    parts: List[str] = []

    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append(f"<title>{_escape(report['title'])}</title>")
    parts.append(f"<style>{STYLE}</style></head><body><main>")

    parts.append(f"<h1>{_escape(report['title'])}</h1>")
    image_meta = report.get("image", {})
    subtitle = [f"Generated {_escape(report['generated'])}",
                f"AstroVision-X {_escape(report['version'])}"]
    if image_meta.get("name"):
        subtitle.insert(0, _escape(image_meta["name"]))
    if observer:
        subtitle.append(f"Observer: {_escape(observer)}")
    parts.append(f"<p class='sub'>{' &middot; '.join(subtitle)}</p>")

    # -- headline numbers --------------------------------------------------
    cards = [("Sources", summary.get("n_sources", 0)),
             ("Transients", summary.get("n_transients", 0)),
             ("Lens candidates", summary.get("n_lens_candidates", 0)),
             ("Anomalies", summary.get("n_anomalies", 0))]
    parts.append("<div class='grid'>")
    for label, number in cards:
        parts.append(f"<div class='card'><div class='l'>{_escape(label)}</div>"
                     f"<div class='n'>{_escape(number)}</div></div>")
    parts.append("</div>")

    parts.append("<h2>Overview</h2>")
    parts.append(f"<p>{_escape(summary.get('narrative', ''))}</p>")

    data_uri = _figure(image, analysis.catalog)
    if data_uri:
        parts.append("<figure>")
        parts.append(f"<img src='{data_uri}' alt='Detected sources in the field'>")
        parts.append("<figcaption>Asinh-stretched field with detections circled: "
                     "blue = star, orange = galaxy, green = other.</figcaption>")
        parts.append("</figure>")

    # -- data quality ------------------------------------------------------
    psf = report.get("psf") or {}
    photometry = report.get("photometry", {})
    parts.append("<h2>Data quality</h2><div class='scroll'><table><tbody>")
    rows = [
        ("PSF FWHM (px)", _number(psf.get("fwhm"), 2)),
        ("PSF stars used", _number(psf.get("n_stars"))),
        ("PSF ellipticity", _number(psf.get("ellipticity"))),
        ("Background RMS", _number(photometry.get("median_rms"))),
        ("Photometric zero point", _number(photometry.get("zero_point"), 2)),
        ("5-sigma limiting magnitude", _number(
            photometry.get("limiting_magnitude_5sigma"), 2)),
        ("Aperture correction applied", _escape(
            photometry.get("aperture_corrected", False))),
    ]
    transient = report.get("transient_summary")
    if transient:
        rows.append(("Subtraction quality (1 = noise-limited)",
                     _number(transient.get("median_subtraction_quality"), 2)))
    for label, value in rows:
        parts.append(f"<tr><th>{_escape(label)}</th><td class='num'>{value}</td></tr>")
    parts.append("</tbody></table></div>")

    # -- field statistics --------------------------------------------------
    field = report.get("field", {})
    if field:
        parts.append("<h2>Field statistics</h2><div class='scroll'><table><tbody>")
        clustering = field.get("clustering", {})
        for label, value in [
            ("Area (sq arcmin)", _number(field.get("area_sq_arcmin"), 2)),
            ("Source density (per sq arcmin)",
             _number(field.get("source_density_per_sq_arcmin"), 1)),
            ("Number-count slope (Euclidean = 0.6)",
             _number(field.get("counts_slope"))),
            ("Completeness turnover (mag)",
             _number(field.get("completeness_limit"), 2)),
            ("Clark-Evans index (&lt;1 clustered)",
             _number(clustering.get("clark_evans"))),
            ("Star/galaxy ratio", _number(field.get("star_galaxy_ratio"), 2)),
        ]:
            parts.append(f"<tr><th>{label}</th><td class='num'>{value}</td></tr>")
        parts.append("</tbody></table></div>")
        morphologies = field.get("morphology_counts")
        if morphologies:
            parts.append("<h3>Galaxy morphologies</h3><div class='scroll'><table>")
            parts.append("<thead><tr><th>Type</th><th class='num'>Count</th>"
                         "</tr></thead><tbody>")
            for name, count in morphologies.items():
                parts.append(f"<tr><td>{_escape(name.replace('_', ' '))}</td>"
                             f"<td class='num'>{count}</td></tr>")
            parts.append("</tbody></table></div>")

    # -- ranked candidates -------------------------------------------------
    parts.append("<h2>Ranked follow-up candidates</h2>")
    if not report["priority"]:
        parts.append("<p class='muted'>No object met the thresholds for follow-up.</p>")
    for item in report["priority"]:
        parts.append("<div class='cand'>")
        heading = f"#{item['rank']} &middot; {_escape(item['kind'].title())}"
        if item.get("source_id") is not None:
            heading += f" &middot; source {item['source_id']}"
        parts.append(f"<h3>{heading}{_verdict_tag(item['verdict'])}</h3>")
        position = item.get("position", [float('nan')] * 2)
        where = f"pixel ({_number(position[0], 1)}, {_number(position[1], 1)})"
        sky = item.get("sky_position")
        if sky:
            where += (f" &middot; RA {_number(sky[0], 5)}&deg;, "
                      f"Dec {_number(sky[1], 5)}&deg;")
        parts.append(f"<p class='muted'>{where} &middot; "
                     f"priority {_number(item['score'], 2)}</p>")
        if item.get("reasons"):
            parts.append("<ul>" + "".join(
                f"<li>{_escape(r)}</li>" for r in item["reasons"]) + "</ul>")
        if item.get("caveats"):
            parts.append("<div class='note'>" + " ".join(
                _escape(c) for c in item["caveats"]) + "</div>")
        parts.append("</div>")

    # -- transient table ---------------------------------------------------
    if report.get("transients"):
        parts.append("<h2>Transient candidates</h2><div class='scroll'><table>")
        parts.append("<thead><tr><th>ID</th><th class='num'>x</th><th class='num'>y</th>"
                     "<th class='num'>sigma</th><th class='num'>R/B</th>"
                     "<th class='num'>epochs</th><th>Classification</th>"
                     "<th>Verdict</th></tr></thead><tbody>")
        for candidate in report["transients"]:
            parts.append(
                f"<tr><td>{candidate['id']}</td>"
                f"<td class='num'>{_number(candidate['x'], 1)}</td>"
                f"<td class='num'>{_number(candidate['y'], 1)}</td>"
                f"<td class='num'>{_number(candidate['significance'], 1)}</td>"
                f"<td class='num'>{_number(candidate['real_bogus'], 2)}</td>"
                f"<td class='num'>{candidate['meta'].get('n_detections', 1)}</td>"
                f"<td>{_escape(candidate['classification'].replace('_', ' '))}</td>"
                f"<td>{_verdict_tag(candidate['verdict'])}</td></tr>")
        parts.append("</tbody></table></div>")

    # -- lenses ------------------------------------------------------------
    if report.get("lens_candidates"):
        parts.append("<h2>Gravitational-lens candidates</h2>")
        for lens in report["lens_candidates"]:
            parts.append("<div class='cand'>")
            parts.append(f"<h3>Source {lens['source_id']}"
                         f"{_verdict_tag(lens['verdict'])}</h3>")
            parts.append(
                f"<p class='muted'>score {_number(lens['score'], 2)} &middot; "
                f"{lens['arc_count']} arc(s) &middot; Einstein radius "
                f"{_number(lens['einstein_radius_arcsec'], 2)} arcsec &middot; "
                f"ring {_number(100 * lens['ring_completeness'], 0)}% complete</p>")
            if lens.get("notes"):
                parts.append("<ul>" + "".join(
                    f"<li>{_escape(n)}</li>" for n in lens["notes"]) + "</ul>")
            parts.append("</div>")

    # -- anomalies ---------------------------------------------------------
    if report.get("anomalies"):
        parts.append("<h2>Novelty candidates</h2><div class='scroll'><table>")
        parts.append("<thead><tr><th class='num'>Rank</th><th class='num'>Source</th>"
                     "<th class='num'>Score</th><th>Type</th><th>Why</th>"
                     "</tr></thead><tbody>")
        for record in report["anomalies"][:top_candidates]:
            parts.append(
                f"<tr><td class='num'>{record['rank']}</td>"
                f"<td class='num'>{record['source_id']}</td>"
                f"<td class='num'>{_number(record['score'])}</td>"
                f"<td>{_escape(record['novelty_type'].replace('_', ' '))}</td>"
                f"<td class='muted'>{_escape(record['explanation'])}</td></tr>")
        parts.append("</tbody></table></div>")

    # -- recommendations ---------------------------------------------------
    parts.append("<h2>Recommended next steps</h2><ol>")
    for action in report["recommendations"]:
        parts.append(f"<li>{_escape(action)}</li>")
    parts.append("</ol>")

    # -- provenance --------------------------------------------------------
    parts.append("<h2>Provenance</h2><div class='scroll'><table>")
    parts.append("<thead><tr><th>Stage</th><th>Status</th><th class='num'>Seconds</th>"
                 "<th>Notes</th></tr></thead><tbody>")
    for stage in report.get("stages", []):
        parts.append(f"<tr><td>{_escape(stage['name'])}</td>"
                     f"<td>{_escape(stage['status'])}</td>"
                     f"<td class='num'>{_number(stage['seconds'], 2)}</td>"
                     f"<td class='muted'>{_escape(stage.get('message', ''))}</td></tr>")
    parts.append("</tbody></table></div>")

    physical = report.get("physical", {})
    if physical.get("assumptions"):
        parts.append("<h3>Assumptions</h3><ul>" + "".join(
            f"<li>{_escape(a)}</li>" for a in physical["assumptions"]) + "</ul>")
    if report["warnings"]:
        parts.append("<div class='note'><strong>Warnings.</strong><ul>" + "".join(
            f"<li>{_escape(w)}</li>" for w in report["warnings"]) + "</ul></div>")

    parts.append(f"<div class='note'><strong>Important.</strong> "
                 f"{_escape(report['disclaimer'])}</div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_html(analysis: FieldAnalysis, path: str, **kwargs) -> str:
    """Write the HTML report to ``path``."""
    document = render_html(analysis, **kwargs)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return path
