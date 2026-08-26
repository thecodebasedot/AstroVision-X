"""Scientific report generation in text, JSON and HTML."""

from __future__ import annotations

import os
from typing import Dict, Sequence

from ..core.logging import get_logger
from ..core.types import FieldAnalysis
from ..io.catalog import write_catalog
from .html import render_html, write_html
from .json_report import render_json, write_json
from .schema import build_report, flatten_for_display
from .text import render_text, write_text

log = get_logger("report")

__all__ = [
    "build_report", "flatten_for_display",
    "render_text", "write_text",
    "render_json", "write_json",
    "render_html", "write_html",
    "generate_reports",
]


def generate_reports(analysis: FieldAnalysis, output_dir: str,
                     formats: Sequence[str] = ("text", "json"),
                     title: str = "AstroVision-X Field Analysis",
                     observer: str = "", top_candidates: int = 10,
                     include_embeddings: bool = False,
                     image=None, basename: str = "report") -> Dict[str, str]:
    """Write every requested format plus the catalog; returns the paths."""
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, str] = {}
    options = {"title": title, "observer": observer, "top_candidates": top_candidates}

    for fmt in formats:
        key = str(fmt).lower()
        try:
            if key in ("text", "txt"):
                written["text"] = write_text(
                    analysis, os.path.join(output_dir, f"{basename}.txt"), **options)
            elif key == "json":
                written["json"] = write_json(
                    analysis, os.path.join(output_dir, f"{basename}.json"),
                    include_embeddings=include_embeddings, **options)
            elif key == "html":
                written["html"] = write_html(
                    analysis, os.path.join(output_dir, f"{basename}.html"),
                    image=image, **options)
            else:
                log.warning("unknown report format '%s'; skipping", fmt)
        except Exception as exc:                    # noqa: BLE001 - reported
            log.error("failed to write the %s report: %s", key, exc)
            analysis.warn(f"could not write the {key} report: {exc}")

    if len(analysis.catalog):
        for extension in ("csv", "json"):
            path = os.path.join(output_dir, f"catalog.{extension}")
            try:
                written[f"catalog_{extension}"] = write_catalog(analysis.catalog, path)
            except Exception as exc:                # noqa: BLE001 - reported
                log.error("failed to write catalog.%s: %s", extension, exc)

    log.info("wrote %d report file(s) to %s", len(written), output_dir)
    return written
