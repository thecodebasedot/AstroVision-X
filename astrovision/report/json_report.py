"""Machine-readable JSON report, for archiving and downstream tooling."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ..core.types import FieldAnalysis
from .schema import build_report


def _default(value: Any) -> Any:
    """Make NumPy types and non-finite floats JSON-safe."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "value"):          # enums
        return value.value
    return str(value)


def _clean(value: Any) -> Any:
    """Replace NaN/inf with null so the output is valid JSON everywhere."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def render_json(analysis: FieldAnalysis, indent: int = 2, **kwargs) -> str:
    """Render the report as a JSON string."""
    report = _clean(build_report(analysis, **kwargs))
    return json.dumps(report, indent=indent, default=_default)


def write_json(analysis: FieldAnalysis, path: str, indent: int = 2, **kwargs) -> str:
    """Write the JSON report to ``path``."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_json(analysis, indent, **kwargs))
    return path
