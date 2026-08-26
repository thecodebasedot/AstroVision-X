"""Optional dependency management.

AstroVision-X has exactly one hard dependency: NumPy.  Everything else --
SciPy, Astropy, scikit-learn, PyTorch, Matplotlib -- is optional.  Modules
import through this layer so that a missing package degrades a *feature*
rather than breaking the whole import graph.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Optional

from .exceptions import MissingDependencyError

_CACHE: Dict[str, Optional[ModuleType]] = {}

#: Maps an importable module name to the ``pip install 'astrovision-x[extra]'``
#: extra that provides it.
EXTRAS = {
    "scipy": "science",
    "astropy": "science",
    "sklearn": "ml",
    "torch": "deep",
    "matplotlib": "viz",
    "skimage": "all",
    "pandas": "all",
}


def try_import(name: str) -> Optional[ModuleType]:
    """Import ``name``, returning ``None`` instead of raising when absent."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        module: Optional[ModuleType] = importlib.import_module(name)
    except Exception:  # pragma: no cover - depends on environment
        module = None
    _CACHE[name] = module
    return module


def require(name: str, feature: str = "") -> ModuleType:
    """Import ``name`` or raise :class:`MissingDependencyError`."""
    module = try_import(name)
    if module is None:
        raise MissingDependencyError(name, feature, EXTRAS.get(name, ""))
    return module


def has(name: str) -> bool:
    """Return ``True`` when optional dependency ``name`` is importable."""
    return try_import(name) is not None


def capabilities() -> Dict[str, bool]:
    """Report which optional backends are available in this environment."""
    return {name: has(name) for name in EXTRAS}


def describe_capabilities() -> str:
    """Human-readable capability banner used by the CLI."""
    lines = []
    for name, available in sorted(capabilities().items()):
        mark = "available" if available else "missing"
        extra = EXTRAS.get(name, "")
        hint = "" if available else f"  (pip install 'astrovision-x[{extra}]')"
        lines.append(f"  {name:<12} {mark}{hint}")
    return "\n".join(lines)
