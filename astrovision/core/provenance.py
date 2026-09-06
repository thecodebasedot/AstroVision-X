"""What would be needed to get this exact result again.

A catalog without its provenance is a number without a unit. Six months on,
the question is never "what did the pipeline find" but "could I reproduce it,
and if I cannot, what changed" -- and the answer has to come from something
written down *at the time*, because the environment that produced the result
will not survive the interval.

The manifest records the things that determine a run's output:

* the **configuration**, as a content hash and in full, because the hash
  says whether two runs were configured alike and the full text says how they
  differed;
* the **code**, as the package version and the git revision, with a flag when
  the working tree was dirty -- a result from uncommitted code has no
  reproducible source;
* the **dependencies** that do arithmetic: NumPy, and whichever of SciPy,
  Astropy, scikit-learn and PyTorch were present, because a NumPy upgrade can
  change a sum in the last bit and a PyTorch one can change a model's answer;
* the **random seeds**, since every stochastic stage draws from a seeded
  generator and the seed is the only thing standing between "random" and
  "reproducible";
* the **inputs**, as file checksums, since "the same image" is a claim that
  needs checking.

The check that makes this worth having is in :func:`same_result`: two runs
whose manifests match must produce identical catalogs, and the test suite
asserts that they do.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .backend import try_import
from .logging import get_logger

log = get_logger("core.provenance")

#: Packages whose version can change a numerical result.
NUMERICAL_DEPENDENCIES: Sequence[str] = ("numpy", "scipy", "astropy", "sklearn",
                                          "torch", "photutils", "sep")


def _package_version(name: str) -> Optional[str]:
    module = try_import(name)
    if module is None:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _git(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
    try:
        output = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0:
        return None
    return output.stdout.strip()


def git_state(path: Optional[str] = None) -> Dict[str, Any]:
    """The revision the code was run from, and whether it was clean."""
    root = path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    revision = _git(["rev-parse", "HEAD"], cwd=root)
    if revision is None:
        return {"revision": None, "dirty": None,
                "note": "not a git checkout, or git is unavailable"}
    status = _git(["status", "--porcelain"], cwd=root)
    return {"revision": revision, "dirty": bool(status),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)}


def file_checksum(path: str, algorithm: str = "sha256",
                  chunk: int = 1 << 20) -> str:
    """Content hash of a file, streamed so a gigabyte frame does not need RAM."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return f"{algorithm}:{digest.hexdigest()}"


def config_hash(config: Any) -> str:
    """A content hash of a configuration, stable across key order."""
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    if isinstance(payload, dict):
        # How many processes did the work is not part of what the result is.
        payload = {k: v for k, v in payload.items() if k != "n_workers"}
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    """Everything that determined one run's output."""

    created: str = ""
    package_version: str = ""
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    git: Dict[str, Any] = field(default_factory=dict)
    python: str = ""
    platform: str = ""
    dependencies: Dict[str, Optional[str]] = field(default_factory=dict)
    seeds: Dict[str, int] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def reproducibility_key(self) -> str:
        """The part of the manifest that decides whether results can match.

        Timestamps and output paths do not affect the result; the
        configuration, code, numerical dependencies, seeds and inputs do.
        """
        payload = {"config_hash": self.config_hash,
                   "package_version": self.package_version,
                   "git": self.git.get("revision"),
                   "dependencies": self.dependencies,
                   "seeds": self.seeds, "inputs": self.inputs}
        text = json.dumps(payload, sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def differences(self, other: "Manifest") -> List[str]:
        """Human-readable reasons two runs might not agree."""
        reasons: List[str] = []
        if self.config_hash != other.config_hash:
            reasons.append("configuration differs")
        if self.package_version != other.package_version:
            reasons.append(f"package version {self.package_version} vs "
                           f"{other.package_version}")
        if self.git.get("revision") != other.git.get("revision"):
            reasons.append("code revision differs")
        if self.git.get("dirty") or other.git.get("dirty"):
            reasons.append("at least one run used uncommitted code")
        for name in sorted(set(self.dependencies) | set(other.dependencies)):
            if self.dependencies.get(name) != other.dependencies.get(name):
                reasons.append(f"{name} {self.dependencies.get(name)} vs "
                               f"{other.dependencies.get(name)}")
        if self.seeds != other.seeds:
            reasons.append("random seeds differ")
        for name in sorted(set(self.inputs) | set(other.inputs)):
            if self.inputs.get(name) != other.inputs.get(name):
                reasons.append(f"input {name} differs")
        return reasons


def build_manifest(config: Any, inputs: Optional[Iterable[str]] = None,
                   seeds: Optional[Dict[str, int]] = None,
                   notes: Optional[Sequence[str]] = None) -> Manifest:
    """Record the state of the world before a run starts."""
    from ..version import __version__

    manifest = Manifest(
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        package_version=str(__version__),
        config_hash=config_hash(config),
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config or {}),
        git=git_state(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        dependencies={name: _package_version(name) for name in NUMERICAL_DEPENDENCIES},
        seeds=dict(seeds or {}),
        notes=list(notes or []))
    random_state = manifest.config.get("random_state")
    if random_state is not None and "random_state" not in manifest.seeds:
        manifest.seeds["random_state"] = int(random_state)
    for path in inputs or []:
        if os.path.exists(path):
            manifest.inputs[os.path.basename(path)] = file_checksum(path)
        else:
            manifest.notes.append(f"input {path} not found at manifest time")
    if manifest.git.get("dirty"):
        manifest.notes.append("working tree had uncommitted changes; the "
                              "revision alone does not reproduce this run")
    return manifest


def catalog_digest(catalog: Any, fields: Sequence[str] = ("x", "y"),
                   precision: int = 6) -> str:
    """A content hash of a catalog's measurements.

    Rounded to ``precision`` decimals so that a last-bit difference in a
    floating-point sum -- which is not a reproducibility failure -- does not
    register as one, while anything larger does.
    """
    rows = []
    for source in catalog:
        rows.append([round(float(getattr(source, name, float("nan"))), precision)
                     for name in fields])
        photometry = getattr(source, "photometry", None)
        if photometry is not None:
            rows[-1].append(round(float(getattr(photometry, "flux", float("nan"))),
                                  precision))
    rows.sort()
    text = json.dumps(rows, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def same_result(first: Any, second: Any, **kwargs) -> bool:
    """True when two catalogs carry the same measurements."""
    return catalog_digest(first, **kwargs) == catalog_digest(second, **kwargs)
