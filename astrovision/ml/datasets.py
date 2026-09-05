"""Labelled stamp sets, from simulated fields or from real survey files.

Everything trained in this package so far has been trained on simulated data,
where the label is exact, the stamp is clean, and the instrument is the one the
model will be used on. Real training data is none of those things, and the gap
is not a detail:

* **Labels are votes, not facts.** Galaxy Zoo gives a fraction of volunteers
  who chose each answer. A stamp where 51 % said spiral is not the same
  training example as one where 98 % did, and treating them alike teaches the
  model the disagreement.
* **Stamps have holes.** Chip gaps, saturated columns, masked cosmic rays --
  real cutouts arrive with NaNs, and a NaN entering a network makes every
  weight downstream NaN on the first backward pass. It has to be handled at
  the door.
* **Units are arbitrary and vary between files.** Counts, nanomaggies,
  calibrated flux. The per-stamp asinh stretch already makes the input
  scale-free, which is worth knowing: it means what survives is the *shape*
  of the object, so a domain gap is about optics and noise rather than units.
* **The instrument is different.** Different seeing, pixel scale, depth and
  background structure from whatever the model was trained on.

This module provides the loaders and the bookkeeping. What it deliberately
does not do is pretend that any of it has been run against real survey files:
no external data was reachable from the environment this was written in, so
the loaders are exercised against files written in the same formats, and the
domain shift is measured between two *simulated* instruments. The
:mod:`astrovision.ml.transfer` module measures what that shift costs and what
it takes to recover from it, which is the question a real dataset would be
used to answer.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import try_import
from ..core.logging import get_logger
from ..core.numeric import as_float_image
from ..core.types import ObjectClass

log = get_logger("ml.datasets")

#: A stamp with more than this fraction of unusable pixels is dropped rather
#: than filled.  Filling a few masked pixels with the local background is
#: harmless; filling half a stamp invents an object.
MAX_BAD_FRACTION = 0.25

#: Minimum agreement among labellers before a vote-based label is used.  Below
#: this the example is not a hard case to learn from, it is a case the
#: labellers themselves did not settle.
MIN_VOTE_AGREEMENT = 0.60


@dataclass
class StampSet:
    """Stamps with labels, weights and provenance."""

    stamps: List[np.ndarray] = field(default_factory=list)
    labels: List[ObjectClass] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)
    source: str = ""

    def __len__(self) -> int:
        return len(self.stamps)

    def counts(self) -> Dict[str, int]:
        """How many examples of each class."""
        result: Dict[str, int] = {}
        for label in self.labels:
            key = label.value if isinstance(label, ObjectClass) else str(label)
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def add(self, stamp: np.ndarray, label: ObjectClass, weight: float = 1.0,
            identifier: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        self.stamps.append(stamp)
        self.labels.append(label)
        self.weights.append(float(weight))
        self.ids.append(str(identifier))
        self.meta.append(dict(meta or {}))

    def subset(self, index: Sequence[int]) -> "StampSet":
        chosen = StampSet(source=self.source)
        for i in index:
            chosen.add(self.stamps[i], self.labels[i], self.weights[i],
                       self.ids[i], self.meta[i])
        return chosen

    def report(self) -> Dict[str, Any]:
        return {"n": len(self), "source": self.source, "classes": self.counts(),
                "dropped": dict(self.dropped),
                "mean_weight": float(np.mean(self.weights)) if self.weights else float("nan"),
                "stamp_shapes": sorted({s.shape for s in self.stamps})[:5]}


def clean_stamp(stamp: np.ndarray,
                max_bad_fraction: float = MAX_BAD_FRACTION
                ) -> Tuple[Optional[np.ndarray], str]:
    """Make one stamp safe to put through a network, or refuse it.

    Returns ``(stamp, reason)``; the stamp is ``None`` when it was refused.

    A NaN reaching an optimiser turns every weight downstream into a NaN on the
    first backward pass, and the failure appears far from its cause -- so bad
    pixels are dealt with here, at the door, and never passed on.

    >>> stamp = np.ones((8, 8)); stamp[0, 0] = np.nan
    >>> cleaned, reason = clean_stamp(stamp)
    >>> bool(np.isfinite(cleaned).all()), reason
    (True, 'filled 1 bad pixel')
    """
    data = as_float_image(stamp)
    if data.ndim != 2 or data.size == 0:
        return None, "not a two-dimensional stamp"
    bad = ~np.isfinite(data)
    fraction = float(bad.mean())
    if fraction > float(max_bad_fraction):
        return None, f"{100 * fraction:.0f}% of pixels unusable"
    if not bad.any():
        return data, "clean"
    good = data[~bad]
    # The local median, not zero: a hole filled with zero is a dark patch, and
    # a network learns dark patches as a feature of whatever class had the most
    # chip gaps.
    data = np.where(bad, float(np.median(good)) if good.size else 0.0, data)
    count = int(bad.sum())
    return data, f"filled {count} bad pixel" + ("s" if count != 1 else "")


def read_label_table(path: str, id_column: str = "id",
                     class_column: Optional[str] = None,
                     vote_columns: Optional[Dict[str, str]] = None,
                     min_agreement: float = MIN_VOTE_AGREEMENT
                     ) -> Dict[str, Tuple[str, float]]:
    """Read labels from a CSV, either as a class or as vote fractions.

    ``vote_columns`` maps a class name to the column holding its vote
    fraction, which is the form crowd-sourced catalogues actually come in. The
    winning class becomes the label and its fraction becomes the *weight*, so
    a stamp 98 % of labellers agreed on counts for more than one they split
    over -- and one below ``min_agreement`` is dropped, because it is not a
    hard example, it is an unsettled one.

    Returns ``{id: (class_name, weight)}``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"label table not found: {path}")
    table: Dict[str, Tuple[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return table
        for row in reader:
            key = str(row.get(id_column, "")).strip()
            if not key:
                continue
            if vote_columns:
                votes: Dict[str, float] = {}
                for name, column in vote_columns.items():
                    try:
                        votes[name] = float(row.get(column, "nan"))
                    except (TypeError, ValueError):
                        continue
                votes = {k: v for k, v in votes.items() if np.isfinite(v)}
                if not votes:
                    continue
                total = sum(max(v, 0.0) for v in votes.values())
                if total <= 0:
                    continue
                best = max(votes, key=lambda k: votes[k])
                agreement = max(votes[best], 0.0) / total
                if agreement < float(min_agreement):
                    continue
                table[key] = (best, float(agreement))
            elif class_column:
                value = str(row.get(class_column, "")).strip()
                if value:
                    table[key] = (value, 1.0)
    return table


def load_fits_cutouts(directory: str, labels: Dict[str, Tuple[str, float]],
                      class_map: Optional[Dict[str, ObjectClass]] = None,
                      pattern: str = "*.fits", hdu: int = 0,
                      max_bad_fraction: float = MAX_BAD_FRACTION,
                      limit: Optional[int] = None) -> StampSet:
    """Load a directory of FITS cutouts and attach labels by filename stem.

    This is the shape real training sets arrive in: one file per object, named
    by catalogue identifier, and a separate table of labels. Files without a
    label, labels without a file, unreadable files and stamps too damaged to
    use are all counted and reported rather than silently skipped -- a loader
    that quietly drops a third of a class produces a model that has never seen
    it and no message saying so.
    """
    fits = try_import("astropy.io.fits")
    if fits is None:                                    # pragma: no cover
        raise ImportError("astropy is required to read FITS cutouts")
    mapping = dict(class_map or {})
    dataset = StampSet(source=f"fits:{directory}")

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    seen: set = set()
    for path in paths:
        if limit is not None and len(dataset) >= limit:
            break
        stem = os.path.basename(path).split(".")[0]
        seen.add(stem)
        entry = labels.get(stem)
        if entry is None:
            dataset.drop("no label for this file")
            continue
        name, weight = entry
        label = mapping.get(name)
        if label is None:
            try:
                label = ObjectClass(name)
            except ValueError:
                dataset.drop(f"unknown class {name!r}")
                continue
        try:
            with fits.open(path, memmap=False) as handle:
                data = np.asarray(handle[hdu].data, dtype=float)
        except Exception as error:                      # pragma: no cover
            log.warning("could not read %s: %s", path, error)
            dataset.drop("unreadable file")
            continue
        if data.ndim == 3:
            # A stamp cube: survey cutout services often stack bands or the
            # science/reference/difference triplet.  The first plane is the
            # convention here and the rest are recorded, not thrown away.
            planes = data.shape[0]
            data = data[0]
        else:
            planes = 1
        cleaned, reason = clean_stamp(data, max_bad_fraction)
        if cleaned is None:
            dataset.drop(reason)
            continue
        dataset.add(cleaned, label, weight, stem,
                    {"path": path, "planes": planes, "cleaning": reason})

    for stem in labels:
        if stem not in seen:
            dataset.drop("label with no file")
    log.info("loaded %d stamps from %s (%s)", len(dataset), directory,
             ", ".join(f"{v} {k}" for k, v in dataset.dropped.items()) or "nothing dropped")
    return dataset


def load_alert_stamps(directory: str, labels: Dict[str, Tuple[str, float]],
                      class_map: Optional[Dict[str, ObjectClass]] = None,
                      channel: str = "difference",
                      max_bad_fraction: float = MAX_BAD_FRACTION) -> StampSet:
    """Load alert-broker stamp triplets.

    A transient alert carries three stamps: the science image, a reference,
    and their difference. Which one to train on is a real choice, and the
    default here is the difference, because that is the image the real/bogus
    decision is actually about -- the science stamp is dominated by the host
    galaxy, which is a property of where the transient is rather than of
    whether it is real.

    Files are expected as ``<id>_<channel>.npy`` or ``<id>_<channel>.fits``,
    which is the flattened form these are usually unpacked into.
    """
    mapping = dict(class_map or {})
    dataset = StampSet(source=f"alerts:{directory}:{channel}")
    fits = try_import("astropy.io.fits")

    for identifier, (name, weight) in sorted(labels.items()):
        label = mapping.get(name)
        if label is None:
            try:
                label = ObjectClass(name)
            except ValueError:
                dataset.drop(f"unknown class {name!r}")
                continue
        data = None
        for extension in (".npy", ".fits"):
            path = os.path.join(directory, f"{identifier}_{channel}{extension}")
            if not os.path.exists(path):
                continue
            try:
                if extension == ".npy":
                    data = np.load(path)
                elif fits is not None:
                    with fits.open(path, memmap=False) as handle:
                        data = np.asarray(handle[0].data, dtype=float)
            except Exception as error:                  # pragma: no cover
                log.warning("could not read %s: %s", path, error)
            break
        if data is None:
            dataset.drop(f"no {channel} stamp")
            continue
        cleaned, reason = clean_stamp(np.asarray(data, dtype=float), max_bad_fraction)
        if cleaned is None:
            dataset.drop(reason)
            continue
        dataset.add(cleaned, label, weight, identifier,
                    {"channel": channel, "cleaning": reason})
    return dataset


def write_fits_cutouts(directory: str, dataset: StampSet,
                       labels_path: Optional[str] = None) -> str:
    """Write a stamp set out in the directory-of-cutouts form.

    Used to exercise the loaders against files rather than against arrays,
    which is the only way to test that the reading path works at all when no
    real archive is reachable.
    """
    fits = try_import("astropy.io.fits")
    if fits is None:                                    # pragma: no cover
        raise ImportError("astropy is required to write FITS cutouts")
    os.makedirs(directory, exist_ok=True)
    for i, stamp in enumerate(dataset.stamps):
        identifier = dataset.ids[i] or f"obj{i:05d}"
        fits.PrimaryHDU(np.asarray(stamp, dtype=np.float32)).writeto(
            os.path.join(directory, f"{identifier}.fits"), overwrite=True)
    if labels_path:
        with open(labels_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "class", "agreement"])
            for i, label in enumerate(dataset.labels):
                writer.writerow([dataset.ids[i] or f"obj{i:05d}",
                                 label.value if isinstance(label, ObjectClass) else label,
                                 f"{dataset.weights[i]:.3f}"])
    return directory


#: Simulator truth kinds that map onto trainable classes.
SIMULATED_CLASSES: Dict[str, ObjectClass] = {
    "star": ObjectClass.STAR,
    "galaxy": ObjectClass.GALAXY,
    "nebula": ObjectClass.NEBULA,
    "cluster": ObjectClass.STAR_CLUSTER,
}


def stamps_from_fields(config_factory, seeds: Sequence[int], cutout: int = 48,
                       min_flux: float = 1500.0, margin: int = 24,
                       classes: Optional[Dict[str, ObjectClass]] = None,
                       source: str = "simulated") -> StampSet:
    """Cut labelled stamps out of simulated fields.

    ``config_factory`` takes a seed and returns a :class:`SkyConfig`, which is
    what lets a caller define two *instruments* -- different seeing, pixel
    scale and depth -- and get comparable stamp sets from each.
    """
    from ..preprocess import Preprocessor
    from ..simulate import SkySimulator

    mapping = dict(classes or SIMULATED_CLASSES)
    dataset = StampSet(source=source)
    preprocessor = Preprocessor()
    for seed in seeds:
        config = config_factory(seed)
        image, truth = SkySimulator(config).generate()
        clean = preprocessor.run(image, estimate_psf=False)
        height, width = clean.data.shape
        for entry in truth:
            label = mapping.get(entry.kind)
            if label is None:
                continue
            if entry.flux < float(min_flux):
                dataset.drop("too faint")
                continue
            if not (margin < entry.x < width - margin
                    and margin < entry.y < height - margin):
                dataset.drop("too close to the edge")
                continue
            stamp = clean.cutout(entry.x, entry.y, cutout, subtract_background=True)
            cleaned, reason = clean_stamp(stamp)
            if cleaned is None:
                dataset.drop(reason)
                continue
            dataset.add(cleaned, label, 1.0, f"{seed}_{entry.id}",
                        {"kind": entry.kind, "flux": float(entry.flux),
                         "seed": int(seed)})
    return dataset


def split_dataset(dataset: StampSet, fractions: Sequence[float] = (0.7, 0.15, 0.15),
                  seed: int = 0, by_group: Optional[str] = None
                  ) -> List[StampSet]:
    """Split a stamp set, keeping the class balance and optionally grouping.

    ``by_group`` names a metadata key -- ``"seed"``, usually -- whose values
    must not be split across parts. Stamps from one field share its noise
    realisation, its PSF and its background, so putting some in training and
    the rest in test measures how well the model memorised that field. The
    grouped split is the honest one and reports a lower number.
    """
    rng = np.random.default_rng(int(seed))
    weights = np.asarray(fractions, dtype=float)
    weights = weights / weights.sum()

    if by_group:
        groups: Dict[Any, List[int]] = {}
        for i, meta in enumerate(dataset.meta):
            groups.setdefault(meta.get(by_group), []).append(i)
        keys = list(groups)
        rng.shuffle(keys)
        edges = np.cumsum(weights)[:-1] * len(keys)
        chunks = np.split(np.arange(len(keys)), [int(round(e)) for e in edges])
        return [dataset.subset([i for k in chunk for i in groups[keys[k]]])
                for chunk in chunks]

    parts: List[List[int]] = [[] for _ in weights]
    by_class: Dict[str, List[int]] = {}
    for i, label in enumerate(dataset.labels):
        key = label.value if isinstance(label, ObjectClass) else str(label)
        by_class.setdefault(key, []).append(i)
    for index in by_class.values():
        order = np.array(index)
        rng.shuffle(order)
        edges = np.cumsum(weights)[:-1] * len(order)
        for part, chunk in zip(parts, np.split(order, [int(round(e)) for e in edges])):
            part.extend(int(i) for i in chunk)
    return [dataset.subset(sorted(part)) for part in parts]


def class_balance_report(dataset: StampSet) -> Dict[str, Any]:
    """How imbalanced a set is, and what that costs.

    The imbalance ratio is the plain statement of the problem: at 20:1 a model
    that never predicts the rare class is already 95 % accurate, so accuracy
    stops being a useful number and per-class recall takes over.
    """
    counts = dataset.counts()
    if not counts:
        return {"n": 0, "counts": {}, "imbalance": float("nan"),
                "majority_accuracy": float("nan")}
    values = np.array(list(counts.values()), dtype=float)
    total = float(values.sum())
    return {"n": int(total), "counts": counts,
            "imbalance": float(values.max() / max(values.min(), 1.0)),
            "majority_class": max(counts, key=lambda k: counts[k]),
            "majority_accuracy": float(values.max() / total),
            "note": ("accuracy above the majority-class fraction is the only "
                     "part of it that means anything")}
