"""Turning an astronomer's decisions back into training data.

Everything in this package that scores an object produces a *recommendation*.
A person then looks at the ranked list and decides. That decision is the most
expensive data the project will ever have -- an expert's judgement on a
specific object -- and until now it was thrown away the moment the screen was
closed. This module keeps it, feeds it back as a label, and measures whether
choosing *which* objects to show makes the labelling effort go further.

Two rules shape the design, and both are about not lying to ourselves.

**A model's verdict is not a label.** ``Verdict`` on a candidate is what the
pipeline recommended; ``HumanVerdict`` here is what a person concluded. They
are separate types on purpose. Feeding the first back in as training data is
self-training: the model's errors get relabelled as truth, confidently, and
the next model is more certain about them. :func:`verdicts_to_labels` refuses
records without a named reviewer for exactly that reason.

**Selection strategy must be measured, not assumed -- and here it did not
work.** Uncertainty sampling, showing the astronomer whatever the model is
least sure about, is the textbook answer. Measured over six repeats at
budgets from 20 to 100 labels, it lost to plain random selection at three of
the four budgets and won at one, with spreads that overlap everywhere.

The class composition says why, and the mechanism is the useful part: the
decision boundary is crowded with faint *stars*, which are the majority class
and individually uninformative, so uncertainty sampling spent 58 of its 100
labels on them against random selection's 42, while galaxies fell from 32 to
22. It bought more of what there was already plenty of.

Quotas per predicted class were the obvious fix and they also failed --
+0.04, -0.03, -0.06, +0.01 against random -- because the quota is on what the
model *predicts*, and early on the model predicts the majority class for
almost everything, so the quota rebalances nothing.

So the default is random selection: it measured at least as well as everything
tried against it, it is unbiased about the distribution the model will meet,
and it is simpler. The other strategies stay, with their numbers, because this
is one problem at one set of budgets and the question deserves reopening on a
harder one. The full table is in `docs/validation.md`.
"""

from __future__ import annotations

import json
import os
import time
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import ObjectClass, Verdict
from .datasets import StampSet

log = get_logger("ml.active")


@dataclass
class HumanVerdict:
    """One decision by one person about one object.

    ``reviewer`` is required and is not decoration: it is what distinguishes
    a label from a model output, and it is what makes disagreement between
    reviewers visible later instead of silently averaged.
    """

    source_id: int
    label: str                              # what the reviewer concluded it is
    reviewer: str
    confident: bool = True                  # False for "I am not sure"
    model_label: str = ""                   # what the model had said
    model_confidence: float = float("nan")
    model_verdict: str = ""                 # the pipeline's recommendation
    note: str = ""
    timestamp: float = field(default_factory=time.time)

    def agrees_with_model(self) -> bool:
        return bool(self.model_label) and self.label == self.model_label

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerdictLog:
    """An append-only record of human decisions.

    Append-only because a review is a historical fact: an astronomer looked at
    this object on this date and concluded this. Overwriting it when they
    change their mind loses the disagreement, which is data about how hard the
    object is.
    """

    def __init__(self, records: Optional[Sequence[HumanVerdict]] = None):
        self.records: List[HumanVerdict] = list(records or [])

    def __len__(self) -> int:
        return len(self.records)

    def add(self, verdict: HumanVerdict) -> "VerdictLog":
        if not str(verdict.reviewer).strip():
            raise ValueError(
                "a verdict needs a reviewer; an unattributed decision cannot be "
                "distinguished from the model's own output, and training on "
                "that is self-training")
        self.records.append(verdict)
        return self

    def latest(self) -> Dict[int, HumanVerdict]:
        """The most recent verdict per source, whoever gave it."""
        newest: Dict[int, HumanVerdict] = {}
        for record in sorted(self.records, key=lambda r: r.timestamp):
            newest[int(record.source_id)] = record
        return newest

    def disagreements(self) -> List[Tuple[int, List[str]]]:
        """Objects two reviewers labelled differently.

        Worth surfacing rather than resolving: an object experts disagree on is
        either genuinely ambiguous or badly presented, and both are more useful
        to know than a majority vote would be.
        """
        by_source: Dict[int, List[str]] = {}
        for record in self.records:
            by_source.setdefault(int(record.source_id), []).append(record.label)
        return [(source, labels) for source, labels in sorted(by_source.items())
                if len(set(labels)) > 1]

    def agreement_with_model(self) -> Dict[str, Any]:
        """How often the reviewers confirmed the model, and where they did not.

        The confusion here is the most valuable diagnostic the system produces,
        because it is measured on real decisions rather than on a held-out
        split of the training distribution.
        """
        scored = [r for r in self.records if r.model_label]
        if not scored:
            return {"n": 0, "agreement": float("nan"), "confusion": {}}
        confusion: Dict[str, Dict[str, int]] = {}
        for record in scored:
            confusion.setdefault(record.model_label, {}).setdefault(record.label, 0)
            confusion[record.model_label][record.label] += 1
        agreed = [r.agrees_with_model() for r in scored]
        overconfident = [r for r in scored
                         if not r.agrees_with_model()
                         and np.isfinite(r.model_confidence)
                         and r.model_confidence > 0.9]
        return {"n": len(scored), "agreement": float(np.mean(agreed)),
                "confusion": confusion,
                "confidently_wrong": len(overconfident),
                "note": ("objects the model called with over 0.9 confidence and "
                         "a reviewer overruled are where a calibration problem "
                         "shows up first")}

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([r.to_dict() for r in self.records], handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "VerdictLog":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls([HumanVerdict(**row) for row in payload])


def verdicts_to_labels(log: VerdictLog, dataset: StampSet,
                       confident_only: bool = True,
                       classes: Optional[Sequence[ObjectClass]] = None
                       ) -> StampSet:
    """Build a training set from what people actually decided.

    Matches verdicts to stamps by source id. A verdict marked ``confident=False``
    is dropped by default: "I am not sure" is a real and useful answer, and
    training on it teaches the model the reviewer's uncertainty as though it
    were a class.
    """
    allowed = {c.value for c in classes} if classes else None
    newest = log.latest()
    labelled = StampSet(source=f"verdicts:{len(log)}")
    for i, identifier in enumerate(dataset.ids):
        try:
            key = int(str(identifier).split("_")[-1])
        except ValueError:
            continue
        record = newest.get(key)
        if record is None:
            continue
        if confident_only and not record.confident:
            labelled.drop("reviewer was not sure")
            continue
        if allowed is not None and record.label not in allowed:
            labelled.drop(f"label {record.label!r} outside the class set")
            continue
        try:
            label = ObjectClass(record.label)
        except ValueError:
            labelled.drop(f"unknown class {record.label!r}")
            continue
        labelled.add(dataset.stamps[i], label, 1.0, str(identifier),
                     {**dataset.meta[i], "reviewer": record.reviewer})
    return labelled


# -- choosing what to show -------------------------------------------------

def uncertainty_scores(probabilities: np.ndarray, method: str = "margin"
                       ) -> np.ndarray:
    """How unsure the model is about each object, higher meaning less sure.

    ``margin`` -- the gap between the best and second-best class -- is the
    default because it asks the question a reviewer can settle. Entropy is
    dominated by how the *tail* of the distribution is spread, which is a
    property of the softmax rather than of the decision; least-confidence
    ignores the runner-up entirely and so cannot tell a genuine two-way tie
    from a diffuse guess.

    >>> import numpy as np
    >>> p = np.array([[0.5, 0.5], [0.99, 0.01]])
    >>> scores = uncertainty_scores(p)
    >>> bool(scores[0] > scores[1])
    True
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be (n_samples, n_classes >= 2)")
    if method == "entropy":
        safe = np.clip(values, 1e-12, 1.0)
        return -np.sum(safe * np.log(safe), axis=1)
    if method == "least_confident":
        return 1.0 - values.max(axis=1)
    ordered = np.sort(values, axis=1)
    return 1.0 - (ordered[:, -1] - ordered[:, -2])


def select_for_review(probabilities: np.ndarray, n: int,
                      strategy: str = "random",
                      embeddings: Optional[np.ndarray] = None,
                      seed: int = 0, method: str = "margin") -> np.ndarray:
    """Choose which objects to put in front of a person.

    ``random`` is the default because it won. It is not a straw man: it samples
    the distribution the model will actually meet, while uncertainty sampling
    deliberately does not, and over six repeats nothing here beat it. See the
    module docstring for the numbers and the reason.

    ``diverse`` picks uncertain objects that are also far apart in the
    embedding, which addresses the failure where a batch of twenty questions
    turns out to be twenty views of the same confusion.

    ``balanced`` takes the most uncertain objects *within each predicted
    class*, in equal quota. It exists because of what the measurement found:
    plain uncertainty sampling spent its budget on the majority class -- 58
    stars against random selection's 42, with galaxies falling from 32 to 22 --
    since the decision boundary is crowded with faint stars, which are numerous
    and uninformative. Quotas per predicted class are the direct fix for that.
    """
    values = np.asarray(probabilities, dtype=float)
    count = int(min(max(n, 0), len(values)))
    rng = np.random.default_rng(int(seed))
    if count == 0:
        return np.zeros(0, dtype=int)

    if strategy == "random":
        return np.sort(rng.choice(len(values), size=count, replace=False))
    if strategy == "confident":
        # Deliberately available: showing a reviewer what the model is *sure*
        # about is how a systematic error gets caught, and is a different job
        # from making the model better.
        return np.sort(np.argsort(values.max(axis=1))[::-1][:count])

    scores = uncertainty_scores(values, method=method)

    if strategy == "balanced":
        predicted = np.argmax(values, axis=1)
        quota = int(math.ceil(count / max(values.shape[1], 1)))
        chosen: List[int] = []
        for label in range(values.shape[1]):
            members = np.flatnonzero(predicted == label)
            if members.size == 0:
                continue
            order = members[np.argsort(scores[members])[::-1]]
            chosen.extend(int(i) for i in order[:quota])
        # Any shortfall -- a class the model predicted for nobody -- is filled
        # from the overall uncertainty ranking rather than left unspent.
        if len(chosen) < count:
            for i in np.argsort(scores)[::-1]:
                if int(i) not in set(chosen):
                    chosen.append(int(i))
                if len(chosen) >= count:
                    break
        return np.sort(np.asarray(chosen[:count], dtype=int))

    if strategy == "uncertainty" or embeddings is None:
        return np.sort(np.argsort(scores)[::-1][:count])

    if strategy != "diverse":
        raise ValueError(f"unknown strategy {strategy!r}")

    # Greedy: take the most uncertain, then repeatedly take the most uncertain
    # object that is far from everything already chosen.
    matrix = np.asarray(embeddings, dtype=float)
    if len(matrix) != len(values):
        raise ValueError("one embedding per candidate is required")
    order = list(np.argsort(scores)[::-1])
    chosen = [int(order[0])]
    while len(chosen) < count and len(chosen) < len(order):
        distances = np.min(
            np.linalg.norm(matrix[:, None, :] - matrix[None, chosen, :], axis=2),
            axis=1)
        # Normalise both terms so neither dominates by units alone.
        combined = (_unit(scores) + _unit(distances)) / 2.0
        combined[chosen] = -np.inf
        chosen.append(int(np.argmax(combined)))
    return np.sort(np.asarray(chosen, dtype=int))


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def review_queue(catalog, probabilities: np.ndarray, classes: Sequence[ObjectClass],
                 n: int = 20, strategy: str = "random",
                 embeddings: Optional[np.ndarray] = None,
                 seed: int = 0) -> List[Dict[str, Any]]:
    """The list a person is actually shown, with the model's reasoning attached.

    Each entry carries what the model thinks and how sure it is, so the
    reviewer is judging a claim rather than a bare picture -- and so the
    verdict that comes back can be compared against what was claimed.
    """
    sources = list(catalog)
    picked = select_for_review(probabilities, n, strategy=strategy,
                               embeddings=embeddings, seed=seed)
    scores = uncertainty_scores(np.asarray(probabilities, dtype=float))
    queue: List[Dict[str, Any]] = []
    for index in picked:
        row = np.asarray(probabilities[int(index)], dtype=float)
        best = int(np.argmax(row))
        source = sources[int(index)]
        queue.append({
            "source_id": int(getattr(source, "id", index)),
            "index": int(index),
            "model_label": classes[best].value,
            "model_confidence": float(row[best]),
            "runner_up": classes[int(np.argsort(row)[-2])].value,
            "uncertainty": float(scores[int(index)]),
            "model_verdict": str(getattr(source, "verdict", Verdict.WORTH_A_LOOK).value
                                 if hasattr(source, "verdict") else ""),
        })
    return queue


# -- measuring whether any of it helps -------------------------------------

@dataclass
class ActiveLearningRun:
    """One strategy's learning curve."""

    strategy: str = ""
    budgets: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    spread: List[float] = field(default_factory=list)
    class_counts: List[Dict[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "budgets": list(self.budgets),
                "scores": list(self.scores), "spread": list(self.spread),
                "class_counts": list(self.class_counts)}


def run_active_learning(pool: StampSet, test: StampSet,
                        classes: Sequence[ObjectClass],
                        strategy: str = "uncertainty",
                        seed_size: int = 20, rounds: int = 4,
                        batch: int = 20, epochs: int = 30,
                        repeats: int = 2, seed: int = 0,
                        width: int = 16, cutout: int = 48) -> ActiveLearningRun:
    """Simulate the loop: train, select, label, retrain.

    The pool's true labels stand in for the astronomer, which is the only way
    to measure a strategy without spending a person's time -- and it is worth
    naming as an idealisation, because a real reviewer is slower on hard
    objects, sometimes wrong, and sometimes says "not sure". The oracle here is
    instant, always right, and always decisive, so these curves are the best
    case for every strategy alike.
    """
    from .cnn import StampClassifier
    from .transfer import _stratified_sample, evaluate

    run = ActiveLearningRun(strategy=strategy)
    budgets = [seed_size + i * batch for i in range(int(rounds) + 1)]
    per_budget: Dict[int, List[float]] = {b: [] for b in budgets}
    per_counts: Dict[int, Dict[str, int]] = {b: {} for b in budgets}

    for trial in range(max(int(repeats), 1)):
        rng = np.random.default_rng(int(seed) + 97 * trial)
        # The seed set is drawn at random and stratified, identically for every
        # strategy at a given trial: no strategy can select from a model that
        # does not exist yet, and different seed sets would measure the seed
        # set rather than the strategy.
        chosen = _indices_of(pool, _stratified_sample(pool, seed_size, rng))
        classifier = None

        for step, budget in enumerate(budgets):
            if step > 0:
                remaining = [i for i in range(len(pool)) if i not in set(chosen)]
                if not remaining or classifier is None:
                    break
                # The model doing the selecting is the one trained on what has
                # been labelled so far -- which is the model fitted at the end
                # of the previous round, not a fresh one.
                probabilities = classifier.predict_proba(
                    [pool.stamps[i] for i in remaining])
                embeddings = (classifier.embed([pool.stamps[i] for i in remaining])
                              if strategy == "diverse" else None)
                picked = select_for_review(probabilities, batch, strategy=strategy,
                                           embeddings=embeddings,
                                           seed=int(seed) + 13 * step + trial)
                chosen = chosen + [remaining[int(i)] for i in picked]

            classifier = _fit(StampClassifier, pool, chosen, classes, epochs,
                              width, cutout, int(seed) + trial)
            per_budget[budget].append(
                float(evaluate(classifier, test)["balanced_accuracy"]))
            counts: Dict[str, int] = {}
            for i in chosen:
                key = pool.labels[i].value
                counts[key] = counts.get(key, 0) + 1
            per_counts[budget] = counts

    for budget in budgets:
        values = per_budget[budget]
        if not values:
            continue
        run.budgets.append(budget)
        run.scores.append(float(np.mean(values)))
        run.spread.append(float(np.std(values)))
        run.class_counts.append(per_counts[budget])
    return run


def _indices_of(pool: StampSet, subset: StampSet) -> List[int]:
    """Positions in ``pool`` of the members of ``subset``, matched by id."""
    lookup = {identifier: i for i, identifier in enumerate(pool.ids)}
    return [lookup[identifier] for identifier in subset.ids if identifier in lookup]


def _fit(factory, pool: StampSet, index: Sequence[int],
         classes: Sequence[ObjectClass], epochs: int, width: int, cutout: int,
         seed: int):
    classifier = factory(backbone="cnn", classes=list(classes), cutout=cutout,
                         width=width, random_state=int(seed))
    classifier.fit([pool.stamps[i] for i in index],
                   [pool.labels[i] for i in index], epochs=epochs,
                   batch_size=16, verbose=False)
    return classifier


def compare_strategies(pool: StampSet, test: StampSet,
                       classes: Sequence[ObjectClass],
                       strategies: Sequence[str] = ("random", "uncertainty"),
                       **kwargs) -> Dict[str, Any]:
    """Run several strategies over the same budgets and compare them.

    The comparison that matters is at *equal labelling effort*: an astronomer's
    hour is the resource being spent, so a strategy is only better if it gets
    further on the same number of decisions.
    """
    runs = {name: run_active_learning(pool, test, classes, strategy=name, **kwargs)
            for name in strategies}
    baseline = runs.get("random")
    summary: Dict[str, Any] = {"runs": {k: v.to_dict() for k, v in runs.items()}}
    if baseline is not None:
        summary["advantage"] = {
            name: [round(a - b, 4) for a, b in zip(run.scores, baseline.scores)]
            for name, run in runs.items() if name != "random"}
    return summary
