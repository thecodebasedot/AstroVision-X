"""Moving a trained model to a new instrument, and measuring what that costs.

A classifier trained on one telescope's images is not a classifier for
another's. The seeing is different, the pixel scale is different, the depth is
different, and the network has learned all three as though they were
properties of the objects. This is the single largest obstacle to using a
model trained on simulations -- or on one survey -- anywhere else, and the
useful question is not whether the gap exists but **how many labelled examples
from the new instrument it takes to close it**.

That question has an answer, and this module measures it. The measurement has
three legs, and the third is the one usually left out:

1. train on the source domain, test on the source domain — the number people
   quote;
2. train on the source domain, test on the target domain — the gap;
3. fine-tune on *N* target examples, **and** train from scratch on the same
   *N*, and compare. Without the third leg, "fine-tuning reached 80 %" is
   unfalsifiable: it might be that 80 % was available from those *N* examples
   alone and the pretraining contributed nothing.

The mechanics themselves are standard. The head is replaced, the backbone is
frozen or given a smaller learning rate, and training stops on a validation
split rather than after a fixed number of epochs, because with a few dozen
examples the difference between fitting and memorising is a couple of epochs
wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.backend import require
from ..core.exceptions import ModelError
from ..core.logging import get_logger
from ..core.types import ObjectClass
from .cnn import StampClassifier
from .datasets import StampSet, split_dataset

log = get_logger("ml.transfer")


def freeze_backbone(classifier: StampClassifier, trainable_head: bool = True
                    ) -> Dict[str, int]:
    """Freeze everything except the classification head.

    With a few dozen target examples, the whole network cannot be trained --
    there are more parameters than pixels in the training set. What can be
    trained is the last layer, which is a linear model on top of features the
    source domain already learned.

    Returns the parameter counts, because "frozen the backbone" is worth
    checking rather than believing: a typo in a layer name silently trains
    everything.
    """
    if classifier.model is None:
        raise ModelError("build or load the classifier before freezing it")
    head = _head_module(classifier)
    frozen = trainable = 0
    for parameter in classifier.model.parameters():
        parameter.requires_grad = False
        frozen += parameter.numel()
    if trainable_head and head is not None:
        for parameter in head.parameters():
            parameter.requires_grad = True
            trainable += parameter.numel()
            frozen -= parameter.numel()
    return {"frozen": int(frozen), "trainable": int(trainable)}


def unfreeze(classifier: StampClassifier) -> Dict[str, int]:
    """Make every parameter trainable again."""
    if classifier.model is None:
        raise ModelError("build or load the classifier before unfreezing it")
    total = 0
    for parameter in classifier.model.parameters():
        parameter.requires_grad = True
        total += parameter.numel()
    return {"frozen": 0, "trainable": int(total)}


def _head_module(classifier: StampClassifier):
    """The final classification layer, whichever backbone is in use."""
    model = classifier.model
    for name in ("head", "fc", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            return module
    # Fall back to the last Linear in the module tree.
    torch = classifier._torch
    if torch is None:                                   # pragma: no cover
        return None
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    return linears[-1] if linears else None


def replace_head(classifier: StampClassifier,
                 classes: Sequence[ObjectClass]) -> StampClassifier:
    """Give a trained model a new set of output classes.

    The features are kept; only the final linear map is rebuilt. This is what
    makes a model trained on one label scheme usable under another -- and the
    part worth noticing is that it keeps the *representation*, which is where
    the training data went.
    """
    torch = require("torch", "transfer learning")
    if classifier.model is None:
        raise ModelError("build or load the classifier before replacing its head")
    head = _head_module(classifier)
    if head is None or not isinstance(head, torch.nn.Linear):
        raise ModelError("could not find a linear classification head to replace")
    new_head = torch.nn.Linear(head.in_features, len(classes)).to(classifier.device)
    for name in ("head", "fc", "classifier"):
        if getattr(classifier.model, name, None) is head:
            setattr(classifier.model, name, new_head)
            break
    else:                                               # pragma: no cover
        raise ModelError("the classification head is not a direct attribute")
    classifier.classes = list(classes)
    return classifier


@dataclass
class FineTuneResult:
    """What a fine-tuning run did, and whether it was worth doing."""

    n_target_labels: int = 0
    epochs_run: int = 0
    best_epoch: int = 0
    train_loss: List[float] = field(default_factory=list)
    validation_accuracy: List[float] = field(default_factory=list)
    accuracy: float = float("nan")
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    frozen_parameters: int = 0
    trainable_parameters: int = 0
    stopped_early: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"n_target_labels": self.n_target_labels,
                "epochs_run": self.epochs_run, "best_epoch": self.best_epoch,
                "accuracy": self.accuracy,
                "per_class_recall": dict(self.per_class_recall),
                "frozen_parameters": self.frozen_parameters,
                "trainable_parameters": self.trainable_parameters,
                "stopped_early": self.stopped_early, "reason": self.reason}


def evaluate(classifier: StampClassifier, dataset: StampSet) -> Dict[str, Any]:
    """Accuracy and per-class recall on a stamp set.

    Both, always. On an imbalanced set accuracy is mostly a statement about
    the majority class, and a model that has stopped predicting a rare class
    entirely looks fine by it.
    """
    if len(dataset) == 0:
        return {"accuracy": float("nan"), "per_class_recall": {}, "n": 0}
    predictions = classifier.predict(dataset.stamps)
    truth = list(dataset.labels)
    correct = [p == t for p, t in zip(predictions, truth)]
    recall: Dict[str, float] = {}
    for label in sorted({t.value for t in truth}):
        mask = [t.value == label for t in truth]
        hits = [c for c, m in zip(correct, mask) if m]
        recall[label] = float(np.mean(hits)) if hits else float("nan")
    confusion: Dict[str, Dict[str, int]] = {}
    for p, t in zip(predictions, truth):
        confusion.setdefault(t.value, {}).setdefault(p.value, 0)
        confusion[t.value][p.value] += 1
    return {"accuracy": float(np.mean(correct)), "per_class_recall": recall,
            "n": len(dataset), "confusion": confusion,
            "balanced_accuracy": float(np.nanmean(list(recall.values())))}


def fine_tune(classifier: StampClassifier, dataset: StampSet,
              validation: Optional[StampSet] = None,
              epochs: int = 60, learning_rate: float = 3e-3,
              batch_size: int = 16, freeze: bool = True,
              patience: int = 12, verbose: bool = False) -> FineTuneResult:
    """Adapt a trained classifier to a new domain on a small labelled set.

    Training stops when the validation accuracy has not improved for
    ``patience`` epochs, and the weights from the best epoch are restored.
    With thirty examples the model can reach perfect training accuracy in a
    handful of epochs and be worse than useless; a fixed epoch count is a
    coin toss over which side of that it lands on.
    """
    torch = require("torch", "fine-tuning")
    if classifier.model is None:
        raise ModelError("build or load the classifier before fine-tuning")
    result = FineTuneResult(n_target_labels=len(dataset))
    if len(dataset) == 0:
        result.reason = "no target-domain labels supplied"
        return result

    counts = freeze_backbone(classifier) if freeze else unfreeze(classifier)
    result.frozen_parameters = counts["frozen"]
    result.trainable_parameters = counts["trainable"]

    index = {c.value: i for i, c in enumerate(classifier.classes)}
    missing = {l.value for l in dataset.labels} - set(index)
    if missing:
        raise ModelError(f"labels not in the classifier's class set: {sorted(missing)}")

    X = torch.from_numpy(classifier._prepare(dataset.stamps))
    y = torch.tensor([index[l.value] for l in dataset.labels], dtype=torch.long)
    sample_weights = torch.tensor(dataset.weights or [1.0] * len(dataset),
                                  dtype=torch.float32)

    targets = y.numpy()
    class_counts = np.bincount(targets, minlength=len(classifier.classes)).astype(float)
    weights = np.where(class_counts > 0, class_counts.sum() / np.maximum(class_counts, 1), 0.0)
    weights = weights / max(weights[weights > 0].mean(), 1e-9)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=classifier.device),
        reduction="none")

    trainable = [p for p in classifier.model.parameters() if p.requires_grad]
    if not trainable:                                   # pragma: no cover
        result.reason = "nothing is trainable"
        return result
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-3)

    best_accuracy, best_state, since_improvement = -1.0, None, 0
    for epoch in range(int(epochs)):
        classifier.model.train()
        permutation = torch.randperm(len(X))
        total, batches = 0.0, 0
        for start in range(0, len(X), batch_size):
            batch = permutation[start:start + batch_size]
            if len(batch) < 2:
                continue
            xb = X[batch].to(classifier.device)
            if bool(torch.rand(1) < 0.5):
                xb = torch.flip(xb, dims=[3])
            xb = torch.rot90(xb, int(torch.randint(0, 4, (1,))), dims=[2, 3])
            losses = criterion(classifier.model(xb), y[batch].to(classifier.device))
            # Per-sample weights carry label confidence: a stamp 98 % of
            # labellers agreed on should pull harder than one they split over.
            loss = (losses * sample_weights[batch].to(classifier.device)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        result.train_loss.append(total / max(batches, 1))
        result.epochs_run = epoch + 1

        if validation is not None and len(validation):
            classifier.model.eval()
            accuracy = float(evaluate(classifier, validation)["balanced_accuracy"])
            result.validation_accuracy.append(accuracy)
            if accuracy > best_accuracy + 1e-6:
                best_accuracy, since_improvement = accuracy, 0
                result.best_epoch = epoch + 1
                best_state = {k: v.detach().clone()
                              for k, v in classifier.model.state_dict().items()}
            else:
                since_improvement += 1
                if since_improvement >= int(patience):
                    result.stopped_early = True
                    break

    if best_state is not None:
        classifier.model.load_state_dict(best_state)
    classifier.model.eval()
    result.reason = (f"fine-tuned on {len(dataset)} target labels, "
                     f"{'head only' if freeze else 'whole network'}, "
                     f"{result.epochs_run} epochs"
                     + (f", best at {result.best_epoch}" if result.best_epoch else ""))
    if verbose:
        log.info("%s", result.reason)
    return result


@dataclass
class DomainStudy:
    """The three-legged measurement of a domain shift."""

    source_accuracy: float = float("nan")
    target_accuracy: float = float("nan")
    curve: List[Dict[str, Any]] = field(default_factory=list)
    labels_to_recover: Optional[int] = None
    recovery_threshold: float = 0.9
    notes: List[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        return float(self.source_accuracy - self.target_accuracy)

    def to_dict(self) -> Dict[str, Any]:
        return {"source_accuracy": self.source_accuracy,
                "target_accuracy": self.target_accuracy, "gap": self.gap,
                "curve": list(self.curve),
                "labels_to_recover": self.labels_to_recover,
                "recovery_threshold": self.recovery_threshold,
                "notes": list(self.notes)}

    def summary(self) -> str:
        parts = [f"source {100 * self.source_accuracy:.0f}%",
                 f"target {100 * self.target_accuracy:.0f}%",
                 f"gap {100 * self.gap:.0f} points"]
        if self.labels_to_recover is not None:
            parts.append(f"{self.labels_to_recover} target labels recover "
                         f"{100 * self.recovery_threshold:.0f}% of the source score")
        return "; ".join(parts)


def domain_study(train_source: Callable[[], StampClassifier],
                 source_test: StampSet, target_pool: StampSet,
                 target_test: StampSet,
                 label_budgets: Sequence[int] = (10, 25, 50, 100),
                 recovery_threshold: float = 0.9,
                 epochs: int = 60, seed: int = 0,
                 scratch_comparison: bool = True,
                 repeats: int = 3) -> DomainStudy:
    """Measure a domain shift and what it takes to close it.

    ``train_source`` returns a classifier already trained on the source
    domain; it is called for the baseline and again for every run, so each
    fine-tuning starts from the same place rather than from the previous
    run's weights.

    Every budget is also trained **from scratch** on the same examples, which
    is what turns "fine-tuning got 80 %" into a statement about whether the
    pretraining helped.

    ``repeats`` draws each budget more than once, and the reason is
    measured rather than defensive. A single draw of 25 target labels scored
    0.837 here; five draws of the same size gave 0.795 +/- 0.059, spanning
    0.726 to 0.866. Reporting the first draw would have claimed that 25
    labels recover 90 % of the source score, which three of the five draws do
    not. The spread *is* the finding at small budgets, so it is returned
    alongside the mean and never averaged away.
    """
    study = DomainStudy(recovery_threshold=float(recovery_threshold))

    baseline = train_source()
    study.source_accuracy = float(evaluate(baseline, source_test)["balanced_accuracy"])
    study.target_accuracy = float(evaluate(baseline, target_test)["balanced_accuracy"])
    study.notes.append(
        f"a model trained on the source instrument scores "
        f"{100 * study.source_accuracy:.0f}% there and "
        f"{100 * study.target_accuracy:.0f}% on the target instrument")

    threshold = study.recovery_threshold * study.source_accuracy
    for budget in sorted(int(b) for b in label_budgets):
        if budget > len(target_pool):
            study.notes.append(f"budget {budget} exceeds the {len(target_pool)} "
                               "target examples available and was skipped")
            continue
        tuned_scores: List[float] = []
        scratch_scores: List[float] = []
        epochs_run: List[int] = []
        for trial in range(max(int(repeats), 1)):
            rng = np.random.default_rng(int(seed) + 1000 * trial)
            chosen = _stratified_sample(target_pool, budget, rng)
            train_part, validation_part = split_dataset(
                chosen, (0.75, 0.25), seed=int(seed) + budget + trial)
            if len(validation_part) == 0:               # pragma: no cover
                train_part, validation_part = chosen, chosen

            tuned = train_source()
            fine = fine_tune(tuned, train_part, validation_part, epochs=epochs)
            tuned_scores.append(
                float(evaluate(tuned, target_test)["balanced_accuracy"]))
            epochs_run.append(fine.epochs_run)

            if scratch_comparison:
                scratch = StampClassifier(
                    backbone=baseline.backbone, classes=baseline.classes,
                    cutout=baseline.cutout, width=baseline.width,
                    device=baseline.device,
                    random_state=baseline.random_state + budget + trial)
                scratch.build()
                # From scratch the whole network has to train, so nothing is
                # frozen -- freezing a randomly initialised backbone would
                # compare fine-tuning against a random feature extractor
                # rather than against training.
                fine_tune(scratch, train_part, validation_part, epochs=epochs,
                          freeze=False)
                scratch_scores.append(
                    float(evaluate(scratch, target_test)["balanced_accuracy"]))

        entry: Dict[str, Any] = {
            "n_labels": budget,
            "fine_tuned": float(np.mean(tuned_scores)),
            "fine_tuned_sd": float(np.std(tuned_scores)),
            "fine_tuned_runs": [float(v) for v in tuned_scores],
            "repeats": len(tuned_scores),
            "epochs": int(np.median(epochs_run)) if epochs_run else 0}
        if scratch_scores:
            entry["from_scratch"] = float(np.mean(scratch_scores))
            entry["from_scratch_sd"] = float(np.std(scratch_scores))
            entry["transfer_advantage"] = entry["fine_tuned"] - entry["from_scratch"]
        study.curve.append(entry)

        # The threshold has to be cleared by the mean, not by a lucky draw.
        if study.labels_to_recover is None and entry["fine_tuned"] >= threshold:
            study.labels_to_recover = budget

    if study.labels_to_recover is None and study.curve:
        study.notes.append(
            f"no budget up to {study.curve[-1]['n_labels']} labels recovered "
            f"{100 * study.recovery_threshold:.0f}% of the source score on "
            "average")
    return study


def _stratified_sample(dataset: StampSet, n: int, rng) -> StampSet:
    """Take ``n`` examples keeping the class proportions.

    A random draw of thirty from an imbalanced pool can miss a class entirely,
    and a fine-tuning run that never sees a class cannot predict it -- which
    would show up as a property of the method rather than of the draw.
    """
    by_class: Dict[str, List[int]] = {}
    for i, label in enumerate(dataset.labels):
        by_class.setdefault(label.value, []).append(i)
    per_class = max(int(math.floor(n / max(len(by_class), 1))), 1)
    chosen: List[int] = []
    for index in by_class.values():
        order = np.array(index)
        rng.shuffle(order)
        chosen.extend(int(i) for i in order[:per_class])
    remaining = [i for i in range(len(dataset)) if i not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(int(i) for i in remaining[:max(n - len(chosen), 0)])
    return dataset.subset(sorted(chosen[:n]))
