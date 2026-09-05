"""Learning a representation from unlabelled cutouts.

Labels are the scarce resource in astronomy. A survey produces millions of
cutouts a night and a few thousand of them will ever be looked at by a person,
so a representation that can be learned *without* labels and then adapted with
a handful of them is worth more than a better classifier trained on the labels
that exist.

The method is contrastive, in the SimCLR family: take one stamp, make two
augmented views of it, and train the network so those two views land close
together in the embedding while views of *different* stamps land apart. No
label is involved anywhere. What the network is forced to learn is whatever
survives the augmentations -- which makes the choice of augmentation the whole
design, not a detail.

**Astronomy differs from natural images here, in a way that is easy to state
and turned out to be hard to demonstrate.** The standard SimCLR recipe uses a
random resized crop, which teaches the network that scale does not matter. In
a photograph that is correct: a cat is a cat near or far. In a survey cutout
angular size is *the* discriminator between a star and a galaxy, since a star
is by definition unresolved -- so a scale-changing crop appears to be teaching
the network to discard exactly the measurement that separates the two
commonest classes.

That argument is clean, and the measurement does not support it. Across three
seeds per policy, adding resized crops left star and galaxy recall unchanged
(0.784 +/- 0.016 against 0.790 +/- 0.012 without them) and slightly *improved*
overall balanced accuracy. The prediction was wrong, or at least too small to
see at this stamp size and crop range, and saying so is more useful than
quietly deleting the experiment.

The default remains no resized crop, on the physical argument rather than on
a measured advantage, and the switch stays so the question can be reopened
with more classes, larger stamps or a wider crop range. The full numbers are
in `docs/validation.md`.

What is safe is what a telescope could have done differently on a different
night to the *same* object:

* **rotation and reflection** -- the sky has no preferred orientation, so
  these are exact symmetries, not approximations;
* **sub-pixel and few-pixel translation** -- the object never lands on the
  same pixel twice;
* **noise** -- a shorter exposure of the same field;
* **PSF blur** -- worse seeing, which is real, though it must stay mild: blur
  a star enough and it becomes a galaxy, which is teaching the network a lie;
* **brightness scaling** -- distance and exposure time, and harmless because
  the per-stamp stretch already removes the overall scale.

Everything in that list changes the *observation* and leaves the *object*
alone. That is the test an augmentation has to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import require
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, gaussian_kernel, convolve
from ..core.types import ObjectClass
from .cnn import STAMP_CLASSES, StampClassifier, build_cnn
from .datasets import StampSet

log = get_logger("ml.selfsupervised")

#: Temperature of the contrastive loss.  Low values sharpen the distinction
#: between the positive pair and the negatives; 0.1-0.5 is the usual range and
#: the result is not sensitive inside it.
TEMPERATURE = 0.2


@dataclass
class AugmentationPolicy:
    """Which transformations count as "the same object, observed again".

    ``resized_crop`` is off by default. The reason is physical -- it destroys
    angular size, which is what distinguishes an unresolved star from a
    resolved galaxy -- and it is worth being clear that the measurement here
    did **not** find the cost that argument predicts: with crops the probe
    scored 0.776 +/- 0.014 against 0.755 +/- 0.015 without, and star/galaxy
    recall was unchanged. The switch stays so the question can be reopened.
    """

    rotate: bool = True                 # 90-degree turns: exact sky symmetry
    flip: bool = True
    translate: int = 3                  # pixels, uniform in each direction
    noise: float = 0.35                 # added noise, in units of the stamp's own
    blur: float = 0.8                   # maximum extra PSF sigma, pixels
    brightness: float = 0.25            # fractional scaling either way
    resized_crop: bool = False          # scale-destroying; see the class doc
    crop_range: Tuple[float, float] = (0.6, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"rotate": self.rotate, "flip": self.flip,
                "translate": self.translate, "noise": self.noise,
                "blur": self.blur, "brightness": self.brightness,
                "resized_crop": self.resized_crop}


def augment(stamp: np.ndarray, policy: AugmentationPolicy,
            rng: np.random.Generator) -> np.ndarray:
    """One plausible re-observation of the same object."""
    from ..core.numeric import bilinear_resize

    data = as_float_image(stamp).copy()
    height, width = data.shape

    if policy.resized_crop:
        # A crop that changes the sampling changes the apparent size of the
        # object, which is the measurement that separates stars from galaxies.
        # Off by default for that reason -- though see the class docstring:
        # the predicted cost did not show up in the measurement.
        low, high = policy.crop_range
        scale = float(rng.uniform(low, high))
        size = max(int(round(min(height, width) * scale)), 8)
        top = int(rng.integers(0, max(height - size, 1)))
        left = int(rng.integers(0, max(width - size, 1)))
        data = bilinear_resize(data[top:top + size, left:left + size],
                               (height, width))

    if policy.rotate:
        data = np.rot90(data, int(rng.integers(0, 4)))
    if policy.flip and bool(rng.random() < 0.5):
        data = data[:, ::-1]

    if policy.translate:
        shift = int(policy.translate)
        dy = int(rng.integers(-shift, shift + 1))
        dx = int(rng.integers(-shift, shift + 1))
        data = np.roll(np.roll(data, dy, axis=0), dx, axis=1)

    level = float(np.median(data))
    spread = float(np.std(data - level)) or 1.0

    if policy.blur > 0:
        sigma = float(rng.uniform(0.0, policy.blur))
        if sigma > 0.15:
            data = convolve(data, gaussian_kernel(sigma))
    if policy.noise > 0:
        data = data + rng.normal(0.0, policy.noise * spread, data.shape)
    if policy.brightness > 0:
        factor = 1.0 + float(rng.uniform(-policy.brightness, policy.brightness))
        data = level + (data - level) * factor
    return np.ascontiguousarray(data)


def build_projection_head(torch, dimension: int, output: int = 64):
    """The small MLP the contrastive loss actually acts on.

    The loss is applied to a projection of the embedding rather than the
    embedding itself, and the projection is thrown away afterwards. That looks
    wasteful and is not: the loss demands invariance to the augmentations, and
    an embedding forced to be exactly invariant would have discarded the
    information those augmentations vary. Putting a couple of layers in
    between lets the projection be invariant while the embedding underneath
    keeps more than that.
    """
    nn = torch.nn
    return nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(inplace=True),
                         nn.Linear(dimension, output))


def nt_xent_loss(torch, projections, temperature: float = TEMPERATURE):
    """Normalised temperature-scaled cross entropy, over 2N views.

    Rows ``i`` and ``i + N`` are the two views of one stamp and must be each
    other's nearest neighbour among all the others. Everything else in the
    batch is a negative -- including, unavoidably, other stamps of the same
    class, which the loss will push apart. That is a known and accepted flaw
    of the method: with no labels there is no way to know two stars are both
    stars, so the representation is learned in spite of the loss being wrong
    about some pairs.
    """
    n = projections.shape[0] // 2
    normalised = torch.nn.functional.normalize(projections, dim=1)
    similarity = normalised @ normalised.t() / float(temperature)
    similarity.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(
        projections.device)
    return torch.nn.functional.cross_entropy(similarity, targets)


@dataclass
class PretrainResult:
    """What a pretraining run did."""

    epochs: int = 0
    loss: List[float] = field(default_factory=list)
    n_stamps: int = 0
    policy: Dict[str, Any] = field(default_factory=dict)
    embedding_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"epochs": self.epochs, "n_stamps": self.n_stamps,
                "final_loss": self.loss[-1] if self.loss else float("nan"),
                "first_loss": self.loss[0] if self.loss else float("nan"),
                "policy": dict(self.policy),
                "embedding_dim": self.embedding_dim}


class ContrastiveEncoder:
    """A stamp encoder trained without labels.

    Produces the same kind of embedding the supervised classifier does, and
    can be handed to a :class:`~astrovision.ml.cnn.StampClassifier` as a
    starting point, so a few labels turn it into a classifier.
    """

    def __init__(self, cutout: int = 48, width: int = 16,
                 projection: int = 64, device: str = "cpu",
                 policy: Optional[AugmentationPolicy] = None,
                 random_state: int = 42):
        self.cutout = int(cutout)
        self.width = int(width)
        self.projection = int(projection)
        self.device = device
        self.policy = policy or AugmentationPolicy()
        self.random_state = int(random_state)
        self.model = None
        self.head = None
        self._torch = None
        self.result_ = PretrainResult()

    def build(self) -> "ContrastiveEncoder":
        torch = require("torch", "contrastive pretraining")
        self._torch = torch
        torch.manual_seed(self.random_state)
        # The classifier head is never used here; only the feature stack is.
        self.model = build_cnn(torch, n_classes=2, width=self.width,
                               cutout=self.cutout).to(self.device)
        self.head = build_projection_head(torch, self.model.embedding_dim,
                                          self.projection).to(self.device)
        return self

    def _prepare(self, stamps: Sequence[np.ndarray]) -> np.ndarray:
        from ..core.numeric import pad_or_crop
        from ..preprocess.normalize import asinh_stretch

        prepared = [pad_or_crop(asinh_stretch(as_float_image(s)),
                                (self.cutout, self.cutout)) for s in stamps]
        return np.stack(prepared).astype(np.float32)[:, None]

    def fit(self, stamps: Sequence[np.ndarray], epochs: int = 60,
            batch_size: int = 64, learning_rate: float = 1e-3,
            verbose: bool = False) -> PretrainResult:
        """Train on unlabelled stamps.

        No labels are accepted by this method, deliberately: passing them
        would make it possible to leak them into a run that claims not to use
        any.
        """
        if self.model is None:
            self.build()
        torch = self._torch
        if len(stamps) < 4:
            raise ModelError("contrastive training needs at least four stamps")

        rng = np.random.default_rng(self.random_state)
        parameters = list(self.model.parameters()) + list(self.head.parameters())
        optimiser = torch.optim.AdamW(parameters, lr=learning_rate,
                                      weight_decay=1e-4)
        self.result_ = PretrainResult(n_stamps=len(stamps),
                                      policy=self.policy.to_dict(),
                                      embedding_dim=self.model.embedding_dim)
        self.model.train()
        for epoch in range(int(epochs)):
            order = rng.permutation(len(stamps))
            total, batches = 0.0, 0
            for start in range(0, len(order), batch_size):
                chunk = order[start:start + batch_size]
                if len(chunk) < 4:
                    continue
                first = [augment(stamps[i], self.policy, rng) for i in chunk]
                second = [augment(stamps[i], self.policy, rng) for i in chunk]
                views = torch.from_numpy(
                    self._prepare(list(first) + list(second))).to(self.device)
                projections = self.head(self.model.embed(views))
                loss = nt_xent_loss(torch, projections)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1
            self.result_.loss.append(total / max(batches, 1))
            self.result_.epochs = epoch + 1
            if verbose and (epoch % max(1, epochs // 10) == 0
                            or epoch == epochs - 1):
                log.info("contrastive epoch %3d/%d loss=%.4f", epoch + 1, epochs,
                         self.result_.loss[-1])
        self.model.eval()
        log.info("pretrained on %d unlabelled stamps: loss %.3f -> %.3f",
                 len(stamps), self.result_.loss[0], self.result_.loss[-1])
        return self.result_

    def embed(self, stamps: Sequence[np.ndarray]) -> np.ndarray:
        """The learned representation, which is what all this was for."""
        if self.model is None:
            raise NotFittedError("build or fit the encoder before embedding")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self._prepare(stamps)).to(self.device)
            return self.model.embed(x).cpu().numpy()

    def to_classifier(self, classes: Optional[Sequence[ObjectClass]] = None
                      ) -> StampClassifier:
        """Wrap the pretrained features in a classifier with a fresh head.

        The head is new and random; the features are what the unlabelled data
        paid for. Fine-tuning this with a handful of labels is the whole point
        of the exercise.
        """
        torch = require("torch", "contrastive pretraining")
        if self.model is None:
            raise NotFittedError("fit the encoder before converting it")
        classes = list(classes or STAMP_CLASSES)
        classifier = StampClassifier(backbone="cnn", classes=classes,
                                     cutout=self.cutout, width=self.width,
                                     device=self.device,
                                     random_state=self.random_state)
        classifier.build()
        classifier.model.features.load_state_dict(self.model.features.state_dict())
        classifier.model.head = torch.nn.Linear(self.model.embedding_dim,
                                                len(classes)).to(self.device)
        classifier._torch = torch
        return classifier


def linear_probe(embeddings: np.ndarray, labels: Sequence[str],
                 test_embeddings: np.ndarray, test_labels: Sequence[str],
                 seed: int = 0) -> Dict[str, Any]:
    """Fit a linear classifier on frozen features and score it.

    The standard way to ask what a representation *contains*, as opposed to
    what a whole network can be trained to do. Nothing but a linear map is
    allowed, so a high score means the information was already there and
    linearly available; a low one means the features would need real work to
    use, whatever a fine-tuned network might eventually reach.
    """
    X = np.asarray(embeddings, dtype=float)
    Z = np.asarray(test_embeddings, dtype=float)
    classes = sorted(set(labels))
    index = {name: i for i, name in enumerate(classes)}
    y = np.array([index[str(l)] for l in labels])

    # Multinomial logistic regression by gradient descent: a dependency-free
    # linear probe, so the measurement does not change with what is installed.
    rng = np.random.default_rng(int(seed))
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mean) / scale
    Zs = (Z - mean) / scale
    W = rng.normal(0.0, 0.01, (Xs.shape[1], len(classes)))
    b = np.zeros(len(classes))
    one_hot = np.zeros((len(y), len(classes)))
    one_hot[np.arange(len(y)), y] = 1.0
    for _ in range(600):
        scores = Xs @ W + b
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        error = probabilities - one_hot
        W -= 0.5 * (Xs.T @ error / len(Xs) + 1e-3 * W)
        b -= 0.5 * error.mean(axis=0)

    predictions = np.argmax(Zs @ W + b, axis=1)
    truth = np.array([index.get(str(l), -1) for l in test_labels])
    known = truth >= 0
    accuracy = float(np.mean(predictions[known] == truth[known])) if known.any() \
        else float("nan")
    recalls = {}
    for name, i in index.items():
        mask = truth == i
        recalls[name] = float(np.mean(predictions[mask] == i)) if mask.any() \
            else float("nan")
    return {"accuracy": accuracy, "per_class_recall": recalls,
            "balanced_accuracy": float(np.nanmean(list(recalls.values()))),
            "n_train": len(X), "n_test": int(known.sum())}


def anomaly_ranking_quality(embeddings: np.ndarray, is_anomalous: Sequence[bool]
                            ) -> Dict[str, Any]:
    """How well an embedding separates known oddities, as an ROC area.

    Uses the k-nearest-neighbour distance as the anomaly score, which is the
    part of the anomaly engine that depends purely on the representation --
    so the number compares *embeddings*, not detectors.
    """
    matrix = np.asarray(embeddings, dtype=float)
    flags = np.asarray(is_anomalous, dtype=bool)
    if matrix.ndim != 2 or len(matrix) < 5 or flags.sum() == 0 or flags.all():
        return {"auc": float("nan"), "n_anomalies": int(flags.sum())}

    scores = []
    for i in range(len(matrix)):
        distances = np.linalg.norm(matrix - matrix[i], axis=1)
        distances[i] = np.inf
        scores.append(float(np.mean(np.sort(distances)[:3])))
    scores = np.asarray(scores)

    positive = scores[flags]
    negative = scores[~flags]
    # The area under the ROC curve is the probability that a random anomaly
    # scores above a random ordinary object; computing it that way avoids a
    # threshold sweep and is exact.
    wins = float(np.sum(positive[:, None] > negative[None, :]))
    ties = float(np.sum(positive[:, None] == negative[None, :]))
    auc = (wins + 0.5 * ties) / (len(positive) * len(negative))
    return {"auc": float(auc), "n_anomalies": int(flags.sum()),
            "n_ordinary": int((~flags).sum())}


def label_efficiency(encoder: ContrastiveEncoder, labelled: StampSet,
                     test: StampSet, budgets: Sequence[int] = (10, 25, 50, 100),
                     seed: int = 0, repeats: int = 3) -> List[Dict[str, Any]]:
    """Compare a probe on pretrained features against training from scratch.

    The claim self-supervision makes is about *labels*, so this is the
    measurement that tests it: at each budget, how well does a linear probe on
    the unlabelled-pretrained features do, and how well does the same budget
    do with no pretraining at all.
    """
    from .transfer import _stratified_sample, evaluate

    rows: List[Dict[str, Any]] = []
    test_embeddings = encoder.embed(test.stamps)
    test_labels = [l.value for l in test.labels]

    for budget in sorted(int(b) for b in budgets):
        if budget > len(labelled):
            continue
        probe_scores, scratch_scores = [], []
        for trial in range(max(int(repeats), 1)):
            rng = np.random.default_rng(int(seed) + 100 * trial)
            chosen = _stratified_sample(labelled, budget, rng)
            probe = linear_probe(encoder.embed(chosen.stamps),
                                 [l.value for l in chosen.labels],
                                 test_embeddings, test_labels,
                                 seed=int(seed) + trial)
            probe_scores.append(probe["balanced_accuracy"])

            scratch = StampClassifier(backbone="cnn",
                                      classes=list(dict.fromkeys(test.labels)),
                                      cutout=encoder.cutout, width=encoder.width,
                                      device=encoder.device,
                                      random_state=int(seed) + trial)
            scratch.fit(chosen.stamps, chosen.labels, epochs=40, batch_size=16,
                        verbose=False)
            scratch_scores.append(evaluate(scratch, test)["balanced_accuracy"])
        rows.append({
            "n_labels": budget,
            "linear_probe": float(np.mean(probe_scores)),
            "linear_probe_sd": float(np.std(probe_scores)),
            "from_scratch": float(np.mean(scratch_scores)),
            "from_scratch_sd": float(np.std(scratch_scores)),
            "advantage": float(np.mean(probe_scores) - np.mean(scratch_scores)),
            "repeats": len(probe_scores)})
    return rows
