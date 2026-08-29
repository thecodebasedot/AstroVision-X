"""Why a model said what it said, and whether that answer can be trusted.

An astronomer will not act on a number a black box produced, and they are
right not to. Every model in this package that scores an object therefore has
a matching explanation here:

* the **stamp classifier** gets Grad-CAM, a map of which pixels the class
  score depended on;
* the **gradient-boosted model** gets Shapley values, the contribution of each
  measured feature to this object's prediction;
* the **anomaly ranking** gets retrieval -- "this looks like these three
  objects, and here is how far from them it is".

The hard part is not producing any of those. It is knowing whether they are
true. A saliency map is an image, and an image is convincing whether or not it
describes the model; published saliency methods have been shown to produce
confident-looking maps that are unchanged when the model's weights are
randomised. So every explanation here is checked against the model's own
behaviour rather than against intuition:

* a Grad-CAM map is checked by **deletion**: erase the pixels it calls
  important and the class score must fall further than erasing the same number
  of random pixels. If it does not, the map is decoration.
* Shapley values are checked by **additivity**: they must sum, with the base
  rate, to the model's actual output for that object. A method that does not
  close this books is attributing something other than the prediction.
* retrieval is checked by **purity**: the neighbours it returns must share the
  query's class more often than chance, on a set where the classes are known.

Those checks are functions here, not comments, and they are what the tests
assert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import require
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, bilinear_resize
from ..core.types import ObjectClass, Source, SourceCatalog

log = get_logger("ml.explain")


@dataclass
class SaliencyMap:
    """A per-pixel importance map for one stamp and one class."""

    heatmap: np.ndarray                     # same shape as the stamp, 0-1
    predicted_class: str = ""
    probability: float = float("nan")
    layer: str = ""
    native_shape: Tuple[int, int] = (0, 0)  # resolution before upsampling
    method: str = "grad-cam"

    @property
    def peak(self) -> Tuple[int, int]:
        """Row and column of the most important pixel."""
        index = int(np.argmax(self.heatmap))
        return divmod(index, self.heatmap.shape[1])

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "layer": self.layer,
                "predicted_class": self.predicted_class,
                "probability": self.probability,
                "native_shape": list(self.native_shape),
                "peak": list(self.peak),
                "concentration": float(self.heatmap.max() - self.heatmap.mean())}


def grad_cam(classifier, stamp: np.ndarray, class_index: Optional[int] = None
             ) -> SaliencyMap:
    """Gradient-weighted class activation mapping (Selvaraju et al. 2017).

    The class score is differentiated with respect to the last convolutional
    feature map; each channel is weighted by how much raising it would raise
    that score, and the weighted sum is the map.

    Two honest notes about what comes back:

    **The resolution is the feature map's, not the stamp's.** After two
    stride-2 blocks a 48-pixel stamp has a 12 x 12 map, so the heatmap is
    twelve pixels across, smoothly upsampled. It can say "the light in the
    middle" and it cannot say "this spiral arm". Reading structure into the
    interpolation is reading the interpolation.

    **For this architecture Grad-CAM and plain CAM coincide.** The network
    global-average-pools before a single linear head, so the gradient of the
    class score with respect to a channel's pooled activation is exactly that
    channel's head weight. The gradient form is computed anyway, because it is
    the one that stays correct if the head ever stops being a single linear
    layer -- and :func:`cam_matches_head_weights` checks the equality holds.
    """
    torch = require("torch", "Grad-CAM")
    if classifier.model is None:
        raise NotFittedError("build or load the classifier before explaining it")
    features = getattr(classifier.model, "features", None)
    if features is None:
        raise ModelError(
            "Grad-CAM needs a convolutional feature stack; this backbone has "
            "none. An attention model needs attention rollout instead, which "
            "is a different method and is not implemented here.")

    classifier.model.eval()
    x = torch.from_numpy(classifier._prepare([stamp])).to(classifier.device)

    # Everything but the final pooling: that is the last spatial feature map.
    body = torch.nn.Sequential(*list(features.children())[:-1])
    pool = list(features.children())[-1]

    maps = body(x)
    maps.retain_grad()
    pooled = pool(maps).flatten(1)
    logits = classifier.model.head(pooled)
    probabilities = torch.softmax(logits, dim=1)
    index = int(torch.argmax(logits, dim=1).item()) if class_index is None \
        else int(class_index)

    classifier.model.zero_grad(set_to_none=True)
    logits[0, index].backward()
    gradients = maps.grad[0].detach().cpu().numpy()
    activations = maps[0].detach().cpu().numpy()

    weights = gradients.mean(axis=(1, 2))
    # ReLU on the combination, not on the channels: a channel that argues
    # *against* the class is evidence about the class, but Grad-CAM asks
    # specifically what supports it.
    combined = np.maximum((weights[:, None, None] * activations).sum(axis=0), 0.0)
    native = combined.shape
    if combined.max() > combined.min():
        combined = (combined - combined.min()) / (combined.max() - combined.min())
    else:
        combined = np.zeros_like(combined)

    stamp_shape = as_float_image(stamp).shape
    heatmap = bilinear_resize(combined, stamp_shape)
    return SaliencyMap(
        heatmap=np.clip(heatmap, 0.0, 1.0),
        predicted_class=classifier.classes[index].value,
        probability=float(probabilities[0, index].item()),
        layer="last convolutional block", native_shape=tuple(native))


def cam_matches_head_weights(classifier, stamp: np.ndarray,
                             tolerance: float = 1e-3) -> Dict[str, Any]:
    """Check that the gradient weights equal the head weights, as they must.

    With global average pooling before one linear layer this is an identity,
    not an approximation, so a mismatch means the gradient is not flowing
    where it is assumed to -- a hook on the wrong layer, or a head that is no
    longer a single linear map. Cheap, exact, and it fails loudly.
    """
    torch = require("torch", "Grad-CAM")
    features = getattr(classifier.model, "features", None)
    if features is None:
        raise ModelError("this backbone has no convolutional feature stack")
    classifier.model.eval()
    x = torch.from_numpy(classifier._prepare([stamp])).to(classifier.device)
    body = torch.nn.Sequential(*list(features.children())[:-1])
    pool = list(features.children())[-1]
    maps = body(x)
    maps.retain_grad()
    logits = classifier.model.head(pool(maps).flatten(1))
    index = int(torch.argmax(logits, dim=1).item())
    classifier.model.zero_grad(set_to_none=True)
    logits[0, index].backward()

    n_pixels = maps.shape[2] * maps.shape[3]
    gradient_weights = maps.grad[0].detach().cpu().numpy().mean(axis=(1, 2))
    head_weights = (classifier.model.head.weight[index].detach().cpu().numpy()
                    / n_pixels)
    difference = float(np.max(np.abs(gradient_weights - head_weights)))
    return {"max_difference": difference, "agrees": difference < float(tolerance),
            "n_channels": int(gradient_weights.size),
            "note": ("global average pooling into one linear head makes these "
                     "identical; a difference means the gradient is not "
                     "flowing where the map assumes")}


def deletion_curve(classifier, stamp: np.ndarray, heatmap: np.ndarray,
                   fractions: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
                   class_index: Optional[int] = None,
                   random_baseline: bool = True, fill: str = "noise",
                   seed: int = 0) -> Dict[str, Any]:
    """Erase the pixels a map calls important and watch the score fall.

    This is the test that separates an explanation from a picture. If the
    highlighted pixels really carry the decision, erasing them should drop the
    class probability faster than erasing the same number of randomly chosen
    pixels. If the two curves lie on top of each other, the map is not
    describing the model.

    **What "erase" means decides the answer**, which was discovered the
    embarrassing way. Filling with a constant -- the obvious choice, and the
    usual one -- narrows the stamp's noise distribution, and the classifier's
    asinh stretch computes its softening from exactly that distribution. So a
    map that happens to rank background pixels highly changes the *stretch*
    rather than removing information, and scores an advantage it has not
    earned. Measured both ways on the same maps: a constant fill reported a
    mean advantage of 0.109 and 32 of 40 stamps beating chance, while filling
    with noise drawn from the stamp's own background gave 0.044 and 25 of 40.
    More than half the apparent effect was the test measuring itself.

    ``fill="noise"`` is therefore the default. ``fill="constant"`` is kept
    because it is what the literature usually does, and because the gap
    between the two is worth being able to reproduce.

    Returns both curves and the area between them: positive means the map beat
    chance, zero or negative means it did not.
    """
    data = as_float_image(stamp)
    rng = np.random.default_rng(int(seed))
    order = np.argsort(np.asarray(heatmap, dtype=float).ravel())[::-1]
    _, level, spread = _background_stats(data)

    if class_index is None:
        probabilities = classifier.predict_proba([data])[0]
        class_index = int(np.argmax(probabilities))

    def score_after(mask_index: np.ndarray, noise_seed: int) -> float:
        damaged = data.copy().ravel()
        if fill == "constant":
            damaged[mask_index] = level
        else:
            local = np.random.default_rng(int(noise_seed))
            damaged[mask_index] = level + local.normal(0.0, spread,
                                                       size=len(mask_index))
        return float(classifier.predict_proba(
            [damaged.reshape(data.shape)])[0, class_index])

    guided, random_scores = [], []
    for fraction in fractions:
        count = int(round(float(fraction) * data.size))
        guided.append(score_after(order[:count], int(seed)))
        if random_baseline:
            chosen = rng.choice(data.size, size=count, replace=False)
            # The same noise seed for both curves, so the comparison is
            # between the two orderings and not between two noise draws.
            random_scores.append(score_after(chosen, int(seed)))

    result: Dict[str, Any] = {
        "fractions": [float(f) for f in fractions], "fill": str(fill),
        "guided": guided, "class_index": int(class_index)}
    if random_baseline:
        result["random"] = random_scores
        # Area between the curves: the random curve should stay above the
        # guided one, so a positive number means the map found the pixels
        # that mattered.
        result["advantage"] = float(np.trapezoid(random_scores, fractions)
                                    - np.trapezoid(guided, fractions))
        result["beats_chance"] = bool(result["advantage"] > 0)
    return result


def _background_stats(data: np.ndarray) -> Tuple[float, float, float]:
    """Mean, median and robust spread of a stamp's background."""
    from ..core.numeric import sigma_clipped_stats

    mean, median, spread = sigma_clipped_stats(data[np.isfinite(data)])
    if not np.isfinite(spread) or spread <= 0:
        spread = float(np.std(data)) or 1.0
    return float(mean), float(median), float(spread)


def occlusion_map(classifier, stamp: np.ndarray, patch: int = 8, stride: int = 4,
                  class_index: Optional[int] = None, seed: int = 0
                  ) -> SaliencyMap:
    """Importance by covering the image up, one patch at a time.

    Slower than Grad-CAM by the number of patches, and worth it here. It makes
    no assumption about the architecture and no claim about gradients: it
    measures the thing an explanation is supposed to be about, which is how
    much the model's answer depends on a region. Each patch is replaced with
    background noise rather than a constant, for the reason described in
    :func:`deletion_curve`.

    On this classifier it produces maps that are measurably more faithful than
    Grad-CAM's -- see `docs/validation.md` -- which is not a surprise: it is
    optimising the same quantity the faithfulness test measures. What it
    cannot do is run at interactive speed on a large catalog.
    """
    data = as_float_image(stamp)
    _, level, spread = _background_stats(data)
    rng = np.random.default_rng(int(seed))

    base = classifier.predict_proba([data])[0]
    if class_index is None:
        class_index = int(np.argmax(base))
    reference = float(base[class_index])

    height, width = data.shape
    scores = np.zeros_like(data)
    counts = np.zeros_like(data)
    half = int(patch)
    positions = [(y, x)
                 for y in range(0, max(height - half + 1, 1), int(stride))
                 for x in range(0, max(width - half + 1, 1), int(stride))]
    batch, boxes = [], []
    for y, x in positions:
        covered = data.copy()
        covered[y:y + half, x:x + half] = level + rng.normal(
            0.0, spread, size=covered[y:y + half, x:x + half].shape)
        batch.append(covered)
        boxes.append((y, x))
    probabilities = classifier.predict_proba(batch)[:, class_index]
    for (y, x), value in zip(boxes, probabilities):
        scores[y:y + half, x:x + half] += reference - float(value)
        counts[y:y + half, x:x + half] += 1.0

    heatmap = np.where(counts > 0, scores / np.maximum(counts, 1e-9), 0.0)
    heatmap = np.maximum(heatmap, 0.0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return SaliencyMap(heatmap=heatmap,
                       predicted_class=classifier.classes[class_index].value,
                       probability=reference, layer=f"{patch}px occlusion",
                       native_shape=(len(range(0, max(height - half + 1, 1), stride)),
                                     len(range(0, max(width - half + 1, 1), stride))),
                       method="occlusion")


def explain_stamp(classifier, stamp: np.ndarray, method: str = "occlusion",
                  class_index: Optional[int] = None, **kwargs) -> SaliencyMap:
    """Explain one stamp, defaulting to the method that measured better.

    Grad-CAM is the famous one and it is nearly useless here. Measured on 40
    stamps against a noise-preserving deletion test, it beat chance on 21 of
    them with a mean advantage of 0.03, its correlation with where the
    object's light actually is came out at -0.04, and it put 0.15 of its mass
    on the central sixteenth of the stamp where a uniform map would put 0.11.
    Occlusion beat chance on 37 of 40 with an advantage of 0.23, a correlation
    of 0.30, and 0.60 of its mass on the object.

    Part of that gap is expected -- occlusion optimises the quantity the
    deletion test measures -- but the correlation and the concentration are
    not what it optimises, and it wins those too. The cause is structural: a
    48-pixel stamp leaves Grad-CAM a 12 x 12 map, one to four cells of which
    cover a compact source, and global average pooling means the decision
    genuinely draws on the whole frame.

    So the default is occlusion, and Grad-CAM remains available for the case
    where its speed matters more than its fidelity -- with this docstring
    attached to it.
    """
    if method == "grad-cam":
        return grad_cam(classifier, stamp, class_index=class_index)
    if method == "occlusion":
        return occlusion_map(classifier, stamp, class_index=class_index, **kwargs)
    raise ValueError(f"unknown saliency method {method!r}; "
                     "expected 'occlusion' or 'grad-cam'")


@dataclass
class Attribution:
    """Per-feature contributions to one prediction."""

    values: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, float] = field(default_factory=dict)
    base_value: float = float("nan")
    prediction: float = float("nan")
    predicted_class: str = ""
    n_samples: int = 0
    converged: bool = False

    def top(self, n: int = 5) -> List[Tuple[str, float]]:
        """The features that moved this prediction most, either way."""
        return sorted(self.values.items(), key=lambda kv: -abs(kv[1]))[:n]

    def additivity_error(self) -> float:
        """How far the attributions are from explaining the prediction.

        Shapley values sum with the base value to exactly the model's output.
        Any residual is estimation error, and quoting it is the difference
        between an attribution and a decoration.
        """
        return float(abs(self.base_value + sum(self.values.values())
                         - self.prediction))

    def explain(self, n: int = 3) -> str:
        """A sentence an astronomer can read."""
        if not self.values:
            return "no attribution available"
        parts = []
        for name, value in self.top(n):
            direction = "raised" if value > 0 else "lowered"
            parts.append(f"{name} {direction} it by {abs(value):.3f}")
        return (f"predicted {self.predicted_class} at {self.prediction:.3f} "
                f"against a base rate of {self.base_value:.3f}; "
                + ", ".join(parts))

    def to_dict(self) -> Dict[str, Any]:
        return {"values": dict(self.values), "errors": dict(self.errors),
                "base_value": self.base_value, "prediction": self.prediction,
                "predicted_class": self.predicted_class,
                "n_samples": self.n_samples, "converged": self.converged,
                "additivity_error": self.additivity_error(),
                "explanation": self.explain()}


def shapley_values(predict_proba: Callable[[np.ndarray], np.ndarray],
                   x: np.ndarray, background: np.ndarray,
                   feature_names: Optional[Sequence[str]] = None,
                   class_index: int = 0, n_samples: int = 200,
                   tolerance: float = 0.02, seed: int = 0) -> Attribution:
    """Shapley attributions by permutation sampling.

    The Shapley value of a feature is its average marginal contribution over
    every order in which features could be revealed. Enumerating those orders
    is factorial, so they are sampled -- which makes the result an estimate,
    and an estimate without an error bar is a number pretending to be a fact.
    The standard error of the mean over permutations is returned per feature,
    and ``converged`` says whether every one of them is below ``tolerance``.

    Model-agnostic on purpose: it takes a ``predict_proba`` callable, so it
    works whether the boosted trees came from XGBoost, scikit-learn or the
    NumPy fallback, and works on any other scorer too.

    The error falls as one over the square root of the sample count, and it is
    worth knowing the constant before trusting a small run. On an eight-feature
    model here:

    ====== ===========
    draws  max std err
    ====== ===========
    50     0.049
    100    0.035
    200    0.025
    400    0.018
    800    0.013
    ====== ===========

    The residual in :meth:`Attribution.additivity_error` -- 0.005 to 0.010
    across all of those -- is a different quantity: it is how far the sampled
    reference rows sit from the full background mean, and it does not shrink
    with more permutations, only with a larger background set.

    An error of exactly zero is meaningful rather than missing: it says the
    model's output did not move in *any* permutation when that feature was
    revealed, which is what a feature no tree splits on looks like.
    """
    x = np.asarray(x, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    if background.shape[1] != x.size:
        raise ValueError("background rows must have the same width as x")
    names = list(feature_names) if feature_names is not None \
        else [f"f{i}" for i in range(x.size)]
    if len(names) != x.size:
        raise ValueError("one name per feature is required")

    rng = np.random.default_rng(int(seed))
    n_features = x.size
    contributions = np.zeros((int(n_samples), n_features))

    base = float(np.mean(predict_proba(background)[:, class_index]))
    prediction = float(predict_proba(x.reshape(1, -1))[0, class_index])

    for sample in range(int(n_samples)):
        reference = background[rng.integers(0, len(background))]
        order = rng.permutation(n_features)
        current = reference.copy()
        previous = float(predict_proba(current.reshape(1, -1))[0, class_index])
        for feature in order:
            current[feature] = x[feature]
            value = float(predict_proba(current.reshape(1, -1))[0, class_index])
            contributions[sample, feature] = value - previous
            previous = value

    means = contributions.mean(axis=0)
    errors = contributions.std(axis=0) / math.sqrt(max(int(n_samples), 1))
    attribution = Attribution(
        values={name: float(v) for name, v in zip(names, means)},
        errors={name: float(e) for name, e in zip(names, errors)},
        base_value=base, prediction=prediction, n_samples=int(n_samples),
        converged=bool(np.all(errors < float(tolerance))))
    return attribution


def explain_prediction(model, x: np.ndarray, background: np.ndarray,
                       feature_names: Optional[Sequence[str]] = None,
                       n_samples: int = 200, seed: int = 0) -> Attribution:
    """Attributions for one row through a fitted classifier.

    The background set is what the attribution is *against*: a feature's
    contribution is meaningful only relative to some reference population.
    Passing the training set answers "why this object rather than a typical
    one", which is the question an astronomer is actually asking.
    """
    if getattr(model, "classes_", None) is None:
        raise NotFittedError("fit the model before explaining it")
    probabilities = model.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))
    class_index = int(np.argmax(probabilities[0]))
    attribution = shapley_values(model.predict_proba, x, background,
                                 feature_names=feature_names,
                                 class_index=class_index, n_samples=n_samples,
                                 seed=seed)
    attribution.predicted_class = str(model.classes_[class_index])
    return attribution


@dataclass
class Neighbours:
    """The objects a query most resembles, and how far away they are."""

    indices: List[int] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)
    typical_distance: float = float("nan")
    isolation: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {"indices": list(self.indices), "distances": list(self.distances),
                "labels": list(self.labels), "ids": list(self.ids),
                "typical_distance": self.typical_distance,
                "isolation": self.isolation, "explanation": self.explain()}

    def explain(self) -> str:
        if not self.indices:
            return "no neighbours found"
        described = ", ".join(
            f"#{i} ({label}, {distance:.2f})"
            for i, label, distance in zip(self.ids or self.indices,
                                          self.labels or ["?"] * len(self.indices),
                                          self.distances))
        text = f"nearest in the embedding: {described}"
        if np.isfinite(self.isolation):
            text += (f"; its closest neighbour is {self.isolation:.1f} times the "
                     "typical separation in this field, which is what makes it "
                     "stand out")
        return text


def retrieve_similar(embeddings: np.ndarray, query_index: int, n: int = 3,
                     labels: Optional[Sequence[str]] = None,
                     ids: Optional[Sequence[int]] = None) -> Neighbours:
    """Find the objects nearest a query in embedding space.

    This is the explanation an anomaly score cannot give by itself. A number
    saying "0.97 unusual" is unactionable; "its nearest analogues are these
    three, and even they are four times further away than objects normally
    are from each other" is a description a person can check by looking.

    The isolation ratio is the second part of that: the distance to the
    nearest neighbour, divided by the typical nearest-neighbour distance in
    the same set. Above about two, the object genuinely has no close analogue
    here; near one, it is ordinary and the anomaly score is about something
    else.
    """
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2:
        return Neighbours()
    query = matrix[int(query_index)]
    separation = np.linalg.norm(matrix - query, axis=1)
    separation[int(query_index)] = np.inf
    order = np.argsort(separation)[:int(n)]

    # The typical nearest-neighbour distance in this set, for scale.
    nearest = []
    for i in range(len(matrix)):
        distances = np.linalg.norm(matrix - matrix[i], axis=1)
        distances[i] = np.inf
        nearest.append(float(distances.min()))
    typical = float(np.median(nearest))

    result = Neighbours(
        indices=[int(i) for i in order],
        distances=[float(separation[i]) for i in order],
        labels=[str(labels[i]) for i in order] if labels is not None else [],
        ids=[int(ids[i]) for i in order] if ids is not None else [],
        typical_distance=typical)
    if typical > 0 and result.distances:
        result.isolation = float(result.distances[0] / typical)
    return result


def retrieval_purity(embeddings: np.ndarray, labels: Sequence[str],
                     n: int = 3) -> Dict[str, Any]:
    """Do retrieved neighbours share the query's class more often than chance?

    The check that makes retrieval an explanation rather than a list. Chance
    is the probability two random objects share a class, computed from the
    label distribution -- not one over the number of classes, which would be
    the right baseline only if the classes were balanced, and they never are.
    """
    matrix = np.asarray(embeddings, dtype=float)
    labels = [str(l) for l in labels]
    if len(matrix) < 3:
        return {"purity": float("nan"), "chance": float("nan"), "n": len(matrix)}
    hits = []
    for i in range(len(matrix)):
        found = retrieve_similar(matrix, i, n=n, labels=labels)
        if found.labels:
            hits.append(float(np.mean([l == labels[i] for l in found.labels])))
    counts = np.array([labels.count(l) for l in sorted(set(labels))], dtype=float)
    fractions = counts / counts.sum()
    chance = float(np.sum(fractions ** 2))
    purity = float(np.mean(hits)) if hits else float("nan")
    return {"purity": purity, "chance": chance, "n": len(matrix),
            "lift": float(purity / chance) if chance > 0 else float("nan"),
            "beats_chance": bool(purity > chance)}


def explain_catalog(catalog: SourceCatalog, n: int = 3,
                    attribute: str = "anomaly_score",
                    top: int = 10) -> List[Dict[str, Any]]:
    """Attach a retrieval explanation to the most unusual sources.

    Only to the top few: computing neighbours for every source in a large
    catalog is quadratic and the answer is uninteresting for the ordinary
    ones, which by definition have close analogues everywhere.
    """
    sources: List[Source] = [s for s in catalog
                             if getattr(s, "embedding", None) is not None]
    if len(sources) < 3:
        return []
    embeddings = np.vstack([np.asarray(s.embedding, dtype=float) for s in sources])
    scores = np.array([float(getattr(s, attribute, np.nan) or np.nan)
                       for s in sources])
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]

    explanations: List[Dict[str, Any]] = []
    for position in order[:int(top)]:
        found = retrieve_similar(
            embeddings, int(position), n=n,
            labels=[s.object_class.value if isinstance(s.object_class, ObjectClass)
                    else str(s.object_class) for s in sources],
            ids=[int(s.id) for s in sources])
        payload = found.to_dict()
        payload["source_id"] = int(sources[int(position)].id)
        payload[attribute] = float(scores[int(position)])
        sources[int(position)].meta["similar_to"] = payload
        explanations.append(payload)
    return explanations
