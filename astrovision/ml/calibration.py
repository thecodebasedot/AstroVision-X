"""Turning classifier scores into probabilities that mean what they say.

A classifier's output is a number between 0 and 1.  That does not make it a
probability.  A model is *calibrated* when the objects it scores 0.8 are
right 80% of the time, and most classifiers are not: neural networks are
famously overconfident, boosted trees pull toward the extremes, and a rule
system built from hand-tuned logistic votes -- like the one in
:mod:`astrovision.classify.rules` -- has no reason to be calibrated at all.

This matters more here than in a typical machine-learning setting, because
the numbers are used for *ranking what a human should look at*.  An
overconfident 0.95 that is really 0.6 sends an astronomer to a telescope.
And it matters for the boundary this package keeps: a candidate reported
with a confidence is making a quantitative claim, and the claim should be
true.

Two methods, both standard:

* **Platt scaling** fits a one-parameter logistic to the scores.  It assumes
  the miscalibration has a particular shape, which makes it stable with very
  little validation data -- a few dozen labelled objects.
* **Isotonic regression** fits any monotonic map.  It is strictly more
  general and strictly hungrier for data; below a few hundred points it
  overfits, reproducing the validation set's noise as structure.

The right choice is a function of how much labelled data there is, so
:func:`fit_calibrator` picks by counting it rather than leaving a
consequential default to whoever calls first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from ..core.exceptions import DataError
from ..core.logging import get_logger

log = get_logger("ml.calibration")

#: Below this many labelled examples, isotonic regression fits noise.
ISOTONIC_MINIMUM = 200


@dataclass
class Calibrator:
    """A fitted map from raw scores to calibrated probabilities."""

    method: str = "identity"
    #: Platt parameters ``(a, b)`` for ``sigmoid(a * score + b)``.
    slope: float = 1.0
    intercept: float = 0.0
    #: Isotonic breakpoints, as parallel arrays.
    knots_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    knots_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_train: int = 0
    reason: str = ""

    def __call__(self, scores) -> np.ndarray:
        return self.transform(scores)

    def transform(self, scores) -> np.ndarray:
        """Map raw scores to calibrated probabilities."""
        values = np.asarray(scores, dtype=float)
        if self.method == "platt":
            z = self.slope * to_log_odds(values) + self.intercept
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        if self.method == "isotonic" and self.knots_x.size:
            # Held flat outside the fitted range: the validation set contains
            # no evidence about scores it never saw, and extrapolating a
            # monotone fit invents some.
            return np.interp(values, self.knots_x, self.knots_y,
                             left=float(self.knots_y[0]), right=float(self.knots_y[-1]))
        return np.clip(values, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method, "slope": float(self.slope),
            "intercept": float(self.intercept), "n_train": int(self.n_train),
            "n_knots": int(self.knots_x.size), "reason": self.reason,
        }


def to_log_odds(scores, epsilon: float = 1e-6) -> np.ndarray:
    """Map probabilities to the unbounded log-odds scale.

    Platt scaling belongs on this scale, not on the probability itself.  The
    method was defined for an SVM's unbounded decision value, and a logistic
    of a linear function of a *bounded* input cannot represent the maps that
    calibration usually needs.  The commonest miscalibration -- an
    overconfident model, where the reported log-odds are the true ones
    multiplied by some factor -- becomes a pure slope here, which one
    parameter fits exactly.  Fed probabilities directly, the same fit made
    the calibration error worse in testing rather than better.
    """
    p = np.clip(np.asarray(scores, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def fit_platt(scores: Sequence[float], labels: Sequence[int],
              iterations: int = 100, tolerance: float = 1e-7) -> Calibrator:
    """Fit ``sigmoid(a * logit(score) + b)`` by Newton-Raphson.

    The targets are smoothed away from 0 and 1 in Platt's original way --
    ``(n+1)/(n+2)`` for positives -- which stops the fit running the
    coefficients to infinity when the classes happen to be separable in the
    validation set.  Perfect separation on a hundred points is not evidence
    of infinite confidence; it is evidence of a hundred points.
    """
    x = to_log_odds(scores)
    y = np.asarray(labels, dtype=float)
    if x.size != y.size or x.size == 0:
        raise DataError("Platt scaling needs matching, non-empty scores and labels")
    n_positive = float(np.sum(y > 0.5))
    n_negative = float(y.size - n_positive)
    high = (n_positive + 1.0) / (n_positive + 2.0)
    low = 1.0 / (n_negative + 2.0)
    target = np.where(y > 0.5, high, low)

    def objective(slope: float, intercept: float) -> float:
        z = np.clip(slope * x + intercept, -60.0, 60.0)
        # The numerically stable form of the cross-entropy: written as
        # log(1 + exp(z)) it overflows for large positive z, which is exactly
        # where a diverging fit spends its time.
        return float(np.sum((1.0 - target) * z + np.logaddexp(0.0, -z)))

    a, b = 1.0, 0.0
    current = objective(a, b)
    for _ in range(int(iterations)):
        z = np.clip(a * x + b, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))
        residual = p - target
        weight = np.clip(p * (1.0 - p), 1e-12, None)
        gradient = np.array([float(np.sum(residual * x)), float(np.sum(residual))])
        hessian = np.array([
            [float(np.sum(weight * x * x)), float(np.sum(weight * x))],
            [float(np.sum(weight * x)), float(np.sum(weight))],
        ])
        try:
            step = np.linalg.solve(hessian + 1e-10 * np.eye(2), gradient)
        except np.linalg.LinAlgError:                          # pragma: no cover
            break
        # Newton's method on this likelihood diverges without a line search
        # whenever the validation classes are close to separable: the Hessian
        # goes flat, the step goes to infinity, and the fit lands on a slope
        # of ten million -- a calibrator that returns only 0 and 1.  Halving
        # the step until the objective actually decreases is the standard
        # remedy and costs nothing when the plain step was already fine.
        scale = 1.0
        improved = False
        for _ in range(24):
            trial_a, trial_b = a - scale * float(step[0]), b - scale * float(step[1])
            trial = objective(trial_a, trial_b)
            if trial <= current:
                a, b, current = trial_a, trial_b, trial
                improved = True
                break
            scale *= 0.5
        if not improved or scale * float(np.max(np.abs(step))) < tolerance:
            break
    return Calibrator(method="platt", slope=a, intercept=b, n_train=int(x.size),
                      reason="Platt scaling (logistic in log-odds)")


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> Calibrator:
    """Fit a monotone step function by pool-adjacent-violators.

    The algorithm is exact and needs no iteration: sort by score, then
    repeatedly merge any adjacent pair whose means are out of order, which
    provably converges to the least-squares monotone fit.
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size != y.size or x.size == 0:
        raise DataError("isotonic regression needs matching, non-empty inputs")
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    values: List[float] = []
    weights: List[float] = []
    for value in y:
        values.append(float(value))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            weight = weights[-1] + weights[-2]
            merged = (values[-1] * weights[-1] + values[-2] * weights[-2]) / weight
            values[-2:] = [merged]
            weights[-2:] = [weight]

    fitted = np.repeat(values, [int(w) for w in weights])
    return Calibrator(method="isotonic", knots_x=x, knots_y=np.clip(fitted, 0.0, 1.0),
                      n_train=int(x.size),
                      reason="isotonic regression (pool-adjacent-violators)")


def fit_calibrator(scores: Sequence[float], labels: Sequence[int],
                   method: str = "auto") -> Calibrator:
    """Fit whichever calibrator the amount of labelled data can support.

    ``auto`` uses isotonic regression above :data:`ISOTONIC_MINIMUM`
    examples and Platt scaling below it.  The threshold is not arbitrary:
    isotonic regression has as many free parameters as it has distinct
    scores, so on a small validation set it reproduces that set exactly and
    generalises worse than the uncalibrated model it replaced.
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10:
        return Calibrator(method="identity", n_train=int(x.size),
                          reason=f"only {x.size} labelled examples; left uncalibrated")
    if len(np.unique(y > 0.5)) < 2:
        return Calibrator(method="identity", n_train=int(x.size),
                          reason="validation set has only one class")

    chosen = method
    if method == "auto":
        chosen = "isotonic" if x.size >= ISOTONIC_MINIMUM else "platt"
    if chosen == "isotonic":
        return fit_isotonic(x, y)
    if chosen == "platt":
        return fit_platt(x, y)
    if chosen in ("identity", "none"):
        return Calibrator(method="identity", n_train=int(x.size),
                          reason="calibration disabled")
    raise DataError(f"unknown calibration method {method!r}")


# --------------------------------------------------------------------------
# how well calibrated is it?
# --------------------------------------------------------------------------
def reliability_curve(probabilities: Sequence[float], labels: Sequence[int],
                      n_bins: int = 10, strategy: str = "quantile"
                      ) -> Dict[str, np.ndarray]:
    """Observed frequency against predicted probability, per bin.

    ``strategy="quantile"`` puts an equal number of points in each bin, which
    is what you want for a diagnostic: uniform-width bins leave the extreme
    bins nearly empty, and those are precisely the bins where overconfidence
    lives and where a noisy estimate is most misleading.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    if p.size == 0:
        empty = np.zeros(0)
        return {"predicted": empty, "observed": empty, "counts": empty}

    if strategy == "quantile":
        edges = np.unique(np.percentile(p, np.linspace(0, 100, int(n_bins) + 1)))
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        edges = np.array([p.min(), p.max() + 1e-9])
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, edges.size - 2)

    predicted, observed, counts = [], [], []
    for bin_index in range(edges.size - 1):
        inside = index == bin_index
        if not inside.any():
            continue
        predicted.append(float(np.mean(p[inside])))
        observed.append(float(np.mean(y[inside] > 0.5)))
        counts.append(float(inside.sum()))
    return {"predicted": np.asarray(predicted), "observed": np.asarray(observed),
            "counts": np.asarray(counts)}


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int],
                               n_bins: int = 10) -> float:
    """Mean absolute gap between predicted and observed, weighted by bin size.

    Zero means perfectly calibrated.  Around 0.05 is respectable; above 0.15
    the numbers should not be presented as probabilities.

    >>> scores = [0.1] * 100 + [0.9] * 100
    >>> labels = [0] * 90 + [1] * 10 + [0] * 10 + [1] * 90
    >>> round(expected_calibration_error(scores, labels), 3)
    0.0
    """
    curve = reliability_curve(probabilities, labels, n_bins)
    if curve["counts"].size == 0:
        return float("nan")
    total = float(curve["counts"].sum())
    gaps = np.abs(curve["predicted"] - curve["observed"])
    return float(np.sum(gaps * curve["counts"]) / total)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probabilities: accuracy and calibration together.

    Unlike the calibration error it also penalises a *useless* model: a
    classifier that predicts the base rate for everything is perfectly
    calibrated and has a poor Brier score, which is the right verdict.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y)
    if not finite.any():
        return float("nan")
    return float(np.mean((p[finite] - (y[finite] > 0.5)) ** 2))


def calibration_report(probabilities: Sequence[float], labels: Sequence[int],
                       n_bins: int = 10) -> Dict[str, Any]:
    """Everything needed to say whether a confidence can be believed."""
    curve = reliability_curve(probabilities, labels, n_bins)
    error = expected_calibration_error(probabilities, labels, n_bins)
    worst = 0.0
    if curve["counts"].size:
        worst = float(np.max(np.abs(curve["predicted"] - curve["observed"])))
    return {
        "expected_calibration_error": error,
        "worst_bin_error": worst,
        "brier_score": brier_score(probabilities, labels),
        "n_bins": int(curve["counts"].size),
        "n_samples": int(np.sum(curve["counts"])),
        "curve": {"predicted": curve["predicted"].tolist(),
                  "observed": curve["observed"].tolist(),
                  "counts": curve["counts"].tolist()},
        "usable_as_probability": bool(np.isfinite(error) and error <= 0.15),
    }


def calibrate_catalog(catalog, calibrator: Calibrator,
                      key: str = "class_confidence") -> int:
    """Apply a calibrator to a catalog's confidences, in place.

    The raw value is kept in ``meta["raw_confidence"]``: a calibrated number
    is the one to act on, but the uncalibrated one is what the model actually
    said, and losing it makes the calibration impossible to check later.
    """
    updated = 0
    for source in catalog:
        raw = getattr(source, key, None)
        if raw is None or not np.isfinite(raw):
            continue
        source.meta["raw_confidence"] = float(raw)
        setattr(source, key, float(calibrator.transform([raw])[0]))
        updated += 1
    if hasattr(catalog, "meta"):
        catalog.meta["calibration"] = calibrator.to_dict()
    return updated
