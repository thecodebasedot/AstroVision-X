"""Gradient-boosted decision trees for tabular prediction.

Boosted trees are the workhorse for the tabular half of this platform --
predicting a class or a continuous quantity from measured features.  The
implementation prefers XGBoost, then scikit-learn, and otherwise falls back
to a self-contained NumPy version so the capability is never simply absent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.backend import try_import
from ..core.exceptions import NotFittedError
from ..core.logging import get_logger
from ..core.numeric import softmax
from .scaler import RobustScaler

log = get_logger("ml.gbdt")


class _RegressionStumpTree:
    """A shallow regression tree fitted by exhaustive split search."""

    __slots__ = ("max_depth", "min_samples", "feature", "threshold", "value",
                 "left", "right", "n_nodes")

    def __init__(self, max_depth: int = 3, min_samples: int = 5):
        self.max_depth = int(max_depth)
        self.min_samples = int(min_samples)
        self.feature: List[int] = []
        self.threshold: List[float] = []
        self.value: List[float] = []
        self.left: List[int] = []
        self.right: List[int] = []
        self.n_nodes = 0

    def _node(self, value: float) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.value.append(float(value))
        self.left.append(-1)
        self.right.append(-1)
        self.n_nodes += 1
        return self.n_nodes - 1

    def fit(self, X: np.ndarray, residual: np.ndarray,
            n_bins: int = 24) -> "_RegressionStumpTree":
        self._build(X, residual, 0, n_bins)
        return self

    def _build(self, X: np.ndarray, residual: np.ndarray, depth: int,
               n_bins: int) -> int:
        node = self._node(float(residual.mean()) if residual.size else 0.0)
        if depth >= self.max_depth or len(residual) < 2 * self.min_samples:
            return node

        best = (0.0, -1, 0.0)     # (gain, feature, threshold)
        parent_sse = float(((residual - residual.mean()) ** 2).sum())
        for j in range(X.shape[1]):
            column = X[:, j]
            finite = column[np.isfinite(column)]
            if finite.size < 2 * self.min_samples:
                continue
            # Quantile candidate splits: far cheaper than all unique values
            # and just as effective for boosting.
            candidates = np.unique(np.quantile(finite, np.linspace(0.05, 0.95, n_bins)))
            for threshold in candidates:
                mask = column <= threshold
                left_n, right_n = int(mask.sum()), int((~mask).sum())
                if left_n < self.min_samples or right_n < self.min_samples:
                    continue
                left, right = residual[mask], residual[~mask]
                sse = (float(((left - left.mean()) ** 2).sum()) +
                       float(((right - right.mean()) ** 2).sum()))
                gain = parent_sse - sse
                if gain > best[0]:
                    best = (gain, j, float(threshold))

        if best[1] < 0:
            return node
        _, feature, threshold = best
        mask = X[:, feature] <= threshold
        self.feature[node] = feature
        self.threshold[node] = threshold
        self.left[node] = self._build(X[mask], residual[mask], depth + 1, n_bins)
        self.right[node] = self._build(X[~mask], residual[~mask], depth + 1, n_bins)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        index = np.zeros(len(X), dtype=int)
        for _ in range(self.max_depth + 1):
            features = np.array([self.feature[i] for i in index])
            leaf = features < 0
            if leaf.any():
                out[leaf] = [self.value[i] for i in index[leaf]]
            if leaf.all():
                break
            active = ~leaf
            nodes = index[active]
            thresholds = np.array([self.threshold[i] for i in nodes])
            values = X[active, np.array([self.feature[i] for i in nodes])]
            index[active] = np.where(
                values <= thresholds,
                np.array([self.left[i] for i in nodes]),
                np.array([self.right[i] for i in nodes]))
        out[np.array([self.feature[i] for i in index]) < 0] = [
            self.value[i] for i in index[np.array([self.feature[i] for i in index]) < 0]]
        return out


class GradientBoostedClassifier:
    """Multi-class boosted trees with a NumPy fallback.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(4, 1, (60, 3))])
    >>> y = np.array([0] * 60 + [1] * 60)
    >>> model = GradientBoostedClassifier(n_estimators=20, backend="numpy").fit(X, y)
    >>> float((model.predict(X) == y).mean()) > 0.9
    True
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, backend: str = "auto", random_state: int = 42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.backend = str(backend)
        self.random_state = int(random_state)
        self.classes_: Optional[np.ndarray] = None
        self.scaler_: Optional[RobustScaler] = None
        self._impl = None
        self._trees: List[List[_RegressionStumpTree]] = []
        self._init_scores: Optional[np.ndarray] = None
        self.backend_used_: str = "none"
        self.n_features_: int = 0

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if try_import("xgboost") is not None:
            return "xgboost"
        if try_import("sklearn") is not None:
            return "sklearn"
        return "numpy"

    def fit(self, X: np.ndarray, y: Sequence) -> "GradientBoostedClassifier":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        self.scaler_ = RobustScaler(clip=None).fit(data)
        Z = self.scaler_.transform(data)
        self.n_features_ = int(Z.shape[1])

        backend = self._resolve_backend()
        if backend == "xgboost":
            xgboost = try_import("xgboost")
            self._impl = xgboost.XGBClassifier(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, random_state=self.random_state,
                verbosity=0, eval_metric="mlogloss")
            self._impl.fit(Z, encoded)
        elif backend == "sklearn":
            ensemble = try_import("sklearn.ensemble")
            self._impl = ensemble.GradientBoostingClassifier(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, random_state=self.random_state)
            self._impl.fit(Z, encoded)
        else:
            self._fit_numpy(Z, encoded)
        self.backend_used_ = backend
        log.debug("gradient boosting fitted with the %s backend on %d samples",
                  backend, len(Z))
        return self

    def _fit_numpy(self, Z: np.ndarray, encoded: np.ndarray) -> None:
        """One-vs-rest logistic boosting, the classic multi-class formulation."""
        n_classes = len(self.classes_)
        one_hot = np.zeros((len(Z), n_classes))
        one_hot[np.arange(len(Z)), encoded] = 1.0
        prior = np.clip(one_hot.mean(axis=0), 1e-6, 1 - 1e-6)
        self._init_scores = np.log(prior)
        scores = np.tile(self._init_scores, (len(Z), 1))
        self._trees = []
        for _ in range(self.n_estimators):
            probability = np.apply_along_axis(softmax, 1, scores)
            round_trees: List[_RegressionStumpTree] = []
            for k in range(n_classes):
                residual = one_hot[:, k] - probability[:, k]
                tree = _RegressionStumpTree(self.max_depth).fit(Z, residual)
                scores[:, k] += self.learning_rate * tree.predict(Z)
                round_trees.append(tree)
            self._trees.append(round_trees)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise NotFittedError("call fit before predict_proba")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        Z = self.scaler_.transform(data)
        if self._impl is not None:
            return np.asarray(self._impl.predict_proba(Z), dtype=float)
        scores = np.tile(self._init_scores, (len(Z), 1))
        for round_trees in self._trees:
            for k, tree in enumerate(round_trees):
                scores[:, k] += self.learning_rate * tree.predict(Z)
        return np.apply_along_axis(softmax, 1, scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def feature_importance(self, names: Optional[Sequence[str]] = None
                           ) -> Dict[str, float]:
        """Relative importance per feature, summed over all trees."""
        if self._impl is not None and hasattr(self._impl, "feature_importances_"):
            importance = np.asarray(self._impl.feature_importances_, dtype=float)
        elif self._trees:
            # Size from the fitted input, not from the splits actually used:
            # a feature no tree ever split on has zero importance, and
            # saying so is more informative than omitting it.
            importance = np.zeros(max(self.n_features_, 1))
            for round_trees in self._trees:
                for tree in round_trees:
                    for feature in tree.feature:
                        if feature >= 0:
                            importance[feature] += 1.0
        else:
            raise NotFittedError("call fit before feature_importance")
        total = importance.sum()
        if total > 0:
            importance = importance / total
        if names is not None:
            labels = list(names)
            if len(labels) != len(importance):
                raise ValueError(
                    f"expected {len(importance)} feature names, got {len(labels)}")
        else:
            labels = [f"f{i}" for i in range(len(importance))]
        return dict(sorted(zip(labels, importance.tolist()), key=lambda kv: -kv[1]))
