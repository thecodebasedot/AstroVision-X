"""Isolation Forest, implemented in NumPy.

Outliers are easier to *isolate*: a random split is more likely to cut them
off from the bulk early, so their average path length through a random tree
is shorter.  That is the whole idea, and it needs no distance metric --
which matters for astronomical feature vectors whose columns are on wildly
different scales and are often missing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.exceptions import NotFittedError
from ..core.logging import get_logger

log = get_logger("ml.isolation_forest")


def _average_path_length(n: int) -> float:
    """Expected path length of an unsuccessful BST search over ``n`` points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler) - 2.0 * (n - 1) / n


class _IsolationTree:
    """One randomly-split isolation tree, stored as flat arrays."""

    __slots__ = ("feature", "threshold", "left", "right", "size", "depth_limit", "n_nodes")

    def __init__(self, depth_limit: int):
        self.depth_limit = int(depth_limit)
        self.feature: List[int] = []
        self.threshold: List[float] = []
        self.left: List[int] = []
        self.right: List[int] = []
        self.size: List[int] = []
        self.n_nodes = 0

    def _new_node(self) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(0)
        self.n_nodes += 1
        return self.n_nodes - 1

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "_IsolationTree":
        self._build(X, 0, rng)
        return self

    def _build(self, X: np.ndarray, depth: int, rng: np.random.Generator) -> int:
        node = self._new_node()
        self.size[node] = len(X)
        if depth >= self.depth_limit or len(X) <= 1:
            return node

        # Split only on features that actually vary in this subset.
        spread = X.max(axis=0) - X.min(axis=0)
        usable = np.nonzero(spread > 1e-12)[0]
        if usable.size == 0:
            return node
        feature = int(rng.choice(usable))
        low, high = float(X[:, feature].min()), float(X[:, feature].max())
        threshold = float(rng.uniform(low, high))

        mask = X[:, feature] < threshold
        if mask.all() or (~mask).all():
            return node
        self.feature[node] = feature
        self.threshold[node] = threshold
        self.left[node] = self._build(X[mask], depth + 1, rng)
        self.right[node] = self._build(X[~mask], depth + 1, rng)
        return node

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """Path length for each row, with the BST correction at leaves."""
        lengths = np.zeros(len(X), dtype=float)
        indices = np.zeros(len(X), dtype=int)
        active = np.ones(len(X), dtype=bool)
        depth = 0
        while active.any() and depth <= self.depth_limit + 1:
            nodes = indices[active]
            features = np.array([self.feature[n] for n in nodes])
            leaves = features < 0
            if leaves.any():
                sizes = np.array([self.size[n] for n in nodes[leaves]], dtype=float)
                target = np.nonzero(active)[0][leaves]
                lengths[target] = depth + np.array([_average_path_length(int(s)) for s in sizes])
                active[target] = False
            if not active.any():
                break
            nodes = indices[active]
            features = np.array([self.feature[n] for n in nodes])
            thresholds = np.array([self.threshold[n] for n in nodes])
            values = X[active, features]
            go_left = values < thresholds
            indices[active] = np.where(
                go_left,
                np.array([self.left[n] for n in nodes]),
                np.array([self.right[n] for n in nodes]))
            depth += 1
        lengths[active] = depth
        return lengths


class IsolationForest:
    """Ensemble of isolation trees producing an outlier score in ``[0, 1]``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 2)), np.array([[9.0, 9.0]])])
    >>> scores = IsolationForest(n_estimators=64, random_state=0).fit(X).score(X)
    >>> int(np.argmax(scores))
    200
    """

    def __init__(self, n_estimators: int = 128, max_samples: int = 256,
                 random_state: int = 42, contamination: float = 0.02):
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.random_state = int(random_state)
        self.contamination = float(contamination)
        self.trees_: List[_IsolationTree] = []
        self.subsample_size_: int = 0
        self.threshold_: float = float("nan")

    def fit(self, X: np.ndarray) -> "IsolationForest":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        n_samples = len(data)
        if n_samples == 0:
            raise ValueError("cannot fit an IsolationForest on an empty array")

        rng = np.random.default_rng(self.random_state)
        self.subsample_size_ = int(min(self.max_samples, n_samples))
        depth_limit = max(1, int(np.ceil(np.log2(max(self.subsample_size_, 2)))))

        self.trees_ = []
        for _ in range(self.n_estimators):
            index = rng.choice(n_samples, self.subsample_size_,
                               replace=self.subsample_size_ > n_samples)
            self.trees_.append(_IsolationTree(depth_limit).fit(data[index], rng))

        scores = self.score(data)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        log.debug("isolation forest: %d trees, subsample %d, threshold %.3f",
                  self.n_estimators, self.subsample_size_, self.threshold_)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in ``[0, 1]``; larger means more anomalous."""
        if not self.trees_:
            raise NotFittedError("call fit before score")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        lengths = np.mean([tree.path_length(data) for tree in self.trees_], axis=0)
        normaliser = _average_path_length(self.subsample_size_)
        if normaliser <= 0:
            return np.zeros(len(data))
        return np.power(2.0, -lengths / normaliser)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``True`` for rows flagged as outliers at the fitted contamination."""
        return self.score(X) >= self.threshold_

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).score(X)
