"""Clustering of catalog embeddings.

Grouping the catalog in feature space finds populations the classifier was
never told about -- a globular cluster's member stars, a galaxy group, or a
family of similar-looking artefacts.  Points that join no cluster are
themselves interesting, which is why DBSCAN-style noise labels are kept.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.exceptions import NotFittedError
from ..core.logging import get_logger

log = get_logger("ml.clustering")


class KMeans:
    """K-means with k-means++ initialisation."""

    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 tol: float = 1e-5, n_init: int = 4, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.centres_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")

    def _init_centres(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++: seed centres far apart so the fit does not collapse."""
        n = len(X)
        centres = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            distance = np.min(
                np.stack([np.sum((X - c) ** 2, axis=1) for c in centres]), axis=0)
            total = distance.sum()
            if total <= 0:
                centres.append(X[rng.integers(n)])
                continue
            centres.append(X[rng.choice(n, p=distance / total)])
        return np.array(centres, dtype=float)

    def fit(self, X: np.ndarray) -> "KMeans":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if len(data) == 0:
            raise ValueError("cannot cluster an empty array")
        self.n_clusters = max(1, min(self.n_clusters, len(data)))
        rng = np.random.default_rng(self.random_state)

        for _ in range(max(1, self.n_init)):
            centres = self._init_centres(data, rng)
            labels = np.zeros(len(data), dtype=int)
            for _ in range(self.max_iter):
                distance = np.linalg.norm(data[:, None, :] - centres[None, :, :], axis=2)
                new_labels = np.argmin(distance, axis=1)
                new_centres = np.array([
                    data[new_labels == k].mean(axis=0) if np.any(new_labels == k)
                    else data[rng.integers(len(data))]
                    for k in range(self.n_clusters)])
                shift = float(np.linalg.norm(new_centres - centres))
                centres, labels = new_centres, new_labels
                if shift < self.tol:
                    break
            inertia = float(np.sum(
                (data - centres[labels]) ** 2))
            if inertia < self.inertia_:
                self.inertia_, self.centres_, self.labels_ = inertia, centres, labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centres_ is None:
            raise NotFittedError("call fit before predict")
        data = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return np.argmin(np.linalg.norm(data[:, None, :] - self.centres_[None, :, :], axis=2),
                         axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


class DBSCAN:
    """Density-based clustering; label ``-1`` marks noise points."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        data = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = len(data)
        labels = np.full(n, -1, dtype=int)
        if n == 0:
            self.labels_ = labels
            return self

        distance = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        neighbours = [np.nonzero(row <= self.eps)[0] for row in distance]
        core = np.array([len(nb) >= self.min_samples for nb in neighbours])

        cluster = 0
        visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i] or not core[i]:
                continue
            # Flood-fill the density-connected component from this core point.
            queue = [i]
            visited[i] = True
            labels[i] = cluster
            while queue:
                point = queue.pop()
                for j in neighbours[point]:
                    if labels[j] == -1:
                        labels[j] = cluster
                    if not visited[j]:
                        visited[j] = True
                        if core[j]:
                            queue.append(int(j))
            cluster += 1
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


class HDBSCANLite:
    """Hierarchical density clustering over a minimum spanning tree.

    Full HDBSCAN* selects clusters by excess of mass; this keeps the useful
    part -- mutual-reachability distances, which stop a thin bridge of noise
    from merging two real groups -- and cuts the tree at a stability-selected
    scale.  Unlike plain DBSCAN it needs no ``eps``.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = None):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples or min_cluster_size)
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "HDBSCANLite":
        data = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = len(data)
        if n < max(2, self.min_cluster_size):
            self.labels_ = np.full(n, -1, dtype=int)
            self.probabilities_ = np.zeros(n)
            return self

        distance = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        k = min(self.min_samples, n - 1)
        core_distance = np.sort(distance, axis=1)[:, k]
        # Mutual reachability inflates distances in sparse regions, which is
        # what makes the result robust to varying density.
        reach = np.maximum(distance, np.maximum(core_distance[:, None],
                                                core_distance[None, :]))

        parent, edges = self._minimum_spanning_tree(reach)
        lengths = np.array([w for _, _, w in edges], dtype=float)
        if lengths.size == 0:
            self.labels_ = np.full(n, -1, dtype=int)
            self.probabilities_ = np.zeros(n)
            return self

        best_labels = np.full(n, -1, dtype=int)
        best_score = -1.0
        for quantile in np.linspace(0.35, 0.95, 13):
            cut = float(np.quantile(lengths, quantile))
            labels = self._components(n, edges, cut)
            score = self._stability(labels)
            if score > best_score:
                best_score, best_labels = score, labels

        self.labels_ = best_labels
        # Membership strength: how far inside its cluster a point sits.
        probabilities = np.zeros(n)
        for value in set(best_labels) - {-1}:
            member = best_labels == value
            local = core_distance[member]
            if local.size:
                worst = float(local.max())
                probabilities[member] = 1.0 - (local / worst if worst > 0 else 0.0)
        self.probabilities_ = probabilities
        return self

    @staticmethod
    def _minimum_spanning_tree(weights: np.ndarray):
        """Prim's algorithm on a dense weight matrix."""
        n = len(weights)
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        best = weights[0].copy()
        parent = np.zeros(n, dtype=int)
        edges: List[Tuple[int, int, float]] = []
        for _ in range(n - 1):
            candidate = np.where(in_tree, np.inf, best)
            j = int(np.argmin(candidate))
            if not np.isfinite(candidate[j]):
                break
            in_tree[j] = True
            edges.append((int(parent[j]), j, float(best[j])))
            improve = weights[j] < best
            parent = np.where(improve, j, parent)
            best = np.minimum(best, weights[j])
        return parent, edges

    def _components(self, n: int, edges, cut: float) -> np.ndarray:
        """Connected components of the MST after removing long edges."""
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b, weight in edges:
            if weight <= cut:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

        roots: Dict[int, List[int]] = {}
        for i in range(n):
            roots.setdefault(find(i), []).append(i)
        labels = np.full(n, -1, dtype=int)
        cluster = 0
        for members in roots.values():
            if len(members) >= self.min_cluster_size:
                labels[members] = cluster
                cluster += 1
        return labels

    @staticmethod
    def _stability(labels: np.ndarray) -> float:
        """Prefer partitions that assign many points to few, balanced clusters."""
        clustered = labels >= 0
        if not clustered.any():
            return 0.0
        sizes = np.bincount(labels[clustered])
        sizes = sizes[sizes > 0]
        if sizes.size < 1:
            return 0.0
        fraction = float(clustered.mean())
        balance = float(sizes.min() / sizes.max()) if sizes.size > 1 else 1.0
        return fraction * (0.5 + 0.5 * balance) / (1.0 + 0.05 * sizes.size)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


def cluster(X: np.ndarray, method: str = "kmeans", **kwargs) -> Dict[str, np.ndarray]:
    """Run a named clustering algorithm; returns labels and diagnostics."""
    method = str(method).lower()
    if method == "kmeans":
        model = KMeans(n_clusters=kwargs.get("n_clusters", 8),
                       random_state=kwargs.get("random_state", 42))
        labels = model.fit_predict(X)
        return {"labels": labels, "centres": model.centres_,
                "inertia": np.array([model.inertia_])}
    if method == "dbscan":
        model = DBSCAN(eps=kwargs.get("eps", 0.5),
                       min_samples=kwargs.get("min_cluster_size", 5))
        return {"labels": model.fit_predict(X)}
    if method == "hdbscan":
        model = HDBSCANLite(min_cluster_size=kwargs.get("min_cluster_size", 5))
        labels = model.fit_predict(X)
        return {"labels": labels, "probabilities": model.probabilities_}
    raise ValueError(f"unknown clustering method '{method}'")


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient, ignoring noise points."""
    data = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return float("nan")
    data, labels = data[mask], labels[mask]
    distance = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    scores = []
    for i in range(len(data)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = float(distance[i, same].mean())
        b = min(float(distance[i, labels == other].mean())
                for other in set(labels) if other != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else float("nan")
