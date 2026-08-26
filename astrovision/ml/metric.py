"""Similarity search over catalog embeddings.

"Show me everything that looks like this" is one of the most useful
operations in a survey: it turns a single interesting object into a sample.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import NotFittedError
from ..core.numeric import normalise_unit
from ..core.types import Source, SourceCatalog


class SimilaritySearch:
    """Cosine nearest-neighbour lookup over an embedding matrix.

    >>> import numpy as np
    >>> X = np.eye(4)
    >>> search = SimilaritySearch().fit(X)
    >>> search.query(X[2], k=1)[0][0]
    2
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = str(metric)
        self.matrix_: Optional[np.ndarray] = None
        self.ids_: Optional[List[int]] = None

    def fit(self, embeddings: np.ndarray,
            ids: Optional[Sequence[int]] = None) -> "SimilaritySearch":
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self.matrix_ = normalise_unit(matrix, axis=1) if self.metric == "cosine" else matrix
        self.ids_ = list(ids) if ids is not None else list(range(len(matrix)))
        return self

    @classmethod
    def from_catalog(cls, catalog: SourceCatalog,
                     metric: str = "cosine") -> "SimilaritySearch":
        matrix = catalog.embeddings()
        if matrix is None:
            raise ValueError("every source needs an embedding before indexing")
        return cls(metric).fit(matrix, [s.id for s in catalog])

    def query(self, vector: np.ndarray, k: int = 5,
              exclude_self: bool = False) -> Tuple[List[int], np.ndarray]:
        """Return ``(ids, similarities)`` for the ``k`` closest entries."""
        if self.matrix_ is None:
            raise NotFittedError("call fit before query")
        query = np.nan_to_num(np.asarray(vector, dtype=float).ravel(), nan=0.0)
        if self.metric == "cosine":
            query = normalise_unit(query.reshape(1, -1))[0]
            similarity = self.matrix_ @ query
        else:
            similarity = -np.linalg.norm(self.matrix_ - query, axis=1)
        order = np.argsort(similarity)[::-1]
        if exclude_self and order.size and np.isclose(similarity[order[0]],
                                                      similarity.max()):
            order = order[1:]
        order = order[:int(k)]
        return [self.ids_[i] for i in order], similarity[order]

    def neighbours(self, k: int = 5) -> np.ndarray:
        """``(N, k)`` matrix of each entry's nearest neighbours, excluding itself."""
        if self.matrix_ is None:
            raise NotFittedError("call fit before neighbours")
        if self.metric == "cosine":
            similarity = self.matrix_ @ self.matrix_.T
        else:
            similarity = -np.linalg.norm(
                self.matrix_[:, None, :] - self.matrix_[None, :, :], axis=2)
        np.fill_diagonal(similarity, -np.inf)
        return np.argsort(similarity, axis=1)[:, ::-1][:, :int(k)]

    def knn_distance(self, k: int = 8) -> np.ndarray:
        """Mean distance to the ``k`` nearest neighbours -- a novelty score.

        An object with no close analogues in the catalog is, by definition,
        one nobody has seen before in this field.
        """
        if self.matrix_ is None:
            raise NotFittedError("call fit before knn_distance")
        n = len(self.matrix_)
        if n < 2:
            return np.zeros(n)
        distance = np.linalg.norm(
            self.matrix_[:, None, :] - self.matrix_[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        k = int(min(max(k, 1), n - 1))
        return np.sort(distance, axis=1)[:, :k].mean(axis=1)


def find_similar(catalog: SourceCatalog, source: Source, k: int = 5
                 ) -> List[Tuple[int, float]]:
    """Convenience wrapper: the ``k`` sources most like ``source``."""
    if source.embedding is None:
        raise ValueError("the query source has no embedding")
    search = SimilaritySearch.from_catalog(catalog)
    ids, similarity = search.query(source.embedding, k + 1)
    return [(i, float(s)) for i, s in zip(ids, similarity) if i != source.id][:k]
