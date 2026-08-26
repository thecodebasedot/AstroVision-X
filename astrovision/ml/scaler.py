"""Feature scaling and imputation.

Astronomical feature tables are full of NaNs: a Sersic fit fails, a
morphology statistic is undefined for a point source, a colour needs a band
that was not observed.  Every model here therefore imputes before it scales,
and records which features were missing so that can be reported.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.exceptions import NotFittedError
from ..core.numeric import MAD_TO_SIGMA


class RobustScaler:
    """Centre on the median and scale by the MAD.

    Outliers are the *signal* in an anomaly search, so a mean/std scaler --
    whose statistics the outliers themselves distort -- is the wrong tool.
    """

    def __init__(self, clip: Optional[float] = 8.0):
        self.centre_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.fill_: Optional[np.ndarray] = None
        self.clip = clip
        self.missing_fraction_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        finite = np.isfinite(data)
        self.missing_fraction_ = 1.0 - finite.mean(axis=0)
        self.fill_ = np.array([
            np.median(data[finite[:, j], j]) if finite[:, j].any() else 0.0
            for j in range(data.shape[1])
        ])
        filled = self._impute(data)
        self.centre_ = np.median(filled, axis=0)
        mad = np.median(np.abs(filled - self.centre_), axis=0)
        scale = MAD_TO_SIGMA * mad
        # A feature with zero spread would divide by zero; fall back to the
        # standard deviation, then to unity.
        fallback = np.std(filled, axis=0)
        scale = np.where(scale > 1e-9, scale, fallback)
        self.scale_ = np.where(scale > 1e-9, scale, 1.0)
        return self

    def _impute(self, data: np.ndarray) -> np.ndarray:
        if self.fill_ is None:
            raise NotFittedError("call fit before transform")
        out = np.array(data, dtype=float, copy=True)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(self.fill_, np.nonzero(bad)[1])
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centre_ is None:
            raise NotFittedError("call fit before transform")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        scaled = (self._impute(data) - self.centre_) / self.scale_
        if self.clip is not None:
            scaled = np.clip(scaled, -self.clip, self.clip)
        return scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.centre_ is None:
            raise NotFittedError("call fit before inverse_transform")
        return np.asarray(X, dtype=float) * self.scale_ + self.centre_


class PCA:
    """Principal component analysis via the SVD, for embedding compression."""

    def __init__(self, n_components: int = 8):
        self.n_components = int(n_components)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.mean_ = data.mean(axis=0)
        centred = data - self.mean_
        n_components = min(self.n_components, *centred.shape)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        self.components_ = vt[:n_components]
        variance = (singular ** 2) / max(len(centred) - 1, 1)
        self.explained_variance_ = variance[:n_components]
        total = variance.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0
            else np.zeros_like(self.explained_variance_))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise NotFittedError("call fit before transform")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return (data - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise NotFittedError("call fit before inverse_transform")
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
