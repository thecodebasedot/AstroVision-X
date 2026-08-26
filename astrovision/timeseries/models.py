"""Sequence models for light-curve classification.

Light curves are irregularly sampled sequences, which is exactly the shape
recurrent and attention models handle well.  Both accept the *time deltas*
alongside the fluxes, because when a measurement was taken carries as much
information as its value.

PyTorch is optional; a NumPy nearest-centroid classifier over the
variability statistics stands in when it is absent, so the capability
degrades rather than disappearing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import require, try_import
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.types import LightCurve
from ..ml.scaler import RobustScaler
from .features import variability_features

log = get_logger("timeseries.models")

#: Variability classes the sequence models predict.
VARIABLE_CLASSES: List[str] = [
    "non_variable", "periodic_pulsator", "eclipsing",
    "eruptive", "secular_trend", "stochastic",
]


def encode_curve(curve: LightCurve, length: int = 64) -> np.ndarray:
    """Resample a light curve to a fixed-length ``(length, 3)`` sequence.

    The channels are normalised flux, the time gap to the previous epoch,
    and a validity mask, so a padded short curve is unambiguous.
    """
    clean = curve.clean()
    out = np.zeros((int(length), 3), dtype=np.float32)
    if len(clean) == 0:
        return out
    flux = clean.normalized() - 1.0
    scale = float(np.std(flux)) if len(flux) > 1 else 0.0
    if scale > 1e-9:
        flux = flux / scale
    gaps = np.diff(clean.times, prepend=clean.times[0])
    baseline = max(clean.baseline, 1e-9)
    gaps = gaps / baseline

    n = min(len(clean), int(length))
    if len(clean) <= length:
        out[:n, 0] = flux[:n]
        out[:n, 1] = gaps[:n]
        out[:n, 2] = 1.0
    else:
        # Longer curves are interpolated onto a regular grid rather than
        # truncated, so the whole baseline stays represented.
        grid = np.linspace(clean.times[0], clean.times[-1], length)
        out[:, 0] = np.interp(grid, clean.times, flux)
        out[:, 1] = (grid[1] - grid[0]) / baseline
        out[:, 2] = 1.0
    return out


def build_sequence_model(torch, n_classes: int, architecture: str = "lstm",
                         hidden: int = 64, layers: int = 2, length: int = 64):
    """LSTM or Transformer encoder over an encoded light curve."""
    nn = torch.nn

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(3, hidden, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden * 2),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            output, _ = self.rnn(x)
            # Mean-pool over valid epochs only.
            mask = x[..., 2:3]
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.head(pooled)

    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(3, hidden)
            self.positions = nn.Parameter(torch.zeros(1, length, hidden))
            nn.init.trunc_normal_(self.positions, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 3,
                dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                                 enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

        def forward(self, x):
            tokens = self.project(x) + self.positions[:, :x.shape[1]]
            encoded = self.encoder(tokens, src_key_padding_mask=(x[..., 2] < 0.5))
            mask = x[..., 2:3]
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.head(pooled)

    return TransformerClassifier() if architecture == "transformer" else LSTMClassifier()


class NearestCentroidVariabilityClassifier:
    """NumPy fallback: nearest class centroid in variability-statistic space."""

    def __init__(self, classes: Optional[Sequence[str]] = None):
        self.classes = list(classes or VARIABLE_CLASSES)
        self.centroids_: Optional[np.ndarray] = None
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []

    @staticmethod
    def _features(curves: Sequence[LightCurve]) -> Tuple[np.ndarray, List[str]]:
        rows = [variability_features(c) for c in curves]
        names = list(rows[0]) if rows else []
        matrix = np.array([[row.get(k, np.nan) for k in names] for row in rows], dtype=float)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix, names

    def fit(self, curves: Sequence[LightCurve],
            labels: Sequence[str]) -> "NearestCentroidVariabilityClassifier":
        if len(curves) != len(labels) or not curves:
            raise ModelError("curves and labels must be non-empty and equal in length")
        matrix, names = self._features(curves)
        self.feature_names_ = names
        self.scaler_ = RobustScaler().fit(matrix)
        Z = self.scaler_.transform(matrix)
        labels = np.asarray(labels)
        self.centroids_ = np.vstack([
            Z[labels == c].mean(axis=0) if np.any(labels == c) else np.full(Z.shape[1], np.inf)
            for c in self.classes])
        return self

    def predict_proba(self, curves: Sequence[LightCurve]) -> np.ndarray:
        if self.centroids_ is None:
            raise NotFittedError("call fit before predict_proba")
        matrix, _ = self._features(curves)
        Z = self.scaler_.transform(matrix)
        distance = np.linalg.norm(Z[:, None, :] - self.centroids_[None, :, :], axis=2)
        distance = np.where(np.isfinite(distance), distance, 1e9)
        similarity = 1.0 / (1.0 + distance)
        return similarity / similarity.sum(axis=1, keepdims=True)

    def predict(self, curves: Sequence[LightCurve]) -> List[str]:
        return [self.classes[i] for i in np.argmax(self.predict_proba(curves), axis=1)]


class SequenceClassifier:
    """Trainable light-curve classifier (LSTM or Transformer)."""

    def __init__(self, architecture: str = "lstm", classes: Optional[Sequence[str]] = None,
                 length: int = 64, hidden: int = 64, layers: int = 2,
                 device: str = "cpu", random_state: int = 42):
        self.architecture = str(architecture)
        self.classes = list(classes or VARIABLE_CLASSES)
        self.length = int(length)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.device = device
        self.random_state = int(random_state)
        self.model = None
        self.history_: List[float] = []
        self._torch = None

    @property
    def available(self) -> bool:
        return try_import("torch") is not None

    def build(self) -> "SequenceClassifier":
        torch = require("torch", "the light-curve sequence classifier")
        self._torch = torch
        torch.manual_seed(self.random_state)
        self.model = build_sequence_model(torch, len(self.classes), self.architecture,
                                          self.hidden, self.layers,
                                          self.length).to(self.device)
        log.info("built %s light-curve classifier: %d classes, %.2fM parameters",
                 self.architecture, len(self.classes),
                 sum(p.numel() for p in self.model.parameters()) / 1e6)
        return self

    def _encode(self, curves: Sequence[LightCurve]) -> np.ndarray:
        return np.stack([encode_curve(c, self.length) for c in curves])

    def fit(self, curves: Sequence[LightCurve], labels: Sequence[str],
            epochs: int = 60, learning_rate: float = 2e-3, batch_size: int = 32,
            verbose: bool = True) -> List[float]:
        if self.model is None:
            self.build()
        torch = self._torch
        index = {c: i for i, c in enumerate(self.classes)}
        unknown = {l for l in labels if l not in index}
        if unknown:
            raise ModelError(f"labels not in the class set: {sorted(unknown)}")

        X = torch.from_numpy(self._encode(curves))
        y = torch.tensor([index[l] for l in labels], dtype=torch.long)
        counts = np.bincount(y.numpy(), minlength=len(self.classes)).astype(float)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / max(weights[weights > 0].mean(), 1e-9)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device))
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)

        self.model.train()
        self.history_ = []
        for epoch in range(int(epochs)):
            permutation = torch.randperm(len(X))
            total, batches = 0.0, 0
            for start in range(0, len(X), batch_size):
                batch = permutation[start:start + batch_size]
                if len(batch) < 2:
                    continue
                loss = criterion(self.model(X[batch].to(self.device)),
                                 y[batch].to(self.device))
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimiser.step()
                total += float(loss.item())
                batches += 1
            self.history_.append(total / max(batches, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                log.info("epoch %3d/%d loss=%.4f", epoch + 1, epochs, self.history_[-1])
        self.model.eval()
        return self.history_

    def predict_proba(self, curves: Sequence[LightCurve]) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("build or load the classifier before predicting")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(self._encode(curves)).to(self.device)
            return torch.softmax(self.model(X), dim=1).cpu().numpy()

    def predict(self, curves: Sequence[LightCurve]) -> List[str]:
        return [self.classes[i] for i in np.argmax(self.predict_proba(curves), axis=1)]

    def save(self, path: str) -> str:
        if self.model is None:
            raise NotFittedError("build or load the classifier before saving")
        torch = self._torch or require("torch", "saving the classifier")
        torch.save({"state_dict": self.model.state_dict(), "classes": self.classes,
                    "architecture": self.architecture, "length": self.length,
                    "hidden": self.hidden, "layers": self.layers}, path)
        return path

    def load(self, path: str) -> "SequenceClassifier":
        torch = require("torch", "the light-curve sequence classifier")
        self._torch = torch
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = list(payload["classes"])
        self.architecture = payload.get("architecture", self.architecture)
        self.length = int(payload.get("length", self.length))
        self.hidden = int(payload.get("hidden", self.hidden))
        self.layers = int(payload.get("layers", self.layers))
        self.build()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return self
