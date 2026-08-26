"""Autoencoders for anomaly detection.

An autoencoder trained on ordinary objects learns to reconstruct them.
Anything it reconstructs badly is, by construction, unlike what it was
trained on -- which is exactly the question a novelty search asks.  Two
implementations are provided: a linear (PCA-equivalent) one that always
works, and a non-linear PyTorch one when the deep extra is installed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..core.backend import require, try_import
from ..core.exceptions import NotFittedError
from ..core.logging import get_logger
from .scaler import PCA, RobustScaler

log = get_logger("ml.autoencoder")


class LinearAutoencoder:
    """PCA-based reconstruction error -- a linear autoencoder in closed form.

    It has no hyperparameters to tune, cannot fail to converge, and gives a
    principled baseline: an object is anomalous when it does not lie in the
    low-dimensional subspace the ordinary population occupies.
    """

    def __init__(self, latent_dim: int = 8, scale: bool = True):
        self.latent_dim = int(latent_dim)
        self.scale = bool(scale)
        self.scaler_: Optional[RobustScaler] = None
        self.pca_: Optional[PCA] = None
        self.baseline_: float = float("nan")

    def fit(self, X: np.ndarray) -> "LinearAutoencoder":
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.scaler_ = RobustScaler().fit(data) if self.scale else None
        Z = self.scaler_.transform(data) if self.scaler_ else np.nan_to_num(data)
        self.pca_ = PCA(min(self.latent_dim, *Z.shape)).fit(Z)
        errors = self._errors(Z)
        self.baseline_ = float(np.median(errors)) if errors.size else 0.0
        return self

    def _errors(self, Z: np.ndarray) -> np.ndarray:
        latent = self.pca_.transform(Z)
        reconstruction = self.pca_.inverse_transform(latent)
        return np.sqrt(np.mean((Z - reconstruction) ** 2, axis=1))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise NotFittedError("call fit before reconstruction_error")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        Z = self.scaler_.transform(data) if self.scaler_ else np.nan_to_num(data)
        return self._errors(Z)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: reconstruction error relative to the training median."""
        errors = self.reconstruction_error(X)
        if not np.isfinite(self.baseline_) or self.baseline_ <= 1e-12:
            return errors
        return errors / self.baseline_

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Latent representation, usable as a compact embedding."""
        if self.pca_ is None:
            raise NotFittedError("call fit before encode")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        Z = self.scaler_.transform(data) if self.scaler_ else np.nan_to_num(data)
        return self.pca_.transform(Z)


def build_autoencoder(torch, input_dim: int, latent_dim: int = 8,
                      hidden: Sequence[int] = (64, 32)):
    """A symmetric fully-connected autoencoder."""
    nn = torch.nn
    dims = [int(input_dim)] + [int(h) for h in hidden]

    encoder_layers: List = []
    for a, b in zip(dims[:-1], dims[1:]):
        encoder_layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.LeakyReLU(0.1)]
    encoder_layers.append(nn.Linear(dims[-1], int(latent_dim)))

    decoder_dims = [int(latent_dim)] + list(reversed(dims[1:]))
    decoder_layers: List = []
    for a, b in zip(decoder_dims[:-1], decoder_dims[1:]):
        decoder_layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.LeakyReLU(0.1)]
    decoder_layers.append(nn.Linear(decoder_dims[-1], int(input_dim)))

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            latent = self.encoder(x)
            return self.decoder(latent), latent

    return Autoencoder()


class DeepAutoencoder:
    """Non-linear autoencoder for tabular features (requires PyTorch)."""

    def __init__(self, latent_dim: int = 8, hidden: Sequence[int] = (64, 32),
                 epochs: int = 200, learning_rate: float = 1e-3,
                 batch_size: int = 64, device: str = "cpu", random_state: int = 42):
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.device = device
        self.random_state = int(random_state)
        self.model = None
        self.scaler_: Optional[RobustScaler] = None
        self.baseline_: float = float("nan")
        self.history_: List[float] = []
        self._torch = None

    @property
    def available(self) -> bool:
        return try_import("torch") is not None

    def fit(self, X: np.ndarray, verbose: bool = False) -> "DeepAutoencoder":
        torch = require("torch", "the deep autoencoder")
        self._torch = torch
        torch.manual_seed(self.random_state)

        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.scaler_ = RobustScaler().fit(data)
        Z = self.scaler_.transform(data).astype(np.float32)
        if len(Z) < 4:
            raise ValueError("the deep autoencoder needs at least 4 samples")

        self.model = build_autoencoder(torch, Z.shape[1], self.latent_dim,
                                       self.hidden).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        tensor = torch.from_numpy(Z).to(self.device)
        batch = min(self.batch_size, max(2, len(Z)))

        self.model.train()
        self.history_ = []
        for epoch in range(self.epochs):
            permutation = torch.randperm(len(Z))
            total, batches = 0.0, 0
            for start in range(0, len(Z), batch):
                index = permutation[start:start + batch]
                if len(index) < 2:      # BatchNorm needs more than one sample
                    continue
                x = tensor[index]
                reconstruction, _ = self.model(x)
                loss = criterion(reconstruction, x)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1
            self.history_.append(total / max(batches, 1))
            if verbose and epoch % max(1, self.epochs // 10) == 0:
                log.info("autoencoder epoch %3d/%d loss=%.5f",
                         epoch + 1, self.epochs, self.history_[-1])

        self.model.eval()
        errors = self.reconstruction_error(data)
        self.baseline_ = float(np.median(errors)) if errors.size else 0.0
        return self

    def _forward(self, X: np.ndarray):
        if self.model is None:
            raise NotFittedError("call fit before using the autoencoder")
        torch = self._torch
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        Z = self.scaler_.transform(data).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(Z).to(self.device)
            reconstruction, latent = self.model(tensor)
            return Z, reconstruction.cpu().numpy(), latent.cpu().numpy()

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        Z, reconstruction, _ = self._forward(X)
        return np.sqrt(np.mean((Z - reconstruction) ** 2, axis=1))

    def score(self, X: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X)
        if not np.isfinite(self.baseline_) or self.baseline_ <= 1e-12:
            return errors
        return errors / self.baseline_

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[2]
