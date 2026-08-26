"""Image classifiers for postage stamps: a small CNN and a compact ViT.

Both consume a fixed-size cutout and predict an object class.  They also
expose their penultimate representation, which is the *embedding* the
anomaly and similarity stages use -- a learned descriptor is far more
sensitive to unusual structure than any hand-crafted summary.

PyTorch is optional; this module imports without it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..core.backend import require, try_import
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, pad_or_crop
from ..core.types import ObjectClass, SourceCatalog
from ..preprocess.normalize import asinh_stretch

log = get_logger("ml.cnn")

#: Default class set for the stamp classifier.
STAMP_CLASSES: List[ObjectClass] = [
    ObjectClass.STAR,
    ObjectClass.GALAXY,
    ObjectClass.NEBULA,
    ObjectClass.STAR_CLUSTER,
    ObjectClass.ARTIFACT,
]


def build_cnn(torch, n_classes: int, width: int = 32, cutout: int = 48):
    """A compact residual CNN sized for astronomical postage stamps."""
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
            )
            self.shortcut = (nn.Sequential() if stride == 1 and cin == cout
                             else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride,
                                                          bias=False),
                                                nn.BatchNorm2d(cout)))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.body(x) + self.shortcut(x))

    class StampCNN(nn.Module):
        embedding_dim = width * 4

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, width, 5, padding=2, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                Block(width, width),
                Block(width, width * 2, stride=2),
                Block(width * 2, width * 2),
                Block(width * 2, width * 4, stride=2),
                Block(width * 4, width * 4),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(width * 4, n_classes)

        def embed(self, x):
            return self.features(x).flatten(1)

        def forward(self, x):
            return self.head(self.embed(x))

    return StampCNN()


def build_vit(torch, n_classes: int, cutout: int = 48, patch: int = 8,
              dim: int = 96, depth: int = 4, heads: int = 4):
    """A small Vision Transformer over stamp patches.

    Self-attention relates distant parts of a stamp directly, which suits
    structures such as tidal tails and lensed arcs whose meaning depends on
    how far-apart pieces relate to each other.
    """
    nn = torch.nn
    n_patches = (cutout // patch) ** 2

    class ViT(nn.Module):
        embedding_dim = dim

        def __init__(self):
            super().__init__()
            self.patch = patch
            self.project = nn.Conv2d(1, dim, patch, stride=patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.positions = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
            nn.init.trunc_normal_(self.positions, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 3,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            # norm_first layers cannot use the nested-tensor fast path.

            self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                                 enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, n_classes)

        def embed(self, x):
            tokens = self.project(x).flatten(2).transpose(1, 2)
            cls = self.cls.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + self.positions[:, :tokens.shape[1]]
            # The class token aggregates the whole stamp; it is the embedding.
            return self.norm(self.encoder(tokens))[:, 0]

        def forward(self, x):
            return self.head(self.embed(x))

    return ViT()


class StampClassifier:
    """Trainable postage-stamp classifier (CNN or ViT backbone).

    >>> clf = StampClassifier(backbone="cnn")     # doctest: +SKIP
    >>> clf.fit(stamps, labels, epochs=30)        # doctest: +SKIP
    >>> clf.predict(stamps)                       # doctest: +SKIP
    """

    def __init__(self, backbone: str = "cnn", classes: Optional[Sequence[ObjectClass]] = None,
                 cutout: int = 48, width: int = 32, device: str = "cpu",
                 random_state: int = 42):
        self.backbone = str(backbone)
        self.classes = list(classes or STAMP_CLASSES)
        self.cutout = int(cutout)
        self.width = int(width)
        self.device = device
        self.random_state = int(random_state)
        self.model = None
        self.history_: List[float] = []
        self._torch = None

    @property
    def available(self) -> bool:
        return try_import("torch") is not None

    def build(self) -> "StampClassifier":
        torch = require("torch", "the stamp classifier")
        self._torch = torch
        torch.manual_seed(self.random_state)
        if self.backbone == "vit":
            self.model = build_vit(torch, len(self.classes), self.cutout).to(self.device)
        else:
            self.model = build_cnn(torch, len(self.classes), self.width,
                                   self.cutout).to(self.device)
        log.info("built %s stamp classifier: %d classes, %.2fM parameters",
                 self.backbone, len(self.classes),
                 sum(p.numel() for p in self.model.parameters()) / 1e6)
        return self

    def _prepare(self, stamps: Sequence[np.ndarray]) -> np.ndarray:
        """Stretch and resize stamps into a uniform network input."""
        prepared = []
        for stamp in stamps:
            data = asinh_stretch(as_float_image(stamp))
            prepared.append(pad_or_crop(data, (self.cutout, self.cutout)))
        return np.stack(prepared).astype(np.float32)[:, None]

    def fit(self, stamps: Sequence[np.ndarray], labels: Sequence,
            epochs: int = 40, learning_rate: float = 1e-3, batch_size: int = 32,
            augment: bool = True, verbose: bool = True) -> List[float]:
        """Train on labelled stamps; ``labels`` may be enums or class strings."""
        if self.model is None:
            self.build()
        torch = self._torch
        index = {c.value: i for i, c in enumerate(self.classes)}
        targets = []
        for label in labels:
            key = label.value if isinstance(label, ObjectClass) else str(label)
            if key not in index:
                raise ModelError(f"label '{key}' is not in the classifier's class set")
            targets.append(index[key])
        if not targets:
            raise ModelError("no training samples supplied")

        X = torch.from_numpy(self._prepare(stamps))
        y = torch.tensor(targets, dtype=torch.long)

        # Class weights: astronomical samples are heavily imbalanced -- stars
        # outnumber everything -- and an unweighted loss simply predicts them.
        counts = np.bincount(targets, minlength=len(self.classes)).astype(float)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / max(weights[weights > 0].mean(), 1e-9)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device))
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)

        self.model.train()
        self.history_ = []
        n = len(X)
        for epoch in range(int(epochs)):
            permutation = torch.randperm(n)
            total, batches = 0.0, 0
            for start in range(0, n, batch_size):
                batch = permutation[start:start + batch_size]
                if len(batch) < 2:
                    continue
                xb = X[batch].to(self.device)
                if augment:
                    # Astronomical images have no preferred orientation, so
                    # flips and 90-degree rotations are exactly valid.
                    if bool(torch.rand(1) < 0.5):
                        xb = torch.flip(xb, dims=[3])
                    if bool(torch.rand(1) < 0.5):
                        xb = torch.flip(xb, dims=[2])
                    xb = torch.rot90(xb, int(torch.randint(0, 4, (1,))), dims=[2, 3])
                loss = criterion(self.model(xb), y[batch].to(self.device))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1
            self.history_.append(total / max(batches, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                log.info("epoch %3d/%d loss=%.4f", epoch + 1, epochs, self.history_[-1])
        self.model.eval()
        return self.history_

    def predict_proba(self, stamps: Sequence[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("build or load the classifier before predicting")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(self._prepare(stamps)).to(self.device)
            return torch.softmax(self.model(X), dim=1).cpu().numpy()

    def predict(self, stamps: Sequence[np.ndarray]) -> List[ObjectClass]:
        return [self.classes[int(i)] for i in np.argmax(self.predict_proba(stamps), axis=1)]

    def embed(self, stamps: Sequence[np.ndarray]) -> np.ndarray:
        """Penultimate-layer representation, for anomaly and similarity work."""
        if self.model is None:
            raise NotFittedError("build or load the classifier before embedding")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(self._prepare(stamps)).to(self.device)
            return self.model.embed(X).cpu().numpy()

    def annotate(self, catalog: SourceCatalog, image, min_confidence: float = 0.0,
                 store_embedding: bool = True) -> SourceCatalog:
        """Classify every source in a catalog and write the result back."""
        if len(catalog) == 0:
            return catalog
        stamps = [image.cutout(s.x, s.y, self.cutout, subtract_background=True)
                  for s in catalog]
        probabilities = self.predict_proba(stamps)
        embeddings = self.embed(stamps) if store_embedding else None
        for i, source in enumerate(catalog):
            scores = {c.value: float(probabilities[i, j]) for j, c in enumerate(self.classes)}
            best = int(np.argmax(probabilities[i]))
            confidence = float(probabilities[i, best])
            source.class_scores = scores
            if confidence >= min_confidence:
                source.object_class = self.classes[best]
                source.class_confidence = confidence
            if embeddings is not None:
                source.embedding = embeddings[i]
        return catalog

    def save(self, path: str) -> str:
        if self.model is None:
            raise NotFittedError("build or load the classifier before saving")
        torch = self._torch or require("torch", "saving the classifier")
        torch.save({"state_dict": self.model.state_dict(),
                    "classes": [c.value for c in self.classes],
                    "backbone": self.backbone, "cutout": self.cutout,
                    "width": self.width}, path)
        return path

    def load(self, path: str) -> "StampClassifier":
        torch = require("torch", "the stamp classifier")
        self._torch = torch
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = [ObjectClass(c) for c in payload["classes"]]
        self.backbone = payload.get("backbone", self.backbone)
        self.cutout = int(payload.get("cutout", self.cutout))
        self.width = int(payload.get("width", self.width))
        self.build()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        log.info("loaded stamp classifier from %s", path)
        return self
