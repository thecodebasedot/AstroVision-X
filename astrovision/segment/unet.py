"""U-Net semantic segmentation of astronomical images.

Where thresholding asks "is this pixel above the noise?", a U-Net asks
"what *kind* of structure is this pixel part of?" -- separating a galaxy's
core from its arms, or diffuse nebulosity from the sky, in a way that
intensity alone cannot.  PyTorch is optional; this module imports cleanly
without it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import require, try_import
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, pad_or_crop
from ..preprocess.normalize import asinh_stretch

log = get_logger("segment.unet")

#: Semantic classes predicted by the default segmentation head.
SEGMENT_CLASSES: List[str] = [
    "background", "star", "galaxy_core", "galaxy_arm", "diffuse", "artifact",
]


def build_unet(torch, in_channels: int = 1, n_classes: int = len(SEGMENT_CLASSES),
               width: int = 16, depth: int = 3):
    """Construct a U-Net with ``depth`` down/up stages.

    The default is deliberately small: astronomical segmentation is a
    texture problem at a few tens of pixels, not an ImageNet-scale
    semantic problem, and small models train on modest labelled sets.
    """
    nn = torch.nn

    def conv_block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth = int(depth)
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool2d(2)

            channels = in_channels
            widths: List[int] = []
            for level in range(self.depth):
                out = width * (2 ** level)
                self.encoders.append(conv_block(channels, out))
                widths.append(out)
                channels = out

            self.bottleneck = conv_block(channels, channels * 2)
            channels *= 2

            for out in reversed(widths):
                self.ups.append(nn.ConvTranspose2d(channels, out, 2, stride=2))
                self.decoders.append(conv_block(out * 2, out))
                channels = out

            self.head = nn.Conv2d(channels, n_classes, 1)

        def forward(self, x):
            skips = []
            for encoder in self.encoders:
                x = encoder(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[-2:],
                                                        mode="bilinear", align_corners=False)
                x = decoder(torch.cat([skip, x], dim=1))
            return self.head(x)

    return UNet()


def dice_loss(torch, logits, target, eps: float = 1e-6):
    """Soft Dice loss -- essential when most pixels are background sky."""
    probability = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(
        target.long(), probability.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    union = probability.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * intersection + eps) / (union + eps)).mean()


class UNetSegmenter:
    """Trainable semantic segmenter for astronomical structure.

    >>> segmenter = UNetSegmenter()      # doctest: +SKIP
    >>> segmenter.load("unet.pt")        # doctest: +SKIP
    >>> classes = segmenter.predict(image_array)   # doctest: +SKIP
    """

    def __init__(self, classes: Optional[Sequence[str]] = None, width: int = 16,
                 depth: int = 3, device: str = "cpu", tile_size: int = 128):
        self.classes = list(classes or SEGMENT_CLASSES)
        self.width = int(width)
        self.depth = int(depth)
        self.device = device
        self.tile_size = int(tile_size)
        self.model = None
        self._torch = None

    @property
    def available(self) -> bool:
        return try_import("torch") is not None

    def build(self) -> "UNetSegmenter":
        torch = require("torch", "U-Net segmentation")
        self._torch = torch
        self.model = build_unet(torch, 1, len(self.classes), self.width, self.depth).to(self.device)
        log.info("built U-Net: %d classes, %.2fM parameters", len(self.classes),
                 sum(p.numel() for p in self.model.parameters()) / 1e6)
        return self

    def save(self, path: str) -> str:
        if self.model is None:
            raise NotFittedError("build or load the segmenter before saving it")
        torch = self._torch or require("torch", "saving the segmenter")
        torch.save({"state_dict": self.model.state_dict(), "classes": self.classes,
                    "width": self.width, "depth": self.depth}, path)
        return path

    def load(self, path: str) -> "UNetSegmenter":
        torch = require("torch", "U-Net segmentation")
        self._torch = torch
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = list(payload.get("classes", self.classes))
        self.width = int(payload.get("width", self.width))
        self.depth = int(payload.get("depth", self.depth))
        self.model = build_unet(torch, 1, len(self.classes), self.width, self.depth).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        log.info("loaded U-Net from %s", path)
        return self

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        return asinh_stretch(as_float_image(image)).astype(np.float32)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Per-class probability maps with shape ``(n_classes, ny, nx)``."""
        if self.model is None:
            raise NotFittedError("build or load the segmenter before predicting")
        torch = self._torch
        data = self._prepare(image)
        ny, nx = data.shape
        block = 2 ** self.depth
        tile = max(block * 4, self.tile_size)
        overlap = block * 2

        accumulator = np.zeros((len(self.classes), ny, nx), dtype=np.float32)
        weights = np.zeros((ny, nx), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for y0 in range(0, ny, tile - overlap):
                for x0 in range(0, nx, tile - overlap):
                    y1, x1 = min(ny, y0 + tile), min(nx, x0 + tile)
                    patch = data[y0:y1, x0:x1]
                    py = (-patch.shape[0]) % block
                    px = (-patch.shape[1]) % block
                    if py or px:
                        patch = np.pad(patch, ((0, py), (0, px)), mode="reflect")
                    tensor = torch.from_numpy(patch[None, None]).to(self.device)
                    probability = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
                    accumulator[:, y0:y1, x0:x1] += probability[:, :y1 - y0, :x1 - x0]
                    weights[y0:y1, x0:x1] += 1.0
        return accumulator / np.maximum(weights, 1.0)[None, ...]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Hard class labels, one integer per pixel."""
        return np.argmax(self.predict_proba(image), axis=0).astype(np.int32)

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
            epochs: int = 30, learning_rate: float = 1e-3, batch_size: int = 4,
            patch: int = 64, class_weights: Optional[Sequence[float]] = None,
            verbose: bool = True) -> List[float]:
        """Train on labelled images; ``masks`` hold integer class indices."""
        if self.model is None:
            self.build()
        torch = self._torch
        if len(images) != len(masks):
            raise ModelError("images and masks must be the same length")

        block = 2 ** self.depth
        patch = max(block * 2, int(patch) // block * block)
        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for image, mask in zip(images, masks):
            data = self._prepare(image)
            target = np.asarray(mask, dtype=np.int64)
            if data.shape != target.shape:
                raise ModelError("image and mask shapes differ")
            samples.append((pad_or_crop(data, (patch, patch)).astype(np.float32),
                            pad_or_crop(target, (patch, patch), fill=0).astype(np.int64)))
        if not samples:
            raise ModelError("no training samples supplied")

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32,
                                         device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        history: List[float] = []

        self.model.train()
        for epoch in range(int(epochs)):
            permutation = np.random.permutation(len(samples))
            total, batches = 0.0, 0
            for start in range(0, len(samples), batch_size):
                batch = permutation[start:start + batch_size]
                x = torch.from_numpy(
                    np.stack([samples[i][0] for i in batch])[:, None]).to(self.device)
                y = torch.from_numpy(
                    np.stack([samples[i][1] for i in batch])).to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y) + dice_loss(torch, logits, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1
            history.append(total / max(batches, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                log.info("epoch %3d/%d  loss=%.4f", epoch + 1, epochs, history[-1])
        self.model.eval()
        return history


def labels_from_segmentation(segmentation: np.ndarray, catalog, image: np.ndarray,
                             classes: Sequence[str] = SEGMENT_CLASSES) -> np.ndarray:
    """Bootstrap U-Net training labels from a classical run.

    Hand-labelled astronomical segmentation masks are scarce.  Turning the
    classical pipeline's own output into weak labels gives a U-Net a
    starting point that can then be refined on a small curated set.
    """
    data = as_float_image(image)
    index = {name: i for i, name in enumerate(classes)}
    labels = np.zeros(data.shape, dtype=np.int32)
    for source in catalog:
        footprint = segmentation == source.segment_label
        if not footprint.any():
            continue
        value = source.object_class.value
        if value == "star":
            labels[footprint] = index.get("star", 1)
        elif value in ("nebula", "star_cluster"):
            labels[footprint] = index.get("diffuse", 4)
        elif value == "artifact":
            labels[footprint] = index.get("artifact", 5)
        elif value == "galaxy":
            # Split the galaxy footprint at its half-light isophote.
            values = data[footprint]
            if values.size:
                level = float(np.percentile(values, 70))
                core = footprint & (data >= level)
                labels[footprint] = index.get("galaxy_arm", 3)
                labels[core] = index.get("galaxy_core", 2)
        else:
            labels[footprint] = index.get("galaxy_arm", 3)
    return labels
