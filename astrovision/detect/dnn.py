"""Deep object detection for astronomical images.

The architecture is an anchor-free, CenterNet-style detector: a small
residual backbone predicts a per-pixel *centre heatmap*, a size map and a
sub-pixel offset map.  Anchor-free suits astronomy far better than a YOLO
anchor grid, because sources range from unresolved points to galaxies
spanning hundreds of pixels and are never axis-aligned in any meaningful
way.

PyTorch is an optional dependency.  Everything here imports cleanly
without it; only construction of a model requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.backend import require, try_import
from ..core.exceptions import ModelError, NotFittedError
from ..core.logging import get_logger
from ..core.numeric import as_float_image, maximum_filter
from ..core.types import BoundingBox, ObjectClass, Source, SourceCatalog
from ..preprocess.normalize import asinh_stretch
from .nms import nms

log = get_logger("detect.dnn")

#: Classes the deep detector predicts, in output-channel order.
DETECTOR_CLASSES: List[ObjectClass] = [
    ObjectClass.STAR,
    ObjectClass.GALAXY,
    ObjectClass.NEBULA,
    ObjectClass.STAR_CLUSTER,
    ObjectClass.ARTIFACT,
]


def build_backbone(torch, in_channels: int = 1, width: int = 32):
    """A small residual encoder-decoder that preserves spatial resolution.

    Astronomical detection needs fine localisation far more than deep
    semantic abstraction, so the network keeps a stride-4 bottleneck and
    upsamples back to half resolution.
    """
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.body(x))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, width, 5, padding=2, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            )
            self.down1 = nn.Sequential(
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
                ResidualBlock(width * 2),
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
                ResidualBlock(width * 4),
            )
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(width * 4, width * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(width * 4, width * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
                ResidualBlock(width * 2),
            )

        def forward(self, x):
            stem = self.stem(x)
            d1 = self.down1(stem)          # stride 2
            d2 = self.down2(d1)            # stride 4
            up = self.up(d2)               # back to stride 2
            return self.fuse(torch.cat([up, d1], dim=1))

    return Backbone()


def build_detector(torch, n_classes: int = len(DETECTOR_CLASSES), width: int = 32):
    """Assemble the full detector: backbone plus heatmap/size/offset heads."""
    nn = torch.nn

    class CenterDetector(nn.Module):
        """Predicts object centres, sizes and sub-pixel offsets at stride 2."""

        stride = 2

        def __init__(self):
            super().__init__()
            self.backbone = build_backbone(torch, 1, width)
            channels = width * 2

            def head(out_channels: int, bias: float = 0.0):
                layer = nn.Conv2d(channels, out_channels, 1)
                nn.init.constant_(layer.bias, bias)
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), layer)

            # The -4.6 bias makes the initial heatmap sigmoid ~0.01, which
            # stops the focal loss from being swamped by background early on.
            self.heatmap = head(n_classes, bias=-4.6)
            self.size = head(2)
            self.offset = head(2)

        def forward(self, x):
            features = self.backbone(x)
            return {
                "heatmap": self.heatmap(features),
                "size": self.size(features),
                "offset": self.offset(features),
            }

    return CenterDetector()


def focal_loss(torch, prediction, target, alpha: float = 2.0, beta: float = 4.0):
    """CornerNet/CenterNet penalty-reduced focal loss for centre heatmaps.

    Pixels near a true centre are penalised less than distant ones, which
    matters when object centres are only defined to a pixel or two.
    """
    positive = target.eq(1.0).float()
    negative = 1.0 - positive
    negative_weight = torch.pow(1.0 - target, beta)
    probability = torch.clamp(torch.sigmoid(prediction), 1e-6, 1 - 1e-6)

    positive_loss = -torch.log(probability) * torch.pow(1 - probability, alpha) * positive
    negative_loss = (-torch.log(1 - probability) * torch.pow(probability, alpha) *
                     negative_weight * negative)
    n_positive = positive.sum()
    if n_positive == 0:
        return negative_loss.sum()
    return (positive_loss.sum() + negative_loss.sum()) / n_positive


def gaussian_target(shape: Tuple[int, int], centres: Sequence[Tuple[float, float]],
                    sigmas: Sequence[float]) -> np.ndarray:
    """Render the ground-truth heatmap: a Gaussian bump per object centre."""
    target = np.zeros(shape, dtype=np.float32)
    ny, nx = shape
    for (cx, cy), sigma in zip(centres, sigmas):
        sigma = max(float(sigma), 0.8)
        radius = int(np.ceil(3 * sigma))
        x0, x1 = max(0, int(cx) - radius), min(nx, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(ny, int(cy) + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], bump)
        # The exact centre pixel must be exactly 1 so the focal loss sees a positive.
        py, px = int(round(cy)), int(round(cx))
        if 0 <= py < ny and 0 <= px < nx:
            target[py, px] = 1.0
    return target


def decode_heatmap(heatmap: np.ndarray, size: np.ndarray, offset: np.ndarray,
                   stride: int = 2, score_threshold: float = 0.3,
                   max_detections: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert network outputs into ``(boxes, scores, class_ids)``.

    Peaks are found with a 3x3 maximum filter -- the standard anchor-free
    alternative to NMS -- then refined with the predicted sub-pixel offset.
    """
    heat = np.asarray(heatmap, dtype=float)
    if heat.ndim == 2:
        heat = heat[None, ...]
    n_classes = heat.shape[0]

    boxes: List[List[float]] = []
    scores: List[float] = []
    class_ids: List[int] = []
    for c in range(n_classes):
        plane = heat[c]
        peaks = (plane >= maximum_filter(plane, 3)) & (plane >= score_threshold)
        ys, xs = np.nonzero(peaks)
        for y, x in zip(ys, xs):
            ox = float(offset[0, y, x]) if offset is not None else 0.0
            oy = float(offset[1, y, x]) if offset is not None else 0.0
            w = float(np.exp(np.clip(size[0, y, x], -6, 8))) if size is not None else 6.0
            h = float(np.exp(np.clip(size[1, y, x], -6, 8))) if size is not None else 6.0
            cx = (x + ox) * stride
            cy = (y + oy) * stride
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(float(plane[y, x]))
            class_ids.append(c)

    if not boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    order = np.argsort(scores)[::-1][:max_detections]
    return (np.array(boxes, dtype=float)[order],
            np.array(scores, dtype=float)[order],
            np.array(class_ids, dtype=int)[order])


@dataclass
class DetectorWeights:
    """Serialised detector state, portable across environments."""

    state_dict: Dict[str, Any]
    classes: List[str]
    width: int
    normalization: str = "asinh"
    tile_size: int = 256


class DeepDetector:
    """Anchor-free deep detector with a NumPy-side decoding path.

    >>> detector = DeepDetector()            # doctest: +SKIP
    >>> detector.load("model.pt")            # doctest: +SKIP
    >>> catalog = detector.detect(image)     # doctest: +SKIP
    """

    def __init__(self, width: int = 32, classes: Optional[Sequence[ObjectClass]] = None,
                 score_threshold: float = 0.3, nms_iou: float = 0.4,
                 tile_size: int = 256, device: str = "cpu",
                 normalization: str = "asinh"):
        self.width = int(width)
        self.classes = list(classes or DETECTOR_CLASSES)
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.tile_size = int(tile_size)
        self.device = device
        self.normalization = normalization
        self.model = None
        self._torch = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether PyTorch is importable in this environment."""
        return try_import("torch") is not None

    def build(self) -> "DeepDetector":
        """Instantiate the network (requires PyTorch)."""
        torch = require("torch", "the deep object detector")
        self._torch = torch
        self.model = build_detector(torch, len(self.classes), self.width).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("built deep detector: %d classes, %.2fM parameters",
                 len(self.classes), n_params / 1e6)
        return self

    def save(self, path: str) -> str:
        """Persist weights and configuration to ``path``."""
        if self.model is None:
            raise NotFittedError("build or load the detector before saving it")
        torch = self._torch or require("torch", "saving the detector")
        torch.save({
            "state_dict": self.model.state_dict(),
            "classes": [c.value for c in self.classes],
            "width": self.width,
            "normalization": self.normalization,
            "tile_size": self.tile_size,
        }, path)
        return path

    def load(self, path: str) -> "DeepDetector":
        """Restore weights and configuration from ``path``."""
        torch = require("torch", "the deep object detector")
        self._torch = torch
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.classes = [ObjectClass(c) for c in payload.get("classes",
                                                            [c.value for c in DETECTOR_CLASSES])]
        self.width = int(payload.get("width", self.width))
        self.normalization = payload.get("normalization", self.normalization)
        self.tile_size = int(payload.get("tile_size", self.tile_size))
        self.model = build_detector(torch, len(self.classes), self.width).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        log.info("loaded deep detector from %s", path)
        return self

    # -- inference ---------------------------------------------------------
    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Normalise raw counts into the network's input range."""
        data = as_float_image(image)
        if self.normalization == "asinh":
            return asinh_stretch(data).astype(np.float32)
        return np.nan_to_num(data, nan=0.0).astype(np.float32)

    def predict_maps(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the network over a (possibly large) image, tile by tile."""
        if self.model is None:
            raise NotFittedError("build or load the detector before predicting")
        torch = self._torch
        data = self._prepare(image)
        ny, nx = data.shape
        stride = self.model.stride
        tile = max(64, self.tile_size)
        overlap = 32

        heat = np.zeros((len(self.classes), ny // stride, nx // stride), dtype=np.float32)
        size = np.zeros((2, ny // stride, nx // stride), dtype=np.float32)
        offset = np.zeros((2, ny // stride, nx // stride), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for y0 in range(0, ny, tile - overlap):
                for x0 in range(0, nx, tile - overlap):
                    y1 = min(ny, y0 + tile)
                    x1 = min(nx, x0 + tile)
                    # Pad to a multiple of 4 so the stride-4 bottleneck is exact.
                    patch = data[y0:y1, x0:x1]
                    py = (-patch.shape[0]) % 4
                    px = (-patch.shape[1]) % 4
                    if py or px:
                        patch = np.pad(patch, ((0, py), (0, px)), mode="reflect")
                    tensor = torch.from_numpy(patch[None, None]).to(self.device)
                    out = self.model(tensor)
                    h = torch.sigmoid(out["heatmap"])[0].cpu().numpy()
                    s = out["size"][0].cpu().numpy()
                    o = torch.tanh(out["offset"])[0].cpu().numpy()

                    hy0, hx0 = y0 // stride, x0 // stride
                    hy1 = min(heat.shape[1], hy0 + h.shape[1])
                    hx1 = min(heat.shape[2], hx0 + h.shape[2])
                    sy, sx = hy1 - hy0, hx1 - hx0
                    if sy <= 0 or sx <= 0:
                        continue
                    # Overlapping tiles are combined by maximum on the
                    # heatmap so a source split across a seam still peaks.
                    heat[:, hy0:hy1, hx0:hx1] = np.maximum(heat[:, hy0:hy1, hx0:hx1],
                                                           h[:, :sy, :sx])
                    size[:, hy0:hy1, hx0:hx1] = s[:, :sy, :sx]
                    offset[:, hy0:hy1, hx0:hx1] = o[:, :sy, :sx]
        return {"heatmap": heat, "size": size, "offset": offset}

    def detect(self, image, score_threshold: Optional[float] = None) -> SourceCatalog:
        """Detect sources and return them as a :class:`SourceCatalog`."""
        data = image.data if hasattr(image, "data") else image
        maps = self.predict_maps(data)
        boxes, scores, class_ids = decode_heatmap(
            maps["heatmap"], maps["size"], maps["offset"],
            stride=self.model.stride,
            score_threshold=self.score_threshold if score_threshold is None else score_threshold)
        if len(boxes) == 0:
            return SourceCatalog(meta={"detection_backend": "dnn", "n_raw": 0})

        keep = nms(boxes, scores, self.nms_iou)
        catalog = SourceCatalog(meta={"detection_backend": "dnn",
                                      "n_raw": int(len(boxes)),
                                      "n_kept": len(keep)})
        ny, nx = as_float_image(data).shape
        for new_id, index in enumerate(keep, start=1):
            box = BoundingBox(*map(float, boxes[index])).clip((ny, nx))
            cx, cy = box.center
            source = Source(
                id=new_id, x=cx, y=cy, bbox=box,
                object_class=self.classes[int(class_ids[index])],
                class_confidence=float(scores[index]),
                meta={"detector": "dnn"},
            )
            if hasattr(image, "wcs") and image.wcs is not None:
                ra, dec = image.wcs.pixel_to_world(cx, cy)
                source.ra, source.dec = float(ra), float(dec)
            catalog.append(source)
        log.info("deep detector kept %d of %d raw detections", len(keep), len(boxes))
        return catalog

    # -- training ----------------------------------------------------------
    def fit(self, images: Sequence[np.ndarray], annotations: Sequence[List[Dict[str, Any]]],
            epochs: int = 20, learning_rate: float = 1e-3, batch_size: int = 4,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the detector on labelled cutouts.

        Each entry of ``annotations`` is a list of dicts with keys ``x``,
        ``y``, ``width``, ``height`` and ``class`` (an :class:`ObjectClass`
        or its value).  Returns the loss history.
        """
        if self.model is None:
            self.build()
        torch = self._torch
        model = self.model
        stride = model.stride
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history: Dict[str, List[float]] = {"total": [], "heatmap": [], "size": [], "offset": []}
        class_index = {c.value: i for i, c in enumerate(self.classes)}

        tensors, targets = [], []
        for image, objects in zip(images, annotations):
            data = self._prepare(image)
            py, px = (-data.shape[0]) % 4, (-data.shape[1]) % 4
            if py or px:
                data = np.pad(data, ((0, py), (0, px)), mode="reflect")
            hy, hx = data.shape[0] // stride, data.shape[1] // stride
            heat = np.zeros((len(self.classes), hy, hx), dtype=np.float32)
            size = np.zeros((2, hy, hx), dtype=np.float32)
            offset = np.zeros((2, hy, hx), dtype=np.float32)
            weight = np.zeros((1, hy, hx), dtype=np.float32)

            for obj in objects:
                value = obj.get("class", ObjectClass.UNKNOWN)
                key = value.value if isinstance(value, ObjectClass) else str(value)
                channel = class_index.get(key)
                if channel is None:
                    continue
                cx, cy = float(obj["x"]) / stride, float(obj["y"]) / stride
                w = max(float(obj.get("width", 6.0)), 1.0)
                h = max(float(obj.get("height", 6.0)), 1.0)
                sigma = max(np.sqrt(w * h) / stride / 6.0, 0.8)
                heat[channel] = np.maximum(
                    heat[channel], gaussian_target((hy, hx), [(cx, cy)], [sigma]))
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= iy < hy and 0 <= ix < hx:
                    size[0, iy, ix] = np.log(w)
                    size[1, iy, ix] = np.log(h)
                    offset[0, iy, ix] = cx - ix
                    offset[1, iy, ix] = cy - iy
                    weight[0, iy, ix] = 1.0

            tensors.append(torch.from_numpy(data[None]))
            targets.append((torch.from_numpy(heat), torch.from_numpy(size),
                            torch.from_numpy(offset), torch.from_numpy(weight)))

        if not tensors:
            raise ModelError("no training samples supplied")

        model.train()
        n = len(tensors)
        for epoch in range(int(epochs)):
            permutation = np.random.permutation(n)
            epoch_losses = {"total": 0.0, "heatmap": 0.0, "size": 0.0, "offset": 0.0}
            n_batches = 0
            for start in range(0, n, batch_size):
                batch = permutation[start:start + batch_size]
                # Only images of identical shape can be stacked in one batch.
                shapes = {tuple(tensors[i].shape) for i in batch}
                if len(shapes) > 1:
                    batch = [i for i in batch if tuple(tensors[i].shape) == tuple(tensors[batch[0]].shape)]
                x = torch.stack([tensors[i] for i in batch]).to(self.device)
                heat_t = torch.stack([targets[i][0] for i in batch]).to(self.device)
                size_t = torch.stack([targets[i][1] for i in batch]).to(self.device)
                offset_t = torch.stack([targets[i][2] for i in batch]).to(self.device)
                weight_t = torch.stack([targets[i][3] for i in batch]).to(self.device)

                out = model(x)
                loss_heat = focal_loss(torch, out["heatmap"], heat_t)
                denominator = weight_t.sum().clamp(min=1.0)
                loss_size = (torch.abs(out["size"] - size_t) * weight_t).sum() / denominator
                loss_offset = (torch.abs(torch.tanh(out["offset"]) - offset_t) * weight_t).sum() / denominator
                loss = loss_heat + 0.1 * loss_size + loss_offset

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()

                epoch_losses["total"] += float(loss.item())
                epoch_losses["heatmap"] += float(loss_heat.item())
                epoch_losses["size"] += float(loss_size.item())
                epoch_losses["offset"] += float(loss_offset.item())
                n_batches += 1

            for key in history:
                history[key].append(epoch_losses[key] / max(n_batches, 1))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                log.info("epoch %3d/%d  loss=%.4f (heat=%.4f size=%.4f off=%.4f)",
                         epoch + 1, epochs, history["total"][-1], history["heatmap"][-1],
                         history["size"][-1], history["offset"][-1])
        model.eval()
        return history
