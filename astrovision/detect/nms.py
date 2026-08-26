"""Non-maximum suppression, in NumPy so it needs no deep-learning stack."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.types import BoundingBox


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4,
        max_detections: int = 10_000) -> List[int]:
    """Greedy non-maximum suppression.

    ``boxes`` is ``(N, 4)`` as ``(x_min, y_min, x_max, y_max)``.  Returns
    the indices of the boxes to keep, highest score first.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).ravel()
    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(scores)[::-1]

    keep: List[int] = []
    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[current], x1[rest])
        iy1 = np.maximum(y1[current], y1[rest])
        ix2 = np.minimum(x2[current], x2[rest])
        iy2 = np.minimum(y2[current], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5,
             score_threshold: float = 0.05) -> Tuple[List[int], np.ndarray]:
    """Gaussian soft-NMS: overlapping boxes are demoted, not deleted.

    In crowded star fields hard suppression deletes genuine close pairs;
    soft-NMS keeps them at a reduced score so the vetting stage can decide.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.array(scores, dtype=float).ravel()
    if boxes.shape[0] == 0:
        return [], scores

    areas = (np.clip(boxes[:, 2] - boxes[:, 0], 0, None) *
             np.clip(boxes[:, 3] - boxes[:, 1], 0, None))
    indices = np.arange(len(scores))
    keep: List[int] = []
    working = scores.copy()

    while indices.size > 0:
        best = int(indices[np.argmax(working[indices])])
        if working[best] < score_threshold:
            break
        keep.append(best)
        indices = indices[indices != best]
        if indices.size == 0:
            break
        ix1 = np.maximum(boxes[best, 0], boxes[indices, 0])
        iy1 = np.maximum(boxes[best, 1], boxes[indices, 1])
        ix2 = np.minimum(boxes[best, 2], boxes[indices, 2])
        iy2 = np.minimum(boxes[best, 3], boxes[indices, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[best] + areas[indices] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        working[indices] *= np.exp(-(iou ** 2) / max(sigma, 1e-6))
    return keep, working


def boxes_to_array(boxes: Sequence[BoundingBox]) -> np.ndarray:
    """Convert :class:`BoundingBox` objects to an ``(N, 4)`` array."""
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.array([[b.x_min, b.y_min, b.x_max, b.y_max] for b in boxes], dtype=float)


def array_to_boxes(array: np.ndarray) -> List[BoundingBox]:
    """Convert an ``(N, 4)`` array back to :class:`BoundingBox` objects."""
    return [BoundingBox(*map(float, row)) for row in np.asarray(array).reshape(-1, 4)]


def match_detections(predicted: np.ndarray, truth: np.ndarray,
                     iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IoU matching used for detector evaluation.

    Returns ``(matches, unmatched_predictions, unmatched_truth)``.
    """
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 4)
    truth = np.asarray(truth, dtype=float).reshape(-1, 4)
    if predicted.size == 0 or truth.size == 0:
        return [], list(range(len(predicted))), list(range(len(truth)))

    iou = np.zeros((len(predicted), len(truth)), dtype=float)
    for i, box in enumerate(predicted):
        ix1 = np.maximum(box[0], truth[:, 0])
        iy1 = np.maximum(box[1], truth[:, 1])
        ix2 = np.minimum(box[2], truth[:, 2])
        iy2 = np.minimum(box[3], truth[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area_p = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
        area_t = np.clip(truth[:, 2] - truth[:, 0], 0, None) * np.clip(truth[:, 3] - truth[:, 1], 0, None)
        union = area_p + area_t - inter
        iou[i] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

    matches: List[Tuple[int, int]] = []
    used_p, used_t = set(), set()
    while True:
        i, j = np.unravel_index(int(np.argmax(iou)), iou.shape)
        if iou[i, j] < iou_threshold:
            break
        matches.append((int(i), int(j)))
        used_p.add(int(i))
        used_t.add(int(j))
        iou[i, :] = -1
        iou[:, j] = -1
    return (matches,
            [i for i in range(len(predicted)) if i not in used_p],
            [j for j in range(len(truth)) if j not in used_t])


def average_precision(predicted: np.ndarray, scores: np.ndarray, truth: np.ndarray,
                      iou_threshold: float = 0.5) -> float:
    """Average precision at a single IoU threshold (VOC-style integration)."""
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).reshape(-1, 4)
    if len(truth) == 0:
        return 0.0
    if len(predicted) == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    matches, _, _ = match_detections(predicted[order], truth, iou_threshold)
    matched_pred = {i for i, _ in matches}

    tp = np.array([1.0 if i in matched_pred else 0.0 for i in range(len(predicted))])
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(1.0 - tp)
    recall = cumulative_tp / len(truth)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

    # Monotone-decreasing precision envelope, then integrate over recall.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    change = np.nonzero(np.diff(np.concatenate([[0.0], recall])))[0]
    return float(np.sum((recall[change] - np.concatenate([[0.0], recall])[change]) * precision[change]))
