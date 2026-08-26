"""Connected-component labelling.

SciPy's ``ndimage.label`` is used when available; otherwise a two-pass
union-find implementation gives identical results in pure NumPy, which
keeps detection working in a minimal install.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.backend import try_import

#: 8-connectivity structuring element (objects touching at corners merge).
CONNECT_8 = np.ones((3, 3), dtype=int)
#: 4-connectivity structuring element.
CONNECT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


class _UnionFind:
    """Disjoint-set forest with path compression."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def add(self, item: int) -> int:
        self.parent.setdefault(item, item)
        return item

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:      # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def label(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Label connected ``True`` regions; returns ``(labels, count)``.

    Labels start at 1; background is 0.
    """
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32), 0

    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        structure = CONNECT_8 if connectivity == 8 else CONNECT_4
        labels, count = scipy_ndimage.label(binary, structure=structure)
        return labels.astype(np.int32), int(count)

    ny, nx = binary.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    forest = _UnionFind()
    next_label = 1

    for y in range(ny):
        for x in range(nx):
            if not binary[y, x]:
                continue
            neighbours: List[int] = []
            if x > 0 and labels[y, x - 1]:
                neighbours.append(int(labels[y, x - 1]))
            if y > 0 and labels[y - 1, x]:
                neighbours.append(int(labels[y - 1, x]))
            if connectivity == 8 and y > 0:
                if x > 0 and labels[y - 1, x - 1]:
                    neighbours.append(int(labels[y - 1, x - 1]))
                if x < nx - 1 and labels[y - 1, x + 1]:
                    neighbours.append(int(labels[y - 1, x + 1]))
            if not neighbours:
                labels[y, x] = next_label
                forest.add(next_label)
                next_label += 1
            else:
                smallest = min(neighbours)
                labels[y, x] = smallest
                for other in neighbours:
                    forest.union(smallest, other)

    # Second pass: flatten the equivalence classes to consecutive labels.
    remap: Dict[int, int] = {}
    count = 0
    flat = labels.ravel()
    for index in range(flat.size):
        value = int(flat[index])
        if value == 0:
            continue
        root = forest.find(value)
        if root not in remap:
            count += 1
            remap[root] = count
        flat[index] = remap[root]
    return labels, count


def find_objects(labels: np.ndarray, count: Optional[int] = None
                 ) -> List[Optional[Tuple[slice, slice]]]:
    """Bounding-box slices for each label, indexed from label 1."""
    scipy_ndimage = try_import("scipy.ndimage")
    if scipy_ndimage is not None:
        return list(scipy_ndimage.find_objects(labels))

    data = np.asarray(labels)
    n = int(count if count is not None else data.max())
    boxes: List[Optional[Tuple[slice, slice]]] = [None] * n
    for value in range(1, n + 1):
        ys, xs = np.nonzero(data == value)
        if ys.size == 0:
            continue
        boxes[value - 1] = (slice(int(ys.min()), int(ys.max()) + 1),
                            slice(int(xs.min()), int(xs.max()) + 1))
    return boxes


def label_sizes(labels: np.ndarray, count: Optional[int] = None) -> np.ndarray:
    """Pixel count per label; index 0 holds the background count."""
    data = np.asarray(labels, dtype=np.int64).ravel()
    n = int(count if count is not None else (data.max() if data.size else 0))
    return np.bincount(data, minlength=n + 1)


def remove_small(labels: np.ndarray, min_size: int,
                 count: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Drop labels below ``min_size`` pixels and renumber consecutively."""
    sizes = label_sizes(labels, count)
    keep = np.nonzero(sizes >= int(min_size))[0]
    keep = keep[keep > 0]
    remap = np.zeros(len(sizes), dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return remap[labels], int(len(keep))


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Grow a boolean mask by ``iterations`` pixels (8-connectivity)."""
    scipy_ndimage = try_import("scipy.ndimage")
    binary = np.asarray(mask, dtype=bool)
    if iterations <= 0:
        return binary
    if scipy_ndimage is not None:
        return scipy_ndimage.binary_dilation(binary, structure=CONNECT_8.astype(bool),
                                             iterations=int(iterations))
    out = binary
    for _ in range(int(iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy:1 + dy + out.shape[0],
                                1 + dx:1 + dx + out.shape[1]]
        out = grown
    return out
