"""Lightweight person ReID-style embedding (downsample + normalize)."""

from __future__ import annotations

import math

try:  # pragma: no cover
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def compute_person_embedding(image, size: int = 32) -> list[float]:
    if image is None:
        return []
    if np is not None:
        return _compute_person_np(image, size)
    return _compute_person_py(image, size)


def _compute_person_np(image, size: int) -> list[float]:
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        return []
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    resized = cv2.resize(arr, (size, size))
    vec = resized.astype(np.float32).flatten()
    norm = np.linalg.norm(vec) or 1e-8
    return (vec / norm).tolist()


def _compute_person_py(image, size: int) -> list[float]:
    if not image:
        return []
    # naive nearest neighbor downsample
    h = len(image)
    w = len(image[0]) if h else 0
    if h == 0 or w == 0:
        return []
    y_idx = [int(round(i * (h - 1) / (size - 1))) for i in range(size)]
    x_idx = [int(round(i * (w - 1) / (size - 1))) for i in range(size)]
    vec = []
    for y in y_idx:
        for x in x_idx:
            pixel = image[y][x]
            if len(pixel) >= 3:
                vec.extend([float(pixel[0]), float(pixel[1]), float(pixel[2])])
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-8
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length))) or 1e-8
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length))) or 1e-8
    return dot / (norm_a * norm_b)


class PersonMatcher:
    def __init__(self, references: list[list[float]]) -> None:
        self.references = [list(ref) for ref in references if ref]

    def similarity(self, vector: list[float]) -> float:
        if not self.references or not vector:
            return 0.0
        return max(cosine_similarity(vector, ref) for ref in self.references)
