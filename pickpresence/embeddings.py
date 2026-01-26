"""Shared embedding utilities for detector + reference tools."""

from __future__ import annotations

import math
from typing import Any, Sequence, TYPE_CHECKING

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy optional for CLI/tests
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    import numpy as np_typing

    ArrayLike = np_typing.ndarray
else:
    ArrayLike = Any

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

EMBED_HEIGHT = 16
EMBED_WIDTH = 16
EMBED_CHANNELS = 3
EMBED_SIZE = EMBED_HEIGHT * EMBED_WIDTH * EMBED_CHANNELS


def compute_embedding(face_roi: ArrayLike) -> ArrayLike:
    """Compute normalized embedding for a face ROI."""

    if np is None:
        raise RuntimeError("numpy is required for compute_embedding; install numpy.")

    resized = _resize_roi(face_roi)
    vec = resized.astype(np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)
    return vec / norm


def _resize_roi(face_roi: ArrayLike) -> ArrayLike:
    if np is None:
        raise RuntimeError("numpy is required for compute_embedding; install numpy.")
    if face_roi.size == 0:
        return np.zeros((EMBED_HEIGHT, EMBED_WIDTH, EMBED_CHANNELS), dtype=np.float32)
    roi = face_roi
    if roi.ndim == 2:  # grayscale -> stack channels
        roi = np.stack([roi] * EMBED_CHANNELS, axis=-1)
    elif roi.shape[2] == 1:
        roi = np.repeat(roi, EMBED_CHANNELS, axis=2)

    if cv2 is not None:
        return cv2.resize(roi, (EMBED_WIDTH, EMBED_HEIGHT))

    # Fallback: nearest-neighbor resize implemented with numpy.
    y_idx = np.linspace(0, roi.shape[0] - 1, EMBED_HEIGHT).round().astype(int)
    x_idx = np.linspace(0, roi.shape[1] - 1, EMBED_WIDTH).round().astype(int)
    return roi[y_idx][:, x_idx]


__all__ = [
    "compute_embedding",
    "EMBED_SIZE",
    "combine_embeddings",
]


def combine_embeddings(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("At least one embedding is required.")
    if np is None:
        return _combine_embeddings_py(vectors)
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    normalized = arr / norms
    avg = normalized.mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return avg.tolist()
    return (avg / norm).tolist()


def _combine_embeddings_py(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Pure python fallback for CLI/tests when numpy is unavailable."""

    length = len(vectors[0])
    normalized: list[list[float]] = []
    for vec in vectors:
        if len(vec) != length:
            raise ValueError("All embeddings must share the same length.")
        norm = math.sqrt(sum(value * value for value in vec)) or 1e-8
        normalized.append([value / norm for value in vec])
    avg = [0.0] * length
    for vec in normalized:
        for idx, value in enumerate(vec):
            avg[idx] += value
    avg = [value / len(normalized) for value in avg]
    norm = math.sqrt(sum(value * value for value in avg))
    if norm == 0:
        return avg
    return [value / norm for value in avg]
