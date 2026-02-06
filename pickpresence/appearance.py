"""Lightweight appearance embedding utilities (color histograms)."""

from __future__ import annotations

import math
from typing import Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def compute_appearance_vector(image, bins_per_channel: int = 16) -> list[float]:
    """Compute normalized RGB histogram for a given image/ROI."""
    if image is None:
        return []
    if np is not None:
        return _compute_appearance_np(image, bins_per_channel)
    return _compute_appearance_py(image, bins_per_channel)


def _compute_appearance_np(image, bins_per_channel: int) -> list[float]:
    arr = np.asarray(image)
    if arr.size == 0:
        return []
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    hist = []
    for ch in range(3):
        channel = arr[:, :, ch].flatten()
        h, _ = np.histogram(channel, bins=bins_per_channel, range=(0, 255))
        hist.extend(h.tolist())
    return _normalize(hist)


def _compute_appearance_py(image, bins_per_channel: int) -> list[float]:
    # Expect nested list of [H][W][C]
    if not image:
        return []
    bins = [[0] * bins_per_channel for _ in range(3)]
    for row in image:
        for pixel in row:
            if len(pixel) < 3:
                continue
            for ch in range(3):
                val = int(pixel[ch])
                idx = min(bins_per_channel - 1, max(0, int(val * bins_per_channel / 256)))
                bins[ch][idx] += 1
    hist = [v for ch in bins for v in ch]
    return _normalize(hist)


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1e-8
    return [v / norm for v in values]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length))) or 1e-8
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length))) or 1e-8
    return dot / (norm_a * norm_b)


class AppearanceMatcher:
    def __init__(self, references: Sequence[Sequence[float]]) -> None:
        self.references = [list(ref) for ref in references if ref]

    def similarity(self, vector: Sequence[float]) -> float:
        if not self.references or not vector:
            return 0.0
        return max(cosine_similarity(vector, ref) for ref in self.references)
