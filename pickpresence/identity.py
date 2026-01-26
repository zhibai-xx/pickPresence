"""Identity utilities (embeddings + matcher) for PickPresence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        raise ValueError("Embedding vector cannot be zero.")
    return [v / norm for v in vec]


@dataclass
class ReferenceEmbedding:
    name: str
    vector: list[float]


def load_reference_embedding(path: str | Path) -> ReferenceEmbedding:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "embedding" not in data:
        raise ValueError("Reference embedding JSON must contain 'embedding'.")
    name = data.get("name") or Path(path).stem
    vector = [float(v) for v in data["embedding"]]
    return ReferenceEmbedding(name=name, vector=_normalize(vector))


class FaceMatcher:
    """Simple cosine-similarity matcher for embeddings."""

    def __init__(self, reference: ReferenceEmbedding, threshold: float = 0.8) -> None:
        if threshold < 0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.reference = ReferenceEmbedding(reference.name, reference.vector.copy())
        self.threshold = threshold

    def similarity(self, vector: Sequence[float]) -> float:
        query = _normalize([float(v) for v in vector])
        score = sum(a * b for a, b in zip(self.reference.vector, query))
        return max(0.0, min(1.0, score))

    def is_match(self, vector: Sequence[float]) -> bool:
        return self.similarity(vector) >= self.threshold
