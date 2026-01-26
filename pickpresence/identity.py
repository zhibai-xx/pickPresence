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
    ref_id: str | None = None


def load_reference_embedding(path: str | Path) -> ReferenceEmbedding:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "embedding" not in data:
        raise ValueError("Reference embedding JSON must contain 'embedding'.")
    name = data.get("name") or Path(path).stem
    ref_id = data.get("id") or data.get("ref_id") or name or Path(path).stem
    vector = [float(v) for v in data["embedding"]]
    return ReferenceEmbedding(name=name, vector=_normalize(vector), ref_id=str(ref_id))


def load_reference_embeddings(paths: Sequence[str | Path]) -> list[ReferenceEmbedding]:
    return [load_reference_embedding(path) for path in paths]


@dataclass
class MatchDetails:
    score: float
    best_ref_id: str | None
    best_ref_sim: float | None
    topk_avg: float | None


class FaceMatcher:
    """Cosine-similarity matcher supporting multi-reference aggregation."""

    def __init__(
        self,
        reference: ReferenceEmbedding | Sequence[ReferenceEmbedding],
        threshold: float = 0.8,
        agg: str = "max",
        topk: int = 3,
    ) -> None:
        if threshold < 0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if agg not in {"max", "topk_avg"}:
            raise ValueError("agg must be 'max' or 'topk_avg'")
        if topk <= 0:
            raise ValueError("topk must be positive")
        refs = list(reference) if isinstance(reference, Iterable) and not isinstance(reference, ReferenceEmbedding) else [reference]  # type: ignore[arg-type]
        self.references = [
            ReferenceEmbedding(ref.name, ref.vector.copy(), ref.ref_id or ref.name)
            for ref in refs
        ]
        self.reference = _combined_reference(self.references)
        self.threshold = threshold
        self.agg = agg
        self.topk = topk

    def similarity(self, vector: Sequence[float]) -> float:
        return self.match_details(vector).score

    def match_details(self, vector: Sequence[float]) -> MatchDetails:
        query = _normalize([float(v) for v in vector])
        sims = [
            max(0.0, min(1.0, sum(a * b for a, b in zip(ref.vector, query))))
            for ref in self.references
        ]
        if not sims:
            return MatchDetails(score=0.0, best_ref_id=None, best_ref_sim=None, topk_avg=None)
        best_idx = max(range(len(sims)), key=lambda idx: sims[idx])
        best_ref = self.references[best_idx]
        best_ref_id = best_ref.ref_id or best_ref.name
        best_ref_sim = sims[best_idx]
        topk_avg = None
        if self.agg == "topk_avg":
            k = min(self.topk, len(sims))
            topk_avg = sum(sorted(sims, reverse=True)[:k]) / k
            score = topk_avg
        else:
            score = best_ref_sim
        return MatchDetails(
            score=score,
            best_ref_id=best_ref_id,
            best_ref_sim=best_ref_sim,
            topk_avg=topk_avg,
        )

    def is_match(self, vector: Sequence[float]) -> bool:
        return self.similarity(vector) >= self.threshold


def _combined_reference(references: Sequence[ReferenceEmbedding]) -> ReferenceEmbedding:
    if not references:
        raise ValueError("At least one reference embedding is required.")
    if len(references) == 1:
        ref = references[0]
        return ReferenceEmbedding(ref.name, ref.vector.copy(), ref.ref_id or ref.name)
    length = len(references[0].vector)
    avg = [0.0] * length
    for ref in references:
        if len(ref.vector) != length:
            raise ValueError("All embeddings must share the same length.")
        for idx, value in enumerate(ref.vector):
            avg[idx] += value
    avg = [value / len(references) for value in avg]
    return ReferenceEmbedding(name="template", vector=_normalize(avg), ref_id="template")
