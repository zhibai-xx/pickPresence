"""Track-level selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .detections import DetectionEntry
from .identity import FaceMatcher


@dataclass
class TrackSelection:
    track_id: str
    label: str | None
    entries: List[DetectionEntry]
    avg_similarity: float
    total_duration: float
    score: float
    p90_similarity: float
    max_similarity: float


class TrackSelector:
    """Aggregates detection entries per track and filters using heuristics."""

    def __init__(
        self,
        matcher: FaceMatcher | None,
        policy: str = "best",
        min_avg_similarity: float = 0.75,
        min_total_duration: float = 0.5,
    ) -> None:
        if policy not in {"best", "all"}:
            raise ValueError("policy must be 'best' or 'all'")
        if min_avg_similarity < 0 or min_avg_similarity > 1:
            raise ValueError("min_avg_similarity must be within [0, 1]")
        if min_total_duration < 0:
            raise ValueError("min_total_duration must be non-negative")
        self.matcher = matcher
        self.policy = policy
        self.min_avg_similarity = min_avg_similarity
        self.min_total_duration = min_total_duration

    def select(self, detections: Sequence[DetectionEntry]) -> List[TrackSelection]:
        grouped: Dict[str, List[DetectionEntry]] = {}
        for entry in detections:
            key = str(entry.track_id)
            grouped.setdefault(key, []).append(entry)
        summaries: List[TrackSelection] = []
        for track_id, entries in grouped.items():
            sims = [self._score(entry) for entry in entries]
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            p90 = _percentile(sims, 0.9)
            max_sim = max(sims) if sims else 0.0
            score = max(avg_sim, p90, max_sim)
            total_duration = sum(entry.duration() for entry in entries)
            label = self._resolve_label(entries)
            summaries.append(
                TrackSelection(
                    track_id=track_id,
                    label=label,
                    entries=sorted(entries, key=lambda e: e.start),
                    avg_similarity=avg_sim,
                    total_duration=total_duration,
                    score=score,
                    p90_similarity=p90,
                    max_similarity=max_sim,
                )
            )
        if not summaries:
            return []
        filtered = [
            summary
            for summary in summaries
            if summary.score >= self.min_avg_similarity
            and summary.total_duration >= self.min_total_duration
        ]
        if self.policy == "all":
            return filtered if filtered else summaries
        # best policy
        candidates = filtered if filtered else summaries
        best = max(
            candidates,
            key=lambda summary: (summary.avg_similarity, summary.total_duration),
        )
        return [best]

    def _score(self, entry: DetectionEntry) -> float:
        if self.matcher:
            return self.matcher.similarity(entry.embedding)
        return entry.base_score

    def _resolve_label(self, entries: Sequence[DetectionEntry]) -> str | None:
        for entry in entries:
            if entry.label:
                return entry.label
        return None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    clamped = max(0.0, min(1.0, percentile))
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    index = int(round(clamped * (len(sorted_vals) - 1)))
    index = max(0, min(len(sorted_vals) - 1, index))
    return sorted_vals[index]
