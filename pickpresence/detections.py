"""Detection log helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class DetectionEntry:
    start: float
    end: float
    embedding: list[float]
    track_id: int | str
    sources: list[str] = field(default_factory=list)
    base_score: float = 0.0
    label: str | None = None
    force_keep: bool = False
    original_track_id: int | str | None = None
    appearance: list[float] | None = None
    appearance_similarity: float | None = None
    person_embedding: list[float] | None = None
    person_similarity: float | None = None
    bbox: list[float] | None = None
    frame_index: int | None = None
    similarity: float | None = None
    best_ref_id: str | None = None
    best_ref_sim: float | None = None
    ref_topk_avg: float | None = None
    frame_size: list[float] | None = None

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def load_detection_log(path: str | Path) -> List[DetectionEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Detection log must be a list.")
    entries: List[DetectionEntry] = []
    for item in raw:
            entries.append(
                DetectionEntry(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    embedding=[float(v) for v in item["embedding"]],
                    track_id=item.get("track_id", 0),
                    sources=list(item.get("sources", [])),
                    base_score=float(item.get("score", 0.0)),
                    label=item.get("label"),
                    original_track_id=item.get("original_track_id"),
                    appearance=item.get("appearance"),
                    person_embedding=item.get("person_embedding"),
                    bbox=item.get("bbox"),
                    frame_index=item.get("frame_index"),
                    frame_size=item.get("frame_size"),
                )
            )
    entries.sort(key=lambda entry: entry.start)
    return entries
