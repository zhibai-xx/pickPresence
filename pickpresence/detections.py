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
    bbox: list[float] | None = None
    frame_index: int | None = None
    similarity: float | None = None

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
                    bbox=item.get("bbox"),
                    frame_index=item.get("frame_index"),
                )
            )
    entries.sort(key=lambda entry: entry.start)
    return entries
