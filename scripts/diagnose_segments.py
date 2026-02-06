#!/usr/bin/env python3
"""Compute similarity/track stability diagnostics for segments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pickpresence.identity import FaceMatcher, load_reference_embeddings
from pickpresence.detections import load_detection_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment diagnostics (similarity + track stability).")
    parser.add_argument("--detection-log", required=True, help="Detections JSON.")
    parser.add_argument("--segments-json", required=True, help="JSON list of {start,end,label}.")
    parser.add_argument("--output", required=True, help="Output diagnostics JSON.")
    parser.add_argument("--reference-dir", help="Directory containing reference embedding JSONs.")
    parser.add_argument("--reference-embeddings", nargs="*", help="Explicit reference embedding JSONs.")
    return parser.parse_args()


def _load_segments(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict] = []
    for item in raw:
        segments.append(
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "label": str(item.get("label") or f"{item['start']}-{item['end']}"),
            }
        )
    return segments


def _collect_reference_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        if ref_dir.is_dir():
            paths.extend(sorted(ref_dir.glob("*.json")))
    if args.reference_embeddings:
        paths.extend(Path(p) for p in args.reference_embeddings)
    dedup: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "embedding" not in payload:
            continue
        dedup.append(path)
    if not dedup:
        raise RuntimeError("No reference embeddings found. Provide --reference-dir or --reference-embeddings.")
    return dedup


def _stats(values: Sequence[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "p90": None}
    ordered = sorted(values)
    mean = sum(values) / len(values)
    p90 = ordered[int(round(0.9 * (len(ordered) - 1)))]
    return {
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(mean, 6),
        "p90": round(p90, 6),
    }


def main() -> int:
    args = parse_args()
    detections = load_detection_log(args.detection_log)
    segments = _load_segments(Path(args.segments_json))
    refs = _collect_reference_paths(args)
    matcher = FaceMatcher(load_reference_embeddings(refs), threshold=0.0, agg="topk_avg", topk=2)

    output = {"detection_log": args.detection_log, "segments": []}
    for seg in segments:
        start, end = seg["start"], seg["end"]
        entries = [d for d in detections if d.start < end and d.end > start]
        sims = [matcher.similarity(d.embedding) for d in entries]
        scores = [d.base_score for d in entries]
        track_ids = [str(d.track_id) for d in entries]
        switches = 0
        for prev, cur in zip(track_ids, track_ids[1:]):
            if cur != prev:
                switches += 1
        output["segments"].append(
            {
                "label": seg["label"],
                "start": start,
                "end": end,
                "detection_count": len(entries),
                "track_ids": sorted(set(track_ids)),
                "track_switches": switches,
                "similarity": _stats(sims),
                "det_score": _stats(scores),
            }
        )

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[diagnose_segments] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
