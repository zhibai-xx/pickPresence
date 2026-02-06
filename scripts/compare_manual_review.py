#!/usr/bin/env python3
"""Compare timeline.json against manual review segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare timeline with manual review segments.")
    parser.add_argument("--manual", required=True, help="Manual review JSON path.")
    parser.add_argument("--timeline", required=True, help="Timeline JSON path.")
    parser.add_argument("--output-json", required=True, help="Write comparison JSON here.")
    parser.add_argument("--output-md", help="Optional markdown summary output.")
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Overlap seconds to count a duplicate pair (default 0.5).",
    )
    return parser.parse_args()


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    manual = load_json(args.manual)
    timeline = load_json(args.timeline)

    manual_segments = manual.get("segments", [])
    timeline_segments = timeline.get("segments", [])

    comparisons = []
    for seg in manual_segments:
        ms = float(seg["start"])
        me = float(seg["end"])
        duration = max(0.0, me - ms)
        covered = 0.0
        best = None
        for out in timeline_segments:
            os = float(out["start"])
            oe = float(out["end"])
            ov = overlap(ms, me, os, oe)
            if ov > 0:
                covered += ov
                if best is None or ov > best["overlap"]:
                    best = {
                        "start": os,
                        "end": oe,
                        "overlap": round(ov, 3),
                    }
        ratio = covered / duration if duration else 0.0
        comparisons.append(
            {
                "manual_start": ms,
                "manual_end": me,
                "manual_status": seg.get("status"),
                "manual_note": seg.get("note"),
                "duration": round(duration, 3),
                "covered": round(covered, 3),
                "coverage_ratio": round(ratio, 3),
                "best_overlap": best,
            }
        )

    overlap_pairs = []
    for i, a in enumerate(timeline_segments):
        for j in range(i + 1, len(timeline_segments)):
            b = timeline_segments[j]
            ov = overlap(a["start"], a["end"], b["start"], b["end"])
            if ov >= args.overlap_threshold:
                overlap_pairs.append(
                    {
                        "a_index": i,
                        "b_index": j,
                        "overlap": round(ov, 3),
                        "a_start": a["start"],
                        "a_end": a["end"],
                        "b_start": b["start"],
                        "b_end": b["end"],
                    }
                )

    summary = {
        "manual_count": len(manual_segments),
        "timeline_count": len(timeline_segments),
        "overlap_pairs": len(overlap_pairs),
        "coverage_avg": round(
            sum(item["coverage_ratio"] for item in comparisons) / len(comparisons)
            if comparisons
            else 0.0,
            3,
        ),
    }

    output = {
        "summary": summary,
        "comparisons": comparisons,
        "overlap_pairs": overlap_pairs[:50],
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.output_md:
        lines = [
            "# Manual Review Comparison",
            "",
            f"- Manual segments: {summary['manual_count']}",
            f"- Timeline segments: {summary['timeline_count']}",
            f"- Overlap pairs (>{args.overlap_threshold}s): {summary['overlap_pairs']}",
            f"- Average coverage ratio: {summary['coverage_avg']}",
            "",
            "## Manual Segment Coverage",
        ]
        for item in comparisons:
            lines.append(
                f"- {item['manual_start']:.2f}-{item['manual_end']:.2f} "
                f"{item['manual_status']} coverage={item['coverage_ratio']:.2f} "
                f"best={item['best_overlap']}"
            )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[compare_manual_review] Wrote {args.output_json}")
    if args.output_md:
        print(f"[compare_manual_review] Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
