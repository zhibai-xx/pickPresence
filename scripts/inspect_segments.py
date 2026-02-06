#!/usr/bin/env python3
"""Inspect problematic segments by extracting frames and overlaying similarity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pickpresence.identity import FaceMatcher, load_reference_embeddings
from pickpresence.detections import load_detection_log

try:  # pragma: no cover - optional dependency for annotation
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class SegmentSpec:
    start: float
    end: float
    label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect segments with annotated frames.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--segments-json", required=True, help="JSON list of {start,end,label}.")
    parser.add_argument("--reference-dir", help="Directory containing reference embedding JSONs.")
    parser.add_argument("--reference-embeddings", nargs="*", help="Explicit reference embedding JSONs.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Detector sample fps.")
    parser.add_argument("--det-threshold", type=float, default=0.35, help="Detector threshold.")
    parser.add_argument("--providers", help="Provider list for detector.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Detector device.")
    parser.add_argument("--model-root", help="InsightFace model root.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames per segment (0 = dump all sampled frames).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write annotated frames with bbox/track/sim (requires cv2).",
    )
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Only write annotated frames (skip raw frames).",
    )
    return parser.parse_args()


def _load_segments(path: Path) -> list[SegmentSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments: list[SegmentSpec] = []
    for item in raw:
        segments.append(
            SegmentSpec(
                start=float(item["start"]),
                end=float(item["end"]),
                label=str(item.get("label") or f"{item['start']:.2f}-{item['end']:.2f}"),
            )
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


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def _extract_clip(video: Path, start: float, end: float, out_path: Path) -> None:
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(out_path),
    ]
    _run(cmd)


def _run_detector(
    root: Path,
    clip_path: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    detector = root / "detectors" / "insightface_detector.py"
    cmd = [
        str(root / ".venv-detector" / "bin" / "python"),
        str(detector),
        "--video",
        str(clip_path),
        "--output",
        str(out_path),
        "--sample-fps",
        str(args.sample_fps),
        "--det-threshold",
        str(args.det_threshold),
        "--device",
        str(args.device),
    ]
    if args.providers:
        cmd += ["--providers", args.providers]
    if args.model_root:
        cmd += ["--model-root", args.model_root]
    _run(cmd)


def _extract_frame(clip_path: Path, timestamp: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    _run(cmd)


def _annotate_frame(image_path: Path, detections: Sequence[dict]) -> None:
    if cv2 is None:
        raise RuntimeError("Annotation requires OpenCV (cv2).")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read frame {image_path}")
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"id:{det.get('track_id')} sim:{det.get('similarity'):.2f} score:{det.get('score'):.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    out_path = image_path.with_name(image_path.stem + "_annotated.png")
    cv2.imwrite(str(out_path), frame)


def _pick_detections(detections: list[dict], max_frames: int) -> list[dict]:
    if max_frames <= 0:
        return detections
    if len(detections) <= max_frames:
        return detections
    sorted_by_time = sorted(detections, key=lambda d: d["start"])
    sorted_by_sim = sorted(detections, key=lambda d: d["similarity"], reverse=True)
    picks: list[dict] = []
    picks.extend(sorted_by_sim[: max_frames // 2])
    picks.extend(sorted_by_time[:: max(1, len(sorted_by_time) // (max_frames // 2))][: max_frames // 2])
    # dedup by frame index
    seen = set()
    unique: list[dict] = []
    for item in picks:
        key = item.get("frame_index", item["start"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max_frames:
            break
    return unique


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = _load_segments(Path(args.segments_json))
    refs = _collect_reference_paths(args)
    matcher = FaceMatcher(load_reference_embeddings(refs), threshold=0.0, agg="topk_avg", topk=2)

    report: dict = {"video": str(args.video), "segments": []}
    for idx, seg in enumerate(segments):
        seg_dir = out_dir / f"seg_{idx:02d}_{seg.label.replace(' ', '_')}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        clip_path = seg_dir / "clip.mp4"
        detections_path = seg_dir / "detections.json"
        _extract_clip(Path(args.video), seg.start, seg.end, clip_path)
        _run_detector(root, clip_path, detections_path, args)
        detections = load_detection_log(detections_path)
        enriched: list[dict] = []
        for det in detections:
            similarity = matcher.similarity(det.embedding)
            enriched.append(
                {
                    "start": det.start,
                    "end": det.end,
                    "track_id": det.track_id,
                    "score": det.base_score,
                    "similarity": similarity,
                    "bbox": det.bbox,
                    "frame_index": det.frame_index,
                    "frame_size": det.frame_size,
                }
            )
        enriched.sort(key=lambda d: d["start"])
        # Group detections by frame index (fallback to start time)
        groups: dict[str, dict] = {}
        for det in enriched:
            key = str(det.get("frame_index", det["start"]))
            group = groups.get(key)
            if group is None:
                groups[key] = {"timestamp": det["start"], "detections": [det]}
            else:
                group["detections"].append(det)
        grouped_frames = list(groups.values())
        grouped_frames.sort(key=lambda g: g["timestamp"])
        # Pick frames to dump
        if args.max_frames > 0 and len(grouped_frames) > args.max_frames:
            step = max(1, len(grouped_frames) // args.max_frames)
            grouped_frames = grouped_frames[::step][: args.max_frames]

        frames_dir = seg_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_idx, group in enumerate(grouped_frames):
            timestamp = group["timestamp"]
            dets = group["detections"]
            sim_max = max((d["similarity"] for d in dets), default=0.0)
            frame_path = frames_dir / f"frame_{frame_idx:03d}_t{timestamp:.2f}_sim{sim_max:.3f}.png"
            _extract_frame(clip_path, timestamp, frame_path)
            if args.annotate:
                _annotate_frame(frame_path, dets)
                if args.annotate_only:
                    try:
                        frame_path.unlink(missing_ok=True)
                    except TypeError:
                        if frame_path.exists():
                            frame_path.unlink()

        report["segments"].append(
            {
                "label": seg.label,
                "start": seg.start,
                "end": seg.end,
                "detections": enriched,
                "frames_dir": str(frames_dir),
            }
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
