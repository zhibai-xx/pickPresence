"""Command-line interface for PickPresence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .pipeline import run_pipeline
from .detector_runner import run_detector_script
from pickpresence.embeddings import combine_embeddings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pickpresence",
        description="Generate presence timelines and export clips from a video.",
    )
    parser.add_argument(
        "--track-policy",
        choices=["best", "all"],
        default="best",
        help="Whether to keep only the best-matching track or every track over threshold.",
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", required=True, help="Directory for artifacts.")
    parser.add_argument(
        "--annotations",
        help="Path to optional JSON file containing pre-computed segments.",
    )
    detector_group = parser.add_argument_group("detector generation")
    parser.add_argument(
        "--detection-log",
        help="Path to JSON list of detection entries (with embeddings + timing).",
    )
    detector_group.add_argument(
        "--detector-script",
        help="Executable that takes --video/--output[/--reference] and writes detections JSON.",
        default=os.environ.get("PICKPRESENCE_DETECTOR_SCRIPT"),
    )
    detector_group.add_argument(
        "--detector-output",
        help="Where to write detections when running --detector-script "
        "(defaults to <output-dir>/detections.json).",
    )
    detector_group.add_argument(
        "--detector-args",
        help="Extra CLI args appended to --detector-script invocation (quoted string, optional).",
    )
    parser.add_argument(
        "--reference-embedding",
        help="Path to JSON file describing the target embedding (name + vector).",
    )
    parser.add_argument(
        "--reference-embeddings",
        nargs="+",
        help="Additional reference embeddings to combine into a template.",
    )
    parser.add_argument(
        "--reference-dir",
        help="Directory of reference embedding JSON files to include.",
    )
    parser.add_argument(
        "--reference-list-file",
        help="Text file containing reference embedding paths (one per line).",
    )
    parser.add_argument(
        "--reference-agg",
        choices=["max", "topk_avg"],
        default="max",
        help="Multi-reference aggregation strategy (default max).",
    )
    parser.add_argument(
        "--reference-topk",
        type=int,
        default=3,
        help="Top-k size for reference aggregation when using topk_avg (default 3).",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Human-readable identifier for the desired person.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum duration (seconds) to keep a segment.",
    )
    parser.add_argument(
        "--bridge-gap",
        type=float,
        default=0.5,
        help="Maximum gap (seconds) to merge segments for continuity.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.8,
        help="Minimum cosine similarity to treat detection as a match.",
    )
    parser.add_argument(
        "--match-threshold-start",
        type=float,
        help="Hysteresis start threshold (defaults to --match-threshold).",
    )
    parser.add_argument(
        "--match-threshold-keep",
        type=float,
        help="Hysteresis keep threshold (defaults to 0.5 * start).",
    )
    parser.add_argument(
        "--segment-policy",
        choices=["per_detection", "track_first", "hysteresis"],
        default="per_detection",
        help="How to convert detections into segments (default per_detection).",
    )
    parser.add_argument(
        "--merge-policy",
        choices=["none", "union"],
        default="none",
        help="Optional post-processing rule to merge segments across tracks (union ignores track_id gaps).",
    )
    parser.add_argument(
        "--export-end-eps",
        type=float,
        default=0.2,
        help="Seconds to subtract from segment end when exporting clips (safety padding).",
    )
    parser.add_argument(
        "--trim-policy",
        choices=["none", "head_tail"],
        default="none",
        help="Optional head/tail trimming policy applied after merge (default none).",
    )
    parser.add_argument(
        "--trim-source",
        choices=["detections", "video"],
        default="detections",
        help="Data source used for trimming (video scans require InsightFace).",
    )
    parser.add_argument(
        "--trim-threshold-start",
        type=float,
        help="Similarity threshold to start a trimmed segment (defaults to match/hysteresis start).",
    )
    parser.add_argument(
        "--trim-threshold-keep",
        type=float,
        help="Similarity threshold to keep the tail during trimming (defaults to hysteresis keep).",
    )
    parser.add_argument(
        "--trim-min-run",
        type=int,
        default=2,
        help="Minimum consecutive detections that must pass the threshold before trimming (default 2).",
    )
    parser.add_argument(
        "--trim-pad",
        type=float,
        default=0.2,
        help="Seconds to pad head/tail after trimming succeeds (default 0.2).",
    )
    parser.add_argument(
        "--trim-scan-window",
        type=float,
        default=0.6,
        help="Seconds to scan at each boundary when --trim-source video.",
    )
    parser.add_argument(
        "--trim-scan-step",
        type=float,
        default=0.04,
        help="Step interval (seconds) for video scanning when --trim-source video.",
    )
    parser.add_argument(
        "--min-track-duration",
        type=float,
        default=0.5,
        help="Minimum accumulated duration (seconds) for a track to be considered.",
    )
    parser.add_argument(
        "--min-track-similarity",
        type=float,
        default=None,
        help="Override similarity threshold for track-level filtering (defaults to --match-threshold).",
    )
    parser.add_argument(
        "--force-placeholder-export",
        action="store_true",
        help="Skip ffmpeg invocation even if it exists (useful for tests).",
    )
    return parser.parse_args(argv)


def _load_annotations(path: str | None) -> list[dict] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Annotations JSON must be a list.")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    annotations = _load_annotations(args.annotations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path, reference_paths = _resolve_reference_paths(args, output_dir)

    detection_log = args.detection_log
    if detection_log is None and args.detector_script:
        detection_log = _run_detector_and_get_path(args, reference_path)

    artifacts = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        target_name=args.target_name,
        annotations=annotations,
        detection_log=detection_log,
        reference_embedding=reference_path,
        reference_embeddings=reference_paths,
        reference_agg=args.reference_agg,
        reference_topk=args.reference_topk,
        min_duration=args.min_duration,
        bridge_gap=args.bridge_gap,
        prefer_ffmpeg=not args.force_placeholder_export,
        match_threshold=args.match_threshold,
        match_threshold_start=args.match_threshold_start,
        match_threshold_keep=args.match_threshold_keep,
        segment_policy=args.segment_policy,
        track_policy=args.track_policy,
        min_track_duration=args.min_track_duration,
        min_track_similarity=args.min_track_similarity,
        merge_policy=args.merge_policy,
        export_end_eps=args.export_end_eps,
        trim_policy=args.trim_policy,
        trim_source=args.trim_source,
        trim_threshold_start=args.trim_threshold_start,
        trim_threshold_keep=args.trim_threshold_keep,
        trim_min_run=args.trim_min_run,
        trim_pad=args.trim_pad,
        trim_scan_window=args.trim_scan_window,
        trim_scan_step=args.trim_scan_step,
    )
    print(f"Wrote timeline -> {artifacts.timeline_path}")
    print(f"Generated {len(artifacts.clip_paths)} clip artifact(s).")
    return 0


def _run_detector_and_get_path(args: argparse.Namespace, reference_path: str | None) -> str:
    output_dir = Path(args.output_dir)
    detector_output = (
        Path(args.detector_output)
        if args.detector_output
        else output_dir / "detector_output" / "detections.json"
    )
    env = os.environ.copy()
    run_detector_script(
        script_path=args.detector_script,
        video_path=args.video,
        output_path=detector_output,
        reference_embedding=reference_path,
        extra_args=args.detector_args,
        env=env,
    )
    return str(detector_output)


def _resolve_reference_paths(
    args: argparse.Namespace, output_dir: Path
) -> tuple[str | None, list[str]]:
    paths = _gather_reference_paths(args)
    if not paths:
        return None, []
    if len(paths) == 1:
        return paths[0], paths
    template = _combine_reference_embeddings(paths, args.target_name)
    out_path = output_dir / "reference_template.json"
    out_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return str(out_path), paths


def _combine_reference_embeddings(paths: Sequence[str], name: str | None) -> dict:
    vectors = []
    ref_name = name
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vectors.append(data["embedding"])
        if not ref_name:
            ref_name = data.get("name")
    combined = combine_embeddings(vectors)
    return {"name": ref_name or "template", "embedding": combined}


def _gather_reference_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.reference_embedding:
        paths.append(args.reference_embedding)
    if args.reference_embeddings:
        paths.extend(args.reference_embeddings)
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        if ref_dir.is_dir():
            paths.extend(str(path) for path in sorted(ref_dir.glob("*.json")))
    if args.reference_list_file:
        list_path = Path(args.reference_list_file)
        if list_path.exists():
            for line in list_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(line)
    dedup: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        dedup.append(path)
    return dedup


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
