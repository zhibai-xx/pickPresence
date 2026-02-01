"""Command-line interface for PickPresence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence

from .pipeline import run_pipeline
from .detector_runner import run_detector_script
from pickpresence.embeddings import combine_embeddings
from pickpresence.detections import DetectionEntry, load_detection_log


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
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Optional chunk size (seconds) for long videos (default 0 = disabled).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume chunked processing by skipping existing chunk outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Alias for --resume (skip chunks with existing outputs).",
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

    if args.chunk_seconds and args.chunk_seconds > 0:
        artifacts = _run_chunked_pipeline(
            args=args,
            output_dir=output_dir,
            annotations=annotations,
            detection_log=detection_log,
            reference_path=reference_path,
            reference_paths=reference_paths,
        )
        print(f"Wrote timeline -> {artifacts.timeline_path}")
        print(f"Generated {len(artifacts.clip_paths)} clip artifact(s).")
    else:
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


def _run_chunked_pipeline(
    args: argparse.Namespace,
    output_dir: Path,
    annotations: list[dict] | None,
    detection_log: str | None,
    reference_path: str | None,
    reference_paths: list[str],
):
    chunk_seconds = max(1.0, float(args.chunk_seconds))
    chunk_root = output_dir / "chunks"
    clips_dir = output_dir / "clips"
    chunk_root.mkdir(parents=True, exist_ok=True)
    if not chunk_root.exists():
        raise RuntimeError("Chunked pipeline failed to create chunks/ directory.")
    print(f"[chunk] root={chunk_root}", flush=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume or args.skip_existing

    detections_all: list[DetectionEntry] | None = None
    if detection_log:
        detections_all = load_detection_log(detection_log)
        video_duration = max((entry.end for entry in detections_all), default=0.0)
    else:
        video_duration = _probe_video_duration(args.video)

    if video_duration <= 0:
        raise RuntimeError("Unable to determine video duration for chunking.")

    chunk_specs = _build_chunk_specs(video_duration, chunk_seconds)
    merged_segments: list[dict] = []
    merged_tracks: list[dict] = []
    merged_clips: list[Path] = []
    clip_index = 0
    per_chunk_stats: list[dict] = []
    skipped_chunks = 0
    processed_chunks = 0
    empty_chunks = 0
    total_elapsed = 0.0
    total_detections = 0
    total_segments = 0
    total_clips = 0

    for chunk_id, start, end in chunk_specs:
        chunk_dir = chunk_root / chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        det_path = chunk_dir / "detections.json"
        timeline_path = chunk_dir / "timeline.json"
        chunk_video_path = chunk_dir / "video.mp4"
        chunk_duration = max(0.0, end - start)
        print(
            f"[chunk] start {chunk_id} {start:.3f}-{end:.3f} "
            f"det={det_path} timeline={timeline_path} "
            f"chunk_seconds={chunk_seconds:.1f} detector={args.detector_script} "
            f"detector_args={args.detector_args or ''}"
            ,
            flush=True,
        )

        if resume and det_path.exists() and timeline_path.exists():
            skipped_chunks += 1
            status = "skipped"
            elapsed = 0.0
        else:
            status = "processed"
            t0 = time.perf_counter()
            if detections_all is None:
                if args.detector_script is None:
                    raise RuntimeError("Chunked processing requires --detector-script or --detection-log.")
                _extract_video_chunk(args.video, chunk_video_path, start, end)
                env = os.environ.copy()
                watchdog_limit = chunk_duration * 3 if chunk_duration > 0 else None
                stdout_log = chunk_dir / "detector_stdout.log"
                stderr_log = chunk_dir / "detector_stderr.log"
                try:
                    run_detector_script(
                        script_path=args.detector_script,
                        video_path=chunk_video_path,
                        output_path=det_path,
                        reference_embedding=reference_path,
                        extra_args=args.detector_args,
                        env=env,
                        timeout_seconds=watchdog_limit,
                        stdout_path=stdout_log,
                        stderr_path=stderr_log,
                    )
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    _write_empty_detections(det_path)
                    _write_empty_chunk_timeline(
                        timeline_path=timeline_path,
                        video=args.video,
                        target_name=args.target_name,
                    )
                except Exception as exc:
                    status = "detector_failed"
                    _write_empty_detections(det_path)
                    _write_empty_chunk_timeline(
                        timeline_path=timeline_path,
                        video=args.video,
                        target_name=args.target_name,
                    )
                    print(f"[chunk] detector_failed {chunk_id}: {exc}", file=sys.stderr)

                if status == "processed":
                    if not det_path.exists():
                        raise RuntimeError(f"[chunk] detector did not write {det_path}")
                    detections = load_detection_log(det_path)
                    detections = _offset_detections(detections, start)
                    _write_detections(det_path, detections)
            else:
                detections = _slice_detections(detections_all, start, end)
                _write_detections(det_path, detections)

            if status == "processed" and detections:
                artifacts = run_pipeline(
                    video_path=args.video,
                    output_dir=str(chunk_dir),
                    target_name=args.target_name,
                    annotations=annotations,
                    detection_log=str(det_path),
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
                chunk_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            else:
                empty_chunks += 1
                chunk_timeline = _write_empty_chunk_timeline(
                    timeline_path=timeline_path,
                    video=args.video,
                    target_name=args.target_name,
                )
            elapsed = time.perf_counter() - t0
            processed_chunks += 1
            total_elapsed += elapsed

        chunk_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        if det_path.exists():
            print(f"[chunk] wrote detections -> {det_path}", flush=True)
        print(f"[chunk] wrote timeline -> {timeline_path}", flush=True)
        chunk_segments = chunk_timeline.get("segments", [])
        chunk_tracks = chunk_timeline.get("tracks", [])
        chunk_clip_files = sorted(chunk_dir.glob("clip_*"))
        segment_count = len(chunk_segments)
        detection_count = len(json.loads(det_path.read_text(encoding="utf-8"))) if det_path.exists() else 0
        total_detections += detection_count

        clip_paths_for_chunk = []
        for seg_idx, seg in enumerate(chunk_segments):
            seg = _prefix_segment_tracks(seg, chunk_id)
            seg["source_chunk"] = chunk_id
            if seg_idx < len(chunk_clip_files):
                source_clip = chunk_clip_files[seg_idx]
                dest = clips_dir / f"clip_{clip_index:03d}{source_clip.suffix}"
                if not dest.exists():
                    shutil.copy2(source_clip, dest)
                seg["clip_path"] = str(dest)
                merged_clips.append(dest)
                clip_paths_for_chunk.append(dest)
                clip_index += 1
            chunk_segments[seg_idx] = seg
        merged_segments.extend(chunk_segments)
        total_segments += len(chunk_segments)
        total_clips += len(clip_paths_for_chunk)

        for track in chunk_tracks:
            track = dict(track)
            track["track_id"] = _prefix_track_id(track.get("track_id"), chunk_id)
            track["source_chunk"] = chunk_id
            merged_tracks.append(track)

        fps = (detection_count / elapsed) if elapsed > 0 else 0.0
        per_chunk_stats.append(
            {
                "chunk_id": chunk_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(chunk_duration, 3),
                "status": status,
                "detections": detection_count,
                "segments": segment_count,
                "clips": len(clip_paths_for_chunk),
                "elapsed_sec": round(elapsed, 3),
                "fps": round(fps, 3),
            }
        )

    merged_segments = sorted(merged_segments, key=lambda seg: seg.get("start", 0.0))
    summary = _summarize_segments(merged_segments)
    timeline_path = output_dir / "timeline.json"
    payload = {
        "video": str(args.video),
        "target": args.target_name or "unknown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "tracks": merged_tracks,
        "segments": merged_segments,
    }
    timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    totals = {
        "chunks": len(chunk_specs),
        "processed": processed_chunks,
        "skipped": skipped_chunks,
        "empty": empty_chunks,
        "elapsed_sec": round(total_elapsed, 3),
        "detections": total_detections,
        "segments": total_segments,
        "clips": total_clips,
        "real_time_factor": round((video_duration / total_elapsed), 3) if total_elapsed > 0 else 0.0,
    }
    chunk_summary_path = chunk_root / "summary.json"
    chunk_summary_path.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "chunk_seconds": chunk_seconds,
                "detector_args": args.detector_args,
                "chunks": per_chunk_stats,
                "totals": totals,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[chunk] Summary -> {chunk_summary_path}")
    return type("Artifacts", (), {"timeline_path": timeline_path, "clip_paths": merged_clips, "segments": merged_segments})()


def _build_chunk_specs(video_duration: float, chunk_seconds: float) -> list[tuple[str, float, float]]:
    specs = []
    idx = 0
    start = 0.0
    while start < video_duration - 1e-6:
        end = min(video_duration, start + chunk_seconds)
        specs.append((f"segment_{idx:03d}", start, end))
        idx += 1
        start = end
    return specs


def _offset_detections(entries: list[DetectionEntry], offset: float) -> list[DetectionEntry]:
    if offset == 0:
        return entries
    for entry in entries:
        entry.start += offset
        entry.end += offset
    return entries


def _slice_detections(entries: list[DetectionEntry], start: float, end: float) -> list[DetectionEntry]:
    sliced: list[DetectionEntry] = []
    for entry in entries:
        if entry.end <= start or entry.start >= end:
            continue
        clipped = DetectionEntry(
            start=max(entry.start, start),
            end=min(entry.end, end),
            embedding=entry.embedding,
            track_id=entry.track_id,
            sources=list(entry.sources),
            base_score=entry.base_score,
            label=entry.label,
            bbox=entry.bbox,
            frame_index=entry.frame_index,
        )
        sliced.append(clipped)
    return sliced


def _write_detections(path: Path, entries: list[DetectionEntry]) -> None:
    payload = []
    for entry in entries:
        payload.append(
            {
                "start": round(entry.start, 3),
                "end": round(entry.end, 3),
                "embedding": entry.embedding,
                "track_id": entry.track_id,
                "sources": entry.sources,
                "score": entry.base_score,
                "label": entry.label,
                "bbox": entry.bbox,
                "frame_index": entry.frame_index,
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_empty_detections(path: Path) -> None:
    path.write_text("[]", encoding="utf-8")


def _extract_video_chunk(video_path: str, output_path: Path, start: float, end: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for chunked detector runs without a detection log.")
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _probe_video_duration(video_path: str) -> float:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required to probe video duration.") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps > 0 and frame_count > 0:
        return float(frame_count / fps)
    return 0.0


def _summarize_segments(segments: list[dict]) -> dict:
    total_duration = 0.0
    total_conf = 0.0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        total_duration += max(0.0, end - start)
        total_conf += float(seg.get("confidence", 0.0))
    return {
        "segment_count": len(segments),
        "total_duration": round(total_duration, 3),
        "average_confidence": round(total_conf / len(segments), 3) if segments else 0.0,
    }


def _prefix_track_id(track_id: str | None, chunk_id: str) -> str | None:
    if track_id is None:
        return None
    return f"{chunk_id}:{track_id}"


def _prefix_segment_tracks(seg: dict, chunk_id: str) -> dict:
    seg = dict(seg)
    seg["track_id"] = _prefix_track_id(seg.get("track_id"), chunk_id)
    seg["primary_track_id"] = _prefix_track_id(seg.get("primary_track_id"), chunk_id)
    if seg.get("contrib_track_ids"):
        seg["contrib_track_ids"] = [
            _prefix_track_id(track_id, chunk_id) for track_id in seg["contrib_track_ids"]
        ]
    return seg


def _write_empty_chunk_timeline(
    timeline_path: Path,
    video: str,
    target_name: str | None,
) -> dict:
    payload = {
        "video": str(video),
        "target": target_name or "unknown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {"segment_count": 0, "total_duration": 0.0, "average_confidence": 0.0},
        "tracks": [],
        "segments": [],
    }
    timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
