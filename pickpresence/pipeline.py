"""Core timeline generation and clip export utilities for PickPresence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Sequence

try:  # pragma: no cover - optional dependency for video trimming/audit stats
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .clustering import TrackSelector, TrackSelection
from .detections import DetectionEntry, load_detection_log
from .identity import FaceMatcher, load_reference_embedding, load_reference_embeddings


@dataclass
class Segment:
    """Time-bounded presence segment."""

    start: float
    end: float
    confidence: float
    sources: list[str] = field(default_factory=list)
    track_id: str | None = None
    contrib_track_ids: list[str] | None = None
    primary_track_id: str | None = None
    match_avg: float | None = None
    match_max: float | None = None
    match_p90: float | None = None
    best_ref_id: str | None = None
    best_ref_sim: float | None = None
    best_ref_p90: float | None = None
    ref_topk_avg: float | None = None
    ref_hits: list[dict] | None = None
    export_start: float | None = None
    export_end: float | None = None

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        payload = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "track_id": self.track_id,
        }
        if self.contrib_track_ids:
            payload["contrib_track_ids"] = self.contrib_track_ids
        payload["primary_track_id"] = self.primary_track_id
        payload["match_avg"] = round(self.match_avg, 3) if self.match_avg is not None else None
        payload["match_max"] = round(self.match_max, 3) if self.match_max is not None else None
        payload["match_p90"] = round(self.match_p90, 3) if self.match_p90 is not None else None
        if self.best_ref_id is not None:
            payload["best_ref_id"] = self.best_ref_id
        if self.best_ref_sim is not None:
            payload["best_ref_sim"] = round(self.best_ref_sim, 3)
        if self.best_ref_p90 is not None:
            payload["best_ref_p90"] = round(self.best_ref_p90, 3)
        if self.ref_topk_avg is not None:
            payload["ref_topk_avg"] = round(self.ref_topk_avg, 3)
        if self.ref_hits is not None:
            payload["ref_hits"] = self.ref_hits
        if self.export_start is not None or self.export_end is not None:
            payload["export_start"] = (
                round(self.export_start, 3) if self.export_start is not None else None
            )
            payload["export_end"] = (
                round(self.export_end, 3) if self.export_end is not None else None
            )
        return payload


@dataclass
class TimelineArtifacts:
    """Holds output artifact locations for downstream reporting."""

    timeline_path: Path
    clip_paths: List[Path]
    segments: List[Segment]


class TimelineBuilder:
    """Generates simplified identity timelines with bridge/filter rules."""

    def __init__(
        self,
        min_duration: float = 1.0,
        bridge_gap: float = 0.5,
        policy: str = "per_detection",
        match_threshold_start: float | None = None,
        match_threshold_keep: float | None = None,
    ) -> None:
        if min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if bridge_gap < 0:
            raise ValueError("bridge_gap must be non-negative")
        if policy not in {"per_detection", "track_first", "hysteresis"}:
            raise ValueError("Invalid segment policy")
        self.min_duration = min_duration
        self.bridge_gap = bridge_gap
        self.policy = policy
        self.match_threshold_start = match_threshold_start
        self.match_threshold_keep = match_threshold_keep

    def build(
        self,
        video_path: str | Path,
        target_name: str | None = None,
        annotations: Sequence[dict] | None = None,
        detections: Sequence[DetectionEntry] | None = None,
        matcher: FaceMatcher | None = None,
        track_summaries: Sequence[TrackSelection] | None = None,
    ) -> List[Segment]:
        if annotations:
            segments = self._from_annotations(annotations)
        elif detections and matcher:
            if self.policy == "track_first" and track_summaries:
                segments = self._from_track_summaries(track_summaries)
            elif self.policy == "hysteresis":
                segments = self._from_hysteresis(detections, matcher)
            else:
                segments = self._from_detections(detections, matcher)
            if not segments:
                segments = self._generate_stub_segments(video_path, target_name)
        else:
            segments = self._generate_stub_segments(video_path, target_name)
        bridged = self._bridge_segments(segments)
        filtered = [seg for seg in bridged if seg.duration() >= self.min_duration]
        return filtered

    def _from_annotations(self, annotations: Sequence[dict]) -> List[Segment]:
        parsed = []
        for raw in annotations:
            track_id = str(raw["track_id"]) if "track_id" in raw else None
            parsed.append(
                Segment(
                    start=float(raw["start"]),
                    end=float(raw["end"]),
                    confidence=float(raw.get("confidence", 0.5)),
                    sources=list(raw.get("sources", ["provided"])),
                    track_id=track_id,
                    contrib_track_ids=[track_id] if track_id else None,
                )
            )
        parsed.sort(key=lambda seg: seg.start)
        return parsed

    def _generate_stub_segments(
        self, video_path: str | Path, target_name: str | None
    ) -> List[Segment]:
        path = Path(video_path)
        file_size = max(path.stat().st_size, 1)
        base = max(file_size / 250.0, self.min_duration * 1.1)
        segments = [
            Segment(
                start=0.0,
                end=round(base, 3),
                confidence=0.85,
                sources=["stub-face", target_name or "unknown"],
            ),
            Segment(
                start=round(base + 0.2, 3),
                end=round(base * 1.6 + 0.2, 3),
                confidence=0.7,
                sources=["stub-tracker"],
            ),
        ]
        return segments

    def _from_detections(
        self,
        detections: Sequence[DetectionEntry],
        matcher: FaceMatcher,
    ) -> List[Segment]:
        segments: List[Segment] = []
        for entry in detections:
            if entry is None or entry.track_id is None:
                continue
            score = matcher.similarity(entry.embedding)
            if score < matcher.threshold and not entry.force_keep:
                continue
            confidence = max(score, entry.base_score)
            sources = [*entry.sources, f"track:{entry.track_id}", "face-match"]
            segments.append(
                Segment(
                    start=entry.start,
                    end=entry.end,
                    confidence=confidence,
                    sources=sources,
                    track_id=str(entry.track_id),
                    contrib_track_ids=[str(entry.track_id)],
                )
            )
        return segments

    def _from_track_summaries(self, tracks: Sequence[TrackSelection]) -> List[Segment]:
        segments: List[Segment] = []
        for summary in tracks:
            confidence = max(summary.score, summary.avg_similarity)
            for entry in summary.entries:
                sources = [*entry.sources, f"track:{summary.track_id}", "track-first"]
                segments.append(
                    Segment(
                        start=entry.start,
                        end=entry.end,
                        confidence=max(confidence, entry.base_score),
                        sources=sources,
                        track_id=str(summary.track_id),
                        contrib_track_ids=[str(summary.track_id)],
                    )
                )
        return segments

    def _from_hysteresis(
        self,
        detections: Sequence[DetectionEntry],
        matcher: FaceMatcher,
    ) -> List[Segment]:
        start_th = self.match_threshold_start or matcher.threshold
        keep_th = (
            self.match_threshold_keep
            if self.match_threshold_keep is not None
            else max(0.2, start_th * 0.5)
        )
        segments: List[Segment] = []
        active_track: str | None = None
        for entry in detections:
            if entry.track_id is None:
                continue
            score = matcher.similarity(entry.embedding)
            track_id = str(entry.track_id)
            should_keep = score >= keep_th or entry.force_keep
            should_start = score >= start_th or entry.force_keep
            if active_track == track_id:
                if should_keep:
                    segments.append(
                        Segment(
                            start=entry.start,
                            end=entry.end,
                            confidence=max(score, entry.base_score),
                            sources=[*entry.sources, f"track:{track_id}", "hysteresis"],
                            track_id=track_id,
                            contrib_track_ids=[track_id],
                        )
                    )
                else:
                    active_track = None
            else:
                if should_start:
                    active_track = track_id
                    segments.append(
                        Segment(
                            start=entry.start,
                            end=entry.end,
                            confidence=max(score, entry.base_score),
                            sources=[*entry.sources, f"track:{track_id}", "hysteresis"],
                            track_id=track_id,
                            contrib_track_ids=[track_id],
                        )
                    )
        return segments

    def _bridge_segments(self, segments: Sequence[Segment]) -> List[Segment]:
        bridged: List[Segment] = []
        for seg in segments:
            if not bridged:
                bridged.append(seg)
                continue
            prev = bridged[-1]
            gap = seg.start - prev.end
            same_track = (
                seg.track_id == prev.track_id or seg.track_id is None or prev.track_id is None
            )
            if gap <= self.bridge_gap and same_track:
                prev.end = max(prev.end, seg.end)
                prev.confidence = max(prev.confidence, seg.confidence)
                prev.sources = list(dict.fromkeys([*prev.sources, *seg.sources]))
                prev.contrib_track_ids = _merge_contrib(
                    prev.contrib_track_ids, seg.contrib_track_ids
                )
            else:
                bridged.append(seg)
        return bridged


def merge_segments_union(
    segments: Sequence[Segment], bridge_gap: float, tolerance: float = 1e-6
) -> List[Segment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda seg: seg.start)
    merged: List[Segment] = []
    current: Segment | None = None
    for seg in ordered:
        if current is None:
            current = _clone_for_union(seg)
            continue
        gap = seg.start - current.end
        if gap <= bridge_gap + tolerance:
            current.end = max(current.end, seg.end)
            current.confidence = max(current.confidence, seg.confidence)
            current.sources = list(dict.fromkeys([*current.sources, *seg.sources]))
            current.contrib_track_ids = _merge_contrib(
                current.contrib_track_ids, _segment_contrib_ids(seg)
            )
        else:
            _finalize_union_segment(current)
            merged.append(current)
            current = _clone_for_union(seg)
    if current is not None:
        _finalize_union_segment(current)
        merged.append(current)
    return merged


def _clone_for_union(seg: Segment) -> Segment:
    return Segment(
        start=seg.start,
        end=seg.end,
        confidence=seg.confidence,
        sources=list(dict.fromkeys(seg.sources)),
        track_id=None,
        contrib_track_ids=_segment_contrib_ids(seg),
    )


def _segment_contrib_ids(seg: Segment) -> list[str] | None:
    if seg.contrib_track_ids:
        return list(seg.contrib_track_ids)
    if seg.track_id:
        return [str(seg.track_id)]
    return None


def _finalize_union_segment(seg: Segment) -> None:
    seg.track_id = None
    seg.contrib_track_ids = _merge_contrib(seg.contrib_track_ids, None)
    if "union-merge" not in seg.sources:
        seg.sources.append("union-merge")
    seg.sources = list(dict.fromkeys(seg.sources))



def _merge_contrib(
    first: list[str] | None, second: list[str] | None
) -> list[str] | None:
    if not first and not second:
        return None
    items = []
    if first:
        items.extend(first)
    if second:
        items.extend(second)
    dedup = sorted(set(items))
    return dedup


def apply_trim_policy(
    segments: Sequence[Segment],
    detections: Sequence[DetectionEntry],
    matcher: FaceMatcher | None,
    policy: str,
    start_threshold: float,
    keep_threshold: float,
    min_run: int,
    pad: float,
    min_duration: float,
    video_path: str | Path,
    trim_source: str,
    scan_window: float,
    scan_step: float,
    model_name: str = "buffalo_l",
) -> List[Segment]:
    if not segments:
        return []
    if policy not in {"none", "head_tail"}:
        raise ValueError("trim policy must be 'none' or 'head_tail'")
    if trim_source not in {"detections", "video"}:
        raise ValueError("trim source must be 'detections' or 'video'")
    trimmed: List[Segment] = []
    has_matcher = matcher is not None
    video_scanner: VideoTrimScanner | None = None
    use_video = trim_source == "video" and scan_window > 0 and scan_step > 0
    if policy == "head_tail" and use_video:
        if matcher is None:
            raise RuntimeError("Video-based trimming requires a reference embedding.")
        video_scanner = VideoTrimScanner(
            video_path=video_path,
            matcher=matcher,
            scan_window=scan_window,
            scan_step=scan_step,
            model_name=model_name,
        )
    for seg in segments:
        relevant_entries = _collect_entries_for_segment(seg, detections)
        stats = _compute_match_stats(relevant_entries, matcher)
        seg.match_avg = stats.avg
        seg.match_max = stats.max_score
        seg.match_p90 = stats.p90
        seg.best_ref_id = stats.best_ref_id
        seg.best_ref_sim = stats.best_ref_sim
        seg.best_ref_p90 = stats.best_ref_p90
        seg.ref_topk_avg = stats.ref_topk_avg
        seg.ref_hits = stats.ref_hits
        if seg.primary_track_id is None:
            seg.primary_track_id = stats.primary_track_id or seg.track_id
        if seg.primary_track_id is None and seg.contrib_track_ids:
            seg.primary_track_id = seg.contrib_track_ids[0]
        if policy != "head_tail" or not has_matcher:
            trimmed.append(seg)
            continue
        bounds: tuple[float, float] | None = None
        sources_to_add: list[str] = []
        if trim_source == "video" and video_scanner is not None:
            bounds = video_scanner.trim_segment(
                seg=seg,
                start_threshold=start_threshold,
                keep_threshold=keep_threshold,
                min_run=min_run,
                pad=pad,
            )
            if bounds:
                sources_to_add.append("video-trim")
        if bounds is None:
            bounds = _trim_segment_bounds(
                seg=seg,
                entries=relevant_entries,
                matcher=matcher,
                start_threshold=start_threshold,
                keep_threshold=keep_threshold,
                min_run=min_run,
                pad=pad,
            )
            if bounds:
                sources_to_add.append("det-trim")
        if bounds is None:
            seg.sources = list(dict.fromkeys([*seg.sources, "trim-failed"]))
            continue
        seg.start, seg.end = bounds
        seg.sources = list(dict.fromkeys([*seg.sources, *sources_to_add]))
        if seg.duration() >= min_duration:
            trimmed.append(seg)
    if video_scanner is not None:
        video_scanner.close()
    return sorted(trimmed, key=lambda seg: seg.start)


def _collect_entries_for_segment(
    segment: Segment, detections: Sequence[DetectionEntry], tolerance: float = 1e-6
) -> List[DetectionEntry]:
    if not detections:
        return []
    allowed = None
    if segment.contrib_track_ids:
        allowed = {str(track_id) for track_id in segment.contrib_track_ids}
    elif segment.track_id:
        allowed = {str(segment.track_id)}
    relevant: List[DetectionEntry] = []
    for entry in detections:
        track_key = str(entry.track_id)
        if allowed is not None and track_key not in allowed:
            continue
        if entry.end < segment.start - tolerance:
            continue
        if entry.start > segment.end + tolerance:
            continue
        relevant.append(entry)
    relevant.sort(key=lambda entry: entry.start)
    return relevant


def _trim_segment_bounds(
    seg: Segment,
    entries: Sequence[DetectionEntry],
    matcher: FaceMatcher,
    start_threshold: float,
    keep_threshold: float,
    min_run: int,
    pad: float,
) -> tuple[float, float] | None:
    if not entries:
        return None
    start_time = _find_forward_run(entries, matcher, start_threshold, min_run)
    end_time = _find_backward_run(entries, matcher, keep_threshold, min_run)
    if start_time is None or end_time is None or end_time <= start_time:
        return None
    trimmed_start = max(seg.start, start_time - pad)
    trimmed_end = min(seg.end, end_time + pad)
    if trimmed_end <= trimmed_start:
        return None
    return trimmed_start, trimmed_end


def _find_forward_run(
    entries: Sequence[DetectionEntry],
    matcher: FaceMatcher,
    threshold: float,
    min_run: int,
) -> float | None:
    count = 0
    run_start = None
    for entry in entries:
        score = _score_entry(entry, matcher)
        if score >= threshold:
            count += 1
            if count == 1:
                run_start = entry.start
            if count >= min_run and run_start is not None:
                return run_start
        else:
            count = 0
            run_start = None
    return None


def _find_backward_run(
    entries: Sequence[DetectionEntry],
    matcher: FaceMatcher,
    threshold: float,
    min_run: int,
) -> float | None:
    count = 0
    run_end = None
    for entry in reversed(entries):
        score = _score_entry(entry, matcher)
        if score >= threshold:
            count += 1
            if count == 1:
                run_end = entry.end
            if count >= min_run and run_end is not None:
                return run_end
        else:
            count = 0
            run_end = None
    return None


def _compute_match_stats(
    entries: Sequence[DetectionEntry],
    matcher: FaceMatcher | None,
) -> _MatchStats:
    if matcher is None or not entries:
        return _MatchStats()
    scores: List[float] = []
    per_track: dict[str, List[float]] = {}
    best_ref_ids: List[str] = []
    best_ref_sims: List[float] = []
    topk_avgs: List[float] = []
    for entry in entries:
        score = _score_entry(entry, matcher)
        scores.append(score)
        key = str(entry.track_id)
        per_track.setdefault(key, []).append(score)
        if entry.best_ref_id is not None:
            best_ref_ids.append(entry.best_ref_id)
        if entry.best_ref_sim is not None:
            best_ref_sims.append(entry.best_ref_sim)
        if entry.ref_topk_avg is not None:
            topk_avgs.append(entry.ref_topk_avg)
    if not scores:
        return _MatchStats()
    avg = sum(scores) / len(scores)
    max_score = max(scores)
    p90 = _percentile(scores, 0.9)
    primary = None
    if per_track:
        primary = max(
            per_track.items(),
            key=lambda item: (sum(item[1]) / len(item[1]), len(item[1])),
        )[0]
    ref_stats = _summarize_best_refs(best_ref_ids, best_ref_sims, topk_avgs)
    return _MatchStats(
        avg=avg,
        max_score=max_score,
        p90=p90,
        primary_track_id=primary,
        best_ref_id=ref_stats.best_ref_id,
        best_ref_sim=ref_stats.best_ref_sim,
        best_ref_p90=ref_stats.best_ref_p90,
        ref_topk_avg=ref_stats.ref_topk_avg,
        ref_hits=ref_stats.ref_hits,
    )


def _score_entry(entry: DetectionEntry, matcher: FaceMatcher) -> float:
    if entry.similarity is None:
        details = matcher.match_details(entry.embedding)
        entry.similarity = details.score
        entry.best_ref_id = details.best_ref_id
        entry.best_ref_sim = details.best_ref_sim
        entry.ref_topk_avg = details.topk_avg
    return entry.similarity


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


class VideoTrimScanner:
    """Extracts presence confidence by sampling frames near segment bounds."""

    def __init__(
        self,
        video_path: str | Path,
        matcher: FaceMatcher,
        scan_window: float,
        scan_step: float,
        model_name: str = "buffalo_l",
    ) -> None:
        if scan_window <= 0 or scan_step <= 0:
            raise ValueError("scan_window and scan_step must be positive for video trimming.")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Video-based trimming requires OpenCV (cv2).") from exc
        from detectors.insightface_detector import _load_insightface_app

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video for trimming: {video_path}")
        if np is None:
            raise RuntimeError("Video-based trimming requires numpy to compute similarities.")
        self.app = _load_insightface_app(model_name, device="cpu")
        self.matcher = matcher
        self.scan_window = scan_window
        self.scan_step = scan_step

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def trim_segment(
        self,
        seg: Segment,
        start_threshold: float,
        keep_threshold: float,
        min_run: int,
        pad: float,
    ) -> tuple[float, float] | None:
        if self.scan_window <= 0:
            return None
        start_candidate = self._scan_forward(
            window_start=seg.start,
            window_end=min(seg.end, seg.start + self.scan_window),
            threshold=start_threshold,
            min_run=min_run,
        )
        end_candidate = self._scan_backward(
            window_start=max(seg.start, seg.end - self.scan_window),
            window_end=seg.end,
            threshold=keep_threshold,
            min_run=min_run,
        )
        if start_candidate is None or end_candidate is None or end_candidate <= start_candidate:
            return None
        trimmed_start = max(seg.start, start_candidate - pad)
        trimmed_end = min(seg.end, end_candidate + pad)
        if trimmed_end <= trimmed_start:
            return None
        return trimmed_start, trimmed_end

    def _scan_forward(
        self,
        window_start: float,
        window_end: float,
        threshold: float,
        min_run: int,
    ) -> float | None:
        if window_end <= window_start:
            return None
        t = window_start
        hits = 0
        run_start = None
        while t <= window_end + 1e-6:
            score = self._frame_similarity(t)
            if score is not None and score >= threshold:
                hits += 1
                if hits == 1:
                    run_start = t
                if hits >= min_run and run_start is not None:
                    return run_start
            else:
                hits = 0
                run_start = None
            t += self.scan_step
        return None

    def _scan_backward(
        self,
        window_start: float,
        window_end: float,
        threshold: float,
        min_run: int,
    ) -> float | None:
        if window_end <= window_start:
            return None
        t = window_end
        hits = 0
        run_end = None
        while t >= window_start - 1e-6:
            score = self._frame_similarity(t)
            if score is not None and score >= threshold:
                hits += 1
                if hits == 1:
                    run_end = t
                if hits >= min_run and run_end is not None:
                    return run_end
            else:
                hits = 0
                run_end = None
            t -= self.scan_step
        return None

    def _frame_similarity(self, timestamp: float) -> float | None:
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ret, frame = self.cap.read()
        if not ret:
            return None
        faces = self.app.get(frame)
        if not faces:
            return None
        scores = [
            self.matcher.match_details(face.normed_embedding).score for face in faces
        ]
        if not scores:
            return None
        return max(scores)


class ClipExporter:
    """Handles ffmpeg (or placeholder) export of per-segment clips."""

    def __init__(self, prefer_ffmpeg: bool = True) -> None:
        self.prefer_ffmpeg = prefer_ffmpeg

    def export(
        self, video_path: str | Path, output_dir: str | Path, segments: Sequence[Segment]
    ) -> List[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[Path] = []
        use_ffmpeg = self._should_use_ffmpeg()
        for idx, seg in enumerate(segments):
            clip_name = f"clip_{idx:03d}"
            start = seg.export_start if seg.export_start is not None else seg.start
            end = seg.export_end if seg.export_end is not None else seg.end
            if end <= start:
                continue
            duration = end - start
            if use_ffmpeg:
                clip_path = out_dir / f"{clip_name}.mp4"
                self._run_ffmpeg(video_path, clip_path, start, duration)
            else:
                clip_path = out_dir / f"{clip_name}.txt"
                clip_path.write_text(
                    (
                        f"placeholder clip for {Path(video_path).name}\n"
                        f"start={start}, end={end}, confidence={seg.confidence}\n"
                        f"sources={','.join(seg.sources)}\n"
                    ),
                    encoding="utf-8",
                )
            artifacts.append(clip_path)
        return artifacts

    def _run_ffmpeg(self, video_path: str | Path, clip_path: Path, start: float, duration: float) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{max(0.01, duration):.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _should_use_ffmpeg(self) -> bool:
        if not self.prefer_ffmpeg:
            return False
        if os.environ.get("PICKPRESENCE_FORCE_PLACEHOLDER") == "1":
            return False
        return shutil.which("ffmpeg") is not None


def _write_timeline(
    timeline_path: Path,
    video_path: str | Path,
    target_name: str | None,
    segments: Sequence[Segment],
    summary: dict,
    track_summaries: Sequence[dict],
) -> None:
    payload = {
        "video": str(video_path),
        "target": target_name or "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "tracks": track_summaries,
        "segments": [seg.to_dict() for seg in segments],
    }
    timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    target_name: str | None = None,
    annotations: Sequence[dict] | None = None,
    detection_log: str | Path | None = None,
    reference_embedding: str | Path | None = None,
    reference_embeddings: Sequence[str | Path] | None = None,
    reference_agg: str = "max",
    reference_topk: int = 3,
    min_duration: float = 1.0,
    bridge_gap: float = 0.5,
    prefer_ffmpeg: bool = True,
    match_threshold: float = 0.8,
    match_threshold_start: float | None = None,
    match_threshold_keep: float | None = None,
    segment_policy: str = "per_detection",
    track_policy: str = "best",
    min_track_duration: float = 0.5,
    min_track_similarity: float | None = None,
    merge_policy: str = "none",
    trim_policy: str = "none",
    trim_threshold_start: float | None = None,
    trim_threshold_keep: float | None = None,
    trim_min_run: int = 2,
    trim_pad: float = 0.2,
    export_end_eps: float = 0.2,
    trim_source: str = "detections",
    trim_scan_window: float = 0.6,
    trim_scan_step: float = 0.04,
) -> TimelineArtifacts:
    """High-level helper tying together timeline + export."""

    if merge_policy not in {"none", "union"}:
        raise ValueError("merge_policy must be 'none' or 'union'")
    detections = load_detection_log(detection_log) if detection_log else None
    matcher = None
    reference_paths: list[str | Path] = []
    if reference_embedding:
        reference_paths.append(reference_embedding)
    if reference_embeddings:
        reference_paths.extend(reference_embeddings)
    dedup_paths: list[str | Path] = []
    seen_paths: set[str] = set()
    for path in reference_paths:
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        dedup_paths.append(path)
    if dedup_paths:
        references = load_reference_embeddings(dedup_paths)
        matcher = FaceMatcher(
            references,
            threshold=match_threshold,
            agg=reference_agg,
            topk=reference_topk,
        )
        if target_name is None and references:
            target_name = references[0].name
    selected_tracks = None
    if detections:
        selector = TrackSelector(
            matcher=matcher,
            policy=track_policy,
            min_avg_similarity=min_track_similarity
            if min_track_similarity is not None
            else match_threshold,
            min_total_duration=min_track_duration,
        )
        selected_tracks = selector.select(detections)
        detections = _flatten_selections(selected_tracks)
        if target_name is None and selected_tracks:
            target_name = selected_tracks[0].label or f"track_{selected_tracks[0].track_id}"

    start_th = match_threshold_start or match_threshold
    builder = TimelineBuilder(
        min_duration=min_duration,
        bridge_gap=bridge_gap,
        policy=segment_policy,
        match_threshold_start=start_th,
        match_threshold_keep=match_threshold_keep,
    )
    segments = builder.build(
        video_path=video_path,
        target_name=target_name,
        annotations=annotations,
        detections=detections,
        matcher=matcher,
        track_summaries=selected_tracks,
    )
    if merge_policy == "union":
        segments = merge_segments_union(segments, bridge_gap=bridge_gap)
    trim_start = trim_threshold_start or start_th
    keep_default = (
        trim_threshold_keep
        if trim_threshold_keep is not None
        else (
            match_threshold_keep
            if match_threshold_keep is not None
            else max(0.2, start_th * 0.5)
        )
    )
    segments = apply_trim_policy(
        segments=segments,
        detections=detections or [],
        matcher=matcher,
        policy=trim_policy,
        start_threshold=trim_start,
        keep_threshold=keep_default,
        min_run=max(1, trim_min_run),
        pad=max(0.0, trim_pad),
        min_duration=min_duration,
        video_path=video_path,
        trim_source=trim_source,
        scan_window=max(0.0, trim_scan_window),
        scan_step=max(0.01, trim_scan_step),
        model_name="buffalo_l",
    )
    segments = apply_export_padding(segments, export_end_eps)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = out_dir / "timeline.json"
    summary = _summarize_timeline(segments)
    track_summaries = _summarize_tracks(selected_tracks, segments)
    _write_timeline(timeline_path, video_path, target_name, segments, summary, track_summaries)

    exporter = ClipExporter(prefer_ffmpeg=prefer_ffmpeg)
    clips = exporter.export(video_path, out_dir, segments)

    return TimelineArtifacts(timeline_path=timeline_path, clip_paths=clips, segments=segments)


def apply_export_padding(segments: Sequence[Segment], export_end_eps: float) -> List[Segment]:
    padding = max(0.0, export_end_eps)
    for seg in segments:
        seg.export_start = seg.start
        if padding > 0:
            seg.export_end = max(seg.start, seg.end - padding)
        else:
            seg.export_end = seg.end
    return list(segments)


def _flatten_selections(selections: Sequence[TrackSelection]) -> List[DetectionEntry]:
    flattened: List[DetectionEntry] = []
    for summary in selections:
        for entry in summary.entries:
            entry.force_keep = True
            flattened.append(entry)
    return sorted(flattened, key=lambda entry: entry.start)


def _summarize_timeline(segments: Sequence[Segment]) -> dict:
    total_duration = round(sum(seg.duration() for seg in segments), 3)
    return {
        "segment_count": len(segments),
        "total_duration": total_duration,
        "average_confidence": round(
            sum(seg.confidence for seg in segments) / len(segments), 3
        )
        if segments
        else 0.0,
    }


def _summarize_tracks(
    selections: Sequence[TrackSelection] | None, segments: Sequence[Segment]
) -> List[dict]:
    grouped: dict[str, List[Segment]] = {}
    for seg in segments:
        key = seg.track_id or seg.primary_track_id or (
            "merged" if seg.contrib_track_ids else "unknown"
        )
        grouped.setdefault(key, []).append(seg)
    selection_lookup = {selection.track_id: selection for selection in selections or []}
    summaries = []
    seen: set[str] = set()
    for track_id, segs in grouped.items():
        selection = selection_lookup.get(track_id)
        summaries.append(
            {
                "track_id": track_id,
                "label": selection.label if selection else None,
                "avg_similarity": round(selection.avg_similarity, 3) if selection else None,
                "p90_similarity": round(selection.p90_similarity, 3) if selection else None,
                "max_similarity": round(selection.max_similarity, 3) if selection else None,
                "score": round(selection.score, 3) if selection else None,
                "best_ref_id": selection.best_ref_id if selection else None,
                "best_ref_sim": round(selection.best_ref_sim, 3) if selection and selection.best_ref_sim is not None else None,
                "best_ref_p90": round(selection.best_ref_p90, 3) if selection and selection.best_ref_p90 is not None else None,
                "ref_topk_avg": round(selection.ref_topk_avg, 3) if selection and selection.ref_topk_avg is not None else None,
                "ref_hits": selection.ref_hits if selection else None,
                "total_duration": round(sum(seg.duration() for seg in segs), 3),
                "segment_count": len(segs),
                "detection_count": len(selection.entries) if selection else None,
            }
        )
        if selection:
            seen.add(track_id)
    for selection in selections or []:
        if selection.track_id in seen:
            continue
        summaries.append(
            {
                "track_id": selection.track_id,
                "label": selection.label,
                "avg_similarity": round(selection.avg_similarity, 3),
                "p90_similarity": round(selection.p90_similarity, 3),
                "max_similarity": round(selection.max_similarity, 3),
                "score": round(selection.score, 3),
                "best_ref_id": selection.best_ref_id,
                "best_ref_sim": round(selection.best_ref_sim, 3) if selection.best_ref_sim is not None else None,
                "best_ref_p90": round(selection.best_ref_p90, 3) if selection.best_ref_p90 is not None else None,
                "ref_topk_avg": round(selection.ref_topk_avg, 3) if selection.ref_topk_avg is not None else None,
                "ref_hits": selection.ref_hits,
                "total_duration": round(selection.total_duration, 3),
                "segment_count": 0,
                "detection_count": len(selection.entries),
            }
        )
    return summaries
@dataclass
class _MatchStats:
    avg: float | None = None
    max_score: float | None = None
    p90: float | None = None
    primary_track_id: str | None = None
    best_ref_id: str | None = None
    best_ref_sim: float | None = None
    best_ref_p90: float | None = None
    ref_topk_avg: float | None = None
    ref_hits: list[dict] | None = None


@dataclass
class _RefSummary:
    best_ref_id: str | None = None
    best_ref_sim: float | None = None
    best_ref_p90: float | None = None
    ref_topk_avg: float | None = None
    ref_hits: list[dict] | None = None


def _summarize_best_refs(
    best_ref_ids: Sequence[str],
    best_ref_sims: Sequence[float],
    topk_avgs: Sequence[float],
) -> _RefSummary:
    if not best_ref_ids or not best_ref_sims:
        return _RefSummary()
    counts: dict[str, int] = {}
    for ref_id in best_ref_ids:
        counts[ref_id] = counts.get(ref_id, 0) + 1
    best_ref_id = max(counts.items(), key=lambda item: item[1])[0]
    ref_hits = [
        {"ref_id": ref_id, "hits": count}
        for ref_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    best_ref_sim = max(best_ref_sims)
    best_ref_p90 = _percentile(best_ref_sims, 0.9)
    ref_topk_avg = sum(topk_avgs) / len(topk_avgs) if topk_avgs else None
    return _RefSummary(
        best_ref_id=best_ref_id,
        best_ref_sim=best_ref_sim,
        best_ref_p90=best_ref_p90,
        ref_topk_avg=ref_topk_avg,
        ref_hits=ref_hits,
    )
