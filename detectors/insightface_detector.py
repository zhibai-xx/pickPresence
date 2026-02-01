#!/usr/bin/env python3
"""InsightFace-based detector + tracker for PickPresence."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV (cv2) is required for the insightface detector.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsightFace detector for PickPresence.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", required=True, help="Path to write detections JSON.")
    parser.add_argument("--reference", help="Reference embedding JSON for similarity dumps.")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Frame sampling rate (frames per second).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device for InsightFace (cuda uses GPU if available).",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model pack name (see insightface model zoo).",
    )
    parser.add_argument(
        "--providers",
        help="Comma-separated provider list (overrides --device), e.g. CUDAExecutionProvider,CPUExecutionProvider.",
    )
    parser.add_argument(
        "--model-root",
        help="InsightFace model cache root (defaults to ~/.insightface or PICKPRESENCE_INSIGHTFACE_ROOT).",
    )
    parser.add_argument(
        "--embedding-gate",
        type=float,
        default=0.65,
        help="Minimum cosine similarity to treat a detection as the same track.",
    )
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for marking detections as the reference label.",
    )
    parser.add_argument(
        "--iou-gate",
        type=float,
        default=0.35,
        help="Minimum IoU to reuse the same track.",
    )
    parser.add_argument(
        "--max-track-gap",
        type=float,
        default=1.5,
        help="Maximum seconds between observations to keep a track alive.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process at most N sampled frames (useful for quick tests).",
    )
    parser.add_argument(
        "--dump-frames",
        type=int,
        default=0,
        help="Dump the first N sampled frames with overlays for debugging.",
    )
    parser.add_argument(
        "--dump-dir",
        default="detector_dumps",
        help="Directory to store debug frames when --dump-frames > 0.",
    )
    parser.add_argument(
        "--mock-detections",
        help="Path to an existing detections JSON to copy (for tests).",
    )
    return parser.parse_args()


@dataclass
class Track:
    track_id: str
    bbox: np.ndarray
    embedding: np.ndarray
    last_time: float
    last_frame: int


def main() -> int:
    args = parse_args()

    if args.mock_detections:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(Path(args.mock_detections).read_text(), encoding="utf-8")
        return 0

    providers_override = args.providers or os.environ.get("PICKPRESENCE_PROVIDER_ORDER")
    model_root = args.model_root or os.environ.get("PICKPRESENCE_INSIGHTFACE_ROOT")
    app = _load_insightface_app(args.model_name, args.device, providers_override, model_root)

    reference_name, reference_vec = _load_reference(args.reference)

    detections = run_detector(
        video_path=Path(args.video),
        sample_fps=args.sample_fps,
        app=app,
        embedding_gate=args.embedding_gate,
        iou_gate=args.iou_gate,
        max_track_gap=args.max_track_gap,
        max_frames=args.max_frames,
        reference=(reference_name, reference_vec),
        reference_threshold=args.reference_threshold,
        dump_frames=args.dump_frames,
        dump_dir=Path(args.dump_dir) if args.dump_frames else None,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return 0


def run_detector(
    video_path: Path,
    sample_fps: float,
    app,
    embedding_gate: float,
    iou_gate: float,
    max_track_gap: float,
    max_frames: Optional[int],
    reference: tuple[Optional[str], Optional[np.ndarray]],
    reference_threshold: float,
    dump_frames: int,
    dump_dir: Optional[Path],
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    if video_fps <= 0:
        video_fps = sample_fps
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    tracker = _Tracker(embedding_gate, iou_gate, max_track_gap)
    dump_remaining = dump_frames

    detections: List[dict] = []
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps
        faces = app.get(frame)

        assignments = tracker.assign(
            faces=faces,
            timestamp=timestamp,
            frame_index=frame_idx,
        )

        for assign in assignments:
            det = assign.detection
            embedding = assign.embedding
            start = timestamp
            end = timestamp + (1.0 / sample_fps)
            similarity = None
            label = None
            ref_name, ref_vec = reference
            if ref_vec is not None:
                similarity = cosine_similarity(embedding, ref_vec)
                if similarity >= reference_threshold and ref_name:
                    label = ref_name
            entry = {
                "start": round(start, 3),
                "end": round(end, 3),
                "embedding": embedding.round(6).tolist(),
                "track_id": assign.track_id,
                "sources": ["insightface"],
                "score": round(float(det.det_score), 4),
                "label": label,
                "bbox": det.bbox.astype(float).round(2).tolist(),
                "frame_index": frame_idx,
            }
            detections.append(entry)

        if dump_remaining > 0 and dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            _dump_frame(
                frame=frame,
                frame_index=frame_idx,
                assignments=assignments,
                dump_dir=dump_dir,
                reference=reference,
            )
            dump_remaining -= 1

        sampled += 1
        if max_frames is not None and sampled >= max_frames:
            break
        frame_idx += 1

    cap.release()
    return detections


class _DetectionAssignment:
    def __init__(
        self,
        track_id: str,
        detection,
        embedding: np.ndarray,
        score: float,
    ) -> None:
        self.track_id = track_id
        self.detection = detection
        self.embedding = embedding
        self.score = score


class _Tracker:
    def __init__(self, emb_gate: float, iou_gate: float, max_gap: float) -> None:
        self.emb_gate = emb_gate
        self.iou_gate = iou_gate
        self.max_gap = max_gap
        self.tracks: List[Track] = []
        self.next_id = 0

    def assign(
        self,
        faces,
        timestamp: float,
        frame_index: int,
    ) -> List[_DetectionAssignment]:
        assignments: List[_DetectionAssignment] = []
        for det in faces:
            bbox = det.bbox.astype(float)
            embedding = det.embedding.astype(np.float32)
            track = self._match_track(bbox, embedding, timestamp)
            if track is None:
                track = self._create_track(bbox, embedding, timestamp, frame_index)
            else:
                track.bbox = bbox
                track.embedding = 0.6 * track.embedding + 0.4 * embedding
                track.embedding /= np.linalg.norm(track.embedding) + 1e-6
                track.last_time = timestamp
                track.last_frame = frame_index
            assignments.append(
                _DetectionAssignment(
                    track_id=track.track_id,
                    detection=det,
                    embedding=embedding,
                    score=float(det.det_score),
                )
            )
        self._prune(timestamp)
        return assignments

    def _match_track(
        self, bbox: np.ndarray, embedding: np.ndarray, timestamp: float
    ) -> Optional[Track]:
        best_track = None
        best_score = -math.inf
        for track in self.tracks:
            if timestamp - track.last_time > self.max_gap:
                continue
            iou = bbox_iou(track.bbox, bbox)
            if iou < self.iou_gate:
                continue
            emb_sim = cosine_similarity(track.embedding, embedding)
            if emb_sim < self.emb_gate:
                continue
            combined = 0.6 * emb_sim + 0.4 * iou
            if combined > best_score:
                best_score = combined
                best_track = track
        return best_track

    def _create_track(
        self,
        bbox: np.ndarray,
        embedding: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> Track:
        track = Track(
            track_id=str(self.next_id),
            bbox=bbox,
            embedding=embedding / (np.linalg.norm(embedding) + 1e-6),
            last_time=timestamp,
            last_frame=frame_index,
        )
        self.tracks.append(track)
        self.next_id += 1
        return track

    def _prune(self, timestamp: float) -> None:
        self.tracks = [t for t in self.tracks if timestamp - t.last_time <= self.max_gap]


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _load_insightface_app(
    model_name: str,
    device: str,
    providers_override: Optional[str] = None,
    model_root: Optional[str] = None,
):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "InsightFace is not installed. Run scripts/setup_detector.sh to install dependencies."
        ) from exc

    available = None
    try:  # pragma: no cover - optional dependency
        import onnxruntime as ort  # type: ignore

        available = ort.get_available_providers()
    except Exception:
        available = None

    from pickpresence.provider_utils import resolve_providers

    providers, missing, desired = resolve_providers(device, providers_override, available)
    if "CUDAExecutionProvider" in missing:
        print(
            "[detector] CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider.",
            flush=True,
        )
    if available is not None:
        print(f"[detector] providers desired={desired} available={available} using={providers}", flush=True)
    else:
        print(f"[detector] providers desired={desired} using={providers}", flush=True)

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    if model_root:
        app = FaceAnalysis(name=model_name, providers=providers, root=model_root)
    else:
        app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=ctx_id)
    return app


def _load_reference(path: Optional[str]) -> tuple[Optional[str], Optional[np.ndarray]]:
    if not path:
        return None, None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = data.get("name")
    vec = np.asarray(data["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return name, vec
    return name, vec / norm


def _dump_frame(
    frame: np.ndarray,
    frame_index: int,
    assignments: Sequence[_DetectionAssignment],
    dump_dir: Path,
    reference: tuple[Optional[str], Optional[np.ndarray]],
) -> None:
    overlay = frame.copy()
    ref_name, ref_vec = reference
    for assign in assignments:
        det = assign.detection
        bbox = det.bbox.astype(int)
        cv2.rectangle(
            overlay,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )
        text = f"ID:{assign.track_id}"
        if ref_vec is not None:
            sim = cosine_similarity(assign.embedding, ref_vec)
            text += f" sim:{sim:.2f}"
            if ref_name:
                text += f" ref:{ref_name}"
        cv2.putText(
            overlay,
            text,
            (bbox[0], max(0, bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    out_path = dump_dir / f"frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    raise SystemExit(main())
