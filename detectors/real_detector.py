#!/usr/bin/env python3
"""Lightweight face detector + embedding generator for PickPresence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pickpresence.embeddings import compute_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PickPresence detector v0")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output", required=True, help="Path to write detections JSON.")
    parser.add_argument(
        "--reference",
        help="Optional reference embedding JSON. Accepted but unused in v0.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sampling FPS for frame extraction.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold to keep an existing track.",
    )
    parser.add_argument(
        "--max-track-gap",
        type=float,
        default=2.0,
        help="Maximum seconds between sightings to keep reusing the same track.",
    )
    return parser.parse_args()


@dataclass
class Track:
    track_id: str
    embedding: np.ndarray
    last_time: float


def main() -> None:
    args = parse_args()
    detections = process_video(
        video_path=Path(args.video),
        sample_fps=args.sample_fps,
        similarity_threshold=args.similarity_threshold,
        max_track_gap=args.max_track_gap,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(detections, indent=2), encoding="utf-8")


def process_video(
    video_path: Path,
    sample_fps: float,
    similarity_threshold: float,
    max_track_gap: float,
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    if video_fps <= 0:
        video_fps = sample_fps
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    detections: List[dict] = []
    tracks: List[Track] = []
    frame_idx = 0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    frame_area = width * height
    next_track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]
            embedding = compute_embedding(face_roi)
            track, next_track_id = _assign_track(
                embedding=embedding,
                timestamp=timestamp,
                tracks=tracks,
                next_track_id=next_track_id,
                similarity_threshold=similarity_threshold,
                max_track_gap=max_track_gap,
            )

            duration = 1.0 / sample_fps
            score = float(min(0.99, max(0.3, (w * h) / frame_area)))
            detections.append(
                {
                    "start": round(timestamp, 3),
                    "end": round(timestamp + duration, 3),
                    "embedding": embedding.round(6).tolist(),
                    "track_id": track.track_id,
                    "sources": ["face-det"],
                    "score": round(score, 3),
                    "label": None,
                }
            )

        frame_idx += 1

    cap.release()
    return detections


def _assign_track(
    embedding: np.ndarray,
    timestamp: float,
    tracks: List[Track],
    next_track_id: int,
    similarity_threshold: float,
    max_track_gap: float,
) -> tuple[Track, int]:
    best_track: Optional[Track] = None
    best_score = -1.0
    for track in tracks:
        if timestamp - track.last_time > max_track_gap:
            continue
        score = float(np.dot(track.embedding, embedding))
        if score > best_score:
            best_score = score
            best_track = track

    if best_track and best_score >= similarity_threshold:
        best_track.embedding = embedding
        best_track.last_time = timestamp
        return best_track, next_track_id

    new_track_id = str(next_track_id)
    next_track_id += 1
    new_track = Track(track_id=new_track_id, embedding=embedding, last_time=timestamp)
    tracks.append(new_track)
    return new_track, next_track_id


if __name__ == "__main__":
    main()
