#!/usr/bin/env python3
"""Reference embedding generator for PickPresence."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from pickpresence.embeddings import compute_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reference embedding JSON.")
    parser.add_argument("--name", required=True, help="Target name for the embedding.")
    parser.add_argument("--output", required=True, help="Path to write reference JSON.")
    parser.add_argument("--image", help="Path to a reference image.")
    parser.add_argument("--video", help="Path to a reference video (optional).")
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Timestamp (seconds) to grab frame from video when --video is used.",
    )
    parser.add_argument(
        "--assume-face",
        dest="assume_face",
        action="store_true",
        help="Skip face detection and treat the whole frame as the face ROI (toy backend).",
    )
    parser.add_argument(
        "--backend",
        choices=["toy", "insightface"],
        default="toy",
        help="Embedding backend. Use 'insightface' to match the detector embeddings.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device for the insightface backend.",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model name when using --backend insightface.",
    )
    parser.add_argument(
        "--model-root",
        help="InsightFace model cache root (defaults to ~/.insightface or PICKPRESENCE_INSIGHTFACE_ROOT).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = None
    if args.image:
        frame = _load_image(args.image)
        if frame is None:
            print(f"[make_reference] Failed to read image: {args.image}")
            return 1
    elif args.video:
        frame = _grab_frame(args.video, args.time)
        if frame is None:
            print(f"[make_reference] Failed to read frame at {args.time}s from {args.video}")
            return 1
    else:
        print("[make_reference] Provide --image or --video.")
        return 1

    try:
        embedding = _compute_embedding(frame, args)
    except RuntimeError as exc:
        print(f"[make_reference] {exc}")
        return 1

    payload = {"name": args.name, "embedding": embedding}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[make_reference] Wrote reference -> {args.output}")
    return 0


def _grab_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _extract_face(frame: np.ndarray) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return frame[y : y + h, x : x + w]


def _load_image(path: str) -> Optional[np.ndarray]:
    if cv2 is not None:
        image = cv2.imread(path)
        if image is not None:
            return image
    return _load_ppm(path)


def _load_ppm(path: str) -> Optional[np.ndarray]:
    """Minimal P6 PPM reader used as a cv2 fallback for tests."""

    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P6":
            return None

        def _read_non_comment():
            line = fh.readline()
            while line.startswith(b"#") or not line.strip():
                line = fh.readline()
            return line

        dims = _read_non_comment()
        width, height = map(int, dims.split())
        maxval = int(_read_non_comment())
        if maxval > 255:
            return None
        data = fh.read(width * height * 3)
        if len(data) != width * height * 3:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr.reshape((height, width, 3))


def _compute_embedding(frame: np.ndarray, args: argparse.Namespace) -> list[float]:
    if args.backend == "toy":
        roi = frame if args.assume_face else _extract_face(frame)
        if roi is None or roi.size == 0:
            raise RuntimeError("Could not locate a face ROI. Provide --assume-face or use insightface.")
        return compute_embedding(roi).round(6).tolist()

    embedding = _insightface_embedding(frame, args)
    return embedding.round(6).tolist()


def _insightface_embedding(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("InsightFace is not installed. Run scripts/setup_detector.sh first.") from exc

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0
    if args.device == "cpu":
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    elif args.device == "auto":
        ctx_id = 0
    elif args.device == "cuda":
        ctx_id = 0

    model_root = args.model_root or os.environ.get("PICKPRESENCE_INSIGHTFACE_ROOT")
    if model_root:
        app = FaceAnalysis(name=args.model_name, providers=providers, root=model_root)
    else:
        app = FaceAnalysis(name=args.model_name, providers=providers)
    app.prepare(ctx_id=ctx_id)
    faces = app.get(frame)
    if not faces:
        raise RuntimeError("No face detected with insightface. Try another frame.")
    best = max(faces, key=lambda face: float(face.det_score))
    vec = best.embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


if __name__ == "__main__":
    raise SystemExit(main())
