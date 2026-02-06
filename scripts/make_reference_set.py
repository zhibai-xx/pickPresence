#!/usr/bin/env python3
"""Batch reference embedding generator for PickPresence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - optional for real images
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

from pickpresence.embeddings import compute_embedding
from pickpresence.appearance import compute_appearance_vector
from pickpresence.reid import compute_person_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a set of reference embeddings from images.")
    parser.add_argument("--input-dir", required=True, help="Directory with reference images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write reference JSONs.")
    parser.add_argument("--name", default="Target", help="Target name for the reference set.")
    parser.add_argument(
        "--backend",
        choices=["toy", "insightface"],
        default="toy",
        help="Embedding backend. Use 'insightface' to match detector embeddings.",
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
    parser.add_argument(
        "--assume-face",
        action="store_true",
        help="Skip face detection and treat the whole frame as the face ROI (toy backend).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}])
    if not images:
        print(f"[make_reference_set] No images found in {input_dir}")
        return 1

    items = []
    for idx, image_path in enumerate(images):
        frame = _load_image(str(image_path))
        if frame is None:
            print(f"[make_reference_set] Skipping unreadable image: {image_path}")
            continue
        try:
            embedding, appearance, person_embedding = _compute_embedding(frame, args)
        except RuntimeError as exc:
            print(f"[make_reference_set] {image_path}: {exc}")
            continue
        payload = {
            "name": args.name,
            "ref_id": f"{args.name}_{idx:02d}",
            "source": str(image_path),
            "embedding": embedding,
            "appearance": appearance,
            "person_embedding": person_embedding,
        }
        out_path = output_dir / f"ref_{idx:02d}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        items.append({"path": str(out_path), "source": str(image_path), "ref_id": payload["ref_id"]})

    index = {
        "name": args.name,
        "count": len(items),
        "items": items,
    }
    index_path = output_dir / "reference_set.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[make_reference_set] Wrote {len(items)} references -> {output_dir}")
    print(f"[make_reference_set] Index -> {index_path}")
    return 0


def _load_image(path: str) -> Optional[Any]:
    if cv2 is not None:
        image = cv2.imread(path)
        if image is not None:
            return image
    return _load_ppm(path)


def _load_ppm(path: str) -> Optional[Any]:
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
        try:
            import numpy as np  # type: ignore
        except ImportError:
            # Build nested list fallback.
            pixels = []
            idx = 0
            for _ in range(height):
                row = []
                for _ in range(width):
                    row.append([data[idx], data[idx + 1], data[idx + 2]])
                    idx += 3
                pixels.append(row)
            return pixels
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr.reshape((height, width, 3))


def _compute_embedding(frame: Any, args: argparse.Namespace) -> tuple[list[float], list[float], list[float]]:
    if args.backend == "toy":
        roi = frame if args.assume_face else _extract_face(frame)
        if roi is None:
            raise RuntimeError("Could not locate a face ROI. Use --assume-face or insightface backend.")
        if hasattr(roi, "size") and getattr(roi, "size", 0) == 0:
            raise RuntimeError("Could not locate a face ROI. Use --assume-face or insightface backend.")
        if isinstance(roi, list) and len(roi) == 0:
            raise RuntimeError("Could not locate a face ROI. Use --assume-face or insightface backend.")
        return _compute_embedding_toy(roi), compute_appearance_vector(roi), compute_person_embedding(roi)

    embedding, roi = _insightface_embedding(frame, args)
    appearance = compute_appearance_vector(roi if roi is not None else frame)
    person_embedding = compute_person_embedding(roi if roi is not None else frame)
    return embedding.round(6).tolist(), appearance, person_embedding


def _extract_face(frame: Any) -> Optional[Any]:
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


def _insightface_embedding(frame: Any, args: argparse.Namespace):
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
    import numpy as np  # type: ignore
    vec = best.embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        embedding = vec
    else:
        embedding = vec / norm
    roi = None
    try:
        x1, y1, x2, y2 = best.bbox.astype(int).tolist()
        roi = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    except Exception:
        roi = None
    return embedding, roi


def _compute_embedding_toy(roi) -> list[float]:
    """Compute a small deterministic embedding without numpy."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        # Fallback: use simple RGB means and normalize.
        if roi is None:
            raise RuntimeError("Missing ROI for toy embedding.")
        # roi is likely a nested list of lists if cv2 is unavailable
        # Compute mean per channel manually.
        total = [0.0, 0.0, 0.0]
        count = 0.0
        for row in roi:
            for pixel in row:
                if len(pixel) >= 3:
                    total[0] += float(pixel[0])
                    total[1] += float(pixel[1])
                    total[2] += float(pixel[2])
                    count += 1.0
        if count == 0:
            raise RuntimeError("Empty ROI for toy embedding.")
        vec = [v / count for v in total]
        norm = (sum(v * v for v in vec) ** 0.5) or 1e-8
        return [round(v / norm, 6) for v in vec]

    emb = compute_embedding(roi)
    return emb.round(6).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
