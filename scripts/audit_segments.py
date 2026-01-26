#!/usr/bin/env python3
"""Audit exported clips by sampling last frames and nearby original frames."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from pickpresence.identity import load_reference_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PickPresence clip boundaries.")
    parser.add_argument("--timeline", required=True, help="Path to timeline.json.")
    parser.add_argument("--video", required=True, help="Original video path.")
    parser.add_argument("--reference", required=True, help="Reference embedding JSON.")
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Directory containing clip_XXX.mp4 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store audit images (e.g., out_lab/audit_images).",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.3,
        help="Similarity threshold marking PASS for clip_last frame.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary path (defaults to ffmpeg in PATH).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    reference = load_reference_embedding(args.reference)
    analyzer = FrameAnalyzer(model_name="buffalo_l")
    analyzer.prepare(reference)

    clips_dir = Path(args.clips_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    video_duration = _probe_duration(args.video)

    segments = timeline.get("segments", [])
    offsets = [(-0.20, "m0p20"), (-0.08, "m0p08"), (-0.001, "m0p001"), (0.08, "p0p08")]

    for idx, seg in enumerate(segments):
        seg_dir = out_root / f"seg_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        export_start = seg.get("export_start", seg["start"])
        export_end = seg.get("export_end", seg["end"])
        clip_path = clips_dir / f"clip_{idx:03d}.mp4"
        if not clip_path.exists():
            print(f"[audit] Segment {idx:03d}: clip not found at {clip_path}", file=sys.stderr)
            continue
        duration = max(0.0, export_end - export_start)
        clip_last_time = max(0.0, duration - 0.2)
        clip_last_path = seg_dir / "clip_last.png"
        clip_ok, clip_err = _extract_frame_ffmpeg(
            binary=args.ffmpeg_bin,
            video=clip_path,
            timestamp=clip_last_time,
            output=clip_last_path,
            stderr_path=seg_dir / "clip_last.stderr",
        )
        clip_last_sim = _safe_similarity(analyzer, clip_last_path, clip_ok, clip_err, "clip_last")
        status = "PASS" if clip_last_sim is not None and clip_last_sim >= args.keep_threshold else "FAIL"
        print(
            f"[audit] Segment {idx:03d} export_end={export_end:.3f}s "
            f"clip_last_sim={clip_last_sim if clip_last_sim is not None else 'NA'} {status}"
        )
        if status == "FAIL":
            print(
                "         Suggest increasing --export-end-eps or --trim-scan-window.",
            )

        for delta, label in offsets:
            ts = max(0.0, min(video_duration, export_end + delta))
            out_path = seg_dir / f"orig_end_{label}.png"
            ok, err = _extract_frame_ffmpeg(
                binary=args.ffmpeg_bin,
                video=Path(args.video),
                timestamp=ts,
                output=out_path,
                stderr_path=out_path.with_suffix(".stderr"),
            )
            sim = _safe_similarity(analyzer, out_path, ok, err, f"orig_end_{label}")
            print(
                f"         frame {label} @ {ts:.3f}s similarity={sim if sim is not None else 'NA'}"
            )

    return 0


def _extract_frame_ffmpeg(
    binary: str,
    video: Path,
    timestamp: float,
    output: Path,
    retries: int = 1,
    retry_delay_s: float = 0.15,
    stderr_path: Path | None = None,
) -> tuple[bool, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = retries + 1
    last_err = ""
    for _ in range(attempts):
        if output.exists():
            output.unlink()
        cmd = [
            binary,
            "-y",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(video),
            "-vframes",
            "1",
            "-update",
            "1",
            str(output),
        ]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        last_err = result.stderr.decode(errors="replace")
        if stderr_path is not None:
            stderr_path.write_text(last_err, encoding="utf-8")
        if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
            return True, last_err
        time.sleep(retry_delay_s)
    return False, last_err


def _probe_duration(video: str | Path) -> float:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Audit script requires OpenCV (cv2).") from exc
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps > 0 and frame_count > 0:
        return float(frame_count / fps)
    return 0.0


class FrameAnalyzer:
    """Detect faces in still frames and compute similarity to the reference embedding."""

    def __init__(self, model_name: str = "buffalo_l") -> None:
        self.model_name = model_name
        self.app = None
        self.cv2 = None
        self.reference_vec: np.ndarray | None = None

    def prepare(self, reference) -> None:
        from detectors.insightface_detector import _load_insightface_app

        if reference.vector is None:
            raise RuntimeError("Reference embedding is required for audit.")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Audit script requires OpenCV (cv2).") from exc
        self.cv2 = cv2
        self.app = _load_insightface_app(self.model_name, device="cpu")
        self.reference_vec = np.asarray(reference.vector, dtype=np.float32)

    def similarity(self, image_path: Path) -> float | None:
        if self.cv2 is None or self.app is None or self.reference_vec is None:
            raise RuntimeError("FrameAnalyzer not prepared.")
        frame = self.cv2.imread(str(image_path))
        if frame is None:
            return None
        faces = self.app.get(frame)
        if not faces:
            return None
        sims = [float(np.dot(face.normed_embedding, self.reference_vec)) for face in faces]
        if not sims:
            return None
        return max(sims)


def _safe_similarity(
    analyzer: "FrameAnalyzer",
    image_path: Path,
    extracted: bool,
    stderr: str,
    label: str,
) -> float | None:
    if not extracted:
        if stderr:
            stderr = stderr.strip().replace("\n", " ")
            stderr = (stderr[:200] + "...") if len(stderr) > 200 else stderr
        print(
            f"[audit] {label}: extract_failed path={image_path} "
            f"stderr={stderr if stderr else 'NA'}",
            file=sys.stderr,
        )
        return None
    if not image_path.exists() or image_path.stat().st_size == 0:
        print(f"[audit] {label}: extract_failed path={image_path} size=0", file=sys.stderr)
        return None
    return analyzer.similarity(image_path)


if __name__ == "__main__":
    raise SystemExit(main())
