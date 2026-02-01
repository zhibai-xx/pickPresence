#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_PATH="${1:-/home/zhibai/videos/S01E09_180.mp4}"
OUT_DIR="${2:-$ROOT_DIR/out/run_180}"

export INSIGHTFACE_HOME="$ROOT_DIR/out/insightface"
export MPLCONFIGDIR="$ROOT_DIR/out/matplotlib"
export PICKPRESENCE_INSIGHTFACE_ROOT="$ROOT_DIR/out/insightface"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT_DIR"

DETECTIONS_PATH="$OUT_DIR/detections.json"

mkdir -p "$OUT_DIR"

"$ROOT_DIR/.venv-detector/bin/python" "$ROOT_DIR/detectors/insightface_detector.py" \
  --video "$VIDEO_PATH" \
  --output "$DETECTIONS_PATH" \
  --sample-fps 5.0

"$ROOT_DIR/.venv-detector/bin/python" -m pickpresence.cli \
  --video "$VIDEO_PATH" \
  --output-dir "$OUT_DIR" \
  --reference-dir "$ROOT_DIR/out/refs" \
  --reference-agg topk_avg \
  --reference-topk 2 \
  --detection-log "$DETECTIONS_PATH" \
  --track-policy all \
  --min-track-similarity 0.5 \
  --merge-policy union \
  --trim-policy head_tail \
  --trim-source video \
  --trim-scan-window 0.6 \
  --trim-scan-step 0.04 \
  --bridge-gap 2.0 \
  --export-end-eps 0.2

echo "[run_m10_style_180] Done. Timeline: $OUT_DIR/timeline.json"
