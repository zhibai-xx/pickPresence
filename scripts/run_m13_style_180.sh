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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

DETECTIONS_PATH="$OUT_DIR/detections.json"

mkdir -p "$OUT_DIR"

"$ROOT_DIR/.venv-detector/bin/python" "$ROOT_DIR/detectors/insightface_detector.py" \
  --video "$VIDEO_PATH" \
  --output "$DETECTIONS_PATH" \
  --sample-fps 7.5 \
  --det-threshold 0.35

"$ROOT_DIR/.venv-detector/bin/python" -m pickpresence.cli \
  --video "$VIDEO_PATH" \
  --output-dir "$OUT_DIR" \
  --reference-dir "$ROOT_DIR/out/refs" \
  --reference-agg topk_avg \
  --reference-topk 2 \
  --detection-log "$DETECTIONS_PATH" \
  --segment-policy track_first \
  --track-policy all \
  --min-track-similarity 0.0 \
  --merge-policy none \
  --trim-policy head_tail \
  --trim-source video \
  --trim-scan-window 0.6 \
  --trim-scan-step 0.04 \
  --trim-device cuda \
  --bridge-gap 1.4 \
  --match-threshold 0.75 \
  --min-duration 1.0 \
  --export-end-eps 0.2 \
  --side-threshold-start 0.50 \
  --side-threshold-keep 0.40 \
  --side-scale-min 0.001 \
  --small-face-max-scale 0.002 \
  --medium-face-max-scale 0.01 \
  --low-light-score 0.5 \
  --side-bridge-gap 3.0 \
  --side-profile-ratio-min 0.5 \
  --side-fill-gap 8.0 \
  --side-fill-ratio-min 0.4 \
  --track-fill-gap 8.0 \
  --track-fill-min-similarity 0.20 \
  --small-face-ratio-max 0.0 \
  --small-face-max-match 0.0 \
  --small-face-min-side-ratio 0.0

echo "[run_m13_style_180] Done. Timeline: $OUT_DIR/timeline.json"
