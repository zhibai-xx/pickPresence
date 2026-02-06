#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_PATH="${1:-/home/zhibai/videos/S01E09_180.mp4}"
OUT_DIR="${2:-$ROOT_DIR/out/inspect_segments}"
SEGMENTS_JSON="${3:-$ROOT_DIR/docs/acceptance/S01E09_180_problem_segments.json}"
REF_DIR="${4:-$ROOT_DIR/out/refs}"

export INSIGHTFACE_HOME="$ROOT_DIR/out/insightface"
export MPLCONFIGDIR="$ROOT_DIR/out/matplotlib"
export PICKPRESENCE_INSIGHTFACE_ROOT="$ROOT_DIR/out/insightface"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT_DIR"

mkdir -p "$OUT_DIR"

"$ROOT_DIR/.venv-detector/bin/python" "$ROOT_DIR/scripts/inspect_segments.py" \
  --video "$VIDEO_PATH" \
  --output-dir "$OUT_DIR" \
  --segments-json "$SEGMENTS_JSON" \
  --reference-dir "$REF_DIR" \
  --sample-fps 5.0 \
  --det-threshold 0.35 \
  --device auto \
  --annotate \
  --annotate-only

echo "[inspect_segments] Done. Output: $OUT_DIR"
