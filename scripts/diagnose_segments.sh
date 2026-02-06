#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTIONS_PATH="${1:-$ROOT_DIR/out/run_180/detections.json}"
SEGMENTS_JSON="${2:-$ROOT_DIR/docs/acceptance/S01E09_180_problem_segments.json}"
OUT_PATH="${3:-$ROOT_DIR/out/inspect_segments/diagnostics.json}"
REF_DIR="${4:-$ROOT_DIR/out/refs}"

export PYTHONPATH="$ROOT_DIR"

"$ROOT_DIR/.venv-detector/bin/python" "$ROOT_DIR/scripts/diagnose_segments.py" \
  --detection-log "$DETECTIONS_PATH" \
  --segments-json "$SEGMENTS_JSON" \
  --output "$OUT_PATH" \
  --reference-dir "$REF_DIR"

echo "[diagnose_segments] Done. Output: $OUT_PATH"
