#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_PATH="${VIDEO_PATH:-/home/zhibai/videos/S01E09_180.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/out/run_180}"
REF_IMAGES_DIR="${REF_IMAGES_DIR:-$ROOT_DIR/out/refs_images}"
REF_OUT_DIR="${REF_OUT_DIR:-$ROOT_DIR/out/refs}"
INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-$ROOT_DIR/out/insightface}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-detector/bin/python}"
TRACK_FILL_MAX_DURATION="${TRACK_FILL_MAX_DURATION:-0}"
TRACK_FILL_MAX_CHAIN="${TRACK_FILL_MAX_CHAIN:-0}"
TRACK_FILL_GAP="${TRACK_FILL_GAP:-4.0}"
TRACK_FILL_MIN_SIM="${TRACK_FILL_MIN_SIM:-0.20}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
FACE_CONFIRM_THRESHOLD="${FACE_CONFIRM_THRESHOLD:-0.0}"
FACE_CONFIRM_WINDOW="${FACE_CONFIRM_WINDOW:-0.0}"

mkdir -p "$OUTPUT_DIR"

if [[ -d "$REF_IMAGES_DIR" ]]; then
  bash "$ROOT_DIR/scripts/build_reference_set.sh" "$REF_IMAGES_DIR" "$REF_OUT_DIR" "Target"
else
  echo "[run_m15_style_180] WARN: $REF_IMAGES_DIR not found; skipping reference rebuild."
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/detectors/insightface_detector.py" \
  --video "$VIDEO_PATH" \
  --output "$OUTPUT_DIR/detections.json" \
  --sample-fps 7.5 \
  --det-threshold 0.35 \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --model-root "$INSIGHTFACE_ROOT"

extra_args=()
if [[ "$TRACK_FILL_MAX_DURATION" != "0" && "$TRACK_FILL_MAX_DURATION" != "0.0" ]]; then
  extra_args+=(--track-fill-max-duration "$TRACK_FILL_MAX_DURATION")
fi
if [[ "$TRACK_FILL_MAX_CHAIN" != "0" ]]; then
  extra_args+=(--track-fill-max-chain "$TRACK_FILL_MAX_CHAIN")
fi
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  extra_args+=(--clean-output)
fi
if [[ "$FACE_CONFIRM_THRESHOLD" != "0" && "$FACE_CONFIRM_THRESHOLD" != "0.0" ]]; then
  extra_args+=(--face-confirm-threshold "$FACE_CONFIRM_THRESHOLD")
fi
if [[ "$FACE_CONFIRM_WINDOW" != "0" && "$FACE_CONFIRM_WINDOW" != "0.0" ]]; then
  extra_args+=(--face-confirm-window "$FACE_CONFIRM_WINDOW")
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -m pickpresence.cli \
  --video "$VIDEO_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --reference-dir "$REF_OUT_DIR" \
  --reference-agg topk_avg --reference-topk 2 \
  --detection-log "$OUTPUT_DIR/detections.json" \
  --segment-policy track_first \
  --track-policy all --min-track-similarity 0.0 --min-track-duration 0.0 \
  --merge-policy none --trim-policy head_tail --trim-source video \
  --trim-scan-window 0.6 --trim-scan-step 0.04 --trim-device cuda \
  --bridge-gap 1.4 --min-duration 1.0 --export-end-eps 0.2 \
  --match-threshold 0.75 \
  --side-threshold-start 0.50 --side-threshold-keep 0.40 --side-scale-min 0.001 \
  --small-face-max-scale 0.002 --medium-face-max-scale 0.01 --low-light-score 0.5 \
  --side-bridge-gap 3.0 --side-profile-ratio-min 0.5 \
  --side-fill-gap 8.0 --side-fill-ratio-min 0.4 \
  --track-fill-gap "$TRACK_FILL_GAP" --track-fill-min-similarity "$TRACK_FILL_MIN_SIM" \
  --person-fallback --person-threshold 0.6 \
  "${extra_args[@]}"

echo "[run_m15_style_180] Done. Timeline: $OUTPUT_DIR/timeline.json"
