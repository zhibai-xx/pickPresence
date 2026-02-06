#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-$ROOT_DIR/out/refs_images}"
OUTPUT_DIR="${2:-$ROOT_DIR/out/refs}"
NAME="${3:-Target}"

export PICKPRESENCE_INSIGHTFACE_ROOT="$ROOT_DIR/out/insightface"
export PYTHONPATH="$ROOT_DIR"

"$ROOT_DIR/.venv-detector/bin/python" "$ROOT_DIR/scripts/make_reference_set.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --name "$NAME" \
  --backend insightface \
  --device auto

echo "[build_reference_set] Done. Output: $OUTPUT_DIR"
