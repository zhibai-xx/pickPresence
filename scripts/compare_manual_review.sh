#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUAL_PATH="${1:-$ROOT_DIR/docs/acceptance/S01E09_180_manual_review.json}"
TIMELINE_PATH="${2:-$ROOT_DIR/out/run_180/timeline.json}"
OUTPUT_JSON="${3:-$ROOT_DIR/out/run_180/manual_compare.json}"
OUTPUT_MD="${4:-$ROOT_DIR/out/run_180/manual_compare.md}"

python3 "$ROOT_DIR/scripts/compare_manual_review.py" \
  --manual "$MANUAL_PATH" \
  --timeline "$TIMELINE_PATH" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
