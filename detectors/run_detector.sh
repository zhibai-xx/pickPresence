#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-detector"
PY_BIN="${VENV_DIR}/bin/python"
TARGET="${PICKPRESENCE_DETECTOR_IMPL:-$SCRIPT_DIR/insightface_detector.py}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "[run_detector] Detector venv python not found at $PY_BIN" >&2
  echo "Run scripts/setup_detector.sh to provision dependencies." >&2
  exit 1
fi

exec "$PY_BIN" "$TARGET" "$@"
