#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-detector}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup_detector] Using venv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements-detector.txt"

echo "[setup_detector] Checking ffmpeg availability..."
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1
else
  echo "[setup_detector] WARNING: ffmpeg not found in PATH."
fi

echo "[setup_detector] Validating Python dependencies..."
python - <<'PY'
import cv2
import numpy
print("cv2 version:", cv2.__version__)
print("numpy version:", numpy.__version__)
PY

echo "[setup_detector] Completed."
