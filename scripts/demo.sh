#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR"
fi

pre_ref="${PICKPRESENCE_REFERENCE_EMBEDDING-}"
pre_det="${PICKPRESENCE_DETECTOR_SCRIPT-}"
pre_out="${PICKPRESENCE_OUTPUT_DIR-}"
pre_args="${PICKPRESENCE_DETECTOR_ARGS-}"
pre_cli="${CLI_PYTHON-}"
pre_cli_args="${PICKPRESENCE_CLI_ARGS-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  env_map="$(
    python3 - <<'PY' "$ROOT_DIR/.env"
import sys, json
path = sys.argv[1]
data = {}
with open(path, "r", encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        data[key.strip()] = value
print(json.dumps(data))
PY
  )"
  for key in PICKPRESENCE_REFERENCE_EMBEDDING PICKPRESENCE_DETECTOR_SCRIPT PICKPRESENCE_OUTPUT_DIR PICKPRESENCE_DETECTOR_ARGS CLI_PYTHON PICKPRESENCE_CLI_ARGS
  do
    value="$(python3 - <<PY "$env_map" "$key"
import json, sys
data = json.loads(sys.argv[1])
key = sys.argv[2]
print(data.get(key, ""))
PY
)"
    if [[ -n "$value" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done
fi

PICKPRESENCE_REFERENCE_EMBEDDING="${pre_ref:-${PICKPRESENCE_REFERENCE_EMBEDDING:-}}"
PICKPRESENCE_DETECTOR_SCRIPT="${pre_det:-${PICKPRESENCE_DETECTOR_SCRIPT:-}}"
PICKPRESENCE_OUTPUT_DIR="${pre_out:-${PICKPRESENCE_OUTPUT_DIR:-}}"
PICKPRESENCE_DETECTOR_ARGS="${pre_args:-${PICKPRESENCE_DETECTOR_ARGS:-}}"
CLI_PYTHON="${pre_cli:-${CLI_PYTHON:-}}"
PICKPRESENCE_CLI_ARGS="${pre_cli_args:-${PICKPRESENCE_CLI_ARGS:-}}"
DETECTOR_VENV_DIR="${DETECTOR_VENV_DIR:-$ROOT_DIR/.venv-detector}"
DEFAULT_VENV_PY="$DETECTOR_VENV_DIR/bin/python"

if [[ -z "${CLI_PYTHON:-}" && -x "$DEFAULT_VENV_PY" ]]; then
  CLI_PYTHON="$DEFAULT_VENV_PY"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.mp4 [reference.json]"
  exit 1
fi

VIDEO_PATH="$1"
OUT_DIR="${PICKPRESENCE_OUTPUT_DIR:-$ROOT_DIR/out}"
DETECTIONS_PATH="$OUT_DIR/detections.json"
DETECTOR_SCRIPT="${PICKPRESENCE_DETECTOR_SCRIPT:-$ROOT_DIR/detectors/run_detector.sh}"
REFERENCE_ARG="${2:-}"
if [[ -z "${REFERENCE_ARG:-}" ]]; then
  REFERENCE_ARG="${PICKPRESENCE_REFERENCE_EMBEDDING:-}"
fi
REFERENCE_EMBED="$REFERENCE_ARG"
PYTHON_BIN="${CLI_PYTHON:-python3}"
DETECTOR_EXTRA=()
CLI_EXTRA=()
if [[ -n "${PICKPRESENCE_DETECTOR_ARGS:-}" ]]; then
  while IFS= read -r arg; do
    DETECTOR_EXTRA+=("$arg")
  done < <(python3 - <<'PY'
import os, shlex
value = os.environ.get("PICKPRESENCE_DETECTOR_ARGS", "")
for token in shlex.split(value):
    print(token)
PY
)
fi
if [[ -n "${PICKPRESENCE_CLI_ARGS:-}" ]]; then
  while IFS= read -r arg; do
    CLI_EXTRA+=("$arg")
  done < <(python3 - <<'PY'
import os, shlex
value = os.environ.get("PICKPRESENCE_CLI_ARGS", "")
for token in shlex.split(value):
    print(token)
PY
)
fi

if [[ "$*" == *"--chunk-seconds"* ]]; then
  chunked_mode=1
fi

chunked_mode=0
for arg in "${CLI_EXTRA[@]}"; do
  if [[ "$arg" == "--chunk-seconds" ]]; then
    chunked_mode=1
    break
  fi
done

if [[ ! -x "$DETECTOR_SCRIPT" ]]; then
  echo "Detector script not executable: $DETECTOR_SCRIPT"
  echo "Make sure to run: chmod +x $DETECTOR_SCRIPT"
  exit 1
fi

if [[ -z "$REFERENCE_EMBED" ]]; then
  echo "Set PICKPRESENCE_REFERENCE_EMBEDDING to a target embedding JSON before running the demo."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for clip export. Install ffmpeg and ensure it's available in PATH."
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ "$chunked_mode" -eq 0 ]]; then
  echo "[demo] Running detector..."
  detector_cmd=(
    "$DETECTOR_SCRIPT"
    --video "$VIDEO_PATH"
    --output "$DETECTIONS_PATH"
    --reference "$REFERENCE_EMBED"
  )
  if [[ ${#DETECTOR_EXTRA[@]} -gt 0 ]]; then
    detector_cmd+=("${DETECTOR_EXTRA[@]}")
  fi
  "${detector_cmd[@]}"
else
  echo "[demo] Chunked mode detected; skipping full detector run."
fi

echo "[demo] Running PickPresence CLI..."
cli_cmd=(
  "$PYTHON_BIN"
  -m
  pickpresence.cli
  --video "$VIDEO_PATH"
  --output-dir "$OUT_DIR"
  --reference-embedding "$REFERENCE_EMBED"
)
if [[ "$chunked_mode" -eq 0 ]]; then
  cli_cmd+=(--detection-log "$DETECTIONS_PATH")
fi
if [[ ${#CLI_EXTRA[@]} -gt 0 ]]; then
  cli_cmd+=("${CLI_EXTRA[@]}")
fi
"${cli_cmd[@]}"

TIMELINE_PATH="$OUT_DIR/timeline.json"
python3 - <<PY
import json
from pathlib import Path
timeline = json.loads(Path("$TIMELINE_PATH").read_text())
summary = timeline.get("summary", {})
print(f"[demo] Timeline: $TIMELINE_PATH")
print(f"[demo] Clips directory: $OUT_DIR")
print(f"[demo] Target: {timeline.get('target')}")
print(
    "[demo] Segments:",
    summary.get("segment_count", 0),
    "Total duration:",
    summary.get("total_duration", 0),
    "seconds",
)
PY
