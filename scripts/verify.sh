#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
echo "[verify] Running pytest..."
if command -v python >/dev/null 2>&1; then
  python scripts/verify_pytest.py
else
  python3 scripts/verify_pytest.py
fi
