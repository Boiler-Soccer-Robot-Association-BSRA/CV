#!/usr/bin/env bash
# One-command dev environment setup. Works on Apple Silicon Macs and Linux
# (anywhere python3.10 and venv are available).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3.10 not found, falling back to python3 (see .python-version for the pinned version)" >&2
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install

echo
echo "Done. Activate this environment in new shells with: source venv/bin/activate"
echo "Verify it with:    python scripts/quickstart.py"
echo "Run tests with:     pytest testing-deployment-team/tests/"
