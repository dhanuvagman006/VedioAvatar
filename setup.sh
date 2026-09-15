#!/usr/bin/env bash
# Linux / WSL2 bootstrap. Extra arguments are passed to scripts/setup_envs.py.
set -euo pipefail
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "python3 (3.10+) is required"; exit 1; }
command -v git >/dev/null || { echo "git is required"; exit 1; }

[ -d .venv ] || python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/setup_envs.py "$@"
./.venv/bin/python -m avatar_pipeline.cli doctor || true
echo
echo "Ready. Use ./avatar (wraps the CLI), or: source .venv/bin/activate"
echo "Then:  ./avatar serve   (web UI on http://localhost:8000)"
