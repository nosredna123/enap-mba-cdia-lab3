#!/usr/bin/env bash
# Simple project setup: create .venv, activate it, and install dependencies.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_cmd="${PYTHON:-python3}"

if ! command -v "$python_cmd" >/dev/null 2>&1; then
  echo "Python command not found: $python_cmd" >&2
  echo "Set PYTHON to a valid interpreter (e.g., python3.11)." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv using $python_cmd ..."
  "$python_cmd" -m venv .venv
else
  echo "Virtual environment .venv already exists; reusing it."
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo "Installing project requirements ..."
pip install -r requirements.txt

cat <<'EOF'

Environment ready.
To activate later, run:
  source .venv/bin/activate
EOF
