#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ -d "${VENV_DIR}" ]]; then
  source "${VENV_DIR}/bin/activate"
elif ! command -v python >/dev/null 2>&1; then
  echo "No python executable found in the current shell."
  echo "Activate your conda environment or run: bash scripts/setup_venv.sh"
  exit 1
fi

cd "${ROOT_DIR}"

python -m remote_lab.cli "$@"
