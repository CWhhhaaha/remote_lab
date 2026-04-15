#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}"
  echo "Run: bash scripts/setup_venv.sh"
  exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${ROOT_DIR}"

python -m remote_lab.cli "$@"
