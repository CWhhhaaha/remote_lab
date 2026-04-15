#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT_DIR}/data}"
RAW_DIR="${DATA_ROOT}/raw/jigsaw"

mkdir -p "${RAW_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Missing kaggle CLI. Install it first in your active environment:"
  echo "  python -m pip install kaggle"
  exit 1
fi

echo "Downloading Jigsaw competition archive into ${RAW_DIR}"
kaggle competitions download \
  -c jigsaw-toxic-comment-classification-challenge \
  -p "${RAW_DIR}"

echo "Download complete."
echo "Expected archive:"
echo "  ${RAW_DIR}/jigsaw-toxic-comment-classification-challenge.zip"
