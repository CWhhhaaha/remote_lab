#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT_DIR}/data}"
RAW_DIR="${DATA_ROOT}/raw/jigsaw"
COMPETITION="jigsaw-toxic-comment-classification-challenge"
FILES=(
  "train.csv.zip"
  "test.csv.zip"
  "test_labels.csv.zip"
  "sample_submission.csv.zip"
)

mkdir -p "${RAW_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Missing kaggle CLI. Install it first in your active environment:"
  echo "  python -m pip install kaggle"
  exit 1
fi

echo "Downloading Jigsaw competition files into ${RAW_DIR}"
for filename in "${FILES[@]}"; do
  echo "  -> ${filename}"
  kaggle competitions download \
    -c "${COMPETITION}" \
    -f "${filename}" \
    -p "${RAW_DIR}"
done

echo "Download complete."
echo "Downloaded files:"
printf '  %s\n' "${FILES[@]}"
