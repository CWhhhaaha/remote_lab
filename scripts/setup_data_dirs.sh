#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT_DIR}/data}"

mkdir -p \
  "${DATA_ROOT}/raw/jigsaw" \
  "${DATA_ROOT}/processed/train" \
  "${DATA_ROOT}/processed/test" \
  "${DATA_ROOT}/cache/huggingface" \
  "${DATA_ROOT}/cache/tmp"

echo "Data directories ready under ${DATA_ROOT}"
