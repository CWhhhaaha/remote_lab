#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

python scripts/tokenize_jigsaw.py \
  --train-input data/processed/train/jigsaw.json.gz \
  --test-input data/processed/test/jigsaw.json.gz \
  --train-output data/cache/jigsaw/train_tokens.pt \
  --test-output data/cache/jigsaw/test_tokens.pt \
  --max-length 128

ls -lh data/cache/jigsaw
