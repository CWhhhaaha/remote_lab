#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYMM_GPU="${SYMM_GPU:-4}"
RUN_DIR="${ROOT_DIR}/runs/jigsaw_symm_gpu${SYMM_GPU}"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${ROOT_DIR}/data/cache/jigsaw/train_tokens.pt" || ! -f "${ROOT_DIR}/data/cache/jigsaw/test_tokens.pt" ]]; then
  echo "Missing token cache under data/cache/jigsaw. Run scripts/prepare_jigsaw_cache.sh first."
  exit 1
fi

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${SYMM_GPU}
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_symm_init_v1.json \
  --output-dir runs/jigsaw_symm_gpu${SYMM_GPU}
" > "${RUN_DIR}/train.log" 2>&1 &
SYMM_PID=$!

echo "symm_pid=${SYMM_PID}"
echo "symm_log=${RUN_DIR}/train.log"
