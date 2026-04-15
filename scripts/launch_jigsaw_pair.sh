#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_GPU="${BASELINE_GPU:-2}"
INTERVAL_GPU="${INTERVAL_GPU:-3}"

BASELINE_RUN_DIR="${ROOT_DIR}/runs/jigsaw_baseline_gpu${BASELINE_GPU}"
INTERVAL_RUN_DIR="${ROOT_DIR}/runs/jigsaw_interval_gpu${INTERVAL_GPU}"

mkdir -p "${BASELINE_RUN_DIR}" "${INTERVAL_RUN_DIR}"

if [[ ! -f "${ROOT_DIR}/data/cache/jigsaw/train_tokens.pt" || ! -f "${ROOT_DIR}/data/cache/jigsaw/test_tokens.pt" ]]; then
  echo "Missing token cache under data/cache/jigsaw. Run scripts/prepare_jigsaw_cache.sh first."
  exit 1
fi

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${BASELINE_GPU}
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_baseline_v1.json \
  --output-dir runs/jigsaw_baseline_gpu${BASELINE_GPU}
" > "${BASELINE_RUN_DIR}/train.log" 2>&1 &
BASELINE_PID=$!

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${INTERVAL_GPU}
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_interval_reg_v1.json \
  --output-dir runs/jigsaw_interval_gpu${INTERVAL_GPU}
" > "${INTERVAL_RUN_DIR}/train.log" 2>&1 &
INTERVAL_PID=$!

echo "baseline_pid=${BASELINE_PID}"
echo "baseline_log=${BASELINE_RUN_DIR}/train.log"
echo "interval_pid=${INTERVAL_PID}"
echo "interval_log=${INTERVAL_RUN_DIR}/train.log"
