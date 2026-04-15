#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINEAR_GPU="${LINEAR_GPU:-2}"
BASELINE_GPU="${BASELINE_GPU:-3}"

LINEAR_RUN_DIR="${ROOT_DIR}/runs/jigsaw_interval_linear_lambda05_bs256_gpu${LINEAR_GPU}"
BASELINE_RUN_DIR="${ROOT_DIR}/runs/jigsaw_baseline_bs256_gpu${BASELINE_GPU}"

mkdir -p "${LINEAR_RUN_DIR}" "${BASELINE_RUN_DIR}"

if [[ ! -f "${ROOT_DIR}/data/cache/jigsaw/train_tokens.pt" || ! -f "${ROOT_DIR}/data/cache/jigsaw/test_tokens.pt" ]]; then
  echo "Missing token cache under data/cache/jigsaw. Run scripts/prepare_jigsaw_cache.sh first."
  exit 1
fi

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${LINEAR_GPU}
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_interval_reg_linear_lambda05_bs256_acc1_eval1_v1.json \
  --output-dir runs/jigsaw_interval_linear_lambda05_bs256_gpu${LINEAR_GPU}
" > "${LINEAR_RUN_DIR}/train.log" 2>&1 &
LINEAR_PID=$!

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${BASELINE_GPU}
python -m remote_lab.cli \
  --config configs/experiments/jigsaw_bert_small_encoder_baseline_bs256_acc1_eval1_v1.json \
  --output-dir runs/jigsaw_baseline_bs256_gpu${BASELINE_GPU}
" > "${BASELINE_RUN_DIR}/train.log" 2>&1 &
BASELINE_PID=$!

echo "linear_pid=${LINEAR_PID}"
echo "linear_log=${LINEAR_RUN_DIR}/train.log"
echo "baseline_pid=${BASELINE_PID}"
echo "baseline_log=${BASELINE_RUN_DIR}/train.log"
