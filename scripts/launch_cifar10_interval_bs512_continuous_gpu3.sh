#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_GPU="${INTERVAL_GPU:-3}"
RUN_DIR="${ROOT_DIR}/runs/cifar10_vit6_interval_bs512_continuous_gpu${INTERVAL_GPU}"

mkdir -p "${RUN_DIR}" "${ROOT_DIR}/data/raw/cifar10"

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${INTERVAL_GPU}
python -m remote_lab.cli \
  --config configs/experiments/cifar10_vit6_interval_reg_linear_lambda05_bs512_continuous_v1.json \
  --output-dir runs/cifar10_vit6_interval_bs512_continuous_gpu${INTERVAL_GPU}
" > "${RUN_DIR}/train.log" 2>&1 &
INTERVAL_PID=$!

echo "interval_pid=${INTERVAL_PID}"
echo "interval_log=${RUN_DIR}/train.log"
