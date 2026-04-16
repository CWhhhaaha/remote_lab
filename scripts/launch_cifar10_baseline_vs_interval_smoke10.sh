#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_GPU="${BASELINE_GPU:-2}"
INTERVAL_GPU="${INTERVAL_GPU:-6}"

BASELINE_RUN_DIR="${ROOT_DIR}/runs/cifar10_vit6_baseline_smoke10_gpu${BASELINE_GPU}"
INTERVAL_RUN_DIR="${ROOT_DIR}/runs/cifar10_vit6_interval_smoke10_gpu${INTERVAL_GPU}"

mkdir -p "${BASELINE_RUN_DIR}" "${INTERVAL_RUN_DIR}" "${ROOT_DIR}/data/raw/cifar10"

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${BASELINE_GPU}
python -m remote_lab.cli \
  --config configs/experiments/cifar10_vit6_baseline_smoke10_v1.json \
  --output-dir runs/cifar10_vit6_baseline_smoke10_gpu${BASELINE_GPU}
" > "${BASELINE_RUN_DIR}/train.log" 2>&1 &
BASELINE_PID=$!

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${INTERVAL_GPU}
python -m remote_lab.cli \
  --config configs/experiments/cifar10_vit6_interval_reg_linear_lambda05_smoke10_v1.json \
  --output-dir runs/cifar10_vit6_interval_smoke10_gpu${INTERVAL_GPU}
" > "${INTERVAL_RUN_DIR}/train.log" 2>&1 &
INTERVAL_PID=$!

echo "baseline_pid=${BASELINE_PID}"
echo "baseline_log=${BASELINE_RUN_DIR}/train.log"
echo "interval_pid=${INTERVAL_PID}"
echo "interval_log=${INTERVAL_RUN_DIR}/train.log"
