#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
RUN_DIR="${ROOT_DIR}/runs/cifar10_vit12_baseline_b2048_recipe_gpu${GPU_ID}"

mkdir -p "${RUN_DIR}" "${ROOT_DIR}/data/raw/cifar10"

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python -m remote_lab.cli \
  --config configs/experiments/cifar10_vit12_baseline_b2048_recipe_v1.json \
  --output-dir runs/cifar10_vit12_baseline_b2048_recipe_gpu${GPU_ID}
" > "${RUN_DIR}/train.log" 2>&1 &
RUN_PID=$!

echo "run_pid=${RUN_PID}"
echo "run_log=${RUN_DIR}/train.log"
