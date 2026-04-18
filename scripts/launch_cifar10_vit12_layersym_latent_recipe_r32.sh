#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-2}"
RUN_DIR="${ROOT_DIR}/runs/cifar10_vit12_layersym_latent_b2048_recipe_r32_gpu${GPU_ID}"

mkdir -p "${RUN_DIR}"

nohup bash -lc "
cd '${ROOT_DIR}'
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python -m remote_lab.cli \
  --config configs/experiments/cifar10_vit12_layersym_latent_b2048_recipe_r32_v1.json \
  --output-dir runs/cifar10_vit12_layersym_latent_b2048_recipe_r32_gpu${GPU_ID}
" > "${RUN_DIR}/train.log" 2>&1 &

echo "run_dir=${RUN_DIR}"
echo "log_path=${RUN_DIR}/train.log"
