#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="imagenet1k_vit12_bmb_recipe_r64_30ep_gpu${GPU_ID}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"

nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${GPU_ID}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m remote_lab.cli \
  --config configs/experiments/imagenet1k_vit12_bmb_recipe_r64_30ep_v1.json \
  --output-dir runs/${RUN_NAME}
" > "${LOG_PATH}" 2>&1 &

echo "gpu_id=${GPU_ID}"
echo "run_name=${RUN_NAME}"
echo "log_path=${LOG_PATH}"
