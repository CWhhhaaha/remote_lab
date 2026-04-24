#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-4,5}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/chenwei/.conda/envs/remote_lab/bin/python" ]]; then
    PYTHON_BIN="/data/chenwei/.conda/envs/remote_lab/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
RUN_NAME="imagenet1k_vit12_bbt_recipe_r64_dp2_bs768_gpu45"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"

nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${GPU_IDS}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m remote_lab.cli \
  --config configs/experiments/imagenet1k_vit12_bbt_recipe_r64_dp2_bs768_30ep_v1.json \
  --output-dir runs/${RUN_NAME}
" > "${LOG_PATH}" 2>&1 &

echo "gpu_ids=${GPU_IDS}"
echo "run_name=${RUN_NAME}"
echo "log_path=${LOG_PATH}"
