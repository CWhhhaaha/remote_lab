#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASELINE_RUN_NAME="imagenet1k_vit12_baseline_recipe_30ep_gpu${GPU_ID}"
BMB_RUN_NAME="imagenet1k_vit12_bmb_recipe_r64_30ep_gpu${GPU_ID}"
BASELINE_OUTPUT_DIR="${PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}"
BMB_OUTPUT_DIR="${PROJECT_ROOT}/runs/${BMB_RUN_NAME}"
PAIR_LOG_PATH="${PROJECT_ROOT}/runs/imagenet1k_vit12_baseline_then_bmb_r64_30ep_gpu${GPU_ID}.log"

mkdir -p "${BASELINE_OUTPUT_DIR}" "${BMB_OUTPUT_DIR}" "$(dirname "${PAIR_LOG_PATH}")"

nohup bash -lc "
set -euo pipefail
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${GPU_ID}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"

echo '[pair] starting baseline at '\"\$(date -Is)\"
'${PYTHON_BIN}' -m remote_lab.cli \
  --config configs/experiments/imagenet1k_vit12_baseline_recipe_30ep_v1.json \
  --output-dir runs/${BASELINE_RUN_NAME} \
  > runs/${BASELINE_RUN_NAME}/train.log 2>&1
echo '[pair] finished baseline at '\"\$(date -Is)\"

echo '[pair] starting bmb_r64 at '\"\$(date -Is)\"
'${PYTHON_BIN}' -m remote_lab.cli \
  --config configs/experiments/imagenet1k_vit12_bmb_recipe_r64_30ep_v1.json \
  --output-dir runs/${BMB_RUN_NAME} \
  > runs/${BMB_RUN_NAME}/train.log 2>&1
echo '[pair] finished bmb_r64 at '\"\$(date -Is)\"
" > "${PAIR_LOG_PATH}" 2>&1 &

echo "gpu_id=${GPU_ID}"
echo "baseline_run_name=${BASELINE_RUN_NAME}"
echo "bmb_run_name=${BMB_RUN_NAME}"
echo "pair_log_path=${PAIR_LOG_PATH}"
echo "baseline_log_path=${BASELINE_OUTPUT_DIR}/train.log"
echo "bmb_log_path=${BMB_OUTPUT_DIR}/train.log"
