#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/chenwei/.conda/envs/remote_lab/bin/python" ]]; then
    PYTHON_BIN="/data/chenwei/.conda/envs/remote_lab/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

DATASET_PATH="${DATASET_PATH:-/data/chenwei/datasets/text/c4-realnewslike-gpt2-1024}"
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"

GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_FULLSHR="${GPU_FULLSHR:-1}"
GPU_PARTSHR48="${GPU_PARTSHR48:-2}"
GPU_BMB64="${GPU_BMB64:-3}"
GPU_BMBUV64="${GPU_BMBUV64:-4}"
GPU_BBT64="${GPU_BBT64:-5}"
GPU_LOWRANK32="${GPU_LOWRANK32:-6}"
GPU_BMBUV32="${GPU_BMBUV32:-7}"

launch_experiment() {
  local gpu_id="$1"
  local run_name="$2"
  local variant="$3"
  local extra_args="${4:-}"

  local output_dir="${PROJECT_ROOT}/runs/${run_name}"
  local log_path="${output_dir}/train.log"

  mkdir -p "${output_dir}"

  nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${gpu_id}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m remote_lab.train_gpt2_c4 \
  --variant ${variant} \
  --dataset-path '${DATASET_PATH}' \
  --output-dir '${output_dir}' \
  --max-steps ${MAX_STEPS} \
  --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --learning-rate ${LEARNING_RATE} \
  --warmup-steps ${WARMUP_STEPS} \
  --eval-steps ${EVAL_STEPS} \
  --save-steps ${SAVE_STEPS} \
  --logging-steps ${LOGGING_STEPS} \
  --bf16 \
  --gradient-checkpointing \
  ${extra_args}
" > "${log_path}" 2>&1 &

  echo "Launched: gpu=${gpu_id} run=${run_name} variant=${variant}"
  echo "  log=${log_path}"
}

launch_experiment "${GPU_BASELINE}" "gpt2_c4_baseline_50k" "baseline" ""
launch_experiment "${GPU_FULLSHR}" "gpt2_c4_fullshr_50k" "fullyshared" ""
launch_experiment "${GPU_PARTSHR48}" "gpt2_c4_partshr48_50k" "partialshared" "--shared-dim 48"
launch_experiment "${GPU_BMB64}" "gpt2_c4_bmb_r64_50k" "bmb" "--rank 64"
launch_experiment "${GPU_BMBUV64}" "gpt2_c4_bmbuv_r64s64_50k" "bmbuv" "--rank 64 --factor-rank 64"
launch_experiment "${GPU_BBT64}" "gpt2_c4_bbt_r64_50k" "bbt" "--rank 64"
launch_experiment "${GPU_LOWRANK32}" "gpt2_c4_lowrank_r32_50k" "lowrank" "--rank 32"
launch_experiment "${GPU_BMBUV32}" "gpt2_c4_bmbuv_r32s32_50k" "bmbuv" "--rank 32 --factor-rank 32"

echo ""
echo "All 8 GPT-2/C4 runs launched."
echo "Dataset path: ${DATASET_PATH}"
echo "Check processes with:"
echo "  ps aux | grep 'remote_lab.train_gpt2_c4' | grep -v grep"
