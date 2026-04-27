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

# GPU allocation: use free GPUs (0 and 1 have ~25-42GB free)
launch_experiment() {
  local gpu_id="$1"
  local variant="$2"
  local extra_args="${3:-}"
  local run_name="gpt2_c4_${variant}_50k_gpu${gpu_id}"

  local output_dir="${PROJECT_ROOT}/runs/${run_name}"
  local log_path="${output_dir}/train.log"

  mkdir -p "${output_dir}"

  nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${gpu_id}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m remote_lab.train_gpt2_c4 \
  --variant ${variant} \
  --output-dir ${output_dir} \
  --max-steps 50000 \
  --per-device-batch-size 32 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2.5e-4 \
  --warmup-steps 2000 \
  --eval-steps 1000 \
  --save-steps 5000 \
  --logging-steps 100 \
  --bf16 \
  --gradient-checkpointing \
  ${extra_args}
" > "${log_path}" 2>&1 &

  echo "Launched: gpu=${gpu_id} variant=${variant} log=${log_path}"
}

# GPU 0: baseline
launch_experiment 0 "baseline" ""

# GPU 1: lowrank r32
launch_experiment 1 "lowrank" "--rank 32"

# GPU 4: bmbuv r32s32 (if enough memory; ~9GB free is tight for 124M + checkpointing)
# Uncomment when GPU 4/5 are free:
# launch_experiment 4 "bmbuv" "--rank 32 --factor-rank 32"

# GPU 5: fullyshared
# Uncomment when GPU 4/5 are free:
# launch_experiment 5 "fullyshared" ""

echo ""
echo "Experiments launched. Checking processes in 3s..."
sleep 3
ps aux | grep "train_gpt2_c4" | grep -v grep || true
