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

launch_experiment() {
  local gpu_id="$1"
  local config_name="$2"
  local run_name="$3"

  local output_dir="${PROJECT_ROOT}/runs/${run_name}"
  local log_path="${output_dir}/train.log"

  mkdir -p "${output_dir}"

  nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${gpu_id}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m remote_lab.cli \
  --config configs/experiments/${config_name} \
  --output-dir runs/${run_name}
" > "${log_path}" 2>&1 &

  echo "Launched: gpu=${gpu_id} run=${run_name}"
}

# GPU 2: Fully Shared
launch_experiment 2 \
  "imagenet1k_vit12_fullyshared_recipe_30ep_v1.json" \
  "imagenet1k_vit12_fullyshared_recipe_30ep_gpu2"

# GPU 3: LowRank r32
launch_experiment 3 \
  "imagenet1k_vit12_lowrank_r32_recipe_30ep_v1.json" \
  "imagenet1k_vit12_lowrank_r32_recipe_30ep_gpu3"

# GPU 6: BMB-UV r32s32
launch_experiment 6 \
  "imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_v1.json" \
  "imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_gpu6"

# GPU 7: Partial Shared r48
launch_experiment 7 \
  "imagenet1k_vit12_partialshared_r48_recipe_30ep_v1.json" \
  "imagenet1k_vit12_partialshared_r48_recipe_30ep_gpu7"

echo ""
echo "All 4 experiments launched. Checking processes in 3s..."
sleep 3
ps aux | grep "remote_lab.cli" | grep -v grep || true
