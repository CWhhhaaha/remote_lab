#!/usr/bin/env bash
# Run 10 experiments in parallel across GPU 1, 2, 3.
# Usage: conda activate remote_lab && bash run_parallel_3gpu.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/experiments"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# Map experiments to GPUs (GPU 1, 2, 3)
# Round-robin distribution: 4 jobs on GPU1, 3 on GPU2, 3 on GPU3
GPU1_EXPS=(
    "imagenet1k_vit12_fullyshared_recipe_30ep_v1"
    "imagenet1k_vit12_partialshared_r32_recipe_30ep_v1"
    "imagenet1k_vit12_partialshared_r48_recipe_30ep_v1"
    "imagenet1k_vit12_lowrank_r64_recipe_30ep_v1"
)
GPU2_EXPS=(
    "imagenet1k_vit12_lowrank_r32_recipe_30ep_v1"
    "imagenet1k_vit12_lowrank_r16_recipe_30ep_v1"
    "imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_v1"
)
GPU3_EXPS=(
    "imagenet1k_vit12_bmbuv_recipe_r16_s16_30ep_v1"
    "imagenet1k_vit12_bbt_recipe_r32_30ep_v1"
    "imagenet1k_vit12_bbt_recipe_r16_30ep_v1"
)

run_on_gpu() {
    local gpu_id="$1"
    shift
    local exps=("$@")
    for exp_name in "${exps[@]}"; do
        local config_file="${CONFIG_DIR}/${exp_name}.json"
        local output_dir="${RUN_DIR}/${exp_name}"
        local log_file="${LOG_DIR}/${exp_name}.gpu${gpu_id}.log"

        if [ ! -f "${config_file}" ]; then
            echo "[GPU${gpu_id} SKIP] Config not found: ${config_file}"
            continue
        fi

        echo "[GPU${gpu_id} START] ${exp_name}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python -m remote_lab.cli \
            --config "${config_file}" \
            --output-dir "${output_dir}" \
            > "${log_file}" 2>&1
        echo "[GPU${gpu_id} DONE]  ${exp_name} -> exit_code=$?"
    done
}

# Launch three background processes, one per GPU
run_on_gpu 1 "${GPU1_EXPS[@]}" &
PID1=$!
run_on_gpu 2 "${GPU2_EXPS[@]}" &
PID2=$!
run_on_gpu 3 "${GPU3_EXPS[@]}" &
PID3=$!

echo "All 10 experiments launched in parallel."
echo "GPU1 PID=$PID1, GPU2 PID=$PID2, GPU3 PID=$PID3"
echo "Waiting for completion..."
wait $PID1
wait $PID2
wait $PID3
echo "All experiments completed. Logs are in ${LOG_DIR}"
