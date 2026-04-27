#!/usr/bin/env bash
# Launch 4 pilot experiments in parallel on GPU 0,1,2,3.
# Usage: bash run_4gpu_pilot.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/experiments"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# GPU -> experiment mapping
GPU0_EXP="imagenet1k_vit12_fullyshared_recipe_30ep_v1"
GPU1_EXP="imagenet1k_vit12_lowrank_r32_recipe_30ep_v1"
GPU2_EXP="imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_v1"
GPU3_EXP="imagenet1k_vit12_partialshared_r48_recipe_30ep_v1"

run_one() {
    local gpu_id="$1"
    local exp_name="$2"
    local config_file="${CONFIG_DIR}/${exp_name}.json"
    local output_dir="${RUN_DIR}/${exp_name}"
    local log_file="${LOG_DIR}/${exp_name}.gpu${gpu_id}.log"

    if [ ! -f "${config_file}" ]; then
        echo "[ERROR GPU${gpu_id}] Config not found: ${config_file}" >&2
        return 1
    fi

    echo "[GPU${gpu_id} START] ${exp_name}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -m remote_lab.cli \
        --config "${config_file}" \
        --output-dir "${output_dir}" \
        > "${log_file}" 2>&1
    local exit_code=$?
    echo "[GPU${gpu_id} DONE]  ${exp_name} -> exit_code=${exit_code}"
    return ${exit_code}
}

# Launch all 4 in parallel
run_one 0 "${GPU0_EXP}" &
PID0=$!
run_one 1 "${GPU1_EXP}" &
PID1=$!
run_one 2 "${GPU2_EXP}" &
PID2=$!
run_one 3 "${GPU3_EXP}" &
PID3=$!

echo "========================================"
echo "4 experiments launched in parallel"
echo "GPU0 PID=$PID0  -> ${GPU0_EXP}"
echo "GPU1 PID=$PID1  -> ${GPU1_EXP}"
echo "GPU2 PID=$PID2  -> ${GPU2_EXP}"
echo "GPU3 PID=$PID3  -> ${GPU3_EXP}"
echo "========================================"
echo "Waiting for completion..."
wait $PID0
wait $PID1
wait $PID2
wait $PID3
echo "All experiments completed. Logs: ${LOG_DIR}"
