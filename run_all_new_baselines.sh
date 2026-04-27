#!/usr/bin/env bash
# Batch script to run all 10 new baseline/ablation experiments on ImageNet-1K.
# Usage: bash run_all_new_baselines.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/experiments"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_DIR="${PROJECT_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

# List of new experiment configs
EXPERIMENTS=(
    "imagenet1k_vit12_fullyshared_recipe_30ep_v1"
    "imagenet1k_vit12_partialshared_r32_recipe_30ep_v1"
    "imagenet1k_vit12_partialshared_r48_recipe_30ep_v1"
    "imagenet1k_vit12_lowrank_r64_recipe_30ep_v1"
    "imagenet1k_vit12_lowrank_r32_recipe_30ep_v1"
    "imagenet1k_vit12_lowrank_r16_recipe_30ep_v1"
    "imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_v1"
    "imagenet1k_vit12_bmbuv_recipe_r16_s16_30ep_v1"
    "imagenet1k_vit12_bbt_recipe_r32_30ep_v1"
    "imagenet1k_vit12_bbt_recipe_r16_30ep_v1"
)

# Run experiments sequentially with nohup.
# If you have multiple GPUs and want parallel runs, modify this loop.
for exp_name in "${EXPERIMENTS[@]}"; do
    config_file="${CONFIG_DIR}/${exp_name}.json"
    output_dir="${RUN_DIR}/${exp_name}"
    log_file="${LOG_DIR}/${exp_name}.log"

    if [ ! -f "${config_file}" ]; then
        echo "[SKIP] Config not found: ${config_file}"
        continue
    fi

    echo "[START] ${exp_name} -> ${output_dir}"
    python -m remote_lab.cli \
        --config "${config_file}" \
        --output-dir "${output_dir}" \
        > "${log_file}" 2>&1
    echo "[DONE]  ${exp_name} -> exit_code=$?"
done

echo "All experiments completed. Logs are in ${LOG_DIR}"
