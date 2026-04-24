#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${PROJECT_ROOT}/scripts/launch_imagenet1k_vit12_bmbuv_recipe_r64_s64_dp2_bs768_gpu01.sh"
bash "${PROJECT_ROOT}/scripts/launch_imagenet1k_vit12_baseline_recipe_dp2_bs768_gpu23.sh"
bash "${PROJECT_ROOT}/scripts/launch_imagenet1k_vit12_bbt_recipe_r64_dp2_bs768_gpu45.sh"
