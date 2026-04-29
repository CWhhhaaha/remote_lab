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

INPUT_PATH="${INPUT_PATH:-/data/chenwei/datasets/text/c4-realnewslike}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/chenwei/datasets/text/c4-realnewslike-gpt2-1024}"
NUM_PROC="${NUM_PROC:-32}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
VAL_FRACTION="${VAL_FRACTION:-0.005}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_PATH="${LOG_DIR}/prepare_c4_gpt2_cache.log"

mkdir -p "${LOG_DIR}"

nohup bash -lc "
cd '${PROJECT_ROOT}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' scripts/prepare_c4_gpt2_cache.py \
  --input-path '${INPUT_PATH}' \
  --output-path '${OUTPUT_PATH}' \
  --seq-length ${SEQ_LENGTH} \
  --val-fraction ${VAL_FRACTION} \
  --num-proc ${NUM_PROC} \
  --seed ${SEED}
" > "${LOG_PATH}" 2>&1 &

echo "Launched cache preparation in background."
echo "log=${LOG_PATH}"
echo "output=${OUTPUT_PATH}"
echo "pid=$!"
