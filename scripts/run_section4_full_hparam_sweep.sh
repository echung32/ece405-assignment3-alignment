#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section4_sft_full_hparam_$(date +%Y%m%d-%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true

LOG_DIR="logs/section4/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section4/sft_experiment/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"
LEARNING_RATES=(5e-6 1e-5 2e-5 3e-5 5e-5)

EXTRA_ARGS=()
for arg in "$@"; do
    if [[ -n "$arg" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR} output_root=${OUTPUT_ROOT} normalize_constant=1.0" | tee -a "$ORCHESTRATOR_LOG"

for learning_rate in "${LEARNING_RATES[@]}"; do
    run_label="lr_${learning_rate//./p}"
    run_label="${run_label//-/m}"
    run_output="${OUTPUT_ROOT}/${run_label}"
    run_log="${LOG_DIR}/${run_label}.log"
    mkdir -p "$run_output"
    echo "[start] lr=${learning_rate} log=${run_log} output=${run_output} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
    uv run python scripts/sft_experiment.py \
        --skip-filtered-run \
        --subset-sizes full \
        --learning-rate "$learning_rate" \
        --per-device-batch-size 2 \
        --gradient-accumulation-steps 8 \
        --num-epochs 3 \
        --output-root "$run_output" \
        --log-root "$LOG_DIR" \
        --wandb-group "$CAMPAIGN_NAME" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$run_log"
    echo "[done] lr=${learning_rate} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
done

echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"