#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section5_ei_sweep_$(date +%Y%m%d-%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true

LOG_DIR="logs/section5/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section5/expert_iteration/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"

RUN_CONFIGS=(
    "g4 db512 ep1"
    "g4 db1024 ep2"
    "g8 db2048 ep1"
    "g8 db1024 ep2"
)

EXTRA_ARGS=()
for arg in "$@"; do
    if [[ -n "$arg" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR} output_root=${OUTPUT_ROOT}" | tee -a "$ORCHESTRATOR_LOG"

for config in "${RUN_CONFIGS[@]}"; do
    read -r rollout_cfg batch_cfg epoch_cfg <<<"$config"
    rollout_count="${rollout_cfg#g}"
    questions_per_step="${batch_cfg#db}"
    sft_epochs="${epoch_cfg#ep}"
    run_label="rollout_g${rollout_count}_db${questions_per_step}_ep${sft_epochs}"
    run_output="${OUTPUT_ROOT}"
    run_log="${LOG_DIR}/${run_label}.log"
    echo "[start] run=${run_label} log=${run_log} output=${run_output} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
    uv run python scripts/expert_iteration_experiment.py \
        --rollouts-per-question "$rollout_count" \
        --questions-per-step "$questions_per_step" \
        --sft-epochs-per-step "$sft_epochs" \
        --no-save-final-model \
        --output-root "$run_output" \
        --log-root "$LOG_DIR" \
        --wandb-group "$CAMPAIGN_NAME" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$run_log"
    echo "[done] run=${run_label} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
done

uv run python scripts/generate_section5_report.py --campaign "$CAMPAIGN_NAME" 2>&1 | tee -a "$ORCHESTRATOR_LOG"

echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"