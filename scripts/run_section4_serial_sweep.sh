#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section4_sft_serial_$(date +%Y%m%d-%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true
LOG_DIR="logs/section4/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section4/sft_experiment/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"

EXTRA_ARGS=()
for arg in "$@"; do
    if [[ -n "$arg" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR} output_root=${OUTPUT_ROOT} normalize_constant=1.0" | tee -a "$ORCHESTRATOR_LOG"

for subset in 128 256 512 1024 full; do
    RUN_LOG="${LOG_DIR}/run_${subset}.log"
    RUN_OUTPUT="${OUTPUT_ROOT}/${subset}"
    mkdir -p "$RUN_OUTPUT"
    echo "[start] subset=${subset} log=${RUN_LOG} output=${RUN_OUTPUT} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$RUN_LOG"
    uv run python scripts/sft_experiment.py \
        --skip-filtered-run \
        --num-epochs 3 \
        --subset-sizes "$subset" \
        --output-root "$RUN_OUTPUT" \
        --log-root "$LOG_DIR" \
        --wandb-group "$CAMPAIGN_NAME" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$RUN_LOG"
    echo "[done] subset=${subset} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$RUN_LOG"
done

echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"