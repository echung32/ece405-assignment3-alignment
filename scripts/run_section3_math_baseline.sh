#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section3_math_baseline_$(date +%Y%m%d-%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true

LOG_DIR="logs/section3/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section3/math_baseline/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"
RUN_LOG="${LOG_DIR}/baseline.log"
PREDICTIONS_PATH="${OUTPUT_ROOT}/math_baseline_predictions.jsonl"
SUMMARY_PATH="${OUTPUT_ROOT}/math_baseline_summary.json"

EXTRA_ARGS=()
for arg in "$@"; do
    if [[ -n "$arg" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR} output_root=${OUTPUT_ROOT} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"
echo "[start] run=qwen2.5_math_1.5b_r1_zero_val log=${RUN_LOG} summary=${SUMMARY_PATH} predictions=${PREDICTIONS_PATH} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$RUN_LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
uv run python scripts/math_baseline.py \
    --model-path data/Qwen/Qwen2.5-Math-1.5B \
    --input-path data/math/val.jsonl \
    --output-path "$PREDICTIONS_PATH" \
    --summary-path "$SUMMARY_PATH" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$RUN_LOG"

echo "[done] run=qwen2.5_math_1.5b_r1_zero_val $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$RUN_LOG"
echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"