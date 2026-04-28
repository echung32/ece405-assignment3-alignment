#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GRAD_ACCUM="${GRAD_ACCUM:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_GPU_ID="${TRAIN_GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
N_GRPO_STEPS="${N_GRPO_STEPS:-1}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-8}"
NUM_LOG_GENERATIONS="${NUM_LOG_GENERATIONS:-2}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-section7_throughput_probe_ga${GRAD_ACCUM}_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-ga${GRAD_ACCUM}}"
LOG_ROOT="logs/section7/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section7/grpo_experiment/${CAMPAIGN_NAME}"
MONITOR_LOG="${LOG_ROOT}/nvidia_smi.csv"
RUN_LOG="${LOG_ROOT}/run.log"
if command -v rg >/dev/null 2>&1; then
    SEARCH_CMD=(rg)
else
    SEARCH_CMD=(grep -E)
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

cleanup() {
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill "$MONITOR_PID" >/dev/null 2>&1 || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "[probe] campaign=${CAMPAIGN_NAME} grad_accum=${GRAD_ACCUM} train_batch=${TRAIN_BATCH_SIZE} train_gpu=${TRAIN_GPU_ID} eval_gpu=${EVAL_GPU_ID}" | tee "$LOG_ROOT/probe.log"
echo "[probe] microbatch_size=$((TRAIN_BATCH_SIZE / GRAD_ACCUM))" | tee -a "$LOG_ROOT/probe.log"

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$MONITOR_LOG" &
MONITOR_PID=$!

WANDB_MODE=offline uv run python scripts/grpo_experiment.py \
    --learning-rate 1e-5 \
    --loss-type reinforce_with_baseline \
    --epochs-per-rollout-batch 1 \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --n-grpo-steps "$N_GRPO_STEPS" \
    --max-eval-examples "$MAX_EVAL_EXAMPLES" \
    --num-log-generations "$NUM_LOG_GENERATIONS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --train-gpu-id "$TRAIN_GPU_ID" \
    --eval-gpu-id "$EVAL_GPU_ID" \
    --no-save-final-model \
    --output-root "$OUTPUT_ROOT" \
    --log-root "$LOG_ROOT" \
    --wandb-group "$CAMPAIGN_NAME" \
    > "$RUN_LOG" 2>&1

cleanup
MONITOR_PID=""

RUN_DIR="${OUTPUT_ROOT}/lr_1em05_loss_reinforce_with_baseline_std_g8_rb256_ep1"
SUMMARY_PATH="${RUN_DIR}/summary.json"
STEP_SUMMARY_PATH="${RUN_DIR}/step_0001/summary.json"

echo "--- step trace ---"
"${SEARCH_CMD[@]}" "\\[step-trace\\]" "$RUN_LOG" || true

echo "--- rollout progress ---"
"${SEARCH_CMD[@]}" "Processed prompts: 100%.*256/256" "$RUN_LOG" || true

echo "--- gpu peaks ---"
awk -F',' '
{
    gsub(/^ +| +$/, "", $1)
    if ($2 + 0 > max[$1]) {
        max[$1] = $2 + 0
    }
    if ($3 + 0 > total[$1]) {
        total[$1] = $3 + 0
    }
}
END {
    for (gpu in max) {
        printf("gpu=%s peak_used_mb=%.0f total_mb=%.0f\n", gpu, max[gpu], total[gpu])
    }
}
' "$MONITOR_LOG" | sort

echo "--- summary ---"
cat "$SUMMARY_PATH"

if [[ -f "$STEP_SUMMARY_PATH" ]]; then
    echo "--- step summary ---"
    cat "$STEP_SUMMARY_PATH"
fi