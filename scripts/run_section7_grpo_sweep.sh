#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section7_grpo_sweep_$(date +%Y%m%d-%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true

LOG_DIR="logs/section7/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section7/grpo_experiment/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"

RUN_CONFIGS=(
    "label=reinforce_lr1e-5_std lr=1e-5 loss=reinforce_with_baseline std=1 epochs=1 train_batch=256 grad_accum=128"
    "label=reinforce_lr5e-6_std lr=5e-6 loss=reinforce_with_baseline std=1 epochs=1 train_batch=256 grad_accum=128"
    "label=no_baseline_lr1e-5_std lr=1e-5 loss=no_baseline std=1 epochs=1 train_batch=256 grad_accum=128"
    "label=no_baseline_lr5e-6_std lr=5e-6 loss=no_baseline std=1 epochs=1 train_batch=256 grad_accum=128"
    "label=reinforce_lr1e-5_mean lr=1e-5 loss=reinforce_with_baseline std=0 epochs=1 train_batch=256 grad_accum=128"
    "label=grpo_clip_lr1e-5_ep2_tb128 lr=1e-5 loss=grpo_clip std=1 epochs=2 train_batch=128 grad_accum=64"
    "label=grpo_clip_lr5e-6_ep4_tb128 lr=5e-6 loss=grpo_clip std=1 epochs=4 train_batch=128 grad_accum=64"
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
    unset run_label learning_rate loss_type use_std epochs_per_rollout_batch train_batch_size gradient_accumulation_steps
    for kv in $config; do
        key="${kv%%=*}"
        value="${kv#*=}"
        case "$key" in
            label) run_label="$value" ;;
            lr) learning_rate="$value" ;;
            loss) loss_type="$value" ;;
            std) use_std="$value" ;;
            epochs) epochs_per_rollout_batch="$value" ;;
            train_batch) train_batch_size="$value" ;;
            grad_accum) gradient_accumulation_steps="$value" ;;
        esac
    done

    run_log="${LOG_DIR}/${run_label}.log"
    echo "[start] run=${run_label} lr=${learning_rate} loss=${loss_type} std=${use_std} epochs=${epochs_per_rollout_batch} train_batch=${train_batch_size} grad_accum=${gradient_accumulation_steps} log=${run_log} output=${OUTPUT_ROOT} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"

    std_flag="--use-std-normalization"
    if [[ "$use_std" == "0" ]]; then
        std_flag="--no-use-std-normalization"
    fi

    uv run python scripts/grpo_experiment.py \
        --learning-rate "$learning_rate" \
        --loss-type "$loss_type" \
        --epochs-per-rollout-batch "$epochs_per_rollout_batch" \
        --train-batch-size "$train_batch_size" \
        --gradient-accumulation-steps "$gradient_accumulation_steps" \
        "$std_flag" \
        --output-root "$OUTPUT_ROOT" \
        --log-root "$LOG_DIR" \
        --wandb-group "$CAMPAIGN_NAME" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$run_log"

    echo "[done] run=${run_label} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
done

uv run python scripts/generate_section7_report.py --campaign "$CAMPAIGN_NAME" 2>&1 | tee -a "$ORCHESTRATOR_LOG"

echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"