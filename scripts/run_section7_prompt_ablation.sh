#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_NAME="${1:-section7_grpo_prompt_ablation_$(date +%Y%m%d_%H%M%S)}"
shift $(( $# > 0 ? 1 : 0 )) || true

BASE_LOSS_TYPE="${BASE_LOSS_TYPE:-grpo_clip}"
BASE_EPOCHS="${BASE_EPOCHS:-4}"
BASE_TRAIN_BATCH="${BASE_TRAIN_BATCH:-128}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-32}"
BASE_STEPS="${BASE_STEPS:-200}"

LOG_DIR="logs/section7/${CAMPAIGN_NAME}"
OUTPUT_ROOT="data/section7/grpo_experiment/${CAMPAIGN_NAME}"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"

RUN_CONFIGS=(
    "label=r1_zero_prompt lr=1e-5 loss=${BASE_LOSS_TYPE} reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_normalize std=0 epochs=${BASE_EPOCHS} train_batch=${BASE_TRAIN_BATCH} grad_accum=${BASE_GRAD_ACCUM} steps=${BASE_STEPS}"
    "label=question_only_prompt lr=1e-5 loss=${BASE_LOSS_TYPE} reward=question_only prompt=cs336_alignment/prompts/question_only.prompt length_norm=masked_normalize std=0 epochs=${BASE_EPOCHS} train_batch=${BASE_TRAIN_BATCH} grad_accum=${BASE_GRAD_ACCUM} steps=${BASE_STEPS}"
)

EXTRA_ARGS=()
for arg in "$@"; do
    if [[ -n "$arg" ]]; then
        EXTRA_ARGS+=("$arg")
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR} output_root=${OUTPUT_ROOT} base_loss=${BASE_LOSS_TYPE} base_epochs=${BASE_EPOCHS} base_train_batch=${BASE_TRAIN_BATCH} base_grad_accum=${BASE_GRAD_ACCUM}" | tee -a "$ORCHESTRATOR_LOG"

for config in "${RUN_CONFIGS[@]}"; do
    unset run_label learning_rate loss_type reward_function prompt_path length_normalization use_std epochs_per_rollout_batch train_batch_size gradient_accumulation_steps n_grpo_steps
    for kv in $config; do
        key="${kv%%=*}"
        value="${kv#*=}"
        case "$key" in
            label) run_label="$value" ;;
            lr) learning_rate="$value" ;;
            loss) loss_type="$value" ;;
            reward) reward_function="$value" ;;
            prompt) prompt_path="$value" ;;
            length_norm) length_normalization="$value" ;;
            std) use_std="$value" ;;
            epochs) epochs_per_rollout_batch="$value" ;;
            train_batch) train_batch_size="$value" ;;
            grad_accum) gradient_accumulation_steps="$value" ;;
            steps) n_grpo_steps="$value" ;;
        esac
    done

    run_log="${LOG_DIR}/${run_label}.log"
    echo "[start] run=${run_label} reward=${reward_function} prompt=${prompt_path} loss=${loss_type} steps=${n_grpo_steps} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"

    std_flag="--use-std-normalization"
    if [[ "${use_std:-1}" == "0" ]]; then
        std_flag="--no-use-std-normalization"
    fi

    cmd=(
        uv run python scripts/grpo_experiment.py
        --learning-rate "$learning_rate"
        --loss-type "$loss_type"
        --reward-function "$reward_function"
        --prompt-path "$prompt_path"
        --length-normalization "$length_normalization"
        --epochs-per-rollout-batch "$epochs_per_rollout_batch"
        --train-batch-size "$train_batch_size"
        --gradient-accumulation-steps "$gradient_accumulation_steps"
        --n-grpo-steps "$n_grpo_steps"
        "$std_flag"
        --output-root "$OUTPUT_ROOT"
        --log-root "$LOG_DIR"
        --wandb-group "$CAMPAIGN_NAME"
    )

    cmd+=("${EXTRA_ARGS[@]}")
    "${cmd[@]}" 2>&1 | tee -a "$run_log"

    echo "[done] run=${run_label} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
done

uv run python scripts/generate_section7_report.py --campaign "$CAMPAIGN_NAME" 2>&1 | tee -a "$ORCHESTRATOR_LOG"

echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"