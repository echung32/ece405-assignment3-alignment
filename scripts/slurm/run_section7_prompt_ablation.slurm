#!/bin/bash
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --job-name=s7-prompt
#SBATCH --account=bggw-dtai-gh
#SBATCH --output=/projects/bggw/echung1/ece405-assignment3-alignment/slurm-logs/%x-%A_%a.out
#SBATCH --error=/projects/bggw/echung1/ece405-assignment3-alignment/slurm-logs/%x-%A_%a.err
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest
#SBATCH --array=0-1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SUBMIT_DIR}/scripts/slurm/section7_slurm_common.sh"

echo "job is starting on $(hostname)"

BASE_LOSS_TYPE="${BASE_LOSS_TYPE:-grpo_clip}"
BASE_EPOCHS="${BASE_EPOCHS:-4}"
BASE_TRAIN_BATCH="${BASE_TRAIN_BATCH:-128}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-32}"
BASE_STEPS="${BASE_STEPS:-200}"

RUN_CONFIGS=(
    "label=r1_zero_prompt lr=1e-5 loss=${BASE_LOSS_TYPE} reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_normalize std=0 epochs=${BASE_EPOCHS} train_batch=${BASE_TRAIN_BATCH} grad_accum=${BASE_GRAD_ACCUM} steps=${BASE_STEPS}"
    "label=question_only_prompt lr=1e-5 loss=${BASE_LOSS_TYPE} reward=question_only prompt=cs336_alignment/prompts/question_only.prompt length_norm=masked_normalize std=0 epochs=${BASE_EPOCHS} train_batch=${BASE_TRAIN_BATCH} grad_accum=${BASE_GRAD_ACCUM} steps=${BASE_STEPS}"
)

section7_init_campaign "section7_grpo_prompt_ablation"
section7_log_launch "array_job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-na}} task=${SLURM_ARRAY_TASK_ID:-na} base_loss=${BASE_LOSS_TYPE} base_epochs=${BASE_EPOCHS} base_train_batch=${BASE_TRAIN_BATCH} base_grad_accum=${BASE_GRAD_ACCUM}"
section7_select_config "${#RUN_CONFIGS[@]}"
section7_parse_config
section7_build_extra_args "$@"
section7_run_selected_config
section7_maybe_finalize_campaign "${#RUN_CONFIGS[@]}"