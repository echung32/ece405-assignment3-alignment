#!/bin/bash
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --job-name=s7-noclip
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

BEST_EPOCHS="${BEST_EPOCHS:-4}"
BEST_TRAIN_BATCH="${BEST_TRAIN_BATCH:-128}"
BEST_GRAD_ACCUM="${BEST_GRAD_ACCUM:-32}"
BEST_STEPS="${BEST_STEPS:-200}"

RUN_CONFIGS=(
    "label=grpo_clip_best lr=1e-5 loss=grpo_clip reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_normalize std=0 epochs=${BEST_EPOCHS} train_batch=${BEST_TRAIN_BATCH} grad_accum=${BEST_GRAD_ACCUM} steps=${BEST_STEPS}"
    "label=grpo_no_clip_best lr=1e-5 loss=grpo_no_clip reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_normalize std=0 epochs=${BEST_EPOCHS} train_batch=${BEST_TRAIN_BATCH} grad_accum=${BEST_GRAD_ACCUM} steps=${BEST_STEPS}"
)

section7_init_campaign "section7_grpo_no_clip_ablation"
section7_log_launch "array_job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-na}} task=${SLURM_ARRAY_TASK_ID:-na} best_epochs=${BEST_EPOCHS} best_train_batch=${BEST_TRAIN_BATCH} best_grad_accum=${BEST_GRAD_ACCUM}"
section7_select_config "${#RUN_CONFIGS[@]}"
section7_parse_config
section7_build_extra_args "$@"
section7_run_selected_config
section7_maybe_finalize_campaign "${#RUN_CONFIGS[@]}"