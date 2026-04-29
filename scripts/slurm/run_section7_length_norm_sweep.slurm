#!/bin/bash
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --job-name=s7-lnorm
#SBATCH --account=bggw-dtai-gh
#SBATCH --output=/projects/bggw/echung1/ece405-assignment3-alignment/slurm-logs/%x-%A_%a.out
#SBATCH --error=/projects/bggw/echung1/ece405-assignment3-alignment/slurm-logs/%x-%A_%a.err
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest
#SBATCH --array=0-1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SUBMIT_DIR}/scripts/slurm/section7_slurm_common.sh"
section7_reject_extra_args "$@"

echo "job is starting on $(hostname)"

RUN_CONFIGS=(
    "label=masked_mean lr=1e-5 loss=reinforce_with_baseline reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_mean std=1 epochs=1 train_batch=256 grad_accum=64 steps=50"
    "label=masked_normalize lr=1e-5 loss=reinforce_with_baseline reward=r1_zero prompt=cs336_alignment/prompts/r1_zero.prompt length_norm=masked_normalize length_const=1024 std=1 epochs=1 train_batch=256 grad_accum=64 steps=50"
)

section7_init_campaign "section7_grpo_lengthnorm"
section7_log_launch "array_job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-na}} task=${SLURM_ARRAY_TASK_ID:-na}"
section7_select_config "${#RUN_CONFIGS[@]}"
section7_parse_config
section7_run_selected_config
section7_maybe_finalize_campaign "${#RUN_CONFIGS[@]}"