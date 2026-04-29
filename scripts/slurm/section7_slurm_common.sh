#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
MODULESHOME_DEFAULT="/opt/cray/pe/lmod/lmod"
LMOD_INIT_BASH="${MODULESHOME:-$MODULESHOME_DEFAULT}/init/bash"
SECTION7_TORCH_LIB_DIR="${SECTION7_TORCH_LIB_DIR:-${ROOT_DIR}/.venv/lib/python3.12/site-packages/torch/lib}"

section7_require_uv() {
    if [[ -x "$UV_BIN" ]]; then
        return 0
    fi
    echo "uv executable not found at ${UV_BIN}. Set UV_BIN=/absolute/path/to/uv when submitting." >&2
    exit 1
}

section7_prepend_env_path() {
    local var_name="$1"
    local path_value="$2"

    if [[ ! -d "$path_value" ]]; then
        return 0
    fi

    local current_value="${!var_name:-}"
    if [[ -z "$current_value" ]]; then
        printf -v "$var_name" '%s' "$path_value"
    elif [[ ":$current_value:" != *":$path_value:"* ]]; then
        printf -v "$var_name" '%s:%s' "$path_value" "$current_value"
    fi
    export "$var_name"
}

section7_select_host_compiler() {
    local gcc_bin
    local gxx_bin

    gcc_bin="$(command -v gcc || true)"
    gxx_bin="$(command -v g++ || true)"

    if [[ -n "$gcc_bin" && -n "$gxx_bin" ]]; then
        export CC="$gcc_bin"
        export CXX="$gxx_bin"
        export CUDAHOSTCXX="$gxx_bin"
    fi
}

section7_prepare_runtime_env() {
    if [[ -f "$LMOD_INIT_BASH" ]]; then
        # shellcheck disable=SC1090
        source "$LMOD_INIT_BASH"
    fi

    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1 || true
        module load cuda/13 cudatoolkit >/dev/null 2>&1 || true
    fi

    section7_prepend_env_path PATH "$HOME/.local/bin"
    section7_prepend_env_path LD_LIBRARY_PATH "$SECTION7_TORCH_LIB_DIR"
    section7_select_host_compiler
}

section7_require_array_index() {
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "SLURM_ARRAY_TASK_ID must be set" >&2
        exit 1
    fi
}

section7_reject_extra_args() {
    if (( $# == 0 )); then
        return 0
    fi
    echo "Submission-time extra args are not supported for Section 7 Slurm sweeps. Edit RUN_CONFIGS in the Slurm file instead." >&2
    exit 1
}

section7_init_campaign() {
    local campaign_prefix="$1"
    local default_campaign="${campaign_prefix}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
    CAMPAIGN_NAME="${CAMPAIGN_NAME:-$default_campaign}"
    LOG_DIR="${ROOT_DIR}/logs/section7/${CAMPAIGN_NAME}"
    OUTPUT_ROOT="${ROOT_DIR}/data/section7/grpo_experiment/${CAMPAIGN_NAME}"
    ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator.log"
    mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
}

section7_select_config() {
    local total_configs="$1"
    section7_require_array_index
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= total_configs )); then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for ${total_configs} configs" >&2
        exit 1
    fi
    SELECTED_CONFIG="${RUN_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
}

section7_parse_config() {
    unset run_label learning_rate loss_type reward_function prompt_path length_normalization length_normalize_constant use_std
    unset epochs_per_rollout_batch train_batch_size gradient_accumulation_steps n_grpo_steps

    for kv in $SELECTED_CONFIG; do
        key="${kv%%=*}"
        value="${kv#*=}"
        case "$key" in
            label) run_label="$value" ;;
            lr) learning_rate="$value" ;;
            loss) loss_type="$value" ;;
            reward) reward_function="$value" ;;
            prompt) prompt_path="$value" ;;
            length_norm) length_normalization="$value" ;;
            length_const) length_normalize_constant="$value" ;;
            std) use_std="$value" ;;
            epochs) epochs_per_rollout_batch="$value" ;;
            train_batch) train_batch_size="$value" ;;
            grad_accum) gradient_accumulation_steps="$value" ;;
            steps) n_grpo_steps="$value" ;;
        esac
    done
}

section7_log_launch() {
    local metadata="$1"
    echo "[launch] campaign=${CAMPAIGN_NAME} log_dir=${LOG_DIR#$ROOT_DIR/} output_root=${OUTPUT_ROOT#$ROOT_DIR/} ${metadata}" | tee -a "$ORCHESTRATOR_LOG"
}

section7_run_selected_config() {
    local run_log="${LOG_DIR}/${run_label}.log"
    local std_flag="--use-std-normalization"
    local srun_export="ALL"

    section7_require_uv
    section7_prepare_runtime_env

    echo "[start] run=${run_label} loss=${loss_type} reward=${reward_function:-r1_zero} epochs=${epochs_per_rollout_batch} train_batch=${train_batch_size} grad_accum=${gradient_accumulation_steps} steps=${n_grpo_steps} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
    echo "[env] loaded_modules=${LOADEDMODULES:-unset} cc=${CC:-unset} cxx=${CXX:-unset} torch_lib=${SECTION7_TORCH_LIB_DIR} ld_library_path=${LD_LIBRARY_PATH:-unset}" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"

    if [[ "${use_std:-1}" == "0" ]]; then
        std_flag="--no-use-std-normalization"
    fi

    local cmd=(
        "$UV_BIN" run python scripts/grpo_experiment.py
        --learning-rate "$learning_rate"
        --loss-type "$loss_type"
        --epochs-per-rollout-batch "$epochs_per_rollout_batch"
        --train-batch-size "$train_batch_size"
        --gradient-accumulation-steps "$gradient_accumulation_steps"
        --n-grpo-steps "$n_grpo_steps"
        "$std_flag"
        --output-root "$OUTPUT_ROOT"
        --log-root "$LOG_DIR"
        --wandb-group "$CAMPAIGN_NAME"
    )

    if [[ -n "${reward_function:-}" ]]; then
        cmd+=(--reward-function "$reward_function")
    fi
    if [[ -n "${prompt_path:-}" ]]; then
        cmd+=(--prompt-path "$prompt_path")
    fi
    if [[ -n "${length_normalization:-}" ]]; then
        cmd+=(--length-normalization "$length_normalization")
    fi
    if [[ -n "${length_normalize_constant:-}" ]]; then
        cmd+=(--length-normalize-constant "$length_normalize_constant")
    fi

    (
        cd "$ROOT_DIR"
        srun --export="$srun_export" --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-72}" --gpus-per-task=2 "${cmd[@]}"
    ) 2>&1 | tee -a "$run_log"

    echo "[done] run=${run_label} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG" | tee -a "$run_log"
}

section7_maybe_finalize_campaign() {
    local expected_runs="$1"
    local lock_file="${LOG_DIR}/report.lock"
    local done_marker="${LOG_DIR}/report.complete"

    (
        flock -n 9 || exit 0

        if [[ -f "$done_marker" ]]; then
            exit 0
        fi

        local actual_runs
        actual_runs=$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name summary.json | wc -l | tr -d ' ')
        if (( actual_runs < expected_runs )); then
            exit 0
        fi

        echo "[report] campaign=${CAMPAIGN_NAME} actual_runs=${actual_runs} expected_runs=${expected_runs} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"
        (
            cd "$ROOT_DIR"
            section7_require_uv
            "$UV_BIN" run python scripts/generate_section7_report.py --campaign "$CAMPAIGN_NAME"
        ) 2>&1 | tee -a "$ORCHESTRATOR_LOG"
        touch "$done_marker"
        echo "[complete] campaign=${CAMPAIGN_NAME} $(date -Is)" | tee -a "$ORCHESTRATOR_LOG"
    ) 9>"$lock_file"
}