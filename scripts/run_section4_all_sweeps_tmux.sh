#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_NAME="${1:-section4_sft_all_$(date +%Y%m%d-%H%M%S)}"
SESSION_NAME="${2:-${BASE_NAME}}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true

SERIAL_CAMPAIGN="${BASE_NAME}_serial"
HPARAM_CAMPAIGN="${BASE_NAME}_hparam"
EXTRA_ARGS=("$@")

COMMAND=(
    "cd '$ROOT_DIR'"
    "bash scripts/run_section4_serial_sweep.sh '$SERIAL_CAMPAIGN'"
)

for arg in "${EXTRA_ARGS[@]}"; do
    if [[ -n "$arg" ]]; then
        quoted_arg=$(printf '%q' "$arg")
        COMMAND[1]+=" $quoted_arg"
    fi
done

COMMAND+=("bash scripts/run_section4_full_hparam_sweep.sh '$HPARAM_CAMPAIGN'")

for arg in "${EXTRA_ARGS[@]}"; do
    if [[ -n "$arg" ]]; then
        quoted_arg=$(printf '%q' "$arg")
        COMMAND[2]+=" $quoted_arg"
    fi
done

tmux new-session -d -s "$SESSION_NAME" \
    "${COMMAND[0]} && ${COMMAND[1]} && ${COMMAND[2]}"

printf 'session=%s\n' "$SESSION_NAME"
printf 'serial_campaign=%s\n' "$SERIAL_CAMPAIGN"
printf 'hparam_campaign=%s\n' "$HPARAM_CAMPAIGN"