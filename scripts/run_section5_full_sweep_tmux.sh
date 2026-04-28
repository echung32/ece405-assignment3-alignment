#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_NAME="${1:-section5_ei_all_$(date +%Y%m%d-%H%M%S)}"
SESSION_NAME="${2:-${BASE_NAME}}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true

CAMPAIGN_NAME="${BASE_NAME}_sweep"
EXTRA_ARGS=("$@")

COMMAND=(
    "cd '$ROOT_DIR'"
    "bash scripts/run_section5_expert_iteration_sweep.sh '$CAMPAIGN_NAME'"
)

for arg in "${EXTRA_ARGS[@]}"; do
    if [[ -n "$arg" ]]; then
        quoted_arg=$(printf '%q' "$arg")
        COMMAND[1]+=" $quoted_arg"
    fi
done

tmux new-session -d -s "$SESSION_NAME" \
    "${COMMAND[0]} && ${COMMAND[1]}"

printf 'session=%s\n' "$SESSION_NAME"
printf 'campaign=%s\n' "$CAMPAIGN_NAME"