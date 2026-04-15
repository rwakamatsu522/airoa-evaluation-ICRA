#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required}"

HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
  ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
fi

exec python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
