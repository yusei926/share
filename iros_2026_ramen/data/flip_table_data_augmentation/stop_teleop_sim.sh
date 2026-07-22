#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
CONFIG_PATH="${FLIP_TABLE_TELEOP_CONFIG:-$FEATURE_DIR/teleop/configs/teleop_v1.json}"
DR_PROFILE="${FLIP_TABLE_TELEOP_DR_PROFILE:-full}"
SEED="${FLIP_TABLE_TELEOP_SEED:-42}"

[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: missing teleop config: $CONFIG_PATH" >&2; exit 1; }
case "$DR_PROFILE" in
  mild|medium|full) ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_DR_PROFILE must be mild, medium, or full" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: FLIP_TABLE_TELEOP_SEED must be non-negative" >&2; exit 2; }

readarray -t VALUES < <(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
workstation = value["workstation"]
print(workstation["tailscale_host"])
print(workstation["ssh_user"])
print(workstation["remote_repo"])
print(workstation["sim_port"])
PY
)
TAILSCALE_HOST="${FLIP_TABLE_WORKSTATION_HOST:-${VALUES[0]}}"
SSH_USER="${FLIP_TABLE_WORKSTATION_USER:-${VALUES[1]}}"
REMOTE_REPO="${FLIP_TABLE_WORKSTATION_REPO:-${VALUES[2]}}"
REMOTE_PORT="${FLIP_TABLE_TELEOP_REMOTE_PORT:-${VALUES[3]}}"
CONFIG_DIGEST="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"
SIM_EXECUTION="${FLIP_TABLE_SIM_EXECUTION:-remote}"

case "$SIM_EXECUTION" in
  local)
    REMOTE_STAGE="${FLIP_TABLE_REMOTE_STAGE:-$ROOT_DIR/.issue70_teleop_stage/$CONFIG_DIGEST}"
    ;;
  remote)
    REMOTE_STAGE="${FLIP_TABLE_REMOTE_STAGE:-${REMOTE_REPO%/}/.issue70_teleop_stage/$CONFIG_DIGEST}"
    ;;
  *) echo "ERROR: FLIP_TABLE_SIM_EXECUTION must be local or remote" >&2; exit 2 ;;
esac

REMOTE_CONTAINER="iros-issue70-teleop-${REMOTE_PORT}-${CONFIG_DIGEST:0:12}-${DR_PROFILE}-${SEED}"
HOST_SCRIPT="$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh"

if [[ "$SIM_EXECUTION" == local ]]; then
  bash "$HOST_SCRIPT" stop 0 "$REMOTE_STAGE" "$REMOTE_CONTAINER"
else
  SSH_TARGET="$SSH_USER@$TAILSCALE_HOST"
  SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" bash "$HOST_SCRIPT" stop 0 "$REMOTE_STAGE" "$REMOTE_CONTAINER"
fi

echo "Stopped persistent Isaac Sim container: $REMOTE_CONTAINER"
