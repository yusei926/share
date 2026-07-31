#!/usr/bin/env bash
# Compatibility entrypoint. Physical actuation has exactly one canonical path.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CANONICAL="$ROOT_DIR/data/flip_table_data_augmentation/run_real_teleop.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--check-only | --run]

Deprecated compatibility wrapper. Both modes delegate to the canonical,
pinned real AVP teleoperation launcher and its complete fail-closed preflight.
EOF
}

case "${1:---check-only}" in
  --check-only) export FLIP_TABLE_REAL_PREFLIGHT_ONLY=1 ;;
  --run) export FLIP_TABLE_REAL_PREFLIGHT_ONLY=0 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

echo "NOTICE: delegating to canonical real teleop launcher: $CANONICAL" >&2
exec "$CANONICAL"
