#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  echo "Usage: $(basename "$0") real|sim"
  echo "Compatibility dispatcher; prefer run_real_teleop.sh or run_sim_teleop.sh."
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "$1" in
  real)
    exec "$ROOT_DIR/data/flip_table_data_augmentation/run_real_teleop.sh"
    ;;
  sim)
    exec "$ROOT_DIR/data/flip_table_data_augmentation/run_sim_teleop.sh"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
