#!/usr/bin/env bash
set -euo pipefail

training_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$training_root/scripts/run_h100_flip_table_groot_n17.sh"
work_root="${GROOT_H100_WORK_ROOT:-/dev/shm/iros_2026_ramen_groot_n17}"
persistent_root="${GROOT_PERSISTENT_RESULT_ROOT:-$HOME/.cache/iros_groot_n17_transfer/results}"
runtime_root="$persistent_root/runtime"
pid_file="$runtime_root/runner.pid"
log_link="$runtime_root/latest.log"
action="${1:-start}"

mkdir -p "$runtime_root"

read_pid() {
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

runner_is_active() {
  local pid
  pid="$(read_pid)" || return 1
  kill -0 "$pid" 2>/dev/null
}

case "$action" in
  status)
    if runner_is_active; then
      pid="$(read_pid)"
      echo "active pid=$pid"
      ps -o pid,etime,%cpu,%mem,cmd -p "$pid"
      [[ -L "$log_link" || -f "$log_link" ]] && tail -n 30 "$log_link"
      exit 0
    fi
    echo "inactive"
    [[ -L "$log_link" || -f "$log_link" ]] && tail -n 50 "$log_link"
    exit 1
    ;;
  start)
    ;;
  *)
    echo "usage: $0 [start|status]" >&2
    exit 2
    ;;
esac

if runner_is_active; then
  echo "ERROR: GR00T runner is already active (pid=$(read_pid))" >&2
  exit 1
fi

mkdir -p "$work_root"
if [[ "$(stat -f -c %T "$work_root")" != "tmpfs" ]]; then
  echo "ERROR: H100 work root must reside on tmpfs: $work_root" >&2
  exit 1
fi
tmpfiles_rule="/etc/tmpfiles.d/iros-groot-n17.conf"
expected_rule="x $work_root - - - - -"
if [[ ! -r "$tmpfiles_rule" ]] || ! grep -Fxq "$expected_rule" "$tmpfiles_rule"; then
  echo "ERROR: tmpfiles exclusion is missing: $expected_rule" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="$runtime_root/runner_${timestamp}.log"
ln -sfn "$log_path" "$log_link"
rm -f "$pid_file"

nohup setsid env \
  GROOT_H100_WORK_ROOT="$work_root" \
  GROOT_PERSISTENT_RESULT_ROOT="$persistent_root" \
  bash "$runner" \
  </dev/null >"$log_path" 2>&1 &
pid="$!"
printf '%s\n' "$pid" >"$pid_file"

sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: detached GR00T runner exited during startup" >&2
  tail -n 100 "$log_path" >&2
  exit 1
fi
echo "started pid=$pid"
echo "log=$log_path"
