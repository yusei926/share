#!/usr/bin/env bash
set -euo pipefail

work_root="${GROOT_H100_WORK_ROOT:-/dev/shm/iros_2026_ramen_groot_n17}"
persistent_root="${GROOT_PERSISTENT_RESULT_ROOT:-$HOME/.cache/iros_groot_n17_transfer/results}"
runtime_root="$persistent_root/runtime"
state_root="${GROOT_AUTOSTOP_STATE_ROOT:-$persistent_root/autostop}"
checkpoint_repo="${GROOT_BASELINE_BACKUP_REPO_ID:-Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints}"
expected_step="${GROOT_EXPECTED_FINAL_STEP:-20000}"
poll_seconds="${GROOT_AUTOSTOP_POLL_SECONDS:-60}"
ack_grace_seconds="${GROOT_AUTOSTOP_ACK_GRACE_SECONDS:-1800}"

mkdir -p "$state_root"
exec 9>"$state_root/watcher.lock"
if ! flock -n 9; then
  echo "ERROR: another GR00T autostop watcher is already active" >&2
  exit 1
fi

exec >>"$state_root/watcher.log" 2>&1
printf '%s watcher started\n' "$(date --iso-8601=seconds)"
printf '%s\n' "$$" >"$state_root/watcher.pid"
rm -f "$state_root/ready.json" "$state_root/failure.json" "$state_root/shutdown.ack"

fail_without_shutdown() {
  local reason="$1"
  python3 - "$state_root/failure.json" "$reason" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "groot_n17_autostop_failure_v1",
            "timestamp": datetime.now().astimezone().isoformat(),
            "reason": sys.argv[2],
            "server_stopped": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
  printf '%s refusing shutdown: %s\n' "$(date --iso-8601=seconds)" "$reason"
  exit 1
}

pid_file="$runtime_root/runner.pid"
[[ -s "$pid_file" ]] || fail_without_shutdown "runner PID file is missing"
runner_pid="$(<"$pid_file")"
[[ "$runner_pid" =~ ^[0-9]+$ ]] || fail_without_shutdown "runner PID is invalid"
kill -0 "$runner_pid" 2>/dev/null || fail_without_shutdown "runner is not active at watcher startup"

runner_log="$(readlink -f "$runtime_root/latest.log" 2>/dev/null || true)"
[[ -f "$runner_log" ]] || fail_without_shutdown "runner log is missing"

while kill -0 "$runner_pid" 2>/dev/null; do
  latest_metric="$(tr '\r' '\n' <"$runner_log" | grep 'ot_train.py:673 step:' | tail -1 || true)"
  printf '%s runner active %s\n' "$(date --iso-8601=seconds)" "$latest_metric"
  sleep "$poll_seconds"
done
sleep 5

exit_record="$(ls -1t "$persistent_root"/runner_exit_*.txt 2>/dev/null | head -1 || true)"
[[ -f "$exit_record" ]] || fail_without_shutdown "runner exit record is missing"
exit_status="$(sed -n 's/^exit_status=//p' "$exit_record" | tail -1)"
[[ "$exit_status" == "0" ]] || fail_without_shutdown "runner exited with status ${exit_status:-unknown}"

grep -Fq 'Completed baseline training and verified its resumable Hub checkpoint.' "$runner_log" \
  || fail_without_shutdown "baseline completion marker is missing"

receipt="$persistent_root/baseline_checkpoint_backup.json"
[[ -f "$receipt" ]] || fail_without_shutdown "checkpoint verification receipt is missing"
python3 - "$receipt" "$checkpoint_repo" "$expected_step" <<'PY' \
  || fail_without_shutdown "checkpoint receipt validation failed"
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["repo_id"] == sys.argv[2]
assert receipt["checkpoint_step"] == int(sys.argv[3])
assert receipt["checkpoint_tag"] == f"{int(sys.argv[3]):06d}"
assert receipt["private"] is True
assert receipt["resumable"] is True
assert len(receipt["required_files"]) >= 10
PY

venv_python="$work_root/venv/bin/python"
[[ -x "$venv_python" ]] || fail_without_shutdown "training Python environment is missing"
"$venv_python" - "$checkpoint_repo" "$expected_step" <<'PY' \
  || fail_without_shutdown "Hub checkpoint validation failed"
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
name = f"{int(sys.argv[2]):06d}"
required = {
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
}
api = HfApi()
info = api.model_info(repo_id, files_metadata=True)
assert info.private
files = {item.rfilename for item in info.siblings}
prefix = f"checkpoints/{name}/"
assert {prefix + relative for relative in required} <= files
tags = {tag.name for tag in api.list_repo_refs(repo_id).tags}
assert name in tags
PY

gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
[[ -z "$gpu_processes" ]] || fail_without_shutdown "GPU process remains after training: $gpu_processes"

python3 - "$state_root/ready.json" "$runner_log" "$exit_record" "$receipt" "$checkpoint_repo" "$expected_step" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "groot_n17_autostop_ready_v1",
            "ready_at": datetime.now().astimezone().isoformat(),
            "runner_log": sys.argv[2],
            "runner_exit_record": sys.argv[3],
            "checkpoint_receipt": sys.argv[4],
            "checkpoint_repo": sys.argv[5],
            "checkpoint_step": int(sys.argv[6]),
            "hub_verified": True,
            "gpu_processes_remaining": 0,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

printf '%s completion verified; waiting up to %ss for recorder acknowledgement\n' \
  "$(date --iso-8601=seconds)" "$ack_grace_seconds"
deadline="$(( $(date +%s) + ack_grace_seconds ))"
while (( $(date +%s) < deadline )); do
  [[ -f "$state_root/shutdown.ack" ]] && break
  sleep 15
done

printf '%s powering off server\n' "$(date --iso-8601=seconds)"
sync
sudo -n poweroff -f
