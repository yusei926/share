#!/usr/bin/env bash
set -euo pipefail

# Keep one Isaac Sim process alive for replay/CV iteration. Each queued job
# still performs a clean environment reset, so this trades only startup time,
# never episode isolation.

FEATURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$FEATURE_DIR/../.." && pwd)"
ACTION="${1:-}"
PERSISTENT_ROOT="$(realpath -m "${FLIP_TABLE_PERSISTENT_EVAL_ROOT:-$ROOT_DIR/outputs/flip_table_real_to_sim}")"
RUNTIME_DIR="$PERSISTENT_ROOT/persistent_jobs"
PID_FILE="$RUNTIME_DIR/worker.pid"
LOG_FILE="$PERSISTENT_ROOT/persistent_worker.log"
EXIT_FILE="$RUNTIME_DIR/last_exit.json"
LIFECYCLE_FILE="$RUNTIME_DIR/last_lifecycle.json"
FOUNDATION_ENV_FILE="${FLIP_TABLE_PERSISTENT_FOUNDATION_ENV_FILE:-$PERSISTENT_ROOT/persistent_foundation.env}"

load_foundation_environment() {
  [[ -f "$FOUNDATION_ENV_FILE" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ ! "$line" =~ ^(FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE|FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE|FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE|FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM)=([^[:space:]]+)$ ]]; then
      echo "ERROR: invalid persistent foundation setting in $FOUNDATION_ENV_FILE" >&2
      return 2
    fi
    export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
  done <"$FOUNDATION_ENV_FILE"
}

usage() {
  echo "Usage: $(basename "$0") start|ensure|status|submit|teleop|render-trajectory|stop|restart|force-stop [options]" >&2
  exit 2
}

worker_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}

wait_until_ready() {
  local deadline=$((SECONDS + ${FLIP_TABLE_PERSISTENT_START_TIMEOUT_S:-420}))
  local state=""
  while (( SECONDS < deadline )); do
    if ! worker_running; then
      echo "ERROR: persistent Isaac worker exited during startup; inspect $LOG_FILE" >&2
      return 1
    fi
    if [[ -f "$RUNTIME_DIR/ready.json" ]]; then
      state="$(python3 - "$RUNTIME_DIR/ready.json" <<'PY'
import json
import sys

try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))
except (OSError, ValueError, TypeError):
    print("")
PY
)"
      if [[ "$state" == "ready" ]]; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "ERROR: persistent Isaac worker did not become ready within ${FLIP_TABLE_PERSISTENT_START_TIMEOUT_S:-420}s; inspect $LOG_FILE" >&2
  return 1
}

remove_stale_state() {
  rm -f "$PID_FILE" "$RUNTIME_DIR/ready.json" "$RUNTIME_DIR/STOP" "$EXIT_FILE"
}

case "$ACTION" in
  start)
    mkdir -p "$RUNTIME_DIR"
    load_foundation_environment
    if worker_running; then
      if [[ -f "$RUNTIME_DIR/ready.json" ]]; then
        echo "Persistent Isaac worker is already running (pid $(<"$PID_FILE"))."
        cat "$RUNTIME_DIR/ready.json"
      else
        echo "Persistent Isaac worker is starting (pid $(<"$PID_FILE"))."
        wait_until_ready
        echo "Persistent Isaac worker is ready. It will remain running until stop is requested."
      fi
      exit 0
    fi
    remove_stale_state
    rm -f "$LIFECYCLE_FILE"
    : >"$LOG_FILE"
    # A remote caller commonly launches this through SSH.  Do not leave the
    # Isaac process in that shell's job group: SSH sends SIGHUP to its children
    # when the command returns, which would silently destroy a just-started
    # persistent worker.
    nohup setsid env \
      FLIP_TABLE_PERSISTENT_EVAL_WORKER=true \
      FLIP_TABLE_SIM_OUTPUT_DIR="$PERSISTENT_ROOT" \
      FLIP_TABLE_POLICY_NAME=RecordedJointTargetPolicy \
      FLIP_TABLE_TEST_NUM=1 \
      FLIP_TABLE_EVAL_MODE=nominal \
      FLIP_TABLE_PERSISTENT_HOST_UID="$(id -u)" \
      FLIP_TABLE_PERSISTENT_HOST_GID="$(id -g)" \
      bash -c '
        set +e
        "$1"
        status=$?
        python3 - "$2" "$status" <<"PY"
import json
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": "team_ramen_persistent_evaluation_exit/v1",
    "exit_code": int(sys.argv[2]),
    "finished_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
temporary.replace(path)
PY
        exit "$status"
      ' bash "$FEATURE_DIR/run_eval.sh" "$EXIT_FILE" \
      </dev/null >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "Started persistent Isaac worker (pid $(<"$PID_FILE"))."
    echo "Log: $LOG_FILE"
    wait_until_ready
    echo "Persistent Isaac worker is ready. It will remain running until stop is requested."
    ;;
  ensure)
    # This is intentionally idempotent so calibration wrappers can reuse an
    # existing Isaac process without needing their caller to manage startup.
    # `start` already validates readiness and refuses to replace a live worker.
    "$0" start
    ;;
  status)
    if worker_running && [[ -f "$RUNTIME_DIR/ready.json" ]]; then
      cat "$RUNTIME_DIR/ready.json"
    elif worker_running; then
      echo "starting (pid $(<"$PID_FILE"))"
    else
      if [[ -f "$EXIT_FILE" ]]; then
        cat "$EXIT_FILE"
      fi
      if [[ -f "$LIFECYCLE_FILE" ]]; then
        cat "$LIFECYCLE_FILE"
      fi
      remove_stale_state
      echo "stopped"
      exit 1
    fi
    ;;
  submit)
    shift
    python3 "$FEATURE_DIR/tools/persistent_eval_client.py" \
      --persistent-root "$PERSISTENT_ROOT" \
      "$@"
    ;;
  teleop)
    # Submit a live AVP job to the already-started Isaac application.  The
    # worker reconstructs only its Gym environment in AVP-direct mode, so a
    # normal replay/calibration job can follow without another Isaac cold
    # start.  The job is intentionally asynchronous: the caller waits for the
    # websocket port before starting the desktop-side AVP process.
    shift
    [[ "${1:-}" == "--output-dir" && -n "${2:-}" ]] || usage
    output_dir="$2"
    shift 2
    [[ $# -eq 0 ]] || usage
    "$0" ensure >/dev/null
    port="${FLIP_TABLE_TELEOP_REMOTE_PORT:-59611}"
    seed="${FLIP_TABLE_TELEOP_SEED:-42}"
    profile="${FLIP_TABLE_TELEOP_DR_PROFILE:-mild}"
    case "$profile" in
      mild) randomization_level=0.25 ;;
      medium) randomization_level=0.55 ;;
      full) randomization_level=1.0 ;;
      *) echo "ERROR: FLIP_TABLE_TELEOP_DR_PROFILE must be mild, medium, or full" >&2; exit 2 ;;
    esac
    python3 "$FEATURE_DIR/tools/persistent_eval_client.py" \
      --persistent-root "$PERSISTENT_ROOT" \
      --output-dir "$output_dir" \
      --policy-name AvpTeleopPolicy \
      --time-out-limit 360000 \
      --seed "$seed" \
      --environment "FLIP_TABLE_TELEOP_PORT=$port" \
      --environment "FLIP_TABLE_TELEOP_PERSISTENT=true" \
      --environment "FLIP_TABLE_TELEOP_PREVIEW_HZ=${FLIP_TABLE_TELEOP_PREVIEW_HZ:-24}" \
      --environment "FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ=${FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ:-5}" \
      --environment "FLIP_TABLE_RL_RANDOMIZATION_LEVEL=$randomization_level" \
      --environment "FLIP_TABLE_SIM_BODY_MODE=balanced_wbc" \
      --environment "FLIP_TABLE_LOCK_LOWER_BODY=false" \
      --environment "FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS=base_,hip,knee,ankle,waist" \
      --environment "FLIP_TABLE_REQUIRE_WAIST_LOCK=false" \
      --environment "FLIP_TABLE_LOCK_ROBOT_ROOT=false" \
      --environment "FLIP_TABLE_FIX_ROOT_LINK=false" \
      --environment "FLIP_TABLE_SIM_PHYSICS_HZ=200" \
      --environment "FLIP_TABLE_SIM_RENDER_INTERVAL=${FLIP_TABLE_SIM_RENDER_INTERVAL:-2}" \
      --environment "FLIP_TABLE_EVAL_MODE=randomized"
    ;;
  render-trajectory)
    # Re-render an accepted 30 Hz command trajectory after the AVP session has
    # ended.  This is deliberately separate from the live preview: rendering
    # four RTX cameras must never add a backlog to hand teleoperation.
    shift
    trajectory="${1:-}"
    output_root="${2:-}"
    [[ -n "$trajectory" && -n "$output_root" && $# -eq 2 ]] || usage
    trajectory="$(realpath -e "$trajectory")"
    output_root="$(realpath -m "$output_root")"
    render_plan="$(python3 - "$trajectory" "$PERSISTENT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

trajectory = Path(sys.argv[1])
root = Path(sys.argv[2])
payload = json.loads(trajectory.read_text(encoding="utf-8"))
if payload.get("schema_version") != "team_ramen_flip_table_offline_replay_trajectory/v2":
    raise SystemExit("unsupported replay trajectory schema")
samples = payload.get("samples")
if not isinstance(samples, list) or len(samples) < 2:
    raise SystemExit("replay trajectory must contain at least two samples")
count = len(samples)
indices = [round(index * 50.0 / 30.0) for index in range(count)]
timeout = indices[-1] + 2
episode = str(payload.get("episode_id", trajectory.stem))
if not episode.replace("_", "").replace("-", "").isalnum():
    raise SystemExit("unsafe replay episode id")
output = root / "offline_replay_render" / f"{episode}_{trajectory.stat().st_mtime_ns}"
print(json.dumps({"count": count, "indices": ",".join(map(str, indices)), "timeout": timeout, "output": str(output)}))
PY
)"
    render_output="$(python3 - "$render_plan" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["output"])
PY
)"
    frame_indices="$(python3 - "$render_plan" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["indices"])
PY
)"
    timeout_steps="$(python3 - "$render_plan" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["timeout"])
PY
)"
    "$0" ensure >/dev/null
    python3 "$FEATURE_DIR/tools/persistent_eval_client.py" \
      --persistent-root "$PERSISTENT_ROOT" \
      --output-dir "$render_output" \
      --policy-name RecordedJointTargetPolicy \
      --time-out-limit "$timeout_steps" \
      --wait \
      --environment "FLIP_TABLE_REPLAY_ACTION_PATH=$trajectory" \
      --environment "FLIP_TABLE_REPLAY_HZ=30" \
      --environment "FLIP_TABLE_SAVE_CAMERA_FRAMES=true" \
      --environment "FLIP_TABLE_CAMERA_FRAME_INDICES=$frame_indices" \
      --environment "FLIP_TABLE_SAVE_CAMERA_NAMES=first_person_camera,head_right_camera,left_hand_camera,right_hand_camera,global_camera" \
      --environment "FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES=true" \
      --environment "FLIP_TABLE_SIM_PHYSICS_HZ=200" \
      --environment "FLIP_TABLE_SIM_RENDER_INTERVAL=2"
    python3 -m data.flip_table_data_augmentation.teleop.materialize_sim_replay_episode \
      --trajectory "$trajectory" \
      --camera-frames "$render_output/test_0/camera_frames" \
      --output-root "$output_root" \
      --success
    ;;
  stop)
    if ! worker_running; then
      echo "Persistent Isaac worker is not running."
      exit 0
    fi
    touch "$RUNTIME_DIR/STOP"
    deadline=$((SECONDS + ${FLIP_TABLE_PERSISTENT_STOP_TIMEOUT_S:-180}))
    while worker_running && (( SECONDS < deadline )); do
      sleep 1
    done
    if worker_running; then
      echo "ERROR: worker did not exit after STOP within timeout; inspect $LOG_FILE" >&2
      exit 1
    fi
    rm -f "$PID_FILE"
    echo "Stopped persistent Isaac worker."
    ;;
  force-stop)
    if worker_running; then
      kill "$(<"$PID_FILE")" 2>/dev/null || true
      deadline=$((SECONDS + ${FLIP_TABLE_PERSISTENT_FORCE_STOP_TIMEOUT_S:-30}))
      while worker_running && (( SECONDS < deadline )); do
        sleep 1
      done
      if worker_running; then
        kill -KILL "$(<"$PID_FILE")" 2>/dev/null || true
      fi
    fi
    remove_stale_state
    echo "Force-stopped persistent Isaac worker."
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  *) usage ;;
esac
