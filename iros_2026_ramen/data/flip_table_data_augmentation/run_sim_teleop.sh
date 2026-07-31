#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0")

Launch only the Isaac/RoboFinals AVP runtime. Machine-specific values are
provided through environment variables.
EOF
}

[[ $# -eq 0 ]] || { usage >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
CONFIG_PATH="${FLIP_TABLE_TELEOP_CONFIG:-$FEATURE_DIR/teleop/configs/teleop_v1.json}"
XR_REVISION="7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
XR_ROOT="${XR_TELEOP_ROOT:-$HOME/.cache/iros_2026_ramen/xr_teleoperate-$XR_REVISION}"
if [[ -n "${XR_TELEOP_CONDA:-}" ]]; then
  CONDA_EXE="$XR_TELEOP_CONDA"
elif [[ -x "$HOME/miniforge3/condabin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/condabin/conda"
else
  CONDA_EXE="$HOME/miniconda3/condabin/conda"
fi
XR_CERT="${XR_TELEOP_CERT:-$HOME/.config/xr_teleoperate_avp/cert.pem}"
XR_KEY="${XR_TELEOP_KEY:-$HOME/.config/xr_teleoperate_avp/key.pem}"
XR_DISPLAY_MODE="${FLIP_TABLE_TELEOP_XR_DISPLAY_MODE:-ego}"
# The AVP preview stays below the separate offline 30 Hz dataset render to
# leave Isaac physics/rendering headroom.
PREVIEW_HZ="${FLIP_TABLE_TELEOP_PREVIEW_HZ:-24}"
AVP_DESKTOP_IP="${AVP_DESKTOP_IP:-}"

if [[ -z "$AVP_DESKTOP_IP" ]]; then
  mapfile -t desktop_ipv4 < <(
    ip -o -4 addr show up scope global | awk '{sub(/\/.*/, "", $4); print $4}' | sort -u
  )
  if (( ${#desktop_ipv4[@]} != 1 )); then
    echo "ERROR: set AVP_DESKTOP_IP explicitly when the Desktop has multiple IPv4 interfaces." >&2
    printf 'Detected: %s\n' "${desktop_ipv4[*]:-none}" >&2
    exit 2
  fi
  AVP_DESKTOP_IP="${desktop_ipv4[0]}"
fi

[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: missing teleop config: $CONFIG_PATH" >&2; exit 1; }
[[ -x "$CONDA_EXE" ]] || { echo "ERROR: missing conda executable: $CONDA_EXE" >&2; exit 1; }
if [[ -n "${XR_TELEOP_ENV:-}" ]]; then
  XR_ENV="$XR_TELEOP_ENV"
elif "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq tv; then
  XR_ENV=tv
elif "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq xr-teleop; then
  XR_ENV=xr-teleop
else
  echo "ERROR: neither the tv nor xr-teleop conda environment exists" >&2
  exit 1
fi
[[ -d "$XR_ROOT/.git" ]] || {
  echo "ERROR: pinned XR runtime is absent. Run $FEATURE_DIR/setup_teleop_runtime.sh" >&2
  exit 1
}
[[ "$(git -C "$XR_ROOT" rev-parse HEAD)" == "$XR_REVISION" ]] || {
  echo "ERROR: XR_TELEOP_ROOT is not pinned to $XR_REVISION" >&2
  exit 1
}
[[ -r "$XR_CERT" && -r "$XR_KEY" ]] || {
  echo "ERROR: AVP TLS certificate/key are missing. Run inference/desktop/xr/generate_avp_tls.sh" >&2
  exit 1
}
[[ "$AVP_DESKTOP_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || {
  echo "ERROR: AVP_DESKTOP_IP must be an IPv4 address" >&2
  exit 2
}
case "$XR_DISPLAY_MODE" in
  ego|immersive) ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_XR_DISPLAY_MODE must be ego or immersive" >&2; exit 2 ;;
esac
python3 - "$PREVIEW_HZ" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 5.0 <= value <= 30.0:
    raise SystemExit("ERROR: FLIP_TABLE_TELEOP_PREVIEW_HZ must be in [5,30]")
PY
export FLIP_TABLE_TELEOP_PREVIEW_HZ="$PREVIEW_HZ"
export FLIP_TABLE_TELEOP_XR_DISPLAY_MODE="$XR_DISPLAY_MODE"
openssl x509 -in "$XR_CERT" -noout -checkip "$AVP_DESKTOP_IP" >/dev/null || {
  echo "ERROR: AVP TLS certificate does not include $AVP_DESKTOP_IP." >&2
  echo "Regenerate it with XR_DESKTOP_IP=$AVP_DESKTOP_IP inference/desktop/xr/generate_avp_tls.sh" >&2
  exit 1
}

readarray -t TELEOP_VALUES < <(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
workstation = value["workstation"]
runtime = value["runtime"]
print(workstation["tailscale_host"])
print(workstation["ssh_user"])
print(workstation["remote_repo"])
print(workstation["sim_port"])
print(runtime["robofinals_image"])
print(runtime["robofinals_digest"])
print(value["rates"]["physics_hz"])
PY
)
TAILSCALE_HOST="${FLIP_TABLE_WORKSTATION_HOST:-${TELEOP_VALUES[0]}}"
SSH_USER="${FLIP_TABLE_WORKSTATION_USER:-${TELEOP_VALUES[1]}}"
REMOTE_REPO="${FLIP_TABLE_WORKSTATION_REPO:-${TELEOP_VALUES[2]}}"
REMOTE_PORT="${FLIP_TABLE_TELEOP_REMOTE_PORT:-${TELEOP_VALUES[3]}}"
ROBOFINALS_IMAGE="${TELEOP_VALUES[4]}"
ROBOFINALS_DIGEST="${TELEOP_VALUES[5]}"
SIM_PHYSICS_HZ="${FLIP_TABLE_TELEOP_PHYSICS_HZ:-${TELEOP_VALUES[6]}}"
SIM_RENDER_INTERVAL="${FLIP_TABLE_TELEOP_RENDER_INTERVAL:-}"
SIMPLIFY_WHITE_COLLISION="${FLIP_TABLE_TELEOP_SIMPLIFY_WHITE_COLLISION:-true}"
SUCCESS_CHECK_INTERVAL_STEPS="${FLIP_TABLE_TELEOP_SUCCESS_CHECK_INTERVAL_STEPS:-10}"
LOCAL_PORT="${FLIP_TABLE_TELEOP_LOCAL_PORT:-$REMOTE_PORT}"
DR_PROFILE="${FLIP_TABLE_TELEOP_DR_PROFILE:-full}"
SEED="${FLIP_TABLE_TELEOP_SEED:-42}"
SIM_START_TIMEOUT_S="${FLIP_TABLE_TELEOP_SIM_START_TIMEOUT_S:-300}"
OUTPUT_ROOT="${FLIP_TABLE_TELEOP_OUTPUT_ROOT:-$ROOT_DIR/outputs/flip_table_teleop/raw}"
FOOT_PEDAL_CONFIG="${FLIP_TABLE_TELEOP_FOOT_PEDAL_CONFIG:-$HOME/.config/iros_2026_ramen/avp_footswitch.json}"
FOOT_PEDAL_ENABLED="${FLIP_TABLE_TELEOP_FOOT_PEDAL_ENABLED:-}"
SIM_OWNER="${FLIP_TABLE_TELEOP_SIM_OWNER:-one-shot}"

case "$DR_PROFILE" in
  mild) DR_LEVEL="0.25" ;;
  medium) DR_LEVEL="0.60" ;;
  full) DR_LEVEL="1.0" ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_DR_PROFILE must be mild, medium, or full" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: FLIP_TABLE_TELEOP_SEED must be non-negative" >&2; exit 2; }
[[ "$LOCAL_PORT" =~ ^[0-9]+$ && "$REMOTE_PORT" =~ ^[0-9]+$ ]] || {
  echo "ERROR: teleop ports must be integers" >&2
  exit 2
}
[[ "$SIM_START_TIMEOUT_S" =~ ^[0-9]+$ && "$SIM_START_TIMEOUT_S" -ge 30 ]] || {
  echo "ERROR: FLIP_TABLE_TELEOP_SIM_START_TIMEOUT_S must be an integer of at least 30" >&2
  exit 2
}
[[ "$SIM_PHYSICS_HZ" == 200 ]] || {
  echo "ERROR: balanced_wbc requires FLIP_TABLE_TELEOP_PHYSICS_HZ=200" >&2
  exit 2
}
if [[ -z "$SIM_RENDER_INTERVAL" ]]; then
  SIM_RENDER_INTERVAL=$((SIM_PHYSICS_HZ / 30))
fi
[[ "$SIM_RENDER_INTERVAL" =~ ^[1-9][0-9]*$ ]] \
  && (( SIM_RENDER_INTERVAL <= SIM_PHYSICS_HZ / 10 )) || {
  echo "ERROR: FLIP_TABLE_TELEOP_RENDER_INTERVAL must preserve at least 10 Hz" >&2
  exit 2
}
[[ "$SUCCESS_CHECK_INTERVAL_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: FLIP_TABLE_TELEOP_SUCCESS_CHECK_INTERVAL_STEPS must be a positive integer" >&2
  exit 2
}
case "$SIMPLIFY_WHITE_COLLISION" in
  true|TRUE|1|yes|YES) SIMPLIFY_WHITE_COLLISION=true ;;
  false|FALSE|0|no|NO) SIMPLIFY_WHITE_COLLISION=false ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_SIMPLIFY_WHITE_COLLISION must be boolean" >&2; exit 2 ;;
esac
case "$SIM_OWNER" in
  one-shot|persistent) ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_SIM_OWNER must be one-shot or persistent" >&2; exit 2 ;;
esac
export XR_TELEOP_CERT="$XR_CERT" XR_TELEOP_KEY="$XR_KEY"
if ss -ltn | grep -q ':8012\b'; then
  echo "ERROR: TCP/8012 is already occupied by another AVP session" >&2
  exit 1
fi
PYTHONPATH_VALUE="$ROOT_DIR:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src"
SESSION=(
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV"
  env "PYTHONPATH=$PYTHONPATH_VALUE"
  python -m data.flip_table_data_augmentation.teleop.sim.runner
  --config "$CONFIG_PATH" --xr-root "$XR_ROOT" --output-root "$OUTPUT_ROOT"
  --dr-profile "$DR_PROFILE" --seed "$SEED"
)
TRANSPORT_PROBE=false
case "${FLIP_TABLE_TELEOP_TRANSPORT_PROBE:-false}" in
  true|TRUE|1|yes|YES) TRANSPORT_PROBE=true; SESSION+=(--transport-probe) ;;
  false|FALSE|0|no|NO|"") ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_TRANSPORT_PROBE must be boolean" >&2; exit 2 ;;
esac
CONTROL_PROBE=false
case "${FLIP_TABLE_TELEOP_CONTROL_PROBE:-false}" in
  true|TRUE|1|yes|YES) CONTROL_PROBE=true; SESSION+=(--control-probe) ;;
  false|FALSE|0|no|NO|"") ;;
  *) echo "ERROR: FLIP_TABLE_TELEOP_CONTROL_PROBE must be boolean" >&2; exit 2 ;;
esac
if [[ "$TRANSPORT_PROBE" == true && "$CONTROL_PROBE" == true ]]; then
  echo "ERROR: transport and control probes are mutually exclusive" >&2
  exit 2
fi

LOCAL_RUNTIME_OUTPUT="${FLIP_TABLE_TELEOP_RUNTIME_OUTPUT:-$ROOT_DIR/outputs/flip_table_teleop/runtime/$(date +%Y%m%d_%H%M%S)_${DR_PROFILE}_${SEED}}"
LOCAL_RUNTIME_OUTPUT="$(realpath -m "$LOCAL_RUNTIME_OUTPUT")"
RUN_OUTPUT_ID="$(basename "$LOCAL_RUNTIME_OUTPUT")"
[[ "$RUN_OUTPUT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "ERROR: FLIP_TABLE_TELEOP_RUNTIME_OUTPUT must end in a safe run directory name" >&2
  exit 2
}
mkdir -p "$LOCAL_RUNTIME_OUTPUT"
SESSION_REPORT="$LOCAL_RUNTIME_OUTPUT/operator_session_report.json"

# Simulation pedals are opt-in so unattended probes are not coupled to USB.
FOOT_PEDAL_ENABLED="${FOOT_PEDAL_ENABLED:-false}"
case "$FOOT_PEDAL_ENABLED" in
  true|TRUE|1|yes|YES) FOOT_PEDAL_ENABLED=true ;;
  false|FALSE|0|no|NO) FOOT_PEDAL_ENABLED=false ;;
  *)
    echo "ERROR: FLIP_TABLE_TELEOP_FOOT_PEDAL_ENABLED must be boolean" >&2
    exit 2
    ;;
esac
if [[ "$FOOT_PEDAL_ENABLED" == true ]]; then
  [[ -r "$FOOT_PEDAL_CONFIG" ]] || {
    echo "ERROR: calibrated foot-pedal mapping is missing: $FOOT_PEDAL_CONFIG" >&2
    echo "Run: python -m data.flip_table_data_augmentation.teleop.pedal_setup" >&2
    exit 1
  }
  SESSION+=(--foot-pedal-config "$FOOT_PEDAL_CONFIG")
fi

SIM_EXECUTION="${FLIP_TABLE_SIM_EXECUTION:-auto}"
case "$SIM_EXECUTION" in
  auto)
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ "$GPU_NAME" == *"RTX 5090"* ]] && docker info >/dev/null 2>&1; then
      SIM_EXECUTION=local
    else
      SIM_EXECUTION=remote
    fi
    ;;
  local|remote) ;;
  *) echo "ERROR: FLIP_TABLE_SIM_EXECUTION must be auto, local, or remote" >&2; exit 2 ;;
esac

if [[ "$SIM_OWNER" == persistent && "$SIM_EXECUTION" != local ]]; then
  echo "ERROR: FLIP_TABLE_TELEOP_SIM_OWNER=persistent requires FLIP_TABLE_SIM_EXECUTION=local" >&2
  exit 2
fi

SSH_TARGET="$SSH_USER@$TAILSCALE_HOST"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
if [[ "$SIM_EXECUTION" == remote ]]; then
  if ! tailscale status --json >/dev/null 2>&1; then
    echo "ERROR: Tailscale is not connected" >&2
    exit 1
  fi
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" true
fi

REMOTE_PID=""
TUNNEL_PID=""
REMOTE_STAGE=""
REMOTE_OUTPUT=""
REMOTE_CONTAINER=""
SIM_TARGET=""
PERSISTENT_EVAL_SCRIPT="$ROOT_DIR/evaluate/flip_table_simulation/persistent_eval.sh"
PERSISTENT_EVAL_ROOT="${FLIP_TABLE_PERSISTENT_EVAL_ROOT:-$ROOT_DIR/outputs/flip_table_real_to_sim}"
PERSISTENT_JOB_ID=""
PERSISTENT_JOB_OUTPUT="$PERSISTENT_EVAL_ROOT/avp_teleop/$RUN_OUTPUT_ID"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$REMOTE_PID" ]]; then
    if [[ "$SIM_EXECUTION" == local ]]; then
      bash "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
        stop "$REMOTE_PID" "$REMOTE_STAGE" "$REMOTE_CONTAINER" || true
    else
      ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" bash \
        "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
        stop "$REMOTE_PID" "$REMOTE_STAGE" "$REMOTE_CONTAINER" || true
    fi
  fi
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  mkdir -p "$LOCAL_RUNTIME_OUTPUT"
  if [[ "$SIM_OWNER" == one-shot ]]; then
    if [[ "$SIM_EXECUTION" == local ]]; then
      rsync -a "$REMOTE_OUTPUT/" "$LOCAL_RUNTIME_OUTPUT/" 2>/dev/null || true
    else
      rsync -az -e "ssh ${SSH_OPTIONS[*]}" "$SSH_TARGET:$REMOTE_OUTPUT/" "$LOCAL_RUNTIME_OUTPUT/" 2>/dev/null || true
    fi
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

RSYNC_EXCLUDES=(
  --exclude='.git/' --exclude='.pytest_cache/' --exclude='outputs/' \
  --exclude='runtime_output/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.venv/' \
  --exclude='.venv_lerobot060/' --exclude='.issue70_teleop_stage/' \
  --exclude='data/flip_table_data_augmentation/outputs/' \
  --exclude='model/subtask_policy_training/.venv/' \
  --exclude='model/subtask_policy_training/.venv_lerobot060/' \
  --exclude='model/subtask_policy_training/logs/' \
  --exclude='model/subtask_policy_training/outputs/'
)
if [[ "$SIM_OWNER" == persistent ]]; then
  [[ -x "$PERSISTENT_EVAL_SCRIPT" ]] || {
    echo "ERROR: persistent Isaac wrapper is missing: $PERSISTENT_EVAL_SCRIPT" >&2
    exit 1
  }
  PERSISTENT_SUBMISSION="$(FLIP_TABLE_TELEOP_REMOTE_PORT="$REMOTE_PORT" \
    FLIP_TABLE_TELEOP_SEED="$SEED" \
    FLIP_TABLE_TELEOP_DR_PROFILE="$DR_PROFILE" \
    FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ="${FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ:-5}" \
    FLIP_TABLE_TELEOP_PREVIEW_HZ="$PREVIEW_HZ" \
    bash "$PERSISTENT_EVAL_SCRIPT" teleop --output-dir "$PERSISTENT_JOB_OUTPUT")"
  printf '%s\n' "$PERSISTENT_SUBMISSION"
  PERSISTENT_JOB_ID="$(python3 - "$PERSISTENT_SUBMISSION" <<'PY'
import json
import sys

text = sys.argv[1]
decoder = json.JSONDecoder()
objects = []
for index, char in enumerate(text):
    if char != "{":
        continue
    try:
        value, _end = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and isinstance(value.get("job_id"), str):
        objects.append(value)
if not objects:
    raise SystemExit("persistent teleop submission did not return a job_id")
print(objects[-1]["job_id"])
PY
  )"
  echo "Queued AVP job $PERSISTENT_JOB_ID on the existing Isaac worker."
else
  # One-shot simulation owns an rsync staging tree and a dedicated container.
  # Persistent mode uses the already running worker directly and must never
  # touch this tree: release cleanup may intentionally remove it.
  CONFIG_DIGEST="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"
  if [[ "$SIM_EXECUTION" == local ]]; then
    REMOTE_STAGE="${FLIP_TABLE_REMOTE_STAGE:-$ROOT_DIR/.issue70_teleop_stage/$CONFIG_DIGEST}"
    SIM_TARGET="this RTX 5090 workstation"
    mkdir -p "$REMOTE_STAGE"
    rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$ROOT_DIR/" "$REMOTE_STAGE/"
  else
    REMOTE_STAGE="${FLIP_TABLE_REMOTE_STAGE:-${REMOTE_REPO%/}/.issue70_teleop_stage/$CONFIG_DIGEST}"
    SIM_TARGET="$SSH_TARGET"
    ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" mkdir -p "$REMOTE_STAGE"
    rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
      -e "ssh ${SSH_OPTIONS[*]}" "$ROOT_DIR/" "$SSH_TARGET:$REMOTE_STAGE/"
  fi
  # Docker writes its output as root. Keep it outside the rsync-managed source
  # tree and isolate each launch so a previous video can never block a new sync.
  REMOTE_OUTPUT="$REMOTE_STAGE/runtime_output/$RUN_OUTPUT_ID"
  REMOTE_CONTAINER="iros-issue70-teleop-${REMOTE_PORT}-${SEED}"
  LAUNCH_ARGS=(
    "$REMOTE_STAGE" "$REMOTE_OUTPUT" "$REMOTE_PORT" "$DR_LEVEL" "$SEED" \
    "$ROBOFINALS_IMAGE" "$ROBOFINALS_DIGEST" "$REMOTE_CONTAINER" \
    "$SIM_PHYSICS_HZ" "$SIM_RENDER_INTERVAL" "$SIMPLIFY_WHITE_COLLISION" \
    "$SUCCESS_CHECK_INTERVAL_STEPS"
  )
  if [[ "$SIM_EXECUTION" == local ]]; then
    REMOTE_PID="$(bash \
      "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
      launch "${LAUNCH_ARGS[@]}")"
  else
    REMOTE_PID="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" bash \
      "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
      launch "${LAUNCH_ARGS[@]}")"
  fi
fi

if [[ "$SIM_OWNER" == one-shot && "$SIM_EXECUTION" == remote ]]; then
  ssh "${SSH_OPTIONS[@]}" -N -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$SSH_TARGET" \
    >"$LOCAL_RUNTIME_OUTPUT/ssh_tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  sleep 1
  kill -0 "$TUNNEL_PID"
fi

if [[ "$SIM_OWNER" == persistent ]]; then
  echo "Waiting for queued AVP job to expose the existing worker camera bridge..."
else
  echo "Simulator PID $REMOTE_PID started on $SIM_TARGET (execution=$SIM_EXECUTION)."
  echo "Waiting for the simulator camera bridge (cold Isaac Sim startup may take several minutes)..."
fi
SIM_WAIT_STARTED=$SECONDS
SIM_WAIT_NEXT_REPORT=15
while true; do
  if [[ "$SIM_OWNER" == persistent ]]; then
    if ss -H -ltn "sport = :$REMOTE_PORT" | grep -q .; then
      SIM_STATE=ready
    elif [[ -f "$PERSISTENT_EVAL_ROOT/persistent_jobs/failed/$PERSISTENT_JOB_ID.job.json" ]]; then
      echo "ERROR: queued AVP job failed before opening the bridge:" >&2
      cat "$PERSISTENT_EVAL_ROOT/persistent_jobs/failed/$PERSISTENT_JOB_ID.job.json" >&2
      exit 1
    elif ! bash "$PERSISTENT_EVAL_SCRIPT" status >/dev/null 2>&1; then
      SIM_STATE=failed
    else
      SIM_STATE=waiting
    fi
  elif [[ "$SIM_EXECUTION" == local ]]; then
    SIM_STATE="$(bash \
      "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
      state "$REMOTE_PID" "$REMOTE_PORT" "$REMOTE_CONTAINER")"
  else
    SIM_STATE="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" bash \
      "$REMOTE_STAGE/data/flip_table_data_augmentation/teleop/simulator_host.sh" \
      state "$REMOTE_PID" "$REMOTE_PORT" "$REMOTE_CONTAINER")"
  fi
  SIM_WAIT_ELAPSED=$((SECONDS - SIM_WAIT_STARTED))
  case "$SIM_STATE" in
    ready) break ;;
    waiting) ;;
    *)
      echo "ERROR: simulator exited before opening the camera bridge." >&2
      if [[ "$SIM_OWNER" == persistent ]]; then
        tail -n 80 "$PERSISTENT_EVAL_ROOT/persistent_worker.log" >&2 || true
      elif [[ "$SIM_EXECUTION" == local ]]; then
        tail -n 80 "$REMOTE_OUTPUT/simulator.log" >&2 || true
      else
        ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" tail -n 80 "$REMOTE_OUTPUT/simulator.log" >&2 || true
      fi
      exit 1
      ;;
  esac
  if (( SIM_WAIT_ELAPSED >= SIM_START_TIMEOUT_S )); then
    echo "ERROR: simulator camera bridge did not start within ${SIM_START_TIMEOUT_S}s." >&2
    if [[ "$SIM_OWNER" == persistent ]]; then
      tail -n 80 "$PERSISTENT_EVAL_ROOT/persistent_worker.log" >&2 || true
    elif [[ "$SIM_EXECUTION" == local ]]; then
      tail -n 80 "$REMOTE_OUTPUT/simulator.log" >&2 || true
    else
      ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" tail -n 80 "$REMOTE_OUTPUT/simulator.log" >&2 || true
    fi
    exit 1
  fi
  if (( SIM_WAIT_ELAPSED >= SIM_WAIT_NEXT_REPORT )); then
    echo "Simulator is still starting (${SIM_WAIT_ELAPSED}s elapsed)."
    SIM_WAIT_NEXT_REPORT=$((SIM_WAIT_NEXT_REPORT + 15))
  fi
  sleep 3
done

echo "Simulator camera bridge is ready after ${SIM_WAIT_ELAPSED}s."
if [[ "$SIM_OWNER" == persistent ]]; then
  echo "AVP exits with q, then the same Isaac worker remains ready for the next queued evaluation."
fi
if [[ "$TRANSPORT_PROBE" == false && "$CONTROL_PROBE" == false ]]; then
  if [[ "$XR_DISPLAY_MODE" == "ego" ]]; then
    echo "AVP uses pass-through with an unmodified head-left/head-right stereo window."
  else
    echo "AVP uses immersive unmodified head-left/head-right stereo."
  fi
  echo "While recording, clean simulator review video saves head stereo, both wrists, and the global view."
  echo "Open on Apple Vision Pro: https://$AVP_DESKTOP_IP:8012/?ws=wss://$AVP_DESKTOP_IP:8012&grid=False"
fi
cd "$XR_ROOT/teleop"
set +e
"${SESSION[@]}" --sim-host 127.0.0.1 --sim-port "$LOCAL_PORT" \
  --session-report "$SESSION_REPORT" \
  2>&1 | tee "$LOCAL_RUNTIME_OUTPUT/operator.log"
SESSION_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$SIM_OWNER" == persistent && -n "$PERSISTENT_JOB_ID" ]]; then
  JOB_DEADLINE=$((SECONDS + 60))
  while (( SECONDS < JOB_DEADLINE )); do
    if [[ -f "$PERSISTENT_EVAL_ROOT/persistent_jobs/completed/$PERSISTENT_JOB_ID.job.json" ]]; then
      echo "AVP job completed; persistent Isaac worker remains running."
      cat "$PERSISTENT_EVAL_ROOT/persistent_jobs/completed/$PERSISTENT_JOB_ID.job.json"
      break
    fi
    if [[ -f "$PERSISTENT_EVAL_ROOT/persistent_jobs/failed/$PERSISTENT_JOB_ID.job.json" ]]; then
      cat "$PERSISTENT_EVAL_ROOT/persistent_jobs/failed/$PERSISTENT_JOB_ID.job.json" >&2
      SESSION_STATUS=1
      break
    fi
    sleep 1
  done
fi
if [[ "$SIM_OWNER" == persistent && "$SESSION_STATUS" -eq 0 ]]; then
  # The interactive process wrote accepted 30 Hz command trajectories under
  # replay_pending/.  Render each one only after `q` releases the AVP job, so
  # four-camera collection cannot contend with the live latest-frame preview.
  AUTO_MATERIALIZE="${FLIP_TABLE_TELEOP_AUTO_MATERIALIZE:-true}"
  case "$AUTO_MATERIALIZE" in
    true|TRUE|1|yes|YES)
      REPLAY_PENDING_ROOT="$(dirname "$OUTPUT_ROOT")/replay_pending"
      shopt -s nullglob
      REPLAY_TRAJECTORIES=("$REPLAY_PENDING_ROOT"/*.json)
      shopt -u nullglob
      for trajectory in "${REPLAY_TRAJECTORIES[@]}"; do
        if ! python3 - "$trajectory" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("collection_disposition") == "pending_offline_render" else 1)
PY
        then
          continue
        fi
        echo "Materializing 30 Hz four-camera dataset from $trajectory ..."
        bash "$PERSISTENT_EVAL_SCRIPT" render-trajectory "$trajectory" "$OUTPUT_ROOT"
      done
      ;;
    false|FALSE|0|no|NO) ;;
    *) echo "ERROR: FLIP_TABLE_TELEOP_AUTO_MATERIALIZE must be boolean" >&2; SESSION_STATUS=2 ;;
  esac
fi
exit "$SESSION_STATUS"
