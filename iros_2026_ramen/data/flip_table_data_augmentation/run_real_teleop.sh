#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0")

Launch only the physical G1 + Dex1 AVP runtime. This script never starts or
validates Isaac Sim, Docker, a GPU, domain randomization, or a Sim socket.
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
PREVIEW_HZ="${FLIP_TABLE_TELEOP_PREVIEW_HZ:-30}"
PREFLIGHT_ONLY="${FLIP_TABLE_REAL_PREFLIGHT_ONLY:-0}"
AVP_DESKTOP_IP="${AVP_DESKTOP_IP:-}"
OUTPUT_ROOT="${FLIP_TABLE_TELEOP_OUTPUT_ROOT:-$ROOT_DIR/outputs/flip_table_teleop/raw}"
ORIN_SSH_TARGET="${G1_ORIN_SSH_TARGET:-g1-orin}"
RUNTIME_OUTPUT="${FLIP_TABLE_TELEOP_RUNTIME_OUTPUT:-$ROOT_DIR/outputs/flip_table_teleop/runtime/$(date +%Y%m%d_%H%M%S)_real}"
FOOT_PEDAL_CONFIG="${FLIP_TABLE_TELEOP_FOOT_PEDAL_CONFIG:-$HOME/.config/iros_2026_ramen/avp_footswitch.json}"
: "${G1_DDS_INTERFACE:?Set G1_DDS_INTERFACE to the robot DDS network interface}"
: "${G1_IMAGE_SERVER_IP:?Set G1_IMAGE_SERVER_IP to the Orin TeleImager IPv4 address}"

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
# Resolve the environment interpreter before taking the physical-G1 lock.
# ``conda run`` launches commands through Python ``subprocess`` with
# ``close_fds=True`` and would silently close descriptor 9.  The final runner
# must inherit that exact locked descriptor so the Python-side inode/owner
# validation remains fail-closed.
XR_PYTHON="$("$CONDA_EXE" run -n "$XR_ENV" python -c 'import sys; print(sys.executable)')"
[[ -x "$XR_PYTHON" ]] || {
  echo "ERROR: could not resolve Python for conda environment $XR_ENV" >&2
  exit 1
}
[[ -d "$XR_ROOT/.git" ]] || {
  echo "ERROR: pinned XR runtime is absent. Run $FEATURE_DIR/setup_teleop_runtime.sh" >&2
  exit 1
}
[[ "$(git -C "$XR_ROOT" rev-parse HEAD)" == "$XR_REVISION" ]] || {
  echo "ERROR: XR_TELEOP_ROOT is not pinned to $XR_REVISION" >&2
  exit 1
}
[[ -r "$XR_CERT" && -r "$XR_KEY" ]] || {
  echo "ERROR: AVP TLS certificate/key are missing" >&2
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
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *) echo "ERROR: FLIP_TABLE_REAL_PREFLIGHT_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
python3 - "$PREVIEW_HZ" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 5.0 <= value <= 30.0:
    raise SystemExit("ERROR: FLIP_TABLE_TELEOP_PREVIEW_HZ must be in [5,30]")
PY
openssl x509 -in "$XR_CERT" -noout -checkip "$AVP_DESKTOP_IP" >/dev/null || {
  echo "ERROR: AVP TLS certificate does not include $AVP_DESKTOP_IP" >&2
  exit 1
}
if ss -ltn | grep -q ':8012\b'; then
  echo "ERROR: TCP/8012 is already occupied by another AVP session" >&2
  exit 1
fi

# The final Python process inherits this descriptor. Other physical-G1 entry
# points use the same lock and therefore cannot publish concurrently.
CONTROL_LOCK_PATH="${IROS_G1_CONTROL_LOCK_PATH:-/run/user/$(id -u)/iros_2026_ramen_g1_control.lock}"
mkdir -p "$(dirname "$CONTROL_LOCK_PATH")"
exec 9>"$CONTROL_LOCK_PATH"
if ! flock -n 9; then
  echo "ERROR: another IROS RAMEN physical-G1 controller owns the control lock" >&2
  exit 1
fi
export IROS_G1_CONTROL_LOCK_PATH="$CONTROL_LOCK_PATH"
export IROS_G1_CONTROL_LOCK_FD=9

DDS_INTERFACE="$G1_DDS_INTERFACE"
IMAGE_SERVER_IP="$G1_IMAGE_SERVER_IP"
CHECK_DIR="$ROOT_DIR/inference/desktop/xr"
ip link show dev "$DDS_INTERFACE" >/dev/null
ip -o link show dev "$DDS_INTERFACE" | grep -q 'UP'
ping -c 1 -W 1 "$IMAGE_SERVER_IP" >/dev/null
if ps -eo args= | grep -Eq '[t]eleop_hand_and_arm\.py|[r]un_g1_walk_to_table|[r]un_g1_arm_pose_smoke|[i]nference\.desktop\.entrypoint|[g]1_loco_client_example\.py|[g]1_arm(5|7|_sdk).*example\.py|[g]1_low_level_example\.py|[t]eleop\.real\.runner'; then
  echo "ERROR: another known G1 controller is running" >&2
  exit 1
fi

[[ -r "$FOOT_PEDAL_CONFIG" ]] || {
  echo "ERROR: calibrated foot-pedal mapping is missing: $FOOT_PEDAL_CONFIG" >&2
  echo "Run: pixi run -e runtime python -m data.flip_table_data_augmentation.teleop.pedal_setup" >&2
  exit 1
}
RUNTIME_OUTPUT="$(realpath -m "$RUNTIME_OUTPUT")"
RUN_OUTPUT_ID="$(basename "$RUNTIME_OUTPUT")"
[[ "$RUN_OUTPUT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "ERROR: FLIP_TABLE_TELEOP_RUNTIME_OUTPUT must end in a safe run directory name" >&2
  exit 2
}
mkdir -p "$RUNTIME_OUTPUT"
SESSION_REPORT="$RUNTIME_OUTPUT/operator_session_report.json"
PYTHONPATH_VALUE="$ROOT_DIR:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src"
export XR_TELEOP_CERT="$XR_CERT" XR_TELEOP_KEY="$XR_KEY"
export FLIP_TABLE_TELEOP_PREVIEW_HZ="$PREVIEW_HZ"
export FLIP_TABLE_TELEOP_XR_DISPLAY_MODE="$XR_DISPLAY_MODE"
export G1_ORIN_SSH_TARGET="$ORIN_SSH_TARGET"
RECORDER_CONTROL_HOST="${G1_RECORDER_CONTROL_HOST:-$(
  ssh -G -T "$ORIN_SSH_TARGET" | awk '$1 == "hostname" {print $2; exit}'
)}"
[[ -n "$RECORDER_CONTROL_HOST" ]] || {
  echo "ERROR: could not resolve recorder control host from $ORIN_SSH_TARGET" >&2
  exit 1
}
export G1_RECORDER_CONTROL_HOST="$RECORDER_CONTROL_HOST"

"$XR_PYTHON" -c \
  'from importlib.metadata import version; assert version("mcap") == "1.3.0"' \
  >/dev/null || {
  echo "ERROR: mcap==1.3.0 is missing from the XR environment." >&2
  echo "Run: $FEATURE_DIR/setup_teleop_runtime.sh" >&2
  exit 1
}
ssh -o BatchMode=yes -o ConnectTimeout=3 "$ORIN_SSH_TARGET" true || {
  echo "ERROR: passwordless SSH to the Orin recorder target failed: $ORIN_SSH_TARGET" >&2
  echo "Set G1_ORIN_SSH_TARGET to the working SSH host/alias." >&2
  exit 1
}

# Both checks are read-only. Publishers are created only after these pass and
# the real runner starts; motion remains gated behind the operator's r pedal.
PYTHONPATH="$PYTHONPATH_VALUE" "$XR_PYTHON" \
  "$CHECK_DIR/check_g1_regular_mode.py" --interface "$DDS_INTERFACE"
PYTHONPATH="$PYTHONPATH_VALUE" "$XR_PYTHON" \
  "$CHECK_DIR/check_dex1_state.py" --interface "$DDS_INTERFACE"
PYTHONPATH="$PYTHONPATH_VALUE" "$XR_PYTHON" \
  "$CHECK_DIR/check_avp_collection_stream.py" --host "$IMAGE_SERVER_IP"
PYTHONPATH="$PYTHONPATH_VALUE" "$XR_PYTHON" \
  "$CHECK_DIR/check_lossless_camera_recorder.py" --host "$RECORDER_CONTROL_HOST"

echo "Non-actuating physical-G1 preflight passed. No Unitree command has been sent."
echo "AVP endpoint: https://$AVP_DESKTOP_IP:8012/?ws=wss://$AVP_DESKTOP_IP:8012&grid=False"
echo "Real-session output: $RUNTIME_OUTPUT/operator.log"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  exit 0
fi
cd "$XR_ROOT/teleop"
exec > >(tee "$RUNTIME_OUTPUT/operator.log") 2>&1
exec env "PYTHONPATH=$PYTHONPATH_VALUE" \
  "$XR_PYTHON" -m data.flip_table_data_augmentation.teleop.real.runner \
  --config "$CONFIG_PATH" --xr-root "$XR_ROOT" --output-root "$OUTPUT_ROOT" \
  --foot-pedal-config "$FOOT_PEDAL_CONFIG" \
  --dds-interface "$DDS_INTERFACE" --image-server-ip "$IMAGE_SERVER_IP" \
  --session-report "$SESSION_REPORT"
