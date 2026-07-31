#!/usr/bin/env bash
# Start the camera-only TeleImager compatibility launcher on the Orin host.
# This script never creates a Unitree DDS publisher or sends a robot command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SAFE_LAUNCHER="$SCRIPT_DIR/orin_teleimager_safe_launcher.py"
TELEIMAGER_PYTHON="${XR_TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/teleimager/bin/python}"
CONFIG_PATH="${XR_TELEIMAGER_CONFIG:-/home/unitree/teleimager/cam_config_server.yaml}"
MODE="run"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--check-only | --run]

Run this script on the Orin host.
  --check-only  Verify one native-MJPEG stereo head camera and both D405s.
  --run         Verify the cameras, then start the camera-only image server.

The head camera is discovered from its 1280x480 MJPEG capability and UVC
serial, so /dev/videoN changes are harmless. Wrist cameras use D405 serials
from the configuration. A camera timeout stops the server instead of
publishing stale frames. When installed as avp_teleimager.service, systemd
restarts this camera-only process after two seconds.
EOF
}

case "${1:---run}" in
  --check-only) MODE="check" ;;
  --run) MODE="run" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

[[ -x "$TELEIMAGER_PYTHON" ]] || {
  echo "ERROR: TeleImager Python is missing: $TELEIMAGER_PYTHON" >&2
  exit 1
}
[[ -f "$SAFE_LAUNCHER" ]] || {
  echo "ERROR: safe TeleImager launcher is missing: $SAFE_LAUNCHER" >&2
  exit 1
}
[[ -f "$CONFIG_PATH" ]] || {
  echo "ERROR: camera configuration is missing: $CONFIG_PATH" >&2
  exit 1
}
command -v v4l2-ctl >/dev/null || {
  echo "ERROR: v4l2-ctl is required; install v4l-utils on Orin." >&2
  exit 1
}
"$TELEIMAGER_PYTHON" -c 'import mcap' >/dev/null 2>&1 || {
  echo "ERROR: mcap==1.3.0 is required in the Orin TeleImager environment." >&2
  echo "Run inference/orin/scripts/install_avp_teleimager_service.sh first." >&2
  exit 1
}

# logging_mp uses a helper process that can briefly retain the launcher's
# command line after the camera-owning parent has exited. Process-name matching
# therefore creates a false singleton. The authoritative camera-server
# ownership evidence is its fixed control/WebRTC/ZMQ listen ports.
if ss -ltnH | awk '
  {
    endpoint=$4
    sub(/^.*:/, "", endpoint)
    if (endpoint == "60000" || endpoint == "60001" || endpoint == "60010" ||
        endpoint == "55555" || endpoint == "55556" ||
        endpoint == "55557") {
      found=1
    }
  }
  END { exit(found ? 0 : 1) }
'; then
  echo "ERROR: a TeleImager listen port is already owned; do not start a second camera server." >&2
  ss -ltnp | grep -E ':(60000|60001|60010|55555|55556|55557)\\b' >&2 || true
  exit 1
fi
if pgrep -af '[u]sb_cam_node_exe|[r]ealsense2_camera(_node)?' >/dev/null; then
  echo "ERROR: a ROS camera process already owns a head or wrist camera." >&2
  pgrep -af '[u]sb_cam_node_exe|[r]ealsense2_camera(_node)?' >&2
  exit 1
fi

args=(--config "$CONFIG_PATH")
if [[ -n "${HEAD_CAMERA_SERIAL:-}" ]]; then
  args+=(--head-serial "$HEAD_CAMERA_SERIAL")
fi
if [[ "$MODE" == check ]]; then
  args+=(--check-only)
else
  echo "Starting head-stereo + bilateral D405 image server (no robot-control publisher)."
fi
export PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$TELEIMAGER_PYTHON" "$SAFE_LAUNCHER" "${args[@]}"
