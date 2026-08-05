#!/usr/bin/env bash
# Read-only Desktop GUI for manually focusing the G1 stereo head camera.
set -euo pipefail

XR_ENV="${XR_TELEOP_ENV:-xr-teleop}"
CONDA_EXE="${XR_TELEOP_CONDA:-$HOME/miniconda3/condabin/conda}"
XR_REVISION="7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
XR_ROOT="${XR_TELEOP_ROOT:-$HOME/.cache/iros_2026_ramen/xr_teleoperate-$XR_REVISION}"
: "${G1_IMAGE_SERVER_IP:?Set G1_IMAGE_SERVER_IP to the Orin TeleImager IPv4 address}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "ERROR: this tool must run from the graphical Desktop session." >&2
  exit 1
fi
if [[ ! -x "$CONDA_EXE" ]]; then
  echo "ERROR: conda executable is missing: $CONDA_EXE" >&2
  exit 1
fi
if [[ ! -d "$XR_ROOT/.git" ]]; then
  echo "ERROR: pinned XR runtime is absent. Run data/flip_table_data_augmentation/setup_teleop_runtime.sh" >&2
  exit 1
fi
if [[ "$(git -C "$XR_ROOT" rev-parse HEAD)" != "$XR_REVISION" ]]; then
  echo "ERROR: XR runtime is not pinned to $XR_REVISION" >&2
  exit 1
fi
if [[ -n "$(git -C "$XR_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: pinned XR runtime has tracked local changes: $XR_ROOT" >&2
  exit 1
fi

PYTHONPATH_VALUE="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src"

exec env PYTHONPATH="$PYTHONPATH_VALUE${PYTHONPATH:+:$PYTHONPATH}" \
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python \
  "$SCRIPT_DIR/head_camera_focus_assistant.py" \
  --host "$G1_IMAGE_SERVER_IP" "$@"
