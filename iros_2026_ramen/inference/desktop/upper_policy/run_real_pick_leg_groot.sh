#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

: "${G1_DDS_INTERFACE:?Set G1_DDS_INTERFACE to the G1-facing NIC}"
: "${G1_IMAGE_SERVER_IP:?Set G1_IMAGE_SERVER_IP to the Orin image-server IP}"

MODEL_REPO="Team-RAMEN/groot-n1.7-pick-legs-ver2-lora"
MODEL_REVISION="8f875028fffb1b35ebaf6a95d1575ccec86c12a1"
CHECKPOINT="${GROOT_PICK_LEG_CHECKPOINT:-$REPO_ROOT/.checkpoints/groot-n1.7-pick-legs-ver2-lora}"
WORKER_PYTHON="$REPO_ROOT/model/subtask_policy_training/.venv/bin/python"

if [[ ! -x "$WORKER_PYTHON" ]]; then
  echo "ERROR: LeRobot 0.6.0 environment is missing: $WORKER_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "Downloading the pinned GR00T N1.7 checkpoint (~9.5 GB)..." >&2
  "$WORKER_PYTHON" - "$MODEL_REPO" "$MODEL_REVISION" "$CHECKPOINT" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    allow_patterns=[
        "README.md",
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
    ],
)
PY
fi

for required in \
  config.json processor_config.json statistics.json embodiment_id.json \
  model.safetensors.index.json model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors; do
  if [[ ! -s "$CHECKPOINT/$required" ]]; then
    echo "ERROR: incomplete GR00T checkpoint: $CHECKPOINT/$required" >&2
    exit 1
  fi
done

XR_REVISION="$(
  pixi run -e runtime python - <<'PY'
from data.flip_table_data_augmentation.teleop.config import load_teleop_config
print(load_teleop_config().runtime.xr_revision)
PY
)"
XR_ROOT="$HOME/.cache/iros_2026_ramen/xr_teleoperate-$XR_REVISION"
if [[ -n "${XR_TELEOP_CONDA:-}" ]]; then
  CONDA_EXE="$XR_TELEOP_CONDA"
elif [[ -x "$HOME/miniforge3/condabin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/condabin/conda"
else
  CONDA_EXE="$HOME/miniconda3/condabin/conda"
fi
if [[ -n "${XR_TELEOP_ENV:-}" ]]; then
  XR_ENV="$XR_TELEOP_ENV"
elif "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq tv; then
  XR_ENV=tv
else
  XR_ENV=xr-teleop
fi
XR_PYTHON="$("$CONDA_EXE" run -n "$XR_ENV" python -c 'import sys; print(sys.executable)')"
if [[ ! -x "$XR_PYTHON" ]]; then
  echo "ERROR: xr_teleoperate Python is unavailable: $XR_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$XR_ROOT/.git" ]]; then
  echo "ERROR: pinned xr_teleoperate checkout is missing: $XR_ROOT" >&2
  exit 1
fi
if [[ "$(git -C "$XR_ROOT" rev-parse HEAD)" != "$XR_REVISION" ]]; then
  echo "ERROR: xr_teleoperate revision mismatch in $XR_ROOT" >&2
  exit 1
fi

# Read-only FSM check. The inference process sends no command unless --actuate
# is present and the second confirmation check also passes.
PYTHONPATH="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src" \
  "$XR_PYTHON" inference/desktop/xr/check_g1_regular_mode.py \
  --interface "$G1_DDS_INTERFACE"

export PYTHONPATH="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$XR_PYTHON" -m inference.desktop.upper_policy.run_pick_leg_groot \
  --interface "$G1_DDS_INTERFACE" \
  --image-server-ip "$G1_IMAGE_SERVER_IP" \
  --checkpoint "$CHECKPOINT" \
  "$@"
