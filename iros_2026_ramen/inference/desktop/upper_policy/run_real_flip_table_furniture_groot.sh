#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

: "${G1_DDS_INTERFACE:?Set G1_DDS_INTERFACE to the G1-facing NIC}"
: "${G1_IMAGE_SERVER_IP:?Set G1_IMAGE_SERVER_IP to the Orin image-server IP}"

CHECKPOINT="${FLIP_TABLE_GROOT_CHECKPOINT:-$REPO_ROOT/.checkpoints/flip_table_groot_n17_2}"
MODEL_PYTHON="$REPO_ROOT/model/subtask_policy_training/.venv/bin/python"
if [[ ! -x "$MODEL_PYTHON" ]]; then
  echo "ERROR: run model/subtask_policy_training/scripts/setup_env.sh first" >&2
  exit 1
fi

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

if [[ ! -x "$XR_PYTHON" || ! -d "$XR_ROOT/.git" ]]; then
  echo "ERROR: pinned xr_teleoperate runtime is unavailable" >&2
  exit 1
fi
if [[ "$(git -C "$XR_ROOT" rev-parse HEAD)" != "$XR_REVISION" ]]; then
  echo "ERROR: xr_teleoperate revision mismatch in $XR_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/training_manifest.json" ]]; then
  echo "ERROR: finalized Furniture-GR00T checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi

PYTHONPATH="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src" \
  "$XR_PYTHON" inference/desktop/xr/check_g1_regular_mode.py \
  --interface "$G1_DDS_INTERFACE"

export PYTHONPATH="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$XR_PYTHON" -m \
  inference.desktop.upper_policy.run_flip_table_furniture_groot \
  --interface "$G1_DDS_INTERFACE" \
  --image-server-ip "$G1_IMAGE_SERVER_IP" \
  --checkpoint "$CHECKPOINT" \
  --worker-python "$MODEL_PYTHON" \
  "$@"
