#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 0 ]] || {
  echo "Usage: $(basename "$0")" >&2
  exit 2
}

XR_REVISION="4c0afdb81f2c19709f053c9c9890a98aca687b44"
TELEVUER_REVISION="948f65f6852410610483345e69715a0c673a99eb"
XR_SOURCE="${XR_TELEOP_SOURCE:-$HOME/GitHub/unitree/xr_teleoperate}"
XR_ROOT="${XR_TELEOP_ROOT:-$HOME/.cache/iros_2026_ramen/xr_teleoperate-$XR_REVISION}"
if [[ -n "${XR_TELEOP_CONDA:-}" ]]; then
  CONDA_EXE="$XR_TELEOP_CONDA"
elif [[ -x "$HOME/miniforge3/condabin/conda" ]]; then
  CONDA_EXE="$HOME/miniforge3/condabin/conda"
else
  CONDA_EXE="$HOME/miniconda3/condabin/conda"
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "ERROR: conda executable is missing: $CONDA_EXE" >&2
  exit 1
fi
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
if [[ ! -d "$XR_SOURCE/.git" ]]; then
  XR_SOURCE="https://github.com/unitreerobotics/xr_teleoperate.git"
fi

new_clone=false
if [[ ! -d "$XR_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$XR_ROOT")"
  git clone --no-checkout "$XR_SOURCE" "$XR_ROOT"
  new_clone=true
fi
# A --no-checkout clone reports every tracked path as deleted until its first
# checkout. Existing caches must be clean, but that initial empty worktree is
# intentional and must be populated before checking its status.
if [[ "$new_clone" != true ]] && [[ -n "$(git -C "$XR_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: pinned XR runtime has local changes: $XR_ROOT" >&2
  exit 1
fi
git -C "$XR_ROOT" checkout --detach "$XR_REVISION"
git -C "$XR_ROOT" submodule sync --recursive
git -C "$XR_ROOT" submodule update --init --recursive

[[ "$(git -C "$XR_ROOT" rev-parse HEAD)" == "$XR_REVISION" ]] || exit 1
[[ "$(git -C "$XR_ROOT/teleop/televuer" rev-parse HEAD)" == "$TELEVUER_REVISION" ]] || exit 1
if [[ -n "$(git -C "$XR_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: pinned XR runtime is not clean after checkout: $XR_ROOT" >&2
  exit 1
fi

PYTHONPATH="$REPO_ROOT:$XR_ROOT:$XR_ROOT/teleop/televuer/src:$XR_ROOT/teleop/teleimager/src" \
  "$CONDA_EXE" run --no-capture-output -n "$XR_ENV" python - <<'PY'
from data.flip_table_data_augmentation.teleop.upstream_compat import install_logging_mp_compat

install_logging_mp_compat()
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from televuer import TeleVuerWrapper
from teleimager.image_client import ImageClient
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

print("pinned-xr-runtime-imports-ok")
PY

cat <<EOF
Pinned XR runtime is ready:
  $XR_ROOT

Set XR_TELEOP_ROOT only when overriding this default.
EOF
