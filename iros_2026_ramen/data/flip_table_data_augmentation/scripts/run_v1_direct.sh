#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
CONFIG_HOST="${FLIP_TABLE_AUG_CONFIG:-$FEATURE_DIR/configs/pipeline_v1.json}"
ROOTFS="${ROBOFINALS_ROOTFS:-/workspace/robofinals_rootfs_v1}"
OCI_LAYOUT="${ROBOFINALS_OCI_LAYOUT:-/workspace/robofinals_oci_v1}"
ROBOFINALS_ROOT="$ROOTFS/workspace/robofinals"
OFFICIAL_BACKUP_ROOT="$ROOTFS/workspace/robofinalsbak"
ROOM_ASSETS="$ROOT_DIR/evaluate/flip_table_simulation/assets/room"
TASK_OVERLAY="$ROOT_DIR/evaluate/flip_table_simulation/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
RL_OVERLAY="$ROOT_DIR/model/flip_table_reinforcement_learning/container_overlay/robofinals_rl/flip_table"
CAMERA_PATCH="$ROOT_DIR/evaluate/flip_table_simulation/container_overlay/patches/patch_g1_global_camera.py"
RECORDER_API_PATCH="$FEATURE_DIR/scripts/patch_v1_recorder_api.py"
OUTPUTS_HOST="${FLIP_TABLE_AUG_OUTPUTS:-$FEATURE_DIR/outputs}"
HF_CACHE_HOST="${HF_HOME:-/workspace/.hf_home}"
LOCK_FILE="${FLIP_TABLE_AUG_RUNTIME_LOCK:-/workspace/.flip-table-v1-runtime.lock}"

usage() {
  cat <<'EOF'
Usage:
  data/flip_table_data_augmentation/scripts/run_v1_direct.sh COMMAND [args]

Commands:
  verify-runtime, audit-source, audit-source-raw, audit-source-fk,
  audit-source-camera, export-table-mesh, prepare-pose-input, annotate-source,
  merge-source-annotations, export-source, generate, render

This entrypoint runs an extracted, digest-verified RoboFinals-IKEA-V1 rootfs on
a Vast base-image instance where nested Docker is unavailable.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
COMMAND="$1"
shift
case "$COMMAND" in
  verify-runtime|audit-source|audit-source-raw|audit-source-fk|audit-source-camera|export-table-mesh|prepare-pose-input|annotate-source|merge-source-annotations|export-source|generate|render) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown command: $COMMAND" >&2
    exit 2
    ;;
esac
FULL_RUNTIME=false
case "$COMMAND" in
  verify-runtime|generate|render) FULL_RUNTIME=true ;;
esac

if [[ "$COMMAND" == "prepare-pose-input" ]]; then
  exec "$FEATURE_DIR/scripts/run_object_pose_runtime.sh" prepare "$@"
fi

for command_name in python3 flock; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "ERROR: required command is unavailable: $command_name" >&2
    exit 1
  }
done
for path in "$CONFIG_HOST"; do
  [[ -f "$path" ]] || { echo "ERROR: required file is missing: $path" >&2; exit 1; }
done
for path in "$ROBOFINALS_ROOT"; do
  [[ -d "$path" ]] || { echo "ERROR: required directory is missing: $path" >&2; exit 1; }
done
if [[ "$FULL_RUNTIME" == true ]]; then
  for path in "$TASK_OVERLAY" "$CAMERA_PATCH" "$RECORDER_API_PATCH"; do
    [[ -f "$path" ]] || { echo "ERROR: required file is missing: $path" >&2; exit 1; }
  done
  for path in "$RL_OVERLAY" "$ROOM_ASSETS/textures" "$OFFICIAL_BACKUP_ROOT/robofinals"; do
    [[ -d "$path" ]] || { echo "ERROR: required directory is missing: $path" >&2; exit 1; }
  done
fi

EXPECTED_DIGEST="$(python3 - "$CONFIG_HOST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["container_digest"])
PY
)"
OBSERVED_DIGEST="sha256:$(sha256sum "$OCI_LAYOUT/manifest.json" | awk '{print $1}')"
if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
  echo "ERROR: extracted V1 OCI manifest digest mismatch" >&2
  echo "  expected: $EXPECTED_DIGEST" >&2
  echo "  observed: $OBSERVED_DIGEST" >&2
  exit 1
fi

CONDA_ROOT="$ROOTFS/opt/conda"
PYTHON_BIN="$CONDA_ROOT/envs/robofinals/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: V1 Python is unavailable: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$OUTPUTS_HOST" "$HF_CACHE_HOST" "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
flock 9

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all,graphics,display,compute,utility
export ROBOFINALS_IMAGE_DIGEST="$OBSERVED_DIGEST"
export ROBOFINALS_ROOT
export FLIP_TABLE_AUGMENTATION_ROOT="$FEATURE_DIR"
export FLIP_TABLE_AUG_RL_OVERLAY_ROOT="$ROBOFINALS_ROOT/robofinals_rl/flip_table"
export FLIP_TABLE_AUG_ROOM_ASSET_ROOT="$ROOM_ASSETS"
export FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT="$OFFICIAL_BACKUP_ROOT"
export FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES=true
export FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA=true
export FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS=true
export FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL=true
export FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES=true
export HF_HOME="$HF_CACHE_HOST"
export PATH="$CONDA_ROOT/bin:$CONDA_ROOT/envs/robofinals/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:$ROBOFINALS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/flip-table-augmentation-runtime}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if [[ "$FULL_RUNTIME" == true ]]; then
  "$PYTHON_BIN" "$CAMERA_PATCH"
  "$PYTHON_BIN" "$RECORDER_API_PATCH"

  target_task="$ROBOFINALS_ROOT/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
  cp "$TASK_OVERLAY" "$target_task"
  rm -rf "$ROBOFINALS_ROOT/robofinals_rl/flip_table"
  mkdir -p "$ROBOFINALS_ROOT/robofinals_rl"
  cp -a "$RL_OVERLAY" "$ROBOFINALS_ROOT/robofinals_rl/flip_table"
fi

cd "$ROBOFINALS_ROOT"
case "$COMMAND" in
  verify-runtime)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/verify_runtime.py" \
      --config "$CONFIG_HOST" --observed-image-digest "$OBSERVED_DIGEST" "$@"
    ;;
  audit-source)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_source.py" --config "$CONFIG_HOST" "$@"
    ;;
  audit-source-raw)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_raw_source_bindings.py" --config "$CONFIG_HOST" "$@"
    ;;
  audit-source-fk)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_source_fk.py" --config "$CONFIG_HOST" "$@"
    ;;
  audit-source-camera)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_source_camera_projection.py" --config "$CONFIG_HOST" "$@"
    ;;
  export-table-mesh)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/export_v1_assembled_table_mesh.py" "$@"
    ;;
  prepare-pose-input)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/prepare_foundationpose_episode.py" \
      --config "$CONFIG_HOST" --observed-image-digest "$OBSERVED_DIGEST" "$@"
    ;;
  annotate-source)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/build_automatic_source_annotation.py" \
      --config "$CONFIG_HOST" "$@"
    ;;
  merge-source-annotations)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/merge_source_annotations.py" "$@"
    ;;
  export-source)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/export_mimic_source.py" --config "$CONFIG_HOST" "$@"
    ;;
  generate)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/run_mimic_generation.py" --config "$CONFIG_HOST" "$@"
    ;;
  render)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/render_accepted_trajectories.py" \
      --config "$CONFIG_HOST" --room-assets "$ROOM_ASSETS" "$@"
    ;;
esac
