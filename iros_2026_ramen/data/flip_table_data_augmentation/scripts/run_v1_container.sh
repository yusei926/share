#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
CONFIG_HOST="${FLIP_TABLE_AUG_CONFIG:-$FEATURE_DIR/configs/pipeline_v1.json}"
ROOM_ASSETS="$ROOT_DIR/evaluate/flip_table_simulation/assets/room"
TASK_OVERLAY="$ROOT_DIR/evaluate/flip_table_simulation/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
RL_OVERLAY="$ROOT_DIR/model/flip_table_reinforcement_learning/container_overlay/robofinals_rl/flip_table"
CAMERA_PATCH="$ROOT_DIR/evaluate/flip_table_simulation/container_overlay/patches/patch_g1_global_camera.py"
RECORDER_API_PATCH="$FEATURE_DIR/scripts/patch_v1_recorder_api.py"
OUTPUTS_HOST="${FLIP_TABLE_AUG_OUTPUTS:-$FEATURE_DIR/outputs}"
HF_CACHE_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
OV_CACHE_HOST="${OV_CACHE_DIR:-$HOME/.cache/ov}"
KIT_CACHE_HOST="${KIT_CACHE_DIR:-$HOME/.cache/Kit}"
NVIDIA_CACHE_HOST="${NVIDIA_CACHE_DIR:-$HOME/.cache/NVIDIA}"

usage() {
  cat <<'EOF'
Usage:
  data/flip_table_data_augmentation/scripts/run_v1_container.sh verify-runtime [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh audit-source [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh audit-source-raw [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh audit-source-fk [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh audit-source-camera [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh export-table-mesh [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh prepare-pose-input [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh annotate-source [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh merge-source-annotations [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh export-source [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh generate [args]
  data/flip_table_data_augmentation/scripts/run_v1_container.sh render [args]

Paths passed to generate/render are container paths. The repository is mounted
at /repo and persistent outputs at /outputs.

Examples:
  .../run_v1_container.sh verify-runtime --output /outputs/runtime.json
  .../run_v1_container.sh audit-source --include-videos --output /outputs/source/audit.json
  .../run_v1_container.sh audit-source-raw --output /outputs/source/raw-binding-audit.json
  .../run_v1_container.sh audit-source-fk --output /outputs/source/fk-audit.json
  .../run_v1_container.sh audit-source-camera --episode-index 23 \
    --output-dir /outputs/source/camera-projection-audit-ep23
  .../run_v1_container.sh prepare-pose-input --episode-index 23 \
    --output-dir /outputs/source/foundationpose-input/episode-000023
  .../run_v1_container.sh export-source --source-root /root/.cache/huggingface/... \
    --annotations /outputs/source/annotations.json \
    --fk-audit /outputs/source/fk-audit.json \
    --output /outputs/source/mimic_source.hdf5
  .../run_v1_container.sh generate --input-file /outputs/source/mimic_source.hdf5 \
    --output-file /outputs/accepted/accepted.hdf5 --ledger-root /outputs/ledger \
    --runtime-manifest /outputs/runtime.json --run-manifest /outputs/runs/smoke.json \
    --run-id smoke --num-trials 1 --headless
  .../run_v1_container.sh render --input-file /outputs/accepted/accepted.hdf5 \
    --ledger-root /outputs/ledger --runtime-manifest /outputs/runtime.json \
    --output-root /outputs/rendered --candidate-limit 1 --headless
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
    usage >&2
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

for path in "$CONFIG_HOST"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file is missing: $path" >&2
    exit 1
  fi
done
if [[ "$FULL_RUNTIME" == true ]]; then
  for path in "$TASK_OVERLAY" "$CAMERA_PATCH" "$RECORDER_API_PATCH"; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: required file is missing: $path" >&2
      exit 1
    fi
  done
  for path in "$RL_OVERLAY" "$ROOM_ASSETS/textures"; do
    if [[ ! -d "$path" ]]; then
      echo "ERROR: required directory is missing: $path" >&2
      exit 1
    fi
  done
  if [[ ! -f "$ROOM_ASSETS/room_props.usda" ]]; then
    echo "ERROR: room prop asset is missing: $ROOM_ASSETS/room_props.usda" >&2
    exit 1
  fi
fi

EXPECTED_DIGEST="$(${PYTHON_BIN:-python3} - "$CONFIG_HOST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["container_digest"])
PY
)"
OBSERVED_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
  echo "ERROR: V1 image digest mismatch" >&2
  echo "  expected: $EXPECTED_DIGEST" >&2
  echo "  observed: $OBSERVED_DIGEST" >&2
  exit 1
fi

mkdir -p "$OUTPUTS_HOST" "$HF_CACHE_HOST" "$OV_CACHE_HOST" "$KIT_CACHE_HOST" "$NVIDIA_CACHE_HOST"
docker_tty_args=()
if [[ -t 0 ]]; then
  docker_tty_args=(-it)
fi
docker_secret_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_secret_args=(-e HF_TOKEN)
fi
runtime_mount_args=()
if [[ "$FULL_RUNTIME" == true ]]; then
  runtime_mount_args=(
    -v "$TASK_OVERLAY":/workspace/robofinals/robofinals_tasks/local_auto_tasks/assemble_table_task.py:ro
    -v "$RL_OVERLAY":/workspace/robofinals/robofinals_rl/flip_table:ro
    -v "$ROOM_ASSETS":/workspace/flip_table_room_assets:ro
  )
fi

docker run --rm "${docker_tty_args[@]}" \
  "${docker_secret_args[@]}" \
  "${runtime_mount_args[@]}" \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all,graphics,display \
  -e ROBOFINALS_IMAGE_DIGEST="$OBSERVED_DIGEST" \
  -e FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT=/workspace/robofinalsbak \
  -e FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES=true \
  -e FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA=true \
  -e FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS=true \
  -e FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL=true \
  -e FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES=true \
  -v "$ROOT_DIR":/repo:ro \
  -v "$OUTPUTS_HOST":/outputs \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -v "$OV_CACHE_HOST":/root/.cache/ov \
  -v "$KIT_CACHE_HOST":/root/.cache/Kit \
  -v "$NVIDIA_CACHE_HOST":/root/.cache/NVIDIA \
  "$IMAGE" \
  -lc '
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate robofinals
cd /workspace/robofinals
export PYTHONPATH=/repo:/workspace/robofinals
command_name="$1"
shift
case "$command_name" in
  verify-runtime|generate|render)
    python /repo/evaluate/flip_table_simulation/container_overlay/patches/patch_g1_global_camera.py
    python /repo/data/flip_table_data_augmentation/scripts/patch_v1_recorder_api.py
    ;;
esac
case "$command_name" in
  verify-runtime)
    exec python /repo/data/flip_table_data_augmentation/scripts/verify_runtime.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --observed-image-digest "$ROBOFINALS_IMAGE_DIGEST" "$@"
    ;;
  audit-source)
    exec python /repo/data/flip_table_data_augmentation/scripts/audit_source.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  audit-source-raw)
    exec python /repo/data/flip_table_data_augmentation/scripts/audit_raw_source_bindings.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  audit-source-fk)
    exec python /repo/data/flip_table_data_augmentation/scripts/audit_source_fk.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  audit-source-camera)
    exec python /repo/data/flip_table_data_augmentation/scripts/audit_source_camera_projection.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  export-table-mesh)
    exec python /repo/data/flip_table_data_augmentation/scripts/export_v1_assembled_table_mesh.py "$@"
    ;;
  prepare-pose-input)
    exec python /repo/data/flip_table_data_augmentation/scripts/prepare_foundationpose_episode.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --observed-image-digest "$ROBOFINALS_IMAGE_DIGEST" "$@"
    ;;
  annotate-source)
    exec python /repo/data/flip_table_data_augmentation/scripts/build_automatic_source_annotation.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  merge-source-annotations)
    exec python /repo/data/flip_table_data_augmentation/scripts/merge_source_annotations.py "$@"
    ;;
  export-source)
    exec python /repo/data/flip_table_data_augmentation/scripts/export_mimic_source.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  generate)
    exec python /repo/data/flip_table_data_augmentation/scripts/run_mimic_generation.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  render)
    exec python /repo/data/flip_table_data_augmentation/scripts/render_accepted_trajectories.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --room-assets /workspace/flip_table_room_assets "$@"
    ;;
esac
' bash "$COMMAND" "$@"
