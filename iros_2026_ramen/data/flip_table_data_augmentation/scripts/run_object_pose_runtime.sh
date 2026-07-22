#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
CONFIG_HOST="${FLIP_TABLE_AUG_CONFIG:-$FEATURE_DIR/configs/pipeline_v1.json}"
IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
RUNTIME_ROOT="${FLIP_TABLE_OBJECT_POSE_RUNTIME:-$HOME/.cache/team-ramen/flip-table-object-pose}"
OUTPUTS_HOST="${FLIP_TABLE_AUG_OUTPUTS:-$FEATURE_DIR/outputs}"
HF_CACHE_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
# The matching compiler is provisioned by setup_object_pose_runtime.sh inside
# the isolated runtime cache, never by changing the host CUDA alternative.
CUDA_HOME_HOST="${CUDA_HOME_HOST:-$RUNTIME_ROOT/cuda-12.8}"
RUNTIME_MODE="${FLIP_TABLE_AUG_RUNTIME_MODE:-auto}"
ROOTFS="${ROBOFINALS_ROOTFS:-/workspace/robofinals_rootfs_v1}"
OCI_LAYOUT="${ROBOFINALS_OCI_LAYOUT:-/workspace/robofinals_oci_v1}"

usage() {
  cat <<'EOF'
Usage:
  run_object_pose_runtime.sh prepare [args]
  run_object_pose_runtime.sh masks [args]
  run_object_pose_runtime.sh track [args]
  run_object_pose_runtime.sh head-cad [args]
  run_object_pose_runtime.sh state-timing [args]
  run_object_pose_runtime.sh handeye [args]
  run_object_pose_runtime.sh handeye-cad [args]
  run_object_pose_runtime.sh handeye-consensus [args]

Examples:
  .../run_object_pose_runtime.sh masks \
    --input-dir /outputs/source/foundationpose-input/episode-000023 \
    --output-dir /outputs/source/foundationpose-masks/episode-000023 --resume
  .../run_object_pose_runtime.sh track \
    --input-dir /outputs/source/foundationpose-input/episode-000023 \
    --mask-dir /outputs/source/foundationpose-masks/episode-000023 \
    --mesh /outputs/source/v1-table-mesh/Table001_assembled_body_frame.obj \
    --initial-root-from-table /outputs/real_to_sim/source_cad_alignment.json \
    --output-dir /outputs/source/foundationpose-tracks/episode-000023 --resume
  .../run_object_pose_runtime.sh head-cad \
    --source-root /root/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/<revision> \
    --episode-index 23 --frames 0 10 20 30 40 50 \
    --urdf /workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf \
    --stereo-calibration /outputs/calibration/head_camera_params.yaml \
    --output-dir /outputs/real_to_sim/source_cad_alignment_0023
  .../run_object_pose_runtime.sh state-timing \
    --source-root /root/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/<revision> \
    --episode-index 23 \
    --urdf /workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf \
    --output /outputs/real_to_sim/state_timing_0023.json
  .../run_object_pose_runtime.sh handeye \
    --input-manifest /outputs/source/foundationpose-input/episode-000023/manifest.json \
    --mask-manifest /outputs/source/foundationpose-masks/episode-000023/manifest.json \
    --source-alignment /outputs/real_to_sim/source_cad_alignment.json \
    --output /outputs/real_to_sim/source/wrist_handeye_proposal.json
  .../run_object_pose_runtime.sh handeye-cad \
    --input-manifest /outputs/source/foundationpose-input/episode-000023/manifest.json \
    --mask-manifest /outputs/source/foundationpose-masks/episode-000023/manifest.json \
    --source-alignment /outputs/real_to_sim/source_cad_alignment.json \
    --mesh /outputs/source/v1-table-mesh/Table001_assembled_body_frame.obj \
    --output-dir /outputs/real_to_sim/source/wrist_handeye_cad
  .../run_object_pose_runtime.sh handeye-consensus \
    --report /outputs/real_to_sim/source/episode-000023-wrist-handeye-cad/wrist_handeye_cad_alignment.json \
    --report /outputs/real_to_sim/source/episode-000024-wrist-handeye-cad/wrist_handeye_cad_alignment.json \
    --output /outputs/real_to_sim/source/wrist_handeye_cad_consensus.json
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
COMMAND="$1"
shift
case "$COMMAND" in
  prepare|masks|track|head-cad|state-timing|handeye|handeye-cad|handeye-consensus) ;;
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

for path in "$CONFIG_HOST" "$RUNTIME_ROOT/runtime-manifest.json" \
  "$RUNTIME_ROOT/compiled-runtime-manifest.json" "$RUNTIME_ROOT/venv/bin/activate"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required object-pose runtime artifact is missing: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$RUNTIME_ROOT/FoundationPose/mycpp/build" ]]; then
  echo "ERROR: compiled FoundationPose mycpp module is missing" >&2
  exit 1
fi
EXPECTED_DIGEST="$(python3 - "$CONFIG_HOST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["container_digest"])
PY
)"
if [[ "$RUNTIME_MODE" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    RUNTIME_MODE="docker"
  else
    RUNTIME_MODE="direct"
  fi
fi
if [[ "$RUNTIME_MODE" != "docker" && "$RUNTIME_MODE" != "direct" ]]; then
  echo "ERROR: FLIP_TABLE_AUG_RUNTIME_MODE must be auto, docker, or direct" >&2
  exit 2
fi

mkdir -p "$OUTPUTS_HOST" "$HF_CACHE_HOST"
if [[ "$RUNTIME_MODE" == "docker" ]]; then
  [[ -d "$CUDA_HOME_HOST" ]] || {
    echo "ERROR: pinned CUDA toolkit is missing: $CUDA_HOME_HOST" >&2
    exit 1
  }
  OBSERVED_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
    echo "ERROR: V1 image digest mismatch: $OBSERVED_DIGEST" >&2
    exit 1
  fi
  network_mode=none
  if [[ "$COMMAND" == "prepare" ]]; then
    network_mode=host
  fi
  offline_arguments=()
  if [[ "$COMMAND" != "prepare" ]]; then
    offline_arguments+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
  fi
  diagnostic_arguments=()
  for name in \
    FLIP_TABLE_FOUNDATIONPOSE_VERBOSE \
    FLIP_TABLE_TEMPORAL_SELECTION_TRACE \
    FLIP_TABLE_TRACK_TRACE; do
    if [[ -v "$name" ]]; then
      diagnostic_arguments+=(-e "$name=${!name}")
    fi
  done
  exec docker run --rm --gpus all --network "$network_mode" --ipc host \
  "${offline_arguments[@]}" \
  "${diagnostic_arguments[@]}" \
  -e CUDA_HOME=/usr/local/cuda-12.8 \
  -v "$ROOT_DIR":/repo:ro \
  -v "$RUNTIME_ROOT":/runtime \
  -v "$OUTPUTS_HOST":/outputs \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -v "$CUDA_HOME_HOST":/usr/local/cuda-12.8:ro \
  "$IMAGE" -lc '
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate robofinals
source /runtime/venv/bin/activate
export PYTHONPATH=/runtime/FoundationPose:/runtime/FoundationPose/mycpp/build:/repo
command_name="$1"
shift
observed_digest="$1"
shift
case "$command_name" in
  prepare)
    exec python /repo/data/flip_table_data_augmentation/scripts/prepare_foundationpose_episode.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --runtime-root /runtime --observed-image-digest "$observed_digest" "$@"
    ;;
  masks)
    exec python /repo/data/flip_table_data_augmentation/scripts/run_grounded_sam2_masks.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --runtime-root /runtime "$@"
    ;;
  track)
    exec python /repo/data/flip_table_data_augmentation/scripts/track_foundationpose_episode.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
      --runtime-root /runtime "$@"
    ;;
  head-cad)
    exec python /repo/evaluate/flip_table_simulation/real_to_sim_calibration/source_cad_alignment.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  state-timing)
    exec python /repo/evaluate/flip_table_simulation/real_to_sim_calibration/state_timing_audit.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  handeye)
    exec python /repo/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_calibration.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  handeye-cad)
    exec python /repo/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_cad_alignment.py \
      --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json "$@"
    ;;
  handeye-consensus)
    exec python /repo/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_consensus.py "$@"
    ;;
esac
' bash "$COMMAND" "$EXPECTED_DIGEST" "$@"
fi

for path in \
  "$OCI_LAYOUT/manifest.json" \
  "$ROOTFS/opt/conda/envs/robofinals/bin/python" \
  "$ROOTFS/workspace/robofinals"; do
  [[ -e "$path" ]] || {
    echo "ERROR: direct V1 runtime is incomplete; run setup_vast.sh first: $path" >&2
    exit 1
  }
done
OBSERVED_DIGEST="sha256:$(sha256sum "$OCI_LAYOUT/manifest.json" | awk '{print $1}')"
if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
  echo "ERROR: extracted V1 OCI manifest digest mismatch: $OBSERVED_DIGEST" >&2
  exit 1
fi
CUDA_HOME_DIRECT="${CUDA_HOME_DIRECT:-/usr/local/cuda}"
[[ -d "$CUDA_HOME_DIRECT" ]] || {
  echo "ERROR: direct CUDA runtime path is missing: $CUDA_HOME_DIRECT" >&2
  exit 1
}

if [[ "$COMMAND" == "prepare" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export CUDA_HOME="$CUDA_HOME_DIRECT"
export PATH="$CUDA_HOME/bin:$ROOTFS/opt/conda/envs/robofinals/bin:$PATH"
export LD_LIBRARY_PATH="$ROOTFS/opt/conda/envs/robofinals/lib:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROBOFINALS_ROOT="$ROOTFS/workspace/robofinals"
export PYTHONPATH="$RUNTIME_ROOT/FoundationPose:$RUNTIME_ROOT/FoundationPose/mycpp/build:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
case "$COMMAND" in
  prepare)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$FEATURE_DIR/scripts/prepare_foundationpose_episode.py" \
      --config "$CONFIG_HOST" --runtime-root "$RUNTIME_ROOT" \
      --observed-image-digest "$OBSERVED_DIGEST" "$@"
    ;;
  masks)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$FEATURE_DIR/scripts/run_grounded_sam2_masks.py" \
      --config "$CONFIG_HOST" --runtime-root "$RUNTIME_ROOT" "$@"
    ;;
  track)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$FEATURE_DIR/scripts/track_foundationpose_episode.py" \
      --config "$CONFIG_HOST" --runtime-root "$RUNTIME_ROOT" "$@"
    ;;
  head-cad)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$ROOT_DIR/evaluate/flip_table_simulation/real_to_sim_calibration/source_cad_alignment.py" \
      --config "$CONFIG_HOST" "$@"
    ;;
  state-timing)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$ROOT_DIR/evaluate/flip_table_simulation/real_to_sim_calibration/state_timing_audit.py" \
      --config "$CONFIG_HOST" "$@"
    ;;
  handeye)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$ROOT_DIR/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_calibration.py" \
      --config "$CONFIG_HOST" "$@"
    ;;
  handeye-cad)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$ROOT_DIR/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_cad_alignment.py" \
      --config "$CONFIG_HOST" "$@"
    ;;
  handeye-consensus)
    exec "$RUNTIME_ROOT/venv/bin/python" \
      "$ROOT_DIR/evaluate/flip_table_simulation/real_to_sim_calibration/wrist_handeye_consensus.py" "$@"
    ;;
esac
