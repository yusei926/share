#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/evaluate/flip_table_simulation"

IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
OUTPUT_DIR="$(realpath -m "${FLIP_TABLE_SIM_OUTPUT_DIR:-$ROOT_DIR/outputs/flip_table_simulation/eval_result}")"
CONFIG_HOST="${FLIP_TABLE_EVAL_CONFIG:-$FEATURE_DIR/configs/flip_table_eval.yml}"
OVERLAY_TASK="$FEATURE_DIR/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
POLICY_OVERLAY="$FEATURE_DIR/container_overlay/policy/flip_table_eval_policy.py"
CV_POLICY_PACKAGE="$FEATURE_DIR/container_overlay/policy/cv_rule_based"
TELEOP_PACKAGE="$ROOT_DIR/data/flip_table_data_augmentation/teleop"
FLOW_PACKAGE="$ROOT_DIR/model/subtask_policy_training/flow_matching"
RLPD_PACKAGE="$ROOT_DIR/model/flip_table_reinforcement_learning/rlpd"
CAMERA_PATCH="$FEATURE_DIR/container_overlay/patches/patch_g1_global_camera.py"
WBC_CONTINUITY_PATCH="$FEATURE_DIR/container_overlay/patches/patch_g1_wbc_action_continuity.py"
BALANCED_WBC_ACTION="$FEATURE_DIR/container_overlay/mdp/team_ramen_balanced_wbc_action.py"
SCENE_PREPARE_TOOL="$FEATURE_DIR/tools/prepare_assembled_table_scene.py"
IN_PROCESS_EVAL_TOOL="$FEATURE_DIR/tools/run_in_process_eval.py"
PERSISTENT_EVAL_WORKER="$FEATURE_DIR/tools/persistent_eval_worker.py"
ROOM_ASSETS="$FEATURE_DIR/assets/room"
GROOT_RUNTIME_SOURCE="$FEATURE_DIR/groot_runtime"
GROOT_RUNTIME_DIR="${FLIP_TABLE_GROOT_RUNTIME_DIR:-$HOME/.cache/iros_2026_ramen/flip_table_groot_runtime}"
GROOT_SHARED_PACKAGE="$ROOT_DIR/model/subtask_policy_training/gr00t"
FURNITURE_GROOT_PLUGIN="$ROOT_DIR/model/subtask_policy_training/lerobot_policy_furniture_groot"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# RoboFinals uses a process-global IPC endpoint at 127.0.0.1:50000.  A
# second evaluator cannot coexist with it and otherwise fails later, after
# starting a heavyweight Isaac container.  Hold a host-side advisory lock for
# the entire wrapper lifetime so callers fail immediately and never reuse a
# stale server from another evaluation.
IPC_LOCK_PATH="${FLIP_TABLE_IPC_LOCK_PATH:-$ROOT_DIR/outputs/flip_table_simulation/.robofinals_env_server_50000.lock}"
mkdir -p "$(dirname "$IPC_LOCK_PATH")"
exec 9>"$IPC_LOCK_PATH"
if ! flock -n 9; then
  echo "ERROR: another RoboFinals evaluator owns IPC port 50000; wait for it to finish." >&2
  exit 3
fi
REPLAY_ACTION_HOST="${FLIP_TABLE_REPLAY_ACTION_PATH:-}"
REPLAY_ACTION_CONTAINER=""
replay_mount_args=()
if [[ -n "$REPLAY_ACTION_HOST" ]]; then
  if [[ ! -f "$REPLAY_ACTION_HOST" ]]; then
    echo "ERROR: FLIP_TABLE_REPLAY_ACTION_PATH is not a file: $REPLAY_ACTION_HOST" >&2
    exit 1
  fi
  REPLAY_ACTION_HOST="$(realpath "$REPLAY_ACTION_HOST")"
  REPLAY_ACTION_CONTAINER="/workspace/flip_table_replay/replay_actions.json"
  replay_mount_args=(-v "$REPLAY_ACTION_HOST:$REPLAY_ACTION_CONTAINER:ro")
fi

if [[ ! -f "$CONFIG_HOST" ]]; then
  echo "ERROR: evaluation config not found: $CONFIG_HOST" >&2
  exit 1
fi
for required_tool in "$WBC_CONTINUITY_PATCH" "$BALANCED_WBC_ACTION" "$SCENE_PREPARE_TOOL" "$IN_PROCESS_EVAL_TOOL" "$PERSISTENT_EVAL_WORKER"; do
  if [[ ! -f "$required_tool" ]]; then
    echo "ERROR: simulation tool not found: $required_tool" >&2
    exit 1
  fi
done
if [[ ! -f "$ROOM_ASSETS/room_props.usda" || ! -d "$ROOM_ASSETS/textures" ]]; then
  echo "Room randomization assets are incomplete under $ROOM_ASSETS" >&2
  exit 1
fi
for required_package in "$FLOW_PACKAGE" "$RLPD_PACKAGE" "$GROOT_SHARED_PACKAGE" "$FURNITURE_GROOT_PLUGIN"; do
  if [[ ! -d "$required_package" ]]; then
    echo "ERROR: evaluation policy package not found: $required_package" >&2
    exit 1
  fi
done
if [[ ! -d "$CV_POLICY_PACKAGE" ]]; then
  echo "ERROR: CV rule-based package is missing" >&2
  exit 1
fi
if [[ ! -f "$TELEOP_PACKAGE/configs/teleop_v1.json" ]]; then
  echo "ERROR: shared teleoperation package is incomplete: $TELEOP_PACKAGE" >&2
  exit 1
fi
TEST_NUM="${FLIP_TABLE_TEST_NUM:-10}"
if [[ ! "$TEST_NUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FLIP_TABLE_TEST_NUM must be a positive integer" >&2
  exit 2
fi
EVAL_SEED="${FLIP_TABLE_EVAL_SEED:-42}"
if [[ ! "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: FLIP_TABLE_EVAL_SEED must be a non-negative integer" >&2
  exit 2
fi
GROOT_INFERENCE_SEED="${FLIP_TABLE_GROOT_INFERENCE_SEED:-$EVAL_SEED}"
if [[ ! "$GROOT_INFERENCE_SEED" =~ ^[0-9]+$ ]] || (( GROOT_INFERENCE_SEED > 4294967295 )); then
  echo "ERROR: FLIP_TABLE_GROOT_INFERENCE_SEED must fit uint32" >&2
  exit 2
fi
EVAL_MODE="${FLIP_TABLE_EVAL_MODE:-randomized}"
case "$EVAL_MODE" in
  nominal|randomized|unseen_dr) ;;
  *)
    echo "ERROR: FLIP_TABLE_EVAL_MODE must be nominal, randomized, or unseen_dr" >&2
    exit 2
    ;;
esac
GROOT_DR_PROFILE="${FLIP_TABLE_GROOT_DR_PROFILE:-generic_v1}"
if [[ ! "$GROOT_DR_PROFILE" =~ ^[a-z0-9_]+$ ]]; then
  echo "ERROR: FLIP_TABLE_GROOT_DR_PROFILE must be a lowercase profile identifier" >&2
  exit 2
fi
export FLIP_TABLE_GROOT_DR_PROFILE="$GROOT_DR_PROFILE"
if [[ "$EVAL_MODE" == "nominal" ]]; then
  # A nominal evaluation is deliberately deterministic. Do not rely on the
  # curriculum width alone: several DR helpers retain a non-zero minimum range.
  export FLIP_TABLE_RL_RANDOMIZATION_LEVEL=0
  export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS=false
  export FLIP_TABLE_EVAL_RANDOMIZE_MASS=false
  export FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES=false
  export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY=false
  export FLIP_TABLE_RL_ENABLE_SENSOR_NOISE=false
  export FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS=0
  export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS=0
  export FLIP_TABLE_TABLE_LONG_RANGE_M=0
  export FLIP_TABLE_TABLE_DEPTH_RANGE_M=0
  export FLIP_TABLE_TABLE_YAW_RANGE_RAD=0
  export FLIP_TABLE_ROBOT_DISTANCE_RANGE_M=0
  export FLIP_TABLE_ROBOT_LATERAL_RANGE_M=0
  export FLIP_TABLE_ROBOT_YAW_RANGE_RAD=0
  export FLIP_TABLE_JOINT_NOISE_RAD=0
  export FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE=false
  export FLIP_TABLE_DEX1_FINGER_NOISE_M=0
  export FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS=false
  export FLIP_TABLE_RANDOMIZE_ROOM=false
  export FLIP_TABLE_RANDOMIZE_ROOM_PROPS=false
  export FLIP_TABLE_RANDOMIZE_LIGHTING=false
fi
if [[ -z "${FLIP_TABLE_POLICY_NAME:-}" ]]; then
  FLIP_TABLE_POLICY_NAME="$(awk '/^policy_name:/ {gsub(/\"/, "", $2); print $2; exit}' "$CONFIG_HOST")"
  export FLIP_TABLE_POLICY_NAME
fi
FLIP_TABLE_SIM_BODY_MODE="${FLIP_TABLE_SIM_BODY_MODE:-balanced_wbc}"
case "$FLIP_TABLE_SIM_BODY_MODE" in
  balanced_wbc)
    export FLIP_TABLE_LOCK_LOWER_BODY=false
    export FLIP_TABLE_LOCK_ROBOT_ROOT=false
    export FLIP_TABLE_FIX_ROOT_LINK=false
    export FLIP_TABLE_REQUIRE_WAIST_LOCK=false
    export FLIP_TABLE_SIM_PHYSICS_HZ=200
    ;;
  fixed_diagnostic) ;;
  full_body_diagnostic) ;;
  *) echo "ERROR: FLIP_TABLE_SIM_BODY_MODE must be balanced_wbc, fixed_diagnostic, or full_body_diagnostic" >&2; exit 2 ;;
esac
export FLIP_TABLE_SIM_BODY_MODE
CHECKPOINT_HOST="${FLIP_TABLE_POLICY_CHECKPOINT:-}"
checkpoint_mount_args=()
CHECKPOINT_CONTAINER=""
if [[ -n "$CHECKPOINT_HOST" ]]; then
  if [[ ! -d "$CHECKPOINT_HOST" ]]; then
    echo "ERROR: FLIP_TABLE_POLICY_CHECKPOINT is not a host directory: $CHECKPOINT_HOST"
    exit 1
  fi
  CHECKPOINT_HOST="$(realpath "$CHECKPOINT_HOST")"
  CHECKPOINT_CONTAINER="/workspace/flip_table_policy_checkpoint"
  checkpoint_mount_args=(-v "$CHECKPOINT_HOST:$CHECKPOINT_CONTAINER:ro")
fi

if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(NoOpPolicy|ScriptedJointPolicy|RecordedJointTargetPolicy|RecordedFullBodyTargetPolicy|AvpTeleopPolicy|LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy|LeRobotGrootN17Policy|TeleopPerformanceBenchmarkPolicy|Dex1ForceCalibrationPolicy)$ && -z "${FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION+x}" ]]; then
  export FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=true
fi
if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "CvRuleBasedPolicy" ]]; then
  export FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=false
  export FLIP_TABLE_USE_PINK_EEF_ACTION=true
fi
if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy|LeRobotGrootN17Policy)$ && -z "$CHECKPOINT_HOST" ]]; then
  echo "ERROR: ${FLIP_TABLE_POLICY_NAME} requires FLIP_TABLE_POLICY_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$GROOT_RUNTIME_DIR" "$HF_CACHE_DIR"

RUNTIME_CONFIG="$OUTPUT_DIR/flip_table_eval_seed_${EVAL_SEED}_${EVAL_MODE}.yml"
python3 - "$CONFIG_HOST" "$RUNTIME_CONFIG" "$EVAL_SEED" <<'PY'
from pathlib import Path
import sys

source, target, seed = (Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
lines = source.read_text(encoding="utf-8").splitlines()
replaced_top = 0
replaced_env = 0
output = []
for line in lines:
    if line.startswith("seed:"):
        output.append(f"seed: {seed}")
        replaced_top += 1
    elif line.startswith("  seed:"):
        output.append(f"  seed: {seed}")
        replaced_env += 1
    else:
        output.append(line)
if replaced_top != 1 or replaced_env != 1:
    raise SystemExit("evaluation config must contain exactly top-level and env_cfg seed fields")
target.write_text("\n".join(output) + "\n", encoding="utf-8")
PY
CONFIG_HOST="$RUNTIME_CONFIG"

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +SI:localuser:root >/dev/null 2>&1 || true
fi

DOCKER_XAUTH="${DOCKER_XAUTH:-${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}}"
if [[ ! -f "$DOCKER_XAUTH" ]]; then
  echo "ERROR: Xauthority file not found: $DOCKER_XAUTH"
  echo "Set DOCKER_XAUTH=/actual/path/to/Xauthority, as described in docs/simulation.pdf."
  exit 1
fi

docker_tty_args=()
if [[ -t 0 ]]; then
  docker_tty_args=(-it)
fi
docker_name_args=()
if [[ -n "${FLIP_TABLE_DOCKER_CONTAINER_NAME:-}" ]]; then
  if [[ ! "$FLIP_TABLE_DOCKER_CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: FLIP_TABLE_DOCKER_CONTAINER_NAME contains invalid characters" >&2
    exit 2
  fi
  docker_name_args=(--name "$FLIP_TABLE_DOCKER_CONTAINER_NAME")
fi

# Camera frames are written by the policy process inside the container.  Only
# ``OUTPUT_DIR`` is mounted writable, so translate an optional host-relative
# destination into that mount rather than silently writing disposable files to
# the container filesystem.  This is especially important for calibration
# evidence, which must survive container teardown.
CAMERA_FRAME_OUTPUT_CONTAINER=""
if [[ -n "${FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR:-}" ]]; then
  CAMERA_FRAME_OUTPUT_HOST="$(realpath -m "$FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR")"
  if [[ "$CAMERA_FRAME_OUTPUT_HOST" == "$OUTPUT_DIR" ]]; then
    CAMERA_FRAME_OUTPUT_CONTAINER="/workspace/robofinals/eval_result"
  elif [[ "$CAMERA_FRAME_OUTPUT_HOST" == "$OUTPUT_DIR/"* ]]; then
    CAMERA_FRAME_OUTPUT_CONTAINER="/workspace/robofinals/eval_result/${CAMERA_FRAME_OUTPUT_HOST#"$OUTPUT_DIR"/}"
  else
    echo "ERROR: FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR must be inside FLIP_TABLE_SIM_OUTPUT_DIR" >&2
    exit 2
  fi
fi

docker --context default run --rm "${docker_tty_args[@]}" \
  "${docker_name_args[@]}" \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e XAUTHORITY=/tmp/docker.xauth \
  -e FLIP_TABLE_POLICY_CHECKPOINT="$CHECKPOINT_CONTAINER" \
  -e FLIP_TABLE_POLICY_NAME="${FLIP_TABLE_POLICY_NAME:-}" \
  -e FLIP_TABLE_EVAL_SEED="$EVAL_SEED" \
  -e FLIP_TABLE_TELEOP_PORT="${FLIP_TABLE_TELEOP_PORT:-59610}" \
  -e FLIP_TABLE_TELEOP_BIND_HOST="${FLIP_TABLE_TELEOP_BIND_HOST:-0.0.0.0}" \
  -e FLIP_TABLE_TELEOP_PERSISTENT="${FLIP_TABLE_TELEOP_PERSISTENT:-false}" \
  -e FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ="${FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ:-5}" \
  -e FLIP_TABLE_PERSISTENT_EVAL_WORKER="${FLIP_TABLE_PERSISTENT_EVAL_WORKER:-false}" \
  -e FLIP_TABLE_PERSISTENT_HOST_UID="${FLIP_TABLE_PERSISTENT_HOST_UID:-}" \
  -e FLIP_TABLE_PERSISTENT_HOST_GID="${FLIP_TABLE_PERSISTENT_HOST_GID:-}" \
  -e FLIP_TABLE_RL_RANDOMIZATION_LEVEL="${FLIP_TABLE_RL_RANDOMIZATION_LEVEL:-1.0}" \
  -e FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS="${FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS:-true}" \
  -e FLIP_TABLE_RL_RANDOMIZE_MASS=false \
  -e FLIP_TABLE_EVAL_RANDOMIZE_MASS=false \
  -e FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES="${FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES:-true}" \
  -e FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY="${FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY:-true}" \
  -e FLIP_TABLE_RL_ENABLE_SENSOR_NOISE="${FLIP_TABLE_RL_ENABLE_SENSOR_NOISE:-true}" \
  -e FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS="${FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS:-2}" \
  -e FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS="${FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS:-2}" \
  -e FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION="${FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION:-true}" \
  -e FLIP_TABLE_CV_SIM_CONTROL_HZ="${FLIP_TABLE_CV_SIM_CONTROL_HZ:-50}" \
  -e FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE="${FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE:-pinocchio}" \
  -e FLIP_TABLE_CV_MIN_CONFIDENCE="${FLIP_TABLE_CV_MIN_CONFIDENCE:-0.03}" \
  -e FLIP_TABLE_CV_MIN_LEG_CONFIDENCE="${FLIP_TABLE_CV_MIN_LEG_CONFIDENCE:-0.20}" \
  -e FLIP_TABLE_CV_WARMUP_STEPS="${FLIP_TABLE_CV_WARMUP_STEPS:-50}" \
  -e FLIP_TABLE_CV_SETTLED_SELECTION_STEPS="${FLIP_TABLE_CV_SETTLED_SELECTION_STEPS:-20}" \
  -e FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS="${FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS:-10}" \
  -e FLIP_TABLE_CV_REDETECT_ALPHA="${FLIP_TABLE_CV_REDETECT_ALPHA:-0.30}" \
  -e FLIP_TABLE_CV_REDETECT_MAX_TRANSLATION_M="${FLIP_TABLE_CV_REDETECT_MAX_TRANSLATION_M:-0.12}" \
  -e FLIP_TABLE_CV_REDETECT_MAX_YAW_RAD="${FLIP_TABLE_CV_REDETECT_MAX_YAW_RAD:-0.35}" \
  -e FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD="${FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD:--0.017}" \
  -e FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS="${FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS:-8}" \
  -e FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL="${FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL:-2}" \
  -e FLIP_TABLE_WBC_JOINT_CONTINUITY_FILTER="${FLIP_TABLE_WBC_JOINT_CONTINUITY_FILTER:-false}" \
  -e FLIP_TABLE_WBC_MAX_JOINT_SPEED_RAD_S="${FLIP_TABLE_WBC_MAX_JOINT_SPEED_RAD_S:-2.0}" \
  -e FLIP_TABLE_WBC_MAX_JOINT_ACCELERATION_RAD_S2="${FLIP_TABLE_WBC_MAX_JOINT_ACCELERATION_RAD_S2:-8.0}" \
  -e FLIP_TABLE_USE_PINK_EEF_ACTION="${FLIP_TABLE_USE_PINK_EEF_ACTION:-false}" \
  -e FLIP_TABLE_REPLAY_ACTION_PATH="$REPLAY_ACTION_CONTAINER" \
  -e FLIP_TABLE_REPLAY_HOLD_INDEX="${FLIP_TABLE_REPLAY_HOLD_INDEX:-}" \
  -e FLIP_TABLE_REPLAY_HZ="${FLIP_TABLE_REPLAY_HZ:-30}" \
  -e FLIP_TABLE_REPLAY_WARMUP_STEPS="${FLIP_TABLE_REPLAY_WARMUP_STEPS:-0}" \
  -e FLIP_TABLE_INITIAL_UPPER_BODY_STATE="${FLIP_TABLE_INITIAL_UPPER_BODY_STATE:-}" \
  -e FLIP_TABLE_PREPARE_ASSEMBLED_SCENE="${FLIP_TABLE_PREPARE_ASSEMBLED_SCENE:-true}" \
  -e FLIP_TABLE_SIMPLIFY_WHITE_COLLISION="${FLIP_TABLE_SIMPLIFY_WHITE_COLLISION:-false}" \
  -e FLIP_TABLE_BENCHMARK_WARMUP_STEPS="${FLIP_TABLE_BENCHMARK_WARMUP_STEPS:-40}" \
  -e FLIP_TABLE_BENCHMARK_MEASURE_STEPS="${FLIP_TABLE_BENCHMARK_MEASURE_STEPS:-180}" \
  -e FLIP_TABLE_TEST_NUM="$TEST_NUM" \
  -e FLIP_TABLE_EVAL_MODE="$EVAL_MODE" \
  -e FLIP_TABLE_GROOT_DR_PROFILE="$GROOT_DR_PROFILE" \
  -e FLIP_TABLE_TIME_OUT_LIMIT="${FLIP_TABLE_TIME_OUT_LIMIT:-}" \
  -e FLIP_TABLE_TABLE_XY_RANGE_M="${FLIP_TABLE_TABLE_XY_RANGE_M:-}" \
  -e FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL="${FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL:-}" \
  -e FLIP_TABLE_TABLE_YAW_OFFSET_RAD="${FLIP_TABLE_TABLE_YAW_OFFSET_RAD:-0}" \
  -e FLIP_TABLE_TABLE_LONG_RANGE_M="${FLIP_TABLE_TABLE_LONG_RANGE_M:-0.12}" \
  -e FLIP_TABLE_TABLE_DEPTH_RANGE_M="${FLIP_TABLE_TABLE_DEPTH_RANGE_M:-0.035}" \
  -e FLIP_TABLE_TABLE_YAW_RANGE_RAD="${FLIP_TABLE_TABLE_YAW_RANGE_RAD:-3.141592653589793}" \
  -e FLIP_TABLE_WORKBENCH_FRONT_AXIS="${FLIP_TABLE_WORKBENCH_FRONT_AXIS:--y}" \
  -e FLIP_TABLE_ROBOT_DISTANCE_M="${FLIP_TABLE_ROBOT_DISTANCE_M:-0.26}" \
  -e FLIP_TABLE_ROBOT_DISTANCE_RANGE_M="${FLIP_TABLE_ROBOT_DISTANCE_RANGE_M:-0.04}" \
  -e FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M="${FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M:-0.62}" \
  -e FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M="${FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M:-0.20}" \
  -e FLIP_TABLE_ROBOT_LATERAL_RANGE_M="${FLIP_TABLE_ROBOT_LATERAL_RANGE_M:-0.10}" \
  -e FLIP_TABLE_ROBOT_YAW_RANGE_RAD="${FLIP_TABLE_ROBOT_YAW_RANGE_RAD:-0.08}" \
  -e FLIP_TABLE_ROBOT_YAW_OFFSET_RAD="${FLIP_TABLE_ROBOT_YAW_OFFSET_RAD:-0.0}" \
  -e FLIP_TABLE_ROBOT_BASE_HEIGHT_M="${FLIP_TABLE_ROBOT_BASE_HEIGHT_M:-0.78}" \
  -e FLIP_TABLE_ROBOT_ROOT_POS_LOCAL="${FLIP_TABLE_ROBOT_ROOT_POS_LOCAL:-}" \
  -e FLIP_TABLE_ROBOT_ROOT_YAW_RAD="${FLIP_TABLE_ROBOT_ROOT_YAW_RAD:-}" \
  -e FLIP_TABLE_USE_DEFAULT_ROBOT_POSE="${FLIP_TABLE_USE_DEFAULT_ROBOT_POSE:-false}" \
  -e FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS="${FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS:-0}" \
  -e FLIP_TABLE_DEFAULT_ROBOT_FORWARD_CELLS="${FLIP_TABLE_DEFAULT_ROBOT_FORWARD_CELLS:-0}" \
  -e FLIP_TABLE_DEFAULT_ROBOT_YAW_OFFSET_RAD="${FLIP_TABLE_DEFAULT_ROBOT_YAW_OFFSET_RAD:-0.0}" \
  -e FLIP_TABLE_DEBUG_GRID_CELL_M="${FLIP_TABLE_DEBUG_GRID_CELL_M:-0.25}" \
  -e FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA="${FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA:-true}" \
  -e FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION="${FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION:-}" \
  -e FLIP_TABLE_SIM_PHYSICS_HZ="${FLIP_TABLE_SIM_PHYSICS_HZ:-200}" \
  -e FLIP_TABLE_SIM_RENDER_INTERVAL="${FLIP_TABLE_SIM_RENDER_INTERVAL:-}" \
  -e FLIP_TABLE_ACT_N_ACTION_STEPS="${FLIP_TABLE_ACT_N_ACTION_STEPS:-10}" \
  -e FLIP_TABLE_ACT_POLICY_HZ="${FLIP_TABLE_ACT_POLICY_HZ:-30}" \
  -e FLIP_TABLE_ACT_SIM_CONTROL_HZ="${FLIP_TABLE_ACT_SIM_CONTROL_HZ:-50}" \
  -e FLIP_TABLE_ACT_DEVICE="${FLIP_TABLE_ACT_DEVICE:-}" \
  -e FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE="${FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE:-1.0}" \
  -e FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2:-100.0}" \
  -e FLIP_TABLE_FLOW_N_ACTION_STEPS="${FLIP_TABLE_FLOW_N_ACTION_STEPS:-6}" \
  -e FLIP_TABLE_FLOW_POLICY_HZ="${FLIP_TABLE_FLOW_POLICY_HZ:-30}" \
  -e FLIP_TABLE_FLOW_SIM_CONTROL_HZ="${FLIP_TABLE_FLOW_SIM_CONTROL_HZ:-50}" \
  -e FLIP_TABLE_FLOW_TARGET_VELOCITY_SCALE="${FLIP_TABLE_FLOW_TARGET_VELOCITY_SCALE:-1.0}" \
  -e FLIP_TABLE_FLOW_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_FLOW_TARGET_ACCELERATION_RAD_S2:-100.0}" \
  -e FLIP_TABLE_GROOT_SOCKET="${FLIP_TABLE_GROOT_SOCKET:-/tmp/flip_table_groot_n17.sock}" \
  -e FLIP_TABLE_GROOT_DEVICE="${FLIP_TABLE_GROOT_DEVICE:-cuda:0}" \
  -e FLIP_TABLE_GROOT_N_ACTION_STEPS="${FLIP_TABLE_GROOT_N_ACTION_STEPS:-10}" \
  -e FLIP_TABLE_GROOT_INFERENCE_SEED="$GROOT_INFERENCE_SEED" \
  -e FLIP_TABLE_GROOT_POLICY_HZ="${FLIP_TABLE_GROOT_POLICY_HZ:-30}" \
  -e FLIP_TABLE_GROOT_TEMPORAL_LAMBDA="${FLIP_TABLE_GROOT_TEMPORAL_LAMBDA:--0.1}" \
  -e FLIP_TABLE_FURNITURE_GROOT_PLUGIN_DIR=/workspace/flip_table_furniture_groot_plugin \
  -e FLIP_TABLE_GROOT_SIM_CONTROL_HZ="${FLIP_TABLE_GROOT_SIM_CONTROL_HZ:-50}" \
  -e FLIP_TABLE_GROOT_TARGET_VELOCITY_SCALE="${FLIP_TABLE_GROOT_TARGET_VELOCITY_SCALE:-1.0}" \
  -e FLIP_TABLE_GROOT_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_GROOT_TARGET_ACCELERATION_RAD_S2:-100.0}" \
  -e FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS="${FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS:-true}" \
  -e FLIP_TABLE_CAMERA_WIDTH="${FLIP_TABLE_CAMERA_WIDTH:-640}" \
  -e FLIP_TABLE_CAMERA_HEIGHT="${FLIP_TABLE_CAMERA_HEIGHT:-480}" \
  -e FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_POS="${FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_POS:-}" \
  -e FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_ROT="${FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_ROT:-}" \
  -e FLIP_TABLE_HEAD_LEFT_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_HEAD_LEFT_CAMERA_FOCAL_LENGTH:-}" \
  -e FLIP_TABLE_HEAD_LEFT_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_HEAD_LEFT_CAMERA_HORIZONTAL_APERTURE:-}" \
  -e FLIP_TABLE_HEAD_LEFT_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_HEAD_LEFT_CAMERA_VERTICAL_APERTURE:-}" \
  -e FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_POS="${FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_POS:-}" \
  -e FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_ROT="${FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_ROT:-}" \
  -e FLIP_TABLE_HEAD_RIGHT_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_HEAD_RIGHT_CAMERA_FOCAL_LENGTH:-}" \
  -e FLIP_TABLE_HEAD_RIGHT_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_HEAD_RIGHT_CAMERA_HORIZONTAL_APERTURE:-}" \
  -e FLIP_TABLE_HEAD_RIGHT_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_HEAD_RIGHT_CAMERA_VERTICAL_APERTURE:-}" \
  -e FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS:-}" \
  -e FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT:-}" \
  -e FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_POS:-}" \
  -e FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_ROT:-}" \
  -e FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_POS:-}" \
  -e FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_ROT:-}" \
  -e FLIP_TABLE_DEX1_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_DEX1_WRIST_CAMERA_FOCAL_LENGTH:-}" \
  -e FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE:-}" \
  -e FLIP_TABLE_DEX1_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_DEX1_WRIST_CAMERA_VERTICAL_APERTURE:-}" \
  -e FLIP_TABLE_LEFT_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_LEFT_WRIST_CAMERA_FOCAL_LENGTH:-}" \
  -e FLIP_TABLE_LEFT_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_LEFT_WRIST_CAMERA_HORIZONTAL_APERTURE:-}" \
  -e FLIP_TABLE_LEFT_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_LEFT_WRIST_CAMERA_VERTICAL_APERTURE:-}" \
  -e FLIP_TABLE_RIGHT_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_RIGHT_WRIST_CAMERA_FOCAL_LENGTH:-}" \
  -e FLIP_TABLE_RIGHT_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_RIGHT_WRIST_CAMERA_HORIZONTAL_APERTURE:-}" \
  -e FLIP_TABLE_RIGHT_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_RIGHT_WRIST_CAMERA_VERTICAL_APERTURE:-}" \
  -e FLIP_TABLE_SAVE_CAMERA_FRAMES="${FLIP_TABLE_SAVE_CAMERA_FRAMES:-}" \
  -e FLIP_TABLE_CAMERA_FRAME_INDICES="${FLIP_TABLE_CAMERA_FRAME_INDICES:-}" \
  -e FLIP_TABLE_CAMERA_FRAME_INDEX="${FLIP_TABLE_CAMERA_FRAME_INDEX:-}" \
  -e FLIP_TABLE_SAVE_CAMERA_NAMES="${FLIP_TABLE_SAVE_CAMERA_NAMES:-}" \
  -e FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR="$CAMERA_FRAME_OUTPUT_CONTAINER" \
  -e FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT="${FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT:-false}" \
  -e FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY="${FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY:-false}" \
  -e FLIP_TABLE_APPLY_RECORDED_CAMERA_GEOMETRY="${FLIP_TABLE_APPLY_RECORDED_CAMERA_GEOMETRY:-true}" \
  -e FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON="${FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON:-}" \
  -e FLIP_TABLE_CALIBRATION_NUM_ENVS="${FLIP_TABLE_CALIBRATION_NUM_ENVS:-}" \
  -e FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M="${FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M:-0.05}" \
  -e FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION="${FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION:-0.70}" \
  -e FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES="${FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES:-}" \
  -e FLIP_TABLE_SAVE_ACTION_STATE_TRACE="${FLIP_TABLE_SAVE_ACTION_STATE_TRACE:-}" \
  -e FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE="${FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE:-false}" \
  -e FLIP_TABLE_JOINT_NOISE_RAD="${FLIP_TABLE_JOINT_NOISE_RAD:-0.02}" \
  -e FLIP_TABLE_SIM_BODY_MODE="$FLIP_TABLE_SIM_BODY_MODE" \
  -e FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE="${FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE:-true}" \
  -e FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE="${FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE:-0.5}" \
  -e FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD="${FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD:-}" \
  -e FLIP_TABLE_DEX1_FINGER_NOISE_M="${FLIP_TABLE_DEX1_FINGER_NOISE_M:-0.002}" \
  -e FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS="${FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS:-}" \
  -e FLIP_TABLE_LOCK_LOWER_BODY="${FLIP_TABLE_LOCK_LOWER_BODY:-false}" \
  -e FLIP_TABLE_LOCK_ROBOT_ROOT="${FLIP_TABLE_LOCK_ROBOT_ROOT:-false}" \
  -e FLIP_TABLE_FIX_ROOT_LINK="${FLIP_TABLE_FIX_ROOT_LINK:-false}" \
  -e FLIP_TABLE_ENABLE_ROBOT_COLLISIONS="${FLIP_TABLE_ENABLE_ROBOT_COLLISIONS:-true}" \
  -e FLIP_TABLE_ENABLE_ROBOT_SELF_COLLISIONS="${FLIP_TABLE_ENABLE_ROBOT_SELF_COLLISIONS:-false}" \
  -e FLIP_TABLE_G1_USD_PATH="${FLIP_TABLE_G1_USD_PATH:-}" \
  -e FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS="${FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS:-true}" \
  -e FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL="${FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL:-true}" \
  -e FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES="${FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES:-true}" \
  -e FLIP_TABLE_DISABLE_COLLISION_FILTER="${FLIP_TABLE_DISABLE_COLLISION_FILTER:-false}" \
  -e FLIP_TABLE_DISABLE_WORKBENCH_COLLISION="${FLIP_TABLE_DISABLE_WORKBENCH_COLLISION:-false}" \
  -e FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS="${FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS:-true}" \
  -e FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE:-0.65,0.95}" \
  -e FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE:-0.48,0.64}" \
  -e FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE:-0.02,0.08}" \
  -e FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE:-0.50,0.75}" \
  -e FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE:-0.35,0.46}" \
  -e FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE:-0.01,0.05}" \
  -e FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE:-0.60,0.90}" \
  -e FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE:-0.42,0.56}" \
  -e FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE:-0.02,0.08}" \
  -e FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE="${FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE:-}" \
  -e FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE="${FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE:-}" \
  -e FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE="${FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE:-}" \
  -e FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM="${FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM:-}" \
  -e FLIP_TABLE_CALIBRATION_DEX1_STIFFNESS_SCALE="${FLIP_TABLE_CALIBRATION_DEX1_STIFFNESS_SCALE:-}" \
  -e FLIP_TABLE_CALIBRATION_DEX1_DAMPING_SCALE="${FLIP_TABLE_CALIBRATION_DEX1_DAMPING_SCALE:-}" \
  -e FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS="${FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS:-base_,hip,knee,ankle,waist}" \
  -e FLIP_TABLE_REQUIRE_WAIST_LOCK="${FLIP_TABLE_REQUIRE_WAIST_LOCK:-false}" \
  -e FLIP_TABLE_RANDOMIZE_ROOM="${FLIP_TABLE_RANDOMIZE_ROOM:-true}" \
  -e FLIP_TABLE_RANDOMIZE_LIGHTING="${FLIP_TABLE_RANDOMIZE_LIGHTING:-true}" \
  -e FLIP_TABLE_ROOM_ASSET_ROOT=/workspace/flip_table_room_assets \
  -e FLIP_TABLE_RANDOMIZE_ROOM_PROPS="${FLIP_TABLE_RANDOMIZE_ROOM_PROPS:-true}" \
  -e FLIP_TABLE_ROOM_PROP_SLOTS="${FLIP_TABLE_ROOM_PROP_SLOTS:-10}" \
  -e FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY="${FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY:-0.62}" \
  -e FLIP_TABLE_ROOM_PROP_ASSETS="${FLIP_TABLE_ROOM_PROP_ASSETS:-Chair,Desk,Shelf,Cabinet,Crates,Plant}" \
  -e FLIP_TABLE_ROOM_PROP_X_RANGE_M="${FLIP_TABLE_ROOM_PROP_X_RANGE_M:--4.8,4.8}" \
  -e FLIP_TABLE_ROOM_PROP_Y_RANGE_M="${FLIP_TABLE_ROOM_PROP_Y_RANGE_M:--4.8,4.8}" \
  -e FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD="${FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD:--3.14159,3.14159}" \
  -e FLIP_TABLE_ROOM_PROP_SCALE_RANGE="${FLIP_TABLE_ROOM_PROP_SCALE_RANGE:-0.80,1.18}" \
  -e FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M="${FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M:-2.20}" \
  -e FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M="${FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M:-0.30}" \
  -e FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M="${FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M:-0.20}" \
  -e FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M="${FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M:-0.50}" \
  -e FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG="${FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG:-80}" \
  -e FLIP_TABLE_ROOM_PROP_FRONT_AXIS="${FLIP_TABLE_ROOM_PROP_FRONT_AXIS:-+x}" \
  -e FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY="${FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY:-0.72}" \
  -e FLIP_TABLE_ROOM_FLOOR_MATERIALS="${FLIP_TABLE_ROOM_FLOOR_MATERIALS:-oak_wood,rough_concrete,ceramic_tile,industrial_vinyl}" \
  -e FLIP_TABLE_ROOM_WALL_MATERIALS="${FLIP_TABLE_ROOM_WALL_MATERIALS:-painted_plaster,rough_concrete,red_brick,oak_panels}" \
  -e FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M="${FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M:-5.5,7.5}" \
  -e FLIP_TABLE_ROOM_WALL_HEIGHT_M="${FLIP_TABLE_ROOM_WALL_HEIGHT_M:-4.0,5.5}" \
  -e FLIP_TABLE_ROOM_TILE_SIZE_M="${FLIP_TABLE_ROOM_TILE_SIZE_M:-0.35,0.9}" \
  -e FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M="${FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M:-0.008,0.025}" \
  -e FLIP_TABLE_ROOM_COLOR_JITTER="${FLIP_TABLE_ROOM_COLOR_JITTER:-0.08}" \
  -e FLIP_TABLE_ROOM_FLOOR_PATTERNS="${FLIP_TABLE_ROOM_FLOOR_PATTERNS:-grid,checker,planks,border}" \
  -e FLIP_TABLE_ROOM_WALL_PATTERNS="${FLIP_TABLE_ROOM_WALL_PATTERNS:-plain,baseboard,horizontal_stripes,vertical_panels,wainscot}" \
  -e FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS="${FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS:-96}" \
  -e FLIP_TABLE_LIGHT_INTENSITY_RANGE="${FLIP_TABLE_LIGHT_INTENSITY_RANGE:-450,1200}" \
  -e FLIP_TABLE_LIGHT_COLOR_RANGE="${FLIP_TABLE_LIGHT_COLOR_RANGE:-0.82,1.0}" \
  -e FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K="${FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K:-3800,6500}" \
  -e FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE="${FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE:-2,4}" \
  -e FLIP_TABLE_SUN_VISIBLE_PROBABILITY="${FLIP_TABLE_SUN_VISIBLE_PROBABILITY:-0.78}" \
  -e FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K="${FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K:-5000,7000}" \
  -e FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE="${FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE:-180,750}" \
  -e FLIP_TABLE_SUN_ELEVATION_DEG="${FLIP_TABLE_SUN_ELEVATION_DEG:-18,72}" \
  -e FLIP_TABLE_SUN_AZIMUTH_DEG="${FLIP_TABLE_SUN_AZIMUTH_DEG:--180,180}" \
  -e FLIP_TABLE_LIGHT_EXPOSURE_RANGE="${FLIP_TABLE_LIGHT_EXPOSURE_RANGE:--0.35,0.35}" \
  -e FLIP_TABLE_SUCCESS_DOT_THRESHOLD="${FLIP_TABLE_SUCCESS_DOT_THRESHOLD:--0.95}" \
  -e FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M="${FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M:-0.35}" \
  -e FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S="${FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S:-0.15}" \
  -e FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S="${FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S:-0.50}" \
  -e FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M="${FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M:-0.03}" \
  -e FLIP_TABLE_SUCCESS_HOLD_STEPS="${FLIP_TABLE_SUCCESS_HOLD_STEPS:-20}" \
  -e FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS="${FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS:-1}" \
  -e FLIP_TABLE_SUCCESS_DEBUG_EVERY="${FLIP_TABLE_SUCCESS_DEBUG_EVERY:-0}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$DOCKER_XAUTH":/tmp/docker.xauth:ro \
  -v "$HOME/.cache/lightwheel_sdk:/root/.cache/lightwheel_sdk" \
  -v "$HOME/.cache/ov:/root/.cache/ov" \
  -v "$HOME/.cache/Kit:/root/.cache/Kit" \
  -v "$HOME/.cache/NVIDIA:/root/.cache/NVIDIA" \
  -v "$CONFIG_HOST":/workspace/flip_table_simulation/flip_table_eval.yml:ro \
  -v "$OVERLAY_TASK":/workspace/robofinals/robofinals_tasks/local_auto_tasks/assemble_table_task.py:ro \
  -v "$POLICY_OVERLAY":/workspace/robofinals/policy/flip_table_eval_policy.py:ro \
  -v "$CV_POLICY_PACKAGE":/workspace/robofinals/policy/cv_rule_based:ro \
  -v "$TELEOP_PACKAGE":/workspace/robofinals/policy/teleop:ro \
  -v "$FLOW_PACKAGE":/workspace/robofinals/policy/flow_matching:ro \
  -v "$RLPD_PACKAGE":/workspace/robofinals/policy/rlpd:ro \
  -v "$GROOT_SHARED_PACKAGE":/workspace/robofinals/policy/team_ramen_groot:ro \
  -v "$FURNITURE_GROOT_PLUGIN":/workspace/flip_table_furniture_groot_plugin:ro \
  -v "$CAMERA_PATCH":/workspace/flip_table_simulation/patch_g1_global_camera.py:ro \
  -v "$WBC_CONTINUITY_PATCH":/workspace/flip_table_simulation/patch_g1_wbc_action_continuity.py:ro \
  -v "$BALANCED_WBC_ACTION":/workspace/flip_table_simulation/team_ramen_balanced_wbc_action.py:ro \
  -v "$SCENE_PREPARE_TOOL":/workspace/flip_table_simulation/prepare_assembled_table_scene.py:ro \
  -v "$IN_PROCESS_EVAL_TOOL":/workspace/flip_table_simulation/run_in_process_eval.py:ro \
  -v "$PERSISTENT_EVAL_WORKER":/workspace/flip_table_simulation/persistent_eval_worker.py:ro \
  -v "$ROOM_ASSETS":/workspace/flip_table_room_assets:ro \
  -v "$GROOT_RUNTIME_SOURCE":/workspace/flip_table_groot_runtime_source:ro \
  -v "$GROOT_RUNTIME_DIR":/workspace/flip_table_groot_runtime \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -v "$OUTPUT_DIR":/workspace/robofinals/eval_result \
  "${replay_mount_args[@]}" \
  "${checkpoint_mount_args[@]}" \
  "$IMAGE" \
  -lc "$(cat <<'CONTAINER_SCRIPT'
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate robofinals
cd /workspace/robofinals

SERVER_PID=""
GROOT_SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$GROOT_SERVER_PID" ]]; then
    kill "$GROOT_SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f "${FLIP_TABLE_GROOT_SOCKET:-/tmp/flip_table_groot_n17.sock}"
}
trap cleanup EXIT

if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "LeRobotGrootN17Policy" ]]; then
  /workspace/flip_table_groot_runtime_source/setup_runtime.sh /workspace/flip_table_groot_runtime
  rm -f "$FLIP_TABLE_GROOT_SOCKET"
  /workspace/flip_table_groot_runtime/.venv/bin/python \
    /workspace/flip_table_groot_runtime_source/groot_inference_server.py \
    --checkpoint "$FLIP_TABLE_POLICY_CHECKPOINT" \
    --socket "$FLIP_TABLE_GROOT_SOCKET" \
    --device "$FLIP_TABLE_GROOT_DEVICE" \
    --n-action-steps "$FLIP_TABLE_GROOT_N_ACTION_STEPS" \
    --seed "$FLIP_TABLE_GROOT_INFERENCE_SEED" \
    > /workspace/robofinals/eval_result/groot_inference_server.log 2>&1 &
  GROOT_SERVER_PID=$!
  GROOT_DEADLINE=$((SECONDS + 1800))
  while [[ ! -S "$FLIP_TABLE_GROOT_SOCKET" ]]; do
    if ! kill -0 "$GROOT_SERVER_PID" >/dev/null 2>&1; then
      echo "ERROR: GR00T inference server exited during startup" >&2
      cat /workspace/robofinals/eval_result/groot_inference_server.log >&2
      exit 1
    fi
    if (( SECONDS >= GROOT_DEADLINE )); then
      echo "ERROR: timed out waiting for GR00T inference server: $FLIP_TABLE_GROOT_SOCKET" >&2
      cat /workspace/robofinals/eval_result/groot_inference_server.log >&2
      exit 1
    fi
    sleep 1
  done
fi

target_wbc_adapter=/workspace/robofinals/robofinals/core/mdp/actions/team_ramen_balanced_wbc_action.py
cp /workspace/flip_table_simulation/team_ramen_balanced_wbc_action.py "$target_wbc_adapter"
python /workspace/flip_table_simulation/patch_g1_global_camera.py
python /workspace/flip_table_simulation/patch_g1_wbc_action_continuity.py

# The organizer layout is immutable inside the pinned image.  Build the
# assembled, kinematic-workbench variant under /tmp and point this run's
# configuration at it before Isaac Lab composes the stage.
CONFIG_PATH=/tmp/flip_table_eval.yml
cp /workspace/flip_table_simulation/flip_table_eval.yml "$CONFIG_PATH"
python - "$CONFIG_PATH" <<'PY'
import os
from pathlib import Path

import yaml

raw = os.environ.get("FLIP_TABLE_CALIBRATION_NUM_ENVS", "").strip()
if raw:
    count = int(raw)
    if not 1 <= count <= 64:
        raise SystemExit("FLIP_TABLE_CALIBRATION_NUM_ENVS must be in [1, 64]")
    config_path = Path(__import__("sys").argv[1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["env_cfg"]["num_envs"] = count
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
if [[ "${FLIP_TABLE_PREPARE_ASSEMBLED_SCENE:-true}" == "true" ]]; then
  read -r layout_source layout_output < <(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(config["env_cfg"]["layout"])
output = Path("/tmp") / f"{source.stem}_flip_table_assembled.usd"
print(source, output)
PY
  )
  scene_prepare_args=(
    --source "$layout_source"
    --output "$layout_output"
  )
  if [[ "${FLIP_TABLE_SIMPLIFY_WHITE_COLLISION:-false}" == "true" ]]; then
    scene_prepare_args+=(--simplify-white-collision)
  fi
  if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "Dex1ForceCalibrationPolicy" ]]; then
    scene_prepare_args+=(--dex1-force-calibration)
  fi
  python /workspace/flip_table_simulation/prepare_assembled_table_scene.py \
    "${scene_prepare_args[@]}"
  python - "$CONFIG_PATH" "$layout_output" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["env_cfg"]["layout"] = sys.argv[2]
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
fi

python - <<'PY'
from pathlib import Path

server_path = Path("/workspace/robofinals/robofinals/scripts/env_server.py")
text = server_path.read_text(encoding="utf-8")
needle = "            rtsp_height=args_cli.rtsp_height if args_cli.enable_rtsp else None,\n"
if "            enable_full_local_scene=True,\n" not in text:
    if needle not in text:
        raise SystemExit("env_server.py parse_env_cfg call changed; cannot enable full local scene")
    text = text.replace(needle, needle + "            enable_full_local_scene=True,\n", 1)

render_needle = "    env: ManagerBasedEnv = gym_env.unwrapped\n    warmup_rendering(env)\n"
render_replacement = (
    "    env: ManagerBasedEnv = gym_env.unwrapped\n"
    "    # Match the organizer teleop/RL render path so PhysX articulation\n"
    "    # transforms are copied into Fabric before camera rendering.\n"
    "    from robofinals.utils.render_utils import optimize_rendering\n"
    "    args_cli.disable_fabric = cfg.disable_fabric\n"
    "    optimize_rendering(env, args_cli)\n"
    "    warmup_rendering(env)\n"
)
if "    optimize_rendering(env, args_cli)\n" not in text:
    if render_needle not in text:
        raise SystemExit("env_server.py make_env changed; cannot enable Fabric render synchronization")
    text = text.replace(render_needle, render_replacement, 1)
elif "    args_cli.disable_fabric = cfg.disable_fabric\n" not in text:
    text = text.replace(
        "    optimize_rendering(env, args_cli)\n",
        "    args_cli.disable_fabric = cfg.disable_fabric\n"
        "    optimize_rendering(env, args_cli)\n",
        1,
    )

teleop_marker = "    # Team RAMEN diagnostic-only teleoperation hook.\n"
if teleop_marker not in text:
    return_needle = "    return env\n"
    hook = (
        teleop_marker
        + "    def flip_table_teleop_diagnostics():\n"
        + "        task = getattr(getattr(env.cfg, 'isaaclab_arena_env', None), 'task', None)\n"
        + "        callback = getattr(task, 'get_flip_table_teleop_diagnostics', None)\n"
        + "        if callback is None:\n"
        + "            raise RuntimeError('flip-table teleoperation diagnostics are unavailable')\n"
        + "        return callback(env)\n"
        + "    env.flip_table_teleop_diagnostics = flip_table_teleop_diagnostics\n"
    )
    if return_needle not in text:
        raise SystemExit("env_server.py make_env return changed; cannot attach teleop diagnostics")
    text = text.replace(return_needle, hook + return_needle, 1)

force_hook_marker = "    # Team RAMEN servo-rate force diagnostic hook.\n"
if force_hook_marker not in text:
    anchor = "    env.flip_table_teleop_diagnostics = flip_table_teleop_diagnostics\n"
    hook = (
        force_hook_marker
        + "    def flip_table_teleop_force_diagnostics():\n"
        + "        task = getattr(getattr(env.cfg, 'isaaclab_arena_env', None), 'task', None)\n"
        + "        callback = getattr(task, 'get_flip_table_teleop_force_diagnostics', None)\n"
        + "        if callback is None:\n"
        + "            raise RuntimeError('flip-table force diagnostics are unavailable')\n"
        + "        return callback(env)\n"
        + "    env.flip_table_teleop_force_diagnostics = flip_table_teleop_force_diagnostics\n"
    )
    if anchor not in text:
        raise SystemExit("env_server.py teleoperation hook changed; cannot attach force diagnostics")
    text = text.replace(anchor, anchor + hook, 1)

server_path.write_text(text, encoding="utf-8")
PY

python - <<'PY'
from pathlib import Path

init_path = Path("/workspace/robofinals/policy/__init__.py")
marker = "# flip_table_eval_policy_overlay"
text = init_path.read_text(encoding="utf-8")
if marker not in text:
    text += f"""

{marker}
_POLICIES.update({{
    "NoOpPolicy": ".flip_table_eval_policy",
    "ScriptedJointPolicy": ".flip_table_eval_policy",
    "CvRuleBasedPolicy": ".flip_table_eval_policy",
    "AvpTeleopPolicy": ".flip_table_eval_policy",
    "TeleopPerformanceBenchmarkPolicy": ".flip_table_eval_policy",
    "Dex1ForceCalibrationPolicy": ".flip_table_eval_policy",
    "RecordedJointTargetPolicy": ".flip_table_eval_policy",
    "RecordedFullBodyTargetPolicy": ".flip_table_eval_policy",
    "RecordedWBCPolicy": ".flip_table_eval_policy",
    "LeRobotACTPolicy": ".flip_table_eval_policy",
    "FlowMatchingBCPolicy": ".flip_table_eval_policy",
    "FlowMatchingRLPDPolicy": ".flip_table_eval_policy",
    "LeRobotGrootN17Policy": ".flip_table_eval_policy",
}})
__all__ = list(_POLICIES)
"""
    init_path.write_text(text, encoding="utf-8")
else:
    changed = False
    if '"AvpTeleopPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "CvRuleBasedPolicy": ".flip_table_eval_policy",'
        text = text.replace(anchor, anchor + '\n    "AvpTeleopPolicy": ".flip_table_eval_policy",', 1)
        changed = True
    if '"TeleopPerformanceBenchmarkPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "AvpTeleopPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "TeleopPerformanceBenchmarkPolicy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if '"Dex1ForceCalibrationPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "TeleopPerformanceBenchmarkPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "Dex1ForceCalibrationPolicy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if '"CvRuleBasedPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "ScriptedJointPolicy": ".flip_table_eval_policy",'
        text = text.replace(anchor, anchor + '\n    "CvRuleBasedPolicy": ".flip_table_eval_policy",', 1)
        changed = True
    if '"RecordedJointTargetPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "LeRobotACTPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            '    "RecordedJointTargetPolicy": ".flip_table_eval_policy",\n' + anchor,
            1,
        )
    if '"RecordedFullBodyTargetPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "RecordedJointTargetPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "RecordedFullBodyTargetPolicy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if '"RecordedWBCPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "LeRobotACTPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            '    "RecordedWBCPolicy": ".flip_table_eval_policy",\n' + anchor,
            1,
        )
        changed = True
    if '"LeRobotGrootN17Policy": ".flip_table_eval_policy"' not in text:
        anchor = '    "LeRobotACTPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "LeRobotGrootN17Policy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if '"FlowMatchingBCPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "LeRobotACTPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "FlowMatchingBCPolicy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if '"FlowMatchingRLPDPolicy": ".flip_table_eval_policy"' not in text:
        anchor = '    "FlowMatchingBCPolicy": ".flip_table_eval_policy",'
        text = text.replace(
            anchor,
            anchor + '\n    "FlowMatchingRLPDPolicy": ".flip_table_eval_policy",',
            1,
        )
        changed = True
    if changed:
        init_path.write_text(text, encoding="utf-8")
PY

if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(AvpTeleopPolicy|TeleopPerformanceBenchmarkPolicy|Dex1ForceCalibrationPolicy)$ ]]; then
  in_process_args=(
    --config "$CONFIG_PATH"
    --policy-name "$FLIP_TABLE_POLICY_NAME"
    --test-num "$FLIP_TABLE_TEST_NUM"
  )
  if [[ -n "${FLIP_TABLE_TIME_OUT_LIMIT:-}" ]]; then
    in_process_args+=(--time-out-limit "$FLIP_TABLE_TIME_OUT_LIMIT")
  fi
  python /workspace/flip_table_simulation/run_in_process_eval.py "${in_process_args[@]}"
  exit 0
fi

if [[ "${FLIP_TABLE_PERSISTENT_EVAL_WORKER:-false}" == "true" ]]; then
  python /workspace/flip_table_simulation/persistent_eval_worker.py \
    --config "$CONFIG_PATH" \
    --job-root /workspace/robofinals/eval_result
  exit 0
fi

python robofinals/scripts/env_server.py --enable_cameras &
SERVER_PID=$!

# The official evaluator uses a fixed local IPC port.  Always reap its server
# when this wrapper exits so a failed calibration attempt cannot poison the
# next run with an address-in-use error.
cleanup_env_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_env_server EXIT INT TERM

python - <<'PY'
import socket
import time

deadline = time.time() + 240
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", 50000)) == 0:
            raise SystemExit(0)
    time.sleep(2)
raise SystemExit("Timed out waiting for RoboFinals env_server on 127.0.0.1:50000")
PY

overrides=(--test_num "$FLIP_TABLE_TEST_NUM" --record_video True)
if [[ -n "${FLIP_TABLE_POLICY_NAME:-}" ]]; then
  overrides+=(--policy_name "$FLIP_TABLE_POLICY_NAME")
fi
if [[ -n "${FLIP_TABLE_TIME_OUT_LIMIT:-}" ]]; then
  overrides+=(--time_out_limit "$FLIP_TABLE_TIME_OUT_LIMIT")
fi
if [[ -n "${FLIP_TABLE_POLICY_CHECKPOINT:-}" ]]; then
  overrides+=(--checkpoint "$FLIP_TABLE_POLICY_CHECKPOINT")
fi

python robofinals/scripts/policy/eval_policy.py \
  --config "$CONFIG_PATH" \
  --overrides "${overrides[@]}"
CONTAINER_SCRIPT
)"
