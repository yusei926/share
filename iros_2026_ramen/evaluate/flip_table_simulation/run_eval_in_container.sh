#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/evaluate/flip_table_simulation"
ROBOFINALS_ROOT="${ROBOFINALS_ROOT:-/workspace/robofinals}"
OUTPUT_DIR="${FLIP_TABLE_SIM_OUTPUT_DIR:-$ROOT_DIR/outputs/flip_table_simulation/eval_result}"
CONFIG_HOST="${FLIP_TABLE_EVAL_CONFIG:-$FEATURE_DIR/configs/flip_table_eval.yml}"
OVERLAY_TASK="$FEATURE_DIR/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
POLICY_OVERLAY="$FEATURE_DIR/container_overlay/policy/flip_table_eval_policy.py"
CV_POLICY_PACKAGE="$FEATURE_DIR/container_overlay/policy/cv_rule_based"
TELEOP_PACKAGE="$ROOT_DIR/data/flip_table_data_augmentation/teleop"
FLOW_PACKAGE="$ROOT_DIR/model/subtask_policy_training/flow_matching"
RLPD_PACKAGE="$ROOT_DIR/model/flip_table_reinforcement_learning/rlpd"
CAMERA_PATCH="$FEATURE_DIR/container_overlay/patches/patch_g1_global_camera.py"
BALANCED_WBC_ACTION="$FEATURE_DIR/container_overlay/mdp/team_ramen_balanced_wbc_action.py"
SCENE_PREPARE_TOOL="$FEATURE_DIR/tools/prepare_assembled_table_scene.py"
IN_PROCESS_EVAL_TOOL="$FEATURE_DIR/tools/run_in_process_eval.py"
PERSISTENT_EVAL_WORKER="$FEATURE_DIR/tools/persistent_eval_worker.py"
GROOT_RUNTIME_SOURCE="$FEATURE_DIR/groot_runtime"
GROOT_RUNTIME_DIR="${FLIP_TABLE_GROOT_RUNTIME_DIR:-/workspace/flip_table_groot_runtime}"
GROOT_SHARED_PACKAGE="$ROOT_DIR/model/subtask_policy_training/gr00t"
FURNITURE_GROOT_PLUGIN="$ROOT_DIR/model/subtask_policy_training/lerobot_policy_furniture_groot"
TEST_NUM="${FLIP_TABLE_TEST_NUM:-10}"

for required_path in "$CONFIG_HOST" "$OVERLAY_TASK" "$POLICY_OVERLAY" "$CAMERA_PATCH" "$BALANCED_WBC_ACTION" "$SCENE_PREPARE_TOOL" "$IN_PROCESS_EVAL_TOOL" "$PERSISTENT_EVAL_WORKER"; do
  if [[ ! -f "$required_path" ]]; then
    echo "ERROR: required flip-table file not found: $required_path" >&2
    exit 1
  fi
done
if [[ ! -d "$CV_POLICY_PACKAGE" ]]; then
  echo "ERROR: required CV policy package not found: $CV_POLICY_PACKAGE" >&2
  exit 1
fi
if [[ ! -f "$TELEOP_PACKAGE/configs/teleop_v1.json" ]]; then
  echo "ERROR: shared teleoperation package is incomplete: $TELEOP_PACKAGE" >&2
  exit 1
fi
for required_path in "$FLOW_PACKAGE" "$RLPD_PACKAGE" "$GROOT_SHARED_PACKAGE" "$FURNITURE_GROOT_PLUGIN"; do
  if [[ ! -d "$required_path" ]]; then
    echo "ERROR: required flip-table package not found: $required_path" >&2
    exit 1
  fi
done
if [[ ! "$TEST_NUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FLIP_TABLE_TEST_NUM must be a positive integer" >&2
  exit 2
fi
if [[ -z "${FLIP_TABLE_POLICY_NAME:-}" ]]; then
  FLIP_TABLE_POLICY_NAME="$(awk '/^policy_name:/ {gsub(/\"/, "", $2); print $2; exit}' "$CONFIG_HOST")"
  export FLIP_TABLE_POLICY_NAME
fi
if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy|LeRobotGrootN17Policy)$ ]]; then
  if [[ -z "${FLIP_TABLE_POLICY_CHECKPOINT:-}" || ! -d "$FLIP_TABLE_POLICY_CHECKPOINT" ]]; then
    echo "ERROR: ${FLIP_TABLE_POLICY_NAME} requires a local FLIP_TABLE_POLICY_CHECKPOINT directory" >&2
    exit 1
  fi
fi

EVAL_MODE="${FLIP_TABLE_EVAL_MODE:-randomized}"
if [[ "$EVAL_MODE" == "nominal" ]]; then
  export FLIP_TABLE_TABLE_LONG_RANGE_M=0
  export FLIP_TABLE_TABLE_DEPTH_RANGE_M=0
  export FLIP_TABLE_TABLE_YAW_RANGE_RAD=0
  export FLIP_TABLE_ROBOT_DISTANCE_RANGE_M=0
  export FLIP_TABLE_ROBOT_LATERAL_RANGE_M=0
  export FLIP_TABLE_ROBOT_YAW_RANGE_RAD=0
  export FLIP_TABLE_JOINT_NOISE_RAD=0
  export FLIP_TABLE_DEX1_FINGER_NOISE_M=0
  export FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE=false
  export FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS=false
  export FLIP_TABLE_RANDOMIZE_ROOM=false
  export FLIP_TABLE_RANDOMIZE_ROOM_PROPS=false
  export FLIP_TABLE_RANDOMIZE_LIGHTING=false
  export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS=false
  export FLIP_TABLE_RL_RANDOMIZE_MASS=false
  export FLIP_TABLE_EVAL_RANDOMIZE_MASS=false
  export FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES=false
  export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY=false
  export FLIP_TABLE_RL_ENABLE_SENSOR_NOISE=false
  export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS=0
else
  case "$EVAL_MODE" in
    randomized|unseen_dr) ;;
    *)
      echo "ERROR: FLIP_TABLE_EVAL_MODE must be nominal, randomized, or unseen_dr" >&2
      exit 2
      ;;
  esac
fi
export FLIP_TABLE_EVAL_MODE="$EVAL_MODE"
GROOT_DR_PROFILE="${FLIP_TABLE_GROOT_DR_PROFILE:-generic_v1}"
if [[ ! "$GROOT_DR_PROFILE" =~ ^[a-z0-9_]+$ ]]; then
  echo "ERROR: FLIP_TABLE_GROOT_DR_PROFILE must be a lowercase profile identifier" >&2
  exit 2
fi
export FLIP_TABLE_GROOT_DR_PROFILE="$GROOT_DR_PROFILE"
export FLIP_TABLE_SIM_BODY_MODE="${FLIP_TABLE_SIM_BODY_MODE:-balanced_wbc}"
case "$FLIP_TABLE_SIM_BODY_MODE" in
  balanced_wbc)
    export FLIP_TABLE_LOCK_LOWER_BODY=false
    export FLIP_TABLE_LOCK_ROBOT_ROOT=false
    export FLIP_TABLE_FIX_ROOT_LINK=false
    export FLIP_TABLE_REQUIRE_WAIST_LOCK=false
    export FLIP_TABLE_SIM_PHYSICS_HZ=200
    ;;
  fixed_diagnostic) ;;
  *) echo "ERROR: unsupported FLIP_TABLE_SIM_BODY_MODE=$FLIP_TABLE_SIM_BODY_MODE" >&2; exit 2 ;;
esac

if [[ ! -d "$ROBOFINALS_ROOT" ]]; then
  echo "ERROR: ROBOFINALS_ROOT not found: $ROBOFINALS_ROOT"
  echo "Run this script inside the paperc/robofinals:RoboFinals-IKEA-V1 container."
  exit 1
fi
ROBOFINALS_REAL_ROOT="$(realpath "$ROBOFINALS_ROOT")"
OFFICIAL_V1_BACKUP_ROOT="${FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT:-$(dirname "$ROBOFINALS_REAL_ROOT")/robofinalsbak}"
if [[ ! -d "$OFFICIAL_V1_BACKUP_ROOT/robofinals" ]]; then
  echo "ERROR: immutable RoboFinals-IKEA-V1 backup not found: $OFFICIAL_V1_BACKUP_ROOT" >&2
  exit 1
fi
export FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT="$OFFICIAL_V1_BACKUP_ROOT"
export FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES=true

mkdir -p "$OUTPUT_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all,graphics,display}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-:1}"
export PATH="/opt/conda/bin:/opt/conda/envs/robofinals/bin:$PATH"
export FLIP_TABLE_CAMERA_WIDTH="${FLIP_TABLE_CAMERA_WIDTH:-640}"
export FLIP_TABLE_CAMERA_HEIGHT="${FLIP_TABLE_CAMERA_HEIGHT:-480}"
export FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS="${FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS:-true}"
export FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_POS="${FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_POS:-}"
export FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_ROT="${FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_ROT:-}"
export FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_POS="${FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_POS:-}"
export FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_ROT="${FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_ROT:-}"
export FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS:-}"
export FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT:-}"
export FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_POS:-}"
export FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_LEFT_WRIST_CAMERA_OFFSET_ROT:-}"
export FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_POS="${FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_POS:-}"
export FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_ROT="${FLIP_TABLE_RIGHT_WRIST_CAMERA_OFFSET_ROT:-}"
export FLIP_TABLE_DEX1_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_DEX1_WRIST_CAMERA_FOCAL_LENGTH:-}"
export FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE:-}"
export FLIP_TABLE_DEX1_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_DEX1_WRIST_CAMERA_VERTICAL_APERTURE:-}"
export FLIP_TABLE_LEFT_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_LEFT_WRIST_CAMERA_FOCAL_LENGTH:-}"
export FLIP_TABLE_LEFT_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_LEFT_WRIST_CAMERA_HORIZONTAL_APERTURE:-}"
export FLIP_TABLE_LEFT_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_LEFT_WRIST_CAMERA_VERTICAL_APERTURE:-}"
export FLIP_TABLE_RIGHT_WRIST_CAMERA_FOCAL_LENGTH="${FLIP_TABLE_RIGHT_WRIST_CAMERA_FOCAL_LENGTH:-}"
export FLIP_TABLE_RIGHT_WRIST_CAMERA_HORIZONTAL_APERTURE="${FLIP_TABLE_RIGHT_WRIST_CAMERA_HORIZONTAL_APERTURE:-}"
export FLIP_TABLE_RIGHT_WRIST_CAMERA_VERTICAL_APERTURE="${FLIP_TABLE_RIGHT_WRIST_CAMERA_VERTICAL_APERTURE:-}"
export FLIP_TABLE_SAVE_CAMERA_FRAMES="${FLIP_TABLE_SAVE_CAMERA_FRAMES:-false}"
export FLIP_TABLE_CAMERA_FRAME_INDICES="${FLIP_TABLE_CAMERA_FRAME_INDICES:-}"
export FLIP_TABLE_CAMERA_FRAME_INDEX="${FLIP_TABLE_CAMERA_FRAME_INDEX:-10}"
export FLIP_TABLE_SAVE_CAMERA_NAMES="${FLIP_TABLE_SAVE_CAMERA_NAMES:-}"
export FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR="${FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR:-}"
export FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT="${FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT:-false}"
export FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY="${FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY:-false}"
export FLIP_TABLE_APPLY_RECORDED_CAMERA_GEOMETRY="${FLIP_TABLE_APPLY_RECORDED_CAMERA_GEOMETRY:-true}"
export FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON="${FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON:-}"
export FLIP_TABLE_CALIBRATION_NUM_ENVS="${FLIP_TABLE_CALIBRATION_NUM_ENVS:-}"
export FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES="${FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES:-true}"
export FLIP_TABLE_SAVE_ACTION_STATE_TRACE="${FLIP_TABLE_SAVE_ACTION_STATE_TRACE:-true}"
export FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE="${FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE:-false}"
export FLIP_TABLE_REPLAY_ACTION_PATH="${FLIP_TABLE_REPLAY_ACTION_PATH:-}"
export FLIP_TABLE_REPLAY_HOLD_INDEX="${FLIP_TABLE_REPLAY_HOLD_INDEX:-}"
export FLIP_TABLE_REPLAY_HZ="${FLIP_TABLE_REPLAY_HZ:-30}"
export FLIP_TABLE_REPLAY_WARMUP_STEPS="${FLIP_TABLE_REPLAY_WARMUP_STEPS:-0}"
export FLIP_TABLE_INITIAL_UPPER_BODY_STATE="${FLIP_TABLE_INITIAL_UPPER_BODY_STATE:-}"
export FLIP_TABLE_PREPARE_ASSEMBLED_SCENE="${FLIP_TABLE_PREPARE_ASSEMBLED_SCENE:-true}"
export FLIP_TABLE_SIMPLIFY_WHITE_COLLISION="${FLIP_TABLE_SIMPLIFY_WHITE_COLLISION:-false}"
export FLIP_TABLE_BENCHMARK_WARMUP_STEPS="${FLIP_TABLE_BENCHMARK_WARMUP_STEPS:-40}"
export FLIP_TABLE_BENCHMARK_MEASURE_STEPS="${FLIP_TABLE_BENCHMARK_MEASURE_STEPS:-180}"
export FLIP_TABLE_TABLE_XY_RANGE_M="${FLIP_TABLE_TABLE_XY_RANGE_M:-}"
export FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL="${FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL:-}"
export FLIP_TABLE_TABLE_YAW_OFFSET_RAD="${FLIP_TABLE_TABLE_YAW_OFFSET_RAD:-0}"
export FLIP_TABLE_TABLE_LONG_RANGE_M="${FLIP_TABLE_TABLE_LONG_RANGE_M:-0.12}"
export FLIP_TABLE_TABLE_DEPTH_RANGE_M="${FLIP_TABLE_TABLE_DEPTH_RANGE_M:-0.035}"
export FLIP_TABLE_TABLE_YAW_RANGE_RAD="${FLIP_TABLE_TABLE_YAW_RANGE_RAD:-3.141592653589793}"
export FLIP_TABLE_WORKBENCH_FRONT_AXIS="${FLIP_TABLE_WORKBENCH_FRONT_AXIS:--y}"
export FLIP_TABLE_ROBOT_DISTANCE_M="${FLIP_TABLE_ROBOT_DISTANCE_M:-0.26}"
export FLIP_TABLE_ROBOT_DISTANCE_RANGE_M="${FLIP_TABLE_ROBOT_DISTANCE_RANGE_M:-0.04}"
export FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M="${FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M:-0.62}"
export FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M="${FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M:-0.20}"
export FLIP_TABLE_ROBOT_LATERAL_RANGE_M="${FLIP_TABLE_ROBOT_LATERAL_RANGE_M:-0.10}"
export FLIP_TABLE_ROBOT_YAW_RANGE_RAD="${FLIP_TABLE_ROBOT_YAW_RANGE_RAD:-0.08}"
export FLIP_TABLE_ROBOT_YAW_OFFSET_RAD="${FLIP_TABLE_ROBOT_YAW_OFFSET_RAD:-0.0}"
export FLIP_TABLE_ROBOT_BASE_HEIGHT_M="${FLIP_TABLE_ROBOT_BASE_HEIGHT_M:-0.78}"
export FLIP_TABLE_ROBOT_ROOT_POS_LOCAL="${FLIP_TABLE_ROBOT_ROOT_POS_LOCAL:-}"
export FLIP_TABLE_ROBOT_ROOT_YAW_RAD="${FLIP_TABLE_ROBOT_ROOT_YAW_RAD:-}"
export FLIP_TABLE_USE_DEFAULT_ROBOT_POSE="${FLIP_TABLE_USE_DEFAULT_ROBOT_POSE:-false}"
export FLIP_TABLE_JOINT_NOISE_RAD="${FLIP_TABLE_JOINT_NOISE_RAD:-0.02}"
export FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE="${FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE:-true}"
export FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE="${FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE:-0.5}"
export FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD="${FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD:-}"
export FLIP_TABLE_DEX1_FINGER_NOISE_M="${FLIP_TABLE_DEX1_FINGER_NOISE_M:-0.002}"
export FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS="${FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS:-}"
if [[ "$FLIP_TABLE_SIM_BODY_MODE" == fixed_diagnostic ]]; then
  if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(NoOpPolicy|ScriptedJointPolicy|RecordedJointTargetPolicy|RecordedFullBodyTargetPolicy|AvpTeleopPolicy|LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy|LeRobotGrootN17Policy|TeleopPerformanceBenchmarkPolicy|Dex1ForceCalibrationPolicy)$ && -z "${FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION+x}" ]]; then
    export FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=true
  fi
  if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "CvRuleBasedPolicy" ]]; then
    export FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=false
    export FLIP_TABLE_USE_PINK_EEF_ACTION=true
  fi
else
  # These switches select legacy diagnostic action managers. Production WBC
  # paths always use the 16-D absolute arm/hand adapter.
  export FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=false
  export FLIP_TABLE_USE_PINK_EEF_ACTION=false
fi
export FLIP_TABLE_CV_SIM_CONTROL_HZ="${FLIP_TABLE_CV_SIM_CONTROL_HZ:-50}"
export FLIP_TABLE_CV_MIN_CONFIDENCE="${FLIP_TABLE_CV_MIN_CONFIDENCE:-0.03}"
export FLIP_TABLE_CV_MIN_LEG_CONFIDENCE="${FLIP_TABLE_CV_MIN_LEG_CONFIDENCE:-0.20}"
export FLIP_TABLE_CV_WARMUP_STEPS="${FLIP_TABLE_CV_WARMUP_STEPS:-50}"
export FLIP_TABLE_CV_SETTLED_SELECTION_STEPS="${FLIP_TABLE_CV_SETTLED_SELECTION_STEPS:-20}"
export FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS="${FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS:-10}"
export FLIP_TABLE_CV_REDETECT_ALPHA="${FLIP_TABLE_CV_REDETECT_ALPHA:-0.30}"
export FLIP_TABLE_CV_REDETECT_MAX_TRANSLATION_M="${FLIP_TABLE_CV_REDETECT_MAX_TRANSLATION_M:-0.12}"
export FLIP_TABLE_CV_REDETECT_MAX_YAW_RAD="${FLIP_TABLE_CV_REDETECT_MAX_YAW_RAD:-0.35}"
export FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD="${FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD:--0.017}"
export FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS="${FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS:-8}"
export FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL="${FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL:-2}"
export FLIP_TABLE_ACT_N_ACTION_STEPS="${FLIP_TABLE_ACT_N_ACTION_STEPS:-10}"
export FLIP_TABLE_ACT_POLICY_HZ="${FLIP_TABLE_ACT_POLICY_HZ:-30}"
export FLIP_TABLE_ACT_SIM_CONTROL_HZ="${FLIP_TABLE_ACT_SIM_CONTROL_HZ:-50}"
export FLIP_TABLE_ACT_DEVICE="${FLIP_TABLE_ACT_DEVICE:-}"
export FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE="${FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE:-1.0}"
export FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2:-100.0}"
export FLIP_TABLE_FLOW_N_ACTION_STEPS="${FLIP_TABLE_FLOW_N_ACTION_STEPS:-6}"
export FLIP_TABLE_FLOW_POLICY_HZ="${FLIP_TABLE_FLOW_POLICY_HZ:-30}"
export FLIP_TABLE_FLOW_SIM_CONTROL_HZ="${FLIP_TABLE_FLOW_SIM_CONTROL_HZ:-50}"
export FLIP_TABLE_FLOW_TARGET_VELOCITY_SCALE="${FLIP_TABLE_FLOW_TARGET_VELOCITY_SCALE:-1.0}"
export FLIP_TABLE_FLOW_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_FLOW_TARGET_ACCELERATION_RAD_S2:-100.0}"
export FLIP_TABLE_GROOT_SOCKET="${FLIP_TABLE_GROOT_SOCKET:-/tmp/flip_table_groot_n17.sock}"
export FLIP_TABLE_GROOT_DEVICE="${FLIP_TABLE_GROOT_DEVICE:-cuda:0}"
export FLIP_TABLE_GROOT_N_ACTION_STEPS="${FLIP_TABLE_GROOT_N_ACTION_STEPS:-10}"
export FLIP_TABLE_GROOT_INFERENCE_SEED="${FLIP_TABLE_GROOT_INFERENCE_SEED:-${FLIP_TABLE_EVAL_SEED:-42}}"
if [[ ! "$FLIP_TABLE_GROOT_INFERENCE_SEED" =~ ^[0-9]+$ ]] || (( FLIP_TABLE_GROOT_INFERENCE_SEED > 4294967295 )); then
  echo "ERROR: FLIP_TABLE_GROOT_INFERENCE_SEED must fit uint32" >&2
  exit 2
fi
export FLIP_TABLE_GROOT_POLICY_HZ="${FLIP_TABLE_GROOT_POLICY_HZ:-30}"
export FLIP_TABLE_GROOT_SIM_CONTROL_HZ="${FLIP_TABLE_GROOT_SIM_CONTROL_HZ:-50}"
export FLIP_TABLE_GROOT_TEMPORAL_LAMBDA="${FLIP_TABLE_GROOT_TEMPORAL_LAMBDA:--0.1}"
export FLIP_TABLE_FURNITURE_GROOT_PLUGIN_DIR="$FURNITURE_GROOT_PLUGIN"
export FLIP_TABLE_GROOT_TARGET_VELOCITY_SCALE="${FLIP_TABLE_GROOT_TARGET_VELOCITY_SCALE:-1.0}"
export FLIP_TABLE_GROOT_TARGET_ACCELERATION_RAD_S2="${FLIP_TABLE_GROOT_TARGET_ACCELERATION_RAD_S2:-100.0}"
export FLIP_TABLE_TELEOP_PORT="${FLIP_TABLE_TELEOP_PORT:-59610}"
export FLIP_TABLE_TELEOP_BIND_HOST="${FLIP_TABLE_TELEOP_BIND_HOST:-0.0.0.0}"
export FLIP_TABLE_RL_RANDOMIZATION_LEVEL="${FLIP_TABLE_RL_RANDOMIZATION_LEVEL:-1.0}"
export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS="${FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS:-true}"
export FLIP_TABLE_RL_RANDOMIZE_MASS=false
export FLIP_TABLE_EVAL_RANDOMIZE_MASS=false
export FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES="${FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES:-true}"
export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY="${FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY:-true}"
export FLIP_TABLE_RL_ENABLE_SENSOR_NOISE="${FLIP_TABLE_RL_ENABLE_SENSOR_NOISE:-true}"
export FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS="${FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS:-2}"
export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS="${FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS:-2}"
export FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION="${FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION:-true}"
export FLIP_TABLE_SUCCESS_DOT_THRESHOLD="${FLIP_TABLE_SUCCESS_DOT_THRESHOLD:--0.95}"
export FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M="${FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M:-0.35}"
export FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S="${FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S:-0.15}"
export FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S="${FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S:-0.50}"
export FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M="${FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M:-0.03}"
export FLIP_TABLE_SUCCESS_HOLD_STEPS="${FLIP_TABLE_SUCCESS_HOLD_STEPS:-20}"
export FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS="${FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS:-1}"
export FLIP_TABLE_SUCCESS_DEBUG_EVERY="${FLIP_TABLE_SUCCESS_DEBUG_EVERY:-0}"
export FLIP_TABLE_FIX_ROOT_LINK="${FLIP_TABLE_FIX_ROOT_LINK:-false}"
export FLIP_TABLE_ENABLE_ROBOT_COLLISIONS="${FLIP_TABLE_ENABLE_ROBOT_COLLISIONS:-true}"
export FLIP_TABLE_ENABLE_ROBOT_SELF_COLLISIONS="${FLIP_TABLE_ENABLE_ROBOT_SELF_COLLISIONS:-false}"
export FLIP_TABLE_G1_USD_PATH="${FLIP_TABLE_G1_USD_PATH:-}"
export FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS="${FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS:-true}"
export FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL="${FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL:-true}"
export FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES="${FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES:-true}"
export FLIP_TABLE_DISABLE_COLLISION_FILTER="${FLIP_TABLE_DISABLE_COLLISION_FILTER:-false}"
export FLIP_TABLE_DISABLE_WORKBENCH_COLLISION="${FLIP_TABLE_DISABLE_WORKBENCH_COLLISION:-false}"
export FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS="${FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS:-true}"
export FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE:-0.65,0.95}"
export FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE:-0.48,0.64}"
export FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE:-0.02,0.08}"
export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE:-0.50,0.75}"
export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE:-0.35,0.46}"
export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE:-0.01,0.05}"
export FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE:-0.60,0.90}"
export FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE:-0.42,0.56}"
export FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE="${FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE:-0.02,0.08}"
export FLIP_TABLE_LOCK_LOWER_BODY="${FLIP_TABLE_LOCK_LOWER_BODY:-false}"
export FLIP_TABLE_LOCK_ROBOT_ROOT="${FLIP_TABLE_LOCK_ROBOT_ROOT:-false}"
export FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS="${FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS:-base_,hip,knee,ankle,waist}"
export FLIP_TABLE_RANDOMIZE_ROOM="${FLIP_TABLE_RANDOMIZE_ROOM:-true}"
export FLIP_TABLE_ROOM_ASSET_ROOT="${FLIP_TABLE_ROOM_ASSET_ROOT:-$FEATURE_DIR/assets/room}"
export FLIP_TABLE_RANDOMIZE_ROOM_PROPS="${FLIP_TABLE_RANDOMIZE_ROOM_PROPS:-true}"
export FLIP_TABLE_ROOM_PROP_SLOTS="${FLIP_TABLE_ROOM_PROP_SLOTS:-10}"
export FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY="${FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY:-0.62}"
export FLIP_TABLE_ROOM_PROP_ASSETS="${FLIP_TABLE_ROOM_PROP_ASSETS:-Chair,Desk,Shelf,Cabinet,Crates,Plant}"
export FLIP_TABLE_ROOM_PROP_X_RANGE_M="${FLIP_TABLE_ROOM_PROP_X_RANGE_M:--4.8,4.8}"
export FLIP_TABLE_ROOM_PROP_Y_RANGE_M="${FLIP_TABLE_ROOM_PROP_Y_RANGE_M:--4.8,4.8}"
export FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD="${FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD:--3.14159,3.14159}"
export FLIP_TABLE_ROOM_PROP_SCALE_RANGE="${FLIP_TABLE_ROOM_PROP_SCALE_RANGE:-0.80,1.18}"
export FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M="${FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M:-2.20}"
export FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M="${FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M:-0.30}"
export FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M="${FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M:-0.20}"
export FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M="${FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M:-0.50}"
export FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG="${FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG:-80}"
export FLIP_TABLE_ROOM_PROP_FRONT_AXIS="${FLIP_TABLE_ROOM_PROP_FRONT_AXIS:-+x}"
export FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY="${FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY:-0.72}"
export FLIP_TABLE_ROOM_FLOOR_MATERIALS="${FLIP_TABLE_ROOM_FLOOR_MATERIALS:-oak_wood,rough_concrete,ceramic_tile,industrial_vinyl}"
export FLIP_TABLE_ROOM_WALL_MATERIALS="${FLIP_TABLE_ROOM_WALL_MATERIALS:-painted_plaster,rough_concrete,red_brick,oak_panels}"
export FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M="${FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M:-5.5,7.5}"
export FLIP_TABLE_ROOM_WALL_HEIGHT_M="${FLIP_TABLE_ROOM_WALL_HEIGHT_M:-4.0,5.5}"
export FLIP_TABLE_ROOM_TILE_SIZE_M="${FLIP_TABLE_ROOM_TILE_SIZE_M:-0.35,0.9}"
export FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M="${FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M:-0.008,0.025}"
export FLIP_TABLE_ROOM_COLOR_JITTER="${FLIP_TABLE_ROOM_COLOR_JITTER:-0.08}"
export FLIP_TABLE_ROOM_FLOOR_PATTERNS="${FLIP_TABLE_ROOM_FLOOR_PATTERNS:-grid,checker,planks,border}"
export FLIP_TABLE_ROOM_WALL_PATTERNS="${FLIP_TABLE_ROOM_WALL_PATTERNS:-plain,baseboard,horizontal_stripes,vertical_panels,wainscot}"
export FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS="${FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS:-96}"
export FLIP_TABLE_LIGHT_INTENSITY_RANGE="${FLIP_TABLE_LIGHT_INTENSITY_RANGE:-450,1200}"
export FLIP_TABLE_LIGHT_COLOR_RANGE="${FLIP_TABLE_LIGHT_COLOR_RANGE:-0.82,1.0}"
export FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K="${FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K:-3800,6500}"
export FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE="${FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE:-2,4}"
export FLIP_TABLE_SUN_VISIBLE_PROBABILITY="${FLIP_TABLE_SUN_VISIBLE_PROBABILITY:-0.78}"
export FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K="${FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K:-5000,7000}"
export FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE="${FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE:-180,750}"
export FLIP_TABLE_SUN_ELEVATION_DEG="${FLIP_TABLE_SUN_ELEVATION_DEG:-18,72}"
export FLIP_TABLE_SUN_AZIMUTH_DEG="${FLIP_TABLE_SUN_AZIMUTH_DEG:--180,180}"
export FLIP_TABLE_LIGHT_EXPOSURE_RANGE="${FLIP_TABLE_LIGHT_EXPOSURE_RANGE:--0.35,0.35}"

source /opt/conda/etc/profile.d/conda.sh
conda activate robofinals
PYTHON_BIN="${CONDA_PREFIX:-/opt/conda/envs/robofinals}/bin/python"

if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "LeRobotGrootN17Policy" ]]; then
  "$GROOT_RUNTIME_SOURCE/setup_runtime.sh" "$GROOT_RUNTIME_DIR"
fi

target_wbc_adapter="$ROBOFINALS_ROOT/robofinals/core/mdp/actions/team_ramen_balanced_wbc_action.py"
cp "$BALANCED_WBC_ACTION" "$target_wbc_adapter"
WBC_CKPT_DIR="$ROBOFINALS_ROOT/robofinals/data/ckpts/nv_wbc_v0904/homie_v2"
for required_wbc in "$WBC_CKPT_DIR/stand.onnx" "$WBC_CKPT_DIR/walk.onnx"; do
  [[ -f "$required_wbc" ]] || { echo "ERROR: official WBC asset missing: $required_wbc" >&2; exit 1; }
done
export FLIP_TABLE_WBC_STAND_ONNX_SHA256="$(sha256sum "$WBC_CKPT_DIR/stand.onnx" | awk '{print $1}')"
export FLIP_TABLE_WBC_WALK_ONNX_SHA256="$(sha256sum "$WBC_CKPT_DIR/walk.onnx" | awk '{print $1}')"
export FLIP_TABLE_WBC_ADAPTER_SHA256="$(sha256sum "$target_wbc_adapter" | awk '{print $1}')"
ROBOFINALS_ROOT="$ROBOFINALS_ROOT" "$PYTHON_BIN" "$CAMERA_PATCH"

ROBOFINALS_ROOT="$ROBOFINALS_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

server_path = Path(os.environ["ROBOFINALS_ROOT"]) / "robofinals" / "scripts" / "env_server.py"
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

target_task="$ROBOFINALS_ROOT/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
backup_task="$target_task.original_assemble_table_task"
if [[ ! -f "$backup_task" ]]; then
  cp "$target_task" "$backup_task"
fi
cp "$OVERLAY_TASK" "$target_task"

target_policy="$ROBOFINALS_ROOT/policy/flip_table_eval_policy.py"
cp "$POLICY_OVERLAY" "$target_policy"
rm -rf "$ROBOFINALS_ROOT/policy/cv_rule_based"
cp -a "$CV_POLICY_PACKAGE" "$ROBOFINALS_ROOT/policy/cv_rule_based"
rm -rf "$ROBOFINALS_ROOT/policy/teleop"
cp -a "$TELEOP_PACKAGE" "$ROBOFINALS_ROOT/policy/teleop"
rm -rf "$ROBOFINALS_ROOT/policy/flow_matching"
cp -a "$FLOW_PACKAGE" "$ROBOFINALS_ROOT/policy/flow_matching"
rm -rf "$ROBOFINALS_ROOT/policy/rlpd"
cp -a "$RLPD_PACKAGE" "$ROBOFINALS_ROOT/policy/rlpd"
rm -rf "$ROBOFINALS_ROOT/policy/team_ramen_groot"
cp -a "$GROOT_SHARED_PACKAGE" "$ROBOFINALS_ROOT/policy/team_ramen_groot"
ROBOFINALS_ROOT="$ROBOFINALS_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

init_path = Path(os.environ["ROBOFINALS_ROOT"]) / "policy" / "__init__.py"
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

config_path="/tmp/flip_table_eval.yml"
cp "$CONFIG_HOST" "$config_path"
"$PYTHON_BIN" - "$config_path" <<'PY'
import os
from pathlib import Path
import sys

import yaml

raw = os.environ.get("FLIP_TABLE_CALIBRATION_NUM_ENVS", "").strip()
if raw:
    count = int(raw)
    if not 1 <= count <= 64:
        raise SystemExit("FLIP_TABLE_CALIBRATION_NUM_ENVS must be in [1, 64]")
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["env_cfg"]["num_envs"] = count
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

if [[ "$FLIP_TABLE_PREPARE_ASSEMBLED_SCENE" == "true" ]]; then
  read -r layout_source layout_output < <("$PYTHON_BIN" - "$config_path" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(config["env_cfg"]["layout"])
output = source.with_name(f"{source.stem}_flip_table_assembled.usd")
print(source, output)
PY
  )
  scene_prepare_args=(--source "$layout_source" --output "$layout_output")
  if [[ "$FLIP_TABLE_SIMPLIFY_WHITE_COLLISION" == "true" ]]; then
    scene_prepare_args+=(--simplify-white-collision)
  fi
  if [[ "${FLIP_TABLE_POLICY_NAME:-}" == "Dex1ForceCalibrationPolicy" ]]; then
    scene_prepare_args+=(--dex1-force-calibration)
  fi
  "$PYTHON_BIN" "$SCENE_PREPARE_TOOL" "${scene_prepare_args[@]}"
  "$PYTHON_BIN" - "$config_path" "$layout_output" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["env_cfg"]["layout"] = sys.argv[2]
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
fi

cd "$ROBOFINALS_ROOT"

if [[ -e eval_result && ! -L eval_result ]]; then
  rm -rf eval_result
fi
ln -sfn "$OUTPUT_DIR" eval_result

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
  rm -f "$FLIP_TABLE_GROOT_SOCKET"
  "$GROOT_RUNTIME_DIR/.venv/bin/python" \
    "$GROOT_RUNTIME_SOURCE/groot_inference_server.py" \
    --checkpoint "$FLIP_TABLE_POLICY_CHECKPOINT" \
    --socket "$FLIP_TABLE_GROOT_SOCKET" \
    --device "$FLIP_TABLE_GROOT_DEVICE" \
    --n-action-steps "$FLIP_TABLE_GROOT_N_ACTION_STEPS" \
    --seed "$FLIP_TABLE_GROOT_INFERENCE_SEED" \
    > "$OUTPUT_DIR/groot_inference_server.log" 2>&1 &
  GROOT_SERVER_PID=$!
  GROOT_DEADLINE=$((SECONDS + 1800))
  while [[ ! -S "$FLIP_TABLE_GROOT_SOCKET" ]]; do
    if ! kill -0 "$GROOT_SERVER_PID" >/dev/null 2>&1; then
      echo "ERROR: GR00T inference server exited during startup" >&2
      cat "$OUTPUT_DIR/groot_inference_server.log" >&2
      exit 1
    fi
    if (( SECONDS >= GROOT_DEADLINE )); then
      echo "ERROR: timed out waiting for GR00T inference server: $FLIP_TABLE_GROOT_SOCKET" >&2
      cat "$OUTPUT_DIR/groot_inference_server.log" >&2
      exit 1
    fi
    sleep 1
  done
fi

if [[ "${FLIP_TABLE_POLICY_NAME:-}" =~ ^(AvpTeleopPolicy|TeleopPerformanceBenchmarkPolicy|Dex1ForceCalibrationPolicy)$ ]]; then
  in_process_args=(
    --config "$config_path"
    --policy-name "$FLIP_TABLE_POLICY_NAME"
    --test-num "$TEST_NUM"
  )
  if [[ -n "${FLIP_TABLE_TIME_OUT_LIMIT:-}" ]]; then
    in_process_args+=(--time-out-limit "$FLIP_TABLE_TIME_OUT_LIMIT")
  fi
  "$PYTHON_BIN" "$IN_PROCESS_EVAL_TOOL" "${in_process_args[@]}"
  echo "Evaluation outputs: $OUTPUT_DIR"
  exit 0
fi

if [[ "${FLIP_TABLE_PERSISTENT_EVAL_WORKER:-false}" == "true" ]]; then
  "$PYTHON_BIN" "$PERSISTENT_EVAL_WORKER" \
    --config "$config_path" \
    --job-root "$OUTPUT_DIR"
  exit 0
fi

server_args=(--enable_cameras)
if [[ "${FLIP_TABLE_SIM_LIVESTREAM:-true}" == "true" ]]; then
  server_args+=(--livestream 2)
fi

"$PYTHON_BIN" robofinals/scripts/env_server.py "${server_args[@]}" &
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

"$PYTHON_BIN" - <<'PY'
import socket
import time

deadline = time.time() + 300
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", 50000)) == 0:
            raise SystemExit(0)
    time.sleep(2)
raise SystemExit("Timed out waiting for RoboFinals env_server on 127.0.0.1:50000")
PY

overrides=(--test_num "$TEST_NUM" --record_video True)
if [[ -n "${FLIP_TABLE_POLICY_NAME:-}" ]]; then
  overrides+=(--policy_name "$FLIP_TABLE_POLICY_NAME")
fi
if [[ -n "${FLIP_TABLE_TIME_OUT_LIMIT:-}" ]]; then
  overrides+=(--time_out_limit "$FLIP_TABLE_TIME_OUT_LIMIT")
fi
if [[ -n "${FLIP_TABLE_POLICY_CHECKPOINT:-}" ]]; then
  overrides+=(--checkpoint "$FLIP_TABLE_POLICY_CHECKPOINT")
fi

"$PYTHON_BIN" robofinals/scripts/policy/eval_policy.py \
  --config "$config_path" \
  --overrides "${overrides[@]}"

echo "Evaluation outputs: $OUTPUT_DIR"
