#!/usr/bin/env bash
set -euo pipefail

# Canonical entrypoint for the maintained V1 flip-table experiments.  It keeps
# the organizer image immutable on disk, applies only repository-owned overlays,
# and records the exact source/configuration used by each run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/model/flip_table_reinforcement_learning"
SIM_DIR="$ROOT_DIR/evaluate/flip_table_simulation"
ROBOFINALS_ROOT="${ROBOFINALS_ROOT:-/workspace/robofinals}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/robofinals/bin/python}"
MODE="${1:-}"
OUTPUT_DIR="${FLIP_TABLE_RL_OUTPUT_DIR:-$ROOT_DIR/outputs/flip_table_reinforcement_learning}"
STAGE="${FLIP_TABLE_RL_STAGE:-full}"
NUM_ENVS="${FLIP_TABLE_RL_NUM_ENVS:-1}"
MAX_ITERATIONS="${FLIP_TABLE_RL_MAX_ITERATIONS:-100}"
POLICY_MODE="${FLIP_TABLE_RL_POLICY_MODE:-visual}"
DEMO_PATH="${FLIP_TABLE_RL_DEMO_ACTION_PATH:-$ROOT_DIR/.checkpoints/flip_table_episode0_actions.json}"

usage() {
  cat <<'EOF'
Usage: run_train_in_container.sh MODE

Supported modes:
  audit_contract       Verify V1 physics, action, sensor, and reset contracts.
  audit_partial_reset  Verify per-environment reset isolation.
  smoke                Replay the recorded action prior as an environment smoke test.
  evaluate             Evaluate a legacy PPO checkpoint.
  train                Train the legacy PPO baseline.
  evaluate_rlpd_stage  Evaluate a Flow checkpoint or Flow+RLPD checkpoint.
  train_rlpd           Train the Flow+RLPD comparison baseline.

All successful demonstrations and Mimic integration belong to the next branch.
This entrypoint intentionally contains no CEM, fixed-trajectory, or privileged
teacher-search mode.
EOF
}

case "$MODE" in
  audit_contract|audit_partial_reset|smoke|evaluate|train|evaluate_rlpd_stage|train_rlpd) ;;
  -h|--help|help|'') usage; exit 0 ;;
  *) echo "ERROR: unsupported mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FLIP_TABLE_RL_NUM_ENVS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FLIP_TABLE_RL_MAX_ITERATIONS must be a positive integer" >&2
  exit 2
fi
if [[ "$MODE" == "train_rlpd" || "$MODE" == "evaluate_rlpd_stage" ]]; then
  POLICY_MODE="visual"
  export FLIP_TABLE_RLPD_USE_FLOW_BASE=true
  export FLIP_TABLE_RL_CONTROL_HZ="${FLIP_TABLE_RLPD_SIM_CONTROL_HZ:-50}"
fi
if [[ "$MODE" == "audit_contract" ]]; then
  NUM_ENVS=1
  export FLIP_TABLE_RLPD_USE_FLOW_BASE=true
  export FLIP_TABLE_RL_CONTROL_HZ="${FLIP_TABLE_SIM_AUDIT_CONTROL_HZ:-50}"
fi
if [[ "$MODE" == "evaluate_rlpd_stage" && "$NUM_ENVS" != "1" ]]; then
  echo "ERROR: evaluate_rlpd_stage requires FLIP_TABLE_RL_NUM_ENVS=1" >&2
  exit 2
fi

if [[ "$POLICY_MODE" != "visual" && "$POLICY_MODE" != "state" ]]; then
  echo "ERROR: FLIP_TABLE_RL_POLICY_MODE must be visual or state" >&2
  exit 2
fi
if [[ ! -d "$ROBOFINALS_ROOT" ]]; then
  echo "ERROR: RoboFinals root not found: $ROBOFINALS_ROOT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: simulator Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

ROBOFINALS_REAL_ROOT="$(realpath "$ROBOFINALS_ROOT")"
OFFICIAL_V1_BACKUP_ROOT="${FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT:-$(dirname "$ROBOFINALS_REAL_ROOT")/robofinalsbak}"
if [[ ! -f "$OFFICIAL_V1_BACKUP_ROOT/robofinals/core/robots/unitree/g1.py" || ! -f "$OFFICIAL_V1_BACKUP_ROOT/robofinals/core/robots/unitree/assets_cfg.py" ]]; then
  echo "ERROR: immutable RoboFinals-IKEA-V1 robot backup is missing under $OFFICIAL_V1_BACKUP_ROOT" >&2
  exit 1
fi

prepare_demo_prior() {
  if [[ -f "$DEMO_PATH" ]]; then
    return
  fi
  if [[ "${FLIP_TABLE_RL_PREPARE_DEMO_IF_MISSING:-true}" != "true" ]]; then
    echo "ERROR: action prior is missing: $DEMO_PATH" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$DEMO_PATH")"
  args=(
    --repo-id "${FLIP_TABLE_RL_DATASET_REPO_ID:-Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1}" \
    --episode "${FLIP_TABLE_RL_DEMO_EPISODE:-0}" \
    --output "$DEMO_PATH"
  )
  if [[ -n "${FLIP_TABLE_RL_DATASET_REVISION:-}" ]]; then
    args+=(--revision "$FLIP_TABLE_RL_DATASET_REVISION")
  fi
  if [[ -n "${FLIP_TABLE_RL_SOURCE_DATASET_ROOT:-}" ]]; then
    args+=(--dataset-root "$FLIP_TABLE_RL_SOURCE_DATASET_ROOT")
  fi
  "$PYTHON_BIN" "$FEATURE_DIR/scripts/prepare_demo_actions.py" "${args[@]}"
}

prepare_demo_prior

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all,graphics,display}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONPATH="$ROOT_DIR:$ROBOFINALS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT="$OFFICIAL_V1_BACKUP_ROOT"
export FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES=true
export FLIP_TABLE_RL_DEMO_ACTION_PATH="$DEMO_PATH"
export FLIP_TABLE_RL_STAGE="$STAGE"
export FLIP_TABLE_RL_CONTROL_HZ="${FLIP_TABLE_RL_CONTROL_HZ:-50}"
export FLIP_TABLE_RL_DEMO_HZ="${FLIP_TABLE_RL_DEMO_HZ:-30}"
export FLIP_TABLE_RL_PHASE_MODE="${FLIP_TABLE_RL_PHASE_MODE:-clock}"
export FLIP_TABLE_RL_DEMO_START_INDEX="${FLIP_TABLE_RL_DEMO_START_INDEX:-0}"
export FLIP_TABLE_RL_DEMO_END_INDEX="${FLIP_TABLE_RL_DEMO_END_INDEX:-866}"
export FLIP_TABLE_RL_EPISODE_SECONDS="${FLIP_TABLE_RL_EPISODE_SECONDS:-32.0}"
export FLIP_TABLE_RL_ZERO_INIT_POLICY_OUTPUT="${FLIP_TABLE_RL_ZERO_INIT_POLICY_OUTPUT:-true}"
export FLIP_TABLE_RL_POLICY_RESIDUAL_RANGE_MULTIPLIER="${FLIP_TABLE_RL_POLICY_RESIDUAL_RANGE_MULTIPLIER:-1.0}"

if [[ "$MODE" == audit_contract || "$MODE" == audit_partial_reset || "$MODE" == smoke ]]; then
  export FLIP_TABLE_RL_EVAL_MODE=fixed
fi
if [[ "${FLIP_TABLE_RL_EVAL_MODE:-randomized}" == "fixed" ]]; then
  export FLIP_TABLE_RL_RANDOMIZATION_LEVEL=0
  export FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS=false
  export FLIP_TABLE_RANDOMIZE_ROOM=false
  export FLIP_TABLE_RANDOMIZE_ROOM_PROPS=false
  export FLIP_TABLE_RANDOMIZE_LIGHTING=false
  export FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE=false
  export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS=false
  export FLIP_TABLE_RL_RANDOMIZE_MASS=false
  export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY=false
  export FLIP_TABLE_RL_ENABLE_SENSOR_NOISE=false
  export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS=0
  export FLIP_TABLE_TABLE_LONG_RANGE_M=0
  export FLIP_TABLE_TABLE_DEPTH_RANGE_M=0
  export FLIP_TABLE_TABLE_YAW_RANGE_RAD=0
  export FLIP_TABLE_ROBOT_DISTANCE_RANGE_M=0
  export FLIP_TABLE_ROBOT_LATERAL_RANGE_M=0
  export FLIP_TABLE_ROBOT_YAW_RANGE_RAD=0
elif [[ "${FLIP_TABLE_RL_EVAL_MODE:-randomized}" == "randomized" ]]; then
  export FLIP_TABLE_RL_RANDOMIZATION_LEVEL="${FLIP_TABLE_RL_RANDOMIZATION_LEVEL:-1.0}"
  export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS="${FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS:-true}"
  export FLIP_TABLE_RL_RANDOMIZE_MASS=false
  export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY="${FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY:-true}"
  export FLIP_TABLE_RANDOMIZE_LIGHTING="${FLIP_TABLE_RANDOMIZE_LIGHTING:-true}"
else
  echo "ERROR: FLIP_TABLE_RL_EVAL_MODE must be fixed or randomized" >&2
  exit 2
fi
if [[ "$POLICY_MODE" == state ]]; then
  export FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS=false
  export FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY=false
fi

TASK_TARGET="$ROBOFINALS_ROOT/robofinals_tasks/local_auto_tasks/assemble_table_task.py"
if [[ ! -f "${TASK_TARGET}.original_flip_table_rl" ]]; then
  cp "$TASK_TARGET" "${TASK_TARGET}.original_flip_table_rl"
fi
cp "$SIM_DIR/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py" "$TASK_TARGET"
rm -rf "$ROBOFINALS_ROOT/robofinals_rl/flip_table"
cp -a "$FEATURE_DIR/container_overlay/robofinals_rl/flip_table" "$ROBOFINALS_ROOT/robofinals_rl/flip_table"

"$PYTHON_BIN" - "$ROBOFINALS_ROOT/robofinals_rl/__init__.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
line = "from . import flip_table as flip_table\n"
text = path.read_text(encoding="utf-8")
if line not in text:
    path.write_text(text.rstrip() + "\n\n" + line, encoding="utf-8")
PY

"$PYTHON_BIN" "$SIM_DIR/container_overlay/patches/patch_g1_global_camera.py"
"$PYTHON_BIN" "$FEATURE_DIR/scripts/patch_rl_train_full_scene.py" --robofinals-root "$ROBOFINALS_ROOT"

SCENE_SOURCE="/workspace/IROS_IKEA_V13_20260702/Scene02.usd"
SCENE_OUTPUT="/workspace/IROS_IKEA_V13_20260702/Scene02_flip_table_assembled.usd"
"$PYTHON_BIN" "$SIM_DIR/tools/prepare_assembled_table_scene.py" --source "$SCENE_SOURCE" --output "$SCENE_OUTPUT"

CONFIG_TARGET="$ROBOFINALS_ROOT/configs/rl/skrl/flip_table_rl.yml"
AGENT_CONFIG_TARGET="$ROBOFINALS_ROOT/robofinals_rl/flip_table/agents/skrl_ppo_cfg.yaml"
cp "$FEATURE_DIR/configs/flip_table_rl.yml" "$CONFIG_TARGET"
"$PYTHON_BIN" - "$CONFIG_TARGET" "$AGENT_CONFIG_TARGET" "$SCENE_OUTPUT" "$NUM_ENVS" "$MAX_ITERATIONS" "$OUTPUT_DIR" "$STAGE" "${FLIP_TABLE_RL_CHECKPOINT:-}" "$POLICY_MODE" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config_path, agent_path = map(Path, sys.argv[1:3])
scene, num_envs, iterations, output_dir, stage, checkpoint, policy_mode = sys.argv[3:]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["layout"] = scene
config["num_envs"] = int(num_envs)
config["rl"] = "FlipTableResidualVisualRL" if policy_mode == "visual" else "FlipTableResidualStateRL"
config["enable_cameras"] = policy_mode == "visual"
if checkpoint:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    config["checkpoint"] = str(path.resolve())
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
timesteps = int(iterations) * int(agent["agent"]["rollouts"])
if timesteps <= 0:
    raise ValueError("FLIP_TABLE_RL_MAX_ITERATIONS must be positive")
agent["trainer"]["timesteps"] = timesteps
policy = agent["models"]["policy"]
initial_log_std = float(os.environ.get("FLIP_TABLE_RL_INITIAL_LOG_STD", policy["initial_log_std"]))
if not float(policy["min_log_std"]) <= initial_log_std <= float(policy["max_log_std"]):
    raise ValueError("FLIP_TABLE_RL_INITIAL_LOG_STD must be within policy bounds")
policy["initial_log_std"] = initial_log_std
agent["agent"]["experiment"]["directory"] = str(Path(output_dir).resolve())
agent["agent"]["experiment"]["experiment_name"] = f"{stage}_{policy_mode}_residual_ppo"
agent_path.write_text(yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
PY

RUN_DIR="${FLIP_TABLE_RL_RUN_DIR:-$OUTPUT_DIR/$MODE/$STAGE}"
# Evaluation and RLPD training intentionally reject a pre-existing output
# directory. Keep the manifest beside that directory, not inside it.
RUN_MANIFEST_PATH="${FLIP_TABLE_RL_RUN_MANIFEST_PATH:-$RUN_DIR.run_manifest.json}"
"$PYTHON_BIN" "$FEATURE_DIR/scripts/write_run_manifest.py" \
  --output "$RUN_MANIFEST_PATH" \
  --repository "$ROOT_DIR" \
  --mode "$MODE" \
  --stage "$STAGE" \
  --policy-mode "$POLICY_MODE" \
  --source-root "$FEATURE_DIR" \
  --source-root "$SIM_DIR/container_overlay" \
  --input "demo=$DEMO_PATH" \
  --input "task_config=$CONFIG_TARGET" \
  --input "agent_config=$AGENT_CONFIG_TARGET" \
  --input "assembled_scene=$SCENE_OUTPUT"

cd "$ROBOFINALS_ROOT"
case "$MODE" in
  audit_contract)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_simulation_contract.py" \
      --task_config flip_table_rl --num-envs 1 --sim-control-hz "$FLIP_TABLE_RL_CONTROL_HZ" \
      --seed "${FLIP_TABLE_SIM_AUDIT_SEED:-42}" \
      --friction-steps "${FLIP_TABLE_SIM_AUDIT_FRICTION_STEPS:-20}" \
      --friction-force-n "${FLIP_TABLE_SIM_AUDIT_FRICTION_FORCE_N:-5.0}" \
      --output "${FLIP_TABLE_SIM_AUDIT_OUTPUT:-$RUN_DIR/simulation_contract_audit.json}" --headless
    ;;
  audit_partial_reset)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/audit_partial_reset_contract.py" \
      --task_config flip_table_rl --num-envs "$NUM_ENVS" \
      --settle_steps "${FLIP_TABLE_RL_RESET_SETTLE_STEPS:-4}" \
      --output "$RUN_DIR/partial_reset_contract.json"
    ;;
  smoke)
    args=(--task_config flip_table_rl --num_envs "$NUM_ENVS" --steps "${FLIP_TABLE_RL_SMOKE_STEPS:-200}" --output "$RUN_DIR/smoke.json")
    [[ "$POLICY_MODE" == visual ]] && args+=(--enable_cameras)
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/smoke_test_env.py" "${args[@]}"
    ;;
  evaluate)
    [[ -n "${FLIP_TABLE_RL_CHECKPOINT:-}" ]] || { echo "ERROR: evaluate requires FLIP_TABLE_RL_CHECKPOINT" >&2; exit 1; }
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/evaluate_policy.py" \
      --task_config flip_table_rl --num_envs 1 --episodes "${FLIP_TABLE_RL_EVAL_EPISODES:-1}" \
      --checkpoint "$FLIP_TABLE_RL_CHECKPOINT" --output "$RUN_DIR" --enable_cameras
    ;;
  evaluate_rlpd_stage)
    if [[ -n "${FLIP_TABLE_RLPD_COMBINED_CHECKPOINT:-}" ]]; then
      checkpoint_args=(--checkpoint "$FLIP_TABLE_RLPD_COMBINED_CHECKPOINT")
    elif [[ -n "${FLIP_TABLE_FLOW_CHECKPOINT:-}" ]]; then
      checkpoint_args=(--flow-checkpoint "$FLIP_TABLE_FLOW_CHECKPOINT")
    else
      echo "ERROR: evaluate_rlpd_stage requires FLIP_TABLE_FLOW_CHECKPOINT or FLIP_TABLE_RLPD_COMBINED_CHECKPOINT" >&2
      exit 1
    fi
    args=(--task_config flip_table_rl --output "$RUN_DIR" --episodes "${FLIP_TABLE_RLPD_EVAL_EPISODES:-3}" --num-envs 1 --policy-hz "${FLIP_TABLE_RLPD_POLICY_HZ:-30}" --sim-control-hz "$FLIP_TABLE_RL_CONTROL_HZ" --max-sim-steps "${FLIP_TABLE_RLPD_EVAL_MAX_SIM_STEPS:-0}" --hard-reset-finger-force-n "${FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N:-15.1}" --reset-settle-steps "${FLIP_TABLE_RLPD_RESET_SETTLE_STEPS:-4}" --residual-mode "${FLIP_TABLE_RLPD_EVAL_RESIDUAL_MODE:-policy}" --constant-residual="${FLIP_TABLE_RLPD_EVAL_CONSTANT_RESIDUAL:-}" --seed "${FLIP_TABLE_RLPD_SEED:-42}" --episode-seeds="${FLIP_TABLE_RLPD_EVAL_EPISODE_SEEDS:-}" --enable_cameras)
    if [[ "${FLIP_TABLE_RLPD_RECORD_VIDEO:-true}" == "true" ]]; then
      args+=(--record-video)
    elif [[ "${FLIP_TABLE_RLPD_RECORD_VIDEO}" != "false" ]]; then
      echo "ERROR: FLIP_TABLE_RLPD_RECORD_VIDEO must be true or false" >&2
      exit 2
    fi
    if [[ "${FLIP_TABLE_RLPD_STOP_ON_CURRICULUM_STAGE_SUCCESS:-false}" == "true" ]]; then
      args+=(--stop-on-curriculum-stage-success)
    fi
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/evaluate_flow_residual_rlpd.py" "${checkpoint_args[@]}" "${args[@]}"
    ;;
  train_rlpd)
    [[ -n "${FLIP_TABLE_FLOW_CHECKPOINT:-}" ]] || { echo "ERROR: train_rlpd requires FLIP_TABLE_FLOW_CHECKPOINT" >&2; exit 1; }
    args=(--task_config flip_table_rl --flow-checkpoint "$FLIP_TABLE_FLOW_CHECKPOINT" --output "$RUN_DIR" --num-envs "$NUM_ENVS" --total-transitions "${FLIP_TABLE_RLPD_TOTAL_TRANSITIONS:-1000000}" --prior-transitions "${FLIP_TABLE_RLPD_PRIOR_TRANSITIONS:-50000}" --learning-starts "${FLIP_TABLE_RLPD_LEARNING_STARTS:-2000}" --batch-size "${FLIP_TABLE_RLPD_BATCH_SIZE:-256}" --update-to-data-ratio "${FLIP_TABLE_RLPD_UTD_RATIO:-1.0}" --online-replay-capacity "${FLIP_TABLE_RLPD_ONLINE_REPLAY_CAPACITY:-1000000}" --prior-replay-capacity "${FLIP_TABLE_RLPD_PRIOR_REPLAY_CAPACITY:-100000}" --random-residual-std "${FLIP_TABLE_RLPD_RANDOM_RESIDUAL_STD:-0.15}" --critic-warmup-updates "${FLIP_TABLE_RLPD_CRITIC_WARMUP_UPDATES:-1000}" --prior-bc-weight "${FLIP_TABLE_RLPD_PRIOR_BC_WEIGHT:-10.0}" --reference-bc-weight "${FLIP_TABLE_RLPD_REFERENCE_BC_WEIGHT:-20.0}" --actor-learning-rate "${FLIP_TABLE_RLPD_ACTOR_LEARNING_RATE:-0.0001}" --actor-q-normalization "${FLIP_TABLE_RLPD_ACTOR_Q_NORMALIZATION:-1.0}" --initial-temperature "${FLIP_TABLE_RLPD_INITIAL_TEMPERATURE:-0.001}" --prior-residual="${FLIP_TABLE_RLPD_PRIOR_RESIDUAL:-}" --prior-action-source "${FLIP_TABLE_RLPD_PRIOR_ACTION_SOURCE:-constant}" --policy-hz "${FLIP_TABLE_RLPD_POLICY_HZ:-30}" --sim-control-hz "$FLIP_TABLE_RL_CONTROL_HZ" --reset-settle-steps "${FLIP_TABLE_RLPD_RESET_SETTLE_STEPS:-4}" --log-every-transitions "${FLIP_TABLE_RLPD_LOG_EVERY_TRANSITIONS:-10000}" --save-every-transitions "${FLIP_TABLE_RLPD_SAVE_EVERY_TRANSITIONS:-250000}" --seed "${FLIP_TABLE_RLPD_SEED:-42}" --enable_cameras)
    if [[ "${FLIP_TABLE_RLPD_REUSE_PREFIX_STATE:-false}" == "true" ]]; then
      args+=(--reuse-prefix-state)
    fi
    if [[ -n "${FLIP_TABLE_RLPD_ACTOR_INIT_CHECKPOINT:-}" ]]; then
      args+=(--actor-init-checkpoint "$FLIP_TABLE_RLPD_ACTOR_INIT_CHECKPOINT")
    fi
    if [[ -n "${FLIP_TABLE_RLPD_RESUME:-}" ]]; then
      args+=(--resume "$FLIP_TABLE_RLPD_RESUME")
    fi
    exec "$PYTHON_BIN" "$FEATURE_DIR/scripts/train_flow_residual_rlpd.py" "${args[@]}"
    ;;
  train)
    args=(--task_config flip_table_rl --num_envs "$NUM_ENVS")
    [[ "$POLICY_MODE" == visual ]] && args+=(--enable_cameras)
    exec "$PYTHON_BIN" robofinals/scripts/rl/train.py "${args[@]}"
    ;;
esac
