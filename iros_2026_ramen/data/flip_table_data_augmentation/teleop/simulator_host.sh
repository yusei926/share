#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
shift || true

case "$action" in
  launch)
    [[ $# -eq 12 ]] || { echo "ERROR: simulator launch requires 12 arguments" >&2; exit 2; }
    stage="$1"
    output="$2"
    port="$3"
    dr_level="$4"
    seed="$5"
    image="$6"
    digest="$7"
    container="$8"
    physics_hz="$9"
    render_interval="${10}"
    simplify_white_collision="${11}"
    success_check_interval_steps="${12}"

    mkdir -p "$output"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      docker pull "$image" >/dev/null
    fi
    if ! docker image inspect --format '{{json .RepoDigests}}' "$image" \
      | grep -Fq "@$digest"; then
      echo "ERROR: RoboFinals V1 image digest differs from $digest" >&2
      exit 1
    fi
    docker rm -f "$container" >/dev/null 2>&1 || true
    nohup env \
      DISPLAY="${DISPLAY:-:0}" \
      ROBOFINALS_IMAGE="paperc/robofinals@$digest" \
      FLIP_TABLE_POLICY_NAME=AvpTeleopPolicy \
      FLIP_TABLE_TELEOP_PERSISTENT=true \
      FLIP_TABLE_TEST_NUM=1 \
      FLIP_TABLE_TIME_OUT_LIMIT=360000 \
      FLIP_TABLE_SIM_OUTPUT_DIR="$output" \
      FLIP_TABLE_DOCKER_CONTAINER_NAME="$container" \
      FLIP_TABLE_TELEOP_PORT="$port" \
      FLIP_TABLE_TELEOP_PREVIEW_HZ="${FLIP_TABLE_TELEOP_PREVIEW_HZ:-24}" \
      FLIP_TABLE_RL_RANDOMIZATION_LEVEL="$dr_level" \
      FLIP_TABLE_TABLE_YAW_RANGE_RAD=3.141592653589793 \
      FLIP_TABLE_SIM_BODY_MODE=balanced_wbc \
      FLIP_TABLE_LOCK_LOWER_BODY=false \
      FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS=base_,hip,knee,ankle,waist \
      FLIP_TABLE_REQUIRE_WAIST_LOCK=false \
      FLIP_TABLE_LOCK_ROBOT_ROOT=false \
      FLIP_TABLE_FIX_ROOT_LINK=false \
      FLIP_TABLE_RL_RANDOMIZE_MASS=false \
      FLIP_TABLE_EVAL_RANDOMIZE_MASS=false \
      FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=false \
      FLIP_TABLE_SIM_PHYSICS_HZ="$physics_hz" \
      FLIP_TABLE_SIM_RENDER_INTERVAL="$render_interval" \
      FLIP_TABLE_SIMPLIFY_WHITE_COLLISION="$simplify_white_collision" \
      FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS="$success_check_interval_steps" \
      FLIP_TABLE_EVAL_MODE=randomized \
      FLIP_TABLE_EVAL_SEED="$seed" \
      "$stage/evaluate/flip_table_simulation/run_eval.sh" \
      >"$output/simulator.log" 2>&1 </dev/null &
    echo $!
    ;;
  stop)
    [[ $# -eq 3 ]] || { echo "ERROR: simulator stop requires 3 arguments" >&2; exit 2; }
    pid="$1"
    stage="$2"
    container="$3"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
      command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
      if [[ "$command_line" == *"$stage/evaluate/flip_table_simulation/run_eval.sh"* ]]; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    fi
    docker rm -f "$container" >/dev/null 2>&1 || true
    ;;
  state)
    [[ $# -eq 3 ]] || { echo "ERROR: simulator state requires 3 arguments" >&2; exit 2; }
    pid="$1"
    port="$2"
    container="$3"
    process_running=false
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
      process_running=true
    fi
    if ss -H -ltn "sport = :$port" | grep -q .; then
      echo ready
    elif [[ "$process_running" == true ]] \
      || [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" == true ]]; then
      echo waiting
    else
      echo failed
    fi
    ;;
  *)
    echo "Usage: $(basename "$0") launch|stop|state ..." >&2
    exit 2
    ;;
esac
