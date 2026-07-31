#!/usr/bin/env bash
set -euo pipefail

# Run the maintained RoboFinals-in-container entrypoint on a local Linux host.
# The image itself supplies /workspace/robofinals and the Scene02 asset; this
# wrapper bind-mounts only repository-owned code and outputs. A fresh container
# starts from the official image, then saves the two pristine V1 robot sources
# required by the patcher before any repository overlay touches the writable
# container layer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
INNER_RUNNER="/workspace/iros_2026_ramen/model/flip_table_reinforcement_learning/run_train_in_container.sh"

if ! docker --context default info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: Docker is unavailable to this shell. Log out/in after the installer adds
your account to the docker group, or run this command from a shell with Docker
access.
EOF
  exit 1
fi

docker_tty_args=()
if [[ -t 0 ]]; then
  docker_tty_args=(-it)
fi

docker_args=(
  --gpus all
  --network host
  --ipc host
  --privileged
  -e "LOCAL_UID=$(id -u)"
  -e "LOCAL_GID=$(id -g)"
  -v "$ROOT_DIR:/workspace/iros_2026_ramen"
  -w /workspace/iros_2026_ramen
)

# A prepared demonstration prior is normally generated outside the repository.
# Bind it to a stable in-container path so a local host path never leaks into
# the runner.  Without it, the inner runner may fetch the private dataset and
# therefore needs one of the standard Hugging Face token variables below.
if [[ -n "${FLIP_TABLE_RL_DEMO_ACTION_PATH:-}" ]]; then
  demo_action_path="$(realpath "$FLIP_TABLE_RL_DEMO_ACTION_PATH")"
  if [[ ! -f "$demo_action_path" ]]; then
    echo "ERROR: FLIP_TABLE_RL_DEMO_ACTION_PATH is not a file: $demo_action_path" >&2
    exit 1
  fi
  docker_args+=(
    -v "$demo_action_path:/workspace/flip_table_inputs/demo_actions.json:ro"
    -e FLIP_TABLE_RL_DEMO_ACTION_PATH=/workspace/flip_table_inputs/demo_actions.json
  )
fi

# The repository is the only writable host bind mount.  Translate an output
# path inside it to the container namespace and reject paths that would be
# written only into the disposable container layer.
if [[ -n "${FLIP_TABLE_RL_OUTPUT_DIR:-}" ]]; then
  if [[ "$FLIP_TABLE_RL_OUTPUT_DIR" = /* ]]; then
    host_output_dir="$(realpath -m "$FLIP_TABLE_RL_OUTPUT_DIR")"
  else
    host_output_dir="$(realpath -m "$ROOT_DIR/$FLIP_TABLE_RL_OUTPUT_DIR")"
  fi
  case "$host_output_dir" in
    "$ROOT_DIR"/*)
      container_output_dir="/workspace/iros_2026_ramen/${host_output_dir#"$ROOT_DIR"/}"
      docker_args+=(-e "FLIP_TABLE_RL_OUTPUT_DIR=$container_output_dir")
      ;;
    *)
      echo "ERROR: FLIP_TABLE_RL_OUTPUT_DIR must be inside $ROOT_DIR" >&2
      exit 1
      ;;
  esac
fi
for token_variable in HF_TOKEN HUGGINGFACE_HUB_TOKEN; do
  if [[ -n "${!token_variable:-}" ]]; then
    docker_args+=(-e "$token_variable")
  fi
done

# Keep the local wrapper behavior identical to invoking the in-container
# runner directly. Docker does not inherit arbitrary host variables, so pass
# only the project-scoped settings that define a flip-table run.
while IFS='=' read -r variable_name _; do
  case "$variable_name" in
    FLIP_TABLE_RL_DEMO_ACTION_PATH|FLIP_TABLE_RL_OUTPUT_DIR)
      # Host paths above are deliberately remapped to container paths.
      ;;
    FLIP_TABLE_*|ROBOFINALS_*|WANDB_*)
      docker_args+=(-e "$variable_name")
      ;;
  esac
done < <(env)

if [[ -n "${DISPLAY:-}" ]]; then
  DOCKER_XAUTH="${DOCKER_XAUTH:-${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}}"
  if [[ ! -f "$DOCKER_XAUTH" ]]; then
    echo "ERROR: Xauthority file not found: $DOCKER_XAUTH" >&2
    exit 1
  fi
  xhost +SI:localuser:root >/dev/null 2>&1 || true
  docker_args+=(
    -e "DISPLAY=$DISPLAY"
    -e XAUTHORITY=/tmp/docker.xauth
    -v "$DOCKER_XAUTH:/tmp/docker.xauth:ro"
  )
fi

docker --context default run --rm "${docker_tty_args[@]}" "${docker_args[@]}" "$IMAGE" \
  -lc '
    set -euo pipefail
    backup=/workspace/robofinalsbak/robofinals/core/robots/unitree
    mkdir -p "$backup"
    cp -a /workspace/robofinals/robofinals/core/robots/unitree/g1.py "$backup/g1.py"
    cp -a /workspace/robofinals/robofinals/core/robots/unitree/assets_cfg.py "$backup/assets_cfg.py"
    mkdir -p /workspace/iros_2026_ramen/.checkpoints \
      /workspace/iros_2026_ramen/outputs
    chown -R "$LOCAL_UID:$LOCAL_GID" \
      /workspace/iros_2026_ramen/.checkpoints \
      /workspace/iros_2026_ramen/outputs
    exec "$@"
  ' bash "$INNER_RUNNER" "$@"
