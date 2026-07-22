#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/workspace/iros_2026_ramen}"
SSH_PORT="${VAST_PORT:-${SSH_PORT:-22}}"
LOCAL_CHECKPOINT="${FLIP_TABLE_POLICY_CHECKPOINT:-}"
REMOTE_CHECKPOINT_DIR="${REMOTE_POLICY_CHECKPOINT_DIR:-$REMOTE_REPO_DIR/.checkpoints/act_flip_table_upper_body}"
LOCAL_FLOW_CHECKPOINT="${FLIP_TABLE_FLOW_CHECKPOINT:-}"
REMOTE_FLOW_CHECKPOINT_DIR="${REMOTE_FLOW_CHECKPOINT_DIR:-$REMOTE_REPO_DIR/.checkpoints/flow_matching_flip_table}"
LOCAL_RLPD_COMBINED_CHECKPOINT="${FLIP_TABLE_RLPD_COMBINED_CHECKPOINT:-}"
REMOTE_RLPD_COMBINED_CHECKPOINT_DIR="${REMOTE_RLPD_COMBINED_CHECKPOINT_DIR:-$REMOTE_REPO_DIR/.checkpoints/flow_residual_rlpd_flip_table}"

usage() {
  cat <<'EOF'
Usage:
  evaluate/flip_table_simulation/tools/sync_to_vast.sh [-p PORT] user@host

Environment:
  REMOTE_REPO_DIR  Remote repo path. Default: /workspace/iros_2026_ramen
  FLIP_TABLE_POLICY_CHECKPOINT  Optional local ACT checkpoint to sync separately
  REMOTE_POLICY_CHECKPOINT_DIR  Remote checkpoint path. Defaults to
                                $REMOTE_REPO_DIR/.checkpoints/act_flip_table_upper_body
  FLIP_TABLE_FLOW_CHECKPOINT    Optional local Flow Matching checkpoint to sync separately
  REMOTE_FLOW_CHECKPOINT_DIR    Remote Flow checkpoint path. Defaults to
                                $REMOTE_REPO_DIR/.checkpoints/flow_matching_flip_table
  FLIP_TABLE_RLPD_COMBINED_CHECKPOINT  Optional local combined Flow + RLPD checkpoint
  REMOTE_RLPD_COMBINED_CHECKPOINT_DIR  Remote combined checkpoint path. Defaults to
                                $REMOTE_REPO_DIR/.checkpoints/flow_residual_rlpd_flip_table
  VAST_PORT        SSH port. Equivalent to -p.

Example:
  evaluate/flip_table_simulation/tools/sync_to_vast.sh -p 12345 root@example.vast.ai
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      SSH_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      REMOTE="${1:-}"
      shift
      ;;
  esac
done

if [[ -z "${REMOTE:-}" ]]; then
  echo "ERROR: missing user@host." >&2
  usage >&2
  exit 2
fi

command -v rsync >/dev/null 2>&1 || {
  echo "ERROR: rsync is required." >&2
  exit 1
}

SSH_CMD=(ssh -p "$SSH_PORT")
RSYNC_RSH="ssh -p $SSH_PORT"

"${SSH_CMD[@]}" "$REMOTE" "mkdir -p '$REMOTE_REPO_DIR'"

rsync -az --delete --delete-delay \
  --exclude='.git/' \
  --exclude='.checkpoints/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.venv/' \
  --exclude='.venv_*/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='wandb/' \
  --exclude='third_party/' \
  --exclude='data/vit_phase1/results/' \
  --exclude='data/vit_phase1/audit/' \
  --exclude='data/vit_phase1/hf_cache/' \
  --exclude='data/vit_phase1/frames/' \
  --exclude='data/bitrobot_lerobot_subtask_datasets/outputs/' \
  --exclude='model/subtask_policy_training/.venv/' \
  --exclude='model/subtask_policy_training/.venv_lerobot060/' \
  --exclude='model/subtask_policy_training/outputs/' \
  --exclude='model/subtask_policy_training/logs/' \
  -e "$RSYNC_RSH" \
  "$ROOT_DIR"/ "$REMOTE:$REMOTE_REPO_DIR"/

sync_checkpoint() {
  local source="$1"
  local destination="$2"

  if [[ -z "$source" ]]; then
    return
  fi
  if [[ ! -d "$source" ]]; then
    echo "ERROR: checkpoint directory not found: $source" >&2
    exit 1
  fi

  "${SSH_CMD[@]}" "$REMOTE" "mkdir -p '$destination'"
  rsync -az --delete --delete-delay \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    -e "$RSYNC_RSH" \
    "$source"/ "$REMOTE:$destination"/
}

sync_checkpoint "$LOCAL_CHECKPOINT" "$REMOTE_CHECKPOINT_DIR"
sync_checkpoint "$LOCAL_FLOW_CHECKPOINT" "$REMOTE_FLOW_CHECKPOINT_DIR"
sync_checkpoint "$LOCAL_RLPD_COMBINED_CHECKPOINT" "$REMOTE_RLPD_COMBINED_CHECKPOINT_DIR"

"${SSH_CMD[@]}" "$REMOTE" "test -d /workspace/robofinals && echo 'RoboFinals root: /workspace/robofinals' || echo 'WARN: /workspace/robofinals not found; launch paperc/robofinals:RoboFinals-IKEA-V1.'"

cat <<EOF
Synced local repo to:
  $REMOTE:$REMOTE_REPO_DIR

Next on the Vast instance:
  cd $REMOTE_REPO_DIR
EOF

if [[ -n "$LOCAL_RLPD_COMBINED_CHECKPOINT" ]]; then
  cat <<EOF
  FLIP_TABLE_RLPD_COMBINED_CHECKPOINT=$REMOTE_RLPD_COMBINED_CHECKPOINT_DIR \\
  FLIP_TABLE_RL_STAGE=full \\
  FLIP_TABLE_RL_NUM_ENVS=1 \\
  FLIP_TABLE_RL_EVAL_MODE=fixed \\
  FLIP_TABLE_RLPD_EVAL_EPISODES=3 \\
  model/flip_table_reinforcement_learning/run_train_in_container.sh evaluate_rlpd_stage
EOF
elif [[ -n "$LOCAL_FLOW_CHECKPOINT" ]]; then
  cat <<EOF
  FLIP_TABLE_FLOW_CHECKPOINT=$REMOTE_FLOW_CHECKPOINT_DIR \\
  FLIP_TABLE_RLPD_EVAL_RESIDUAL_MODE=zero \\
  FLIP_TABLE_RL_STAGE=full \\
  FLIP_TABLE_RL_NUM_ENVS=1 \\
  FLIP_TABLE_RL_EVAL_MODE=fixed \\
  FLIP_TABLE_RLPD_EVAL_EPISODES=3 \\
  model/flip_table_reinforcement_learning/run_train_in_container.sh evaluate_rlpd_stage
EOF
elif [[ -n "$LOCAL_CHECKPOINT" ]]; then
  cat <<EOF
  FLIP_TABLE_POLICY_CHECKPOINT=$REMOTE_CHECKPOINT_DIR \\
  FLIP_TABLE_POLICY_NAME=LeRobotACTPolicy \\
  FLIP_TABLE_TEST_NUM=1 \\
  evaluate/flip_table_simulation/run_eval_in_container.sh
EOF
else
  cat <<EOF
  FLIP_TABLE_POLICY_NAME=NoOpPolicy FLIP_TABLE_TEST_NUM=1 evaluate/flip_table_simulation/run_eval_in_container.sh
EOF
fi
