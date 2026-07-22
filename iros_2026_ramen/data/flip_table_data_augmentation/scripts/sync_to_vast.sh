#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/workspace/iros_2026_ramen_issue70}"
SSH_PORT="${VAST_PORT:-${SSH_PORT:-22}}"
SYNC_MARKER=".flip-table-augmentation-sync-root"

usage() {
  cat <<'EOF'
Usage:
  data/flip_table_data_augmentation/scripts/sync_to_vast.sh [-p PORT] user@host

The default destination is dedicated to Issue #70. Existing repositories and
the remote augmentation outputs directory are never overwritten.
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
      exit 2
      ;;
    *)
      REMOTE="$1"
      shift
      ;;
  esac
done
if [[ -z "${REMOTE:-}" ]]; then
  usage >&2
  exit 2
fi
command -v rsync >/dev/null 2>&1 || { echo "ERROR: rsync is required" >&2; exit 1; }

ssh -p "$SSH_PORT" "$REMOTE" "
set -eu
if [ -e '$REMOTE_REPO_DIR/.git' ] && [ ! -f '$REMOTE_REPO_DIR/$SYNC_MARKER' ]; then
  echo 'ERROR: refusing to sync over a Git worktree without the Issue #70 marker: $REMOTE_REPO_DIR' >&2
  exit 1
fi
mkdir -p '$REMOTE_REPO_DIR'
touch '$REMOTE_REPO_DIR/$SYNC_MARKER'
"
rsync -az --delete --delete-delay \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.venv/' \
  --exclude='.venv_*/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='wandb/' \
  --exclude='third_party/' \
  --exclude='data/flip_table_data_augmentation/outputs/' \
  --filter="P $SYNC_MARKER" \
  -e "ssh -p $SSH_PORT" \
  "$ROOT_DIR"/ "$REMOTE:$REMOTE_REPO_DIR"/

cat <<EOF
Synced code to $REMOTE:$REMOTE_REPO_DIR
Next:
  cd $REMOTE_REPO_DIR
  data/flip_table_data_augmentation/scripts/setup_vast.sh
EOF
