#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/data/flip_table_data_augmentation"
IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
CONFIG="$FEATURE_DIR/configs/pipeline_v1.json"
RUNTIME_MODE="${FLIP_TABLE_AUG_RUNTIME_MODE:-auto}"
FULL_RUN_MIN_FREE_GB="${FLIP_TABLE_AUG_MIN_FREE_GB:-900}"
DIRECT_SETUP_MIN_FREE_GB="${FLIP_TABLE_AUG_DIRECT_SETUP_MIN_FREE_GB:-230}"
OCI_LAYOUT="${ROBOFINALS_OCI_LAYOUT:-/workspace/robofinals_oci_v1}"
ROOTFS="${ROBOFINALS_ROOTFS:-/workspace/robofinals_rootfs_v1}"

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

for value in "$FULL_RUN_MIN_FREE_GB" "$DIRECT_SETUP_MIN_FREE_GB"; do
  positive_integer "$value" || {
    echo "ERROR: storage thresholds must be positive integers" >&2
    exit 2
  }
done

EXPECTED_DIGEST="$(python3 - "$CONFIG" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["container_digest"])
PY
)"
IMAGE_REPOSITORY="${IMAGE%:*}"
PINNED_IMAGE="${ROBOFINALS_PINNED_IMAGE:-${IMAGE_REPOSITORY}@${EXPECTED_DIGEST}}"

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

FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
FREE_GB=$((FREE_KB / 1024 / 1024))
if (( FREE_GB < FULL_RUN_MIN_FREE_GB )); then
  echo "WARNING: ${FREE_GB} GiB is free; the full run gate requires ${FULL_RUN_MIN_FREE_GB} GiB." >&2
  echo "         Smoke and pilot runs may proceed, but full generation must not start yet." >&2
fi

mkdir -p "$FEATURE_DIR/outputs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ "$RUNTIME_MODE" == "docker" ]]; then
  for command_name in docker nvidia-smi python3; do
    command -v "$command_name" >/dev/null 2>&1 || {
      echo "ERROR: required command is unavailable: $command_name" >&2
      exit 1
    }
  done
  docker pull "$PINNED_IMAGE"
  OBSERVED_DIGEST="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
  if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
    echo "ERROR: pulled V1 image digest mismatch: $OBSERVED_DIGEST" >&2
    exit 1
  fi
  docker image tag "$PINNED_IMAGE" "$IMAGE"
  "$FEATURE_DIR/scripts/run_v1_container.sh" verify-runtime \
    --output /outputs/runtime.json
else
  for command_name in skopeo nvidia-smi python3 sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || {
      echo "ERROR: direct Vast runtime requires: $command_name" >&2
      exit 1
    }
  done

  if [[ ! -f "$OCI_LAYOUT/manifest.json" ]]; then
    if (( FREE_GB < DIRECT_SETUP_MIN_FREE_GB )); then
      echo "ERROR: direct V1 extraction needs at least ${DIRECT_SETUP_MIN_FREE_GB} GiB free" >&2
      exit 1
    fi
    incomplete_layout="${OCI_LAYOUT}.incomplete"
    if [[ -e "$incomplete_layout" ]]; then
      echo "ERROR: stale incomplete OCI layout must be reviewed: $incomplete_layout" >&2
      exit 1
    fi
    skopeo copy --insecure-policy "docker://$PINNED_IMAGE" "dir:$incomplete_layout"
    observed="sha256:$(sha256sum "$incomplete_layout/manifest.json" | awk '{print $1}')"
    if [[ "$observed" != "$EXPECTED_DIGEST" ]]; then
      echo "ERROR: downloaded OCI manifest digest mismatch: $observed" >&2
      exit 1
    fi
    mv "$incomplete_layout" "$OCI_LAYOUT"
  fi
  OBSERVED_DIGEST="sha256:$(sha256sum "$OCI_LAYOUT/manifest.json" | awk '{print $1}')"
  if [[ "$OBSERVED_DIGEST" != "$EXPECTED_DIGEST" ]]; then
    echo "ERROR: V1 OCI manifest digest mismatch: $OBSERVED_DIGEST" >&2
    exit 1
  fi

  if [[ ! -d "$ROOTFS/workspace/robofinals" ]]; then
    "$FEATURE_DIR/scripts/extract_oci_rootfs.py" \
      --layout "$OCI_LAYOUT" \
      --output "$ROOTFS" \
      --expected-manifest-digest "$EXPECTED_DIGEST"
  fi
  for required in \
    "$ROOTFS/opt/conda/envs/robofinals/bin/python" \
    "$ROOTFS/workspace/robofinals" \
    "$ROOTFS/workspace/robofinalsbak/robofinals"; do
    [[ -e "$required" ]] || { echo "ERROR: incomplete V1 rootfs: $required" >&2; exit 1; }
  done

  ensure_runtime_link() {
    local target="$1"
    local link="$2"
    if [[ -L "$link" ]]; then
      if [[ "$(readlink -f "$link")" != "$(readlink -f "$target")" ]]; then
        echo "ERROR: runtime link points at another image: $link" >&2
        exit 1
      fi
    elif [[ -e "$link" ]]; then
      echo "ERROR: refusing to replace existing runtime path: $link" >&2
      exit 1
    else
      mkdir -p "$(dirname "$link")"
      ln -s "$target" "$link"
    fi
  }

  ensure_runtime_link "$ROOTFS/opt/conda" /opt/conda
  for name in robofinals Dexmal_Scene_Task IROS_IKEA_V13_20260702 openpi lerobot Scene.usd; do
    if [[ -e "$ROOTFS/workspace/$name" ]]; then
      ensure_runtime_link "$ROOTFS/workspace/$name" "/workspace/$name"
    fi
  done
  ensure_runtime_link "$FEATURE_DIR/outputs" /outputs
  "$FEATURE_DIR/scripts/run_v1_direct.sh" verify-runtime \
    --output "$FEATURE_DIR/outputs/runtime.json"
fi

cat <<EOF
Vast setup completed.
  runtime mode: $RUNTIME_MODE
  image: $PINNED_IMAGE
  observed digest: $OBSERVED_DIGEST
  runtime audit: $FEATURE_DIR/outputs/runtime.json
  free storage: ${FREE_GB} GiB
EOF
