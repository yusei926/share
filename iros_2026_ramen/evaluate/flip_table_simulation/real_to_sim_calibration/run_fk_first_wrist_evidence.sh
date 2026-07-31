#!/usr/bin/env bash
# Produce fresh, source-only D405 table evidence after a head/scene calibration.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_ROOT="${1:?usage: $0 SOURCE_ROOT OUTPUT_DIR EPISODE [EPISODE ...]}"
OUTPUT_DIR="${2:?usage: $0 SOURCE_ROOT OUTPUT_DIR EPISODE [EPISODE ...]}"
shift 2

if [[ $# -lt 2 ]]; then
  echo "at least two calibration episodes are required for held-out wrist validation" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "source dataset root does not exist: $SOURCE_ROOT" >&2
  exit 2
fi

CALIBRATION="${FLIP_TABLE_HEAD_STEREO_CALIBRATION:?set FLIP_TABLE_HEAD_STEREO_CALIBRATION to the pinned head_camera_params.yaml}"
if [[ ! -f "$CALIBRATION" ]]; then
  echo "pinned head stereo calibration does not exist: $CALIBRATION" >&2
  exit 2
fi

container_path() {
  local value="$1"
  case "$value" in
    "$HOME"/.cache/huggingface/*)
      printf '/root/.cache/huggingface/%s\n' "${value#"$HOME/.cache/huggingface/"}"
      ;;
    "$ROOT_DIR"/outputs/*)
      printf '/outputs/%s\n' "${value#"$ROOT_DIR/outputs/"}"
      ;;
    *)
      echo "path is not mounted into the V1 container: $value" >&2
      return 2
      ;;
  esac
}

SOURCE_ROOT_CONTAINER="$(container_path "$SOURCE_ROOT")"
OUTPUT_CONTAINER="$(container_path "$OUTPUT_DIR")"
CALIBRATION_CONTAINER="$(container_path "$CALIBRATION")"

export FLIP_TABLE_AUG_RUNTIME_MODE=docker
export FLIP_TABLE_AUG_OUTPUTS="$ROOT_DIR/outputs"

for episode in "$@"; do
  if [[ ! "$episode" =~ ^[0-9]+$ ]]; then
    echo "episode must be a non-negative integer: $episode" >&2
    exit 2
  fi
  input_dir="$OUTPUT_CONTAINER/input/episode-${episode}"
  mask_dir="$OUTPUT_CONTAINER/masks/episode-${episode}"
  "$ROOT_DIR/data/flip_table_data_augmentation/scripts/run_object_pose_runtime.sh" prepare \
    --source-root "$SOURCE_ROOT_CONTAINER" \
    --calibration "$CALIBRATION_CONTAINER" \
    --episode-index "$episode" \
    --output-dir "$input_dir" \
    --resume
  "$ROOT_DIR/data/flip_table_data_augmentation/scripts/run_object_pose_runtime.sh" masks \
    --input-dir "$input_dir" \
    --output-dir "$mask_dir" \
    --resume
done
