#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EPISODE_BUNDLE="${1:?usage: run_anchor_replay.sh /path/to/episodes/anchor.json [output-dir]}"
OUTPUT_DIR="${2:-$ROOT_DIR/outputs/flip_table_real_to_sim/replay_$(date +%Y%m%d_%H%M%S)}"

resolve_calibration_python() {
  if [[ -n "${FLIP_TABLE_CALIBRATION_PYTHON:-}" ]]; then
    [[ -x "$FLIP_TABLE_CALIBRATION_PYTHON" ]] || {
      echo "ERROR: FLIP_TABLE_CALIBRATION_PYTHON is not executable: $FLIP_TABLE_CALIBRATION_PYTHON" >&2
      exit 2
    }
    printf '%s\n' "$FLIP_TABLE_CALIBRATION_PYTHON"
    return
  fi

  # Development workstations use ``tv``; the RTX simulation workstation uses
  # the Unitree environment. Select an installed environment rather than
  # hard-coding one user's home-directory layout.
  local candidate
  for candidate in \
    "$HOME/miniforge3/envs/tv/bin/python" \
    "$HOME/miniconda3/envs/unitree/bin/python" \
    "$HOME/miniconda3/envs/xr-teleop/bin/python" \
    "$(command -v python3)"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c 'import numpy, pyarrow' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: no Python with numpy and pyarrow was found; set FLIP_TABLE_CALIBRATION_PYTHON" >&2
  exit 2
}

PYTHON_BIN="$(resolve_calibration_python)"
MATERIALIZE_ARGS=(
  --episode-bundle "$EPISODE_BUNDLE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "${FLIP_TABLE_REPLAY_CAMERA_SOURCE_FRAMES:-}" ]]; then
  IFS=',' read -r -a CAMERA_SOURCE_FRAMES <<<"$FLIP_TABLE_REPLAY_CAMERA_SOURCE_FRAMES"
  for source_frame in "${CAMERA_SOURCE_FRAMES[@]}"; do
    [[ "$source_frame" =~ ^[0-9]+$ ]] || {
      echo "ERROR: FLIP_TABLE_REPLAY_CAMERA_SOURCE_FRAMES must be comma-separated non-negative integers" >&2
      exit 2
    }
    MATERIALIZE_ARGS+=(--camera-source-frame "$source_frame")
  done
fi
"$PYTHON_BIN" -m evaluate.flip_table_simulation.real_to_sim_calibration.replay materialize \
  "${MATERIALIZE_ARGS[@]}"

# This file contains only generated numeric values and a local replay path.
# shellcheck disable=SC1090
source "$OUTPUT_DIR/replay_runtime.env"
export FLIP_TABLE_TIME_OUT_LIMIT="${FLIP_TABLE_TIME_OUT_LIMIT:-1800}"
if [[ -n "${FLIP_TABLE_PERSISTENT_EVAL_ROOT:-}" ]]; then
  "$ROOT_DIR/evaluate/flip_table_simulation/persistent_eval.sh" ensure
  "$PYTHON_BIN" "$ROOT_DIR/evaluate/flip_table_simulation/tools/persistent_eval_client.py" \
    --runtime-env "$OUTPUT_DIR/replay_runtime.env" \
    --persistent-root "$FLIP_TABLE_PERSISTENT_EVAL_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed "${FLIP_TABLE_EVAL_SEED:-42}" \
    --wait
else
  "$ROOT_DIR/evaluate/flip_table_simulation/run_eval.sh"
fi

TRACE_PATH="$OUTPUT_DIR/test_0/action_state_trace.jsonl"
if [[ ! -f "$TRACE_PATH" ]]; then
  echo "ERROR: replay finished without an action-state trace: $TRACE_PATH" >&2
  exit 1
fi
CAMERA_DIR="$OUTPUT_DIR/test_0/camera_frames"
if [[ ! -d "$CAMERA_DIR" ]]; then
  echo "ERROR: calibration replay finished without camera-frame evidence: $CAMERA_DIR" >&2
  exit 1
fi
"$PYTHON_BIN" -m evaluate.flip_table_simulation.real_to_sim_calibration.replay analyze-trace \
  --trace "$TRACE_PATH" \
  --output "$OUTPUT_DIR/joint_tracking_report.json"
