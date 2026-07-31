#!/usr/bin/env bash
set -euo pipefail

# Run one bounded, deterministic actuator-identification replay through the
# persistent Isaac worker. This is intentionally not an optimizer: callers
# enumerate a small, documented PD grid and compare the same source interval.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EPISODE_BUNDLE="${1:?usage: run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]}"
OUTPUT_DIR="${2:?usage: run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]}"
SOURCE_FRAME_END="${3:?usage: run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]}"
STIFFNESS_SCALE="${4:?usage: run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]}"
DAMPING_SCALE="${5:?usage: run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]}"
SOURCE_FRAME_START="${6:-0}"
ARMATURE_SCALE="${7:-1}"

for value in "$SOURCE_FRAME_END" "$SOURCE_FRAME_START"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "ERROR: source frames must be non-negative integers" >&2; exit 2; }
done
(( SOURCE_FRAME_START < SOURCE_FRAME_END )) || {
  echo "ERROR: SOURCE_FRAME_START must be less than SOURCE_FRAME_END" >&2
  exit 2
}
for value in "$STIFFNESS_SCALE" "$DAMPING_SCALE" "$ARMATURE_SCALE"; do
  python3 - "$value" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit("ERROR: PD scales must be finite positive values")
PY
done

resolve_python() {
  local candidate
  for candidate in \
    "${FLIP_TABLE_CALIBRATION_PYTHON:-}" \
    "$HOME/miniforge3/envs/tv/bin/python" \
    "$HOME/miniconda3/envs/unitree/bin/python" \
    "$(command -v python3)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c 'import numpy, pyarrow' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "ERROR: no Python with numpy and pyarrow was found" >&2
  exit 2
}

PYTHON_BIN="$(resolve_python)"
PERSISTENT_ROOT="$(realpath -m "${FLIP_TABLE_PERSISTENT_EVAL_ROOT:-$ROOT_DIR/outputs/flip_table_real_to_sim}")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

"$PYTHON_BIN" -m evaluate.flip_table_simulation.real_to_sim_calibration.replay materialize \
  --episode-bundle "$EPISODE_BUNDLE" \
  --output-dir "$OUTPUT_DIR"

# ``materialize`` preserves the complete source stream. The worker timeout
# bounds only simulated time, so each candidate sees an identical prefix while
# avoiding a full contact-heavy replay during free-space PD identification.
CONTROL_STEPS=$((120 + (SOURCE_FRAME_END - 1) * 5 / 3 + 1))
"$ROOT_DIR/evaluate/flip_table_simulation/persistent_eval.sh" ensure
"$PYTHON_BIN" "$ROOT_DIR/evaluate/flip_table_simulation/tools/persistent_eval_client.py" \
  --runtime-env "$OUTPUT_DIR/replay_runtime.env" \
  --persistent-root "$PERSISTENT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "${FLIP_TABLE_EVAL_SEED:-42}" \
  --time-out-limit "$CONTROL_STEPS" \
  --environment "FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES=true" \
  --environment "FLIP_TABLE_RL_RANDOMIZATION_LEVEL=0" \
  --environment "FLIP_TABLE_ARM_STIFFNESS_SCALE_RANGE=$STIFFNESS_SCALE,$STIFFNESS_SCALE" \
  --environment "FLIP_TABLE_ARM_DAMPING_SCALE_RANGE=$DAMPING_SCALE,$DAMPING_SCALE" \
  --environment "FLIP_TABLE_ARM_ARMATURE_SCALE_RANGE=$ARMATURE_SCALE,$ARMATURE_SCALE" \
  --environment "FLIP_TABLE_ARM_FRICTION_SCALE_RANGE=1,1" \
  --environment "FLIP_TABLE_PERSISTENT_RECREATE_ENV=true" \
  --wait

TRACE_PATH="$OUTPUT_DIR/test_0/action_state_trace.jsonl"
[[ -f "$TRACE_PATH" ]] || { echo "ERROR: actuator probe did not produce a trace" >&2; exit 1; }
"$PYTHON_BIN" -m evaluate.flip_table_simulation.real_to_sim_calibration.actuator_identification \
  --episode-bundle "$EPISODE_BUNDLE" \
  --trace "$TRACE_PATH" \
  --source-frame-start "$SOURCE_FRAME_START" \
  --source-frame-end "$SOURCE_FRAME_END" \
  --output "$OUTPUT_DIR/actuator_identification.json"
