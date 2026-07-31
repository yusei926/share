#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/evaluate/flip_table_simulation"
source "$FEATURE_DIR/groot_dr_profiles.sh"
VALIDATION_MODE=finalized
if [[ "${1:-}" == "--candidate" ]]; then
  VALIDATION_MODE=candidate
  shift
fi
CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
  echo "Usage: $0 [--candidate] /path/to/furniture_groot_checkpoint [output_dir]" >&2
  exit 2
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - "$CHECKPOINT" "$VALIDATION_MODE" <<'PY'
import sys
from pathlib import Path

from inference.desktop.upper_policy.furniture_groot_contract import (
    validate_checkpoint_metadata,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    validate_furniture_training_candidate,
)

checkpoint = Path(sys.argv[1])
mode = sys.argv[2]
if mode == "candidate":
    contract = validate_furniture_training_candidate(checkpoint)
    logical = contract["logical_action_dim"]
    horizon = contract["action_horizon"]
    executable = 16
else:
    contract = validate_checkpoint_metadata(checkpoint)
    logical = contract["logical_action_dim"]
    horizon = contract["action_horizon"]
    executable = contract["executable_action_dim"]
print(
    f"Validated {mode} Furniture-GR00T checkpoint: "
    f"H{horizon}, {logical}D logical, {executable}D executable"
)
PY
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${2:-$ROOT_DIR/outputs/flip_table_groot_release/$RUN_ID}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: output directory already exists: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

SCRIPTED_DIR="$OUTPUT_ROOT/scripted_controller_tracking"
groot_apply_dr_profile nominal_v1
FLIP_TABLE_POLICY_NAME=ScriptedJointPolicy \
FLIP_TABLE_SIM_OUTPUT_DIR="$SCRIPTED_DIR" \
FLIP_TABLE_TEST_NUM=1 \
FLIP_TABLE_EVAL_MODE=nominal \
FLIP_TABLE_EVAL_SEED=91001 \
FLIP_TABLE_GROOT_INFERENCE_SEED=91001 \
FLIP_TABLE_TIME_OUT_LIMIT=300 \
FLIP_TABLE_SAVE_ACTION_STATE_TRACE=true \
"$FEATURE_DIR/run_eval.sh"

SWEEP_ROOT="$OUTPUT_ROOT/temporal_validation"
mkdir -p "$SWEEP_ROOT"
groot_apply_dr_profile validation_v1
for temporal_lambda in none -0.25 -0.1 0; do
  lambda_label="${temporal_lambda//-/neg}"
  lambda_label="${lambda_label//./p}"
  for execution_steps in 5 10 20; do
    candidate="$SWEEP_ROOT/lambda_${lambda_label}_exec_${execution_steps}"
    mkdir -p "$candidate"
    python3 - "$candidate/candidate_manifest.json" "$temporal_lambda" "$execution_steps" <<'PY'
import json
import sys
from pathlib import Path

path, decay, execution = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path.write_text(
    json.dumps(
        {
            "temporal_lambda": decay,
            "execution_steps": execution,
            "seed": 92001,
            "policy_inference_seed": 92001,
            "episodes": 5,
            "episode_ids": [f"92001:{index}" for index in range(5)],
            "mode": "randomized_validation",
            "domain_randomization_profile": "validation_v1",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
    FLIP_TABLE_POLICY_NAME=LeRobotGrootN17Policy \
    FLIP_TABLE_POLICY_CHECKPOINT="$CHECKPOINT" \
    FLIP_TABLE_SIM_OUTPUT_DIR="$candidate" \
    FLIP_TABLE_TEST_NUM=5 \
    FLIP_TABLE_EVAL_MODE=randomized \
    FLIP_TABLE_EVAL_SEED=92001 \
    FLIP_TABLE_GROOT_INFERENCE_SEED=92001 \
    FLIP_TABLE_GROOT_TEMPORAL_LAMBDA="$temporal_lambda" \
    FLIP_TABLE_GROOT_N_ACTION_STEPS="$execution_steps" \
    FLIP_TABLE_SAVE_ACTION_STATE_TRACE=true \
    "$FEATURE_DIR/run_eval.sh"
  done
done

SWEEP_REPORT="$OUTPUT_ROOT/temporal_selection.json"
python3 "$FEATURE_DIR/summarize_groot_release_evaluation.py" \
  --sweep-root "$SWEEP_ROOT" \
  --scripted-dir "$SCRIPTED_DIR" \
  --output "$SWEEP_REPORT"

selected_lambda="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["temporal_lambda"])' "$SWEEP_REPORT")"
selected_steps="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["execution_steps"])' "$SWEEP_REPORT")"

FIXED_DIR="$OUTPUT_ROOT/fixed_scene_3ep"
mkdir -p "$FIXED_DIR"
groot_apply_dr_profile nominal_v1
python3 - "$FIXED_DIR/candidate_manifest.json" "$selected_lambda" "$selected_steps" <<'PY'
import json
import sys
from pathlib import Path

path, decay, execution = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path.write_text(
    json.dumps(
        {
            "temporal_lambda": decay,
            "execution_steps": execution,
            "seed": 93001,
            "policy_inference_seed": 93001,
            "episodes": 3,
            "episode_ids": [f"93001:{index}" for index in range(3)],
            "mode": "nominal",
            "domain_randomization_profile": "nominal_v1",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
FLIP_TABLE_POLICY_NAME=LeRobotGrootN17Policy \
FLIP_TABLE_POLICY_CHECKPOINT="$CHECKPOINT" \
FLIP_TABLE_SIM_OUTPUT_DIR="$FIXED_DIR" \
FLIP_TABLE_TEST_NUM=3 \
FLIP_TABLE_EVAL_MODE=nominal \
FLIP_TABLE_EVAL_SEED=93001 \
FLIP_TABLE_GROOT_INFERENCE_SEED=93001 \
FLIP_TABLE_GROOT_TEMPORAL_LAMBDA="$selected_lambda" \
FLIP_TABLE_GROOT_N_ACTION_STEPS="$selected_steps" \
FLIP_TABLE_SAVE_ACTION_STATE_TRACE=true \
"$FEATURE_DIR/run_eval.sh"

FIXED_SUMMARY="$FIXED_DIR/summary.json"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python3 "$FEATURE_DIR/summarize_groot_release_evaluation.py" \
  --candidate-dir "$FIXED_DIR" \
  --output "$FIXED_SUMMARY"

python3 - "$FIXED_SUMMARY" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("test_count") != 3 or result.get("success_count") != 3:
    raise SystemExit(
        "fixed-scene gate failed: expected 3/3; unseen-DR evaluation was not started"
    )
PY

DR_DIR="$OUTPUT_ROOT/unseen_dr_50ep"
mkdir -p "$DR_DIR"
groot_apply_dr_profile held_out_v1
python3 - "$DR_DIR/candidate_manifest.json" "$selected_lambda" "$selected_steps" <<'PY'
import json
import sys
from pathlib import Path

path, decay, execution = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path.write_text(
    json.dumps(
        {
            "temporal_lambda": decay,
            "execution_steps": execution,
            "seed": 94001,
            "policy_inference_seed": 94001,
            "episodes": 50,
            "episode_ids": [f"94001:{index}" for index in range(50)],
            "mode": "unseen_dr",
            "domain_randomization_profile": "held_out_v1",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
FLIP_TABLE_POLICY_NAME=LeRobotGrootN17Policy \
FLIP_TABLE_POLICY_CHECKPOINT="$CHECKPOINT" \
FLIP_TABLE_SIM_OUTPUT_DIR="$DR_DIR" \
FLIP_TABLE_TEST_NUM=50 \
FLIP_TABLE_EVAL_MODE=unseen_dr \
FLIP_TABLE_EVAL_SEED=94001 \
FLIP_TABLE_GROOT_INFERENCE_SEED=94001 \
FLIP_TABLE_GROOT_TEMPORAL_LAMBDA="$selected_lambda" \
FLIP_TABLE_GROOT_N_ACTION_STEPS="$selected_steps" \
FLIP_TABLE_SAVE_ACTION_STATE_TRACE=true \
"$FEATURE_DIR/run_eval.sh"

DR_SUMMARY="$DR_DIR/summary.json"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python3 "$FEATURE_DIR/summarize_groot_release_evaluation.py" \
  --candidate-dir "$DR_DIR" \
  --output "$DR_SUMMARY"

python3 - "$OUTPUT_ROOT/release_evaluation.json" "$CHECKPOINT" "$SWEEP_REPORT" "$FIXED_SUMMARY" "$DR_SUMMARY" "$VALIDATION_MODE" "${GROOT_CANDIDATE_NAME:-}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, checkpoint, sweep_path, fixed_path, dr_path = map(Path, sys.argv[1:6])
validation_mode, candidate_name = sys.argv[6:8]
model = checkpoint / "model.safetensors"
digest = hashlib.sha256(model.read_bytes()).hexdigest()
sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
dr = json.loads(dr_path.read_text(encoding="utf-8"))
payload = {
    "schema_version": "team_ramen_groot_n17_release_evaluation/v1",
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_validation_mode": validation_mode,
    "candidate_name": candidate_name or None,
    "model_safetensors_sha256": digest,
    "selected_temporal_setting": sweep["selected"],
    "temporal_validation": sweep,
    "temporal_validation_sha256": hashlib.sha256(
        sweep_path.read_bytes()
    ).hexdigest(),
    "scripted_controller_tracking": sweep["scripted_controller_tracking"],
    "fixed_scene": fixed,
    "unseen_dr": dr,
    "release_goal": {
        "fixed_scene_required": "3/3",
        "unseen_dr_target": "40/50",
        "unseen_dr_passed": (
            dr.get("test_count") == 50 and int(dr.get("success_count", -1)) >= 40
        ),
    },
    "claim_scope": "simulator evaluation only; no Sim-to-Real success claim",
}
output.write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload["release_goal"], indent=2))
PY

echo "GR00T release evaluation complete: $OUTPUT_ROOT"
