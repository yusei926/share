#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_DIR="$ROOT_DIR/evaluate/flip_table_simulation"
source "$FEATURE_DIR/groot_dr_profiles.sh"
BASELINE="${1:-}"
AUXILIARY="${2:-}"
BASELINE_VALIDATION="${3:-}"
AUXILIARY_VALIDATION="${4:-}"
OUTPUT_ROOT="${5:-}"
if [[ ! -d "$BASELINE" || ! -d "$AUXILIARY" ]]; then
  echo "ERROR: baseline and auxiliary checkpoint directories are required" >&2
  exit 2
fi
if [[ ! -f "$BASELINE_VALIDATION" || ! -f "$AUXILIARY_VALIDATION" ]]; then
  echo "ERROR: both offline validation reports are required" >&2
  exit 2
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$ROOT_DIR/outputs/flip_table_groot_candidate_comparison/$(date +%Y%m%d_%H%M%S)"
fi
BASELINE="$(realpath "$BASELINE")"
AUXILIARY="$(realpath "$AUXILIARY")"
BASELINE_VALIDATION="$(realpath "$BASELINE_VALIDATION")"
AUXILIARY_VALIDATION="$(realpath "$AUXILIARY_VALIDATION")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: output directory already exists: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

python3 "$ROOT_DIR/model/subtask_policy_training/scripts/validate_groot_n17_candidate.py" \
  --checkpoint "$BASELINE" \
  --expected-progress disabled \
  --output "$OUTPUT_ROOT/baseline_candidate_audit.json"
python3 "$ROOT_DIR/model/subtask_policy_training/scripts/validate_groot_n17_candidate.py" \
  --checkpoint "$AUXILIARY" \
  --expected-progress enabled \
  --output "$OUTPUT_ROOT/auxiliary_progress_candidate_audit.json"

SEED=95001
EPISODES=5
groot_apply_dr_profile validation_v1
for candidate_name in baseline auxiliary_progress; do
  if [[ "$candidate_name" == "baseline" ]]; then
    checkpoint="$BASELINE"
  else
    checkpoint="$AUXILIARY"
  fi
  candidate_dir="$OUTPUT_ROOT/$candidate_name"
  mkdir -p "$candidate_dir"
  python3 - "$candidate_dir/candidate_manifest.json" "$candidate_name" "$SEED" "$EPISODES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
name = sys.argv[2]
seed = int(sys.argv[3])
count = int(sys.argv[4])
path.write_text(
    json.dumps(
        {
            "candidate_name": name,
            "temporal_lambda": "-0.1",
            "execution_steps": 10,
            "seed": seed,
            "policy_inference_seed": seed,
            "episodes": count,
            "episode_ids": [f"{seed}:{index}" for index in range(count)],
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
  FLIP_TABLE_POLICY_CHECKPOINT="$checkpoint" \
  FLIP_TABLE_SIM_OUTPUT_DIR="$candidate_dir" \
  FLIP_TABLE_TEST_NUM="$EPISODES" \
  FLIP_TABLE_EVAL_MODE=randomized \
  FLIP_TABLE_EVAL_SEED="$SEED" \
  FLIP_TABLE_GROOT_INFERENCE_SEED="$SEED" \
  FLIP_TABLE_GROOT_TEMPORAL_LAMBDA=-0.1 \
  FLIP_TABLE_GROOT_N_ACTION_STEPS=10 \
  FLIP_TABLE_SAVE_ACTION_STATE_TRACE=true \
  "$FEATURE_DIR/run_eval.sh"
done

COMPARISON="$OUTPUT_ROOT/sim_candidate_selection.json"
python3 "$FEATURE_DIR/summarize_groot_candidate_comparison.py" \
  --baseline-dir "$OUTPUT_ROOT/baseline" \
  --auxiliary-dir "$OUTPUT_ROOT/auxiliary_progress" \
  --baseline-audit "$OUTPUT_ROOT/baseline_candidate_audit.json" \
  --auxiliary-audit "$OUTPUT_ROOT/auxiliary_progress_candidate_audit.json" \
  --baseline-validation "$BASELINE_VALIDATION" \
  --auxiliary-validation "$AUXILIARY_VALIDATION" \
  --output "$COMPARISON"

selected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$COMPARISON")"
if [[ "$selected" == "auxiliary_progress" ]]; then
  selected_checkpoint="$AUXILIARY"
else
  selected_checkpoint="$BASELINE"
fi

GROOT_CANDIDATE_NAME="$selected" \
"$FEATURE_DIR/run_groot_release_evaluation.sh" \
  --candidate \
  "$selected_checkpoint" \
  "$OUTPUT_ROOT/release"
cp "$OUTPUT_ROOT/release/release_evaluation.json" \
  "$OUTPUT_ROOT/sim_release_evaluation.json"

cat >"$OUTPUT_ROOT/H100_RESUME.txt" <<EOF
Copy this entire directory to:
  <H100 run root>/sim_evaluation_bundle
Copy:
  sim_candidate_selection.json -> <H100 run root>/sim_candidate_selection.json
  sim_release_evaluation.json  -> <H100 run root>/sim_release_evaluation.json
Then rerun model/subtask_policy_training/scripts/run_h100_flip_table_groot_n17.sh.
The immutable test split and Hugging Face upload have not been used by this script.
EOF
echo "Candidate comparison and selected release gate complete: $OUTPUT_ROOT"
