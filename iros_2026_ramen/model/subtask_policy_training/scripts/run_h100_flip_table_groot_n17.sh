#!/usr/bin/env bash
set -euo pipefail

training_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$training_root/../.." && pwd)"
cd "$training_root"

work_root="${GROOT_H100_WORK_ROOT:-/dev/shm/iros_2026_ramen_groot_n17}"
source_root="${SOURCE_DATASET_ROOT:-/home/ubuntu/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_2/snapshots/0dc47877dfb2efbea796a059c81290c649bc773c}"
venv_dir="${VENV_DIR:-$work_root/venv}"
run_root="$work_root/runs"
training_view="$work_root/training_view"
sidecar_root="$work_root/progress_sidecar"
overlay_root="$work_root/groot_overlay"
video_cache_root="$work_root/video_cache_640x480_lossless_all_i"
final_repo_id="${GROOT_FINAL_REPO_ID:-Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2}"
baseline_backup_repo_id="${GROOT_BASELINE_BACKUP_REPO_ID:-${final_repo_id}_baseline_checkpoints}"
auxiliary_backup_repo_id="${GROOT_AUXILIARY_BACKUP_REPO_ID:-${final_repo_id}_auxiliary_checkpoints}"
overfit_episodes="${GROOT_OVERFIT_EPISODES:-0,1,8,20}"
seed="${GROOT_SEED:-42}"
training_target="${GROOT_TRAINING_TARGET:-both}"
persistent_result_root="${GROOT_PERSISTENT_RESULT_ROOT:-$HOME/.cache/iros_groot_n17_transfer/results}"
run_started_utc="$(date -u +%Y%m%dT%H%M%SZ)"

case "$training_target" in
  baseline|both) ;;
  *)
    echo "ERROR: GROOT_TRAINING_TARGET must be baseline or both" >&2
    exit 2
    ;;
esac

mkdir -p "$work_root" "$run_root" "$sidecar_root" "$work_root/tmp"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$work_root/wandb_data}"
mkdir -p "$WANDB_DATA_DIR"
persist_run_artifacts() {
  local status="$?"
  mkdir -p "$persistent_result_root"
  tar \
    --exclude="*/checkpoints" \
    --exclude="*/wandb" \
    --exclude="*/pretrained_model" \
    -czf "$persistent_result_root/groot_n17_${run_started_utc}.tar.gz" \
    -C "$work_root" runs progress_sidecar 2>/dev/null || true
  {
    echo "schema_version=groot_n17_runner_exit_v1"
    echo "started_utc=$run_started_utc"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=$status"
    echo "git_head=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
  } >"$persistent_result_root/runner_exit_${run_started_utc}.txt"
  return "$status"
}
trap persist_run_artifacts EXIT

tmpfiles_rule="/etc/tmpfiles.d/iros-groot-n17.conf"
expected_tmpfiles_rule="x $work_root - - - - -"
if [[ ! -r "$tmpfiles_rule" ]] || ! grep -Fxq "$expected_tmpfiles_rule" "$tmpfiles_rule"; then
  echo "ERROR: tmpfiles exclusion is missing: $expected_tmpfiles_rule" >&2
  exit 1
fi

audit_path="$run_root/preflight.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "branch=$(git -C "$repo_root" branch --show-current)"
  echo "head=$(git -C "$repo_root" rev-parse HEAD)"
  echo "training_target=$training_target"
  echo "baseline_checkpoint_repo=$baseline_backup_repo_id"
  echo "auxiliary_checkpoint_repo=$auxiliary_backup_repo_id"
  echo "git_status_begin"
  git -C "$repo_root" status --short
  echo "git_status_end"
  nvidia-smi
  df -h / /dev/shm
  ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -20
} | tee "$audit_path"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if [[ "$gpu_name" != *H100* || "$gpu_memory_mib" -lt 75000 ]]; then
  echo "ERROR: this run requires one H100 with at least 75 GiB VRAM; found $gpu_name ($gpu_memory_mib MiB)" >&2
  exit 1
fi
free_shm_bytes="$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')"
existing_work_bytes="$(du -sb "$work_root" | cut -f1)"
if [[ "$((free_shm_bytes + existing_work_bytes))" -lt 90000000000 ]]; then
  echo "ERROR: this run needs a 90 GB /dev/shm budget; found $free_shm_bytes free and $existing_work_bytes reusable bytes" >&2
  exit 1
fi
if [[ ! -f "$source_root/meta/info.json" ]]; then
  echo "ERROR: pinned v2 dataset snapshot is unavailable: $source_root" >&2
  exit 1
fi
export VENV_DIR="$venv_dir"
export UV_CACHE_DIR="$work_root/uv_cache"
export HF_HOME="$work_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TORCH_HOME="$work_root/torch"
export XDG_CACHE_HOME="$work_root/xdg_cache"
export TMPDIR="$work_root/tmp"
export WANDB_DIR="$work_root/wandb"
export YOLO_CONFIG_DIR="$work_root/ultralytics"
export ULTRALYTICS_OFFLINE=true
export PYTHONPATH="$repo_root"
mkdir -p "$YOLO_CONFIG_DIR"
if [[ -n "$persistent_result_root" ]]; then
  mkdir -p "$persistent_result_root"
  export GROOT_CONTACT_SHEET_REVIEW_COPY="$persistent_result_root/orientation_contact_sheet.jpg"
  export GROOT_CONTACT_SHEET_APPROVAL_FILE="$persistent_result_root/orientation_contact_sheet.approved"
  export GROOT_REQUIRE_CONTACT_SHEET_REVIEW=true
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required to create the isolated Python 3.12 environment" >&2
    exit 1
  fi
  scripts/setup_env.sh
fi
source "$venv_dir/bin/activate"
python scripts/patch_lerobot_groot_relative_eef.py --check
python scripts/patch_lerobot_furniture_groot_plugin.py
python scripts/patch_lerobot_gradient_accumulation.py --check
python scripts/patch_lerobot_checkpoint_retention.py
python scripts/audit_groot_n17_contract.py \
  --dataset-root "$source_root" \
  --output "$run_root/groot_contract_preflight.json"
eef_fk_audit="$run_root/eef_fk_audit.json"
if [[ ! -f "$eef_fk_audit" ]]; then
  eef_fk_python="${GROOT_EEF_FK_AUDIT_PYTHON:-$venv_dir/bin/python}"
  if ! "$eef_fk_python" -c "import pinocchio, scipy" >/dev/null 2>&1; then
    echo "ERROR: GROOT_EEF_FK_AUDIT_PYTHON must provide pinocchio and scipy" >&2
    exit 1
  fi
  "$eef_fk_python" scripts/audit_groot_eef_fk_consistency.py \
    --source-root "$source_root" \
    --output "$eef_fk_audit"
fi
python -c '
import json
import sys

from model.subtask_policy_training.gr00t.n17_contract import (
    validate_eef_fk_release_audit,
)

validate_eef_fk_release_audit(json.load(open(sys.argv[1], encoding="utf-8")))
' "$eef_fk_audit"

if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
  echo "ERROR: W&B credentials are required; set WANDB_API_KEY or log in before training" >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required for the private dataset/model repositories" >&2
  exit 1
fi

export SUBTASK=flip_table
export POLICY_TYPE=groot
export DEVICE=cuda
export SOURCE_DATASET_ROOT="$source_root"
export TRAINING_VIEW_ROOT="$training_view"
export TRAINING_VIEW_FORCE=false
export TRAINING_VALIDATION_FRACTION=0.1
export GROOT_PROGRESS_SIDECAR="$sidecar_root/progress.jsonl"
export GROOT_VISUAL_ROTATION_SIDECAR="$sidecar_root/visual_rotation.jsonl"
export GROOT_PROCESSOR_OVERLAY_ROOT="$overlay_root"
export GROOT_VISUAL_ROTATION_PYTHON="$venv_dir/bin/python"
export GROOT_CHUNK_SIZE=40
export GROOT_N_ACTION_STEPS=10
export GROOT_VALID_ACTION_DIM=46
export GROOT_USE_BF16=true
full_image_transforms="${GROOT_FULL_IMAGE_TRANSFORMS_ENABLE:-false}"
full_gpu_augmentation="${GROOT_FULL_GPU_AUGMENTATION:-true}"
export GROOT_IMAGE_TRANSFORMS_ENABLE="$full_image_transforms"
export GROOT_CONSISTENT_GPU_AUGMENTATION="$full_gpu_augmentation"
export GROOT_OPTIMIZER_LR=0.0001
export GROOT_OPTIMIZER_WEIGHT_DECAY=0.00001
export GROOT_WARMUP_RATIO=0.05
export GROOT_LOG_FREQ=10
export GROOT_ENV_EVAL_FREQ=0
export WANDB_ENABLE=true
export WANDB_PROJECT=iros2026-ramen-flip-table
export WANDB_DISABLE_ARTIFACT=true
export PUSH_TO_HUB=false
export UPLOAD_AFTER_TRAIN=false
export PRIVATE=true

validation_episodes="139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155"
test_episodes="156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173"

prepare_training_view() {
  eval "$(python scripts/resolve_training_config.py \
    --config configs/subtask_training.json \
    --format shell)"

  if [[ ! -s "$GROOT_VISUAL_ROTATION_SIDECAR" ]]; then
    "$GROOT_VISUAL_ROTATION_PYTHON" scripts/extract_flip_table_visual_rotation.py \
      --repo-id "$DATASET_REPO_ID" \
      --revision "$DATASET_REVISION" \
      --weight-repo "$GROOT_VISUAL_ROTATION_WEIGHT_REPO" \
      --weight-file "$GROOT_VISUAL_ROTATION_WEIGHT_FILE" \
      --output-root "$(dirname "$GROOT_VISUAL_ROTATION_SIDECAR")" \
      --source-root "$SOURCE_DATASET_ROOT"
  fi
  if [[ ! -s "$GROOT_PROGRESS_SIDECAR" ]]; then
    python scripts/build_flip_table_progress_sidecar.py \
      --repo-id "$DATASET_REPO_ID" \
      --revision "$DATASET_REVISION" \
      --output-root "$(dirname "$GROOT_PROGRESS_SIDECAR")" \
      --visual-rotation-sidecar "$GROOT_VISUAL_ROTATION_SIDECAR" \
      --source-root "$SOURCE_DATASET_ROOT"
  fi
  if [[ ! -s "$TRAINING_VIEW_ROOT/meta/team_ramen_training_view.json" ]]; then
    python scripts/materialize_lerobot_training_view.py \
      --config configs/subtask_training.json \
      --repo-id "$DATASET_REPO_ID" \
      --output-root "$TRAINING_VIEW_ROOT" \
      --policy-type "$POLICY_TYPE" \
      --action-representation "$ACTION_REPRESENTATION" \
      --validation-fraction "$TRAINING_VALIDATION_FRACTION" \
      --progress-sidecar "$GROOT_PROGRESS_SIDECAR" \
      --revision "$DATASET_REVISION" \
      --source-root "$SOURCE_DATASET_ROOT"
  fi
}

model_dir_for_output() {
  local output_dir="$1"
  local model_dir="$output_dir/checkpoints/last/pretrained_model"
  if [[ ! -d "$model_dir" ]]; then
    echo "ERROR: completed checkpoint not found: $model_dir" >&2
    return 1
  fi
  readlink -f "$model_dir"
}

checkpoint_step_for_output() {
  local output_dir="$1"
  local checkpoint_dir="$output_dir/checkpoints/last"
  [[ -e "$checkpoint_dir" ]] || return 1
  local name
  name="$(basename "$(readlink -f "$checkpoint_dir")")"
  [[ "$name" =~ ^[0-9]+$ ]] || return 1
  printf '%d\n' "$((10#$name))"
}

hub_latest_checkpoint_step() {
  local repo_id="$1"
  python - "$repo_id" <<'PY'
import re
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

try:
    files = HfApi().list_repo_files(sys.argv[1], repo_type="model")
except RepositoryNotFoundError:
    print(0)
    raise SystemExit
steps = {
    int(match.group(1))
    for name in files
    if (match := re.match(r"^checkpoints/([0-9]+)/", name))
}
print(max(steps, default=0))
PY
}

evaluation_matches_model() {
  local report="$1"
  local model_dir="$2"
  [[ -f "$report" ]] || return 1
  python -c '
import hashlib
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
checkpoint = pathlib.Path(sys.argv[2]) / "model.safetensors"
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(block)
digest = hasher.hexdigest()
raise SystemExit(0 if report.get("model_safetensors_sha256") == digest else 1)
' "$report" "$model_dir"
}

prune_completed_training_state() {
  local output_dir="$1"
  local checkpoints_dir="$output_dir/checkpoints"
  local checkpoint_dir
  checkpoint_dir="$(readlink -f "$checkpoints_dir/last")"
  model_dir_for_output "$output_dir" >/dev/null
  if [[ -d "$checkpoint_dir/training_state" ]]; then
    rm -rf -- "$checkpoint_dir/training_state"
    echo "Removed Hub-backed completed optimizer state: $checkpoint_dir/training_state"
  fi
  local candidate
  for candidate in "$checkpoints_dir"/*; do
    [[ -d "$candidate" ]] || continue
    [[ "$(readlink -f "$candidate")" == "$checkpoint_dir" ]] && continue
    rm -rf -- "$candidate"
    echo "Removed older Hub-backed checkpoint: $candidate"
  done
}

run_training() {
  local name="$1"
  local progress_enabled="$2"
  local steps="$3"
  local micro_batch_size="$4"
  local accumulation_steps="$5"
  local save_freq="$6"
  local num_workers="${7:-${GROOT_NUM_WORKERS:-12}}"
  local durable_backup="${8:-false}"
  local output_dir="$run_root/train_$name"
  local checkpoint_repo_id="$final_repo_id"
  local remote_step=0
  local -a resume_args=()
  local wandb_disable_artifact=true
  if [[ "$durable_backup" == "true" ]]; then
    wandb_disable_artifact=false
    if [[ "$name" == "baseline" ]]; then
      checkpoint_repo_id="$baseline_backup_repo_id"
    elif [[ "$name" == "auxiliary_progress" ]]; then
      checkpoint_repo_id="$auxiliary_backup_repo_id"
    else
      echo "ERROR: no durable checkpoint repository is defined for $name" >&2
      return 1
    fi
  fi

  local local_step=0
  local resume_config=""
  local_step="$(checkpoint_step_for_output "$output_dir" 2>/dev/null || printf '0')"
  if (( local_step == 0 )) && [[ -d "$output_dir" ]]; then
    local interrupted_output="${output_dir}_uncheckpointed_${run_started_utc}"
    if [[ -e "$interrupted_output" ]]; then
      echo "ERROR: interrupted output archive already exists: $interrupted_output" >&2
      return 1
    fi
    mv "$output_dir" "$interrupted_output"
    echo "Archived uncheckpointed $name attempt: $interrupted_output"
  fi
  if (( local_step >= steps )) && [[ -d "$output_dir/checkpoints/last/pretrained_model" ]]; then
    echo "Reusing completed training run: $output_dir"
    if [[ "$durable_backup" == "true" ]]; then
      verify_training_backup "$name" "$checkpoint_repo_id" "$output_dir" "$steps"
    fi
    return
  fi

  if (( local_step > 0 )); then
    resume_config="$output_dir/checkpoints/last/pretrained_model/train_config.json"
    echo "Resuming $name from local checkpoint step $local_step"
  elif [[ "$durable_backup" == "true" ]]; then
    remote_step="$(hub_latest_checkpoint_step "$checkpoint_repo_id")"
    if (( remote_step > 0 )); then
      resume_config="$checkpoint_repo_id"
      echo "Restoring $name from private Hub checkpoint step $remote_step"
    fi
  fi

  if [[ -n "$resume_config" ]]; then
    resume_args+=(--resume=true --config_path="$resume_config")
  fi
  OUTPUT_DIR="$output_dir" \
  JOB_NAME="groot_n17_${name}_seed${seed}" \
  POLICY_REPO_ID="$checkpoint_repo_id" \
  GROOT_PROGRESS_ENABLED="$progress_enabled" \
  GROOT_STEPS="$steps" \
  GROOT_BATCH_SIZE="$micro_batch_size" \
  GROOT_SAVE_FREQ="$save_freq" \
  GROOT_EVAL_STEPS="$save_freq" \
  GROOT_MAX_EVAL_SAMPLES=512 \
  SAVE_CHECKPOINT_TO_HUB="$durable_backup" \
  WANDB_DISABLE_ARTIFACT="$wandb_disable_artifact" \
  LEROBOT_LOCAL_CHECKPOINT_KEEP="${GROOT_LOCAL_CHECKPOINT_KEEP:-1}" \
  LEROBOT_GRADIENT_ACCUMULATION_STEPS="$accumulation_steps" \
  scripts/train_lerobot.sh \
    --seed="$seed" \
    --num_workers="$num_workers" \
    --prefetch_factor="${GROOT_PREFETCH_FACTOR:-4}" \
    --persistent_workers=true \
    "${resume_args[@]}"
  if [[ "$durable_backup" == "true" ]]; then
    verify_training_backup "$name" "$checkpoint_repo_id" "$output_dir" "$steps"
  fi
}

verify_training_backup() {
  local name="$1"
  local repo_id="$2"
  local output_dir="$3"
  local expected_step="$4"
  local actual_step
  actual_step="$(checkpoint_step_for_output "$output_dir")"
  if (( actual_step != expected_step )); then
    echo "ERROR: $name ended at checkpoint step $actual_step, expected $expected_step" >&2
    return 1
  fi
  local checkpoint_dir
  checkpoint_dir="$(readlink -f "$output_dir/checkpoints/last")"
  local receipt="$run_root/${name}_checkpoint_backup.json"
  python scripts/verify_hf_training_checkpoint.py \
    --repo-id "$repo_id" \
    --step "$expected_step" \
    --local-checkpoint "$checkpoint_dir" \
    --output "$receipt"
  install -m 0644 \
    "$receipt" \
    "$persistent_result_root/${name}_checkpoint_backup.json"
}

prepare_training_view

overfit_output="$run_root/train_overfit_aux"
overfit_eval="$run_root/eval_overfit_aux"
if [[ ! -f "$overfit_eval/gate.json" ]]; then
  export GROOT_OVERFIT_EPISODES="$overfit_episodes"
  export GROOT_OVERFIT_VALIDATION_COUNT=1
  export GROOT_IMAGE_TRANSFORMS_ENABLE=false
  export GROOT_CONSISTENT_GPU_AUGMENTATION=false
  run_training \
    overfit_aux \
    true \
    "${GROOT_OVERFIT_STEPS:-1000}" \
    "${GROOT_OVERFIT_BATCH_SIZE:-8}" \
    1 \
    "${GROOT_OVERFIT_STEPS:-1000}" \
    "${GROOT_OVERFIT_NUM_WORKERS:-4}" \
    false
  unset GROOT_OVERFIT_EPISODES GROOT_OVERFIT_VALIDATION_COUNT
  export GROOT_IMAGE_TRANSFORMS_ENABLE="$full_image_transforms"
  export GROOT_CONSISTENT_GPU_AUGMENTATION="$full_gpu_augmentation"
  overfit_model="$(model_dir_for_output "$overfit_output")"
  if [[ ! -f "$overfit_eval/report.json" ]]; then
    python scripts/evaluate_groot_n17_offline.py \
      --model-dir "$overfit_model" \
      --dataset-root "$training_view" \
      --episodes "$overfit_episodes" \
      --output-dir "$overfit_eval" \
      --execution-steps 10 \
      --evaluation-split train \
      --device cuda
  fi
  python scripts/check_groot_overfit_gate.py \
    --report "$overfit_eval/report.json" \
    --output "$overfit_eval/gate.json"
fi
rm -rf -- "$overfit_output/checkpoints"

python scripts/build_lossless_intra_video_cache.py \
  --training-view "$training_view" \
  --cache-root "$video_cache_root" \
  --workers "${GROOT_VIDEO_CACHE_WORKERS:-3}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${GROOT_VIDEO_DECODER_CACHE_SIZE:-10}"

full_micro_batch="${GROOT_MICRO_BATCH_SIZE:-32}"
full_global_batch="${GROOT_GLOBAL_BATCH_SIZE:-64}"
full_steps="${GROOT_FULL_STEPS:-20000}"
full_save_freq="${GROOT_FULL_SAVE_FREQ:-5000}"
if (( full_global_batch % full_micro_batch != 0 )); then
  echo "ERROR: GROOT_GLOBAL_BATCH_SIZE must be divisible by GROOT_MICRO_BATCH_SIZE" >&2
  exit 1
fi
full_accumulation_steps="$((full_global_batch / full_micro_batch))"
run_training \
  baseline \
  false \
  "$full_steps" \
  "$full_micro_batch" \
  "$full_accumulation_steps" \
  "$full_save_freq" \
  "${GROOT_NUM_WORKERS:-12}" \
  true
prune_completed_training_state "$run_root/train_baseline"
if [[ "$training_target" == "baseline" ]]; then
  echo "Completed baseline training and verified its resumable Hub checkpoint."
  echo "Checkpoint repository: $baseline_backup_repo_id"
  exit 0
fi
run_training \
  auxiliary_progress \
  true \
  "$full_steps" \
  "$full_micro_batch" \
  "$full_accumulation_steps" \
  "$full_save_freq" \
  "${GROOT_NUM_WORKERS:-12}" \
  true
prune_completed_training_state "$run_root/train_auxiliary_progress"
baseline_output="$run_root/train_baseline"
auxiliary_output="$run_root/train_auxiliary_progress"
baseline_model="$(model_dir_for_output "$baseline_output")"
auxiliary_model="$(model_dir_for_output "$auxiliary_output")"
baseline_validation="$run_root/eval_validation_baseline"
auxiliary_validation="$run_root/eval_validation_auxiliary_progress"

if ! evaluation_matches_model "$baseline_validation/report.json" "$baseline_model"; then
  python scripts/evaluate_groot_n17_offline.py \
    --model-dir "$baseline_model" \
    --dataset-root "$training_view" \
    --episodes "$validation_episodes" \
    --output-dir "$baseline_validation" \
    --execution-steps 10 \
    --evaluation-split validation \
    --device cuda
fi
if ! evaluation_matches_model "$auxiliary_validation/report.json" "$auxiliary_model"; then
  python scripts/evaluate_groot_n17_offline.py \
    --model-dir "$auxiliary_model" \
    --dataset-root "$training_view" \
    --episodes "$validation_episodes" \
    --output-dir "$auxiliary_validation" \
    --execution-steps 10 \
    --evaluation-split validation \
    --device cuda
fi

selection_report="$run_root/candidate_selection.json"
python scripts/select_groot_n17_candidate.py \
  --baseline-report "$baseline_validation/report.json" \
  --auxiliary-report "$auxiliary_validation/report.json" \
  --sim-report "$run_root/sim_candidate_selection.json" \
  --sim-release-report "$run_root/sim_release_evaluation.json" \
  --output "$selection_report"
selected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$selection_report")"
if [[ "$selected" == "auxiliary_progress" ]]; then
  selected_output="$auxiliary_output"
  selected_model="$auxiliary_model"
else
  selected_output="$baseline_output"
  selected_model="$baseline_model"
fi

test_output="$run_root/eval_test_selected"
if ! evaluation_matches_model "$test_output/report.json" "$selected_model"; then
  python scripts/evaluate_groot_n17_offline.py \
    --model-dir "$selected_model" \
    --dataset-root "$training_view" \
    --episodes "$test_episodes" \
    --output-dir "$test_output" \
    --execution-steps 10 \
    --device cuda
fi

python scripts/finalize_groot_n17_checkpoint.py \
  --model-dir "$selected_model" \
  --training-output "$selected_output" \
  --training-view "$training_view" \
  --evaluation-report "$test_output/report.json" \
  --selection-report "$selection_report" \
  --contract-report "$selected_output/groot_contract.json" \
  --progress-manifest "$sidecar_root/progress_manifest.json" \
  --visual-rotation-manifest "$sidecar_root/visual_rotation_manifest.json" \
  --video-cache-manifest "$video_cache_root/manifest.json" \
  --sim-comparison-report "$run_root/sim_candidate_selection.json" \
  --sim-release-report "$run_root/sim_release_evaluation.json" \
  --sim-evaluation-bundle "$run_root/sim_evaluation_bundle"

python scripts/upload_policy.py \
  --repo-id "$final_repo_id" \
  --model-dir "$selected_model" \
  --private \
  --commit-message "Upload contract-preserving GR00T N1.7 flip_table policy"

python scripts/verify_policy_hub_roundtrip.py \
  --repo-id "$final_repo_id" \
  --dataset-root "$training_view" \
  --episodes "$test_episodes" \
  --output-root "$run_root/hf_roundtrip"

cat >"$run_root/NEXT_SIM_EVALUATION.txt" <<EOF
Model: $final_repo_id
Selection: $selected
Passed same-seed baseline/auxiliary simulator comparison before test selection.
Passed scripted-controller tracking, fixed scene 3/3, and held-out DR 50 episodes.
Temporal ensemble sweep: lambda none/-0.25/-0.1/0 and execution 5/10/20.
Do not claim Sim-to-Real success before staged real-robot evaluation.
EOF
echo "Completed H100 training, selection, held-out evaluation, and HF roundtrip."
echo "Artifacts: $run_root"
