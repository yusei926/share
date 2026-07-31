#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" && -d ".venv_lerobot060" ]]; then
  VENV_DIR=".venv_lerobot060"
fi
source "$VENV_DIR/bin/activate"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/subtask_training.json}"
TOLERANCE_S="${TOLERANCE_S:-0.001}"
eval "$(python scripts/resolve_training_config.py --config "$TRAIN_CONFIG" --format shell)"

echo "subtask: $SUBTASK"
echo "task: $TASK"
echo "dataset: $DATASET_REPO_ID"
echo "dataset_revision: ${DATASET_REVISION:-latest}"
echo "policy: $POLICY_REPO_ID"
echo "output_dir: $OUTPUT_DIR"
echo "policy_type: $POLICY_TYPE"
echo "control_scope: $CONTROL_SCOPE"
echo "state_dim: $STATE_DIM"
echo "action_dim: $ACTION_DIM ($ACTION_SEMANTICS)"
echo "action_representation: $ACTION_REPRESENTATION"
echo "cameras: $CAMERAS"
echo "source_camera_map: $SOURCE_CAMERA_MAP"
echo "policy_view_layout: $POLICY_VIEW_LAYOUT"
echo "policy_input_features: $POLICY_INPUT_FEATURES"
echo "wandb_project: $WANDB_PROJECT"
echo "tolerance_s: $TOLERANCE_S"
echo "upload_after_train: $UPLOAD_AFTER_TRAIN"
echo "save_checkpoint_to_hub: ${SAVE_CHECKPOINT_TO_HUB:-false}"
echo "local_checkpoint_keep: ${LEROBOT_LOCAL_CHECKPOINT_KEEP:-unbounded}"
echo "venv: $VENV_DIR"

policy_args=()
prepare_groot_base_cmd=()
GROOT_CANONICAL_BASE_MODEL_PATH=""
GROOT_RUNTIME_BASE_MODEL_PATH=""
if [[ "$POLICY_TYPE" == "groot" ]]; then
  GROOT_CANONICAL_BASE_MODEL_PATH="$GROOT_BASE_MODEL_PATH"
  GROOT_RUNTIME_BASE_MODEL_PATH="$GROOT_PROCESSOR_OVERLAY_ROOT"
  prepare_groot_base_cmd=(
    python scripts/prepare_groot_n17_real_g1_overlay.py
    --model-path "$GROOT_CANONICAL_BASE_MODEL_PATH"
    --revision "$GROOT_BASE_MODEL_REVISION"
    --output-root "$GROOT_PROCESSOR_OVERLAY_ROOT"
  )
  if [[ "${DRY_RUN:-false}" != "true" ]]; then
    python scripts/patch_lerobot_groot_relative_eef.py --check
    python scripts/patch_lerobot_furniture_groot_plugin.py --check
    python scripts/patch_lerobot_gradient_accumulation.py --check
    python scripts/patch_lerobot_checkpoint_retention.py --check
    GROOT_RUNTIME_BASE_MODEL_PATH="$("${prepare_groot_base_cmd[@]}")"
    GROOT_CONTRACT_REPORT="${GROOT_CONTRACT_REPORT:-${OUTPUT_DIR}.groot_contract.json}"
    python scripts/audit_groot_n17_contract.py \
      --dataset-root "$SOURCE_DATASET_ROOT" \
      --output "$GROOT_CONTRACT_REPORT"
  fi
  echo "groot_base_model: $GROOT_CANONICAL_BASE_MODEL_PATH@$GROOT_BASE_MODEL_REVISION"
  echo "groot_runtime_overlay: $GROOT_RUNTIME_BASE_MODEL_PATH"
  echo "groot_embodiment_tag: $GROOT_EMBODIMENT_TAG"
  echo "groot_relative_eef_processor: $GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR"
  echo "groot_dataset_source: shared LeRobot v3 DATASET_REPO_ID"
  echo "groot_policy_impl: $GROOT_POLICY_IMPL"
  echo "groot_valid_action_dim: $GROOT_VALID_ACTION_DIM"
  echo "groot_progress_enabled: $GROOT_PROGRESS_ENABLED"
  echo "groot_progress_sidecar: $GROOT_PROGRESS_SIDECAR"
  echo "groot_visual_rotation_sidecar: $GROOT_VISUAL_ROTATION_SIDECAR"
  policy_args+=(
    --dataset.image_transforms.enable="$GROOT_IMAGE_TRANSFORMS_ENABLE"
    --policy.base_model_path="$GROOT_RUNTIME_BASE_MODEL_PATH"
    --policy.embodiment_tag="$GROOT_EMBODIMENT_TAG"
    --policy.chunk_size="$GROOT_CHUNK_SIZE"
    --policy.n_action_steps="$GROOT_N_ACTION_STEPS"
    --policy.valid_action_dim="$GROOT_VALID_ACTION_DIM"
    --policy.progress_enabled="$GROOT_PROGRESS_ENABLED"
    --policy.progress_loss_weight="$GROOT_PROGRESS_LOSS_WEIGHT"
    --policy.progress_monotonicity_weight="$GROOT_PROGRESS_MONOTONICITY_WEIGHT"
    --policy.progress_hidden_dim="$GROOT_PROGRESS_HIDDEN_DIM"
    --policy.consistent_gpu_augmentation="$GROOT_CONSISTENT_GPU_AUGMENTATION"
    --policy.use_relative_actions="$GROOT_USE_RELATIVE_ACTIONS"
    --policy.relative_exclude_joints="$GROOT_RELATIVE_EXCLUDE_JOINTS"
    --policy.use_bf16="$GROOT_USE_BF16"
    --policy.tune_llm="$GROOT_TUNE_LLM"
    --policy.tune_visual="$GROOT_TUNE_VISUAL"
    --policy.tune_projector="$GROOT_TUNE_PROJECTOR"
    --policy.tune_diffusion_model="$GROOT_TUNE_DIFFUSION_MODEL"
    --policy.tune_vlln="$GROOT_TUNE_VLLN"
    --policy.tune_top_llm_layers="$GROOT_TUNE_TOP_LLM_LAYERS"
    --policy.optimizer_lr="$GROOT_OPTIMIZER_LR"
    --policy.optimizer_weight_decay="$GROOT_OPTIMIZER_WEIGHT_DECAY"
    --policy.warmup_ratio="$GROOT_WARMUP_RATIO"
    --policy.max_steps="$GROOT_STEPS"
    --batch_size="$GROOT_BATCH_SIZE"
    --steps="$GROOT_STEPS"
    --save_freq="$GROOT_SAVE_FREQ"
    --env_eval_freq="$GROOT_ENV_EVAL_FREQ"
    --eval_steps="$GROOT_EVAL_STEPS"
    --max_eval_samples="$GROOT_MAX_EVAL_SAMPLES"
    --log_freq="$GROOT_LOG_FREQ"
  )
else
  policy_args+=(
    --dataset.image_transforms.enable="$TRAIN_IMAGE_TRANSFORMS_ENABLE"
    --batch_size="$TRAIN_BATCH_SIZE"
    --steps="$TRAIN_STEPS"
    --save_freq="$TRAIN_SAVE_FREQ"
    --eval_steps="$TRAIN_EVAL_STEPS"
    --max_eval_samples="$TRAIN_MAX_EVAL_SAMPLES"
    --log_freq="$TRAIN_LOG_FREQ"
  )
  if [[ "$POLICY_TYPE" == "act" ]]; then
    policy_args+=(
      --policy.chunk_size="$ACT_CHUNK_SIZE"
      --policy.n_action_steps="$ACT_N_ACTION_STEPS"
    )
  fi
fi

dataset_args=(--dataset.repo_id="$DATASET_REPO_ID")
prepare_training_view_cmd=()
prepare_progress_cmd=()
prepare_visual_rotation_cmd=()
VISUAL_ROTATION_PYTHON="${GROOT_VISUAL_ROTATION_PYTHON:-$ROOT_DIR/../yolo_obb/.pixi/envs/default/bin/python}"
training_condition="${TRAINING_CONDITION:-}"
sampling_plan=""
if [[ "$MATERIALIZE_TRAINING_VIEW" == "true" ]]; then
  echo "training_view_root: $TRAINING_VIEW_ROOT"
  echo "training_view_force: $TRAINING_VIEW_FORCE"
  prepare_training_view_cmd=(
    python scripts/materialize_lerobot_training_view.py
    --config "$TRAIN_CONFIG"
    --repo-id "$DATASET_REPO_ID"
    --output-root "$TRAINING_VIEW_ROOT"
    --policy-type "$POLICY_TYPE"
    --action-representation "$ACTION_REPRESENTATION"
    --validation-fraction "$TRAINING_VALIDATION_FRACTION"
  )
  if [[ "$POLICY_TYPE" == "groot" ]]; then
    prepare_training_view_cmd+=(--progress-sidecar "$GROOT_PROGRESS_SIDECAR")
    prepare_progress_cmd=(
      python scripts/build_flip_table_progress_sidecar.py
      --repo-id "$DATASET_REPO_ID"
      --revision "$DATASET_REVISION"
      --output-root "$(dirname "$GROOT_PROGRESS_SIDECAR")"
      --visual-rotation-sidecar "$GROOT_VISUAL_ROTATION_SIDECAR"
    )
    prepare_visual_rotation_cmd=(
      "$VISUAL_ROTATION_PYTHON" scripts/extract_flip_table_visual_rotation.py
      --repo-id "$DATASET_REPO_ID"
      --revision "$DATASET_REVISION"
      --weight-repo "$GROOT_VISUAL_ROTATION_WEIGHT_REPO"
      --weight-file "$GROOT_VISUAL_ROTATION_WEIGHT_FILE"
      --output-root "$(dirname "$GROOT_VISUAL_ROTATION_SIDECAR")"
    )
    if [[ -n "${SOURCE_DATASET_ROOT:-}" ]]; then
      prepare_progress_cmd+=(--source-root "$SOURCE_DATASET_ROOT")
      prepare_visual_rotation_cmd+=(--source-root "$SOURCE_DATASET_ROOT")
    fi
  fi
  if [[ -n "$HELDOUT_TEST_EPISODES" ]]; then
    prepare_training_view_cmd+=(--test-episodes "$HELDOUT_TEST_EPISODES")
  fi
  if [[ -n "$training_condition" ]]; then
    prepare_training_view_cmd+=(--training-condition "$training_condition")
    sampling_plan="$TRAINING_VIEW_ROOT/meta/team_ramen_lineage_sampling_plan.json"
    export TEAM_RAMEN_SAMPLING_PLAN="$sampling_plan"
    echo "training_condition: $training_condition"
    echo "sampling_plan: $sampling_plan"
  fi
  if [[ -n "$DATASET_REVISION" ]]; then
    prepare_training_view_cmd+=(--revision "$DATASET_REVISION")
  fi
  if [[ -n "${SOURCE_DATASET_ROOT:-}" ]]; then
    prepare_training_view_cmd+=(--source-root "$SOURCE_DATASET_ROOT")
  fi
  if [[ "$TRAINING_VIEW_FORCE" == "true" ]]; then
    prepare_training_view_cmd+=(--force)
  fi
  dataset_args+=(--dataset.root="$TRAINING_VIEW_ROOT")
fi

trainer=(lerobot-train)
if [[ -n "$training_condition" ]]; then
  trainer=(python scripts/train_lerobot_lineage.py)
fi

cmd=(
  "${trainer[@]}"
  "${dataset_args[@]}"
  --policy.type="${GROOT_POLICY_IMPL:-$POLICY_TYPE}"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --policy.device="$DEVICE"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT:-true}"
  --save_checkpoint_to_hub="${SAVE_CHECKPOINT_TO_HUB:-false}"
  --policy.repo_id="$POLICY_REPO_ID"
  --policy.push_to_hub="$PUSH_TO_HUB"
  --policy.private="$PRIVATE"
  --policy.input_features="$POLICY_INPUT_FEATURES"
  --policy.output_features="$POLICY_OUTPUT_FEATURES"
  --tolerance_s="$TOLERANCE_S"
  "${policy_args[@]}"
  "$@"
)

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  if [[ "${#prepare_groot_base_cmd[@]}" -gt 0 ]]; then
    printf "prepare_groot_base:"
    printf " %q" "${prepare_groot_base_cmd[@]}"
    printf "\n"
  fi
  if [[ "${#prepare_training_view_cmd[@]}" -gt 0 ]]; then
    printf "prepare_training_view:"
    printf " %q" "${prepare_training_view_cmd[@]}"
    printf "\n"
    printf "resolve_training_split: python scripts/resolve_training_split.py --dataset-root %q --format shell" \
      "$TRAINING_VIEW_ROOT"
    if [[ "$TRAINING_VALIDATION_FRACTION" == "0" ]]; then
      printf " --allow-empty-validation"
    fi
    if [[ -n "${GROOT_OVERFIT_EPISODES:-}" ]]; then
      printf " --overfit-train-episodes %q --overfit-validation-count %q" \
        "$GROOT_OVERFIT_EPISODES" "${GROOT_OVERFIT_VALIDATION_COUNT:-1}"
    fi
    printf "\n"
  fi
  if [[ "${#prepare_progress_cmd[@]}" -gt 0 ]]; then
    printf "prepare_progress:"
    printf " %q" "${prepare_progress_cmd[@]}"
    printf "\n"
  fi
  if [[ "${#prepare_visual_rotation_cmd[@]}" -gt 0 ]]; then
    printf "prepare_visual_rotation:"
    printf " %q" "${prepare_visual_rotation_cmd[@]}"
    printf "\n"
  fi
  printf "command:"
  printf " %q" "${cmd[@]}"
  printf "\n"
  if [[ "$UPLOAD_AFTER_TRAIN" == "true" ]]; then
    upload_cmd=(
      python scripts/upload_policy.py
      --repo-id "$POLICY_REPO_ID"
      --output-dir "$OUTPUT_DIR"
      --commit-message "Upload $POLICY_TYPE $SUBTASK LeRobot checkpoint"
    )
    if [[ "$PRIVATE" == "true" ]]; then
      upload_cmd+=(--private)
    fi
    printf "post_train_upload:"
    printf " %q" "${upload_cmd[@]}"
    printf "\n"
  fi
  exit 0
fi

if [[ "${#prepare_visual_rotation_cmd[@]}" -gt 0 && ! -f "$GROOT_VISUAL_ROTATION_SIDECAR" ]]; then
  if [[ ! -x "$VISUAL_ROTATION_PYTHON" ]]; then
    echo "ERROR: locked YOLO environment is missing: $VISUAL_ROTATION_PYTHON" >&2
    echo "Run: pixi install --manifest-path ../yolo_obb/pixi.toml --locked" >&2
    exit 1
  fi
  "${prepare_visual_rotation_cmd[@]}"
fi
if [[ "$POLICY_TYPE" == "groot" && -n "${GROOT_CONTACT_SHEET_REVIEW_COPY:-}" ]]; then
  contact_sheet="$(dirname "$GROOT_VISUAL_ROTATION_SIDECAR")/orientation_contact_sheet.jpg"
  if [[ ! -f "$contact_sheet" ]]; then
    echo "ERROR: visual-rotation contact sheet is missing: $contact_sheet" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$GROOT_CONTACT_SHEET_REVIEW_COPY")"
  install -m 0644 "$contact_sheet" "$GROOT_CONTACT_SHEET_REVIEW_COPY"
  contact_sheet_sha256="$(sha256sum "$contact_sheet" | cut -d ' ' -f1)"
  if [[ "${GROOT_REQUIRE_CONTACT_SHEET_REVIEW:-false}" == "true" ]]; then
    approval_file="${GROOT_CONTACT_SHEET_APPROVAL_FILE:?review approval file is required}"
    echo "Waiting for contact-sheet approval: $GROOT_CONTACT_SHEET_REVIEW_COPY"
    echo "Expected SHA256: $contact_sheet_sha256"
    while [[ ! -f "$approval_file" ]] || [[ "$(<"$approval_file")" != "$contact_sheet_sha256" ]]; do
      sleep 10
    done
    install -m 0644 \
      "$approval_file" \
      "$(dirname "$GROOT_VISUAL_ROTATION_SIDECAR")/orientation_contact_sheet.approved"
    echo "Contact-sheet review approved for SHA256 $contact_sheet_sha256"
  fi
fi
if [[ "${#prepare_progress_cmd[@]}" -gt 0 && ! -f "$GROOT_PROGRESS_SIDECAR" ]]; then
  "${prepare_progress_cmd[@]}"
fi
if [[ "${#prepare_training_view_cmd[@]}" -gt 0 ]]; then
  "${prepare_training_view_cmd[@]}"
fi
if [[ -n "$sampling_plan" && ! -f "$sampling_plan" ]]; then
  echo "ERROR: lineage sampling plan not found: $sampling_plan" >&2
  exit 1
fi

split_args=(python scripts/resolve_training_split.py --dataset-root "$TRAINING_VIEW_ROOT" --format shell)
if [[ "$TRAINING_VALIDATION_FRACTION" == "0" ]]; then
  split_args+=(--allow-empty-validation)
fi
if [[ -n "${GROOT_OVERFIT_EPISODES:-}" ]]; then
  if [[ "$POLICY_TYPE" != "groot" ]]; then
    echo "ERROR: GROOT_OVERFIT_EPISODES is only supported for the GR00T policy" >&2
    exit 1
  fi
  split_args+=(
    --overfit-train-episodes "$GROOT_OVERFIT_EPISODES"
    --overfit-validation-count "${GROOT_OVERFIT_VALIDATION_COUNT:-1}"
  )
fi
eval "$("${split_args[@]}")"
echo "split_sha256: $SPLIT_SHA256"
echo "episodes: train=$TRAIN_EPISODE_COUNT validation=$VALIDATION_EPISODE_COUNT test=$TEST_EPISODE_COUNT"
echo "overfit_mode: $OVERFIT_MODE"
cmd+=(
  --dataset.episodes="$DATASET_EPISODES_JSON"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
)

"${cmd[@]}"

if [[ "$POLICY_TYPE" == "groot" ]]; then
  install -m 0644 "$GROOT_CONTRACT_REPORT" "$OUTPUT_DIR/groot_contract.json"
  python scripts/restore_groot_base_model_path.py \
    --output-dir "$OUTPUT_DIR" \
    --runtime-path "$GROOT_RUNTIME_BASE_MODEL_PATH" \
    --canonical-path "$GROOT_CANONICAL_BASE_MODEL_PATH"
fi

if [[ "$UPLOAD_AFTER_TRAIN" == "true" ]]; then
  upload_cmd=(
    python scripts/upload_policy.py
    --repo-id "$POLICY_REPO_ID"
    --output-dir "$OUTPUT_DIR"
    --commit-message "Upload $POLICY_TYPE $SUBTASK LeRobot checkpoint"
  )
  if [[ "$PRIVATE" == "true" ]]; then
    upload_cmd+=(--private)
  fi
  "${upload_cmd[@]}"
fi
