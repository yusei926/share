#!/usr/bin/env bash
# Train the two reproducible H100 baselines on the immutable real-data snapshot.
# Arms use chunk-start-relative targets; Dex1 commands remain absolute.
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "act" && "$1" != "diffusion" ) || ( "$2" != "smoke" && "$2" != "full" ) ]]; then
  echo "usage: $0 {act|diffusion} {smoke|full}" >&2
  exit 2
fi

policy_type="$1"
run_kind="$2"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
if [[ -n "${VENV_PYTHON:-}" ]]; then
  python_bin="$VENV_PYTHON"
elif [[ -x "$root_dir/.venv_h100_py312_shared/bin/python" ]]; then
  python_bin="$root_dir/.venv_h100_py312_shared/bin/python"
elif [[ -x "$root_dir/.venv_h100_py312/bin/python" ]]; then
  python_bin="$root_dir/.venv_h100_py312/bin/python"
else
  python_bin="$root_dir/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: Python environment not found at $python_bin" >&2
  exit 1
fi

revision="10a6ec05f9993b8d59faad2957e47153b0f15f37"
steps=1000
save_freq=1000
if [[ "$run_kind" == "full" ]]; then
  if [[ "$policy_type" == "act" ]]; then
    steps=200000
  else
    steps=250000
  fi
  # Retain periodic recovery checkpoints without exhausting the 99 GB H100 disk.
  save_freq=50000
fi

run_root="outputs/h100_flip_table_chunk_relative/${policy_type}_${run_kind}"
repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_${policy_type}_chunk_relative_1"

export SUBTASK=flip_table
export POLICY_TYPE="$policy_type"
export DATASET_REPO_ID=Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1
export DATASET_REVISION="$revision"
# The materialized view deliberately retains the source's executable absolute
# targets.  The trainer transforms every action chunk relative to q_current[t].
export ACTION_REPRESENTATION=absolute_target
export MODEL_ACTION_REPRESENTATION=chunk_relative_arm_absolute_gripper
export HELDOUT_TEST_EPISODES=6,96,299
export TRAINING_VALIDATION_FRACTION=0
export TRAIN_EVAL_STEPS=0
export TRAIN_MAX_EVAL_SAMPLES=0
export TRAINING_VIEW_ROOT="$run_root/training_view"
# All variants consume the same immutable RGB source. Reuse the source-hash
# verified cache produced by the earlier action-representation experiment;
# action labels never affect this RGB-only cache.
export TRAINING_VIDEO_CACHE_ROOT="outputs/h100_flip_table_delta/video_cache_320x240_gop8"
export TRAINING_VIEW_FORCE=true
export OUTPUT_DIR="$run_root/train"
export JOB_NAME="flip_table_${policy_type}_chunk_relative_${run_kind}"
export POLICY_REPO_ID="$repo_id"
export WANDB_ENABLE=true
export WANDB_PROJECT=iros2026-ramen-flip-table
export PUSH_TO_HUB=false
export UPLOAD_AFTER_TRAIN=false
export PRIVATE=true
export TRAIN_STEPS="$steps"
export TRAIN_SAVE_FREQ="$save_freq"
export TRAIN_LOG_FREQ=100
export TRAIN_IMAGE_TRANSFORMS_ENABLE=true
export DEVICE=cuda
# There are ten RGB MP4 chunks in the immutable source. Retaining all ten in
# each worker avoids repeated TorchCodec eviction/recreation under random
# sampling, while four workers still cap total decoder ownership.
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=10
export TRAIN_NUM_WORKERS=4

wandb_args=(--no-wandb-enable)
if [[ "$WANDB_ENABLE" == "true" ]]; then
  wandb_args=(--wandb-enable)
fi

build_video_cache() {
  "$python_bin" scripts/build_resized_video_cache.py \
    --training-view "$TRAINING_VIEW_ROOT" \
    --cache-root "$TRAINING_VIDEO_CACHE_ROOT" \
    --workers 3
}

if [[ "$policy_type" == "act" ]]; then
  # The native ACT extension provides the required two-frame observations, three
  # separate ResNet-18 encoders, and seven transformer decoder layers.
  export TRAIN_BATCH_SIZE=64
else
  export TRAIN_BATCH_SIZE=128
fi

if [[ "$policy_type" == "act" ]]; then
  "$python_bin" scripts/materialize_lerobot_training_view.py \
    --config configs/subtask_training.json \
    --repo-id "$DATASET_REPO_ID" \
    --revision "$DATASET_REVISION" \
    --output-root "$TRAINING_VIEW_ROOT" \
    --policy-type act \
    --action-representation "$ACTION_REPRESENTATION" \
    --validation-fraction "$TRAINING_VALIDATION_FRACTION" \
    --test-episodes "$HELDOUT_TEST_EPISODES" \
    --force
  build_video_cache
  "$python_bin" scripts/train_native_act_delta.py \
    --dataset-root "$TRAINING_VIEW_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$TRAIN_STEPS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --save-freq "$TRAIN_SAVE_FREQ" \
    --log-freq "$TRAIN_LOG_FREQ" \
    --warmup-steps 5000 \
    --freeze-backbone-steps 10000 \
    --lr 2e-4 \
    --backbone-lr 1e-5 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --worker-restart-steps 400 \
    --action-representation "$MODEL_ACTION_REPRESENTATION" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$JOB_NAME" \
    "${wandb_args[@]}" \
    --device cuda
else
  "$python_bin" scripts/materialize_lerobot_training_view.py \
    --config configs/subtask_training.json \
    --repo-id "$DATASET_REPO_ID" \
    --revision "$DATASET_REVISION" \
    --output-root "$TRAINING_VIEW_ROOT" \
    --policy-type diffusion \
    --action-representation "$ACTION_REPRESENTATION" \
    --validation-fraction "$TRAINING_VALIDATION_FRACTION" \
    --test-episodes "$HELDOUT_TEST_EPISODES" \
    --force
  build_video_cache
  "$python_bin" scripts/train_native_diffusion_delta.py \
    --dataset-root "$TRAINING_VIEW_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$TRAIN_STEPS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --save-freq "$TRAIN_SAVE_FREQ" \
    --log-freq "$TRAIN_LOG_FREQ" \
    --warmup-steps 10000 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --worker-restart-steps 400 \
    --action-representation "$MODEL_ACTION_REPRESENTATION" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$JOB_NAME" \
    "${wandb_args[@]}" \
    --device cuda
fi

model_dir="$OUTPUT_DIR/checkpoints/last/pretrained_model"
eval_dir="$run_root/evaluation_local"
"$python_bin" scripts/evaluate_delta_chunk_reset.py \
  --model-dir "$model_dir" \
  --dataset-root "$TRAINING_VIEW_ROOT" \
  --episodes "$HELDOUT_TEST_EPISODES" \
  --output-dir "$eval_dir" \
  --device cuda

if [[ "$run_kind" == "full" ]]; then
  wandb_url="$($python_bin -c 'import json; from pathlib import Path; path = Path("'"$OUTPUT_DIR"'") / "summary.json"; print(json.loads(path.read_text()).get("wandb_url", ""))')"
  "$python_bin" scripts/write_delta_model_card.py \
    --model-dir "$model_dir" \
    --training-view "$TRAINING_VIEW_ROOT" \
    --evaluation-report "$eval_dir/report.json" \
    --wandb-url "$wandb_url"

  "$python_bin" scripts/upload_policy.py \
    --repo-id "$POLICY_REPO_ID" \
    --model-dir "$model_dir" \
    --private \
    --commit-message "Upload ${policy_type} chunk-relative flip_table checkpoint"

  "$python_bin" scripts/verify_policy_hub_roundtrip.py \
    --repo-id "$POLICY_REPO_ID" \
    --dataset-root "$TRAINING_VIEW_ROOT" \
    --episodes "$HELDOUT_TEST_EPISODES" \
    --output-root "$run_root/hf_roundtrip"
fi
