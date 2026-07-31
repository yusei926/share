#!/usr/bin/env bash
# Reproducible H100 training for the curated flip-table v2 dataset.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "overfit" && "$1" != "full" ) ]]; then
  echo "usage: $0 {overfit|full}" >&2
  exit 2
fi

kind="$1"
training_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$training_dir/../.." && pwd)"
cd "$training_dir"
python_bin="${VENV_PYTHON:-/home/ubuntu/iros_2026_ramen/model/subtask_policy_training/.venv_h100_py312_shared/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "missing H100 Python environment: $python_bin" >&2
  exit 1
fi

readonly dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
readonly dataset_revision="0dc47877dfb2efbea796a059c81290c649bc773c"
readonly heldout_episodes="156,157,163"
readonly run_root="$repo_root/outputs/flip_table_diffusion_chunk_relative_2/${kind}"
readonly view_root="$run_root/training_view"
readonly cache_root="$repo_root/outputs/flip_table_diffusion_chunk_relative_2/video_cache_320x240_gop8"
readonly output_dir="$run_root/train"
readonly model_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_2"

source_root="$(PYTHONPATH="$repo_root" "$python_bin" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2",
    repo_type="dataset",
    revision="0dc47877dfb2efbea796a059c81290c649bc773c",
    allow_patterns=("README.md", "meta/**", "data/**", "videos/observation.images.cam_0/**", "videos/observation.images.cam_2/**", "videos/observation.images.cam_3/**"),
))
PY
)"

if [[ "$kind" == "overfit" ]]; then
  train_episodes="0,1,2"
  # The smoke run must reach the constant-learning-rate regime.  The prior
  # 8k/10k setting ended during warmup and was not a valid memorization test.
  steps=25000
  batch_size=16
  save_freq=5000
  warmup_steps=1000
  job_name="flip_table_diffusion_v2_overfit"
else
  train_episodes="$(SOURCE_ROOT="$source_root" HELDOUT_EPISODES="$heldout_episodes" "$python_bin" - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

root = Path(os.environ["SOURCE_ROOT"])
heldout = {int(value) for value in os.environ["HELDOUT_EPISODES"].split(",")}
indices = []
for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    indices.extend(int(value) for value in pq.read_table(path, columns=["episode_index"])["episode_index"])
if len(indices) != 174 or len(set(indices)) != len(indices):
    raise SystemExit(f"expected exactly 174 unique episodes, got {len(indices)}")
if not heldout.issubset(indices):
    raise SystemExit(f"held-out episode is absent: {sorted(heldout - set(indices))}")
print(",".join(str(index) for index in sorted(set(indices) - heldout)))
PY
)"
  steps=250000
  batch_size=128
  save_freq=25000
  warmup_steps=10000
  job_name="flip_table_diffusion_v2_full"
fi

rm -rf "$view_root" "$output_dir"
PYTHONPATH="$repo_root" "$python_bin" scripts/materialize_lerobot_training_view.py \
  --config configs/subtask_training.json \
  --repo-id "$dataset_repo_id" \
  --revision "$dataset_revision" \
  --source-root "$source_root" \
  --output-root "$view_root" \
  --policy-type diffusion \
  --action-representation absolute_target \
  --validation-fraction 0 \
  --train-episodes "$train_episodes" \
  --test-episodes "$heldout_episodes" \
  --force

PYTHONPATH="$repo_root" "$python_bin" scripts/audit_diffusion_action_contract.py \
  --dataset-root "$view_root" \
  --output "$run_root/preflight/action_contract.json"

PYTHONPATH="$repo_root" "$python_bin" scripts/build_resized_video_cache.py \
  --training-view "$view_root" \
  --cache-root "$cache_root" \
  --workers 3

export LEROBOT_VIDEO_DECODER_CACHE_SIZE=10
export PYTHONPATH="$repo_root"
wandb_args=(
  --wandb-project iros2026-ramen-flip-table
  --wandb-name "$job_name"
)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_args+=(--wandb-enable)
else
  # Training remains reproducible without credentials: metrics.jsonl and the
  # checkpoint summary are always written locally, while W&B is opt-in here.
  echo "WANDB_API_KEY is not set; writing local metrics only."
  wandb_args+=(--no-wandb-enable)
fi
"$python_bin" scripts/train_native_diffusion_delta.py \
  --dataset-root "$view_root" \
  --output-dir "$output_dir" \
  --steps "$steps" \
  --batch-size "$batch_size" \
  --save-freq "$save_freq" \
  --log-freq 100 \
  --warmup-steps "$warmup_steps" \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --ema-decay 0.9999 \
  --grad-clip 1.0 \
  --num-workers 4 \
  --worker-restart-steps 400 \
  --action-representation chunk_relative_arm_absolute_gripper \
  "${wandb_args[@]}" \
  --device cuda

model_dir="$output_dir/checkpoints/last/pretrained_model"
for episodes in "$train_episodes" "$heldout_episodes"; do
  if [[ "$episodes" == "$train_episodes" ]]; then
    name="train"
  else
    name="heldout"
  fi
  PYTHONPATH="$repo_root" "$python_bin" scripts/evaluate_delta_chunk_reset.py \
    --model-dir "$model_dir" \
    --dataset-root "$view_root" \
    --episodes "$episodes" \
    --output-dir "$run_root/evaluation_${name}" \
    --device cuda
done

if [[ "$kind" == "full" ]]; then
  "$python_bin" scripts/write_training_run_record.py \
    --model-dir "$model_dir" \
    --training-output "$output_dir" \
    --action-contract "$run_root/preflight/action_contract.json" \
    --repository-root "$repo_root"
  "$python_bin" scripts/write_delta_model_card.py \
    --model-dir "$model_dir" \
    --training-view "$view_root" \
    --evaluation-report "$run_root/evaluation_heldout/report.json" \
    --training-run-record "$model_dir/training_run_record.json" \
    --wandb-url "$($python_bin -c 'import json; from pathlib import Path; print(json.loads(Path("'"$output_dir"'/summary.json").read_text()).get("wandb_url", ""))')"
  "$python_bin" scripts/upload_policy.py \
    --repo-id "$model_repo_id" \
    --model-dir "$model_dir" \
    --private \
    --commit-message "Upload v2 z-score unclipped chunk-relative flip_table Diffusion checkpoint"
  PYTHONPATH="$repo_root" "$python_bin" scripts/verify_policy_hub_roundtrip.py \
    --repo-id "$model_repo_id" \
    --dataset-root "$view_root" \
    --episodes "$heldout_episodes" \
    --output-root "$run_root/hf_roundtrip"
fi
