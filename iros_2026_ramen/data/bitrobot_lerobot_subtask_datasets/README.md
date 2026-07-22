# BitRobot LeRobot Subtask Datasets

This folder builds Team RAMEN subtask datasets directly from the official
BitRobot LeRobotDataset v3 dataset:

```text
BitRobot/G1_WBT_Dex1_Building-Children-Table
```

The output repos are:

```text
Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1
```

The conversion keeps the official LeRobot v3 feature names and data layout.
It only slices each full-table-building episode into subtask intervals using
timestamps from:

```text
BitRobot/2026-humanoid-ikea-assembly-challenge
```

No camera renaming, state/action concatenation, GR00T-specific layout, or
camera dropping is performed here. Training and inference scripts should select
and map model-specific inputs.

Parquet files are written with the LeRobot v3 `data_files_size_in_mb` metadata
set to 100 MB. Videos are clipped per subtask and then merged per camera into
LeRobot v3 video files with `video_files_size_in_mb` set to 500 MB.

## Usage

Smoke check without videos:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --max-episodes 2 \
  --skip-videos \
  --force
```

Build one full subtask locally with videos:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --force
```

Upload after a successful build:

```bash
model/subtask_policy_training/.venv/bin/python \
  data/bitrobot_lerobot_subtask_datasets/scripts/build_subtask_datasets.py \
  --subtasks flip_table \
  --upload \
  --private \
  --video-files-size-mb 500 \
  --force
```

Full all-subtask conversion clips many videos and can take a long time. Use
`--video-keys` to limit cameras when intentionally making a model-specific
derived dataset; omit it to preserve all official RGB and IR cameras.
