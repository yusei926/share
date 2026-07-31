# data/yolo_obb — YOLO-OBB Training Dataset Build

Epic #43 / Issue #45 (v1、100 枚固定)。

`data/annotation_tool/` の LS annotations と HF frames を pull して、
ultralytics YOLO-OBB 学習用 dataset を組む pipeline。

## Layout

```
data/yolo_obb/
├── configs/dataset.yaml       # ultralytics 標準 (class + train/val path)
├── scripts/build_dataset.py   # HF pull → split → ultralytics 標準 layout
├── dataset/                   # .gitignore、run で生成
│   ├── images/{train,val}/{skill}/ep{XX}/frame_YY.png
│   ├── labels/{train,val}/{skill}/ep{XX}/frame_YY.txt
│   └── split_manifest.json    # seed + train/val 割り振り
└── .hf_cache/                 # .gitignore、HF DL の一時 cache
```

## Usage

```bash
# YOLO sub-workspace の環境を使う（root の `train` 環境は存在しない）
cd model/yolo_obb
pixi install
PYTHONPATH=../.. pixi run python -m data.yolo_obb.scripts.build_dataset
```

optional flags:
- `--out data/yolo_obb/dataset` (出力 root)
- `--yaml data/yolo_obb/configs/dataset.yaml` (ultralytics config)
- `--val-ratio 0.10` (default 90/10)
- `--seed 42` (train/val 割り振り再現性)

## Data source

- **Frames**: HF `Team-RAMEN/IROS2026_RAMEN_Hara_skillsplitframes_upperpolicy` (private)
- **Annotations**: HF `Team-RAMEN/IROS2026_RAMEN_Hara_upperpolicy_annotations` (private)
- ep 60 / 100 枚 / 5 skill (skill_5=move_table_base は ep 60 に不在)
- annotation は 100 枚 / 1,114 boxes (avg 11 boxes/frame)

## Split strategy

**skill stratified random split** (v1):
- 各 skill 内で `val_ratio` 分を val に割り振り
- 各 skill 最低 1 枚 val (少なすぎ回避)
- seed 固定で再現可能

100 枚 × val_ratio=0.10:
- 各 skill 20 枚 → 2 枚 val = 5 skill × 2 = 10 val (train 90 val 10)
- 完全 random だと val に居ない skill が出得る → stratified 採用

## Class balance report (100 枚 / seed 42 / val_ratio 0.10 実測)

```
train: 90 frames, 1009 boxes
val:   10 frames,  105 boxes

class          train  val    axis-aligned 外接矩形の平均 size (normalized)
─────────────  ─────  ───    ─────────────────────────────────────────────
workspace       92    10     w=1.00 h=0.72 (画面全体)
leg            347    40     w=0.17 h=0.32 (縦長)
leg_tip        165    13     w=0.08 h=0.08 (小)
hole           137    13     w=0.07 h=0.07 (小)
table_top       86     9     w=0.55 h=0.53
hand_right      92    10     w=0.22 h=0.25
hand_left       90    10     w=0.20 h=0.24
```

- **全 7 class が train/val 両方に存在** ✓
- **min class の box 数 (train)**: table_top = 86 (>> 30 の acceptance criteria クリア)
- leg 系 (leg, leg_tip) が上位 → 予想通り (工程で最も現れる)
- workspace は 1 frame につき 1 box、size 大 (画面全体) → 学習容易だが情報量少

## Ultralytics load 検証

```python
from ultralytics.data.utils import check_det_dataset
data = check_det_dataset('data/yolo_obb/configs/dataset.yaml')
# → pass、nc=7 / names 完全一致
```

`YOLO('yolo11n-obb.pt').train(data='data/yolo_obb/configs/dataset.yaml', ...)` の直投入 OK。

## Acceptance (Issue #45)

- [x] `dataset.yaml` が `ultralytics YOLO('yolov11n-obb.pt').train(data=dataset.yaml, ...)` の load を通す
- [x] train/val split が seed 固定で再現可能 (split_manifest.json)
- [x] class balance report 出力
- [x] min class の box 数 > 30 (avg 11 boxes × 20 枚 = 200 期待、少ない class でも十分)

## Out of scope (v2)

- 追加 annotation → train 結果 (val mAP) 見てから判断 (#45 v2)
- Test split の準備 (v1 は train + val のみ、test は Sakura 本走で判断)
- HIW-500 汎化 pretrain
