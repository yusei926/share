# data/annotation_tool — Label Studio + YOLO-OBB Annotation Pipeline

IROS 2026 Upper Policy (YOLO-OBB + Logic) 用のアノテーション環境。
親 Epic: [#43](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/issues/43)。
本 Issue: [#44](https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen/issues/44)。

## Goal

- LeRobot v3.0 head RGB フレームを PNG に抽出
- Label Studio で **7 class OBB (Oriented BBox)** をアノテ
- LS export → **YOLO OBB txt** 形式に変換 (`ultralytics` 学習用)

チーム共有するもの: **クラス定義 (labels_config.xml / class_map.json)** と **コード**。
共有しないもの (`.gitignore`): 画像 / LS 内部 DB / アノテ結果 (LS export JSON, YOLO txt)。

## Class Space (v1, 7 class)

| ID | class_name | 定義 |
|---|---|---|
| 0 | `workspace` | 作業場所 (床マット) |
| 1 | `leg` | 足 (棒全体) |
| 2 | `leg_tip` | 足の先端 (ねじ跡がある挿入側の端) |
| 3 | `hole` | 天板の脚穴 |
| 4 | `table_top` | 天板 |
| 5 | `hand_right` | ロボットの右手 |
| 6 | `hand_left` | ロボットの左手 |

正式定義: [class_map.json](class_map.json)、LS UI 用 config: [labels_config.xml](labels_config.xml)。

## Setup

```bash
cd data/annotation_tool
pixi install
```

## Usage

### 1) フレーム抽出

```bash
pixi run dump-frames --source ../../../research/data/lerobot_sample/Home3/... \
                    --ep 0 --fps 1 --out workspace/images/
```

### 2) LS 起動 (別ターミナル、起動しっぱなし)

```bash
export LABEL_STUDIO_USERNAME='<local username>'
export LABEL_STUDIO_PASSWORD='<local password>'
export LABEL_STUDIO_USER_TOKEN='<random local API token>'
pixi run serve
```

- 認証情報はローカル環境変数だけに置き、Gitへ保存しない
- LS のデータは `workspace/ls_data/` に隔離 (`~/.local/share/label-studio/` は使わない)
- 「Starting development server at 0.0.0.0:8080」まで待つ

### 3) プロジェクト自動セットアップ (ep 別)

```bash
# 単一 ep
pixi run bootstrap --ep 62

# 複数
pixi run bootstrap --eps 62,64,66

# 範囲
pixi run bootstrap --ep-range 62-99
```

各 ep について:
- **HF から該当 frames を DL** (`Team-RAMEN/IROS2026_RAMEN_Hara_skillsplitframes_upperpolicy`)
  → `workspace/images/{skill}/ep{XX}/*.png` に配置 (既存 skip)
- LS project `upperpolicy_ep{XX:04d}` を作成 or 再利用
- `labels_config.xml` (7-class OBB) を適用
- Local storage を ep 別 regex で登録 + sync
- 開くべき URL を表示 (`http://localhost:8080/projects/<id>/data`)
- ログイン: `LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD` に設定した値

`--skip-dl` で HF DL を skip (local に既存前提)、`--workers N` で並列 DL 数調整。

### 4) UI でアノテ

- `Create Rectangle` → 回転ハンドルで角度指定 → クラス選択
- ショートカット: `Ctrl+N` 新規矩形、`D`/`A` 次/前画像

### 5) Export → YOLO OBB txt

- LS UI の Data Manager → Export → JSON (full) を `workspace/exports/` に保存
- 変換:

```bash
pixi run ls-to-yolo --export workspace/exports/project-1.json \
                    --out-dir datasets/v0_trial/labels/
```

## Testing

```bash
pixi run test
```

Round-trip test: LS JSON → YOLO OBB → LS JSON で bit-perfect 一致を確認 (10 tests)。

## Troubleshooting

### `pixi run serve` が数分反応しない
LS 初回起動は **Django DB migrations で 5-10 分程度**かかる (画面に何も出ずに待たされる)。
`logs/ls_serve.log` を tail して `Starting development server` or `Serving on http://` を待つ。
2 回目以降は `workspace/ls_data/label_studio.sqlite3` が残っていれば数十秒。

### `Connection refused` on `redis-server`
LS の RQ (background queue) が redis 6379 に接続する。`pixi.toml` の `[tasks.serve]`
に `depends-on = ["redis"]` を仕込んでいるので `pixi run serve` で自動起動されるが、
明示停止したい場合は `pixi run redis-stop`。

### `pixi run bootstrap --ep N` でエラー
LS が完全起動してから呼ぶこと (`wait_for_ls()` で 120s 待つが、それでもタイムアウトするなら serve log 確認)。

### workspace/images に古い ep が残っていて disk 圧迫
`workspace/images/` 全体は約 300 KB × frames なので、全 287 ep で ~65GB。
不要 ep は `rm -rf workspace/images/*/ep00XX/` で削除可 (HF に上げてあるので取り直し可能)。

## Full extract (背景実行、~4-5h)

Trial ではなく **全 6 skill 揃う 287 ep の全 frame** を抽出したい場合:

```bash
# 背景実行 (途中 kill 可、resume で続きから)
nohup pixi run python scripts/dump_head_frames.py \
    --ep-range 0-532 --full-skill-eps-only \
    > logs/full_run.log 2>&1 &

# 監視
tail -f logs/full_run.log
```

- `--full-skill-eps-only` で全 6 skill 揃う ep のみ処理 (それ以外は skip)
- 各 ep で `per_skill = min skill count in that ep` を自動決定 → 完全均等
- `--resume` (default True) で HF に上げ済 ep は skip
- 実測完走: **198,666 frames / 287 eps / 4h 40min / 65 GB PNG / ~72 GB HF**

## Reference

- LS 運用パターン起源: `tech_lab/cv/roadside_semseg_ft_bootstrap/` (semseg 用途、OBB 対応で fork)
- Ultralytics YOLO OBB 形式: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (正規化座標)

## Trial 30 枚 (v0_trial) 結果メモ

`datasets/v0_trial/` の 30 枚アノテ実施後、以下を記録:

- 1 枚あたりの平均所要時間: <TBD>
- 迷った box / ambiguous case: <TBD>
- クラス漏れ候補 (v2 で検討): <TBD>
- OBB の rotation UI 操作感: <TBD>
