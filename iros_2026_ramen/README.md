# IROS 2026 — Team RAMEN

Unitree G1 + Dex1-1 を用いた IKEA UTTER 組立タスクの開発リポジトリです。実機操作、知覚、データ整備、模倣学習、シミュレーション評価を一つのリポジトリで管理します。

生成物（動画、重み、ログ、Isaac 出力）は `outputs/`、`model/*/runs/`、または各ツールのキャッシュに保存し、Git には入れません。リポジトリ直下にはワークスペース設定と全体設定だけを置き、実装は下記の機能フォルダに配置します。

## 構成

| 場所 | 役割 |
| --- | --- |
| `data/annotation_tool/` | Label Studio と YOLO OBB アノテーション |
| `data/curation_tool/` | 収集エピソードの選別・修正・HF 同期 UI |
| `data/flip_table_data_augmentation/` | AVP テレオペ、raw episode、データセット変換 |
| `data/yolo_obb/` | YOLO 用のデータセット構築・拡張 |
| `model/yolo_obb/` | YOLO OBB 学習・重み管理 |
| `model/subtask_policy_training/` | ACT / Flow Matching BC などのサブタスク方策学習 |
| `model/flip_table_reinforcement_learning/` | bounded residual RLPD と教師軌道関連 |
| `inference/desktop/` | Desktop 側の知覚・オーケストレータ・G1 SDK 制御 |
| `inference/orin/` | Orin 側 ROS 2 bringup、カメラ、実機 bridge |
| `evaluate/flip_table_simulation/` | RoboFinals Isaac Sim overlay、Sim-to-Real 校正、評価 |
| `docs/` | 実行記録、設計、運用手順 |

入口は [inference/README.md](inference/README.md)、実機 bringup は [inference/orin/README.md](inference/orin/README.md)、flip-table のシミュレーションは [evaluate/flip_table_simulation/README.md](evaluate/flip_table_simulation/README.md) を参照してください。

## 環境

```bash
# 軽量な Desktop lower-policy テスト
pixi install
pixi run test-lower-policy

# 実機 G1 / ROS 2 subscriber / YOLO 推論用
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python
pixi install -e runtime
```

YOLO 学習は root 環境と分離しています。

```bash
cd model/yolo_obb
pixi install
```

`runtime` は Python 3.10 の Unitree SDK / CycloneDDS 専用です。学習依存を root の default 環境へ追加しないでください。

## 実機の基本確認

G1 / Orin が同一 L2 ネットワーク上で camera bringup 済みの場合、Desktop から次を実行します。

```bash
pixi run -e runtime python -m evaluate.perception.subscribe_smoke_check \
  --topic /head/camera/color/image_raw/compressed \
  --network <G1側NIC名> --count 30 --timeout 10
```

`<G1側NIC名>` は `ip a` で確認します。複数 NIC の PC では省略しません。

## AVP テレオペとシミュレーション

Apple Vision Pro の上半身テレオペは [docs/inference/apple_vision_pro_upper_body_teleop.md](docs/inference/apple_vision_pro_upper_body_teleop.md) を正とします。シミュレーションでは、表示用ステレオ映像を低遅延・最新フレーム優先で送信し、保存対象の4カメラ30 Hzデータは成功後に同一コマンド軌道をオフライン再レンダリングして生成します。

永続 Isaac worker は次の場所に限定します。

```bash
bash evaluate/flip_table_simulation/persistent_eval.sh start
bash evaluate/flip_table_simulation/persistent_eval.sh status
```

ワーカーを止めるのは、USD、Docker image、基礎シミュレーション設定を変更する前か、明示的に不要になったときだけです。

## リリース前チェック

```bash
pixi run test-lower-policy
pixi run -e runtime python -m pytest inference/desktop/perception/tests -q
python3 -m py_compile evaluate/flip_table_simulation/tools/persistent_eval_worker.py
bash -n data/flip_table_data_augmentation/run_teleop.sh \
  evaluate/flip_table_simulation/persistent_eval.sh
git diff --check
```

実機を動かすコマンドは必ずハーネス、E-stop、周囲クリアランス、現在のFSMを確認してから実行してください。Sim-to-Real の制約・実験証跡の記録要件は `docs/flip_table/` の実行ログと監査文書に従います。

開発規約は [CLAUDE.md](CLAUDE.md) を参照してください。
