# inference/ — Streaming state-machine Orchestrator (Issue #47)

YOLO-OBB による perception → skill 遷移判定 → lower policy への dispatch を 1 frame ずつ streaming で回すパイプライン。real-time (ROS2 実 G1) と replay (LeRobot ep mp4) を同じ code path で処理する。

## Layout (Issue #61 で再編)

- **`desktop/`** — Desktop (RTX 5090) 側 pipeline (本 README の主対象)
- **`orin/`** — Orin (ROS2 Humble) 側 I/O 資産、Docker / apt deploy。詳細は [`orin/README.md`](orin/README.md)
- **`_deprecated/`** — topology β では未使用、削除は別 Issue で consensus 後 (`_old_vit_planner` / `g1_task_runner` / `g1_motion_adapter`)

**以下 path は特記無き限り `inference/desktop/` 配下**。

## 責務境界

```
frame (rgb, t)
   │
   ├── perception/ ─── YOLO 推論 + detection cleaning + FrameSource 抽象
   │                    (yolo_obb.py, cleaner.py, stream.py, frame_source.py)
   │
   ├── skill_planner/ ─ 履歴 flag / 遷移 rule (state.py + enter_conditions.py + geometry.py)
   │
   ├── lower_policy/ ── Skill lifecycle (start/step/stop) + Actuator abstract
   │                    (dispatcher.py, skills/, actuators/)
   │
   ├── orchestrator.py ─ tick pipeline 本体 (TRANSITIONS graph + fire dispatch)
   │
   └── entrypoint.py   ── 実行 entry point (実 G1 / sim 等、旧 `run/g1_orchestrator.py`)
```

## Data flow (per tick)

```
frame.rgb → YoloObbPerception.predict(rgb) → raw dets
          ↓
        DetectionStream.push(raw) → Optional[cleaned]  (2 frame delay if median enabled)
          ↓ (cleaned が None なら skip)
        SkillState.update(cleaned)                     (履歴 flag latch)
          ↓
        for cand in TRANSITIONS[current]:
            if enter_check[cand](cleaned, state):
                state.transition(cand, ctx)             (atomic 副作用)
                dispatcher.start(cand, params)
                break                                    (1 tick 1 transition)
          ↓
        dispatcher.step(obs) → Optional[np.ndarray]     (action tensor or None)
          ↓
        actuator_send_fn(action)                        (Type B のみ)
        JSONL log
```

## Skill pattern (Type A / Type B)

Skill の per-tick behavior は 2 pattern 混在を許容 (`Skill.step` の default None がその設計):

| Type | 適した skill | 書き方 |
|---|---|---|
| **A: 1 発発行完結** | walk / stand 系 (SDK servo が持続保持) | `_on_start` で actuator に 1 発発行、`step` は default None、`_on_stop` で停止指令 |
| **B: per-tick 推論** | VLA / GR00T / ACT / diffusion | `_on_start` で model.reset、`step(obs)` で毎 tick 推論、action tensor を返す |

現状の実装例:
- Type A: `skills/move_to_table.py` (`MoveToTable`)、SDK `Move(continous_move=True)` で 10 日持続
- Type B: 未実装 (別 Epic で `Gr00tSkill` 等を追加予定)

詳細は `lower_policy/skills/base.py` の `Skill.step` docstring 参照。

## 現状の実行 mode

### 1. Offline smoke (LeRobot mp4 replay + MockSkill)

```
pixi run -e runtime python -m evaluate.orchestrator.smoke_run \
    --video data/vit_phase1/hf_cache/videos/observation.images.cam_0/chunk-000/file-000.mp4 \
    --weight model/yolo_obb/runs/m_lowaug_v3/weights/best.pt \
    --out outputs/orchestrator/smoke.jsonl \
    --max-frames 500
```

- mp4 を cv2 で 1 frame ずつ read → `LerobotFrameSource` で pull → Orchestrator に流す
- 6 skill 全て `MockSkill` で埋める (hardware / VLA 依存無し)
- fire event の順序を JSONL log から目視で検証

### 2. 実 G1 orchestrator run (runtime env、`inference/desktop/entrypoint.py`)

```
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 \
    --weight model/yolo_obb/runs/m_lowaug_v3/weights/best.pt \
    --topic /head/camera/color/image_raw/compressed \
    --log outputs/orch_run.jsonl
```

- **前提** (operator 責任): G1 が walking FSM state に既に入っている、ROS2 camera driver が publish 中、ハーネス/E-stop/clearance 確保
- 歩行速度と最大時間は `inference/desktop/lower_policy/configs/skill_config.yaml` の
  `skills.move_to_table.vx` / `max_dwell_sec` で設定する (`entrypoint.py` に `--vx` はない)
- Ros2FrameSource (latest-only) で subscribe、Orchestrator tick を 30Hz で回す
- walk skill (`move_to_table`) のみ実 SDK、他 5 skill は MockSkill (log のみ) — VLA 実装は別 Epic
- Ctrl+Cで`actuator.set_velocity(0)`。Dampには入れずwalking balancerを維持

### 3. 単体 walk test (実 G1、Issue #64)

```
pixi run -e runtime python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table \
    --interface eth0 --duration 1 --vx 0.15
```

- Orchestrator経由せず、preflight後に`MoveToTable`を1秒動かすdry-run対応test tool

### 4. GR00T N1.7 pick-table-leg 実機推論（腕・Dex1のみ）

今後追加する異種モデルを含む共通の登録・artifact検証・offline dry-runは
[`desktop/model_evaluation/README.md`](desktop/model_evaluation/README.md)を参照する。
pick-leg v1/v2、coarse-insert relative-EEF GR00T、flip-table
chunk-relative Diffusion v2は同じregistryから選択できる。共通化するのは
HF revision管理、カメラrole、artifact検査、offline dry-runと最終16D安全境界だけで、
38D absolute、53D relative-EEF、16D chunk-relativeの変換はfamilyごとに分離する。

新しい共通CLIはHF pathまたはURLを直接受け取る。resolverとartifact監査には
Unitree SDK/DDSを含まない`model-eval`環境を使う。

```bash
MODEL=Team-RAMEN/groot-n1.7-pick-legs-ver2-lora

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli resolve "$MODEL"
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli adapter-dry-run "$MODEL"
```

`adapter-dry-run`は合成データだけを使い、weight、Unitree SDK、DDS、カメラを
初期化せず、ロボット指令を送らない。実weightを使う
`offline-model-dry-run`、実機カメラを使う`--actuate`なしのlive preflight、
空Enter確認付き実機試験の順に進める。通常はpre-motion開始とpolicy開始の2段階、
coarse insert / tightenは「ハンド全開でpre-motion」「dataset初期把持幅へ閉じる」
「policy開始」の3段階とする。

HF repoに`iros_ramen_deployment.json`があるonboard済みモデルは、catalogへの
コード追加なしでHF pathから解決できる。ただし未知の次元・意味論を自動推測して
実機へ接続することはしない。manifestがない、またはtrusted family pluginがない
モデルはfail-closedで拒否し、`cli onboard Team-RAMEN/...`で安全な追加作業用draftを
生成する。詳細なprepare、seal、offline model推論、live評価手順はリンク先を参照する。

`Team-RAMEN/groot-n1.7-pick-legs-ver2-lora` は、ACT/Diffusionとは別の専用workerで
LeRobot 0.6.0からロードする。既存policyのconfig、processor、checkpointには変更を
加えない。

学習時の契約は次のとおり。

- task: `pick table leg`
- camera: `cam_0=head-left`, `cam_1=head-right`, `cam_2=left-wrist`,
  `cam_3=right-wrist`
- state: `robot_q[36] + hand[2] = 38D`
- decoded action: absolute `robot_q[36] + hand[2] = 38D`
- 実機へ送るaction: `robot_q[22:36]`の腕14Dと`hand[0:2]`だけ
- root 7D、脚12D、腰3Dの予測値は必ず破棄する。下半身はRegular Modeが所有する

checkpointは初回起動時に固定revisionから約9.5 GBを
`.checkpoints/groot-n1.7-pick-legs-ver2-lora/`へ取得する。先にロボットなしで
実ロードを確認する場合:

```bash
model/subtask_policy_training/.venv/bin/python -m \
  model.subtask_policy_training.deployment.smoke_groot_n17_checkpoint \
  --checkpoint .checkpoints/groot-n1.7-pick-legs-ver2-lora \
  --device cuda:0
```

このsmokeは合成4カメラだけを使い、G1指令を一切送らない。成功時は
`decoded=(16, 38) executable=(16, 16) ... robot_commands_sent=0`と表示される。

実機カメラ・stateを使う非駆動preflight:

```bash
G1_DDS_INTERFACE=enx18c2bf548e45 \
G1_IMAGE_SERVER_IP=192.168.29.159 \
bash inference/desktop/upper_policy/run_real_pick_leg_groot.sh
```

`--actuate`がなければ、モデルロード、4カメラ、38D state、16-step予測、
関節範囲、初期差分を検査して終了し、関節指令は送らない。実機駆動はハーネス、
E-stop、clearanceを確保した上でのみ、同じコマンドへ`--actuate`を追加する。
起動後はハーネス、E-stop、clearanceを再確認して空のEnterを押すとpre-motionを
開始する。文字が入力されている場合は送信しない。確認後はshoulder pitchを後方へ
回す、腕を高く横へ広げる、高い前方待機姿勢へ
伸ばす、学習datasetの0フレーム目における腕target中央値へ移す、の順でarm-only
pre-motionを行う。最終姿勢を重力補償付きで保持し、
空のEnterを押した後にfresh state/camera/predictionの安全検査が通った場合だけ
policyを開始する。実行中もaction envelopeは腕14D＋Dex1 2Dだけで、Regular Modeを
二重確認する。終了時は同じwaypointを逆順にたどって起動直前の実測腕姿勢まで
戻してからarm_sdkを段階解放し、Regular Mode `(501,0)`への復帰を確認する。

ログは既定で`outputs/real_groot_pick_leg/last_run.jsonl`。モデル初回推論はwarm-upを
含むが、このRTX 5090での非駆動実測は2回目以降約85 msだった。

### 5. Furniture-GR00T flip-table 実機推論（腕・Dex1のみ）

最終化済みflip-table checkpointには、49-D state、53-D logical action、
132-D packed action、H40、valid mask 46-D、固定GR00T revision、Dex1 synergy、
held-out評価のmanifestが含まれる。実機adapterはhead-leftと左右D405の
`[-20,0]`履歴だけを使い、task文字列を`flip table`へ固定する。head-right、
object pose、global camera、接触GTは入力しない。

非駆動preflight:

```bash
G1_DDS_INTERFACE=enx18c2bf548e45 \
G1_IMAGE_SERVER_IP=192.168.29.159 \
FLIP_TABLE_GROOT_CHECKPOINT=.checkpoints/flip_table_groot_n17_2 \
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh
```

`--actuate`なしではモデル、3カメラ履歴、datasetと同じFKによる49-D state、
H40予測、関節範囲、Dex1極性を検証するだけで、指令を送らない。実機駆動時は
ハーネス、E-stop、clearanceを確保して`--actuate`を追加し、起動後に
`RUN ARMS ONLY`を入力する。その後、肩を後方へ退避、腕を高く横へ展開、高い前方
待機姿勢へ移行し、空のEnterを押した場合だけpolicyを開始する。再計画間隔と
temporal decayはcheckpoint内の
release評価で選択された値を自動使用し、CLIから未検証値へ変更しない。重複chunkは
物理absolute targetへ復号してからtimestamp整列・平均する。位置・速度・加速度
制限は平均後に適用する。非同期推論中にH40の有効範囲を使い切った場合は直前の
安全targetを保持し、H40相当時間を超えて応答しなければ古いchunkを実行せず停止する。
送信envelopeは常に腕14DとDex1-1左右2Dだけで、腰・脚・rootはRegular Modeが
所有する。

実機試験はrelease simulation gate合格後に、毎回ハーネスとE-stopを確認して
次の順序で進める。各コマンドは起動後に`RUN ARMS ONLY`の入力が必要であり、
途中停止ではCtrl+CまたはE-stopを使用する。

```bash
export G1_DDS_INTERFACE=enx18c2bf548e45
export G1_IMAGE_SERVER_IP=192.168.29.159
export FLIP_TABLE_GROOT_CHECKPOINT=.checkpoints/flip_table_groot_n17_2

# 1. 非駆動preflight
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh

# 2. 肩後方→高い横展開→高い前方待機の低速移行だけを確認
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh \
  --actuate --pre-motion-only \
  --log outputs/real_furniture_groot/stage_1_pre_motion.jsonl

# 3. policyを1秒、3秒、5秒の順で小振幅追従させる
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh \
  --actuate --max-seconds 1 \
  --policy-arm-velocity-rad-s 0.15 \
  --policy-arm-acceleration-rad-s2 0.5 \
  --log outputs/real_furniture_groot/stage_2_1s.jsonl
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh \
  --actuate --max-seconds 3 \
  --policy-arm-velocity-rad-s 0.30 \
  --policy-arm-acceleration-rad-s2 1.0 \
  --log outputs/real_furniture_groot/stage_3_3s.jsonl
bash inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh \
  --actuate --max-seconds 5 \
  --policy-arm-velocity-rad-s 0.50 \
  --policy-arm-acceleration-rad-s2 2.0 \
  --log outputs/real_furniture_groot/stage_4_5s.jsonl
```

上記3回で追従、Dex1極性、安全停止を確認した後にのみ、通常上限で独立10
episodeを実施する。各episodeのJSONLと外部動画を別名で保存し、成功率を記録する。
この実機10 episodeが完了するまではSim-to-Real成功と表現しない。

## 依存 env

| env | Python | 主な依存 | 用途 |
|---|---|---|---|
| **main (default)** | 3.12 | ultralytics, opencv, numpy | unit test、offline smoke、YOLO 学習/推論 |
| **runtime** | 3.10 | + `unitree_sdk2py`, `cyclonedds` | 実 G1 動作 (`inference/desktop/entrypoint.py`) |

- **rclpy は install しない方針** (Issue #58 で決定)。ROS2 camera topic の subscribe は cyclonedds を直接使う (SDK が既に使ってる DDS layer に相乗り)。詳細は Issue #58 の「Scope 決定の経緯」参照
- `Ros2FrameSource` / `G1SDKWalkActuator` は module top-level では cyclonedds / SDK を import せず lazy import → main env でも import 可能 (test collection / `--help` が通る)

## 実機投入で今後追加が必要な作業

### 次 Issue で対応予定 (Issue #58「実機 G1 動作環境セットアップ + Orchestrator 実機調整」)

- [x] cyclonedds 直接 subscribe 方式の実装 (2026-07-14 完了、commit 4548853):
  - `inference/desktop/perception/sensor_msgs_idl/` に `CompressedImage_` の cyclonedds IDL 定義 (`Header_` / `Time_` は `unitree_sdk2py.idl` の既存を流用、typename は ROS2 canonical と一致 = wire 互換)
  - `Ros2FrameSource` を rclpy 前提から SDK ChannelFactory 経由の cyclonedds subscribe に書き換え、default QoS を SensorDataQoS 相当 (BestEffort / KeepLast(1) / Volatile) で明示
- [x] runtime env に pytest 追加 (IDL test 実行のため、2026-07-14 完了)
- [ ] `inference/desktop/entrypoint.py` を実 G1 で動作確認 (walk skill のみ、他 5 は Mock のまま) — **Phase 2 (別 Issue) で対応**
- [ ] 実測 tuning (camera topic 名 / QoS / Orchestrator tick 実速度 / `MoveToTable` の vx 妥当性 / fire event 挙動)
- [ ] このREADMEに実測結果を反映

**なぜ rclpy を諦めたか**: G1 = Humble + cyclonedds 0.10.2 固定 (Unitree 公式)、SDK が cyclonedds==0.10.2 hard pin → runtime env は Python 3.10 縛り、robostack-humble に py310 build 不在で詰み。SDK fork や Python 3.9 落としより、rclpy を使わず cyclonedds 直接方式 (SDK と同じ DDS layer で subscribe) が最小変更。TF2 / ROS2 service / rosbag2 が使えなくなるが、VLA skill 想定 (image + joint state → action の end-to-end) では TF2 不要と確認済

**Topology β 前提 (Issue #58)**: Desktop (RTX 5090) で orchestrator を動かし、Orin
(G1 onboard) から publish された ROS2 camera topic を Ros2FrameSource で subscribe
する構成に固定。`Ros2FrameSource` (cyclonedds direct) が primary source。

### Head camera の runtime view（2026-07-18 実機確認済み）

Orin の `/head/camera/color/image_raw/compressed` は head stereo packed RGB
`1280x480` を publishする。YOLO-OBBの学習入力はLeRobot `cam_0`、すなわちpacked
画像の**左眼 `640x480`**である。

`inference.desktop.entrypoint` は `HEAD_CAMERA_STEREO_VIEW = "left"` に固定し、
`Ros2FrameSource(..., stereo_view="left")` でdecode直後に左半分だけを取り出す。
右眼とpacked全幅は上位オーケストレータのYOLO入力には使用しない。汎用の
`Ros2FrameSource` は他用途との互換性のため `stereo_view="packed"` をdefaultに
保つが、実機entrypointからは必ずleftを明示する。

実機＋Mock actuator試験では入力shape `(480, 640, 3)`、YOLO cleanup後の
`move_to_table → move_table_base` 遷移、有限速度指令に続くゼロ速度停止を確認済み。

### さらに先 (別 Epic)

- [ ] VLA skill 実装 (Type B: `Gr00tSkill` / `ActSkill` / diffusion 等、5 skill 分の model)
- [x] runtime monitor: `desktop/lower_policy/actuators/safety.py:SafeActuator` に wrapper 実装 (Issue #61)、wrap 対象の manipulation actuator 実装で active 化
- [ ] Safety layer 残り: joint velocity check / E-stop hook / physical fail-safe

## Test

```
# lower_policy 単体 (SDK-lazy、default env で軽量)
pixi run test-lower-policy

# perception tests (cyclonedds subscribe / IDL テスト、runtime env)
pixi run -e runtime python -m pytest inference/desktop/perception/tests/test_frame_source.py inference/desktop/perception/tests/test_sensor_msgs_idl.py -v
```

- `inference/desktop/tests/`, `inference/desktop/*/tests/`: pure unit test (mock 依存、CI で自動走行、fast)
- 実データ smoke は `evaluate/orchestrator/` 側 (手動実行、GPU 使用)
- `inference/desktop/` 全体 (ultralytics/opencv 込み) を走らせるには model/yolo_obb
  sub-workspace env、または issue/64 merge 後の runtime env が必要

## 実機起動 前チェック (最小)

**前提**: 環境構築 (`third_party/unitree_sdk2_python` clone / `pixi install -e runtime` /
実 G1 setup) は事前完了、詳細は [root `README.md` の「環境構築」](../README.md#環境構築) 参照。
物理準備 (ハーネス / E-stop / clearance / バッテリ / G1 walking FSM 状態) は operator
事前確保の前提。

上記が揃っている前提で、orchestrator 起動者側は起動直前に以下だけ確認:

- [ ] `--interface` が Desktop ↔ G1 の DDS 疎通する NIC 名になっている (`ip a` で確認)
- [ ] `--topic` が Orin 側の camera driver が publish する topic 名と一致
- [ ] 単体testで`--vx 0.15 --duration 1`を確認する。`0.10`以下はRPC成功でも
  この実機のgait開始閾値未満だったため、成功確認値として使わない
- [ ] `--weight` の YOLO weight file と `--log` 出力先が意図通り

### 起動コマンド

```
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 \
    --weight model/yolo_obb/runs/m_lowaug_v3/weights/best.pt \
    --topic /head/camera/color/image_raw/compressed \
    --vx 0.15 \
    --log outputs/orch_run.jsonl
```

**Ctrl+C 停止動作の事前確認** (初回のみ、上記コマンドを 3-5 秒で `Ctrl+C` 押す):

```
[shutdown] set_velocity(0); keeping walking FSM
```

がstderrに出て、実機が停止後もRegular Modeで自立を維持すればOK。Dampへは
切り替えない。--interface / --weight / camera系flagは本番と同じ値にする。

その他 flag 詳細は `pixi run -e runtime python -m inference.desktop.entrypoint --help` 参照。

## Reference

- **Issue #47** (this): streaming state-machine Orchestrator + Mock e2e
- **Issue #64**: G1 walk tuning (`lower_policy/`、実測閾値と安全停止)
- **Epic #43**: YOLO-OBB parent epic
- プロジェクト全体の rule: [`/CLAUDE.md`](../CLAUDE.md)
