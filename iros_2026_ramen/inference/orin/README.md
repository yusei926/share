# inference/orin/ — Orin 側 ROS2 Humble deploy

topology β (Issue #58) における Orin 側の deploy guide。Orin は I/O layer 専任 (camera publish / hand SDK / motor bridge)、上位 orchestrator は Desktop 側 (`inference/desktop/`) で走る。

## Status

- **topology β 確定** (Issue #58): Desktop (RTX 5090) = pixi runtime + cyclonedds direct、Orin = ROS2 I/O layer
- **Docker deploy 実装済**、apt install (host native) 選択肢は Phase 2 #A で Orin 実物見て評価
- 詳細な ROS2 system 設計 (mermaid diagram / topic contract / GR00T observation) は [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md) 参照 (元 topology γ 前提だが Orin 側 package 詳細は流用可)

## Active packages

### Production (`ros2_ws/src/`)

| package | 責務 |
|---|---|
| `g1_bringup` | launch entry (`launch/system.launch.py`, `launch/dataset_replay_rviz.launch.py`) + `system.yaml` |
| `g1_hw_bridge` | Unitree G1 hardware I/O bridge (`mock_hw_bridge_node` = 実機無し zero publish / `real_hw_bridge_node` = unitree_sdk2py 経由で rt/lowstate 購読)。`mock_hardware:=false` で real に切替 |
| `g1_description` | Unitree G1 + Dex1 hand URDF / mesh / RViz config |
| `g1_msgs` | 共有 msg 定義 (`PolicyOutput.msg`) |

### Dev tools (`dev_tools/`、実運用 deploy に不要)

| package | 責務 |
|---|---|
| `g1_dataset_replay` | LeRobot parquet を `/joint_states` 等に replay → RViz で recorded episode 可視化 (mock G1、実機不要) |

### Deprecated (`_deprecated/`、topology β では未使用)

- `g1_task_runner` — planner + subtask executor、Desktop orchestrator と重複。去就は Phase 2 #B-2 で判断予定 ([`../_deprecated/README.md`](../_deprecated/README.md) 参照)
- `g1_motion_adapter` — PolicyOutput → joint_target 変換 + safety filter。safety filter は [`desktop/lower_policy/actuators/safety.py:SafeActuator`](../desktop/lower_policy/actuators/safety.py) に純 Python として migration 済

## Deploy 手順 (Docker)

現状 Orin 側 deploy は Docker approach で実装済。apt install (host native) 選択肢は Phase 2 #A で Orin 実物見て trade-off を評価する。

### 起動 (Orin 実機で、G1 body は OFF でも OK)

```bash
cd inference/orin
docker compose build
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py
```

- `ros:humble-ros-base` (ARM64 対応) base で ROS2 Humble + cyclonedds + camera driver + `unitree_sdk2py` を container 内に入れる (`Dockerfile` 参照、SDK は pinned commit で `git clone` される)
- `g1_hw_bridge` は default で mock 起動。`mock_hardware:=false` で real bridge (`real_hw_bridge_node`) に切替 → G1 body の `rt/lowstate` を購読して `/joint_states` + `/g1/robot_state` に publish (motor 命令は一切発行しない = Damp state 安全性を bridge レベルで保証)
- **実 HBVCAM camera** (USB2.0 Camera RGB) を `usb_cam` が MJPG grab (`pixel_format: mjpeg2rgb`)、`image_transport-plugins` が `/head/camera/color/image_raw/compressed` (CompressedImage) を自動生成 (Desktop 側 `Ros2FrameSource` が subscribe する topic 名と一致)
- 詳細な起動 flag / 動作は [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md#quick-start) 参照 (**注: topology γ 前提の記述、`auto_start_task` などは廃止済**)

#### ヘッドカメラの安定指定（実機・評価時に確認必須）

`/dev/videoN` は USB の接続順、接続ポート、同時に接続されるカメラによって変わるため、launch や Compose へ固定しない。開発機の default は、現在の HBVCAM capture node を指す次の by-id path とする。

```text
/dev/v4l/by-id/usb-USB2.0_Camera_RGB_USB2.0_Camera_RGB_01.00.00-video-index0
```

`docker-compose.yml` はこの値を `HEAD_CAMERA_DEVICE` として container へ渡し、`system.launch.py` の `head_camera_device` launch argument が参照する。launch 時に by-id symlink を正規化してから `usb_cam` へ渡す。`/dev/v4l` は container に bind mount 済みで、数値 device の割当が変わっても by-id symlink が現在の capture node を解決する。

コンペ本番を含む別環境では製品名、VID/PID、シリアル番号が開発機と異なる可能性があるため、default path の存在だけで判断せず、起動前に必ず実デバイスを確認すること。

```bash
ls -l /dev/v4l/by-id
v4l2-ctl --list-devices
v4l2-ctl -d /dev/v4l/by-id/<capture-node> --list-formats-ext
```

ヘッドカメラの capture node（metadata node ではない方）であり、使用する `MJPG`・`1280x480`・`30 fps` を提供できることを確認する。本番機の by-id path が異なる場合は、リポジトリを編集せず起動時に環境変数で上書きする。

```bash
HEAD_CAMERA_DEVICE=/dev/v4l/by-id/<capture-node> \
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false
```

一度だけ別deviceを試す場合は launch argumentでも上書きできる。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false \
    head_camera_device:=/dev/video0
```

device path の上書きだけなら image rebuild は不要。対応解像度やpixel formatも異なる機種へ交換した場合は、camera parameterも実機仕様に合わせる。

### 実機 deploy (real bridge、Issue #65)

G1 body 電源 ON + EtherCAT 接続後に real bridge で lowstate を購読する。

```bash
cd inference/orin
docker compose build   # 初回のみ (unitree_sdk2py を container 内に clone)
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false
```

ヘッドUSBカメラが未接続の状態でbridgeだけを検証する場合は、camera node を起動せずに次を使う。

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py mock_hardware:=false enable_camera:=false
```

- `mock_hardware:=false` で `g1_real_hw_bridge` が起動 (mock は起動しない)。
- G1 と Orin を繋ぐ NIC 名は `system.yaml` の `g1_real_hw_bridge.ros__parameters.interface` で指定する。
  - 現在の Orin 実機の default は `eth0`。本番機 / 別開発機では別名になるので、起動前に必ず `ip a` で NIC 名を確認して書き換えること。書き換え忘れ時は lowstate を受け取れず bridge が publish しないので、`ros2 topic hz /joint_states` で ~50 Hz を必ず確認する。
- **motor 命令は一切発行しない** ので、G1 が Damp state / walking FSM いずれでも安全に joint 位置を Desktop に流せる。walk 動作テストは Issue #64 (Phase 2 #C) で。
- 疎通確認: Desktop 側 `evaluate/perception/subscribe_smoke_check` の joint state 版が今後追加予定 (Phase 2 #A-2)。取り急ぎは `ros2 topic hz /joint_states` で ~50Hz 確認可能。

### Wrist camera 起動 (D405 × 2、Issue #75)

両手 wrist RealSense D405 (RGB + IR stereo) を起動する。**default off** = 明示的に `enable_wrist_cameras:=true` を渡す:

```bash
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py \
    mock_hardware:=false \
    enable_wrist_cameras:=true
```

Publish topics (namespace `/wrist_{side}/camera/` prefix):
- `color/image_raw` + `color/image_raw/compressed` (mono RGB、YOLO 学習入力形式)
- `infra1/image_raw` + `infra1/image_raw/compressed` (left IR)
- `infra2/image_raw` + `infra2/image_raw/compressed` (right IR、stereo pair)

**serial → 左右 mapping**: default は Issue #75 実機疎通機体 (2026-07-19) の物理 bus 番号順の初期 guess:
- `wrist_left_serial` = `128422271048` (bus 2-2.1)
- `wrist_right_serial` = `128422271925` (bus 2-2.2)

`/dev/v4l/by-id`に見える文字列はRealSense driverが要求する固有serialと
異なることがある。ここには`realsense2_camera`起動ログで確認したserialを設定する。

この実機のD405はRGB/IRとも`848x480x30` profileを広告する。640幅を要求しても
driverが848x480へfallbackするため、launchは実profileを明示している。

実映像で左右対応付け確認後、間違ってたら **launch arg で swap 可能** (image rebuild 不要):

```bash
# 例: 実映像で左右逆と判明した場合
WRIST_LEFT_SERIAL=128422271925 WRIST_RIGHT_SERIAL=128422271048 \
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup system.launch.py \
    mock_hardware:=false enable_wrist_cameras:=true
```

**CLAUDE.md Key Facts (確定事項)**:
- **IR projector OFF** (`depth_module.emitter_enabled=0`)。dataset G1_WBT の passive stereo と整合、active stereo は使わない
- Depth stream は `enable_depth=false` で未 enable (YAGNI、必要になったら enable)

**疎通確認** (Desktop 側):
```bash
pixi run -e runtime python -m evaluate.perception.subscribe_smoke_check \
    --topic /wrist_left/camera/color/image_raw/compressed --count 30
pixi run -e runtime python -m evaluate.perception.subscribe_smoke_check \
    --topic /wrist_right/camera/color/image_raw/compressed --count 30
```

## Mock G1 (実機不要、RViz で動作確認)

LeRobot parquet を replay して RViz で G1 の動きを再現 (`g1_dataset_replay` 経由):

```bash
cd inference/orin
docker compose run --rm ramen_inference \
    ros2 launch g1_bringup dataset_replay_rviz.launch.py \
    parquet_file:=/path/to/episode.parquet
```

## Desktop 側との接続 (topology β)

- **Camera stream**: Orin 側 `g1_bringup` が camera driver 起動 → ROS2 topic publish → Desktop 側 orchestrator (`inference/desktop/perception/frame_source.py:Ros2FrameSource`) が cyclonedds direct で subscribe
- **SDK 命令 (walk / manipulate)**: Desktop 側 SDK direct call (`inference/desktop/lower_policy/actuators/g1_sdk.py`) → cyclonedds 経由で G1 SDK。**Orin 側の `task_runner` / `motion_adapter` chain は経由しない** (topology γ 時代の chain は `_deprecated/` へ)
- **Hand I/O**: Orin 側 hand SDK が担当
- Desktop 側 entry: `inference/desktop/entrypoint.py`、詳細は [`inference/README.md`](../README.md) の "実 G1 orchestrator run" section

### Dex1-1 serial-to-DDS service

AVP上半身テレオペでは、Orin host上の公式`dex1_1_service`がDex1-1のserial通信と
`rt/dex1/{left,right}/{cmd,state}`の変換を担当する。Docker内のROS bringupとは別serviceで
あり、状態topicが出ているだけでは左右の位置指令への追従は証明できない。

公式revision `cdd9fc5`のtopic名、motor ID（右=0、左=1）、gain、gear ratioは変更せず、
DDS callbackとmotor threadの同期、左右両motorの起動時検出、serial通信失敗とmotor errorの
可視化だけを行うpatchを適用する。

```bash
cd /home/ubuntu/GitHub/iros_2026_ramen
bash inference/orin/scripts/install_dex1_service_hardening.sh
```

この時点ではbuildとsystemd unit配置のみで、稼働processは変更しない。ロボット停止中かつ
Dex1周囲を監督できる保守時間にだけ、明示的に再起動する。

```bash
bash inference/orin/scripts/install_dex1_service_hardening.sh --restart
journalctl -u dex1_1_gripper_server.service -n 100 --no-pager
```

片側motorが未検出、連続serial通信失敗、またはmotor errorがある場合はテレオペを開始せず、
journalの`left`/`right`表示、motor ID、serial port、error codeを確認する。小振幅の片側診断は
[`apple_vision_pro_upper_body_teleop.md`](../../docs/inference/apple_vision_pro_upper_body_teleop.md)
の手順を使う。installerは既知の未追跡runtime logを無視するが、公式checkout内の追跡済み
source変更は上書きしない。

## Reference

- [`ROS2_SYSTEM_REFERENCE.md`](ROS2_SYSTEM_REFERENCE.md) — 元 ROS2 system 設計 doc (topology γ 前提だが Orin 側 package 詳細 / topic 一覧 / dataset contract は流用可)
- [`../README.md`](../README.md) — `inference/` 全体 overview
- [`../../CLAUDE.md`](../../CLAUDE.md) — project 全体 rule
