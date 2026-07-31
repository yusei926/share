# Apple Vision ProによるG1上半身テレオペレーション

## 対応範囲

Apple Vision Proの両手首姿勢とピンチを、G1の14腕関節と左右Dex1-1へ
変換します。実機とシミュレーションは、誤った実行系をロードできないように
別の入口から起動します。

```bash
data/flip_table_data_augmentation/run_real_teleop.sh
data/flip_table_data_augmentation/run_sim_teleop.sh
```

両者が共有するのは、AVP入力、Unitree公式G1 29DoF IK、関節順序、14D腕位置・
14Dフィードフォワードトルク・2D Dex1開閉値のmessage、pause/resume/recordの
状態遷移、dataset schemaだけです。**安全フィルタ、watchdog後の停止処理、servo、
actuator、camera観測実装は共有しません。** 操作者・policyの出力は14腕関節と
左右Dex1だけで、腰、脚、歩行の目標値は生成しません。
操作開始には毎回`r`による明示的なクラッチが必要です。

## 実機とSimの実行境界

```text
shared: contracts / AVP+official IK / operator state / dataset schema
   ├─ real: 250 Hz rt/arm_sdk + 200 Hz Dex1 DDS + official q/torque + Regular mode
   └─ sim:  Isaac socket + 50 Hz action filter + simulated gripper/watchdog
```

実体は`teleop/real/runner.py`と`teleop/sim/runner.py`に分かれています。
実機runnerをimportしてもSim backend/safetyはロードされず、Sim runnerをimportしても
Unitree DDS/real backendはロードされないことをテストで固定しています。従来の
`run_teleop.sh real|sim`は互換用dispatcherとしてのみ残し、新しい手順や自動化では
上記2つの専用launcherを使います。

実機の`OfficialG1CommandFilter`は公式の実測姿勢基準global q scalingだけを担当し、
Simの`CommandSafetyFilter`はIsaac用の関節別速度・加速度制限だけを担当します。
同じクラスをflagで切り替える実装には戻しません。共有config内の関節範囲とtimeoutは
共通message契約の上限であり、actuator実装を共有していることを意味しません。

## 腕と下半身の制御境界

実機は公式`xr_teleoperate --motion`と同じ`rt/arm_sdk`経路を使います。
G1は**Regular mode**で起立させてください。公式が明記する通りRunning modeは
対象外です。腕packetは公式`G1_29_ArmController`に合わせて、LowStateから取得した
35 motor slot（実関節29 + protocol/reserved 6）、`mode_machine`、gainを初回に
snapshotし、腕15..28だけをAVP目標で上書きし、slot 29をarm-sdk weightとして
使います。この全身形式は`rt/arm_sdk`のprotocol要件であり、操作者が腰・脚を
指令することを意味しません。

- AVP/policyが変更できる関節: 腕15..28の14Dだけ
- Regular controllerが継続して担当: 起立、バランス、腰・脚、歩行
- 実機arm-sdk publish: 公式と同じ250 Hz
- 実機Dex1 publish: 公式と同じ200 Hz（arm-sdk clockとは独立）
- AVP目標、camera、dataset: 30 Hz
- sim servo / physics: 50 Hz / 200 Hz（実機DDS周期とは別契約）

実機の腕指令は公式`G1_29_ArmIK.solve_ik()`が返す14D関節位置と14D
フィードフォワードトルク（Pinocchio RNEAによる重力補償）を両方`rt/arm_sdk`へ
渡します。位置は公式IK内の4 sample weighted moving averageを1回だけ通し、
arm-sdk側では公式と同じく実測姿勢との差分全体を1つの係数でscaleする
20→30 rad/s（5秒ramp）の速度制限を使います。関節別の追加速度・加速度
平滑化は実機経路には重ねません。Dex1は公式と同じ実測値±0.18 rad制限と
`[0.5, 0.3, 0.2]` weighted moving averageだけを使います。

Regular controllerによる正常なバランス補正や歩行を腕の異常と誤判定しないため、
脚・腰の実測速度だけを理由にarm-sdkを解除しません。腰変位と下半身peak速度は
session reportへ診断値として残します。代わりに、追従中の`mode_machine`変更、
G1 lowstate停止、arm-sdk publish失敗は腕interlockをlatchし、制御されたweight解放へ
移行します。weight=0のDDS送信成功を確認するまでは、`r`を押しても再armしません。
下半身の自律制御と腕テレオペを同時使用する場合も、R3のsoft E-stopなど
公式の実機安全手段を常に優先してください。

## 固定バージョン

- `unitreerobotics/xr_teleoperate`: `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`
- TeleVuer: `766de45e74373ae0ea66321d942ce538385655a5`
- TeleImager: `2aab15d9601865ab6bee334ae26839e0306b0770`
- RoboFinals: `paperc/robofinals:RoboFinals-IKEA-V1`
- RoboFinals digest: `sha256:6f62a73c48a0b1f97e15e9cc76a9fee8a052c0b3b9dcd324d25c1ed01020e9f1`

2026-07-22の腕・下半身reviewでは、実行pinだけでなく公式`xr_teleoperate`
main `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`、公式
`unitree_sdk2_python` `e4cd91f051aaa77a70600e3d2bf7f50889db1980`の
`g1_arm7_sdk_dds_example.py`ともpacket、joint index、gain、250 Hz arm loop、
weight解放、Regular-mode境界を照合しています。公式checkoutは変更せず、本リポジトリ側の
明示的な起動gate、watchdog、記録だけを追加しています。

固定値の正本は
`data/flip_table_data_augmentation/teleop/configs/teleop_v1.json`です。
head-yaw-relative arm referenceと`motion_data_ready`を含む最新公式runtimeを
そのままpinし、`TeleVuerWrapper(arm_reference_mode="head_yaw")`を使います。
独自の座標変換backportや、公式`HAND_MOVE`処理を部分的に置き換える処理は
ありません。頭のpitch/rollは腕IKへ入れずyawだけを基準にするため、手を
見下ろした際に腕目標がねじれる問題を避けます。

## 操作PCの初回準備

```bash
data/flip_table_data_augmentation/setup_teleop_runtime.sh
XR_DESKTOP_IP=<AVPから見える操作PCのIPv4> \
  inference/desktop/xr/generate_avp_tls.sh
```

生成された公開CA
`~/.config/xr_teleoperate_avp/rootCA.pem`だけをAVPへ入れ、証明書を信頼
します。`rootCA.key`と`key.pem`は操作PCから持ち出しません。

`run_sim_teleop.sh`はRTX 5090端末を自動判定します。別端末から実行する
場合はTailscale SSHで`ramen-workstation`へ同期し、端末上で実行する場合
はDockerを直接使います。どちらもソースcheckoutを変更せず、隔離した
stageを使います。

## 常駐Isaac workerとの切替

Sim-to-Realの再生・CV評価とAVPは、同じ**既存の永続Isaac worker**の
ジョブキューで直列化できます。workerを先に起動しておけば、最初のIsaac
起動だけ待てばよく、AVPを`q`で終えてもworkerは次の評価を受け付けたままです。
AVPと通常評価を切り替える際は、観測cameraの契約を保つためGym環境だけを
再作成します。Isaacアプリ本体を二重起動したり、同時にsceneを所有したりは
しません。

```bash
# 最初の一回だけ（cold start）
bash evaluate/flip_table_simulation/persistent_eval.sh start

# AVP。`q`の後もworkerは残る。
FLIP_TABLE_SIM_EXECUTION=local \
FLIP_TABLE_TELEOP_SIM_OWNER=persistent \
FLIP_TABLE_TELEOP_REMOTE_PORT=59611 \
FLIP_TABLE_TELEOP_LOCAL_PORT=59611 \
FLIP_TABLE_TELEOP_FOOT_PEDAL_ENABLED=true \
FLIP_TABLE_TELEOP_DR_PROFILE=mild \
bash data/flip_table_data_augmentation/run_sim_teleop.sh

# 次のSim-to-Real評価は通常どおりqueueへsubmitする。
bash evaluate/flip_table_simulation/persistent_eval.sh submit ...

# 作業完了時だけ明示的に停止する。
bash evaluate/flip_table_simulation/persistent_eval.sh stop
```

AVP用ジョブは`FLIP_TABLE_TELEOP_PERSISTENT=true`で実行されます。これは
**worker内でのAVP sessionの寿命を明示する値**です。AVPの`q`はジョブを安全に終了し、
worker自体は次のジョブのためにready状態へ戻ります。状態とworker logはそれぞれ
`outputs/flip_table_real_to_sim/persistent_jobs/ready.json`と
`outputs/flip_table_real_to_sim/persistent_worker.log`です。

## 実機カメラサーバ

実機収録ではOrin上のTeleImagerが次をZMQ配信します。

| role | source | frame |
|---|---|---|
| `head_camera` | G1 head stereo | 1280x480 side-by-side, 30 Hz |
| `left_wrist_camera` | left D405 | 640x480, 30 Hz |
| `right_wrist_camera` | right D405 | 640x480, 30 Hz |

Orinで実デバイスを確認し、役割を明示した設定を一度生成します。

```bash
python inference/desktop/xr/prepare_avp_collection_camera_config.py \
  --head-video-id <N> \
  --left-d405-serial <LEFT_SERIAL> \
  --right-d405-serial <RIGHT_SERIAL> \
  --output /home/unitree/teleimager/cam_config_server.yaml
```

公式checkoutそのものは変更しません。Orin上の本リポジトリから、外付けの
安全launcherを使って起動します。

```bash
inference/desktop/xr/launch_orin_avp_image_server.sh --check-only
inference/desktop/xr/launch_orin_avp_image_server.sh --run
```

競技運用では、カメラ取得例外から自動復旧するcamera-only serviceをOrinへ
インストールします。これはカメラだけを所有し、Unitree DDS publisherや関節指令を
作成しません。

```bash
cd /home/ubuntu/GitHub/iros_2026_ramen
bash inference/orin/scripts/install_avp_teleimager_service.sh

# ロボット停止中の保守時間にだけ初回起動または再起動
bash inference/orin/scripts/install_avp_teleimager_service.sh --restart
journalctl -u avp_teleimager.service -f
```

serviceは取得例外時に非ゼロ終了し、2秒後に再起動します。journalの
`[safe-teleimager] health`にはhead・左右D405それぞれのserial、frame sequence、
最終frame age、例外が記録されます。Desktopは映像復旧後も腕を自動再開せず、
操作者が`r`で再anchorするまで直前targetを保持します。

同じcamera-only serviceには、学習用の損失なしMCAP recorderも含まれます。
制御portはTCP `60010`で、`start`、`stop`、`status`、`clock_sync`だけを扱い、
Unitree DDSや関節指令には触れません。MCAPは既定で
`/home/unitree/teleimager/lossless_recordings`へ書きます。開始時に10 GiB未満しか
空きがない場合、queue overflow、writer失敗、source sequence欠落のいずれかが
起きた場合は収録を不合格にしますが、腕制御は継続します。

D405にはUSB descriptor serialとRealSense API serialの2種類があり、この実機では
次の対応です。TeleImagerと`realsense2_camera`へ指定するのは右列です。

| USB descriptor (`lsusb`/udev) | RealSense API (`pyrealsense2`) |
|---|---|
| `130523070154` | `128422271925` |
| `128223071636` | `128422271048` |

片側がUSB構成に失敗すると、`pyrealsense2` 2.50.0のRSUSB backendは正常側を
列挙できても、壊れたslotの`get_info()`で`failed to set power state`を送出します。
safe launcherはdeviceごとに例外を分離し、正常側のAPI serial、失敗slot、
USB descriptor serial、`bConfigurationValue`を同じfatal logへ出します。

USB再接続でheadの`/dev/videoN`が変化または一時的に消えた場合も、launcherは
1280x480 MJPEG対応V4L nodeのudev serialとlibuvc identityを相関して再発見します。
別Webカメラが存在してもheadを一意に相関できれば起動できます。相関が曖昧な場合だけ
`HEAD_CAMERA_SERIAL=<serial>`を明示し、推測では起動しません。左右D405はserialで
固定され、片方でも見つからなければ起動しません。既にcamera serverまたは
ROS camera nodeが動いている場合も、二重所有を避けるため起動を拒否します。

実機で使う`orin_teleimager_safe_launcher.py`は、headをlibuvcのネイティブ
1280x480 MJPEGとして配信します。OpenCVでdecode→再encodeすると配信上は30 Hzでも
固有frameが約15 Hzになるため、この経路は使用しません。各取得frameはZMQへ1回だけ
publishします。左右D405は独立threadで同時取得します。30 Hzの2台を1本のlockで
直列化すると片側がready timeoutするため、取得処理を直列化してはいけません。
D405がtimeoutした場合は、公式TeleImagerと同様にcamera server全体を停止します。
古いframeを再送したり、同一process内でlibrealsense pipelineを再起動したりしません。
systemdが新しいprocessとして再起動し、USB serialからdeviceを再発見します。
Desktopは各streamを独立監視し、100 msで警告、200 msで腕とDex1を現在姿勢に
HOLD、750 msでoutageとして記録します。収録中episodeは破棄され、復旧後も
勝手に追従せず、`r`による再anchorが必要です。データセットには、各JPEGを初めて
観測した時刻の実測skewが33.3 ms以内で、全streamが前sampleから更新されたbundle
だけを書き込みます。

Dex1-1 serial-to-DDS serviceは
`inference/desktop/xr/systemd/dex1_1_gripper_server.service`をOrinへ配置し、
実機手順に従って有効化します。

公式`dex1_1_service`の`cdd9fc5`では、DDS callbackと500 Hzのモータthreadが
受信messageと更新時刻を同期せず共有しています。また、左右どちらか一方だけを
検出した場合も起動成功になり、実行中のserial通信失敗も無視します。本リポジトリは
公式topic、motor ID、gain、gear ratioを変更せず、これらの競合とhealth checkだけを
修正するpatchを管理します。Orinで次を実行してください。

```bash
cd /home/ubuntu/GitHub/iros_2026_ramen
bash inference/orin/scripts/install_dex1_service_hardening.sh
```

上のコマンドはbuildとsystemd unitの配置だけを行い、実行中serviceを変更しません。
ロボット周囲とDex1の指挟みリスクを確認した保守時間に限り、明示的に再起動します。

```bash
bash inference/orin/scripts/install_dex1_service_hardening.sh --restart
journalctl -u dex1_1_gripper_server.service -n 100 --no-pager
```

patchは以下を保証します。

- 左右motor ID 0/1が両方検出されなければ起動しない
- DDS commandの更新時刻とmessageを同一mutex下でsnapshotする
- serial通信が連続失敗した場合は異常終了し、systemdに再起動させる
- motor error codeが変化した場合、左右、mode、温度とともにjournalへ記録する

状態topicが500 Hzで見えることだけでは、位置指令への追従は証明できません。まず
非操作確認を行い、その後、指や物体をDex1から完全に外した状態で片側ずつ小振幅診断を
行います。`--execute`を付けない限りpublisherは作成されません。

```bash
# 読み取りのみ
pixi run -e runtime python -m inference.desktop.xr.check_dex1_motion \
  --interface <ROBOT_DDS_INTERFACE> --side left

# 監督下で左を0.18 radだけ動かして元へ戻す
pixi run -e runtime python -m inference.desktop.xr.check_dex1_motion \
  --interface <ROBOT_DDS_INTERFACE> --side left --execute
```

通常のテレオペでも、各Dex1目標は公式実装と同じく最新実測値から±0.18 rad以内に
制限します。モータ停止や物体接触時に目標だけが端点まで積算されることを防ぎ、左右の
DDS送信回数・失敗数・追従誤差・feedback制限状態をsession diagnosticsと
`operator_session_report.json`へ保存します。pin済みUnitree Python SDKのwriter内部で
一般例外処理が再度例外を出す既知経路もbackend側で隔離し、servo threadを黙って停止
させません。G1 lowstateまたはarm-sdk送信の障害だけを腕interlockへ接続します。
Dex1片側だけのfeedback/送信障害は、その手を最終実測位置で保持して再試行し、
`dex1_state_stale_seen_left_right`とDDS failure countをreportへ残します。ハンドだけの
障害を理由に両腕のarm-sdk所有権を落とさないためです。

## 実行前検査

シミュレーションの非操作camera probe:

```bash
FLIP_TABLE_TELEOP_TRANSPORT_PROBE=true \
FLIP_TABLE_TELEOP_PROBE_FRAMES=180 \
data/flip_table_data_augmentation/run_sim_teleop.sh
```

シミュレーションの小振幅control probe:

```bash
FLIP_TABLE_TELEOP_CONTROL_PROBE=true \
data/flip_table_data_augmentation/run_sim_teleop.sh
```

`real`はDDS interface、G1のhigh-level FSM 501/mode 0、G1/Dex1 state、
3系統の実機camera、競合controllerを非操作で検査してからAVPを開始します。
接続先は誤操作を防ぐため明示必須です。複数NICがある場合はAVPから見える
`AVP_DESKTOP_IP`も明示必須です。全実機launcherは共通file lockを取得するため、
本リポジトリの歩行・arm smoke・orchestrator・テレオペを同時起動できません。
検査中は指令をpublishしません。

```bash
G1_DDS_INTERFACE=<ROBOT_DDS_INTERFACE> \
G1_IMAGE_SERVER_IP=<ORIN_IPV4> \
AVP_DESKTOP_IP=<AVPから見えるDESKTOP_IPV4> \
  data/flip_table_data_augmentation/run_real_teleop.sh
```

MCAP転送先は既定でSSH alias `g1-orin`です。起動scriptは`ssh -G`でそのhostを解決し、
同じ有線経路をTCP 60010の時計同期にも使います。この環境ではWi-Fi側
`192.168.29.159`の同期不確かさが約2〜3 msだったのに対し、有線
`192.168.123.164`では約0.3 msでした。別環境では
`G1_ORIN_SSH_TARGET=unitree@<ORIN管理IP>`を指定し、必要なら
`G1_RECORDER_CONTROL_HOST=<低RTTのORIN IP>`を個別指定してください。起動前検査は
passwordless SSHを要求し、保存時にパスワードpromptで制御loopを止めません。

## 操作

- `r`: 両手が安定した後に現在の実測腕姿勢へ再anchorして追従開始
- `s`: 収録開始、もう一度押して成功episodeを保存
- `d`: 収録破棄、simではscene reset
- `q`: 追従中・HOLD中のどちらでも、追従停止後にarm_sdkを段階的に
  regular controllerへ戻して最終終了（左ペダル）
- `Ctrl+C`: 追従中・HOLD中のどちらでも、追従停止後にarm_sdkを段階的に
  regular controllerへ戻して最終終了

実機の`q`と`Ctrl+C`は同じ最終終了処理です。追従中でも受け付け、追従を停止して
IDLE/QUITを送り、固定した公式XR controllerと同じ約2秒をかけてarm_sdkのweightを
段階的に0へ戻します。ログにweight 0.75、0.50、0.25、0.00を出し、0.00のpublish完了を
待ってからprocessを終了します。一時停止して同じ姿勢から再開したい場合は`r`を
押します。1回目の`r`でHOLDし、2回目の`r`で再anchorして再開します。最終終了後、
強制kill、PC電源断、通信断後までソフトウェアだけで保持し続けることはできません。

AVPを外す、手が隠れる、WebSocketが切れるなどでtrackingが失われると、
新しい指令を停止して実測姿勢を保持します。再開時は`r`を押し、左右の
有効なhand eventが安定するまで待ちます。切断前のanchorやIK履歴は再利用
しません。公式TeleVuerでは左右の手首とhand stateが1つの`HAND_MOVE` eventに
含まれるため、共有メモリへの書き込みと読み取りもそのevent全体を1つのlockで
扱います。読み取りが一時的に間に合わない場合もprocessを終了せずHOLDへ移行し、
同じtracking generationでは追従を再開しません。手が再び安定してから`r`で
新しいgenerationを発行した場合だけ再anchorします。

追従中は、固定した公式`xr_teleoperate`のTeleVuerが返す左右の絶対手首姿勢を
公式G1_29 IKへ直接渡します。開始・再開直後の1 targetだけは実測姿勢を保持して
IK履歴を初期化しますが、独自の相対座標offsetやdeadband変換は挟みません。
制御processはXR/motor stateからIKを先に計算して親processへ返し、その後にcamera
JPEG decode、HUD合成、AVP/Desktop描画を行います。描画遅延がIK targetの生成時刻や
応答待ち時間へ加算されない構成です。

保存側の`s`はファイル確定だけを行い、追従やarm_sdk所有権を解除しません。
したがって保存直後も同じanchorでテレオペを継続できます。別episodeを取り始める場合は
もう一度`s`を押します。

Simの`s`もライブAVP encoder設定を変更しません。dataset用30 Hz映像はoffline replayで
生成し、ライブ表示は録画中も低遅延JPEG profileのままです。

## 映像と保存契約

`sim`では低遅延を優先して、従来どおりhead-left/head-rightだけを真の
立体ペアとして表示します。`real`では、少し小さくした中央の真の立体ヘッド映像の
左右に、コンパクトな**単眼診断HUD**を重ねます。

- 左: 左D405リスト映像と左腕7関節の実測角度（度）
- 中央: 対応するhead-left/head-rightの640x480立体映像
- 右: 右D405リスト映像と右腕7関節の実測角度（度）

リスト映像と角度HUDは左右眼に同一内容を複製するため、偽の視差を作りません。

AVP表示は公式TeleImagerの`SNDHWM=1`、`RCVHWM=1`というlatest-only経路を維持し、
低遅延を優先します。Desktop側は`request_bgr=False`でJPEGだけを120 Hz pollingし、
ヘッドJPEGを一度だけdecodeして左右へ分割します。この表示経路のframe dropは許容し、
学習データの完全性判定には使いません。

学習用映像はOrinのcamera acquisition threadから直接、次の3物理streamをMCAPへ
保存します。

- `head_stereo`: 1280x480 JPEG。変換時だけ左右640x480へ分割
- `left_wrist`: 640x480 D405 JPEG
- `right_wrist`: 640x480 D405 JPEG

各recordにはUSB serial、source sequence、Orin capture monotonic時刻、JPEG SHA-256、
D405のdevice frame counter/timestamp/domainを保存します。Desktopの数値traceと
Orin時刻はTCP 60010の複数回ping-pongから最小RTT群を選んで対応付け、1秒ごとに
offsetを更新し、drift、RTT、推定誤差をmanifestへ残します。wall clockは使用しません。

実機HUDは、視野周辺へ行き過ぎないようリスト映像を小さな補助パネルにしています。
中央のヘッドステレオ像も従来より小さく表示します。HUDに **HANDS READY** と表示される
まではハンド姿勢を使った追従開始を受け付けません。両手の追跡が安定してから `r`
（右ペダル）を押してください。**HANDS WAIT** の間に `r` を押しても、後から勝手に
追従を開始することはありません。
中央のヘッド映像だけがステレオです。既定の公式TeleVuer `ego`モードを使うため、
Virtual Reality中もこの表示の周囲はAVPのパススルーです。HUDは操作者の確認用で、
policy入力や学習データの画像内容を変更しません。policy入力は次の3枚です。

実機teleopを起動すると、Desktopにも
`IROS 2026 RAMEN - Teleoperation Monitor`という独立windowを自動表示します。
上段にhead-left/head-right、下段にleft/right D405を配置し、各wrist映像へ対応する
左右7関節ずつの実測角度と`HANDS READY/WAIT`を重ねます。Desktop表示は別processの
最新frame-only経路であり、描画が遅れてもAVP IK・制御・30 Hz収録を待たせません。
windowを閉じてもteleopは継続します。GUIを使わない場合だけ、起動前に
`FLIP_TABLE_TELEOP_DESKTOP_PREVIEW=false`を指定してください。

- `cam_0`: head-left RGB 640x480
- `cam_2`: left D405 RGB 640x480
- `cam_3`: right D405 RGB 640x480

head-right、global camera、object pose、contact、simulator state、成功判定は
policy schemaへ入りません。sim固有値は診断sidecarとoffline検証だけに
保存します。各収録はcamera timestamp、30 Hz cadence、RGB/JPEG形式、
privileged情報の非混入を監査し、違反したepisodeはrelease対象外です。
実機ではさらに、source sequence、D405 frame counter、MCAP message count、
JPEG SHA-256を照合します。旧latest-only Desktop収録の
`recording_sample_rate_below_28hz`判定は使いません。

確定時はhead stereoを基準に正確な`1/30秒`のcanonical timelineを作り、左右D405の
最も近い未使用frameを20 ms以内で一対一対応させます。画像補間や光学フロー生成は
行いません。欠落時に固定長出力が必要なら直前画像をplaceholderとして置きますが、
`camera_valid=false`と元sequence/timestampを必ず残します。同じsource sequenceを
正常frameとして二度使いません。有効率99.5%以上かつ連続欠落1 frame以下だけを
候補とし、それ以外は`raw/rejected/`へ隔離します。無効行と、それをまたぐ履歴windowは
`camera_valid`列を読むFlow Matching dataset adapterが現在画像・履歴・action horizonを
まとめて学習対象から除外します。従来の27.693 Hz episodeを画像複製で昇格させることは
ありません。

3台の物理cameraは別々のclockで動き、D405には外部trigger同期がないため、全物理frameを
20 ms以内で常に100%対応させられるとは限りません。これは同じ画像を複製して隠さず、
上記のvalidity情報とepisode合否へ反映します。各物理streamが30 Hzで連続保存できたかと、
3 streamを学習行へ対応付けられた割合は別々の指標です。

`s`の1回目ではclock syncとOrin recorder ACKをbackgroundで実行し、ACK完了後から
Desktop数値traceとOrin MCAPを同時収録します。通信timeout中も腕指令loopは止まりません。
TRACKから安全HOLDへ移った場合も、実際に送信したHOLD targetを30 Hzで記録し続け、
数値traceの空白を後から補間しません。2回目の`s`でOrin側を先に確定します。
`d`による破棄、MCAP finish/fsync/SHA、Desktop転送、30 Hz変換も追従loop外の
background threadで行うため、開始・保存・破棄操作で腕指令を止めません。
`operator_session_report.json`では
`preview_transition_hz`、`preview_bundle_hz`と、各episodeの
`recorded_source_hz`を別々に確認できます。

simで`r`の後に` s `を押して収録を開始すると、AVP表示用の低遅延stereo映像とは別に、
**加工前のRGB review video**も記録します。head-left/head-right、left/right
wrist、global（俯瞰）の5視点で、検出枠・UI・camera randomizationは入りません。
制御と学習用cameraの30 Hz経路を妨げないようreview videoは標準5 Hzです
（`FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ=1..10`で変更可能）。収録停止までだけ
保存され、出力はその起動の
one-shot時は`outputs/flip_table_teleop/.../runtime_output/...`、常駐worker時は
`outputs/flip_table_real_to_sim/avp_teleop/.../test_0/`以下にRoboFinalsのvideoとして
生成されます。これは操作レビュー用であり、policyやcriticの入力には使いません。

成功として確定したsim収録では、live表示のfpsに依存しない30 Hzの19D command
trajectoryを保存し、AVP job終了後に永続workerがhead stereo・左右D405・globalを
再レンダリングします。生成されたraw episodeだけが学習データ候補であり、previewや
review videoをデータセットのフレームとして流用しません。

詳細な収録、Mimic、HF release手順は
`data/flip_table_data_augmentation/README.md`を参照してください。
