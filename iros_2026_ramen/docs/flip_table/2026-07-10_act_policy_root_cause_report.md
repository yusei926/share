# ACT Policy Root-Cause Report

> **状態: 歴史的な原因調査（現行の実行仕様ではない）。**
> 調査期間は2026-07-10から2026-07-14である。定量値は当時のcheckpoint、V1 overlay、
> 固定条件に対する証跡として残す。シミュレーション契約の最終的な扱いは
> [2026-07-14 simulation contract audit](2026-07-14_simulation_contract_audit.md)、
> 現在の方針と未達事項は
> [2026-07-17 simulation investigation summary](2026-07-17_simulation_investigation_summary.md)
> を正とする。

- 最終レビュー: 2026-07-17
- 対象: IROS 2026 RAMEN `flip_table`用ACTポリシーの初期シミュレーション評価

## 1. エグゼクティブサマリー

ACTポリシーがシミュレーション内でほとんど有効動作を生成しない主因は、**実機学習画像とシミュレーション画像の分布差（sim-real visual gap）**である。

実機画像を入力した場合、学習済みACTはGTに近い大きさの腕動作chunkを生成し、no-op baselineより腕chunk MSEを82.9%削減する。一方、同じcheckpointへsim画像を入力すると、予測される腕動作rangeは実機画像入力時の約1/4.5まで縮小する。stateだけを差し替えても回復せず、画像だけを実機へ差し替えると大きく回復するため、主因を画像分布差と特定した。

当時の監査では、19次元action経路、上半身関節順序、target適用、関節応答が正常だった。
したがって当時の条件では「simulatorがactionを無視している」「17D/19Dの対応が壊れている」
という説明は否定できた。ただし、これはシミュレーション全体、接触物理、またはSim-to-Real妥当性を
保証する結論ではない。後続監査で複数のsimulator実装不具合が見つかっている。

二次要因として、次の2件を確定した。

1. `n_action_steps=100`により、最大100 actionの間、新しい観測を使わないopen-loop実行になっている。学習30 Hz、sim制御50 Hz、simカメラ20 Hzの時間軸も一致していない。
2. 評価adapterがG1 Dex1の`gripper_pos`を左右手状態として誤解釈し、実際には右手の2関節だけから両手stateを作っている。

また、現checkpointはaugmentationなし、validationなし、実効0.687 epochであり、simへの頑健性を十分に獲得していない。ただしこれは主因ではなく、まずadapter・時間軸・visual alignmentを修正してから再学習を判断すべきである。

## 2. 調査対象

### 2.1 学習データとモデル

- データセット: 531 episodes、290,941 frames、30 fps
- policy入力画像: `head_left`、`left_wrist`、`right_wrist`
- 画像解像度: 640 x 480
- state: `robot_q_current`
- action: `robot_q_desired`
- 上半身順序: waist 3 + left arm 7 + right arm 7 + hands 2
- checkpoint: `act_flip_table_upper_body`
- `chunk_size=100`
- `n_action_steps=100`（当時の評価値。現行評価の既定値は10）

### 2.2 評価環境

- 運営提供RoboFinals IKEA V1ベース
- G1 + Dex1-1
- 下半身はpolicy対象外で固定
- policy画像: head 1枚 + left wrist + right wristの3枚
- global cameraは録画・診断専用で、policy入力ではない

## 3. 調査方法

以下を独立に切り分けた。

1. 実機画像に対してACTが意味のあるactionを出すか
2. 実機画像とsim画像を交換したとき、予測chunkがどう変わるか
3. head・left wrist・right wristのどの画像が出力低下に寄与するか
4. stateだけを実機・simで交換したときに回復するか
5. adapter出力とsimへ送信される19D actionが一致するか
6. sim内のtargetとactual jointが追従するか
7. `n_action_steps=100`と10で動作量が変化するか
8. Dex1 hand stateの取得元が左右手を正しく表すか
9. checkpoint推移からモデルが学習中か、崩壊しているか

## 4. 定量結果

### 4.1 実機画像に対するACT性能

実機54サンプルの集計結果は次の通りである。

| 指標 | ACT | GT |
|---|---:|---:|
| arm chunk range平均 | 1.734 rad | 1.871 rad |
| arm chunk MSE | 0.007687 | - |

54 episodeのframe 10だけに揃えた比較:

| 指標 | 値 |
|---|---:|
| ACT arm chunk range平均 | 1.399 rad |
| GT arm chunk range平均 | 1.555 rad |
| ACT arm chunk MSE平均 | 0.007832 |
| hold-current/no-op baseline MSE平均 | 0.045849 |
| no-op比MSE削減率 | 82.9% |

この結果から、ACTは実機ドメインでは有効な動作分布を学習している。

### 4.2 checkpoint推移

固定した実機12サンプルに対する結果:

| Step | first arm error | arm chunk MSE | predicted range |
|---:|---:|---:|---:|
| 10k | 0.318804 | 0.018745 | 1.064532 |
| 50k | 0.169910 | 0.011409 | 1.367478 |
| 80k | 0.150270 | 0.009297 | 1.333442 |
| 90k | 0.143576 | 0.008900 | 1.327374 |
| 100k | 0.129245 | 0.008070 | 1.444386 |

10k/50k/100k比較時のGT rangeは1.62308である。100k時点でもerrorとMSEは改善しており、学習が崩壊または完全に飽和しているとは判断できない。

### 4.3 sim-realクロス入力

中盤の実機サンプルを用いた比較:

| 画像 | state | arm chunk range |
|---|---|---:|
| real | real | 2.537 |
| sim | sim | 0.344 |
| sim | real | 0.483 |
| real | sim | 1.641 |

初期状態に近いnearest real frame-10での比較:

| 画像 | state | arm chunk range |
|---|---|---:|
| real | real | 1.505 |
| sim | sim | 0.344 |
| sim | real | 0.219 |
| real | sim | 1.548 |

stateを固定した状態で画像だけをsimからrealへ変えると、腕chunk rangeは約4.5倍になる。逆にsim画像のままreal stateへ変更しても回復しない。このため、state差ではなく画像差が主因である。

### 4.4 カメラablation

sim stateを固定し、各画像だけを実機画像へ交換した。

| 画像構成 | arm chunk range |
|---|---:|
| all sim | 0.3438 |
| real head only | 0.6889 |
| real left wrist only | 0.7657 |
| real right wrist only | 0.4199 |
| real wrists | 0.8841 |
| real head + left wrist | 1.5291 |
| real head + right wrist | 0.7639 |
| all real | 1.5479 |

headとleft wristを実機画像へ交換した時点でall realに近い出力まで回復する。最重要のvisual gapはheadとleft wristであり、right wristの寄与は相対的に小さい。

### 4.5 action経路とsim応答

runtimeで確認したaction manager shapeは19で、構成は次の通りである。

- waist: 3
- left arm: 7
- right arm: 7
- left hand: 1
- right hand: 1

関節index:

- waist: `[2, 5, 8]`
- left arm: `[11, 15, 19, 21, 23, 25, 27]`
- right arm: `[12, 16, 20, 22, 24, 26, 28]`
- left hand joints: `[29, 30]`
- right hand joints: `[31, 32]`

adapter raw arm actionとsimへ送信される19D actionの最大差は0だった。

固定条件、`n_action_steps=100`の実測:

| 指標 | 値 |
|---|---:|
| start-to-end arm norm | 0.3175 |
| full arm state range norm | 0.5233 |
| target range | 0.5020 |

simはactionを適用し、関節も応答している。見た目でほぼ静止する理由は、ACTが現在姿勢付近の保守的targetしか出さず、対象へのリーチや接触に必要な動作を生成していないためである。

### 4.6 n_action_steps比較

学習データは30 Hzなので100 stepは3.33秒である。一方、当時のsimは`dt=0.005`、decimation 4で制御50 Hzとなり、100 stepは2.0秒で消費された。カメラ更新は0.05秒、20 Hzだった。

LeRobotの`select_action`はaction queueが空になるまで新しい観測から`predict_action_chunk`を呼ばない。このため`n_action_steps=100`では、最大100 actionをopen-loopで実行する。

固定初期条件での比較:

| 指標 | n=100 | n=10 |
|---|---:|---:|
| start/end arm norm | 0.3175 | 0.5105 |
| state range | 0.5233 | 0.7136 |
| max single-joint range | 0.2001 | 0.3657 |
| mean target delta | 0.0950 | 0.1237 |
| success | 0/1 | 0/1 |

`n_action_steps=10`で動作量は増えたが成功には至らない。したがって100-step open-loopは悪化要因だが、唯一の原因ではない。

## 5. 根本原因

### 5.1 主因: sim-real画像分布差

実機head画像では、銀色の前腕、しわのある黒い布、作業場の背景、腕とテーブルが画面内を大きく占有する構図が見られる。sim画像では、滑らかな黒平面、単純化した部屋、黒い手形状、清潔な白テーブルとなっている。

実機wrist画像は腕、テーブル脚、作業対象による近接・遮蔽が大きい。sim wrist画像は広角で対称的、対象が相対的に遠く、画像が清潔すぎる。

現在の`x115 z70 down20`候補は目視上の改善はあるが、ACT出力を指標にすると依然OODである。カメラ候補の採用判断をcontact sheetや色比率だけで行うのは不十分である。

学習時head入力は実機`cam_0/head_left`である。sim評価では`first_person_camera`を対応させているが、実機左ステレオカメラと光学・取付・歪み・露出が厳密に同じとは確認できていない。

### 5.2 二次要因: hand state adapterの誤読

評価adapterは`gripper_pos`が存在するとそれを左右手状態として使う。公式V1の`gripper_pos`は`joint_pos[-1]`と`-joint_pos[-2]`を返す。

実runtimeの33 joint末尾は次の順序だった。

| index | joint |
|---:|---|
| 29 | left_finger_1 |
| 30 | left_finger_2 |
| 31 | right_finger_1 |
| 32 | right_finger_2 |

実測tail valuesは`[-0.0020, -0.0121, -0.0096, -0.0200]`、`gripper_pos`は`[-0.0200, +0.0096]`だった。つまり左右手ではなく、right finger 2と符号反転したright finger 1を返している。adapterはこれを左右手と解釈するため、両hand state channelが破損する。

action command変換は公式Dex1 conventionの`-1=open`、`+1=closed`と、`OPEN_POS=0.0245`、`CLOSE_POS=-0.02`に対して正しい。

hand stateだけをoffline補正した場合のarm range:

| hand state | arm range |
|---|---:|
| bugged | 0.3438 |
| dataset frame0 median | 0.3228 |
| direct LR example | 0.3333 |
| both closed | 0.2817 |
| both open | 0.4482 |

hand state bugは腕静止の主因ではないが、左右hand状態とgripper制御の整合性に関わるため、再評価前に必ず修正する。

### 5.3 二次要因: 時間軸不整合

- dataset: 30 Hz
- sim control: 50 Hz
- sim camera: 20 Hz
- action chunk: 100
- queued action execution: 100

観測更新、policy再推論、action消費の周期が一致していない。`n_action_steps`を減らすだけでなく、30 Hzの学習action列を50 Hz制御へどうresampleするか、20 Hz画像で何回推論するかを仕様として固定する必要がある。

### 5.4 増幅要因: 学習設定

- 100k steps x batch 2 = 200,000 sampled observations
- dataset frames比で約0.687 epoch
- `image_transforms.enable=false`
- `eval_split=0.0`
- `eval_steps=0`
- environment evaluationなし
- 最終runはW&B disabled
- validation/checkpoint selectionなし

100k時点でも改善が続いているため、学習量は十分とは言えない。augmentationなしで実機画像だけを学習したモデルに強いsim domain randomizationを与えると、かえってOODを増やす可能性がある。

## 6. 否定できた原因

| 候補 | 判定 | 根拠 |
|---|---|---|
| upper-body joint ordering間違い | 否定 | runtime indexと19D構成を確認 |
| actionがsimで捨てられる | 否定 | raw actionと送信actionが一致し、jointが応答 |
| 17D/19D mismatch | 否定 | waist 3 + arms 14 + hands 2を確認 |
| state/action sourceが逆 | 否定 | state=`robot_q_current`、action=`robot_q_desired` |
| normalizationの二重適用・欠落 | 否定 | configとcheckpoint statsが一致 |
| 画像解像度不一致 | 否定 | 学習・評価とも640 x 480を確認 |
| モデルが完全に未学習 | 否定 | 実機画像でGTに近い出力、no-op比82.9%改善 |
| lower-body lockがarmを上書き | 否定 | lock対象はhip/knee/ankle/rootのみ |
| global camera | 否定 | policy入力ではなく録画・診断専用 |

## 7. 修正優先順位

### P0: 再学習前に修正

1. `joint_pos`からnamed Dex1 left/right jointsを直接取得する。
2. finger 1/2のどちらを使うか、または平均するかを明示してdataset hand conventionへ変換する。
3. G1 Dex1ではgeneric `gripper_pos`を使用しない。
4. `ACT_N_ACTION_STEPS`を明示設定可能にし、初期候補を10とする。
5. episode reset時にaction queueを必ずclearする。
6. dataset 30 Hz、sim control 50 Hz、camera 20 Hzの時間軸を整合する。
7. camera role/resolution、state/action dimension/order、hand range、policy inference count、target/actualをruntime assert・logへ出す。
8. randomizationを無効化したdeterministic smoke testを用意する。

### P1: visual alignment

1. 実機head-leftとD405 wristのintrinsic/extrinsicを計測値、CAD、bracket情報から決定する。
2. crop、FOV、distortion、exposure、取り付け位置、物理的occlusionを実機に合わせる。
3. 作業机の布・texture、G1の銀/黒外観、背景clutter、照明を実機へ近づける。
4. headとleft wristを優先して整合する。
5. 候補比較には画像の目視だけでなく、feature/domain distanceとACT arm chunk responseを使う。
6. 原因切り分け中はroom randomizationを弱め、固定条件で比較する。

### P2: 学習改善

1. episode単位のheld-out splitを作る。
2. brightness、contrast、saturation、hue、sharpness、affine等を慎重に有効化する。
3. 実機held-outとsimの両方でcheckpointを定量選定する。
4. 1 effective epoch以上へ継続学習する。
5. 必要ならreal + sim、またはimage-translated dataでfine-tuneする。
6. `n_action_steps=5-10`またはtemporal ensembleを検討する。`chunk_size=100`自体は保持可能。
7. P0/P1完了前の再学習は避ける。

## 8. 再評価手順

1. Scripted joint action testで19D経路とtarget/actual追従を確認する。
2. randomization offの固定sceneでACTを3 episodes実行する。
3. head、left wrist、right wristのACT応答をablationで再確認する。
4. 固定sceneで明確なreach/contactが出た後、randomized 10 episodesを実行する。
5. 各episodeで4カメラ映像、success state、policy action、sim target、actual jointを保存する。

### 合格基準

- scripted testでtargetとactualが期待方向へ追従する。
- 固定3 episodesすべてで、少なくとも明確なreach/contactを確認する。
- sim入力時のarm chunk rangeが、実機frame-10平均1.399 radへ近づく。
- 単なる関節微動を成功と判定しない。
- randomized 10 episodesで成功率を記録する。
- 成功率0%の場合、再学習前にvisual/state/timebase traceを再診断する。

## 9. 証拠ログと成果物

### Vastログ

- `/workspace/logs/flip_table_act_root_cause_current.log`
- `/workspace/logs/flip_table_act_root_cause_n10.log`
- `/workspace/logs/flip_table_act_root_cause_fixed_n100.log`
- `/workspace/logs/flip_table_act_root_cause_fixed_n10.log`
- `/workspace/logs/flip_table_act_gripper_source_diag.log`

### Vast出力

- `/workspace/iros_2026_ramen/outputs/flip_table_simulation/act_root_cause_current`
- `/workspace/iros_2026_ramen/outputs/flip_table_simulation/act_root_cause_n10`
- `/workspace/iros_2026_ramen/outputs/flip_table_simulation/act_root_cause_fixed_n100`
- `/workspace/iros_2026_ramen/outputs/flip_table_simulation/act_root_cause_fixed_n10`

### ローカル

- `/tmp/act_flip_table_output_diagnosis_current_54samples.json`
- `model/subtask_policy_training/outputs/train/act_flip_table_upper_body`

### 関連コード

- `evaluate/flip_table_simulation/container_overlay/policy/flip_table_eval_policy.py`
- `evaluate/flip_table_simulation/container_overlay/patches/patch_g1_global_camera.py`
- `evaluate/flip_table_simulation/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py`
- `model/subtask_policy_training/scripts/diagnose_act_policy_outputs.py`
- `model/subtask_policy_training/configs/subtask_training.json`
- `model/subtask_policy_training/outputs/train/act_flip_table_upper_body/config.json`
- `model/subtask_policy_training/outputs/train/act_flip_table_upper_body/train_config.json`
- `model/subtask_policy_training/outputs/training_views/act_flip_table_upper_body/meta/info.json`

## 10. 制約事項

- camera extrinsicはまだ計測ベースの完全校正ではない。
- `n_action_steps=100/10`比較は固定条件の限定試験である。
- trainingにvalidation splitがなく、generalizationの正式評価ではない。
- 現状は根本原因調査までで、恒久修正後の10 episode成功率は未測定である。
- `x115 z70 down20`は目視候補として採用した値であり、ACT応答ベースでは最終確定値ではない。

## 11. 最終判断

再学習を直ちに行うべきではない。まずP0のhand state・action queue・時間軸を修正し、P1でhead/left wristを中心に実機分布へ合わせる。その状態で固定scene評価を行い、sim入力に対するACTのarm chunk rangeが回復するかを確認する。

P0/P1後も実機画像では動くがsim画像で動かない場合、sim画像を含むfine-tuningまたはdomain adaptationが必要である。実機held-outでも性能不足が確認された場合に限り、学習量・augmentation・validationを含むP2へ進む。

## 12. 2026-07-11追加検証

### 12.1 G1/Dex1外観の恒久修正

> 2026-07-14訂正: OCI最終レイヤーを監査した結果、Dex1 classと`G1_GRIPPER.usd`は
> `RoboFinals-IKEA-V1`原本には含まれず、競技実機に合わせて後から追加された資産だった。
> 以下の「運営V1のG1_GRIPPER」という表現は履歴上の記述である。現在の正確な由来と
> シミュレーション監査結果は
> [2026-07-14 simulation contract audit](2026-07-14_simulation_contract_audit.md)を参照すること。

運営V1の`G1_GRIPPER.usd`はメッシュ、マテリアル参照先、variantを持つが、39個のvisual primで`UsdShade.MaterialBindingAPI`が欠落し、レンダリング時に無彩色に見えていた。現在のstartup patchは公式V1の幾何、binding target、collision、massを変更せず、欠落schemaのみを補う。色・metallic・roughness・specularはUnitree公式`unitree_sim_isaaclab` G1/Dex1 USDの値に合わせた。運営の`Physics=PhysX`、`Robot=Robot`、`Sensor=Sensors` variantは維持し、元ファイルはバックアップ後にパッチする。

### 12.2 ロボット配置の座標系を修正

V1の作業台はワールドで約90度回転している。従来のworkbench-local `-y`を正面と解釈すると、ロボットが黒い作業台の反対側へ配置されていた。`+y`側へ変更したframe 10では、head/left wrist/right wristの各視点に白いテーブルと黒い作業台が入り、手先も作業領域へ到達する。±0.1 mの距離・横方向randomizationは維持する。

### 12.3 実機episode 0のq_desired再生

実機episode 0の映像は、開始時の天板・脚の状態から途中で天板が立ち上がるため、失敗例とは考えにくい。19-Dに切り出した同じ`robot_q_desired`をVast上のV1へ再生した。

- resetを全データ中央値の姿勢にした場合: 指先との脚距離は約0.16〜0.45 m、接触力0 N、テーブル姿勢不変。
- episode 0先頭targetをreset姿勢へ適用した場合: 初期の一時的接触は発生するが、継続接触はなく、テーブル姿勢は不変。
- 正しい`+y`側配置でも、episode 0全再生のsuccessは`0/1`。
- runtimeでは天板・4脚がすべて`rigid=True, kinematic=False`、4本のFixedJointが有効である。

この結果から、action index/target適用だけを原因とする説明は不十分である。残る主な候補は、データセットに記録されていない実機のroot座標・向き、実機とV1の接触形状/摩擦/テーブル配置、及び腕q軌跡とシミュレーションの手先位置の差である。episode replayは初期rootを推定している診断であり、最終評価ではない。

元の36-D `robot_q_desired`も確認した。episode 0の先頭有効フレーム以降はroot poseが概ね`(x,y,z)=(2.1,-2.56,0.70)`、quaternionのyawが約`-0.797 rad`で推移する。一方、現在の学習viewはroot 7次元を意図的に除外して19-D上半身へ変換している。実機座標とV1座標の対応関係が未校正のままこの値を本番へ直接使うと、歩行/root制御を隠れて導入することになるため、現時点では本番デフォルトに採用しない。

### 12.4 配置修正後ACTの再評価

Vast上で公式USD、`+y`側配置、3入力640x480、`n_action_steps=10`のACTを80 sim steps実行した。

| 指標 | 旧配置 | `+y`側配置 |
|---|---:|---:|
| sim ACT arm target range norm | 0.611 | 0.810 |
| 実機frame-10 ACT arm target range平均 | 1.399 | 1.399 |
| success | 0/1 | 0/1 |

配置修正によりACT出力は増えたが、実機分布へは未到達である。従って、現時点で「ACTの学習失敗だけ」または「配置だけ」を単独の根本原因とは断定しない。次は、実機root poseを固定した接触校正、D405/headの画像幾何差の定量比較、固定sceneでのACT ablationを行う。

### 12.5 Sim-to-Real用domain randomization

背景はworld-fixed座標で床、壁、照明、窓、視覚的な家具を変化させる。床と壁は木目、コンクリート、タイル、ビニール、塗装、レンガ等の再生可能なtextureを使う。家具はpolicy画像に寄与するロボット前方のみに配置し、タスク領域から安全距離を取る視覚専用アセットとした。診断用`global_camera`もworld-fixedで、policy入力には使わない。

接触はPhysXの`average` combineを使い、次の実在的なpair範囲から各surfaceの係数を逆算する。

| pair | static friction | dynamic friction | restitution |
|---|---:|---:|---:|
| Dex1 hand - white table | 0.55-0.85 | 0.40-0.68 | 0.02-0.08 |
| white table - black workbench top | 0.35-0.60 | 0.25-0.48 | 0.01-0.05 |
| black workbench top - Dex1 hand | 0.50-0.80 | 0.35-0.65 | 0.02-0.08 |

対象collision shapeはDex1 8、白テーブル926、黒作業机天板1であり、黒作業机の脚や下部supportは対象外である。reset中はmaterialの新規作成やbinding変更を行わず、startupで事前bindしたattributeのみ更新する。これによりPhysX tensor viewの無効化を避けた。`contactOffset`/`restOffset`は実体パラメータではなく数値接触生成用のため、Sim-to-Real randomization対象に含めない。

### 12.6 最新ACTランダム条件評価

2026-07-11に部屋、照明、ロボット/テーブル初期位置姿勢、接触係数をすべてrandomizeし、ACTを3 episodes評価した。入力はhead-left、left/right D405の各640x480、出力は19-D上半身関節target、`n_action_steps=10`、下半身とrootは固定である。

| episode | 成功 | 腕関節最大range | target/actual MAE |
|---|---:|---:|---:|
| 0 | 0 | 1.288 rad | 0.028 rad |
| 1 | 0 | 0.579 rad | 0.033 rad |
| 2 | 0 | 0.736 rad | 0.029 rad |

成功率は`0/3`で、白テーブルへの有効なreach/contact/flipは発生しなかった。一方、各episodeは500 sim steps、301 policy inferencesを完走し、関節targetとactualの追従も正常で、CUDA、tensor view、randomizationエラーは0だった。従って現時点の失敗は「シミュレータがactionを適用していない」ことではなく、現ACTがランダム化したsim画像から有効な接触軌道を生成できないことにある。

## 13. 2026-07-14シミュレーション契約再監査

上記の「action適用経路は正常」という結論は維持するが、「シミュレーション全体に問題がない」という
意味ではない。追加監査で、Vastに残った未管理arm actuator、fixed評価時のDex1接触材質欠落、途中まで
傾いただけでも成立し得る成功判定を発見し、修正した。

修正後はpolicy非依存試験で、19-D action、下半身/root lock、4本の固定関節、天板・脚・作業台寸法、
外力・トルク応答、接触材質binding、低/高摩擦応答、固定およびrandomized reset invariantを検査する。
詳細とVast上の証拠パスは
[2026-07-14 simulation contract audit](2026-07-14_simulation_contract_audit.md)を参照すること。

それでもDex1 collision geometry、実機摩擦、腕step response、camera hand-eyeは実測未照合であり、
クリーン環境で成功したscripted flipもまだない。従って、現ACT/Flowの失敗をpolicyだけの問題とも、
simだけの問題とも断定しない。

## 14. 2026-07-14追加訂正: simulator側の実装不具合

その後の監査で、policy評価へ影響する次の問題も確認して修正した。

- GPU非対応の白脚collision-shape filterにより、白脚限定contact forceが常に0だった。
- reset後も前episodeのactuator targetとPhysX contact cacheが短時間残っていた。
- 非同期部分resetが、継続中の他環境のinitial table normalを0へ破壊していた。
- RL経路ではlower-body lockの呼び出しが消える場合があった。
- multi-env照明で全環境が再抽選され、距離減衰しない光が環境数ぶん重なっていた。
- 診断用table poseが共有tensor viewで、実変位を0と誤記録していた。

修正後のpolicy非依存監査は17/17 gate合格である。また単一環境をresetから再生した既知の
19-D関節軌道で、右Dex1の把持品質最大`0.99983`、最大finger force`10.51 N`、白天板中心の
最大上昇`0.05654 m`を確認した。下半身関節とrobot rootの変位は0だった。

これにより「actionが物理へ届かず物体が全く動かない」という説明は否定できる。一方、これは片側で
白テーブルを傾けたpositive controlであり、両手把持、180度flip、安定設置、実機一致の証明ではない。
過去の非同期vector RL、旧contact filter、旧multi-env照明で得た結論は再評価する。詳細は
[2026-07-14 simulation contract audit](2026-07-14_simulation_contract_audit.md)を正とする。

## 15. 2026-07-17時点の扱い

本書の実機画像とsim画像を入れ替えた比較は、**当時のACT checkpointがsim画像へ弱かった**ことを
示す有力な証拠である。ただし、現在も「visual gapだけが根本原因」とは断定しない。後続監査で
contact filter、reset、lower-body lock、照明、診断snapshotなどに不具合が見つかり、修正済みである。

現行の評価契約は、`head_left`、`left_wrist`、`right_wrist`の各640x480 RGBと19-D上半身関節状態を
入力し、19-D上半身絶対関節targetだけを出力する。`head_right`、global camera、object pose、
segmentation、contactなどはpolicy入力に使わない。評価時のACT action queueは既定で
`FLIP_TABLE_ACT_N_ACTION_STEPS=10`であり、runnerはV1 taskの実際のcontrol rateを読み取って
policy adapterへ同期する。

この変更は、旧checkpointの成功を意味しない。full flip、固定scene 3/3、randomized 10 episode、
実機での再現はいずれも未達である。新規学習または比較実験は、必ずV1 image/overlay hash、seed、
camera動画、action/state trace、成功判定を保存し、契約監査を再実行してから比較する。
