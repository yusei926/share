# Simulation Contract Audit

> **状態: V1 overlayの内部契約監査。end-to-end成功またはSim-to-Realの証明ではない。**
> 証拠取得日は2026-07-14、最終レビュー日は2026-07-17である。数値は当時のV1 image、
> overlay、seed、単一環境条件に固有であり、asset、physics、overlay、scene生成を変更した場合は
> 監査を再実行する。

## 結論

「policyだけが悪く、シミュレーションは正しい」という前提は誤りだった。方策非依存監査と
既知関節軌道による試験で、シミュレーション・評価・診断経路に複数の実装不具合を確認して修正した。

修正後は、次の範囲を実測で確認している。

- 19-D上半身関節targetが正しい関節順で適用される。
- Dex1左右hand commandが独立して正しい開閉量へ変換される。
- 下半身12関節とrobot rootはpolicy実行中に移動しない。
- 白テーブルは4本のfixed jointで組み立てられ、外力・トルク・摩擦差へ応答する。
- Dex1と白脚の接触相手、接触力、把持幾何を取得できる。
- 既知の右手軌道だけで、安全力範囲内の把持と白テーブル中心の5.65 cm上昇を再現できる。
- 2環境の部分resetで、非reset環境のtask state、質量、接触材質、部屋、照明を変更しない。

したがって、現在は「関節targetが物理へ届かない」「白テーブルが固定されている」「接触が常に0」
という種類の根本不具合を否定できる。ただし、ロボットによる180度反転と安定設置はまだ未達であり、
実機のcamera hand-eye、collision geometry、摩擦、関節応答との一致も未証明である。現時点の正確な
表現は、**主要な内部契約は合格したが、end-to-end物理とSim-to-Real妥当性は未完了**である。

## 発見・修正した問題

### 1. V1内Python APIの不整合

`paperc/robofinals:RoboFinals-IKEA-V1`最終レイヤーの`g1.py`は、同梱runtimeに存在しない
`use_newton_gripper_action()`と削除済み`VBDSolverCfg`を参照していた。未変更ソースでは対象環境を
生成できないため、同梱runtimeが持つstrategy hookへ最小限に接続した。

- `configure_g1_hand_action_cfg(...)`
- `customize_g1_controller_physics_cfg(...)`
- FrameTransformer rootを、USDに存在しない`base_link`から`pelvis`へ変更

### 2. Dex1資産の由来が曖昧だった

OCI最終レイヤーにはDex1用classと`g1_urdf_gripper/G1_GRIPPER.usd`が含まれない。現在の環境は、
競技実機G1 + Dex1-1に合わせるための生成済みDex1資産とcompatibility patchを使う。
「V1を完全に無変更で使用」とは表現しない。V1原本PythonのSHA-256は起動時に固定し、Dex1差分を
決定的に再適用する。

### 3. Vastのmutable runtimeに未管理actuator設定が残っていた

旧`assets_cfg.py`に、GitとV1原本のどちらにもない腕actuator設定が残留していた。

- `LW_G1_ARM_ACTUATOR`
- `G1_IMPLICIT_ARM_JOINT_PARAMS`
- `new_implicit_arms`
- 左右非対称なjoint別stiffness、damping、effort limit

起動ごとにV1 Python原本を復元し、管理された差分だけを適用するよう修正した。現在の腕は
`IdealPDActuatorCfg` / `IdealPDActuator`である。

### 4. fixed評価でnominal接触材質が欠落していた

旧実装は`FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS=false`で、係数samplingだけでなくDex1・白テーブル・
黒作業台へのphysics material bindingまで無効にしていた。bindingは常時適用し、randomization flagは
episodeごとの係数samplingだけを制御するよう分離した。

### 5. 成功判定が不完全だった

旧判定は約110度傾いて手が離れた一瞬でも成功になり得た。現在は以下をすべて要求する。

- initial tabletop normalとのdotが`-0.95`以下
- tabletop centerがinitialから`0.35 m`以上上昇
- linear speed `0.15 m/s`以下、angular speed `0.50 rad/s`以下
- yawを反映した天板全面が3 cm margin込みで作業台footprint内
- 両gripperが脚から離れている
- 50 Hzで20 step、すなわち0.4秒連続

### 6. 接触相手filterがPhysX GPUで機能していなかった

旧finger sensorは白脚の深いcollision-shape pathをfilterに使っていたが、使用中のPhysX GPU backendでは
この構成が解決されず、白脚限定forceが常に0だった。さらに、準備済みUSDの`Leg001` bodyとvisualが
同名になり、cloned environmentのsensor初期化を壊していた。

visualを`Leg001_visual`へ決定的にrenameし、4本の白脚body sensorを4本のDex1 finger bodyでfilterする
reverse構成へ変更した。runtime shapeは各sensor `[num_envs, 1, 4, 3]`である。全表面forceは別の
unfiltered finger sensorで安全判定に使用する。

### 7. reset後に前episodeのactuator targetが残っていた

joint stateだけを書き戻してもactuatorのposition/velocity targetは更新されない。reset直後の1 tickで
前episodeのcommandを実行し得た。resetと初期姿勢randomizationの双方でstateとtargetを同期するよう修正した。

意図的に`3.9407`相当ずらしたcommandを与えてからresetした監査で、position target-state errorと
velocity target-state errorはいずれも厳密に0だった。

### 8. RL経路でlower-body lockが実行されない場合があった

元のlockはtask success checkから呼ばれていたが、RL configがsuccess termを置換すると実行経路が消えた。
RL actionの各制御tickでも必ずlockするよう修正した。また、全関節stateを書き戻す実装をやめ、
下半身12関節だけを`joint_ids`指定で固定し、上半身の物理積分へ触れないようにした。

### 9. 部分resetが他環境のflip rewardを破壊していた

1環境のresetごとに`_initial_table_normal`を全環境ゼロで再確保し、reset対象だけを書き戻していた。
非reset環境ではnormal dotが0となり、実際には未回転でも`table_flip_progress=0.5`相当になる重大な
reward corruptionだった。配列はshape変更時だけ確保し、`env_ids`部分だけ更新するよう修正した。

この不具合を含む過去の非同期vector PPO/RL学習結果は、手法の有効性を判断する証拠に使えない。

### 10. reset直後にPhysX contact cacheが残っていた

強い接触後のresetで、raw finger forceが最大約212 N残り、数step後に0へ収束することを確認した。
sensor値そのものは診断用に保持しつつ、reset後4 stepのcontact reward・force penalty・terminationを
warm-up maskする。通常実行のreset settleも4 stepへ統一した。

### 11. 診断開始poseが共有tensor参照だった

`_table_body_pose()`がsimulator-backed viewを返し、診断が保存した「開始座標」までphysics進行後の
最終値へ書き換わっていた。このため実際には白テーブルが約5.5 cm上昇していても、報告値が`0.0 m`に
なっていた。pose APIをsnapshot契約に変更し、position/quaternionをcloneして返す。

### 12. vector環境の照明が相互干渉していた

1環境の部分resetで全環境の照明を再抽選していた。また環境ごとの`DistantLight`は距離減衰しないため、
並列環境数だけ太陽光が全sceneへ重なっていた。共通公式ライトはepisode中に変更せず、reset対象環境の
有限距離`SphereLight`だけを更新する。light/shadow linkも対象`/World/envs/env_N`へ限定した。

## 最新の方策非依存監査

最新結果:

- `/workspace/evaluations/flip_table_sim_contract_audit_v224_lower_subset.json`
- `/workspace/evaluations/flip_table_sim_contract_audit_v224_lower_subset.run_manifest.json`

条件はseed 42、1 env、50 Hz、policyなしである。sim固有stateは診断だけに使い、actor、critic、planner、
推論分岐には与えていない。全17 gateが合格した。

| gate | result |
| --- | --- |
| runtime identity / actuator | pass |
| prepared scene initial velocity | pass |
| reset actuator target clearing | pass |
| direct target / solver / effort saturation | pass |
| 50 Hz timebase | pass |
| scene geometry / fixed and randomized reset | pass |
| static stability | pass |
| complete-flip success contract | pass |
| 17-D body action mapping / tracking | pass |
| left/right Dex1 mapping / tracking | pass |
| white-leg contact partner filter | pass |
| contact material binding / friction response | pass |
| assembled-table force / torque response | pass |

主要値:

| metric | value |
| --- | ---: |
| sim dt / decimation / control dt | `0.005 s / 4 / 0.020 s` |
| 5秒静止時table translation | `3.70e-6 m` |
| 5秒静止時assembly position drift | `1.56e-6 m` |
| lower-body joint error | `0 rad` |
| robot root translation / rotation | `0 m / 0 rad` |
| reset position/velocity target-state error | `0 / 0` |
| 30 N上向き外力、8 step | `0.11916 m` |
| 30 N support unload + 0.5 Nm、8 step | `0.10049 rad` |
| low-friction 5 N横力、20 step | `0.13480 m` |
| high-friction 5 N横力、20 step | `1.19e-7 m` |

scene geometryは黒作業台`1.80 x 0.75 x 0.76 m`、白天板約`0.58 x 0.42 x 0.04 m`、
tabletop `1.10 kg`、脚`0.124 kg x 4`、作業台`29.0 kg`かつkinematicである。4本のfixed jointの
assembly driftは概ね`1e-5 m`以下である。

## 部分reset分離監査

最新結果:

- `/workspace/evaluations/flip_table_partial_reset_audit_v222_light_link.json`

2環境のうちenv 1だけをresetし、env 0が完全に維持されることを確認した。

| contract | result |
| --- | --- |
| initial table normal / position error | `0 / 0` |
| lighting unchanged in untouched env | pass |
| contact materials unchanged in untouched env | pass |
| room appearance unchanged in untouched env | pass |
| mass error in untouched env | `0 kg` |
| reset env values resampled | pass |
| per-env infinite-distance light | none |
| incorrect light/shadow link | none |

この監査はUSDのlight/shadow link、属性値、部分reset前後の状態を検査している。複数環境を同時に
レンダリングし、env 1の照明変更がenv 0の画素を変えないことを画像差分で確認する試験は未実施である。

## ロボットによる接触・持ち上げpositive control

単一環境をresetから再生した結果:

- `/workspace/evaluations/flip_table_v215_lift_single_origin_v223_lower_subset.json`

実行は記録済み19-D上半身関節targetのみを使い、object poseやforceはpolicy入力ではなく診断にだけ使った。

| metric | value |
| --- | ---: |
| right grasp quality, max / final | `0.99983 / 0.99718` |
| right grasp quality, minimum during lift | `0.49141` |
| max finger force | `10.5099 N` |
| max right hand upward displacement | `0.11728 m` |
| max tabletop-center height delta | `0.05654 m` |
| final tabletop-center translation | `(-0.0441, -0.0552, +0.0547) m` |
| lower-body joint drift | `0 rad` |
| robot root translation | `0 m` |
| flip progress | `0.00972` |

これは右手が脚へ物理接触し、関節運動に追従して白テーブルを傾けて持ち上げられることを示す。
一方、両手把持、18 cm lift、180度flip、安定設置は示していない。

## 過去結果の扱い

- reverse leg-body filter導入前のcontact/grasp値は無効。
- reset target同期前のepisode先頭挙動は再評価が必要。
- 部分reset修正前の非同期vector PPO/RL rewardと学習曲線は無効。
- pose snapshot修正前の`candidate_table_translation`と`max_table_height_delta`は無効。
- 旧照明下のmulti-env visual RLは、並列数依存の画像分布だったため再学習または再評価が必要。
- 単一env ACTで腕target rangeが小さかった事実は残るが、それだけでsim物理を原因から除外してはならない。

## 未解決事項

1. ロボット関節targetだけで、両手把持から180度反転・安定設置までを1回通す。
2. Dex1 collision meshとD405 bracket/camera extrinsicを実機CAD・計測値へ合わせる。
3. 実機で腕step response、指先摩擦、把持力を測り、sim監査値と照合する。
4. simの固定root/lower bodyと、実機G1 balance controllerのcompliance差を評価する。
5. multi-env照明分離をrendered pixel差分でも検証する。
6. randomized visual environmentのcamera pixel分布を実機held-out画像と定量比較する。
7. 実機互換なV1 teleoperation demonstrationを収集し、object-relative subtask境界を監査可能な
   offline annotationとして付与する。Mimicによるvariationは物理検証済みのものだけを使い、
   visual imitation policyをまず成立させる。Residual RLは、そのBC方策を改善する根拠がある場合だけ
   追加する。この作業は未実装であり、成功を主張しない。
8. 新しい学習・評価runでは、V1 image/overlay hash、commit、seed、camera動画、action/state trace、
   success traceを同じ出力ディレクトリへ保存する。固定scene 3 episodeとrandomized 10 episodeを
   同じ成功契約で評価する。

## 2026-07-17の運用上の注意

本監査の17 gate合格は、**policyなしの内部契約**に対する結果である。full flipを実行できる
scripted policy、学習済みpolicy、または実機での成功を意味しない。新しいV1 instanceやoverlayへ
切り替えた後は、学習前に`audit_contract`と`audit_partial_reset`を再実行し、run manifestで
image、overlay、commit、seedを固定すること。

global camera、object pose、segmentation、contact、table poseは、成功判定・監査・解析にのみ使用する。
policy、critic、planner、推論時分岐には渡さない。deploy候補の入力はhead-left RGB、左右D405 RGB、
上半身/Dex1関節状態に限定し、出力は速度・加速度・範囲制限付きの19-D上半身関節targetに限定する。

## 関連コード

- `evaluate/flip_table_simulation/container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py`
- `evaluate/flip_table_simulation/tools/prepare_assembled_table_scene.py`
- `model/flip_table_reinforcement_learning/scripts/audit_simulation_contract.py`
- `model/flip_table_reinforcement_learning/scripts/audit_partial_reset_contract.py`
- `model/flip_table_reinforcement_learning/run_train_in_container.sh`
- `evaluate/flip_table_simulation/tests/test_flip_table_simulation_config.py`
- `model/flip_table_reinforcement_learning/tests/test_flip_table_rl.py`
- `model/flip_table_reinforcement_learning/teacher/`
- [2026-07-17 simulation investigation summary](2026-07-17_simulation_investigation_summary.md)
