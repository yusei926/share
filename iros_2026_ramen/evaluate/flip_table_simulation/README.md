# Flip-table simulation evaluation

RoboFinals-IKEA-V1 に対する Team RAMEN の評価 overlay、Sim-to-Real 校正、AVPテレオペ再生をここで管理します。

## 実行上の原則

- 固定 trajectory と scripted policy は物理診断または教師探索であり、学習済み方策の成功とは扱いません。
- policy/critic/planner の入力は実機相当の head-left RGB、左右D405 RGB、17上半身関節、Dex1状態、履歴だけです。物体pose、contact、global camera、segmentation、sim ground truth は成功判定・報酬・診断・offline teacher探索に限ります。
- policy出力は両腕14D＋左右Dex1 2Dの16D絶対targetです。腰3Dは19D観測stateに含めますが指令しません。Simでは公式G1 WBC、実機ではRegular modeがroot・脚・腰を所有します。
- 全通常Sim経路は `balanced_wbc`（physics 200 Hz / WBC 50 Hz、速度0、base height 0.74 m、torso RPY 0）を使います。固定root・固定下半身は `fixed_diagnostic` の故障診断だけです。
- すべての実験でcommit、seed、checkpoint hash、最大接触力、速度、動画、action/state traceを `outputs/` に残します。生成物はGit管理しません。

詳細な契約と測定済みの制約は `docs/flip_table/`、特に simulation contract audit と execution log を参照してください。

## 入口

| 用途 | 入口 |
| --- | --- |
| 単発評価 | `bash evaluate/flip_table_simulation/run_eval.sh` |
| コンテナ内評価 | `bash evaluate/flip_table_simulation/run_eval_in_container.sh` |
| Furniture-GR00Tリリース評価 | `bash evaluate/flip_table_simulation/run_groot_release_evaluation.sh <checkpoint>` |
| 永続Isaac worker | `bash evaluate/flip_table_simulation/persistent_eval.sh start` |
| 実データ固定scene校正 | `evaluate/flip_table_simulation/real_to_sim_calibration/` |
| AVP 収録 | `bash data/flip_table_data_augmentation/run_sim_teleop.sh` |

このPC（RTX 5090）で実行することを標準とします。Vast.ai を通常手順・再現手順として使いません。

## 永続 worker

永続workerはIsaacのcold startを避けるためだけに使います。各queue jobは必ず `env.reset` し、policy state・physics state・出力を共有しません。

```bash
cd /home/ubuntu/GitHub/iros_2026_ramen
bash evaluate/flip_table_simulation/persistent_eval.sh start
bash evaluate/flip_table_simulation/persistent_eval.sh status
```

AVP jobはdirect camera layout、通常のreplay/校正jobはstandard layoutを使います。切替時はGym環境だけを再作成し、SimulationAppは維持します。USD/image/configを変更したときはworkerを停止してから再起動してください。

```bash
bash evaluate/flip_table_simulation/persistent_eval.sh stop
```

## AVPデータ収集

live AVP表示はhead stereoの最新フレームだけを送ります。既定は、full sceneでdeadlineを取りこぼして不均一になる30 Hzではなく、実測で安定する24 Hzです。表示fpsを落としても操作の滞留を増やさない設計です。`s`で成功収録を確定すると、30 Hzの16D arm/hand command trajectoryと19D観測stateを保存し、AVP終了後に同一trajectoryをoffline replayして head stereo・左右D405・global の画像を生成します。これにより4カメラ30 Hzの学習raw episodeと、低遅延のAVP表示を混同しません。

詳しい操作・安全条件は [Apple Vision Pro手順](../../docs/inference/apple_vision_pro_upper_body_teleop.md) を参照してください。
Sim AVPは`teleop/sim/`内のsocket backendと50 Hz action safetyだけを使い、
実機のUnitree DDS、250 Hz arm_sdk、Dex1 service、Regular-mode ownership処理を
importしません。

## Furniture-GR00Tリリース評価

H100で作成したbaselineとaux-progressの未finalize候補は、同一seedの
randomized 5 episodeで先に比較します。scene randomizationとGR00Tの
flow-matching乱数をともに固定し、各episodeは`base_seed + episode_index`で
独立に再seedします。使用seedは各action traceへ記録され、不一致があれば
候補選択を失敗させます。

```bash
bash evaluate/flip_table_simulation/run_groot_candidate_comparison.sh \
  /path/to/baseline/pretrained_model \
  /path/to/auxiliary_progress/pretrained_model \
  /path/to/eval_validation_baseline/report.json \
  /path/to/eval_validation_auxiliary_progress/report.json \
  outputs/flip_table_groot_candidate_comparison/release_candidate
```

この比較はtest splitを使いません。Sim成功率、平均後のaction jerk・加速度、
追従誤差、offline validation scoreの順で候補を決め、選択候補について下記の
完全release評価も続けて実行します。

通常のrelease評価はH100でfinalize済みのcheckpointを受け付けます。起動前に49D state、
53D logical action、132D packed action、H40、46D valid mask、3 policy
cameras、Dex1 synergy、EEF/FK監査を再検証します。

```bash
bash evaluate/flip_table_simulation/run_groot_release_evaluation.sh \
  /absolute/path/to/finalized/checkpoint
```

実行順はscripted-controller追従、temporal-ensemble validation sweep
（`none/-0.25/-0.1/0` × `5/10/20`）、固定scene 3 episode、未使用DR
50 episodeです。重複chunkは各観測時点の `q_current` から物理絶対腕targetへ
復号してtimestamp整列後に平均し、速度・加速度・関節範囲制限は平均後に
適用します。固定sceneが3/3未満ならDR評価へ進みません。DRのrelease基準は
40/50以上です。temporal sweep、固定scene、未使用DRの各episodeは独立した
flow-matching seedをtraceへ保存し、欠落や不一致があればreleaseを拒否します。
候補選定は`validation_v1`、固定sceneは`nominal_v1`、最終50 episodeは
`held_out_v1`を使用します。最終profileではvalidationに出さない床・壁材、
模様、propと、現実的な低摩擦・高反発端条件を使用します。profile IDは
manifestだけでなくrollout traceにも保存し、実行時設定との不一致を拒否します。

global camera、object pose、contact、sim ground truthは成功判定と診断だけに
保存され、policy入力には入りません。評価はRTX 5090端末で実行し、この
リポジトリを開いているローカル端末ではIsaac Simを起動しないでください。

## 最低限の検証

```bash
python3 -m py_compile \
  evaluate/flip_table_simulation/tools/persistent_eval_worker.py \
  evaluate/flip_table_simulation/tools/persistent_eval_client.py
bash -n evaluate/flip_table_simulation/persistent_eval.sh
git diff --check
```

Isaacを起動する検証はGPU/VRAMと実行中workerを確認し、重複起動しないでください。
