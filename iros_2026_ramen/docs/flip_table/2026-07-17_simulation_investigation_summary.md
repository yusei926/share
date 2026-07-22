# Flip Table Simulation Investigation Summary

> **状態: 2026-07-17時点の判断記録。成功報告ではない。**
> 詳細な当時の証跡は[ACT policy root-cause report](2026-07-10_act_policy_root_cause_report.md)、
> simulator内部契約は[simulation contract audit](2026-07-14_simulation_contract_audit.md)を参照する。

## 現在の結論

`flip_table`を安定して完了するpolicyは、まだ存在しない。V1 sceneの主要な内部契約は監査済みだが、
これは物理的な180度反転、randomized sceneでの成功、または実機G1 + Dex1-1への移行を証明しない。

| 項目 | 状態 |
| --- | --- |
| 19-D上半身actionのtarget追従 | 確認済み |
| 下半身・root固定 | 確認済み |
| 組立済み白テーブルの力・トルク・摩擦応答 | 確認済み |
| 既知関節軌道による片手の把持・小さな持ち上げ | 確認済み |
| scripted full flip | 未達 |
| 学習済みpolicyによる固定scene 3/3成功 | 未達 |
| 学習済みpolicyによるrandomized 10 episode成功 | 未達 |
| 実機G1 + Dex1-1での評価 | 未実施 |

## 現行の実機互換契約

- 入力はhead-left RGB、left/right D405 RGB（各640x480）、上半身/Dex1関節状態のみ。
- `head_right`、global camera、object pose、segmentation、contact、table poseなどsim専用情報は、
  成功判定・監査・解析以外に使わない。
- 出力は腰3、左腕7、右腕7、左右Dex1各1の19-D上半身絶対関節targetのみ。
- 下半身はpolicyで制御しない。simではrootと下半身を固定し、実機ではG1標準balance controllerを使う。
- targetには関節範囲、速度、加速度の制限をかける。teleport、物体直接操作、sim専用指令は使わない。

## 確認済みのこと

- organizer提供の`paperc/robofinals:RoboFinals-IKEA-V1`をbaselineとし、起動時に原本robot sourceを
  復元してから、repository管理のDex1-1、scene、camera、policy overlayを決定的に適用する。
- policyのcamera mappingは`head_left <- first_person_camera`、`left_wrist <- left_hand_camera`、
  `right_wrist <- right_hand_camera`で固定され、3入力はraw `640x480x3`のまま扱う。global cameraは
  録画・診断専用である。
- ACTの旧checkpointは実機データ画像には有効な腕action chunkを出した一方、sim画像では大きく保守的に
  なった。当時のadapter誤読と長すぎるaction queueは修正済みだが、旧checkpointのfull flip成功はない。
- `audit_contract`と`audit_partial_reset`は、新しい学習runの前に実行するrelease gateである。

## 失敗した探索と判断

固定source actionを中心にしたCEM、trajectory composition、局所contact searchは、片手liftや一時的な
両手接触を作れても安定したfull flipへ至らなかった。contact-richな両手操作を、この局所探索だけで
見つける根拠は薄いため、関連candidate JSON、生成器、batch search launcherはrelease surfaceから
削除した。

過去のvector PPO/RLの結果は、修正前のpartial reset、contact filter、照明、診断snapshotの影響を
受けるため、手法比較の根拠には使わない。Flow/RLPDとPPOのコードは比較baselineとしてのみ残しており、
成功した方策として扱わない。sim-real画像差は有力な要因だが、唯一の根本原因とは断定しない。

## 次の方針

最有力な次段階は、V1 teleoperationで成功したdemonstrationを収集し、object-relative subtask境界を
auditableなoffline annotationとして付与することだ。その上でIsaac Lab Mimicを使い、物理的に検証した
variationだけを増やして、同じ3カメラ・19-D契約のvisual Diffusion Policyを学習する。policy failureは
実機互換の観測・actionだけでcorrection demonstrationへ戻す。

Residual RLは、BC方策がreach/grasp/flipの連鎖をすでに成立させ、残差が何を改善するか測定できる場合に
限って追加する。Mimic、teleoperation収集、Diffusion Policy学習は、この時点ではまだ実装・成功済み
ではない。

## 再現と受入基準

新しいinstance、V1 image、asset、physics、scene、overlayを変更した場合は、学習前に以下を行う。

1. `audit_contract`と`audit_partial_reset`を実行し、V1 image/overlay hash、commit、seedをrun manifestへ保存する。
2. policy入力・出力が上記19-D/3-camera契約だけであることを、traceとadapter testで確認する。
3. fixed scene 3 episodeを評価し、各episodeのcamera動画、action/state trace、成功判定を保存する。
4. realisticなcamera誤差、照明、texture、初期姿勢、テーブル姿勢、遅延、センサノイズを用いた
   randomized 10 episodeを同じ成功契約で評価する。目標は10/10、最低受入は8/10とする。
5. 実機未評価の項目は「Sim-to-Real成功」と呼ばず、camera hand-eye、collision geometry、摩擦、
   関節応答、G1 balance controllerとの差を実測で確認する。

## リリース確認

```bash
PYTHONPATH=. model/subtask_policy_training/.venv/bin/pytest -q \
  model/flip_table_reinforcement_learning/tests \
  evaluate/flip_table_simulation/tests
git diff --check
```

生成動画、run output、virtual environment、dataset、checkpointはGit管理しない。
