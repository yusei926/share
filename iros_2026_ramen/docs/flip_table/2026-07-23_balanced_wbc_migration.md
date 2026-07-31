# G1 balanced WBC移行記録（2026-07-23）

## 目的

Simのroot・脚・腰を毎tick固定する旧経路を本番経路から外し、実機のRegular Modeと同じ所有境界にそろえる。

- Policy／AVP: 両腕14関節と左右Dex1の16D target
- Sim: RoboFinals公式`G1-Gripper-Controller-DecoupledWBC`
- 実機: Unitree Regular Mode（下半身・腰）と`rt/arm_sdk`（腕）

公式WBC、`stand.onnx`、`walk.onnx`、公式`xr_teleoperate` checkoutは変更しない。Team RAMEN側のadapterが16D絶対targetを公式WBCへ渡す。

RoboFinals V1のWBCは単一環境を前提とする初期化・履歴管理を含むため、adapterは各Isaac環境に独立した公式WBC instanceを割り当てる。これによりONNX推論と公式制御則を変更せず、4/16並列環境の履歴・partial resetを分離する。

## データ契約

| 種別 | 次元 | 順序 |
|---|---:|---|
| state | 19 | 腰3 + 左腕7 + 右腕7 + 左右Dex1状態2 |
| action | 16 | 左腕7 + 右腕7 + 左右Dex1 target 2 |

腰はバランス応答を観測するためstateへ残すが、actionには含めない。既存19D actionは元データを変更せず、腰3Dを決定的に除去し、変換version・元action hash・除去値を記録する。旧19D checkpointは互換対象外であり、Flow Matching BCとResidual RLPDは19D入力・16D出力で再学習する。

## Sim制御契約

- `FLIP_TABLE_SIM_BODY_MODE=balanced_wbc`を本番既定とする。
- physics 200 Hz、WBC/action 50 Hz。
- WBC commandは速度`[0, 0, 0]`、base height `0.74 m`、torso RPY`[0, 0, 0]`。
- episode中のroot teleport、脚・腰state書き戻し、腰lockを禁止する。
- HOLD、手の見失い、AVP pause中もWBC standを継続し、腕・手targetだけを保持する。
- `fixed_diagnostic`は故障切り分け専用で、学習・評価・teleop・replayの成果として扱わない。

raw episodeとreplayにはbody mode、physics/control rate、WBC asset hash、adapter hashを保存する。global camera、object pose、接触、WBC内部stateはreward・成功判定・安全診断にのみ使用し、policy／critic／planner入力へ追加しない。

## 安全修正

- Simの関節制限器を、終端・反転・joint limit境界でも速度と加速度を破らない制動距離ベースへ変更した。
- 天板の中心だけでなく、yaw後の天板投影面全体が作業台上に3 cm以上の余裕を保つようresetを制限した。不可能なyawは範囲内で再サンプルし、明示キャリブレーション姿勢は改変せず拒否する。
- floating-base WBCの初期整定でrootが移動してもテーブルとの最小距離を維持できるよう、reset配置に3 cmの整定余裕を確保した。
- 公式WBCの`upper_body`が腕14関節だけで、腰を含まないことを起動時にfail-closedで検査する。
- 実機はarm_sdk 250 HzとDex1 200 Hzを独立送信し、DDS write成功後だけ適用sequenceを進める。
- arm_sdk weight解放、control lock、FSM、カメラ時刻、入力thread終了をfail-closed化した。

## 検証結果

- runtime単体テスト（teleop・Sim設定・実機較正・下位制御）: 455 passed / 10 skipped
- Flow Matching／Residual RLPDの重点契約テスト: 34 passed
- RoboFinals WBC完全監査: 17/17 gate passed。
  - 60秒stand: root XY drift `0.000201 m`、base height error `0.018979 m`、最大roll `0.000324 rad`、最大pitch `0.020792 rad`。
  - randomized reset 16 trial、腕14D追従、左右Dex1、接触材、摩擦応答、力・トルク応答を含めて合格。
  - 監査時adapter SHA-256: `adabf75ed71151c7827b2736a6d1f6866c110e6b7bdf576bfaa858b5e4dc1d2a`。
- partial reset分離: 4環境・16環境とも合格。未reset環境の初期位置・法線誤差は0。
- 50 step state-policy smoke: 1/4/16環境すべて完走、action dimは16。実測throughputは順に約`9.22`、`30.50`、`37.74 transitions/s`。

生成レポートは`outputs/flip_table_reinforcement_learning/`以下に保存したが、生成物のためGitには含めない。

固定条件3/3および未見seed 10 episodeで8/10以上は、学習済みcheckpointの評価条件である。本移行時点では完全flipを達成したcheckpointはなく、WBC移行やteacher trajectoryだけを成功と表現しない。
