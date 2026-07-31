# Hugging FaceモデルのG1実機評価

このパッケージは、学習方法・次元・カメラ・正規化が異なる上位方策を、
同じ安全な実機境界へ接続します。CLIへ渡す識別子は、短い登録名だけでなく
`Team-RAMEN/repository`またはHugging Face URLを使用できます。

ただし、**任意のHFモデルを形だけ推測して実機へ接続する機能ではありません**。
同じ16次元でも、絶対関節角、相対関節角、EEF、正規化、カメラ順序が違えば
物理的な意味は別物です。既知モデルはローカルcatalog、今後のモデルはHF repo内の
`iros_ramen_deployment.json`で完全な契約を確定します。契約がない、曖昧、
未対応family、hash不一致のどれかなら、実機起動はfail-closedで拒否されます。

## 安全境界

- canonical出力は`両腕14D絶対関節角[rad] + Dex1左右開き率2D`だけです。
- root、脚、腰のcommand次元は0です。実機の腰以下はUnitree Regular Modeが所有します。
- remote manifestは実行するPython moduleやscriptを指定できません。
  runner、worker、adapterはレビュー済みのローカルfamily pluginだけが所有します。
- HF revisionは実行前に40桁commit SHAへ解決します。
- 必要な全weightはHF LFS SHA-256で照合し、ローカル全required fileをsealします。
- seal後に1 byteでも変わったartifactはlive起動できません。
- 共通launcherに任意argument passthroughはなく、検証後のcheckpointやworker差し替えを
  防止しています。速度・加速度など8個の安全制限だけを、レビュー済み上限内で
  model family共通に指定できます。

現在のtrusted familyは次の4種類です。

| family | model入力 | model出力 | canonical化 |
|---|---|---|---|
| `act_absolute_joint16_v1` | 4 RGB + 腕・Dex1 16D | absolute 16D×30 | 学習supportへclampしDex1を開き率化 |
| `groot_absolute_joint_v1` | 4 RGB + 38D | absolute 38D | root/脚/腰を破棄し腕・Dex1だけ |
| `groot_relative_eef_v1` | 3 RGB + 49D | decoded 53D | EEF/腰/base/navigationを破棄 |
| `diffusion_chunk_relative_v1` | 3 RGB×2 + 19D | relative 16D | measured腕へ1回だけanchor |

## 環境

resolver、artifact監査、adapter dry-runにはSDKを含まない専用環境を使います。

```bash
pixi install -e model-eval
```

`model-eval`にはUnitree SDKとCycloneDDSを入れていません。実modelのGPU推論は、
リポジトリに固定されたmodel worker環境へ明示的に移ります。

## HFパスだけを渡す標準フロー

以下の`MODEL`には`Team-RAMEN/...`または
`https://huggingface.co/Team-RAMEN/...`を渡せます。

```bash
MODEL=Team-RAMEN/groot-n1.7-pick-legs-ver2-lora
LOCAL=.checkpoints/groot-n1.7-pick-legs-ver2-lora
```

### 1. 解決と契約監査

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli resolve "$MODEL"

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli inspect "$MODEL"
```

既知モデルならcatalogの固定SHAへ解決します。未知モデルはHFの最新revisionを
full SHAへ解決し、repo内のdeployment manifestを検証します。`main`のまま実行記録へ
残すことはありません。

model cardや小さいconfigだけを読み、weightを取得せず監査する場合:

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli inspect-hf "$MODEL"
```

この出力は契約関連のfieldだけに要約されます。学習機のローカルpath、episode全一覧、
tokenは表示しません。`safe_for_actuation=true`になるのは、manifestが存在するだけで
なく、trusted family契約として実際に検証できた場合だけです。

### 2. artifact取得・seal

まず取得内容を確認し、commit-pinnedの最小snapshotだけを取得します。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli download-plan "$MODEL" \
  --local-dir "$LOCAL"

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli prepare "$MODEL" \
  --local-dir "$LOCAL"
```

`prepare`はdownload、static contract検査、weight hash検査、全required fileの
tamper-evident sealを連続して行います。以後は次で再検査できます。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli validate-artifacts "$MODEL" \
  --local-dir "$LOCAL"
```

### 3. 指令なしdry-run

adapterの次元と除外範囲だけを、合成配列で高速に検査します。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli adapter-dry-run "$MODEL"
```

これはweightをloadしません。実weightをGPUで1回推論するには、記録済み観測bundleを
使います。この経路もUnitree SDK、DDS、live cameraを初期化しません。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli offline-model-dry-run "$MODEL" \
  --local-dir "$LOCAL" \
  --bundle outputs/model_eval_bundle \
  --device cuda:0
```

bundleは次の形式です。

```text
outputs/model_eval_bundle/
├── observation.json
├── head_left.jpg
├── head_right.jpg       # 4-camera familyのみ
├── left_wrist.jpg
└── right_wrist.jpg
```

`observation.json`:

```json
{
  "body_joint_position_rad": [29個],
  "dex1_opening_fraction": [2個],
  "eef_xyz_euler": [12個],
  "camera_jpeg": {
    "head_left": "head_left.jpg",
    "head_right": "head_right.jpg",
    "left_wrist": "left_wrist.jpg",
    "right_wrist": "right_wrist.jpg"
  }
}
```

familyが使わないcamera roleやEEFは、そのmodel contractに合わせます。
camera pathのbundle外参照、NaN/Inf、次元不一致は拒否されます。

### 4. live read-only preflight

ここから先だけはG1とカメラが必要です。まず`--actuate`なしのコマンドを生成します。
`real-command`自身は表示だけで実行しません。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli real-command "$MODEL" \
  --local-dir "$LOCAL"
```

表示されたコマンドの実行時に、seal、固定XR revision、NIC、カメラ、model identity、
fresh observation、prediction形状を再確認します。`--actuate`がないためロボットへ
関節指令は送りません。

### 5. 実機評価

read-only preflightとログ確認が成功した後だけ、コマンド生成側へ`--actuate`を付けます。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli real-command "$MODEL" \
  --local-dir "$LOCAL" \
  --actuate \
  --max-seconds 1
```

生成されたコマンドを実行しても、各runnerでRegular Mode、fresh camera/state、
model hash、初期delta、安全制限、Enter確認を再確認します。最初は必ず
ハーネス・E-stop下の1秒・低振幅試験から進めます。

`--actuate`を使う登録済みrunnerは、policyを開始する前に共通のarm-only
pre-motionを必ず実行します。

1. 左右shoulder pitchだけを後方へ回し、肘・手首の開始姿勢を維持する
2. shoulder pitchを後方に保ち、肘を伸ばしながらshoulder rollで腕を横へ広げる
3. 腕の横幅を維持したまま前方へ移し、肘が机の内側を横切らないようにする
4. 肘と手首を少し下げた前方待機姿勢へ移す
5. 対象モデルで固定した学習開始姿勢へ移す。通常は各episodeの0フレーム目の
   `action.robot_q_desired`中央値、ACT joint16は学習元deploymentで選ばれた
   episode 2101 frame 0のmedoidを使用する
6. その姿勢を重力補償付きで保持し、空のEnter入力後にだけfresh
   camera/state/predictionを再検査してpolicy実行を開始する

最初の空Enterはこの決定論的な退避動作を許可する確認です。通常の登録済みrunnerは
2回目の空Enterをpolicy開始確認とします。coarse insert / tightenだけは、1回目で
Dex1を全開にして退避経路からdataset frame-0腕姿勢へ移動し、2回目で腕を固定したまま
dataset frame-0の左右把持幅へ閉じ、実測収束後の3回目でpolicyを開始します。
Enter待機中も30 Hzで
腕・Dex1の最終targetを再送し、実機状態を
検査するため、watchdogによる意図しないarm_sdk解放は行いません。Ctrl+C、状態異常、
waypoint未到達、Enter後のmodel安全検査失敗はいずれもpolicyを開始せずcontrolled
releaseへ進みます。pre-motionとpolicyのどちらも腰・脚・rootのcommand次元は0です。
正常終了・Ctrl+C・検査失敗時は、policy最終姿勢からdataset frame-0、前方待機、
前方退避、横退避、shoulder後退、起動直前に実測した腕姿勢の順で逆走してから、
arm_sdk weightを段階的に0へ戻します。解放後は`fsm_id=501, fsm_mode=0`への復帰も
読み取り確認し、復帰しなければ成功終了にしません。

起動前のhigh-level検査は`fsm_id=501, fsm_mode=0`だけを許可します。実機firmwareは
`rt/arm_sdk`取得後に同じFSM 501の`fsm_mode=1`を返すため、pre-motion後の再検査だけは
mode 0/1を許可します。これは任意FSMを許可する迂回ではなく、同時にbackendの
`mode_machine`固定、DDS正常、camera freshness、lower-body command 0、Regular所有を
継続検査します。

### W&Bへの実機評価記録

共通launcherから実行したrunは、モデルfamilyに関係なく以下を
`outputs/real_policy_evaluation/runs/<run_id>/` へ自動保存します。

- head-left、head-right、left-wrist、right-wristのMP4
- 29D実関節位置・速度、Dex1状態、適用済み腕・Dex1 target
- backendが受け取ったすべての腕・Dex1 command
- policy runnerの推論、pre-motion、policy action、復帰event
- model/revision、source commit、安全制限、camera cadence、完了判定

W&Bの既定projectは次です。最初の有効runのupload時にW&B側へ
自動作成されます。

```text
entity:  ken05-matuo-llm-88_llm_2025_suzuki
project: iros-2026-ramen-real-policy-evaluation
```

自動uploadは、`--actuate`付き、policy実行10秒以上、runner exit code 0、
`return_motion_complete`確認、収録drop/error 0、4カメラMP4完成のすべてを
満たすrunだけを対象にします。閾値は `--wandb-min-seconds` で変更できます。
タスク成功は条件ではないため、正常に完走・復帰した失敗試行も比較対象に
残ります。1秒・5秒試験、Ctrl+C、例外終了、安全復帰未確認のrunは
ローカル診断にのみ残し、W&Bへは送信しません。

最初に1回だけ認証します。API keyはrepositoryやshell historyへ書かず、
W&Bの非表示入力で入れてください。

```bash
pixi run -e model-eval wandb login
```

未認証・通信失敗時もrobot runの終了codeは変えず、`wandb_upload.json`に
再送可能な状態を残します。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.wandb_export \
  outputs/real_policy_evaluation/runs/<run_id>
```

## 未登録モデルのonboarding

HF repoにmanifestがなければ、自動推測で実機へ進まずdraftを作ります。

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli onboard \
  Team-RAMEN/new-model \
  --output /tmp/iros_ramen_deployment.draft.json
```

draftの`REQUIRED`を、dataset/schema/model作者の情報から確定してください。
`_onboarding`は説明用なので削除し、JSON schema
`deployment_manifest.schema.json`へ適合させます。全weightのLFS SHA-256、camera role、
state/actionの**順序・単位・絶対/相対基準**、Dex1 scale、taskを確定してから
HF repoへ`iros_ramen_deployment.json`としてcommitします。

既存familyと次元が同じだけでは流用できません。意味論まで完全一致する場合だけ既存
familyを指定します。新しいACT、Diffusion、Pi0、VLA表現なら、まずローカルに
trusted family plugin（adapter、worker、runner、static validation、test）を追加します。
remote manifestだけで新しい実行コードを有効化することはできません。

したがって、現在のTeam-RAMEN内の全modelが無条件に即実機推論できるわけではありません。
一度正しくonboardingしてmanifestをrepoへ置けば、それ以降はHFパスだけで同じ
resolve→prepare→dry-run→live preflightフローを再現できます。

## 全24モデルの推測オフライン監査

deployment manifestがまだないモデルも、**実機経路から完全に分離した状態**で
「configを読めるか」「weightをstrict loadできるか」「合成入力から有限のaction tensorが
出るか」までは試せます。ここで推測するのはtensor契約だけであり、関節の順序・単位・
絶対/相対表現が実機に合うとは判定しません。

全repoのmetadata監査:

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli audit-team \
  --namespace Team-RAMEN \
  --output outputs/model_evaluation/team_ramen_model_audit.json
```

1モデルの推測契約:

```bash
MODEL=Team-RAMEN/pana_nakatsuka_act_pick_table_leg
LOCAL=.checkpoints/model-eval-inferred/pana_nakatsuka_act_pick_table_leg

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli infer-offline "$MODEL"

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli offline-prepare "$MODEL" \
  --local-dir "$LOCAL"

pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli inferred-model-dry-run \
  --local-dir "$LOCAL" --device cuda:0
```

全repoの一括試験:

```bash
pixi run -e model-eval python -m \
  inference.desktop.model_evaluation.cli test-team-offline \
  --namespace Team-RAMEN \
  --workspace .checkpoints/model-eval-inferred/all \
  --output outputs/model_evaluation/team_ramen_model_offline_test.json \
  --device cuda:0 --prepare --max-download-gb 20
```

安全境界は次のとおりです。

- inferred用lockは`.iros_ramen_inferred_offline_lock.json`で、実機用lockとは別schemaです。
  実機launcherはこれを読みません。
- CLIに`--actuate`、NIC、G1 IP、live camera引数はありません。
- probe環境で`unitree_sdk2py`または`cyclonedds`がimportされた場合は失敗します。
- `.pt`/`.pth`はpickleコード実行を伴い得るため、自動loadしません。
- release checkpointが複数あって一意に選べないrepoはmetadata監査だけで止めます。
- Pi0.5の独自fieldは、値を含め「学習時だけ」と確認できた既知fieldだけ一時的に除外し、
  元configとseal済みweightは変更しません。推論に影響し得る拡張が有効なら拒否します。
- `weight_inference_passed`でも実機対応を意味しません。実機へ進めるには、従来どおり
  正確なdeployment manifest、trusted family adapter、記録済み実画像dry-runが必要です。

## 登録済みモデル

- `Team-RAMEN/pana_nakatsuka_act_pick_joint16_augxx_s40k_20260730`
- `Team-RAMEN/groot-n1.7-pick-legs-ver1`
- `Team-RAMEN/groot-n1.7-pick-legs-ver2-lora`
- `Team-RAMEN/IROS2026_RAMEN_takada_groot_n17_coarse_insert_100k_dex1_v2`
- `Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_2`

短いaliasは互換用に残していますが、新しい運用ではHF pathを推奨します。
