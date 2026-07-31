# G1 walk-to-table Smoke Test (Issue #64)

実機 G1 で **`move_to_table` skill の base pipeline** を通す最小 smoke test。
依頼された engineer 向けの一気通貫 guide。

対象 branch: `issue/64-real-g1-walk-tuning`
関連: [logic_wrapper.md](logic_wrapper.md) の Lower Policy interface

---

## 0. 目的

**「G1 が低速で1秒間前進 → 停止 → walking FSM維持」** のpipelineを実機で通す。

- manipulation 無し、他 skill 無し
- 対象は純粋にSDK経由のbase locomotion (`LocoClient.SetVelocity`)
- Wrapper / Perception / ViT は経由しない (`SkillDispatchLowerPolicy` を script から直接叩く)

### 期待挙動

- 前進距離 **≈ 15cm** (0.15 m/s × 1 s、指令値からの目安)
- 転倒しない、蛇行しない
- 停止後は既存のRegular Mode / walking balancerを維持
- コンソール: `[start] vx=0.15 m/s, duration=1.0s` → `[stop]`

---

## 1. 事前準備 checklist

### 1.1 コード / env セットアップ

```bash
# 1. リポジトリ + branch
git clone <repo-url> && cd iros_2026_ramen
git checkout issue/64-real-g1-walk-tuning

# 2. SDK clone (third_party/ は gitignore、手動で用意)
git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git \
    third_party/unitree_sdk2_python

# 3. runtime env 構築 (Python 3.10 + cyclonedds + unitree_sdk2py)
#    (main env とは Python version が違うので `-e runtime` で明示活性化)
pixi install -e runtime

# 4. dry-run で疎通確認 (robot 無しで OK、default env でも可)
pixi run python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table --dry-run
```

**dry-run が成功** = 以下の2 eventsが順に出力される:

```
('set_velocity', 0.15, 0.0, 0.0, 1.0)
('set_velocity', 0.0, 0.0, 0.0, 1.0)
```

### 1.2 物理準備

- [ ] **ハーネス or 天井吊り支持** (初回歩行は必須)
- [ ] **E-stop リモコン** (Unitree 標準付属) を人が握って待機
- [ ] 前方 **3m 以上**のクリアランス
- [ ] バッテリ **50% 以上**、満充電推奨
- [ ] 硬い床の場合はマット敷き
- [ ] G1 直結の Ethernet interface 名を控える (`ip a` で確認、例: `eth0` / `enp2s0`)

---

## 2. テスト手順

**前提**: 3-DoF waist機はRegular Mode (`fsm_id=501`) から開始する。
Damp からの startup は operator が Unitree 標準ワイヤレスリモコン等で事前に済ませておく
(本 doc の scope 外)。

### 2.1 walk forward smoke 実行

```bash
pixi run -e runtime python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table \
    --interface eth0 --duration 1 --vx 0.15
```

- **Enter プロンプト**で続行確認 (Ctrl-C で中止可)
- 実行前にsport API version / `fsm_id` / `fsm_mode`をread-only確認
- 実行: 1秒前進 → 速度ゼロ。姿勢高さ・FSMは変更しない

### 2.2 完了後

robotはそのまま **Regular Mode / walking balancer維持**、次のテストに使える。

### 2.3 2026-07-16 実機結果

同じ3-DoF waist G1 (`fsm_id=501`) で以下を確認した。

| `vx` | `duration` | RPC | 実機結果 |
|---:|---:|---|---|
| 0.03 m/s | 1秒 / 2秒 | `code=0` | 不動 |
| 0.10 m/s | 1秒 | `code=0` | 不動 |
| 0.15 m/s | 1秒 | `code=0` | 前進成功 |
| 0.30 m/s | 1秒 | `code=0` | 前進成功（Unitree公式example値） |

したがって、RPCの`code=0`は歩行開始を保証せず、この実機の歩行開始閾値は
`0.10 < threshold <= 0.15 m/s`にある。単体smokeの既定値は実測成功済みの
`0.15 m/s`とする。

---

## 3. 記録してほしい 5 項目

| # | 項目 | 記録方法 |
|---|---|---|
| 1 | **成功 / 失敗** | 上記期待挙動どおりだったか |
| 2 | **前進距離実測** | メジャーで、指令値からの目安15cmとの差 |
| 3 | **挙動の異常** | 揺れ / 蛇行 / 停止時のふらつき等 |
| 4 | **エラーログ** | あれば全文コピー (特に DDS init 失敗 / RPC timeout) |
| 5 | **video (可能なら)** | 側面 or 正面からの動画 |

---

## 4. 危険信号 (即中断)

- **E-stop 即押し** が最優先
- 転倒しそう → 手で支える / ハーネスに頼る
- 想定と違う動き → コンソールCtrl-C (finallyで`set_velocity(0)`、Dampには入らない)
- SDK 側 timeout → network 設定 (interface 名 / domain_id / firewall UDP 47998) 見直し

---

## 5. トラブルシューティング

| 症状 | 原因候補 | 対処 |
|---|---|---|
| `client.Init()` が timeout | interface 名違い / domain_id 違い / firewall | `ip a` で確認、firewall 停止、`--domain-id` 調整 |
| preflightで停止 | `fsm_id != 501` / API版不一致 | Regular ModeとSDK checkoutを確認。ガードを迂回しない |
| RPC code=0だが前進しない | 速度がgait開始閾値未満 / firmware側上書き | 実測成功済みの`0.15m/s × 1s`で再確認する |
| 蛇行する / 傾く | バッテリ低下 / 床摩擦バラつき / IMU calibration | バッテリ充電、床マット、robot 再起動 |

---

## 6. 返答テンプレ

以下を埋めて Slack / Issue コメント / メールで返答:

```
実機テスト完了 (Issue #64, branch issue/64-real-g1-walk-tuning)

- 結果:              成功 / 部分成功 / 失敗
- 前進距離:          XXcm (指令値からの目安15cm)
- interface / iface: XXX
- 特記事項 (挙動):
- エラーログ:
- video URL:
```

---

## 7. 参考

- コード: [inference/desktop/lower_policy/](../../inference/desktop/lower_policy/)
  - Skill 本体: [skills/move_to_table.py](../../inference/desktop/lower_policy/skills/move_to_table.py)
  - SDK actuator: [actuators/g1_sdk.py](../../inference/desktop/lower_policy/actuators/g1_sdk.py)
  - Smoke script: [scripts/run_g1_walk_to_table.py](../../inference/desktop/lower_policy/scripts/run_g1_walk_to_table.py)
- Runtime env: `pixi.toml` の `[feature.runtime]` + `[environments]` セクション (`pixi run -e runtime <cmd>` で活性化)
- SDK 例: `third_party/unitree_sdk2_python/example/g1/high_level/g1_loco_client_example.py`
