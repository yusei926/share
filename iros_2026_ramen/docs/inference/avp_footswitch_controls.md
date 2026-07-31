# AVPテレオペ用PCsensorフットスイッチ

対象はUSB ID `3553:b001` の **PCsensor FootSwitch**（3ペダル）です。
ペダルはOS全体のキーボードショートカットとして使いません。udev設定により通常の
キーボード／マウス入力から外し、AVPテレオペ実行中だけ専用プロセスが排他的に読み取ります。
したがって、テレオペ外ではペダル操作は何もしません。

## 初回セットアップ

ロボットのテレオペプロセスが停止している状態で、一度だけ実行します。

```bash
cd /home/ubuntu/GitHub/iros_2026_ramen
bash inference/desktop/xr/install_pcsensor_footswitch_udev_rule.sh
# USBを一度抜き差しする

pixi run -e runtime python -m \
  data.flip_table_data_augmentation.teleop.pedal_setup
```

画面の指示に従い、左・中央・右の順に各ペダルを一度ずつ押します。対応は次の通りです。

| ペダル | テレオペ操作 | 作用 |
| --- | --- | --- |
| 左 | `q` | 安全終了（腕・Dex1はIDLEへ） |
| 中央 | `s` | 記録開始、または記録保存 |
| 右 | `r` | 追従開始／再センタリング |

マッピングは `~/.config/iros_2026_ramen/avp_footswitch.json` に保存されます。ペダルを
交換した場合は同じキャリブレーションを再実行してください。

`d`（記録破棄・reset）は意図しない破棄を避けるためキーボード専用です。物理E-stopは
常にペダルより優先する最終安全手段です。

## 実行

通常の実機起動コマンドを使用します。`run_real_teleop.sh` はフットスイッチ設定が無い
場合、ロボット指令を送る前に停止します。シミュレーションでは既定で無効です。AVPで
シミュレーションを操作するときだけ、以下の環境変数で明示的に有効にします。

```bash
G1_DDS_INTERFACE=<G1のNIC> \
G1_IMAGE_SERVER_IP=<OrinのIP> \
bash data/flip_table_data_augmentation/run_real_teleop.sh
```

```bash
FLIP_TABLE_TELEOP_FOOT_PEDAL_ENABLED=true \
  bash data/flip_table_data_augmentation/run_sim_teleop.sh
```

テレオペ中だけ、フットスイッチのキーボード入力デバイスを `EVIOCGRAB` で排他的に取得
します。終了・異常終了時には必ずgrabを解除します。
