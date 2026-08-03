# Manual Flip-Table Dataset Curation

`Team-RAMEN/IROS2026_RAMEN_HARA_curation` の人手ラベルから、private
`Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_3` を生成するツールです。

採用するのは、各元episodeの最後の `flip_table`（task index `2`）ラベルが
`success` または `optimal` である区間だけです。元データは
`BitRobot/G1_WBT_Dex1_Building-Children-Table` に固定します。

数値state/actionと4 RGB・4 IRカメラを保持し、導出episodeの timestamp/index だけを
LeRobot v3の連続値に再構成します。ラベルの重複、範囲外、schema不一致はfail closedです。

## セットアップ

```bash
cd data/flip_table_dataset_curation
pixi install
```

HF token はファイルへ保存せず、既存の Hugging Face login または
`HF_TOKEN` 環境変数を使います。

## 実行順

```bash
pixi run audit-labels
```

`audit-labels` は選択結果を `workspace/selection/selected_segments.json` に保存します。
この工程で失敗した場合は、Curation UIでラベルを直して同期し、新しいlabels revisionを
configへ固定してからやり直します。

```bash
pixi run build
pixi run validate
pixi run publish
```

`publish` は、367 episode 未満、ローカル検証失敗、既存 remote main の
manifest 不一致のいずれかで fail closed します。private staging branch を
検証した後だけ main へ昇格します。

## 生成物

すべて `workspace/` 配下で、Gitには追加しません。

- `audit/label_audit.json`
- `selection/selected_segments.json`
- `dataset/Team-RAMEN__IROS2026_RAMEN_suzuki_flip_table_3/`
- `validation/validation.json`
- `publish/publish_report.json`
