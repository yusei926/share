# Flip Table Dataset Curation

`Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1` を、実機で再現する単一の
初期配置・単一の flip 手順へ絞り込んで
`Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2` を生成するツールです。

元の state/action 値は変更しません。次の episode だけを採用し、前後の長い
無動作区間のみを trim します。

- 一歩も歩いていない
- 初期状態でテーブル短辺がロボット正面にある
- 最多の安定した flip 軌道クラスタに属する

個々の成功を人が確認したデータではないため、`success=true` は付与しません。
採用根拠と source lineage は episode metadata と
`meta/curation/manifest.json` に保存します。

## セットアップ

```bash
cd data/flip_table_dataset_curation
pixi install
```

HF token はファイルへ保存せず、既存の Hugging Face login または
`HF_TOKEN` 環境変数を使います。

## 実行順

```bash
pixi run audit-source
pixi run analyze
pixi run review
```

`pixi run review` が生成する `workspace/review/index.html` で、向きクラスタと
軌道クラスタの代表例を確認します。決定は次のように記録します。

```bash
pixi run python -m flip_table_curation.cli decide \
  --orientation-cluster 0 \
  --trajectory-cluster 1 \
  --reviewer yusei926
```

決定後に解析を確定し、dataset を作成・検証・公開します。

```bash
pixi run analyze
pixi run build
pixi run validate
pixi run publish
```

`publish` は、100 episode 未満、ローカル検証失敗、既存 remote main の
manifest 不一致のいずれかで fail closed します。private staging branch を
検証した後だけ main へ昇格します。

## 生成物

すべて `workspace/` 配下で、Gitには追加しません。

- `audit/source_audit.json`
- `analysis/analysis.json`
- `review/index.html`
- `decision.json`
- `dataset/Team-RAMEN__IROS2026_RAMEN_suzuki_flip_table_2/`
- `validation/validation.json`
- `publish/publish_report.json`
