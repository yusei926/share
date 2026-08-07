# IROS 2026 RAMEN — yusei926 snapshot

`iros_2026_ramen/` は、IROS 2026 RAMEN のうち yusei926 が担当した
ソース、設定、テスト、文書を中心に抜き出した共有用スナップショットです。

- 更新日: 2026-08-07
- 元ブランチ: `issue-70-flip-table-data-augmentation`
- 元コミット: `83a0afa`
- 元リポジトリ: <https://github.com/matsuolab-llmcompe2025-team-suzuki/iros_2026_ramen>

- Git 履歴・`.git/`・実行ログ・`outputs/`・学習重み・データセットは含めません。
- 公開サーバーのアドレス、SSH 接続情報、費用台帳などのインフラ運用情報も含めません。
- 原則として、他メンバーが最後に変更したファイルは含めません。
- 例外として、2026-08-07 の明示的な共有依頼に基づき、Desktop側の
  orchestrator実行に不可欠な `entrypoint.py`、`orchestrator.py`、
  `lower_policy`、`perception`、`skill_planner` の最小依存一式とテストを含めます。
- `tereope_by_avp.MOV` は Apple Vision Pro テレオペレーションの記録動画です。

このディレクトリは個人担当分の共有用であり、元リポジトリ全体の完全な配布物では
ありません。ただし、同梱したDesktop orchestrator経路については、rootのPixi環境と
別途取得した `third_party/unitree_sdk2_python` を使って単体テスト・起動できます。

```bash
cd iros_2026_ramen
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  third_party/unitree_sdk2_python
pixi install -e runtime
pixi run -e runtime test-desktop-runtime
```

元リポジトリはprivateです。完全版が必要な場合は上記URLに対するTeam collaborator権限を
管理者へ依頼してください。このsnapshot自体には元リポジトリのGit履歴は含まれません。
