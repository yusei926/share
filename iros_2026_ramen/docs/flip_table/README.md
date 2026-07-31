# Flip Table Documentation

このディレクトリは、`flip_table`の調査証跡、simulator契約、現在の開発判断を管理する。
ファイル名は`YYYY-MM-DD_<scope>.md`で統一し、日付は主な証拠取得日または判断日を表す。

| 文書 | 用途 | 現在の扱い |
| --- | --- | --- |
| [2026-07-10 ACT policy root-cause report](2026-07-10_act_policy_root_cause_report.md) | 旧ACT checkpointのsim画像入力時の不動・保守的出力を切り分けた定量証跡 | 歴史的調査。現行仕様ではない。 |
| [2026-07-14 simulation contract audit](2026-07-14_simulation_contract_audit.md) | V1 overlayのaction、reset、接触、物理、部分resetをpolicy非依存で検査した証跡 | 内部契約の根拠。end-to-end成功の証明ではない。 |
| [2026-07-17 simulation investigation summary](2026-07-17_simulation_investigation_summary.md) | 現在の結論、禁止事項、未達、次の受入基準 | このディレクトリの現行判断の正本。 |
| [2026-07-23 balanced WBC migration](2026-07-23_balanced_wbc_migration.md) | Simと実機の制御所有境界、19D state／16D action契約、WBC移行の検証記録 | 現行のSim制御・policy契約。 |

## 文書を追加・更新する際の規則

- 学習・評価の主張には、commit、V1 image/overlay hash、seed、設定、run manifest、動画、action/state traceを結び付ける。
- `fixed scene`、`randomized scene`、`sim-only diagnostic`、`real robot`を明確に区別する。
- policy入力にsim専用情報を使った場合は、deploy候補の結果と混ぜず、診断またはreward用途として明記する。
- simulator、asset、physics、camera、action adapterを変更した場合は、古い数値を流用せず契約監査を再実行する。
- full flip、fixed 3/3、randomized 10 episode、実機確認のどこまで満たしたかを必ず記す。未実施を成功と表現しない。
