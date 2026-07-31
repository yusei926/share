# Team-RAMEN 24モデル オフライン互換性監査（2026-07-31）

## 目的

Hugging Faceの`Team-RAMEN` namespaceにある24モデルについて、ロボット、DDS、live
cameraへ一切接続せず、利用可能なmetadataからtensor契約を推定し、可能なものは実weightを
GPUで1回推論する。

この監査の成功は実機互換性を意味しない。実機評価には、関節順序、単位、絶対/相対表現、
Dex1 scale、camera roleを作者が確定したdeployment manifestと、レビュー済みadapterが
別途必要である。

## 安全境界

- 実機用ModelSpecと推測用InferredOfflineContractを別型・別lock schemaにした。
- 推測lockを実機launcherへ渡すことはできない。
- `--actuate`、NIC、G1 IP、live cameraのinterfaceは推測CLIに存在しない。
- probe中のUnitree SDK/CycloneDDS importを検査し、見つけた場合は失敗する。
- remote Python codeと`.pt`/`.pth` pickleは自動実行しない。
- HF revisionを40桁commit SHAへ固定し、選択した全artifactをSHA-256でsealする。

## 実装時点の分類

- 登録済み・レビュー済み経路: 4
- LeRobot/native configからオフライン推論を試せる追加候補: 12
- metadata/構造監査のみ: 8

追加候補の内訳はACT、native ACT、native Diffusion、Furniture-GR00T、Pi0.5である。
構造監査のみの内訳は、空repo 3、perception checkpoint repo 2、曖昧または独自pickle
repo 3である。

## 実測結果

`outputs/model_evaluation/team_ramen_24_model_offline_test_20260731.json`へ機械可読結果を
保存した。最終結果は次のとおり。

- 推測契約で実weight推論成功: 11
  - ACT 7
  - native ACT 1
  - Pi0.5 2
  - Furniture-GR00T 1
- 登録済み契約で実weight推論成功: 4
  - raw GR00T 2
  - coarse-insert GR00T 1
  - chunk-relative Diffusion v2 1
- 契約不整合を正しく拒否: 1
- metadata/構造監査のみ: 8

したがって、24 repoすべてを監査し、15モデルは実weightから期待次元の有限action tensorを
生成できた。全15モデルで`robot_command_sent=false`、`dds_initialized=false`、
`physical_transport_imported=false`を確認した。登録済み4モデルのweight実測は
`outputs/model_evaluation/team_ramen_registered_model_weight_probe_20260731.json`へ保存した。

旧`IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_1`は、
`z-score action normalization`に対して`clip_sample=true`であり、現在の安全なDiffusion
loaderが意図的に拒否した。weight破損ではなく学習・推論設定契約の不整合である。

Pi0.5の公開configにはLeRobot 0.6.0標準外の学習用fieldが残っていた。元artifactを変更せず、
無効な学習用fieldだけをメモリ上のconfig投影から除外し、`compile_model=max-autotune`も
一回限りのprobeでは無効化した。2モデルとも32D入力から`[1,10,32]`の有限出力を得た。

Furniture-GR00Tの公開configには学習機固有の`/dev/shm/...` base pathが残っていた。
固定revision `2fc962b...`を確認して`nvidia/GR00T-N1.7-3B`へメモリ上だけで戻し、
serialized processorを読み込んだ結果、49D入力から`[1,10,53]`の有限出力を得た。

なお、実weight推論成功15件のうち、実機mappingまでレビュー済みなのは登録済み4件だけで
ある。残り11件はtensor互換性の確認であり、そのまま実機へ接続してはならない。

## CLI

詳細手順は
`inference/desktop/model_evaluation/README.md`の「全24モデルの推測オフライン監査」を参照。
