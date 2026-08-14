# CURRENT STATE

最終更新: 2026-08-14 JST

- 現在Phase: 1（ローカル読み取り専用inventory完了）。実DB/モデルartifactの所在確認は未完了。
- 正しい作業コピー: このリポジトリの `main`、`origin=https://github.com/yamaguchikei88-web/keiba-ai.git`。
- 今回の変更範囲: Phase 0/1の監査・計画・引継ぎ文書のみ。既存コード、DB、データ、モデル、外部設定は未変更。
- 正しい作業コピーと親プロジェクト範囲にも実DB、学習済みモデル、統計cache、metadata、予測履歴、実験ログは存在しない。実データ件数・日付範囲・実測性能は未確認。
- 最重要リスク: 全履歴統計キャッシュの未来情報、`/result/register`による自動再学習、予測・モデル履歴の未保存。
- 詳細: `docs/PHASE0_AUDIT_REPORT.md`、`docs/PHASE1_DATA_INVENTORY.md`。
