# CURRENT STATE

最終更新: 2026-08-14 JST

- 現在Phase: 2（予想履歴・モデル・実験管理基盤の設計と単体テスト完了）。実DB/モデルartifactの所在確認は未完了。
- 正しい作業コピー: このリポジトリの `main`、`origin=https://github.com/yamaguchikei88-web/keiba-ai.git`。
- 今回の変更範囲: Phase 0/1文書に加え、明示的に実行する研究registry migration/APIと単体テスト。既存API、ML、scraper、DB、データ、モデル、外部設定は未変更。
- 正しい作業コピーと親プロジェクト範囲にも実DB、学習済みモデル、統計cache、metadata、予測履歴、実験ログは存在しない。実データ件数・日付範囲・実測性能は未確認。
- 最重要リスク: 全履歴統計キャッシュの未来情報、`/result/register`による自動再学習、予測・モデル履歴の未保存。
- Phase 2 registryは予測時刻・cutoff・入力snapshot・model/feature/experiment versionを保存できるが、既存APIには未接続である。
- Phase 2実装は `e04d84b` でGitHub `main` に同期済み。`python -m unittest discover -s tests -v` は3件成功。
- 複数PC準備として、artifact pathの環境変数化、非破壊`setup.bat`、README、共有storage方針を追加済み。実DB/model/外部設定は未変更。Python/registry/path testsは5件成功。
- 詳細: `docs/PHASE0_AUDIT_REPORT.md`、`docs/PHASE1_DATA_INVENTORY.md`、`docs/DATA_MODEL.md`。
