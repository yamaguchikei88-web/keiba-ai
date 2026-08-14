# CURRENT TASK

状態: 完了（Phase 2: 予想履歴・モデル・実験管理基盤）

`registry/store.py`へ明示実行型SQLite registry migration/APIを追加し、feature set、model、experiment、prediction run/entry、bet、metricを保存できるようにした。既存API/ML/scraperには未接続で、DB作成はテストの一時領域だけで行った。`tests/test_registry.py`は3件成功。実装は`e04d84b`でGitHub `main`に同期済み。次は実artifactへの読取り確認と、承認後の安全なAPI接続設計が必要。
