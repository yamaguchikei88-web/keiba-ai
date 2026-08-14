# AI HANDOFF

## 現在地

- Phase: 2 予想履歴・モデル・実験管理基盤（完了）
- 現在Task: Phase 2のcode/docs/test commit/push
- 次Task: 実artifactの場所と読取り専用アクセスを確認し、承認後に既存APIへ安全にregistryを接続する設計を行う。

## 完了した作業

- GitHub `main` の公開構成、スクレイパー、SQLite schema、ML、API、デプロイ設定を監査。
- 監査レポート、計画、状態、アーキテクチャ、引継ぎ文書を作成。
- 空の無関係なローカルGit初期化フォルダとは別に、このGitHub作業コピーを安全にcloneした。
- 正しい作業コピーと親プロジェクト範囲を探索し、SQLite/モデル/metadata/予測履歴が0件であることを確認した。既知のWindows Google Drive相当パスも存在しない。
- `registry/store.py`に、明示実行のみのSQLite migration、feature/model/experiment/prediction/metrics registry、明示的production昇格を実装した。`tests/test_registry.py`の3テストは成功。

## 変更範囲

- 変更対象: `registry/`、`tests/test_registry.py`、Phase 2文書、状態/引継ぎ文書。
- 変更していないもの: `api/`、`ml/`、`scraper/`、`frontend/`、既存DB schema、データ、モデル、デプロイ設定、外部サービス。

## 発見した問題

1. `stats_cache.pkl`は全履歴集計で予測時点を切らず、未来情報を含み得る。
2. `/result/register`が自動再学習・artifact上書きをする。
3. 予測履歴、モデルversion台帳、実験・バックテストがない。
4. 学習で使うodds/popularityは推論では欠損である。

## 判断が必要な事項と次の作業

- 実DB/モデルartifactへの読取り専用アクセスを与えるか。
- 既存モデルを基準モデルとして凍結する方法。
- prediction/model/experiment台帳の保存先。

未実装: registryを既存API/DBに接続するmigration、prediction/result入力の記録、as-of特徴量、backtest、評価計算、既存の自動再学習停止。次は実artifactを読取り確認し、互換性・バックアップ・手動承認を含む別taskとして接続を設計する。`/result/register`を今回変更していない点に注意。
