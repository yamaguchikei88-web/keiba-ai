# AI HANDOFF

## 現在地

- Phase: 0 現状監査（完了）
- 現在Task: GitHub同期（この文書群のdoc-only commit/push）
- 次Task: Phase 1開始可否の判断待ち。実DB・モデルinventoryはまだ行わない。

## 完了した作業

- GitHub `main` の公開構成、スクレイパー、SQLite schema、ML、API、デプロイ設定を監査。
- 監査レポート、計画、状態、アーキテクチャ、引継ぎ文書を作成。
- 空の無関係なローカルGit初期化フォルダとは別に、このGitHub作業コピーを安全にcloneした。

## 変更範囲

- 変更対象: `docs/MASTER_PLAN.md`、`docs/CURRENT_STATE.md`、`docs/ARCHITECTURE.md`、本ファイル、`docs/PHASE0_AUDIT_REPORT.md`、`tasks/CURRENT_TASK.md` のみ。
- 変更していないもの: 既存コード、DB schema、データ、モデル、デプロイ設定、外部サービス。

## 発見した問題

1. `stats_cache.pkl`は全履歴集計で予測時点を切らず、未来情報を含み得る。
2. `/result/register`が自動再学習・artifact上書きをする。
3. 予測履歴、モデルversion台帳、実験・バックテストがない。
4. 学習で使うodds/popularityは推論では欠損である。

## 判断が必要な事項と次の作業

- 実DB/モデルartifactへの読取り専用アクセスを与えるか。
- 既存モデルを基準モデルとして凍結する方法。
- prediction/model/experiment台帳の保存先。

次は、承認後にDBを変更せずinventory（hash、schema、件数、日付範囲、NULL率）を取得する。自動再学習停止、schema変更、モデル変更は別task・承認後に行う。
