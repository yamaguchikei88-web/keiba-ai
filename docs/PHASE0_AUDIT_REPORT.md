# 競馬予想AI 現状監査レポート

監査日: 2026-08-14 JST。対象は `yamaguchikei88-web/keiba-ai` のGitHub `main`。Phase 0として、公開コード・設定だけを読取り監査し、既存コード・DB・データ・モデル・デプロイは変更していない。

## 1. プロジェクト概要

JRA競馬を対象に、netkeibaから過去レースを収集してSQLiteへ保存し、LightGBMで各出走馬の「1着」二値分類を行い、FastAPIからNext.js UIへ返す最小構成である。LLM利用箇所は確認できなかった。

## 2. ファイル構成

`api/main.py`（FastAPI）、`frontend/`（Next.js 14/React 18/Tailwind）、`ml/features.py`、`ml/train.py`、`ml/predict.py`、`scraper/netkeiba_scraper.py`、`scraper/horse_scraper.py`、`colab_training.py`、`training.ipynb`、`requirements.txt`、`render.yaml`がある。GitHub上にテスト、CI、Docker、migration、`data/`、`models/`、予測履歴、実験ログは確認できない。frontendの`src/app`、notebook、bat/runtimeの詳細と実デプロイ状態は未監査。

## 3. 現在のアーキテクチャ

`netkeiba -> scraper -> data/keiba.db -> ml/train.py -> models/{keiba_lgbm.pkl,stats_cache.pkl} -> FastAPI /predict -> Next.js`。RenderはPython 3.11で`api/main:app`を起動する。GitHub AboutにVercel URLがある。外部設定は変更・検証していない。

## 4. データ取得構成

- 取得元: `db.netkeiba.com`（過去結果）と`race.netkeiba.com`（出馬表）。
- 方法: requests + BeautifulSoup。過去レースは1秒/件、血統は1.5秒/頭の待機を置く。
- 計画期間: `scrape_all(2015, 2025)`。コメントは過去10年・約10万レースだが、DB実体がないため実データ期間・件数は未確認。
- 保存: 通常は`data/keiba.db`、Colab手順ではGoogle Drive。

コード上の項目は、日付・競馬場・R・距離・芝ダ・向き・天候・馬場・頭数、着順、枠/馬番、馬/騎手/調教師、年齢・性別・馬体重、単勝オッズ・人気、走破タイム、着差、通過順、上がり3F、血統。払戻、オッズ取得時点、取消状態は扱わない。schema上の騎手ID・調教師ID・馬体重増減・grade・調教ログは現行スクレイパーで保存されない。

## 5. データベース構成

SQLite `data/keiba.db`。`races`（PK `race_id`）、`race_results`（PK `id`、`race_id` FK）、`horses`（PK `horse_id`）、`training_logs`（PK `id`、`race_id` FK）。race_resultsには着順、枠/馬番、馬、騎手/調教師、odds/popularity、time/margin/passing_order/last_3fを持つ。FKは宣言のみで有効化はなく、結果一意制約、prediction/model/experiment/payoutテーブルはない。

## 6. 現在の予想ロジック

`POST /predict`がrace_idを作り出馬表を取得する。馬・騎手・調教師・父の統計をキャッシュから読み、特徴量を符号化してLightGBMの`predict`を実行する。返す`win_prob`はレース内で合計1に正規化されない。降順で◎○▲△×を振り、上位3頭から単勝・馬連・三連複・三連単を機械生成する。予測時にオッズ・EV・資金配分・払戻は計算せず、予測も保存しない。

## 7. 使用モデルと特徴量

LightGBM GBDT二値分類、targetは`finish_pos == 1`。主な設定は`num_leaves=127`、`learning_rate=0.05`、最大5000 rounds、early stopping 100。joblibでモデル/統計キャッシュ、JSONで学習時刻・AUC・行数・特徴量一覧を保存する設計だがartifact実体と実測AUCは未確認。

30特徴量は、レース条件、枠/馬番、年齢/性別/馬体重、odds/popularity、馬の過去成績、騎手/調教師成績、父の重馬場成績、父/母父/競馬場のカテゴリコード、脚質、月、曜日。

## 8. 予想保存・バックテスト・評価

予測値、入力、モデルversion、オッズ時刻、買い目、結果照合の保存はない。`/result/register`は既存`race_results.finish_pos`の更新のみ。専用バックテストはなく、時系列順の先頭80%/末尾20%でAUCを1件計算するのみ。ROI、的中率、払戻、購入数、Log Loss、Brier Score、Calibrationは未実装・未保存である。

## 9. データリーケージのリスク

1. **重大**: `build_stats_cache()`が全期間結果を集計し、`predict.py`が予測日で切らず使用する。過去バックテストや過去日予測に将来結果が混入する。
2. **重大**: 学習では結果ページの`odds`と`popularity`を使うが、推論では`None`（-1）。最終オッズの時点も記録されない。
3. `build_horse_stats()`は`date < race_date`を使うため個別行は前進集計の意図があるが、同日レースの時刻を再現できない。
4. カテゴリ符号化を全データで行ってから80/20分割するため、前処理がtrain-onlyでない。
5. `/result/register`は無認可で結果更新・バックグラウンド自動再学習・artifact上書きをする。再現性・公平比較・手動採用の原則に反する。

## 10. 問題分類とロードマップ

- **A 致命的**: 全履歴キャッシュの未来情報、自動本番再学習、予測履歴なし。
- **B 今すぐ設計すべき**: odds/popularityの学習推論不一致、DB/model実体のinventory、データ取得列の対応確認。
- **C 後で改善**: train-only encoder、DB制約/migration、CORS/認証、テスト/CI。
- **D 現時点で問題なし**: LLMが最終予測を決めないこと、SQLite/Colab/Render/Vercelの無料構成、時系列80/20の方向性。

推奨順序は、(1)読取り専用inventoryとデータ辞書、(2)追記専用prediction/model/experiment/result台帳、(3)as-of特徴量とrace単位walk-forward、(4)Log Loss/Brier/Calibration/EV/ROI、(5)旧モデル比較と手動承認による採用。大規模リファクタリング、LLM主導予測、特徴量大量追加、DB置換、デプロイ変更、新モデル導入は現時点で不要。
