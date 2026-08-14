# Phase 1: 実DB・モデル・データ inventory（読み取り専用）

調査日: 2026-08-14 JST
対象作業コピー: `<workspace>/keiba-ai-github`（`main`）
実施範囲: ローカルの正しい作業コピーとその親プロジェクトフォルダ、コードが参照する既知のローカル外部パス。DB接続、スクレイパー、学習、API起動、外部サービス操作は行っていない。

## 結論

この環境で実際に使えるSQLite DB、学習済みモデル、統計キャッシュ、モデルmetadata、予測履歴、実験ログは **0件** だった。従って、実データ期間・レース数・欠損率・実測モデル性能は確定できない。今回得られた数値は「ファイルが存在しない」というinventory結果であり、推定値ではない。

## DB inventory

`.db`、`.sqlite`、`.sqlite3`を、正しい作業コピーおよび親プロジェクトフォルダで再帰的に読取り探索したが0件だった。`data/keiba.db`はコードの既定パスだが、`data/*.db`は`.gitignore`対象で、Git追跡ファイルにも存在しない。

| 項目 | 実測結果 |
|---|---|
| DBファイル | 存在しない |
| サイズ・更新日時・SQLite判定 | 対象ファイルなし |
| tables/schema/PK/FK/行数 | DBなしのため未取得 |
| 最古/最新日・レース/出走/馬/騎手/調教師/odds/result件数 | DBなしのため未取得 |
| 実データ品質（NULL、重複、ID不整合、異常値） | DBなしのため未検査 |

コードが定義する想定schemaは `races`（`race_id` PK）、`race_results`（`id` PK、`race_id` FK）、`horses`（`horse_id` PK）、`training_logs`（`id` PK、`race_id` FK）。prediction/model/experiment/result照合/payoutテーブルは定義されない。定義根拠は`scraper/netkeiba_scraper.py:init_db()`。

### 静的な品質上の注意

実データに対する品質判定ではない。スクレイパーは結果表の`cols[8]`を`weight`と`margin`の両方に参照しており、列対応を実DB・対象HTMLで検証する必要がある。また`race_results`に一意制約がなく、SQLite FK有効化もコード上で行われない。

## モデル inventory

`.pkl`、`.joblib`、`.model`、`.lgb`、`.bin`、`model_meta.json`を同範囲で探索したが0件だった。`models/`もGit追跡対象ではない。

想定artifactは`models/keiba_lgbm.pkl`、`models/stats_cache.pkl`、`models/model_meta.json`。保存コードは`ml/train.py:155,163,174`、読込コードは`ml/predict.py:42,45`。実artifactがないためモデル種類の実体検証、feature数、version、学習期間、ファイルサイズ、更新日時、metadata内容は未確認である。

## feature inventory

targetは`finish_pos == 1`。`ml/features.py:154`の`FEATURE_COLS`は30列で、レース条件（distance/surface/direction/track/weather/頭数）、枠/馬番、年齢/性別/馬体重、odds/popularity、馬の過去成績、騎手・調教師成績、父の重馬場成績、父/母父/競馬場符号、脚質、月、曜日から成る。

前処理は固定マップでのカテゴリ数値化、pandas category codes（father/maternal_father/racecourse）、欠損の`-1`補完である。normalization/scalingはない。rolling/statistical featuresは`build_horse_stats()`の馬・騎手・調教師・血統の過去集計と直近3走平均、脚質である。

## 学習 pipeline

1. `load_raw_data()`がDBの`race_results`、`races`、`horses`を結合し、`finish_pos IS NOT NULL`だけを対象に日付/race_id/着順順で読む。
2. `prepare_dataset()`が特徴量を作り、targetを1着フラグにする。
3. `train()`が行順で先頭80%をtrain、末尾20%をvalidationにする。random splitではなく時系列を意図した順序分割だが、race単位境界、日付境界、walk-forwardは保証しない。
4. LightGBM binary GBDTを最大5000 round/early stopping 100で学習し、validation AUCだけを算出する。
5. joblib/JSONへmodel、stats cache、metadataを書き出す（ただし今回実行していない）。

設定: `num_leaves=127`、`learning_rate=0.05`、`feature_fraction=0.8`、`bagging_fraction=0.8`、`min_child_samples=50`、L1/L2=0.1、metric=AUC。

## prediction pipeline と情報時点

`POST /predict` -> `build_race_id` -> `fetch_shutuba()`（netkeiba出馬表） -> `load_model()` -> `lookup_stats()` -> `encode_features()` -> LightGBM `predict()` -> スコア降順/印/上位3頭買い目 -> API response。

予測時に取得しようとする情報は出馬表の馬、枠/馬番、年齢/性別、馬体重、騎手、調教師、父、レース条件である。`odds`と`popularity`は予測時に`None`である。一方、着順、走破タイム、着差、通過順、上がり3F、結果ページの最終odds/popularityは結果確定後の情報で、予測入力に使ってはならない。

## リーク詳細

`ml/features.py:61`の`build_horse_stats()`は各レースについて`past = df[df["date"] < race_date]`を用いるため、学習行の馬・騎手・調教師・父・脚質特徴は日付より前に限ろうとしている。ただし同日内の時刻は扱えない。

重大なのは、`ml/train.py:48`の`build_stats_cache(raw_df)`である。学習後に`load_raw_data()`の **全期間** `raw_df` を馬ID、騎手名、調教師名、父ごとにgroupbyし、勝率・複勝率・平均着順・直近3走・距離別勝率・馬場別勝率を作る。日付で切らない。これを`stats_cache.pkl`へ保存し、`ml/predict.py:139`の`lookup_stats()`がrace_id/予測日を受けずにそのまま返す。したがって予測対象日より後の結果を含み得る。該当する全履歴統計は以下。

- horse: `race_count`、`win_rate`、`top3_rate`、`avg_finish`、`recent_avg`、距離別win、馬場別win
- jockey: `win_rate`、`top3_rate`
- trainer: `win_rate`
- father_heavy: 重・不良でのwin_rate

加えて、カテゴリcodesは全データを符号化してから80/20分割し、学習のodds/popularityは推論時欠損である。いずれも公平な確率評価を妨げる。

## 自動再学習の追跡

`api/main.py:93`の`POST /result/register`が`data/keiba.db`へ`race_results.finish_pos`をUPDATEしcommitする。直後にdaemon threadを作り、`train.retrain_with_new_data()`（`ml/train.py:179`）を呼ぶ。同関数は`train()`を実行し、`models/keiba_lgbm.pkl`、`models/stats_cache.pkl`、`models/model_meta.json`を同じ固定パスへ上書き保存する。モデルversion、候補比較、承認、バックアップ、入力検証はない。今回はこのendpoint/API/DB/モデルに変更を加えていない。

## 外部データ・Render/Vercel

Colab手順は`/content/drive/MyDrive/keiba-ai/data/keiba.db`と`.../models/`を参照する。既知のWindows Google Drive/My Drive相当パスとColabパスはこの環境に存在しないため、Drive実体にはアクセスできなかった。

frontendはNext.jsで、環境変数名は`NEXT_PUBLIC_API_URL`（example値はRender API URL）。未設定時は`http://localhost:8000`。Render設定は`pip install -r requirements.txt`、`cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`、環境変数名`PYTHON_VERSION`。DBは相対パス`data/keiba.db`、モデルは`models/`を想定し、Render persistent diskやartifact配置の設定はコード・render.yamlにない。Vercel/Renderの実設定・環境変数値・稼働状態は未調査。

## 現在の実データ状況

| 分類 | 確定結果 |
|---|---|
| A. すぐ使える実データ | なし |
| B. 存在するが整理が必要なデータ | なし（この環境で実体未発見） |
| C. 存在するが学習に不十分なデータ | なし（この環境で実体未発見） |
| D. 現在存在しないデータ | SQLite DB、学習済みmodel/cache/meta、予測履歴、実験ログ、払戻・時点付きオッズ |
| E. 外部から必要 | 実DB/モデルartifactの読取り専用提供、時点付き出馬表/odds、払戻、モデル・予測・実験台帳用データ |

## Phase 1結論と次作業

Phase 1は、アクセス可能なローカル範囲でのinventoryとして完了した。実DBとmodelsを入手または読取り専用でマウントできるまで、要求された実データ量・品質数値は測定不能である。次はPhase 2ではなく、まず実artifactの場所と読取り専用アクセスを確認し、ファイルhash、SQLite `PRAGMA`、テーブル件数、日付範囲、NULL/重複を取得する追加inventory taskを承認する。
