# ARCHITECTURE

現状フロー:

`netkeiba過去結果 -> scraper -> SQLite(data/keiba.db) -> features/train -> models(joblib/json) -> FastAPI -> Next.js`

予測時は`ml/predict.py`がnetkeiba出馬表を直接取得し、保存済みLightGBMと統計キャッシュで各馬の1着スコアを出す。APIは`POST /predict`、結果登録は`POST /result/register`。後者はDB更新後に自動再学習を起動する。Phase 0では未変更であり、将来は承認制の候補実験フローを設計・承認してから対応する。

DB tables: `races`、`race_results`、`horses`、`training_logs`。prediction/result/model/experiment専用テーブルはない。

Phase 2 adds a separate, explicit research-registry layer: `feature_sets -> model_registry -> experiment_registry -> prediction_runs -> prediction_entries/bet_records/metric_records`. It is not yet wired to API or ML code, so existing production behavior remains unchanged. See `DATA_MODEL.md`.
