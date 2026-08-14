# Research Data Model

Phase 2 adds an explicit SQLite research registry through `registry/store.py`. It is not imported by `api/`, `ml/`, or `scraper/`; schema creation occurs only when a future approved caller explicitly invokes `initialize_registry(path)`.

The registry can be placed in a separate SQLite file (recommended initially) or, after backup/review, alongside the existing operational tables. It never replaces `races`, `race_results`, `horses`, or `training_logs`.

| Table | Purpose | Key fields |
|---|---|---|
| `schema_migrations` | Registry migration audit | version, applied_at |
| `feature_sets` | Immutable feature contract | feature_set_version, names JSON, code reference, status |
| `model_registry` | Versioned model artifact metadata | model_version, type, feature set, periods, params/metrics JSON, status, path, hash |
| `experiment_registry` | Reproducible experiment context | experiment_id, model/feature version, train/validation/test periods, params, metrics, conclusion |
| `prediction_runs` | Immutable prediction header | prediction_id, race_id, prediction_time, data_cutoff_time, model/feature/experiment, input snapshot JSON/hash, reason |
| `prediction_entries` | Per-horse probability and resolved outcome | horse_id, win/place probabilities, rank, prediction/final odds, actual rank/win/place |
| `bet_records` | Bet and settlement ledger | bet type/amount, payout, profit |
| `metric_records` | Named scalar metrics | scope, metric name/value, details JSON |

`prediction_time` and `data_cutoff_time` are mandatory. The latter is normally equal to prediction time, but is separate so a delayed pipeline cannot claim information was available later than it was. `input_snapshot_json` records the exact pre-race input payload; its SHA-256 can be used to compare it with an external immutable snapshot if one is introduced later.

Metrics are normalized as named records rather than fixed columns. Supported names include `log_loss`, `brier_score`, `roc_auc`, `top1_accuracy`, `top3_accuracy`, `calibration_ece`, `bets`, `investment`, `payout`, `profit`, `roi`, `hit_rate`, and `max_drawdown`. Calculation is deliberately out of scope for Phase 2.

No registry row is automatically created by the current API. Wiring prediction/result events to this registry needs a separate approved task with compatibility tests.
