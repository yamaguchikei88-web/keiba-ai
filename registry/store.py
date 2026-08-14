"""SQLite registry for predictions, models, feature sets, and experiments.

Nothing in the existing API imports this module. A caller must explicitly create
or migrate a database with ``initialize_registry`` before using it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Mapping[str, Any] | list[Any] | None) -> str | None:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None


@contextmanager
def _connect(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(Path(database_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_registry(database_path: str | Path) -> None:
    """Create only the registry tables, idempotently, in an explicitly chosen DB.

    This function deliberately does not run at import time and is not called by
    the current API, scraper, trainer, or prediction code.
    """
    with _connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feature_sets (
                feature_set_version TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                description TEXT NOT NULL,
                feature_names_json TEXT NOT NULL,
                code_reference TEXT,
                status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'retired'))
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                model_version TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_type TEXT NOT NULL,
                feature_set_version TEXT NOT NULL REFERENCES feature_sets(feature_set_version),
                training_start TEXT,
                training_end TEXT,
                validation_start TEXT,
                validation_end TEXT,
                parameters_json TEXT NOT NULL,
                metrics_json TEXT,
                status TEXT NOT NULL CHECK (status IN ('candidate', 'validated', 'production', 'retired')),
                model_path TEXT NOT NULL,
                artifact_sha256 TEXT,
                approved_at TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS experiment_registry (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_version TEXT REFERENCES model_registry(model_version),
                feature_set_version TEXT NOT NULL REFERENCES feature_sets(feature_set_version),
                training_start TEXT,
                training_end TEXT,
                validation_start TEXT,
                validation_end TEXT,
                test_start TEXT,
                test_end TEXT,
                parameters_json TEXT NOT NULL,
                metrics_json TEXT,
                conclusion TEXT,
                status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'rejected'))
            );

            CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_id TEXT PRIMARY KEY,
                race_id TEXT NOT NULL,
                prediction_time TEXT NOT NULL,
                model_version TEXT NOT NULL REFERENCES model_registry(model_version),
                feature_set_version TEXT NOT NULL REFERENCES feature_sets(feature_set_version),
                experiment_id TEXT REFERENCES experiment_registry(experiment_id),
                data_cutoff_time TEXT NOT NULL,
                input_snapshot_json TEXT NOT NULL,
                input_snapshot_sha256 TEXT,
                prediction_reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_entries (
                prediction_id TEXT NOT NULL REFERENCES prediction_runs(prediction_id) ON DELETE CASCADE,
                horse_id TEXT NOT NULL,
                predicted_win_probability REAL NOT NULL CHECK (predicted_win_probability >= 0 AND predicted_win_probability <= 1),
                predicted_place_probability REAL CHECK (predicted_place_probability >= 0 AND predicted_place_probability <= 1),
                predicted_rank INTEGER NOT NULL CHECK (predicted_rank > 0),
                odds_at_prediction REAL,
                final_odds REAL,
                actual_rank INTEGER,
                actual_win INTEGER CHECK (actual_win IN (0, 1)),
                actual_place INTEGER CHECK (actual_place IN (0, 1)),
                PRIMARY KEY (prediction_id, horse_id)
            );

            CREATE TABLE IF NOT EXISTS bet_records (
                bet_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL REFERENCES prediction_runs(prediction_id),
                bet_flag INTEGER NOT NULL CHECK (bet_flag IN (0, 1)),
                bet_type TEXT,
                bet_amount REAL NOT NULL CHECK (bet_amount >= 0),
                payout REAL,
                profit REAL,
                settled_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metric_records (
                metric_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                scope_type TEXT NOT NULL CHECK (scope_type IN ('model', 'experiment', 'prediction_run', 'betting')),
                scope_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_details_json TEXT,
                UNIQUE(scope_type, scope_id, metric_name)
            );

            CREATE INDEX IF NOT EXISTS idx_prediction_runs_race_time
                ON prediction_runs(race_id, prediction_time);
            CREATE INDEX IF NOT EXISTS idx_prediction_entries_prediction
                ON prediction_entries(prediction_id);
            CREATE INDEX IF NOT EXISTS idx_models_status
                ON model_registry(status);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _utc_now()),
        )


class ResearchRegistry:
    """Explicit writer for the new registry tables; no implicit model promotion."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def register_feature_set(
        self, version: str, description: str, feature_names: list[str], *, code_reference: str | None = None,
        status: str = "candidate",
    ) -> None:
        with _connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO feature_sets VALUES (?, ?, ?, ?, ?, ?)",
                (version, _utc_now(), description, _json(feature_names), code_reference, status),
            )

    def register_model(self, *, model_version: str, model_type: str, feature_set_version: str,
                       model_path: str, parameters: Mapping[str, Any], status: str = "candidate",
                       metrics: Mapping[str, Any] | None = None, periods: Mapping[str, str] | None = None,
                       artifact_sha256: str | None = None, notes: str | None = None) -> None:
        periods = periods or {}
        with _connect(self.database_path) as conn:
            conn.execute(
                """INSERT INTO model_registry(
                    model_version, created_at, model_type, feature_set_version, training_start, training_end,
                    validation_start, validation_end, parameters_json, metrics_json, status, model_path,
                    artifact_sha256, approved_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (model_version, _utc_now(), model_type, feature_set_version, periods.get("training_start"),
                 periods.get("training_end"), periods.get("validation_start"), periods.get("validation_end"),
                 _json(parameters), _json(metrics), status, model_path, artifact_sha256, notes),
            )

    def record_experiment(self, *, experiment_id: str, feature_set_version: str,
                          parameters: Mapping[str, Any], status: str = "planned",
                          model_version: str | None = None, periods: Mapping[str, str] | None = None,
                          metrics: Mapping[str, Any] | None = None, conclusion: str | None = None) -> None:
        periods = periods or {}
        with _connect(self.database_path) as conn:
            conn.execute(
                """INSERT INTO experiment_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, _utc_now(), model_version, feature_set_version,
                 periods.get("training_start"), periods.get("training_end"),
                 periods.get("validation_start"), periods.get("validation_end"),
                 periods.get("test_start"), periods.get("test_end"), _json(parameters), _json(metrics),
                 conclusion, status),
            )

    def record_prediction(self, *, prediction_id: str, race_id: str, prediction_time: str,
                          model_version: str, feature_set_version: str,
                          entries: Iterable[Mapping[str, Any]], experiment_id: str | None = None,
                          data_cutoff_time: str | None = None,
                          input_snapshot: Mapping[str, Any] | None = None,
                          input_snapshot_sha256: str | None = None, prediction_reason: str | None = None) -> None:
        """Store the immutable prediction header and one row per horse in one transaction."""
        entries = list(entries)
        with _connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO prediction_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (prediction_id, race_id, prediction_time, model_version, feature_set_version, experiment_id,
                 data_cutoff_time or prediction_time, _json(input_snapshot or {}), input_snapshot_sha256,
                 prediction_reason, _utc_now()),
            )
            conn.executemany(
                """INSERT INTO prediction_entries(
                    prediction_id, horse_id, predicted_win_probability, predicted_place_probability,
                    predicted_rank, odds_at_prediction, final_odds, actual_rank, actual_win, actual_place
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(prediction_id, e["horse_id"], e["predicted_win_probability"],
                  e.get("predicted_place_probability"), e["predicted_rank"], e.get("odds_at_prediction"),
                  e.get("final_odds"), e.get("actual_rank"), e.get("actual_win"), e.get("actual_place"))
                 for e in entries],
            )

    def record_metrics(self, scope_type: str, scope_id: str, metrics: Mapping[str, float | None],
                       details: Mapping[str, Any] | None = None) -> None:
        with _connect(self.database_path) as conn:
            conn.executemany(
                """INSERT INTO metric_records(metric_id, recorded_at, scope_type, scope_id, metric_name,
                    metric_value, metric_details_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope_type, scope_id, metric_name) DO UPDATE SET
                    metric_value=excluded.metric_value, metric_details_json=excluded.metric_details_json,
                    recorded_at=excluded.recorded_at""",
                [(f"{scope_type}:{scope_id}:{name}", _utc_now(), scope_type, scope_id, name, value, _json(details))
                 for name, value in metrics.items()],
            )

    def production_model(self) -> sqlite3.Row | None:
        with _connect(self.database_path) as conn:
            return conn.execute("SELECT * FROM model_registry WHERE status='production'").fetchone()

    def approve_production(self, model_version: str, approved_at: str | None = None) -> None:
        """An explicit human-controlled action; never called by result registration."""
        with _connect(self.database_path) as conn:
            candidate = conn.execute(
                "SELECT 1 FROM model_registry WHERE model_version=? AND status='validated'", (model_version,)
            ).fetchone()
            if candidate is None:
                raise ValueError("only an existing validated model can be promoted")
            conn.execute("UPDATE model_registry SET status='retired' WHERE status='production'")
            result = conn.execute(
                "UPDATE model_registry SET status='production', approved_at=? WHERE model_version=? AND status='validated'",
                (approved_at or _utc_now(), model_version),
            )
            assert result.rowcount == 1
