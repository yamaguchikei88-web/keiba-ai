import sqlite3
import tempfile
import unittest
from pathlib import Path

from registry.store import ResearchRegistry, initialize_registry


class ResearchRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "registry.sqlite"
        initialize_registry(self.db_path)
        self.registry = ResearchRegistry(self.db_path)
        self.registry.register_feature_set("features_v1", "baseline", ["distance", "age"])

    def tearDown(self):
        self.tempdir.cleanup()

    def test_migration_creates_expected_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            conn.close()
        self.assertTrue({"model_registry", "experiment_registry", "prediction_runs", "prediction_entries", "metric_records"} <= tables)

    def test_model_version_requires_explicit_validated_promotion(self):
        self.registry.register_model(
            model_version="v1.0.0", model_type="lightgbm", feature_set_version="features_v1",
            model_path="models/v1.0.0/model.pkl", parameters={"num_leaves": 31}, status="validated",
        )
        self.assertIsNone(self.registry.production_model())
        self.registry.approve_production("v1.0.0", approved_at="2026-08-14T00:00:00+00:00")
        self.assertEqual(self.registry.production_model()["model_version"], "v1.0.0")

    def test_experiment_prediction_and_metrics_are_linked(self):
        self.registry.register_model(
            model_version="v1.0.0", model_type="lightgbm", feature_set_version="features_v1",
            model_path="models/v1.0.0/model.pkl", parameters={}, status="candidate",
        )
        self.registry.record_experiment(
            experiment_id="exp-001", model_version="v1.0.0", feature_set_version="features_v1",
            parameters={"seed": 42}, status="completed",
        )
        self.registry.record_prediction(
            prediction_id="pred-001", race_id="202601010101", prediction_time="2026-01-01T09:00:00+09:00",
            model_version="v1.0.0", feature_set_version="features_v1", experiment_id="exp-001",
            input_snapshot={"race_id": "202601010101", "odds_observed_at": "2026-01-01T09:00:00+09:00"},
            entries=[{"horse_id": "horse-a", "predicted_win_probability": 0.3, "predicted_rank": 1,
                      "odds_at_prediction": 4.2}],
        )
        self.registry.record_metrics("experiment", "exp-001", {"log_loss": 0.61, "brier_score": 0.21})
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM prediction_entries").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM metric_records").fetchone()[0], 2)
            self.assertIsNotNone(conn.execute("SELECT input_snapshot_json FROM prediction_runs").fetchone()[0])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
