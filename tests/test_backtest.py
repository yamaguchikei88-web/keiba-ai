"""
Phase 4 tests: walk-forward backtest, evaluation metrics, model comparison,
promotion gate, and registry integration.

Test priorities (matching config/promotion_rules.METRIC_PRIORITY):
  1. Time-series split correctness (no future data leakage — the most critical)
  2. Probability metrics: LogLoss, Brier, ECE
  3. Discrimination: ROC-AUC
  4. Race-level: Top1/Top3 Accuracy
  5. Betting: ROI
  6. Model comparison and promotion gate
  7. Registry round-trip

Reliability note:
  All DB-dependent tests use synthetic DataFrames, not the real 96-race keiba.db.
  This keeps tests fast, deterministic, and independent of data collection.
"""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.backtest import (
    WalkForwardWindow,
    WindowResult,
    BetStrategy,
    MaxProbStrategy,
    make_walk_forward_windows,
)
from ml.evaluate import (
    compute_ece,
    compute_metrics,
    check_promotion_eligibility,
    aggregate_window_metrics,
)
from ml.compare import ModelSummary, compare_models, promote_to_validated
from config.promotion_rules import METRIC_PRIORITY, PRODUCTION_REQUIRES_HUMAN_APPROVAL


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_race_df(n_races: int = 8, horses_per_race: int = 12, start_date: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic race DataFrame matching load_raw_data() output schema."""
    rows = []
    base = pd.Timestamp(start_date)
    for r in range(n_races):
        race_date = base + pd.Timedelta(days=r * 7)  # one race day per week
        race_id = f"race_{r:03d}"
        for h in range(horses_per_race):
            rows.append({
                "race_id": race_id,
                "date": race_date,
                "racecourse": "東京",
                "race_num": r % 12 + 1,
                "distance": 2000,
                "surface": "芝",
                "direction": "左",
                "weather": "晴",
                "track_condition": "良",
                "num_horses": horses_per_race,
                "finish_pos": h + 1,
                "gate_num": h + 1,
                "horse_num": h + 1,
                "horse_id": f"horse_{h:03d}",
                "horse_name": f"Horse{h:03d}",
                "age": 4,
                "sex": "牡",
                "weight": 480,
                "weight_change": 0,
                "jockey_name": f"Jockey{h % 3}",
                "trainer_name": f"Trainer{h % 5}",
                "odds": float(h + 1) * 2.0,
                "popularity": h + 1,
                "time_seconds": 120.0 + h * 0.1,
                "last_3f": 33.0,
                "passing_order": f"{h+1}-{h+1}-{h+1}",
                "father": f"Father{h % 10}",
                "mother": f"Mother{h % 10}",
                "maternal_father": f"MatFather{h % 5}",
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_predictions(n_races: int = 3, horses_per_race: int = 5, correct_top1: bool = True) -> pd.DataFrame:
    """Synthetic predictions DataFrame matching WindowResult.predictions schema."""
    rows = []
    for r in range(n_races):
        race_id = f"r{r}"
        probs = np.random.dirichlet(np.ones(horses_per_race))
        winner_idx = 0  # horse 0 always wins
        for h in range(horses_per_race):
            # If correct_top1, ensure horse 0 has highest prob
            if correct_top1 and h == 0:
                prob = max(probs)
            elif correct_top1 and h > 0:
                prob = probs[h] * (1 - max(probs)) / (sum(probs[1:]) + 1e-9)
            else:
                prob = probs[h]

            rows.append({
                "race_id": race_id,
                "horse_id": f"h{h}",
                "horse_num": h + 1,
                "finish_pos": h + 1,
                "y_true": 1 if h == winner_idx else 0,
                "y_pred": float(prob),
                "predicted_rank": h + 1,  # will be overwritten
                "odds": float(h + 1) * 3.0,
            })

    df = pd.DataFrame(rows)
    # Compute predicted_rank within each race
    df["predicted_rank"] = df.groupby("race_id")["y_pred"].rank(ascending=False, method="first").astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════
# 1. Time-series split tests (most critical — future data leakage)
# ══════════════════════════════════════════════════════════════════════════

class TestWalkForwardSplit(unittest.TestCase):

    def _make_windows(self, n_races=8, horses=12, n_windows=2, min_train=4):
        df = _make_race_df(n_races=n_races, horses_per_race=horses)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return make_walk_forward_windows(df, n_windows=n_windows, min_train_races=min_train)

    def test_train_end_strictly_before_val_start(self):
        """All windows: train_end < val_start (strict inequality, no same-day overlap)."""
        windows = self._make_windows()
        for w in windows:
            self.assertLess(
                w.train_end, w.val_start,
                f"Window {w.window_id}: train_end({w.train_end.date()}) >= val_start({w.val_start.date()})"
            )

    def test_stats_cutoff_before_val_start(self):
        """stats_cutoff = val_start - 1 day, strictly before val_start."""
        windows = self._make_windows()
        for w in windows:
            self.assertEqual(w.stats_cutoff, w.val_start - pd.Timedelta(days=1))
            self.assertLess(w.stats_cutoff, w.val_start)

    def test_no_race_id_overlap_between_train_and_val(self):
        """No race_id appears in both train and val partitions."""
        df = _make_race_df(n_races=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows = make_walk_forward_windows(df, n_windows=2, min_train_races=4)
        for w in windows:
            train_ids = set(df[df["date"] < w.val_start]["race_id"])
            val_ids = set(df[(df["date"] >= w.val_start) & (df["date"] <= w.val_end)]["race_id"])
            self.assertEqual(len(train_ids & val_ids), 0,
                             f"Window {w.window_id}: race_id overlap between train and val")

    def test_future_data_does_not_affect_earlier_window(self):
        """Adding a future race must not change the train period of earlier windows.

        val_end may shift (more val days to distribute) — that is expected.
        The critical property is that train_start/train_end/val_start are stable:
        these define which data is used for training and thus prevent future leakage.
        """
        df = _make_race_df(n_races=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows_before = make_walk_forward_windows(df, n_windows=2, min_train_races=4)

        # Add a future race (far future — does not fall in existing val periods)
        future_row = df.iloc[0].copy()
        future_row["date"] = pd.Timestamp("2099-01-01")
        future_row["race_id"] = "future_race"
        df_with_future = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows_after = make_walk_forward_windows(df_with_future, n_windows=2, min_train_races=4)

        # Train boundaries (train_start, train_end) and val_start must not change.
        # val_end may change because the algorithm redistributes the extra val days.
        w_before = windows_before[0]
        w_after = windows_after[0]
        self.assertEqual(w_before.train_start, w_after.train_start,
                         "train_start changed after adding future data")
        self.assertEqual(w_before.train_end, w_after.train_end,
                         "train_end changed after adding future data")
        self.assertEqual(w_before.val_start, w_after.val_start,
                         "val_start changed — this could cause future data leakage into training")
        # Also verify the future race is not in the train period of Window 1
        self.assertLess(w_after.train_end, pd.Timestamp("2099-01-01"),
                        "Future race date must not enter the train period")

    def test_stats_cutoff_property(self):
        """WalkForwardWindow.stats_cutoff returns val_start - 1 day."""
        w = WalkForwardWindow(
            window_id=1,
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-02-01"),
            val_start=pd.Timestamp("2024-02-10"),
            val_end=pd.Timestamp("2024-02-28"),
            n_train_races=30,
            n_val_races=10,
        )
        expected = pd.Timestamp("2024-02-09")
        self.assertEqual(w.stats_cutoff, expected)

    def test_small_dataset_raises_value_error(self):
        """Raises ValueError when dataset is too small for min_train_races."""
        df = _make_race_df(n_races=2, horses_per_race=5)
        with self.assertRaises(ValueError):
            make_walk_forward_windows(df, n_windows=2, min_train_races=40)

    def test_reliability_warning_emitted(self):
        """UserWarning is issued when val_races < 30."""
        df = _make_race_df(n_races=8, horses_per_race=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_walk_forward_windows(df, n_windows=2, min_train_races=4)
        warning_texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(any("バックテスト基盤" in t or "動作確認" in t for t in warning_texts),
                        "Expected reliability warning not emitted")


# ══════════════════════════════════════════════════════════════════════════
# 2. Probability metrics: LogLoss, Brier, ECE
# ══════════════════════════════════════════════════════════════════════════

class TestProbabilityMetrics(unittest.TestCase):

    def _simple_preds(self, y_true, y_pred, odds=None):
        n = len(y_true)
        df = pd.DataFrame({
            "race_id": [f"r{i}" for i in range(n)],
            "y_true": y_true,
            "y_pred": y_pred,
            "predicted_rank": [1] * n,
        })
        if odds is not None:
            df["odds"] = odds
        return df

    def test_log_loss_perfect_model(self):
        """A model that outputs near-certainty for the correct class has low log_loss."""
        preds = self._simple_preds(
            y_true=[1, 0, 1, 0],
            y_pred=[0.99, 0.01, 0.99, 0.01],
        )
        m = compute_metrics(preds)
        self.assertIsNotNone(m["log_loss"])
        self.assertLess(m["log_loss"], 0.05)

    def test_log_loss_random_model(self):
        """A model that outputs 0.5 everywhere has higher log_loss than a good model."""
        preds_good = self._simple_preds([1, 0, 1, 0], [0.99, 0.01, 0.99, 0.01])
        preds_random = self._simple_preds([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        m_good = compute_metrics(preds_good)
        m_random = compute_metrics(preds_random)
        self.assertLess(m_good["log_loss"], m_random["log_loss"])

    def test_brier_score_range(self):
        """Brier score is in [0, 1]."""
        preds = self._simple_preds([1, 0, 1, 0], [0.8, 0.2, 0.6, 0.4])
        m = compute_metrics(preds)
        self.assertIsNotNone(m["brier_score"])
        self.assertGreaterEqual(m["brier_score"], 0.0)
        self.assertLessEqual(m["brier_score"], 1.0)

    def test_ece_perfect_calibration(self):
        """A model where acc == confidence in every bin has ECE ≈ 0."""
        # All samples in the same bin: pred=0.5, and exactly 50% are positive
        y_true = np.array([1.0, 0.0] * 10)
        y_pred = np.array([0.5] * 20)
        ece = compute_ece(y_true, y_pred)
        self.assertAlmostEqual(ece, 0.0, places=5,
                               msg="constant prediction == base rate should give ECE=0")

    def test_ece_range(self):
        """ECE is in [0, 1]."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        ece = compute_ece(y_true, y_pred)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 3. ROI tests (bet strategy layer)
# ══════════════════════════════════════════════════════════════════════════

class TestROI(unittest.TestCase):

    def _preds_with_odds(self, races):
        """races: list of [(y_pred, y_true, odds), ...]"""
        rows = []
        for r_i, race in enumerate(races):
            for h_i, (y_pred, y_true, odds) in enumerate(race):
                rows.append({
                    "race_id": f"r{r_i}",
                    "horse_id": f"h{h_i}",
                    "y_pred": y_pred,
                    "y_true": y_true,
                    "odds": odds,
                    "predicted_rank": 0,  # placeholder
                })
        df = pd.DataFrame(rows)
        df["predicted_rank"] = df.groupby("race_id")["y_pred"].rank(ascending=False, method="first").astype(int)
        return df

    def test_roi_all_win(self):
        """All bets win → ROI = (total_odds - n_races) / n_races > 0."""
        strategy = MaxProbStrategy()
        # 3 races, each: bet on horse 0 (highest prob, wins), odds=3.0
        races = [
            [(0.8, 1, 3.0), (0.2, 0, 5.0)],
            [(0.8, 1, 3.0), (0.2, 0, 5.0)],
            [(0.8, 1, 3.0), (0.2, 0, 5.0)],
        ]
        df = self._preds_with_odds(races)
        roi = strategy.compute_roi(df)
        self.assertIsNotNone(roi)
        # stake=3, payout=9, roi=(9-3)/3=2.0
        self.assertAlmostEqual(roi, 2.0, places=9)
        self.assertGreater(roi, 0)

    def test_roi_all_lose(self):
        """All bets lose → ROI = -1.0."""
        strategy = MaxProbStrategy()
        # 3 races: highest-prob horse always loses
        races = [
            [(0.8, 0, 3.0), (0.2, 1, 5.0)],
            [(0.8, 0, 3.0), (0.2, 1, 5.0)],
            [(0.8, 0, 3.0), (0.2, 1, 5.0)],
        ]
        df = self._preds_with_odds(races)
        roi = strategy.compute_roi(df)
        self.assertIsNotNone(roi)
        # stake=3, payout=0, roi=(0-3)/3=-1.0
        self.assertAlmostEqual(roi, -1.0, places=9)

    def test_roi_none_when_no_odds(self):
        """ROI is None when all odds are NaN."""
        strategy = MaxProbStrategy()
        df = pd.DataFrame({
            "race_id": ["r1", "r1"],
            "y_pred": [0.7, 0.3],
            "y_true": [1, 0],
            "odds": [np.nan, np.nan],
        })
        roi = strategy.compute_roi(df)
        self.assertIsNone(roi)


# ══════════════════════════════════════════════════════════════════════════
# 4. Race-level accuracy tests
# ══════════════════════════════════════════════════════════════════════════

class TestRaceLevelAccuracy(unittest.TestCase):

    def _preds(self, race_winner_ranks):
        """race_winner_ranks: {race_id: predicted_rank of the winner}"""
        rows = []
        for race_id, winner_rank in race_winner_ranks.items():
            for rank in range(1, 5):  # 4 horses
                rows.append({
                    "race_id": race_id,
                    "y_true": 1 if rank == winner_rank else 0,
                    "y_pred": 1.0 / rank,
                    "predicted_rank": rank,
                })
        return pd.DataFrame(rows)

    def test_top1_acc_all_correct(self):
        """All races: winner is predicted rank 1 → top1_acc = 1.0."""
        df = self._preds({"r1": 1, "r2": 1, "r3": 1})
        m = compute_metrics(df)
        self.assertAlmostEqual(m["top1_acc"], 1.0, places=9)

    def test_top1_acc_none_correct(self):
        """No race: winner is predicted rank 1 → top1_acc = 0.0."""
        df = self._preds({"r1": 4, "r2": 4, "r3": 4})
        m = compute_metrics(df)
        self.assertAlmostEqual(m["top1_acc"], 0.0, places=9)

    def test_top3_acc_winner_always_in_top3(self):
        """Winner is always rank ≤ 3 → top3_acc = 1.0."""
        df = self._preds({"r1": 2, "r2": 3, "r3": 1})
        m = compute_metrics(df)
        self.assertAlmostEqual(m["top3_acc"], 1.0, places=9)

    def test_top3_acc_winner_never_in_top3(self):
        """Winner is always rank 4 → top3_acc = 0.0."""
        df = self._preds({"r1": 4, "r2": 4})
        m = compute_metrics(df)
        self.assertAlmostEqual(m["top3_acc"], 0.0, places=9)


# ══════════════════════════════════════════════════════════════════════════
# 5. Model comparison tests
# ══════════════════════════════════════════════════════════════════════════

class TestModelComparison(unittest.TestCase):

    def _summary(self, version, metrics):
        return ModelSummary(
            model_version=version,
            feature_set_version="v1",
            training_cutoff_date="2024-02-01",
            n_windows=2,
            total_val_races=40,
            total_val_rows=600,
            metrics=metrics,
            window_metrics=[],
        )

    def test_better_log_loss_ranks_first(self):
        """Model with lower LogLoss ranks first (primary metric)."""
        good = self._summary("vGood", {"log_loss": 0.28, "brier_score": 0.06,
                                       "ece": 0.03, "roc_auc": 0.65,
                                       "top1_acc": 0.18, "top3_acc": 0.60, "roi": -0.10})
        bad  = self._summary("vBad",  {"log_loss": 0.40, "brier_score": 0.09,
                                       "ece": 0.08, "roc_auc": 0.55,
                                       "top1_acc": 0.25, "top3_acc": 0.70, "roi": 0.20})
        df = compare_models([good, bad])
        self.assertEqual(df.iloc[0]["model_version"], "vGood",
                         "Lower LogLoss model must rank first")

    def test_roi_only_improvement_does_not_win(self):
        """A model with worse LogLoss/Brier but better ROI ranks second."""
        # vA: better probs, worse ROI
        vA = self._summary("vA", {"log_loss": 0.28, "brier_score": 0.06, "ece": 0.03,
                                  "roc_auc": 0.65, "top1_acc": 0.18, "top3_acc": 0.60,
                                  "roi": -0.20})
        # vB: worse probs, better ROI — should NOT rank first
        vB = self._summary("vB", {"log_loss": 0.40, "brier_score": 0.09, "ece": 0.08,
                                  "roc_auc": 0.55, "top1_acc": 0.22, "top3_acc": 0.65,
                                  "roi": 0.50})
        df = compare_models([vA, vB])
        self.assertEqual(df.iloc[0]["model_version"], "vA",
                         "Better probability model must win even if ROI is lower")


# ══════════════════════════════════════════════════════════════════════════
# 6. Promotion gate tests
# ══════════════════════════════════════════════════════════════════════════

class TestPromotionGate(unittest.TestCase):

    def _good_metrics(self):
        return {"log_loss": 0.28, "brier_score": 0.055, "ece": 0.03,
                "roc_auc": 0.65, "top1_acc": 0.20, "top3_acc": 0.60, "roi": -0.05}

    def test_good_model_can_promote(self):
        """Model meeting all thresholds promotes from candidate to validated."""
        s = ModelSummary("vOK", "v1", "2024-02-01", 2, 40, 600, self._good_metrics(), [])
        promoted, reasons = promote_to_validated(s)
        self.assertTrue(promoted)
        self.assertEqual(s.status, "validated")
        self.assertEqual(reasons, [])

    def test_bad_log_loss_blocks_promotion(self):
        """Model with LogLoss above threshold is blocked."""
        m = self._good_metrics()
        m["log_loss"] = 0.50  # above max=0.35
        s = ModelSummary("vBad", "v1", "2024-02-01", 2, 40, 600, m, [])
        promoted, reasons = promote_to_validated(s)
        self.assertFalse(promoted)
        self.assertEqual(s.status, "candidate")
        self.assertTrue(any("log_loss" in r for r in reasons))

    def test_high_roi_alone_does_not_promote(self):
        """ROI above threshold doesn't help if LogLoss/Brier are bad."""
        m = {"log_loss": 0.50, "brier_score": 0.09, "ece": 0.08, "roc_auc": 0.55,
             "top1_acc": 0.30, "top3_acc": 0.75, "roi": 1.00}
        s = ModelSummary("vHighROI", "v1", "2024-02-01", 2, 40, 600, m, [])
        promoted, _ = promote_to_validated(s)
        self.assertFalse(promoted, "High ROI alone must not trigger promotion")

    def test_no_auto_production_promotion(self):
        """promote_to_validated never sets status to production."""
        s = ModelSummary("vOK2", "v1", "2024-02-01", 2, 40, 600, self._good_metrics(), [])
        promote_to_validated(s)
        self.assertNotEqual(s.status, "production",
                            "Automatic promotion to production must be impossible")

    def test_double_promote_blocked(self):
        """Promoting an already-validated model raises False (not candidate)."""
        s = ModelSummary("vValid", "v1", "2024-02-01", 2, 40, 600, self._good_metrics(), [])
        promote_to_validated(s)  # first: candidate → validated
        promoted2, reasons2 = promote_to_validated(s)
        self.assertFalse(promoted2)
        self.assertTrue(any("candidate ではありません" in r for r in reasons2))

    def test_production_requires_human_approval(self):
        """PRODUCTION_REQUIRES_HUMAN_APPROVAL constant must be True."""
        self.assertTrue(PRODUCTION_REQUIRES_HUMAN_APPROVAL)

    def test_too_few_val_races_blocks_promotion(self):
        """min_val_races floor blocks promotion when data is insufficient."""
        eligible, reasons = check_promotion_eligibility(
            self._good_metrics(), n_val_races=5
        )
        self.assertFalse(eligible)
        self.assertTrue(any("min_val_races" in r or "統計的信頼性" in r for r in reasons))


# ══════════════════════════════════════════════════════════════════════════
# 7. Registry round-trip test
# ══════════════════════════════════════════════════════════════════════════

class TestRegistryRoundTrip(unittest.TestCase):

    def test_register_and_retrieve_model(self):
        """Register a model, record metrics, then retrieve production model via registry."""
        from registry.store import ResearchRegistry, initialize_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_registry.db"
            initialize_registry(db_path)
            reg = ResearchRegistry(db_path)

            # Register feature set and model
            from ml.features import FEATURE_COLS
            reg.register_feature_set(
                version="v_test",
                description="test feature set",
                feature_names=FEATURE_COLS,
                status="active",
            )
            reg.register_model(
                model_version="v_test_model",
                model_type="lightgbm_binary",
                feature_set_version="v_test",
                model_path="/tmp/model.pkl",
                parameters={"n_estimators": 100},
                status="validated",
                metrics={"log_loss": 0.28, "roc_auc": 0.65},
            )

            # Record metrics
            reg.record_metrics(
                scope_type="model",
                scope_id="v_test_model",
                metrics={"log_loss": 0.28, "brier_score": 0.055, "roc_auc": 0.65},
            )

            # Approve production (human action simulation)
            reg.approve_production("v_test_model")

            # Retrieve production model
            prod = reg.production_model()
            self.assertIsNotNone(prod)
            self.assertEqual(prod["model_version"], "v_test_model")
            self.assertEqual(prod["status"], "production")


# ══════════════════════════════════════════════════════════════════════════
# 8. Walk-forward independence test
#    Adding future race data must not change past window predictions.
# ══════════════════════════════════════════════════════════════════════════

class TestWalkForwardIndependence(unittest.TestCase):
    """Verify that future races do not affect past window y_pred / metrics.

    This is the "independence test" requested in the walk-forward audit:
    if we add races that are strictly after a window's val_end, the
    predictions for that window (y_pred, feature values, metrics) must
    be bit-for-bit identical.

    Implementation:
      1. Build a synthetic dataset large enough for 2 windows.
      2. Run run_window() for Window 1 → record y_pred.
      3. Append future races (dates >> val_end) to the dataset.
      4. Run run_window() again for Window 1 (same window object).
      5. Assert y_pred arrays are exactly equal.
    """

    def _make_large_race_df(self):
        """12 race days × 10 horses — enough for 2 windows with min_train=4."""
        return _make_race_df(n_races=12, horses_per_race=10, start_date="2024-01-01")

    def _add_future_races(self, df: pd.DataFrame, n_future: int = 3) -> pd.DataFrame:
        """Append synthetic races far in the future (2099)."""
        future_rows = []
        base = pd.Timestamp("2099-06-01")
        for r in range(n_future):
            race_date = base + pd.Timedelta(days=r * 7)
            race_id = f"future_race_{r:03d}"
            for h in range(10):
                row = df.iloc[0].copy()
                row["race_id"] = race_id
                row["date"] = race_date
                row["horse_id"] = f"future_horse_{h:03d}"
                row["horse_num"] = h + 1
                row["finish_pos"] = h + 1
                row["odds"] = float(h + 2) * 2.0
                row["popularity"] = h + 1
                future_rows.append(row)
        future_df = pd.DataFrame(future_rows)
        return pd.concat([df, future_df], ignore_index=True)

    def test_future_races_do_not_change_past_window_y_pred(self):
        """Adding future races must not alter y_pred for past windows.

        This is the most critical independence property of walk-forward:
        the model is trained on data < val_start, and features for val
        horses depend only on data <= stats_cutoff (= val_start - 1 day).
        Future races (post-val_end) must be invisible to both.
        """
        from ml.backtest import run_window

        df = self._make_large_race_df()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows = make_walk_forward_windows(df, n_windows=2, min_train_races=4)

        w1 = windows[0]

        # Run Window 1 on the original dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_before = run_window(df, w1)

        y_pred_before = result_before.predictions["y_pred"].values.copy()

        # Add future races (well past val_end of any window)
        df_with_future = self._add_future_races(df, n_future=3)

        # Run the same Window 1 on the augmented dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_after = run_window(df_with_future, w1)

        y_pred_after = result_after.predictions["y_pred"].values.copy()

        # y_pred must be identical: future data must be invisible to Window 1
        self.assertEqual(
            len(y_pred_before), len(y_pred_after),
            "Number of val predictions changed after adding future data"
        )
        np.testing.assert_array_equal(
            y_pred_before, y_pred_after,
            err_msg=(
                "y_pred changed after adding future races — "
                "future data is leaking into past window predictions!"
            ),
        )

    def test_future_races_do_not_change_past_window_metrics(self):
        """Adding future races must not alter computed metrics for past windows."""
        from ml.backtest import run_window
        from ml.evaluate import compute_metrics

        df = self._make_large_race_df()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows = make_walk_forward_windows(df, n_windows=2, min_train_races=4)

        w1 = windows[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_before = run_window(df, w1)

        metrics_before = compute_metrics(result_before.predictions)

        df_with_future = self._add_future_races(df, n_future=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_after = run_window(df_with_future, w1)

        metrics_after = compute_metrics(result_after.predictions)

        for key in metrics_before:
            v_before = metrics_before[key]
            v_after = metrics_after[key]
            if v_before is None and v_after is None:
                continue
            self.assertIsNotNone(v_after, f"metric '{key}' became None after adding future data")
            self.assertIsNotNone(v_before, f"metric '{key}' was None before (test setup issue)")
            self.assertAlmostEqual(
                v_before, v_after, places=10,
                msg=(
                    f"metric '{key}' changed from {v_before} to {v_after} "
                    "after adding future races — independence violation!"
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
