"""
Walk-forward backtesting for keiba-ai.

Architecture (現在 Phase 4):
  Model
    → horse-level probabilities (y_pred per horse per race)
      → WindowResult (horse-level predictions preserved)
        → evaluate.py computes metrics (LogLoss, Brier, ECE, AUC, Top1/3, ROI)
          → compare.py ranks models using METRIC_PRIORITY
            → promotion_rules.py gates candidate → validated
              → registry records all decisions (human approves validated → production)

将来のレイヤー構造 (Phase 6〜9 設計要件):
  Model
    → PredictionStrategy          # Phase 6: 印・自信度・評価理由を生成
        → 印 (◎ ○ ▲ ☆ △)         # 馬ごとの評価ランク
        → 自信度 (S/A/B/C)         # レース単位の信頼度
        → 評価理由                  # 馬ごとのコメント (optional)
    → BetStrategy                 # Phase 7: 買い目選択 (現在実装済み、multi-horse 拡張予定)
        → 単勝 / 馬連 / ワイド / 三連複
    → BankrollAllocator           # Phase 7: 予算別資金配分
        → 1,000円 / 3,000円 / 5,000円 / 10,000円
    → API                         # Phase 9
    → Web                         # Phase 9

Web画面構成 (Phase 9 設計要件):
  1. 予想検索画面  : 開催日 × 競馬場 × レース番号 → 予想実行
  2. 自信度一覧画面: レースごとの自信度 (S/A/B/C) と印・本命馬・買い目概要
                    S/A/B/C フィルタ / 「自信度Sだけ表示」
  3. レース詳細画面: 各馬の印・AI確率・評価理由 / 自信度 / 推奨買い目 / 予算別資金配分

BetStrategy 設計方針:
  Model outputs probabilities → BetStrategy selects bets → ROI computed

  Current: MaxProbStrategy (bet 1 unit on highest-prob horse)
  Future (add without changing this module):
    - PositiveEVStrategy:   bet when EV = p * prediction_time_odds - 1 > 0
    - MaxEVStrategy:        bet the horse with highest EV
    - ThresholdEVStrategy:  bet when EV > threshold
    - KellyStrategy:        f* = (p * odds - 1) / (odds - 1)
    - MultiHorseStrategy:   multiple horses per race (馬連・ワイド・三連複)

horse-level predictions are always preserved in WindowResult so that
any future strategy can be evaluated retroactively without re-running
the model.

Phase 4 実装範囲:
  - walk-forward バックテスト基盤 (このファイル)
  - 評価指標 (evaluate.py): LogLoss / Brier / ECE / AUC / Top1/3 / ROI
  - モデル比較・昇格ゲート (compare.py)
  - レジストリ (registry/store.py)
  - CI 互換 CLI (pipeline.py)
  Phase 6〜9 の印・自信度・買い目・Web は未実装。
  WindowResult.predictions には将来のWeb表示に必要なレース情報を保持済み。
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DB_PATH, MODEL_DIR, model_path

# Import from ml.features using the ml-subdir path pattern
_ML_DIR = Path(__file__).parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))
from features import load_raw_data, build_horse_stats, encode_features, FEATURE_COLS

# build_stats_cache is defined in train.py but we reproduce it here to avoid
# the circular import chain caused by train.py's own sys.path manipulation.
# Keeping it in sync with train.py is verified by the test suite.
def build_stats_cache(df: pd.DataFrame, cutoff_date=None) -> dict:
    """Build horse/jockey/trainer/father_heavy stats cache with temporal cutoff.

    Reproduces ml/train.py:build_stats_cache. Both must stay in sync.
    cutoff_date: include only races on or before this date (None = all data).
    """
    if cutoff_date is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff_date)]
    df = df.copy()
    df["is_win"] = (df["finish_pos"] == 1).astype(int)
    df["is_top3"] = (df["finish_pos"] <= 3).astype(int)

    cache: dict = {"horse": {}, "jockey": {}, "trainer": {}, "father_heavy": {}}

    for horse_id, g in df.groupby("horse_id"):
        if not horse_id:
            continue
        cache["horse"][horse_id] = {
            "race_count": len(g),
            "win_rate": round(g["is_win"].mean(), 4),
            "top3_rate": round(g["is_top3"].mean(), 4),
            "avg_finish": round(g["finish_pos"].mean(), 2),
            "recent_avg": round(g.tail(3)["finish_pos"].mean(), 2),
        }
        for (lo, hi), label in [((1000,1400),"s"),((1400,1800),"m"),((1800,2400),"l"),((2400,4000),"xl")]:
            sub = g[(g["distance"] >= lo) & (g["distance"] < hi)]
            cache["horse"][horse_id][f"dist_{label}_win"] = round(sub["is_win"].mean(), 4) if len(sub) > 0 else 0.0
        for cond in ["良", "稍重", "重", "不良"]:
            sub = g[g["track_condition"] == cond]
            cache["horse"][horse_id][f"cond_{cond}_win"] = round(sub["is_win"].mean(), 4) if len(sub) > 0 else 0.0

    for jockey, g in df.groupby("jockey_name"):
        if not jockey:
            continue
        cache["jockey"][jockey] = {
            "win_rate": round(g["is_win"].mean(), 4),
            "top3_rate": round(g["is_top3"].mean(), 4),
        }

    for trainer, g in df.groupby("trainer_name"):
        if not trainer:
            continue
        cache["trainer"][trainer] = {"win_rate": round(g["is_win"].mean(), 4)}

    if "father" in df.columns:
        heavy = df[df["track_condition"].isin(["重", "不良"])]
        for father, g in heavy.groupby("father"):
            if not father:
                continue
            cache["father_heavy"][father] = round(g["is_win"].mean(), 4)

    return cache


# LightGBM parameters (same as ml/train.py — keep in sync)
_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}
_N_ROUNDS = 5000
_EARLY_STOPPING = 100

logger = logging.getLogger(__name__)

# Minimum val races before issuing a statistical reliability warning.
# 96-race datasets will always trigger this — that is expected and correct.
_MIN_RELIABLE_VAL_RACES = 30


# ── WalkForwardWindow ──────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """One walk-forward time window.

    Invariant (enforced by make_walk_forward_windows):
      train_end < val_start  (strict: no same-day overlap)

    stats_cache cutoff rule:
      build_stats_cache(df, cutoff_date=window.stats_cutoff)
      This ensures no val-period race results contaminate the historical
      statistics used as features during validation.
    """
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # last training day (inclusive)
    val_start: pd.Timestamp      # first validation day (inclusive)
    val_end: pd.Timestamp        # last validation day (inclusive)
    n_train_races: int
    n_val_races: int

    @property
    def stats_cutoff(self) -> pd.Timestamp:
        """build_stats_cache cutoff: the day before val_start.

        Using val_start itself would risk including same-day race results in
        the stats cache before those races have been run (data leakage).
        """
        return self.val_start - pd.Timedelta(days=1)

    def __str__(self) -> str:
        return (
            f"Window {self.window_id}: "
            f"train [{self.train_start.date()} – {self.train_end.date()}] ({self.n_train_races} races) | "
            f"val   [{self.val_start.date()} – {self.val_end.date()}] ({self.n_val_races} races)"
        )


# ── WindowResult ───────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Backtest result for one walk-forward window.

    predictions DataFrame schema (horse-level, one row per horse per race):
      race_id        str    (レースID)
      date           date   (開催日)
      racecourse     str    (競馬場名 — Web表示・自信度一覧フィルタで使用)
      race_num       int    (レース番号 — Web表示・レース特定で使用)
      distance       int    (距離 m)
      surface        str    (芝 / ダ)
      num_horses     int    (出走頭数 — 確率正規化・印付けで使用)
      horse_num      int    (馬番)
      finish_pos     int    (実際の着順)
      horse_id       str
      horse_name     str
      y_true         int    (1 if finish_pos == 1, else 0)
      y_pred         float  (model's win probability estimate)
      odds           float  (win odds recorded in DB — see note below)
      predicted_rank int    (1 = highest probability in the race)

    将来追加予定 (Phase 6〜8 で拡張):
      place_prob          float  (複勝確率 — 馬連・ワイド・三連複の買い目生成に必要)
      prediction_time_odds float (レース前オッズ — EV = y_pred × prediction_time_odds - 1)
      ev                  float  (期待値 — prediction_time_odds × y_pred - 1)
      印                  str    (◎ ○ ▲ ☆ △ — PredictionStrategy が付与)
      confidence          str    (S/A/B/C — レース単位、PredictionStrategy が付与)
      reason              str    (評価理由 — 各馬のコメント、optional)

    Note on `odds` (= final_odds):
      The `odds` column comes from the race RESULT page on db.netkeiba.com
      and represents the FINAL confirmed single-win odds after the race.
      In Japan, this is very close to the last pre-race market odds, but it
      is conceptually a POST-RACE figure ("final_odds"), not a pre-race figure
      ("prediction_time_odds").

      ROI computed from this column is therefore a POST-HOC evaluation:
        "What ROI would we have achieved if we had been able to purchase at
         approximately these odds?"

      Future design: separate prediction_time_odds (available before race,
      used in EV calculation) from final_odds (available only after race,
      used for ROI settlement). EV = predicted_prob × prediction_time_odds - 1.

    This schema is preserved as-is so that any BetStrategy or PredictionStrategy
    can be applied retroactively without re-running the model.
    """
    window: WalkForwardWindow
    predictions: pd.DataFrame    # horse-level, schema above
    metrics: dict[str, float]    # computed by evaluate.py
    n_train_rows: int
    n_val_rows: int

    @property
    def window_id(self) -> int:
        return self.window.window_id


# ── BetStrategy (abstraction layer) ────────────────────────────────────────

class BetStrategy(ABC):
    """Abstract base for bet selection strategies.

    Model outputs horse-level probabilities.
    BetStrategy decides WHICH horses to bet on and HOW MUCH.

    Current implementation: MaxProbStrategy
    Future: PositiveEVStrategy, KellyStrategy, MultiHorseStrategy, etc.

    EV reference formula (for future implementations):
      EV = p * odds - 1
      where p = AI win probability, odds = market win odds (payout per unit)
      Bet when EV > 0 (positive expected value).
    """

    @abstractmethod
    def select_bets(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Select horses to bet on for a single race.

        Args:
            race_df: Rows for one race with columns:
                     horse_id, y_pred, odds (may be NaN), and others.

        Returns:
            Subset of race_df with an added 'bet_amount' column (float).
            Return empty DataFrame to skip the race.
        """
        ...

    def compute_roi(self, predictions: pd.DataFrame) -> float | None:
        """Compute ROI across all races using this strategy.

        The `odds` column in predictions is the final confirmed odds recorded
        in the race result DB — a POST-HOC figure (≈ final_odds). ROI here
        is a post-hoc evaluation: "what ROI if we could buy at these odds?"

        For future EV-based strategies, use prediction_time_odds (pre-race
        market odds) for the buy/no-buy decision, and final_odds only for
        settlement payout. They are currently the same column in the DB.

        Returns None if odds are unavailable for all races.
        """
        if "odds" not in predictions.columns or predictions["odds"].isna().all():
            return None

        total_stake = 0.0
        total_payout = 0.0

        for _, race_df in predictions.groupby("race_id"):
            bets = self.select_bets(race_df.copy())
            if bets.empty:
                continue
            for _, bet_row in bets.iterrows():
                amount = float(bet_row["bet_amount"])
                total_stake += amount
                # Payout: odds × amount if horse won, else 0
                if bet_row["y_true"] == 1 and pd.notna(bet_row.get("odds")):
                    total_payout += float(bet_row["odds"]) * amount

        if total_stake == 0:
            return None
        return (total_payout - total_stake) / total_stake


class MaxProbStrategy(BetStrategy):
    """Bet 1 unit on the horse with the highest predicted win probability.

    This is the simplest baseline strategy. It ignores market odds entirely,
    which means it does not exploit EV. Use as a reference benchmark only.

    To exploit EV, implement PositiveEVStrategy:
      EV = p * odds - 1 > 0  →  bet
    """

    def select_bets(self, race_df: pd.DataFrame) -> pd.DataFrame:
        if race_df.empty:
            return race_df
        best_idx = race_df["y_pred"].idxmax()
        result = race_df.loc[[best_idx]].copy()
        result["bet_amount"] = 1.0
        return result


# ── make_walk_forward_windows ──────────────────────────────────────────────

def make_walk_forward_windows(
    df: pd.DataFrame,
    n_windows: int = 2,
    min_train_races: int = 40,
) -> list[WalkForwardWindow]:
    """Generate walk-forward windows from race data.

    Splits on racing-day boundaries (never within a single calendar day) so
    that all races from one day belong entirely to train or entirely to val.

    Each window i:
      train: races where date < window.val_start
      val:   races where window.val_start <= date <= window.val_end
      stats_cache cutoff = window.stats_cutoff (val_start - 1 day)

    Args:
        df:               DataFrame with 'race_id' and 'date' columns
                          (typically the output of load_raw_data).
        n_windows:        Number of validation windows.
                          2 is recommended for 96-race datasets.
        min_train_races:  Minimum race count that must exist *before* the
                          first validation window begins.

    Returns:
        List of WalkForwardWindow in chronological order.

    Raises:
        ValueError: if the data is too small to satisfy the constraints.

    Warns:
        UserWarning for each window whose n_val_races < 30.
        96-race datasets will always trigger this — it is expected.
        Phase 4 purpose: validate the backtest infrastructure.
        Re-run with 2019-2025 data for statistically meaningful results.
    """
    if "race_id" not in df.columns or "date" not in df.columns:
        raise ValueError("df must contain 'race_id' and 'date' columns.")

    # One row per racing day: date → n_races (unique race_id count)
    by_day = (
        df.groupby("date")["race_id"]
        .nunique()
        .sort_index()
        .reset_index(name="n_races")
    )
    by_day["cum"] = by_day["n_races"].cumsum()
    n_days = len(by_day)

    # Find the first day index i such that cumulative races *before* day[i] >= min_train_races.
    # "before day[i]" = by_day.iloc[i-1]["cum"]  (0 if i==0, excluded since i starts from 1).
    first_val_idx: int | None = None
    for i in range(1, n_days):
        if int(by_day.iloc[i - 1]["cum"]) >= min_train_races:
            first_val_idx = i
            break

    if first_val_idx is None:
        total = int(by_day["n_races"].sum())
        raise ValueError(
            f"min_train_races={min_train_races} を満たせません。"
            f"全{total}レース / {n_days}開催日。min_train_races を下げてください。"
        )

    val_day_count = n_days - first_val_idx
    if val_day_count < n_windows:
        raise ValueError(
            f"val期間の開催日数({val_day_count}) < n_windows({n_windows})。"
            f"n_windows を {val_day_count} 以下に下げてください。"
        )

    # Distribute val days across windows (last window absorbs any remainder)
    base_days, extra_days = divmod(val_day_count, n_windows)

    windows: list[WalkForwardWindow] = []
    cursor = first_val_idx  # index into by_day

    for i in range(n_windows):
        win_n_days = base_days + (1 if i < extra_days else 0)
        val_slice = by_day.iloc[cursor: cursor + win_n_days]

        val_start = pd.Timestamp(val_slice.iloc[0]["date"])
        val_end = pd.Timestamp(val_slice.iloc[-1]["date"])
        train_start = pd.Timestamp(by_day.iloc[0]["date"])
        train_end = pd.Timestamp(by_day.iloc[cursor - 1]["date"])

        n_train = int(by_day.iloc[cursor - 1]["cum"])
        n_val = int(val_slice["n_races"].sum())

        # Core invariant: no same-day or reversed splits
        if train_end >= val_start:
            raise AssertionError(
                f"Window {i + 1}: train_end({train_end.date()}) >= val_start({val_start.date()}). "
                "時系列の逆転が発生しています。バグレポートしてください。"
            )

        if n_val < _MIN_RELIABLE_VAL_RACES:
            warnings.warn(
                f"[Window {i + 1}] val期間は {n_val} レース / {win_n_days} 開催日のみ "
                f"（統計的信頼性の推奨: {_MIN_RELIABLE_VAL_RACES}レース以上）。\n"
                "Phase 4 の目的はバックテスト基盤の動作確認です。\n"
                "この結果を「モデルの本番精度」として扱わないでください。\n"
                "2019〜2025等の大量データ投入後に再実行することを推奨します。",
                UserWarning,
                stacklevel=2,
            )

        windows.append(WalkForwardWindow(
            window_id=i + 1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            n_train_races=n_train,
            n_val_races=n_val,
        ))
        cursor += win_n_days

    return windows


# ── run_window ─────────────────────────────────────────────────────────────

def _train_window_model(
    raw_df: pd.DataFrame,
    window: WalkForwardWindow,
) -> lgb.Booster:
    """Train a LightGBM model on the train portion of a walk-forward window.

    Uses build_horse_stats() with date < race_date (already time-safe).
    stats_cache cutoff = window.stats_cutoff to exclude val-period data.
    """
    train_df = raw_df[raw_df["date"] < window.val_start].copy()
    logger.info(
        f"Window {window.window_id}: 学習データ {len(train_df)}行 "
        f"({train_df['race_id'].nunique()}レース, 〜{window.train_end.date()})"
    )

    # build_horse_stats uses date < race_date internally (no extra cutoff needed)
    featured = build_horse_stats(train_df)
    featured = encode_features(featured)

    X = featured[FEATURE_COLS].fillna(-1)
    y = (featured["finish_pos"] == 1).astype(int)

    if len(X) == 0 or y.sum() == 0:
        raise ValueError(f"Window {window.window_id}: 学習データが空またはpositive sampleなし。")

    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)

    callbacks = [
        lgb.early_stopping(_EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=9999),  # suppress per-round output
    ]

    # Use a small val set (last 20% of training window) for early stopping only
    split = int(len(X) * 0.8)
    train_ds = lgb.Dataset(X.iloc[:split], label=y.iloc[:split], feature_name=FEATURE_COLS)
    es_ds = lgb.Dataset(X.iloc[split:], label=y.iloc[split:], reference=train_ds)

    model = lgb.train(
        _LGB_PARAMS,
        train_ds,
        num_boost_round=_N_ROUNDS,
        valid_sets=[train_ds, es_ds],
        valid_names=["train", "es_val"],
        callbacks=callbacks,
    )
    return model


def run_window(
    raw_df: pd.DataFrame,
    window: WalkForwardWindow,
    model: lgb.Booster | None = None,
) -> WindowResult:
    """Run one walk-forward window: optionally train a model, then predict on val.

    Walk-forward guarantee (why no future data leaks):
      1. Model training uses raw_df["date"] < window.val_start only.
      2. Val feature generation uses build_horse_stats(combined) where combined =
         pre_cutoff_df (date <= stats_cutoff) + val_raw. Inside build_horse_stats,
         each race's stats come from `past = df[df["date"] < race_date]` — never
         the race's own results or later dates.
      3. encode_features() is stateless (fixed maps + hashlib); val data in
         the input DataFrame does not affect the encoding of any other row.
      4. FEATURE_COLS excludes odds and popularity (unknown at prediction time).

    Note on val-period internal ordering:
      If val spans multiple days (e.g. 2024-02-10 and 2024-02-11), the features
      for 2024-02-11 races include the RESULTS of 2024-02-10 races as history.
      This mirrors real-world usage (the previous day's results are available),
      so it is not considered a leak. When val spans only 1 day, no intra-val
      information flows between races.

    Args:
        raw_df:  Full raw DataFrame from load_raw_data() (all dates).
        window:  The walk-forward window defining train/val date ranges.
        model:   Pre-trained model to use. If None, trains a fresh model on
                 the train portion of this window.

    Returns:
        WindowResult with horse-level predictions and empty metrics dict.
        metrics are populated by evaluate.py after this call.
    """
    logger.info(
        f"[Window {window.window_id}] "
        f"train_start={window.train_start.date()} "
        f"train_end={window.train_end.date()} "
        f"val_start={window.val_start.date()} "
        f"val_end={window.val_end.date()} "
        f"n_train_races={window.n_train_races} "
        f"n_val_races={window.n_val_races} "
        f"stats_cutoff={window.stats_cutoff.date()}"
    )

    val_raw = raw_df[
        (raw_df["date"] >= window.val_start) & (raw_df["date"] <= window.val_end)
    ].copy()

    if val_raw.empty:
        raise ValueError(f"Window {window.window_id}: val期間のデータが空です。")

    # Build horse stats for val rows, using only pre-cutoff history
    # build_horse_stats internally uses date < race_date, so pass the full
    # pre-cutoff subset (not just val_raw) so it has historical context.
    pre_cutoff_df = raw_df[raw_df["date"] <= window.stats_cutoff].copy()
    combined = pd.concat([pre_cutoff_df, val_raw], ignore_index=True)
    featured_combined = build_horse_stats(combined)
    featured_combined = encode_features(featured_combined)

    # Select only val rows from the featured output
    val_featured = featured_combined[
        (featured_combined["date"] >= window.val_start)
        & (featured_combined["date"] <= window.val_end)
    ].copy()

    X_val = val_featured[FEATURE_COLS].fillna(-1)
    y_val = (val_featured["finish_pos"] == 1).astype(int)

    n_val_rows = len(X_val)
    logger.info(
        f"Window {window.window_id}: val {n_val_rows}行 "
        f"({val_featured['race_id'].nunique()}レース)"
    )

    # --- Train model if not provided ---
    if model is None:
        model = _train_window_model(raw_df, window)

    train_featured_count = raw_df[raw_df["date"] < window.val_start]
    n_train_rows = len(train_featured_count)

    # --- Predict ---
    y_pred = model.predict(X_val)

    # --- Build predictions DataFrame (horse-level, schema documented in WindowResult) ---
    # Include race-level metadata for future Web display (自信度一覧・レース詳細画面).
    # Using .get() with fallback so synthetic test DataFrames without these columns still work.
    def _col(name, default):
        return val_featured[name] if name in val_featured.columns else pd.Series(default, index=val_featured.index)

    predictions = val_featured[["race_id", "date", "horse_num", "finish_pos"]].copy()
    predictions.insert(2, "racecourse", _col("racecourse", ""))
    predictions.insert(3, "race_num",   _col("race_num", -1))
    predictions.insert(4, "distance",   _col("distance", -1))
    predictions.insert(5, "surface",    _col("surface", ""))
    predictions.insert(6, "num_horses", _col("num_horses", -1))
    predictions["horse_id"] = val_featured.get("horse_id", pd.Series("", index=val_featured.index))
    predictions["horse_name"] = val_featured.get("horse_name", pd.Series("", index=val_featured.index))
    predictions["y_true"] = y_val.values
    predictions["y_pred"] = y_pred
    predictions["odds"] = val_featured.get("odds", pd.Series(np.nan, index=val_featured.index))

    # predicted_rank: within each race, rank by y_pred descending (1 = most likely winner)
    predictions["predicted_rank"] = (
        predictions.groupby("race_id")["y_pred"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    predictions = predictions.reset_index(drop=True)

    return WindowResult(
        window=window,
        predictions=predictions,
        metrics={},   # populated by evaluate.py
        n_train_rows=n_train_rows,
        n_val_rows=n_val_rows,
    )


# ── run_backtest ───────────────────────────────────────────────────────────

def run_backtest(
    n_windows: int = 2,
    min_train_races: int = 40,
    model: lgb.Booster | None = None,
    db_path: Path | str | None = None,
) -> list[WindowResult]:
    """Run a full walk-forward backtest.

    For each window:
      1. Generate window definition (make_walk_forward_windows)
      2. Train a model on the train period (or use the provided model)
      3. Build stats_cache with cutoff = window.stats_cutoff
      4. Predict on val period
      5. Return WindowResult list (metrics filled in by evaluate.py)

    Args:
        n_windows:        Number of walk-forward windows (2 for 96 races).
        min_train_races:  Minimum races before first val window.
        model:            Pre-trained model to evaluate. If None, trains
                          a fresh model per window.
        db_path:          Path to keiba.db. Defaults to project_paths.DB_PATH.

    Returns:
        List of WindowResult, one per walk-forward window.

    Note:
        96-race results have low statistical reliability.
        Phase 4 purpose: validate the backtest infrastructure.
        Re-run with 2019-2025 data for meaningful accuracy estimates.
    """
    _db = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(_db)
    raw_df = load_raw_data(conn)
    conn.close()

    logger.info(f"全データ: {len(raw_df)}行 / {raw_df['race_id'].nunique()}レース")

    windows = make_walk_forward_windows(raw_df, n_windows=n_windows, min_train_races=min_train_races)
    results: list[WindowResult] = []

    for window in windows:
        logger.info(f"=== {window} ===")
        result = run_window(raw_df, window, model=model)
        results.append(result)
        logger.info(
            f"Window {window.window_id} 完了: "
            f"val={result.n_val_rows}行, "
            f"positive={result.predictions['y_true'].sum()}件"
        )

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--min-train-races", type=int, default=40)
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    results = run_backtest(n_windows=args.windows, min_train_races=args.min_train_races)

    summary = []
    for r in results:
        summary.append({
            "window_id": r.window_id,
            "train": {"start": str(r.window.train_start.date()), "end": str(r.window.train_end.date()), "races": r.window.n_train_races},
            "val":   {"start": str(r.window.val_start.date()), "end": str(r.window.val_end.date()), "races": r.window.n_val_races},
            "n_val_rows": r.n_val_rows,
            "metrics": r.metrics,
        })

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for s in summary:
            print(f"\n[Window {s['window_id']}]")
            print(f"  train: {s['train']['start']} ~ {s['train']['end']} ({s['train']['races']} races)")
            print(f"  val:   {s['val']['start']} ~ {s['val']['end']} ({s['val']['races']} races)")
            print(f"  val rows: {s['n_val_rows']}")
            if s['metrics']:
                for k, v in s['metrics'].items():
                    print(f"  {k}: {v}")
