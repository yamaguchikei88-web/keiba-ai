"""
Evaluation metrics for keiba-ai win probability models.

Metric priority (most important first — matches config/promotion_rules.py):
  1. LogLoss     -- probability calibration quality (lower is better)
  2. Brier Score -- mean squared probability error (lower is better)
  3. ECE         -- Expected Calibration Error (lower is better)
  4. ROC-AUC    -- discrimination ability (higher is better)
  5. Top1 Acc   -- race-level: highest-prob horse is the winner (higher is better)
  6. Top3 Acc   -- race-level: winner is in the top-3 predictions (higher is better)
  7. ROI        -- reference only; strategy-dependent (higher is better)

This module is intentionally separate from BetStrategy so that metrics can be
computed for any strategy without changing the evaluation logic.

EV reference (for future PositiveEVStrategy evaluation):
  EV = p * odds - 1
  where p = AI win probability, odds = market win odds (payout per unit stake)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.promotion_rules import CANDIDATE_TO_VALIDATED, METRIC_PRIORITY


# ── ECE ───────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Splits predictions into n_bins confidence intervals and measures the
    weighted average absolute difference between mean confidence and accuracy.

    A perfectly calibrated model predicts p=0.30 and wins 30% of the time →
    ECE ≈ 0. A poorly calibrated model (e.g., always predicts 0.5) has high ECE.

    Args:
        y_true: Binary labels (0/1).
        y_pred: Predicted win probabilities in [0, 1].
        n_bins: Number of confidence bins (default: 10).

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_pred[mask].mean()
        ece += (count / n) * abs(acc - conf)

    return float(ece)


# ── Race-level metrics ─────────────────────────────────────────────────────

def _race_level_accuracy(predictions: pd.DataFrame, top_k: int) -> float:
    """Fraction of races where the actual winner appears in the top-k predicted.

    Args:
        predictions: horse-level DataFrame with race_id, y_true, predicted_rank.
        top_k:       k for top-k accuracy.

    Returns:
        Fraction of races where the winner is in the top-k predicted horses.
    """
    correct = 0
    total = 0
    for _, group in predictions.groupby("race_id"):
        winner_rows = group[group["y_true"] == 1]
        if winner_rows.empty:
            continue
        winner_rank = winner_rows["predicted_rank"].iloc[0]
        correct += int(winner_rank <= top_k)
        total += 1
    return correct / total if total > 0 else 0.0


# ── compute_metrics ────────────────────────────────────────────────────────

def compute_metrics(
    predictions: pd.DataFrame,
    strategy=None,
) -> dict[str, float | None]:
    """Compute all evaluation metrics from horse-level predictions.

    Args:
        predictions: horse-level DataFrame with columns:
                     race_id, y_true (0/1), y_pred (probability),
                     predicted_rank (int, 1=best), odds (float, may be NaN).
        strategy:    BetStrategy instance for ROI computation.
                     If None, uses MaxProbStrategy as default.

    Returns:
        Dictionary of metric_name → value. ROI is None if odds unavailable.

    Metric order follows config/promotion_rules.METRIC_PRIORITY.
    LogLoss/Brier/ECE are the primary gates; ROI is reference only.
    """
    y_true = predictions["y_true"].values.astype(float)
    y_pred = predictions["y_pred"].values.astype(float)

    # Clip to avoid log(0) in log_loss
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    metrics: dict[str, float | None] = {}

    # 1. LogLoss (primary probability quality gate)
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_pred_clipped))
    except Exception:
        metrics["log_loss"] = None

    # 2. Brier Score
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_clipped))
    except Exception:
        metrics["brier_score"] = None

    # 3. ECE (calibration)
    try:
        metrics["ece"] = compute_ece(y_true, y_pred_clipped)
    except Exception:
        metrics["ece"] = None

    # 4. ROC-AUC
    try:
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_clipped))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    # 5. Top1 Accuracy (race-level)
    try:
        metrics["top1_acc"] = _race_level_accuracy(predictions, top_k=1)
    except Exception:
        metrics["top1_acc"] = None

    # 6. Top3 Accuracy (race-level)
    try:
        metrics["top3_acc"] = _race_level_accuracy(predictions, top_k=3)
    except Exception:
        metrics["top3_acc"] = None

    # 7. ROI (reference only — strategy-dependent)
    try:
        if strategy is None:
            from ml.backtest import MaxProbStrategy
            strategy = MaxProbStrategy()
        metrics["roi"] = strategy.compute_roi(predictions)
    except Exception:
        metrics["roi"] = None

    return metrics


# ── check_promotion_eligibility ────────────────────────────────────────────

def check_promotion_eligibility(
    metrics: dict[str, float | None],
    n_val_races: int,
    rules: dict | None = None,
) -> tuple[bool, list[str]]:
    """Check whether a model meets the candidate → validated promotion thresholds.

    Args:
        metrics:      Output of compute_metrics().
        n_val_races:  Number of validation races (statistical floor check).
        rules:        Override for CANDIDATE_TO_VALIDATED (used in tests).

    Returns:
        (eligible: bool, reasons: list[str])
        reasons is empty when eligible=True, or lists each failing condition.

    Note:
        validated → production always requires human approval.
        This function never promotes to production.
    """
    if rules is None:
        rules = CANDIDATE_TO_VALIDATED

    reasons: list[str] = []

    # Statistical reliability floor
    min_races = rules.get("min_val_races", 20)
    if n_val_races < min_races:
        reasons.append(
            f"val races ({n_val_races}) < min_val_races ({min_races}) — "
            "統計的信頼性不足"
        )

    for metric, constraint in rules.items():
        if metric == "min_val_races" or not isinstance(constraint, dict):
            continue
        value = metrics.get(metric)
        if value is None:
            reasons.append(f"{metric}: 計算不能 (None)")
            continue
        if "max" in constraint and value > constraint["max"]:
            reasons.append(
                f"{metric}: {value:.4f} > max={constraint['max']} (不合格)"
            )
        if "min" in constraint and value < constraint["min"]:
            reasons.append(
                f"{metric}: {value:.4f} < min={constraint['min']} (不合格)"
            )

    return len(reasons) == 0, reasons


# ── aggregate_window_metrics ───────────────────────────────────────────────

def aggregate_window_metrics(
    window_metrics: list[dict[str, float | None]],
) -> dict[str, float | None]:
    """Compute mean metrics across multiple walk-forward windows.

    None values are excluded from the mean. If all windows have None for a
    metric, the aggregate is also None.
    """
    if not window_metrics:
        return {}

    all_keys = set().union(*[m.keys() for m in window_metrics])
    aggregated: dict[str, float | None] = {}

    for key in all_keys:
        values = [m[key] for m in window_metrics if m.get(key) is not None]
        aggregated[key] = float(np.mean(values)) if values else None

    return aggregated


# ── evaluate_backtest_results ──────────────────────────────────────────────

def evaluate_backtest_results(
    window_results: list,  # list[WindowResult] from backtest.py
    strategy=None,
) -> dict[str, Any]:
    """Evaluate all walk-forward windows and return a summary.

    Args:
        window_results: list of WindowResult from backtest.run_backtest().
        strategy:       BetStrategy for ROI. Defaults to MaxProbStrategy.

    Returns:
        {
          "window_metrics": [dict per window],
          "aggregated":     dict (mean across windows),
          "n_windows":      int,
          "total_val_races": int,
          "total_val_rows":  int,
        }
    """
    window_metrics_list = []
    total_val_races = 0
    total_val_rows = 0

    for wr in window_results:
        m = compute_metrics(wr.predictions, strategy=strategy)
        wr.metrics.update(m)   # populate WindowResult.metrics in-place
        window_metrics_list.append(m)
        total_val_races += wr.window.n_val_races
        total_val_rows += wr.n_val_rows

    aggregated = aggregate_window_metrics(window_metrics_list)

    return {
        "window_metrics": window_metrics_list,
        "aggregated": aggregated,
        "n_windows": len(window_results),
        "total_val_races": total_val_races,
        "total_val_rows": total_val_rows,
    }


# ── format_metrics_report ──────────────────────────────────────────────────

def format_metrics_report(
    eval_summary: dict[str, Any],
    model_version: str = "unknown",
    include_reliability_warning: bool = True,
) -> str:
    """Format evaluation summary as a human-readable report."""
    lines = [
        f"=== モデル評価レポート: {model_version} ===",
        "",
    ]

    if include_reliability_warning:
        lines += [
            "[警告] 96レースでは統計的信頼性が低く、Phase 4 はバックテスト基盤の動作確認が目的です。",
            "この結果を「モデルの本番精度」として扱わないでください。",
            "2019〜2025等の大量データ投入後に再実行することを推奨します。",
            "",
        ]

    agg = eval_summary.get("aggregated", {})
    lines += [
        f"ウィンドウ数:   {eval_summary.get('n_windows')}",
        f"val レース数:   {eval_summary.get('total_val_races')} (合計)",
        f"val 行数:       {eval_summary.get('total_val_rows')} (合計)",
        "",
        "--- 集計指標 (全ウィンドウ平均) ---",
    ]

    label_map = {
        "log_loss":    "LogLoss    (↓ best)",
        "brier_score": "Brier Score(↓ best)",
        "ece":         "ECE        (↓ best)",
        "roc_auc":     "ROC-AUC   (↑ best)",
        "top1_acc":    "Top1 Acc  (↑ best)",
        "top3_acc":    "Top3 Acc  (↑ best)",
        "roi":         "ROI       (↑ best, 参考)",
    }

    for metric, _ in METRIC_PRIORITY:
        val = agg.get(metric)
        label = label_map.get(metric, metric)
        if val is None:
            lines.append(f"  {label}: N/A")
        else:
            lines.append(f"  {label}: {val:.4f}")

    # Per-window breakdown
    lines += ["", "--- ウィンドウ別指標 ---"]
    for i, wm in enumerate(eval_summary.get("window_metrics", []), 1):
        lines.append(f"  [Window {i}]")
        for metric, _ in METRIC_PRIORITY:
            val = wm.get(metric)
            label = label_map.get(metric, metric)
            if val is None:
                lines.append(f"    {label}: N/A")
            else:
                lines.append(f"    {label}: {val:.4f}")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import logging
    import sqlite3
    from project_paths import DB_PATH
    from ml.backtest import run_backtest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate backtest results")
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--min-train-races", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_backtest(n_windows=args.windows, min_train_races=args.min_train_races)
    summary = evaluate_backtest_results(results)

    eligible, reasons = check_promotion_eligibility(
        summary["aggregated"],
        n_val_races=summary["total_val_races"] // summary["n_windows"],
    )

    if args.json:
        output = {
            "model_version": args.model_version,
            "eligible_for_validated": eligible,
            "reasons": reasons,
            **summary,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_metrics_report(summary, model_version=args.model_version))
        print()
        if eligible:
            print("昇格判定: candidate → validated 昇格条件を満たしています。")
            print("(validated → production は人間の承認が必要です)")
        else:
            print("昇格判定: 昇格条件を満たしていません。")
            for r in reasons:
                print(f"  × {r}")
