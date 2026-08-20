"""
Model promotion rules and metric priority definitions.

Promotion flow:
  candidate  --[all thresholds pass automatically]--> validated
  validated  --[explicit human approve_production() call]--> production

Metric evaluation priority (1 = most important):
  1. LogLoss     -- probability calibration accuracy (lower is better)
  2. Brier Score -- mean squared probability error (lower is better)
  3. ECE         -- expected calibration error (lower is better)
  4. ROC-AUC     -- discrimination ability (higher is better)
  5. Top1 Acc    -- race-level win prediction rate (higher is better)
  6. Top3 Acc    -- race-level top3 inclusion rate (higher is better)
  7. ROI         -- reference only; never sole basis for promotion (higher is better)

WARNING: Do NOT promote a model that improves only ROC-AUC or ROI while
         degrading LogLoss / Brier / ECE. The primary goal of this project
         is accurate probability estimation ("真の勝率推定"), not hit-rate
         maximization. A model that says every horse has 10% win probability
         is not useful even if its top1 accuracy happens to be high.
"""

from __future__ import annotations

# ── candidate → validated: ALL conditions must pass simultaneously ─────────
#
# These thresholds are intentionally conservative for the 96-race dataset.
# When 2019-2025 data is loaded, revisit and tighten these values.
#
CANDIDATE_TO_VALIDATED: dict = {
    # Probability quality (the most important gates)
    "log_loss":    {"max": 0.35},    # primary calibration gate
    "brier_score": {"max": 0.075},   # squared error gate
    "ece":         {"max": 0.05},    # calibration reliability gate
    # Discrimination floor (necessary but not sufficient)
    "roc_auc":     {"min": 0.60},
    # Statistical reliability floor (skip promotion if too few val races)
    "min_val_races": 20,
}

# ── Model comparison priority ──────────────────────────────────────────────
#
# compare.py uses this ordering when ranking candidates.
# A model wins only if it is better on the HIGHER-PRIORITY metrics.
# Improving only roi or top1_acc is not enough.
#
METRIC_PRIORITY: list[tuple[str, str]] = [
    ("log_loss",    "lower_is_better"),
    ("brier_score", "lower_is_better"),
    ("ece",         "lower_is_better"),
    ("roc_auc",     "higher_is_better"),
    ("top1_acc",    "higher_is_better"),
    ("top3_acc",    "higher_is_better"),
    ("roi",         "higher_is_better"),   # reference only
]

# ── Production promotion guard ─────────────────────────────────────────────
#
# This constant documents the architectural constraint.
# Code must call registry.approve_production() explicitly; it is never called
# automatically by backtest, evaluate, or compare.
#
PRODUCTION_REQUIRES_HUMAN_APPROVAL: bool = True
