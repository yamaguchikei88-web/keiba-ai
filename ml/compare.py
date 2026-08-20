"""
Model comparison and promotion gate for keiba-ai.

compare.py ranks candidate models using METRIC_PRIORITY and gates
candidate → validated promotion. It never promotes to production.

Promotion flow:
  candidate  --[compare.py auto]--> validated  (when all thresholds pass)
  validated  --[human: registry.approve_production()]--> production

Design principle:
  A model is "better" only if it improves on the most important metrics first.
  ROI or Top1Acc alone are NEVER sufficient to declare a model better.
  LogLoss, Brier, ECE must improve or hold; ROC-AUC must stay above floor.

  This prevents the common failure mode where a model that ignores probability
  calibration (e.g., always outputs 0.1 for every horse) accidentally wins on
  hit-rate metrics.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.promotion_rules import (
    CANDIDATE_TO_VALIDATED,
    METRIC_PRIORITY,
    PRODUCTION_REQUIRES_HUMAN_APPROVAL,
)
from ml.evaluate import check_promotion_eligibility


# ── ModelSummary ───────────────────────────────────────────────────────────

@dataclass
class ModelSummary:
    """Aggregated backtest result for one model version.

    All metrics here are averages across walk-forward windows.
    Include enough fields so future users can trace why a model was promoted.
    """
    model_version: str
    feature_set_version: str
    training_cutoff_date: str            # last date in train period
    n_windows: int
    total_val_races: int
    total_val_rows: int
    metrics: dict[str, float | None]     # aggregated across windows
    window_metrics: list[dict]           # per-window breakdown
    status: str = "candidate"            # candidate | validated
    notes: str = ""


# ── compare_models ─────────────────────────────────────────────────────────

def compare_models(
    summaries: list[ModelSummary],
    rules: dict | None = None,
) -> pd.DataFrame:
    """Rank model summaries by METRIC_PRIORITY.

    Returns a DataFrame with one row per model, sorted best-first.
    The ranking uses the ordered metric list: a model wins over another
    if it is better on the first metric where they differ (by more than
    the tolerance). This mirrors the priority defined in promotion_rules.py.

    Args:
        summaries:  List of ModelSummary to compare.
        rules:      Override promotion rules (for testing).

    Returns:
        DataFrame with columns: model_version, status, eligible, all metrics,
        plus a 'rank' column (1 = best).
    """
    if rules is None:
        rules = CANDIDATE_TO_VALIDATED

    rows = []
    for s in summaries:
        eligible, reasons = check_promotion_eligibility(
            s.metrics,
            n_val_races=s.total_val_races // max(s.n_windows, 1),
            rules=rules,
        )
        row: dict[str, Any] = {
            "model_version": s.model_version,
            "feature_set_version": s.feature_set_version,
            "status": s.status,
            "eligible": eligible,
            "ineligible_reasons": "; ".join(reasons) if reasons else "",
        }
        for metric, _ in METRIC_PRIORITY:
            row[metric] = s.metrics.get(metric)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by METRIC_PRIORITY (ascending for lower_is_better, descending for higher)
    sort_keys = []
    sort_ascending = []
    for metric, direction in METRIC_PRIORITY:
        if metric not in df.columns:
            continue
        sort_keys.append(metric)
        sort_ascending.append(direction == "lower_is_better")

    if sort_keys:
        # NaN values go last in ranking
        df = df.sort_values(
            by=sort_keys,
            ascending=sort_ascending,
            na_position="last",
        )

    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── promote_to_validated ───────────────────────────────────────────────────

def promote_to_validated(
    summary: ModelSummary,
    registry=None,
    rules: dict | None = None,
) -> tuple[bool, list[str]]:
    """Automatically promote a candidate model to validated if it passes thresholds.

    This function ONLY promotes candidate → validated.
    validated → production ALWAYS requires human approval via registry.approve_production().

    Args:
        summary:   ModelSummary from evaluate_backtest_results.
        registry:  ResearchRegistry instance. If provided, updates model status in DB.
        rules:     Override promotion rules (for testing).

    Returns:
        (promoted: bool, reasons: list[str])
        reasons is empty if promoted=True, otherwise lists failing conditions.
    """
    assert PRODUCTION_REQUIRES_HUMAN_APPROVAL, (
        "Production promotion requires human approval. This should never be False."
    )

    if summary.status != "candidate":
        return False, [f"モデル '{summary.model_version}' は candidate ではありません (status={summary.status})"]

    eligible, reasons = check_promotion_eligibility(
        summary.metrics,
        n_val_races=summary.total_val_races // max(summary.n_windows, 1),
        rules=rules,
    )

    if not eligible:
        return False, reasons

    # Update status in memory
    summary.status = "validated"

    # Persist to registry if provided
    if registry is not None:
        try:
            registry.record_metrics(
                scope_type="model",
                scope_id=summary.model_version,
                metrics={k: v for k, v in summary.metrics.items() if v is not None},
            )
        except Exception as e:
            # Metric recording failure does not block promotion
            print(f"[WARN] registry.record_metrics failed: {e}")

    return True, []


# ── format_comparison_report ───────────────────────────────────────────────

def format_comparison_report(
    ranking_df: pd.DataFrame,
    summaries: list[ModelSummary],
) -> str:
    """Format model comparison as a human-readable report."""
    lines = [
        "=== モデル比較レポート ===",
        "",
        "[指標評価の優先順位]",
    ]
    for rank_i, (m, d) in enumerate(METRIC_PRIORITY, 1):
        arrow = "↓ best" if d == "lower_is_better" else "↑ best"
        suffix = " ← 主要確率品質指標" if rank_i <= 3 else (" ← 参考値のみ" if m == "roi" else "")
        lines.append(f"  {rank_i}. {m} ({arrow}){suffix}")

    lines += [
        "",
        "[注意] LogLoss/Brier/ECEの同時改善なしにROIやTop1Accのみ改善したモデルは採用しないこと。",
        "",
        "--- ランキング ---",
    ]

    metric_cols = [m for m, _ in METRIC_PRIORITY if m in ranking_df.columns]

    for _, row in ranking_df.iterrows():
        status_tag = "[eligible]" if row["eligible"] else "[NG]"
        lines.append(f"\n  #{int(row['rank'])} {row['model_version']} {status_tag} (status={row['status']})")
        for m in metric_cols:
            val = row[m]
            _, direction = next(d for name, d in METRIC_PRIORITY if name == m)
            arrow = "↓" if direction == "lower_is_better" else "↑"
            lines.append(f"    {arrow} {m}: {val:.4f}" if val is not None else f"    - {m}: N/A")
        if row["ineligible_reasons"]:
            lines.append(f"    × 不合格理由: {row['ineligible_reasons']}")

    lines += [
        "",
        "--- 昇格サマリー ---",
        "(candidate → validated): 全条件を満たしたモデルのみ自動昇格",
        "(validated → production): 人間の approve_production() 承認が必要",
    ]

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging
    import sqlite3
    from project_paths import DB_PATH, REGISTRY_DB_PATH
    from ml.backtest import run_backtest
    from ml.evaluate import evaluate_backtest_results
    from config.promotion_rules import CANDIDATE_TO_VALIDATED

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compare and promote models")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--feature-set-version", default="v1")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--min-train-races", type=int, default=40)
    parser.add_argument("--approve", metavar="MODEL_VERSION",
                        help="Promote a validated model to production (human approval)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Approve mode: validated → production (human initiated)
    if args.approve:
        from registry.store import ResearchRegistry
        reg = ResearchRegistry(REGISTRY_DB_PATH)
        print(f"'{args.approve}' を production に昇格します。")
        print("この操作は取り消せません。続行しますか？ [y/N]", end=" ")
        answer = input().strip().lower()
        if answer == "y":
            reg.approve_production(args.approve)
            print(f"'{args.approve}' → production 昇格完了。")
        else:
            print("キャンセルしました。")
        sys.exit(0)

    # Compare mode: run backtest, evaluate, rank
    results = run_backtest(n_windows=args.windows, min_train_races=args.min_train_races)
    eval_summary = evaluate_backtest_results(results)

    # Read training_cutoff_date from model_meta.json if available
    meta_path = PROJECT_ROOT / "models" / "model_meta.json"
    training_cutoff = "unknown"
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text())
        training_cutoff = meta.get("training_cutoff_date", "unknown")

    summary = ModelSummary(
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        training_cutoff_date=training_cutoff,
        n_windows=eval_summary["n_windows"],
        total_val_races=eval_summary["total_val_races"],
        total_val_rows=eval_summary["total_val_rows"],
        metrics=eval_summary["aggregated"],
        window_metrics=eval_summary["window_metrics"],
    )

    ranking_df = compare_models([summary])
    promoted, reasons = promote_to_validated(summary)

    if args.json:
        output = {
            "model_version": args.model_version,
            "promoted_to_validated": promoted,
            "reasons": reasons,
            "metrics": eval_summary["aggregated"],
            "ranking": ranking_df.to_dict(orient="records"),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_comparison_report(ranking_df, [summary]))
        print()
        if promoted:
            print(f"昇格: '{args.model_version}' candidate → validated")
            print("(production昇格は --approve <model_version> で人間が実行してください)")
        else:
            print(f"昇格不可: '{args.model_version}'")
            for r in reasons:
                print(f"  × {r}")
