"""
End-to-end ML pipeline for keiba-ai.

Runs: backtest → evaluate → compare → (auto) candidate→validated → registry record

Usage:
  python -m ml.pipeline [options]

GitHub Actions compatible:
  --json          machine-readable JSON output (exit 0 on success, 1 on failure)
  --dry-run       evaluate only, no registry writes
  --model-version MODEL_VERSION
  --feature-set-version FEATURE_SET_VERSION
  --windows N
  --min-train-races N

The pipeline NEVER promotes to production automatically.
Production promotion requires human action via registry.approve_production().
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DB_PATH, MODEL_DIR, REGISTRY_DB_PATH, model_path
from config.promotion_rules import METRIC_PRIORITY, PRODUCTION_REQUIRES_HUMAN_APPROVAL

logger = logging.getLogger(__name__)


# ── _load_model_meta ───────────────────────────────────────────────────────

def _load_model_meta() -> dict:
    meta_path = model_path("model_meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


# ── _register_to_registry ──────────────────────────────────────────────────

def _register_to_registry(
    registry,
    model_version: str,
    feature_set_version: str,
    summary,
    eval_summary: dict,
    window_results: list,
) -> None:
    """Record model, experiments, and metrics in the registry.

    Tracked information:
      - model_version, feature_set_version, training_cutoff_date
      - backtest windows (as experiments)
      - aggregated and per-window metrics
      - model status (candidate or validated)

    This ensures 「なぜ現在のproductionモデルが採用されたのか」 is traceable.
    """
    meta = _load_model_meta()

    # 1. Ensure feature_set is registered
    try:
        from ml.features import FEATURE_COLS
        registry.register_feature_set(
            version=feature_set_version,
            description=f"Phase 4 feature set: {len(FEATURE_COLS)} columns, odds/popularity excluded",
            feature_names=FEATURE_COLS,
            code_reference="ml/features.py:FEATURE_COLS",
            status="active",
        )
        logger.info(f"feature_set {feature_set_version} registered")
    except Exception as e:
        if "UNIQUE constraint" in str(e) or "already exists" in str(e).lower():
            logger.debug(f"feature_set {feature_set_version} already registered")
        else:
            logger.warning(f"feature_set registration: {e}")

    # 2. Register model
    try:
        # Collect period info from windows
        all_train_starts = [str(wr.window.train_start.date()) for wr in window_results]
        all_val_ends = [str(wr.window.val_end.date()) for wr in window_results]

        registry.register_model(
            model_version=model_version,
            model_type="lightgbm_binary",
            feature_set_version=feature_set_version,
            model_path=str(model_path("keiba_lgbm.pkl")),
            parameters=meta.get("lgb_params", {}),
            status=summary.status,
            metrics=eval_summary["aggregated"],
            periods={
                "training_start": all_train_starts[0] if all_train_starts else "",
                "training_end": meta.get("training_cutoff_date", ""),
                "validation_start": min(all_train_starts) if all_train_starts else "",
                "validation_end": max(all_val_ends) if all_val_ends else "",
            },
            notes=(
                f"Phase 4 backtest: {eval_summary['n_windows']} windows, "
                f"{eval_summary['total_val_races']} val races. "
                "96-race dataset: low statistical reliability. "
                "Re-run with 2019-2025 data for meaningful accuracy estimates."
            ),
        )
        logger.info(f"model {model_version} registered (status={summary.status})")
    except Exception as e:
        if "UNIQUE constraint" in str(e) or "already exists" in str(e).lower():
            logger.warning(f"model {model_version} already in registry (skipped)")
        else:
            logger.warning(f"model registration failed: {e}")

    # 3. Record per-window experiments
    for wr in window_results:
        exp_id = f"bt_{model_version}_w{wr.window_id}"
        try:
            registry.record_experiment(
                experiment_id=exp_id,
                feature_set_version=feature_set_version,
                model_version=model_version,
                parameters={
                    "window_id": wr.window_id,
                    "n_windows": eval_summary["n_windows"],
                },
                status="completed",
                metrics=wr.metrics,
                periods={
                    "training_start": str(wr.window.train_start.date()),
                    "training_end": str(wr.window.train_end.date()),
                    "validation_start": str(wr.window.val_start.date()),
                    "validation_end": str(wr.window.val_end.date()),
                },
                conclusion=(
                    f"val={wr.window.n_val_races}races, "
                    f"top1={wr.metrics.get('top1_acc', 'N/A')}, "
                    f"log_loss={wr.metrics.get('log_loss', 'N/A')}"
                ),
            )
            logger.info(f"experiment {exp_id} recorded")
        except Exception as e:
            if "UNIQUE constraint" in str(e) or "already exists" in str(e).lower():
                logger.debug(f"experiment {exp_id} already recorded")
            else:
                logger.warning(f"experiment recording failed for {exp_id}: {e}")

    # 4. Record aggregated metrics
    try:
        agg_notnone = {k: v for k, v in eval_summary["aggregated"].items() if v is not None}
        registry.record_metrics(
            scope_type="model",
            scope_id=model_version,
            metrics=agg_notnone,
        )
        logger.info(f"metrics recorded for {model_version}: {list(agg_notnone.keys())}")
    except Exception as e:
        logger.warning(f"metrics recording failed: {e}")


# ── run_pipeline ───────────────────────────────────────────────────────────

def run_pipeline(
    model_version: str = "v1",
    feature_set_version: str = "v1",
    n_windows: int = 2,
    min_train_races: int = 40,
    dry_run: bool = False,
    registry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the full evaluation pipeline.

    Steps:
      1. Run walk-forward backtest (training + prediction per window)
      2. Compute metrics (LogLoss, Brier, ECE, AUC, Top1/3, ROI)
      3. Compare and check promotion eligibility
      4. Auto-promote candidate → validated if all thresholds pass
      5. Record to registry (unless dry_run=True)

    Returns a JSON-serializable result dict for GitHub Actions consumption.

    Note:
      96-race results have low statistical reliability.
      Phase 4 purpose: validate the backtest infrastructure.
      This function NEVER promotes to production.
    """
    from ml.backtest import run_backtest
    from ml.evaluate import evaluate_backtest_results, format_metrics_report
    from ml.compare import ModelSummary, promote_to_validated

    logger.info(f"Pipeline start: model={model_version}, windows={n_windows}")

    # Step 1: backtest
    window_results = run_backtest(n_windows=n_windows, min_train_races=min_train_races)

    # Step 2: evaluate
    eval_summary = evaluate_backtest_results(window_results)

    # Read meta
    meta = _load_model_meta()
    training_cutoff = meta.get("training_cutoff_date", "unknown")

    # Step 3: compare / promotion check
    summary = ModelSummary(
        model_version=model_version,
        feature_set_version=feature_set_version,
        training_cutoff_date=training_cutoff,
        n_windows=eval_summary["n_windows"],
        total_val_races=eval_summary["total_val_races"],
        total_val_rows=eval_summary["total_val_rows"],
        metrics=eval_summary["aggregated"],
        window_metrics=eval_summary["window_metrics"],
    )

    # Step 4: promote candidate → validated (if eligible)
    promoted, promotion_reasons = promote_to_validated(summary)

    result: dict[str, Any] = {
        "model_version": model_version,
        "feature_set_version": feature_set_version,
        "training_cutoff_date": training_cutoff,
        "n_windows": eval_summary["n_windows"],
        "total_val_races": eval_summary["total_val_races"],
        "total_val_rows": eval_summary["total_val_rows"],
        "promoted_to_validated": promoted,
        "model_status": summary.status,
        "promotion_reasons": promotion_reasons,
        "metrics": {
            k: (round(v, 6) if v is not None else None)
            for k, v in eval_summary["aggregated"].items()
        },
        "window_metrics": eval_summary["window_metrics"],
        "reliability_warning": (
            "96-race dataset: low statistical reliability. "
            "Phase 4 validates infrastructure only. "
            "Re-run with 2019-2025 data for meaningful accuracy estimates."
        ),
    }

    # Step 5: registry
    if not dry_run:
        try:
            from registry.store import ResearchRegistry, initialize_registry
            _reg_path = Path(registry_path) if registry_path else REGISTRY_DB_PATH
            initialize_registry(_reg_path)
            registry = ResearchRegistry(_reg_path)
            _register_to_registry(
                registry, model_version, feature_set_version,
                summary, eval_summary, window_results,
            )
            result["registry_path"] = str(_reg_path)
            logger.info(f"Registry updated: {_reg_path}")
        except Exception as e:
            logger.error(f"Registry update failed: {e}")
            result["registry_error"] = str(e)
    else:
        logger.info("dry_run=True: registry write skipped")
        result["registry_path"] = None

    logger.info(
        f"Pipeline complete: status={summary.status}, "
        f"promoted={promoted}, "
        f"log_loss={result['metrics'].get('log_loss')}"
    )
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="keiba-ai ML pipeline: backtest → evaluate → compare → registry"
    )
    parser.add_argument("--model-version", default="v1",
                        help="Model version string recorded in registry")
    parser.add_argument("--feature-set-version", default="v1",
                        help="Feature set version string")
    parser.add_argument("--windows", type=int, default=2,
                        help="Number of walk-forward windows")
    parser.add_argument("--min-train-races", type=int, default=40,
                        help="Minimum races before first val window")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip registry writes (evaluate only)")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output for GitHub Actions")
    args = parser.parse_args()

    result = run_pipeline(
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        n_windows=args.windows,
        min_train_races=args.min_train_races,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        sys.exit(0 if result["promoted_to_validated"] or result["model_status"] == "candidate" else 1)
    else:
        print()
        print("[keiba-ai Pipeline Result]")
        print(f"  model_version:       {result['model_version']}")
        print(f"  status:              {result['model_status']}")
        print(f"  promoted_to_valid:   {result['promoted_to_validated']}")
        print(f"  val_races (total):   {result['total_val_races']}")
        print()
        print("[指標 (全ウィンドウ平均)]")
        for metric, _ in METRIC_PRIORITY:
            val = result["metrics"].get(metric)
            label = f"{metric:<14}"
            print(f"  {label}: {val:.4f}" if val is not None else f"  {label}: N/A")
        print()
        print("[信頼性警告]")
        print(f"  {result['reliability_warning']}")
        if result.get("promotion_reasons"):
            print()
            print("[昇格不可の理由]")
            for r in result["promotion_reasons"]:
                print(f"  × {r}")
        if result.get("registry_path"):
            print()
            print(f"[Registry] {result['registry_path']}")
