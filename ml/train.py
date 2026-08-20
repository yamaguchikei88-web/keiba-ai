"""
LightGBMモデルを学習・保存するスクリプト
学習済みモデル + 馬・騎手・調教師の統計キャッシュを保存する
"""

import sqlite3
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import DB_PATH, MODEL_DIR, model_path
from features import prepare_dataset, FEATURE_COLS, load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = model_path("keiba_lgbm.pkl")
STATS_CACHE_PATH = model_path("stats_cache.pkl")
META_PATH = model_path("model_meta.json")

LGB_PARAMS = {
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

N_ROUNDS = 5000
EARLY_STOPPING = 100


def build_stats_cache(df: pd.DataFrame, cutoff_date=None) -> dict:
    """
    予測時にDBなしで使えるよう、馬・騎手・調教師・血統の統計を事前計算してキャッシュする

    cutoff_date: この日付以前のデータのみ使用（None = 全データ）。
                 学習時は training_cutoff_date を渡してval/未来データを除外する。
                 Phase 4 バックテストでも任意の日付を指定して再現性を確保する。
    """
    if cutoff_date is not None:
        df = df[df["date"] <= pd.Timestamp(cutoff_date)]
    logger.info(
        f"統計キャッシュを構築中... "
        f"(cutoff={cutoff_date if cutoff_date is not None else '制限なし'}, "
        f"レース数={df['race_id'].nunique() if 'race_id' in df.columns else '?'})"
    )

    df = df.copy()
    df["is_win"] = (df["finish_pos"] == 1).astype(int)
    df["is_top3"] = (df["finish_pos"] <= 3).astype(int)

    cache = {
        "horse": {},
        "jockey": {},
        "trainer": {},
        "father_heavy": {},
    }

    # 馬ごとの統計
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
        # 距離別・馬場別の勝率
        for dist_range, label in [((1000, 1400), "s"), ((1400, 1800), "m"), ((1800, 2400), "l"), ((2400, 4000), "xl")]:
            sub = g[(g["distance"] >= dist_range[0]) & (g["distance"] < dist_range[1])]
            cache["horse"][horse_id][f"dist_{label}_win"] = round(sub["is_win"].mean(), 4) if len(sub) > 0 else 0.0
        for cond in ["良", "稍重", "重", "不良"]:
            sub = g[g["track_condition"] == cond]
            cache["horse"][horse_id][f"cond_{cond}_win"] = round(sub["is_win"].mean(), 4) if len(sub) > 0 else 0.0

    # 騎手ごとの統計
    for jockey, g in df.groupby("jockey_name"):
        if not jockey:
            continue
        cache["jockey"][jockey] = {
            "win_rate": round(g["is_win"].mean(), 4),
            "top3_rate": round(g["is_top3"].mean(), 4),
        }

    # 調教師ごとの統計
    for trainer, g in df.groupby("trainer_name"):
        if not trainer:
            continue
        cache["trainer"][trainer] = {
            "win_rate": round(g["is_win"].mean(), 4),
        }

    # 父血統の重馬場成績
    if "father" in df.columns:
        heavy = df[df["track_condition"].isin(["重", "不良"])]
        for father, g in heavy.groupby("father"):
            if not father:
                continue
            cache["father_heavy"][father] = round(g["is_win"].mean(), 4)

    logger.info(
        f"キャッシュ完了: 馬{len(cache['horse'])}頭 / 騎手{len(cache['jockey'])}人 / "
        f"調教師{len(cache['trainer'])}人 / 父{len(cache['father_heavy'])}頭"
    )
    return cache


def train():
    logger.info("データセット準備中...")
    X, y, full = prepare_dataset()

    # 時系列順で80/20分割（data leakを防ぐため必ずchronological split）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 学習データの最終日（統計キャッシュのcutoffとして使用）
    training_cutoff_date = full.iloc[:split_idx]["date"].max()
    logger.info(f"学習データ期間: 〜{training_cutoff_date.date()} / 検証データ: {full.iloc[split_idx]['date'].date()}〜")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    logger.info(f"学習開始: {N_ROUNDS}ラウンド")
    logger.info(f"学習データ: {len(X_train)}行 / 検証データ: {len(X_val)}行")

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=True),
        lgb.log_evaluation(period=500),
    ]

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=N_ROUNDS,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    logger.info(f"検証AUC: {auc:.4f}")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    logger.info(f"\n特徴量重要度（上位10）:\n{importance.head(10).to_string(index=False)}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    # 統計キャッシュを構築・保存（予測サーバーがDBなしで使う）
    # cutoff_date = training_cutoff_date: valデータ以降の成績をキャッシュから除外し
    # 「学習時点で知り得た情報のみ」を反映する。Phase 4 バックテストでも同様に利用する。
    conn = sqlite3.connect(DB_PATH)
    raw_df = load_raw_data(conn)
    conn.close()
    stats_cache = build_stats_cache(raw_df, cutoff_date=training_cutoff_date)
    joblib.dump(stats_cache, STATS_CACHE_PATH)
    logger.info(f"統計キャッシュ保存: {STATS_CACHE_PATH}")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "best_iteration": model.best_iteration,
        "val_auc": round(auc, 4),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "training_cutoff_date": training_cutoff_date.strftime("%Y-%m-%d"),
        "feature_cols": FEATURE_COLS,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"学習完了 | AUC: {auc:.4f} | Best iteration: {model.best_iteration}")
    return model, auc


def retrain_with_new_data():
    logger.info("モデル再学習を開始します...")
    model, auc = train()
    logger.info(f"再学習完了 | AUC: {auc:.4f}")
    return model


if __name__ == "__main__":
    train()
