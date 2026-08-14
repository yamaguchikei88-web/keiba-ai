"""
MLモデル用の特徴量エンジニアリング
レースデータ → モデルが学習できる数値特徴量に変換する
"""

import sqlite3
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import DB_PATH

# 馬場状態の数値化
TRACK_COND_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}
SURFACE_MAP = {"芝": 0, "ダ": 1}
DIRECTION_MAP = {"右": 0, "左": 1, "": 2}
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}


def load_raw_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """DBから全レース・着順データを結合して読み込む"""
    query = """
        SELECT
            r.race_id, r.date, r.racecourse, r.race_num, r.distance, r.surface,
            r.direction, r.weather, r.track_condition, r.num_horses,
            rr.finish_pos, rr.gate_num, rr.horse_num, rr.horse_id,
            rr.horse_name, rr.age, rr.sex, rr.weight, rr.weight_change,
            rr.jockey_name, rr.trainer_name, rr.odds, rr.popularity,
            rr.time_seconds, rr.last_3f, rr.passing_order,
            h.father, h.mother, h.maternal_father
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        LEFT JOIN horses h ON rr.horse_id = h.horse_id
        WHERE rr.finish_pos IS NOT NULL
        ORDER BY r.date, r.race_id, rr.finish_pos
    """
    df = pd.read_sql_query(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calc_running_style(passing_order: str) -> int:
    """通過順位から脚質を判定 0:逃げ 1:先行 2:差し 3:追い込み"""
    if not passing_order or str(passing_order) == "nan":
        return -1
    positions = [int(p) for p in str(passing_order).split("-") if p.strip().isdigit()]
    if not positions:
        return -1
    avg = np.mean(positions[:2]) if len(positions) >= 2 else positions[0]
    if avg <= 2:
        return 0
    elif avg <= 5:
        return 1
    elif avg <= 9:
        return 2
    else:
        return 3


def build_horse_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    各馬・騎手・調教師の過去成績を計算
    ※ データリークを防ぐため、各レースより「前」のデータのみ使用
    """
    df = df.sort_values(["date", "race_id", "horse_num"]).copy()
    df["is_win"] = (df["finish_pos"] == 1).astype(int)
    df["is_top3"] = (df["finish_pos"] <= 3).astype(int)

    result_rows = []

    for _, group in df.groupby("race_id"):
        race_date = group["date"].iloc[0]
        past = df[df["date"] < race_date]

        for _, row in group.iterrows():
            features = row.to_dict()

            # -------- 馬の過去成績 --------
            horse_past = past[past["horse_id"] == row["horse_id"]]
            n = len(horse_past)
            features["horse_race_count"] = n
            features["horse_win_rate"] = horse_past["is_win"].mean() if n > 0 else 0.0
            features["horse_top3_rate"] = horse_past["is_top3"].mean() if n > 0 else 0.0
            features["horse_avg_finish"] = horse_past["finish_pos"].mean() if n > 0 else 10.0

            # 同距離での成績
            same_dist = horse_past[
                (horse_past["distance"] - row["distance"]).abs() <= 200
            ]
            nd = len(same_dist)
            features["horse_dist_win_rate"] = same_dist["is_win"].mean() if nd > 0 else 0.0

            # 同馬場状態での成績
            same_cond = horse_past[horse_past["track_condition"] == row["track_condition"]]
            nc = len(same_cond)
            features["horse_cond_win_rate"] = same_cond["is_win"].mean() if nc > 0 else 0.0

            # 直近3走平均着順
            recent = horse_past.tail(3)
            features["horse_recent_avg"] = recent["finish_pos"].mean() if len(recent) > 0 else 10.0

            # -------- 騎手の過去成績 --------
            jockey_past = past[past["jockey_name"] == row["jockey_name"]]
            nj = len(jockey_past)
            features["jockey_win_rate"] = jockey_past["is_win"].mean() if nj > 0 else 0.0
            features["jockey_top3_rate"] = jockey_past["is_top3"].mean() if nj > 0 else 0.0

            # -------- 調教師の過去成績 --------
            trainer_past = past[past["trainer_name"] == row["trainer_name"]]
            nt = len(trainer_past)
            features["trainer_win_rate"] = trainer_past["is_win"].mean() if nt > 0 else 0.0

            # -------- 血統の重馬場適性 --------
            if row.get("father"):
                father_horses = past[past["father"] == row["father"]]
                heavy = father_horses[father_horses["track_condition"].isin(["重", "不良"])]
                features["father_heavy_win_rate"] = heavy["is_win"].mean() if len(heavy) > 0 else 0.0
            else:
                features["father_heavy_win_rate"] = 0.0

            # -------- 脚質 --------
            if n > 0:
                style_series = horse_past["passing_order"].dropna().apply(calc_running_style)
                features["running_style"] = style_series.mode()[0] if len(style_series) > 0 else -1
            else:
                features["running_style"] = -1

            result_rows.append(features)

    return pd.DataFrame(result_rows)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ変数を数値化"""
    df = df.copy()
    df["track_condition_enc"] = df["track_condition"].map(TRACK_COND_MAP).fillna(-1)
    df["weather_enc"] = df["weather"].map(WEATHER_MAP).fillna(-1)
    df["surface_enc"] = df["surface"].map(SURFACE_MAP).fillna(-1)
    df["direction_enc"] = df["direction"].map(DIRECTION_MAP).fillna(2)
    df["sex_enc"] = df["sex"].map(SEX_MAP).fillna(-1)

    # 血統を数値IDにエンコード（Label Encoding）
    for col in ["father", "maternal_father", "racecourse"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes

    # 月・曜日
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    return df


FEATURE_COLS = [
    "distance", "surface_enc", "direction_enc", "track_condition_enc", "weather_enc",
    "num_horses", "gate_num", "horse_num", "age", "sex_enc", "weight",
    "odds", "popularity",
    "horse_race_count", "horse_win_rate", "horse_top3_rate", "horse_avg_finish",
    "horse_dist_win_rate", "horse_cond_win_rate", "horse_recent_avg",
    "jockey_win_rate", "jockey_top3_rate",
    "trainer_win_rate",
    "father_heavy_win_rate",
    "father_enc", "maternal_father_enc", "racecourse_enc",
    "running_style", "month", "dayofweek",
]


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """学習用データセットを作成"""
    conn = sqlite3.connect(DB_PATH)
    raw = load_raw_data(conn)
    conn.close()

    print(f"生データ: {len(raw)}行")
    featured = build_horse_stats(raw)
    featured = encode_features(featured)

    # 欠損値を埋める
    X = featured[FEATURE_COLS].fillna(-1)
    y = (featured["finish_pos"] == 1).astype(int)  # 1着=1, それ以外=0

    print(f"特徴量行列: {X.shape}")
    print(f"勝率（1着の割合）: {y.mean():.3f}")
    return X, y, featured


if __name__ == "__main__":
    X, y, full = prepare_dataset()
    print(X.head())
