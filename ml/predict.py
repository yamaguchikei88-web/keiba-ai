"""
学習済みモデル + 統計キャッシュを使って予想を生成する
サーバー上でDBなしで動作する
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import re
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from project_paths import model_path, PRODUCTION_VERSION_PATH
from features import encode_features, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = model_path("keiba_lgbm.pkl")
STATS_CACHE_PATH = model_path("stats_cache.pkl")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RACECOURSE_NAME_TO_CODE = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

_model = None
_stats = None
_cache_version = None  # 現在キャッシュ済みのモデルに対応する production_version.txt の内容

# バージョン文字列の許可パターン（英数字・ハイフン・アンダースコア、先頭は英数字、最大64文字）
_VERSION_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$')


def _get_effective_version() -> str:
    """production_version.txt から本番採用バージョンを読む。

    MODEL_VERSION 環境変数は ml/train.py 専用。predict.py は一切参照しない。
    本番モデルの切り替えは production_version.txt の更新のみで行う。
    ファイルが存在しない・空の場合は RuntimeError を送出する（安全側に倒す）。
    """
    if not PRODUCTION_VERSION_PATH.exists():
        raise RuntimeError(
            f"本番モデルが未設定です。production_version.txt が見つかりません:\n"
            f"  {PRODUCTION_VERSION_PATH}\n"
            f"設定するには: echo 'v202609' > {PRODUCTION_VERSION_PATH}"
        )
    version = PRODUCTION_VERSION_PATH.read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError(
            f"production_version.txt が空です: {PRODUCTION_VERSION_PATH}\n"
            "有効なバージョン文字列（例: v202609）を書き込んでください。"
        )
    return version


def _resolve_load_paths() -> tuple[Path, Path]:
    """Return (model_pkl, stats_pkl). model + stats + meta の 3 ファイルが揃っているか検証する。

    1 ファイルでも欠けている場合は本番採用を拒否して FileNotFoundError を送出する。
    """
    version = _get_effective_version()
    mp = model_path(f"keiba_lgbm_{version}.pkl")
    sp = model_path(f"stats_cache_{version}.pkl")
    mep = model_path(f"model_meta_{version}.json")
    missing = [p for p in (mp, sp, mep) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"バージョン {version!r} のモデルファイルが不完全です（本番採用を拒否）:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    return mp, sp


def reload_model() -> None:
    """モデルキャッシュをクリアして次回 load_model() 時に強制再読み込みさせる。

    production_version.txt を更新後、または新モデルのデプロイ後に呼び出す。
    """
    global _model, _stats, _cache_version
    _model = None
    _stats = None
    _cache_version = None


def set_production_version(version: str) -> None:
    """production_version.txt をアトミックに更新して本番モデルを切り替える。

    手順:
    1. バージョン文字列の形式を検証する
    2. モデル 3 ファイル（pkl + stats + meta）の存在を確認する
    3. .tmp ファイルに書き込んでからアトミックにリネームする
       （POSIX では真のアトミック操作、Windows では best-effort）

    ロールバック例:
        set_production_version("v202608")  # 旧バージョンへ切り替え
    """
    if not _VERSION_RE.match(version):
        raise ValueError(
            f"不正なバージョン文字列: {version!r}\n"
            "使用可能: 英数字・ハイフン・アンダースコア（先頭は英数字、最大64文字）"
        )
    mp = model_path(f"keiba_lgbm_{version}.pkl")
    sp = model_path(f"stats_cache_{version}.pkl")
    mep = model_path(f"model_meta_{version}.json")
    missing = [p for p in (mp, sp, mep) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"バージョン {version!r} のモデルファイルが存在しません（切り替えを拒否）:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    PRODUCTION_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PRODUCTION_VERSION_PATH.with_name("production_version.txt.tmp")
    tmp_path.write_text(version.strip() + "\n", encoding="utf-8")
    tmp_path.replace(PRODUCTION_VERSION_PATH)  # POSIX: atomic; Windows: best-effort
    logger.info(f"production_version.txt を {version!r} に更新しました (atomic rename)")


def load_model():
    global _model, _stats, _cache_version
    current_version = _get_effective_version()
    # 有効バージョンが変わった場合はキャッシュを無効化して再ロード
    # （production_version.txt の更新も自動検出する）
    if _model is not None and _cache_version != current_version:
        logger.info(
            f"有効バージョン変更を検出 ({_cache_version!r} → {current_version!r}): "
            f"キャッシュを再読み込みします"
        )
        _model = None
        _stats = None
    if _model is None:
        mp, sp = _resolve_load_paths()
        _model = joblib.load(mp)
        _stats = joblib.load(sp)
        _cache_version = current_version
        logger.info(f"モデル読み込み完了 (version={current_version or 'default'})")
    return _model, _stats


def build_race_id(date: str, course: str, kai: int, day: int, race_num: int) -> str:
    y = date.replace("-", "")[:4]
    code = RACECOURSE_NAME_TO_CODE.get(course, "05")
    return f"{y}{code}{kai:02d}{day:02d}{race_num:02d}"


def fetch_shutuba(race_id: str) -> tuple[list[dict], dict]:
    """出馬表とレース情報を取得"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")

    race_info = {
        "race_id": race_id,
        "distance": 2000,
        "surface": "芝",
        "direction": "右",
        "weather": "晴",
        "track_condition": "良",
        "racecourse": "",
        "race_num": int(race_id[-2:]) if race_id[-2:].isdigit() else 0,
        "num_horses": 18,
    }

    data_el = soup.select_one(".RaceData01")
    if data_el:
        text = data_el.get_text()
        m = re.search(r"(\d{3,4})m", text)
        if m:
            race_info["distance"] = int(m.group(1))
        race_info["surface"] = "芝" if "芝" in text else "ダ"
        race_info["direction"] = "右" if "右" in text else "左" if "左" in text else ""
        wm = re.search(r"天候\s*:\s*(\S+)", text)
        if wm:
            race_info["weather"] = wm.group(1)
        cm = re.search(r"馬場\s*:\s*(\S+)", text)
        if cm:
            race_info["track_condition"] = cm.group(1)

    horses = []
    table = soup.select_one(".Shutuba_Table")
    if not table:
        return [], race_info

    for row in table.select("tr.HorseList"):
        cols = row.select("td")
        if len(cols) < 8:
            continue
        horse_link = row.select_one("td.HorseName a")
        horse_id = ""
        if horse_link:
            m = re.search(r"/horse/(\d+)/", horse_link.get("href", ""))
            if m:
                horse_id = m.group(1)

        father = ""
        ped_link = row.select_one("td.Pedigree")
        if ped_link:
            father_el = ped_link.select_one("a")
            if father_el:
                father = father_el.get_text(strip=True)

        age_sex_text = cols[4].get_text(strip=True) if len(cols) > 4 else ""
        sex = age_sex_text[:1] if age_sex_text else ""
        age_str = re.sub(r"\D", "", age_sex_text)
        age = int(age_str) if age_str else 4

        weight_text = cols[8].get_text(strip=True) if len(cols) > 8 else ""
        weight_m = re.match(r"(\d+)", weight_text)
        weight = int(weight_m.group(1)) if weight_m else 480

        horses.append({
            "gate_num": int(cols[0].get_text(strip=True)) if cols[0].get_text(strip=True).isdigit() else 0,
            "horse_num": int(cols[1].get_text(strip=True)) if cols[1].get_text(strip=True).isdigit() else 0,
            "horse_id": horse_id,
            "horse_name": horse_link.get_text(strip=True) if horse_link else "",
            "age": age,
            "sex": sex,
            "weight": weight,
            "jockey_name": cols[6].get_text(strip=True) if len(cols) > 6 else "",
            "trainer_name": cols[18].get_text(strip=True) if len(cols) > 18 else "",
            "father": father,
        })

    race_info["num_horses"] = len(horses)
    return horses, race_info


def lookup_stats(horse_id: str, jockey: str, trainer: str, father: str,
                 distance: int, track_condition: str, stats: dict) -> dict:
    """統計キャッシュから各馬の特徴量を取得"""
    h = stats["horse"].get(horse_id, {})
    j = stats["jockey"].get(jockey, {})
    t = stats["trainer"].get(trainer, {})

    dist_label = "s" if distance < 1400 else "m" if distance < 1800 else "l" if distance < 2400 else "xl"

    return {
        "horse_race_count": h.get("race_count", 0),
        "horse_win_rate": h.get("win_rate", 0.0),
        "horse_top3_rate": h.get("top3_rate", 0.0),
        "horse_avg_finish": h.get("avg_finish", 8.0),
        "horse_recent_avg": h.get("recent_avg", 8.0),
        "horse_dist_win_rate": h.get(f"dist_{dist_label}_win", 0.0),
        "horse_cond_win_rate": h.get(f"cond_{track_condition}_win", 0.0),
        "jockey_win_rate": j.get("win_rate", 0.0),
        "jockey_top3_rate": j.get("top3_rate", 0.0),
        "trainer_win_rate": t.get("win_rate", 0.0),
        "father_heavy_win_rate": stats["father_heavy"].get(father, 0.0),
        "running_style": -1,
    }


def predict_race(race_id: str) -> dict:
    model, stats = load_model()

    horses, race_info = fetch_shutuba(race_id)
    if not horses:
        raise ValueError(f"出馬表が取得できませんでした: {race_id}")

    today = datetime.now()
    rows = []
    for h in horses:
        stat_feats = lookup_stats(
            h["horse_id"], h["jockey_name"], h["trainer_name"],
            h.get("father", ""), race_info["distance"],
            race_info["track_condition"], stats,
        )
        row = {
            **race_info,
            **h,
            **stat_feats,
            "date": pd.Timestamp(today),
            # odds / popularity は FEATURE_COLS から除外済み（レース前は不明）
            # maternal_father は出馬表から取得不可のためデフォルト値
            "maternal_father": "",
            "month": today.month,
            "dayofweek": today.weekday(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = encode_features(df)
    X = df[FEATURE_COLS].fillna(-1)
    probs = model.predict(X)

    results = []
    for h, prob in zip(horses, probs):
        results.append({
            "horse_num": h["horse_num"],
            "gate_num": h["gate_num"],
            "horse_name": h["horse_name"],
            "jockey_name": h["jockey_name"],
            "win_prob": round(float(prob), 4),
        })

    results.sort(key=lambda x: x["win_prob"], reverse=True)

    marks = ["◎", "○", "▲", "△", "×"]
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["mark"] = marks[i] if i < len(marks) else ""

    top3 = [r["horse_num"] for r in results[:3]]
    recommendation = {
        "単勝": str(results[0]["horse_num"]),
        "馬連": f"{min(top3[0], top3[1])}-{max(top3[0], top3[1])}",
        "三連複": "-".join(str(n) for n in sorted(top3)),
        "三連単": f"{top3[0]}→{top3[1]}→{top3[2]}",
    }

    return {
        "race_id": race_id,
        "race_info": race_info,
        "horses": results,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    result = predict_race("202609030411")
    for h in result["horses"][:5]:
        print(f"{h['mark']} {h['horse_name']} ({h['horse_num']}番) 勝率: {h['win_prob']:.1%}")
    print("\n買い目:", result["recommendation"])
