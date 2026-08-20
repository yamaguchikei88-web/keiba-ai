"""
netkeiba.com から過去レースデータを収集するスクレイパー
対象: 過去10年分の全JRAレース（約10万レース）
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import re
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import DB_PATH
BASE_URL = "https://db.netkeiba.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

RACECOURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def _extract_race_date(soup: BeautifulSoup) -> str | None:
    """HTMLから実際の開催日付を取得する。取得できない場合は None を返す。
    race_id からの推測は絶対に行わない。"""
    for get_el in (
        lambda s: s.find("title"),
        lambda s: s.select_one(".RaceData02"),
    ):
        el = get_el(soup)
        if el:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", el.get_text())
            if m:
                return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _parse_weight(text: str) -> tuple[int | None, int | None]:
    """馬体重テキストをパースして (体重, 増減) を返す。
    '478(-2)' → (478, -2) / '480(+4)' → (480, 4) / '500' → (500, None)"""
    if not text:
        return None, None
    m = re.match(r"(\d+)\(([+-]?\d+)\)", text.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r"(\d+)", text.strip())
    if m2:
        return int(m2.group(1)), None
    return None, None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            racecourse TEXT,
            race_num INTEGER,
            race_name TEXT,
            grade TEXT,
            distance INTEGER,
            surface TEXT,
            direction TEXT,
            weather TEXT,
            track_condition TEXT,
            num_horses INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            finish_pos INTEGER,
            gate_num INTEGER,
            horse_num INTEGER,
            horse_id TEXT,
            horse_name TEXT,
            age INTEGER,
            sex TEXT,
            weight INTEGER,
            weight_change INTEGER,
            jockey_id TEXT,
            jockey_name TEXT,
            trainer_id TEXT,
            trainer_name TEXT,
            odds REAL,
            popularity INTEGER,
            time_seconds REAL,
            margin TEXT,
            passing_order TEXT,
            last_3f REAL,
            FOREIGN KEY (race_id) REFERENCES races(race_id),
            UNIQUE (race_id, horse_num)
        );

        -- 既存DBに UNIQUE 制約がない場合でも重複を防ぐため unique index を別途作成
        CREATE UNIQUE INDEX IF NOT EXISTS idx_race_results_dedup
            ON race_results(race_id, horse_num);

        CREATE TABLE IF NOT EXISTS horses (
            horse_id TEXT PRIMARY KEY,
            horse_name TEXT,
            father TEXT,
            mother TEXT,
            maternal_father TEXT,
            birth_year INTEGER,
            sex TEXT,
            color TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            horse_id TEXT,
            training_date TEXT,
            course TEXT,
            time_5f REAL,
            time_3f REAL,
            rank INTEGER,
            total INTEGER,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );

        CREATE INDEX IF NOT EXISTS idx_race_results_race_id ON race_results(race_id);
        CREATE INDEX IF NOT EXISTS idx_race_results_horse_id ON race_results(horse_id);
        CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
    """)
    conn.commit()
    conn.close()
    logger.info(f"DB初期化完了: {DB_PATH}")


def get_race_id_list(year: int, course_code: str) -> list[str]:
    """指定年・競馬場の全レースIDを取得"""
    race_ids = []
    for kai in range(1, 6):       # 開催回数（最大5回）
        for day in range(1, 13):  # 開催日数（最大12日）
            for race_num in range(1, 13):  # 1〜12R
                race_id = f"{year}{course_code}{kai:02d}{day:02d}{race_num:02d}"
                race_ids.append(race_id)
    return race_ids


def parse_time(time_str: str) -> float | None:
    """'1:33.5' → 秒数 93.5"""
    if not time_str or time_str.strip() == "":
        return None
    m = re.match(r"(\d+):(\d+)\.(\d+)", time_str.strip())
    if m:
        minutes, seconds, fraction = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return minutes * 60 + seconds + fraction / 10
    return None


def scrape_race(race_id: str, session: requests.Session) -> dict | None:
    """1レース分のデータを取得"""
    url = f"{BASE_URL}/race/{race_id}/"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        # レース情報
        race_data = {"race_id": race_id}

        # レース名: 非空の <h1> を探す（.RaceName クラスは現 netkeiba では廃止）
        race_name_el = next(
            (h for h in soup.find_all("h1") if h.get_text(strip=True)), None
        )
        if not race_name_el:
            return None
        race_data["race_name"] = race_name_el.get_text(strip=True)

        # レース条件（距離・馬場・天気など）.mainrace_data から取得（.RaceData01 は廃止）
        data_intro = soup.select_one(".mainrace_data")
        if data_intro:
            text = data_intro.get_text()
            dist_m = re.search(r"(\d{3,4})m", text)
            if dist_m:
                race_data["distance"] = int(dist_m.group(1))
            race_data["surface"] = "芝" if "芝" in text else "ダ"
            race_data["direction"] = "右" if "右" in text else "左" if "左" in text else ""
            weather_m = re.search(r"天候\s*:\s*(\S+)", text)
            if weather_m:
                race_data["weather"] = weather_m.group(1)
            # 馬場状態: 芝は「芝 : 良」、ダートは「ダート : 良」から取得
            surface = race_data.get("surface", "")
            if surface == "芝":
                cond_m = re.search(r"芝\s*:\s*(\S+)", text)
            else:
                cond_m = re.search(r"ダート\s*:\s*(\S+)", text)
            if not cond_m:
                cond_m = re.search(r"馬場\s*:\s*(\S+)", text)
            if cond_m:
                race_data["track_condition"] = cond_m.group(1)

        # 日付・競馬場・レース番号
        race_data["date"] = _extract_race_date(soup)  # None if not found in HTML
        race_data["racecourse"] = RACECOURSE_CODES.get(race_id[4:6], "")
        race_data["race_num"] = int(race_id[-2:]) if race_id[-2:].isdigit() else 0

        # 着順テーブル
        result_table = soup.select_one(".race_table_01")
        if not result_table:
            return None

        results = []
        rows = result_table.select("tr")[1:]  # ヘッダーをスキップ
        for row in rows:
            cols = row.select("td")
            if len(cols) < 23:  # cols[22](調教師)まで必要
                continue
            try:
                horse_link = cols[3].select_one("a")
                jockey_link = cols[6].select_one("a")

                horse_id = ""
                if horse_link and horse_link.get("href"):
                    hm = re.search(r"/horse/(\d+)/", horse_link["href"])
                    if hm:
                        horse_id = hm.group(1)

                # netkeiba結果テーブル列構造 実HTML確認済み 25列 (db.netkeiba.com/race/)
                # col0:着順 col1:枠番 col2:馬番 col3:馬名 col4:性齢 col5:斤量 col6:騎手
                # col7:タイム col8:着差 col9-13:タイム指数系(プレミアム/未使用)
                # col14:通過 col15:上り col16:単勝 col17:人気 col18:馬体重
                # col19:調教タイム col20:厩舎ｺﾒﾝﾄ col21:備考 col22:調教師 col23:馬主 col24:賞金
                weight_raw = cols[18].get_text(strip=True) if len(cols) > 18 else ""
                weight, weight_change = _parse_weight(weight_raw)
                trainer_raw = cols[22].get_text(strip=True) if len(cols) > 22 else ""
                trainer_name = re.sub(r"^\[[東西]\]", "", trainer_raw)

                result = {
                    "race_id": race_id,
                    "finish_pos": int(cols[0].get_text(strip=True)) if cols[0].get_text(strip=True).isdigit() else None,
                    "gate_num": int(cols[1].get_text(strip=True)) if cols[1].get_text(strip=True).isdigit() else None,
                    "horse_num": int(cols[2].get_text(strip=True)) if cols[2].get_text(strip=True).isdigit() else None,
                    "horse_id": horse_id,
                    "horse_name": cols[3].get_text(strip=True),
                    "sex": cols[4].get_text(strip=True)[:1] if cols[4].get_text(strip=True) else "",
                    "age": int(re.sub(r"\D", "", cols[4].get_text(strip=True))) if re.sub(r"\D", "", cols[4].get_text(strip=True)) else None,
                    "weight": weight,
                    "weight_change": weight_change,
                    "jockey_name": jockey_link.get_text(strip=True) if jockey_link else cols[6].get_text(strip=True),
                    "time_seconds": parse_time(cols[7].get_text(strip=True)),
                    "margin": cols[8].get_text(strip=True) if len(cols) > 8 else "",
                    "passing_order": cols[14].get_text(strip=True) if len(cols) > 14 else "",
                    "last_3f": float(cols[15].get_text(strip=True)) if len(cols) > 15 and cols[15].get_text(strip=True).replace(".", "").isdigit() else None,
                    "odds": float(cols[16].get_text(strip=True)) if len(cols) > 16 and cols[16].get_text(strip=True).replace(".", "").isdigit() else None,
                    "popularity": int(cols[17].get_text(strip=True)) if len(cols) > 17 and cols[17].get_text(strip=True).isdigit() else None,
                    "trainer_name": trainer_name,
                }
                results.append(result)
            except Exception as e:
                logger.debug(f"行パースエラー: {e}")
                continue

        race_data["results"] = results
        race_data["num_horses"] = len(results)
        return race_data

    except requests.RequestException as e:
        logger.warning(f"リクエストエラー {race_id}: {e}")
        return None


def save_race(race_data: dict, conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO races
        (race_id, date, racecourse, race_num, race_name, distance, surface, direction,
         weather, track_condition, num_horses)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        race_data.get("race_id"), race_data.get("date"), race_data.get("racecourse"),
        race_data.get("race_num"), race_data.get("race_name"), race_data.get("distance"),
        race_data.get("surface"), race_data.get("direction"), race_data.get("weather"),
        race_data.get("track_condition"), race_data.get("num_horses"),
    ))
    for r in race_data.get("results", []):
        c.execute("""
            INSERT OR IGNORE INTO race_results
            (race_id, finish_pos, gate_num, horse_num, horse_id, horse_name, age, sex,
             weight, weight_change, jockey_name, time_seconds, margin, passing_order,
             last_3f, odds, popularity, trainer_name)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["race_id"], r["finish_pos"], r["gate_num"], r["horse_num"], r["horse_id"],
            r["horse_name"], r["age"], r["sex"], r["weight"], r["weight_change"],
            r["jockey_name"], r["time_seconds"], r["margin"], r["passing_order"],
            r["last_3f"], r["odds"], r["popularity"], r["trainer_name"],
        ))
    conn.commit()


def scrape_all(start_year: int = 2015, end_year: int = 2025):
    """指定期間の全JRAレースをスクレイピング"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    session = requests.Session()
    total_saved = 0

    for year in range(start_year, end_year + 1):
        for code in RACECOURSE_CODES.keys():
            race_ids = get_race_id_list(year, code)
            logger.info(f"{year}年 {RACECOURSE_CODES[code]}: {len(race_ids)}件を処理")
            for race_id in race_ids:
                # 既存チェック
                c = conn.cursor()
                c.execute("SELECT 1 FROM races WHERE race_id=?", (race_id,))
                if c.fetchone():
                    continue

                data = scrape_race(race_id, session)
                if data and data.get("results"):
                    save_race(data, conn)
                    total_saved += 1
                    if total_saved % 100 == 0:
                        logger.info(f"保存済み: {total_saved}レース")

                time.sleep(1.0)  # サーバー負荷軽減（1秒待機）

    conn.close()
    logger.info(f"スクレイピング完了: 合計 {total_saved}レース保存")


if __name__ == "__main__":
    scrape_all(start_year=2015, end_year=2025)
