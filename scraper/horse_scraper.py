"""
馬の血統情報を取得するスクレイパー
DBに登録されている horse_id を元に血統データを補完する
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "keiba.db"
BASE_URL = "https://db.netkeiba.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def scrape_horse_pedigree(horse_id: str, session: requests.Session) -> dict | None:
    url = f"{BASE_URL}/horse/ped/{horse_id}/"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")

        pedigree = {"horse_id": horse_id}

        # 血統表から父・母・母父を取得
        ped_table = soup.select_one(".blood_table")
        if not ped_table:
            return None

        cells = ped_table.select("td")
        if len(cells) >= 1:
            pedigree["father"] = cells[0].get_text(strip=True) if len(cells) > 0 else ""
        if len(cells) >= 16:
            pedigree["mother"] = cells[15].get_text(strip=True) if len(cells) > 15 else ""
            pedigree["maternal_father"] = cells[14].get_text(strip=True) if len(cells) > 14 else ""

        # 馬名
        name_el = soup.select_one(".horse_title h1")
        if name_el:
            pedigree["horse_name"] = name_el.get_text(strip=True)

        return pedigree

    except Exception as e:
        logger.warning(f"血統取得エラー {horse_id}: {e}")
        return None


def fill_pedigree():
    """DBの全馬の血統データを補完"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 血統未取得の馬を抽出
    c.execute("""
        SELECT DISTINCT rr.horse_id
        FROM race_results rr
        LEFT JOIN horses h ON rr.horse_id = h.horse_id
        WHERE h.horse_id IS NULL AND rr.horse_id != ''
        LIMIT 5000
    """)
    horse_ids = [row[0] for row in c.fetchall()]
    logger.info(f"血統未取得の馬: {len(horse_ids)}頭")

    session = requests.Session()
    for i, horse_id in enumerate(horse_ids):
        data = scrape_horse_pedigree(horse_id, session)
        if data:
            c.execute("""
                INSERT OR REPLACE INTO horses (horse_id, horse_name, father, mother, maternal_father)
                VALUES (?,?,?,?,?)
            """, (
                data.get("horse_id"), data.get("horse_name", ""),
                data.get("father", ""), data.get("mother", ""), data.get("maternal_father", ""),
            ))
            conn.commit()

        if i % 50 == 0:
            logger.info(f"進捗: {i}/{len(horse_ids)}")

        time.sleep(1.5)

    conn.close()
    logger.info("血統データ補完完了")


if __name__ == "__main__":
    fill_pedigree()
