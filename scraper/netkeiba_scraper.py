"""
netkeiba.com から過去レースデータを収集するスクレイパー
対象: 過去10年分の全JRAレース（約10万レース）

【リトライ方式】
  最大 max_retries=3 回、指数バックオフ (backoff_base × 2^attempt 秒)
  デフォルト待機: 1秒 → 2秒 → 4秒

【リトライ対象エラー】
  - requests.Timeout / ConnectionError / その他 RequestException
  - HTTP 429 (レート制限), 500, 502, 503, 504 (一時的サーバーエラー)

【リトライしないエラー（即時終了）】
  - HTTP 404 → not_found (存在しない race_id)
  - HTTP 400, 401, 403 → client_error (永続的クライアントエラー)
  - HTTP 200 かつ <h1> なし → not_found (netkeiba 空ページ = 非存在 race_id)
  - HTTP 200 かつ .race_table_01 なし → parse_error (HTML 構造変更 or 中止レース)

【非存在 race_id と通信失敗の区別】
  判定可能な範囲:
    1. HTTP 404 → not_found (明確な非存在)
    2. HTTP 200 + <h1> なし → not_found (netkeiba の空ページ応答)
    3. HTTP 200 + <h1> あり + .race_table_01 なし → parse_error (曖昧: 非存在 or HTML 変更)
    4. requests.RequestException → network_error / timeout (通信失敗)
    5. HTTP 5xx → server_error (サーバー側一時エラー)
  ※ netkeiba は非存在 race_id に対して 200 を返すことが多く、
     3 の判定は完全に分離できないため parse_error として記録する。

【失敗ログ】
  data/failed_races.jsonl に JSONL 形式で追記
  フィールド: race_id, status, reason, status_code, attempts, timestamp
"""

import json
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import sqlite3
import time
import re
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import DB_PATH, FAILED_LOG_PATH

BASE_URL = "https://db.netkeiba.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

RACECOURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# JRA 有効競馬場コード（01〜10）— 非開催日に返る偽IDのフィルタ用
VALID_COURSE_CODES = frozenset(f"{i:02d}" for i in range(1, 11))

# リトライを行う HTTP ステータスコード（一時的なサーバーエラー・レート制限）
_RETRY_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

# リトライしない HTTP ステータスコード（存在しないリソース・認証エラー）
_NO_RETRY_HTTP_CODES = frozenset({400, 401, 403, 404})


@dataclass
class ScrapeResult:
    """scrape_race() の結果を保持する。成功時のみ data は非 None。"""
    data: dict | None        # レースデータ（成功時のみ非 None）
    status: str              # "success" | "not_found" | "parse_error" |
                             # "timeout" | "network_error" | "server_error" |
                             # "rate_limited" | "client_error"
    status_code: int | None  # HTTP ステータスコード（ネットワークエラー時は None）
    attempts: int            # 実際の試行回数
    reason: str              # 失敗理由（成功時は ""）


def _extract_race_date(soup: BeautifulSoup) -> str | None:
    """HTML から実際の開催日付を取得する。取得できない場合は None を返す。
    race_id からの推測は行わない。"""
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


def _validate_date_arg(value: str, arg_name: str) -> None:
    """YYYYMMDD 形式かつ実在する日付かを検証する。不正な場合は ValueError を送出。"""
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError(
            f"{arg_name} は 'YYYYMMDD' 形式の8桁数字で指定してください（例: '20190101'）: {value!r}"
        )
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{arg_name} に存在しない日付が指定されました: {value!r}")


def get_race_id_list(year: int, course_code: str) -> list[str]:
    """指定年・競馬場の全レースID候補を生成する。

    生成方式: kai(1-5) × day(1-12) × race_num(1-12) = 720候補/コース/年
    ※ 候補の約55%は実在しない race_id（JRA実際のレース数 ≈ 年間3,400-3,600件 / 10コース）
    ※ 実在 race_id の取得方式（race calendar 等）への変更は Phase 5 Step 3 前に要判断
    """
    race_ids = []
    for kai in range(1, 6):       # 開催回数（最大5回）
        for day in range(1, 13):  # 開催日数（最大12日）
            for race_num in range(1, 13):  # 1〜12R
                race_id = f"{year}{course_code}{kai:02d}{day:02d}{race_num:02d}"
                race_ids.append(race_id)
    return race_ids


def get_race_ids_for_date(date_str: str, session: requests.Session) -> list[str]:
    """'20240127' → その日の有効 JRA race_id 一覧。

    race/list/{date}/ は非開催日でも HTTP 200 + 偽 ID (course code 45+) を返すため、
    course code が VALID_COURSE_CODES (01〜10) に含まれる ID のみ返す。
    非開催日・エラー時は [] を返す。
    """
    url = f"{BASE_URL}/race/list/{date_str}/"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        resp.encoding = "EUC-JP"
        all_ids = re.findall(r'/race/(\d{12})/', resp.text)
        seen: set[str] = set()
        return [r for r in all_ids
                if r[4:6] in VALID_COURSE_CODES and not (r in seen or seen.add(r))]
    except requests.RequestException:
        return []


def get_race_dates_in_range(start_year: int, end_year: int) -> list[str]:
    """start_year/1/1 〜 end_year/12/31 の全日付を 'YYYYMMDD' 形式で返す。"""
    from datetime import date, timedelta
    d = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    dates: list[str] = []
    while d <= end:
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def parse_time(time_str: str) -> float | None:
    """'1:33.5' → 秒数 93.5"""
    if not time_str or time_str.strip() == "":
        return None
    m = re.match(r"(\d+):(\d+)\.(\d+)", time_str.strip())
    if m:
        minutes, seconds, fraction = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return minutes * 60 + seconds + fraction / 10
    return None


def _parse_race_html(race_id: str, html: str) -> tuple[dict | None, str]:
    """HTML テキストからレースデータを抽出する（純粋関数・テスト容易）。

    Returns:
        (data, parse_status):
          data        : レースデータ dict（失敗時は None）
          parse_status: "success" | "not_found" | "parse_error"

    判定方式:
      "not_found"  : <h1> にテキストなし → netkeiba が非存在 race_id に返す空ページ
      "parse_error": <h1> あり + .race_table_01 なし → HTML 構造変更 or 中止レース（曖昧）
      "success"    : 上記以外・正常パース完了
    """
    soup = BeautifulSoup(html, "html.parser")
    race_data: dict = {"race_id": race_id}

    race_name_el = next(
        (h for h in soup.find_all("h1") if h.get_text(strip=True)), None
    )
    if not race_name_el:
        return None, "not_found"
    race_data["race_name"] = race_name_el.get_text(strip=True)

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
        surface = race_data.get("surface", "")
        if surface == "芝":
            cond_m = re.search(r"芝\s*:\s*(\S+)", text)
        else:
            cond_m = re.search(r"ダート\s*:\s*(\S+)", text)
        if not cond_m:
            cond_m = re.search(r"馬場\s*:\s*(\S+)", text)
        if cond_m:
            race_data["track_condition"] = cond_m.group(1)

    race_data["date"] = _extract_race_date(soup)
    race_data["racecourse"] = RACECOURSE_CODES.get(race_id[4:6], "")
    race_data["race_num"] = int(race_id[-2:]) if race_id[-2:].isdigit() else 0

    result_table = soup.select_one(".race_table_01")
    if not result_table:
        return None, "parse_error"

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

            # netkeiba 結果テーブル列構造（db.netkeiba.com/race/）
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
    return race_data, "success"


def scrape_race(
    race_id: str,
    session: requests.Session,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> ScrapeResult:
    """1 レース分のデータを指数バックオフ付きリトライで取得する。

    バックオフ待機時間: backoff_base × 2^attempt 秒
      attempt=0: backoff_base × 1 = 1.0秒（デフォルト）
      attempt=1: backoff_base × 2 = 2.0秒
      attempt=2: backoff_base × 4 = 4.0秒（最終試行はリトライしないのでsleepなし）
    """
    url = f"{BASE_URL}/race/{race_id}/"
    last_exc: Exception | None = None
    last_status: int | None = None

    for attempt in range(max_retries):
        current_attempt = attempt + 1
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
        except requests.Timeout as e:
            last_exc = e
            last_status = None
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    f"タイムアウト {race_id} (試行 {current_attempt}/{max_retries}): "
                    f"{wait:.0f}秒後にリトライ"
                )
                time.sleep(wait)
            continue
        except requests.RequestException as e:
            last_exc = e
            last_status = None
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    f"ネットワークエラー {race_id} (試行 {current_attempt}/{max_retries}): "
                    f"{wait:.0f}秒後にリトライ"
                )
                time.sleep(wait)
            continue

        last_status = resp.status_code

        # 存在しないリソース・永続的クライアントエラー → リトライしない
        if resp.status_code in _NO_RETRY_HTTP_CODES:
            status = "not_found" if resp.status_code == 404 else "client_error"
            return ScrapeResult(
                data=None, status=status, status_code=resp.status_code,
                attempts=current_attempt, reason=f"HTTP {resp.status_code}",
            )

        # 一時的サーバーエラー・レート制限 → リトライ
        if resp.status_code in _RETRY_HTTP_CODES:
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                logger.warning(
                    f"HTTP {resp.status_code} {race_id} (試行 {current_attempt}/{max_retries}): "
                    f"{wait:.0f}秒後にリトライ"
                )
                time.sleep(wait)
                continue
            status = "rate_limited" if resp.status_code == 429 else "server_error"
            return ScrapeResult(
                data=None, status=status, status_code=resp.status_code,
                attempts=current_attempt,
                reason=f"HTTP {resp.status_code} (全{max_retries}回失敗)",
            )

        # その他の非 200
        if resp.status_code != 200:
            return ScrapeResult(
                data=None, status="client_error", status_code=resp.status_code,
                attempts=current_attempt, reason=f"HTTP {resp.status_code}",
            )

        # HTTP 200: HTML パース（EUC-JP デコード後に純粋関数へ渡す）
        resp.encoding = "EUC-JP"
        data, parse_status = _parse_race_html(race_id, resp.text)

        reason_map = {
            "success":     "",
            "not_found":   "レース名が見つからない（非存在 race_id またはページ構造変更）",
            "parse_error": "着順テーブルなし（HTML 構造変更または中止レース）",
        }
        return ScrapeResult(
            data=data,
            status=parse_status,
            status_code=200,
            attempts=current_attempt,
            reason=reason_map.get(parse_status, parse_status),
        )

    # 全試行がネットワーク/タイムアウトエラー
    status = "timeout" if isinstance(last_exc, requests.Timeout) else "network_error"
    reason = f"{type(last_exc).__name__}: {str(last_exc)[:150]}" if last_exc else "不明なエラー"
    return ScrapeResult(
        data=None, status=status, status_code=last_status,
        attempts=max_retries, reason=reason,
    )


def _log_failed_race(race_id: str, result: ScrapeResult, log_path: Path) -> None:
    """取得失敗 race_id を JSONL ファイルに追記する。

    機密情報（URL パラメータ・ユーザー情報等）は記録しない。
    reason は最大 200 文字に切り詰めてログ肥大化を防止する。
    """
    entry = {
        "race_id": race_id,
        "status": result.status,
        "reason": result.reason[:200],
        "status_code": result.status_code,
        "attempts": result.attempts,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


def scrape_all(
    start_year: int = 2015,
    end_year: int = 2025,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_seconds: float = 1.0,
    sleep_calendar: float = 0.5,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    failed_log_path: Path | None = None,
):
    """指定期間の全 JRA レースをスクレイピングする（カレンダー方式）。

    Args:
        start_year:       取得開始年
        end_year:         取得終了年（含む）
        start_date:       取得開始日（'YYYYMMDD' 形式、省略時は start_year/1/1）
        end_date:         取得終了日（'YYYYMMDD' 形式、省略時は end_year/12/31）
                          start_date と end_date は両方指定するか両方省略すること。
                          start_date <= end_date であること。
        sleep_seconds:    レース間待機秒数（デフォルト 1.0 秒）
        sleep_calendar:   カレンダーページ間待機秒数（デフォルト 0.5 秒）
                          netkeiba へのレート制限対策として使用する。
        max_retries:      リトライ最大回数（デフォルト 3）
        backoff_base:     指数バックオフの基底秒数（デフォルト 1.0）
        failed_log_path:  失敗ログ出力先（None の場合 data/failed_races.jsonl）

    カレンダー方式:
        race/list/{YYYYMMDD}/ から実在 race_id を取得するため、
        非存在候補へのリクエストをほぼゼロに削減できる。
        非開催日は有効コードの ID が 0 件となり自動スキップ。

    分割実行:
        start_date / end_date で月単位・週単位の分割実行が可能。
        Colab セッション終了後も races テーブルに保存済みの race_id は
        次回実行時に自動スキップされるため、中断から安全に再開できる。

    再開機能:
        正常保存済み race_id は races テーブルで管理し、次回実行時にスキップする。
        取得失敗 race_id は races テーブルに記録されないため、次回実行時に自動再試行。
        重複 INSERT は INSERT OR IGNORE と UNIQUE 制約で防止済み。
    """
    # 引数検証（init_db より前に行う）
    if (start_date is None) != (end_date is None):
        raise ValueError(
            "start_date と end_date は両方指定するか、両方省略してください"
        )
    if start_date is not None:
        _validate_date_arg(start_date, "start_date")
        _validate_date_arg(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) が end_date ({end_date}) より後です"
            )

    init_db()
    conn = sqlite3.connect(DB_PATH)
    session = requests.Session()

    if failed_log_path is None:
        failed_log_path = FAILED_LOG_PATH  # KEIBA_FAILED_LOG_PATH env var or DATA_DIR/failed_races.jsonl

    total = skipped = saved = not_found = parse_errors = failed = 0
    dates = get_race_dates_in_range(start_year, end_year)
    if start_date is not None:
        dates = [d for d in dates if start_date <= d <= end_date]

    logger.info(
        f"スクレイピング開始: {start_date or f'{start_year}0101'}〜{end_date or f'{end_year}1231'} "
        f"({len(dates)}日間)"
    )

    try:
        for date_str in dates:
            race_ids = get_race_ids_for_date(date_str, session)
            if sleep_calendar > 0:
                time.sleep(sleep_calendar)  # カレンダーページ間のレート制限対策
            if not race_ids:
                continue  # 非開催日または取得エラー

            logger.info(f"{date_str}: {len(race_ids)}件の race_id を取得")

            for race_id in race_ids:
                total += 1

                # 正常取得済みの race_id はスキップ（再起動後も続きから再開可能）
                c = conn.cursor()
                c.execute("SELECT 1 FROM races WHERE race_id=?", (race_id,))
                if c.fetchone():
                    skipped += 1
                    continue

                result = scrape_race(
                    race_id, session,
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                )

                if result.status == "success" and result.data and result.data.get("results"):
                    save_race(result.data, conn)
                    saved += 1
                    if saved % 100 == 0:
                        logger.info(
                            f"進捗: 保存={saved}, スキップ={skipped}, "
                            f"非存在={not_found}, パースエラー={parse_errors}, 失敗={failed}"
                        )
                elif result.status == "not_found":
                    not_found += 1
                    logger.debug(f"非存在 {race_id}: {result.reason}")
                elif result.status == "parse_error":
                    parse_errors += 1
                    logger.debug(f"パースエラー {race_id}: {result.reason}")
                else:
                    # ネットワーク/サーバーエラー → 失敗ログへ記録・次回再試行可能
                    failed += 1
                    _log_failed_race(race_id, result, failed_log_path)
                    logger.warning(f"取得失敗 {race_id}: {result.reason}")

                time.sleep(sleep_seconds)
    finally:
        conn.close()

    logger.info(
        f"スクレイピング完了: 合計候補={total}, 保存={saved}, 取得済みスキップ={skipped}, "
        f"非存在={not_found}, パースエラー={parse_errors}, 通信失敗={failed}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="netkeiba レーススクレイパー")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--start-date", type=str, default=None,
                        help="取得開始日 YYYYMMDD（省略時は start-year/1/1）")
    parser.add_argument("--end-date",   type=str, default=None,
                        help="取得終了日 YYYYMMDD（省略時は end-year/12/31）")
    parser.add_argument("--sleep",      type=float, default=1.0,
                        help="レース間待機秒数（デフォルト 1.0）")
    parser.add_argument("--sleep-calendar", type=float, default=0.5,
                        help="カレンダーページ間待機秒数（デフォルト 0.5）")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    args = parser.parse_args()

    scrape_all(
        start_year=args.start_year,
        end_year=args.end_year,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_seconds=args.sleep,
        sleep_calendar=args.sleep_calendar,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
    )
