"""
Phase 1 単体テスト
対象: scraper/netkeiba_scraper.py の date修正・UNIQUE制約・weight修正
"""

import sqlite3
import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scraper") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scraper"))

from netkeiba_scraper import _extract_race_date, _parse_weight


# ---------------------------------------------------------------------------
# _extract_race_date
# ---------------------------------------------------------------------------

class TestExtractRaceDate(unittest.TestCase):

    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_date_from_title(self):
        """<title> タグから日付を正しく取得できる"""
        soup = self._soup(
            "<html><head><title>2025年5月4日 東京11R ヴィクトリアマイル | netkeiba</title></head></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2025-05-04")

    def test_date_from_title_single_digit_month_day(self):
        """1桁の月日でもゼロパディングされる"""
        soup = self._soup(
            "<html><head><title>2024年1月7日 中山1R | netkeiba</title></head></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2024-01-07")

    def test_date_from_racedata02_fallback(self):
        """.RaceData02 にフォールバックして日付を取得できる"""
        soup = self._soup(
            "<html><body>"
            '<div class="RaceData02"><span>1回</span><span>東京</span>'
            "<span>2025年5月4日</span></div>"
            "</body></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2025-05-04")

    def test_no_date_returns_none(self):
        """日付情報が存在しない場合は None を返す（race_id からの推測は行わない）"""
        soup = self._soup("<html><head><title>ネット競馬</title></head></html>")
        self.assertIsNone(_extract_race_date(soup))

    def test_title_has_priority_over_racedata02(self):
        """<title> の日付が .RaceData02 より優先される"""
        soup = self._soup(
            "<html><head><title>2025年3月15日 阪神5R | netkeiba</title></head>"
            "<body>"
            '<div class="RaceData02"><span>2025年4月1日</span></div>'
            "</body></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2025-03-15")


# ---------------------------------------------------------------------------
# _parse_weight
# ---------------------------------------------------------------------------

class TestParseWeight(unittest.TestCase):

    def test_normal_with_decrease(self):
        self.assertEqual(_parse_weight("478(-2)"), (478, -2))

    def test_normal_with_increase(self):
        self.assertEqual(_parse_weight("480(+4)"), (480, 4))

    def test_no_change(self):
        self.assertEqual(_parse_weight("500(0)"), (500, 0))

    def test_no_parentheses(self):
        """初出走など変動なし表記: 体重のみ返す"""
        self.assertEqual(_parse_weight("456"), (456, None))

    def test_empty_string(self):
        self.assertEqual(_parse_weight(""), (None, None))

    def test_none_input(self):
        self.assertEqual(_parse_weight(None), (None, None))

    def test_whitespace_stripped(self):
        self.assertEqual(_parse_weight("  466(-6)  "), (466, -6))

    def test_non_numeric(self):
        """計不明など数字でない場合は (None, None)"""
        self.assertEqual(_parse_weight("計不明"), (None, None))


# ---------------------------------------------------------------------------
# UNIQUE 制約 (init_db → race_results)
# ---------------------------------------------------------------------------

def _build_in_memory_db() -> sqlite3.Connection:
    """テスト用のインメモリ DB に race_results テーブルを作成する"""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
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
            UNIQUE (race_id, horse_num)
        );
    """)
    return conn


class TestRaceResultsUniqueConstraint(unittest.TestCase):

    def setUp(self):
        self.conn = _build_in_memory_db()

    def tearDown(self):
        self.conn.close()

    def _insert(self, race_id: str, horse_num: int):
        self.conn.execute(
            "INSERT OR IGNORE INTO race_results (race_id, horse_num) VALUES (?, ?)",
            (race_id, horse_num),
        )
        self.conn.commit()

    def _count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]

    def test_first_insert_saved(self):
        self._insert("202505010511", 1)
        self.assertEqual(self._count(), 1)

    def test_duplicate_ignored(self):
        """同じ (race_id, horse_num) を2回 INSERT しても1件のまま"""
        self._insert("202505010511", 1)
        self._insert("202505010511", 1)
        self.assertEqual(self._count(), 1)

    def test_different_horse_num_allowed(self):
        """同じ race_id でも horse_num が違えば別レコード"""
        self._insert("202505010511", 1)
        self._insert("202505010511", 2)
        self.assertEqual(self._count(), 2)

    def test_different_race_id_allowed(self):
        """horse_num が同じでも race_id が違えば別レコード"""
        self._insert("202505010511", 1)
        self._insert("202505010512", 1)
        self.assertEqual(self._count(), 2)

    def test_multiple_duplicates_all_ignored(self):
        """複数回の重複 INSERT がすべて無視される"""
        for _ in range(5):
            self._insert("202506010101", 3)
        self.assertEqual(self._count(), 1)


# ---------------------------------------------------------------------------
# 追加: 漏れていたエッジケース
# ---------------------------------------------------------------------------

class TestParseWeightEdgeCases(unittest.TestCase):

    def test_hyphen_returns_none(self):
        """'-' は馬体重なし。(None, None) を返す"""
        self.assertEqual(_parse_weight("-"), (None, None))

    def test_double_hyphen_returns_none(self):
        self.assertEqual(_parse_weight("--"), (None, None))


class TestExtractRaceDateEdgeCases(unittest.TestCase):

    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_date_with_day_of_week_suffix(self):
        """'2025年5月4日（日）' のように曜日が続いても正しくパースされる"""
        soup = self._soup(
            "<html><head><title>2025年5月4日（日） 東京11R | netkeiba</title></head></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2025-05-04")

    def test_december_two_digit_month(self):
        """12月など2桁月もゼロパディングなしで正しく返る"""
        soup = self._soup(
            "<html><head><title>2024年12月22日 中山11R 有馬記念 | netkeiba</title></head></html>"
        )
        self.assertEqual(_extract_race_date(soup), "2024-12-22")


# ---------------------------------------------------------------------------
# fixture テスト: scrape_race() 全体をモック HTML で検証
# ---------------------------------------------------------------------------

def _make_mock_response(html: str, encoding: str = "UTF-8"):
    """requests.Response のモックを作成する"""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = 200
    resp.text = html
    resp.encoding = encoding
    return resp


# ── 実HTML相当の25列フィクスチャ (プリンシパルS 2024-05-04 東京 202405020511)
# 実際のnetkeiba HTML構造に基づき構築:
#   col0:着順 col1:枠番 col2:馬番 col3:馬名 col4:性齢 col5:斤量 col6:騎手
#   col7:タイム col8:着差 col9-13:タイム指数系(プレミアム/"**")
#   col14:通過 col15:上り col16:単勝 col17:人気 col18:馬体重
#   col19:調教タイム col20:厩舎ｺﾒﾝﾄ col21:備考 col22:調教師 col23:馬主 col24:賞金
_FIXTURE_RACE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>プリンシパルS&#65372;2024年5月4日 | 競馬データベース - netkeiba</title>
</head>
<body>
  <h1></h1>
  <h1>プリンシパルS(L)</h1>
  <div class="mainrace_data">
    <dl class="racedata fc">
      <dd>プリンシパルS(L)芝左2000m&#160;/&#160;
      天候 : 晴&#160;/&#160;
      芝 : 良&#160;&#160;/&#160;
      発走 : 15:45</dd>
    </dl>
  </div>
  <table class="race_table_01 nk_tb_common">
    <tr>
      <th>着順</th><th>枠番</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th>
      <th>騎手</th><th>タイム</th><th>着差</th>
      <th>ﾀｲﾑ指数</th><th>ﾀｲﾑ指数M</th><th>ｽﾀｰﾄ指数</th><th>追走指数</th><th>上がり指数</th>
      <th>通過</th><th>上り</th><th>単勝</th><th>人気</th><th>馬体重</th>
      <th>調教ﾀｲﾑ</th><th>厩舎ｺﾒﾝﾄ</th><th>備考</th><th>調教師</th><th>馬主</th><th>賞金(万円)</th>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>13</td>
      <td><a href="/horse/2021105817/">ダノンエアズロック</a></td>
      <td>牡3</td>
      <td>57</td>
      <td><a href="/jockey/result/recent/05509/">モレイラ</a></td>
      <td>1:59.6</td>
      <td></td>
      <td>**</td><td>**</td><td>**</td><td>**</td><td>**</td>
      <td>4-4-5</td>
      <td>33.4</td>
      <td>2.3</td>
      <td>1</td>
      <td>492(-12)</td>
      <td></td><td></td><td></td>
      <td><a href="/trainer/result/recent/01070/">[東]堀宣行</a></td>
      <td>ダノックス</td>
      <td>2028.7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td><a href="/horse/2021104791/">メリオーレム</a></td>
      <td>牡3</td>
      <td>57</td>
      <td><a href="/jockey/result/recent/05386/">戸崎圭太</a></td>
      <td>1:59.8</td>
      <td>1.1/4</td>
      <td>**</td><td>**</td><td>**</td><td>**</td><td>**</td>
      <td>4-6-5</td>
      <td>33.6</td>
      <td>6.2</td>
      <td>4</td>
      <td>484(-8)</td>
      <td></td><td></td><td></td>
      <td><a href="/trainer/result/recent/01061/">[西]友道康夫</a></td>
      <td>G1レーシング</td>
      <td>808.2</td>
    </tr>
  </table>
</body>
</html>"""

# ダートレース用フィクスチャ（馬場状態パターン検証）
_FIXTURE_DIRT_HTML = """<!DOCTYPE html>
<html>
<head><title>3歳未勝利&#65372;2024年4月20日 | 競馬データベース - netkeiba</title></head>
<body>
  <h1></h1>
  <h1>3歳未勝利</h1>
  <div class="mainrace_data">
    <dl class="racedata fc">
      <dd>3歳未勝利ダ左1600m&#160;/&#160;天候 : 晴&#160;/&#160;ダート : 良&#160;/&#160;発走 : 10:10</dd>
    </dl>
  </div>
  <table class="race_table_01 nk_tb_common">
    <tr>
      <th>着順</th><th>枠番</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th>
      <th>騎手</th><th>タイム</th><th>着差</th>
      <th>c9</th><th>c10</th><th>c11</th><th>c12</th><th>c13</th>
      <th>通過</th><th>上り</th><th>単勝</th><th>人気</th><th>馬体重</th>
      <th>c19</th><th>c20</th><th>c21</th><th>調教師</th><th>馬主</th><th>賞金</th>
    </tr>
    <tr>
      <td>1</td><td>3</td><td>5</td>
      <td><a href="/horse/2021100001/">ダートホース</a></td>
      <td>牝3</td><td>54</td>
      <td><a href="/jockey/result/recent/00001/">テスト騎手</a></td>
      <td>1:38.2</td><td></td>
      <td>**</td><td>**</td><td>**</td><td>**</td><td>**</td>
      <td>5-5</td><td>37.7</td><td>3.0</td><td>1</td><td>438(-8)</td>
      <td></td><td></td><td></td>
      <td><a href="/trainer/result/recent/00099/">[東]竹内正洋</a></td>
      <td>テストオーナー</td><td>100.0</td>
    </tr>
  </table>
</body>
</html>"""


class TestScrapeRaceFixture(unittest.TestCase):
    """25列実HTML相当モックで scrape_race() をエンドツーエンド検証。
    プリンシパルS (2024-05-04 東京 202405020511) の実データに基づくフィクスチャ。
    cols[8]=着差, cols[14]=通過, cols[15]=上り, cols[16]=単勝,
    cols[17]=人気, cols[18]=馬体重, cols[22]=調教師 を固定。"""

    def setUp(self):
        from unittest.mock import MagicMock
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scraper"))
        import netkeiba_scraper as scraper
        self.scraper = scraper

        session = MagicMock()
        session.get.return_value = _make_mock_response(_FIXTURE_RACE_HTML)
        self.result = scraper.scrape_race("202405020511", session)

    # ── レース情報

    def test_returns_race_data(self):
        """scrape_race() が None でなく dict を返す"""
        self.assertIsNotNone(self.result)

    def test_race_name_from_h1(self):
        """レース名が非空 <h1> から取得される（.RaceName 廃止対応）"""
        self.assertEqual(self.result["race_name"], "プリンシパルS(L)")

    def test_date_from_title(self):
        """date が <title> から正しく取得される"""
        self.assertEqual(self.result["date"], "2024-05-04")

    def test_distance(self):
        """距離が .mainrace_data から取得される"""
        self.assertEqual(self.result["distance"], 2000)

    def test_surface_turf(self):
        """芝レースの surface = '芝'"""
        self.assertEqual(self.result["surface"], "芝")

    def test_direction(self):
        self.assertEqual(self.result["direction"], "左")

    def test_track_condition_turf(self):
        """芝の馬場状態が '芝 : 良' から '良' として取得される"""
        self.assertEqual(self.result["track_condition"], "良")

    def test_num_horses(self):
        self.assertEqual(self.result["num_horses"], 2)

    # ── 馬1着: ダノンエアズロック 実データ固定

    def test_finish_pos(self):
        self.assertEqual(self.result["results"][0]["finish_pos"], 1)
        self.assertEqual(self.result["results"][1]["finish_pos"], 2)

    def test_margin_col8_first_empty(self):
        """1着 margin = cols[8] = '' (着差なし)"""
        self.assertEqual(self.result["results"][0]["margin"], "")

    def test_margin_col8_second(self):
        """2着 margin = cols[8] = '1.1/4'"""
        self.assertEqual(self.result["results"][1]["margin"], "1.1/4")

    def test_passing_order_col14(self):
        """passing_order = cols[14] = '4-4-5' (通過)"""
        self.assertEqual(self.result["results"][0]["passing_order"], "4-4-5")

    def test_passing_order_second(self):
        self.assertEqual(self.result["results"][1]["passing_order"], "4-6-5")

    def test_last_3f_col15(self):
        """last_3f = cols[15] = 33.4 (上り)"""
        self.assertAlmostEqual(self.result["results"][0]["last_3f"], 33.4)

    def test_odds_col16(self):
        """odds = cols[16] = 2.3 (単勝)"""
        self.assertAlmostEqual(self.result["results"][0]["odds"], 2.3)

    def test_popularity_col17(self):
        """popularity = cols[17] = 1 (人気)"""
        self.assertEqual(self.result["results"][0]["popularity"], 1)

    def test_weight_col18_not_passing_order(self):
        """weight = cols[18] = 492 (馬体重)。cols[14]='4-4-5'(通過) ではない"""
        w = self.result["results"][0]["weight"]
        self.assertEqual(w, 492)
        self.assertIsInstance(w, int)

    def test_weight_change_col18(self):
        """weight_change = -12 (cols[18]='492(-12)' の括弧内)"""
        self.assertEqual(self.result["results"][0]["weight_change"], -12)

    def test_weight_is_not_none_all_horses(self):
        """全馬の weight が None でないこと"""
        for h in self.result["results"]:
            self.assertIsNotNone(h["weight"], msg=f"weight が None: {h['horse_name']}")

    def test_trainer_col22_east_prefix_stripped(self):
        """trainer_name = cols[22] から '[東]' を除去した値"""
        self.assertEqual(self.result["results"][0]["trainer_name"], "堀宣行")

    def test_trainer_col22_west_prefix_stripped(self):
        """'[西]' プレフィックスも除去される"""
        self.assertEqual(self.result["results"][1]["trainer_name"], "友道康夫")

    def test_second_horse_weight_decrease(self):
        """2着 weight=484, weight_change=-8"""
        h2 = self.result["results"][1]
        self.assertEqual(h2["weight"], 484)
        self.assertEqual(h2["weight_change"], -8)

    def test_time_seconds_parsed(self):
        """'1:59.6' → 119.6秒"""
        self.assertAlmostEqual(self.result["results"][0]["time_seconds"], 119.6)

    def test_jockey_name(self):
        self.assertEqual(self.result["results"][0]["jockey_name"], "モレイラ")

    def test_horse_sex_and_age(self):
        """'牡3' → sex='牡', age=3"""
        h1 = self.result["results"][0]
        self.assertEqual(h1["sex"], "牡")
        self.assertEqual(h1["age"], 3)

    def test_horse_id_extracted(self):
        """馬IDがhrefから抽出される"""
        self.assertEqual(self.result["results"][0]["horse_id"], "2021105817")


class TestScrapeRaceFixtureDirt(unittest.TestCase):
    """ダートレースの馬場状態取得テスト（'ダート : 良' パターン）"""

    def setUp(self):
        from unittest.mock import MagicMock
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scraper"))
        import netkeiba_scraper as scraper

        session = MagicMock()
        session.get.return_value = _make_mock_response(_FIXTURE_DIRT_HTML)
        self.result = scraper.scrape_race("202405020101", session)

    def test_surface_dirt(self):
        """ダートレースの surface = 'ダ'"""
        self.assertEqual(self.result["surface"], "ダ")

    def test_track_condition_dirt(self):
        """ダートの馬場状態が 'ダート : 良' から '良' として取得される"""
        self.assertEqual(self.result["track_condition"], "良")

    def test_weight_dirt_race(self):
        """ダートレースでも weight が cols[18] から正しく取得される"""
        self.assertEqual(self.result["results"][0]["weight"], 438)
        self.assertEqual(self.result["results"][0]["weight_change"], -8)

    def test_trainer_dirt_race(self):
        self.assertEqual(self.result["results"][0]["trainer_name"], "竹内正洋")


if __name__ == "__main__":
    unittest.main()
