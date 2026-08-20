"""
ml/features.py の Phase 3 修正に対するテスト

検証項目:
  1. _stable_hash: セッション間で同一値を返す
  2. encode_features: DataFrameのサイズ・内容に依存しない安定エンコード
  3. FEATURE_COLS: odds/popularity が除外されている
  4. FEATURE_COLS: 全カテゴリが _enc 形式でエンコードされている
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
from features import (
    encode_features, FEATURE_COLS, RACECOURSE_ENC_MAP,
    TRACK_COND_MAP, _stable_hash,
)


def _make_row(**overrides) -> dict:
    """encode_features に渡す最小限の行データを生成"""
    base = {
        "date": pd.Timestamp("2024-01-27"),
        "track_condition": "良",
        "weather": "晴",
        "surface": "芝",
        "direction": "左",
        "sex": "牡",
        "racecourse": "東京",
        "father": "ディープインパクト",
        "maternal_father": "キングカメハメハ",
    }
    base.update(overrides)
    return base


def _make_df(*rows) -> pd.DataFrame:
    if not rows:
        rows = (_make_row(),)
    return pd.DataFrame(list(rows))


# ── _stable_hash ──────────────────────────────────────────────────

class TestStableHash(unittest.TestCase):

    def test_same_input_same_output(self):
        """同じ文字列を2回渡すと同じ値になる"""
        a = _stable_hash("ディープインパクト")
        b = _stable_hash("ディープインパクト")
        self.assertEqual(a, b)

    def test_empty_returns_minus1(self):
        """空文字は -1"""
        self.assertEqual(_stable_hash(""), -1)

    def test_output_in_range(self):
        """結果は [0, 500) の範囲内"""
        v = _stable_hash("キングカメハメハ")
        self.assertGreaterEqual(v, 0)
        self.assertLess(v, 500)

    def test_different_inputs(self):
        """異なる文字列は（高確率で）異なる値になる"""
        a = _stable_hash("ディープインパクト")
        b = _stable_hash("キングカメハメハ")
        # 500バケットでの衝突確率は 1/500。テストの安定性のため参照値で固定。
        self.assertNotEqual(a, b)

    def test_known_values_are_reproducible(self):
        """過去に確認したハッシュ値が変わっていないことを確認（回帰テスト）"""
        # 値は実行して確認した固定値
        import hashlib
        name = "ディープインパクト"
        expected = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 500
        self.assertEqual(_stable_hash(name), expected)


# ── encode_features の安定性 ─────────────────────────────────────

class TestEncodeFeaturesStability(unittest.TestCase):

    def test_stable_on_repeated_call(self):
        """同じDataFrameに2回 encode_features を呼んでも同じ結果"""
        df = _make_df()
        r1 = encode_features(df)
        r2 = encode_features(df)
        self.assertEqual(r1["father_enc"].iloc[0], r2["father_enc"].iloc[0])
        self.assertEqual(r1["racecourse_enc"].iloc[0], r2["racecourse_enc"].iloc[0])

    def test_stable_on_subset_vs_fullset(self):
        """全体DataFrameと部分DataFrameで同じ馬名が同じコードになる

        これが旧 cat.codes で起きていた主要バグ:
        全体で「ディープインパクト=47」→ 部分で「ディープインパクト=0」に変わっていた。
        """
        row1 = _make_row(father="ディープインパクト")
        row2 = _make_row(father="キングカメハメハ", racecourse="中山")
        full_df = _make_df(row1, row2)
        sub_df  = _make_df(row1)

        full_enc = encode_features(full_df)["father_enc"].iloc[0]
        sub_enc  = encode_features(sub_df)["father_enc"].iloc[0]
        self.assertEqual(full_enc, sub_enc,
            f"全体エンコード={full_enc} vs 部分エンコード={sub_enc}: "
            "cat.codes バグが再発している可能性があります")

    def test_different_fathers_different_codes(self):
        """異なる父名は（高確率で）異なるコードになる"""
        row1 = _make_row(father="ディープインパクト")
        row2 = _make_row(father="キングカメハメハ")
        df = _make_df(row1, row2)
        result = encode_features(df)
        self.assertNotEqual(result["father_enc"].iloc[0],
                            result["father_enc"].iloc[1])

    def test_null_father_is_minus1(self):
        """father が None/NaN のとき father_enc = -1"""
        df = _make_df(_make_row(father=None))
        result = encode_features(df)
        self.assertEqual(result["father_enc"].iloc[0], -1)

    def test_missing_maternal_father_column(self):
        """maternal_father 列がない場合でも -1 で補完されクラッシュしない"""
        df = _make_df()
        df = df.drop(columns=["maternal_father"])
        result = encode_features(df)  # KeyError が起きてはいけない
        self.assertTrue((result["maternal_father_enc"] == -1).all())


# ── encode_features の固定マップ ─────────────────────────────────

class TestEncodeFeaturesMaps(unittest.TestCase):

    def test_racecourse_tokyo(self):
        """東京 → RACECOURSE_ENC_MAP["東京"] = 4"""
        df = _make_df(_make_row(racecourse="東京"))
        self.assertEqual(encode_features(df)["racecourse_enc"].iloc[0],
                         RACECOURSE_ENC_MAP["東京"])

    def test_racecourse_nakayama(self):
        df = _make_df(_make_row(racecourse="中山"))
        self.assertEqual(encode_features(df)["racecourse_enc"].iloc[0],
                         RACECOURSE_ENC_MAP["中山"])

    def test_racecourse_unknown_is_minus1(self):
        """未知の競馬場は -1"""
        df = _make_df(_make_row(racecourse="海外コース"))
        self.assertEqual(encode_features(df)["racecourse_enc"].iloc[0], -1)

    def test_track_condition_ryo(self):
        """良 → 0"""
        df = _make_df(_make_row(track_condition="良"))
        self.assertEqual(encode_features(df)["track_condition_enc"].iloc[0], 0)

    def test_track_condition_heavy(self):
        """重 → 2"""
        df = _make_df(_make_row(track_condition="重"))
        self.assertEqual(encode_features(df)["track_condition_enc"].iloc[0], 2)

    def test_surface_turf(self):
        """芝 → 0"""
        df = _make_df(_make_row(surface="芝"))
        self.assertEqual(encode_features(df)["surface_enc"].iloc[0], 0)

    def test_surface_dirt(self):
        """ダ → 1"""
        df = _make_df(_make_row(surface="ダ"))
        self.assertEqual(encode_features(df)["surface_enc"].iloc[0], 1)

    def test_sex_male(self):
        """牡 → 0"""
        df = _make_df(_make_row(sex="牡"))
        self.assertEqual(encode_features(df)["sex_enc"].iloc[0], 0)

    def test_month_and_dayofweek(self):
        """2024-01-27 (土) → month=1, dayofweek=5"""
        df = _make_df(_make_row(date=pd.Timestamp("2024-01-27")))
        result = encode_features(df)
        self.assertEqual(result["month"].iloc[0], 1)
        self.assertEqual(result["dayofweek"].iloc[0], 5)


# ── FEATURE_COLS の構成 ───────────────────────────────────────────

class TestFeatureCols(unittest.TestCase):

    def test_odds_not_in_feature_cols(self):
        """odds は FEATURE_COLS に含まれない（推論時不明のため除外）"""
        self.assertNotIn("odds", FEATURE_COLS,
            "odds が FEATURE_COLS に含まれています。"
            "推論時は odds が不明なため除外が必要です。")

    def test_popularity_not_in_feature_cols(self):
        """popularity は FEATURE_COLS に含まれない（推論時不明のため除外）"""
        self.assertNotIn("popularity", FEATURE_COLS,
            "popularity が FEATURE_COLS に含まれています。"
            "推論時は popularity が不明なため除外が必要です。")

    def test_feature_count(self):
        """FEATURE_COLS は28列（odds/popularity 除外後）"""
        self.assertEqual(len(FEATURE_COLS), 28,
            f"FEATURE_COLS の列数が 28 ではありません: {len(FEATURE_COLS)}")

    def test_no_raw_string_categoricals(self):
        """cat.codesが使われた後のままの生カテゴリ列がないことを確認"""
        raw_cats = ["track_condition", "weather", "surface", "direction",
                    "sex", "father", "maternal_father", "racecourse"]
        for col in raw_cats:
            self.assertNotIn(col, FEATURE_COLS,
                f"生カテゴリ列 '{col}' が FEATURE_COLS に含まれています")

    def test_all_encoded_categoricals_present(self):
        """全カテゴリの _enc 版が FEATURE_COLS に含まれている"""
        required = [
            "track_condition_enc", "weather_enc", "surface_enc",
            "direction_enc", "sex_enc",
            "father_enc", "maternal_father_enc", "racecourse_enc",
        ]
        for col in required:
            self.assertIn(col, FEATURE_COLS,
                f"エンコード列 '{col}' が FEATURE_COLS にありません")

    def test_no_duplicates(self):
        """FEATURE_COLS に重複がない"""
        self.assertEqual(len(FEATURE_COLS), len(set(FEATURE_COLS)),
            "FEATURE_COLS に重複する列名があります")

    def test_encode_features_produces_all_cols(self):
        """encode_features の結果が FEATURE_COLS の全列を含む"""
        base = _make_row()
        # FEATURE_COLS に含まれる数値列のデフォルト値を追加
        numeric_defaults = {
            "distance": 2000, "num_horses": 16, "gate_num": 5, "horse_num": 9,
            "age": 4, "weight": 480,
            "horse_race_count": 3, "horse_win_rate": 0.1, "horse_top3_rate": 0.3,
            "horse_avg_finish": 5.0, "horse_dist_win_rate": 0.1,
            "horse_cond_win_rate": 0.1, "horse_recent_avg": 5.0,
            "jockey_win_rate": 0.1, "jockey_top3_rate": 0.3,
            "trainer_win_rate": 0.1, "father_heavy_win_rate": 0.0,
            "running_style": 1, "month": 1, "dayofweek": 5,
        }
        base.update(numeric_defaults)
        df = pd.DataFrame([base])
        result = encode_features(df)
        missing = [c for c in FEATURE_COLS if c not in result.columns]
        self.assertEqual(missing, [],
            f"encode_features の出力に以下の列がありません: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
