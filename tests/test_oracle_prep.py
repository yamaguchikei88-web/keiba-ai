"""
M1-M7 Oracle Cloud VM 移行準備コードの検証テスト

M1: jobs/weekly_scrape.sh  ― 存在確認・構文チェック・CLIオプション検証
M2: jobs/monthly_train.sh  ― 存在確認・構文チェック・CLIオプション検証
M3: jobs/backup.sh         ― 存在確認・構文チェック
M4: .env.example / project_paths.py / netkeiba_scraper.py の FAILED_LOG_PATH 対応
M5: ml/train.py の _resolve_model_paths() バージョニング
M6: ml/predict.py の _resolve_load_paths() / reload_model() / load_model() バージョン管理
M7: api/main.py の /result/register が自動再学習を起動しない
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from project_paths import MODEL_DIR, model_path


# ─────────────────────────────────────────────────────────────────────────────
# M1 / M2 / M3: shell scripts
# ─────────────────────────────────────────────────────────────────────────────

class TestShellScripts(unittest.TestCase):
    """M1, M2, M3: jobs/*.sh の存在・構文・CLIオプション検証"""

    def test_weekly_scrape_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "jobs" / "weekly_scrape.sh").exists(),
            "jobs/weekly_scrape.sh が存在しません",
        )

    def test_monthly_train_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "jobs" / "monthly_train.sh").exists(),
            "jobs/monthly_train.sh が存在しません",
        )

    def test_backup_exists(self):
        self.assertTrue(
            (PROJECT_ROOT / "jobs" / "backup.sh").exists(),
            "jobs/backup.sh が存在しません",
        )

    @unittest.skipUnless(shutil.which("bash"), "bash が利用できません")
    def test_weekly_scrape_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "jobs" / "weekly_scrape.sh")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"weekly_scrape.sh 構文エラー: {result.stderr}")

    @unittest.skipUnless(shutil.which("bash"), "bash が利用できません")
    def test_monthly_train_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "jobs" / "monthly_train.sh")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"monthly_train.sh 構文エラー: {result.stderr}")

    @unittest.skipUnless(shutil.which("bash"), "bash が利用できません")
    def test_backup_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "jobs" / "backup.sh")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"backup.sh 構文エラー: {result.stderr}")

    def test_weekly_scrape_uses_correct_sleep_option(self):
        """weekly_scrape.sh は実在する --sleep オプションを使う（--sleep-seconds は存在しない）"""
        source = (PROJECT_ROOT / "jobs" / "weekly_scrape.sh").read_text(encoding="utf-8")
        self.assertNotIn("--sleep-seconds", source, "存在しない --sleep-seconds を使っています")
        self.assertIn("--sleep-calendar", source)

    def test_weekly_scrape_has_date_range_options(self):
        source = (PROJECT_ROOT / "jobs" / "weekly_scrape.sh").read_text(encoding="utf-8")
        for opt in ["--start-year", "--end-year", "--start-date", "--end-date"]:
            self.assertIn(opt, source, f"weekly_scrape.sh に {opt} がありません")

    def test_monthly_train_uses_correct_pipeline_options(self):
        """monthly_train.sh は実在する pipeline CLI オプションを使う"""
        source = (PROJECT_ROOT / "jobs" / "monthly_train.sh").read_text(encoding="utf-8")
        for opt in ["--model-version", "--feature-set-version", "--windows", "--json"]:
            self.assertIn(opt, source, f"monthly_train.sh に {opt} がありません")
        self.assertNotIn("--model_version", source, "ハイフンでなくアンダースコアを使っています")

    def test_monthly_train_sets_model_version_env(self):
        """monthly_train.sh は MODEL_VERSION 環境変数をエクスポートする"""
        source = (PROJECT_ROOT / "jobs" / "monthly_train.sh").read_text(encoding="utf-8")
        self.assertIn("MODEL_VERSION", source)
        self.assertIn("export MODEL_VERSION", source)

    def test_backup_uses_sqlite3_backup(self):
        """backup.sh は cp ではなく sqlite3 の .backup コマンドを使う"""
        source = (PROJECT_ROOT / "jobs" / "backup.sh").read_text(encoding="utf-8")
        self.assertIn(".backup", source, "sqlite3 の .backup コマンドを使っていません")

    def test_all_scripts_have_set_euo_pipefail(self):
        for script in ["weekly_scrape.sh", "monthly_train.sh", "backup.sh"]:
            source = (PROJECT_ROOT / "jobs" / script).read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", source, f"{script} に set -euo pipefail がありません")


# ─────────────────────────────────────────────────────────────────────────────
# M4: 環境変数設計
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvDesign(unittest.TestCase):
    """M4: .env.example / project_paths.py / scraper の FAILED_LOG_PATH 対応"""

    def test_env_example_has_all_required_vars(self):
        env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for var in [
            "KEIBA_DATA_DIR",
            "KEIBA_DB_PATH",
            "KEIBA_REGISTRY_DB_PATH",
            "KEIBA_MODEL_DIR",
            "KEIBA_FAILED_LOG_PATH",
            "KEIBA_PRODUCTION_VERSION_PATH",
        ]:
            self.assertIn(var, env_example, f".env.example に {var} がありません")

    def test_env_example_has_oracle_vm_path_examples(self):
        env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("/data/keiba-ai", env_example, ".env.example に Oracle VM パス例がありません")

    def test_project_paths_exports_failed_log_path(self):
        from project_paths import FAILED_LOG_PATH, DATA_DIR
        self.assertEqual(FAILED_LOG_PATH.parent, DATA_DIR)
        self.assertEqual(FAILED_LOG_PATH.name, "failed_races.jsonl")

    def test_scraper_imports_failed_log_path(self):
        scraper_source = (PROJECT_ROOT / "scraper" / "netkeiba_scraper.py").read_text(encoding="utf-8")
        self.assertIn("FAILED_LOG_PATH", scraper_source)
        self.assertIn("from project_paths import", scraper_source)

    def test_scraper_uses_failed_log_path_as_default(self):
        """scraper は DB_PATH.parent 直書きではなく FAILED_LOG_PATH を使う"""
        scraper_source = (PROJECT_ROOT / "scraper" / "netkeiba_scraper.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "DB_PATH.parent / \"failed_races.jsonl\"", scraper_source,
            "FAILED_LOG_PATH を使わずに DB_PATH.parent を直書きしています",
        )


# ─────────────────────────────────────────────────────────────────────────────
# M5: ml/train.py モデルバージョニング
# ─────────────────────────────────────────────────────────────────────────────

class TestModelVersioning(unittest.TestCase):
    """M5: train.py の _resolve_model_paths() バージョニング"""

    def setUp(self):
        import train as train_module
        self.train = train_module

    def test_resolve_model_paths_no_version_returns_defaults(self):
        """バージョン未指定時はデフォルトファイル名を返す"""
        mp, sp, jp = self.train._resolve_model_paths("")
        self.assertEqual(mp.name, "keiba_lgbm.pkl")
        self.assertEqual(sp.name, "stats_cache.pkl")
        self.assertEqual(jp.name, "model_meta.json")

    def test_resolve_model_paths_with_version_returns_versioned_names(self):
        """バージョン指定時はバージョン付きファイル名を返す（MODEL_VERSION=v202509 → keiba_lgbm_v202509.pkl）"""
        mp, sp, jp = self.train._resolve_model_paths("v202509")
        self.assertEqual(mp.name, "keiba_lgbm_v202509.pkl")   # f"keiba_lgbm_{version}.pkl" with version="v202509"
        self.assertEqual(sp.name, "stats_cache_v202509.pkl")
        self.assertEqual(jp.name, "model_meta_v202509.json")

    def test_resolve_model_paths_all_in_model_dir(self):
        """バージョン付きファイルも MODEL_DIR 配下に保存される"""
        mp, sp, jp = self.train._resolve_model_paths("v202501")
        self.assertEqual(mp.parent, MODEL_DIR)
        self.assertEqual(sp.parent, MODEL_DIR)
        self.assertEqual(jp.parent, MODEL_DIR)

    def test_resolve_model_paths_default_also_in_model_dir(self):
        mp, sp, jp = self.train._resolve_model_paths("")
        self.assertEqual(mp.parent, MODEL_DIR)
        self.assertEqual(sp.parent, MODEL_DIR)
        self.assertEqual(jp.parent, MODEL_DIR)

    def test_train_py_reads_model_version_env(self):
        """train.py のソースに os.environ.get('MODEL_VERSION') が含まれる"""
        source = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")
        self.assertIn("MODEL_VERSION", source)
        self.assertIn("_resolve_model_paths", source)

    def test_train_py_versioned_save_only(self):
        """バージョン指定時にデフォルトパス (keiba_lgbm.pkl) を上書きしない

        学習=即本番 のアンチパターン（dual-save）が存在しないことを検証する。
        本番昇格は production_version.txt への書き込みで行う。
        """
        source = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")
        # モジュールレベル定数は存在する（_resolve_model_paths のデフォルト用）
        self.assertIn("MODEL_PATH", source)
        # dual-save パターンが存在しない（学習したら即本番になるアンチパターン）
        self.assertNotIn("joblib.dump(model, MODEL_PATH)", source)
        # 本番昇格方法の案内メッセージが存在する
        self.assertIn("production_version.txt", source)


# ─────────────────────────────────────────────────────────────────────────────
# M6: ml/predict.py バージョン管理・キャッシュ無効化
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictVersioning(unittest.TestCase):
    """M6: predict.py の _get_effective_version / _resolve_load_paths / キャッシュ制御"""

    # ── ヘルパー ──────────────────────────────────────────────────

    @staticmethod
    def _pvp(exists: bool, version: str = "v202601"):
        """production_version.txt モックを返すコンテキストマネージャ"""
        m = MagicMock()
        m.exists.return_value = exists
        m.read_text.return_value = version + "\n"
        return patch("predict.PRODUCTION_VERSION_PATH", m)

    def setUp(self):
        import predict as predict_module
        self.predict = predict_module
        self.predict.reload_model()
        os.environ.pop("MODEL_VERSION", None)

    def tearDown(self):
        self.predict.reload_model()
        os.environ.pop("MODEL_VERSION", None)

    # ── _get_effective_version ────────────────────────────────────

    def test_get_effective_version_raises_without_file(self):
        """production_version.txt がない場合は RuntimeError を送出する（安全側に倒す）"""
        with self._pvp(exists=False):
            with self.assertRaises(RuntimeError):
                self.predict._get_effective_version()

    def test_get_effective_version_raises_when_file_empty(self):
        """空の production_version.txt でも RuntimeError を送出する"""
        m = MagicMock()
        m.exists.return_value = True
        m.read_text.return_value = "   "
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with self.assertRaises(RuntimeError):
                self.predict._get_effective_version()

    def test_get_effective_version_reads_from_file(self):
        """production_version.txt に 'v202601\\n' があれば 'v202601' を返す"""
        with self._pvp(exists=True, version="v202601"):
            result = self.predict._get_effective_version()
        self.assertEqual(result, "v202601")

    def test_get_effective_version_ignores_model_version_env(self):
        """MODEL_VERSION 環境変数は参照しない（production_version.txt のみ有効）"""
        with self._pvp(exists=True, version="v202601"):
            with patch.dict(os.environ, {"MODEL_VERSION": "v202699"}):
                result = self.predict._get_effective_version()
        self.assertEqual(result, "v202601")  # env var は無視

    # ── _resolve_load_paths ───────────────────────────────────────

    def test_resolve_load_paths_raises_without_production_version_file(self):
        """production_version.txt がない場合は RuntimeError を送出する"""
        with self._pvp(exists=False):
            with self.assertRaises(RuntimeError):
                self.predict._resolve_load_paths()

    def test_resolve_load_paths_raises_when_version_files_missing(self):
        """production_version.txt が v999 を指定しているがモデルファイルが存在しない → FileNotFoundError"""
        with self._pvp(exists=True, version="v999"):
            with self.assertRaises(FileNotFoundError):
                self.predict._resolve_load_paths()

    def test_resolve_load_paths_raises_when_only_model_and_stats_exist(self):
        """model + stats は存在するが meta がない場合も本番採用を拒否する"""
        with self._pvp(exists=True, version="v202601"):
            with patch("predict.model_path") as mock_mp:
                def fake_path(name):
                    m = MagicMock(spec=Path)
                    # meta (.json) だけ存在しない
                    m.exists.return_value = not name.endswith(".json")
                    m.__str__ = lambda self: f"/models/{name}"
                    return m
                mock_mp.side_effect = fake_path
                with self.assertRaises(FileNotFoundError):
                    self.predict._resolve_load_paths()

    # ── reload_model / load_model キャッシュ制御 ─────────────────

    def test_reload_model_clears_all_cache(self):
        """reload_model() でモデル・統計・バージョンキャッシュが全てクリアされる"""
        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=(Path("/f/m.pkl"), Path("/f/s.pkl"))):
                with patch("predict.joblib.load", side_effect=[MagicMock(), MagicMock()]):
                    self.predict.load_model()
        self.assertIsNotNone(self.predict._model)

        self.predict.reload_model()
        self.assertIsNone(self.predict._model)
        self.assertIsNone(self.predict._stats)
        self.assertIsNone(self.predict._cache_version)

    def test_load_model_uses_cache_on_second_call(self):
        """2回目の load_model() は joblib.load を再度呼ばない（キャッシュ）"""
        mock_model = MagicMock(name="model")
        mock_stats = MagicMock(name="stats")
        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=(Path("/f/m.pkl"), Path("/f/s.pkl"))):
                with patch("predict.joblib.load", side_effect=[mock_model, mock_stats]) as mock_load:
                    m1, s1 = self.predict.load_model()
                    m2, s2 = self.predict.load_model()
                    self.assertEqual(mock_load.call_count, 2)  # 初回 2 ファイルのみ
        self.assertIs(m1, m2)
        self.assertIs(s1, s2)

    def test_load_model_reloads_on_production_version_change(self):
        """production_version.txt が v202601 → v202602 に変わるとキャッシュを無効化して再ロード"""
        mock_v1 = MagicMock(name="model_v1")
        mock_v2 = MagicMock(name="model_v2")
        fake_paths = (Path("/f/m.pkl"), Path("/f/s.pkl"))

        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v1, MagicMock()]):
                    m_before, _ = self.predict.load_model()
        self.assertIs(m_before, mock_v1)

        with patch("predict._get_effective_version", return_value="v202602"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v2, MagicMock()]):
                    m_after, _ = self.predict.load_model()
        self.assertIs(m_after, mock_v2)

    def test_load_model_does_not_reload_same_production_version(self):
        """production_version.txt の内容が変わらなければキャッシュを無効化しない"""
        mock_model = MagicMock()
        fake_paths = (Path("/f/m.pkl"), Path("/f/s.pkl"))
        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_model, MagicMock()]) as mock_load:
                    self.predict.load_model()
                    self.predict.load_model()  # 2回目
                    self.assertEqual(mock_load.call_count, 2)  # 初回のみ

    def test_predict_py_has_cache_version_tracking(self):
        source = (PROJECT_ROOT / "ml" / "predict.py").read_text(encoding="utf-8")
        self.assertIn("_cache_version", source)
        self.assertIn("reload_model", source)
        self.assertIn("_resolve_load_paths", source)


# ─────────────────────────────────────────────────────────────────────────────
# production_version.txt による本番モデル管理（モデル管理レビュー追加）
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionVersionFile(unittest.TestCase):
    """本番モデル保護の最終安全性テスト

    設計原則:
      - production_version.txt が唯一の本番切り替え操作
      - train.py 実行では production_version.txt は絶対に変更されない
      - MODEL_VERSION 環境変数は predict.py では参照しない
      - 3 ファイル (pkl + stats + meta) が揃わなければ本番採用しない
      - set_production_version() のみが安全なアトミック書き込みを提供する
    """

    def setUp(self):
        import predict as predict_module
        self.predict = predict_module
        self.predict.reload_model()
        os.environ.pop("MODEL_VERSION", None)

    def tearDown(self):
        self.predict.reload_model()
        os.environ.pop("MODEL_VERSION", None)

    # ── production_version.txt がない / 空 → エラー ────────────

    def test_no_production_version_file_raises_runtime_error(self):
        """production_version.txt がない場合は RuntimeError を送出する（安全側に倒す）"""
        m = MagicMock()
        m.exists.return_value = False
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with self.assertRaises(RuntimeError):
                self.predict._get_effective_version()

    def test_empty_production_version_file_raises_runtime_error(self):
        """空の production_version.txt でも RuntimeError を送出する"""
        m = MagicMock()
        m.exists.return_value = True
        m.read_text.return_value = "  \n  "
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with self.assertRaises(RuntimeError):
                self.predict._get_effective_version()

    # ── train.py は production_version.txt を触らない ───────────

    def test_train_does_not_reference_production_version_path(self):
        """train.py は PRODUCTION_VERSION_PATH をインポートも参照もしない"""
        source = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("PRODUCTION_VERSION_PATH", source)

    def test_train_does_not_write_production_version_txt(self):
        """train.py は production_version.txt への書き込み操作を持たない"""
        source = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("set_production_version", source)
        # production_version.txt への write_text / open は存在しない（ログ文言を除く）
        lines_with_pvt = [
            line for line in source.splitlines()
            if "production_version.txt" in line and "write" in line.lower()
        ]
        self.assertEqual(lines_with_pvt, [], "train.py が production_version.txt に書き込んでいます")

    # ── MODEL_VERSION env は本番モデルを変えない ────────────────

    def test_model_version_env_alone_does_not_change_production_model(self):
        """MODEL_VERSION=v202699 をセットしても production_version.txt がなければエラーのまま"""
        m = MagicMock()
        m.exists.return_value = False
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with patch.dict(os.environ, {"MODEL_VERSION": "v202699"}):
                with self.assertRaises(RuntimeError):
                    self.predict._get_effective_version()

    def test_model_version_env_does_not_override_production_version_txt(self):
        """production_version.txt が 'v202601' のとき MODEL_VERSION='v202699' は無視される"""
        m = MagicMock()
        m.exists.return_value = True
        m.read_text.return_value = "v202601\n"
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with patch.dict(os.environ, {"MODEL_VERSION": "v202699"}):
                result = self.predict._get_effective_version()
        self.assertEqual(result, "v202601")

    # ── 3 ファイル揃わなければ本番採用しない ────────────────────

    def test_missing_model_files_raises_file_not_found(self):
        """production_version.txt が v999 を指定しているがファイルが存在しない → FileNotFoundError"""
        m = MagicMock()
        m.exists.return_value = True
        m.read_text.return_value = "v999\n"
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with self.assertRaises(FileNotFoundError):
                self.predict._resolve_load_paths()

    def test_partial_model_files_raises_file_not_found(self):
        """model + stats は存在するが meta がない場合も本番採用を拒否する"""
        m = MagicMock()
        m.exists.return_value = True
        m.read_text.return_value = "v202601\n"
        with patch("predict.PRODUCTION_VERSION_PATH", m):
            with patch("predict.model_path") as mock_mp:
                def fake_path(name):
                    p = MagicMock(spec=Path)
                    p.exists.return_value = not name.endswith(".json")
                    p.__str__ = lambda self: f"/models/{name}"
                    return p
                mock_mp.side_effect = fake_path
                with self.assertRaises(FileNotFoundError):
                    self.predict._resolve_load_paths()

    # ── production_version.txt 切り替え → キャッシュ無効化 ──────

    def test_production_version_change_invalidates_cache(self):
        """production_version.txt を v202601 → v202602 に変えると次の load_model で再ロードされる"""
        mock_v1 = MagicMock(name="model_v1")
        mock_v2 = MagicMock(name="model_v2")
        fake_paths = (Path("/f/m.pkl"), Path("/f/s.pkl"))

        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v1, MagicMock()]):
                    m1, _ = self.predict.load_model()

        with patch("predict._get_effective_version", return_value="v202602"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v2, MagicMock()]):
                    m2, _ = self.predict.load_model()

        self.assertIsNot(m1, m2)
        self.assertIs(m2, mock_v2)

    def test_rollback_invalidates_cache_and_reloads_old_version(self):
        """production_version.txt を v202602 → v202601 に戻すと旧バージョンが再ロードされる"""
        fake_paths = (Path("/f/m.pkl"), Path("/f/s.pkl"))
        mock_v1_1st = MagicMock(name="model_v1_1st")
        mock_v2 = MagicMock(name="model_v2")
        mock_v1_2nd = MagicMock(name="model_v1_2nd")

        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v1_1st, MagicMock()]):
                    m_v1_before, _ = self.predict.load_model()

        with patch("predict._get_effective_version", return_value="v202602"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v2, MagicMock()]):
                    self.predict.load_model()

        with patch("predict._get_effective_version", return_value="v202601"):
            with patch("predict._resolve_load_paths", return_value=fake_paths):
                with patch("predict.joblib.load", side_effect=[mock_v1_2nd, MagicMock()]):
                    m_rolled_back, _ = self.predict.load_model()

        self.assertIs(m_rolled_back, mock_v1_2nd)

    # ── set_production_version (atomic write) ────────────────────

    def test_set_production_version_rejects_invalid_version_strings(self):
        """不正なバージョン文字列は ValueError を送出する"""
        import predict as p
        for invalid in ["", " ", "v 202601", "../etc/passwd", "v202601!@#"]:
            with self.subTest(version=invalid):
                with self.assertRaises(ValueError):
                    p.set_production_version(invalid)

    def test_set_production_version_rejects_missing_model_files(self):
        """モデルファイルが存在しない場合は FileNotFoundError を送出して切り替えを拒否する"""
        import predict as p, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pvp = tmpdir_path / "production_version.txt"
            with patch("predict.PRODUCTION_VERSION_PATH", pvp):
                with patch("predict.model_path", side_effect=lambda n: tmpdir_path / n):
                    with self.assertRaises(FileNotFoundError):
                        p.set_production_version("v999")

    def test_set_production_version_writes_atomically(self):
        """set_production_version は .tmp 経由のアトミックな書き込みを行う（tempdir で実証）"""
        import predict as p, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pvp = tmpdir_path / "production_version.txt"
            # モデル 3 ファイルを作成
            for name in ["keiba_lgbm_v202601.pkl", "stats_cache_v202601.pkl", "model_meta_v202601.json"]:
                (tmpdir_path / name).touch()
            with patch("predict.PRODUCTION_VERSION_PATH", pvp):
                with patch("predict.model_path", side_effect=lambda n: tmpdir_path / n):
                    p.set_production_version("v202601")
            self.assertTrue(pvp.exists())
            self.assertEqual(pvp.read_text(encoding="utf-8").strip(), "v202601")
            # .tmp ファイルが残っていない（rename 完了）
            self.assertFalse((tmpdir_path / "production_version.txt.tmp").exists())

    def test_set_production_version_source_has_atomic_rename(self):
        """predict.py のソースに tmp 経由のアトミックな書き込みパターンが存在する"""
        source = (PROJECT_ROOT / "ml" / "predict.py").read_text(encoding="utf-8")
        self.assertIn("production_version.txt.tmp", source)
        self.assertIn(".replace(", source)  # Path.replace = atomic rename

    # ── ソースコード構造確認 ─────────────────────────────────────

    def test_predict_source_has_set_production_version(self):
        """predict.py に set_production_version() が実装されている"""
        source = (PROJECT_ROOT / "ml" / "predict.py").read_text(encoding="utf-8")
        self.assertIn("def set_production_version(", source)
        self.assertIn("_VERSION_RE", source)

    def test_project_paths_exports_production_version_path(self):
        """project_paths.py が PRODUCTION_VERSION_PATH をエクスポートする"""
        from project_paths import PRODUCTION_VERSION_PATH
        self.assertIn("production_version.txt", str(PRODUCTION_VERSION_PATH))

    def test_predict_py_imports_production_version_path(self):
        """predict.py が project_paths から PRODUCTION_VERSION_PATH をインポートする"""
        source = (PROJECT_ROOT / "ml" / "predict.py").read_text(encoding="utf-8")
        self.assertIn("PRODUCTION_VERSION_PATH", source)
        self.assertIn("_get_effective_version", source)


# ─────────────────────────────────────────────────────────────────────────────
# M7: api/main.py 自動再学習の分離
# ─────────────────────────────────────────────────────────────────────────────

class TestApiNoAutoRetrain(unittest.TestCase):
    """M7: /result/register が自動再学習を起動しない"""

    @classmethod
    def _get_register_result_body(cls) -> str:
        """api/main.py から register_result 関数の本体を抽出して返す"""
        source = (PROJECT_ROOT / "api" / "main.py").read_text(encoding="utf-8")
        func_start = source.find("def register_result(")
        assert func_start != -1, "register_result 関数が見つかりません"
        # 次の @app. または def で終端を探す
        func_end = len(source)
        for marker in ["\n@app.", "\ndef ", "\nif __name__"]:
            pos = source.find(marker, func_start + 1)
            if pos != -1:
                func_end = min(func_end, pos)
        return source[func_start:func_end]

    def test_no_threading_thread_in_register_result(self):
        """register_result に threading.Thread の呼び出しが含まれない"""
        body = self._get_register_result_body()
        self.assertNotIn("threading.Thread", body)
        self.assertNotIn("Thread(", body)

    def test_no_retrain_call_in_register_result(self):
        """register_result から retrain_with_new_data が呼ばれない"""
        body = self._get_register_result_body()
        self.assertNotIn("retrain_with_new_data", body)

    def test_db_write_is_preserved_in_register_result(self):
        """DB 書き込み（UPDATE race_results）は削除されていない"""
        body = self._get_register_result_body()
        self.assertIn("UPDATE race_results", body)

    def test_register_result_returns_accepted_status(self):
        """register_result は status='accepted' を返す"""
        body = self._get_register_result_body()
        self.assertIn("\"status\": \"accepted\"", body)

    def test_retrain_with_new_data_still_in_train_py(self):
        """retrain_with_new_data() は train.py に残っている（削除していない）"""
        source = (PROJECT_ROOT / "ml" / "train.py").read_text(encoding="utf-8")
        self.assertIn("def retrain_with_new_data(", source)


if __name__ == "__main__":
    unittest.main()
