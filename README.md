# 競馬予想AI研究基盤

コード・設計・AI引継ぎ状態の正本はGitHubです。実データ、学習済みモデル、統計cacheなどの大型artifactはGitへ置かず、承認済みの共有Google Drive / Google Cloud storageへ置きます。

## 新PCの安全なセットアップ

1. `git clone https://github.com/yamaguchikei88-web/keiba-ai.git`
2. Python 3.11で `python -m pip install -r requirements.txt` を実行する。
3. Node.js 20 LTS以上で `cd frontend`、`npm install` を実行する。
4. `.env.example` を `.env` へ、`frontend/.env.example` を `frontend/.env.local` へコピーし、共有storageまたは接続先を設定する。
5. artifactが必要なtaskでは、承認済みの共有storageを同期/マウントし、`KEIBA_DATA_DIR`と`KEIBA_MODEL_DIR`（または`KEIBA_DB_PATH`）を設定する。artifactがない状態ではスクレイパー、学習、予測APIを実行しない。
6. `python -m unittest discover -s tests -v` を実行する。開発再開前に `docs/AI_HANDOFF.md` と `tasks/CURRENT_TASK.md` を読む。

`setup.bat`は依存関係を入れるだけで、データ収集や学習は実行しません。

## 環境変数

| Name | Purpose | Default |
|---|---|---|
| `KEIBA_DATA_DIR` | 非Git管理のデータ/registry directory | `<repo>/data` |
| `KEIBA_MODEL_DIR` | 非Git管理のmodel artifact directory | `<repo>/models` |
| `KEIBA_DB_PATH` | race DBの個別パス | `KEIBA_DATA_DIR/keiba.db` |
| `KEIBA_REGISTRY_DB_PATH` | research registry DBの個別パス | `KEIBA_DATA_DIR/research_registry.db` |
| `NEXT_PUBLIC_API_URL` | frontendが呼ぶAPI URL | `http://localhost:8000` |

Colabでは`KEIBA_SHARED_ROOT`で共有Drive上のartifact rootを指定できる。既定の`/content/drive/...`はColabのマウントパスであり、PCローカルpathではないが、共有先はチームで合意してから設定する。

`.env`はPython標準では自動読込されません。OS/シェルで設定するか、将来承認された設定ローダーを導入してください。認証情報はGitへコミットしません。

## 資産の配置

- GitHub: ソース、tests、依存定義、設計、migration、AI handoff、task状態。
- 共有storage: SQLite snapshotとhash/manifest、versioned model artifact、stats cache、model metadata、入力snapshot、実験出力、バックアップ。
- PCローカル: virtualenv、`node_modules`、`.next`、`__pycache__`、一時download、共有storageの同期キャッシュ。唯一の正本にしない。

## 現在の制約

このcloneには実DB・学習済みモデルがありません。`/result/register`の自動再学習/固定path上書きは既存挙動で、Phase 2 registryへまだ接続されていません。モデル採用・データ取得・本番設定変更は、別taskで手動承認してから実施します。
