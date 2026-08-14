# 複数PC開発・共有storage方針

## 正本

GitHub `main`を、コード、依存定義、tests、設計、task状態、AI handoffの正本とする。PCローカルのデータ・model・OneDrive同期フォルダを唯一の正本にしてはならない。

実データとartifactの正本は、将来承認する共有Google DriveまたはGoogle Cloud Storageに置く。各artifactには少なくともversion、作成時刻、SHA-256、作成元experiment、保存先URIを記録する。

## 新PCでの再開

`README.md`の手順でcloneと依存導入を行い、`docs/AI_HANDOFF.md`と`tasks/CURRENT_TASK.md`を読む。DB/modelが必要なtaskだけ、共有storageを同期/マウントして以下をOS環境変数として設定する。

- `KEIBA_DATA_DIR`、`KEIBA_MODEL_DIR`
- 必要なら`KEIBA_DB_PATH`、`KEIBA_REGISTRY_DB_PATH`

Python codeは`project_paths.py`を通じてこれらを使う。値は`.env`に置いてもよいが、現時点でPythonは`.env`を自動読込しない。`.env`と認証情報はGitに置かない。

## 現在できること

- Windowsでclone後、Python依存を導入してregistry testsを実行できる。
- frontendは`frontend/.env.example`と`npm install`で起動準備できる。
- Colab用`colab_training.py`は`KEIBA_SHARED_ROOT`で共有Drive rootを指定でき、model artifactをGitHubへpushしない方針を示す。

## 未解決

- `requirements.txt`は範囲指定であり、hash付きlock fileではない。
- frontendには`package-lock.json`/`pnpm-lock.yaml`がなく、Node依存の完全な再現性は未達。
- `training.ipynb`には旧来のDrive default pathが残る。artifactをGitHubへpushするcellは無効化したが、Phase 3より前に`KEIBA_SHARED_ROOT`に統一する別taskが必要。
- Google Drive/GCSの共有フォルダ、IAM、同期手順、artifact manifest形式は未決定。
- 実DB/modelを置く共有storageはまだ提供されていない。

## 禁止

artifact、DB、model、tokenをGitHubへcommitしない。結果登録だけで学習・production昇格を行わない。共有storageからの大量取得、同期設定変更、デプロイ設定変更は承認済みtaskだけで行う。
