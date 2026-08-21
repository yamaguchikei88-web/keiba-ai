# Oracle Cloud VM 移行手順

競馬予想AI (keiba-ai) を Oracle Cloud Always Free VM へ移行する手順。

## 前提条件

| 項目 | 値 |
|---|---|
| VM スペック | A1 ARM 4 OCPU / 24 GB RAM（Always Free） |
| OS | Ubuntu 22.04 LTS |
| ブロックボリューム | 50 GB（/data/keiba-ai マウント） |
| Python | 3.11 以上 |
| GitHub リポジトリ | keiba-ai |

---

## ① Oracle VM 作成（Oracle Cloud コンソール）

1. Oracle Cloud コンソール → Compute → Instances → Create Instance
2. Shape: `VM.Standard.A1.Flex` (4 OCPU, 24 GB RAM)
3. Image: Canonical Ubuntu 22.04 (ARM)
4. SSH Key: 新規生成 または 既存の公開鍵をアップロード
5. VCN: 既存または新規作成、パブリックIPを割り当て
6. **セキュリティリスト**: SSH (22), API 用ポート (8000) を開放

---

## ② SSH 接続

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<VM_PUBLIC_IP>
```

---

## ③ ブロックボリューム作成・マウント

```bash
# Oracle コンソールでブロックボリューム (50GB) を作成してアタッチ後:
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /data/keiba-ai
sudo mount /dev/sdb /data/keiba-ai
sudo chown ubuntu:ubuntu /data/keiba-ai

# 再起動後も自動マウント
echo '/dev/sdb /data/keiba-ai ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

---

## ④ 基本ツールインストール

```bash
sudo apt-get update && sudo apt-get install -y \
    git python3.11 python3.11-venv python3-pip sqlite3 \
    build-essential libffi-dev libssl-dev curl
```

---

## ⑤ GitHub から clone

```bash
cd /home/ubuntu
git clone https://github.com/<YOUR_USERNAME>/keiba-ai.git
cd keiba-ai
```

---

## ⑥ Python venv 作成・依存インストール

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

ARM (aarch64) では各パッケージのホイールが自動選択される。
LightGBM 4.0+ は ARM Linux 対応のホイールを提供している。

---

## ⑦ .env 設定

```bash
cp .env.example .env
nano .env
```

Oracle VM 標準設定:

```
KEIBA_DATA_DIR=/data/keiba-ai
KEIBA_DB_PATH=/data/keiba-ai/keiba.db
KEIBA_REGISTRY_DB_PATH=/data/keiba-ai/research_registry.db
KEIBA_MODEL_DIR=/data/keiba-ai/models
KEIBA_FAILED_LOG_PATH=/data/keiba-ai/failed_races.jsonl
KEIBA_PRODUCTION_VERSION_PATH=/data/keiba-ai/models/production_version.txt
```

```bash
# 必要なディレクトリを作成
mkdir -p /data/keiba-ai/models /data/keiba-ai/logs /data/keiba-ai/backups
```

---

## ⑧ keiba.db 移行

Windows PC から Oracle VM へ転送:

```bash
# PC 側で実行（PowerShell または bash）
scp -i ~/.ssh/id_rsa "C:\Users\kei.yamaguchi\Documents\keiba-ai\data\keiba.db" \
    ubuntu@<VM_PUBLIC_IP>:/data/keiba-ai/keiba.db
```

VM 側で integrity を確認:

```bash
sqlite3 /data/keiba-ai/keiba.db "PRAGMA integrity_check;"
# → ok が返ればOK
sqlite3 /data/keiba-ai/keiba.db "SELECT COUNT(*) FROM races;"
sqlite3 /data/keiba-ai/keiba.db "SELECT COUNT(*) FROM race_results;"
```

---

## ⑨ research_registry.db 移行

存在する場合のみ転送:

```bash
scp -i ~/.ssh/id_rsa "C:\Users\kei.yamaguchi\Documents\keiba-ai\data\research_registry.db" \
    ubuntu@<VM_PUBLIC_IP>:/data/keiba-ai/research_registry.db
```

存在しない場合は初回起動時に自動作成される。

---

## ⑩ models 移行

学習済みモデルを転送:

```bash
scp -i ~/.ssh/id_rsa \
    "C:\Users\kei.yamaguchi\Documents\keiba-ai\models\keiba_lgbm_*.pkl" \
    "C:\Users\kei.yamaguchi\Documents\keiba-ai\models\stats_cache_*.pkl" \
    "C:\Users\kei.yamaguchi\Documents\keiba-ai\models\model_meta_*.json" \
    ubuntu@<VM_PUBLIC_IP>:/data/keiba-ai/models/
```

---

## ⑪ production_version.txt 設定

**必ずモデルファイル転送後に実行する**:

```bash
# VM 上で実行
# 利用するバージョンを確認
ls /data/keiba-ai/models/keiba_lgbm_*.pkl

# 本番モデルを設定（例: v202609）
source /home/ubuntu/keiba-ai/.venv/bin/activate
cd /home/ubuntu/keiba-ai
python -c "from ml.predict import set_production_version; set_production_version('v202609')"

# または直接書き込み（atomic ではないが可）
echo "v202609" > /data/keiba-ai/models/production_version.txt
```

**production_version.txt は `.gitignore` で除外されており、GitHub へはコミットしない。**
本番モデルの切り替えは VM 上の手動操作のみで行う。

---

## ⑫ migration_check.sh 実行

```bash
cd /home/ubuntu/keiba-ai
source .venv/bin/activate
bash jobs/migration_check.sh
```

- `[OK]` のみ → API 起動可
- `[WARN]` あり → DB・モデル未配置（⑧〜⑪ を再確認）
- `[FAIL]` あり → コード・環境の問題（ログを確認）

---

## ⑬ API 起動

```bash
cd /home/ubuntu/keiba-ai
source .venv/bin/activate
source .env
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

systemd で常駐させる場合:

```bash
sudo tee /etc/systemd/system/keiba-api.service <<'EOF'
[Unit]
Description=keiba-ai FastAPI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/keiba-ai
EnvironmentFile=/home/ubuntu/keiba-ai/.env
ExecStart=/home/ubuntu/keiba-ai/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable keiba-api
sudo systemctl start keiba-api
sudo systemctl status keiba-api
```

---

## ⑭ cron 設定

```bash
crontab -e
```

追加する行:

```cron
# 週次スクレイピング（毎週月曜 03:00 JST = 日曜 18:00 UTC）
0 18 * * 0  /home/ubuntu/keiba-ai/jobs/weekly_scrape.sh

# 月次学習（毎月1日 03:00 JST = 00:00前日 18:00 UTC）
0 18 1 * *  /home/ubuntu/keiba-ai/jobs/monthly_train.sh

# 日次バックアップ（毎日 02:00 JST = 前日 17:00 UTC）
0 17 * * *  /home/ubuntu/keiba-ai/jobs/backup.sh
```

---

## ⑮ バックアップ確認

```bash
bash /home/ubuntu/keiba-ai/jobs/backup.sh

# バックアップ先を確認
ls -lh /data/keiba-ai/backups/
```

---

## ⑯ ロールバック方法

### モデルロールバック（API 再起動不要）

```bash
# 利用可能なバージョン一覧
ls /data/keiba-ai/models/keiba_lgbm_*.pkl

# 旧バージョンへ切り替え
source /home/ubuntu/keiba-ai/.venv/bin/activate
cd /home/ubuntu/keiba-ai
python -c "from ml.predict import set_production_version; set_production_version('v202608')"
# → 次のAPIリクエストから v202608 が自動で使われる
```

### 全体ロールバック（コード）

```bash
git log --oneline -10
git checkout <commit_hash>
sudo systemctl restart keiba-api
```

---

## ⑰ Oracle Cloud から別 VPS へ移行する場合

### データ移行（ブロックボリュームのスナップショットが最速）

```bash
# Oracle コンソール: ブロックボリューム → スナップショット作成

# または rsync でファイル転送
rsync -avz --progress \
    ubuntu@<ORACLE_IP>:/data/keiba-ai/ \
    ubuntu@<NEW_VPS_IP>:/data/keiba-ai/
```

### アプリケーション再セットアップ

1. ⑤〜⑪ を新 VPS で実行（clone → .env → DB 配置 → production_version.txt）
2. migration_check.sh で確認
3. cron を再設定
4. DNS / リバースプロキシ の向き先を変更

### Oracle 固有設定と共通設定の分離

| 項目 | Oracle 固有 | アプリ共通 |
|---|---|---|
| ブロックボリューム | `/dev/sdb` マウント設定 | `/data/keiba-ai` 以下のファイル群 |
| セキュリティリスト | VCN のルール | `0.0.0.0:8000` を開放 |
| SSH 認証 | OCI キーペア | `.ssh/id_rsa` 公開鍵 |
| 課金 | Always Free 枠 | 不要 |

アプリケーションコード・DB・モデルは Oracle に依存しない。
別 VPS でも `.env` のパスを調整するだけで動作する。

---

## 参考: 環境変数一覧

| 変数 | 説明 | Oracle VM 標準値 |
|---|---|---|
| `KEIBA_DATA_DIR` | データディレクトリルート | `/data/keiba-ai` |
| `KEIBA_DB_PATH` | keiba.db フルパス | `/data/keiba-ai/keiba.db` |
| `KEIBA_REGISTRY_DB_PATH` | research_registry.db フルパス | `/data/keiba-ai/research_registry.db` |
| `KEIBA_MODEL_DIR` | モデルファイルディレクトリ | `/data/keiba-ai/models` |
| `KEIBA_FAILED_LOG_PATH` | スクレイピング失敗ログ | `/data/keiba-ai/failed_races.jsonl` |
| `KEIBA_PRODUCTION_VERSION_PATH` | production_version.txt パス | `/data/keiba-ai/models/production_version.txt` |
