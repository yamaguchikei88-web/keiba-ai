#!/bin/bash
# SQLite バックアップジョブ（Oracle VM cron 用）
# cron 設定例: 0 19 * * *  /home/ubuntu/keiba-ai/jobs/backup.sh
#              毎日 04:00 JST に SQLite DB をバックアップ
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# .env 読み込み
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/backup_$(date +%Y%m%d).log"
exec >> "$LOG" 2>&1

echo "=== backup start: $(date) ==="

# 環境変数未設定時はデフォルトパスを使用
DB_KEIBA="${KEIBA_DB_PATH:-${PROJECT_ROOT}/data/keiba.db}"
DB_REGISTRY="${KEIBA_REGISTRY_DB_PATH:-${PROJECT_ROOT}/data/research_registry.db}"

BACKUP_DIR="${PROJECT_ROOT}/backups/$(date +%Y%m)"
mkdir -p "$BACKUP_DIR"

# sqlite3 の .backup コマンドで安全にバックアップ（書き込みロック不要）
backup_db() {
    local src="$1"
    local label="$2"
    local dest="${BACKUP_DIR}/${label}_$(date +%Y%m%d).db"

    if [ ! -f "$src" ]; then
        echo "スキップ (ファイル未存在): $src"
        return 0
    fi

    if sqlite3 "$src" ".backup $dest"; then
        echo "バックアップ完了: $dest ($(du -h "$dest" | cut -f1))"
    else
        echo "バックアップ失敗: $src" >&2
        return 1
    fi
}

backup_db "$DB_KEIBA" "keiba"
backup_db "$DB_REGISTRY" "registry"

# 30日以上前のバックアップを削除
find "${PROJECT_ROOT}/backups" -name "*.db" -mtime +30 -delete 2>/dev/null \
    && echo "30日以上前のバックアップを削除しました" || true

echo "=== backup done: $(date) ==="
