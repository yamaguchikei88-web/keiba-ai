#!/bin/bash
# 週次データ収集ジョブ（Oracle VM cron 用）
# cron 設定例: 0 17 * * 6   /home/ubuntu/keiba-ai/jobs/weekly_scrape.sh
#              毎週日曜 02:00 JST に先週分のレースを収集
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# .env が存在する場合は読み込む（KEIBA_* 環境変数の設定）
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ログディレクトリを作成
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/scrape_$(date +%Y%m%d_%H%M%S).log"
exec >> "$LOG" 2>&1

echo "=== weekly_scrape start: $(date) ==="

# 仮想環境を activate（存在する場合のみ）
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# 先週7日分（昨日から遡って7日間）を収集
# 冪等性あり: 取得済みの race_id は自動スキップされる
END=$(date -d 'yesterday' +%Y%m%d)
START=$(date -d '7 days ago' +%Y%m%d)
START_YEAR=$(date -d "$START" +%Y)
END_YEAR=$(date -d "$END" +%Y)

echo "収集期間: ${START} 〜 ${END} (${START_YEAR}〜${END_YEAR}年)"

cd "$PROJECT_ROOT"

python scraper/netkeiba_scraper.py \
    --start-year     "$START_YEAR"  \
    --end-year       "$END_YEAR"    \
    --start-date     "$START"       \
    --end-date       "$END"         \
    --sleep          1.0            \
    --sleep-calendar 0.5            \
    --max-retries    3

echo "=== weekly_scrape done: $(date) ==="
