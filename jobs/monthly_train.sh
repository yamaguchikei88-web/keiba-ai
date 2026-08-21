#!/bin/bash
# 月次学習・評価ジョブ（Oracle VM cron 用）
# cron 設定例: 0 18 1 * *  /home/ubuntu/keiba-ai/jobs/monthly_train.sh
#              毎月1日 03:00 JST に学習・バックテスト・評価を実行
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# .env 読み込み
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# モデルバージョン（YYYYMM 形式）
export MODEL_VERSION="v$(date +%Y%m)"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/train_${MODEL_VERSION}.log"
exec >> "$LOG" 2>&1

echo "=== monthly_train start: model=${MODEL_VERSION} $(date) ==="

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "$PROJECT_ROOT"

echo "--- Step 1: モデル学習 (MODEL_VERSION=${MODEL_VERSION}) ---"
python ml/train.py

echo "--- Step 2: バックテスト・評価パイプライン ---"
PIPELINE_JSON="${LOG_DIR}/pipeline_${MODEL_VERSION}.json"
python -m ml.pipeline \
    --model-version       "${MODEL_VERSION}" \
    --feature-set-version "v1"               \
    --windows             3                  \
    --json > "$PIPELINE_JSON"

echo "--- Pipeline 結果サマリー ---"
python -c "
import json
with open('${PIPELINE_JSON}') as f:
    r = json.load(f)
print('status=' + str(r['model_status']) + ' promoted=' + str(r['promoted_to_validated']))
m = r.get('metrics', {})
for k in ['log_loss', 'roc_auc', 'brier_score']:
    print('  ' + k + '=' + str(m.get(k)))
"

echo "=== monthly_train done: ${MODEL_VERSION} $(date) ==="
echo ""
echo "【次のステップ】"
echo "  1. 上記のパイプライン結果を確認してください。"
echo "  2. 採用する場合は以下のコマンドで本番モデルを切り替えてください:"
echo "     echo '${MODEL_VERSION}' > \${KEIBA_MODEL_DIR:-${PROJECT_ROOT}/models}/production_version.txt"
echo "  3. APIプロセスを再起動するか、次のリクエストで自動的に切り替わります。"
echo "  4. ロールバックする場合: echo '旧バージョン' > production_version.txt"
echo "  注意: production への自動昇格は行いません。必ず手動で確認・承認してください。"
