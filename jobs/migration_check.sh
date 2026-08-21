#!/bin/bash
# Oracle VM 移行後チェックスクリプト
# 使い方: bash jobs/migration_check.sh
#
# exit code:
#   0 = PASS のみ、または WARN のみ（モデル/DB 未配置は WARN）
#   1 = FAIL が 1 件以上（コード・環境のセットアップ問題）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0
WARN=0

ok()   { echo "[OK]   $*";   PASS=$((PASS+1)); }
fail() { echo "[FAIL] $*";   FAIL=$((FAIL+1)); }
warn() { echo "[WARN] $*";   WARN=$((WARN+1)); }
info() { echo "[INFO] $*"; }
sep()  { echo ""; }

echo "=== keiba-ai migration_check $(date '+%Y-%m-%d %H:%M:%S') ==="
sep

# ─────────────────────────────────────────────────────────────────
# 1. Python バージョン
# ─────────────────────────────────────────────────────────────────
info "--- 1. Python バージョン ---"
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python が見つかりません。python3 または python をインストールしてください"
else
    PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    PY_OK=$("$PYTHON" -c "import sys; print('yes' if sys.version_info >= (3,10) else 'no')")
    if [ "$PY_OK" = "yes" ]; then
        ok "Python $PY_VER"
    else
        fail "Python 3.10+ が必要です (現在: $PY_VER)"
    fi
fi
sep

# ─────────────────────────────────────────────────────────────────
# 2. 必須ディレクトリ確認
# ─────────────────────────────────────────────────────────────────
info "--- 2. 必須ディレクトリ確認 ---"
for dir in \
    "$PROJECT_ROOT" \
    "$PROJECT_ROOT/ml" \
    "$PROJECT_ROOT/api" \
    "$PROJECT_ROOT/jobs" \
    "$PROJECT_ROOT/scraper" \
    "$PROJECT_ROOT/config"; do
    if [ -d "$dir" ]; then
        ok "dir exists: ${dir#$PROJECT_ROOT/}"
    else
        fail "dir missing: ${dir#$PROJECT_ROOT/}"
    fi
done
sep

# ─────────────────────────────────────────────────────────────────
# 3. .env 確認
# ─────────────────────────────────────────────────────────────────
info "--- 3. .env 確認 ---"
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    ok ".env 存在"
    # .env をロード（エラーは無視）
    set -a
    source "$ENV_FILE" 2>/dev/null || true
    set +a
else
    fail ".env が存在しません → .env.example を参考に作成してください"
    warn "(以降のパスはデフォルト値を使用します)"
fi
sep

# ─────────────────────────────────────────────────────────────────
# 4. keiba.db 確認
# ─────────────────────────────────────────────────────────────────
info "--- 4. keiba.db 確認 ---"
DB_PATH="${KEIBA_DB_PATH:-$PROJECT_ROOT/data/keiba.db}"
if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(du -k "$DB_PATH" | cut -f1)
    INTEGRITY=$(sqlite3 "$DB_PATH" "PRAGMA integrity_check;" 2>/dev/null || echo "ERROR")
    if [ "$INTEGRITY" = "ok" ]; then
        RACES=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM races;" 2>/dev/null || echo "?")
        RESULTS=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM race_results;" 2>/dev/null || echo "?")
        ok "keiba.db integrity=ok  size=${DB_SIZE}KB  races=${RACES}  race_results=${RESULTS}"
    else
        fail "keiba.db integrity_check 失敗: $INTEGRITY"
    fi
else
    warn "keiba.db が見つかりません: $DB_PATH"
    warn "  → 移行前の初期状態では正常。keiba.db を配置してください"
fi
sep

# ─────────────────────────────────────────────────────────────────
# 5. research_registry.db 確認
# ─────────────────────────────────────────────────────────────────
info "--- 5. research_registry.db 確認 ---"
REG_PATH="${KEIBA_REGISTRY_DB_PATH:-$PROJECT_ROOT/data/research_registry.db}"
if [ -f "$REG_PATH" ]; then
    INTEGRITY=$(sqlite3 "$REG_PATH" "PRAGMA integrity_check;" 2>/dev/null || echo "ERROR")
    if [ "$INTEGRITY" = "ok" ]; then
        ok "research_registry.db integrity=ok"
    else
        fail "research_registry.db integrity_check 失敗: $INTEGRITY"
    fi
else
    warn "research_registry.db が見つかりません: $REG_PATH"
    warn "  → 存在しない場合は正常（初回起動時に自動作成）"
fi
sep

# ─────────────────────────────────────────────────────────────────
# 6. MODEL_DIR 確認
# ─────────────────────────────────────────────────────────────────
info "--- 6. MODEL_DIR 確認 ---"
MODEL_DIR="${KEIBA_MODEL_DIR:-$PROJECT_ROOT/models}"
if [ -d "$MODEL_DIR" ]; then
    PKL_COUNT=$(find "$MODEL_DIR" -name "keiba_lgbm_*.pkl" 2>/dev/null | wc -l | tr -d ' ')
    ok "MODEL_DIR 存在: $MODEL_DIR  (versioned pkl: $PKL_COUNT 件)"
else
    warn "MODEL_DIR が存在しません: $MODEL_DIR"
    warn "  → 移行前の初期状態では正常。models/ を配置してください"
fi
sep

# ─────────────────────────────────────────────────────────────────
# 7. production_version.txt 確認
# ─────────────────────────────────────────────────────────────────
info "--- 7. production_version.txt 確認 ---"
PVP="${KEIBA_PRODUCTION_VERSION_PATH:-$MODEL_DIR/production_version.txt}"
if [ -f "$PVP" ]; then
    VERSION=$(cat "$PVP" | tr -d '[:space:]')
    if [ -n "$VERSION" ]; then
        # 3 ファイル存在確認
        M_PKL="$MODEL_DIR/keiba_lgbm_${VERSION}.pkl"
        S_PKL="$MODEL_DIR/stats_cache_${VERSION}.pkl"
        J_META="$MODEL_DIR/model_meta_${VERSION}.json"
        MISSING=()
        [ -f "$M_PKL" ] || MISSING+=("keiba_lgbm_${VERSION}.pkl")
        [ -f "$S_PKL" ] || MISSING+=("stats_cache_${VERSION}.pkl")
        [ -f "$J_META" ] || MISSING+=("model_meta_${VERSION}.json")
        if [ ${#MISSING[@]} -eq 0 ]; then
            ok "production_version=${VERSION}  3 ファイルすべて存在"
        else
            fail "production_version=${VERSION} だが不足ファイル: ${MISSING[*]}"
        fi
    else
        fail "production_version.txt が空です → 有効なバージョンを書き込んでください"
    fi
else
    warn "production_version.txt が存在しません: $PVP"
    warn "  → 本番 API は起動できません。移行後に設定が必要です"
    warn "  例: echo 'v202609' > $PVP"
fi
sep

# ─────────────────────────────────────────────────────────────────
# 8. Python import 確認
# ─────────────────────────────────────────────────────────────────
info "--- 8. Python import 確認 ---"
if [ -z "$PYTHON" ]; then
    fail "Python が利用できないため import チェックをスキップ"
else
    IMPORT_RESULT=$("$PYTHON" - <<'PYEOF'
import sys
failures = []
for mod in ["fastapi", "uvicorn", "lightgbm", "joblib", "pandas", "numpy", "sklearn", "requests", "bs4", "pytest"]:
    try:
        __import__(mod)
    except ImportError:
        failures.append(mod)
if failures:
    print("FAIL: " + " ".join(failures))
    sys.exit(1)
else:
    print("OK: " + " ".join(["fastapi","uvicorn","lightgbm","joblib","pandas","numpy","sklearn","requests","bs4","pytest"]))
PYEOF
    )
    if echo "$IMPORT_RESULT" | grep -q "^OK:"; then
        ok "$IMPORT_RESULT"
    else
        fail "import 失敗: $IMPORT_RESULT"
        fail "  → pip install -r requirements.txt を実行してください"
    fi
fi
sep

# ─────────────────────────────────────────────────────────────────
# 9. pytest 実行
# ─────────────────────────────────────────────────────────────────
info "--- 9. pytest ---"
if [ -z "$PYTHON" ]; then
    fail "Python が利用できないため pytest をスキップ"
else
    cd "$PROJECT_ROOT"
    PYTEST_OUT=$("$PYTHON" -m pytest tests/test_oracle_prep.py -q --tb=short 2>&1 || true)
    PYTEST_LAST=$(echo "$PYTEST_OUT" | tail -3)
    if echo "$PYTEST_LAST" | grep -qE "passed"; then
        ok "pytest: $PYTEST_LAST"
    else
        fail "pytest 失敗:"
        echo "$PYTEST_OUT" | tail -20
    fi
fi
sep

# ─────────────────────────────────────────────────────────────────
# 結果サマリー
# ─────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "  PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
echo "════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "FAIL が $FAIL 件あります。上記の [FAIL] 項目を修正してください。"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo ""
    echo "WARN が $WARN 件あります（DB・モデル未配置なら正常）。"
    echo "Oracle VM に keiba.db / models/ を配置後に再実行してください。"
    exit 0
else
    echo ""
    echo "すべてのチェックが PASS しました。API 起動の準備ができています。"
    exit 0
fi
