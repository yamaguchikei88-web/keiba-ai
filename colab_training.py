"""
Google Colab で実行するデータ収集・学習スクリプト
Colabに新しいノートブックを作り、このファイルの各セルをコピーして実行してください
"""

# ============================================================
# セル1: 環境セットアップ（最初に1回だけ実行）
# ============================================================
CELL_1 = """
!pip install requests beautifulsoup4 pandas numpy lightgbm scikit-learn joblib

import os
from google.colab import drive
drive.mount('/content/drive')

# Shared Drive / Cloud Storage mount path. Change this variable for a shared
# location; do not use a PC-local path or copy artifacts into Git.
SHARED_ROOT = os.environ.get('KEIBA_SHARED_ROOT', '/content/drive/MyDrive/keiba-ai')
os.makedirs(f'{SHARED_ROOT}/data', exist_ok=True)
os.makedirs(f'{SHARED_ROOT}/models', exist_ok=True)
print('セットアップ完了')
"""

# ============================================================
# セル2: GitHubからコードを取得
# ============================================================
CELL_2 = """
!git clone https://github.com/yamaguchikei88-web/keiba-ai.git /content/keiba-ai
import sys
sys.path.append('/content/keiba-ai/scraper')
sys.path.append('/content/keiba-ai/ml')
print('コード取得完了')
"""

# ============================================================
# セル3: データ収集（10年分・約10万レース）
# ※ 1回で終わらない場合は start_year を変えて続きから実行
# ============================================================
CELL_3 = """
import sys
sys.path.append('/content/keiba-ai/scraper')

# DBの保存先をGoogle Driveに設定
import sqlite3
from pathlib import Path

# scraper の DB_PATH を Drive に向ける
import netkeiba_scraper as scraper
shared_root = Path(os.environ.get('KEIBA_SHARED_ROOT', '/content/drive/MyDrive/keiba-ai'))
scraper.DB_PATH = shared_root / 'data' / 'keiba.db'
scraper.init_db()

# 収集開始（1年ずつ実行してもOK）
scraper.scrape_all(start_year=2015, end_year=2025)
print('データ収集完了')
"""

# ============================================================
# セル4: 血統データ補完
# ============================================================
CELL_4 = """
import sys
sys.path.append('/content/keiba-ai/scraper')
import horse_scraper as hs
from pathlib import Path

shared_root = Path(os.environ.get('KEIBA_SHARED_ROOT', '/content/drive/MyDrive/keiba-ai'))
hs.DB_PATH = shared_root / 'data' / 'keiba.db'
hs.fill_pedigree()
print('血統データ補完完了')
"""

# ============================================================
# セル5: モデル学習（約30分〜2時間）
# ============================================================
CELL_5 = """
import sys
sys.path.append('/content/keiba-ai/ml')
from pathlib import Path

# パスを Drive に向ける
import features
shared_root = Path(os.environ.get('KEIBA_SHARED_ROOT', '/content/drive/MyDrive/keiba-ai'))
features.DB_PATH = shared_root / 'data' / 'keiba.db'

import train as trainer
trainer.DB_PATH = shared_root / 'data' / 'keiba.db'
trainer.MODEL_DIR = shared_root / 'models'
trainer.MODEL_PATH = trainer.MODEL_DIR / 'keiba_lgbm.pkl'
trainer.STATS_CACHE_PATH = trainer.MODEL_DIR / 'stats_cache.pkl'
trainer.META_PATH = trainer.MODEL_DIR / 'model_meta.json'

model, auc = trainer.train()
print(f'学習完了！ AUC: {auc:.4f}')
"""

# ============================================================
# セル6: artifactを共有storageに残し、Gitへは文書だけをcommit
# ============================================================
CELL_6 = """
from pathlib import Path
import json

shared_root = Path(os.environ.get('KEIBA_SHARED_ROOT', '/content/drive/MyDrive/keiba-ai'))
model_dir = shared_root / 'models'
print('モデルartifactは共有storageに保持します:', model_dir)
print('GitHubへmodel/cacheをコピー・pushしません。')
print('承認済みのmodel registry記録とartifact hashを別taskで作成してください。')
"""

if __name__ == "__main__":
    print("このファイルはColabで実行するスクリプトの参照用です。")
    print("各CELL_N の文字列をColabのセルにコピーして実行してください。")
    for i in range(1, 7):
        print(f"\n{'='*50}")
        print(f"セル {i}:")
        print(eval(f"CELL_{i}"))
