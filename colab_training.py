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

# 保存先フォルダを作成
os.makedirs('/content/drive/MyDrive/keiba-ai/data', exist_ok=True)
os.makedirs('/content/drive/MyDrive/keiba-ai/models', exist_ok=True)
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
scraper.DB_PATH = Path('/content/drive/MyDrive/keiba-ai/data/keiba.db')
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

hs.DB_PATH = Path('/content/drive/MyDrive/keiba-ai/data/keiba.db')
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
features.DB_PATH = Path('/content/drive/MyDrive/keiba-ai/data/keiba.db')

import train as trainer
trainer.DB_PATH = Path('/content/drive/MyDrive/keiba-ai/data/keiba.db')
trainer.MODEL_DIR = Path('/content/drive/MyDrive/keiba-ai/models')
trainer.MODEL_PATH = trainer.MODEL_DIR / 'keiba_lgbm.pkl'
trainer.STATS_CACHE_PATH = trainer.MODEL_DIR / 'stats_cache.pkl'
trainer.META_PATH = trainer.MODEL_DIR / 'model_meta.json'

model, auc = trainer.train()
print(f'学習完了！ AUC: {auc:.4f}')
"""

# ============================================================
# セル6: モデルをGitHubにアップロード
# ============================================================
CELL_6 = """
import shutil
from pathlib import Path

# DriveからリポジトリのmodelsフォルダにコピーS
src = Path('/content/drive/MyDrive/keiba-ai/models')
dst = Path('/content/keiba-ai/models')
dst.mkdir(exist_ok=True)

for f in ['keiba_lgbm.pkl', 'stats_cache.pkl', 'model_meta.json']:
    if (src / f).exists():
        shutil.copy(src / f, dst / f)
        print(f'コピー: {f}')

# GitHubにプッシュ
# ※ 事前にGitHubのPersonal Access Tokenを取得してください
GITHUB_TOKEN = 'ここにGitHub Personal Access Tokenを入力'
GITHUB_USER = 'yamaguchikei88-web'

!cd /content/keiba-ai && git config user.email "your@email.com"
!cd /content/keiba-ai && git config user.name "keiba-ai"
!cd /content/keiba-ai && git add models/
!cd /content/keiba-ai && git commit -m "Update trained model"
!cd /content/keiba-ai && git push https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/keiba-ai.git main

print('GitHubへのアップロード完了！')
"""

if __name__ == "__main__":
    print("このファイルはColabで実行するスクリプトの参照用です。")
    print("各CELL_N の文字列をColabのセルにコピーして実行してください。")
    for i in range(1, 7):
        print(f"\n{'='*50}")
        print(f"セル {i}:")
        print(eval(f"CELL_{i}"))
