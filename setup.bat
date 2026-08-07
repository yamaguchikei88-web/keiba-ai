@echo off
echo ========================================
echo  競馬AI予想システム セットアップ
echo ========================================

echo.
echo [1/3] Pythonライブラリをインストール中...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo エラー: pip installに失敗しました
    pause
    exit /b 1
)

echo.
echo [2/3] データ収集を開始します（数時間かかります）...
echo ※ 裏で実行してください。終了しないでください。
python scraper/netkeiba_scraper.py

echo.
echo [3/3] 完了！次は以下を実行してください:
echo   python ml/train.py      （モデル学習）
echo   python api/main.py      （APIサーバー起動）
echo.
pause
