@echo off
echo ========================================
echo  競馬AI予想システム セットアップ
echo ========================================

echo.
echo [1/2] Pythonライブラリをインストール中...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo エラー: pip installに失敗しました
    pause
    exit /b 1
)

echo.
echo [2/2] 完了
echo.
echo このスクリプトはデータ収集・学習を実行しません。
echo README.mdを読み、共有storageの環境変数を設定してください。
echo.
pause
