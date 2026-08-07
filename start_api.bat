@echo off
echo APIサーバーを起動します...
cd /d %~dp0
python api/main.py
pause
