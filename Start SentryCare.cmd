@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup.ps1 first to install the local environment.
    pause
    exit /b 1
)
echo Opening SentryCare Studio at http://127.0.0.1:8765
echo Keep this window open while using Studio. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m sentrycare.web --open-browser
pause
