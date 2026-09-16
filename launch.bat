@echo off
title TennisPreview
color 0A
cd /d "%~dp0"

echo ============================================================
echo   TennisPreview v1.0.0
echo   Tennis Match Analysis & Value Betting
echo ============================================================
echo.

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    echo [OK] Uso virtual environment venv
) else (
    set "PYTHON=python"
    echo [WARN] venv non trovato - uso python di sistema
)

%PYTHON% -m src.main --gui

echo.
echo Fatto!
pause
