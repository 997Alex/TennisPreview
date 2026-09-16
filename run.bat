@echo off
REM TennisPreview Run Script - Windows
REM Equivalente di run.sh per Windows

setlocal
cd /d "%~dp0"

echo ============================================================
echo   TennisPreview v1.0.0
echo   Tennis Match Analysis ^& Value Betting
echo ============================================================
echo.

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    echo [OK] Uso virtual environment venv
) else (
    set "PYTHON=python"
    echo [WARN] venv non trovato - uso python di sistema
)

REM Verifica dipendenze principali, installa se mancano
%PYTHON% -c "import pandas, numpy, loguru, feedparser, tqdm, yaml, dotenv, aiohttp" 2>nul
if errorlevel 1 (
    echo [INFO] Installazione dipendenze...
    if exist "venv\Scripts\pip.exe" (
        venv\Scripts\pip.exe install -r requirements.txt
    ) else (
        pip install -r requirements.txt
    )
)

REM Crea .env da esempio se manca
if not exist ".env" (
    echo [INFO] Creo .env da .env.example...
    copy ".env.example" ".env" >nul
    echo [WARN] API keys opzionali: il programma funziona in --demo senza chiavi.
    echo [WARN] Modifica .env e aggiungi THEODDSAPI_KEY / API_FOOTBALL_KEY per quote live.
)

echo [INFO] Avvio pipeline TennisPreview (palinsesto Sisal del giorno)...
echo [INFO] Uso: run.bat [--demo] [--date YYYY-MM-DD] [--days N] [--no-progress] [--gui]
echo.
%PYTHON% -m src.main %*

echo.
echo Fatto!
endlocal
