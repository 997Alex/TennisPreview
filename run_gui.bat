@echo off
title TennisPreview GUI
color 0A
cd /d "%~dp0"

echo ============================================================
echo   TennisPreview v1.0.0 - Interfaccia Grafica
echo ============================================================
echo.

if exist "venv\Scripts\streamlit.exe" (
    echo [OK] Streaming app via Streamlit
) else (
    echo [INFO] Installazione streamlit...
    venv\Scripts\pip.exe install streamlit plotly pillow
)

echo.
echo Apri il browser su: http://localhost:8501
echo.

venv\Scripts\streamlit.exe run src\gui\app.py --server.port 8501 --server.headless true

echo.
echo Fatto!
pause
