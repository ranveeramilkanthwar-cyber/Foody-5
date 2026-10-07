@echo off
title FoodSafe AI Server
echo ================================================
echo   Foody - Indian Food Label Analyzer
echo   Dashboard: http://localhost:8000/dashboard/
echo ================================================
echo.

REM Try M: drive venv first (RTX 5050 / CUDA 12.8)
if exist "M:\Projects\ML\ChatBot\venv\Scripts\python.exe" (
    echo Using M: drive GPU environment...
    "M:\Projects\ML\ChatBot\venv\Scripts\python.exe" server.py
    goto end
)

REM Fallback to system python
echo Using system Python...
python server.py

:end
pause
