@echo off
title RPA Bot Monitoring Dashboard
color 0A

echo.
echo ============================================================
echo   RPA Bot Monitoring Dashboard - Starting Server
echo ============================================================
echo.

:: Move to the correct directory (same folder as this .bat file)
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python not found in PATH!
    echo         Install Python from https://python.org
    pause
    exit /b 1
)

:: Check Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Flask not found. Installing now...
    pip install flask werkzeug
    if errorlevel 1 (
        color 0C
        echo [ERROR] Failed to install Flask. Check your internet connection.
        pause
        exit /b 1
    )
)

:: Kill any stale python process on port 5000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo [OK] Python found
echo [OK] Flask found  
echo [OK] Port 5000 cleared
echo.
echo ============================================================
echo   Server starting at: http://127.0.0.1:5000
echo   Open that URL in your browser after this message
echo   Press Ctrl+C in this window to stop the server
echo ============================================================
echo.

:: Start Flask
python app.py

:: If Flask crashes, pause so you can see the error
if errorlevel 1 (
    color 0C
    echo.
    echo [ERROR] Server crashed! See error above.
    pause
)
