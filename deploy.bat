@echo off
setlocal enabledelayedexpansion
title CivicPulse Sight - Deployment Launcher

echo ======================================================================
echo    CIVICPULSE SIGHT - Urban ^& Geospatial Intelligence Platform
echo ======================================================================
echo.

:: 1. Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in your PATH.
    echo Please install Node.js (v20+ recommended) from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

:: 2. Ensure dependencies are installed
if not exist "node_modules\" (
    echo [SETUP] First-time launch detected. Installing dependencies...
    echo Running npm install...
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
    echo [SETUP] Dependencies successfully installed.
    echo.
)

:: 3. Inform user of access links
echo ======================================================================
echo   CIVICPULSE SIGHT IS READY TO LAUNCH!
echo ======================================================================
echo.
echo   Primary 3D Globe Cockpit:
echo   --^> http://localhost:4173/
echo.
echo   Urban Priority ^& Infrastructure Engine:
echo   --^> http://localhost:4173/smartcity/index.html
echo.
echo   Triage Audit ^& Model Validation:
echo   --^> http://localhost:4173/audit.html
echo.
echo ======================================================================
echo   Launching default web browser in 3 seconds...
echo   Press Ctrl+C at any time in this window to stop the server.
echo ======================================================================
echo.

:: 4. Launch browser after a short delay in background
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:4173/"

:: 5. Start the Vite server
call npm run dev

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Server terminated with exit code %errorlevel%.
    pause
)
