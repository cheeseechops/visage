@echo off
title Visage - Face Recognition Photo Management
color 0A

echo.
echo  ██╗   ██╗██╗███████╗ █████╗  ██████╗ ███████╗
echo  ██║   ██║██║██╔════╝██╔══██╗██╔════╝ ██╔════╝
echo  ██║   ██║██║███████╗███████║██║  ███╗█████╗  
echo  ╚██╗ ██╔╝██║╚════██║██╔══██║██║   ██║██╔══╝  
echo   ╚████╔╝ ██║███████║██║  ██║╚██████╔╝███████╗
echo    ╚═══╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
echo.
echo  Face Recognition Photo Management System
echo  ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo    Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if virtual environment exists
if exist "venv_py311\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv_py311\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ⚠️  Virtual environment not found, using system Python
)
echo.

REM Display network choice menu
echo 🌐 Choose how to run Visage:
echo.
echo    [1] Local Only - Accessible only on this computer
echo    [2] Network Mode - Accessible on local network
echo    [3] Custom Port - Choose your own port
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🏠 Starting Visage in LOCAL mode...
    echo    Only accessible from this computer
    echo.
    python app.py --host local
) else if "%choice%"=="2" (
    echo.
    echo 🌐 Starting Visage in NETWORK mode...
    echo    Accessible from other devices on your network
    echo.
    python app.py --host network
) else if "%choice%"=="3" (
    echo.
    set /p port="Enter port number (default 5000): "
    if "%port%"=="" set port=5000
    echo.
    echo 🔧 Choose host mode for port %port%:
    echo    [1] Local Only
    echo    [2] Network Mode
    echo.
    set /p host_choice="Enter choice (1-2): "
    
    if "%host_choice%"=="1" (
        echo.
        echo 🏠 Starting Visage in LOCAL mode on port %port%...
        python app.py --host local --port %port%
    ) else if "%host_choice%"=="2" (
        echo.
        echo 🌐 Starting Visage in NETWORK mode on port %port%...
        python app.py --host network --port %port%
    ) else (
        echo ❌ Invalid choice. Exiting...
        pause
        exit /b 1
    )
) else (
    echo ❌ Invalid choice. Exiting...
    pause
    exit /b 1
)

echo.
echo 👋 Visage has stopped. Press any key to exit...
pause >nul
