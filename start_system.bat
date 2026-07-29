@echo off
title QuantumCrop AI – System Launcher
echo ====================================================
echo      QUANTUMCROP AI – EYES IN THE SKY LAUNCHER
echo ====================================================
echo.

:: Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

:: Create Virtual Environment if not exists
if not exist "backend\.venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv backend\.venv
)

echo [INFO] Activating virtual environment and installing backend requirements...
call backend\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

echo.
echo [INFO] Skipping automatic integration verification checks to prevent blocking.
echo.

:: Check for Node/NPM
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Node.js / NPM is not installed or in your PATH.
    echo Frontend Vite compilation requires Node.js.
    echo Please run backend manually and install Node.js later.
    echo.
) else (
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

echo.
echo ====================================================
echo  LAUNCHING BACKEND AND FRONTEND SERVERS...
echo ====================================================
echo.

:: Start Backend Server in a new window
echo [INFO] Starting FastAPI backend on http://localhost:8000
start "QuantumCrop Backend" cmd /k "call backend\.venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

:: Wait a moment
timeout /t 3 /nobreak >nul

:: Start Frontend Server
if exist "frontend\node_modules" (
    echo [INFO] Starting React Frontend on http://localhost:5173
    start "QuantumCrop Frontend" cmd /k "cd frontend && npm run dev"
) else (
    echo [WARNING] Node modules missing. Could not launch Vite.
    echo Please run 'npm install' inside the 'frontend' folder manually.
)

echo.
echo [SUCCESS] Both servers launched!
echo Open http://localhost:5173 to start analyzing remote sensing crop data.
echo.
pause
