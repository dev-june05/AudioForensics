@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   AI-Generated and Tampered Audio Detection Setup
echo ===================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)
echo [OK] Python is installed.

:: 2. Check NPM
where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] NPM is not installed or not in PATH.
    echo Please install Node.js and try again.
    pause
    exit /b 1
)
echo [OK] NPM is installed.

echo.
echo --- Setting up Backend ---
cd backend

:: 3. Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment in backend\venv...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: 4. Install backend dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo [OK] Backend setup complete.
deactivate
cd ..

echo.
echo --- Setting up Frontend ---
cd frontend
:: 5. Install frontend dependencies
echo Installing NPM packages...
call npm install
echo [OK] Frontend setup complete.
cd ..

echo.
echo ===================================================
echo   Setup Complete!
echo ===================================================
echo.
echo To start the application:
echo.
echo 1. Start the Backend:
echo    cd backend
echo    venv\Scripts\activate
echo    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo 2. Start the Frontend (in a new terminal):
echo    cd frontend
echo    npm run dev
echo.
echo Open http://localhost:5173 in your browser.
echo ===================================================
pause
