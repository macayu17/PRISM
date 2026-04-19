@echo off
echo ========================================
echo Parkinson's Disease Assessment Portal
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo Starting web application...
echo.
echo The application will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python start_server.py

pause
