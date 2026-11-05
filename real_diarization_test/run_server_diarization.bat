@echo off

title SpeakerIdentity IA - Real Diarization Test

cd /d "%~dp0\.."

echo.
echo ================================================================
echo              SpeakerIdentity IA - REAL DIARIZATION TEST
echo ================================================================
echo.

echo [INFO] Project root:
echo %CD%
echo.

echo [INFO] Python:
python --version

echo.

echo [INFO] Checking diarization package...

python -c "import diarization; print('[OK] diarization package loaded')"

if errorlevel 1 (
    echo.
    echo [ERROR] Impossible d'importer diarization.
    pause
    exit /b 1
)

echo.

echo [INFO] Starting real diarization server...
echo.

python real_diarization_test\server.py

echo.
echo ================================================================
echo                    SERVER STOPPED
echo ================================================================
pause