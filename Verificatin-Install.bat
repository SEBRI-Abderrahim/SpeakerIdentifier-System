@echo off
setlocal EnableExtensions
title SpeakerIdentity - Complete Repair
color 0B

cd /d "%~dp0"

echo ================================================================
echo          SPEAKERIDENTITY - COMPLETE REPAIR
echo                   PYTHON 3.12
echo ================================================================
echo.

REM 1. Verify Python

echo [1/7] Checking Python...

py -3.12 --version
if errorlevel 1 goto ERROR

REM 2. Upgrade installer

echo.
echo [2/7] Updating pip...

py -3.12 -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto ERROR

REM 3. Remove incompatible packages

echo.
echo [3/7] Removing old PyTorch packages...

py -3.12 -m pip uninstall -y torch torchaudio torchvision

REM 4. Install matching CPU versions

echo.
echo [4/7] Installing clean PyTorch CPU packages...

py -3.12 -m pip install --no-cache-dir --force-reinstall torch==2.6.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 goto ERROR_TORCH

REM 5. Verify Torch independently

echo.
echo [5/7] Testing PyTorch...

py -3.12 -c "import torch; print('TORCH OK:',torch.__version__)"

if errorlevel 1 goto ERROR_TORCH

REM 6. Verify Torchaudio independently

echo.
echo [6/7] Testing Torchaudio...

if not exist "C:\Python312\Lib\site-packages\torchaudio\lib\libtorchaudio.pyd" (
    echo [ERROR] libtorchaudio.pyd is missing.
    goto ERROR_TORCH
)

py -3.12 -c "import torchaudio; print('TORCHAUDIO OK:',torchaudio.__version__)"

if errorlevel 1 goto ERROR_TORCH

REM 7. Verify complete diarization dependencies

echo.
echo [7/7] Testing project dependencies...

py -3.12 -c "import numpy, speechbrain, webrtcvad, torch, torchaudio; print('NUMPY:',numpy.__version__); print('SPEECHBRAIN:',speechbrain.__version__); print('TORCH:',torch.__version__); print('TORCHAUDIO:',torchaudio.__version__); print('WEBRTC VAD: OK')"

if errorlevel 1 goto ERROR

echo.
echo ================================================================
echo           ALL DEPENDENCIES INSTALLED
echo ================================================================
echo.
echo SpeakerIdentity is ready for testing.
echo.
echo Start command:
echo py -3.12 real_diarization_test\server.py
echo.
pause
exit /b 0

:ERROR_TORCH
echo.
echo ================================================================
echo       PYTORCH / TORCHAUDIO ERROR
echo ================================================================
echo.
echo Check Microsoft Visual C++ Redistributable x64.
echo Download:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo Check the installation logs above.
goto FIN

:ERROR
echo.
echo ================================================================
echo             DEPENDENCY INSTALLATION ERROR
echo ================================================================
echo.
echo Check the Python error displayed above.

:FIN
echo.
pause
exit /b 1
