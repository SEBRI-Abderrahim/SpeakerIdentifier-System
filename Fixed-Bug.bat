@echo off
setlocal EnableExtensions
title SpeakerIdentity - Complete Installation and Repair
color 0B

cd /d "%~dp0"

set "PY=py -3.12"
set "MODEL_DIR=pretrained_models\spkrec-ecapa-voxceleb"

echo.
echo ================================================================
echo          SPEAKERIDENTITY IA - COMPLETE REPAIR
echo                    WINDOWS / PYTHON 3.12
echo ================================================================
echo.

REM ================================================================
REM 1 - CHECK PYTHON
REM ================================================================

echo [1/10] Checking Python 3.12...

%PY% --version
if errorlevel 1 goto ERROR_PYTHON

REM ================================================================
REM 2 - UPDATE PIP
REM ================================================================

echo.
echo [2/10] Updating pip, setuptools and wheel...

%PY% -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto ERROR

REM ================================================================
REM 3 - CLEAN AND INSTALL PYTORCH CPU
REM ================================================================

echo.
echo [3/10] Repairing PyTorch and Torchaudio...

%PY% -m pip uninstall -y torch torchaudio torchvision

%PY% -m pip install --no-cache-dir torch==2.6.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 goto ERROR_TORCH

REM ================================================================
REM 4 - INSTALL NUMPY
REM ================================================================

echo.
echo [4/10] Installing NumPy...

%PY% -m pip install --no-cache-dir numpy==1.26.4
if errorlevel 1 goto ERROR

REM ================================================================
REM 5 - INSTALL SPEECHBRAIN AND HUGGINGFACE
REM ================================================================

echo.
echo [5/10] Installing SpeechBrain dependencies...

%PY% -m pip install --upgrade speechbrain==1.0.3 huggingface_hub==0.25.2

if errorlevel 1 goto ERROR_SPEECHBRAIN

REM ================================================================
REM 6 - INSTALL WEBRTC VAD
REM ================================================================

echo.
echo [6/10] Installing WebRTC VAD...

%PY% -m pip install --upgrade webrtcvad-wheels==2.0.14

if errorlevel 1 goto ERROR

REM ================================================================
REM 7 - INSTALL PROJECT REQUIREMENTS
REM ================================================================

echo.
echo [7/10] Installing project requirements...

if exist requirements.txt (
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 goto ERROR
) else (
    echo [WARNING] requirements.txt not found.
)

echo.
echo Reapplying compatible core versions...

%PY% -m pip install numpy==1.26.4

if errorlevel 1 goto ERROR

%PY% -m pip install torch==2.6.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 goto ERROR_TORCH

%PY% -m pip install speechbrain==1.0.3 huggingface_hub==0.25.2 webrtcvad-wheels==2.0.14

if errorlevel 1 goto ERROR_SPEECHBRAIN

REM ================================================================
REM 8 - TEST PYTHON IMPORTS
REM ================================================================

echo.
echo [8/10] Testing Python dependencies...

%PY% -c "import numpy, torch, torchaudio, speechbrain, webrtcvad; print('NUMPY:',numpy.__version__); print('TORCH:',torch.__version__); print('TORCHAUDIO:',torchaudio.__version__); print('SPEECHBRAIN:',speechbrain.__version__); print('WEBRTC VAD: OK')"

if errorlevel 1 goto ERROR_IMPORT

echo.
echo Testing SpeechBrain EncoderClassifier...

%PY% -c "from speechbrain.inference.speaker import EncoderClassifier; print('ENCODER CLASSIFIER OK')"

if errorlevel 1 goto ERROR_IMPORT

REM ================================================================
REM 9 - DOWNLOAD AND TEST ECAPA MODEL
REM ================================================================

echo.
echo [9/10] Downloading and testing ECAPA-TDNN model...
echo.
echo First download may take several minutes.
echo.

REM Disable symlink warning on Windows.
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM Remove incomplete model directory if it exists.
if exist "%MODEL_DIR%" (
    echo Removing previous incomplete ECAPA model...
    rmdir /s /q "%MODEL_DIR%"
)

REM Create model directory.
if not exist "pretrained_models" mkdir "pretrained_models"

REM IMPORTANT:
REM LocalStrategy.COPY avoids Windows symlink privilege error WinError 1314.

%PY% -c "from speechbrain.inference.speaker import EncoderClassifier; from speechbrain.utils.fetching import LocalStrategy; m=EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb', run_opts={'device':'cpu'}, local_strategy=LocalStrategy.COPY); print('ECAPA MODEL LOADED SUCCESSFULLY')"

if errorlevel 1 goto ERROR_MODEL

REM ================================================================
REM 10 - START SERVER
REM ================================================================

echo.
echo [10/10] Checking server...

if not exist "real_diarization_test\server.py" goto ERROR_SERVER

echo.
echo ================================================================
echo                 ALL CHECKS PASSED
echo ================================================================
echo.
echo Web interface: http://127.0.0.1:8010/
echo Health:        http://127.0.0.1:8010/health
echo WebSocket:     ws://127.0.0.1:8010/ws/diarization
echo.
echo Starting SpeakerIdentity server...
echo.

%PY% real_diarization_test\server.py

if errorlevel 1 goto ERROR_SERVER_START

goto FINISH

REM ================================================================
REM ERROR HANDLERS
REM ================================================================

:ERROR_PYTHON
echo.
echo [ERROR] Python 3.12 not found.
echo Install Python 3.12 and enable the Python launcher.
goto FINISH

:ERROR_TORCH
echo.
echo [ERROR] PyTorch or Torchaudio installation failed.
echo Check Internet connection and Microsoft Visual C++ Redistributable x64.
goto FINISH

:ERROR_SPEECHBRAIN
echo.
echo [ERROR] SpeechBrain installation failed.
echo Check the pip error above.
goto FINISH

:ERROR_IMPORT
echo.
echo [ERROR] Python dependency import failed.
echo Check the traceback above.
goto FINISH

:ERROR_MODEL
echo.
echo [ERROR] ECAPA model could not be loaded.
echo Check Internet connection and Hugging Face access.
echo.
echo The script uses LocalStrategy.COPY to avoid Windows symlink errors.
goto FINISH

:ERROR_SERVER
echo.
echo [ERROR] real_diarization_test\server.py not found.
echo Make sure this BAT file is in the project root folder.
goto FINISH

:ERROR_SERVER_START
echo.
echo [ERROR] SpeakerIdentity server stopped with an error.
echo Check the server traceback above.
goto FINISH

:ERROR
echo.
echo [ERROR] Installation failed.
echo Check the pip error above.
goto FINISH

:FINISH
echo.
echo ================================================================
echo                 PROCESS FINISHED
echo ================================================================
echo.
pause
endlocal
