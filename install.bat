@echo off
setlocal
cd /d "%~dp0"
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" set "UV=uv"
if not exist ".venv\Scripts\python.exe" (
    "%UV%" venv --python 3.10 .venv || exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,10), 'Python 3.10 required'" || exit /b 1
"%UV%" pip install --python ".venv\Scripts\python.exe" -r requirements.txt || exit /b 1
if not exist bin mkdir bin
if not exist jpg mkdir jpg
".venv\Scripts\python.exe" -c "import tensorflow, RangeCoder, skimage, matplotlib; from tensorflow.keras.models import load_model; e=load_model('models/encoder_1_2.h5',compile=False); d=load_model('models/decoder_1_2.h5',compile=False); import numpy as np; y=d(e(np.zeros((1,128,128,3),np.float32))); assert tuple(y.shape)==(1,128,128,3); print('Codec environment ready')"
exit /b %errorlevel%
