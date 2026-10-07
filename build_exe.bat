@echo off
REM Compila RSL_FFT.exe (singolo file, senza console). Richiede Python 3.9+ per Windows.
setlocal
cd /d "%~dp0"

if not exist .venv (
    python -m venv .venv
    if errorlevel 1 goto :err
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :err

python rsl_core.py --selftest
if errorlevel 1 goto :err

pyinstaller --noconfirm --clean --onefile --windowed --name RSL_FFT ^
    --hidden-import matplotlib.backends.backend_tkagg --hidden-import PIL._tkinter_finder rsl_fft_gui.py
if errorlevel 1 goto :err

echo.
echo Fatto: dist\RSL_FFT.exe
goto :eof

:err
echo.
echo ERRORE durante la compilazione.
exit /b 1
