@echo off
REM ============================================================
REM  Gamma Lens - one-click launcher
REM  Double-click this file (or the "Gamma Lens" Desktop icon)
REM  to start the Streamlit app. It opens in your browser.
REM  Close this black window to stop the app.
REM ============================================================
title Gamma Lens
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  Could not find .venv\Scripts\python.exe
    echo  The virtual environment is missing. Create it and install
    echo  requirements before running this launcher.
    echo.
    pause
    exit /b 1
)

echo.
echo  Starting Gamma Lens... a browser tab will open shortly.
echo  Keep this window open while you use the app; close it to stop.
echo.
REM Launch via "python -m streamlit" instead of the streamlit.exe shim:
REM uv builds that shim as a trampoline with a baked-in absolute path that
REM breaks when the venv/OneDrive path shifts ("uv trampoline failed to
REM canonicalize script path"). Going through python.exe sidesteps it.
".venv\Scripts\python.exe" -m streamlit run "streamlit_app.py"

echo.
echo  Gamma Lens has stopped.
pause
