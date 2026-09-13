@echo off
REM Builds the Windows .exe for the YOLO Tactical Tracker.
REM Run this from the repo root, or just double-click it - it cd's into
REM the repo root itself either way.
REM
REM Must be run on Windows: PyInstaller packages for whatever OS it runs
REM on, it does not cross-compile.

cd /d "%~dp0\.."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

pyinstaller --noconfirm packaging\yolo_tracker.spec

echo.
echo Build finished. The app is in dist\YoloTacticalTracker\
echo Ship that whole folder - YoloTacticalTracker.exe depends on the files next to it.
