#!/usr/bin/env bash
# Builds a native executable for the YOLO Tactical Tracker on whatever OS
# this script runs on (Linux/macOS). For a Windows .exe, run build.bat on
# an actual Windows machine instead - PyInstaller packages for the OS it
# runs on, it does not cross-compile.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller

pyinstaller --noconfirm packaging/yolo_tracker.spec

echo
echo "Build finished. The app is in dist/YoloTacticalTracker/"
echo "Ship that whole folder - the executable inside depends on the files next to it."
