# PyInstaller spec for the YOLO Tactical Tracker.
#
# Build (on Windows, from the repo root):
#     pyinstaller packaging/yolo_tracker.spec
#
# Output goes to dist/YoloTacticalTracker/ - ship that whole folder (the
# .exe inside it depends on the other files next to it). --onedir is used
# instead of --onefile: no self-extraction step on every launch, which
# matters for startup speed on a weak industrial PC.

import os
from PyInstaller.utils.hooks import collect_all

# SPECPATH is injected by PyInstaller at exec time: the directory
# containing this .spec file. Using it (rather than a bare relative path)
# means the build works the same whether PyInstaller is invoked from the
# repo root or from inside packaging/.
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

datas = []
binaries = []
hiddenimports = []

# ultralytics/torch/onnxruntime/cv2 all load extra data files or C
# extensions dynamically, which PyInstaller's static analysis misses -
# collect_all pulls in what each package declares it needs.
for pkg in ('ultralytics', 'torch', 'onnxruntime', 'cv2'):
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    except Exception:
        pass

block_cipher = None

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'main.py')],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # This app only uses PyQt5. If any other Qt binding (PyQt6, PySide2,
    # PySide6) happens to be installed in the build environment - often
    # pulled in transitively by an unrelated package - PyInstaller's Qt
    # hook auto-detects it and refuses to bundle two bindings at once.
    # Excluding the others here avoids that, regardless of what else is
    # installed alongside PyQt5.
    excludes=['PyQt6', 'PySide2', 'PySide6', 'PySide'],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoloTacticalTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='YoloTacticalTracker',
)
