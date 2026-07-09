# PyInstaller one-folder spec for estimationFace (OPTIONAL packaging path).
#
# The primary offline deliverable is the portable runtime\ bundle + Launch.bat.
# This spec is for teams that want a single dist\estimationFace\ folder with an
# .exe. Build (offline) from an environment that already has the deps, e.g. the
# PyInstaller shipped in bgsegment\.venv:
#
#   D:\tools-offline\1Project\bgsegment\.venv\Scripts\python.exe -m PyInstaller ^
#       installer\estimationface.spec --noconfirm
#
# Note: build from a full CPython venv that has numpy/opencv/mediapipe/
# onnxruntime/trimesh/PySide6 installed — NOT from the embedded runtime.

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# Bundle model assets and demo meshes/textures next to the exe.
datas = [
    (os.path.join(ROOT, "models"), "models"),
    (os.path.join(ROOT, "assets"), "assets"),
]
datas += collect_data_files("mediapipe")   # .task/.tflite/.binarypb runtime data

hiddenimports = (
    collect_submodules("mediapipe")
    + collect_submodules("onnxruntime")
    + ["trimesh", "scipy.spatial"]
)

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="estimationFace",
    console=False,          # GUI app: no console window
    icon=None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    name="estimationFace",
)
