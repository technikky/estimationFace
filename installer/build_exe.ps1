# OPTIONAL: build a single-folder .exe with PyInstaller (offline).
#
# The portable runtime\ bundle is the primary deliverable and needs no build.
# Use this only if you want a dist\estimationFace\estimationFace.exe.
#
# Requires a full CPython venv that already has the project deps + PyInstaller.
# By default it reuses the PyInstaller shipped in bgsegment\.venv.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)

$venvPy = "D:\tools-offline\1Project\bgsegment\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "Build venv python not found at $venvPy. Point this at a venv that has numpy/opencv/mediapipe/onnxruntime/trimesh/PySide6 + PyInstaller."
    exit 1
}

Write-Host "Building estimationFace with PyInstaller (one-folder)…"
& $venvPy -m PyInstaller (Join-Path $root "installer\estimationface.spec") --noconfirm --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build")

Write-Host ""
Write-Host "Done. Executable: $root\dist\estimationFace\estimationFace.exe"
Write-Host "Copy the whole dist\estimationFace folder to the offline machine."
