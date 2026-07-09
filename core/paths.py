"""Project path resolution (100% offline).

Everything resolves relative to the project root so the app runs from any
working directory and from inside a PyInstaller bundle. The root is the folder
that contains ``models/``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Return the project root.

    * Normal run: this file lives at ``<root>/core/paths.py`` -> one parent up
      from the ``core`` package directory.
    * Frozen (PyInstaller): resources are unpacked next to the executable, so
      the root is the folder holding ``app.exe``.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = ROOT / "logs"
TEST_DIR = ROOT / "test"

# Individual model files / dirs (all local, bundled with the app).
MEDIAPIPE_MODEL = MODELS_DIR / "mediapipe" / "face_landmarker.task"
AGING_MODEL = MODELS_DIR / "aging" / "face_reaging.onnx"
RECOGNITION_MODEL = MODELS_DIR / "recognition" / "w600k_r50.onnx"
DETECTION_DIR = MODELS_DIR / "detection"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require(path: Path, what: str) -> Path:
    """Return ``path`` if it exists, else raise a clear offline-setup error."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. This file must be bundled locally for "
            f"offline operation. See README.md (Model Assets)."
        )
    return path
