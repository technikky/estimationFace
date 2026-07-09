"""estimationFace — application entry point (100% offline).

Launch with the bundled runtime:
    runtime\\python.exe app.py

Offline features:
  * FR-2  Offspring generation  (parent blend / two-people frame)
  * FR-3  Identity-preserving age progression (FRAN)
  * FR-4  3D model & user-image mapping
"""
from __future__ import annotations

import os
import sys

# Belt-and-braces: make sure nothing tries to phone home for models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    log = setup_logging()
    log.info("Starting estimationFace")

    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("estimationFace")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
