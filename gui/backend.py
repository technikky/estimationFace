"""Shared, lazily-loaded model backend + a generic worker thread.

Loading the ONNX / MediaPipe models is expensive, so a single instance of each
is created on first use and reused across all three tabs.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from core import aging, blending, identity, landmarks, mapping
from core.logging_setup import get_logger

log = get_logger("backend")


class Backend:
    """Lazy holder for the heavy models (thread-safe enough for our UI usage)."""

    def __init__(self) -> None:
        self._mesh: Optional[landmarks.FaceMesh] = None
        self._aging: Optional[aging.AgingEngine] = None
        self._mapper: Optional[mapping.TextureMapper] = None
        self._encoder = None
        # Serialises inference across tabs/threads. MediaPipe's IMAGE-mode
        # detector is not safe under concurrent calls, so every heavy job
        # (manual or live) runs while holding this lock.
        self.infer_lock = threading.Lock()

    def mesh(self) -> landmarks.FaceMesh:
        if self._mesh is None:
            self._mesh = landmarks.FaceMesh(max_faces=2)
        return self._mesh

    def aging(self) -> "aging.AgingEngine":
        if self._aging is None:
            self._aging = aging.AgingEngine()
        return self._aging

    def mapper(self) -> mapping.TextureMapper:
        if self._mapper is None:
            self._mapper = mapping.TextureMapper()
        return self._mapper

    def encoder(self):
        if self._encoder is None:
            self._encoder = identity.get_encoder()
        return self._encoder


BACKEND = Backend()


class Worker(QThread):
    """Runs a callable off the UI thread; emits the result or an error string."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # noqa: D401
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:  # surfaced to the user as a message
            log.exception("Worker failed")
            self.failed.emit(str(exc))


def bgr_to_pixmap(image_bgr: np.ndarray, max_side: int = 480) -> QPixmap:
    """Convert a BGR ndarray to a QPixmap, scaled to fit ``max_side``."""
    h, w = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if max(w, h) > max_side:
        pix = pix.scaled(max_side, max_side,
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix
