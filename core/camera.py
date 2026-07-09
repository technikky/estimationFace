"""Threaded webcam capture (latest-frame-wins) for photo / live inputs.

Tries Media Foundation then DirectShow on Windows so most UVC webcams open.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger("camera")

_BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]


def enumerate_cameras(max_probe: int = 6) -> List[int]:
    found = []
    for index in range(max_probe):
        for _, be in _BACKENDS:
            cap = cv2.VideoCapture(index, be)
            ok = cap.isOpened()
            cap.release()
            if ok:
                found.append(index)
                break
    return found


class CameraCapture:
    """Background camera reader serving the most recent frame."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 mirror: bool = True) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False

    def _open(self) -> cv2.VideoCapture:
        for name, be in _BACKENDS:
            cap = cv2.VideoCapture(self.index, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                for _ in range(10):
                    ok, _f = cap.read()
                    if ok:
                        log.info("Camera %d opened via %s", self.index, name)
                        return cap
                    time.sleep(0.05)
                cap.release()
        raise RuntimeError(
            f"Cannot open camera {self.index}. Is it connected and free?"
        )

    def start(self) -> None:
        if self._running:
            return
        self._cap = self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
