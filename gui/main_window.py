"""estimationFace main window — three tabs: Offspring | Age | 3D Map.

Each tab takes local image(s) or a webcam snapshot, runs the relevant offline
core feature on a background thread, and previews / saves the result. Nothing
here ever touches the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from core import blending, camera, mapping
from core.logging_setup import get_logger
from core.paths import ASSETS_DIR, ROOT
from .backend import BACKEND, Worker, bgr_to_pixmap

log = get_logger("ui")

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp)"


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
class ImagePanel(QWidget):
    """A titled preview box that holds a BGR ndarray, loadable from file/webcam."""

    def __init__(self, title: str, with_webcam: bool = True) -> None:
        super().__init__()
        self.image: Optional[np.ndarray] = None
        layout = QVBoxLayout(self)
        self._label = QLabel(f"{title}\n(no image)")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(300, 300)
        self._label.setStyleSheet(
            "border:1px solid #888; background:#1b1b1b; color:#aaa;"
        )
        layout.addWidget(self._label)
        row = QHBoxLayout()
        btn_file = QPushButton("Load Image…")
        btn_file.clicked.connect(self._load_file)
        row.addWidget(btn_file)
        if with_webcam:
            btn_cam = QPushButton("Webcam…")
            btn_cam.clicked.connect(self._load_webcam)
            row.addWidget(btn_cam)
        layout.addLayout(row)
        self._title = title

    def _load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", _IMG_FILTER)
        if path:
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Load failed", f"Could not read {path}")
                return
            self.set_image(img)

    def _load_webcam(self) -> None:
        dlg = CameraDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.frame is not None:
            self.set_image(dlg.frame)

    def set_image(self, image_bgr: np.ndarray) -> None:
        self.image = image_bgr
        self._label.setPixmap(bgr_to_pixmap(image_bgr, 380))

    def set_title(self, title: str) -> None:
        self._title = title
        if self.image is None:
            self._label.setText(f"{title}\n(no image)")

    def has_image(self) -> bool:
        return self.image is not None


class CameraDialog(QDialog):
    """Live webcam preview with a capture button."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Webcam — capture a frame")
        self.frame: Optional[np.ndarray] = None
        self._cam: Optional[camera.CameraCapture] = None

        layout = QVBoxLayout(self)
        self._view = QLabel("Starting camera…")
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setMinimumSize(640, 480)
        self._view.setStyleSheet("background:#000; color:#ccc;")
        layout.addWidget(self._view)

        row = QHBoxLayout()
        self._device = QComboBox()
        for idx in camera.enumerate_cameras() or [0]:
            self._device.addItem(f"Camera {idx}", idx)
        row.addWidget(self._device)
        btn_start = QPushButton("Start")
        btn_start.clicked.connect(self._start)
        row.addWidget(btn_start)
        btn_cap = QPushButton("Capture")
        btn_cap.clicked.connect(self._capture)
        row.addWidget(btn_cap)
        layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _start(self) -> None:
        self._stop_cam()
        try:
            self._cam = camera.CameraCapture(index=self._device.currentData() or 0)
            self._cam.start()
            self._timer.start(33)
        except Exception as exc:
            QMessageBox.warning(self, "Camera", str(exc))

    def _tick(self) -> None:
        if self._cam is None:
            return
        frame = self._cam.read()
        if frame is not None:
            self.frame = frame
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self._view.setPixmap(QPixmap.fromImage(qimg).scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _capture(self) -> None:
        if self.frame is None:
            QMessageBox.information(self, "Capture", "No frame yet — press Start.")
            return
        self.accept()

    def _stop_cam(self) -> None:
        self._timer.stop()
        if self._cam is not None:
            self._cam.stop()
            self._cam = None

    def closeEvent(self, event) -> None:
        self._stop_cam()
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop_cam()
        super().reject()


class ResultPanel(QWidget):
    """Preview of a generated result with a Save button."""

    def __init__(self) -> None:
        super().__init__()
        self.image: Optional[np.ndarray] = None
        layout = QVBoxLayout(self)
        self._label = QLabel("Result")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(400, 400)
        self._label.setStyleSheet(
            "border:1px solid #4a8; background:#111; color:#8c8;"
        )
        layout.addWidget(self._label)
        self._save = QPushButton("Save Result…")
        self._save.clicked.connect(self._save_result)
        self._save.setEnabled(False)
        layout.addWidget(self._save)

    def set_result(self, image_bgr: np.ndarray) -> None:
        self.image = image_bgr
        self._label.setPixmap(bgr_to_pixmap(image_bgr, 460))
        self._save.setEnabled(True)

    def _save_result(self) -> None:
        if self.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", str(ROOT / "result.png"), "PNG (*.png)")
        if path:
            cv2.imwrite(path, self.image)


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setRange(minimum, maximum)
    s.setValue(value)
    return s


class LivePreview:
    """Webcam live-preview helper with a busy-guarded background worker.

    Each tick grabs the latest camera frame and, only if the previous frame
    finished, dispatches ``_live_process(frame)`` to a worker thread — so a slow
    per-frame op (offspring ~1 fps, fast aging ~1 fps) never freezes the UI; it
    just drops intermediate frames. Subclasses implement ``_live_process`` and
    have a ``result`` ResultPanel and a ``status`` QLabel.
    """

    def _live_init(self) -> None:
        self._cam: Optional[camera.CameraCapture] = None
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)
        self._live_busy = False
        self._live_worker: Optional[Worker] = None

    def _live_start(self, interval_ms: int = 120) -> bool:
        try:
            self._cam = camera.CameraCapture(index=0)
            self._cam.start()
            self._live_timer.start(interval_ms)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Camera", str(exc))
            return False

    def _live_stop(self) -> None:
        self._live_timer.stop()
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
        self._live_busy = False

    def _live_tick(self) -> None:
        if self._cam is None or self._live_busy:
            return
        frame = self._cam.read()
        if frame is None:
            return
        self._live_busy = True
        self._live_worker = Worker(lambda f=frame: self._live_process(f))
        self._live_worker.finished_ok.connect(self._live_ok)
        self._live_worker.failed.connect(self._live_err)
        self._live_worker.start()

    def _live_ok(self, image) -> None:
        self._live_busy = False
        if image is not None:
            self.result.set_result(image)

    def _live_err(self, msg: str) -> None:
        self._live_busy = False
        self.status.setText(f"Live: {msg}")

    def _live_process(self, frame):  # pragma: no cover - overridden
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FR-2 tab
# ---------------------------------------------------------------------------
class OffspringTab(QWidget, LivePreview):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[Worker] = None
        self._live_init()
        root = QHBoxLayout(self)

        self.parent_a = ImagePanel("Parent A")
        self.parent_b = ImagePanel("Parent B")
        inputs = QVBoxLayout()
        inputs.addWidget(self.parent_a)
        inputs.addWidget(self.parent_b)

        controls = QVBoxLayout()
        self.single = QCheckBox("Single image with two people (left=A, right=B)")
        self.single.stateChanged.connect(self._on_single_toggled)
        controls.addWidget(self.single)

        # One-shot: grab a webcam frame containing BOTH people (left=A, right=B).
        self.btn_two = QPushButton("📷 Webcam: two people in one shot")
        self.btn_two.clicked.connect(self._webcam_two_people)
        controls.addWidget(self.btn_two)

        self.alpha = _slider(0, 100, 50)
        self._alpha_lbl = QLabel("Blend (A ↔ B): 50%")
        self.alpha.valueChanged.connect(
            lambda v: self._alpha_lbl.setText(f"Blend (A ↔ B): {v}%"))
        controls.addWidget(self._alpha_lbl)
        controls.addWidget(self.alpha)

        self.harmonize = QCheckBox("Color / lighting harmonization")
        self.harmonize.setChecked(True)
        controls.addWidget(self.harmonize)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.go = QPushButton("Generate Offspring")
        self.go.clicked.connect(self._run)
        controls.addWidget(self.go)
        self.live = QCheckBox("Live webcam preview (two people, ~1 fps on CPU)")
        self.live.stateChanged.connect(self._toggle_live)
        controls.addWidget(self.live)
        controls.addWidget(self.status)
        controls.addStretch(1)

        self.result = ResultPanel()

        root.addLayout(inputs, 2)
        root.addLayout(controls, 2)
        root.addWidget(self.result, 3)

    def _on_single_toggled(self) -> None:
        single = self.single.isChecked()
        self.parent_b.setEnabled(not single)
        self.parent_a.set_title(
            "Two people (A=left, B=right)" if single else "Parent A")

    def _webcam_two_people(self) -> None:
        """Capture one webcam frame with both people; auto-split left/right."""
        dlg = CameraDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.frame is not None:
            self.single.setChecked(True)          # both faces come from one frame
            self.parent_a.set_image(dlg.frame)
            self.status.setText(
                "Captured. Left face → Parent A, right face → Parent B.")

    def _run(self) -> None:
        if not self.parent_a.has_image():
            QMessageBox.information(self, "Input", "Load Parent A first.")
            return
        single = self.single.isChecked()
        if not single and not self.parent_b.has_image():
            QMessageBox.information(self, "Input", "Load Parent B (or tick single image).")
            return
        alpha = self.alpha.value() / 100.0
        harmonize = self.harmonize.isChecked()
        img_a = self.parent_a.image
        img_b = None if single else self.parent_b.image

        def job():
            with BACKEND.infer_lock:
                return blending.generate_offspring(
                    img_a, img_b, alpha=alpha, harmonize=harmonize,
                    mesh=BACKEND.mesh())

        self._start(job, "Morphing parents…")

    # -- live mode -----------------------------------------------------------
    def _toggle_live(self) -> None:
        if self.live.isChecked():
            self.single.setChecked(True)          # live uses one two-person frame
            if self._live_start(interval_ms=150):
                self.status.setText("Live: put both faces in view…")
            else:
                self.live.setChecked(False)
        else:
            self._live_stop()
            self.status.setText("Live stopped.")

    def _live_process(self, frame):
        alpha = self.alpha.value() / 100.0
        harmonize = self.harmonize.isChecked()
        with BACKEND.infer_lock:
            try:
                r = blending.generate_offspring(
                    frame, None, alpha=alpha, harmonize=harmonize,
                    canvas=384, mesh=BACKEND.mesh())
                return r.image
            except Exception:
                return frame           # <2 faces yet — show the raw camera feed

    def stop_live(self) -> None:
        self._live_stop()

    def _start(self, job, msg: str) -> None:
        self.go.setEnabled(False)
        self.status.setText(msg)
        self._worker = Worker(job)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, result) -> None:
        self.result.set_result(result.image)
        self.status.setText(f"Done (alpha={result.alpha:.2f}).")
        self.go.setEnabled(True)

    def _error(self, msg: str) -> None:
        self.status.setText(f"Error: {msg}")
        self.go.setEnabled(True)


# ---------------------------------------------------------------------------
# FR-3 tab
# ---------------------------------------------------------------------------
class AgeTab(QWidget, LivePreview):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[Worker] = None
        self._live_init()
        root = QHBoxLayout(self)

        self.input = ImagePanel("Portrait")
        controls = QVBoxLayout()

        self.src = _slider(1, 90, 25)
        self._src_lbl = QLabel("Source age: 25")
        self.src.valueChanged.connect(
            lambda v: self._src_lbl.setText(f"Source age: {v}"))
        self.tgt = _slider(5, 80, 70)
        self._tgt_lbl = QLabel("Target age: 70")
        self.tgt.valueChanged.connect(
            lambda v: self._tgt_lbl.setText(f"Target age: {v}"))
        self.strength = _slider(0, 150, 100)
        self._str_lbl = QLabel("Strength: 100%")
        self.strength.valueChanged.connect(
            lambda v: self._str_lbl.setText(f"Strength: {v}%"))

        for w in (self._src_lbl, self.src, self._tgt_lbl, self.tgt,
                  self._str_lbl, self.strength):
            controls.addWidget(w)

        self.go = QPushButton("Age Face (high quality, ~7s)")
        self.go.clicked.connect(self._run)
        controls.addWidget(self.go)
        self.live = QCheckBox("Live webcam preview (fast mode, ~1 fps on CPU)")
        self.live.stateChanged.connect(self._toggle_live)
        controls.addWidget(self.live)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        self.identity = QLabel("")
        self.identity.setWordWrap(True)
        controls.addWidget(self.identity)
        controls.addStretch(1)

        self.result = ResultPanel()
        root.addWidget(self.input, 2)
        root.addLayout(controls, 2)
        root.addWidget(self.result, 3)

    def _run(self) -> None:
        if not self.input.has_image():
            QMessageBox.information(self, "Input", "Load a portrait first.")
            return
        img = self.input.image
        src, tgt = self.src.value(), self.tgt.value()
        strength = self.strength.value() / 100.0

        def job():
            from core import aging
            with BACKEND.infer_lock:
                return aging.progress_age(
                    img, src, tgt, strength=strength,
                    engine=BACKEND.aging(), mesh=BACKEND.mesh())

        self.go.setEnabled(False)
        self.status.setText("Ageing (a few seconds on CPU)…")
        self.identity.setText("")
        self._worker = Worker(job)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, result) -> None:
        self.result.set_result(result.image)
        self.status.setText(
            f"Aged {result.source_age} → {result.target_age}.")
        if result.resemblance is not None:
            r = result.resemblance
            self.identity.setText("Identity: " + r.summary())
            self.identity.setStyleSheet(
                "color:#6c6;" if r.passed else "color:#e88;")
        self.go.setEnabled(True)

    def _error(self, msg: str) -> None:
        self.status.setText(f"Error: {msg}")
        self.go.setEnabled(True)

    # -- live mode -----------------------------------------------------------
    def _toggle_live(self) -> None:
        if self.live.isChecked():
            if self._live_start(interval_ms=150):
                self.status.setText("Live ageing (fast mode)…")
                self.identity.setText("")
            else:
                self.live.setChecked(False)
        else:
            self._live_stop()
            self.status.setText("Live stopped.")

    def _live_process(self, frame):
        from core import aging
        src, tgt = self.src.value(), self.tgt.value()
        strength = self.strength.value() / 100.0
        with BACKEND.infer_lock:
            try:
                r = aging.progress_age(
                    frame, src, tgt, strength=strength, engine=BACKEND.aging(),
                    mesh=BACKEND.mesh(), fast=True)
                return r.image
            except Exception:
                return frame           # no face yet — show the raw feed

    def stop_live(self) -> None:
        self._live_stop()


# ---------------------------------------------------------------------------
# FR-4 tab
# ---------------------------------------------------------------------------
class MapTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[Worker] = None
        self._overlay: Optional[mapping.MeshOverlay] = None
        self._mesh_path: Optional[str] = None
        self._texture: Optional[np.ndarray] = None
        root = QHBoxLayout(self)

        self.input = ImagePanel("Target face")
        controls = QVBoxLayout()

        self.mode = QComboBox()
        self.mode.addItems(["2D image / mask (homography)", "3D mesh (OBJ/glTF)"])
        controls.addWidget(QLabel("Mapping mode:"))
        controls.addWidget(self.mode)

        btn_tex = QPushButton("Choose texture image…")
        btn_tex.clicked.connect(self._choose_texture)
        controls.addWidget(btn_tex)
        btn_mesh = QPushButton("Choose 3D mesh…")
        btn_mesh.clicked.connect(self._choose_mesh)
        controls.addWidget(btn_mesh)
        self.asset_lbl = QLabel("No asset chosen")
        self.asset_lbl.setWordWrap(True)
        controls.addWidget(self.asset_lbl)

        self.opacity = _slider(0, 100, 85)
        self._op_lbl = QLabel("Opacity: 85%")
        self.opacity.valueChanged.connect(
            lambda v: self._op_lbl.setText(f"Opacity: {v}%"))
        controls.addWidget(self._op_lbl)
        controls.addWidget(self.opacity)

        self.scale = _slider(50, 200, 110)
        self._sc_lbl = QLabel("Mesh scale: 110")
        self.scale.valueChanged.connect(
            lambda v: self._sc_lbl.setText(f"Mesh scale: {v}"))
        controls.addWidget(self._sc_lbl)
        controls.addWidget(self.scale)

        self.go = QPushButton("Map onto face")
        self.go.clicked.connect(self._run)
        controls.addWidget(self.go)
        self.live = QCheckBox("Live webcam mapping")
        self.live.stateChanged.connect(self._toggle_live)
        controls.addWidget(self.live)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch(1)

        self.result = ResultPanel()
        root.addWidget(self.input, 2)
        root.addLayout(controls, 2)
        root.addWidget(self.result, 3)

        self._cam: Optional[camera.CameraCapture] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._live_tick)

    def _choose_texture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose texture", str(ASSETS_DIR / "textures"), _IMG_FILTER)
        if path:
            self._texture = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.asset_lbl.setText(f"Texture: {Path(path).name}")
            self.mode.setCurrentIndex(0)

    def _choose_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose mesh", str(ASSETS_DIR / "meshes"),
            "Meshes (*.obj *.glb *.gltf *.ply *.stl)")
        if path:
            self._mesh_path = path
            self._overlay = None
            self.asset_lbl.setText(f"Mesh: {Path(path).name}")
            self.mode.setCurrentIndex(1)

    def _process(self, frame: np.ndarray) -> np.ndarray:
        with BACKEND.infer_lock:
            face = BACKEND.mesh().detect_one(frame)
            if face is None:
                return frame
            if self.mode.currentIndex() == 0:
                if self._texture is None:
                    raise ValueError("Choose a texture image first.")
                return BACKEND.mapper().map_image(
                    frame, face, self._texture, self.opacity.value() / 100.0)
            if self._mesh_path is None:
                raise ValueError("Choose a 3D mesh first.")
            if self._overlay is None:
                self._overlay = mapping.MeshOverlay(self._mesh_path)
            return self._overlay.render(
                frame, face, scale=float(self.scale.value()),
                opacity=self.opacity.value() / 100.0)

    def _run(self) -> None:
        if not self.input.has_image():
            QMessageBox.information(self, "Input", "Load a target face first.")
            return
        frame = self.input.image

        def job():
            return self._process(frame)

        self.go.setEnabled(False)
        self.status.setText("Mapping…")
        self._worker = Worker(job)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _done(self, image) -> None:
        self.result.set_result(image)
        self.status.setText("Done.")
        self.go.setEnabled(True)

    def _error(self, msg: str) -> None:
        self.status.setText(f"Error: {msg}")
        self.go.setEnabled(True)

    # -- live mode -----------------------------------------------------------
    def _toggle_live(self) -> None:
        if self.live.isChecked():
            try:
                self._cam = camera.CameraCapture(index=0)
                self._cam.start()
                self._timer.start(60)
                self.status.setText("Live mapping…")
            except Exception as exc:
                self.live.setChecked(False)
                QMessageBox.warning(self, "Camera", str(exc))
        else:
            self._timer.stop()
            if self._cam is not None:
                self._cam.stop()
                self._cam = None
            self.status.setText("Live stopped.")

    def _live_tick(self) -> None:
        if self._cam is None:
            return
        frame = self._cam.read()
        if frame is None:
            return
        try:
            self.result.set_result(self._process(frame))
        except Exception as exc:
            self.status.setText(f"Live error: {exc}")

    def stop_live(self) -> None:
        self._timer.stop()
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("estimationFace — Offline Face Estimation Suite")
        self.resize(1280, 760)
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.offspring = OffspringTab()
        self.age = AgeTab()
        self.map = MapTab()
        self.tabs.addTab(self.offspring, "Offspring (FR-2)")
        self.tabs.addTab(self.age, "Age Progression (FR-3)")
        self.tabs.addTab(self.map, "3D / Texture Map (FR-4)")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        footer = QLabel("100% offline · InsightFace ArcFace + MediaPipe 468-mesh "
                        "+ FRAN aging · all inference local")
        footer.setStyleSheet("color:#888; padding:4px;")
        layout.addWidget(footer)

    def _stop_all_live(self) -> None:
        """Only one webcam can be open at a time — stop every tab's live mode."""
        for tab in (self.offspring, self.age, self.map):
            tab.stop_live()
        for cb in (self.offspring.live, self.age.live, self.map.live):
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _on_tab_changed(self, _index: int) -> None:
        self._stop_all_live()

    def closeEvent(self, event) -> None:
        self._stop_all_live()
        super().closeEvent(event)
