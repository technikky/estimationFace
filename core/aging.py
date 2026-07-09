"""FR-3  Identity-preserving age progression (FRAN, offline ONNX Runtime).

Wraps the Face Re-Aging Network (``face_reaging.onnx``). The model is a U-Net
that takes a 5-channel tensor ``[R, G, B, source_age/100, target_age/100]`` at
512x512 and returns a 3-channel **residual** that is added back to the RGB image.
Working at 1024x1024 with a 512 sliding window (stride 256) preserves fine
detail. Re-implemented with numpy/opencv only — no torch, no face_recognition,
no network.

Identity preservation: FRAN keeps identity by construction; we additionally
report an ArcFace resemblance score (aged vs. source) and expose a ``strength``
control that scales the residual so the operator can trade ageing intensity
against resemblance (the PRS "identity lock").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .landmarks import FaceLandmarks, FaceMesh, crop_face
from .logging_setup import get_logger
from .paths import AGING_MODEL, require

log = get_logger("aging")

_WORK = 1024        # crop is processed at this resolution
_WINDOW = 512       # FRAN native window
_STRIDE = 256


@dataclass
class AgingResult:
    image: np.ndarray                 # full BGR image with the aged face composited
    crop_box: tuple                   # (x0, y0, x1, y1) region that was aged
    source_age: int
    target_age: int
    resemblance: Optional[object] = None   # identity.IdentityResult or None


def _hann_window(size: int) -> np.ndarray:
    """2D raised-cosine window (1 at centre -> 0 at edges) for seamless tiling."""
    w = np.hanning(size)
    win = np.outer(w, w).astype(np.float32)
    return np.clip(win, 1e-3, 1.0)


def _oval_mask(size: int, feather: int = 61) -> np.ndarray:
    """Feathered oval that limits the residual to the face and fades at borders."""
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(mask, (size // 2, size // 2),
                (int(size * 0.46), int(size * 0.48)), 0, 0, 360, 1.0, -1)
    k = feather * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


class AgingEngine:
    """Loads FRAN once and ages face crops."""

    def __init__(self) -> None:
        import onnxruntime as ort

        model_path = str(require(AGING_MODEL, "FRAN face_reaging.onnx"))
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._small_mask = _hann_window(_WINDOW)
        self._big_mask = _oval_mask(_WORK)
        log.info("AgingEngine ready (FRAN)")

    def _infer_window(self, window_5ch: np.ndarray) -> np.ndarray:
        """Run FRAN on one (5,512,512) window -> (512,512,3) residual (HWC)."""
        blob = window_5ch[None, ...].astype(np.float32)          # (1,5,512,512)
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        return np.transpose(out[0], (1, 2, 0))                   # (512,512,3)

    def _age_crop_fast(self, crop_bgr: np.ndarray, source_age: int,
                       target_age: int, strength: float) -> np.ndarray:
        """Single-window (512) aging for live preview — one inference, ~1s CPU.

        Lower detail than :meth:`_age_crop` (no 1024 sliding window) but fast
        enough for an interactive webcam preview.
        """
        oh, ow = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        work = cv2.resize(rgb, (_WINDOW, _WINDOW), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(work, (2, 0, 1))
        src_plane = np.full((1, _WINDOW, _WINDOW), source_age / 100.0, np.float32)
        tgt_plane = np.full((1, _WINDOW, _WINDOW), target_age / 100.0, np.float32)
        inp = np.concatenate([chw, src_plane, tgt_plane], axis=0)
        residual = self._infer_window(inp)
        residual *= _oval_mask(_WINDOW)[..., None]
        residual *= float(np.clip(strength, 0.0, 1.5))
        residual = cv2.resize(residual, (ow, oh), interpolation=cv2.INTER_LINEAR)
        aged = np.clip(rgb + residual, 0.0, 1.0)
        return cv2.cvtColor((aged * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _age_crop(self, crop_bgr: np.ndarray, source_age: int,
                  target_age: int, strength: float) -> np.ndarray:
        """Return an aged crop (BGR uint8), same size as the input crop."""
        oh, ow = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        work = cv2.resize(rgb, (_WORK, _WORK), interpolation=cv2.INTER_LINEAR)

        chw = np.transpose(work, (2, 0, 1))                      # (3,1024,1024)
        src_plane = np.full((1, _WORK, _WORK), source_age / 100.0, np.float32)
        tgt_plane = np.full((1, _WORK, _WORK), target_age / 100.0, np.float32)
        inp = np.concatenate([chw, src_plane, tgt_plane], axis=0)  # (5,1024,1024)

        residual = np.zeros((_WORK, _WORK, 3), np.float32)
        count = np.zeros((_WORK, _WORK, 1), np.float32)
        sm = self._small_mask[..., None]
        step = _STRIDE
        add = 2 if _WINDOW % step != 0 else 1
        for y in range(0, _WORK - _WINDOW + add, step):
            for x in range(0, _WORK - _WINDOW + add, step):
                win = inp[:, y:y + _WINDOW, x:x + _WINDOW]
                out = self._infer_window(win)
                residual[y:y + _WINDOW, x:x + _WINDOW] += out * sm
                count[y:y + _WINDOW, x:x + _WINDOW] += sm
        residual /= np.clip(count, 1e-3, None)
        residual *= self._big_mask[..., None]                    # face-limited
        residual *= float(np.clip(strength, 0.0, 1.5))

        residual = cv2.resize(residual, (ow, oh), interpolation=cv2.INTER_LINEAR)
        aged = np.clip(rgb + residual, 0.0, 1.0)
        return cv2.cvtColor((aged * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def age(
        self, image_bgr: np.ndarray, face: FaceLandmarks,
        source_age: int, target_age: int, strength: float = 1.0,
        check_identity: bool = True, fast: bool = False,
    ) -> AgingResult:
        """Age a single detected face and composite it back into the image.

        ``fast=True`` uses a single 512 window (~1s CPU) for live preview;
        the default sliding-window path is higher quality but ~7s.
        """
        crop, box = crop_face(image_bgr, face, margin=0.5)
        if fast:
            aged_crop = self._age_crop_fast(crop, source_age, target_age, strength)
        else:
            aged_crop = self._age_crop(crop, source_age, target_age, strength)

        out = image_bgr.copy()
        x0, y0, x1, y1 = box
        # Feathered composite of the aged crop back over the original.
        h, w = aged_crop.shape[:2]
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.5), int(h * 0.5)),
                    0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (61, 61), 0)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(
            aged_crop.astype(np.float32) * m + region * (1 - m), 0, 255
        ).astype(np.uint8)

        resemblance = None
        if check_identity:
            resemblance = self._resemblance(image_bgr, face, out, box)

        return AgingResult(
            image=out, crop_box=box, source_age=source_age,
            target_age=target_age, resemblance=resemblance,
        )

    def _resemblance(self, src_img, src_face, aged_img, box):
        """ArcFace resemblance between the source face and the aged result."""
        try:
            from . import identity
            enc = identity.get_encoder()
            src_vec = enc.embed_face(src_img, src_face)
            # Re-detect the aged face within its crop box for a fair alignment.
            mesh = FaceMesh(max_faces=1)
            try:
                x0, y0, x1, y1 = box
                aged_face = mesh.detect_one(aged_img[y0:y1, x0:x1])
                if aged_face is None:
                    return None
                aged_vec = enc.embed_face(aged_img[y0:y1, x0:x1], aged_face)
            finally:
                mesh.close()
            res = identity.compare(src_vec, aged_vec)
            log.info("Aging identity check: %s", res.summary())
            return res
        except Exception as exc:  # identity check is best-effort
            log.warning("Resemblance check skipped: %s", exc)
            return None


def progress_age(
    image_bgr: np.ndarray, source_age: int, target_age: int,
    strength: float = 1.0, engine: Optional[AgingEngine] = None,
    mesh: Optional[FaceMesh] = None, fast: bool = False,
) -> AgingResult:
    """High-level entry: age the primary face in ``image_bgr``.

    ``fast=True`` skips the identity check and uses the single-window path for
    interactive live preview.
    """
    own_mesh = mesh is None
    mesh = mesh or FaceMesh(max_faces=1)
    engine = engine or AgingEngine()
    try:
        face = mesh.detect_one(image_bgr)
        if face is None:
            raise ValueError("No face detected for age progression.")
        return engine.age(image_bgr, face, source_age, target_age, strength,
                          check_identity=not fast, fast=fast)
    finally:
        if own_mesh:
            mesh.close()
