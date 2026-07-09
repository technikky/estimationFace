"""Identity embeddings & resemblance validation (offline ArcFace / ONNX Runtime).

Implements the PRS "Identity Vector Validation" commercial requirement: after
generating an aged (or blended) face, compare it against the source to make sure
the person is still recognisable.

Uses the InsightFace ``buffalo_l`` recognition model ``w600k_r50.onnx``
(ResNet-50, 512-d embedding). Runs on CPU via ONNX Runtime; no network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .landmarks import FaceLandmarks, align_arcface
from .logging_setup import get_logger
from .paths import RECOGNITION_MODEL, require

log = get_logger("identity")

# ArcFace preprocessing constants (insightface convention).
_INPUT_MEAN = 127.5
_INPUT_STD = 127.5

# Default acceptance threshold as a resemblance percentage. Cosine similarity is
# mapped linearly to 0-100 (see :func:`resemblance_percent`). 0.75 mirrors the
# PRS target; identity-preserving aging typically lands well above this for the
# same person, while unrelated faces fall far below.
DEFAULT_THRESHOLD = 75.0


@dataclass
class IdentityResult:
    """Outcome of comparing two faces."""

    percent: float          # resemblance 0-100
    cosine: float           # raw cosine similarity -1..1
    passed: bool            # percent >= threshold
    threshold: float

    def summary(self) -> str:
        verdict = "OK" if self.passed else "LOW"
        return (
            f"Resemblance {self.percent:.1f}% (cos={self.cosine:.3f}) "
            f"[{verdict}, threshold {self.threshold:.0f}%]"
        )


class IdentityEncoder:
    """Wraps the ArcFace ONNX session and produces normalised embeddings."""

    def __init__(self) -> None:
        import onnxruntime as ort

        model_path = str(require(RECOGNITION_MODEL, "ArcFace w600k_r50.onnx"))
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info("IdentityEncoder ready (ArcFace r50)")

    def embed_chip(self, chip_bgr: np.ndarray) -> np.ndarray:
        """Embed a 112x112 aligned BGR chip -> L2-normalised 512-d vector."""
        if chip_bgr.shape[:2] != (112, 112):
            chip_bgr = cv2.resize(chip_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        rgb = chip_bgr[:, :, ::-1].astype(np.float32)
        blob = (rgb - _INPUT_MEAN) / _INPUT_STD
        blob = np.transpose(blob, (2, 0, 1))[None, ...]  # (1,3,112,112)
        vec = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_face(self, image_bgr: np.ndarray, face: FaceLandmarks) -> np.ndarray:
        """Align a detected face to the ArcFace template and embed it."""
        return self.embed_chip(align_arcface(image_bgr, face))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already normalised) embeddings."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def resemblance_percent(a: np.ndarray, b: np.ndarray) -> float:
    """Map cosine similarity to a 0-100 resemblance percentage."""
    return float(np.clip(cosine_similarity(a, b), 0.0, 1.0) * 100.0)


def compare(
    a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> IdentityResult:
    cos = cosine_similarity(a, b)
    pct = float(np.clip(cos, 0.0, 1.0) * 100.0)
    return IdentityResult(
        percent=pct, cosine=cos, passed=pct >= threshold, threshold=threshold
    )


# Shared lazily-created encoder so callers don't reload the 167 MB model.
_ENCODER: Optional[IdentityEncoder] = None


def get_encoder() -> IdentityEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = IdentityEncoder()
    return _ENCODER
