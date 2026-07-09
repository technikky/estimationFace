"""Face detection & 468/478-point landmarking (offline, MediaPipe Tasks API).

This is the shared geometry layer used by every feature:

* FR-2 offspring blending needs the full dense mesh of both parents.
* FR-3 aging needs a face crop and an ArcFace-aligned chip for the identity lock.
* FR-4 mapping needs the mesh + a face bounding box.

Built on ``mediapipe.tasks.vision.FaceLandmarker`` in IMAGE running mode. The
model file ``face_landmarker.task`` (bundled under ``models/mediapipe/``) yields
478 landmarks per face (468 mesh + 10 iris). Everything runs on CPU with no
network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .logging_setup import get_logger
from .paths import MEDIAPIPE_MODEL, require

log = get_logger("landmarks")

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    mp = vision = BaseOptions = None
    _IMPORT_ERROR = exc

# --- semantic landmark indices (MediaPipe canonical face mesh) --------------
# Clusters are averaged into a single centroid to be robust to noise.
_EYE_A = [33, 133, 159, 145, 158, 153]     # one eye (side decided by x later)
_EYE_B = [362, 263, 386, 374, 385, 380]    # the other eye
_NOSE_TIP = 1
_MOUTH_L = 61
_MOUTH_R = 291

# ArcFace 112x112 reference template, ordered
# [left_eye, right_eye, nose, mouth_left, mouth_right] in IMAGE coordinates
# (left == smaller x). This is the standard insightface ``arcface_src``.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceLandmarks:
    """Dense landmarks and derived geometry for one detected face."""

    points: np.ndarray          # (N, 2) float32 pixel coordinates (N == 478)
    image_size: tuple           # (width, height) of the source image

    @property
    def bbox(self) -> tuple:
        """Return (x0, y0, x1, y1) integer bounding box clamped to the image."""
        w, h = self.image_size
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return (
            int(max(0, np.floor(x0))), int(max(0, np.floor(y0))),
            int(min(w, np.ceil(x1))), int(min(h, np.ceil(y1))),
        )

    @property
    def center(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def five_points(self) -> np.ndarray:
        """Return the 5 alignment points [L-eye, R-eye, nose, L-mouth, R-mouth].

        'Left'/'right' are decided by image x so the result is invariant to
        mirroring or moderate head roll.
        """
        eye_a = self.points[_EYE_A].mean(axis=0)
        eye_b = self.points[_EYE_B].mean(axis=0)
        left_eye, right_eye = sorted((eye_a, eye_b), key=lambda p: p[0])
        m_a, m_b = self.points[_MOUTH_L], self.points[_MOUTH_R]
        left_mouth, right_mouth = sorted((m_a, m_b), key=lambda p: p[0])
        nose = self.points[_NOSE_TIP]
        return np.array(
            [left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32
        )


class FaceMesh:
    """Static-image face landmarker, cached per ``max_faces`` setting."""

    # MediaPipe's face detector works best when faces are a modest fraction of
    # the frame. Very large frames (or faces filling the frame) can be missed,
    # so detection runs on a copy capped to this longest side and the landmarks
    # are scaled back to full resolution.
    _DETECT_MAX_SIDE = 900

    def __init__(self, max_faces: int = 2, min_confidence: float = 0.5) -> None:
        if vision is None:  # pragma: no cover
            raise RuntimeError(
                "MediaPipe Tasks API unavailable. Install mediapipe. "
                f"Original error: {_IMPORT_ERROR}"
            )
        model_path = str(require(MEDIAPIPE_MODEL, "MediaPipe face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=int(max_faces),
            min_face_detection_confidence=float(min_confidence),
            min_face_presence_confidence=float(min_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        log.info("FaceMesh ready (max_faces=%d)", max_faces)

    def detect(self, image_bgr: np.ndarray) -> List[FaceLandmarks]:
        """Detect faces in a BGR image, returned sorted left-to-right by x.

        Landmarks are always expressed in the *original* image's pixel
        coordinates, even though detection may run on a downscaled copy.
        """
        h, w = image_bgr.shape[:2]
        scale = min(1.0, self._DETECT_MAX_SIDE / float(max(h, w)))
        if scale < 1.0:
            det_img = cv2.resize(image_bgr, (int(round(w * scale)),
                                             int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        else:
            det_img = image_bgr
        dh, dw = det_img.shape[:2]
        rgb = np.ascontiguousarray(det_img[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        faces: List[FaceLandmarks] = []
        for lm in getattr(result, "face_landmarks", []) or []:
            # Normalised landmarks -> full-resolution pixel coordinates.
            pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
            faces.append(FaceLandmarks(points=pts, image_size=(w, h)))
        faces.sort(key=lambda f: f.center[0])
        log.debug("Detected %d face(s) (det %dx%d)", len(faces), dw, dh)
        return faces

    def detect_one(self, image_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        faces = self.detect(image_bgr)
        return faces[0] if faces else None

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "FaceMesh":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def align_arcface(image_bgr: np.ndarray, face: FaceLandmarks) -> np.ndarray:
    """Return a 112x112 BGR face chip aligned to the ArcFace template."""
    src = face.five_points()
    matrix, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:  # degenerate; fall back to a plain resize of the bbox
        x0, y0, x1, y1 = face.bbox
        crop = image_bgr[y0:y1, x0:x1]
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(image_bgr, matrix, (112, 112), borderValue=0)


def crop_face(
    image_bgr: np.ndarray, face: FaceLandmarks, margin: float = 0.4
) -> tuple:
    """Return (crop_bgr, (x0, y0, x1, y1)) with a symmetric margin around the face.

    The margin generously includes forehead/jaw/hair so downstream aging and
    mapping have context. The returned box is clamped to the image.
    """
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = face.bbox
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * margin), int(bh * margin)
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(w, x1 + mx), min(h, y1 + my)
    return image_bgr[cy0:cy1, cx0:cx1].copy(), (cx0, cy0, cx1, cy1)
