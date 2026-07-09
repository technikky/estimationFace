"""FR-2  Offspring face generation (parent blend) — deterministic & offline.

Pipeline (no GAN, no network):

1. Detect the dense mesh for both parents.
2. Similarity-align each parent to a canonical face template on a fixed canvas,
   carrying every landmark through the same transform.
3. ``target = (1-alpha)*alignedA + alpha*alignedB`` — the child's face shape.
4. Delaunay-triangulate the target shape (``cv2.Subdiv2D``) and piecewise-affine
   warp each parent into it.
5. Histogram-match B->A inside the face region (lighting/skin harmonisation) and
   cross-dissolve.
6. Feather-composite the blended face through its convex hull for a natural edge.

``alpha`` biases the child toward parent A (0.0) or parent B (1.0); 0.5 is even.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .landmarks import FaceLandmarks, FaceMesh
from .logging_setup import get_logger

log = get_logger("blending")

# ArcFace-style 5-point template (112 space); scaled to the output canvas below.
_TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32,
)


@dataclass
class OffspringResult:
    image: np.ndarray                 # BGR blended child portrait (canvas sized)
    parent_a_aligned: np.ndarray      # aligned parent A (for the UI preview)
    parent_b_aligned: np.ndarray      # aligned parent B (for the UI preview)
    alpha: float


def _canvas_template(canvas: int) -> np.ndarray:
    """Scale the 112-space template to the output canvas, centred with margin."""
    scale = canvas / 112.0 * 0.85
    tpl = _TEMPLATE_112 * scale
    # Re-centre the template within the canvas.
    tpl += (canvas / 2.0) - tpl.mean(axis=0)
    return tpl.astype(np.float32)


def _boundary_points(canvas: int) -> np.ndarray:
    """8 fixed edge points so the warp covers the whole canvas, not just the face."""
    c = canvas - 1
    m = canvas // 2
    return np.array(
        [[0, 0], [m, 0], [c, 0], [0, m], [c, m], [0, c], [m, c], [c, c]],
        dtype=np.float32,
    )


def _align(image: np.ndarray, face: FaceLandmarks, template: np.ndarray,
           canvas: int) -> Tuple[np.ndarray, np.ndarray]:
    """Warp image+landmarks to the canvas via a similarity fit of the 5 points."""
    matrix, _ = cv2.estimateAffinePartial2D(
        face.five_points(), template, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("Could not align face (degenerate landmarks).")
    aligned_img = cv2.warpAffine(image, matrix, (canvas, canvas), borderValue=0)
    pts = np.hstack([face.points, np.ones((len(face.points), 1), np.float32)])
    aligned_pts = (matrix @ pts.T).T.astype(np.float32)
    return aligned_img, aligned_pts


def _delaunay_triangles(points: np.ndarray, canvas: int) -> List[Tuple[int, int, int]]:
    """Return triangles as index triplets into ``points`` (via cv2.Subdiv2D).

    Vertices are matched back to indices with integer-quantised keys (x4 sub-pixel).
    Keys are built identically for inserted points and for the coordinates
    returned by ``getTriangleList`` so the round-trip is exact — mixing numpy and
    Python float rounding here silently drops almost every triangle.
    """
    def key(x: float, y: float) -> Tuple[int, int]:
        cx = min(max(float(x), 0.0), canvas - 1.0)
        cy = min(max(float(y), 0.0), canvas - 1.0)
        return (int(round(cx * 4)), int(round(cy * 4)))

    subdiv = cv2.Subdiv2D((0, 0, canvas, canvas))
    lookup = {}
    for i, p in enumerate(points):
        cx = min(max(float(p[0]), 0.0), canvas - 1.0)
        cy = min(max(float(p[1]), 0.0), canvas - 1.0)
        subdiv.insert((cx, cy))
        lookup.setdefault(key(cx, cy), i)

    triangles = []
    for t in subdiv.getTriangleList():
        idx = []
        for (x, y) in [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]:
            k = key(x, y)
            if k not in lookup:
                break
            idx.append(lookup[k])
        if len(idx) == 3:
            triangles.append(tuple(idx))
    return triangles


def _warp_triangle(src: np.ndarray, dst: np.ndarray,
                   t_src: np.ndarray, t_dst: np.ndarray) -> None:
    """Affine-warp one triangle from ``src`` into ``dst`` in place."""
    r1 = cv2.boundingRect(t_src.astype(np.float32))
    r2 = cv2.boundingRect(t_dst.astype(np.float32))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    t1 = t_src - np.array([r1[0], r1[1]], np.float32)
    t2 = t_dst - np.array([r2[0], r2[1]], np.float32)

    src_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_crop.size == 0:
        return
    M = cv2.getAffineTransform(t1.astype(np.float32), t2.astype(np.float32))
    warped = cv2.warpAffine(
        src_crop, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.zeros((r2[3], r2[2], 3), np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1.0, 1.0, 1.0), cv2.LINE_AA, 0)
    roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if roi.shape[:2] != mask.shape[:2]:
        return
    roi[:] = roi * (1 - mask) + warped.astype(np.float32) * mask


def _warp_to_target(src_img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                    triangles: List[Tuple[int, int, int]], canvas: int) -> np.ndarray:
    out = np.zeros((canvas, canvas, 3), np.float32)
    for (a, b, c) in triangles:
        _warp_triangle(src_img.astype(np.float32), out,
                       src_pts[[a, b, c]], dst_pts[[a, b, c]])
    return out


def _face_mask(points: np.ndarray, canvas: int, feather: int = 15) -> np.ndarray:
    """Feathered single-channel mask (0-1) over the convex hull of the mesh."""
    hull = cv2.convexHull(points[:468].astype(np.float32)).astype(np.int32)
    mask = np.zeros((canvas, canvas), np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0, cv2.LINE_AA)
    k = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def match_histogram(src: np.ndarray, ref: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Match ``src`` colour distribution to ``ref`` (per LAB channel CDF).

    This is the PRS "Lighting & Color Harmonization" step: it normalises skin
    tone and exposure so the blended child looks cohesive.
    """
    src_lab = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    if mask is not None:
        sel = mask > 0.15
    else:
        sel = np.ones(src.shape[:2], bool)
    out = src_lab.copy()
    for ch in range(3):
        s_vals = src_lab[:, :, ch][sel]
        r_vals = ref_lab[:, :, ch][sel]
        if s_vals.size == 0 or r_vals.size == 0:
            continue
        s_hist = np.bincount(s_vals, minlength=256).astype(np.float64)
        r_hist = np.bincount(r_vals, minlength=256).astype(np.float64)
        s_cdf = np.cumsum(s_hist) / max(s_hist.sum(), 1)
        r_cdf = np.cumsum(r_hist) / max(r_hist.sum(), 1)
        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        out[:, :, ch] = lut[src_lab[:, :, ch]]
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR).astype(np.float32)


def blend_faces(
    image_a: np.ndarray, face_a: FaceLandmarks,
    image_b: np.ndarray, face_b: FaceLandmarks,
    alpha: float = 0.5, canvas: int = 512, harmonize: bool = True,
) -> OffspringResult:
    """Blend two detected parent faces into a single child portrait."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    template = _canvas_template(canvas)

    img_a, pts_a = _align(image_a, face_a, template, canvas)
    img_b, pts_b = _align(image_b, face_b, template, canvas)

    bounds = _boundary_points(canvas)
    pts_a = np.vstack([pts_a, bounds])
    pts_b = np.vstack([pts_b, bounds])
    target = (1.0 - alpha) * pts_a + alpha * pts_b

    triangles = _delaunay_triangles(target, canvas)
    log.info("Morphing with %d triangles (alpha=%.2f)", len(triangles), alpha)

    warped_a = _warp_to_target(img_a, pts_a, target, triangles, canvas)
    warped_b = _warp_to_target(img_b, pts_b, target, triangles, canvas)

    mask = _face_mask(target, canvas)
    if harmonize:
        warped_b = match_histogram(warped_b, warped_a, mask)

    blended = (1.0 - alpha) * warped_a + alpha * warped_b

    # Feather the blended face over parent A's aligned image for a natural border.
    m3 = cv2.merge([mask, mask, mask])
    composed = blended * m3 + warped_a * (1 - m3)
    result = np.clip(composed, 0, 255).astype(np.uint8)

    return OffspringResult(
        image=result,
        parent_a_aligned=np.clip(img_a, 0, 255).astype(np.uint8),
        parent_b_aligned=np.clip(img_b, 0, 255).astype(np.uint8),
        alpha=alpha,
    )


def generate_offspring(
    image_a: np.ndarray, image_b: Optional[np.ndarray] = None,
    alpha: float = 0.5, canvas: int = 512, harmonize: bool = True,
    mesh: Optional[FaceMesh] = None,
) -> OffspringResult:
    """High-level entry: two portraits, or one image containing two people.

    * Two images -> parent A = first, parent B = second.
    * One image  -> the two largest faces are used, ordered left-to-right
      (per the PRS: "man is left, his wife is right").
    """
    own_mesh = mesh is None
    mesh = mesh or FaceMesh(max_faces=2)
    try:
        if image_b is None:
            faces = mesh.detect(image_a)
            if len(faces) < 2:
                raise ValueError(
                    f"Single-image mode needs two faces; found {len(faces)}."
                )
            face_a, face_b = faces[0], faces[1]      # left, right
            return blend_faces(image_a, face_a, image_a, face_b,
                               alpha, canvas, harmonize)
        face_a = mesh.detect_one(image_a)
        face_b = mesh.detect_one(image_b)
        if face_a is None or face_b is None:
            raise ValueError("Could not detect a face in one of the parent images.")
        return blend_faces(image_a, face_a, image_b, face_b, alpha, canvas, harmonize)
    finally:
        if own_mesh:
            mesh.close()
