"""FR-4  Real-time 3D model & user-image mapping (offline).

Two capabilities, both driven by the dense mesh from :mod:`core.landmarks`:

* :class:`TextureMapper` — projects a user image / mask (optionally RGBA) onto
  the localised face region via a **homography** derived from four facial
  anchors, so the decal follows in-plane rotation and perspective.
* :class:`MeshOverlay` — estimates 3D head pose with ``cv2.solvePnP`` and
  renders a user OBJ / glTF (loaded by ``trimesh``) over the face using a
  self-contained software rasteriser (painter's algorithm + Lambert shading).
  A software renderer avoids any OpenGL-context requirement, so it works inside
  a frozen offline executable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .landmarks import FaceLandmarks
from .logging_setup import get_logger

log = get_logger("mapping")

# Generic 3D head model points (mm) with their MediaPipe landmark indices,
# used for solvePnP pose estimation.
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye (image-left) outer corner
        [43.3, 32.7, -26.0],    # right eye outer corner
        [-28.9, -28.9, -24.1],  # left mouth corner
        [28.9, -28.9, -24.1],   # right mouth corner
    ],
    dtype=np.float32,
)
_MODEL_INDICES = [1, 152, 33, 263, 61, 291]

# Face anchors (mesh indices) for the homography decal: forehead L/R, chin L/R.
_ANCHOR_INDICES = [67, 297, 172, 397]


@dataclass
class HeadPose:
    rvec: np.ndarray
    tvec: np.ndarray
    camera_matrix: np.ndarray
    dist: np.ndarray

    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        """Return (pitch, yaw, roll) in degrees."""
        R, _ = cv2.Rodrigues(self.rvec)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        return pitch, yaw, roll


def estimate_pose(face: FaceLandmarks) -> HeadPose:
    """Estimate head pose from the dense mesh via solvePnP."""
    w, h = face.image_size
    image_points = face.points[_MODEL_INDICES].astype(np.float32)
    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float32
    )
    dist = np.zeros((4, 1), np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed to estimate head pose.")
    return HeadPose(rvec=rvec, tvec=tvec, camera_matrix=camera_matrix, dist=dist)


class TextureMapper:
    """Homography-projects a 2D image/mask onto the face region."""

    def map_image(
        self, frame_bgr: np.ndarray, face: FaceLandmarks, texture: np.ndarray,
        opacity: float = 1.0,
    ) -> np.ndarray:
        """Overlay ``texture`` (BGR or BGRA) onto the face using 4 anchors.

        The texture's four corners are mapped to forehead-L/R and chin-L/R, so
        it wraps the face plane and tracks head roll / perspective.
        """
        h_t, w_t = texture.shape[:2]
        src = np.array([[0, 0], [w_t - 1, 0], [w_t - 1, h_t - 1], [0, h_t - 1]],
                       dtype=np.float32)
        # dst order must match src corners: TL, TR, BR, BL.
        fL, fR = face.points[_ANCHOR_INDICES[0]], face.points[_ANCHOR_INDICES[1]]
        cL, cR = face.points[_ANCHOR_INDICES[2]], face.points[_ANCHOR_INDICES[3]]
        dst = np.array([fL, fR, cR, cL], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        fh, fw = frame_bgr.shape[:2]

        if texture.shape[2] == 4:
            bgr = texture[:, :, :3]
            alpha = texture[:, :, 3].astype(np.float32) / 255.0
        else:
            bgr = texture
            alpha = np.ones((h_t, w_t), np.float32)

        warped = cv2.warpPerspective(bgr, H, (fw, fh))
        warped_a = cv2.warpPerspective(alpha, H, (fw, fh)) * float(opacity)
        warped_a = warped_a[..., None]

        out = frame_bgr.astype(np.float32) * (1 - warped_a) + \
            warped.astype(np.float32) * warped_a
        return np.clip(out, 0, 255).astype(np.uint8)


class MeshOverlay:
    """Renders a 3D mesh over the face with a software rasteriser."""

    def __init__(self, mesh_path: str, color: Tuple[int, int, int] = (200, 200, 210)):
        import trimesh

        scene = trimesh.load(mesh_path, force="mesh")
        if isinstance(scene, trimesh.Scene):
            scene = scene.dump(concatenate=True)
        self.vertices = np.asarray(scene.vertices, np.float32)
        self.faces = np.asarray(scene.faces, np.int32)

        # Vertex colours if present, else a flat base colour.
        self.vertex_colors = None
        vc = getattr(scene.visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(self.vertices):
            self.vertex_colors = np.asarray(vc, np.float32)[:, :3][:, ::-1]  # RGB->BGR

        # Normalise: centre at origin, scale to unit width in model space.
        self.vertices -= self.vertices.mean(axis=0)
        extent = np.ptp(self.vertices, axis=0).max()
        if extent > 0:
            self.vertices /= extent
        self.base_color = np.array(color, np.float32)
        log.info("Loaded mesh %s (%d verts, %d faces)",
                 mesh_path, len(self.vertices), len(self.faces))

    def render(
        self, frame_bgr: np.ndarray, face: FaceLandmarks,
        scale: float = 110.0, opacity: float = 1.0,
        light_dir: Tuple[float, float, float] = (0.3, 0.4, 1.0),
    ) -> np.ndarray:
        """Composite the mesh over the face using the estimated head pose."""
        pose = estimate_pose(face)
        R, _ = cv2.Rodrigues(pose.rvec)

        # Model -> camera space (scale mesh to ~face size, apply pose).
        verts_cam = (R @ (self.vertices * scale).T).T + pose.tvec.reshape(1, 3)
        proj, _ = cv2.projectPoints(
            self.vertices * scale, pose.rvec, pose.tvec,
            pose.camera_matrix, pose.dist,
        )
        proj = proj.reshape(-1, 2)

        light = np.array(light_dir, np.float32)
        light /= np.linalg.norm(light)

        overlay = frame_bgr.copy()
        drawn = np.zeros(frame_bgr.shape[:2], np.uint8)

        # Painter's algorithm: far triangles first.
        tri_depth = verts_cam[self.faces][:, :, 2].mean(axis=1)
        order = np.argsort(tri_depth)[::-1]

        for fi in order:
            tri = self.faces[fi]
            p = proj[tri]
            v = verts_cam[tri]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            n_norm = np.linalg.norm(normal)
            if n_norm == 0:
                continue
            normal /= n_norm
            if normal[2] > 0:            # back-face cull (camera looks down +z)
                continue
            shade = float(np.clip(abs(np.dot(normal, light)), 0.25, 1.0))
            if self.vertex_colors is not None:
                col = self.vertex_colors[tri].mean(axis=0)
            else:
                col = self.base_color
            col = np.clip(col * shade, 0, 255)
            cv2.fillConvexPoly(overlay, p.astype(np.int32), col.tolist(), cv2.LINE_AA)
            cv2.fillConvexPoly(drawn, p.astype(np.int32), 255)

        m = (cv2.GaussianBlur(drawn, (5, 5), 0).astype(np.float32) / 255.0
             * float(opacity))[..., None]
        out = frame_bgr.astype(np.float32) * (1 - m) + overlay.astype(np.float32) * m
        return np.clip(out, 0, 255).astype(np.uint8)
