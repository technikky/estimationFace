# estimationFace — Offline Face Estimation Suite

![100% Offline](https://img.shields.io/badge/100%25-offline-brightgreen)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A **100% offline**, self-contained Windows desktop app for:

| Feature | PRS ref | What it does |
|---|---|---|
| **Offspring generation** | FR-2 | Blend two parent portraits (or one photo with two people, left + right) into a plausible child using dense-landmark Delaunay morphing + color harmonization. |
| **Age progression** | FR-3 | Age a face 5→80 with the FRAN re-aging network, preserving identity (ArcFace resemblance check). |
| **3D / texture mapping** | FR-4 | Project a user image/mask (homography) or a 3D mesh (OBJ/glTF) onto a detected face, tracking head pose. Live webcam mode included. |

Everything runs locally through **ONNX Runtime** and **MediaPipe** on CPU. The
app never makes a network request — enforced and verified by
`test/offline_check.py` (which hard-blocks sockets while loading every model).

---

## Quick start

Double-click **`Launch estimationFace.bat`**, or:

```powershell
.\run.ps1
```

Both use the bundled portable Python in `runtime\` — nothing to install.

## Webcam & real-time

Every tab accepts a **webcam** input, and each has a **live preview** mode:

| Tab | Live mode | Speed on CPU |
|---|---|---|
| 3D / Texture map | ✅ true real-time | ~15–30 fps |
| Offspring | ✅ live preview | ~1 fps (drops frames, UI stays responsive) |
| Age progression | ✅ live preview (fast mode) | ~1 fps (single-window FRAN) |

- **Offspring “📷 Webcam: two people in one shot”** captures a single frame with
  **both people in view** and splits it automatically — **left face → Parent A,
  right face → Parent B** (no separate panels). The live checkbox does the same
  continuously.
- **Aging live** uses a fast single-512-window path (~1 s/frame) instead of the
  7 s high-quality sliding window, and skips the identity check for speed.

> Note: offspring morphing and FRAN aging are compute-heavy; **true 30 fps for
> those two needs a GPU**. On CPU the live modes run as fast as the machine
> allows (≈1 fps) without freezing the UI. Only one webcam is used at a time —
> switching tabs stops any running camera.

## Directory layout

```
estimationFace\
├── runtime\                     Embedded Python 3.11 (portable, self-contained)
├── offline-sdk\                 Cached wheels + installers for offline rebuilds
├── models\
│   ├── mediapipe\face_landmarker.task    468/478-point mesh (Apache-2.0)
│   ├── aging\face_reaging.onnx           FRAN re-aging U-Net (MIT)
│   └── recognition\w600k_r50.onnx        ArcFace r50 identity embeddings
├── core\
│   ├── landmarks.py    Face detection + dense mesh + ArcFace alignment
│   ├── identity.py     512-d embeddings + resemblance validation
│   ├── blending.py     FR-2 offspring morphing + color harmonization
│   ├── aging.py        FR-3 FRAN inference + identity lock
│   ├── mapping.py      FR-4 homography decal + 3D software rasteriser
│   └── camera.py       Threaded webcam capture
├── gui\                PySide6 UI (3 tabs) + shared model backend
├── app.py              Entry point
├── requirements_offline.txt
├── Launch estimationFace.bat / run.ps1
└── test\               Smoke test, offline check, sample assets
```

## How it works

**Offspring (FR-2).** Both parents are similarity-aligned to a canonical face
template, all 478 landmarks carried through the same transform. The child shape
is `(1-α)·A + α·B`; each parent is piecewise-affine warped into it over a
Delaunay triangulation (`cv2.Subdiv2D`), histograms are matched inside the face
region for lighting/skin cohesion, and the two are cross-dissolved and
feather-composited through the mesh convex hull.

**Aging (FR-3).** The face crop is resized to 1024², a 5-channel tensor
`[R,G,B, sourceAge/100, targetAge/100]` is built, and FRAN runs over 512²
sliding windows (stride 256) with a raised-cosine blend. The network output is a
**residual** added back to the crop, limited to the face by a feathered oval,
then composited into the original. An ArcFace comparison (aged vs. source)
reports a resemblance %; a `strength` control trades ageing intensity against
resemblance.

**Mapping (FR-4).** `cv2.solvePnP` recovers head pose from the mesh. A 2D
texture is projected with a 4-anchor homography; a 3D mesh is rendered with a
built-in painter's-algorithm rasteriser (Lambert shading, back-face culling) —
no OpenGL context required, so it survives inside a frozen build.

## Rebuilding the runtime offline

```powershell
runtime\python.exe -m pip install --no-index `
    --find-links offline-sdk\python-wheels -r requirements_offline.txt
```

## Verifying

```powershell
runtime\python.exe test\offline_check.py   # asserts local models + zero network
runtime\python.exe test\smoke_test.py      # exercises all four features
```

## Model provenance

| Model | Source | License |
|---|---|---|
| `face_reaging.onnx` (FRAN) | Glat0s/face_reaging-onnx (from timroelofs123/face_re-aging) | MIT |
| `w600k_r50.onnx` (ArcFace) | InsightFace `buffalo_l` | non-commercial research; contact InsightFace for commercial use |
| `face_landmarker.task` | Google MediaPipe | Apache-2.0 |

See `THIRD-PARTY-NOTICES.md`.
