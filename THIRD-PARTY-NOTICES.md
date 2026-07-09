# Third-Party Notices

estimationFace bundles the following third-party components. All are used
locally; none require network access at runtime.

## Models

### FRAN — Face Re-Aging Network (`models/aging/face_reaging.onnx`)
- ONNX export: https://github.com/Glat0s/face_reaging-onnx  (MIT License)
- Original weights/research: https://github.com/timroelofs123/face_reaging
  and Roelofs et al., "Face Re-Aging" — https://huggingface.co/timroelofs123/face_re-aging
- License: MIT.

### ArcFace `w600k_r50` (`models/recognition/w600k_r50.onnx`)
- Part of the InsightFace `buffalo_l` model pack — https://github.com/deepinsight/insightface
- The InsightFace pretrained models are released for **non-commercial research
  purposes**. For commercial deployment, obtain the appropriate license from
  InsightFace (recognition-oss-pack@insightface.ai). This model powers the
  optional identity-resemblance check only; it can be removed or swapped without
  affecting FR-2/FR-3/FR-4 generation.

### MediaPipe Face Landmarker (`models/mediapipe/face_landmarker.task`)
- Google MediaPipe — https://developers.google.com/mediapipe
- License: Apache License 2.0.

## Python packages

Bundled in `runtime/` and cached in `offline-sdk/python-wheels/`:
NumPy (BSD), OpenCV (Apache-2.0), MediaPipe (Apache-2.0),
ONNX Runtime (MIT), trimesh (MIT), PySide6 / Qt (LGPLv3),
protobuf (BSD), sympy (BSD), coloredlogs (MIT).

Qt/PySide6 is used under the LGPLv3; the bundled runtime keeps the Qt libraries
as separate, replaceable shared libraries in `runtime/Lib/site-packages/PySide6`.
