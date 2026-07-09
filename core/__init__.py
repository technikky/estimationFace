"""estimationFace core package.

100% offline face estimation toolkit:

* FR-2  Offspring generation  -> :mod:`core.blending`
* FR-3  Age progression       -> :mod:`core.aging`
* FR-4  3D / texture mapping  -> :mod:`core.mapping`

All heavy inference runs locally through ONNX Runtime / MediaPipe; nothing in
this package ever performs a network request.
"""
from __future__ import annotations

__all__ = [
    "paths",
    "logging_setup",
    "landmarks",
    "identity",
    "blending",
    "aging",
    "mapping",
]
