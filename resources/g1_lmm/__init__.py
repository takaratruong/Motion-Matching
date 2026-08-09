"""Deterministic trainers for the 60 Hz G1 learned-motion-matching bundle."""

import os

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _configured_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
elif _configured_workspace != CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError(
        "G1 LMM requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch"
    )

from .dataset import G1LmmDimensions
from .models import Compressor, Decompressor, Projector, Stepper

__all__ = [
    "Compressor",
    "CUBLAS_WORKSPACE_CONFIG",
    "Decompressor",
    "G1LmmDimensions",
    "Projector",
    "Stepper",
]
