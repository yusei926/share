"""Compose the true stereo head-camera view shown in Apple Vision Pro."""

from __future__ import annotations

import numpy as np


HEAD_SHAPE = (480, 640, 3)


def _rgb(value: np.ndarray, shape: tuple[int, int, int], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8 {shape}, got {result.dtype} {result.shape}")
    return result


def compose_head_stereo_view(
    head_left: np.ndarray,
    head_right: np.ndarray,
) -> np.ndarray:
    """Return the unmodified 1280x480 side-by-side stereo head view."""

    left = _rgb(head_left, HEAD_SHAPE, "head_left")
    right = _rgb(head_right, HEAD_SHAPE, "head_right")
    return np.concatenate((left, right), axis=1)
