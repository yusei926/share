"""Calibrated source-camera views used only for offline object annotation."""

from __future__ import annotations

import numpy as np


POSE_VIEW_NAMES = ("head_left", "left_wrist", "right_wrist")
PRIMARY_POSE_VIEW = POSE_VIEW_NAMES[0]


def inverse_brown_conrady_rectification_maps(
    intrinsic_matrix_px: np.ndarray,
    distortion_coefficients: np.ndarray,
    *,
    width: int,
    height: int,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build raw-pixel remaps for a RealSense inverse Brown-Conrady model.

    The calibrated coefficients map distorted normalized coordinates to
    undistorted coordinates.  OpenCV's standard undistortion API expects the
    opposite model, so the raw source coordinate is solved explicitly for each
    output pinhole pixel.
    """

    intrinsic = np.asarray(intrinsic_matrix_px, dtype=np.float64)
    coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or coefficients.shape != (5,)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(coefficients).all()
        or width <= 0
        or height <= 0
        or iterations <= 0
    ):
        raise ValueError("inverse Brown-Conrady calibration is invalid")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0 or not np.allclose(
        intrinsic[2], (0.0, 0.0, 1.0), atol=1.0e-12
    ):
        raise ValueError("camera intrinsic matrix is invalid")

    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    target_x = (pixel_x - cx) / fx
    target_y = (pixel_y - cy) / fy
    distorted_x = target_x.copy()
    distorted_y = target_y.copy()
    k1, k2, p1, p2, k3 = coefficients
    for _ in range(iterations):
        radius2 = distorted_x * distorted_x + distorted_y * distorted_y
        radial = 1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
        estimated_x = (
            distorted_x * radial
            + 2.0 * p1 * distorted_x * distorted_y
            + p2 * (radius2 + 2.0 * distorted_x * distorted_x)
        )
        estimated_y = (
            distorted_y * radial
            + p1 * (radius2 + 2.0 * distorted_y * distorted_y)
            + 2.0 * p2 * distorted_x * distorted_y
        )
        distorted_x += target_x - estimated_x
        distorted_y += target_y - estimated_y
    return (
        (distorted_x * fx + cx).astype(np.float32),
        (distorted_y * fy + cy).astype(np.float32),
    )
