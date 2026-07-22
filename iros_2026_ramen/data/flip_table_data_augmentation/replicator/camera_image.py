"""Map ideal Omniverse pinhole renders into the recorded camera geometry."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..config import CameraConfig


def ideal_render_intrinsic(camera: CameraConfig) -> np.ndarray:
    """Return the centered square-pixel intrinsic represented by the USD camera."""

    focal_x = camera.focal_length_mm * camera.width / camera.horizontal_aperture_mm
    focal_y = camera.focal_length_mm * camera.height / camera.vertical_aperture_mm
    if not np.isclose(focal_x, focal_y, rtol=1.0e-9):
        raise ValueError("USD render camera must have square pixels")
    return np.asarray(
        (
            (focal_x, 0.0, 0.5 * camera.width),
            (0.0, focal_y, 0.5 * camera.height),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _inverse_brown_to_ideal(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply librealsense's closed-form distorted-to-undistorted mapping."""

    x = (pixels[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixels[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (
        coefficients[0] + r2 * (coefficients[1] + r2 * coefficients[4])
    )
    x_ideal = x * radial + 2.0 * coefficients[2] * x * y + coefficients[3] * (
        r2 + 2.0 * x * x
    )
    y_ideal = y * radial + 2.0 * coefficients[3] * x * y + coefficients[2] * (
        r2 + 2.0 * y * y
    )
    return np.stack((x_ideal, y_ideal), axis=1)


@lru_cache(maxsize=16)
def _remap(
    width: int,
    height: int,
    intrinsic_values: tuple[float, ...],
    distortion_model: str,
    distortion_coefficients: tuple[float, ...],
    ideal_intrinsic_values: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3)
    ideal = np.asarray(ideal_intrinsic_values, dtype=np.float64).reshape(3, 3)
    coefficients = np.asarray(distortion_coefficients, dtype=np.float64)
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    output_pixels = np.stack((u.reshape(-1), v.reshape(-1)), axis=1)
    if distortion_model == "opencv_brown_conrady":
        ideal_pixels = cv2.undistortPoints(
            output_pixels.reshape(-1, 1, 2),
            intrinsic,
            coefficients,
            P=ideal,
        ).reshape(-1, 2)
    elif distortion_model == "realsense_inverse_brown_conrady":
        normalized = _inverse_brown_to_ideal(output_pixels, intrinsic, coefficients)
        ideal_pixels = np.empty_like(normalized)
        ideal_pixels[:, 0] = normalized[:, 0] * ideal[0, 0] + ideal[0, 2]
        ideal_pixels[:, 1] = normalized[:, 1] * ideal[1, 1] + ideal[1, 2]
    else:
        raise ValueError(f"unsupported distortion model: {distortion_model}")
    if not np.isfinite(ideal_pixels).all():
        raise ValueError("camera calibration produced a non-finite image remap")
    return (
        ideal_pixels[:, 0].reshape(height, width).astype(np.float32),
        ideal_pixels[:, 1].reshape(height, width).astype(np.float32),
    )


def apply_recorded_camera_geometry(image: np.ndarray, camera: CameraConfig) -> np.ndarray:
    """Produce raw-size RGB with the pinned real principal point and distortion."""

    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise ValueError(
            f"ideal camera image must be uint8 {(camera.height, camera.width, 3)}, "
            f"got {image.shape}/{image.dtype}"
        )
    ideal = ideal_render_intrinsic(camera)
    map_x, map_y = _remap(
        camera.width,
        camera.height,
        camera.intrinsic_matrix_px,
        camera.distortion_model,
        camera.distortion_coefficients,
        tuple(float(value) for value in ideal.reshape(-1)),
    )
    calibrated = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if calibrated.shape != image.shape or calibrated.dtype != np.uint8:
        raise RuntimeError("camera calibration changed image shape or dtype")
    return calibrated
