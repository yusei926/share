"""Project source G1 forward kinematics into the calibrated head-left image."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import CameraConfig


PROJECTION_AUDIT_SCHEMA_VERSION = "team_ramen_source_camera_projection_audit/v1"
ARM_FRAME_NAMES = {
    side: tuple(
        f"{side}_{joint}_link"
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
    for side in ("left", "right")
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def root_from_camera(
    root_from_parent: np.ndarray,
    camera: CameraConfig,
) -> np.ndarray:
    """Compose the Isaac Lab XYZW camera offset with its FK parent pose."""

    parent = np.asarray(root_from_parent, dtype=np.float64)
    if parent.shape != (4, 4) or not np.isfinite(parent).all():
        raise ValueError("root_from_parent must be a finite 4x4 transform")
    parent_from_camera = np.eye(4, dtype=np.float64)
    parent_from_camera[:3, :3] = Rotation.from_quat(
        camera.offset_quaternion_xyzw
    ).as_matrix()
    parent_from_camera[:3, 3] = camera.offset_position_m
    return parent @ parent_from_camera


def project_root_points(
    points_root: np.ndarray,
    root_from_optical_opengl: np.ndarray,
    camera: CameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project root-frame points through OpenGL optics into raw source pixels."""

    points = np.asarray(points_root, dtype=np.float64)
    transform = np.asarray(root_from_optical_opengl, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_root must be a finite Nx3 array")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("root_from_optical_opengl must be a finite 4x4 transform")
    optical_from_root = np.linalg.inv(transform)
    points_gl = (
        optical_from_root[:3, :3] @ points.T
    ).T + optical_from_root[:3, 3]
    points_cv = np.column_stack((points_gl[:, 0], -points_gl[:, 1], -points_gl[:, 2]))
    pixels = cv2.projectPoints(
        points_cv,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(camera.intrinsic_matrix_px, dtype=np.float64).reshape(3, 3),
        np.asarray(camera.distortion_coefficients, dtype=np.float64),
    )[0].reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("camera projection produced NaN or Inf")
    return pixels, points_cv[:, 2]


def visible_mask(pixels: np.ndarray, depth_m: np.ndarray, camera: CameraConfig) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or depth.shape != (values.shape[0],):
        raise ValueError("pixels and depth have incompatible shapes")
    return (
        (depth > camera.clipping_range_m[0])
        & (depth < camera.clipping_range_m[1])
        & (values[:, 0] >= 0.0)
        & (values[:, 0] < camera.width)
        & (values[:, 1] >= 0.0)
        & (values[:, 1] < camera.height)
    )


def draw_arm_projection(
    image_bgr: np.ndarray,
    projections: dict[str, tuple[np.ndarray, np.ndarray]],
    camera: CameraConfig,
) -> np.ndarray:
    """Draw depth-valid FK chains without changing the evidence image."""

    image = np.asarray(image_bgr)
    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise ValueError("projection evidence must be a raw uint8 BGR camera frame")
    output = image.copy()
    colors = {"left": (0, 255, 0), "right": (0, 0, 255)}
    for side in ("left", "right"):
        pixels, depth = projections[side]
        valid_depth = np.asarray(depth) > camera.clipping_range_m[0]
        for index in range(len(pixels) - 1):
            if valid_depth[index] and valid_depth[index + 1]:
                cv2.line(
                    output,
                    tuple(np.rint(pixels[index]).astype(int)),
                    tuple(np.rint(pixels[index + 1]).astype(int)),
                    colors[side],
                    3,
                    cv2.LINE_AA,
                )
        for pixel, is_visible in zip(pixels, visible_mask(pixels, depth, camera), strict=True):
            if is_visible:
                cv2.circle(
                    output,
                    tuple(np.rint(pixel).astype(int)),
                    6,
                    colors[side],
                    -1,
                    cv2.LINE_AA,
                )
    return output


def projection_json(
    frame_names: tuple[str, ...],
    pixels: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraConfig,
) -> list[dict[str, Any]]:
    visible = visible_mask(pixels, depth_m, camera)
    return [
        {
            "frame": name,
            "pixel_xy": [float(pixel[0]), float(pixel[1])],
            "depth_m": float(depth),
            "in_frame": bool(in_frame),
        }
        for name, pixel, depth, in_frame in zip(
            frame_names, pixels, depth_m, visible, strict=True
        )
    ]
