"""Small, dependency-free SE(3) helpers for offline source calibration."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


EEF_POSE_DIM = 6


def _pose(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (EEF_POSE_DIM,) or not np.isfinite(result).all():
        raise ValueError("pose must be a finite [x, y, z, roll, pitch, yaw] vector")
    return result


def _transform(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all() or not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("transform must be a finite 4x4 homogeneous matrix")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError("transform rotation must be a proper rotation matrix")
    return result


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
         (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
         (-sp, cp * sr, cp * cr)),
        dtype=np.float64,
    )


def matrix_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to the source dataset's XYZ Euler form."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array((roll, pitch, yaw), dtype=np.float64)


def pose_to_transform(pose: Sequence[float]) -> np.ndarray:
    """Convert ``[x, y, z, roll, pitch, yaw]`` to an SE(3) matrix."""

    values = _pose(pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = euler_xyz_to_matrix(*values[3:])
    result[:3, 3] = values[:3]
    return result


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    """Convert an SE(3) matrix to ``[x, y, z, roll, pitch, yaw]``."""

    value = _transform(transform)
    return np.concatenate((value[:3, 3], matrix_to_euler_xyz(value[:3, :3])))


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a validated SE(3) matrix."""

    value = _transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm as a rotation vector."""

    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    vector = np.array((value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]), dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) > 1.0e-6:
        return vector * (0.5 * angle / sine)
    eigenvalues, eigenvectors = np.linalg.eigh((value + np.eye(3)) * 0.5)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return angle * axis / np.linalg.norm(axis)
