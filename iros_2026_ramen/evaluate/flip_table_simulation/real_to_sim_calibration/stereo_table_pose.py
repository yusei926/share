"""Metric table-pose geometry for offline flip-table calibration only.

The source dataset does not contain a table pose.  This module therefore
operates on explicit, auditable left/right image correspondences for four known
tabletop corners.  It is deliberately free of segmentation heuristics and has
no policy-facing API.
"""

from __future__ import annotations

import numpy as np


UTTER_TABLETOP_LENGTH_M = 0.580
UTTER_TABLETOP_WIDTH_M = 0.420


def tabletop_corners_object_m() -> np.ndarray:
    """Return the four CAD tabletop corners in a stable counter-clockwise order."""

    length = UTTER_TABLETOP_LENGTH_M / 2.0
    width = UTTER_TABLETOP_WIDTH_M / 2.0
    return np.asarray(
        ((-length, -width, 0.0), (length, -width, 0.0), (length, width, 0.0), (-length, width, 0.0)),
        dtype=np.float64,
    )


def rigid_transform(target_points: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Solve ``target_from_source`` with a proper least-squares rigid transform."""

    target = np.asarray(target_points, dtype=np.float64)
    source = np.asarray(source_points, dtype=np.float64)
    if target.shape != (4, 3) or source.shape != (4, 3):
        raise ValueError("table pose requires four finite 3-D correspondences")
    if not np.isfinite(target).all() or not np.isfinite(source).all():
        raise ValueError("table correspondences must be finite")
    target_center = target.mean(axis=0)
    source_center = source.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def triangulate_rectified_corners(
    left_px: np.ndarray, right_px: np.ndarray, projection_left: np.ndarray, projection_right: np.ndarray
) -> np.ndarray:
    """Triangulate four rectified stereo correspondences into left-camera metres."""

    left = np.asarray(left_px, dtype=np.float64)
    right = np.asarray(right_px, dtype=np.float64)
    p_left = np.asarray(projection_left, dtype=np.float64)
    p_right = np.asarray(projection_right, dtype=np.float64)
    if left.shape != (4, 2) or right.shape != (4, 2):
        raise ValueError("left/right tabletop clicks must each have shape [4,2]")
    if p_left.shape != (3, 4) or p_right.shape != (3, 4):
        raise ValueError("rectified stereo projection matrices must be [3,4]")
    if not all(np.isfinite(value).all() for value in (left, right, p_left, p_right)):
        raise ValueError("stereo inputs must be finite")
    homogeneous = []
    for point_left, point_right in zip(left, right, strict=True):
        design = np.stack(
            (
                point_left[0] * p_left[2] - p_left[0],
                point_left[1] * p_left[2] - p_left[1],
                point_right[0] * p_right[2] - p_right[0],
                point_right[1] * p_right[2] - p_right[1],
            )
        )
        _, _, right_t = np.linalg.svd(design)
        point = right_t[-1]
        if abs(point[3]) < 1.0e-12:
            raise ValueError("stereo triangulation produced a point at infinity")
        homogeneous.append(point[:3] / point[3])
    result = np.asarray(homogeneous, dtype=np.float64)
    if np.any(result[:, 2] <= 0.0):
        raise ValueError("stereo triangulation produced non-positive depth")
    return result
