"""Recover a source table frame from calibrated stereo correspondences.

The source MCAP records calibrated head-stereo images and root-frame EEF
states, but not a table pose.  This module estimates the missing offline
calibration from explicitly recorded image correspondences.  It is deliberately
separate from deployment code: the resulting table pose can seed an EEF teacher,
never a policy input or runtime planner input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

try:
    from .transforms import transform_to_pose
    from .v1_table_geometry import V1_TABLE001_BODY_FRAME
except ImportError:  # Direct execution from the teacher directory.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transforms import transform_to_pose  # type: ignore[no-redef]
    from v1_table_geometry import V1_TABLE001_BODY_FRAME  # type: ignore[no-redef]


CALIBRATION_SCHEMA_VERSION = "flip_table_source_task_frame_calibration/v1"
_EPSILON = 1.0e-10


@dataclass(frozen=True)
class StereoCalibration:
    """Intrinsics and metre-normalized rectified matrices for one stereo rig."""

    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    rectification_left: np.ndarray
    rectification_right: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray
    linear_unit_to_m: float = 1.0


def _matrix(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {result.shape}")
    return result


def _points(value: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 3:
        raise ValueError(f"{name} must have shape [N>=3,3], got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    centered = result - result.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=1.0e-8) < 2:
        raise ValueError(f"{name} must contain at least three non-collinear points")
    return result


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Load an MCAP head-stereo calibration and normalize its baseline to metres.

    The organizer's OpenCV YAML serializes ``T``, ``P2[0, 3]`` and ``baseline``
    in millimetres.  The recorded EEF labels are in metres.  OpenCV permits
    either unit in a projection matrix, but mixing them silently makes the
    triangulated camera points 1,000 times too large.  Keep focal lengths and
    principal points in pixels, and scale only the translation column.
    """

    calibration_path = Path(path)
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stereo calibration YAML must contain a mapping")
    baseline = float(payload.get("baseline", 0.0))
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("stereo calibration needs a positive finite baseline")
    # The source head-camera calibration has a 60 mm baseline.  Small values
    # are already metres, which keeps this loader usable for metre-native
    # OpenCV exports without guessing a second conversion.
    linear_unit_to_m = 0.001 if baseline > 1.0 else 1.0
    projection_left = _matrix(payload.get("P1"), (3, 4), name="P1").copy()
    projection_right = _matrix(payload.get("P2"), (3, 4), name="P2").copy()
    projection_left[:, 3] *= linear_unit_to_m
    projection_right[:, 3] *= linear_unit_to_m
    focal = float(projection_left[0, 0])
    if focal <= 0.0:
        raise ValueError("P1 focal length must be positive")
    matrix_baseline_m = abs(float(projection_right[0, 3] - projection_left[0, 3])) / focal
    expected_baseline_m = baseline * linear_unit_to_m
    if not math.isclose(matrix_baseline_m, expected_baseline_m, rel_tol=0.05, abs_tol=0.002):
        raise ValueError(
            "stereo baseline is inconsistent between baseline and P1/P2: "
            f"{expected_baseline_m:.6f} m versus {matrix_baseline_m:.6f} m"
        )
    return StereoCalibration(
        camera_matrix_left=_matrix(payload.get("camera_matrix_left"), (3, 3), name="camera_matrix_left"),
        camera_matrix_right=_matrix(payload.get("camera_matrix_right"), (3, 3), name="camera_matrix_right"),
        dist_coeffs_left=_matrix(payload.get("dist_coeffs_left"), (5,), name="dist_coeffs_left"),
        dist_coeffs_right=_matrix(payload.get("dist_coeffs_right"), (5,), name="dist_coeffs_right"),
        rectification_left=_matrix(payload.get("R1"), (3, 3), name="R1"),
        rectification_right=_matrix(payload.get("R2"), (3, 3), name="R2"),
        projection_left=projection_left,
        projection_right=projection_right,
        linear_unit_to_m=linear_unit_to_m,
    )


def rectify_pixels(
    pixels: Sequence[Sequence[float]],
    *,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Undistort raw image pixels into the calibration's rectified image plane."""

    raw = np.asarray(pixels, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0 or not np.isfinite(raw).all():
        raise ValueError("pixels must be a non-empty finite [N,2] array")
    rectified = cv2.undistortPoints(
        raw.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        R=rectification,
        P=projection,
    )
    return np.asarray(rectified, dtype=np.float64).reshape(-1, 2)


def triangulate_rectified_pixels(
    left_pixels: Sequence[Sequence[float]],
    right_pixels: Sequence[Sequence[float]],
    *,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
) -> np.ndarray:
    """Triangulate pixels in the metre units of the supplied projection matrices."""

    left = np.asarray(left_pixels, dtype=np.float64)
    right = np.asarray(right_pixels, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape or left.shape[0] == 0:
        raise ValueError("left/right pixels must be equally shaped non-empty [N,2] arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("left/right pixels must be finite")
    p_left = _matrix(projection_left, (3, 4), name="projection_left")
    p_right = _matrix(projection_right, (3, 4), name="projection_right")
    points: list[np.ndarray] = []
    for (u_left, v_left), (u_right, v_right) in zip(left, right, strict=True):
        system = np.stack(
            (
                u_left * p_left[2] - p_left[0],
                v_left * p_left[2] - p_left[1],
                u_right * p_right[2] - p_right[0],
                v_right * p_right[2] - p_right[1],
            )
        )
        _u, _singular, right_singular_t = np.linalg.svd(system)
        homogeneous = right_singular_t[-1]
        if abs(float(homogeneous[3])) <= _EPSILON:
            raise ValueError("stereo correspondence triangulates at infinity")
        point = homogeneous[:3] / homogeneous[3]
        if not np.isfinite(point).all() or point[2] <= 0.0:
            raise ValueError("stereo correspondence must be in front of the left camera")
        points.append(point)
    return np.asarray(points, dtype=np.float64)


def fit_rigid_transform(from_points: Sequence[Sequence[float]], to_points: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``T_to_from`` with Kabsch and return per-point Euclidean residuals."""

    source = _points(from_points, name="from_points")
    target = _points(to_points, name="to_points")
    if source.shape != target.shape:
        raise ValueError("from_points and to_points must have the same shape")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (target - target_center).T @ (source - source_center)
    left_singular, _singular_values, right_singular_t = np.linalg.svd(covariance)
    rotation = left_singular @ right_singular_t
    if np.linalg.det(rotation) < 0.0:
        left_singular[:, -1] *= -1.0
        rotation = left_singular @ right_singular_t
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    residuals = np.linalg.norm((rotation @ source.T).T + transform[:3, 3] - target, axis=1)
    return transform, residuals


def _raw_pixels(correspondences: Sequence[dict[str, Any]], side: str) -> np.ndarray:
    key = f"{side}_pixel_raw"
    try:
        pixels = [entry[key] for entry in correspondences]
    except KeyError as error:
        raise ValueError(f"every correspondence needs {key}") from error
    result = np.asarray(pixels, dtype=np.float64)
    if result.shape != (len(correspondences), 2) or not np.isfinite(result).all():
        raise ValueError(f"{key} values must be finite [u,v] pairs")
    return result


def _source_points(correspondences: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    try:
        values = [entry[key] for entry in correspondences]
    except KeyError as error:
        raise ValueError(f"every correspondence needs {key}") from error
    return _points(values, name=key)


def _triangulate_raw_correspondences(
    correspondences: Sequence[dict[str, Any]], calibration: StereoCalibration
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(correspondences) < 3:
        raise ValueError("at least three stereo correspondences are required")
    left_raw = _raw_pixels(correspondences, "left")
    right_raw = _raw_pixels(correspondences, "right")
    left_rectified = rectify_pixels(
        left_raw,
        camera_matrix=calibration.camera_matrix_left,
        dist_coeffs=calibration.dist_coeffs_left,
        rectification=calibration.rectification_left,
        projection=calibration.projection_left,
    )
    right_rectified = rectify_pixels(
        right_raw,
        camera_matrix=calibration.camera_matrix_right,
        dist_coeffs=calibration.dist_coeffs_right,
        rectification=calibration.rectification_right,
        projection=calibration.projection_right,
    )
    points = triangulate_rectified_pixels(
        left_rectified,
        right_rectified,
        projection_left=calibration.projection_left,
        projection_right=calibration.projection_right,
    )
    return points, left_rectified, right_rectified


def _metric_summary(residuals: np.ndarray) -> dict[str, float]:
    return {
        "rms_m": float(math.sqrt(float(np.mean(np.square(residuals))))),
        "max_m": float(np.max(residuals)),
        "mean_m": float(np.mean(residuals)),
    }


def calibrate_source_task_frame(
    correspondences_payload: dict[str, Any], calibration: StereoCalibration, *, max_root_rms_m: float, max_table_rms_m: float
) -> dict[str, Any]:
    """Estimate source root-to-camera and source root-to-table transforms.

    ``root_camera_correspondences`` pair stereo-visible, known EEF reference
    points with their measured root-frame positions.  ``table_correspondences``
    pair known tabletop-frame fiducials with the same stereo frame.  The table
    must be static while those table pixels are selected.
    """

    if max_root_rms_m <= 0.0 or max_table_rms_m <= 0.0:
        raise ValueError("RMS acceptance thresholds must be positive")
    if correspondences_payload.get("source_table_body_frame") != V1_TABLE001_BODY_FRAME:
        raise ValueError(f"source_table_body_frame must be {V1_TABLE001_BODY_FRAME}")
    root_entries = correspondences_payload.get("root_camera_correspondences")
    table_entries = correspondences_payload.get("table_correspondences")
    if not isinstance(root_entries, list) or not isinstance(table_entries, list):
        raise ValueError("correspondence payload needs root_camera_correspondences and table_correspondences lists")
    root_camera_points, root_left_rectified, root_right_rectified = _triangulate_raw_correspondences(root_entries, calibration)
    root_points = _source_points(root_entries, "source_root_point_m")
    root_from_camera, root_residuals = fit_rigid_transform(root_camera_points, root_points)

    table_camera_points, table_left_rectified, table_right_rectified = _triangulate_raw_correspondences(table_entries, calibration)
    table_points = _source_points(table_entries, "table_point_m")
    table_root_points = (root_from_camera[:3, :3] @ table_camera_points.T).T + root_from_camera[:3, 3]
    root_from_table, table_residuals = fit_rigid_transform(table_points, table_root_points)

    root_metrics = _metric_summary(root_residuals)
    table_metrics = _metric_summary(table_residuals)
    static_confirmed = correspondences_payload.get("table_is_static_confirmation") is True
    acceptance_eligible = (
        static_confirmed
        and root_metrics["rms_m"] <= max_root_rms_m
        and table_metrics["rms_m"] <= max_table_rms_m
    )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "offline_teacher_only": True,
        "source_episode_index": correspondences_payload.get("source_episode_index"),
        "calibration_endpoint": correspondences_payload.get("calibration_endpoint"),
        "workspace_manifest_sha256": correspondences_payload.get("workspace_manifest_sha256"),
        "raw_source_repo_id": correspondences_payload.get("raw_source_repo_id"),
        "raw_source_revision": correspondences_payload.get("raw_source_revision"),
        "head_stereo_calibration": correspondences_payload.get("head_stereo_calibration"),
        "source_task_pose_root": transform_to_pose(root_from_table).tolist(),
        "source_root_from_rectified_left_camera": root_from_camera.tolist(),
        "source_root_from_table": root_from_table.tolist(),
        "stereo_linear_unit_to_m": float(calibration.linear_unit_to_m),
        "acceptance_eligible": acceptance_eligible,
        "acceptance_requirements": {
            "table_is_static_confirmation": True,
            "max_root_rms_m": float(max_root_rms_m),
            "max_table_rms_m": float(max_table_rms_m),
        },
        "measurements": {
            "root_camera": {
                "count": len(root_entries),
                "metrics": root_metrics,
                "residuals_m": root_residuals.tolist(),
                "rectified_left_pixels": root_left_rectified.tolist(),
                "rectified_right_pixels": root_right_rectified.tolist(),
                "source_root_points_m": root_points.tolist(),
                "triangulated_left_camera_points_m": root_camera_points.tolist(),
            },
            "table": {
                "count": len(table_entries),
                "metrics": table_metrics,
                "residuals_m": table_residuals.tolist(),
                "rectified_left_pixels": table_left_rectified.tolist(),
                "rectified_right_pixels": table_right_rectified.tolist(),
                "table_points_m": table_points.tolist(),
                "triangulated_left_camera_points_m": table_camera_points.tolist(),
            },
        },
        "source_task_frame_provenance": correspondences_payload.get(
            "source_task_frame_provenance",
            "calibrated from recorded head stereo and root-frame EEF/table fiducials",
        ),
        "source_table_body_frame": V1_TABLE001_BODY_FRAME,
        "limitations": (
            "offline calibration only; acceptance requires static-table confirmation and both RMS gates. "
            "The calibrated task frame is forbidden from final policy inputs and runtime planning."
        ),
    }
