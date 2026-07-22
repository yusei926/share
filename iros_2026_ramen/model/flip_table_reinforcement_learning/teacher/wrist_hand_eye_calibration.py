"""Calibrate source table pose from recorded D405 IR stereo and EEF motion.

The source recording has no table TF and the head camera has no recorded
root-frame extrinsic.  Each Dex1 D405 is, however, rigidly mounted to the
recorded EEF frame and provides calibrated IR stereo.  For every static-table
observation we recover ``T_ir1_table`` from named table fiducials.  Across
multiple EEF poses we solve the hand-eye equation:

``T_root_table = T_root_eef[i] @ T_eef_ir1 @ T_ir1_table[i]``.

This is an offline teacher-calibration utility only.  Neither table poses,
camera extrinsics, nor its residuals are permitted policy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from .transforms import (
        EEF_POSE_DIM,
        inverse_transform,
        pose_to_transform,
        rotation_log,
        transform_to_pose,
    )
    from .source_stereo_calibration import fit_rigid_transform
    from .v1_table_geometry import V1_TABLE001_BODY_FRAME
except ImportError:  # Direct execution from the teacher directory.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transforms import (  # type: ignore[no-redef]
        EEF_POSE_DIM,
        inverse_transform,
        pose_to_transform,
        rotation_log,
        transform_to_pose,
    )
    from source_stereo_calibration import fit_rigid_transform  # type: ignore[no-redef]
    from v1_table_geometry import V1_TABLE001_BODY_FRAME  # type: ignore[no-redef]


CALIBRATION_SCHEMA_VERSION = "flip_table_source_wrist_hand_eye_calibration/v1"
REVIEW_SCHEMA_VERSION = "flip_table_source_wrist_hand_eye_review/v1"
_EPSILON = 1.0e-10


@dataclass(frozen=True)
class D405IrStereoCalibration:
    """D405 IR intrinsics and IR2-to-IR1 extrinsic, all in metres."""

    camera_matrix_ir1: np.ndarray
    camera_matrix_ir2: np.ndarray
    rotation_ir2_to_ir1: np.ndarray
    translation_ir2_to_ir1_m: np.ndarray
    serial_number: str


@dataclass(frozen=True)
class WristTableObservation:
    """One IR-stereo table-pose observation at a measured source EEF pose."""

    observation_id: str
    source_eef_state_root: np.ndarray
    camera_from_table: np.ndarray
    table_fit_residuals_m: np.ndarray
    epipolar_errors_px: np.ndarray


@dataclass(frozen=True)
class WristTableFiducialObservation:
    """One measured table point in IR1 at a recorded root-frame EEF pose."""

    observation_id: str
    source_eef_state_root: np.ndarray
    table_point_m: np.ndarray
    camera_point_m: np.ndarray
    epipolar_error_px: float


def _finite_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


def _intrinsic_matrix(payload: dict[str, Any], *, name: str) -> np.ndarray:
    intrinsics = payload.get(name, {}).get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError(f"D405 calibration is missing {name}.intrinsics")
    values = [intrinsics.get(key) for key in ("fx", "fy", "ppx", "ppy")]
    fx, fy, ppx, ppy = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (fx, fy, ppx, ppy)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"D405 {name} intrinsics must be finite with positive focal lengths")
    return np.array(((fx, 0.0, ppx), (0.0, fy, ppy), (0.0, 0.0, 1.0)), dtype=np.float64)


def load_d405_ir_stereo_calibration(path: str | Path) -> D405IrStereoCalibration:
    """Load the recorded RealSense JSON and validate the 18-mm IR baseline."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("D405 calibration JSON must contain an object")
    extrinsics = payload.get("ir2", {}).get("extrinsics_to_ir1")
    if not isinstance(extrinsics, dict):
        raise ValueError("D405 calibration is missing ir2.extrinsics_to_ir1")
    rotation = _finite_array(extrinsics.get("rotation"), (9,), name="ir2_to_ir1.rotation").reshape(3, 3)
    translation = _finite_array(
        extrinsics.get("translation"), (3,), name="ir2_to_ir1.translation"
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-5
    ):
        raise ValueError("D405 IR extrinsic rotation must be proper")
    baseline = float(np.linalg.norm(translation))
    if not 0.010 <= baseline <= 0.030:
        raise ValueError(f"unexpected D405 IR baseline {baseline:.6f} m")
    serial = payload.get("serial_number")
    if not isinstance(serial, str) or not serial:
        raise ValueError("D405 calibration needs serial_number")
    return D405IrStereoCalibration(
        camera_matrix_ir1=_intrinsic_matrix(payload, name="ir1"),
        camera_matrix_ir2=_intrinsic_matrix(payload, name="ir2"),
        rotation_ir2_to_ir1=rotation,
        translation_ir2_to_ir1_m=translation,
        serial_number=serial,
    )


def triangulate_d405_ir_pixels(
    ir1_pixels: Sequence[Sequence[float]],
    ir2_pixels: Sequence[Sequence[float]],
    calibration: D405IrStereoCalibration,
) -> np.ndarray:
    """Triangulate raw IR1/IR2 correspondences into the IR1 frame in metres."""

    left = np.asarray(ir1_pixels, dtype=np.float64)
    right = np.asarray(ir2_pixels, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape or left.shape[0] == 0:
        raise ValueError("IR1/IR2 pixels must be equal non-empty [N,2] arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("IR pixels must be finite")
    rotation_ir1_to_ir2 = calibration.rotation_ir2_to_ir1.T
    translation_ir1_to_ir2 = -rotation_ir1_to_ir2 @ calibration.translation_ir2_to_ir1_m
    projection_ir1 = calibration.camera_matrix_ir1 @ np.column_stack((np.eye(3), np.zeros(3)))
    projection_ir2 = calibration.camera_matrix_ir2 @ np.column_stack(
        (rotation_ir1_to_ir2, translation_ir1_to_ir2)
    )
    homogeneous = cv2.triangulatePoints(projection_ir1, projection_ir2, left.T, right.T)
    if np.any(np.abs(homogeneous[3]) <= _EPSILON):
        raise ValueError("D405 correspondence triangulates at infinity")
    points = (homogeneous[:3] / homogeneous[3]).T
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise ValueError("D405 correspondences must triangulate in front of IR1")
    return points


def d405_epipolar_errors_px(
    ir1_pixels: Sequence[Sequence[float]],
    ir2_pixels: Sequence[Sequence[float]],
    calibration: D405IrStereoCalibration,
) -> np.ndarray:
    """Return symmetric raw-pixel epipolar distances for D405 IR pairs."""

    left = np.asarray(ir1_pixels, dtype=np.float64)
    right = np.asarray(ir2_pixels, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape or left.shape[0] == 0:
        raise ValueError("IR1/IR2 pixels must be equal non-empty [N,2] arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("IR pixels must be finite")
    translation = calibration.translation_ir2_to_ir1_m
    skew = np.array(
        ((0.0, -translation[2], translation[1]), (translation[2], 0.0, -translation[0]), (-translation[1], translation[0], 0.0)),
        dtype=np.float64,
    )
    fundamental = (
        np.linalg.inv(calibration.camera_matrix_ir1).T
        @ skew
        @ calibration.rotation_ir2_to_ir1
        @ np.linalg.inv(calibration.camera_matrix_ir2)
    )
    homogeneous_left = np.column_stack((left, np.ones(len(left))))
    homogeneous_right = np.column_stack((right, np.ones(len(right))))
    line_left = (fundamental @ homogeneous_right.T).T
    line_right = (fundamental.T @ homogeneous_left.T).T
    left_distance = np.abs(np.sum(homogeneous_left * line_left, axis=1)) / np.maximum(
        np.linalg.norm(line_left[:, :2], axis=1), _EPSILON
    )
    right_distance = np.abs(np.sum(homogeneous_right * line_right, axis=1)) / np.maximum(
        np.linalg.norm(line_right[:, :2], axis=1), _EPSILON
    )
    return 0.5 * (left_distance + right_distance)


def _rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = _finite_array(rotvec, (3,), name="rotation vector")
    theta = float(np.linalg.norm(rotvec))
    if theta <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    skew = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _transform_from_parameter(parameter: np.ndarray) -> np.ndarray:
    parameter = _finite_array(parameter, (6,), name="SE(3) parameter")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_rotvec(parameter[3:])
    transform[:3, 3] = parameter[:3]
    return transform


def _parameter_from_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4,4]")
    return np.concatenate((transform[:3, 3], rotation_log(transform[:3, :3])))


def _pose_residual(actual: np.ndarray, expected: np.ndarray, *, rotation_scale_m: float) -> np.ndarray:
    delta = inverse_transform(expected) @ actual
    return np.concatenate((delta[:3, 3], rotation_scale_m * rotation_log(delta[:3, :3])))


def _observation_residuals(
    parameter: np.ndarray,
    observations: Sequence[WristTableObservation],
    *,
    rotation_scale_m: float,
) -> np.ndarray:
    root_from_table = _transform_from_parameter(parameter[:6])
    eef_from_camera = _transform_from_parameter(parameter[6:])
    residuals = []
    for observation in observations:
        root_from_eef = pose_to_transform(observation.source_eef_state_root)
        predicted = root_from_eef @ eef_from_camera @ observation.camera_from_table
        residuals.append(_pose_residual(predicted, root_from_table, rotation_scale_m=rotation_scale_m))
    return np.concatenate(residuals)


def _solve_damped_least_squares(
    initial: np.ndarray,
    observations: Sequence[WristTableObservation],
    *,
    rotation_scale_m: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    parameter = initial.astype(np.float64, copy=True)
    damping = 1.0e-3
    epsilon = 1.0e-5
    residual = _observation_residuals(parameter, observations, rotation_scale_m=rotation_scale_m)
    cost = float(residual @ residual)
    for _ in range(max_iterations):
        jacobian = np.empty((residual.size, parameter.size), dtype=np.float64)
        for column in range(parameter.size):
            shifted = parameter.copy()
            shifted[column] += epsilon
            jacobian[:, column] = (
                _observation_residuals(shifted, observations, rotation_scale_m=rotation_scale_m) - residual
            ) / epsilon
        normal = jacobian.T @ jacobian + damping * np.eye(parameter.size)
        try:
            update = np.linalg.solve(normal, -jacobian.T @ residual)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        candidate = parameter + update
        candidate_residual = _observation_residuals(
            candidate, observations, rotation_scale_m=rotation_scale_m
        )
        candidate_cost = float(candidate_residual @ candidate_residual)
        if candidate_cost < cost:
            parameter, residual, cost = candidate, candidate_residual, candidate_cost
            damping = max(damping * 0.3, 1.0e-9)
            if float(np.linalg.norm(update)) < 1.0e-8:
                break
        else:
            damping *= 10.0
    return parameter, cost


def _direct_fiducial_residuals(
    parameter: np.ndarray,
    observations: Sequence[WristTableFiducialObservation],
) -> np.ndarray:
    root_from_table = _transform_from_parameter(parameter[:6])
    eef_from_camera = _transform_from_parameter(parameter[6:])
    residuals = []
    for observation in observations:
        root_from_eef = pose_to_transform(observation.source_eef_state_root)
        predicted_root_point = (
            root_from_eef @ eef_from_camera @ np.append(observation.camera_point_m, 1.0)
        )[:3]
        expected_root_point = (root_from_table @ np.append(observation.table_point_m, 1.0))[:3]
        residuals.append(predicted_root_point - expected_root_point)
    return np.concatenate(residuals)


def _solve_direct_fiducials(
    observations: Sequence[WristTableFiducialObservation], *, max_iterations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly solve hand-eye and table pose from points spread across views."""

    if len(observations) < 8:
        raise ValueError("direct wrist hand-eye calibration needs at least eight fiducial observations")
    positions = np.asarray([observation.source_eef_state_root[:3] for observation in observations])
    if np.linalg.matrix_rank(positions - positions.mean(axis=0), tol=1.0e-5) < 2:
        raise ValueError("EEF observations lack translational excitation for hand-eye calibration")
    observation_ids = {observation.observation_id for observation in observations}
    if len(observation_ids) < 3:
        raise ValueError("direct wrist hand-eye calibration needs at least three EEF views")
    unique_table_points = np.unique(
        np.asarray([observation.table_point_m for observation in observations]), axis=0
    )
    if unique_table_points.shape[0] < 3:
        raise ValueError("direct wrist hand-eye calibration needs three distinct table points")
    centered = unique_table_points - unique_table_points.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=1.0e-8) < 2:
        raise ValueError("direct wrist hand-eye calibration needs non-collinear table points")

    seed_rotations = (
        np.zeros(3),
        np.array((math.pi, 0.0, 0.0)),
        np.array((0.0, math.pi, 0.0)),
        np.array((0.0, 0.0, math.pi)),
        np.array((math.pi / 2.0, 0.0, 0.0)),
        np.array((0.0, math.pi / 2.0, 0.0)),
        np.array((0.0, 0.0, math.pi / 2.0)),
    )
    first = observations[0]
    root_from_eef = pose_to_transform(first.source_eef_state_root)
    best: tuple[np.ndarray, float] | None = None
    # Both poses have unknown orientation. Multiple physically neutral seeds
    # prevent one arbitrary camera convention from deciding the result.
    for table_rotation_seed in seed_rotations:
        root_from_table = np.eye(4, dtype=np.float64)
        root_from_table[:3, :3] = _rotation_from_rotvec(table_rotation_seed)
        for camera_rotation_seed in seed_rotations:
            eef_from_camera = np.eye(4, dtype=np.float64)
            eef_from_camera[:3, :3] = _rotation_from_rotvec(camera_rotation_seed)
            predicted_root_point = (
                root_from_eef @ eef_from_camera @ np.append(first.camera_point_m, 1.0)
            )[:3]
            root_from_table[:3, 3] = predicted_root_point - root_from_table[:3, :3] @ first.table_point_m
            initial = np.concatenate(
                (_parameter_from_transform(root_from_table), _parameter_from_transform(eef_from_camera))
            )
            parameter = initial.astype(np.float64, copy=True)
            damping = 1.0e-3
            epsilon = 1.0e-5
            residual = _direct_fiducial_residuals(parameter, observations)
            cost = float(residual @ residual)
            for _ in range(max_iterations):
                jacobian = np.empty((residual.size, parameter.size), dtype=np.float64)
                for column in range(parameter.size):
                    shifted = parameter.copy()
                    shifted[column] += epsilon
                    jacobian[:, column] = (_direct_fiducial_residuals(shifted, observations) - residual) / epsilon
                try:
                    update = np.linalg.solve(
                        jacobian.T @ jacobian + damping * np.eye(parameter.size), -jacobian.T @ residual
                    )
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue
                candidate = parameter + update
                candidate_residual = _direct_fiducial_residuals(candidate, observations)
                candidate_cost = float(candidate_residual @ candidate_residual)
                if candidate_cost < cost:
                    parameter, residual, cost = candidate, candidate_residual, candidate_cost
                    damping = max(damping * 0.3, 1.0e-9)
                    if float(np.linalg.norm(update)) < 1.0e-8:
                        break
                else:
                    damping *= 10.0
            if best is None or cost < best[1]:
                best = parameter, cost
    if best is None:
        raise RuntimeError("direct wrist hand-eye solver did not produce a candidate")
    parameter, _cost = best
    residuals = _direct_fiducial_residuals(parameter, observations).reshape(-1, 3)
    return _transform_from_parameter(parameter[:6]), _transform_from_parameter(parameter[6:]), residuals


def solve_wrist_hand_eye(
    observations: Sequence[WristTableObservation],
    *,
    rotation_scale_m: float = 0.08,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Solve source ``root<-table`` and ``eef<-IR1`` from static-table views."""

    if len(observations) < 3:
        raise ValueError("wrist hand-eye calibration needs at least three static-table observations")
    if rotation_scale_m <= 0.0 or max_iterations <= 0:
        raise ValueError("rotation_scale_m and max_iterations must be positive")
    positions = np.asarray([observation.source_eef_state_root[:3] for observation in observations])
    if np.linalg.matrix_rank(positions - positions.mean(axis=0), tol=1.0e-5) < 2:
        raise ValueError("EEF observations lack translational excitation for hand-eye calibration")

    # Multiple orientation seeds avoid deciding an unobserved camera frame by
    # convention. The selected solution is always the lowest residual one.
    seed_rotations = (
        np.zeros(3),
        np.array((math.pi, 0.0, 0.0)),
        np.array((0.0, math.pi, 0.0)),
        np.array((0.0, 0.0, math.pi)),
        np.array((math.pi / 2.0, 0.0, 0.0)),
        np.array((0.0, math.pi / 2.0, 0.0)),
        np.array((0.0, 0.0, math.pi / 2.0)),
    )
    best: tuple[np.ndarray, float] | None = None
    first_root_from_eef = pose_to_transform(observations[0].source_eef_state_root)
    for rotation_seed in seed_rotations:
        eef_from_camera = np.eye(4, dtype=np.float64)
        eef_from_camera[:3, :3] = _rotation_from_rotvec(rotation_seed)
        root_from_table = first_root_from_eef @ eef_from_camera @ observations[0].camera_from_table
        initial = np.concatenate(
            (_parameter_from_transform(root_from_table), _parameter_from_transform(eef_from_camera))
        )
        solved, cost = _solve_damped_least_squares(
            initial, observations, rotation_scale_m=rotation_scale_m, max_iterations=max_iterations
        )
        if best is None or cost < best[1]:
            best = solved, cost
    if best is None:
        raise RuntimeError("hand-eye solver did not produce a candidate")
    parameter, _cost = best
    root_from_table = _transform_from_parameter(parameter[:6])
    eef_from_camera = _transform_from_parameter(parameter[6:])
    diagnostics: list[dict[str, float]] = []
    for observation in observations:
        predicted = pose_to_transform(observation.source_eef_state_root) @ eef_from_camera @ observation.camera_from_table
        delta = inverse_transform(root_from_table) @ predicted
        diagnostics.append(
            {
                "translation_residual_m": float(np.linalg.norm(delta[:3, 3])),
                "rotation_residual_rad": float(np.linalg.norm(rotation_log(delta[:3, :3]))),
            }
        )
    return root_from_table, eef_from_camera, diagnostics


def build_wrist_table_observations(
    payload: dict[str, Any], calibration: D405IrStereoCalibration
) -> list[WristTableObservation]:
    """Triangulate grouped IR fiducials and fit table pose for each observation."""

    groups = payload.get("wrist_table_observations")
    if not isinstance(groups, list):
        raise ValueError("payload needs wrist_table_observations")
    observations: list[WristTableObservation] = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each wrist_table_observation must be an object")
        observation_id = group.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("each wrist_table_observation needs observation_id")
        eef_pose = _finite_array(group.get("source_eef_state_root"), (EEF_POSE_DIM,), name="source_eef_state_root")
        fiducials = group.get("table_fiducials")
        if not isinstance(fiducials, list) or len(fiducials) < 3:
            raise ValueError("each wrist_table_observation needs at least three table_fiducials")
        try:
            table_points = np.asarray([entry["table_point_m"] for entry in fiducials], dtype=np.float64)
            ir1_pixels = np.asarray([entry["ir1_pixel_raw"] for entry in fiducials], dtype=np.float64)
            ir2_pixels = np.asarray([entry["ir2_pixel_raw"] for entry in fiducials], dtype=np.float64)
        except KeyError as error:
            raise ValueError("each wrist fiducial needs table_point_m, ir1_pixel_raw, and ir2_pixel_raw") from error
        if table_points.shape != (len(fiducials), 3):
            raise ValueError("table_point_m must be [N,3]")
        camera_points = triangulate_d405_ir_pixels(ir1_pixels, ir2_pixels, calibration)
        epipolar_errors = d405_epipolar_errors_px(ir1_pixels, ir2_pixels, calibration)
        camera_from_table, residuals = fit_rigid_transform(table_points, camera_points)
        observations.append(
            WristTableObservation(
                observation_id,
                eef_pose,
                camera_from_table,
                residuals,
                epipolar_errors,
            )
        )
    return observations


def build_direct_wrist_table_fiducials(
    payload: dict[str, Any], calibration: D405IrStereoCalibration
) -> list[WristTableFiducialObservation]:
    """Read visible table points without requiring three points per camera view."""

    groups = payload.get("wrist_table_observations")
    if not isinstance(groups, list):
        raise ValueError("payload needs wrist_table_observations")
    observations: list[WristTableFiducialObservation] = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each wrist_table_observation must be an object")
        observation_id = group.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("each wrist_table_observation needs observation_id")
        eef_pose = _finite_array(group.get("source_eef_state_root"), (EEF_POSE_DIM,), name="source_eef_state_root")
        fiducials = group.get("table_fiducials")
        if not isinstance(fiducials, list) or not fiducials:
            raise ValueError("each wrist_table_observation needs at least one table_fiducial")
        try:
            table_points = np.asarray([entry["table_point_m"] for entry in fiducials], dtype=np.float64)
            ir1_pixels = np.asarray([entry["ir1_pixel_raw"] for entry in fiducials], dtype=np.float64)
            ir2_pixels = np.asarray([entry["ir2_pixel_raw"] for entry in fiducials], dtype=np.float64)
        except KeyError as error:
            raise ValueError("each wrist fiducial needs table_point_m, ir1_pixel_raw, and ir2_pixel_raw") from error
        if table_points.shape != (len(fiducials), 3) or not np.isfinite(table_points).all():
            raise ValueError("table_point_m must be finite [N,3]")
        camera_points = triangulate_d405_ir_pixels(ir1_pixels, ir2_pixels, calibration)
        epipolar_errors = d405_epipolar_errors_px(ir1_pixels, ir2_pixels, calibration)
        observations.extend(
            WristTableFiducialObservation(
                observation_id, eef_pose, table_point, camera_point, float(epipolar_error)
            )
            for table_point, camera_point, epipolar_error in zip(
                table_points, camera_points, epipolar_errors, strict=True
            )
        )
    return observations


def calibrate_source_table_from_wrist_ir(
    payload: dict[str, Any],
    calibration: D405IrStereoCalibration,
    *,
    max_hand_eye_translation_rms_m: float,
    max_hand_eye_rotation_rms_rad: float,
    max_table_fit_rms_m: float,
    max_stereo_epipolar_error_px: float = 1.5,
) -> dict[str, Any]:
    """Run a residual-gated offline D405 hand-eye/table calibration."""

    if min(
        max_hand_eye_translation_rms_m,
        max_hand_eye_rotation_rms_rad,
        max_table_fit_rms_m,
        max_stereo_epipolar_error_px,
    ) <= 0.0:
        raise ValueError("all wrist calibration thresholds must be positive")
    provenance = payload.get(
        "source_task_frame_provenance",
        "calibrated from recorded D405 IR stereo table fiducials and root-frame EEF state",
    )
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("source_task_frame_provenance must be a non-empty string")
    if provenance.lower().strip().startswith("replace "):
        raise ValueError("source_task_frame_provenance must not retain the workspace placeholder")
    binding_keys = (
        "source_episode_index",
        "calibration_endpoint",
        "workspace_manifest_sha256",
    )
    binding = {key: payload.get(key) for key in binding_keys}
    if any(value is not None for value in binding.values()) and not all(
        value is not None for value in binding.values()
    ):
        raise ValueError(
            "source episode, calibration endpoint, and annotation workspace hash must be provided together"
        )
    if binding["source_episode_index"] is not None:
        if not isinstance(binding["source_episode_index"], int) or binding["source_episode_index"] < 0:
            raise ValueError("source_episode_index must be a non-negative integer")
        if binding["calibration_endpoint"] not in {"initial", "final"}:
            raise ValueError("calibration_endpoint must be initial or final")
        workspace_sha = binding["workspace_manifest_sha256"]
        if not isinstance(workspace_sha, str) or len(workspace_sha) != 64 or any(
            character not in "0123456789abcdef" for character in workspace_sha
        ):
            raise ValueError("workspace_manifest_sha256 must be 64 lowercase hexadecimal characters")
    source_table_body_frame = payload.get("source_table_body_frame")
    if source_table_body_frame != V1_TABLE001_BODY_FRAME:
        raise ValueError(
            "source_table_body_frame must be "
            f"{V1_TABLE001_BODY_FRAME!r}; calibration points must use the V1 physics body frame"
        )
    groups = payload.get("wrist_table_observations")
    if not isinstance(groups, list):
        raise ValueError("payload needs wrist_table_observations")
    per_view_pose_mode = all(
        isinstance(group, dict)
        and isinstance(group.get("table_fiducials"), list)
        and len(group["table_fiducials"]) >= 3
        for group in groups
    )
    if per_view_pose_mode:
        observations = build_wrist_table_observations(payload, calibration)
        root_from_table, eef_from_camera, diagnostics = solve_wrist_hand_eye(observations)
        translation_residuals = np.asarray([item["translation_residual_m"] for item in diagnostics])
        rotation_residuals = np.asarray([item["rotation_residual_rad"] for item in diagnostics])
        table_residuals = np.concatenate([item.table_fit_residuals_m for item in observations])
        epipolar_errors = np.concatenate([item.epipolar_errors_px for item in observations])
        metrics = {
            "hand_eye_translation_rms_m": float(math.sqrt(float(np.mean(np.square(translation_residuals))))),
            "hand_eye_rotation_rms_rad": float(math.sqrt(float(np.mean(np.square(rotation_residuals))))),
            "table_fit_rms_m": float(math.sqrt(float(np.mean(np.square(table_residuals))))),
            "stereo_epipolar_rms_px": float(math.sqrt(float(np.mean(np.square(epipolar_errors))))),
            "stereo_epipolar_max_px": float(np.max(epipolar_errors)),
        }
        output_observations: list[dict[str, Any]] = [
            {
                "observation_id": observation.observation_id,
                "source_eef_state_root": observation.source_eef_state_root.tolist(),
                "camera_from_table": observation.camera_from_table.tolist(),
                "table_fit_residuals_m": observation.table_fit_residuals_m.tolist(),
                "stereo_epipolar_errors_px": observation.epipolar_errors_px.tolist(),
                **diagnostic,
            }
            for observation, diagnostic in zip(observations, diagnostics, strict=True)
        ]
        calibration_mode = "per_view_table_pose"
    else:
        direct_observations = build_direct_wrist_table_fiducials(payload, calibration)
        root_from_table, eef_from_camera, point_residuals = _solve_direct_fiducials(
            direct_observations, max_iterations=100
        )
        point_norms = np.linalg.norm(point_residuals, axis=1)
        epipolar_errors = np.asarray(
            [observation.epipolar_error_px for observation in direct_observations], dtype=np.float64
        )
        metrics = {
            "hand_eye_translation_rms_m": float(math.sqrt(float(np.mean(np.square(point_norms))))),
            "hand_eye_rotation_rms_rad": None,
            "table_fit_rms_m": float(math.sqrt(float(np.mean(np.square(point_norms))))),
            "stereo_epipolar_rms_px": float(math.sqrt(float(np.mean(np.square(epipolar_errors))))),
            "stereo_epipolar_max_px": float(np.max(epipolar_errors)),
        }
        output_observations = [
            {
                "observation_id": observation.observation_id,
                "source_eef_state_root": observation.source_eef_state_root.tolist(),
                "table_point_m": observation.table_point_m.tolist(),
                "ir1_point_m": observation.camera_point_m.tolist(),
                "stereo_epipolar_error_px": observation.epipolar_error_px,
                "root_point_residual_m": float(residual),
            }
            for observation, residual in zip(direct_observations, point_norms, strict=True)
        ]
        calibration_mode = "direct_multi_view_fiducials"
    review = payload.get("acceptance_review")
    reviewed_ids = review.get("reviewed_observation_ids") if isinstance(review, dict) else None
    annotated_ids = {
        group.get("observation_id")
        for group in groups
        if isinstance(group, dict)
        and isinstance(group.get("observation_id"), str)
        and isinstance(group.get("table_fiducials"), list)
        and bool(group["table_fiducials"])
    }
    reviewed_ids_are_valid = (
        isinstance(review, dict)
        and review.get("schema_version") == REVIEW_SCHEMA_VERSION
        and isinstance(review.get("reviewer_id"), str)
        and bool(review["reviewer_id"].strip())
        and isinstance(review.get("static_table_evidence"), str)
        and bool(review["static_table_evidence"].strip())
        and isinstance(review.get("d405_rigid_mount_evidence"), str)
        and bool(review["d405_rigid_mount_evidence"].strip())
        and isinstance(reviewed_ids, list)
        and len(set(reviewed_ids)) >= 3
        and all(isinstance(item, str) and item for item in reviewed_ids)
        and set(reviewed_ids).issubset(annotated_ids)
    )
    static_confirmed = payload.get("table_is_static_confirmation") is True
    camera_rigid_confirmed = payload.get("d405_is_rigid_to_eef_confirmation") is True
    acceptance_eligible = (
        static_confirmed
        and camera_rigid_confirmed
        and reviewed_ids_are_valid
        and metrics["hand_eye_translation_rms_m"] <= max_hand_eye_translation_rms_m
        and (
            metrics["hand_eye_rotation_rms_rad"] is None
            or metrics["hand_eye_rotation_rms_rad"] <= max_hand_eye_rotation_rms_rad
        )
        and metrics["table_fit_rms_m"] <= max_table_fit_rms_m
        and (
            metrics["stereo_epipolar_max_px"] is None
            or metrics["stereo_epipolar_max_px"] <= max_stereo_epipolar_error_px
        )
    )
    result = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "offline_teacher_only": True,
        "calibration_method": "D405 IR stereo static-table hand-eye calibration",
        "calibration_mode": calibration_mode,
        "source_task_frame_provenance": provenance.strip(),
        "source_table_body_frame": source_table_body_frame,
        "source_task_pose_root": transform_to_pose(root_from_table).tolist(),
        "source_root_from_table": root_from_table.tolist(),
        "source_eef_from_ir1_camera": transform_to_pose(eef_from_camera).tolist(),
        "d405_serial_number": calibration.serial_number,
        "acceptance_eligible": acceptance_eligible,
        "acceptance_requirements": {
            "table_is_static_confirmation": True,
            "d405_is_rigid_to_eef_confirmation": True,
            "accepted_human_review": {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "minimum_distinct_reviewed_observation_ids": 3,
                "required_fields": (
                    "reviewer_id, static_table_evidence, d405_rigid_mount_evidence, "
                    "reviewed_observation_ids"
                ),
            },
            "max_hand_eye_translation_rms_m": max_hand_eye_translation_rms_m,
            "max_hand_eye_rotation_rms_rad": max_hand_eye_rotation_rms_rad,
            "hand_eye_rotation_rms_required": calibration_mode == "per_view_table_pose",
            "max_table_fit_rms_m": max_table_fit_rms_m,
            "max_stereo_epipolar_error_px": max_stereo_epipolar_error_px,
        },
        "metrics": metrics,
        "observations": output_observations,
        "limitations": (
            "offline calibration only; policy, critic, and runtime planner must never read table pose, "
            "D405 extrinsics, calibration residuals, or source EEF targets"
        ),
    }
    if binding["source_episode_index"] is not None:
        result.update(binding)
    return result
