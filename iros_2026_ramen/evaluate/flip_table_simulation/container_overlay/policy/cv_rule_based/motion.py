"""Geometry-only flip-table controller for Pink wrist and Dex1 commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class Phase(str, Enum):
    CLEARANCE_STAGING = "clearance_staging"
    ALIGN_APPROACH = "align_approach"
    ALIGN_GRASP = "align_grasp"
    ALIGN_SHORT_EDGE = "align_short_edge"
    LEFT_LEG_FLIP_90 = "left_leg_flip_90"
    RIGHT_PREGRASP = "right_pregrasp"
    RIGHT_GRASP = "right_grasp"
    HANDOVER = "handover"
    RIGHT_TOP_FLIP_90 = "right_top_flip_90"
    SETTLE_AND_RETREAT = "settle_and_retreat"


@dataclass(frozen=True)
class _Keyframe:
    phase: Phase
    duration_s: float
    left_position: np.ndarray
    left_rotation: np.ndarray
    right_position: np.ndarray
    right_rotation: np.ndarray
    left_hand: float
    right_hand: float


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(unit).all() or norm <= 1.0e-12:
        raise ValueError("rotation axis must be finite and non-zero")
    unit = unit / norm
    skew = np.asarray(
        ((0.0, -unit[2], unit[1]), (unit[2], 0.0, -unit[0]),
         (-unit[1], unit[0], 0.0)), dtype=np.float64
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale), dtype=np.float64
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            result = np.asarray(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                                 (matrix[0, 1] + matrix[1, 0]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale))
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            result = np.asarray(((matrix[0, 2] - matrix[2, 0]) / scale,
                                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                                 (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            result = np.asarray(((matrix[1, 0] - matrix[0, 1]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale,
                                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm <= 1.0e-12:
        raise ValueError("rotation produced an invalid quaternion")
    result /= norm
    return result


def _rotation_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _interpolate_rotation(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two rotations on the shortest quaternion arc."""

    amount = float(np.clip(fraction, 0.0, 1.0))
    start_quaternion = _quaternion_wxyz(start)
    end_quaternion = _quaternion_wxyz(end)
    dot = float(np.dot(start_quaternion, end_quaternion))
    if dot < 0.0:
        end_quaternion = -end_quaternion
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        quaternion = (1.0 - amount) * start_quaternion + amount * end_quaternion
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        quaternion = (
            math.sin((1.0 - amount) * angle) / sine * start_quaternion
            + math.sin(amount * angle) / sine * end_quaternion
        )
    return _rotation_from_quaternion_wxyz(quaternion)


def validate_cartesian_action(action: np.ndarray) -> np.ndarray:
    """Validate the 16-D Pink wrist-pose and Dex1 command contract."""

    validated = np.asarray(action, dtype=np.float32)
    if validated.shape != (16,):
        raise ValueError(f"Cartesian action must have shape (16,), got {validated.shape}")
    if not np.isfinite(validated).all():
        raise ValueError("Cartesian action contains NaN or Inf")
    for name, quaternion_slice in (("left", slice(3, 7)), ("right", slice(10, 14))):
        norm = float(np.linalg.norm(validated[quaternion_slice]))
        if norm <= 1.0e-6:
            raise ValueError(f"{name} wrist quaternion is zero")
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
            raise ValueError(f"{name} wrist quaternion norm is {norm:.8f}, expected 1")
    if np.any(np.abs(validated[14:16]) > 1.0 + 1.0e-6):
        raise ValueError("Dex1 commands must remain in [-1, 1]")
    return validated


def limit_cartesian_action_rate(
    previous_action: np.ndarray,
    requested_action: np.ndarray,
    control_hz: float,
    *,
    max_linear_speed_m_s: float,
    max_angular_speed_rad_s: float,
    max_hand_speed_s: float,
) -> np.ndarray:
    """Bound per-cycle wrist motion before it reaches the IK controller."""

    previous = validate_cartesian_action(previous_action)
    requested = validate_cartesian_action(requested_action)
    if min(
        control_hz,
        max_linear_speed_m_s,
        max_angular_speed_rad_s,
        max_hand_speed_s,
    ) <= 0.0:
        raise ValueError("Cartesian command-rate limits must be positive")
    dt = 1.0 / control_hz
    limited = requested.copy()
    for position_slice, quaternion_slice in (
        (slice(0, 3), slice(3, 7)),
        (slice(7, 10), slice(10, 14)),
    ):
        delta = requested[position_slice].astype(np.float64) - previous[
            position_slice
        ].astype(np.float64)
        distance = float(np.linalg.norm(delta))
        max_distance = max_linear_speed_m_s * dt
        if distance > max_distance:
            limited[position_slice] = previous[position_slice] + delta * (
                max_distance / distance
            )

        previous_rotation = _rotation_from_quaternion_wxyz(previous[quaternion_slice])
        requested_rotation = _rotation_from_quaternion_wxyz(requested[quaternion_slice])
        quaternion_dot = abs(
            float(
                np.dot(
                    _quaternion_wxyz(previous_rotation),
                    _quaternion_wxyz(requested_rotation),
                )
            )
        )
        angular_distance = 2.0 * math.acos(float(np.clip(quaternion_dot, 0.0, 1.0)))
        max_angle = max_angular_speed_rad_s * dt
        fraction = 1.0 if angular_distance <= max_angle else max_angle / angular_distance
        quaternion = _quaternion_wxyz(
            _interpolate_rotation(previous_rotation, requested_rotation, fraction)
        )
        if float(np.dot(quaternion, previous[quaternion_slice])) < 0.0:
            quaternion *= -1.0
        limited[quaternion_slice] = quaternion

    max_hand_delta = max_hand_speed_s * dt
    limited[14:16] = previous[14:16] + np.clip(
        requested[14:16] - previous[14:16],
        -max_hand_delta,
        max_hand_delta,
    )
    return validate_cartesian_action(limited)


def _undirected_yaw_delta(target: float, source: float) -> float:
    """Return the smallest yaw delta for an axis whose two directions are equivalent."""

    return 0.5 * math.atan2(
        math.sin(2.0 * (target - source)),
        math.cos(2.0 * (target - source)),
    )


def blend_table_frames(
    current: np.ndarray,
    candidate: np.ndarray,
    alpha: float,
    *,
    max_translation_m: float,
    max_yaw_rad: float,
) -> np.ndarray:
    """Low-pass an RGB table-frame update after rejecting discontinuous detections."""

    current = np.asarray(current, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if current.shape != (4, 4) or candidate.shape != (4, 4):
        raise ValueError("table frames must be [4,4]")
    if not np.isfinite(current).all() or not np.isfinite(candidate).all():
        raise ValueError("table frames must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")
    translation_delta = candidate[:2, 3] - current[:2, 3]
    translation_norm = float(np.linalg.norm(translation_delta))
    current_yaw = math.atan2(float(current[1, 0]), float(current[0, 0]))
    candidate_yaw = math.atan2(float(candidate[1, 0]), float(candidate[0, 0]))
    yaw_delta = _undirected_yaw_delta(candidate_yaw, current_yaw)
    if translation_norm > max_translation_m:
        raise ValueError(f"table-frame translation jump {translation_norm:.3f} m")
    if abs(yaw_delta) > max_yaw_rad:
        raise ValueError(f"table-frame yaw jump {yaw_delta:.3f} rad")
    yaw = current_yaw + alpha * yaw_delta
    long_axis = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
    short_axis = np.asarray((math.sin(yaw), -math.cos(yaw), 0.0))
    blended = current.copy()
    blended[:2, 3] += alpha * translation_delta
    blended[:3, 0] = long_axis
    blended[:3, 1] = short_axis
    blended[:3, 2] = (0.0, 0.0, -1.0)
    return blended


def validate_static_table_redetection(
    initial_frame: np.ndarray,
    candidate_frame: np.ndarray,
    initial_attachments: dict[str, np.ndarray],
    candidate_attachments: dict[str, np.ndarray],
    *,
    max_center_drift_m: float,
    max_attachment_drift_m: float,
) -> None:
    """Reject RGB updates inconsistent with a stationary pre-grasp table.

    Before either hand closes, the table should not move.  Comparing each new
    monocular estimate with the settled RGB estimate prevents an approaching
    hand or arm from being mistaken for a white table leg.
    """

    initial = np.asarray(initial_frame, dtype=np.float64)
    candidate = np.asarray(candidate_frame, dtype=np.float64)
    if initial.shape != (4, 4) or candidate.shape != (4, 4):
        raise ValueError("table frames must be [4,4]")
    if max_center_drift_m <= 0.0 or max_attachment_drift_m <= 0.0:
        raise ValueError("redetection drift limits must be positive")
    if set(initial_attachments) != {"left", "right"} or set(candidate_attachments) != {
        "left",
        "right",
    }:
        raise ValueError("redetection attachments must contain left and right")

    center_drift = float(np.linalg.norm(candidate[:2, 3] - initial[:2, 3]))
    if center_drift > max_center_drift_m:
        raise ValueError(f"table-center drift {center_drift:.3f} m")
    for side in ("left", "right"):
        anchor = np.asarray(initial_attachments[side], dtype=np.float64).reshape(3)
        update = np.asarray(candidate_attachments[side], dtype=np.float64).reshape(3)
        drift = float(np.linalg.norm(update[:2] - anchor[:2]))
        if drift > max_attachment_drift_m:
            raise ValueError(f"{side} leg-attachment drift {drift:.3f} m")


def update_bounded_integral_offsets(
    offsets: np.ndarray,
    errors: np.ndarray,
    enabled: tuple[bool, bool],
    *,
    gain: float,
    max_step_m: float,
    max_norm_m: float,
) -> np.ndarray:
    """Integrate Cartesian tracking error without retaining stale arm bias."""

    result = np.asarray(offsets, dtype=np.float64).copy()
    error = np.asarray(errors, dtype=np.float64)
    if result.shape != (2, 3) or error.shape != (2, 3):
        raise ValueError("integral wrist offsets and errors must be [2,3]")
    if gain < 0.0 or max_step_m <= 0.0 or max_norm_m <= 0.0:
        raise ValueError("invalid integral wrist-servo setting")
    for side_index, side_enabled in enumerate(enabled):
        if not side_enabled:
            result[side_index] = 0.0
            continue
        delta = gain * error[side_index]
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step_m:
            delta *= max_step_m / delta_norm
        result[side_index] += delta
        offset_norm = float(np.linalg.norm(result[side_index]))
        if offset_norm > max_norm_m:
            result[side_index] *= max_norm_m / offset_norm
    return result


GRASP_RETRY_OFFSETS_TOOL_M = (
    (0.0, 0.0, 0.0),
    (0.03, 0.0, 0.0),
    (0.03, 0.05, 0.0),
    (0.06, 0.05, 0.0),
    (0.09, 0.05, 0.0),
    (0.09, 0.075, 0.0),
    (0.03, 0.075, 0.0),
    (0.06, 0.075, 0.0),
    (0.03, 0.05, -0.03),
    (0.03, 0.05, 0.03),
    (0.06, 0.0, 0.0),
    (0.09, 0.0, 0.0),
    (-0.03, 0.0, 0.0),
    (0.03, -0.05, 0.0),
    (0.06, -0.05, 0.0),
    (0.09, -0.05, 0.0),
)


def apply_tool_position_offset(
    action: np.ndarray,
    offset_tool_m: np.ndarray | tuple[float, float, float],
    side: str,
) -> np.ndarray:
    """Apply a gripper-local translation without changing orientation."""

    adjusted = validate_cartesian_action(action).copy()
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported gripper side: {side}")
    offset = np.asarray(offset_tool_m, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("tool offset must be a finite 3-D vector")
    position_slice = slice(0, 3) if side == "left" else slice(7, 10)
    quaternion_slice = slice(3, 7) if side == "left" else slice(10, 14)
    rotation = _rotation_from_quaternion_wxyz(adjusted[quaternion_slice])
    adjusted[position_slice] += (rotation @ offset).astype(np.float32)
    return validate_cartesian_action(adjusted)


def grasp_retry_action(
    aligned_action: np.ndarray,
    offset_tool_m: tuple[float, float, float],
    step: int,
    control_hz: float,
    *,
    side: str = "left",
    backoff_m: float = 0.08,
    insertion_depth_m: float = 0.035,
) -> tuple[np.ndarray, str, bool]:
    """Generate one real-compatible regrasp attempt for either Dex1 hand.

    The offset is expressed in the RGB-derived gripper frame. The sequence
    opens while backed away from the leg, approaches along tool +X, inserts the
    object between the fingers, then closes and holds for encoder verification.
    It uses no simulator object or contact state.
    """

    action = validate_cartesian_action(aligned_action).copy()
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported gripper side: {side}")
    if control_hz <= 0.0 or backoff_m <= 0.0 or insertion_depth_m < 0.0:
        raise ValueError("invalid grasp retry distance or control rate")
    if step < 0:
        raise ValueError("grasp retry step must be non-negative")
    local_offset = np.asarray(offset_tool_m, dtype=np.float64)
    if local_offset.shape != (3,) or not np.isfinite(local_offset).all():
        raise ValueError("grasp retry offset must be a finite 3-D vector")

    position_slice = slice(0, 3) if side == "left" else slice(7, 10)
    quaternion_slice = slice(3, 7) if side == "left" else slice(10, 14)
    hand_index = 14 if side == "left" else 15
    rotation = _rotation_from_quaternion_wxyz(action[quaternion_slice])
    target = action[position_slice].astype(np.float64) + rotation @ local_offset
    inserted = target + rotation[:, 0] * insertion_depth_m
    backed = target - rotation[:, 0] * backoff_m
    open_steps = max(1, round(0.40 * control_hz))
    approach_steps = max(1, round(0.80 * control_hz))
    close_steps = max(1, round(1.20 * control_hz))
    verify_steps = max(1, round(0.60 * control_hz))
    clamped = min(step, open_steps + approach_steps + close_steps + verify_steps - 1)

    if clamped < open_steps:
        action[position_slice] = backed
        action[hand_index] = -1.0
        stage = "open_backoff"
        verification_ready = False
    elif clamped < open_steps + approach_steps:
        fraction = (clamped - open_steps + 1) / approach_steps
        action[position_slice] = (1.0 - fraction) * backed + fraction * inserted
        # Keep the fingers fully open until the caller's wrist-camera gate has
        # confirmed that the object lies inside the finger corridor.
        action[hand_index] = -1.0
        stage = "approach"
        verification_ready = False
    elif clamped < open_steps + approach_steps + close_steps:
        fraction = (clamped - open_steps - approach_steps + 1) / close_steps
        action[position_slice] = inserted
        action[hand_index] = -1.0 + 1.75 * fraction
        stage = "close"
        verification_ready = False
    else:
        action[position_slice] = inserted
        action[hand_index] = 0.75
        stage = "verify"
        verification_ready = True
    return validate_cartesian_action(action), stage, verification_ready


def grasp_retry_total_steps(control_hz: float) -> int:
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    return sum(
        max(1, round(duration * control_hz))
        for duration in (0.40, 0.80, 1.20, 0.60)
    )


def dex1_enclosure_from_joint_positions(
    finger_positions: np.ndarray,
    block_threshold_rad: float,
    open_rejection_threshold_rad: float = 0.020,
    max_finger_asymmetry_rad: float = 0.015,
) -> tuple[bool, bool]:
    """Detect a two-sided Dex1 enclosure from measured finger joints.

    A single finger can stop on the outside of a leg while the other closes
    completely. Requiring both joints to remain above the unobstructed-close
    threshold rejects that false grasp without using contact sensors.
    """

    fingers = np.asarray(finger_positions, dtype=np.float64)
    if fingers.shape != (2, 2) or not np.isfinite(fingers).all():
        raise ValueError("Dex1 enclosure requires finite [left/right, finger1/finger2]")
    if not -0.02 < block_threshold_rad < 0.0245:
        raise ValueError("Dex1 block threshold must be inside the joint limits")
    if not block_threshold_rad < open_rejection_threshold_rad < 0.0245:
        raise ValueError("Dex1 open rejection threshold must be above block threshold")
    if not 0.0 < max_finger_asymmetry_rad < 0.0445:
        raise ValueError("Dex1 finger asymmetry threshold is outside joint travel")
    return tuple(
        bool(
            np.min(side) > block_threshold_rad
            and np.max(side) < open_rejection_threshold_rad
            and abs(float(side[0] - side[1])) <= max_finger_asymmetry_rad
        )
        for side in fingers
    )


class GeometricFlipPlanner:
    """Generate the complete motion from detected table geometry, without demonstrations."""

    TABLE_LENGTH_M = 0.58
    TABLE_DEPTH_M = 0.42
    LEG_INSET_M = 0.035
    LEG_GRASP_HEIGHT_M = 0.15
    # Dex1-1 URDF: wrist_yaw_link -> dex1_base_link is 41.5 mm and the
    # collision mesh center is about 110 mm farther along finger +X.
    WRIST_TO_GRASP_M = 0.1515
    def __init__(
        self,
        root_from_table: np.ndarray,
        control_hz: float,
        leg_attachment_points: dict[str, np.ndarray] | None = None,
        *,
        table_is_aligned: bool = False,
    ) -> None:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        transform = np.asarray(root_from_table, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("root_from_table must be finite [4,4]")
        self.table_root_z_m = float(transform[2, 3])
        if not -0.45 <= self.table_root_z_m <= 0.20:
            raise ValueError(
                "tabletop height is outside the reachable root-frame range: "
                f"z={self.table_root_z_m:.3f} m"
            )
        self.control_hz = float(control_hz)
        if leg_attachment_points is not None:
            if set(leg_attachment_points) != {"left", "right"}:
                raise ValueError("leg_attachment_points must contain left and right")
            leg_attachment_points = {
                side: np.asarray(point, dtype=np.float64).reshape(3).copy()
                for side, point in leg_attachment_points.items()
            }
            if not all(np.isfinite(point).all() for point in leg_attachment_points.values()):
                raise ValueError("leg attachment points must be finite")
        self._keyframes = self._build_keyframes(
            transform,
            leg_attachment_points,
            table_is_aligned=table_is_aligned,
        )
        self._end_steps = np.cumsum(
            [max(1, round(frame.duration_s * self.control_hz)) for frame in self._keyframes]
        )

    @staticmethod
    def neutral_action() -> np.ndarray:
        action = np.zeros(16, dtype=np.float32)
        action[0:3] = (0.15, 0.24, 0.24)
        action[3] = 1.0
        action[7:10] = (0.15, -0.24, 0.24)
        action[10] = 1.0
        action[14:16] = -1.0
        return validate_cartesian_action(action)

    @property
    def total_steps(self) -> int:
        return int(self._end_steps[-1])

    def phase_start_step(self, phase: Phase) -> int:
        indices = [index for index, frame in enumerate(self._keyframes) if frame.phase is phase]
        if not indices:
            raise ValueError(f"phase is absent from planner: {phase.value}")
        first = indices[0]
        return 0 if first == 0 else int(self._end_steps[first - 1])

    def phase_end_step(self, phase: Phase) -> int:
        """Return the exclusive end of a contiguous planner phase."""

        indices = [index for index, frame in enumerate(self._keyframes) if frame.phase is phase]
        if not indices:
            raise ValueError(f"phase is absent from planner: {phase.value}")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise RuntimeError(f"planner phase is not contiguous: {phase.value}")
        return int(self._end_steps[indices[-1]])

    def phase_at(self, step: int) -> Phase:
        index = min(len(self._keyframes) - 1, int(np.searchsorted(self._end_steps, step, side="right")))
        return self._keyframes[index].phase

    def action_at(self, step: int) -> np.ndarray:
        clamped = min(max(int(step), 0), self.total_steps - 1)
        index = int(np.searchsorted(self._end_steps, clamped, side="right"))
        frame = self._keyframes[index]
        start_step = 0 if index == 0 else int(self._end_steps[index - 1])
        duration = max(1, int(self._end_steps[index]) - start_step)
        fraction = float(np.clip((clamped - start_step + 1) / duration, 0.0, 1.0))
        previous = self._keyframes[max(0, index - 1)]
        left_position = (1.0 - fraction) * previous.left_position + fraction * frame.left_position
        right_position = (1.0 - fraction) * previous.right_position + fraction * frame.right_position
        left_rotation = _interpolate_rotation(previous.left_rotation, frame.left_rotation, fraction)
        right_rotation = _interpolate_rotation(previous.right_rotation, frame.right_rotation, fraction)
        action = np.zeros(16, dtype=np.float32)
        action[0:3] = left_position
        action[3:7] = _quaternion_wxyz(left_rotation)
        action[7:10] = right_position
        action[10:14] = _quaternion_wxyz(right_rotation)
        action[14] = (1.0 - fraction) * previous.left_hand + fraction * frame.left_hand
        action[15] = (1.0 - fraction) * previous.right_hand + fraction * frame.right_hand
        return action

    def _build_keyframes(
        self,
        table: np.ndarray,
        leg_attachment_points: dict[str, np.ndarray] | None,
        *,
        table_is_aligned: bool,
    ) -> list[_Keyframe]:
        center = table[:3, 3].copy()
        long_axis = table[:3, 0].copy()
        long_axis[2] = 0.0
        long_axis /= np.linalg.norm(long_axis)
        # The long axis is undirected.  Orient it toward the robot so adding a
        # positive long offset always selects the reachable near short edge.
        if long_axis[0] > 0.0:
            long_axis *= -1.0
        short_axis = np.asarray((long_axis[1], -long_axis[0], 0.0), dtype=np.float64)
        up = np.asarray((0.0, 0.0, 1.0))
        # Dex1 wrist +X points from wrist toward the grasp. Keep the initial
        # approach rooted in the robot frame instead of table yaw so a long-edge
        # presentation cannot make either arm approach sideways or twist around
        # a near-singular wrist posture.
        approach_axis = center.copy()
        approach_axis[2] = 0.0
        approach_norm = float(np.linalg.norm(approach_axis))
        if approach_norm <= 1.0e-6:
            approach_axis = np.asarray((1.0, 0.0, 0.0))
        else:
            approach_axis /= approach_norm
        robot_left = np.asarray((-approach_axis[1], approach_axis[0], 0.0))
        initial_tool = np.column_stack((approach_axis, robot_left, up))
        edge_pitch = math.radians(30.0)
        edge_tool_x = (
            math.cos(edge_pitch) * approach_axis - math.sin(edge_pitch) * up
        )
        edge_tool_y = robot_left.copy()
        edge_tool_z = np.cross(edge_tool_x, edge_tool_y)
        edge_tool_z /= np.linalg.norm(edge_tool_z)
        edge_tool = np.column_stack((edge_tool_x, edge_tool_y, edge_tool_z))

        half_length = 0.5 * self.TABLE_LENGTH_M
        half_depth = 0.5 * self.TABLE_DEPTH_M
        long_offset = half_length - self.LEG_INSET_M
        depth_offset = half_depth - self.LEG_INSET_M
        leg_height = self.table_root_z_m + self.LEG_GRASP_HEIGHT_M
        if leg_attachment_points is None:
            near_edge = center + long_axis * long_offset
            left_leg = near_edge + short_axis * depth_offset
            right_leg = near_edge - short_axis * depth_offset
        else:
            left_leg = leg_attachment_points["left"].copy()
            right_leg = leg_attachment_points["right"].copy()
        left_leg[2] = right_leg[2] = leg_height

        def wrist(grasp: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            return grasp - rotation[:, 0] * self.WRIST_TO_GRASP_M

        left_safe = np.asarray((0.15, 0.25, 0.26))
        right_safe = np.asarray((0.15, -0.25, 0.26))

        yaw = math.atan2(long_axis[1], long_axis[0])
        target_yaw = math.atan2(-approach_axis[1], -approach_axis[0])
        yaw_delta = 0.0 if table_is_aligned else math.atan2(
            math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)
        )
        align = _rotation_about(up, yaw_delta)
        aligned_long = align @ long_axis
        aligned_short = align @ short_axis
        self.aligned_long_axis = aligned_long.copy()
        self.aligned_short_axis = aligned_short.copy()
        self.alignment_yaw_delta_rad = float(yaw_delta)

        all_legs = []
        for long_sign in (-1.0, 1.0):
            for short_sign in (-1.0, 1.0):
                point = (
                    center
                    + long_sign * long_offset * long_axis
                    + short_sign * depth_offset * short_axis
                )
                point[2] = leg_height
                all_legs.append(point)
        pivot_leg = min(all_legs, key=lambda point: float(np.linalg.norm(point[:2])))
        pivot_side = (
            "left" if float(np.dot(pivot_leg, robot_left)) >= 0.0 else "right"
        )
        moving_side = "right" if pivot_side == "left" else "left"
        self.alignment_pivot_side = pivot_side
        self.alignment_mode = (
            "already_aligned"
            if table_is_aligned
            else "short_edge_pull"
            if abs(yaw_delta) <= math.pi / 4
            else "long_edge_push_recenter_pull"
        )

        def nearest_edge_grasp(
            frame_center: np.ndarray,
            frame_long: np.ndarray,
            frame_short: np.ndarray,
            side: str,
        ) -> np.ndarray:
            edges = (
                (frame_center + half_length * frame_long, frame_short, half_depth),
                (frame_center - half_length * frame_long, frame_short, half_depth),
                (frame_center + half_depth * frame_short, frame_long, half_length),
                (frame_center - half_depth * frame_short, frame_long, half_length),
            )
            midpoint, tangent, half_span = min(
                edges,
                key=lambda item: float(np.dot(item[0] - frame_center, approach_axis)),
            )
            desired_sign = 1.0 if side == "left" else -1.0
            endpoint_sign = (
                desired_sign
                if float(np.dot(tangent, robot_left)) >= 0.0
                else -desired_sign
            )
            grasp = midpoint + endpoint_sign * 0.62 * half_span * tangent
            grasp[2] = self.table_root_z_m + 0.055
            return grasp

        edge_grasp = nearest_edge_grasp(center, long_axis, short_axis, moving_side)
        self.alignment_pivot_grasp_center = pivot_leg.copy()
        self.alignment_edge_grasp_center = edge_grasp.copy()
        self.alignment_moving_side = moving_side
        pivot_contact = wrist(pivot_leg, initial_tool)
        edge_contact = wrist(edge_grasp, edge_tool)

        def side_pair(
            pivot_value: np.ndarray,
            moving_value: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                (pivot_value, moving_value)
                if pivot_side == "left"
                else (moving_value, pivot_value)
            )

        def side_commands(pivot_value: float, moving_value: float) -> tuple[float, float]:
            return (
                (pivot_value, moving_value)
                if pivot_side == "left"
                else (moving_value, pivot_value)
            )

        left_contact, right_contact = side_pair(pivot_contact, edge_contact)
        left_target, right_target = side_pair(pivot_leg, edge_grasp)
        left_target_tool, right_target_tool = side_pair(initial_tool, edge_tool)
        # At reset the open fingers can already be farther into the table than
        # the near legs.  A direct diagonal move to the pre-grasp therefore
        # catches one leg with a single finger and pushes the table away.  Move
        # laterally outside the assembly, rise above the leg tips, retract to
        # the robot side, and only then align and approach along tool +X.
        staging_height = self.table_root_z_m + 0.50
        outside_center_x = 0.30
        left_outside_center = np.asarray(
            (outside_center_x, max(0.40, left_leg[1] + 0.16), staging_height)
        )
        right_outside_center = np.asarray(
            (outside_center_x, min(-0.40, right_leg[1] - 0.16), staging_height)
        )
        approach_clearance_m = 0.16
        left_high = left_target - initial_tool[:, 0] * approach_clearance_m
        right_high = right_target - initial_tool[:, 0] * approach_clearance_m
        left_high[2] = right_high[2] = staging_height
        left_pre_center = left_target - initial_tool[:, 0] * approach_clearance_m
        right_pre_center = right_target - initial_tool[:, 0] * approach_clearance_m
        left_outside = wrist(left_outside_center, initial_tool)
        right_outside = wrist(right_outside_center, initial_tool)
        left_high_wrist = wrist(left_high, left_target_tool)
        right_high_wrist = wrist(right_high, right_target_tool)
        left_pre = wrist(left_pre_center, left_target_tool)
        right_pre = wrist(right_pre_center, right_target_tool)
        aligned_tool = align @ initial_tool

        # With a long edge initially facing G1, the two visible near legs become
        # a near/far pair after alignment. Retaining the RGB-labelled left leg
        # would therefore place the left arm on the robot-right side. Rebuild
        # all four attachments from the calibrated tabletop dimensions, rotate
        # them with the commanded alignment, and explicitly select the new
        # near-left leg for the first roll.
        if table_is_aligned and leg_attachment_points is not None:
            # Post-alignment RGB localization assigns all four CAD corners to
            # the arm workspaces. Preserve that assignment: recomputing only
            # the root-nearest pair can send the left arm across the torso if
            # the tabletop translated while it was being rotated.
            flip_left = left_leg.copy()
        else:
            aligned_legs = [center + align @ (point - center) for point in all_legs]
            near_legs = sorted(
                aligned_legs,
                key=lambda point: float(np.dot(point - center, approach_axis)),
            )[:2]
            flip_left = max(
                near_legs,
                key=lambda point: float(np.dot(point - center, robot_left)),
            )
        flip_left[2] = leg_height
        flip_left_wrist = wrist(flip_left, initial_tool)
        flip_left_pre = flip_left_wrist - initial_tool[:, 0] * approach_clearance_m
        self.first_roll_grasp_center = flip_left.copy()
        # The left hand tips the assembly toward robot-left. The left tabletop
        # edge remains supported by the workbench while the former right edge
        # rises for the right-hand handover.
        pivot = center + aligned_short * (0.5 * self.TABLE_DEPTH_M)

        def supported_left_roll(angle: float) -> tuple[np.ndarray, np.ndarray]:
            """Roll the grasp counterclockwise toward robot-left."""

            rotation = _rotation_about(aligned_long, angle)
            tool = rotation @ initial_tool
            position = pivot + rotation @ (flip_left_wrist - pivot)
            return position, tool

        frames: list[_Keyframe] = []

        def add(phase: Phase, duration: float, lp: np.ndarray, lr: np.ndarray, rp: np.ndarray,
                rr: np.ndarray, lh: float, rh: float) -> None:
            frames.append(_Keyframe(phase, duration, lp.copy(), lr.copy(), rp.copy(), rr.copy(), lh, rh))

        add(Phase.CLEARANCE_STAGING, 0.4, left_safe, initial_tool,
            right_safe, initial_tool, -1, -1)
        add(Phase.CLEARANCE_STAGING, 1.4, left_outside, initial_tool,
            right_outside, initial_tool, -1, -1)
        add(Phase.CLEARANCE_STAGING, 1.4, left_high_wrist, left_target_tool,
            right_high_wrist, right_target_tool, -1, -1)
        add(Phase.ALIGN_APPROACH, 1.2, left_pre, left_target_tool,
            right_pre, right_target_tool, -1, -1)
        add(Phase.ALIGN_GRASP, 1.6, left_contact, left_target_tool,
            right_contact, right_target_tool, -1, -1)
        add(Phase.ALIGN_GRASP, 1.2, left_contact, left_target_tool,
            right_contact, right_target_tool, 1, 1)

        if table_is_aligned:
            # This planner is installed after head-RGB relocalization. Its
            # earlier phases are skipped; the final alignment frames provide
            # the left-leg target used by the encoder-verified regrasp gate.
            add(Phase.ALIGN_SHORT_EDGE, 0.8, left_safe, initial_tool,
                right_safe, initial_tool, -1, -1)
            add(Phase.ALIGN_SHORT_EDGE, 1.0, flip_left_pre, initial_tool,
                right_safe, initial_tool, -1, -1)
            add(Phase.ALIGN_SHORT_EDGE, 1.0, flip_left_wrist, initial_tool,
                right_safe, initial_tool, -1, -1)
            add(Phase.ALIGN_SHORT_EDGE, 0.8, flip_left_wrist, initial_tool,
                right_safe, initial_tool, 1, -1)
        else:
            pivot_tool = initial_tool
            moving_tool = edge_tool
            pivot_world = pivot_leg.copy()
            moving_world = edge_grasp.copy()
            pivot_wrist = pivot_contact.copy()
            moving_wrist = edge_contact.copy()

            def add_compass_pose(
                duration: float,
                angle: float,
                pivot_point: np.ndarray,
                moving_point: np.ndarray,
                pivot_hand: float,
                moving_hand: float,
            ) -> None:
                rotation = _rotation_about(up, angle)
                pivot_rotation = rotation @ pivot_tool
                moving_rotation = rotation @ moving_tool
                rotated_moving = pivot_point + rotation @ (moving_point - pivot_point)
                left_position, right_position = side_pair(
                    wrist(pivot_point, pivot_rotation),
                    wrist(rotated_moving, moving_rotation),
                )
                left_rotation, right_rotation = side_pair(
                    pivot_rotation, moving_rotation
                )
                left_hand, right_hand = side_commands(pivot_hand, moving_hand)
                add(
                    Phase.ALIGN_SHORT_EDGE,
                    duration,
                    left_position,
                    left_rotation,
                    right_position,
                    right_rotation,
                    left_hand,
                    right_hand,
                )

            if self.alignment_mode == "short_edge_pull":
                for angle in np.linspace(0.0, yaw_delta, 4)[1:]:
                    add_compass_pose(
                        0.9,
                        float(angle),
                        pivot_world,
                        moving_world,
                        1.0,
                        1.0,
                    )
                final_rotation = _rotation_about(up, yaw_delta)
                final_pivot_tool = final_rotation @ pivot_tool
                final_moving_tool = final_rotation @ moving_tool
                final_moving = pivot_world + final_rotation @ (
                    moving_world - pivot_world
                )
                final_left, final_right = side_pair(
                    wrist(pivot_world, final_pivot_tool),
                    wrist(final_moving, final_moving_tool),
                )
                final_left_rot, final_right_rot = side_pair(
                    final_pivot_tool, final_moving_tool
                )
                left_hand, right_hand = side_commands(1.0, -1.0)
                add(Phase.ALIGN_SHORT_EDGE, 0.5, final_left, final_left_rot,
                    final_right, final_right_rot, left_hand, right_hand)
            else:
                remaining = math.copysign(math.radians(30.0), yaw_delta)
                push_delta = yaw_delta - remaining
                for angle in np.linspace(0.0, push_delta, 4)[1:]:
                    add_compass_pose(
                        0.8,
                        float(angle),
                        pivot_world,
                        moving_world,
                        1.0,
                        1.0,
                    )

                push_rotation = _rotation_about(up, push_delta)
                pushed_long = push_rotation @ long_axis
                pushed_short = push_rotation @ short_axis
                pushed_center = pivot_world + push_rotation @ (center - pivot_world)
                pushed_moving = pivot_world + push_rotation @ (
                    moving_world - pivot_world
                )
                pushed_pivot_tool = push_rotation @ pivot_tool
                pushed_moving_tool = push_rotation @ moving_tool
                pushed_left, pushed_right = side_pair(
                    wrist(pivot_world, pushed_pivot_tool),
                    wrist(pushed_moving, pushed_moving_tool),
                )
                pushed_left_rot, pushed_right_rot = side_pair(
                    pushed_pivot_tool, pushed_moving_tool
                )
                left_hand, right_hand = side_commands(1.0, -1.0)
                add(Phase.ALIGN_SHORT_EDGE, 0.5, pushed_left, pushed_left_rot,
                    pushed_right, pushed_right_rot, left_hand, right_hand)

                lateral_error = float(np.dot(pushed_center, robot_left))
                recenter = -robot_left * float(np.clip(lateral_error, -0.18, 0.18))
                forward_distance = float(np.dot(pushed_center, approach_axis))
                if forward_distance > 0.66:
                    recenter -= approach_axis * min(forward_distance - 0.66, 0.08)
                recentered_pivot = pivot_world + recenter
                recentered_center = pushed_center + recenter
                recentered_edge = nearest_edge_grasp(
                    recentered_center, pushed_long, pushed_short, moving_side
                )
                recentered_pivot_wrist = wrist(
                    recentered_pivot, pushed_pivot_tool
                )
                recentered_edge_wrist = wrist(
                    recentered_edge, pushed_moving_tool
                )
                recentered_edge_pre = (
                    recentered_edge_wrist
                    - pushed_moving_tool[:, 0] * approach_clearance_m
                )
                left_position, right_position = side_pair(
                    recentered_pivot_wrist, recentered_edge_pre
                )
                left_rotation, right_rotation = side_pair(
                    pushed_pivot_tool, pushed_moving_tool
                )
                add(Phase.ALIGN_SHORT_EDGE, 1.3, left_position, left_rotation,
                    right_position, right_rotation, left_hand, right_hand)
                left_position, right_position = side_pair(
                    recentered_pivot_wrist, recentered_edge_wrist
                )
                add(Phase.ALIGN_SHORT_EDGE, 1.0, left_position, left_rotation,
                    right_position, right_rotation, left_hand, right_hand)
                left_hand, right_hand = side_commands(1.0, 1.0)
                add(Phase.ALIGN_SHORT_EDGE, 0.8, left_position, left_rotation,
                    right_position, right_rotation, left_hand, right_hand)

                for angle in np.linspace(0.0, remaining, 4)[1:]:
                    rotation = _rotation_about(up, float(angle))
                    pivot_rotation = rotation @ pushed_pivot_tool
                    moving_rotation = rotation @ pushed_moving_tool
                    moving_point = recentered_pivot + rotation @ (
                        recentered_edge - recentered_pivot
                    )
                    left_position, right_position = side_pair(
                        wrist(recentered_pivot, pivot_rotation),
                        wrist(moving_point, moving_rotation),
                    )
                    left_rotation, right_rotation = side_pair(
                        pivot_rotation, moving_rotation
                    )
                    add(Phase.ALIGN_SHORT_EDGE, 0.8, left_position, left_rotation,
                        right_position, right_rotation, left_hand, right_hand)
                left_hand, right_hand = side_commands(-1.0, -1.0)

            add(Phase.ALIGN_SHORT_EDGE, 0.6, left_safe, initial_tool,
                right_safe, initial_tool, -1.0, -1.0)
        for angle in np.linspace(0.0, math.pi / 2, 7)[1:]:
            left_arc, left_tool = supported_left_roll(float(angle))
            add(Phase.LEFT_LEG_FLIP_90, 1.20, left_arc, left_tool,
                right_safe, aligned_tool, 1, -1)

        left_90, left_tool_90 = supported_left_roll(math.pi / 2)
        add(Phase.LEFT_LEG_FLIP_90, 1.0, left_90, left_tool_90,
            right_safe, aligned_tool, 1, -1)
        # Once the left-hand roll has raised the tabletop to vertical, the
        # former left lateral edge is the upper edge.  The real procedure uses
        # the right hand on that tabletop edge, not on the low leg next to the
        # workbench.  Approach a point on the robot-side half of the edge.
        upper_edge_base = (
            center
            + aligned_long * 0.15
            - aligned_short * (0.5 * self.TABLE_DEPTH_M)
        )
        upper_edge_rotation = _rotation_about(aligned_long, math.pi / 2)
        upper_edge_grasp = pivot + upper_edge_rotation @ (upper_edge_base - pivot)
        upper_edge_tool = upper_edge_rotation @ aligned_tool
        upper_edge_wrist = wrist(upper_edge_grasp, upper_edge_tool)
        add(Phase.RIGHT_PREGRASP, 0.4, left_90, left_tool_90,
            upper_edge_wrist - upper_edge_tool[:, 0] * 0.08,
            upper_edge_tool, 1, -1)
        add(Phase.RIGHT_GRASP, 0.5, left_90, left_tool_90,
            upper_edge_wrist, upper_edge_tool, 1, 1)
        add(Phase.HANDOVER, 0.4, left_90, left_tool_90,
            upper_edge_wrist, upper_edge_tool, 1, 1)
        add(Phase.HANDOVER, 0.5, left_90 - aligned_long * 0.10, left_tool_90,
            upper_edge_wrist, upper_edge_tool, -1, 1)
        # Push the vertical tabletop beyond its balance point.  Gravity then
        # completes the second 90-degree roll, avoiding an unreachable wrist
        # arc all the way to the workbench surface.
        for fraction in np.linspace(0.0, 1.0, 7)[1:]:
            right_arc = upper_edge_wrist + aligned_short * (0.12 * float(fraction))
            add(Phase.RIGHT_TOP_FLIP_90, 0.55, left_safe, aligned_tool,
                right_arc, upper_edge_tool, -1, 1)
        add(Phase.SETTLE_AND_RETREAT, 1.0, left_safe, aligned_tool,
            right_arc, upper_edge_tool, -1, -1)
        add(Phase.SETTLE_AND_RETREAT, 2.5, left_safe, aligned_tool,
            right_safe, aligned_tool, -1, -1)
        return frames
