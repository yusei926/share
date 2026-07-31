"""Collision-aware arm-only staging before real policy inference.

The waypoints intentionally contain only the fourteen G1 arm joints. The
floating base, waist, and legs remain owned by Unitree Regular Mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Joint order:
# left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw,
# right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
#
# Positive shoulder pitch moves the arms rearward in the G1 joint convention.
# Only the two shoulder-pitch joints are changed in the first stage; every
# other joint is resolved from the measured starting pose. This prevents an
# elbow or wrist from sweeping forward before the shoulders clear the table.
SHOULDER_PITCH_BACKWARD_RAD = 0.85
SHOULDER_ROLL_LATERAL_RAD = 1.60
ELBOW_OUTWARD_CLEARANCE_RAD = 0.40
_PRESERVE_EXCEPT_SHOULDER_PITCH = tuple(
    index for index in range(14) if index not in (0, 7)
)
_PRESERVE_EXCEPT_SHOULDER_PITCH_ROLL_ELBOW = tuple(
    index for index in range(14) if index not in (0, 1, 3, 7, 8, 10)
)
SHOULDER_BACKWARD_TEMPLATE_RAD = (
    SHOULDER_PITCH_BACKWARD_RAD,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    SHOULDER_PITCH_BACKWARD_RAD,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

# With the roots already well rearward, move the shoulder-roll joints outward
# while bending the elbows outward.  Official-model FK shows that increasing
# shoulder roll alone saturates the upper-arm lateral reach near 1.6 rad.  The
# positive 0.4-rad elbow target moves each palm about another 8 cm outward
# compared with a straight elbow, without increasing the rearward shoulder
# pitch. Yaw and wrists retain their measured values.
LATERAL_HIGH_ARM_POSE_RAD = (
    SHOULDER_PITCH_BACKWARD_RAD,
    SHOULDER_ROLL_LATERAL_RAD,
    0.0,
    ELBOW_OUTWARD_CLEARANCE_RAD,
    0.0,
    0.0,
    0.0,
    SHOULDER_PITCH_BACKWARD_RAD,
    -SHOULDER_ROLL_LATERAL_RAD,
    0.0,
    ELBOW_OUTWARD_CLEARANCE_RAD,
    0.0,
    0.0,
    0.0,
)

# Move forward while keeping the elbows laterally outside the table. Only
# after this waypoint converges may the shoulder roll close toward the final
# ready pose. Official-model FK places the left elbow at
# (0.041, 0.320, 0.316) m and wrist at (0.192, 0.376, 0.406) m.
FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD = (
    -0.55,
    SHOULDER_ROLL_LATERAL_RAD,
    0.0,
    ELBOW_OUTWARD_CLEARANCE_RAD,
    0.0,
    0.0,
    0.0,
    -0.55,
    -SHOULDER_ROLL_LATERAL_RAD,
    0.0,
    ELBOW_OUTWARD_CLEARANCE_RAD,
    0.0,
    0.0,
    0.0,
)

# Slightly lower forward ready pose. Compared with the previous
# shoulder-pitch=-0.60/elbow=-0.30 target, FK lowers the elbow by about 6 mm
# and the wrist by about 25 mm while retaining forward table clearance. The
# small outward roll keeps the forearms separated.
FORWARD_HIGH_ARM_POSE_RAD = (
    -0.55,
    0.15,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.0,
    -0.55,
    -0.15,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class ArmPreMotionWaypoint:
    name: str
    arm_position_rad: tuple[float, ...]
    preserve_initial_joint_indices: tuple[int, ...] = ()
    do_not_decrease_from_initial_indices: tuple[int, ...] = ()

    def as_array(self) -> np.ndarray:
        values = np.asarray(self.arm_position_rad, dtype=np.float64)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(f"pre-motion waypoint {self.name!r} must be finite 14-D")
        return values

    def resolve(self, initial_arm_position_rad: np.ndarray) -> np.ndarray:
        initial = np.asarray(initial_arm_position_rad, dtype=np.float64)
        if initial.shape != (14,) or not np.isfinite(initial).all():
            raise ValueError("initial arm pose must be finite 14-D")
        values = self.as_array().copy()
        for index in self.preserve_initial_joint_indices:
            if index < 0 or index >= 14:
                raise ValueError(
                    f"pre-motion waypoint {self.name!r} has invalid preserve index"
                )
            values[index] = initial[index]
        for index in self.do_not_decrease_from_initial_indices:
            if index < 0 or index >= 14:
                raise ValueError(
                    f"pre-motion waypoint {self.name!r} has invalid monotonic index"
                )
            values[index] = max(values[index], initial[index])
        return values


ARM_PRE_MOTION_WAYPOINTS = (
    ArmPreMotionWaypoint(
        "shoulder_pitch_backward_clearance",
        SHOULDER_BACKWARD_TEMPLATE_RAD,
        _PRESERVE_EXCEPT_SHOULDER_PITCH,
        (0, 7),
    ),
    ArmPreMotionWaypoint(
        "lateral_high_clearance",
        LATERAL_HIGH_ARM_POSE_RAD,
        _PRESERVE_EXCEPT_SHOULDER_PITCH_ROLL_ELBOW,
    ),
    ArmPreMotionWaypoint(
        "forward_outward_clearance",
        FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD,
    ),
    ArmPreMotionWaypoint("forward_high_ready", FORWARD_HIGH_ARM_POSE_RAD),
)


def build_arm_pre_motion_waypoints(
    dataset_frame0_arm_rad: tuple[float, ...] | np.ndarray,
) -> tuple[ArmPreMotionWaypoint, ...]:
    """Append the model's dataset frame-zero pose to the clearance path."""

    frame0 = np.asarray(dataset_frame0_arm_rad, dtype=np.float64)
    if frame0.shape != (14,) or not np.isfinite(frame0).all():
        raise ValueError("dataset frame-zero arm pose must be finite 14-D")
    return ARM_PRE_MOTION_WAYPOINTS + (
        ArmPreMotionWaypoint("dataset_frame0_pose", tuple(frame0.tolist())),
    )


def build_arm_return_waypoints(
    initial_arm_position_rad: tuple[float, ...] | np.ndarray,
    dataset_frame0_arm_rad: tuple[float, ...] | np.ndarray,
) -> tuple[ArmPreMotionWaypoint, ...]:
    """Build the exact reverse clearance path back to the measured start.

    The policy may have moved away from frame zero, so frame zero is the first
    return target. The remaining resolved startup path is then traversed in
    reverse before restoring the arm pose captured prior to arm_sdk takeover.
    """

    initial = np.asarray(initial_arm_position_rad, dtype=np.float64)
    if initial.shape != (14,) or not np.isfinite(initial).all():
        raise ValueError("initial arm pose must be finite 14-D")
    startup = build_arm_pre_motion_waypoints(dataset_frame0_arm_rad)
    resolved = [waypoint.resolve(initial) for waypoint in startup]
    return tuple(
        ArmPreMotionWaypoint(
            f"return_{waypoint.name}", tuple(target.tolist())
        )
        for waypoint, target in zip(reversed(startup), reversed(resolved), strict=True)
    ) + (
        ArmPreMotionWaypoint("return_measured_initial_pose", tuple(initial.tolist())),
    )


def validate_arm_pre_motion_waypoints(
    lower_rad: tuple[float, ...],
    upper_rad: tuple[float, ...],
    waypoints: tuple[ArmPreMotionWaypoint, ...] = ARM_PRE_MOTION_WAYPOINTS,
) -> None:
    """Fail before actuation if a waypoint violates the hardware margins."""

    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    if (
        lower.shape != (14,)
        or upper.shape != (14,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("arm safety bounds must be finite ordered 14-D vectors")
    if not waypoints:
        raise ValueError("arm waypoint sequence must not be empty")
    for waypoint in waypoints:
        values = waypoint.as_array()
        outside = (values < lower) | (values > upper)
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"pre-motion waypoint {waypoint.name!r} joint {index}="
                f"{values[index]:.4f} violates "
                f"[{lower[index]:.4f},{upper[index]:.4f}]"
            )
