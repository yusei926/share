"""Backend-neutral command and observation messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


MESSAGE_SCHEMA_VERSION = "team_ramen_flip_table_teleop_message/v1"


class ControlMode(str, Enum):
    IDLE = "idle"
    TRACK = "track"


class ControlEvent(str, Enum):
    NONE = "none"
    RECORD_TOGGLE = "record_toggle"
    DISCARD_RESET = "discard_reset"
    QUIT = "quit"


def _vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain {width} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class ArmHandTarget:
    sequence: int
    monotonic_ns: int
    mode: ControlMode
    event: ControlEvent
    arm_position_rad: tuple[float, ...]
    dex1_opening_fraction: tuple[float, float]

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.monotonic_ns <= 0:
            raise ValueError("command sequence/time must be non-negative and positive")
        if len(self.arm_position_rad) != 14 or not all(map(math.isfinite, self.arm_position_rad)):
            raise ValueError("arm_position_rad must be finite 14-D")
        if len(self.dex1_opening_fraction) != 2 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.dex1_opening_fraction
        ):
            raise ValueError("Dex1 opening targets must be finite left/right fractions in [0,1]")

    def to_message(self) -> dict[str, Any]:
        return {
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "type": "command",
            "sequence": self.sequence,
            "monotonic_ns": self.monotonic_ns,
            "mode": self.mode.value,
            "event": self.event.value,
            "arm_position_rad": list(self.arm_position_rad),
            "dex1_opening_fraction": list(self.dex1_opening_fraction),
        }

    @classmethod
    def from_message(cls, value: Mapping[str, Any]) -> "ArmHandTarget":
        if value.get("schema_version") != MESSAGE_SCHEMA_VERSION or value.get("type") != "command":
            raise ValueError("unsupported command message")
        return cls(
            sequence=int(value["sequence"]),
            monotonic_ns=int(value["monotonic_ns"]),
            mode=ControlMode(value["mode"]),
            event=ControlEvent(value["event"]),
            arm_position_rad=_vector(value["arm_position_rad"], 14, "arm_position_rad"),
            dex1_opening_fraction=_vector(
                value["dex1_opening_fraction"], 2, "dex1_opening_fraction"
            ),
        )


@dataclass(frozen=True)
class TeleopObservation:
    sequence: int
    capture_monotonic_ns: int
    backend: str
    body_joint_position_rad: tuple[float, ...]
    body_joint_velocity_rad_s: tuple[float, ...]
    dex1_opening_fraction: tuple[float, float]
    applied_arm_target_rad: tuple[float, ...]
    applied_dex1_opening_target: tuple[float, float]
    root_pose_xyzw: tuple[float, ...]
    camera_capture_monotonic_ns: Mapping[str, int]
    camera_jpeg: Mapping[str, bytes]
    diagnostic_camera_capture_monotonic_ns: Mapping[str, int] = field(default_factory=dict)
    diagnostic_camera_jpeg: Mapping[str, bytes] = field(default_factory=dict)
    success: bool | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.capture_monotonic_ns <= 0:
            raise ValueError("observation sequence/time is invalid")
        if self.backend not in {"sim", "real"}:
            raise ValueError("backend must be sim or real")
        if len(self.body_joint_position_rad) != 29 or not all(
            map(math.isfinite, self.body_joint_position_rad)
        ):
            raise ValueError("body joint position must be finite 29-D")
        if len(self.body_joint_velocity_rad_s) != 29 or not all(
            map(math.isfinite, self.body_joint_velocity_rad_s)
        ):
            raise ValueError("body joint velocity must be finite 29-D")
        if len(self.root_pose_xyzw) != 7 or not all(map(math.isfinite, self.root_pose_xyzw)):
            raise ValueError("root pose must be finite xyz+xyzw")
        if not math.isclose(
            sum(value * value for value in self.root_pose_xyzw[3:]), 1.0, abs_tol=1e-4
        ):
            raise ValueError("root quaternion must be unit length")
        if len(self.dex1_opening_fraction) != 2 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.dex1_opening_fraction
        ):
            raise ValueError("Dex1 state must be finite left/right opening fractions")
        if len(self.applied_arm_target_rad) != 14 or not all(
            map(math.isfinite, self.applied_arm_target_rad)
        ):
            raise ValueError("applied arm target must be finite 14-D")
        if len(self.applied_dex1_opening_target) != 2 or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.applied_dex1_opening_target
        ):
            raise ValueError("applied Dex1 target must contain two opening fractions")
        camera_roles = set(self.camera_jpeg)
        allowed_roles = (
            {"head_left", "head_right"},
            {"head_left", "head_right", "left_wrist", "right_wrist"},
        )
        if camera_roles not in allowed_roles or set(self.camera_capture_monotonic_ns) != camera_roles:
            raise ValueError(
                "observation must contain true head stereo, optionally with both wrist cameras"
            )
        if any(not isinstance(value, bytes) or not value for value in self.camera_jpeg.values()):
            raise ValueError("camera payloads must be non-empty JPEG bytes")
        if set(self.diagnostic_camera_jpeg) != set(
            self.diagnostic_camera_capture_monotonic_ns
        ):
            raise ValueError("diagnostic camera payloads and timestamps differ")
        if set(self.diagnostic_camera_jpeg) - {"global"}:
            raise ValueError("global is the only supported diagnostic camera")
        if any(
            not isinstance(value, bytes) or not value
            for value in self.diagnostic_camera_jpeg.values()
        ):
            raise ValueError("diagnostic camera payloads must be non-empty JPEG bytes")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("success must be bool or null")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

    @property
    def arm_joint_position_rad(self) -> tuple[float, ...]:
        return self.body_joint_position_rad[15:29]

    @property
    def arm_joint_velocity_rad_s(self) -> tuple[float, ...]:
        return self.body_joint_velocity_rad_s[15:29]

    def to_message(self) -> dict[str, Any]:
        return {
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "type": "observation",
            "sequence": self.sequence,
            "capture_monotonic_ns": self.capture_monotonic_ns,
            "backend": self.backend,
            "body_joint_position_rad": list(self.body_joint_position_rad),
            "body_joint_velocity_rad_s": list(self.body_joint_velocity_rad_s),
            "dex1_opening_fraction": list(self.dex1_opening_fraction),
            "applied_arm_target_rad": list(self.applied_arm_target_rad),
            "applied_dex1_opening_target": list(self.applied_dex1_opening_target),
            "root_pose_xyzw": list(self.root_pose_xyzw),
            "camera_capture_monotonic_ns": dict(self.camera_capture_monotonic_ns),
            "camera_jpeg": dict(self.camera_jpeg),
            "diagnostic_camera_capture_monotonic_ns": dict(
                self.diagnostic_camera_capture_monotonic_ns
            ),
            "diagnostic_camera_jpeg": dict(self.diagnostic_camera_jpeg),
            "success": self.success,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_message(cls, value: Mapping[str, Any]) -> "TeleopObservation":
        if value.get("schema_version") != MESSAGE_SCHEMA_VERSION or value.get("type") != "observation":
            raise ValueError("unsupported observation message")
        camera_jpeg = value["camera_jpeg"]
        if not isinstance(camera_jpeg, Mapping):
            raise ValueError("camera_jpeg must be an object")
        return cls(
            sequence=int(value["sequence"]),
            capture_monotonic_ns=int(value["capture_monotonic_ns"]),
            backend=str(value["backend"]),
            body_joint_position_rad=_vector(
                value["body_joint_position_rad"], 29, "body_joint_position_rad"
            ),
            body_joint_velocity_rad_s=_vector(
                value["body_joint_velocity_rad_s"], 29, "body_joint_velocity_rad_s"
            ),
            dex1_opening_fraction=_vector(
                value["dex1_opening_fraction"], 2, "dex1_opening_fraction"
            ),
            applied_arm_target_rad=_vector(
                value["applied_arm_target_rad"], 14, "applied_arm_target_rad"
            ),
            applied_dex1_opening_target=_vector(
                value["applied_dex1_opening_target"], 2, "applied_dex1_opening_target"
            ),
            root_pose_xyzw=_vector(value["root_pose_xyzw"], 7, "root_pose_xyzw"),
            camera_capture_monotonic_ns={
                str(key): int(item) for key, item in value["camera_capture_monotonic_ns"].items()
            },
            camera_jpeg={str(key): bytes(item) for key, item in camera_jpeg.items()},
            diagnostic_camera_capture_monotonic_ns={
                str(key): int(item)
                for key, item in value.get(
                    "diagnostic_camera_capture_monotonic_ns", {}
                ).items()
            },
            diagnostic_camera_jpeg={
                str(key): bytes(item)
                for key, item in value.get("diagnostic_camera_jpeg", {}).items()
            },
            success=value.get("success"),
            diagnostics=dict(value.get("diagnostics", {})),
        )
