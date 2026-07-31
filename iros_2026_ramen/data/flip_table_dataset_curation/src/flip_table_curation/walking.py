from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BODY_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class StepDetection:
    walked: bool
    step_count: int
    left_step_count: int
    right_step_count: int
    maximum_contact_displacement_m: float
    contact_segments: dict[str, list[dict[str, float | int | list[float]]]]


def _quaternion_wxyz_matrix(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid root quaternion")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class FootKinematics:
    """Reusable G1 foot FK; loading Pinocchio once keeps the full audit practical."""

    def __init__(self, urdf_path: Path) -> None:
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        missing = [
            name for name in BODY_JOINT_ORDER if not self.model.existJointName(name)
        ]
        frames = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
        missing_frames = [
            name for name in frames.values() if not self.model.existFrame(name)
        ]
        if missing or missing_frames:
            raise ValueError(
                "URDF contract mismatch: "
                f"missing_joints={missing}, missing_frames={missing_frames}"
            )
        self.joint_q = np.asarray(
            [
                self.model.joints[self.model.getJointId(name)].idx_q
                for name in BODY_JOINT_ORDER
            ],
            dtype=np.int64,
        )
        self.frame_ids = {
            side: self.model.getFrameId(name) for side, name in frames.items()
        }

    def positions(self, robot_q: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(robot_q, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 36
            or not np.isfinite(values).all()
        ):
            raise ValueError("robot_q must be finite [T,36]")
        output = {
            side: np.empty((len(values), 3), dtype=np.float64)
            for side in self.frame_ids
        }
        q = np.zeros(self.model.nq, dtype=np.float64)
        for index, source in enumerate(values):
            q.fill(0.0)
            q[self.joint_q] = source[7:]
            self.pin.framesForwardKinematics(self.model, self.data, q)
            root_rotation = _quaternion_wxyz_matrix(source[3:7])
            for side, frame_id in self.frame_ids.items():
                local = np.asarray(
                    self.data.oMf[frame_id].translation, dtype=np.float64
                )
                output[side][index] = source[:3] + root_rotation @ local
        return output


def foot_world_positions(robot_q: np.ndarray, urdf_path: Path) -> dict[str, np.ndarray]:
    return FootKinematics(urdf_path).positions(robot_q)


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.median(padded[index : index + window], axis=0) for index in range(len(values))]
    )


def _runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


def detect_steps(
    foot_positions: dict[str, np.ndarray],
    *,
    fps: float,
    median_window: int,
    floor_tolerance_m: float,
    maximum_contact_speed_m_s: float,
    minimum_contact_seconds: float,
    step_displacement_m: float,
) -> StepDetection:
    segments: dict[str, list[dict[str, float | int | list[float]]]] = {}
    side_steps: dict[str, int] = {}
    maximum = 0.0
    minimum_frames = max(1, int(np.ceil(minimum_contact_seconds * fps)))
    for side in ("left", "right"):
        position = _median_filter(np.asarray(foot_positions[side], dtype=np.float64), median_window)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(f"{side} foot position must be [T,3]")
        velocity = np.linalg.norm(
            np.diff(position, axis=0, prepend=position[:1]), axis=1
        ) * fps
        floor = float(np.percentile(position[:, 2], 5.0))
        contact = (position[:, 2] <= floor + floor_tolerance_m) & (
            velocity <= maximum_contact_speed_m_s
        )
        entries: list[dict[str, float | int | list[float]]] = []
        centroids: list[np.ndarray] = []
        for start, end in _runs(contact, minimum_frames):
            centroid = np.median(position[start:end, :2], axis=0)
            centroids.append(centroid)
            entries.append(
                {
                    "start": start,
                    "end": end,
                    "duration_s": (end - start) / fps,
                    "centroid_xy_m": centroid.tolist(),
                }
            )
        count = 0
        for previous, current in zip(centroids, centroids[1:], strict=False):
            displacement = float(np.linalg.norm(current - previous))
            maximum = max(maximum, displacement)
            if displacement >= step_displacement_m:
                count += 1
        segments[side] = entries
        side_steps[side] = count
    total = side_steps["left"] + side_steps["right"]
    return StepDetection(
        walked=total > 0,
        step_count=total,
        left_step_count=side_steps["left"],
        right_step_count=side_steps["right"],
        maximum_contact_displacement_m=maximum,
        contact_segments=segments,
    )
