"""Pure contracts for the flip-table real-to-simulation calibration workflow.

The dataset stores calibrated image streams and robot labels but not a table
pose, contact forces, or a robot-to-camera TF tree.  This module deliberately
keeps recorded facts, inferred quantities, and simulator-only diagnostics
separate so an apparently good replay cannot hide a non-physical shortcut.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "team_ramen_flip_table_real_to_sim_calibration/v1"
SOURCE_REVISION = "10a6ec05f9993b8d59faad2957e47153b0f15f37"
SOURCE_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1"
SOURCE_FPS = 30.0
SOURCE_ROOT_DIM = 7
SOURCE_BODY_DIM = 29
SOURCE_Q_DIM = SOURCE_ROOT_DIM + SOURCE_BODY_DIM
SOURCE_HAND_DIM = 2
SOURCE_EEF_DIM = 12
SOURCE_EEF_ORDER = ("left", "right")
SOURCE_EEF_POSE_FORMAT = "xyz_euler_xyz_rad"
SOURCE_EEF_REFERENCE_FRAME = "robot_root"
UPPER_BODY_SLICE = slice(19, 36)
POLICY_CAMERA_KEYS = (
    "observation.images.cam_0",
    "observation.images.cam_2",
    "observation.images.cam_3",
)
CALIBRATION_CAMERA_KEYS = (
    "observation.images.cam_0",
    "observation.images.cam_1",
    "observation.images.cam_2",
    "observation.images.cam_3",
)
CAMERA_ROLES = {
    "observation.images.cam_0": "head_left",
    "observation.images.cam_1": "head_right",
    "observation.images.cam_2": "left_wrist",
    "observation.images.cam_3": "right_wrist",
}


@dataclass(frozen=True)
class EpisodeSignals:
    """Numerical activity and duration used for deterministic episode selection."""

    episode_index: int
    frames: int
    duration_s: float
    upper_body_path_length_rad: float
    upper_body_peak_to_peak_rad: float
    hand_path_length: float
    timestamp_jitter_s: float

    @property
    def score(self) -> float:
        # Favor complete, dynamic motions but avoid selecting a pathological
        # timestamp sequence merely because it contains very large targets.
        return (
            math.log1p(self.duration_s)
            + 0.45 * math.log1p(self.upper_body_path_length_rad)
            + 0.25 * math.log1p(self.upper_body_peak_to_peak_rad)
            + 0.10 * math.log1p(self.hand_path_length)
            - 20.0 * self.timestamp_jitter_s
        )

    def json(self) -> dict[str, float | int]:
        record = asdict(self)
        record["selection_score"] = self.score
        return record


@dataclass(frozen=True)
class EpisodeSelection:
    anchor: int
    calibration: tuple[int, int]
    validation: tuple[int, int, int, int, int]

    def all_indices(self) -> tuple[int, ...]:
        return (self.anchor, *self.calibration, *self.validation)

    def json(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "calibration": list(self.calibration),
            "validation": list(self.validation),
            "all_unique": len(set(self.all_indices())) == len(self.all_indices()),
        }


@dataclass(frozen=True)
class JointReplayMetrics:
    samples: int
    upper_body_rmse_rad: float
    lower_body_rmse_rad: float | None
    upper_body_p95_abs_error_rad: float
    lower_body_p95_abs_error_rad: float | None
    source_hz: float
    simulator_hz: float

    def passed(self) -> bool:
        return (
            self.upper_body_rmse_rad <= 0.03
            and self.lower_body_rmse_rad is not None
            and self.lower_body_rmse_rad <= 0.05
            and self.upper_body_p95_abs_error_rad <= 0.08
            and self.lower_body_p95_abs_error_rad is not None
            and self.lower_body_p95_abs_error_rad <= 0.12
        )

    def json(self) -> dict[str, object]:
        value = asdict(self)
        value["passed"] = self.passed()
        value["thresholds"] = {
            "upper_body_rmse_rad": 0.03,
            "lower_body_rmse_rad": 0.05,
            "upper_body_p95_abs_error_rad": 0.08,
            "lower_body_p95_abs_error_rad": 0.12,
        }
        return value


def finite_matrix(value: Any, width: int, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite [T,{width}] matrix, got {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{label} must contain at least two frames")
    return matrix


def source_19d_actions(robot_q_desired: Any, hand_cmd: Any) -> np.ndarray:
    """Return the only 19-D command that the fixed-base evaluation may replay."""

    q = finite_matrix(robot_q_desired, SOURCE_Q_DIM, "robot_q_desired")
    hands = finite_matrix(hand_cmd, SOURCE_HAND_DIM, "hand_cmd")
    if len(q) != len(hands):
        raise ValueError("robot_q_desired and hand_cmd must have equal frame counts")
    if np.any(hands < 0.0) or np.any(hands > 4.5):
        raise ValueError("hand_cmd is outside the recorded [0,4.5] range")
    return np.concatenate((q[:, UPPER_BODY_SLICE], hands), axis=1)


def source_19d_observation(robot_q_current: Any, hand_state: Any) -> np.ndarray:
    """Return the measured 19-D upper-body state at each camera timestamp.

    ``hand_state`` is retained verbatim rather than clipped to the command
    interval: a small amount of encoder overshoot is an observed calibration
    fact, while conversion to a simulator-valid finger position happens at the
    scene boundary.  This is deliberately distinct from
    :func:`source_19d_actions`, whose hand command is contractually in
    ``[0, 4.5]``.
    """

    q = finite_matrix(robot_q_current, SOURCE_Q_DIM, "robot_q_current")
    hands = finite_matrix(hand_state, SOURCE_HAND_DIM, "hand_state")
    if len(q) != len(hands):
        raise ValueError("robot_q_current and hand_state must have equal frame counts")
    return np.concatenate((q[:, UPPER_BODY_SLICE], hands), axis=1)


def episode_signals(
    episode_index: int,
    timestamps: Any,
    robot_q_desired: Any,
    hand_cmd: Any,
) -> EpisodeSignals:
    time = np.asarray(timestamps, dtype=np.float64)
    q = finite_matrix(robot_q_desired, SOURCE_Q_DIM, "robot_q_desired")
    hands = finite_matrix(hand_cmd, SOURCE_HAND_DIM, "hand_cmd")
    if time.shape != (len(q),) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be finite, strictly increasing, and match actions")
    if hands.shape[0] != len(q):
        raise ValueError("hand_cmd must match robot_q_desired frames")
    deltas = np.diff(time)
    expected = 1.0 / SOURCE_FPS
    upper = q[:, UPPER_BODY_SLICE]
    return EpisodeSignals(
        episode_index=int(episode_index),
        frames=int(len(q)),
        duration_s=float(time[-1] - time[0] + expected),
        upper_body_path_length_rad=float(np.abs(np.diff(upper, axis=0)).sum()),
        upper_body_peak_to_peak_rad=float(np.ptp(upper, axis=0).max()),
        hand_path_length=float(np.abs(np.diff(hands, axis=0)).sum()),
        timestamp_jitter_s=float(np.max(np.abs(deltas - expected))),
    )


def select_episode_roles(
    signals: Sequence[EpisodeSignals], *, eligible_episode_indices: Iterable[int] | None = None
) -> EpisodeSelection:
    """Pick one representative fit episode and disjoint stratified episodes.

    This uses numeric activity plus an optional source EEF/FK eligibility
    filter. The CLI additionally records visual samples for human and
    automated review; it never claims that this score proves visibility or
    physical success.
    """

    if len(signals) < 8:
        raise ValueError("at least eight episodes are required for anchor/calibration/validation")
    eligible = None if eligible_episode_indices is None else {int(value) for value in eligible_episode_indices}
    candidates = (
        list(signals)
        if eligible is None
        else [item for item in signals if item.episode_index in eligible]
    )
    if len(candidates) < 8:
        raise ValueError(
            "at least eight EEF/FK-eligible episodes are required for anchor/calibration/validation"
        )
    ranked = sorted(candidates, key=lambda item: (item.score, -item.episode_index), reverse=True)
    eligible = ranked[: max(8, math.ceil(len(ranked) * 0.35))]
    # A median high-quality episode is less likely to be an extreme motion than
    # the top score, while still excluding very short/no-op source segments.
    anchor = eligible[len(eligible) // 2]
    remaining = [item for item in ranked if item.episode_index != anchor.episode_index]
    calibration = (remaining[0], remaining[len(remaining) // 2])
    excluded = {anchor.episode_index, *(item.episode_index for item in calibration)}
    residual = [item for item in ranked if item.episode_index not in excluded]
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    validation = tuple(residual[min(len(residual) - 1, round(q * (len(residual) - 1)))] for q in quantiles)
    if len({item.episode_index for item in validation}) != 5:
        # Small synthetic fixtures can make quantiles collide; use stable order.
        validation = tuple(residual[index] for index in range(5))
    result = EpisodeSelection(
        anchor=anchor.episode_index,
        calibration=(calibration[0].episode_index, calibration[1].episode_index),
        validation=tuple(item.episode_index for item in validation),
    )
    if len(set(result.all_indices())) != 8:
        raise AssertionError("episode selection must be disjoint")
    return result


def compute_joint_replay_metrics(
    target_19d: Any,
    actual_19d: Any,
    *,
    source_hz: float = SOURCE_FPS,
    simulator_hz: float = 50.0,
) -> JointReplayMetrics:
    target = finite_matrix(target_19d, 19, "target_19d")
    actual = finite_matrix(actual_19d, 19, "actual_19d")
    if target.shape != actual.shape:
        raise ValueError("target_19d and actual_19d must have equal shapes")
    if source_hz <= 0.0 or simulator_hz <= 0.0:
        raise ValueError("source_hz and simulator_hz must be positive")
    errors = actual[:, :17] - target[:, :17]
    # The production calibration contract has no lower-body command channel.
    # Keep it unavailable rather than treating an absent signal as a success.
    return JointReplayMetrics(
        samples=len(target),
        upper_body_rmse_rad=float(np.sqrt(np.mean(errors * errors))),
        lower_body_rmse_rad=None,
        upper_body_p95_abs_error_rad=float(np.quantile(np.abs(errors), 0.95)),
        lower_body_p95_abs_error_rad=None,
        source_hz=float(source_hz),
        simulator_hz=float(simulator_hz),
    )


def compare_metric_gate(value: Mapping[str, Any]) -> dict[str, object]:
    """Gate only metrics actually present; unavailable real measurements fail closed."""

    required = {
        "camera_reprojection_median_px": 3.0,
        "camera_reprojection_p95_px": 8.0,
        "table_translation_rmse_m": 0.020,
        "table_rotation_rmse_deg": 3.0,
        "phase_timing_max_error_s": 0.100,
        "mask_iou": 0.90,
    }
    outcomes: dict[str, dict[str, object]] = {}
    for name, threshold in required.items():
        raw = value.get(name)
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            outcomes[name] = {"status": "missing", "threshold": threshold}
            continue
        observed = float(raw)
        lower_is_better = name != "mask_iou"
        passed = observed <= threshold if lower_is_better else observed >= threshold
        outcomes[name] = {"status": "pass" if passed else "fail", "value": observed, "threshold": threshold}
    return {"passed": all(item["status"] == "pass" for item in outcomes.values()), "metrics": outcomes}
