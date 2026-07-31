"""Event-based progress labels for flip-table demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MILESTONE_NAMES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
MILESTONE_DESCRIPTIONS = {
    "M0": "active motion begins",
    "M1": "first support-leg grasp",
    "M2": "bimanual load transfer",
    "M3": "first rotation and support-hand release",
    "M4": "mid-flip regrasp",
    "M5": "second rotation and catch",
    "M6": "final stable state",
}


@dataclass(frozen=True)
class Milestone:
    frame: int | None
    confidence: float
    source: str

    @property
    def valid(self) -> bool:
        return self.frame is not None and self.confidence >= 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "confidence": self.confidence,
            "source": self.source,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class ProgressAnnotation:
    episode_index: int
    length: int
    primary_hand: str | None
    milestones: dict[str, Milestone]
    progress: list[float]
    progress_mask: list[bool]
    review_required: bool
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "flip_table_event_progress_v1",
            "episode_index": self.episode_index,
            "length": self.length,
            "primary_hand": self.primary_hand,
            "milestones": {
                name: self.milestones[name].as_dict() for name in MILESTONE_NAMES
            },
            "progress": self.progress,
            "progress_mask": self.progress_mask,
            "review_required": self.review_required,
            "diagnostics": self.diagnostics,
        }


def annotate_episode(
    *,
    episode_index: int,
    hand_cmd: Sequence[Sequence[float]],
    hand_state: Sequence[Sequence[float]],
    ee_action: Sequence[Sequence[float]],
    robot_q_desired: Sequence[Sequence[float]],
    visual_rotation: Sequence[float] | None = None,
    visual_confidence: Sequence[float] | None = None,
    stable_frames: int = 5,
) -> ProgressAnnotation:
    """Infer optional ordered milestones without forcing a fixed phase layout."""
    cmd = _matrix("hand_cmd", hand_cmd, columns=2)
    state = _matrix("hand_state", hand_state, columns=2)
    eef = _matrix("ee_action", ee_action, columns=12)
    joints = _matrix("robot_q_desired", robot_q_desired, columns=36)
    length = len(cmd)
    if not (len(state) == len(eef) == len(joints) == length):
        raise ValueError("all progress signals must have the same episode length")
    if length < 2:
        raise ValueError("progress annotation requires at least two frames")

    commanded_closed = _hysteresis_closed(cmd, stable_frames=stable_frames)
    measured_closed = _hysteresis_closed(state, stable_frames=stable_frames)
    closed = commanded_closed & measured_closed
    runs = _stable_runs(closed, minimum=stable_frames)
    motion = _motion_score(eef, joints)
    motion_threshold = max(float(np.quantile(motion, 0.25)), 0.08 * float(np.quantile(motion, 0.95)))
    m0_frame = _first_sustained(motion > motion_threshold, stable_frames)
    if m0_frame is None:
        m0_frame = 0

    m1_run = next((run for run in runs if sum(run[2]) == 1 and run[0] >= m0_frame), None)
    primary_index = None if m1_run is None else int(np.argmax(m1_run[2]))
    primary_hand = None if primary_index is None else ("left" if primary_index == 0 else "right")
    opposite_index = None if primary_index is None else 1 - primary_index

    m1 = _run_milestone(
        m1_run,
        confidence=0.85,
        source="hand_cmd+hand_state:first_stable_single",
    )
    m2_run = _first_run_after(runs, m1.frame, required=(True, True))
    m2 = _run_milestone(
        m2_run,
        confidence=0.85,
        source="hand_cmd+hand_state:first_stable_bimanual",
    )

    m3_run = None
    if opposite_index is not None:
        required = [False, False]
        required[opposite_index] = True
        m3_run = _first_run_after(runs, m2.frame, required=tuple(required))
    m3 = _run_milestone(
        m3_run,
        confidence=0.65,
        source="hand_cmd+hand_state:primary_release_after_bimanual",
    )

    m4_run = _first_run_after(runs, m3.frame, required=(True, True))
    m4 = _run_milestone(
        m4_run,
        confidence=0.75,
        source="hand_cmd+hand_state:mid_flip_bimanual_regrasp",
    )

    m5_run = _first_single_run_after(runs, m4.frame)
    m5 = _run_milestone(
        m5_run,
        confidence=0.6,
        source="hand_cmd+hand_state:second_release_or_catch",
    )

    if visual_rotation is not None:
        visual = _vector("visual_rotation", visual_rotation, length)
        visual_weight = (
            np.ones(length, dtype=np.float64)
            if visual_confidence is None
            else np.clip(_vector("visual_confidence", visual_confidence, length), 0.0, 1.0)
        )
        visual_speed = np.abs(np.diff(np.unwrap(visual), prepend=visual[0])) * visual_weight
        m3 = _refine_with_visual_peak(m3, visual_speed, length)
        m5 = _refine_with_visual_peak(m5, visual_speed, length)
        visual_used = True
    else:
        visual_used = False

    m6_frame, m6_confidence = _final_stable_frame(motion, closed, stable_frames)
    milestones = {
        "M0": Milestone(m0_frame, 0.8, "eef_arm_velocity:motion_onset"),
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m5,
        "M6": Milestone(m6_frame, m6_confidence, "eef_arm_velocity:final_stability"),
    }
    milestones = _enforce_order(milestones)
    progress, progress_mask = interpolate_progress(length, milestones)
    valid_count = sum(milestone.valid for milestone in milestones.values())
    diagnostics = {
        "hand_transition_runs": [
            {"start": start, "end": end, "closed": list(hand_state_value)}
            for start, end, hand_state_value in runs
        ],
        "motion_threshold": motion_threshold,
        "hand_command_state_disagreement_fraction": float(
            np.mean(commanded_closed != measured_closed)
        ),
        "valid_milestone_count": valid_count,
        "visual_rotation_used": visual_used,
        "optional_approach_present": m0_frame > stable_frames,
        "optional_release_or_retreat_present": bool(m6_frame is not None and m6_frame < length - 1),
    }
    return ProgressAnnotation(
        episode_index=episode_index,
        length=length,
        primary_hand=primary_hand,
        milestones=milestones,
        progress=progress.tolist(),
        progress_mask=progress_mask.tolist(),
        review_required=valid_count < 5 or not milestones["M6"].valid,
        diagnostics=diagnostics,
    )


def interpolate_progress(
    length: int,
    milestones: dict[str, Milestone],
) -> tuple[np.ndarray, np.ndarray]:
    progress = np.zeros(length, dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    valid = [
        (index, milestones[name].frame)
        for index, name in enumerate(MILESTONE_NAMES)
        if milestones[name].valid
    ]
    for (left_index, left_frame), (right_index, right_frame) in zip(valid, valid[1:], strict=False):
        assert left_frame is not None and right_frame is not None
        if right_frame <= left_frame:
            continue
        values = np.linspace(
            left_index / (len(MILESTONE_NAMES) - 1),
            right_index / (len(MILESTONE_NAMES) - 1),
            right_frame - left_frame + 1,
            dtype=np.float32,
        )
        progress[left_frame : right_frame + 1] = values
        mask[left_frame : right_frame + 1] = True
    return progress, mask


def progress_horizons(
    progress: Sequence[float],
    mask: Sequence[bool],
    *,
    horizon: int = 40,
) -> tuple[list[list[float]], list[list[bool]]]:
    values = _vector("progress", progress, len(progress))
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("progress and mask must have the same shape")
    value_rows: list[list[float]] = []
    mask_rows: list[list[bool]] = []
    for start in range(len(values)):
        end = min(len(values), start + horizon)
        row = np.zeros(horizon, dtype=np.float32)
        row_mask = np.zeros(horizon, dtype=bool)
        row[: end - start] = values[start:end]
        row_mask[: end - start] = valid[start:end]
        value_rows.append(row.tolist())
        mask_rows.append(row_mask.tolist())
    return value_rows, mask_rows


def _hysteresis_closed(command: np.ndarray, *, stable_frames: int) -> np.ndarray:
    closed_fraction = np.clip((4.5 - command) / 4.5, 0.0, 1.0)
    result = np.zeros_like(closed_fraction, dtype=bool)
    result[0] = closed_fraction[0] >= 0.2
    close_threshold = 0.25
    open_threshold = 0.15
    for frame in range(1, len(result)):
        result[frame] = result[frame - 1]
        result[frame, closed_fraction[frame] >= close_threshold] = True
        result[frame, closed_fraction[frame] <= open_threshold] = False
    for hand in range(2):
        result[:, hand] = _remove_short_boolean_runs(result[:, hand], stable_frames)
    return result


def _remove_short_boolean_runs(values: np.ndarray, minimum: int) -> np.ndarray:
    output = values.copy()
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end - start >= minimum:
            continue
        replacement = output[start - 1] if start > 0 else (output[end] if end < len(output) else False)
        output[start:end] = replacement
    return output


def _stable_runs(closed: np.ndarray, minimum: int) -> list[tuple[int, int, tuple[bool, bool]]]:
    code = closed[:, 0].astype(np.int8) * 2 + closed[:, 1].astype(np.int8)
    starts = np.r_[0, np.flatnonzero(code[1:] != code[:-1]) + 1]
    ends = np.r_[starts[1:], len(code)]
    return [
        (int(start), int(end), (bool(closed[start, 0]), bool(closed[start, 1])))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= minimum
    ]


def _motion_score(eef: np.ndarray, joints: np.ndarray) -> np.ndarray:
    eef_position = np.concatenate((eef[:, 0:3], eef[:, 6:9]), axis=1)
    arms = joints[:, 22:36]
    eef_speed = np.linalg.norm(np.diff(eef_position, axis=0, prepend=eef_position[:1]), axis=1)
    arm_speed = np.linalg.norm(np.diff(arms, axis=0, prepend=arms[:1]), axis=1)
    score = eef_speed + 0.05 * arm_speed
    if len(score) >= 5:
        score = np.convolve(score, np.ones(5) / 5.0, mode="same")
    return score


def _first_sustained(values: np.ndarray, minimum: int) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run >= minimum:
            return index - minimum + 1
    return None


def _first_run_after(
    runs: list[tuple[int, int, tuple[bool, bool]]],
    frame: int | None,
    *,
    required: tuple[bool, bool],
) -> tuple[int, int, tuple[bool, bool]] | None:
    if frame is None:
        return None
    return next((run for run in runs if run[0] > frame and run[2] == required), None)


def _first_single_run_after(
    runs: list[tuple[int, int, tuple[bool, bool]]],
    frame: int | None,
) -> tuple[int, int, tuple[bool, bool]] | None:
    if frame is None:
        return None
    return next((run for run in runs if run[0] > frame and sum(run[2]) == 1), None)


def _run_milestone(
    run: tuple[int, int, tuple[bool, bool]] | None,
    *,
    confidence: float,
    source: str,
) -> Milestone:
    return Milestone(None if run is None else run[0], 0.0 if run is None else confidence, source)


def _refine_with_visual_peak(
    milestone: Milestone,
    visual_speed: np.ndarray,
    length: int,
) -> Milestone:
    if milestone.frame is None:
        return milestone
    radius = max(5, int(round(0.5 * 30)))
    start = max(0, milestone.frame - radius)
    end = min(length, milestone.frame + radius + 1)
    local = visual_speed[start:end]
    if not len(local) or float(local.max()) <= 1e-6:
        return milestone
    peak = start + int(np.argmax(local))
    return Milestone(
        peak,
        min(1.0, milestone.confidence + 0.05),
        f"{milestone.source}+head_rgb_rotation",
    )


def _final_stable_frame(
    motion: np.ndarray,
    closed: np.ndarray,
    stable_frames: int,
) -> tuple[int | None, float]:
    tail = max(stable_frames, min(len(motion) // 5, 30))
    threshold = max(float(np.quantile(motion, 0.25)), 1e-6)
    for end in range(len(motion), tail - 1, -1):
        start = end - tail
        if float(np.quantile(motion[start:end], 0.75)) <= 1.5 * threshold:
            if np.all(closed[start:end] == closed[start]):
                return start, 0.8
    return None, 0.0


def _enforce_order(milestones: dict[str, Milestone]) -> dict[str, Milestone]:
    result = dict(milestones)
    previous = -1
    for name in MILESTONE_NAMES:
        milestone = result[name]
        if milestone.frame is None:
            continue
        if milestone.frame <= previous:
            result[name] = Milestone(None, 0.0, f"{milestone.source}:rejected_non_monotonic")
            continue
        previous = milestone.frame
    return result


def _matrix(name: str, values: Sequence[Sequence[float]], *, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [T,{columns}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _vector(name: str, values: Sequence[float], length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array
