from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrimResult:
    valid: bool
    start: int
    end: int
    first_active: int | None
    last_active: int | None
    post_roll_complete: bool
    reason: str | None

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _persistent(mask: np.ndarray, window: int, required: int) -> np.ndarray:
    if window <= 0 or required <= 0 or required > window:
        raise ValueError("invalid persistence parameters")
    counts = np.convolve(mask.astype(np.int16), np.ones(window, dtype=np.int16), mode="same")
    return counts >= required


def detect_trim(
    arm_targets: np.ndarray,
    hand_targets: np.ndarray,
    *,
    fps: float,
    arm_velocity_threshold: float,
    hand_velocity_threshold: float,
    persistence_window: int,
    persistence_required: int,
    pre_roll: int,
    post_roll: int,
    minimum_frames: int,
    minimum_terminal_stable_frames: int,
) -> TrimResult:
    arms = np.asarray(arm_targets, dtype=np.float64)
    hands = np.asarray(hand_targets, dtype=np.float64)
    if arms.ndim != 2 or arms.shape[1] != 14:
        raise ValueError("arm_targets must be [T,14]")
    if hands.shape != (len(arms), 2):
        raise ValueError("hand_targets must be [T,2]")
    if len(arms) < 2 or not np.isfinite(arms).all() or not np.isfinite(hands).all():
        return TrimResult(False, 0, 0, None, None, False, "invalid_or_too_short")

    arm_speed = np.max(np.abs(np.diff(arms, axis=0, prepend=arms[:1])) * fps, axis=1)
    hand_speed = np.max(np.abs(np.diff(hands, axis=0, prepend=hands[:1])) * fps, axis=1)
    raw_active = (arm_speed > arm_velocity_threshold) | (
        hand_speed > hand_velocity_threshold
    )
    active = _persistent(raw_active, persistence_window, persistence_required)
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return TrimResult(False, 0, 0, None, None, False, "no_active_motion")
    first = int(indices[0])
    last = int(indices[-1])
    stable_after = len(arms) - 1 - last
    start = max(0, first - pre_roll)
    end = min(len(arms), last + 1 + post_roll)
    if end - start < minimum_frames:
        start = max(0, end - minimum_frames)
        if end - start < minimum_frames:
            return TrimResult(
                False, start, end, first, last, False, "trimmed_too_short"
            )
    # The official subtask slices often end while the arms are retracting even
    # though the table is already stable. Rejecting those rows would select for
    # recorder padding rather than manipulation quality. Keep the source end,
    # record that the requested post-roll was truncated, and let the full
    # trajectory/terminal-image cluster reject genuinely incomplete flips.
    post_roll_complete = stable_after >= max(
        minimum_terminal_stable_frames, post_roll
    )
    return TrimResult(
        True, start, end, first, last, post_roll_complete, None
    )


def resample_trajectory(values: np.ndarray, points: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2 or points < 2:
        raise ValueError("trajectory must be [T,D], T>=2 and points>=2")
    old = np.linspace(0.0, 1.0, len(source))
    new = np.linspace(0.0, 1.0, points)
    return np.stack(
        [np.interp(new, old, source[:, column]) for column in range(source.shape[1])],
        axis=1,
    )
