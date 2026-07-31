"""Canonical deployable state and action contract for flip-table policies.

The high-level policy observes the three waist joints because their motion is
useful proprioception, but never owns them.  Balance is owned by Unitree
Regular mode on hardware and the organizer WBC in simulation.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from typing import Any

import numpy as np


STATE_DIM = 19
ACTION_DIM = 16
WAIST_DIM = 3
ARM_DIM = 14
HAND_DIM = 2
ACTION_CONVERSION_VERSION = "drop_policy_waist_action/v1"
ARM_ORDER = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ACTION_ORDER = ARM_ORDER + ("left_dex1_command", "right_dex1_command")
STATE_ORDER = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
) + ARM_ORDER + ("left_dex1_state", "right_dex1_state")


def finite_vector(value: Iterable[float], width: int, label: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}]")
    return result


def state_19d(
    body_joint_position_rad: Iterable[float],
    dex1_opening_fraction: Iterable[float],
) -> np.ndarray:
    """Build waist3 + arms14 + Dex1-command2 from deployable feedback."""

    body = finite_vector(body_joint_position_rad, 29, "body joint position")
    hands = finite_vector(dex1_opening_fraction, HAND_DIM, "Dex1 opening")
    result = np.concatenate((body[12:15], body[15:29], 4.5 * hands))
    if result.shape != (STATE_DIM,):  # defensive against order changes
        raise AssertionError(f"state contract produced {result.shape}")
    return result


def action_16d(
    arm_position_rad: Iterable[float],
    dex1_opening_fraction: Iterable[float],
) -> np.ndarray:
    """Build arms14 + Dex1-command2; waist is intentionally absent."""

    arms = finite_vector(arm_position_rad, ARM_DIM, "arm target")
    hands = finite_vector(dex1_opening_fraction, HAND_DIM, "Dex1 target")
    result = np.concatenate((arms, 4.5 * hands))
    if result.shape != (ACTION_DIM,):
        raise AssertionError(f"action contract produced {result.shape}")
    return result


def convert_legacy_action_19d(value: Iterable[float]) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically remove the legacy waist command without mutating input."""

    legacy = finite_vector(value, STATE_DIM, "legacy 19-D action")
    canonical = np.concatenate((legacy[WAIST_DIM : WAIST_DIM + ARM_DIM], legacy[-HAND_DIM:]))
    source_bytes = json.dumps(
        [float(item) for item in legacy], separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return canonical, {
        "conversion_version": ACTION_CONVERSION_VERSION,
        "source_action_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "dropped_waist_action_rad": [float(item) for item in legacy[:WAIST_DIM]],
    }
