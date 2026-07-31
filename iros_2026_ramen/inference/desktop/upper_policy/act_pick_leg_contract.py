"""Pinned physical contract for the Nakatsuka joint16 ACT pick-leg policy.

The Hugging Face checkpoint describes tensor shapes but not their physical
meaning.  The mapping below was audited against the training/deployment source
at ``Approach-Release-Squad/IROS-2026_IKEA_PickTableLeg@817e8add...``.
Only the fourteen G1 arm joints and two Dex1-1 opening values are executable;
Unitree Regular Mode remains the sole owner of waist, legs, and balance.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


MODEL_REPO_ID = "Team-RAMEN/pana_nakatsuka_act_pick_joint16_augxx_s40k_20260730"
MODEL_REVISION = "fcf3f0493c64b301f20fb5c803ab18344b62aabf"
SOURCE_REVISION = "817e8addd944b43b9ada6d096feaa93f75179d38"
TASK_TEXT = "pick table leg"
MODEL_STATE_DIM = 16
MODEL_ACTION_DIM = 16
MODEL_ACTION_HORIZON = 30
DEX1_DATASET_OPEN_VALUE = 4.5
# Pinned deployment start state: episode 2101 frame 0 in the source dataset.
DATASET_INITIAL_DEX1_PHYSICAL = np.asarray([4.43, 4.47], dtype=np.float64)
DATASET_INITIAL_DEX1_OPENING_FRACTION = (
    DATASET_INITIAL_DEX1_PHYSICAL / DEX1_DATASET_OPEN_VALUE
)
CAMERA_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")
CAMERA_KEYS = tuple(f"observation.images.cam_{index}" for index in range(4))

# Serialized action support from the pinned postprocessor.  Policy outputs are
# clamped here before conversion to the canonical hardware boundary.  The
# worker independently verifies these values against the safetensors artifact.
ACTION_MIN = np.asarray(
    [
        -1.396479, -0.338903, -0.935873, -1.047199, -1.946133, -0.856259,
        -1.614414, -1.360593, -1.358774, -0.996851, -0.733229, -1.117883,
        -1.095591, -1.016864, 0.0, 0.0,
    ],
    dtype=np.float64,
)
ACTION_MAX = np.asarray(
    [
        0.533885, 1.222892, 0.979839, 1.392741, 0.767448, 1.614429,
        0.600166, 0.421710, 0.427251, 0.785821, 1.398272, 1.972221,
        1.614429, 1.614429, 4.5, 4.5,
    ],
    dtype=np.float64,
)


def compose_model_state(
    body_joint_position_rad: Sequence[float],
    dex1_opening_fraction: Sequence[float],
) -> np.ndarray:
    """Map G1 body29 plus normalized hands to the training state16."""

    body = np.asarray(body_joint_position_rad, dtype=np.float64)
    hands = np.asarray(dex1_opening_fraction, dtype=np.float64)
    if body.shape != (29,) or not np.isfinite(body).all():
        raise ValueError("G1 body state must be finite [29]")
    if hands.shape != (2,) or not np.isfinite(hands).all():
        raise ValueError("Dex1 opening state must be finite [2]")
    if np.any((hands < 0.0) | (hands > 1.0)):
        raise ValueError("Dex1 opening fractions must be in [0,1]")
    return np.concatenate((body[15:29], hands * DEX1_DATASET_OPEN_VALUE)).astype(
        np.float32
    )


def camera_payloads(camera_jpeg: Mapping[str, bytes]) -> dict[str, bytes]:
    """Map physically verified camera roles to the four training keys."""

    if set(camera_jpeg) != set(CAMERA_ROLES):
        raise ValueError(
            f"ACT requires exactly {CAMERA_ROLES}, got {sorted(camera_jpeg)}"
        )
    return {
        key: bytes(camera_jpeg[role])
        for key, role in zip(CAMERA_KEYS, CAMERA_ROLES, strict=True)
    }


def clamp_native_action(action: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and clamp an absolute action chunk to training support."""

    values = np.asarray(action, dtype=np.float64)
    expected = (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"ACT action chunk must be finite {expected}, got {values.shape}")
    return np.clip(values, ACTION_MIN, ACTION_MAX)


def canonical_action(action: Sequence[Sequence[float]]) -> np.ndarray:
    """Return arms14 radians plus normalized Dex1 opening fractions."""

    result = clamp_native_action(action)
    result[:, 14:] /= DEX1_DATASET_OPEN_VALUE
    return result
