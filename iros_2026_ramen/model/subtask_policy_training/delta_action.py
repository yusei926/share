"""Shared arm-delta and absolute-gripper action contract for flip-table policies."""

from __future__ import annotations

import math
from collections.abc import Sequence


ARM_DIM = 14
GRIPPER_DIM = 2
ACTION_DIM = ARM_DIM + GRIPPER_DIM
STATE_ARM_START = 3
STATE_DIM = 19
ACTION_REPRESENTATION = "arm_delta_gripper_absolute"
ACTION_SEMANTICS = (
    "upper-body arm joint deltas; first action is desired-current, future actions are "
    "desired[t+k]-desired[t+k-1]; left/right gripper commands are absolute"
)


def _finite_vector(name: str, values: Sequence[float], expected_dim: int) -> list[float]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} must be {expected_dim}-D, got {len(values)}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def encode_action(
    *,
    state: Sequence[float],
    absolute_action: Sequence[float],
    previous_arm_target: Sequence[float] | None,
) -> list[float]:
    """Encode a target action without leaking state across episode boundaries."""

    state_values = _finite_vector("state", state, STATE_DIM)
    target = _finite_vector("absolute_action", absolute_action, ACTION_DIM)
    if previous_arm_target is None:
        arm_reference = state_values[STATE_ARM_START : STATE_ARM_START + ARM_DIM]
    else:
        arm_reference = _finite_vector("previous_arm_target", previous_arm_target, ARM_DIM)
    return [
        target[index] - arm_reference[index] for index in range(ARM_DIM)
    ] + target[ARM_DIM:]


def decode_action_chunk(
    *,
    initial_state: Sequence[float],
    delta_actions: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Integrate a model chunk into absolute arm targets and absolute gripper commands."""

    state = _finite_vector("initial_state", initial_state, STATE_DIM)
    arm_target = state[STATE_ARM_START : STATE_ARM_START + ARM_DIM]
    decoded: list[list[float]] = []
    for index, action in enumerate(delta_actions):
        encoded = _finite_vector(f"delta_actions[{index}]", action, ACTION_DIM)
        arm_target = [arm_target[joint] + encoded[joint] for joint in range(ARM_DIM)]
        decoded.append(arm_target + encoded[ARM_DIM:])
    return decoded
