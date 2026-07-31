"""Operator state shared by real and simulated teleoperation runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import TeleopObservation


@dataclass
class TrackingAnchorRequest:
    """Issue unique re-anchor generations without reusing a stale anchor."""

    last_issued: int = 0
    generation: int = 0

    def request(self) -> int:
        self.last_issued += 1
        self.generation = self.last_issued
        return self.generation

    def disarm(self) -> None:
        self.generation = 0


def hold_target_from_observation(
    observation: TeleopObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy the actually applied target for a no-motion operator pause."""

    arm = np.asarray(observation.applied_arm_target_rad, dtype=np.float64)
    hand = np.asarray(observation.applied_dex1_opening_target, dtype=np.float64)
    torque = np.asarray(
        observation.diagnostics.get("arm_feedforward_torque_nm", (0.0,) * 14),
        dtype=np.float64,
    )
    if (
        arm.shape != (14,)
        or torque.shape != (14,)
        or hand.shape != (2,)
        or not np.isfinite(arm).all()
        or not np.isfinite(torque).all()
        or not np.isfinite(hand).all()
    ):
        raise ValueError(
            "teleop hold target must be finite arm[14], torque[14], and Dex1[2]"
        )
    if np.any(hand < 0.0) or np.any(hand > 1.0):
        raise ValueError("teleop hold Dex1 target must lie in [0, 1]")
    return arm.copy(), hand.copy(), torque.copy()
