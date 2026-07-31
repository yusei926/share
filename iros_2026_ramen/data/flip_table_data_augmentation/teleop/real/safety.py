"""Official-motion-compatible command filtering for the physical G1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..config import SafetyLimits
from ..contracts import ArmHandTarget, ControlMode
from ..shared.watchdog import WatchdogState


@dataclass(frozen=True)
class RealSafeTarget:
    arm_position_rad: tuple[float, ...]
    dex1_opening_fraction: tuple[float, float]
    arm_feedforward_torque_nm: tuple[float, ...]
    watchdog: WatchdogState


class OfficialG1CommandFilter:
    """Apply the upstream G1 motion-mode q-limit without Sim smoothing.

    Official xr_teleoperate already smooths the IK output.  The physical path
    therefore performs only Unitree's measured-position-relative global delta
    scaling here.  Dex1's feedback-relative bound and weighted filter remain
    in the real DDS backend, where fresh motor feedback is available.
    """

    def __init__(self, limits: SafetyLimits, *, servo_hz: float) -> None:
        if not math.isfinite(servo_hz) or servo_hz <= 0.0:
            raise ValueError("servo_hz must be positive")
        self.limits = limits
        self.dt = 1.0 / servo_hz
        self._last_arm: np.ndarray | None = None
        self._last_hand: np.ndarray | None = None
        self._last_torque = np.zeros(14, dtype=np.float64)
        self._last_sequence = -1

    @staticmethod
    def _array(value: Iterable[float], width: int, label: str) -> np.ndarray:
        result = np.asarray(tuple(value), dtype=np.float64)
        if result.shape != (width,) or not np.isfinite(result).all():
            raise ValueError(f"{label} must be finite [{width}]")
        return result

    def reset(
        self,
        arm_position_rad: Iterable[float],
        dex1_opening_fraction: Iterable[float],
    ) -> None:
        self._last_arm = self._array(
            arm_position_rad, 14, "arm_position_rad"
        ).copy()
        self._last_hand = np.clip(
            self._array(
                dex1_opening_fraction, 2, "dex1_opening_fraction"
            ),
            0.0,
            1.0,
        )
        self._last_torque.fill(0.0)
        self._last_sequence = -1

    def watchdog_state(
        self,
        *,
        now_ns: int,
        last_command_ns: int | None,
        tracking: bool,
    ) -> WatchdogState:
        if not tracking or last_command_ns is None:
            return WatchdogState.STOP
        age_s = max(0.0, (now_ns - last_command_ns) / 1.0e9)
        if age_s >= self.limits.command_stop_timeout_s:
            return WatchdogState.STOP
        if age_s >= self.limits.command_hold_timeout_s:
            return WatchdogState.HOLD
        return WatchdogState.ACTIVE

    def apply(
        self,
        command: ArmHandTarget | None,
        *,
        measured_arm_position_rad: Iterable[float],
        measured_dex1_opening_fraction: Iterable[float],
        now_ns: int,
        last_command_ns: int | None,
        tracking: bool,
        official_arm_velocity_limit_rad_s: float,
    ) -> RealSafeTarget:
        measured_arm = self._array(
            measured_arm_position_rad, 14, "measured arm"
        )
        measured_hand = np.clip(
            self._array(
                measured_dex1_opening_fraction, 2, "measured Dex1"
            ),
            0.0,
            1.0,
        )
        if self._last_arm is None or self._last_hand is None:
            self.reset(measured_arm, measured_hand)
        assert self._last_arm is not None and self._last_hand is not None

        watchdog = self.watchdog_state(
            now_ns=now_ns,
            last_command_ns=last_command_ns,
            tracking=tracking,
        )
        if watchdog is WatchdogState.HOLD:
            return RealSafeTarget(
                tuple(self._last_arm),
                tuple(self._last_hand),
                tuple(self._last_torque),
                watchdog,
            )
        if watchdog is not WatchdogState.ACTIVE or command is None:
            self.reset(measured_arm, measured_hand)
            return RealSafeTarget(
                tuple(measured_arm), tuple(measured_hand), (0.0,) * 14, watchdog
            )
        if command.mode not in {ControlMode.TRACK, ControlMode.HOLD}:
            self.reset(measured_arm, measured_hand)
            return RealSafeTarget(
                tuple(measured_arm),
                tuple(measured_hand),
                (0.0,) * 14,
                WatchdogState.STOP,
            )
        if command.sequence < self._last_sequence:
            raise ValueError(
                f"out-of-order command sequence {command.sequence} "
                f"< {self._last_sequence}"
            )
        if (
            not math.isfinite(official_arm_velocity_limit_rad_s)
            or official_arm_velocity_limit_rad_s <= 0.0
        ):
            raise ValueError("official G1 arm velocity limit must be positive")

        lower = np.asarray(
            self.limits.arm_position_lower_rad, dtype=np.float64
        )
        upper = np.asarray(
            self.limits.arm_position_upper_rad, dtype=np.float64
        )
        desired_arm = np.clip(
            np.asarray(command.arm_position_rad, dtype=np.float64), lower, upper
        )
        desired_hand = np.clip(
            np.asarray(command.dex1_opening_fraction, dtype=np.float64),
            0.0,
            1.0,
        )
        desired_torque = self._array(
            command.arm_feedforward_torque_nm, 14, "arm feedforward torque"
        )

        # Exact G1_29_ArmController direction-preserving q scaling, relative
        # to the latest measured position.  Do not add acceleration smoothing
        # after upstream IK's moving-average filter.
        delta = desired_arm - measured_arm
        motion_scale = float(np.max(np.abs(delta))) / (
            official_arm_velocity_limit_rad_s * self.dt
        )
        arm = measured_arm + delta / max(motion_scale, 1.0)

        self._last_arm = np.clip(arm, lower, upper)
        self._last_hand = desired_hand
        self._last_torque = desired_torque.copy()
        self._last_sequence = command.sequence
        return RealSafeTarget(
            tuple(self._last_arm),
            tuple(self._last_hand),
            tuple(self._last_torque),
            watchdog,
        )
