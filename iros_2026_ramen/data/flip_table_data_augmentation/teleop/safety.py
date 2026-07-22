"""Position, rate, and stale-command limits shared by both backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np

from .config import SafetyLimits
from .contracts import ArmHandTarget, ControlEvent, ControlMode


class WatchdogState(str, Enum):
    ACTIVE = "active"
    HOLD = "hold"
    STOP = "stop"


@dataclass(frozen=True)
class SafeTarget:
    arm_position_rad: tuple[float, ...]
    dex1_opening_fraction: tuple[float, float]
    watchdog: WatchdogState


class CommandSafetyFilter:
    """Rate-limit targets and turn communication gaps into hold then stop."""

    def __init__(self, limits: SafetyLimits, *, servo_hz: float) -> None:
        if not math.isfinite(servo_hz) or servo_hz <= 0.0:
            raise ValueError("servo_hz must be positive")
        self.limits = limits
        self.dt = 1.0 / servo_hz
        self._last_arm: np.ndarray | None = None
        self._last_arm_velocity = np.zeros(14, dtype=np.float64)
        self._last_hand: np.ndarray | None = None
        self._last_hand_velocity = np.zeros(2, dtype=np.float64)
        self._last_sequence = -1

    @staticmethod
    def _array(value: Iterable[float], width: int, label: str) -> np.ndarray:
        result = np.asarray(tuple(value), dtype=np.float64)
        if result.shape != (width,) or not np.isfinite(result).all():
            raise ValueError(f"{label} must be finite [{width}]")
        return result

    def reset(self, arm_position_rad: Iterable[float], dex1_opening_fraction: Iterable[float]) -> None:
        self._last_arm = self._array(arm_position_rad, 14, "arm_position_rad").copy()
        self._last_hand = np.clip(
            self._array(dex1_opening_fraction, 2, "dex1_opening_fraction"), 0.0, 1.0
        )
        self._last_arm_velocity.fill(0.0)
        self._last_hand_velocity.fill(0.0)
        self._last_sequence = -1

    def watchdog_state(self, *, now_ns: int, last_command_ns: int | None, tracking: bool) -> WatchdogState:
        if not tracking or last_command_ns is None:
            return WatchdogState.STOP
        age_s = max(0.0, (now_ns - last_command_ns) / 1.0e9)
        if age_s >= self.limits.command_stop_timeout_s:
            return WatchdogState.STOP
        if age_s >= self.limits.command_hold_timeout_s:
            return WatchdogState.HOLD
        return WatchdogState.ACTIVE

    @staticmethod
    def _rate_limit(
        desired: np.ndarray,
        previous: np.ndarray,
        previous_velocity: np.ndarray,
        *,
        velocity_limit: float,
        acceleration_limit: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired_velocity = np.clip((desired - previous) / dt, -velocity_limit, velocity_limit)
        velocity_delta = np.clip(
            desired_velocity - previous_velocity,
            -acceleration_limit * dt,
            acceleration_limit * dt,
        )
        velocity = np.clip(
            previous_velocity + velocity_delta,
            -velocity_limit,
            velocity_limit,
        )
        candidate = previous + velocity * dt
        reached = np.isclose(candidate, desired, atol=1e-12) | (
            np.sign(desired - previous) != np.sign(desired - candidate)
        )
        candidate = np.where(reached, desired, candidate)
        velocity = np.where(reached, 0.0, velocity)
        return candidate, velocity

    def apply(
        self,
        command: ArmHandTarget | None,
        *,
        measured_arm_position_rad: Iterable[float],
        measured_dex1_opening_fraction: Iterable[float],
        now_ns: int,
        last_command_ns: int | None,
        tracking: bool,
    ) -> SafeTarget:
        measured_arm = self._array(measured_arm_position_rad, 14, "measured arm")
        measured_hand = np.clip(
            self._array(measured_dex1_opening_fraction, 2, "measured Dex1"), 0.0, 1.0
        )
        if self._last_arm is None or self._last_hand is None:
            self.reset(measured_arm, measured_hand)
        assert self._last_arm is not None and self._last_hand is not None

        watchdog = self.watchdog_state(
            now_ns=now_ns, last_command_ns=last_command_ns, tracking=tracking
        )
        if watchdog is not WatchdogState.ACTIVE or command is None:
            self.reset(measured_arm, measured_hand)
            return SafeTarget(tuple(measured_arm), tuple(measured_hand), watchdog)
        if command.mode is not ControlMode.TRACK:
            self.reset(measured_arm, measured_hand)
            return SafeTarget(tuple(measured_arm), tuple(measured_hand), WatchdogState.STOP)
        if command.sequence < self._last_sequence:
            raise ValueError(
                f"out-of-order command sequence {command.sequence} < {self._last_sequence}"
            )

        lower = np.asarray(self.limits.arm_position_lower_rad, dtype=np.float64)
        upper = np.asarray(self.limits.arm_position_upper_rad, dtype=np.float64)
        desired_arm = np.clip(np.asarray(command.arm_position_rad, dtype=np.float64), lower, upper)
        desired_hand = np.clip(np.asarray(command.dex1_opening_fraction, dtype=np.float64), 0.0, 1.0)
        arm, arm_velocity = self._rate_limit(
            desired_arm,
            self._last_arm,
            self._last_arm_velocity,
            velocity_limit=self.limits.arm_velocity_rad_s,
            acceleration_limit=self.limits.arm_acceleration_rad_s2,
            dt=self.dt,
        )
        hand, hand_velocity = self._rate_limit(
            desired_hand,
            self._last_hand,
            self._last_hand_velocity,
            velocity_limit=self.limits.hand_velocity_fraction_s,
            acceleration_limit=self.limits.hand_acceleration_fraction_s2,
            dt=self.dt,
        )
        self._last_arm = np.clip(arm, lower, upper)
        self._last_arm_velocity = arm_velocity
        self._last_hand = np.clip(hand, 0.0, 1.0)
        self._last_hand_velocity = hand_velocity
        self._last_sequence = command.sequence
        return SafeTarget(tuple(self._last_arm), tuple(self._last_hand), watchdog)


def idle_target(
    sequence: int,
    monotonic_ns: int,
    arm_position_rad: Iterable[float],
    dex1_opening_fraction: Iterable[float],
    *,
    event: ControlEvent = ControlEvent.NONE,
) -> ArmHandTarget:
    return ArmHandTarget(
        sequence=sequence,
        monotonic_ns=monotonic_ns,
        mode=ControlMode.IDLE,
        event=event,
        arm_position_rad=tuple(float(value) for value in arm_position_rad),
        dex1_opening_fraction=tuple(float(value) for value in dex1_opening_fraction),
    )
