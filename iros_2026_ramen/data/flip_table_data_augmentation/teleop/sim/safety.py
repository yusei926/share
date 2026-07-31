"""Simulation-only command limits and watchdog behavior."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..config import SafetyLimits
from ..contracts import ArmHandTarget, ControlMode
from ..shared.watchdog import WatchdogState


@dataclass(frozen=True)
class SafeTarget:
    arm_position_rad: tuple[float, ...]
    dex1_opening_fraction: tuple[float, float]
    arm_feedforward_torque_nm: tuple[float, ...]
    watchdog: WatchdogState


class CommandSafetyFilter:
    """Bound commands for the Isaac action path.

    Isaac receives joint actions at its own servo cadence.  Its limits remain
    deliberately independent from Unitree's 250 Hz ``rt/arm_sdk`` contract.
    Feedforward torque is retained in the shared envelope for logging, but the
    simulator action path does not consume it.
    """

    def __init__(self, limits: SafetyLimits, *, servo_hz: float) -> None:
        if not math.isfinite(servo_hz) or servo_hz <= 0.0:
            raise ValueError("servo_hz must be positive")
        self.limits = limits
        self.dt = 1.0 / servo_hz
        self._last_arm: np.ndarray | None = None
        self._last_arm_velocity = np.zeros(14, dtype=np.float64)
        self._last_hand: np.ndarray | None = None
        self._last_hand_velocity = np.zeros(2, dtype=np.float64)
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
        self._last_arm_velocity.fill(0.0)
        self._last_hand_velocity.fill(0.0)
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
        error = desired - previous
        # Brake before the target instead of accelerating until crossing it.
        # sqrt(2*a*d) is the continuous stopping-speed envelope; the following
        # acceleration projection preserves the discrete per-tick bound.
        # Discrete stopping envelope.  The ``-a*dt`` term reserves one full
        # deceleration tick and prevents the final joint-limit clip from
        # silently creating a larger acceleration than configured.
        a_dt = acceleration_limit * dt
        stopping_speed = np.maximum(
            0.0,
            -a_dt
            + np.sqrt(
                np.maximum(0.0, a_dt * a_dt + 2.0 * acceleration_limit * np.abs(error))
            ),
        )
        desired_velocity = np.sign(error) * np.minimum(
            velocity_limit, stopping_speed
        )
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
        crossed = (
            np.sign(desired - previous) != np.sign(desired - candidate)
        )
        landing_velocity = error / dt
        landing_is_feasible = (
            np.abs(landing_velocity) <= velocity_limit + 1.0e-12
        ) & (
            np.abs(landing_velocity - previous_velocity)
            <= acceleration_limit * dt + 1.0e-12
        )
        # When an exact final step is dynamically feasible, retain its actual
        # velocity. The next zero-velocity tick then also satisfies the
        # acceleration bound. Never snap both position and velocity at once.
        land = crossed & landing_is_feasible
        candidate = np.where(land, desired, candidate)
        velocity = np.where(land, landing_velocity, velocity)
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
            return SafeTarget(
                tuple(self._last_arm),
                tuple(self._last_hand),
                tuple(self._last_torque),
                watchdog,
            )
        if watchdog is not WatchdogState.ACTIVE or command is None:
            self.reset(measured_arm, measured_hand)
            return SafeTarget(
                tuple(measured_arm), tuple(measured_hand), (0.0,) * 14, watchdog
            )
        if command.mode not in {ControlMode.TRACK, ControlMode.HOLD}:
            self.reset(measured_arm, measured_hand)
            return SafeTarget(
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
        emitted_arm = np.clip(arm, lower, upper)
        emitted_arm_velocity = (emitted_arm - self._last_arm) / self.dt
        emitted_hand = np.clip(hand, 0.0, 1.0)
        emitted_hand_velocity = (emitted_hand - self._last_hand) / self.dt
        if np.any(
            np.abs(emitted_arm_velocity - self._last_arm_velocity)
            > self.limits.arm_acceleration_rad_s2 * self.dt + 1.0e-9
        ):
            raise RuntimeError("arm limiter could not satisfy acceleration at a joint boundary")
        if np.any(
            np.abs(emitted_hand_velocity - self._last_hand_velocity)
            > self.limits.hand_acceleration_fraction_s2 * self.dt + 1.0e-9
        ):
            raise RuntimeError("hand limiter could not satisfy acceleration at a joint boundary")
        self._last_arm = emitted_arm
        self._last_arm_velocity = emitted_arm_velocity
        self._last_hand = emitted_hand
        self._last_hand_velocity = emitted_hand_velocity
        self._last_torque = np.asarray(
            command.arm_feedforward_torque_nm, dtype=np.float64
        ).copy()
        self._last_sequence = command.sequence
        return SafeTarget(
            tuple(self._last_arm),
            tuple(self._last_hand),
            tuple(self._last_torque),
            watchdog,
        )
