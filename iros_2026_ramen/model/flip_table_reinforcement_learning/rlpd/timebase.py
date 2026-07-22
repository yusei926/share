"""Deterministic scheduling between policy and simulator control rates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyControlClock:
    """Advance a slower policy on top of a faster simulator control loop.

    The first policy target is selected before the first simulator step. Each
    call to :meth:`advance_sim_step` then reports whether the next policy target
    is due. A phase accumulator avoids drift for non-integer ratios such as
    30 Hz policy inference over 50 Hz simulator control.
    """

    policy_hz: float
    sim_control_hz: float
    _phase: float = field(init=False, default=0.0)
    sim_steps: int = field(init=False, default=0)
    completed_policy_intervals: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.policy_hz = float(self.policy_hz)
        self.sim_control_hz = float(self.sim_control_hz)
        if self.policy_hz <= 0.0 or self.sim_control_hz <= 0.0:
            raise ValueError("policy_hz and sim_control_hz must be positive")
        if self.policy_hz > self.sim_control_hz:
            raise ValueError("policy_hz cannot exceed sim_control_hz")

    def advance_sim_step(self) -> bool:
        """Record one simulator step and return whether a policy interval ended."""

        self.sim_steps += 1
        self._phase += self.policy_hz
        if self._phase + 1.0e-12 < self.sim_control_hz:
            return False
        self._phase -= self.sim_control_hz
        self.completed_policy_intervals += 1
        return True

    def reset(self) -> None:
        self._phase = 0.0
        self.sim_steps = 0
        self.completed_policy_intervals = 0

    @property
    def phase(self) -> float:
        return self._phase
