"""Timestamp-aligned temporal ensembling in physical upper-body target space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dex1_hand_synergy import hand_to_dex1

PHYSICAL_ACTION_DIM = 16
LOGICAL_ACTION_DIM = 53


def logical_chunk_to_physical_targets(chunk: Sequence[Sequence[float]]) -> np.ndarray:
    """Decode a postprocessed 53-D chunk to arms14 + absolute Dex1 left/right."""
    logical = np.asarray(chunk, dtype=np.float64)
    if logical.ndim != 2 or logical.shape[1] != LOGICAL_ACTION_DIM:
        raise ValueError(f"logical chunk must have shape [T,53], got {logical.shape}")
    if not np.isfinite(logical).all():
        raise ValueError("logical chunk contains NaN or Inf")
    physical = np.empty((logical.shape[0], PHYSICAL_ACTION_DIM), dtype=np.float64)
    physical[:, :7] = logical[:, 32:39]
    physical[:, 7:14] = logical[:, 39:46]
    physical[:, 14] = [
        hand_to_dex1(row[18:25], side="left", kind="action") for row in logical
    ]
    physical[:, 15] = [
        hand_to_dex1(row[25:32], side="right", kind="action") for row in logical
    ]
    return physical


@dataclass(frozen=True)
class _Candidate:
    origin_step: int
    target: np.ndarray


class PhysicalTargetTemporalEnsembler:
    """Blend overlapping predictions only after absolute physical decoding."""

    def __init__(self, *, decay_lambda: float | None = -0.1) -> None:
        if decay_lambda is not None and not math.isfinite(float(decay_lambda)):
            raise ValueError("temporal ensemble lambda must be finite or None")
        self.decay_lambda = None if decay_lambda is None else float(decay_lambda)
        self._candidates: dict[int, list[_Candidate]] = {}

    def reset(self) -> None:
        self._candidates.clear()

    def add_chunk(self, *, origin_step: int, absolute_targets: Sequence[Sequence[float]]) -> None:
        targets = np.asarray(absolute_targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != PHYSICAL_ACTION_DIM:
            raise ValueError(f"physical target chunk must have shape [T,16], got {targets.shape}")
        if not np.isfinite(targets).all():
            raise ValueError("physical target chunk contains NaN or Inf")
        for offset, target in enumerate(targets):
            step = int(origin_step) + offset
            self._candidates.setdefault(step, []).append(
                _Candidate(origin_step=int(origin_step), target=target.copy())
            )

    def target(self, step: int) -> np.ndarray:
        step = int(step)
        candidates = self._candidates.get(step, [])
        if not candidates:
            raise KeyError(f"no temporal-ensemble target for step {step}")
        if self.decay_lambda is None:
            result = max(candidates, key=lambda candidate: candidate.origin_step).target.copy()
        else:
            weights = np.asarray(
                [
                    math.exp(self.decay_lambda * max(0, step - candidate.origin_step))
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            stacked = np.stack([candidate.target for candidate in candidates], axis=0)
            result = np.average(stacked, axis=0, weights=weights)
        self._discard_before(step)
        return result

    def candidate_count(self, step: int) -> int:
        return len(self._candidates.get(int(step), []))

    def _discard_before(self, step: int) -> None:
        for old_step in [value for value in self._candidates if value < step]:
            del self._candidates[old_step]


class UpperBodySafetyLimiter:
    """Apply position, velocity, then acceleration limits after ensembling."""

    def __init__(
        self,
        *,
        lower: Sequence[float],
        upper: Sequence[float],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        control_hz: float,
    ) -> None:
        self.lower = _vector("lower", lower)
        self.upper = _vector("upper", upper)
        self.max_velocity = _vector("max_velocity", max_velocity)
        self.max_acceleration = _vector("max_acceleration", max_acceleration)
        self.control_hz = float(control_hz)
        if self.control_hz <= 0 or not math.isfinite(self.control_hz):
            raise ValueError("control_hz must be positive and finite")
        if np.any(self.lower > self.upper):
            raise ValueError("lower safety limits exceed upper limits")
        if np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("velocity and acceleration limits must be positive")
        self._previous_target: np.ndarray | None = None
        self._previous_velocity = np.zeros(PHYSICAL_ACTION_DIM, dtype=np.float64)

    def reset(self) -> None:
        self._previous_target = None
        self._previous_velocity.fill(0)

    def apply(self, target: Sequence[float], *, measured: Sequence[float]) -> np.ndarray:
        requested = np.clip(_vector("target", target), self.lower, self.upper)
        measured_array = _vector("measured", measured)
        reference = measured_array if self._previous_target is None else self._previous_target
        dt = 1.0 / self.control_hz
        desired_velocity = np.clip(
            (requested - reference) / dt,
            -self.max_velocity,
            self.max_velocity,
        )
        velocity_delta = np.clip(
            desired_velocity - self._previous_velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        velocity = self._previous_velocity + velocity_delta
        safe = np.clip(reference + velocity * dt, self.lower, self.upper)
        self._previous_target = safe
        self._previous_velocity = velocity
        return safe.copy()


def _vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError(f"{name} must have shape [16], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array
