from __future__ import annotations

import numpy as np
import pytest

from model.subtask_policy_training.gr00t.dex1_hand_synergy import dex1_to_hand
from model.subtask_policy_training.gr00t.temporal_ensemble import (
    PhysicalTargetTemporalEnsembler,
    UpperBodySafetyLimiter,
    logical_chunk_to_physical_targets,
)


def test_logical_action_decodes_full_hand_synergy_to_dex1() -> None:
    logical = np.zeros((2, 53), dtype=np.float64)
    logical[:, 32:46] = np.arange(14)
    logical[0, 18:25] = dex1_to_hand(4.5, side="left", kind="action")
    logical[0, 25:32] = dex1_to_hand(0.0, side="right", kind="action")
    logical[1, 18:25] = dex1_to_hand(1.5, side="left", kind="action")
    logical[1, 25:32] = dex1_to_hand(3.0, side="right", kind="action")

    physical = logical_chunk_to_physical_targets(logical)

    assert physical.shape == (2, 16)
    assert physical[0, :14] == pytest.approx(np.arange(14))
    assert np.allclose(physical[:, 14:], [[4.5, 0.0], [1.5, 3.0]])


def test_temporal_ensemble_aligns_absolute_targets_by_execution_step() -> None:
    ensemble = PhysicalTargetTemporalEnsembler(decay_lambda=-0.1)
    first = np.zeros((3, 16))
    second = np.full((3, 16), 2.0)
    ensemble.add_chunk(origin_step=0, absolute_targets=first)
    ensemble.add_chunk(origin_step=1, absolute_targets=second)

    expected = 2.0 / (1.0 + np.exp(-0.1))
    assert ensemble.target(1) == pytest.approx(np.full(16, expected))
    assert ensemble.candidate_count(2) == 2


def test_no_ensemble_uses_newest_chunk() -> None:
    ensemble = PhysicalTargetTemporalEnsembler(decay_lambda=None)
    ensemble.add_chunk(origin_step=0, absolute_targets=np.zeros((2, 16)))
    ensemble.add_chunk(origin_step=1, absolute_targets=np.ones((2, 16)))
    assert ensemble.target(1) == pytest.approx(np.ones(16))


def test_safety_limits_are_applied_after_ensemble() -> None:
    limiter = UpperBodySafetyLimiter(
        lower=[-2.0] * 16,
        upper=[2.0] * 16,
        max_velocity=[1.0] * 16,
        max_acceleration=[10.0] * 16,
        control_hz=10.0,
    )
    safe = limiter.apply([2.0] * 16, measured=[0.0] * 16)
    assert safe == pytest.approx([0.1] * 16)
