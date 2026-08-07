"""SafeActuator の unit test。

manipulation actuator 未実装なので MockInner (呼び出し記録するだけ) に
対して verify する。実 actuator 実装時に integration test を追加する。
"""

from __future__ import annotations

import pytest

from inference.desktop.lower_policy.actuators.safety import (
    DEFAULT_BLOCKED_JOINTS,
    SafeActuator,
    SafetyLimits,
)


class MockInner:
    """SafeActuator の call を記録するだけの mock。"""

    def __init__(self) -> None:
        self.joint_calls: list[tuple[list[str], list[float]]] = []
        self.hand_calls: list[list[float]] = []

    def send_joint_trajectory(self, names, positions):
        self.joint_calls.append((list(names), list(positions)))

    def send_hand(self, positions):
        self.hand_calls.append(list(positions))


# ---------------------------------------------------------------------------
# joint clamping
# ---------------------------------------------------------------------------


def test_joint_positive_over_limit_clamped_to_limit():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits(joint_max_abs=1.0, blocked_joints=frozenset()))
    safe.send_joint_trajectory(["arm_1"], [10.0])
    assert mock.joint_calls[-1] == (["arm_1"], [1.0])


def test_joint_negative_over_limit_clamped_to_neg_limit():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits(joint_max_abs=1.0, blocked_joints=frozenset()))
    safe.send_joint_trajectory(["arm_1"], [-10.0])
    assert mock.joint_calls[-1] == (["arm_1"], [-1.0])


def test_joint_within_limit_unchanged():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits(joint_max_abs=1.5, blocked_joints=frozenset()))
    safe.send_joint_trajectory(["arm_1", "arm_2"], [0.5, -1.0])
    assert mock.joint_calls[-1] == (["arm_1", "arm_2"], [0.5, -1.0])


# ---------------------------------------------------------------------------
# blocked joints (下半身 filter)
# ---------------------------------------------------------------------------


def test_blocked_joint_filtered_out():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits())  # default blocked = g1_joint_00..11
    safe.send_joint_trajectory(["g1_joint_00", "arm_1"], [0.5, 0.3])
    assert mock.joint_calls[-1] == (["arm_1"], [0.3])


def test_all_blocked_drops_call():
    """全 joint が blocked に該当 → inner を呼ばず drop (空 trajectory 送信しない)。"""
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits())
    safe.send_joint_trajectory(["g1_joint_00", "g1_joint_01"], [0.1, 0.2])
    assert mock.joint_calls == []


def test_default_blocked_joints_covers_lower_body():
    """default で G1 下半身 12 joint (00..11) 全てが blocked。"""
    assert DEFAULT_BLOCKED_JOINTS == frozenset(f"g1_joint_{i:02d}" for i in range(12))
    assert len(DEFAULT_BLOCKED_JOINTS) == 12


# ---------------------------------------------------------------------------
# hand clamping
# ---------------------------------------------------------------------------


def test_hand_positions_clamped_to_limit():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits(hand_max_abs=0.5))
    safe.send_hand([2.0, -3.0, 0.1])
    assert mock.hand_calls[-1] == [0.5, -0.5, 0.1]


def test_hand_empty_list_forwarded_as_empty():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits())
    safe.send_hand([])
    assert mock.hand_calls[-1] == []


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_name_position_length_mismatch_raises():
    mock = MockInner()
    safe = SafeActuator(mock, SafetyLimits(blocked_joints=frozenset()))
    with pytest.raises(ValueError, match="長さ不一致"):
        safe.send_joint_trajectory(["a", "b"], [0.1])


# ---------------------------------------------------------------------------
# default limits
# ---------------------------------------------------------------------------


def test_default_limits_match_motion_adapter_defaults():
    """motion_adapter (`_deprecated/g1_motion_adapter/...`) と同じ default 値を継承。"""
    limits = SafetyLimits()
    assert limits.joint_max_abs == 1.5
    assert limits.hand_max_abs == 1.0
    assert limits.blocked_joints == DEFAULT_BLOCKED_JOINTS


def test_none_limits_uses_default():
    mock = MockInner()
    safe = SafeActuator(mock, limits=None)
    safe.send_joint_trajectory(["arm_1"], [5.0])
    # default joint_max_abs=1.5 で clamp
    assert mock.joint_calls[-1] == (["arm_1"], [1.5])
