"""MoveToTable skill の unit tests。

lifecycle 契約:
  - start は非 idempotent (二重 start → RuntimeError)
  - stop は idempotent (未 start / 停止済でも安全)
  - stop 後の再 start は OK
"""

from __future__ import annotations

import pytest

from inference.desktop.lower_policy.actuators.mock import MockWalkActuator
from inference.desktop.lower_policy.skills.move_to_table import (
    DEFAULT_WALK_VX,
    MoveToTable,
)


def test_name():
    skill = MoveToTable(MockWalkActuator())
    assert skill.name == "move_to_table"


def test_default_vx_is_verified_real_robot_value():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.start({})
    assert actuator.events == [("set_velocity", DEFAULT_WALK_VX, 0.0, 0.0)]


def test_start_issues_move_without_posture_transition():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.1)
    skill.start({})
    assert actuator.events == [("set_velocity", 0.1, 0.0, 0.0)]
    assert skill.is_active


def test_stop_issues_only_zero_velocity():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.1)
    skill.start({})
    n = len(actuator.events)
    skill.stop()
    assert actuator.events[n:] == [("set_velocity", 0.0, 0.0, 0.0, 1.0)]
    assert not skill.is_active


def test_custom_vx_used_on_start():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.05)
    skill.start({})
    assert ("set_velocity", 0.05, 0.0, 0.0) in actuator.events


def test_max_dwell_sec_is_forwarded_to_actuator_duration():
    """max_dwell_sec を SDK set_velocity の duration にも渡す (host / firmware 同期、Issue #81)。"""
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.15, max_dwell_sec=3.0)
    skill.start({})
    assert actuator.events == [("set_velocity", 0.15, 0.0, 0.0, 3.0)]


def test_max_dwell_sec_exposed_as_property():
    """Skill.max_dwell_sec property が constructor 値を返す (Orchestrator lookup 用、Issue #81)。"""
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.15, max_dwell_sec=1.5)
    assert skill.max_dwell_sec == 1.5


def test_max_dwell_sec_none_by_default():
    """max_dwell_sec 未指定なら None (fail-safe 無し、Skill 基底 class の default)。"""
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    assert skill.max_dwell_sec is None


# ---------------------------------------------------------------------------
# from_config: dict (skill_config.yaml の skills.move_to_table section) を受ける
# ---------------------------------------------------------------------------
def test_from_config_reads_vx_and_max_dwell_sec():
    """YAML 由来の dict から vx と max_dwell_sec を注入できる (Phase 2b)。"""
    actuator = MockWalkActuator()
    skill = MoveToTable.from_config(
        {"vx": 0.185, "max_dwell_sec": 1.0}, actuator
    )
    assert skill.max_dwell_sec == 1.0
    # start して events で vx が渡ってるか verify
    skill.start({})
    assert actuator.events[0] == ("set_velocity", 0.185, 0.0, 0.0, 1.0)


def test_from_config_missing_vx_uses_default():
    """vx 未指定なら DEFAULT_WALK_VX が使われる。"""
    from inference.desktop.lower_policy.skills.move_to_table import DEFAULT_WALK_VX

    actuator = MockWalkActuator()
    skill = MoveToTable.from_config({"max_dwell_sec": 1.0}, actuator)
    skill.start({})
    assert actuator.events[0][1] == DEFAULT_WALK_VX


def test_from_config_missing_max_dwell_sec_is_none():
    """max_dwell_sec 未指定なら None (fail-safe 無し、SDK duration も None)。"""
    actuator = MockWalkActuator()
    skill = MoveToTable.from_config({"vx": 0.1}, actuator)
    assert skill.max_dwell_sec is None


def test_from_config_null_max_dwell_sec_is_none():
    """YAML の `max_dwell_sec: null` は None として扱う。"""
    actuator = MockWalkActuator()
    skill = MoveToTable.from_config(
        {"vx": 0.1, "max_dwell_sec": None}, actuator
    )
    assert skill.max_dwell_sec is None


def test_from_config_non_positive_max_dwell_sec_raises():
    """max_dwell_sec <= 0 は ValueError (early validation)。"""
    actuator = MockWalkActuator()
    with pytest.raises(ValueError, match="max_dwell_sec must be > 0"):
        MoveToTable.from_config({"max_dwell_sec": 0.0}, actuator)
    with pytest.raises(ValueError, match="max_dwell_sec must be > 0"):
        MoveToTable.from_config({"max_dwell_sec": -1.0}, actuator)


def test_from_config_non_mapping_raises():
    """cfg が dict でなければ ValueError。"""
    actuator = MockWalkActuator()
    with pytest.raises(ValueError, match="must be a mapping"):
        MoveToTable.from_config([1, 2, 3], actuator)  # type: ignore[arg-type]


def test_double_start_raises():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.start({})
    with pytest.raises(RuntimeError, match="already active"):
        skill.start({})


def test_double_stop_is_idempotent():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.start({})
    skill.stop()
    n = len(actuator.events)
    skill.stop()  # no-op
    assert len(actuator.events) == n
    assert not skill.is_active


def test_stop_without_start_is_idempotent():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.stop()
    assert actuator.events == []
    assert not skill.is_active


def test_restart_after_stop_works():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.start({})
    skill.stop()
    skill.start({})  # 再 start OK (二重 start ではない)
    assert skill.is_active


def test_params_are_accepted_but_ignored():
    """MoveToTable は現仕様で params (leg_index / retry) を無視する。
    signature 互換のため受け取るが、events には影響しない。
    """
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)
    skill.start({"leg_index": 3, "retry": True})
    assert actuator.events == [("set_velocity", DEFAULT_WALK_VX, 0.0, 0.0)]
