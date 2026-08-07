"""SkillDispatchLowerPolicy の unit tests。

dispatch routing の契約:
  - registry key と skill.name の一致を fail-fast (init)
  - 同 skill 再 start は no-op (caller から見て idempotent)
  - 別 skill への遷移は「旧を stop → 新を start」の順
  - DONE_SENTINEL は現 active を stop
  - 未知 skill 名は KeyError
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.mock import MockWalkActuator
from inference.desktop.lower_policy.dispatcher import (
    DONE_SENTINEL,
    SkillDispatchLowerPolicy,
)
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.mock import MockSkill
from inference.desktop.lower_policy.skills.move_to_table import MoveToTable


class DummySkill(Skill):
    """Dispatch routing の test 用 stub。actuator に依存せず、event ログのみ。"""

    def __init__(self, skill_name: str) -> None:
        super().__init__()
        self.name = skill_name  # type: ignore[misc]
        self.events: list[str] = []

    def _on_start(self, params: dict) -> None:
        self.events.append("start")

    def _on_stop(self) -> None:
        self.events.append("stop")


def make_dispatcher() -> tuple[SkillDispatchLowerPolicy, MockWalkActuator]:
    """MoveToTable(vx=0.1) を積んだ dispatcher と actuator の pair を返す。"""
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator, vx=0.1)
    return SkillDispatchLowerPolicy({"move_to_table": skill}), actuator


def test_registry_key_must_match_skill_name():
    actuator = MockWalkActuator()
    skill = MoveToTable(actuator)  # skill.name = "move_to_table"
    with pytest.raises(ValueError, match="skill.name mismatch"):
        SkillDispatchLowerPolicy({"other_name": skill})


def test_start_dispatches_to_skill():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    assert actuator.events == [("set_velocity", 0.1, 0.0, 0.0)]
    assert policy.active_skill_name == "move_to_table"


def test_active_skill_returns_instance(monkeypatch):
    """dispatcher.active_skill が Skill instance を返す (Issue #81、Orchestrator が
    Skill.max_dwell_sec を lookup するために使う)。"""
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy

    skill_a = MockSkill("skill_a", max_dwell_sec=1.5)
    skill_b = MockSkill("skill_b")  # None
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a, "skill_b": skill_b})
    # 未 start
    assert policy.active_skill is None
    # start 後
    policy.start("skill_a", {})
    assert policy.active_skill is skill_a
    assert policy.active_skill.max_dwell_sec == 1.5
    # switch
    policy.start("skill_b", {})
    assert policy.active_skill is skill_b
    assert policy.active_skill.max_dwell_sec is None
    # stop
    policy.stop()
    assert policy.active_skill is None


def test_stop_after_start():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    policy.stop()
    assert actuator.events == [
        ("set_velocity", 0.1, 0.0, 0.0),
        ("set_velocity", 0.0, 0.0, 0.0, 1.0),
    ]
    assert policy.active_skill_name is None


def test_stop_without_active_is_noop():
    policy, actuator = make_dispatcher()
    policy.stop()
    assert actuator.events == []
    assert policy.active_skill_name is None


def test_restart_same_skill_is_noop():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    n = len(actuator.events)
    policy.start("move_to_table", {})
    assert len(actuator.events) == n
    assert policy.active_skill_name == "move_to_table"


def test_done_sentinel_stops_active_skill():
    policy, actuator = make_dispatcher()
    policy.start("move_to_table", {})
    policy.start(DONE_SENTINEL, {})
    assert policy.active_skill_name is None
    # active skill stop の events もこの順で出ている
    assert actuator.events[-1:] == [("set_velocity", 0.0, 0.0, 0.0, 1.0)]


def test_done_sentinel_when_no_active_is_noop():
    policy, actuator = make_dispatcher()
    policy.start(DONE_SENTINEL, {})
    assert actuator.events == []
    assert policy.active_skill_name is None


def test_unknown_skill_raises_keyerror():
    policy, _ = make_dispatcher()
    with pytest.raises(KeyError, match="unknown skill"):
        policy.start("unknown_skill", {})


def test_unknown_skill_does_not_stop_active_skill():
    """caller が未知 skill を渡した時、現 active skill は stop されない
    (atomic 挙動)。KeyError 検証を「old skill を stop する前」に行うことで
    state 不一致を防ぐ。
    """
    skill_a = DummySkill("skill_a")
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a})
    policy.start("skill_a", {})
    assert skill_a.events == ["start"]
    with pytest.raises(KeyError):
        policy.start("unknown_skill", {})
    # old skill は stop されていない
    assert skill_a.events == ["start"]
    assert policy.active_skill_name == "skill_a"


def test_transition_to_different_skill_stops_first():
    """A → B の遷移: A の stop → B の start がこの順で起きる。"""
    skill_a = DummySkill("skill_a")
    skill_b = DummySkill("skill_b")
    policy = SkillDispatchLowerPolicy({"skill_a": skill_a, "skill_b": skill_b})

    policy.start("skill_a", {})
    assert skill_a.events == ["start"]
    assert skill_b.events == []

    policy.start("skill_b", {})
    assert skill_a.events == ["start", "stop"]
    assert skill_b.events == ["start"]
    assert policy.active_skill_name == "skill_b"


# =========================================================================
# Skill.step + Dispatcher.step (Type A / Type B の pass-through 検証)
# =========================================================================
class _TypeBStub(Skill):
    """VLA / GR00T 系を模した stub。step で固定 action tensor を返す。"""

    def __init__(self, skill_name: str, action: np.ndarray) -> None:
        super().__init__()
        self.name = skill_name  # type: ignore[misc]
        self._action = action

    def _on_start(self, params: dict) -> None:
        pass

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> Optional[np.ndarray]:
        return self._action


def test_dispatcher_step_returns_none_when_no_active():
    """未 start 時、Dispatcher.step は None を返す。"""
    policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
    assert policy.step({"any": "obs"}) is None


def test_dispatcher_step_type_a_returns_none():
    """Type A (Skill.step default): 自前で actuator を叩く前提、caller には None。"""
    policy = SkillDispatchLowerPolicy({"m": MockSkill("m")})
    policy.start("m", {})
    assert policy.step({"any": "obs"}) is None


def test_dispatcher_step_type_b_returns_action_tensor():
    """Type B (step override): action tensor を pass-through で返す。"""
    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    stub = _TypeBStub("vla_skill", action=action)
    policy = SkillDispatchLowerPolicy({"vla_skill": stub})
    policy.start("vla_skill", {})
    out = policy.step({"obs": "any"})
    assert out is not None
    np.testing.assert_array_equal(out, action)


def test_dispatcher_step_switches_after_transition():
    """Type B → Type A へ切替後、以降 step は None (Type A の default)。"""
    action = np.array([1.0], dtype=np.float32)
    type_b = _TypeBStub("skill_b", action=action)
    type_a = MockSkill("skill_a")
    policy = SkillDispatchLowerPolicy({"skill_a": type_a, "skill_b": type_b})

    policy.start("skill_b", {})
    np.testing.assert_array_equal(policy.step({}), action)

    policy.start("skill_a", {})
    assert policy.step({}) is None
