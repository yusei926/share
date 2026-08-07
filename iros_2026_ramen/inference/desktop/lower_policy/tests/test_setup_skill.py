"""SetupSkill unit test (Issue #81)。

- Type B contract: step() が np.ndarray (14,) を返す
- 時間ベース stage 切り替え (fake time_fn で決定論的に verify)
- YAML load (real skill_config.yaml round-trip + fixture-based error cases)
- sparse pose_rad の densify (指定 joint 以外は 0.0)
- Skill 基底 class の lifecycle (start / stop / restart / 二重 start)
- total_dwell_sec / defensive copy / safety limit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1_NUM_ARM_JOINTS
from inference.desktop.lower_policy.pose_utils import (
    JOINT_NAMES,
    densify_pose as _densify_pose,
)
from inference.desktop.lower_policy.skills.setup_skill import (
    SetupSkill,
    SetupStage,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_CONFIG_YAML = (
    REPO_ROOT
    / "inference"
    / "desktop"
    / "lower_policy"
    / "configs"
    / "skill_config.yaml"
)


def _dummy_obs() -> dict:
    return {"head_rgb": None, "cleaned": [], "t": 0}


class FakeClock:
    """test 用の決定論的 monotonic clock。tick(sec) で任意に時間を進める。"""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def tick(self, sec: float) -> None:
        self._t += sec


def _pose(**joint_values: float) -> np.ndarray:
    """joint short label → dense 14-D array (test の可読性用)。"""
    return _densify_pose(dict(joint_values), context="test")


def _default_stages() -> list[SetupStage]:
    """典型的な 2 段 stage (production YAML と同じ shape)。"""
    return [
        SetupStage("lateral", 2.0, _pose(**{"L.shoulder_roll": 1.2, "R.shoulder_roll": -1.2})),
        SetupStage(
            "forward",
            2.0,
            _pose(**{
                "L.shoulder_pitch": -0.2, "L.shoulder_roll": 1.2,
                "R.shoulder_pitch": -0.2, "R.shoulder_roll": -1.2,
            }),
        ),
    ]


# ---------------------------------------------------------------------------
# JOINT_NAMES / _densify_pose
# ---------------------------------------------------------------------------
class TestJointNamesAndDensify:
    def test_joint_names_has_14_entries_and_are_unique(self):
        assert len(JOINT_NAMES) == G1_NUM_ARM_JOINTS == 14
        assert len(set(JOINT_NAMES)) == 14

    def test_densify_pose_empty_dict_returns_zeros(self):
        pose = _densify_pose({}, context="s")
        assert pose.shape == (14,)
        assert pose.dtype == np.float64
        assert np.all(pose == 0.0)

    def test_densify_pose_places_values_at_correct_indices(self):
        pose = _densify_pose({"L.shoulder_roll": 1.2, "R.elbow": -0.5}, context="s")
        assert pose[JOINT_NAMES.index("L.shoulder_roll")] == pytest.approx(1.2)
        assert pose[JOINT_NAMES.index("R.elbow")] == pytest.approx(-0.5)
        # 明記していない joint は 0.0
        assert np.count_nonzero(pose) == 2

    def test_densify_pose_unknown_joint_raises(self):
        with pytest.raises(ValueError, match="unknown joint"):
            _densify_pose({"L.finger": 0.5}, context="s")

    def test_densify_pose_non_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _densify_pose([1, 2, 3], context="s")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SetupStage validation
# ---------------------------------------------------------------------------
class TestSetupStageValidation:
    def test_valid_stage_constructs(self):
        s = SetupStage("s", 1.0, np.zeros(14))
        assert s.name == "s"

    def test_non_positive_dwell_raises(self):
        with pytest.raises(ValueError, match="dwell_sec must be > 0"):
            SetupStage("s", 0.0, np.zeros(14))
        with pytest.raises(ValueError, match="dwell_sec must be > 0"):
            SetupStage("s", -1.0, np.zeros(14))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="must have shape"):
            SetupStage("s", 1.0, np.zeros(13))

    def test_nan_inf_raises(self):
        pose = np.zeros(14)
        pose[0] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            SetupStage("s", 1.0, pose)

    def test_exceeds_safety_limit_raises(self):
        pose = np.zeros(14)
        pose[0] = 2.0  # > 1.5 rad
        with pytest.raises(ValueError, match="safety limit"):
            SetupStage("s", 1.0, pose)


# ---------------------------------------------------------------------------
# SetupSkill construction & properties
# ---------------------------------------------------------------------------
class TestSetupSkillConstruction:
    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SetupSkill([])

    def test_default_name_is_setup(self):
        skill = SetupSkill(_default_stages())
        assert skill.name == "setup"

    def test_custom_skill_name(self):
        skill = SetupSkill(_default_stages(), skill_name="custom_setup")
        assert skill.name == "custom_setup"

    def test_total_dwell_sec_is_sum(self):
        skill = SetupSkill(_default_stages())
        assert skill.total_dwell_sec == pytest.approx(4.0)

    def test_stages_property_returns_tuple(self):
        stages = _default_stages()
        skill = SetupSkill(stages)
        assert skill.stages == tuple(stages)


# ---------------------------------------------------------------------------
# Type B contract: step returns 14-D ndarray
# ---------------------------------------------------------------------------
class TestStepContract:
    def test_step_returns_ndarray_of_correct_shape(self):
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        action = skill.step(_dummy_obs())
        assert isinstance(action, np.ndarray)
        assert action.shape == (14,)
        assert action.dtype == np.float64

    def test_step_before_start_returns_first_stage_pose(self):
        """_on_start されていない場合の safe default = 先頭 stage pose。"""
        skill = SetupSkill(_default_stages(), time_fn=FakeClock())
        action = skill.step(_dummy_obs())
        assert np.allclose(action, _default_stages()[0].pose_rad)

    def test_step_returns_fresh_copy_each_call(self):
        """caller が action array を書き換えても内部 stage pose に影響しない。"""
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        a1 = skill.step(_dummy_obs())
        a1[0] = 999.0
        a2 = skill.step(_dummy_obs())
        assert a2[0] != 999.0


# ---------------------------------------------------------------------------
# 時間ベース stage 切り替え (FakeClock で決定論的 verify)
# ---------------------------------------------------------------------------
class TestStageTransition:
    def test_first_stage_at_elapsed_zero(self):
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        # elapsed=0
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[0].pose_rad)

    def test_first_stage_just_before_boundary(self):
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        clock.tick(1.999)  # < 2.0 → まだ stage 0
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[0].pose_rad)

    def test_second_stage_at_first_boundary(self):
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        clock.tick(2.0)  # 境界 = 次 stage の先頭
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[1].pose_rad)

    def test_holds_last_stage_after_total_dwell(self):
        """全 stage 消化後は最終 stage の pose を返し続ける (arm_actuator が hold)。"""
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        clock.tick(100.0)  # 4.0s の遥か先
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[1].pose_rad)

    def test_single_stage_holds_indefinitely(self):
        clock = FakeClock()
        only = [SetupStage("only", 3.0, _pose(**{"L.elbow": -0.5}))]
        skill = SetupSkill(only, time_fn=clock)
        skill.start({})
        for t in (0.0, 1.0, 2.99, 3.0, 10.0):
            clock._t = t
            assert np.allclose(skill.step(_dummy_obs()), only[0].pose_rad)


# ---------------------------------------------------------------------------
# Skill 基底 class lifecycle
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_start_marks_active(self):
        skill = SetupSkill(_default_stages(), time_fn=FakeClock())
        assert not skill.is_active
        skill.start({})
        assert skill.is_active

    def test_double_start_raises(self):
        skill = SetupSkill(_default_stages(), time_fn=FakeClock())
        skill.start({})
        with pytest.raises(RuntimeError, match="already active"):
            skill.start({})

    def test_stop_is_idempotent(self):
        skill = SetupSkill(_default_stages(), time_fn=FakeClock())
        skill.stop()  # 未 start でも OK
        skill.start({})
        skill.stop()
        skill.stop()  # 二重 stop も OK
        assert not skill.is_active

    def test_restart_resets_start_time(self):
        clock = FakeClock()
        skill = SetupSkill(_default_stages(), time_fn=clock)
        skill.start({})
        clock.tick(5.0)  # 全 stage 消化
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[1].pose_rad)
        skill.stop()
        clock.tick(1.0)  # 経過は捨てる
        skill.start({})  # 再 start → 時刻リセット
        assert np.allclose(skill.step(_dummy_obs()), _default_stages()[0].pose_rad)


# ---------------------------------------------------------------------------
# from_yaml: real config round-trip + error cases
# ---------------------------------------------------------------------------
class TestFromYaml:
    def test_load_production_skill_config_yaml(self):
        """`configs/skill_config.yaml` が SetupSkill から load できる。"""
        assert SKILL_CONFIG_YAML.is_file(), f"missing: {SKILL_CONFIG_YAML}"
        skill = SetupSkill.from_yaml(SKILL_CONFIG_YAML, time_fn=FakeClock())
        # production YAML は 2 stage、各 2 秒、合計 4 秒
        assert len(skill.stages) == 2
        assert skill.stages[0].name == "lateral_clearance"
        assert skill.stages[1].name == "forward_clearance"
        assert skill.total_dwell_sec == pytest.approx(4.0)
        # Stage 1 は shoulder roll のみ ±1.2
        s1 = skill.stages[0].pose_rad
        assert s1[JOINT_NAMES.index("L.shoulder_roll")] == pytest.approx(1.2)
        assert s1[JOINT_NAMES.index("R.shoulder_roll")] == pytest.approx(-1.2)
        assert np.count_nonzero(s1) == 2
        # Stage 2 は Stage 1 に shoulder pitch -0.2 を追加
        s2 = skill.stages[1].pose_rad
        assert s2[JOINT_NAMES.index("L.shoulder_pitch")] == pytest.approx(-0.2)
        assert s2[JOINT_NAMES.index("R.shoulder_pitch")] == pytest.approx(-0.2)
        assert s2[JOINT_NAMES.index("L.shoulder_roll")] == pytest.approx(1.2)
        assert s2[JOINT_NAMES.index("R.shoulder_roll")] == pytest.approx(-1.2)
        assert np.count_nonzero(s2) == 4

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "cfg.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_missing_skills_section_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, "other: {}\n")
        with pytest.raises(ValueError, match="skills.setup"):
            SetupSkill.from_yaml(p)

    def test_missing_setup_section_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, "skills: {}\n")
        with pytest.raises(ValueError, match="skills.setup"):
            SetupSkill.from_yaml(p)

    def test_missing_stages_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, "skills:\n  setup: {}\n")
        with pytest.raises(ValueError, match="stages"):
            SetupSkill.from_yaml(p)

    def test_empty_stages_list_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, "skills:\n  setup:\n    stages: []\n")
        with pytest.raises(ValueError, match="non-empty"):
            SetupSkill.from_yaml(p)

    def test_missing_dwell_sec_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            "skills:\n  setup:\n    stages:\n"
            "      - name: s1\n        pose_rad: {}\n",
        )
        with pytest.raises(ValueError, match="dwell_sec"):
            SetupSkill.from_yaml(p)

    def test_unknown_joint_name_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            "skills:\n  setup:\n    stages:\n"
            "      - name: s1\n        dwell_sec: 1.0\n"
            "        pose_rad: {L.finger: 0.5}\n",
        )
        with pytest.raises(ValueError, match="unknown joint"):
            SetupSkill.from_yaml(p)

    def test_pose_out_of_safety_limit_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            "skills:\n  setup:\n    stages:\n"
            "      - name: s1\n        dwell_sec: 1.0\n"
            "        pose_rad: {L.elbow: 2.0}\n",
        )
        with pytest.raises(ValueError, match="safety limit"):
            SetupSkill.from_yaml(p)

    def test_negative_dwell_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            "skills:\n  setup:\n    stages:\n"
            "      - name: s1\n        dwell_sec: -1.0\n"
            "        pose_rad: {}\n",
        )
        with pytest.raises(ValueError, match="dwell_sec must be > 0"):
            SetupSkill.from_yaml(p)

    def test_root_not_mapping_raises(self, tmp_path):
        p = self._write_yaml(tmp_path, "- foo\n- bar\n")
        with pytest.raises(ValueError, match="root must be a mapping"):
            SetupSkill.from_yaml(p)

    def test_from_yaml_accepts_time_fn_and_skill_name(self, tmp_path):
        clock = FakeClock()
        skill = SetupSkill.from_yaml(
            SKILL_CONFIG_YAML, time_fn=clock, skill_name="my_setup"
        )
        assert skill.name == "my_setup"
        skill.start({})
        clock.tick(2.5)
        # 2.5s @ stage 0 = 2.0 → stage 1
        assert np.allclose(skill.step(_dummy_obs()), skill.stages[1].pose_rad)
