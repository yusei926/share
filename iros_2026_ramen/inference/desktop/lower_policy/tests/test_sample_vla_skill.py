"""SampleVLASkill unit test (Issue #75、Phase 4 で default 廃止)。

- Type B contract: step() が np.ndarray を返す (shape (14,))
- fixed_pose の shape validation (14-D 以外は error)
- lifecycle (start / stop / step の順、二重 start が error)
- fixed_pose は **required** (Issue #81 Phase 4、default 廃止)
- from_config で YAML 経由の sparse pose 注入
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1_NUM_ARM_JOINTS
from inference.desktop.lower_policy.skills.sample_vla_skill import SampleVLASkill


def _dummy_obs() -> dict:
    """obs は step で無視されるので中身適当。"""
    return {
        "head_rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "joint_state": None,
        "cleaned": [],
        "t": 0,
    }


def _zero_pose() -> np.ndarray:
    """14-D ゼロ pose (test 用の neutral)。"""
    return np.zeros(G1_NUM_ARM_JOINTS, dtype=np.float64)


class TestConstructor:
    def test_fixed_pose_is_stored_and_returned(self):
        """fixed_pose がそのまま step で返る (VLA drop-in の骨格)。"""
        pose = np.linspace(-0.5, 0.5, G1_NUM_ARM_JOINTS)
        skill = SampleVLASkill("pick_table_leg", fixed_pose=pose)
        skill.start({})
        action = skill.step(_dummy_obs())
        assert np.allclose(action, pose)

    def test_fixed_pose_is_required(self):
        """Phase 4 で default 廃止、fixed_pose 省略は TypeError。"""
        with pytest.raises(TypeError):
            SampleVLASkill("pick_table_leg")  # type: ignore[call-arg]

    def test_wrong_pose_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SampleVLASkill("s", fixed_pose=np.zeros(13))
        with pytest.raises(ValueError, match="shape"):
            SampleVLASkill("s", fixed_pose=np.zeros(29))


class TestStepReturnsActionTensor:
    def test_step_returns_ndarray_of_correct_shape(self):
        """Type B contract: step が np.ndarray (14,) を返す (None を返す Type A と区別)。"""
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.start({})
        action = skill.step(_dummy_obs())
        assert isinstance(action, np.ndarray)
        assert action.shape == (G1_NUM_ARM_JOINTS,)

    def test_step_ignores_obs_in_mock_impl(self):
        """Mock 実装は obs 内容によらず同じ pose を返す (real VLA 実装で差し替えられる)。"""
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.start({})
        # 全く違う obs 2 個を試して同じ返り値になるか
        obs_a = _dummy_obs()
        obs_b = _dummy_obs()
        obs_b["head_rgb"] = np.full((10, 10, 3), 200, dtype=np.uint8)
        obs_b["t"] = 999_999
        action_a = skill.step(obs_a)
        action_b = skill.step(obs_b)
        assert np.allclose(action_a, action_b)

    def test_step_returns_fresh_copy_each_call(self):
        """呼び出し元が受け取った array を書き換えても、次 step で影響しない (defensive copy)。"""
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.start({})
        a1 = skill.step(_dummy_obs())
        a1[0] = 999.0  # caller が書き換え
        a2 = skill.step(_dummy_obs())
        assert a2[0] != 999.0  # 内部 pose は無傷


class TestLifecycle:
    def test_start_marks_active(self):
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        assert not skill.is_active
        skill.start({})
        assert skill.is_active

    def test_double_start_raises(self):
        """base Skill の contract: 既 active で start を呼ぶと RuntimeError。"""
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.start({})
        with pytest.raises(RuntimeError, match="already active"):
            skill.start({})

    def test_stop_is_idempotent(self):
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.stop()  # 未 start でも OK
        skill.start({})
        skill.stop()
        skill.stop()  # 二重 stop も OK
        assert not skill.is_active

    def test_restart_after_stop(self):
        """stop 後に再 start できる (base Skill の docstring 通り)。"""
        skill = SampleVLASkill("test_skill", fixed_pose=_zero_pose())
        skill.start({})
        skill.stop()
        skill.start({})  # RuntimeError 出ない
        assert skill.is_active


# ---------------------------------------------------------------------------
# from_config: dict (skill_config.yaml の skills.<name> section) を受ける (Phase 2b)
# ---------------------------------------------------------------------------
class TestFromConfig:
    def test_from_config_reads_sparse_default_pose_rad(self):
        """YAML の sparse pose_rad が 14-D dense pose として skill に注入される。"""
        skill = SampleVLASkill.from_config(
            {"default_pose_rad": {"L.shoulder_roll": 1.2, "R.shoulder_roll": -1.2}},
            skill_name="move_table_base",
        )
        assert skill.name == "move_table_base"
        skill.start({})
        action = skill.step(_dummy_obs())
        assert action.shape == (14,)
        assert action[1] == pytest.approx(1.2)   # L.shoulder_roll
        assert action[8] == pytest.approx(-1.2)  # R.shoulder_roll
        assert np.count_nonzero(action) == 2

    def test_from_config_missing_pose_defaults_to_zeros(self):
        """default_pose_rad 未指定なら 14-D ゼロ姿勢。"""
        skill = SampleVLASkill.from_config({}, skill_name="mock_skill")
        skill.start({})
        action = skill.step(_dummy_obs())
        assert action.shape == (14,)
        assert np.all(action == 0.0)

    def test_from_config_unknown_joint_raises(self):
        with pytest.raises(ValueError, match="unknown joint"):
            SampleVLASkill.from_config(
                {"default_pose_rad": {"L.finger": 0.5}}, skill_name="s"
            )

    def test_from_config_non_mapping_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            SampleVLASkill.from_config([1, 2, 3], skill_name="s")  # type: ignore[arg-type]
