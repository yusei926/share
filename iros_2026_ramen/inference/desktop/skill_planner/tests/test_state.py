"""SkillState.update / .transition の unit test。"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.state import SkillState


def _make_det(
    class_name: str,
    verts: np.ndarray,
    class_id: int = 0,
    confidence: float = 0.9,
) -> OBBDetection:
    return OBBDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        verts=verts.astype(np.float32),
    )


def _box(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """axis-aligned bbox を OBB 4 頂点 (時計回り) で返す。"""
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )


# =========================================================================
# update: 履歴 flag latch
# =========================================================================
class TestUpdate:
    def test_no_latch_outside_cycle_phases(self):
        """move_to_table 中は左手 ∩ leg overlap があっても latch しない。"""
        state = SkillState(current_skill="move_to_table", n_legs_completed=0)
        dets = [
            _make_det("hand_left", _box(0.10, 0.10, 0.50, 0.50)),
            _make_det("leg", _box(0.20, 0.20, 0.55, 0.55)),  # 大きく重なる
        ]
        state.update(dets)
        assert state.left_hand_touched_leg_since_pick is False

    def test_no_latch_outside_move_table_base(self):
        """move_table_base 中も latch しない。"""
        state = SkillState(current_skill="move_table_base", n_legs_completed=1)
        dets = [
            _make_det("hand_left", _box(0.10, 0.10, 0.50, 0.50)),
            _make_det("leg", _box(0.20, 0.20, 0.55, 0.55)),
        ]
        state.update(dets)
        assert state.left_hand_touched_leg_since_pick is False

    def test_latch_left_hand_touched_leg_within_cycle(self):
        """pick_table_leg 中に左手 ∩ leg overlap → latch True。"""
        state = SkillState(current_skill="pick_table_leg", n_legs_completed=0)
        dets = [
            _make_det("hand_left", _box(0.10, 0.10, 0.50, 0.50)),
            _make_det("leg", _box(0.20, 0.20, 0.55, 0.55)),
        ]
        state.update(dets)
        assert state.left_hand_touched_leg_since_pick is True

    def test_latch_persists_across_frames(self):
        """一度 latch した flag は次 frame の overlap 無しでも True 維持。"""
        state = SkillState(current_skill="pick_table_leg", n_legs_completed=0)
        # frame T: overlap あり → latch
        dets_hit = [
            _make_det("hand_left", _box(0.10, 0.10, 0.50, 0.50)),
            _make_det("leg", _box(0.20, 0.20, 0.55, 0.55)),
        ]
        state.update(dets_hit)
        assert state.left_hand_touched_leg_since_pick is True
        # frame T+1: overlap 無し (離れた位置) → 依然 True
        dets_miss = [
            _make_det("hand_left", _box(0.10, 0.10, 0.20, 0.20)),
            _make_det("leg", _box(0.80, 0.80, 0.95, 0.95)),
        ]
        state.update(dets_miss)
        assert state.left_hand_touched_leg_since_pick is True

    def test_latch_leg_tip_touched_hole(self):
        state = SkillState(current_skill="insert_table_leg", n_legs_completed=0)
        dets = [
            _make_det("leg_tip", _box(0.30, 0.30, 0.40, 0.40)),
            _make_det("hole", _box(0.32, 0.32, 0.42, 0.42)),
        ]
        state.update(dets)
        assert state.leg_tip_touched_hole_since_pick is True

    def test_latch_left_right_hand_overlap(self):
        state = SkillState(
            current_skill="rotate_leg_to_tighten", n_legs_completed=0
        )
        dets = [
            _make_det("hand_left", _box(0.10, 0.10, 0.30, 0.30)),
            _make_det("hand_right", _box(0.12, 0.12, 0.32, 0.32)),
        ]
        state.update(dets)
        assert state.left_right_hand_overlapped_since_pick is True

    def test_update_missing_class_no_change(self):
        """該当 class が無検出 → latch されない (0.0 IoU で threshold 未達)。"""
        state = SkillState(current_skill="pick_table_leg", n_legs_completed=0)
        state.update([])  # 空 detection
        assert state.left_hand_touched_leg_since_pick is False
        assert state.leg_tip_touched_hole_since_pick is False
        assert state.left_right_hand_overlapped_since_pick is False


# =========================================================================
# transition: skill 切替 + 副作用
# =========================================================================
class TestTransition:
    def test_transition_updates_current_skill(self):
        state = SkillState(current_skill="move_to_table", n_legs_completed=0)
        state.transition("move_table_base")
        assert state.current_skill == "move_table_base"

    def test_pick_transition_resets_all_latch_flags(self):
        """pick_table_leg に遷移 → 3 履歴 flag が全 reset される。"""
        state = SkillState(
            current_skill="move_table_base",
            n_legs_completed=1,
            left_hand_touched_leg_since_pick=True,
            leg_tip_touched_hole_since_pick=True,
            left_right_hand_overlapped_since_pick=True,
        )
        state.transition("pick_table_leg")
        assert state.left_hand_touched_leg_since_pick is False
        assert state.leg_tip_touched_hole_since_pick is False
        assert state.left_right_hand_overlapped_since_pick is False

    def test_non_pick_transition_preserves_flags(self):
        """pick 以外への遷移では履歴 flag を保持。"""
        state = SkillState(
            current_skill="pick_table_leg",
            n_legs_completed=0,
            left_hand_touched_leg_since_pick=True,
        )
        state.transition("insert_table_leg")
        assert state.left_hand_touched_leg_since_pick is True

    def test_n_legs_increment_after_rotate(self):
        """prev == rotate_leg_to_tighten → n_legs_completed +1 (move_table_base)。"""
        state = SkillState(
            current_skill="rotate_leg_to_tighten", n_legs_completed=0
        )
        state.transition("move_table_base")
        assert state.n_legs_completed == 1

    def test_n_legs_increment_after_rotate_to_flip(self):
        """prev == rotate_leg_to_tighten → n_legs_completed +1 (flip_table)。
        rotate→flip でも rotate 完了扱いで +1 (4 leg 完成の flip 遷移)。"""
        state = SkillState(
            current_skill="rotate_leg_to_tighten", n_legs_completed=3
        )
        state.transition("flip_table")
        assert state.n_legs_completed == 4

    def test_n_legs_not_incremented_from_non_rotate(self):
        """prev != rotate → n_legs_completed 不変。"""
        state = SkillState(
            current_skill="pick_table_leg", n_legs_completed=1
        )
        state.transition("insert_table_leg")
        assert state.n_legs_completed == 1

    def test_transition_snapshot_from_ctx(self):
        """ctx に base_rotation_start_table_top_verts があれば snapshot 反映。"""
        state = SkillState(
            current_skill="rotate_leg_to_tighten", n_legs_completed=0
        )
        verts = np.array(
            [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]], dtype=np.float32
        )
        state.transition(
            "move_table_base",
            ctx={"base_rotation_start_table_top_verts": verts},
        )
        assert state.base_rotation_start_table_top_verts is not None
        np.testing.assert_array_equal(
            state.base_rotation_start_table_top_verts, verts
        )

    def test_transition_no_ctx_preserves_snapshot(self):
        """ctx=None なら既存の snapshot を保持。"""
        verts_old = np.array(
            [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]], dtype=np.float32
        )
        state = SkillState(
            current_skill="move_table_base",
            n_legs_completed=1,
            base_rotation_start_table_top_verts=verts_old,
        )
        state.transition("pick_table_leg")  # ctx なし
        np.testing.assert_array_equal(
            state.base_rotation_start_table_top_verts, verts_old
        )

    def test_transition_pick_and_rotate_combined(self):
        """1 pick cycle 全体で: pick → insert → rotate → move_base の副作用連鎖。"""
        state = SkillState(current_skill="move_table_base", n_legs_completed=0)
        # pick 遷移: 3 flag reset
        state.transition("pick_table_leg")
        assert state.n_legs_completed == 0
        # update で latch (dummy)
        state.left_hand_touched_leg_since_pick = True
        # insert 遷移: 何も変わらない
        state.transition("insert_table_leg")
        assert state.left_hand_touched_leg_since_pick is True
        assert state.n_legs_completed == 0
        # rotate 遷移: 何も変わらない
        state.transition("rotate_leg_to_tighten")
        assert state.n_legs_completed == 0
        # move_base 遷移: prev==rotate なので n_legs +1
        state.transition("move_table_base")
        assert state.n_legs_completed == 1
        assert state.left_hand_touched_leg_since_pick is True  # まだ保持

    def test_second_pick_cycle_flag_reset(self):
        """2 pick cycle 目の開始で 1 cycle 目の履歴 flag が全 reset。"""
        state = SkillState(current_skill="move_table_base", n_legs_completed=1)
        state.left_hand_touched_leg_since_pick = True
        state.leg_tip_touched_hole_since_pick = True
        state.left_right_hand_overlapped_since_pick = True
        state.transition("pick_table_leg")
        assert state.left_hand_touched_leg_since_pick is False
        assert state.leg_tip_touched_hole_since_pick is False
        assert state.left_right_hand_overlapped_since_pick is False
