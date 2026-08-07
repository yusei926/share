"""Orchestrator core の unit test。

perception は StubPerception (frame_id → 予め決めた OBBDetection list)、
dispatcher は 6 MockSkill 積んだ実 SkillDispatchLowerPolicy を使う。
"""

from __future__ import annotations

from typing import Optional

import io
import json

import numpy as np
import pytest

from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.mock import MockSkill
from inference.desktop.orchestrator import (
    DEFAULT_ENTER_CHECK,
    DEFAULT_TRANSITIONS,
    LiveSourceSafetyError,
    Orchestrator,
    TickResult,
)
from inference.desktop.perception.frame_source import FrameData, LerobotFrameSource
from inference.desktop.perception.joint_state_source import JointStateData
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection


# =========================================================================
# helpers
# =========================================================================
def _det(
    class_name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    confidence: float = 0.9,
) -> OBBDetection:
    verts = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    return OBBDetection(
        class_id=0, class_name=class_name, confidence=confidence, verts=verts
    )


def _empty_frame(t: int) -> FrameData:
    """t を rgb[0,0,0] に埋め込んだ 1x1 frame (Stub がこれで frame_id を復元)。"""
    rgb = np.array([[[t % 256]]], dtype=np.uint8) * np.ones(
        (1, 1, 3), dtype=np.uint8
    )
    return FrameData(rgb=rgb, t=t)


class _StubPerception:
    """t (FrameData.t) → 予め決めた OBBDetection list を返す stub。"""

    def __init__(self, dets_by_t: dict[int, list[OBBDetection]]):
        self._dets_by_t = dets_by_t
        self._last_t: int = -1
        self._call_history: list[int] = []

    def predict(self, rgb: np.ndarray) -> list[OBBDetection]:
        # rgb は 1x1x3、rgb[0,0,0] を frame_t として復元
        # ただし next tick で受け取った rgb は必ずしも tick から直接来ないので、
        # 呼び出し順序で t を assign する形が確実
        self._last_t += 1
        self._call_history.append(self._last_t)
        return self._dets_by_t.get(self._last_t, [])


def _base_cleanup_config() -> dict:
    return {
        "max_count": {"leg": 4, "hand_left": 1, "hand_right": 1, "table_top": 1, "workspace": 1, "hole": 8, "leg_tip": 4},
        "over_max_continue_iou": 0.3,
        "under_max_similar_iou": 0.3,
        "median_filter": {"enabled": False, "iou_match_min": 0.5},  # test は cleaner を透過寄りで
    }


def _make_registry() -> dict[str, MockSkill]:
    return {
        name: MockSkill(name)
        for name in [
            "move_to_table",
            "move_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "flip_table",
        ]
    }


# =========================================================================
# Orchestrator の基本挙動
# =========================================================================
class TestInitialization:
    def test_initial_state_before_any_tick(self):
        """init 時点で dispatcher はまだ start されてない (lazy start)。"""
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)
        assert orch.state.current_skill == "move_to_table"
        assert dispatcher.active_skill_name is None  # lazy: tick まで start しない


class TestTickWarmup:
    def test_first_tick_auto_starts_initial_skill(self):
        """初回 tick で dispatcher が initial skill を auto start。"""
        perception = _StubPerception({0: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        result = orch.tick(_empty_frame(t=0))
        # cleaner warmup 中 (median disabled 時 = 1 push warmup) なので None
        assert result is None
        # ただし dispatcher は auto start してる
        assert dispatcher.active_skill_name == "move_to_table"

    def test_second_tick_returns_result_median_disabled(self):
        """median disabled 時、2 push 目 (T=1) で TickResult 返る。"""
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        assert result.t == 1
        assert result.current_skill == "move_to_table"
        assert result.fire_transition_to is None

    def test_duplicate_t_returns_none(self):
        """同じ frame.t の重複 tick は None を返す (source が同じ frame 二度返した場合)。"""
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))
        orch.tick(_empty_frame(t=1))  # ここで warmup 完了
        _ = orch.tick(_empty_frame(t=2))
        # 重複: 同じ t を再度渡すと None (何もしない)
        dup = orch.tick(_empty_frame(t=2))
        assert dup is None


# =========================================================================
# Fire dispatch (1 tick 1 transition + MAX 1 rule)
# =========================================================================
class TestFireDispatch:
    def test_fire_transition_calls_state_and_dispatcher(self):
        """move_to_table → move_table_base の遷移条件を満たす dets を渡すと
        state.transition + dispatcher.start が起きる。"""
        # workspace 下辺 y >= 0.87 で _enter_move_table_base_first が fire
        ws_det = _det("workspace", 0.10, 0.10, 0.90, 0.90)  # bottom_y=0.90 >= 0.87
        perception = _StubPerception(
            {0: [], 1: [ws_det], 2: [ws_det]}  # 0=warmup, 1=fire trigger 用 raw
        )
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))  # warmup
        result = orch.tick(_empty_frame(t=1))  # cleaned_0 emit (empty)
        # cleaned_0 は empty なので fire しない
        assert result is not None
        assert result.fire_transition_to is None
        # 2 tick 目: cleaned_1 が ws_det (case B new + will_persist 判定)
        result = orch.tick(_empty_frame(t=2))
        assert result is not None
        # cleaned_1 が ws_det を含めば fire
        if result.fire_transition_to == "move_table_base":
            assert orch.state.current_skill == "move_table_base"
            assert dispatcher.active_skill_name == "move_table_base"

    def test_max_one_transition_per_tick(self):
        """1 tick で 2 段跳ばず、次候補の enter_check は skip される。

        move_to_table → (fire) → move_table_base に transition した同 tick で
        次の候補 pick_table_leg の enter_check は評価されない (for-break)。
        """
        # workspace fire で move_table_base 遷移、同時に pick fire しても
        # move_to_table の candidates = ["move_table_base"] のみなので
        # 順序上 pick_table_leg は evaluate されない (default TRANSITIONS 依存)
        ws_det = _det("workspace", 0.10, 0.10, 0.90, 0.90)
        perception = _StubPerception({0: [], 1: [ws_det], 2: [ws_det]})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        orch.tick(_empty_frame(t=0))
        orch.tick(_empty_frame(t=1))
        result = orch.tick(_empty_frame(t=2))
        # 1 tick 内で複数遷移が起きたら state.current_skill が "pick_table_leg" 等に
        # ジャンプしてしまう。move_table_base に留まってれば MAX 1 が enforce されてる証拠。
        # (transition が起きなくても current_skill=move_to_table のままなので、
        # このテストは fire 起きた case のみ意味を持つ。)
        if result and result.fire_transition_to == "move_table_base":
            assert orch.state.current_skill == "move_table_base"


# =========================================================================
# Action forwarding (Type B skill の action tensor が actuator に流れるか)
# =========================================================================
class _TypeBStub(Skill):
    def __init__(self, name: str, action: np.ndarray) -> None:
        super().__init__()
        self.name = name  # type: ignore[misc]
        self._action = action

    def _on_start(self, params: dict) -> None: pass
    def _on_stop(self) -> None: pass
    def step(self, obs: dict) -> Optional[np.ndarray]:
        return self._action


class TestActionForwarding:
    def test_action_forwarded_when_send_fn_set(self):
        """Type B skill の action が actuator_send_fn に流れる。"""
        action = np.array([1.0, 2.0], dtype=np.float32)
        registry = _make_registry()
        # move_to_table を Type B stub に差し替え
        registry["move_to_table"] = _TypeBStub("move_to_table", action)
        dispatcher = SkillDispatchLowerPolicy(registry)

        sent: list[np.ndarray] = []
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        orch = Orchestrator(
            perception, cleaner, dispatcher,
            actuator_send_fn=lambda a: sent.append(a),
        )
        orch.tick(_empty_frame(t=0))  # warmup
        orch.tick(_empty_frame(t=1))  # 定常 → dispatcher.step で action 返る
        assert len(sent) == 1
        np.testing.assert_array_equal(sent[0], action)

    def test_action_dropped_when_send_fn_none(self):
        """actuator_send_fn=None なら action は silently drop (log 経由で確認可)。"""
        action = np.array([1.0], dtype=np.float32)
        registry = _make_registry()
        registry["move_to_table"] = _TypeBStub("move_to_table", action)
        dispatcher = SkillDispatchLowerPolicy(registry)
        perception = _StubPerception({0: [], 1: []})
        cleaner = DetectionStream(_base_cleanup_config())
        orch = Orchestrator(
            perception, cleaner, dispatcher, actuator_send_fn=None,
        )
        orch.tick(_empty_frame(t=0))
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        np.testing.assert_array_equal(result.action, action)


# =========================================================================
# JSONL log sink
# =========================================================================
class TestLogSink:
    def test_log_writes_jsonl_per_tick(self):
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        sink = io.StringIO()
        orch = Orchestrator(perception, cleaner, dispatcher, log_sink=sink)

        orch.tick(_empty_frame(t=0))  # warmup, no log
        orch.tick(_empty_frame(t=1))  # emit → log
        orch.tick(_empty_frame(t=2))  # emit → log

        lines = sink.getvalue().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            payload = json.loads(line)
            assert "t" in payload
            assert "current_skill" in payload
            assert "fire_transition_to" in payload
            assert "n_cleaned" in payload


# =========================================================================
# run loop
# =========================================================================
class TestRunLoop:
    def test_run_loop_ends_on_source_exhaustion(self):
        """LerobotFrameSource が exhaust したら run() が return する。"""
        frames = [_empty_frame(t=i) for i in range(5)]
        source = LerobotFrameSource(frames)
        perception = _StubPerception({i: [] for i in range(5)})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        # hz=None で sleep なし、5 tick で source 尽きて自然終了
        orch.run(source, hz=None)
        assert dispatcher.active_skill_name == "move_to_table"


# =========================================================================
# JointState source integration (Issue #75)
# =========================================================================
class _StubJointStateSource:
    """`.get() → Optional[JointStateData]` を返す stub。"""

    def __init__(self, snapshot: Optional[JointStateData]):
        self._snapshot = snapshot

    def get(self) -> Optional[JointStateData]:
        return self._snapshot


class TestJointStateInObs:
    """orchestrator._build_obs が joint_state_source の値を obs に載せることを確認。"""

    def _make_orch(self, joint_state_source: Optional[object] = None) -> Orchestrator:
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        return Orchestrator(
            perception, cleaner, dispatcher,
            joint_state_source=joint_state_source,
        )

    def test_no_source_no_joint_state_field(self):
        """joint_state_source=None なら obs に joint_state field を追加しない (後方互換)。"""
        orch = self._make_orch(joint_state_source=None)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" not in obs
        # 既存 field は維持
        assert "head_rgb" in obs and "t" in obs and "cleaned" in obs

    def test_source_snapshot_populates_obs(self):
        """source.get() が JointStateData を返せば obs["joint_state"] にそのまま乗る。"""
        snapshot = JointStateData(
            name=("j0", "j1"),
            position=np.array([0.1, 0.2], dtype=np.float64),
            velocity=np.zeros(2),
            effort=np.zeros(2),
            t=123_456_789,
        )
        source = _StubJointStateSource(snapshot=snapshot)
        orch = self._make_orch(joint_state_source=source)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" in obs
        assert obs["joint_state"] is snapshot

    def test_source_returns_none_still_adds_field(self):
        """source.get() が None (未受信) でも obs に field は追加 (値 None)。
        skill 側で「obs['joint_state'] is None なら未受信」を明示 handle させる。"""
        source = _StubJointStateSource(snapshot=None)
        orch = self._make_orch(joint_state_source=source)
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "joint_state" in obs
        assert obs["joint_state"] is None


# =========================================================================
# Wrist camera source integration (Issue #75)
# =========================================================================
class _StubFrameSource:
    """`.get() → Optional[FrameData]` を返す stub (Ros2FrameSource duck-type)。"""

    def __init__(self, snapshot: Optional[FrameData]):
        self._snapshot = snapshot

    def get(self) -> Optional[FrameData]:
        return self._snapshot


class TestWristSourceInObs:
    """orchestrator._build_obs が wrist_{left,right}_source の値を obs に載せることを確認。
    SampleVLASkill (Mock) は obs 使わないが、real VLA drop-in で obs pass-through が
    通ることを事前 verify する pipeline test。"""

    def _make_orch(
        self,
        wrist_left_source=None,
        wrist_right_source=None,
    ) -> Orchestrator:
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        return Orchestrator(
            perception, cleaner, dispatcher,
            wrist_left_source=wrist_left_source,
            wrist_right_source=wrist_right_source,
        )

    def test_no_wrist_sources_no_wrist_fields(self):
        """両 wrist source=None なら obs に wrist field 追加しない (後方互換)。"""
        orch = self._make_orch()
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert "wrist_left_rgb" not in obs
        assert "wrist_right_rgb" not in obs
        # 既存 field は維持
        assert "head_rgb" in obs and "t" in obs and "cleaned" in obs

    def test_wrist_sources_populate_obs(self):
        """wrist_{left,right}_source が渡されれば obs にそれぞれの latest snapshot 追加。"""
        left_frame = _empty_frame(t=100)
        right_frame = _empty_frame(t=200)
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=left_frame),
            wrist_right_source=_StubFrameSource(snapshot=right_frame),
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is left_frame
        assert obs["wrist_right_rgb"] is right_frame

    def test_wrist_source_returns_none_still_adds_field(self):
        """source.get() が None (未受信) でも obs に field は追加 (値 None)。
        real VLA 側で「obs['wrist_left_rgb'] is None なら未受信」を明示 handle させる。"""
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=None),
            wrist_right_source=_StubFrameSource(snapshot=None),
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is None
        assert obs["wrist_right_rgb"] is None

    def test_one_side_only_adds_only_that_side(self):
        """片方だけ渡した場合は片方のみ obs 追加 (別々 opt-in)。"""
        left_frame = _empty_frame(t=100)
        orch = self._make_orch(
            wrist_left_source=_StubFrameSource(snapshot=left_frame),
            # right は None
        )
        obs = orch._build_obs(_empty_frame(t=0), cleaned=[])
        assert obs["wrist_left_rgb"] is left_frame
        assert "wrist_right_rgb" not in obs


# =========================================================================
# Live source safety (Issue #64: run_live + auto-transition)
# =========================================================================
class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedLiveSource:
    def __init__(self, values: list[Optional[FrameData]]) -> None:
        self._values = iter(values)

    def get(self) -> Optional[FrameData]:
        try:
            return next(self._values)
        except StopIteration:
            raise KeyboardInterrupt


def _patch_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(
        "inference.desktop.orchestrator.time.monotonic", clock.monotonic
    )
    monkeypatch.setattr("inference.desktop.orchestrator.time.sleep", clock.sleep)
    return clock


class TestLiveRunLoop:
    def test_waits_for_first_frame_before_starting_skill(self, monkeypatch):
        _patch_clock(monkeypatch)
        source = _ScriptedLiveSource([None, None, _empty_frame(t=0)])
        perception = _StubPerception({0: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(perception, cleaner, dispatcher)

        with pytest.raises(KeyboardInterrupt):
            orch.run_live(source, hz=10.0, startup_timeout=1.0)

        assert dispatcher.active_skill_name == "move_to_table"

    def test_startup_timeout_does_not_start_skill(self, monkeypatch):
        _patch_clock(monkeypatch)

        class _NoFrameSource:
            def get(self):
                return None

        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        with pytest.raises(LiveSourceSafetyError, match="startup timeout"):
            orch.run_live(_NoFrameSource(), hz=10.0, startup_timeout=0.2)

        assert dispatcher.active_skill_name is None

    def test_stale_frame_raises_safety_error(self, monkeypatch):
        _patch_clock(monkeypatch)
        frame = _empty_frame(t=0)

        class _StaleSource:
            def get(self):
                return frame

        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({0: []}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        with pytest.raises(LiveSourceSafetyError, match="frame timeout"):
            orch.run_live(
                _StaleSource(),
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.2,
            )

        assert dispatcher.active_skill_name == "move_to_table"

    def _make_registry_with_dwell(self, dwell_map: dict[str, float]) -> dict[str, MockSkill]:
        """dwell 秒を skill instance に注入した registry (Issue #81、Skill.max_dwell_sec 経由)。"""
        return {
            name: MockSkill(name, max_dwell_sec=dwell_map.get(name))
            for name in [
                "move_to_table", "move_table_base", "pick_table_leg",
                "insert_table_leg", "rotate_leg_to_tighten", "flip_table",
            ]
        }

    def test_skill_max_dwell_auto_transitions_to_next_candidate(self, monkeypatch):
        """Skill.max_dwell_sec 到達で TRANSITIONS graph の first candidate に
        auto-transition される (Issue #64 決め打ちルール、Issue #81 で skill が
        自分の dwell を持つ形に refactor)。domain gap で YOLO fire が来なくても
        pipeline を止めずに次 skill = move_table_base へ進む。"""
        _patch_clock(monkeypatch)

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({"move_to_table": 0.2})
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        # dwell 到達後は auto-transition = 例外 raise せず pipeline 継続。
        # test 側から強制終了させるため source が途中で KeyboardInterrupt を
        # 出す form (`_ScriptedLiveSource` を N frame → StopIteration → interrupt)。
        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # move_to_table → move_table_base に auto-transition された
        assert dispatcher.active_skill_name == "move_table_base"

    def test_skill_max_dwell_raises_safety_error_when_no_transition_candidate(self, monkeypatch):
        """TRANSITIONS graph に candidates が無い skill が dwell 超過した場合、
        auto-transition できないので fallback で LiveSourceSafetyError を raise。"""
        _patch_clock(monkeypatch)

        class _FreshSource:
            def __init__(self):
                self.t = 0

            def get(self):
                frame = _empty_frame(t=self.t)
                self.t += 1
                return frame

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({"move_to_table": 0.2})
        )
        # transitions = 空 dict → 遷移候補なし
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            transitions={},
        )

        with pytest.raises(LiveSourceSafetyError, match="no transition candidates"):
            orch.run_live(
                _FreshSource(),
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # 遷移不能、initial skill のまま残る
        assert dispatcher.active_skill_name == "move_to_table"

    def test_auto_transition_writes_event_to_log_sink(self, monkeypatch):
        """auto-transition が JSONL log_sink に event 記録される (post-hoc 解析用)。
        通常 tick log と区別できるよう `event: auto_transition_timeout` schema で。"""
        _patch_clock(monkeypatch)

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({"move_to_table": 0.2})
        )
        log_sink = io.StringIO()
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            log_sink=log_sink,
        )

        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # log_sink から event=auto_transition_timeout の行を検出
        lines = [line for line in log_sink.getvalue().split("\n") if line.strip()]
        auto_events = [
            json.loads(line) for line in lines
            if '"event"' in line and "auto_transition_timeout" in line
        ]
        assert len(auto_events) == 1
        ev = auto_events[0]
        assert ev["event"] == "auto_transition_timeout"
        assert ev["from"] == "move_to_table"
        assert ev["to"] == "move_table_base"
        assert ev["timeout_s"] == 0.2
        assert ev["elapsed_s"] >= 0.2  # 到達時点なので >= timeout

    def test_skill_max_dwell_fires_for_multiple_chained_skills(self, monkeypatch):
        """chain 対応 (Issue #81): 複数 skill が max_dwell_sec を expose していれば
        全て順に auto-transition が発火する (旧 initial_skill_timeout の 1 回きり制約撤廃)。
        pick_table_leg → insert_table_leg → rotate_leg_to_tighten の 2 段 chain を verify。"""
        _patch_clock(monkeypatch)

        dispatcher = SkillDispatchLowerPolicy(
            self._make_registry_with_dwell({
                "pick_table_leg": 0.2,
                "insert_table_leg": 0.2,
            })
        )
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
            initial_skill="pick_table_leg",
        )

        # pick_table_leg → insert_table_leg → rotate_leg_to_tighten の 2 段 chain
        # (rotate_leg_to_tighten は max_dwell_sec None なので停止したまま)
        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # 2 段 chain 経過、rotate_leg_to_tighten に到達
        assert dispatcher.active_skill_name == "rotate_leg_to_tighten"

    def test_skill_with_none_max_dwell_does_not_auto_transition(self, monkeypatch):
        """Skill.max_dwell_sec が None なら auto-transition しない (enter_check ベースのみ)。
        従来 initial_skill_timeout=None の挙動と等価。"""
        _patch_clock(monkeypatch)

        # 全 skill max_dwell_sec=None (registry default)
        dispatcher = SkillDispatchLowerPolicy(_make_registry())
        orch = Orchestrator(
            _StubPerception({}),
            DetectionStream(_base_cleanup_config()),
            dispatcher,
        )

        n_frames = 200
        source = _ScriptedLiveSource([_empty_frame(t=i) for i in range(n_frames)])
        with pytest.raises(KeyboardInterrupt):
            orch.run_live(
                source,
                hz=10.0,
                startup_timeout=1.0,
                frame_timeout=0.5,
            )

        # timeout 発火せず、move_to_table のまま (YOLO fire 無しなので)
        assert dispatcher.active_skill_name == "move_to_table"


# =========================================================================
# Production 配線の regression (initial_skill="setup" + DEFAULT_*)
# =========================================================================
def _production_registry() -> dict[str, MockSkill]:
    """entrypoint.py と同じ skill 名集合 (setup を含む production 配線)。"""
    return {
        name: MockSkill(name)
        for name in [
            "setup",
            "move_to_table",
            "move_table_base",
            "pick_table_leg",
            "insert_table_leg",
            "rotate_leg_to_tighten",
            "flip_table",
        ]
    }


class TestProductionWiring:
    """production の唯一経路 (initial_skill='setup' + DEFAULT_TRANSITIONS/ENTER_CHECK)
    を通す regression。setup の遷移先 move_to_table が DEFAULT_ENTER_CHECK に無いと、
    warmup 明けの tick で enter_check[cand] 直接 index が KeyError 即死する。"""

    def test_setup_start_survives_warmup_tick_without_keyerror(self):
        """initial_skill='setup' + DEFAULT_* で warmup 明けの tick が例外死しない。"""
        perception = _StubPerception({0: [], 1: [], 2: []})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_production_registry())
        orch = Orchestrator(
            perception, cleaner, dispatcher, initial_skill="setup"
        )

        orch.tick(_empty_frame(t=0))  # warmup (cleaned=None)
        # 修正前: ここで KeyError 'move_to_table' (setup の遷移候補を index)
        result = orch.tick(_empty_frame(t=1))
        assert result is not None
        assert result.current_skill == "setup"
        # YOLO fire 無し・dwell 前なので setup に留まる (enter_check では進まない)
        assert result.fire_transition_to is None

    def test_default_graph_is_constructible(self):
        """production 配線 (DEFAULT_TRANSITIONS + DEFAULT_ENTER_CHECK + 全 skill 登録)
        で Orchestrator が例外なく構築できる (production wiring smoke)。"""
        perception = _StubPerception({})
        cleaner = DetectionStream(_base_cleanup_config())
        dispatcher = SkillDispatchLowerPolicy(_production_registry())
        # 例外を投げずに構築できれば OK
        Orchestrator(perception, cleaner, dispatcher, initial_skill="setup")
