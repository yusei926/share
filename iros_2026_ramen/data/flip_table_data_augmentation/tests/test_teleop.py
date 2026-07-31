from __future__ import annotations

from collections import deque
from dataclasses import replace
import inspect
import json
import io
import socket
import subprocess
import sys
import time
from pathlib import Path
import threading
import types
import queue

import numpy as np
import pytest

from data.flip_table_data_augmentation.teleop.config import (
    OFFICIAL_G1_29_ARM_LOWER_RAD,
    OFFICIAL_G1_29_ARM_UPPER_RAD,
    load_teleop_config,
)
from data.flip_table_data_augmentation.teleop.contracts import (
    ArmHandTarget,
    ControlEvent,
    ControlMode,
    MESSAGE_SCHEMA_VERSION,
    TeleopObservation,
)
from data.flip_table_data_augmentation.teleop.operator_view import (
    HUD_PANEL_WIDTH,
    REAL_DESKTOP_SHAPE,
    REAL_EYE_WIDTH,
    compose_head_stereo_view,
    compose_real_desktop_view,
    compose_real_operator_stereo_view,
)
from data.flip_table_data_augmentation.teleop.operator_process import (
    OperatorProcess,
    _TrackingFaultLatch,
    _desktop_preview_enabled,
    _operator_hand_status,
    operator_camera_roles,
)
from data.flip_table_data_augmentation.teleop.desktop_preview import _offer_latest
from data.flip_table_data_augmentation.teleop.xr_runtime import (
    IncoherentBilateralHandFrame,
)
from data.flip_table_data_augmentation.teleop.numeric import (
    compose_robot_q,
    demo_hand_value,
    desired_body_q,
    numeric_features,
)
from data.flip_table_data_augmentation.teleop.raw_episode import (
    EpisodeIdentity,
    FrameSynchronizationError,
    RawEpisodeWriter,
)
from data.flip_table_data_augmentation.teleop.sim.safety import (
    CommandSafetyFilter,
    WatchdogState,
)
from data.flip_table_data_augmentation.teleop.real.safety import (
    OfficialG1CommandFilter,
)
from data.flip_table_data_augmentation.teleop.real.network import counter_delta
from data.flip_table_data_augmentation.teleop.backend import TransientObservationError
from data.flip_table_data_augmentation.teleop.sim.backend import SimSocketBackend
from data.flip_table_data_augmentation.teleop.real.backend import (
    ARM_INDICES,
    ARM_SDK_BLEND_OUT_S,
    ARM_SDK_MAX_WEIGHT,
    BODY_KD,
    BODY_KP,
    DEX1_MAX_TARGET_OFFSET_FRACTION,
    DEX1_MOTOR_OPEN_RAD,
    G1_29_LOWCMD_MOTOR_COUNT,
    LOWER_BODY_INDICES,
    RealDdsBackend,
    REAL_ARM_SDK_HZ,
    REAL_DEX1_HZ,
    WAIST_GUARD_INDICES,
    WEAK_INDICES,
    WEAK_KD,
    WEAK_KP,
    WRIST_INDICES,
    WRIST_KD,
    WRIST_KP,
    camera_bundle_status,
)
from data.flip_table_data_augmentation.teleop.real.teleimager import (
    LatestCameraSample,
    LatestCameraTracker,
    receive_teleimage,
)
from data.flip_table_data_augmentation.teleop.transport import FramedSocket
from data.flip_table_data_augmentation.teleop.upstream_compat import install_logging_mp_compat
from data.flip_table_data_augmentation.teleop import xr_runtime as xr_runtime_module
from data.flip_table_data_augmentation.teleop.xr_runtime import XrInput
from data.flip_table_data_augmentation.teleop.xr_runtime import (
    _hand_payload_assessment,
    _install_heartbeat_patch,
    _valid_webxr_wrist_payload,
    _xr_display_mode,
)
from data.flip_table_data_augmentation.teleop.session import (
    ObservationStream,
    TrackingAnchorRequest,
    _assembled_table_mass_kg,
    _camera_alert_roles,
    _control_probe_target,
    _hold_target_from_observation,
    _remote_capture_seconds,
    _write_session_report,
)
from data.flip_table_data_augmentation.teleop.timing import bounded_delay_steps


def test_tracking_anchor_request_never_reuses_a_generation_after_disarm() -> None:
    request = TrackingAnchorRequest()

    assert request.request() == 1
    request.disarm()
    assert request.generation == 0
    assert request.request() == 2


def test_independent_camera_liveness_identifies_only_the_stopped_role() -> None:
    warning, hold, outage = _camera_alert_roles(
        {
            "head_left": 21.0,
            "head_right": 21.0,
            "left_wrist": 205.0,
            "right_wrist": 19.0,
        },
        hold_timeout_s=0.2,
        outage_timeout_s=0.75,
    )
    assert warning == ["left_wrist"]
    assert hold == ["left_wrist"]
    assert outage == []

    warning, hold, outage = _camera_alert_roles(
        {
            "head_left": 24.0,
            "head_right": 24.0,
            "left_wrist": 751.0,
            "right_wrist": 20.0,
        },
        hold_timeout_s=0.2,
        outage_timeout_s=0.75,
    )
    assert warning == ["left_wrist"]
    assert hold == ["left_wrist"]
    assert outage == ["left_wrist"]


def test_pause_holds_the_safety_filtered_backend_target_not_avp_input() -> None:
    observation = _observation()
    observed = replace(
        observation,
        applied_arm_target_rad=tuple(np.linspace(-0.4, 0.4, 14)),
        applied_dex1_opening_target=(0.25, 0.75),
    )

    arm, hand, torque = _hold_target_from_observation(observed)

    np.testing.assert_allclose(arm, observed.applied_arm_target_rad)
    np.testing.assert_allclose(hand, observed.applied_dex1_opening_target)
    np.testing.assert_allclose(torque, np.zeros(14))
    arm[:] = 0.0
    hand[:] = 0.0
    assert observed.applied_arm_target_rad[0] == pytest.approx(-0.4)
    assert observed.applied_dex1_opening_target == (0.25, 0.75)


def test_assembled_table_mass_is_fixed_to_official_usd_value() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["physical_randomization"]["table_part_mass_scale"] == [1.0, 1.0]
    randomization = {
        "table_part_masses": [
            {"body_mass_kg": [1.1]},
            *({"body_mass_kg": [0.124]} for _ in range(4)),
        ]
    }
    assert _assembled_table_mass_kg(randomization) == pytest.approx(1.596)


def test_hands_component_is_registered_once_without_custom_reassertion() -> None:
    source = inspect.getsource(xr_runtime_module._install_heartbeat_patch)
    assert 'session.remove("hands")' not in source
    assert "reasserting the Hands component" not in source
    assert "_hand_stream_refresh_due" not in source


def test_webxr_wrist_payload_matches_upstream_nonsingular_transform_rule() -> None:
    valid = np.tile(np.eye(4, dtype=np.float64).reshape(-1), 25)
    assert _valid_webxr_wrist_payload(valid) is True

    zero_wrist = valid.copy()
    zero_wrist[:16] = 0.0
    assert _valid_webxr_wrist_payload(zero_wrist) is False

    scaled_rotation = valid.copy()
    scaled_rotation[0] = 2.0
    assert _valid_webxr_wrist_payload(scaled_rotation) is True

    unused_invalid = valid.copy()
    unused_invalid[16] = float("nan")
    event = types.SimpleNamespace(
        value={
            "left": unused_invalid,
            "right": valid,
            "leftState": {"pinchValue": 0.06},
            "rightState": {"pinchValue": 0.07},
        }
    )
    issue, diagnostics = _hand_payload_assessment(event)
    assert issue is None
    assert diagnostics == ("invalid_left_unused_skeleton",)


def _jpeg(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (640, 480), color)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


SIM_CONTROL_CONTRACT = {
    "body_mode": "balanced_wbc",
    "physics_hz": 200.0,
    "control_hz": 50.0,
    "wbc_navigation_velocity_m_s_rad_s": [0.0, 0.0, 0.0],
    "wbc_base_height_m": 0.74,
    "wbc_torso_rpy_rad": [0.0, 0.0, 0.0],
    "wbc_stand_onnx_sha256": "a" * 64,
    "wbc_walk_onnx_sha256": "b" * 64,
    "team_adapter_sha256": "c" * 64,
}


def _observation(sequence: int = 1, *, now_ns: int | None = None) -> TeleopObservation:
    now = time.monotonic_ns() if now_ns is None else now_ns
    cameras = {
        "head_left": b"left-jpeg",
        "head_right": b"right-jpeg",
        "left_wrist": b"left-wrist-jpeg",
        "right_wrist": b"right-wrist-jpeg",
    }
    return TeleopObservation(
        sequence=sequence,
        capture_monotonic_ns=now,
        backend="sim",
        body_joint_position_rad=(0.0,) * 29,
        body_joint_velocity_rad_s=(0.0,) * 29,
        dex1_opening_fraction=(1.0, 1.0),
        applied_arm_target_rad=(0.0,) * 14,
        applied_dex1_opening_target=(1.0, 1.0),
        root_pose_xyzw=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        camera_capture_monotonic_ns={role: now for role in cameras},
        camera_jpeg=cameras,
        success=True,
        diagnostics={
            "randomization": {"yaw_rad": 0.2},
            "sim_control_contract": SIM_CONTROL_CONTRACT,
        },
    )


def _recordable_observation(
    sequence: int = 1, *, now_ns: int | None = None
) -> TeleopObservation:
    colors = {
        "head_left": (220, 40, 20),
        "head_right": (20, 180, 60),
        "left_wrist": (30, 60, 220),
        "right_wrist": (220, 180, 20),
    }
    return replace(
        _observation(sequence, now_ns=now_ns),
        camera_jpeg={role: _jpeg(color) for role, color in colors.items()},
    )


def _target(sequence: int = 1) -> ArmHandTarget:
    return ArmHandTarget(
        sequence=sequence,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=(0.0,) * 14,
        dex1_opening_fraction=(1.0, 1.0),
    )


def test_operator_view_keeps_true_stereo_eyes_distinct() -> None:
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    left[..., 0] = 211
    right[..., 2] = 173
    view = compose_head_stereo_view(left, right)

    assert view.shape == (480, 1280, 3)
    assert np.array_equal(view[:, :640], left)
    assert np.array_equal(view[:, 640:], right)
    assert not np.array_equal(view[:, :640], view[:, 640:])


def test_real_operator_view_keeps_head_stereo_and_adds_wrist_angle_huds() -> None:
    head_left = np.zeros((480, 640, 3), dtype=np.uint8)
    head_right = np.zeros_like(head_left)
    left_wrist = np.zeros_like(head_left)
    right_wrist = np.zeros_like(head_left)
    head_left[..., 0] = 211
    head_right[..., 2] = 173
    left_wrist[..., 1] = 149
    right_wrist[..., :] = (29, 67, 191)

    view = compose_real_operator_stereo_view(
        head_left,
        head_right,
        left_wrist,
        right_wrist,
        np.linspace(-1.0, 1.0, 14),
        "HANDS READY",
    )

    assert view.shape == (480, REAL_EYE_WIDTH * 2, 3)
    # The true head stereo pixels are untouched in the central area of each
    # eye. The wrist/angle panels are duplicated monocular diagnostics.
    np.testing.assert_array_equal(
        view[:, HUD_PANEL_WIDTH : HUD_PANEL_WIDTH + 640], head_left
    )
    np.testing.assert_array_equal(
        view[
            :,
            REAL_EYE_WIDTH + HUD_PANEL_WIDTH : REAL_EYE_WIDTH + HUD_PANEL_WIDTH + 640,
        ],
        head_right,
    )
    assert not np.array_equal(view[:, :HUD_PANEL_WIDTH], 0)
    assert not np.array_equal(
        view[:, HUD_PANEL_WIDTH + 640 : REAL_EYE_WIDTH],
        0,
    )


def test_real_desktop_view_shows_all_four_cameras_and_joint_overlays() -> None:
    head_left = np.full((480, 640, 3), (211, 0, 0), dtype=np.uint8)
    head_right = np.full((480, 640, 3), (0, 0, 173), dtype=np.uint8)
    left_wrist = np.full((480, 640, 3), (0, 149, 0), dtype=np.uint8)
    right_wrist = np.full((480, 640, 3), (29, 67, 191), dtype=np.uint8)

    view = compose_real_desktop_view(
        head_left,
        head_right,
        left_wrist,
        right_wrist,
        np.linspace(-1.0, 1.0, 14),
        "HANDS READY",
    )

    assert view.shape == REAL_DESKTOP_SHAPE
    # Uncovered camera pixels prove each native image occupies its own tile;
    # labelled/angle areas intentionally differ from their source pixels.
    np.testing.assert_array_equal(view[300, 400], head_left[300, 400])
    np.testing.assert_array_equal(view[300, 1000], head_right[300, 360])
    np.testing.assert_array_equal(view[800, 400], left_wrist[320, 400])
    np.testing.assert_array_equal(view[800, 1000], right_wrist[320, 360])
    assert not np.array_equal(view[540, 20], left_wrist[60, 20])


def test_desktop_preview_defaults_on_and_has_explicit_boolean_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FLIP_TABLE_TELEOP_DESKTOP_PREVIEW", raising=False)
    assert _desktop_preview_enabled() is True
    monkeypatch.setenv("FLIP_TABLE_TELEOP_DESKTOP_PREVIEW", "false")
    assert _desktop_preview_enabled() is False
    monkeypatch.setenv("FLIP_TABLE_TELEOP_DESKTOP_PREVIEW", "invalid")
    with pytest.raises(ValueError, match="must be boolean"):
        _desktop_preview_enabled()


def test_desktop_preview_queue_always_retains_latest_update() -> None:
    updates: queue.Queue[object] = queue.Queue(maxsize=1)
    _offer_latest(updates, "old")
    _offer_latest(updates, "new")
    assert updates.get_nowait() == "new"


def test_real_operator_camera_contract_includes_both_wrist_cameras() -> None:
    assert operator_camera_roles("sim") == ("head_left", "head_right")
    assert operator_camera_roles("real") == (
        "head_left",
        "head_right",
        "left_wrist",
        "right_wrist",
    )
    with pytest.raises(ValueError, match="unsupported operator backend"):
        operator_camera_roles("invalid")


def test_observation_stream_counts_every_received_frame_before_latest_sampling() -> None:
    class Backend:
        def __init__(self) -> None:
            self.values = iter(
                (
                    _observation(2, now_ns=2_000_000_000),
                    _observation(3, now_ns=2_033_333_333),
                    _observation(5, now_ns=2_100_000_000),
                )
            )
            self.release = threading.Event()

        def observe(self, timeout_s):
            try:
                return next(self.values)
            except StopIteration:
                self.release.wait(timeout_s)
                return _observation(5, now_ns=2_100_000_000)

    backend = Backend()
    stream = ObservationStream(
        backend,
        _observation(1, now_ns=1_966_666_667),
    )
    stream.start()
    deadline = time.monotonic() + 1.0
    while stream.stats().received_count < 4 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert stream.latest().observation.sequence == 5
    stats = stream.stats()
    assert stats.received_count == 4
    assert stats.sequences == (1, 2, 3, 5)
    assert stats.missing_sequences == 1
    stream.reset_stats()
    reset_stats = stream.stats()
    assert reset_stats.received_count == 1
    assert reset_stats.sequences == (5,)
    assert reset_stats.missing_sequences == 0

    stream.request_stop()
    backend.release.set()
    stream.close()


def test_observation_stream_retains_last_sample_during_recoverable_camera_gap() -> None:
    recovered = _observation(2, now_ns=2_000_000_000)

    class Backend:
        def __init__(self) -> None:
            self.attempt = 0
            self.release = threading.Event()

        def observe(self, timeout_s):
            self.attempt += 1
            if self.attempt == 1:
                raise TransientObservationError("right wrist stale")
            if self.attempt == 2:
                return recovered
            self.release.wait(timeout_s)
            return recovered

    backend = Backend()
    stream = ObservationStream(
        backend,
        _observation(1, now_ns=1_966_666_667),
    )
    stream.start()
    deadline = time.monotonic() + 1.0
    saw_transient = False
    while time.monotonic() < deadline:
        health = stream.health()
        saw_transient = saw_transient or health.transient_error == "right wrist stale"
        if stream.latest().observation.sequence == 2:
            break
        time.sleep(0.001)

    assert saw_transient is True
    assert stream.latest().observation.sequence == 2
    assert stream.health().transient_error is None
    stream.request_stop()
    backend.release.set()
    stream.close()


def test_operator_process_keeps_only_latest_pending_state_and_camera() -> None:
    operator = OperatorProcess(Path("/tmp/xr-unused"), load_teleop_config())
    operator._transport = object()
    state = {
        "arm_joint_position_rad": (0.0,) * 14,
        "arm_joint_velocity_rad_s": (0.0,) * 14,
        "dex1_opening_fraction": (1.0, 1.0),
        "tracking_generation": 0,
    }

    operator.submit(
        camera_jpeg={"head_left": b"left", "head_right": b"right"},
        **state,
    )
    operator.submit(camera_jpeg=None, **state)

    assert operator._pending_update is not None
    assert operator._pending_update["sequence"] == 2
    assert operator._pending_update["camera_jpeg"] == {
        "head_left": b"left",
        "head_right": b"right",
    }


def test_operator_tracking_fault_stays_latched_until_a_new_generation() -> None:
    fault = _TrackingFaultLatch()

    assert fault.blocks(1) is False
    fault.trip(1)
    assert fault.blocks(1) is True
    # Parent disarm cannot accidentally clear the fault before its response is
    # observed, while the next explicit r creates a new usable generation.
    assert fault.blocks(0) is False
    assert fault.blocks(2) is False
    with pytest.raises(ValueError, match="armed tracking generation"):
        fault.trip(0)


def test_operator_hud_distinguishes_wait_ready_anchor_track_and_manual_resume() -> None:
    assert (
        _operator_hand_status(
            avp_live=False,
            requested_generation=0,
            produced_generation=0,
            had_tracking=False,
        )
        == "HANDS WAIT"
    )
    assert (
        _operator_hand_status(
            avp_live=True,
            requested_generation=0,
            produced_generation=0,
            had_tracking=False,
        )
        == "HANDS READY"
    )
    assert (
        _operator_hand_status(
            avp_live=True,
            requested_generation=2,
            produced_generation=0,
            had_tracking=True,
        )
        == "ANCHORING"
    )
    assert (
        _operator_hand_status(
            avp_live=True,
            requested_generation=2,
            produced_generation=2,
            had_tracking=True,
        )
        == "TRACKING"
    )
    assert (
        _operator_hand_status(
            avp_live=True,
            requested_generation=0,
            produced_generation=0,
            had_tracking=True,
        )
        == "PRESS R"
    )


def test_real_operator_process_requires_head_and_both_wrist_payloads() -> None:
    operator = OperatorProcess(
        Path("/tmp/xr-unused"), load_teleop_config(), backend="real"
    )
    operator._transport = object()
    state = {
        "arm_joint_position_rad": (0.0,) * 14,
        "arm_joint_velocity_rad_s": (0.0,) * 14,
        "dex1_opening_fraction": (1.0, 1.0),
        "tracking_generation": 0,
    }

    with pytest.raises(ValueError, match="unexpected roles"):
        operator.submit(
            camera_jpeg={"head_left": b"left", "head_right": b"right"},
            **state,
        )
    payload = {
        "head_left": b"head-left",
        "head_right": b"head-right",
        "left_wrist": b"left-wrist",
        "right_wrist": b"right-wrist",
    }
    operator.submit(camera_jpeg=payload, **state)
    assert operator._pending_update is not None
    assert operator._pending_update["camera_jpeg"] == payload


def test_avp_wrist_discontinuity_is_detected_before_ik() -> None:
    previous = (np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64))
    current = tuple(value.copy() for value in previous)
    current[1][0, 3] = 0.07

    assert XrInput._wrist_pair_has_discontinuity(previous, current) is True
    assert XrInput._wrist_pair_has_discontinuity(previous, previous) is False


def test_avp_uses_latest_official_head_yaw_runtime() -> None:
    config = load_teleop_config()

    assert config.runtime.xr_revision == "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
    assert config.runtime.televuer_revision == "766de45e74373ae0ea66321d942ce538385655a5"
    source = Path(xr_runtime_module.__file__).read_text()
    assert 'arm_reference_mode="head_yaw"' in source
    assert "_official_head_yaw_wrist_reference" not in source


def test_avp_first_anchored_target_is_continuous_before_official_absolute_ik() -> None:
    class TeleData:
        head_pose = np.eye(4, dtype=np.float64)
        left_wrist_pose = np.eye(4, dtype=np.float64)
        right_wrist_pose = np.eye(4, dtype=np.float64)
        left_hand_pinchValue = 6.0
        right_hand_pinchValue = 6.0

    class Wrapper:
        data = TeleData()
        tvuer = types.SimpleNamespace(
            team_ramen_hand_event_count=types.SimpleNamespace(value=0),
            team_ramen_hand_frame_generation=types.SimpleNamespace(value=0),
        )

        def get_tele_data(self):
            return self.data

    class Ik:
        def __init__(self) -> None:
            self.targets = []
            self.init_data = np.full(14, 4.0)
            self.smooth_filter = types.SimpleNamespace(
                _data_queue=[np.full(14, 2.0), np.full(14, 3.0)],
                _filtered_data=np.full(14, 3.0),
            )

        def solve_ik(self, left, right, arm_q, arm_dq):
            self.targets.append((left.copy(), right.copy()))
            return arm_q + 0.1, np.zeros_like(arm_q)

    xr = XrInput.__new__(XrInput)
    xr.wrapper = Wrapper()
    xr.ik = Ik()
    xr._tracking_generation = 0
    xr._pending_tracking_generation = 0
    xr._pending_hand_event_count = -1
    xr._pending_wrist_samples = []
    xr._last_avp_wrist_poses = None
    arm_q = np.linspace(-0.3, 0.3, 14)
    arm_dq = np.zeros(14)
    opening = np.asarray((0.25, 0.75))

    for event_count in range(1, 5):
        xr.wrapper.tvuer.team_ramen_hand_event_count.value = event_count
        assert xr.target(arm_q, arm_dq, opening, 1) is None
    xr.wrapper.tvuer.team_ramen_hand_event_count.value = 5
    first = xr.target(arm_q, arm_dq, opening, 1)
    assert first is not None
    first_arm, first_torque, first_hand = first
    xr.wrapper.data.left_wrist_pose = np.eye(4, dtype=np.float64)
    xr.wrapper.data.left_wrist_pose[0, 3] = 0.05
    xr.wrapper.tvuer.team_ramen_hand_event_count.value = 6
    second = xr.target(arm_q, arm_dq, opening, 1)
    assert second is not None
    second_arm, second_torque, second_hand = second

    np.testing.assert_allclose(first_arm, arm_q)
    np.testing.assert_allclose(first_hand, opening)
    np.testing.assert_allclose(first_torque, np.zeros(14))
    np.testing.assert_allclose(second_arm, arm_q + 0.1)
    np.testing.assert_allclose(second_hand, (0.5, 0.5))
    np.testing.assert_allclose(second_torque, np.zeros(14))
    assert len(xr.ik.targets) == 1
    # Match upstream: TeleVuer's absolute wrist matrices reach G1_29 IK
    # unchanged after the one-target safe start hold.
    np.testing.assert_allclose(xr.ik.targets[0][0], xr.wrapper.data.left_wrist_pose)
    np.testing.assert_allclose(xr.ik.targets[0][1], xr.wrapper.data.right_wrist_pose)
    np.testing.assert_allclose(xr.ik.init_data, arm_q)
    assert len(xr.ik.smooth_filter._data_queue) == 1
    np.testing.assert_allclose(xr.ik.smooth_filter._data_queue[0], arm_q)


def test_avp_reanchor_restarts_stability_window_after_wrist_jump() -> None:
    xr = XrInput.__new__(XrInput)
    xr._tracking_generation = 0
    xr._pending_tracking_generation = 0
    xr._pending_hand_event_count = -1
    xr._pending_wrist_samples = []
    xr._last_avp_wrist_poses = None
    steady = (np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64))

    assert xr._stable_anchor_candidate(1, 1, steady) is None
    jumped = tuple(pose.copy() for pose in steady)
    jumped[0][0, 3] = 0.10
    assert xr._stable_anchor_candidate(1, 2, jumped) is None
    for event_count in range(3, 6):
        assert xr._stable_anchor_candidate(1, event_count, jumped) is None
    candidate = xr._stable_anchor_candidate(1, 6, jumped)

    assert candidate is not None
    np.testing.assert_allclose(candidate[0][:3, 3], (0.10, 0.0, 0.0))


def test_avp_tele_data_snapshot_retries_a_torn_bilateral_frame() -> None:
    generation = types.SimpleNamespace(value=2)
    first = object()
    second = object()

    class Wrapper:
        tvuer = types.SimpleNamespace(team_ramen_hand_frame_generation=generation)

        def __init__(self) -> None:
            self.calls = 0

        def get_tele_data(self):
            self.calls += 1
            if self.calls == 1:
                generation.value = 4
                return first
            return second

    xr = XrInput.__new__(XrInput)
    xr.wrapper = Wrapper()

    assert xr._tele_data_snapshot() is second
    assert xr.wrapper.calls == 2


def test_avp_tele_data_snapshot_uses_one_outer_bilateral_lock() -> None:
    data = object()

    class FrameLock:
        def __init__(self) -> None:
            self.acquired = 0
            self.released = 0

        def acquire(self, *, timeout):
            assert timeout == pytest.approx(0.05)
            self.acquired += 1
            return True

        def release(self):
            self.released += 1

    frame_lock = FrameLock()

    class Wrapper:
        tvuer = types.SimpleNamespace(team_ramen_hand_frame_lock=frame_lock)

        @staticmethod
        def get_tele_data():
            assert frame_lock.acquired == 1
            assert frame_lock.released == 0
            return data

    xr = XrInput.__new__(XrInput)
    xr.wrapper = Wrapper()

    assert xr._tele_data_snapshot() is data
    assert frame_lock.acquired == 1
    assert frame_lock.released == 1


def test_avp_tele_data_snapshot_times_out_without_reading_partial_frame() -> None:
    class FrameLock:
        @staticmethod
        def acquire(*, timeout):
            assert timeout == pytest.approx(0.05)
            return False

        @staticmethod
        def release():
            raise AssertionError("an unacquired frame lock must not be released")

    class Wrapper:
        tvuer = types.SimpleNamespace(team_ramen_hand_frame_lock=FrameLock())

        @staticmethod
        def get_tele_data():
            raise AssertionError("a partial bilateral frame must not be read")

    xr = XrInput.__new__(XrInput)
    xr.wrapper = Wrapper()

    with pytest.raises(IncoherentBilateralHandFrame, match="bilateral AVP hand event"):
        xr._tele_data_snapshot()


def test_avp_tele_data_snapshot_rejects_a_persistently_torn_bilateral_frame() -> None:
    generation = types.SimpleNamespace(value=1)

    class Wrapper:
        tvuer = types.SimpleNamespace(team_ramen_hand_frame_generation=generation)

        def get_tele_data(self):
            raise AssertionError("must not read a frame while its writer is active")

    xr = XrInput.__new__(XrInput)
    xr.wrapper = Wrapper()

    with pytest.raises(IncoherentBilateralHandFrame, match="coherent bilateral"):
        xr._tele_data_snapshot()


def test_avp_disarm_discards_active_and_pending_tracking_anchors() -> None:
    xr = XrInput.__new__(XrInput)
    xr._tracking_generation = 3
    xr._pending_tracking_generation = 4
    xr._pending_hand_event_count = 42
    xr._pending_wrist_samples = [
        (np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64))
    ]
    xr._last_avp_wrist_poses = (
        np.eye(4, dtype=np.float64),
        np.eye(4, dtype=np.float64),
    )

    xr.disarm()

    assert xr._tracking_generation == 0
    assert xr._pending_tracking_generation == 0
    assert xr._pending_hand_event_count == -1
    assert xr._pending_wrist_samples == []
    assert xr._last_avp_wrist_poses is None


def test_avp_pinch_distance_maps_both_dex1_grippers_independently() -> None:
    np.testing.assert_allclose(
        XrInput._dex1_opening_from_pinch(np.asarray((5.0, 7.0))),
        (0.0, 1.0),
    )


def test_control_probe_moves_both_wrists_and_dex1_independently() -> None:
    initial = np.zeros(14, dtype=np.float64)
    start_arm, start_hand = _control_probe_target(initial, 0, 5)
    quarter_arm, quarter_hand = _control_probe_target(initial, 1, 5)
    half_arm, half_hand = _control_probe_target(initial, 2, 5)

    np.testing.assert_allclose(start_arm, initial)
    np.testing.assert_allclose(start_hand, (1.0, 0.0))
    assert quarter_arm[4] == pytest.approx(0.12)
    assert quarter_arm[11] == pytest.approx(-0.12)
    np.testing.assert_allclose(quarter_hand, (0.5, 0.5))
    np.testing.assert_allclose(half_hand, (0.0, 1.0))
    np.testing.assert_allclose(
        XrInput._dex1_opening_from_pinch(np.asarray((6.0, 4.0))),
        (0.5, 0.0),
    )
    np.testing.assert_allclose(
        XrInput._dex1_opening_from_pinch(np.asarray((8.0, 6.5))),
        (1.0, 0.75),
    )


def test_dex1_target_is_acceleration_limited_and_stale_input_stops() -> None:
    config = load_teleop_config()
    safety = CommandSafetyFilter(
        config.safety,
        servo_hz=config.rates.servo_hz,
    )
    measured_arm = np.zeros(14)
    measured_hand = np.zeros(2)
    safety.reset(measured_arm, measured_hand)
    now_ns = 10_000_000_000
    command = ArmHandTarget(
        sequence=1,
        monotonic_ns=now_ns,
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=(0.0,) * 14,
        dex1_opening_fraction=(1.0, 1.0),
    )

    first = safety.apply(
        command,
        measured_arm_position_rad=measured_arm,
        measured_dex1_opening_fraction=measured_hand,
        now_ns=now_ns,
        last_command_ns=now_ns,
        tracking=True,
    )
    expected_first_step = (
        config.safety.hand_acceleration_fraction_s2
        / config.rates.servo_hz**2
    )
    np.testing.assert_allclose(
        first.dex1_opening_fraction,
        (expected_first_step, expected_first_step),
    )
    assert first.watchdog is WatchdogState.ACTIVE

    stale = safety.apply(
        command,
        measured_arm_position_rad=measured_arm,
        measured_dex1_opening_fraction=measured_hand,
        now_ns=now_ns + int(config.safety.command_stop_timeout_s * 1.0e9),
        last_command_ns=now_ns,
        tracking=True,
    )
    assert stale.watchdog is WatchdogState.STOP
    np.testing.assert_allclose(stale.dex1_opening_fraction, measured_hand)


def test_real_motion_filter_matches_official_global_arm_scaling_and_keeps_torque() -> None:
    config = load_teleop_config()
    safety = OfficialG1CommandFilter(
        config.safety, servo_hz=REAL_ARM_SDK_HZ
    )
    measured_arm = np.zeros(14)
    measured_hand = np.asarray((0.25, 0.75))
    safety.reset(measured_arm, measured_hand)
    now_ns = 10_000_000_000
    desired = np.linspace(0.1, 1.4, 14)
    torque = np.linspace(-2.0, 2.0, 14)
    result = safety.apply(
        ArmHandTarget(
            sequence=1,
            monotonic_ns=now_ns,
            mode=ControlMode.TRACK,
            event=ControlEvent.NONE,
            arm_position_rad=tuple(desired),
            dex1_opening_fraction=(1.0, 0.0),
            arm_feedforward_torque_nm=tuple(torque),
        ),
        measured_arm_position_rad=measured_arm,
        measured_dex1_opening_fraction=measured_hand,
        now_ns=now_ns,
        last_command_ns=now_ns,
        tracking=True,
        official_arm_velocity_limit_rad_s=20.0,
    )

    maximum_step = 20.0 / REAL_ARM_SDK_HZ
    scale = np.max(np.abs(desired)) / maximum_step
    np.testing.assert_allclose(result.arm_position_rad, desired / scale)
    np.testing.assert_allclose(result.dex1_opening_fraction, (1.0, 0.0))
    np.testing.assert_allclose(result.arm_feedforward_torque_nm, torque)

    held = safety.apply(
        None,
        measured_arm_position_rad=measured_arm,
        measured_dex1_opening_fraction=measured_hand,
        now_ns=now_ns + int(config.safety.command_hold_timeout_s * 1.0e9),
        last_command_ns=now_ns,
        tracking=True,
        official_arm_velocity_limit_rad_s=20.0,
    )
    assert held.watchdog is WatchdogState.HOLD
    np.testing.assert_allclose(held.arm_position_rad, result.arm_position_rad)
    np.testing.assert_allclose(held.arm_feedforward_torque_nm, torque)


def test_real_and_sim_runtime_import_boundaries_are_one_way() -> None:
    package = "data.flip_table_data_augmentation.teleop"
    real_probe = f"""
import sys
from {package}.real import runner
assert '{package}.sim.backend' not in sys.modules
assert '{package}.sim.safety' not in sys.modules
assert 'unitree_sdk2py' not in sys.modules
"""
    sim_probe = f"""
import sys
from {package}.sim import runner
assert '{package}.real_backend' not in sys.modules
assert '{package}.real.backend' not in sys.modules
assert 'unitree_sdk2py' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", real_probe], check=True)
    subprocess.run([sys.executable, "-c", sim_probe], check=True)


def test_sim_filter_cannot_enable_the_real_g1_motion_rule() -> None:
    import inspect

    parameters = inspect.signature(CommandSafetyFilter).parameters
    assert "official_g1_motion" not in parameters
    assert "official_arm_velocity_limit_rad_s" not in inspect.signature(
        CommandSafetyFilter.apply
    ).parameters


def test_sim_rate_limiter_never_snaps_through_acceleration_limit() -> None:
    rng = np.random.default_rng(20260723)
    dt = 0.01
    velocity_limit = 2.0
    acceleration_limit = 8.0
    position = np.zeros(14)
    velocity = np.zeros(14)
    desired = np.zeros(14)

    for step in range(4000):
        if step % 17 == 0:
            desired = rng.uniform(-1.0, 1.0, size=14)
        previous_velocity = velocity.copy()
        position, velocity = CommandSafetyFilter._rate_limit(
            desired,
            position,
            velocity,
            velocity_limit=velocity_limit,
            acceleration_limit=acceleration_limit,
            dt=dt,
        )
        assert np.max(np.abs(velocity)) <= velocity_limit + 1.0e-10
        acceleration = (velocity - previous_velocity) / dt
        assert np.max(np.abs(acceleration)) <= acceleration_limit + 1.0e-10


def test_sim_filter_respects_limits_at_joint_boundaries_under_random_targets() -> None:
    config = load_teleop_config()
    servo_hz = 50.0
    dt = 1.0 / servo_hz
    lower = np.asarray(config.safety.arm_position_lower_rad)
    upper = np.asarray(config.safety.arm_position_upper_rad)
    arm = 0.5 * (lower + upper)
    hand = np.asarray((0.5, 0.5))
    previous_arm_velocity = np.zeros(14)
    previous_hand_velocity = np.zeros(2)
    safety = CommandSafetyFilter(config.safety, servo_hz=servo_hz)
    safety.reset(arm, hand)
    rng = np.random.default_rng(20260723)

    for sequence in range(3000):
        if sequence % 13 == 0:
            # Exercise clipping as well as reversals near every limit.
            desired_arm = rng.uniform(lower - 2.0, upper + 2.0)
            desired_hand = rng.uniform(0.0, 1.0, size=2)
        command = ArmHandTarget(
            sequence=sequence,
            monotonic_ns=sequence + 1,
            mode=ControlMode.TRACK,
            event=ControlEvent.NONE,
            arm_position_rad=tuple(desired_arm),
            dex1_opening_fraction=tuple(desired_hand),
        )
        output = safety.apply(
            command,
            measured_arm_position_rad=arm,
            measured_dex1_opening_fraction=hand,
            now_ns=sequence + 1,
            last_command_ns=sequence + 1,
            tracking=True,
        )
        emitted_arm = np.asarray(output.arm_position_rad)
        emitted_hand = np.asarray(output.dex1_opening_fraction)
        arm_velocity = (emitted_arm - arm) / dt
        hand_velocity = (emitted_hand - hand) / dt

        assert np.all(emitted_arm >= lower) and np.all(emitted_arm <= upper)
        assert np.all(emitted_hand >= 0.0) and np.all(emitted_hand <= 1.0)
        assert np.max(np.abs(arm_velocity)) <= config.safety.arm_velocity_rad_s + 1.0e-9
        assert np.max(np.abs(hand_velocity)) <= config.safety.hand_velocity_fraction_s + 1.0e-9
        assert (
            np.max(np.abs(arm_velocity - previous_arm_velocity)) / dt
            <= config.safety.arm_acceleration_rad_s2 + 1.0e-8
        )
        assert (
            np.max(np.abs(hand_velocity - previous_hand_velocity)) / dt
            <= config.safety.hand_acceleration_fraction_s2 + 1.0e-8
        )
        arm, hand = emitted_arm, emitted_hand
        previous_arm_velocity, previous_hand_velocity = arm_velocity, hand_velocity


def test_pinned_teleimage_adapter_uses_named_fields_and_receive_timestamp() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class TeleImageLike:
        bgr = frame
        jpg = b"jpeg"
        fps = 29.75

    before = time.monotonic_ns()
    received = receive_teleimage(lambda: TeleImageLike())
    after = time.monotonic_ns()

    assert received.bgr is frame
    assert received.jpg == b"jpeg"
    assert received.fps == pytest.approx(29.75)
    assert before <= received.received_monotonic_ns <= after

    with pytest.raises(RuntimeError, match="unsupported TeleImager API"):
        receive_teleimage(lambda: (frame, 30.0))


def test_jpeg_only_teleimage_adapter_does_not_touch_bgr_property() -> None:
    class JpegOnlyTeleImage:
        jpg = b"jpeg"
        fps = 30.0

        @property
        def bgr(self):
            raise AssertionError("JPEG-only consumer must not access .bgr")

    received = receive_teleimage(
        lambda: JpegOnlyTeleImage(),
        include_bgr=False,
    )

    assert received.bgr is None
    assert received.jpg == b"jpeg"
    assert received.fps == pytest.approx(30.0)


def test_camera_tracker_timestamps_only_unique_jpeg_transitions(
    monkeypatch,
) -> None:
    class TeleImageLike:
        bgr = None
        fps = 30.0

        def __init__(self, jpg):
            self.jpg = jpg

    values = iter((TeleImageLike(b"a"), TeleImageLike(b"a"), TeleImageLike(b"b")))
    times = iter((1_000_000_000, 1_500_000_000, 2_000_000_000))
    monkeypatch.setattr(
        "data.flip_table_data_augmentation.teleop.real.teleimager.time.monotonic_ns",
        lambda: next(times),
    )
    tracker = LatestCameraTracker("head", lambda: next(values))

    first, changed = tracker.poll()
    assert changed is True
    assert first is not None
    assert first.jpeg_generation == 1
    assert first.first_observed_monotonic_ns == 1_000_000_000

    duplicate, changed = tracker.poll()
    assert changed is False
    assert duplicate == first
    assert duplicate.first_observed_monotonic_ns == 1_000_000_000

    second, changed = tracker.poll()
    assert changed is True
    assert second is not None
    assert second.jpeg_generation == 2
    assert second.first_observed_monotonic_ns == 2_000_000_000
    assert second.transition_hz == pytest.approx(1.0)


def test_camera_bundle_accepts_phase_offset_and_rejects_stale_or_false_skew() -> None:
    def sample(role: str, generation: int, observed_ns: int) -> LatestCameraSample:
        return LatestCameraSample(
            role=role,
            jpg=role.encode(),
            source_fps=30.0,
            jpeg_generation=generation,
            first_observed_monotonic_ns=observed_ns,
            transition_hz=30.0,
        )

    baseline = {"head": 4, "left_wrist": 7, "right_wrist": 9}
    phased = {
        "head": sample("head", 5, 1_000_000_000),
        "left_wrist": sample("left_wrist", 8, 1_012_000_000),
        "right_wrist": sample("right_wrist", 10, 1_025_000_000),
    }
    valid, skew_ms, generations = camera_bundle_status(
        phased, baseline, camera_hz=30.0
    )
    assert valid is True
    assert skew_ms == pytest.approx(25.0)
    assert generations == {"head": 5, "left_wrist": 8, "right_wrist": 10}

    stale = dict(phased)
    stale["right_wrist"] = sample("right_wrist", 9, 1_025_000_000)
    assert camera_bundle_status(stale, baseline, camera_hz=30.0)[0] is False

    excessive_skew = dict(phased)
    excessive_skew["right_wrist"] = sample(
        "right_wrist", 10, 1_040_000_000
    )
    valid, skew_ms, _generations = camera_bundle_status(
        excessive_skew, baseline, camera_hz=30.0
    )
    assert valid is False
    assert skew_ms == pytest.approx(40.0)


def test_network_counter_delta_never_invents_negative_wraparound() -> None:
    assert counter_delta(
        {"rx_dropped": 5, "tcp_retransmitted_segments": 10},
        {
            "rx_dropped": 8,
            "tcp_retransmitted_segments": 7,
            "rx_errors": 2,
        },
    ) == {
        "rx_dropped": 3,
        "tcp_retransmitted_segments": 0,
        "rx_errors": 0,
    }


def test_head_camera_uvc_identity_is_correlated_not_assumed_unique(
    monkeypatch,
) -> None:
    from inference.desktop.xr import orin_teleimager_safe_launcher as launcher

    monkeypatch.setitem(
        sys.modules,
        "uvc",
        types.SimpleNamespace(
            device_list=lambda: [
                {"serialNumber": "head-60mm", "uid": "1", "name": "HBVCAM"},
                {"serialNumber": "webcam", "uid": "2", "name": "USB Webcam"},
            ]
        ),
    )
    monkeypatch.setattr(
        launcher, "v4l_device_serial", lambda _node: "head-60mm"
    )

    assert (
        launcher.find_stereo_head_uvc_serial(Path("/dev/video6"))
        == "head-60mm"
    )
    with pytest.raises(RuntimeError, match="does not match"):
        launcher.find_stereo_head_uvc_serial(
            Path("/dev/video6"), serial_override="webcam"
        )


def test_safe_launcher_reuses_configured_head_serial_when_v4l_is_detached(
    monkeypatch, tmp_path
) -> None:
    from inference.desktop.xr import orin_teleimager_safe_launcher as launcher

    config_path = tmp_path / "cameras.yaml"
    config_path.write_text(
        """
head_camera:
  serial_number: head-configured
left_wrist_camera:
  serial_number: left-d405
right_wrist_camera:
  serial_number: right-d405
""".strip(),
        encoding="utf-8",
    )
    received = {}
    monkeypatch.setattr(launcher, "find_stereo_head_node", lambda: None)

    def select(_node, serial_override=None):
        received["serial_override"] = serial_override
        return str(serial_override)

    monkeypatch.setattr(launcher, "find_stereo_head_uvc_serial", select)
    config, identity, wrist_serials = launcher.prepare_config(config_path)

    assert received["serial_override"] == "head-configured"
    assert identity == "uvc:head-configured"
    assert config["head_camera"]["serial_number"] == "head-configured"
    assert wrist_serials == {"left-d405", "right-d405"}


def test_safe_launcher_reports_expected_serials_on_realsense_enumeration_failure(
    monkeypatch,
) -> None:
    from inference.desktop.xr import orin_teleimager_safe_launcher as launcher

    def fail_enumeration():
        raise RuntimeError("failed to set power state")

    monkeypatch.setattr(
        launcher, "inspect_realsense_devices", fail_enumeration
    )
    with pytest.raises(RuntimeError, match="left-d405.*right-d405.*power state"):
        launcher.wait_for_realsense({"left-d405", "right-d405"}, timeout_s=1.0)


def test_real_dex1_target_matches_official_feedback_relative_step_limit() -> None:
    desired = np.asarray((1.0, 0.0))
    measured = np.asarray((0.4, 0.6))

    limited, active = RealDdsBackend._limit_dex1_target_around_measured(
        desired,
        measured,
    )

    np.testing.assert_allclose(
        limited,
        (
            measured[0] + DEX1_MAX_TARGET_OFFSET_FRACTION,
            measured[1] - DEX1_MAX_TARGET_OFFSET_FRACTION,
        ),
    )
    np.testing.assert_array_equal(active, (True, True))

    unchanged, unchanged_active = (
        RealDdsBackend._limit_dex1_target_around_measured(
            np.asarray((0.42, 0.58)),
            measured,
        )
    )
    np.testing.assert_allclose(unchanged, (0.42, 0.58))
    np.testing.assert_array_equal(unchanged_active, (False, False))


def test_real_dex1_write_preserves_left_right_topics_and_reports_failures() -> None:
    class MotorCommand:
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 0.0
        kd = 0.0

    class MotorCommands:
        def __init__(self) -> None:
            self.cmds = []

    class Publisher:
        def __init__(self, result: bool = True, error: Exception | None = None) -> None:
            self.result = result
            self.error = error
            self.messages = []

        def Write(self, message):
            self.messages.append(message)
            if self.error is not None:
                raise self.error
            return self.result

    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend._types = {"MotorCmds": MotorCommands, "MotorCmd": MotorCommand}
    backend._arm_message = lambda *_args, **_kwargs: "arm-message"
    backend._arm_pub = Publisher()
    backend._left_pub = Publisher(result=False)
    backend._right_pub = Publisher()
    backend._command_lock = threading.Lock()
    backend._dds_write_count = np.zeros(3, dtype=np.int64)
    backend._dds_write_failure_count = np.zeros(3, dtype=np.int64)
    backend._dds_write_error_reported = set()
    backend._published = False

    status = backend._write(
        np.zeros(29),
        5,
        np.zeros(14),
        np.asarray((0.25, 0.75)),
    )

    assert status == (True, False, True)
    assert backend._left_pub.messages[0].cmds[0].q == pytest.approx(
        0.25 * DEX1_MOTOR_OPEN_RAD
    )
    assert backend._right_pub.messages[0].cmds[0].q == pytest.approx(
        0.75 * DEX1_MOTOR_OPEN_RAD
    )
    np.testing.assert_array_equal(backend._dds_write_count, (1, 1, 1))
    np.testing.assert_array_equal(backend._dds_write_failure_count, (0, 1, 0))
    assert backend._published is True

    backend._left_pub = Publisher()
    backend._right_pub = Publisher(error=RuntimeError("writer failed"))
    status = backend._write(
        np.zeros(29),
        5,
        np.zeros(14),
        np.asarray((0.4, 0.6)),
    )
    assert status == (True, True, False)
    np.testing.assert_array_equal(backend._dds_write_count, (2, 2, 2))
    np.testing.assert_array_equal(backend._dds_write_failure_count, (0, 1, 1))

    left_count = len(backend._left_pub.messages)
    right_count = len(backend._right_pub.messages)
    status = backend._write(
        np.zeros(29),
        5,
        np.zeros(14),
        np.asarray((0.4, 0.6)),
        publish_dex1=False,
    )
    assert status == (True, None, None)
    assert len(backend._left_pub.messages) == left_count
    assert len(backend._right_pub.messages) == right_count
    np.testing.assert_array_equal(backend._dds_write_count, (3, 2, 2))


def test_real_release_is_not_reported_complete_when_arm_dds_write_fails() -> None:
    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend.config = load_teleop_config()
    backend._arm_sdk_weight = 0.1
    backend._release_log_bucket = 0
    backend._published = True
    backend._waist_anchor = np.zeros(3)
    backend._whole_body_hold_q = np.zeros(G1_29_LOWCMD_MOTOR_COUNT)
    backend._tracking = False
    backend._command_lock = threading.Lock()
    backend._release_complete = threading.Event()
    backend._write = lambda *_args, **_kwargs: (False, True, True)
    backend._dex1_publish_due = lambda: True

    backend._release_arm_sdk(
        np.zeros(G1_29_LOWCMD_MOTOR_COUNT), np.zeros(29), 5, np.zeros(2)
    )

    assert backend._published is True
    assert backend._release_complete.is_set() is False
    assert backend._arm_sdk_weight == pytest.approx(0.1)


def test_real_dex1_schedule_is_independent_official_200hz() -> None:
    assert RealDdsBackend.dex1_publish_period_s() == pytest.approx(
        1.0 / REAL_DEX1_HZ
    )
    assert REAL_DEX1_HZ != REAL_ARM_SDK_HZ


def test_real_applied_sequences_advance_only_after_successful_dds_write() -> None:
    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend._command_lock = threading.Lock()
    backend._applied_arm = np.zeros(14)
    backend._applied_arm_torque = np.zeros(14)
    backend._applied_hand = np.zeros(2)
    backend._last_applied_command_sequence = None
    backend._last_applied_dex1_command_sequence = [None, None]
    backend._dex1_tracking_error = np.zeros(2)
    target = ArmHandTarget(
        sequence=9,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=(0.1,) * 14,
        dex1_opening_fraction=(0.2, 0.8),
    )
    arm = np.full(14, 0.1)
    torque = np.full(14, 0.2)
    hand = np.asarray((0.2, 0.8))

    backend._confirm_successful_targets(
        target, arm, torque, hand, np.zeros(2), (False, True, None)
    )
    assert backend._last_applied_command_sequence is None
    assert backend._last_applied_dex1_command_sequence == [9, None]
    np.testing.assert_allclose(backend._applied_arm, 0.0)
    np.testing.assert_allclose(backend._applied_hand, (0.2, 0.0))

    backend._confirm_successful_targets(
        target, arm, torque, hand, np.zeros(2), (True, None, True)
    )
    assert backend._last_applied_command_sequence == 9
    assert backend._last_applied_dex1_command_sequence == [9, 9]
    np.testing.assert_allclose(backend._applied_arm, arm)
    np.testing.assert_allclose(backend._applied_hand, hand)


def test_hold_mode_keeps_the_captured_arm_and_dex1_target_under_safety_limits() -> None:
    config = load_teleop_config()
    safety = CommandSafetyFilter(config.safety, servo_hz=config.rates.servo_hz)
    measured_arm = np.zeros(14)
    measured_hand = np.ones(2)
    safety.reset(measured_arm, measured_hand)
    now_ns = 10_000_000_000
    captured_arm = tuple(np.linspace(-0.2, 0.2, 14))
    captured_hand = (0.6, 0.4)

    held = safety.apply(
        ArmHandTarget(
            sequence=1,
            monotonic_ns=now_ns,
            mode=ControlMode.HOLD,
            event=ControlEvent.NONE,
            arm_position_rad=captured_arm,
            dex1_opening_fraction=captured_hand,
        ),
        measured_arm_position_rad=measured_arm,
        measured_dex1_opening_fraction=measured_hand,
        now_ns=now_ns,
        last_command_ns=now_ns,
        tracking=True,
    )

    assert held.watchdog is WatchdogState.ACTIVE
    # The captured target is rate limited just like a normal tracking target;
    # HOLD never becomes the release-to-regular-controller IDLE path.
    assert np.max(np.abs(np.asarray(held.arm_position_rad))) > 0.0
    assert np.max(np.abs(np.asarray(held.arm_position_rad))) < 0.2
    assert np.all(np.asarray(held.dex1_opening_fraction) < 1.0)


def test_config_separates_operator_stereo_from_three_policy_cameras() -> None:
    config = load_teleop_config()

    assert config.operator_camera_roles == ("head_left", "head_right")
    assert config.policy_camera_keys == (
        "observation.images.cam_0",
        "observation.images.cam_2",
        "observation.images.cam_3",
    )
    assert config.rates.camera_hz == 30
    assert config.rates.camera_poll_hz == 120
    assert config.rates.record_hz == 30


def test_runtime_setup_handles_a_new_no_checkout_clone_before_dirty_check() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "setup_teleop_runtime.sh"
    ).read_text(encoding="utf-8")

    assert "new_clone=false" in script
    assert 'if [[ "$new_clone" != true ]]' in script
    assert 'git -C "$XR_ROOT" checkout --detach "$XR_REVISION"' in script
    assert "pinned XR runtime is not clean after checkout" in script
    assert 'XR_ENV=xr-teleop' in script
    assert "pinned-xr-runtime-imports-and-teleimage-contract-ok" in script
    assert "receive_teleimage" in script


def test_separate_teleop_launchers_enforce_their_own_runtime_contracts() -> None:
    sim_script = (
        Path(__file__).resolve().parents[1] / "run_sim_teleop.sh"
    ).read_text(encoding="utf-8")
    real_script = (
        Path(__file__).resolve().parents[1] / "run_real_teleop.sh"
    ).read_text(encoding="utf-8")

    for script in (real_script, sim_script):
        assert "AVP_DESKTOP_IP" in script
        assert (
            "openssl x509 -in \"$XR_CERT\" -noout -checkip "
            "\"$AVP_DESKTOP_IP\"" in script
        )
        assert '--session-report "$SESSION_REPORT"' in script
        assert "192.168.29.159" not in script

    assert "set AVP_DESKTOP_IP explicitly" in sim_script
    assert "set AVP_DESKTOP_IP explicitly" in real_script
    assert "iros_2026_ramen_g1_control.lock" in real_script
    assert "check_g1_regular_mode.py" in real_script
    assert 'XR_PYTHON="$("$CONDA_EXE" run -n "$XR_ENV"' in real_script
    assert 'exec env "PYTHONPATH=$PYTHONPATH_VALUE"' in real_script
    assert (
        '"$CONDA_EXE" run --no-capture-output -n "$XR_ENV"' not in real_script
    )

    assert "Open on Apple Vision Pro: https://$AVP_DESKTOP_IP:8012/" in sim_script
    assert "--exclude='model/subtask_policy_training/outputs/'" in sim_script
    assert "--exclude='model/subtask_policy_training/.venv_lerobot060/'" in sim_script
    assert 'RUN_OUTPUT_ID="$(basename "$LOCAL_RUNTIME_OUTPUT")"' in sim_script
    assert 'REMOTE_OUTPUT="$REMOTE_STAGE/runtime_output/$RUN_OUTPUT_ID"' in sim_script
    assert "--exclude='runtime_output/'" in sim_script
    assert "Waiting for the simulator camera bridge" in sim_script
    assert (
        'if [[ "$TRANSPORT_PROBE" == false && "$CONTROL_PROBE" == false ]]'
        in sim_script
    )
    assert 'tee "$LOCAL_RUNTIME_OUTPUT/operator.log"' in sim_script
    assert (
        'SESSION_REPORT="$LOCAL_RUNTIME_OUTPUT/operator_session_report.json"'
        in sim_script
    )
    assert 'FLIP_TABLE_SIM_EXECUTION' in sim_script
    assert 'SIM_EXECUTION=local' in sim_script
    assert 'SIM_EXECUTION=remote' in sim_script
    assert "G1_DDS_INTERFACE" not in sim_script
    assert "G1_IMAGE_SERVER_IP" not in sim_script

    assert '${G1_DDS_INTERFACE:?' in real_script
    assert '${G1_IMAGE_SERVER_IP:?' in real_script
    assert "data.flip_table_data_augmentation.teleop.real.runner" in real_script
    assert "data.flip_table_data_augmentation.teleop.sim.runner" not in real_script
    assert "docker info" not in real_script
    assert "nvidia-smi" not in real_script


def test_sim_recording_never_switches_live_avp_to_archival_jpeg() -> None:
    sim_script = (
        Path(__file__).resolve().parents[1] / "run_sim_teleop.sh"
    ).read_text(encoding="utf-8")
    policy = (
        Path(__file__).resolve().parents[3]
        / "evaluate/flip_table_simulation/container_overlay/policy/flip_table_eval_policy.py"
    ).read_text(encoding="utf-8")

    assert 'self._jpeg(image, recording=payload["recording"])' not in policy
    assert policy.count("self._jpeg(image, recording=False)") >= 2
    host_script = (
        Path(__file__).resolve().parents[1]
        / "teleop"
        / "simulator_host.sh"
    ).read_text(encoding="utf-8")
    assert 'ss -H -ltn "sport = :$port"' in host_script
    assert 'FLIP_TABLE_SIM_RENDER_INTERVAL="$render_interval"' in host_script
    assert (
        'FLIP_TABLE_SIMPLIFY_WHITE_COLLISION="$simplify_white_collision"'
        in host_script
    )
    assert 'FLIP_TABLE_TELEOP_PERSISTENT=true' in host_script
    assert 'process_running=false' in host_script
    assert '[[ "$pid" =~ ^[1-9][0-9]*$ ]]' in host_script
    assert 'SIM_OWNER="${FLIP_TABLE_TELEOP_SIM_OWNER:-one-shot}"' in sim_script
    assert 'FLIP_TABLE_TELEOP_SIM_OWNER=persistent' in sim_script
    assert 'Queued AVP job $PERSISTENT_JOB_ID on the existing Isaac worker.' in sim_script
    assert 'render-trajectory "$trajectory" "$OUTPUT_ROOT"' in sim_script
    assert sim_script.index("Simulator camera bridge is ready") < sim_script.index(
        "Open on Apple Vision Pro"
    )

    stop_script = (
        Path(__file__).resolve().parents[1] / "stop_teleop_sim.sh"
    ).read_text(encoding="utf-8")
    assert 'stop 0 "$REMOTE_STAGE" "$REMOTE_CONTAINER"' in stop_script
    assert 'Stopped persistent Isaac Sim container' in stop_script

    session = (
        Path(__file__).resolve().parents[1] / "teleop" / "session.py"
    ).read_text(encoding="utf-8")
    assert "camera_source_hz=" in session
    assert "camera_transport_hz=" in session
    assert "missing_camera_sequences=" in session
    assert "simulator camera source rate is too low" in session


def test_avp_policy_returns_completed_session_to_persistent_worker() -> None:
    policy = (
        Path(__file__).resolve().parents[3]
        / "evaluate"
        / "flip_table_simulation"
        / "container_overlay"
        / "policy"
        / "flip_table_eval_policy.py"
    ).read_text(encoding="utf-8")

    assert "def _persistent_sessions_enabled" in policy
    assert "def _release_client" in policy
    assert "def close(self)" in policy
    assert "while True:\n            self._accept_ready_client()" in policy
    assert "session ended safely; returning the queued job" in policy
    assert "so the persistent Isaac worker can accept the next job" in policy
    assert "self._last_event_sequence = -1" in policy


def test_real_collection_camera_config_is_role_explicit(tmp_path) -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "inference"
        / "desktop"
        / "xr"
        / "prepare_avp_collection_camera_config.py"
    )
    output = tmp_path / "cam_config_server.yaml"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--head-video-id",
            "6",
            "--left-d405-serial",
            "323622270214",
            "--right-d405-serial",
            "409122273599",
            "--output",
            str(output),
        ],
        check=True,
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert set(config) == {
        "head_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    }
    assert config["head_camera"]["image_shape"] == [480, 1280]
    assert config["head_camera"]["binocular"] is True
    for role in ("left_wrist_camera", "right_wrist_camera"):
        assert config[role]["type"] == "realsense"
        assert config[role]["enable_zmq"] is True
        assert config[role]["image_shape"] == [480, 640]

    launcher = script.with_name("launch_orin_avp_image_server.sh").read_text(
        encoding="utf-8"
    )
    assert "orin_teleimager_safe_launcher.py" in launcher
    assert "--check-only" in launcher
    assert "head-stereo + bilateral D405" in launcher
    assert "cam_config_server.avp_head_only.yaml" not in launcher
    root = Path(__file__).resolve().parents[3]
    service = (
        root
        / "inference/desktop/xr/systemd/avp_teleimager.service.in"
    ).read_text(encoding="utf-8")
    installer = (
        root / "inference/orin/scripts/install_avp_teleimager_service.sh"
    ).read_text(encoding="utf-8")
    assert "Restart=on-failure" in service
    assert "RestartSec=2" in service
    assert "launch_orin_avp_image_server.sh --run" in service
    assert "No running camera process was changed." in installer
    assert "--restart" in installer

    discovery_patch = (
        Path(__file__).resolve().parents[3]
        / "inference/desktop/xr/patches/teleimager_skip_uvc_discovery.patch"
    ).read_text(encoding="utf-8")
    assert "self.rs_serial_numbers = self._list_realsense_serial_numbers()" in discovery_patch


def test_dex1_service_hardening_is_pinned_and_thread_safe() -> None:
    root = Path(__file__).resolve().parents[3]
    patch = (
        root
        / "inference/desktop/xr/patches/dex1_1_service_thread_safety.patch"
    ).read_text(encoding="utf-8")
    installer = (
        root / "inference/orin/scripts/install_dex1_service_hardening.sh"
    ).read_text(encoding="utf-8")

    assert "cdd9fc5a78d51521eb262a56e0c5c19770700932" in installer
    assert "git -C \"$SOURCE_DIR\" apply --check" in installer
    assert "--untracked-files=no" in installer
    assert "--restart" in installer
    assert "copy_latest(MessageType& output)" in patch
    assert "std::lock_guard<std::mutex> lock(mutex_)" in patch
    assert "Both left and right Dex1-1 gripper motors must be online" in patch
    assert "motors_.size() == 2" in patch
    assert "Unexpected Dex1-1 motor ID" in patch
    assert "consecutive_comm_failures_" in patch
    assert "state_.merror" in patch


def test_avp_acceptance_report_requires_bilateral_arm_and_dex1_motion(tmp_path) -> None:
    config = load_teleop_config()
    timestamps = deque((index / 30.0 for index in range(61)), maxlen=61)
    arm_max = np.full(14, 0.1, dtype=np.float64)
    hand_max = np.full(2, 0.5, dtype=np.float64)
    audit = {
        "started_at_utc": "2026-07-20T00:00:00Z",
        "started_monotonic": time.monotonic() - 2.0,
        "avp_connected": True,
        "hand_tracking_hz": deque([27.4] * 61, maxlen=301),
        "hand_event_count_max": 61,
        "hand_invalid_event_count_max": 4,
        "hand_missing_pose_count_max": 2,
        "hand_invalid_wrist_count_max": 1,
        "hand_invalid_pinch_count_max": 1,
        "track_command_times": deque(timestamps, maxlen=301),
        "camera_receive_times": deque(timestamps, maxlen=61),
        "camera_capture_times": deque(timestamps, maxlen=61),
        "camera_sequences": deque(range(61), maxlen=61),
        "camera_received_count": 61,
        "missing_camera_sequences": 0,
        "sim_sender_drops": 0,
        "command_observation_latency_ms": deque([50.0] * 61, maxlen=301),
        "arm_tracking_errors_rad": deque([0.01] * 61, maxlen=4214),
        "hand_tracking_errors": deque([0.01] * 61, maxlen=602),
        "tracking_enabled_count": 2,
        "stale_stop_count": 0,
        "track_command_count": 61,
        "track_command_sequences": set(range(61)),
        "commanded_arm_min": np.zeros(14),
        "commanded_arm_max": arm_max,
        "observed_arm_min": np.zeros(14),
        "observed_arm_max": arm_max,
        "commanded_hand_min": np.zeros(2),
        "commanded_hand_max": hand_max,
        "observed_hand_min": np.zeros(2),
        "observed_hand_max": hand_max,
        "record_started_count": 1,
        "record_saved_count": 0,
        "record_discarded_count": 1,
        "reset_count": 1,
        "safe_quit_requested": True,
    }
    report = tmp_path / "passing.json"
    args = types.SimpleNamespace(backend="sim", dr_profile="mild", seed=101)

    _write_session_report(
        report,
        args=args,
        config=config,
        audit=audit,
        termination_reason="safe_quit",
        error=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["transport"]["valid_hand_events"] == 61
    assert payload["transport"]["invalid_hand_events"] == 4
    assert payload["transport"]["hand_event_rejections"] == {
        "missing_bilateral_pose": 2,
        "invalid_wrist_matrix": 1,
        "invalid_pinch_state": 1,
        "invalid_unused_skeleton_diagnostic": 0,
        "by_side_and_reason": {},
    }
    assert payload["checks"]["hand_tracking_hz_gte_25"] is True
    assert payload["checks"]["left_dex1_observed"] is True
    assert payload["checks"]["right_dex1_observed"] is True
    assert payload["checks"]["no_dds_write_failure"] is True

    audit["dds_write_count_arm_left_right"] = np.array([100, 100, 100])
    audit["dds_write_failure_count_arm_left_right"] = np.array([0, 1, 0])
    report = tmp_path / "dds_failure.json"
    _write_session_report(
        report,
        args=args,
        config=config,
        audit=audit,
        termination_reason="safe_quit",
        error=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["checks"]["no_dds_write_failure"] is False
    assert payload["tracking"]["dds_write_failure_count_arm_left_right"] == [0, 1, 0]
    audit["dds_write_failure_count_arm_left_right"] = np.zeros(3, dtype=np.int64)

    audit["commanded_hand_max"] = np.array([0.5, 0.1])
    audit["observed_hand_max"] = np.array([0.5, 0.1])
    report = tmp_path / "one_hand_only.json"
    _write_session_report(
        report,
        args=args,
        config=config,
        audit=audit,
        termination_reason="safe_quit",
        error=None,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["checks"]["left_dex1_commanded_continuously"] is True
    assert payload["checks"]["right_dex1_commanded_continuously"] is False


def test_televuer_patch_treats_a_closed_websocket_as_a_normal_reconnect(monkeypatch) -> None:
    class FakeTeleVuer:
        async def main_image_binocular_zmq(self, session):
            raise AssertionError("Websocket session is missing.")

        def __init__(self):
            self.img2display = np.ones((2, 2, 3), dtype=np.uint8)

        async def on_hand_move(self, event, session, fps=60):
            return fps

    module = types.SimpleNamespace(TeleVuer=FakeTeleVuer)
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)

    patched = _install_heartbeat_patch()
    instance = patched()
    stale_session = types.SimpleNamespace(
        CURRENT_WS_ID="old", vuer=types.SimpleNamespace(ws={"new": object()})
    )

    import asyncio

    assert asyncio.run(instance.main_image_binocular_zmq(stale_session)) is None
    assert np.count_nonzero(instance.img2display) == 0
    assert hasattr(instance, "team_ramen_session_heartbeat_ns")
    assert hasattr(instance, "team_ramen_hand_tracking_hz")


def test_televuer_hand_heartbeat_accepts_only_complete_finite_bilateral_frames(
    monkeypatch,
) -> None:
    class FakeTeleVuer:
        def __init__(self):
            self.upstream_hand_calls = 0

        async def on_hand_move(self, event, session, fps=60):
            self.upstream_hand_calls += 1
            return fps

        async def main_image_binocular_zmq(self, session):
            return None

    module = types.SimpleNamespace(TeleVuer=FakeTeleVuer)
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)
    instance = _install_heartbeat_patch()()

    import asyncio

    invalid = types.SimpleNamespace(value={"left": [0.0] * 400})
    asyncio.run(instance.on_hand_move(invalid, None))
    assert instance.team_ramen_hand_heartbeat_ns.value == 0
    assert instance.team_ramen_hand_window_event_count.value == 0
    assert instance.team_ramen_hand_invalid_event_count.value == 1
    assert instance.team_ramen_hand_missing_pose_count.value == 1
    assert instance.upstream_hand_calls == 0

    pose = np.tile(np.eye(4, dtype=np.float64).reshape(-1), 25).tolist()
    valid = types.SimpleNamespace(
        value={
            "left": pose,
            "right": pose,
            "leftState": {"pinchValue": 0.06},
            "rightState": {"pinchValue": 0.07},
        }
    )
    asyncio.run(instance.on_hand_move(valid, None))
    first_ns = instance.team_ramen_hand_heartbeat_ns.value
    asyncio.run(instance.on_hand_move(valid, None))

    assert first_ns > 0
    assert instance.team_ramen_hand_heartbeat_ns.value >= first_ns
    assert instance.team_ramen_hand_tracking_hz.value > 0.0
    upstream_calls_after_valid_events = instance.upstream_hand_calls

    asyncio.run(instance.on_hand_move(invalid, None))
    assert instance.team_ramen_hand_heartbeat_ns.value >= first_ns
    assert instance.team_ramen_hand_tracking_hz.value > 0.0
    assert instance.team_ramen_hand_window_event_count.value == 2
    assert instance.team_ramen_hand_invalid_event_count.value == 2
    assert instance.team_ramen_hand_missing_pose_count.value == 2
    assert instance.upstream_hand_calls == upstream_calls_after_valid_events

    invalid_matrix = types.SimpleNamespace(
        value={
            "left": [0.0] * 400,
            "right": pose,
            "leftState": {"pinchValue": 0.06},
            "rightState": {"pinchValue": 0.07},
        }
    )
    invalid_pinch = types.SimpleNamespace(
        value={
            "left": pose,
            "right": pose,
            "leftState": {"pinchValue": float("nan")},
            "rightState": {"pinchValue": 0.07},
        }
    )
    asyncio.run(instance.on_hand_move(invalid_matrix, None))
    asyncio.run(instance.on_hand_move(invalid_pinch, None))
    assert instance.team_ramen_hand_invalid_wrist_count.value == 1
    assert instance.team_ramen_hand_invalid_pinch_count.value == 1
    assert instance.team_ramen_hand_heartbeat_ns.value >= first_ns
    assert instance.upstream_hand_calls == upstream_calls_after_valid_events

    unused_invalid_pose = np.asarray(pose, dtype=np.float64)
    unused_invalid_pose[16] = float("nan")
    unused_invalid = types.SimpleNamespace(
        value={
            "left": unused_invalid_pose,
            "right": pose,
            "leftState": {"pinchValue": 0.06},
            "rightState": {"pinchValue": 0.07},
        }
    )
    asyncio.run(instance.on_hand_move(unused_invalid, None))
    assert instance.upstream_hand_calls == upstream_calls_after_valid_events + 1
    assert instance.team_ramen_hand_invalid_unused_skeleton_count.value == 1


def test_televuer_hand_rate_restarts_after_headset_inactivity(monkeypatch) -> None:
    class FakeTeleVuer:
        def __init__(self):
            pass

        async def on_hand_move(self, event, session, fps=60):
            return fps

        async def main_image_binocular_zmq(self, session):
            return None

    module = types.SimpleNamespace(TeleVuer=FakeTeleVuer)
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)
    instance = _install_heartbeat_patch()()
    timestamps = iter((1_000_000_000, 1_100_000_000, 2_000_000_000, 2_100_000_000))
    monkeypatch.setattr(xr_runtime_module.time, "monotonic_ns", lambda: next(timestamps))
    pose = np.tile(np.eye(4, dtype=np.float64).reshape(-1), 25).tolist()
    event = types.SimpleNamespace(
        value={
            "left": pose,
            "right": pose,
            "leftState": {"pinchValue": 0.06},
            "rightState": {"pinchValue": 0.07},
        }
    )

    import asyncio

    asyncio.run(instance.on_hand_move(event, None))
    asyncio.run(instance.on_hand_move(event, None))
    assert instance.team_ramen_hand_tracking_hz.value == pytest.approx(10.0)
    asyncio.run(instance.on_hand_move(event, None))
    assert instance.team_ramen_hand_tracking_hz.value == 0.0
    asyncio.run(instance.on_hand_move(event, None))
    assert instance.team_ramen_hand_tracking_hz.value == pytest.approx(10.0)
    assert instance.team_ramen_hand_event_count.value == 4


def test_televuer_hand_patch_preserves_complete_official_hand_handler(monkeypatch) -> None:
    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class SharedArray:
        def __init__(self, length):
            self.values = np.zeros(length, dtype=np.float64)
            self.lock = Lock()

        def __setitem__(self, key, value):
            self.values[key] = value

        def get_lock(self):
            return self.lock

    class SharedValue:
        def __init__(self, value):
            self.value = value
            self.lock = Lock()

        def get_lock(self):
            return self.lock

    class FakeTeleVuer:
        def __init__(self):
            self.official_hand_calls = 0
            self.left_arm_pose_shared = SharedArray(16)
            self.right_arm_pose_shared = SharedArray(16)
            for side in ("left", "right"):
                for name, initial in (
                    ("pinch", False),
                    ("pinchValue", 0.0),
                    ("squeeze", False),
                    ("squeezeValue", 0.0),
                ):
                    setattr(self, f"{side}_hand_{name}_shared", SharedValue(initial))

        async def on_hand_move(self, event, _session, fps=60):
            assert fps == 60
            self.official_hand_calls += 1
            for side in ("left", "right"):
                pose_value = np.asarray(event.value[side], dtype=np.float64)
                state = event.value[f"{side}State"]
                arm_pose = getattr(self, f"{side}_arm_pose_shared")
                with arm_pose.get_lock():
                    arm_pose[:] = pose_value[:16]
                for name, default, converter in (
                    ("pinch", False, bool),
                    ("pinchValue", 0.0, float),
                    ("squeeze", False, bool),
                    ("squeezeValue", 0.0, float),
                ):
                    shared = getattr(self, f"{side}_hand_{name}_shared")
                    with shared.get_lock():
                        shared.value = converter(state.get(name, default))

        async def main_image_binocular_zmq(self, session):
            return session

    module = types.SimpleNamespace(TeleVuer=FakeTeleVuer)
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)
    instance = _install_heartbeat_patch()()
    left = np.tile(np.eye(4, dtype=np.float64).reshape(-1), 25)
    right = left.copy()
    right[12] = 0.25
    event = types.SimpleNamespace(
        value={
            "left": left.tolist(),
            "right": right.tolist(),
            "leftState": {
                "pinch": True,
                "pinchValue": 0.052,
                "squeeze": False,
                "squeezeValue": 0.1,
            },
            "rightState": {
                "pinch": False,
                "pinchValue": 0.071,
                "squeeze": True,
                "squeezeValue": 0.8,
            },
        }
    )

    import asyncio

    asyncio.run(instance.on_hand_move(event, None))

    assert instance.left_arm_pose_shared.values == pytest.approx(left[:16])
    assert instance.right_arm_pose_shared.values == pytest.approx(right[:16])
    assert instance.left_hand_pinch_shared.value is True
    assert instance.left_hand_pinchValue_shared.value == pytest.approx(0.052)
    assert instance.right_hand_pinch_shared.value is False
    assert instance.right_hand_pinchValue_shared.value == pytest.approx(0.071)
    assert instance.team_ramen_hand_event_count.value == 1
    assert instance.official_hand_calls == 1


def test_televuer_patch_does_not_hide_an_active_session_failure(monkeypatch) -> None:
    class FakeTeleVuer:
        async def main_image_binocular_zmq(self, session):
            raise AssertionError("Websocket session is missing.")

        def __init__(self):
            pass

        async def on_hand_move(self, event, session, fps=60):
            return fps

    module = types.SimpleNamespace(TeleVuer=FakeTeleVuer)
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)
    instance = _install_heartbeat_patch()()
    active_session = types.SimpleNamespace(
        CURRENT_WS_ID="active", vuer=types.SimpleNamespace(ws={"active": object()})
    )

    import asyncio
    import pytest

    with pytest.raises(AssertionError, match="Websocket session is missing"):
        asyncio.run(instance.main_image_binocular_zmq(active_session))


def test_xr_display_mode_defaults_to_pass_through_stereo_window(monkeypatch) -> None:
    monkeypatch.delenv("FLIP_TABLE_TELEOP_XR_DISPLAY_MODE", raising=False)
    assert _xr_display_mode() == "ego"
    assert _xr_display_mode("immersive") == "immersive"
    with pytest.raises(ValueError, match="must be one of"):
        _xr_display_mode("pass-through")


@pytest.mark.parametrize(
    ("stream_name", "expected_height", "expected_distance"),
        (
            ("main_image_binocular_zmq", 1.0, 1.0),
            ("main_image_binocular_zmq_ego", 0.82, 1.60),
    ),
)
def test_televuer_stereo_stream_uses_two_eye_layers_and_deadline_pacing(
    monkeypatch, stream_name, expected_height, expected_distance
) -> None:
    elements = []

    class Element:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            elements.append(self)

    class FakeTeleVuer:
        def __init__(self):
            self.img2display = np.zeros((2, 4, 3), dtype=np.uint8)
            self.img_width = 2
            self.aspect_ratio = 1.0
            self.display_fps = 30.0

        async def on_hand_move(self, event, session, fps=60):
            return event, session, fps

        async def main_image_binocular_zmq(self, session):
            return session

        async def main_image_binocular_zmq_ego(self, session):
            return session

    module = types.SimpleNamespace(
        TeleVuer=FakeTeleVuer,
        Hands=Element,
        ImageBackground=Element,
    )
    monkeypatch.setitem(sys.modules, "televuer.televuer", module)
    instance = _install_heartbeat_patch()()

    class Session:
        CURRENT_WS_ID = "active"

        def __init__(self):
            self.vuer = types.SimpleNamespace(ws={"active": object()})
            self.calls = []

        def upsert(self, value, *, to):
            self.calls.append((value, to))
            if len(self.calls) == 2:
                self.vuer.ws.clear()
                raise AssertionError("Websocket session is missing.")

    import asyncio

    session = Session()
    assert asyncio.run(getattr(instance, stream_name)(session)) is None
    assert len(session.calls) == 2
    eye_elements = session.calls[1][0]
    assert [element.kwargs["layers"] for element in eye_elements] == [1, 2]
    assert all(element.kwargs["quality"] == 80 for element in eye_elements)
    assert all(element.kwargs["height"] == expected_height for element in eye_elements)
    assert all(
        element.kwargs["distanceToCamera"] == expected_distance
        for element in eye_elements
    )
    assert eye_elements[0].args[0].shape == (2, 2, 3)
    assert eye_elements[1].args[0].shape == (2, 2, 3)


def test_avp_liveness_requires_a_live_session_and_recent_hand_tracking() -> None:
    now_ns = 10_000_000_000
    xr = XrInput.__new__(XrInput)
    xr.wrapper = types.SimpleNamespace(
        tvuer=types.SimpleNamespace(
            team_ramen_session_heartbeat_ns=types.SimpleNamespace(
                value=now_ns - 50_000_000
            ),
            team_ramen_hand_heartbeat_ns=types.SimpleNamespace(
                value=now_ns - 500_000_000
            ),
            team_ramen_hand_tracking_hz=types.SimpleNamespace(value=29.5),
            team_ramen_hand_event_count=types.SimpleNamespace(value=42),
            team_ramen_hand_window_event_count=types.SimpleNamespace(value=16),
        )
    )

    assert xr.connected(now_ns, 0.20, 0.75) is True
    assert xr.connected(now_ns, 0.02, 0.75) is False
    assert xr.connected(now_ns, 0.20, 0.25) is False
    assert xr.connected(now_ns, 0.20, 0.20) is False
    assert xr.liveness(now_ns)["hand_tracking_hz"] == 29.5
    assert xr.liveness(now_ns)["hand_event_count"] == 42
    assert xr.liveness(now_ns)["hand_invalid_event_count"] == 0


def test_avp_liveness_treats_a_concurrent_heartbeat_update_as_age_zero() -> None:
    now_ns = 10_000_000_000
    xr = XrInput.__new__(XrInput)
    xr.wrapper = types.SimpleNamespace(
        tvuer=types.SimpleNamespace(
            team_ramen_session_heartbeat_ns=types.SimpleNamespace(value=now_ns + 1),
            team_ramen_hand_heartbeat_ns=types.SimpleNamespace(value=now_ns + 10_000),
            team_ramen_hand_tracking_hz=types.SimpleNamespace(value=29.5),
            team_ramen_hand_event_count=types.SimpleNamespace(value=7),
            team_ramen_hand_window_event_count=types.SimpleNamespace(value=7),
        )
    )

    assert xr.liveness(now_ns)["session_age_s"] == 0.0
    assert xr.liveness(now_ns)["hand_age_s"] == 0.0
    assert xr.connected(now_ns, 0.20, 0.75) is True


def test_avp_liveness_has_no_false_stale_during_ten_virtual_minutes() -> None:
    start_ns = 10_000_000_000
    session_heartbeat = types.SimpleNamespace(value=start_ns)
    hand_heartbeat = types.SimpleNamespace(value=start_ns)
    hand_rate = types.SimpleNamespace(value=30.0)
    hand_count = types.SimpleNamespace(value=1)
    hand_window_count = types.SimpleNamespace(value=3)
    xr = XrInput.__new__(XrInput)
    xr.wrapper = types.SimpleNamespace(
        tvuer=types.SimpleNamespace(
            team_ramen_session_heartbeat_ns=session_heartbeat,
            team_ramen_hand_heartbeat_ns=hand_heartbeat,
            team_ramen_hand_tracking_hz=hand_rate,
            team_ramen_hand_event_count=hand_count,
            team_ramen_hand_window_event_count=hand_window_count,
        )
    )

    poll_period_ns = round(1.0e9 / 60.0)
    hand_period_ns = round(1.0e9 / 30.0)
    session_period_ns = 50_000_000
    polls = 10 * 60 * 60
    for poll in range(polls):
        now_ns = start_ns + poll * poll_period_ns
        session_heartbeat.value = (
            start_ns
            + ((now_ns - start_ns) // session_period_ns) * session_period_ns
        )
        hand_index = (now_ns - start_ns) // hand_period_ns
        hand_heartbeat.value = start_ns + hand_index * hand_period_ns
        hand_count.value = int(hand_index) + 1
        hand_window_count.value = int(hand_index) + 3
        # Reproduce the worker update that can land after the monitor sampled
        # now_ns. This was the race that stopped the seed-140 AVP session.
        if poll % 257 == 0:
            hand_heartbeat.value = now_ns + 10_000
        assert xr.connected(now_ns, 0.20, 0.75) is True

    brief_gap_ns = hand_heartbeat.value + 199_000_000
    session_heartbeat.value = brief_gap_ns
    assert xr.connected(brief_gap_ns, 0.20, 0.20) is True

    stale_now_ns = hand_heartbeat.value + 200_000_001
    session_heartbeat.value = stale_now_ns
    assert xr.connected(stale_now_ns, 0.20, 0.20) is False


def test_logging_mp_compat_adapts_pinned_xr_api_without_replacing_logger(monkeypatch) -> None:
    calls = []

    class Logger:
        def setLevel(self, level):
            calls.append(("level", level))

    logger = Logger()
    module = types.SimpleNamespace(
        getLogger=lambda name: calls.append(("get", name)) or logger,
        basicConfig=lambda **kwargs: calls.append(("basic", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "logging_mp", module)

    install_logging_mp_compat()
    assert module.get_logger("teleimager", level=20) is logger
    module.basic_config(level=10)
    assert calls == [("get", "teleimager"), ("level", 20), ("basic", {"level": 10})]


def test_logging_mp_compat_adapts_latest_official_camel_case_api(monkeypatch) -> None:
    calls = []
    logger = object()
    module = types.SimpleNamespace(
        get_logger=lambda name: calls.append(("get", name)) or logger,
        basic_config=lambda **kwargs: calls.append(("basic", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "logging_mp", module)

    install_logging_mp_compat()
    assert module.getLogger("teleimager") is logger
    module.basicConfig(level=10)
    assert calls == [("get", "teleimager"), ("basic", {"level": 10})]


def test_observation_transport_preserves_both_eyes() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = FramedSocket(left_socket)
    receiver = FramedSocket(right_socket)
    try:
        source = replace(
            _observation(),
            camera_bundle_valid=False,
            camera_skew_ms=12.5,
            stale_roles=("left_wrist",),
            camera_stream_metadata={
                "left_wrist": {
                    "role": "left_wrist",
                    "jpeg_generation": 7,
                    "first_observed_ns": 123,
                    "age_ms": 205.0,
                    "fresh": False,
                    "source_fps": 30.0,
                }
            },
        )
        sender.send(source.to_message())
        restored = TeleopObservation.from_message(receiver.receive(timeout_s=1.0))
    finally:
        sender.close()
        receiver.close()

    assert restored.camera_jpeg["head_left"] == b"left-jpeg"
    assert restored.camera_jpeg["head_right"] == b"right-jpeg"
    assert restored.camera_jpeg["head_left"] != restored.camera_jpeg["head_right"]
    assert restored.camera_bundle_valid is False
    assert restored.camera_skew_ms == 12.5
    assert restored.stale_roles == ("left_wrist",)
    assert restored.camera_stream_metadata["left_wrist"]["jpeg_generation"] == 7
    assert restored.success is True


def test_observation_rejects_camera_metadata_for_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="camera_stream_metadata"):
        replace(
            _observation(),
            camera_stream_metadata={"not_a_camera": {"jpeg_generation": 1}},
        )


def test_sim_backend_retries_an_ssh_tunnel_before_the_remote_port_is_ready() -> None:
    config = load_teleop_config()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    host, port = listener.getsockname()
    ready = threading.Event()

    def server() -> None:
        first, _ = listener.accept()
        first.close()
        second, _ = listener.accept()
        transport = FramedSocket(second)
        transport.send(
            {
                "schema_version": MESSAGE_SCHEMA_VERSION,
                "type": "hello",
                "backend": "sim",
                "config_sha256": config.digest,
                "runtime_digest": config.runtime.robofinals_digest,
                "servo_hz": config.rates.servo_hz,
                "camera_hz": config.rates.camera_hz,
            }
        )
        ready.set()
        transport.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    try:
        backend = SimSocketBackend(host, port, config, connect_timeout_s=3.0)
        backend.close()
    finally:
        listener.close()
    assert ready.wait(timeout=1.0)
    thread.join(timeout=1.0)


def test_sim_backend_normalizes_remote_monotonic_time_before_action_alignment() -> None:
    """The sim and operator hosts cannot compare raw monotonic timestamps."""

    remote_capture = 9_000_000_000_000
    observation = TeleopObservation(
        sequence=1,
        capture_monotonic_ns=remote_capture,
        backend="sim",
        body_joint_position_rad=(0.0,) * 29,
        body_joint_velocity_rad_s=(0.0,) * 29,
        dex1_opening_fraction=(1.0, 1.0),
        applied_arm_target_rad=(0.0,) * 14,
        applied_dex1_opening_target=(1.0, 1.0),
        root_pose_xyzw=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        camera_capture_monotonic_ns={
            "head_left": remote_capture - 33_000_000,
            "head_right": remote_capture - 33_000_000,
            "left_wrist": remote_capture,
            "right_wrist": remote_capture,
        },
        camera_jpeg={
            "head_left": b"left-jpeg",
            "head_right": b"right-jpeg",
            "left_wrist": b"left-wrist-jpeg",
            "right_wrist": b"right-wrist-jpeg",
        },
        diagnostics={"privileged_policy_features": []},
    )

    normalized = SimSocketBackend._normalize_remote_clock(
        observation,
        local_receive_ns=3_000_000_000,
    )

    assert normalized.capture_monotonic_ns == 3_000_000_000
    assert normalized.camera_capture_monotonic_ns == {
        "head_left": 2_967_000_000,
        "head_right": 2_967_000_000,
        "left_wrist": 3_000_000_000,
        "right_wrist": 3_000_000_000,
    }
    assert normalized.diagnostics["transport_timing"] == {
        "clock_domain": "operator_monotonic_receive_time",
        "remote_observation_monotonic_ns": remote_capture,
        "remote_camera_capture_monotonic_ns": dict(observation.camera_capture_monotonic_ns),
        "remote_diagnostic_camera_capture_monotonic_ns": {},
        "local_receive_monotonic_ns": 3_000_000_000,
    }
    assert _remote_capture_seconds(normalized) == pytest.approx(
        remote_capture / 1.0e9
    )


def test_raw_episode_keeps_head_right_out_of_policy_schema(tmp_path) -> None:
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="sim",
            dr_profile="mild",
            seed=7,
            config_sha256="a" * 64,
            runtime_digest="sha256:" + "b" * 64,
        ),
    )
    writer.append(_recordable_observation(1), _target(1))
    writer.append(_recordable_observation(2), _target(2))
    output = writer.save(diagnostics={"checked": True}, success=True)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_camera_keys"] == [
        "observation.images.cam_0",
        "observation.images.cam_2",
        "observation.images.cam_3",
    ]
    assert manifest["operator_only_cameras"] == ["head_right"]
    assert manifest["camera_frame_contract"] == {
        "encoding": "JPEG",
        "width": 640,
        "height": 480,
        "color_mode": "RGB",
    }
    assert (output / "diagnostics" / "head_right" / "000000.jpg").is_file()
    assert not (output / "policy_cameras" / "head_right").exists()
    assert manifest["diagnostic_cameras"] == []
    assert not (output / "diagnostics" / "global").exists()


def test_real_episode_rejects_stale_dex1_feedback_without_dropping_arm_control(
    tmp_path,
) -> None:
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="real",
            dr_profile="real",
            seed=8,
            config_sha256="a" * 64,
            runtime_digest="xr-revision",
        ),
    )
    now_ns = 6_000_000_000
    observation = replace(
        _recordable_observation(1, now_ns=now_ns),
        backend="real",
        diagnostics={
            "privileged_policy_features": [],
            "dex1_state_stale_left_right": [True, False],
        },
    )

    with pytest.raises(FrameSynchronizationError, match="Dex1 feedback was stale"):
        writer.append(
            observation,
            replace(_target(1), monotonic_ns=now_ns),
        )
    writer.discard()


def test_real_episode_rejects_a_nonfresh_camera_bundle(tmp_path) -> None:
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="real",
            dr_profile="real",
            seed=8,
            config_sha256="a" * 64,
            runtime_digest="xr-revision",
        ),
    )
    now_ns = 6_000_000_000
    observation = replace(
        _recordable_observation(1, now_ns=now_ns),
        backend="real",
        camera_bundle_valid=False,
        camera_skew_ms=40.0,
        stale_roles=("right_wrist",),
    )

    with pytest.raises(FrameSynchronizationError, match="not a new synchronized"):
        writer.append(
            observation,
            replace(_target(1), monotonic_ns=now_ns),
        )
    writer.discard()


def test_unsuccessful_sim_trial_is_persisted_outside_the_training_discovery_root(tmp_path) -> None:
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="sim",
            dr_profile="mild",
            seed=7,
            config_sha256="a" * 64,
            runtime_digest="sha256:" + "b" * 64,
        ),
    )
    writer.append(_recordable_observation(1), _target(1))
    writer.append(_recordable_observation(2), _target(2))
    output = writer.save(
        diagnostics={"rejection_reasons": ["simulator_success_not_reached"]},
        success=False,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert output.parent == tmp_path / "rejected"
    assert manifest["success"] is False
    assert manifest["collection_disposition"] == "rejected_diagnostic"
    assert list(tmp_path.glob("*/manifest.json")) == []


def test_raw_episode_accepts_two_frame_camera_delay_with_scheduler_jitter(
    tmp_path,
) -> None:
    now = 5_000_000_000
    delayed = replace(
        _recordable_observation(1, now_ns=now),
        camera_capture_monotonic_ns={
            role: now - 75_000_000
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        },
    )
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="sim",
            dr_profile="mild",
            seed=7,
            config_sha256="a" * 64,
            runtime_digest="sha256:" + "b" * 64,
        ),
    )
    writer.append(delayed, replace(_target(1), monotonic_ns=now))
    writer.discard()


def test_raw_episode_rejects_camera_older_than_delay_and_scheduler_budget(
    tmp_path,
) -> None:
    now = 5_000_000_000
    stale = replace(
        _recordable_observation(1, now_ns=now),
        camera_capture_monotonic_ns={
            role: now - 110_000_000
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        },
    )
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="sim",
            dr_profile="mild",
            seed=7,
            config_sha256="a" * 64,
            runtime_digest="sha256:" + "b" * 64,
        ),
    )
    with pytest.raises(FrameSynchronizationError, match="ages_ms"):
        writer.append(stale, replace(_target(1), monotonic_ns=now))
    writer.discard()


def test_camera_delay_is_bounded_by_elapsed_time_not_only_frame_count() -> None:
    captures = (1_000_000_000, 1_040_000_000, 1_080_000_000)

    assert bounded_delay_steps(
        captures,
        2,
        maximum_age_ns=66_666_667,
    ) == 1
    assert bounded_delay_steps(
        captures,
        2,
        maximum_age_ns=80_000_000,
    ) == 2


def test_camera_delay_rejects_non_monotonic_capture_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        bounded_delay_steps(
            (1_000_000_000, 1_000_000_000),
            1,
            maximum_age_ns=66_666_667,
        )


def test_numeric_conversion_preserves_body_order_and_source_hand_scale() -> None:
    body = np.arange(29, dtype=np.float64) / 100.0
    arm = np.arange(14, dtype=np.float64) / 10.0
    desired = desired_body_q(body, arm)

    assert np.array_equal(desired[:15], body[:15])
    assert np.array_equal(desired[15:], arm)
    assert np.array_equal(
        compose_robot_q((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), body)[7:],
        body.astype(np.float32),
    )
    assert np.array_equal(demo_hand_value((0.0, 1.0)), np.array([0.0, 4.5], np.float32))


def test_numeric_features_use_same_frame_commanded_targets_and_only_six_source_keys() -> None:
    class FakeFk:
        def __call__(self, body):
            body = np.asarray(body)
            return np.full(12, body[15], dtype=np.float32)

    frame = {
        "root_pose_xyzw": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "body_joint_position_rad": [0.0] * 29,
        "dex1_opening_state": [0.25, 0.75],
        "commanded_arm_target_rad": [0.4] * 14,
        "commanded_dex1_opening_target": [0.0, 1.0],
        # These are one servo period old diagnostics and must never be used as
        # labels for the image/state captured in this frame.
        "applied_arm_target_rad": [-0.4] * 14,
        "applied_dex1_opening_target": [1.0, 0.0],
    }
    values = numeric_features(frame, fk=FakeFk())

    assert tuple(values) == (
        "observation.state.ee_state",
        "observation.state.hand_state",
        "observation.state.robot_q_current",
        "action.ee_action",
        "action.hand_cmd",
        "action.robot_q_desired",
    )
    assert np.all(values["observation.state.ee_state"] == 0.0)
    assert np.all(values["action.ee_action"] == 0.4)
    assert np.all(values["action.robot_q_desired"][22:] == 0.4)


def test_first_record_key_starts_but_does_not_immediately_save(monkeypatch, tmp_path) -> None:
    from data.flip_table_data_augmentation.teleop import session

    observation = _observation()
    session_report = tmp_path / "operator_session_report.json"

    class FakeBackend:
        def __init__(self):
            self.commands = []
            self.observe_count = 0
            self.closed = threading.Event()

        def observe(self, timeout_s):
            if self.observe_count == 0:
                assert timeout_s == session.INITIAL_OBSERVATION_TIMEOUT_S
            else:
                assert timeout_s == session.WARMUP_OBSERVATION_TIMEOUT_S
                self.closed.wait()
            self.observe_count += 1
            return observation

        def apply(self, target):
            self.commands.append(target)

        def close(self):
            self.closed.set()

    class FakeKeys:
        def __init__(self):
            self.values = iter(("r", None, "s", "q"))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            return next(self.values)

    class FakeWriter:
        instances = []

        def __init__(self, *args, **kwargs):
            self.episode_id = "test"
            self.frame_count = 0
            self.saved = False
            self.discarded = False
            self.instances.append(self)

        def append(self, observation, target):
            self.frame_count += 1

        def save(self, **kwargs):
            self.saved = True
            return tmp_path / "saved"

        def discard(self):
            self.discarded = True

    backend = FakeBackend()
    initialization_order = []

    class FakeOperator:
        instances = []

        def __init__(self, root, config, *, backend="sim"):
            assert backend == "sim"
            self.closed = False
            self.tracking_generation = 0
            self.camera_updates = []
            self.instances.append(self)

        def start(self):
            initialization_order.append("xr")

        def submit(self, **kwargs):
            self.tracking_generation = kwargs["tracking_generation"]
            self.camera_updates.append(kwargs["camera_jpeg"])

        def latest_target(self):
            return types.SimpleNamespace(
                monotonic_ns=time.monotonic_ns(),
                avp_live=True,
                tracking_generation=self.tracking_generation,
                arm_position_rad=(0.0,) * 14,
                arm_feedforward_torque_nm=(0.0,) * 14,
                dex1_opening_fraction=(1.0, 1.0),
                source_sequence=len(self.camera_updates),
                image_processing_ms=1.0,
                ik_processing_ms=2.0,
                total_processing_ms=3.0,
                session_age_s=0.01,
                hand_age_s=0.01,
                hand_tracking_hz=30.0,
                hand_event_count=120,
                hand_contiguous_event_count=120,
                hand_invalid_event_count=0,
                hand_missing_pose_count=0,
                hand_invalid_wrist_count=0,
                hand_invalid_pinch_count=0,
                hand_invalid_unused_skeleton_count=0,
                hand_invalid_details={},
            )

        def submitted_sequence(self):
            return len(self.camera_updates)

        def close(self):
            self.closed = True

    def make_backend(args, config):
        initialization_order.append("backend")
        return backend

    monkeypatch.setattr(session, "_make_backend", make_backend)
    monkeypatch.setattr(session, "OperatorProcess", FakeOperator)
    monkeypatch.setattr(session, "KeyReader", FakeKeys)
    monkeypatch.setattr(session, "ReplayTrajectoryWriter", FakeWriter)
    monkeypatch.setattr(
        session,
        "_decode_jpeg",
        lambda payload: np.zeros((480, 640, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "session.py",
            "sim",
            "--xr-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path),
            "--session-report",
            str(session_report),
        ],
    )

    assert session.main() == 0
    assert initialization_order == ["xr", "backend"]
    assert backend.commands[0].mode is ControlMode.IDLE
    assert backend.commands[0].event is ControlEvent.NONE
    assert backend.observe_count == 2
    # The same simulator frame is rendered once; subsequent high-rate monitor
    # ticks update AVP IK without resending or decoding stale JPEGs.
    assert len(FakeOperator.instances) == 1
    assert FakeOperator.instances[0].camera_updates[0] == {
        "head_left": b"left-jpeg",
        "head_right": b"right-jpeg",
    }
    assert FakeOperator.instances[0].camera_updates[1:] == [None, None, None]
    assert len(FakeWriter.instances) == 1
    assert FakeWriter.instances[0].saved is False
    assert FakeWriter.instances[0].discarded is True
    report = json.loads(session_report.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["operator_controls"]["record_started_count"] == 1
    assert report["operator_controls"]["record_discarded_count"] == 1
    assert report["operator_controls"]["safe_quit_requested"] is True


def test_sim_transport_probe_receives_head_stereo_without_xr(monkeypatch, tmp_path) -> None:
    from argparse import Namespace

    from data.flip_table_data_augmentation.teleop import session

    class FakeBackend:
        def __init__(self):
            self.commands = []
            self.closed = False

        def apply(self, target):
            self.commands.append(target)

        def observe(self, timeout_s):
            assert timeout_s == session.INITIAL_OBSERVATION_TIMEOUT_S
            observation = _observation()
            message = observation.to_message()
            message["camera_jpeg"] = {
                role: payload
                for role, payload in observation.camera_jpeg.items()
                if role in {"head_left", "head_right"}
            }
            message["camera_capture_monotonic_ns"] = {
                role: timestamp
                for role, timestamp in observation.camera_capture_monotonic_ns.items()
                if role in {"head_left", "head_right"}
            }
            return TeleopObservation.from_message(message)

        def close(self):
            self.closed = True

    backend = FakeBackend()
    monkeypatch.setattr(session, "_make_backend", lambda args, config: backend)
    monkeypatch.setattr(
        session,
        "_decode_jpeg",
        lambda payload: np.zeros((480, 640, 3), dtype=np.uint8),
    )

    assert session._run_transport_probe(
        Namespace(
            backend="sim",
            output_root=tmp_path / "raw",
            dr_profile="mild",
            seed=1,
        ),
        load_teleop_config(),
    ) == 0
    assert [command.event for command in backend.commands] == [
        ControlEvent.NONE,
        ControlEvent.QUIT,
    ]
    assert all(command.mode is ControlMode.IDLE for command in backend.commands)
    assert backend.closed is True


def test_golden_arm_hand_target_is_identical_for_sim_and_real_backends() -> None:
    """Both backends receive the exact same real-compatible target contract."""

    target = ArmHandTarget(
        sequence=73,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=tuple(np.linspace(-0.6, 0.7, 14)),
        dex1_opening_fraction=(0.2, 0.8),
        arm_feedforward_torque_nm=tuple(np.linspace(-2.0, 2.0, 14)),
    )

    class FakeTransport:
        def __init__(self) -> None:
            self.messages = []

        def send(self, message) -> None:
            self.messages.append(message)

    sim = SimSocketBackend.__new__(SimSocketBackend)
    sim.transport = FakeTransport()
    sim.apply(target)
    restored = ArmHandTarget.from_message(sim.transport.messages[0])
    assert restored == target

    real = RealDdsBackend.__new__(RealDdsBackend)
    real._command_lock = threading.Lock()
    real._latest_command = None
    real._last_command_ns = None
    real._tracking = False
    real._arm_interlock_reason = None
    real._arm_sdk_weight = 0.0
    real._release_complete = threading.Event()
    real._release_complete.set()
    real._arm_speed_ramp_started_ns = None
    real._whole_body_hold_q = None
    real._waist_anchor = None
    real._waist_guard_tripped = False
    real._mode_machine_anchor = None
    real._lower_body_peak_speed_rad_s = 0.0
    real._dex1_filter_history = []
    real._release_log_bucket = None
    real.apply(target)

    assert sim.transport.messages == [target.to_message()]
    assert real._latest_command == target
    assert real._tracking is True
    assert real._last_command_ns is not None

    held = replace(target, sequence=74, mode=ControlMode.HOLD)
    real.apply(held)

    assert real._latest_command == held
    assert real._tracking is True
    assert real._arm_sdk_weight == 0.0


def test_real_arm_interlock_rejects_repeated_track_until_idle_edge() -> None:
    target = ArmHandTarget(
        sequence=1,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=(0.0,) * 14,
        dex1_opening_fraction=(0.5, 0.5),
    )
    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend._command_lock = threading.Lock()
    backend._latest_command = None
    backend._last_command_ns = None
    backend._tracking = False
    backend._arm_sdk_weight = 0.5
    backend._whole_body_hold_q = np.zeros(G1_29_LOWCMD_MOTOR_COUNT)
    backend._waist_anchor = np.zeros(3)
    backend._waist_guard_tripped = False
    backend._arm_interlock_reason = "mode_machine changed"
    backend._release_complete = threading.Event()

    backend.apply(target)
    assert backend._tracking is False
    assert backend._arm_interlock_reason == "mode_machine changed"

    backend.apply(replace(target, sequence=2, mode=ControlMode.IDLE))
    assert backend._tracking is False
    assert backend._arm_interlock_reason == "mode_machine changed"

    backend.apply(replace(target, sequence=3))
    assert backend._tracking is False
    assert backend._arm_interlock_reason == "mode_machine changed"

    # This is set only after the servo confirms a successful weight=0 packet.
    backend._arm_sdk_weight = 0.0
    backend._release_complete.set()
    backend._arm_speed_ramp_started_ns = None
    backend._mode_machine_anchor = None
    backend._lower_body_peak_speed_rad_s = 0.0
    backend._dex1_filter_history = []
    backend._release_log_bucket = None
    backend.apply(replace(target, sequence=4))
    assert backend._tracking is True
    assert backend._arm_interlock_reason is None
    assert backend._release_complete.is_set() is False


def test_real_arm_sdk_message_matches_official_whole_body_hold_protocol() -> None:
    class Motor:
        def __init__(self) -> None:
            self.mode = None
            self.q = None
            self.dq = None
            self.tau = None
            self.kp = None
            self.kd = None

    class Message:
        def __init__(self) -> None:
            self.motor_cmd = [
                Motor() for _ in range(G1_29_LOWCMD_MOTOR_COUNT)
            ]
            self.mode_pr = None
            self.mode_machine = None
            self.crc = None

    class Crc:
        @staticmethod
        def Crc(message):
            del message
            return 123

    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend._types = {"LowCmd": Message}
    backend._crc = Crc()
    all_motor = (
        np.arange(G1_29_LOWCMD_MOTOR_COUNT, dtype=np.float64) / 10.0
    )
    arm = np.arange(14, dtype=np.float64) / 20.0

    torque = np.linspace(-1.3, 1.3, 14)
    message = backend._arm_message(
        all_motor, 5, arm, 0.4, feedforward_torque=torque
    )

    assert message.motor_cmd[29].q == pytest.approx(0.4)
    assert message.motor_cmd[29].mode == 1
    assert message.motor_cmd[29].kp == BODY_KP
    assert message.motor_cmd[29].kd == BODY_KD
    assert message.mode_pr == 0
    assert message.mode_machine == 5
    assert message.crc == 123
    for index in range(G1_29_LOWCMD_MOTOR_COUNT):
        if index == 29:
            continue
        command = message.motor_cmd[index]
        expected = arm[index - ARM_INDICES[0]] if index in ARM_INDICES else all_motor[index]
        assert command.q == pytest.approx(expected)
        if index in WEAK_INDICES:
            assert command.kp == WEAK_KP
            assert command.kd == WEAK_KD
        elif index in WRIST_INDICES:
            assert command.kp == WRIST_KP
            assert command.kd == WRIST_KD
        else:
            assert command.kp == BODY_KP
            assert command.kd == BODY_KD
        expected_tau = torque[index - ARM_INDICES[0]] if index in ARM_INDICES else 0.0
        assert command.tau == pytest.approx(expected_tau)
    for offset, index in enumerate(ARM_INDICES):
        command = message.motor_cmd[index]
        assert command.q == pytest.approx(arm[offset])
        if index in WRIST_INDICES:
            assert command.kp == WRIST_KP
            assert command.kd == WRIST_KD
        else:
            assert command.kp == WEAK_KP
            assert command.kd == WEAK_KD


def test_real_arm_sdk_waist_motion_diagnostic_detects_torso_deviation() -> None:
    body = np.zeros(29, dtype=np.float64)
    anchor = body[list(WAIST_GUARD_INDICES)].copy()
    assert RealDdsBackend._waist_deviation_exceeded(body, anchor) is False

    body[14] = 0.13
    assert RealDdsBackend._waist_deviation_exceeded(body, anchor) is True


def test_real_arm_sdk_state_watchdog_detects_stale_g1_or_dex1_feedback() -> None:
    now_ns = 10_000_000_000
    assert RealDdsBackend._state_is_stale(
        (now_ns - 100_000_000, now_ns - 100_000_000, now_ns - 100_000_000),
        now_ns=now_ns,
        maximum_age_s=0.2,
    ) is False
    assert RealDdsBackend._state_is_stale(
        (now_ns - 201_000_000, now_ns, now_ns),
        now_ns=now_ns,
        maximum_age_s=0.2,
    ) is True


def test_real_arm_sdk_uses_official_250hz_blend_clock() -> None:
    backend = RealDdsBackend.__new__(RealDdsBackend)
    backend._arm_sdk_weight = ARM_SDK_MAX_WEIGHT - 0.001

    assert backend._next_arm_sdk_weight(active=True, releasing=False) == pytest.approx(
        ARM_SDK_MAX_WEIGHT
    )
    backend._arm_sdk_weight = 1.0
    release_steps = int(REAL_ARM_SDK_HZ * ARM_SDK_BLEND_OUT_S)
    for _ in range(release_steps - 1):
        backend._next_arm_sdk_weight(active=False, releasing=True)
    assert backend._arm_sdk_weight == pytest.approx(1.0 / release_steps)
    assert backend._next_arm_sdk_weight(
        active=False, releasing=True
    ) == pytest.approx(0.0)


def test_real_lower_body_motion_is_observed_not_used_as_an_arm_target() -> None:
    # Official --motion keeps Regular mode responsible for balance/walking.
    # The only operator-controlled indices in the LowCmd remain 15..28.
    assert ARM_INDICES == tuple(range(15, 29))
    assert set(ARM_INDICES).isdisjoint(LOWER_BODY_INDICES)


def test_arm_limits_keep_margin_inside_official_g1_29_urdf() -> None:
    safety = load_teleop_config().safety
    lower_margin = np.asarray(safety.arm_position_lower_rad) - np.asarray(
        OFFICIAL_G1_29_ARM_LOWER_RAD
    )
    upper_margin = np.asarray(OFFICIAL_G1_29_ARM_UPPER_RAD) - np.asarray(
        safety.arm_position_upper_rad
    )

    np.testing.assert_allclose(lower_margin, 0.03, atol=1.0e-9)
    np.testing.assert_allclose(upper_margin, 0.03, atol=1.0e-9)


def test_collection_audit_validates_policy_isolation_and_camera_files(tmp_path) -> None:
    from data.flip_table_data_augmentation.scripts.audit_teleop_collection import (
        audit_collection,
    )

    config = load_teleop_config()
    now = time.monotonic_ns()
    payloads = {
        "head_left": _jpeg((255, 0, 0)),
        "head_right": _jpeg((0, 255, 0)),
        "left_wrist": _jpeg((0, 0, 255)),
        "right_wrist": _jpeg((255, 255, 0)),
    }
    randomization = {
        "profile_level": 0.25,
        "table": {"yaw_delta_rad": 0.2},
        "robot": {},
        "contact_materials": {"pairs": {}},
        "camera_mounts": {},
        "camera_image": {
            "rigs": {"head_stereo": {}, "left_wrist": {}, "right_wrist": {}}
        },
        "control": {},
        "lighting": {},
        "room": {},
        "policy_uses_privileged_values": False,
    }
    observation = TeleopObservation(
        sequence=1,
        capture_monotonic_ns=now,
        backend="sim",
        body_joint_position_rad=(0.0,) * 29,
        body_joint_velocity_rad_s=(0.0,) * 29,
        dex1_opening_fraction=(1.0, 1.0),
        applied_arm_target_rad=(0.0,) * 14,
        applied_dex1_opening_target=(1.0, 1.0),
        root_pose_xyzw=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        camera_capture_monotonic_ns={role: now for role in payloads},
        camera_jpeg=payloads,
        success=True,
        diagnostics={
            "randomization": randomization,
            "privileged_policy_features": [],
            "sim_control_contract": SIM_CONTROL_CONTRACT,
        },
    )
    writer = RawEpisodeWriter(
        tmp_path,
        EpisodeIdentity(
            backend="sim",
            dr_profile="mild",
            seed=7,
            config_sha256=config.digest,
            runtime_digest=config.runtime.robofinals_digest,
        ),
    )
    writer.append(observation, _target(1))
    writer.append(_recordable_observation(2, now_ns=now + 33_000_000), _target(2))
    writer.save(
        diagnostics={
            "success_source": "simulator_validation",
            "randomization": randomization,
        },
        success=True,
    )

    report = audit_collection(tmp_path)

    # One valid episode is recognized but cannot pass the deliberately strict
    # 10/10/10 collection release gate.
    assert report["errors"] == {}
    assert report["actual_successes_by_profile"] == {"mild": 1}
    assert report["status"] == "failed"
