from __future__ import annotations

import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from data.flip_table_data_augmentation.teleop.config import load_teleop_config
from data.flip_table_data_augmentation.teleop.shared.policy_contract import (
    action_16d,
    state_19d,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (
    CommandSequence,
    MODEL_DEX1_OPEN_VALUE,
    PolicyActionLimiter,
    PolicyStartPoseHold,
    command_from_action,
    current_camera_skew_ms,
    is_fresh_policy_observation,
    run_blocking_check_with_pose_hold,
    run_arm_pre_motion,
    return_arms_before_release,
    validate_policy_chunk,
    validate_runtime_backend,
    validate_state_distribution,
    verify_regular_mode,
    verify_regular_mode_after_release,
    wait_for_policy_start_with_hold,
)
from inference.desktop.upper_policy.act_pick_leg_contract import (
    DATASET_INITIAL_DEX1_OPENING_FRACTION,
)
from inference.desktop.upper_policy.pre_motion import (
    ARM_PRE_MOTION_WAYPOINTS,
    ArmPreMotionWaypoint,
    FORWARD_HIGH_ARM_POSE_RAD,
    FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD,
    ELBOW_OUTWARD_CLEARANCE_RAD,
    LATERAL_HIGH_ARM_POSE_RAD,
    SHOULDER_PITCH_BACKWARD_RAD,
    SHOULDER_ROLL_LATERAL_RAD,
    validate_arm_pre_motion_waypoints,
    build_arm_pre_motion_waypoints,
    build_arm_return_waypoints,
)
from inference.desktop.upper_policy.subtask_start_pose import (
    COARSE_INSERT_FRAME0,
    FLIP_TABLE_V1_FRAME0,
    FLIP_TABLE_V2_FRAME0,
    PICK_LEG_FRAME0,
    PICK_LEG_ACT_EP2101_FRAME0,
    subtask_start_pose_for_model,
)
from inference.desktop.upper_policy.worker_protocol import (
    receive_message,
    send_message,
)


def test_worker_protocol_round_trip() -> None:
    stream = io.BytesIO()
    value = {"images": [b"jpeg"], "state": list(range(19))}
    send_message(stream, value)
    stream.seek(0)
    restored = receive_message(stream)
    assert restored["images"] == [b"jpeg"]
    assert restored["state"] == list(range(19))


def test_policy_chunk_is_arms_and_absolute_dex1_only() -> None:
    config = load_teleop_config()
    measured = np.zeros(14)
    actions = np.zeros((16, 16))
    actions[:, 14:] = MODEL_DEX1_OPEN_VALUE * np.asarray([0.25, 0.75])
    report = validate_policy_chunk(
        actions,
        measured_arm=measured,
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
    )
    assert report["initial_arm_delta_max_rad"] == 0.0
    command = command_from_action(1, actions[0])
    assert len(command.arm_position_rad) == 14
    assert command.dex1_opening_fraction == pytest.approx((0.25, 0.75))
    assert command.arm_feedforward_torque_nm == pytest.approx((0.0,) * 14)
    assert not hasattr(command, "waist_position_rad")


def test_policy_chunk_checks_only_the_prefix_that_will_be_executed() -> None:
    config = load_teleop_config()
    actions = np.zeros((16, 16), dtype=np.float64)
    actions[:, 14:] = 2.0
    # This discontinuity is in the unused tail when execution_steps=8. It is
    # still diagnosed, but cannot reject a prefix that will be replanned first.
    actions[10:, 3] = 0.26

    report = validate_policy_chunk(
        actions,
        measured_arm=np.zeros(14),
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
        execution_steps=8,
    )

    assert report["chunk_step_delta_max_rad"] == pytest.approx(0.0)
    assert report["full_chunk_step_delta_max_rad"] == pytest.approx(0.26)
    assert report["validated_execution_steps"] == pytest.approx(8)


def test_coarse_raw_step_is_plausibility_checked_before_rate_limiting() -> None:
    config = load_teleop_config()
    actions = np.zeros((16, 16), dtype=np.float64)
    actions[:, 14:] = 2.0
    actions[1:, 4] = 0.2606

    report = validate_policy_chunk(
        actions,
        measured_arm=np.zeros(14),
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.30,
        execution_steps=8,
    )
    assert report["chunk_step_delta_max_rad"] == pytest.approx(0.2606)

    limiter = PolicyActionLimiter(
        np.zeros(14),
        np.full(2, 2.0 / MODEL_DEX1_OPEN_VALUE),
        command_hz=30.0,
        arm_velocity_rad_s=0.5,
        arm_acceleration_rad_s2=2.0,
        hand_velocity_fraction_s=0.5,
        hand_acceleration_fraction_s2=2.0,
    )
    transmitted = np.stack([limiter.apply(action) for action in actions[:8]])
    transmitted_velocity = np.diff(
        np.vstack((np.zeros((1, 16)), transmitted)), axis=0
    )[:, :14] * 30.0
    transmitted_acceleration = np.diff(
        np.vstack((np.zeros((1, 14)), transmitted_velocity)), axis=0
    ) * 30.0
    assert np.max(np.abs(transmitted_velocity)) <= 0.5 + 1.0e-9
    assert np.max(np.abs(transmitted_acceleration)) <= 2.0 + 1.0e-9

    actions[1:, 4] = 0.31
    with pytest.raises(
        ValueError,
        match=r"0\.3100 rad step.*transition=0->1, arm joint=4",
    ):
        validate_policy_chunk(
            actions,
            measured_arm=np.zeros(14),
            config=config,
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.30,
            execution_steps=8,
        )


def test_state_and_action_dimension_order_and_dex1_units() -> None:
    body = np.arange(29, dtype=np.float64)
    state = state_19d(body, (0.25, 0.75))
    assert state.shape == (19,)
    assert state[:3].tolist() == [12.0, 13.0, 14.0]
    assert state[3:17].tolist() == list(np.arange(15.0, 29.0))
    assert state[17:].tolist() == pytest.approx([1.125, 3.375])

    arms = np.arange(14, dtype=np.float64) / 10.0
    action = action_16d(arms, (0.25, 0.75))
    assert action.shape == (16,)
    assert action[:14].tolist() == pytest.approx(arms)
    assert action[14:].tolist() == pytest.approx([1.125, 3.375])
    command = command_from_action(7, action)
    assert command.arm_position_rad == pytest.approx(arms)
    assert command.dex1_opening_fraction == pytest.approx((0.25, 0.75))
    torque = np.linspace(-2.0, 2.0, 14)
    compensated = command_from_action(
        8, action, arm_feedforward_torque_nm=torque
    )
    assert compensated.arm_feedforward_torque_nm == pytest.approx(torque)


def _camera_observation(
    generations: tuple[int, int, int],
    times_ns: tuple[int, int, int],
) -> SimpleNamespace:
    return SimpleNamespace(
        stale_roles=(),
        camera_stream_metadata={
            role: {"jpeg_generation": generation}
            for role, generation in zip(
                ("head_left", "left_wrist", "right_wrist"),
                generations,
                strict=True,
            )
        },
        camera_capture_monotonic_ns={
            role: timestamp
            for role, timestamp in zip(
                ("head_left", "left_wrist", "right_wrist"),
                times_ns,
                strict=True,
            )
        },
    )


def test_policy_camera_history_requires_every_role_to_advance_and_actual_skew() -> None:
    first = _camera_observation((10, 20, 30), (1_000_000_000,) * 3)
    assert current_camera_skew_ms(first) == 0.0
    assert is_fresh_policy_observation(first, None, maximum_skew_ms=33.4)

    only_head_advanced = _camera_observation(
        (11, 20, 30), (1_033_000_000, 1_000_000_000, 1_000_000_000)
    )
    assert not is_fresh_policy_observation(
        only_head_advanced, (10, 20, 30), maximum_skew_ms=33.4
    )
    excessive_skew = _camera_observation(
        (11, 21, 31), (1_000_000_000, 1_020_000_000, 1_040_000_000)
    )
    assert not is_fresh_policy_observation(
        excessive_skew, (10, 20, 30), maximum_skew_ms=33.4
    )


def test_policy_action_limiter_respects_velocity_and_acceleration() -> None:
    limiter = PolicyActionLimiter(
        np.zeros(14),
        np.zeros(2),
        command_hz=30,
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        hand_velocity_fraction_s=1.0,
        hand_acceleration_fraction_s2=4.0,
    )
    previous_arm = limiter.arm.copy()
    previous_hand = limiter.hand.copy()
    previous_arm_velocity = np.zeros(14)
    previous_hand_velocity = np.zeros(2)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        desired = np.concatenate(
            (rng.uniform(-2.0, 2.0, 14), 4.5 * rng.uniform(0.0, 1.0, 2))
        )
        emitted = limiter.apply(desired)
        arm = emitted[:14]
        hand = emitted[14:] / 4.5
        arm_velocity = (arm - previous_arm) * 30.0
        hand_velocity = (hand - previous_hand) * 30.0
        assert np.max(np.abs(arm_velocity)) <= 1.0 + 1.0e-9
        assert np.max(np.abs(hand_velocity)) <= 1.0 + 1.0e-9
        assert np.max(np.abs((arm_velocity - previous_arm_velocity) * 30.0)) <= 4.0 + 1.0e-8
        assert np.max(np.abs((hand_velocity - previous_hand_velocity) * 30.0)) <= 4.0 + 1.0e-8
        previous_arm = arm
        previous_hand = hand
        previous_arm_velocity = arm_velocity
        previous_hand_velocity = hand_velocity


def test_live_state_guard_allows_tiny_regular_waist_motion_only(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    stats = {
        "observation.state": {
            "min": [0.0] * 19,
            "max": [1.0] * 19,
            "mean": [0.5] * 19,
            "std": [0.2] * 19,
        }
    }
    (checkpoint / "normalization.json").write_text(json.dumps(stats))

    def observation(first_state_value: float) -> SimpleNamespace:
        body = np.full(29, 0.5)
        body[12] = first_state_value
        return SimpleNamespace(
            body_joint_position_rad=body,
            dex1_opening_fraction=(0.5 / 4.5, 0.5 / 4.5),
            camera_capture_monotonic_ns={
                "head_left": 1_000_000_000,
                "left_wrist": 1_001_000_000,
                "right_wrist": 1_002_000_000,
            },
        )

    report = validate_state_distribution(
        [observation(-0.01), observation(-0.01)], checkpoint
    )
    assert report["state_outside_training_value_count"] == 2
    assert report["state_training_range_excursion_max"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="materially outside"):
        validate_state_distribution(
            [observation(-1.0), observation(-1.0)], checkpoint
        )


def test_runtime_backend_guard_is_fail_closed() -> None:
    healthy = SimpleNamespace(
        stale_roles=(),
        diagnostics={
            "lower_body_policy_command_dimensions": 0,
            "regular_mode_owns_lower_body": True,
            "arm_interlock_reason": None,
            "dex1_thread_error": None,
            "dds_write_failure_count_arm_left_right": [0, 0, 0],
        },
    )
    validate_runtime_backend(healthy)
    for key, value, match in (
        ("arm_interlock_reason", "lowstate stale", "interlock"),
        ("dex1_thread_error", "publisher failed", "Dex1"),
        ("dds_write_failure_count_arm_left_right", [1, 0, 0], "DDS"),
    ):
        broken = SimpleNamespace(
            stale_roles=(),
            diagnostics={**healthy.diagnostics, key: value},
        )
        with pytest.raises(RuntimeError, match=match):
            validate_runtime_backend(broken)
    with pytest.raises(RuntimeError, match="camera"):
        validate_runtime_backend(
            SimpleNamespace(stale_roles=("left_wrist",), diagnostics=healthy.diagnostics)
        )


def test_regular_recheck_inherits_the_same_control_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.lower_policy.actuators import g1_control_lock
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(g1_control_lock, "current_g1_control_lock_fd", lambda: 17)
    monkeypatch.setattr(
        g1_control_lock,
        "G1_CONTROL_LOCK_PATH",
        Path("/tmp/test-g1-control.lock"),
    )
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    verify_regular_mode("test0")
    assert captured["pass_fds"] == (17,)
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["IROS_G1_CONTROL_LOCK_FD"] == "17"
    assert environment["IROS_G1_CONTROL_LOCK_PATH"] == "/tmp/test-g1-control.lock"
    assert captured["command"][-2:] == ["--interface", "test0"]


def test_post_acquisition_regular_recheck_allows_only_arm_sdk_submodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.lower_policy.actuators import g1_control_lock
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(g1_control_lock, "current_g1_control_lock_fd", lambda: 17)
    monkeypatch.setattr(
        g1_control_lock,
        "G1_CONTROL_LOCK_PATH",
        Path("/tmp/test-g1-control.lock"),
    )
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    verify_regular_mode("test0", arm_sdk_active=True)
    assert captured["command"][-6:] == [
        "--interface",
        "test0",
        "--allowed-fsm-mode",
        "0",
        "--allowed-fsm-mode",
        "1",
    ]


def test_post_release_regular_check_retries_then_requires_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    calls: list[tuple[str, bool]] = []

    def fake_verify(interface: str, *, arm_sdk_active: bool = False) -> None:
        calls.append((interface, arm_sdk_active))
        if len(calls) < 3:
            raise runner.subprocess.CalledProcessError(1, ["check"])

    monkeypatch.setattr(runner, "verify_regular_mode", fake_verify)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    verify_regular_mode_after_release("test0", attempts=3)
    assert calls == [("test0", False)] * 3


def test_policy_chunk_rejects_large_first_or_interstep_motion() -> None:
    config = load_teleop_config()
    measured = np.zeros(14)
    actions = np.zeros((16, 16))
    with pytest.raises(ValueError, match="first policy arm target"):
        bad = actions.copy()
        bad[0:, 0] = 0.21
        validate_policy_chunk(
            bad,
            measured_arm=measured,
            config=config,
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
        )
    with pytest.raises(ValueError, match="policy chunk contains"):
        bad = actions.copy()
        bad[1:, 0] = 0.21
        validate_policy_chunk(
            bad,
            measured_arm=measured,
            config=config,
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
        )


def test_read_only_chunk_validation_reports_initial_delta_without_enabling_it() -> None:
    config = load_teleop_config()
    measured = np.zeros(14)
    actions = np.zeros((16, 16))
    actions[:, 0] = 0.21

    report = validate_policy_chunk(
        actions,
        measured_arm=measured,
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
        enforce_initial_delta=False,
    )

    assert report["initial_arm_delta_max_rad"] == pytest.approx(0.21)


def test_read_only_chunk_validation_reports_step_delta_without_enabling_it() -> None:
    config = load_teleop_config()
    actions = np.zeros((16, 16))
    actions[1:, 0] = 0.21

    report = validate_policy_chunk(
        actions,
        measured_arm=np.zeros(14),
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
        enforce_step_delta=False,
    )

    assert report["chunk_step_delta_max_rad"] == pytest.approx(0.21)


def test_policy_chunk_rejects_invalid_gripper_or_joint_limit() -> None:
    config = load_teleop_config()
    measured = np.zeros(14)
    actions = np.zeros((16, 16))
    actions[0, 14] = MODEL_DEX1_OPEN_VALUE * 1.06
    with pytest.raises(ValueError, match="Dex1"):
        validate_policy_chunk(
            actions,
            measured_arm=measured,
            config=config,
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
        )
    actions = np.zeros((16, 16))
    actions[:, 0] = config.safety.arm_position_upper_rad[0] + 0.01
    with pytest.raises(ValueError, match="hardware margin"):
        validate_policy_chunk(
            actions,
            measured_arm=actions[0, :14],
            config=config,
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
        )


def test_policy_chunk_allows_and_reports_small_zscore_gripper_extrapolation() -> None:
    config = load_teleop_config()
    measured = np.zeros(14)
    actions = np.zeros((16, 16))
    actions[:, 14] = MODEL_DEX1_OPEN_VALUE + 0.01
    actions[:, 15] = -0.01

    report = validate_policy_chunk(
        actions,
        measured_arm=measured,
        config=config,
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
    )

    assert report["dex1_min_fraction"] == 0.0
    assert report["dex1_max_fraction"] == 1.0
    assert report["dex1_raw_min_scalar"] == pytest.approx(-0.01)
    assert report["dex1_raw_max_scalar"] == pytest.approx(
        MODEL_DEX1_OPEN_VALUE + 0.01
    )
    assert report["dex1_clamp_max_scalar"] == pytest.approx(0.01)


def _healthy_runtime_observation(
    arm: np.ndarray,
    hand: np.ndarray,
) -> SimpleNamespace:
    return SimpleNamespace(
        arm_joint_position_rad=tuple(arm),
        dex1_opening_fraction=tuple(hand),
        stale_roles=(),
        diagnostics={
            "lower_body_policy_command_dimensions": 0,
            "regular_mode_owns_lower_body": True,
            "arm_interlock_reason": None,
            "dex1_thread_error": None,
            "dds_write_failure_count_arm_left_right": [0, 0, 0],
        },
    )


class _FollowingPreMotionBackend:
    def __init__(self, *, follow: bool = True) -> None:
        self.arm = np.zeros(14, dtype=np.float64)
        self.hand = np.asarray([0.31, 0.72], dtype=np.float64)
        self.follow = follow
        self.commands: list[object] = []

    def observe(self, timeout_s: float) -> SimpleNamespace:
        assert timeout_s > 0.0
        return _healthy_runtime_observation(self.arm, self.hand)

    def apply(self, command: object) -> None:
        self.commands.append(command)
        if self.follow:
            self.arm = np.asarray(command.arm_position_rad, dtype=np.float64)
            self.hand = np.asarray(
                command.dex1_opening_fraction, dtype=np.float64
            )


class _FakeGravityCompensator:
    def torque_nm(self, arm_position_rad: np.ndarray) -> np.ndarray:
        arm = np.asarray(arm_position_rad, dtype=np.float64)
        assert arm.shape == (14,)
        return np.linspace(-1.3, 1.3, 14)


def test_pre_motion_waypoints_are_arm_only_and_collision_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_teleop_config()
    validate_arm_pre_motion_waypoints(
        config.safety.arm_position_lower_rad,
        config.safety.arm_position_upper_rad,
    )
    assert [waypoint.name for waypoint in ARM_PRE_MOTION_WAYPOINTS] == [
        "shoulder_pitch_backward_clearance",
        "lateral_high_clearance",
        "forward_outward_clearance",
        "forward_high_ready",
    ]
    initial = np.linspace(-0.3, 0.3, 14)
    shoulder_only = ARM_PRE_MOTION_WAYPOINTS[0].resolve(initial)
    assert shoulder_only[0] == pytest.approx(SHOULDER_PITCH_BACKWARD_RAD)
    assert shoulder_only[7] == pytest.approx(SHOULDER_PITCH_BACKWARD_RAD)
    np.testing.assert_allclose(
        shoulder_only[[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]],
        initial[[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]],
    )
    already_rearward = initial.copy()
    already_rearward[[0, 7]] = (0.95, 1.0)
    np.testing.assert_allclose(
        ARM_PRE_MOTION_WAYPOINTS[0].resolve(already_rearward)[[0, 7]],
        (0.95, 1.0),
    )
    assert len(LATERAL_HIGH_ARM_POSE_RAD) == 14
    assert LATERAL_HIGH_ARM_POSE_RAD[0] == pytest.approx(
        SHOULDER_PITCH_BACKWARD_RAD
    )
    assert LATERAL_HIGH_ARM_POSE_RAD[7] == pytest.approx(
        SHOULDER_PITCH_BACKWARD_RAD
    )
    assert SHOULDER_PITCH_BACKWARD_RAD == pytest.approx(0.85)
    assert LATERAL_HIGH_ARM_POSE_RAD[1] == pytest.approx(
        SHOULDER_ROLL_LATERAL_RAD
    )
    assert LATERAL_HIGH_ARM_POSE_RAD[8] == pytest.approx(
        -SHOULDER_ROLL_LATERAL_RAD
    )
    lateral = ARM_PRE_MOTION_WAYPOINTS[1].resolve(initial)
    assert lateral[0] == pytest.approx(SHOULDER_PITCH_BACKWARD_RAD)
    assert lateral[1] == pytest.approx(SHOULDER_ROLL_LATERAL_RAD)
    assert lateral[7] == pytest.approx(SHOULDER_PITCH_BACKWARD_RAD)
    assert lateral[8] == pytest.approx(-SHOULDER_ROLL_LATERAL_RAD)
    assert lateral[3] == pytest.approx(ELBOW_OUTWARD_CLEARANCE_RAD)
    assert lateral[10] == pytest.approx(ELBOW_OUTWARD_CLEARANCE_RAD)
    np.testing.assert_allclose(
        lateral[[2, 4, 5, 6, 9, 11, 12, 13]],
        initial[[2, 4, 5, 6, 9, 11, 12, 13]],
    )
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[0] == pytest.approx(-0.55)
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[1] == pytest.approx(1.60)
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[3] == pytest.approx(
        ELBOW_OUTWARD_CLEARANCE_RAD
    )
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[7] == pytest.approx(-0.55)
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[8] == pytest.approx(-1.60)
    assert FORWARD_OUTWARD_CLEARANCE_ARM_POSE_RAD[10] == pytest.approx(
        ELBOW_OUTWARD_CLEARANCE_RAD
    )
    assert FORWARD_HIGH_ARM_POSE_RAD[0] == pytest.approx(-0.55)
    assert FORWARD_HIGH_ARM_POSE_RAD[3] == pytest.approx(-0.2)
    assert FORWARD_HIGH_ARM_POSE_RAD[7] == pytest.approx(-0.55)
    assert FORWARD_HIGH_ARM_POSE_RAD[10] == pytest.approx(-0.2)

    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    backend = _FollowingPreMotionBackend()
    log_path = tmp_path / "pre_motion.jsonl"
    command_sequence = CommandSequence(9)
    latest = run_arm_pre_motion(
        backend,
        config=config,
        log_path=log_path,
        command_sequence=command_sequence,
        gravity_compensator=_FakeGravityCompensator(),
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        waypoint_tolerance_rad=0.05,
        stage_timeout_s=2.0,
        stable_samples_required=2,
    )
    assert command_sequence.value == 9 + len(backend.commands)
    assert np.asarray(latest.arm_joint_position_rad) == pytest.approx(
        FORWARD_HIGH_ARM_POSE_RAD, abs=0.05
    )
    assert np.asarray(backend.commands[-1].arm_position_rad) == pytest.approx(
        FORWARD_HIGH_ARM_POSE_RAD, abs=0.05
    )
    for command in backend.commands:
        assert len(command.arm_position_rad) == 14
        assert command.dex1_opening_fraction == pytest.approx((0.31, 0.72))
        assert command.arm_feedforward_torque_nm == pytest.approx(
            np.linspace(-1.3, 1.3, 14)
        )
        assert not hasattr(command, "waist_position_rad")
        assert not hasattr(command, "leg_position_rad")

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [
        record["stage"]
        for record in records
        if record["event"] == "pre_motion_stage_started"
    ] == [waypoint.name for waypoint in ARM_PRE_MOTION_WAYPOINTS]
    assert all(
        record["lower_body_command_dimensions"] == 0
        for record in records
        if record["event"] in {"pre_motion_stage_started", "pre_motion_command"}
    )


def test_model_specific_frame0_pose_and_reverse_path_are_deterministic() -> None:
    pose = subtask_start_pose_for_model(
        "Team-RAMEN/groot-n1.7-pick-legs-ver1"
    )
    assert pose is PICK_LEG_FRAME0
    assert len(pose.sha256) == 64
    initial = np.linspace(-0.25, 0.25, 14)
    startup = build_arm_pre_motion_waypoints(pose.arm_position_rad)
    assert startup[-1].name == "dataset_frame0_pose"
    np.testing.assert_allclose(startup[-1].as_array(), pose.arm_position_rad)

    reverse = build_arm_return_waypoints(initial, pose.arm_position_rad)
    assert [waypoint.name for waypoint in reverse] == [
        "return_dataset_frame0_pose",
        "return_forward_high_ready",
        "return_forward_outward_clearance",
        "return_lateral_high_clearance",
        "return_shoulder_pitch_backward_clearance",
        "return_measured_initial_pose",
    ]
    np.testing.assert_allclose(reverse[0].as_array(), pose.arm_position_rad)
    np.testing.assert_allclose(reverse[-1].as_array(), initial)


def test_all_registered_dataset_frame0_poses_fit_hardware_margins() -> None:
    config = load_teleop_config()
    for pose in (
        PICK_LEG_FRAME0,
        PICK_LEG_ACT_EP2101_FRAME0,
        COARSE_INSERT_FRAME0,
        FLIP_TABLE_V1_FRAME0,
        FLIP_TABLE_V2_FRAME0,
    ):
        validate_arm_pre_motion_waypoints(
            config.safety.arm_position_lower_rad,
            config.safety.arm_position_upper_rad,
            build_arm_pre_motion_waypoints(pose.arm_position_rad),
        )
        if pose.dex1_opening_fraction is not None:
            hand = np.asarray(pose.dex1_opening_fraction)
            assert hand.shape == (2,)
            assert np.all(hand >= 0.0)
            assert np.all(hand <= 1.0)
    with pytest.raises(ValueError, match="no verified dataset frame-zero"):
        subtask_start_pose_for_model("Team-RAMEN/unregistered-model")


def test_flip_table_v2_start_pose_pins_dataset_frame0_hands() -> None:
    pose = subtask_start_pose_for_model(
        "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_2"
    )
    assert pose is FLIP_TABLE_V2_FRAME0
    assert pose.dex1_opening_fraction == pytest.approx(
        (4.167656421661377 / MODEL_DEX1_OPEN_VALUE, 1.0)
    )


def test_reverse_return_sends_arm_only_and_restores_initial_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    backend = _FollowingPreMotionBackend()
    initial = np.linspace(-0.2, 0.2, 14)
    backend.arm = np.asarray(PICK_LEG_FRAME0.arm_position_rad, dtype=np.float64)
    sequence = CommandSequence(100)
    ok = return_arms_before_release(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "return.jsonl",
        command_sequence=sequence,
        gravity_compensator=_FakeGravityCompensator(),
        initial_arm_position_rad=initial,
        dataset_frame0_arm_rad=PICK_LEG_FRAME0.arm_position_rad,
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        waypoint_tolerance_rad=0.05,
        stage_timeout_s=2.0,
    )
    assert ok
    np.testing.assert_allclose(backend.arm, initial, atol=0.05)
    assert all(len(command.arm_position_rad) == 14 for command in backend.commands)
    assert all(not hasattr(command, "waist_position_rad") for command in backend.commands)


def test_act_pre_motion_opens_hands_then_sets_dataset_start_and_returns_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    backend = _FollowingPreMotionBackend()
    initial_arm = backend.arm.copy()
    waypoints = (
        ArmPreMotionWaypoint(
            "hands_full_open_before_clearance",
            (0.0,) * 14,
            tuple(range(14)),
        ),
    ) + build_arm_pre_motion_waypoints(
        PICK_LEG_ACT_EP2101_FRAME0.arm_position_rad
    )
    targets = {waypoint.name: (1.0, 1.0) for waypoint in waypoints}
    targets["dataset_frame0_pose"] = tuple(
        DATASET_INITIAL_DEX1_OPENING_FRACTION.tolist()
    )
    sequence = CommandSequence()
    run_arm_pre_motion(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "act_start.jsonl",
        command_sequence=sequence,
        gravity_compensator=_FakeGravityCompensator(),
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        waypoint_tolerance_rad=0.10,
        stage_timeout_s=2.0,
        waypoints=waypoints,
        hand_targets_by_waypoint=targets,
    )
    np.testing.assert_allclose(
        backend.hand, DATASET_INITIAL_DEX1_OPENING_FRACTION, atol=0.05
    )
    assert any(
        np.min(command.dex1_opening_fraction) >= 0.95
        for command in backend.commands
    )

    assert return_arms_before_release(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "act_return.jsonl",
        command_sequence=sequence,
        gravity_compensator=_FakeGravityCompensator(),
        initial_arm_position_rad=initial_arm,
        dataset_frame0_arm_rad=PICK_LEG_ACT_EP2101_FRAME0.arm_position_rad,
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        waypoint_tolerance_rad=0.10,
        stage_timeout_s=2.0,
        dex1_return_opening_fraction=(1.0, 1.0),
    )
    np.testing.assert_allclose(backend.hand, (1.0, 1.0), atol=0.05)


def test_policy_start_gate_refreshes_hold_until_empty_enter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    responses = iter((None, "not-enter", None, ""))
    backend = _FollowingPreMotionBackend()
    latest = backend.observe(0.1)
    command_sequence = CommandSequence(20)

    returned = wait_for_policy_start_with_hold(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "gate.jsonl",
        command_sequence=command_sequence,
        gravity_compensator=_FakeGravityCompensator(),
        latest=latest,
        enter_poll=lambda: next(responses),
    )

    assert returned is not None
    assert len(backend.commands) == 3
    assert command_sequence.value == 23
    assert all(command.mode.value == "track" for command in backend.commands)
    records = [
        json.loads(line)
        for line in (tmp_path / "gate.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "policy_start_gate_waiting",
        "policy_start_gate_confirmed",
    ]


def test_custom_grasp_gate_holds_pose_and_uses_distinct_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    responses = iter((None, ""))
    backend = _FollowingPreMotionBackend()
    initial_arm = backend.arm.copy()
    initial_hand = backend.hand.copy()
    command_sequence = CommandSequence(40)

    returned = wait_for_policy_start_with_hold(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "grasp_gate.jsonl",
        command_sequence=command_sequence,
        gravity_compensator=_FakeGravityCompensator(),
        latest=backend.observe(0.1),
        enter_poll=lambda: next(responses),
        prompt="Press Enter to grasp: ",
        invalid_prompt="Press Enter to grasp: ",
        waiting_event="dataset_grasp_gate_waiting",
        confirmed_event="dataset_grasp_gate_confirmed",
    )

    np.testing.assert_allclose(returned.arm_joint_position_rad, initial_arm)
    np.testing.assert_allclose(returned.dex1_opening_fraction, initial_hand)
    assert len(backend.commands) == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "grasp_gate.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "dataset_grasp_gate_waiting",
        "dataset_grasp_gate_confirmed",
    ]


def test_coarse_insert_hand_only_grasp_preserves_dataset_arm_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.desktop.upper_policy import run_flip_table_diffusion as runner

    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    backend = _FollowingPreMotionBackend()
    dataset_arm = np.asarray(COARSE_INSERT_FRAME0.arm_position_rad)
    backend.arm = dataset_arm.copy()
    backend.hand = np.ones(2, dtype=np.float64)
    target_hand = COARSE_INSERT_FRAME0.dex1_opening_fraction
    assert target_hand is not None
    waypoint = ArmPreMotionWaypoint(
        "dataset_frame0_hand_grasp",
        (0.0,) * 14,
        tuple(range(14)),
    )

    latest = run_arm_pre_motion(
        backend,
        config=load_teleop_config(),
        log_path=tmp_path / "coarse_grasp.jsonl",
        command_sequence=CommandSequence(),
        gravity_compensator=_FakeGravityCompensator(),
        arm_velocity_rad_s=1.0,
        arm_acceleration_rad_s2=4.0,
        waypoint_tolerance_rad=0.10,
        stage_timeout_s=2.0,
        stable_samples_required=2,
        waypoints=(waypoint,),
        hand_targets_by_waypoint={waypoint.name: target_hand},
        hand_velocity_fraction_s=1.0,
        hand_acceleration_fraction_s2=4.0,
    )

    np.testing.assert_allclose(latest.arm_joint_position_rad, dataset_arm)
    np.testing.assert_allclose(
        latest.dex1_opening_fraction, target_hand, atol=0.05
    )
    assert all(
        np.asarray(command.arm_position_rad) == pytest.approx(dataset_arm)
        for command in backend.commands
    )
    assert all(not hasattr(command, "waist_position_rad") for command in backend.commands)
    assert all(not hasattr(command, "leg_position_rad") for command in backend.commands)


def test_blocking_policy_start_check_keeps_watchdog_fresh() -> None:
    backend = _FollowingPreMotionBackend()
    config = load_teleop_config()
    latest = backend.observe(0.1)
    command_sequence = CommandSequence(30)
    pose_hold = PolicyStartPoseHold(
        backend,
        command_sequence=command_sequence,
        gravity_compensator=_FakeGravityCompensator(),
        latest=latest,
    )

    def slow_check() -> str:
        time.sleep(0.09)
        return "checked"

    result, returned = run_blocking_check_with_pose_hold(
        slow_check,
        backend=backend,
        config=config,
        pose_hold=pose_hold,
        latest=latest,
    )

    assert result == "checked"
    assert returned is not None
    assert len(backend.commands) >= 2
    assert command_sequence.value == 30 + len(backend.commands)
    assert all(command.mode.value == "track" for command in backend.commands)


def test_pre_motion_fails_closed_if_measured_arm_does_not_follow(
    tmp_path: Path,
) -> None:
    backend = _FollowingPreMotionBackend(follow=False)
    command_sequence = CommandSequence()
    with pytest.raises(TimeoutError, match="did not converge"):
        run_arm_pre_motion(
            backend,
            config=load_teleop_config(),
            log_path=tmp_path / "timeout.jsonl",
            command_sequence=command_sequence,
            gravity_compensator=_FakeGravityCompensator(),
            arm_velocity_rad_s=1.0,
            arm_acceleration_rad_s2=4.0,
            waypoint_tolerance_rad=0.05,
            stage_timeout_s=0.04,
            stable_samples_required=2,
        )
    assert backend.commands
    assert command_sequence.value == backend.commands[-1].sequence
    assert command_sequence.next() > backend.commands[-1].sequence
