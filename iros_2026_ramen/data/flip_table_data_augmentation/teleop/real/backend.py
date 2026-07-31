"""Non-actuating-until-armed G1 + Dex1 DDS backend.

This intentionally does not instantiate the upstream controller classes: the
v1.5 constructors start publisher threads immediately. The message types,
topics, gains, joint ordering, and Dex1 mapping below are the same official
contract, with an explicit publish gate controlled by `r`.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Mapping

import numpy as np

from ..backend import TeleopBackend, TransientObservationError
from ..config import TeleopConfig
from ..contracts import ArmHandTarget, ControlMode, TeleopObservation
from .safety import OfficialG1CommandFilter
from .network import counter_delta, network_snapshot, route_interface
from .teleimager import (
    LatestCameraSample,
    LatestCameraTracker,
    create_image_client,
)
from ..shared.watchdog import WatchdogState
from ..upstream_compat import install_logging_mp_compat


ARM_INDICES = tuple(range(15, 29))
LOWER_BODY_INDICES = tuple(range(15))
G1_29_LOWCMD_MOTOR_COUNT = 35
# Unitree's official G1_29_ArmController publishes rt/arm_sdk at 250 Hz.
# Keep this independent from the simulator's 50 Hz physics-side servo clock:
# the two transports have different contracts even though they share the same
# 30 Hz operator command stream.
REAL_ARM_SDK_HZ = 250.0
REAL_DEX1_HZ = 200.0
OFFICIAL_ARM_VELOCITY_INITIAL_RAD_S = 20.0
OFFICIAL_ARM_VELOCITY_FINAL_RAD_S = 30.0
OFFICIAL_ARM_VELOCITY_RAMP_S = 5.0
DEX1_MOTOR_OPEN_RAD = 5.4
# The official Dex1_1_Gripper_Controller never asks either motor to move more
# than 0.18 rad away from its latest measured position in one control update.
# This bound is feedback-relative, not merely a velocity limit: when a gripper
# is obstructed or its motor does not respond, the target must not continue to
# integrate toward an endpoint and build a multi-radian tracking error.
DEX1_MAX_TARGET_OFFSET_RAD = 0.18
DEX1_MAX_TARGET_OFFSET_FRACTION = DEX1_MAX_TARGET_OFFSET_RAD / DEX1_MOTOR_OPEN_RAD
WRIST_INDICES = frozenset((19, 20, 21, 26, 27, 28))
WEAK_INDICES = frozenset((4, 10, 15, 16, 17, 18, 22, 23, 24, 25))
BODY_KP = 300.0
BODY_KD = 3.0
WEAK_KP = 80.0
WEAK_KD = 3.0
WRIST_KP = 40.0
WRIST_KD = 1.5

# ``rt/arm_sdk`` blends the operator arm target with the regular-mode motion
# controller through motor_cmd[29].q.  Stepping this from 0 to 1 is unsafe:
# the SDK's own motion-mode guidance requires a gradual transition.  These are
# deliberately conservative for a standing robot; they affect arm ownership,
# not locomotion.
ARM_SDK_BLEND_IN_S = 1.0
# Pinned upstream xr_teleoperate releases through 101 values at 20 ms each.
# Match that roughly two-second transition instead of dropping ownership in
# 0.5 s, which is perceptually abrupt when every arm joint transfers at once.
ARM_SDK_BLEND_OUT_S = 2.0
# Unitree's motion-mode examples use full arm_sdk authority after the
# transition.  Keep the gradual blend but preserve normal arm tracking range.
ARM_SDK_MAX_WEIGHT = 1.0

# The official G1 arm_sdk packet initializes all 35 LowCmd motor slots from the
# same LowState sample (29 real joints plus six protocol/reserved slots), then
# overwrites joints 15..28 with arm targets and slot 29 with the blend weight.
# The operator target is still strictly 14-D: legs and waist remain owned by
# the G1 Regular-mode controller, exactly as official ``--motion``.
WAIST_GUARD_INDICES = (12, 13, 14)
WAIST_GUARD_MAX_DEVIATION_RAD = 0.12


def camera_bundle_status(
    samples: Mapping[str, LatestCameraSample],
    last_synchronized_generation: Mapping[str, int],
    *,
    camera_hz: float,
) -> tuple[bool, float, dict[str, int]]:
    """Evaluate one host-observed physical bundle without rewriting timestamps."""

    if camera_hz <= 0.0 or not samples:
        raise ValueError("camera_hz and samples must be non-empty and positive")
    if set(samples) != set(last_synchronized_generation):
        raise ValueError("camera samples and generation baseline roles differ")
    source_times = {
        role: sample.first_observed_monotonic_ns
        for role, sample in samples.items()
    }
    generations = {
        role: sample.jpeg_generation for role, sample in samples.items()
    }
    skew_ms = (max(source_times.values()) - min(source_times.values())) / 1.0e6
    all_new = all(
        generations[role] > last_synchronized_generation[role]
        for role in generations
    )
    return all_new and skew_ms <= 1000.0 / camera_hz, skew_ms, generations


def camera_bundle_status_from_histories(
    histories: Mapping[str, tuple[LatestCameraSample, ...]],
    last_synchronized_generation: Mapping[str, int],
    *,
    camera_hz: float,
) -> tuple[bool, float, dict[str, int]]:
    """Consume one nearest, unique host-observed sample from each role.

    This is preview diagnostics only. The authoritative dataset is the Orin
    MCAP, but retaining and consuming a short ring prevents the Desktop's
    120-Hz poller from reducing three independent latest-value streams back to
    one latest sample before skew/transition diagnostics are calculated.
    """

    if camera_hz <= 0.0 or set(histories) != set(
        last_synchronized_generation
    ):
        raise ValueError("camera histories and generation baseline differ")
    if not histories or any(not values for values in histories.values()):
        return False, float("inf"), dict(last_synchronized_generation)
    anchor_role = "head" if "head" in histories else next(iter(histories))
    maximum_skew_ns = int(round(1.0e9 / camera_hz))
    for anchor in histories[anchor_role]:
        if (
            anchor.jpeg_generation
            <= last_synchronized_generation[anchor_role]
        ):
            continue
        selected = {anchor_role: anchor}
        for role, candidates in histories.items():
            if role == anchor_role:
                continue
            eligible = [
                sample
                for sample in candidates
                if sample.jpeg_generation
                > last_synchronized_generation[role]
            ]
            if not eligible:
                break
            nearest = min(
                eligible,
                key=lambda sample: (
                    abs(
                        sample.first_observed_monotonic_ns
                        - anchor.first_observed_monotonic_ns
                    ),
                    sample.jpeg_generation,
                ),
            )
            if (
                abs(
                    nearest.first_observed_monotonic_ns
                    - anchor.first_observed_monotonic_ns
                )
                > maximum_skew_ns
            ):
                break
            selected[role] = nearest
        if set(selected) != set(histories):
            continue
        times = [
            sample.first_observed_monotonic_ns
            for sample in selected.values()
        ]
        return (
            True,
            (max(times) - min(times)) / 1.0e6,
            {
                role: sample.jpeg_generation
                for role, sample in selected.items()
            },
        )
    newest_times = [
        values[-1].first_observed_monotonic_ns
        for values in histories.values()
    ]
    return (
        False,
        (max(newest_times) - min(newest_times)) / 1.0e6,
        dict(last_synchronized_generation),
    )


class RealDdsBackend(TeleopBackend):
    def __init__(self, interface: str, image_server_ip: str, config: TeleopConfig) -> None:
        from inference.desktop.lower_policy.actuators.g1_control_lock import (
            acquire_g1_control_lock,
        )

        acquire_g1_control_lock()
        install_logging_mp_compat()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, networkInterface=interface)
        self.config = config
        # The sealed model-evaluation launcher enables this optional recorder
        # through an environment variable. Normal teleoperation and all other
        # backend users retain exactly the same control path.
        from inference.desktop.model_evaluation.recording import (
            RealEvaluationRecorder,
        )

        self._evaluation_recorder = RealEvaluationRecorder.from_environment()
        self._types = {
            "MotorCmds": MotorCmds_,
            "MotorCmd": unitree_go_msg_dds__MotorCmd_,
            "LowCmd": unitree_hg_msg_dds__LowCmd_,
        }
        self._crc = CRC()
        self._low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._left_state_sub = ChannelSubscriber("rt/dex1/left/state", MotorStates_)
        self._right_state_sub = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
        for subscriber in (self._low_state_sub, self._left_state_sub, self._right_state_sub):
            subscriber.Init()
        self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._left_pub = ChannelPublisher("rt/dex1/left/cmd", MotorCmds_)
        self._right_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        for publisher in (self._arm_pub, self._left_pub, self._right_pub):
            publisher.Init()
        self._image_client = create_image_client(image_server_ip)
        self._image_network_interface = route_interface(image_server_ip)
        self._image_network_baseline = network_snapshot(
            self._image_network_interface
        )
        self._image_network_current = dict(self._image_network_baseline)
        self._image_network_last_sample_ns = time.monotonic_ns()
        self._camera_trackers = {
            "head": LatestCameraTracker(
                "head", self._image_client.get_head_frame
            ),
            "left_wrist": LatestCameraTracker(
                "left_wrist", self._image_client.get_left_wrist_frame
            ),
            "right_wrist": LatestCameraTracker(
                "right_wrist", self._image_client.get_right_wrist_frame
            ),
        }
        self._camera_ready = False
        self._head_payload_generation = 0
        self._head_eye_jpeg: dict[str, bytes] = {}
        self._last_synchronized_camera_generation = {
            role: 0 for role in self._camera_trackers
        }
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._body_q: np.ndarray | None = None
        self._body_dq: np.ndarray | None = None
        self._all_motor_q: np.ndarray | None = None
        self._body_state_ns: int | None = None
        self._mode_machine = 0
        self._mode_machine_anchor: int | None = None
        self._dex1 = np.zeros(2, dtype=np.float64)
        self._dex1_state_ns: list[int | None] = [None, None]
        self._latest_command: ArmHandTarget | None = None
        self._last_command_ns: int | None = None
        self._tracking = False
        self._published = False
        self._arm_sdk_weight = 0.0
        self._arm_speed_ramp_started_ns: int | None = None
        self._release_complete = threading.Event()
        self._release_complete.set()
        self._release_log_bucket: int | None = None
        self._whole_body_hold_q: np.ndarray | None = None
        self._waist_anchor: np.ndarray | None = None
        self._waist_guard_tripped = False
        self._lower_body_peak_speed_rad_s = 0.0
        self._arm_interlock_reason: str | None = None
        self._last_applied_command_sequence: int | None = None
        self._last_applied_dex1_command_sequence: list[int | None] = [None, None]
        self._dex1_tracking_error = np.zeros(2, dtype=np.float64)
        self._dex1_feedback_limit_active = np.zeros(2, dtype=bool)
        self._dex1_state_stale = np.zeros(2, dtype=bool)
        self._dex1_thread_error: str | None = None
        self._dds_write_count = np.zeros(3, dtype=np.int64)
        self._dds_write_failure_count = np.zeros(3, dtype=np.int64)
        self._dds_write_error_reported: set[str] = set()
        self._last_valid_state: (
            tuple[np.ndarray, np.ndarray, int, np.ndarray] | None
        ) = None
        self._stop = threading.Event()
        self._sequence = 0
        self._applied_arm = np.zeros(14, dtype=np.float64)
        self._applied_arm_torque = np.zeros(14, dtype=np.float64)
        self._applied_hand = np.zeros(2, dtype=np.float64)
        self._requested_arm = np.zeros(14, dtype=np.float64)
        self._requested_arm_torque = np.zeros(14, dtype=np.float64)
        self._requested_hand = np.zeros(2, dtype=np.float64)
        self._dex1_filter_history: list[np.ndarray] = []
        self._safety = OfficialG1CommandFilter(
            config.safety, servo_hz=REAL_ARM_SDK_HZ
        )
        self._state_thread = threading.Thread(target=self._read_state, name="g1-dds-state", daemon=True)
        self._state_thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._state_lock:
                state_ready = (
                    self._body_q is not None
                    and self._body_dq is not None
                    and self._all_motor_q is not None
                    and self._body_state_ns is not None
                    and all(value is not None for value in self._dex1_state_ns)
                )
            if state_ready:
                break
            time.sleep(0.02)
        else:
            self.close()
            raise TimeoutError("G1 lowstate or one of the two Dex1 states was not received")
        self._servo_thread = threading.Thread(target=self._servo, name="g1-dds-servo", daemon=True)
        self._servo_thread.start()
        self._dex1_thread = threading.Thread(
            target=self._dex1_servo, name="g1-dex1-servo", daemon=True
        )
        self._dex1_thread.start()

    def _read_state(self) -> None:
        while not self._stop.is_set():
            low = self._low_state_sub.Read()
            left = self._left_state_sub.Read()
            right = self._right_state_sub.Read()
            with self._state_lock:
                if low is not None and len(low.motor_state) >= G1_29_LOWCMD_MOTOR_COUNT:
                    self._all_motor_q = np.asarray(
                        [
                            low.motor_state[i].q
                            for i in range(G1_29_LOWCMD_MOTOR_COUNT)
                        ],
                        dtype=np.float64,
                    )
                    self._body_q = np.asarray([low.motor_state[i].q for i in range(29)], dtype=np.float64)
                    self._body_dq = np.asarray([low.motor_state[i].dq for i in range(29)], dtype=np.float64)
                    self._mode_machine = int(low.mode_machine)
                    self._body_state_ns = time.monotonic_ns()
                if left is not None and getattr(left, "states", None):
                    self._dex1[0] = float(left.states[0].q)
                    self._dex1_state_ns[0] = time.monotonic_ns()
                if right is not None and getattr(right, "states", None):
                    self._dex1[1] = float(right.states[0].q)
                    self._dex1_state_ns[1] = time.monotonic_ns()
            time.sleep(0.002)

    def _state(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        tuple[int, int, int],
    ]:
        with self._state_lock:
            timestamps = (self._body_state_ns, *self._dex1_state_ns)
            if (
                self._body_q is None
                or self._body_dq is None
                or self._all_motor_q is None
                or any(value is None for value in timestamps)
            ):
                raise RuntimeError("G1 or Dex1 state is unavailable")
            return (
                self._body_q.copy(),
                self._body_dq.copy(),
                self._all_motor_q.copy(),
                self._dex1.copy(),
                self._mode_machine,
                tuple(int(value) for value in timestamps),
            )

    def _command_snapshot(self) -> tuple[ArmHandTarget | None, int | None, bool]:
        with self._command_lock:
            return self._latest_command, self._last_command_ns, self._tracking

    @staticmethod
    def _gains_for_joint(index: int) -> tuple[float, float]:
        if index in WRIST_INDICES:
            return WRIST_KP, WRIST_KD
        if index in WEAK_INDICES:
            return WEAK_KP, WEAK_KD
        return BODY_KP, BODY_KD

    def _arm_message(
        self,
        whole_body_hold_q: np.ndarray,
        mode_machine: int,
        target: np.ndarray,
        weight: float,
        feedforward_torque: np.ndarray | None = None,
    ):
        msg = self._types["LowCmd"]()
        msg.mode_pr = 0
        msg.mode_machine = mode_machine
        hold = np.asarray(whole_body_hold_q, dtype=np.float64)
        requested = np.asarray(target, dtype=np.float64)
        torque = np.zeros(len(ARM_INDICES), dtype=np.float64)
        if feedforward_torque is not None:
            torque = np.asarray(feedforward_torque, dtype=np.float64)
        if hold.shape != (G1_29_LOWCMD_MOTOR_COUNT,) or not np.isfinite(hold).all():
            raise ValueError("G1_29 LowCmd hold position must be finite 35-D")
        if requested.shape != (len(ARM_INDICES),) or not np.isfinite(requested).all():
            raise ValueError("arm target must be finite 14-D")
        if torque.shape != (len(ARM_INDICES),) or not np.isfinite(torque).all():
            raise ValueError("arm feedforward torque must be finite 14-D")
        if not np.isfinite(weight) or not 0.0 <= weight <= ARM_SDK_MAX_WEIGHT:
            raise ValueError("arm_sdk blend weight must be finite in [0,1]")
        # Match Unitree's G1_29 arm_sdk contract exactly: capture a stationary
        # 29-D hold pose once, then overwrite only the arm targets below.  In
        # particular, a mode=1 command with zero waist gains is not a valid
        # substitute for this protocol and was observed to destabilize the
        # regular-mode torso controller.
        for index in range(G1_29_LOWCMD_MOTOR_COUNT):
            command = msg.motor_cmd[index]
            command.mode = 1
            command.q = float(hold[index])
            command.dq = 0.0
            command.tau = 0.0
            command.kp, command.kd = self._gains_for_joint(index)
        for offset, index in enumerate(ARM_INDICES):
            command = msg.motor_cmd[index]
            command.mode = 1
            command.q = float(requested[offset])
            command.dq = 0.0
            command.tau = float(torque[offset])
            command.kp, command.kd = self._gains_for_joint(index)
        msg.motor_cmd[29].q = float(weight)
        msg.crc = self._crc.Crc(msg)
        return msg

    def _gripper_message(self, target_rad: float):
        msg = self._types["MotorCmds"]()
        command = self._types["MotorCmd"]()
        command.q = float(target_rad)
        command.dq = 0.0
        command.tau = 0.0
        command.kp = 5.0
        command.kd = 0.05
        msg.cmds = [command]
        return msg

    @staticmethod
    def _limit_dex1_target_around_measured(
        desired_fraction: np.ndarray,
        measured_fraction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired = np.asarray(desired_fraction, dtype=np.float64)
        measured = np.asarray(measured_fraction, dtype=np.float64)
        if desired.shape != (2,) or measured.shape != (2,):
            raise ValueError("Dex1 desired and measured targets must be 2-D")
        if not np.isfinite(desired).all() or not np.isfinite(measured).all():
            raise ValueError("Dex1 desired and measured targets must be finite")
        lower = measured - DEX1_MAX_TARGET_OFFSET_FRACTION
        upper = measured + DEX1_MAX_TARGET_OFFSET_FRACTION
        limited = np.clip(np.clip(desired, 0.0, 1.0), lower, upper)
        limited = np.clip(limited, 0.0, 1.0)
        active = np.abs(limited - desired) > 1.0e-9
        return limited, active

    def _filter_dex1_official(self, target: np.ndarray) -> np.ndarray:
        """Apply the official Dex1 [0.5, 0.3, 0.2] moving filter once."""

        value = np.asarray(target, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("Dex1 filter target must be finite 2-D")
        if self._dex1_filter_history and np.array_equal(
            value, self._dex1_filter_history[-1]
        ):
            return self._applied_hand.copy()
        self._dex1_filter_history.append(value.copy())
        if len(self._dex1_filter_history) > 3:
            self._dex1_filter_history.pop(0)
        if len(self._dex1_filter_history) < 3:
            return value.copy()
        # np.convolve(..., [0.5, 0.3, 0.2], valid) weights newest,
        # middle, oldest as 0.5, 0.3, 0.2 in the official implementation.
        return (
            0.2 * self._dex1_filter_history[0]
            + 0.3 * self._dex1_filter_history[1]
            + 0.5 * self._dex1_filter_history[2]
        )

    def _official_arm_velocity_limit(self, now_ns: int) -> float:
        started_ns = self._arm_speed_ramp_started_ns
        if started_ns is None:
            return OFFICIAL_ARM_VELOCITY_INITIAL_RAD_S
        elapsed_s = max(0.0, (now_ns - started_ns) / 1.0e9)
        fraction = min(1.0, elapsed_s / OFFICIAL_ARM_VELOCITY_RAMP_S)
        return OFFICIAL_ARM_VELOCITY_INITIAL_RAD_S + fraction * (
            OFFICIAL_ARM_VELOCITY_FINAL_RAD_S
            - OFFICIAL_ARM_VELOCITY_INITIAL_RAD_S
        )

    def _write(
        self,
        whole_body_hold_q: np.ndarray,
        mode_machine: int,
        arm: np.ndarray,
        hand: np.ndarray,
        weight: float = 1.0,
        arm_feedforward_torque: np.ndarray | None = None,
        publish_dex1: bool = True,
    ) -> tuple[bool, bool | None, bool | None]:
        writes = (
            (
                "arm_sdk",
                self._arm_pub,
                lambda: self._arm_message(
                    whole_body_hold_q,
                    mode_machine,
                    arm,
                    weight,
                    arm_feedforward_torque,
                ),
            ),
            (
                "dex1_left",
                self._left_pub,
                lambda: self._gripper_message(
                    float(np.clip(hand[0], 0.0, 1.0) * DEX1_MOTOR_OPEN_RAD)
                ),
            ),
            (
                "dex1_right",
                self._right_pub,
                lambda: self._gripper_message(
                    float(np.clip(hand[1], 0.0, 1.0) * DEX1_MOTOR_OPEN_RAD)
                ),
            ),
        )
        status_list: list[bool | None] = []
        for index, (label, publisher, make_message) in enumerate(writes):
            if index > 0 and not publish_dex1:
                status_list.append(None)
                continue
            try:
                ok = publisher.Write(make_message()) is not False
                status_list.append(ok)
                if ok:
                    self._dds_write_error_reported.discard(label)
            except Exception as exc:
                # Pinned unitree_sdk2_python has an exception-handler defect
                # (it calls ``e.args`` as a function), so a writer failure can
                # otherwise escape and silently kill this daemon servo thread.
                if label not in self._dds_write_error_reported:
                    print(f"[safety] {label} DDS write raised: {exc}", flush=True)
                    self._dds_write_error_reported.add(label)
                status_list.append(False)
        status = (status_list[0], status_list[1], status_list[2])
        with self._command_lock:
            attempted = np.asarray(
                [value is not None for value in status], dtype=np.int64
            )
            failed = np.asarray(
                [value is False for value in status], dtype=np.int64
            )
            self._dds_write_count += attempted
            self._dds_write_failure_count += failed
            # Only a successful arm write proves that this process has ever
            # acquired arm_sdk authority. Dex1 publishers are independent.
            self._published = self._published or status[0]
        return status

    def _write_dex1(self, hand: np.ndarray) -> tuple[bool, bool]:
        statuses: list[bool] = []
        for offset, (label, publisher) in enumerate(
            (("dex1_left", self._left_pub), ("dex1_right", self._right_pub))
        ):
            try:
                message = self._gripper_message(
                    float(np.clip(hand[offset], 0.0, 1.0) * DEX1_MOTOR_OPEN_RAD)
                )
                ok = publisher.Write(message) is not False
                if ok:
                    self._dds_write_error_reported.discard(label)
            except Exception as exc:
                ok = False
                if label not in self._dds_write_error_reported:
                    print(f"[safety] {label} DDS write raised: {exc}", flush=True)
                    self._dds_write_error_reported.add(label)
            statuses.append(ok)
        with self._command_lock:
            self._dds_write_count[1:3] += 1
            self._dds_write_failure_count[1:3] += np.asarray(
                [not status for status in statuses], dtype=np.int64
            )
        return statuses[0], statuses[1]

    def _confirm_successful_targets(
        self,
        command: ArmHandTarget | None,
        arm: np.ndarray,
        arm_torque: np.ndarray,
        hand: np.ndarray,
        measured_hand: np.ndarray,
        write_status: tuple[bool, bool | None, bool | None],
    ) -> None:
        """Advance applied diagnostics only for channels actually written."""

        with self._command_lock:
            if write_status[0]:
                self._applied_arm = arm.copy()
                self._applied_arm_torque = arm_torque.copy()
                if command is not None:
                    self._last_applied_command_sequence = command.sequence
            for side in range(2):
                if write_status[side + 1]:
                    self._applied_hand[side] = hand[side]
                    if command is not None:
                        self._last_applied_dex1_command_sequence[side] = (
                            command.sequence
                        )
            self._dex1_tracking_error = np.abs(
                self._applied_hand - measured_hand
            )

    @staticmethod
    def _waist_deviation_exceeded(
        body_q: np.ndarray, waist_anchor: np.ndarray | None
    ) -> bool:
        if waist_anchor is None:
            return False
        current = np.asarray(body_q, dtype=np.float64)
        if current.shape[0] < max(WAIST_GUARD_INDICES) + 1:
            raise ValueError("G1 body position lacks waist joints")
        anchor = np.asarray(waist_anchor, dtype=np.float64)
        if anchor.shape != (len(WAIST_GUARD_INDICES),):
            raise ValueError("waist anchor must contain yaw, roll, and pitch")
        return bool(
            np.max(np.abs(current[list(WAIST_GUARD_INDICES)] - anchor))
            > WAIST_GUARD_MAX_DEVIATION_RAD
        )

    @staticmethod
    def _state_is_stale(
        timestamps_ns: tuple[int, ...], *, now_ns: int, maximum_age_s: float
    ) -> bool:
        if maximum_age_s <= 0.0:
            raise ValueError("maximum state age must be positive")
        maximum_age_ns = int(maximum_age_s * 1.0e9)
        return any(now_ns - timestamp > maximum_age_ns for timestamp in timestamps_ns)

    def _trip_arm_interlock(self, reason: str) -> None:
        """Latch an arm release until the session has sent an explicit IDLE."""

        with self._command_lock:
            if self._arm_interlock_reason is None:
                self._arm_interlock_reason = reason
                print(f"[safety] {reason}; releasing arm control.", flush=True)
            self._tracking = False

    def _next_arm_sdk_weight(self, *, active: bool, releasing: bool) -> float:
        if active:
            self._arm_sdk_weight = min(
                ARM_SDK_MAX_WEIGHT,
                self._arm_sdk_weight
                + 1.0 / (REAL_ARM_SDK_HZ * ARM_SDK_BLEND_IN_S),
            )
        elif releasing:
            self._arm_sdk_weight = max(
                0.0,
                self._arm_sdk_weight
                - 1.0 / (REAL_ARM_SDK_HZ * ARM_SDK_BLEND_OUT_S),
            )
        return self._arm_sdk_weight

    def _release_arm_sdk(
        self,
        all_motor_q: np.ndarray,
        body_q: np.ndarray,
        mode_machine: int,
        measured_hand: np.ndarray,
    ) -> None:
        """Blend arm ownership back to regular mode using a measured hold."""

        previous_weight = self._arm_sdk_weight
        release_weight = self._next_arm_sdk_weight(active=False, releasing=True)
        if self._release_log_bucket is None:
            print(
                "[shutdown] arm_sdk controlled release started "
                f"(weight={previous_weight:.2f}, duration≈{ARM_SDK_BLEND_OUT_S:.1f}s).",
                flush=True,
            )
        bucket = int(np.ceil(release_weight * 4.0 - 1.0e-12))
        if bucket != self._release_log_bucket and bucket < 4:
            print(
                f"[shutdown] arm_sdk release weight={release_weight:.2f}",
                flush=True,
            )
        self._release_log_bucket = bucket
        write_status = self._write(
            all_motor_q,
            mode_machine,
            body_q[list(ARM_INDICES)],
            measured_hand,
            weight=release_weight,
            publish_dex1=False,
        )
        if not write_status[0]:
            # Local state represents the last weight confirmed written, never
            # merely the next requested blend value.
            self._arm_sdk_weight = previous_weight
            # Do not report a successful transfer to the regular controller
            # unless the final arm_sdk blend packet actually left this process.
            # The servo keeps retrying until close()'s bounded timeout.
            if release_weight <= 0.0:
                print(
                    "[shutdown] WARNING: final arm_sdk release packet failed; "
                    "retrying until the shutdown timeout.",
                    flush=True,
                )
            return
        if release_weight <= 0.0:
            with self._command_lock:
                self._published = False
                self._tracking = False
                self._waist_anchor = None
                self._whole_body_hold_q = None
                self._mode_machine_anchor = None
            self._release_complete.set()
            print("[shutdown] arm_sdk release complete (weight=0.00).", flush=True)

    def _dex1_servo(self) -> None:
        """Run the official Dex1 cadence independently from 250 Hz arm_sdk."""

        period = self.dex1_publish_period_s()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                _body, _dq, _all, dex1_rad, _mode, timestamps = self._state()
                command, _last_ns, tracking = self._command_snapshot()
                measured = np.clip(dex1_rad / DEX1_MOTOR_OPEN_RAD, 0.0, 1.0)
                now_ns = time.monotonic_ns()
                stale = np.asarray(
                    [
                        self._state_is_stale(
                            (timestamp,),
                            now_ns=now_ns,
                            maximum_age_s=self.config.safety.command_hold_timeout_s,
                        )
                        for timestamp in timestamps[1:]
                    ],
                    dtype=bool,
                )
                with self._command_lock:
                    requested = self._requested_hand.copy()
                limited, feedback_limited = self._limit_dex1_target_around_measured(
                    requested, measured
                )
                limited = np.where(stale, measured, limited)
                enabled = (
                    tracking
                    and command is not None
                    and command.mode in {ControlMode.TRACK, ControlMode.HOLD}
                )
                if enabled:
                    filtered = self._filter_dex1_official(limited)
                    left_ok, right_ok = self._write_dex1(filtered)
                    with self._command_lock:
                        for side, ok in enumerate((left_ok, right_ok)):
                            if ok:
                                self._applied_hand[side] = filtered[side]
                                self._last_applied_dex1_command_sequence[side] = (
                                    command.sequence
                                )
                with self._command_lock:
                    self._dex1_state_stale = stale
                    self._dex1_feedback_limit_active = np.logical_or(
                        feedback_limited, stale
                    )
                    self._dex1_tracking_error = np.abs(
                        self._applied_hand - measured
                    )
            except Exception as exc:
                # Dex1 is an independent official 200 Hz channel.  A hand-only
                # failure must not release arm_sdk and make both arms fall.
                # Latch it for diagnostics/exit status, stop issuing new hand
                # targets, and keep the arm servo alive so shutdown can blend
                # ownership back to Regular mode in a controlled way.
                message = f"Dex1 publisher thread failed ({exc})"
                with self._command_lock:
                    if self._dex1_thread_error is None:
                        self._dex1_thread_error = message
                        print(f"[safety] {message}; holding arm control.", flush=True)
                break
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    @staticmethod
    def dex1_publish_period_s() -> float:
        """Return the official Dex1 period used by its independent thread."""

        return 1.0 / REAL_DEX1_HZ

    def _servo(self) -> None:
        period = 1.0 / REAL_ARM_SDK_HZ
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                (
                    body_q,
                    body_dq,
                    all_motor_q,
                    dex1_rad,
                    mode_machine,
                    state_timestamps,
                ) = self._state()
            except RuntimeError as exc:
                # Never leave the last nonzero arm_sdk blend latched merely
                # because DDS state access failed.  The last validated state
                # is used only to send the documented gradual release packet.
                self._trip_arm_interlock(f"G1 or Dex1 state unavailable ({exc})")
                if self._published and self._last_valid_state is not None:
                    last_all_motor, last_body, last_mode, last_hand = self._last_valid_state
                    self._release_arm_sdk(
                        last_all_motor, last_body, last_mode, last_hand
                    )
                time.sleep(max(0.0, period - (time.monotonic() - started)))
                continue
            measured_hand = np.clip(dex1_rad / DEX1_MOTOR_OPEN_RAD, 0.0, 1.0)
            self._last_valid_state = (
                all_motor_q.copy(),
                body_q.copy(),
                mode_machine,
                measured_hand.copy(),
            )
            now_ns = time.monotonic_ns()
            command, last_command_ns, tracking = self._command_snapshot()
            if self._state_is_stale(
                (state_timestamps[0],),
                now_ns=now_ns,
                maximum_age_s=self.config.safety.command_hold_timeout_s,
            ):
                self._trip_arm_interlock("G1 lowstate became stale")
                tracking = False
            dex1_state_stale = np.asarray(
                [
                    self._state_is_stale(
                        (timestamp,),
                        now_ns=now_ns,
                        maximum_age_s=self.config.safety.command_hold_timeout_s,
                    )
                    for timestamp in state_timestamps[1:]
                ],
                dtype=bool,
            )
            if tracking and self._waist_anchor is None:
                self._waist_anchor = body_q[list(WAIST_GUARD_INDICES)].copy()
                self._waist_guard_tripped = False
            if tracking and self._whole_body_hold_q is None:
                self._whole_body_hold_q = all_motor_q.copy()
            if tracking and self._mode_machine_anchor is None:
                self._mode_machine_anchor = mode_machine
            if (
                tracking
                and self._mode_machine_anchor is not None
                and mode_machine != self._mode_machine_anchor
            ):
                self._trip_arm_interlock(
                    "G1 mode_machine changed while arm_sdk was active "
                    f"({self._mode_machine_anchor} -> {mode_machine})"
                )
                tracking = False
            lower_peak = float(
                np.max(np.abs(body_dq[list(LOWER_BODY_INDICES)]))
            )
            self._lower_body_peak_speed_rad_s = max(
                self._lower_body_peak_speed_rad_s, lower_peak
            )
            if tracking and self._waist_deviation_exceeded(body_q, self._waist_anchor):
                # Regular mode legitimately moves the lower body for balance
                # and walking. Record unexpected torso motion for acceptance
                # diagnostics, but do not fight it by dropping arm ownership.
                self._waist_guard_tripped = True
            safe = self._safety.apply(
                command,
                measured_arm_position_rad=body_q[list(ARM_INDICES)],
                measured_dex1_opening_fraction=measured_hand,
                now_ns=now_ns,
                last_command_ns=last_command_ns,
                tracking=tracking,
                official_arm_velocity_limit_rad_s=(
                    self._official_arm_velocity_limit(now_ns)
                ),
            )
            applied_arm = np.asarray(safe.arm_position_rad)
            applied_arm_torque = np.asarray(safe.arm_feedforward_torque_nm)
            requested_hand = np.asarray(safe.dex1_opening_fraction)
            with self._command_lock:
                self._requested_arm = applied_arm.copy()
                self._requested_arm_torque = applied_arm_torque.copy()
                self._requested_hand = requested_hand.copy()
                applied_hand = self._applied_hand.copy()
            if safe.watchdog is WatchdogState.ACTIVE or (
                safe.watchdog is WatchdogState.HOLD and self._published
            ):
                whole_body_hold_q = self._whole_body_hold_q
                if whole_body_hold_q is None:
                    # The first active tick captures a verified lowstate; this
                    # fallback is defensive and never fabricates a pose.
                    whole_body_hold_q = all_motor_q
                previous_weight = self._arm_sdk_weight
                requested_weight = self._next_arm_sdk_weight(
                    active=safe.watchdog is WatchdogState.ACTIVE,
                    releasing=False,
                )
                write_status = self._write(
                    whole_body_hold_q,
                    self._mode_machine_anchor
                    if self._mode_machine_anchor is not None
                    else mode_machine,
                    applied_arm,
                    applied_hand,
                    weight=requested_weight,
                    arm_feedforward_torque=applied_arm_torque,
                    publish_dex1=False,
                )
                # Losing arm_sdk publication can leave stale upper-body
                # ownership and therefore trips the arm interlock. A Dex1-only
                # write failure remains visible in diagnostics and is retried,
                # but must not make both arms fall by releasing arm_sdk.
                if not write_status[0]:
                    self._arm_sdk_weight = previous_weight
                    self._trip_arm_interlock("arm_sdk DDS publisher write failed")
                    if not self._published and previous_weight == 0.0:
                        self._release_complete.set()
                self._confirm_successful_targets(
                    command,
                    applied_arm,
                    applied_arm_torque,
                    applied_hand,
                    measured_hand,
                    write_status,
                )
            elif safe.watchdog is WatchdogState.STOP and self._published:
                # Release arm_sdk ownership smoothly while holding its last
                # measured arm/hand posture.  This is the motion-mode blend
                # specified by Unitree, rather than an abrupt ownership step.
                self._release_arm_sdk(
                    all_motor_q, body_q, mode_machine, measured_hand
                )
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    @staticmethod
    def _jpeg(frame: np.ndarray) -> bytes:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("failed to encode camera frame")
        return encoded.tobytes()

    def _head_eye_payloads(self, sample: LatestCameraSample) -> dict[str, bytes]:
        if sample.jpeg_generation == self._head_payload_generation:
            return dict(self._head_eye_jpeg)
        import cv2

        head = cv2.imdecode(
            np.frombuffer(sample.jpg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if head is None or head.shape != (480, 1280, 3):
            shape = None if head is None else tuple(head.shape)
            raise TransientObservationError(
                "head JPEG did not decode to 1280x480 BGR "
                f"(shape={shape}, generation={sample.jpeg_generation})"
            )
        self._head_eye_jpeg = {
            "head_left": self._jpeg(head[:, :640]),
            "head_right": self._jpeg(head[:, 640:]),
        }
        self._head_payload_generation = sample.jpeg_generation
        return dict(self._head_eye_jpeg)

    @staticmethod
    def _camera_role_metadata(
        sample: LatestCameraSample,
        *,
        now_ns: int,
        fresh: bool,
    ) -> dict[str, object]:
        return {
            "role": sample.role,
            "jpeg_generation": sample.jpeg_generation,
            "first_observed_ns": sample.first_observed_monotonic_ns,
            "age_ms": round(sample.age_ms(now_ns), 3),
            "fresh": fresh,
            "source_fps": round(sample.source_fps, 3),
            "transition_hz": round(sample.transition_hz, 3),
        }

    def observe(self, timeout_s: float) -> TeleopObservation:
        """Return the latest complete camera snapshot without bundle blocking.

        Only cold start waits for every physical stream to advance once. Once
        initialized, a stopped wrist camera cannot freeze head preview, robot
        state observation, or safety evaluation: per-role age is carried in
        the observation and the session enters HOLD from that evidence.
        """

        deadline = time.monotonic() + timeout_s
        poll_fresh = {role: False for role in self._camera_trackers}
        poll_errors: dict[str, str] = {}
        while True:
            for role, tracker in self._camera_trackers.items():
                try:
                    _sample, changed = tracker.poll()
                    poll_fresh[role] = poll_fresh[role] or changed
                    poll_errors.pop(role, None)
                except Exception as exc:  # noqa: BLE001
                    poll_errors[role] = f"{type(exc).__name__}: {exc}"
            samples = {
                role: tracker.sample
                for role, tracker in self._camera_trackers.items()
            }
            all_present = all(sample is not None for sample in samples.values())
            # The getter exposes a latest-value slot with no source timestamp.
            # Requiring one post-connection transition per role prevents a
            # pre-existing cached JPEG from being presented as a fresh bundle.
            all_advanced = all(
                sample is not None and sample.jpeg_generation >= 2
                for sample in samples.values()
            )
            if self._camera_ready or (all_present and all_advanced):
                self._camera_ready = True
                break
            if time.monotonic() >= deadline:
                states = {
                    role: (
                        None
                        if sample is None
                        else sample.jpeg_generation
                    )
                    for role, sample in samples.items()
                }
                raise TransientObservationError(
                    "real camera cold start did not observe one new JPEG from "
                    f"every role (generation={states}, errors={poll_errors})"
                )
            time.sleep(0.005)

        typed_samples = {
            role: sample
            for role, sample in samples.items()
            if sample is not None
        }
        if set(typed_samples) != set(self._camera_trackers):
            raise TransientObservationError(
                f"real camera sample unavailable (errors={poll_errors})"
            )
        head_sample = typed_samples["head"]
        left_sample = typed_samples["left_wrist"]
        right_sample = typed_samples["right_wrist"]
        payload = {
            **self._head_eye_payloads(head_sample),
            "left_wrist": left_sample.jpg,
            "right_wrist": right_sample.jpg,
        }
        body_q, body_dq, _all_motor_q, dex1_rad, _mode_machine, state_times = self._state()
        now = time.monotonic_ns()
        maximum_state_age_ns = int(self.config.safety.command_hold_timeout_s * 1.0e9)
        if now - state_times[0] > maximum_state_age_ns:
            raise RuntimeError("real G1 lowstate is stale")
        self._sequence += 1
        camera_bundle_valid, camera_skew_ms, generations = (
            camera_bundle_status_from_histories(
                {
                    role: tracker.history
                    for role, tracker in self._camera_trackers.items()
                },
                self._last_synchronized_camera_generation,
                camera_hz=float(self.config.rates.camera_hz),
            )
        )
        if camera_bundle_valid:
            self._last_synchronized_camera_generation = dict(generations)
        stream_metadata = {
            "head_left": self._camera_role_metadata(
                head_sample, now_ns=now, fresh=poll_fresh["head"]
            ),
            "head_right": self._camera_role_metadata(
                head_sample, now_ns=now, fresh=poll_fresh["head"]
            ),
            "left_wrist": self._camera_role_metadata(
                left_sample, now_ns=now, fresh=poll_fresh["left_wrist"]
            ),
            "right_wrist": self._camera_role_metadata(
                right_sample, now_ns=now, fresh=poll_fresh["right_wrist"]
            ),
        }
        camera_age_ms = {
            role: float(metadata["age_ms"])
            for role, metadata in stream_metadata.items()
        }
        hold_threshold_ms = self.config.safety.command_hold_timeout_s * 1000.0
        stale_roles = tuple(
            role
            for role, age_ms in camera_age_ms.items()
            if age_ms > hold_threshold_ms
        )
        camera_receive_ns = {
            "head_left": head_sample.first_observed_monotonic_ns,
            "head_right": head_sample.first_observed_monotonic_ns,
            "left_wrist": left_sample.first_observed_monotonic_ns,
            "right_wrist": right_sample.first_observed_monotonic_ns,
        }
        hand = np.clip(dex1_rad / DEX1_MOTOR_OPEN_RAD, 0.0, 1.0)
        with self._command_lock:
            applied_arm = self._applied_arm.copy()
            applied_arm_torque = self._applied_arm_torque.copy()
            applied_hand = self._applied_hand.copy()
            requested_arm = self._requested_arm.copy()
            requested_arm_torque = self._requested_arm_torque.copy()
            requested_hand = self._requested_hand.copy()
            arm_sdk_weight = float(self._arm_sdk_weight)
            waist_guard_tripped = bool(self._waist_guard_tripped)
            lower_body_peak_speed_rad_s = float(
                self._lower_body_peak_speed_rad_s
            )
            arm_interlock_reason = self._arm_interlock_reason
            last_applied_command_sequence = self._last_applied_command_sequence
            last_applied_dex1_command_sequence = tuple(
                self._last_applied_dex1_command_sequence
            )
            dex1_tracking_error = self._dex1_tracking_error.copy()
            dex1_feedback_limit_active = self._dex1_feedback_limit_active.copy()
            dex1_state_stale = self._dex1_state_stale.copy()
            dex1_thread_error = self._dex1_thread_error
            dds_write_count = self._dds_write_count.copy()
            dds_write_failure_count = self._dds_write_failure_count.copy()
        # Network counters are diagnostics, not control input. Reading every
        # sysfs counter and /proc/net/snmp at the 60 Hz operator rate adds
        # needless work to the latency-sensitive process, so sample at 1 Hz.
        if now - self._image_network_last_sample_ns >= 1_000_000_000:
            self._image_network_current = network_snapshot(
                self._image_network_interface
            )
            self._image_network_last_sample_ns = now
        image_network_current = dict(self._image_network_current)
        image_network_delta = counter_delta(
            self._image_network_baseline, image_network_current
        )
        observation = TeleopObservation(
            sequence=self._sequence,
            capture_monotonic_ns=now,
            backend="real",
            body_joint_position_rad=tuple(body_q),
            body_joint_velocity_rad_s=tuple(body_dq),
            dex1_opening_fraction=tuple(hand),
            applied_arm_target_rad=tuple(applied_arm),
            applied_dex1_opening_target=tuple(applied_hand),
            root_pose_xyzw=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            camera_capture_monotonic_ns={
                **camera_receive_ns,
            },
            camera_jpeg=payload,
            camera_bundle_valid=camera_bundle_valid,
            camera_skew_ms=camera_skew_ms,
            stale_roles=stale_roles,
            camera_stream_metadata=stream_metadata,
            diagnostics={
                "root_pose_source": "unavailable_stationary_identity",
                "privileged_policy_features": [],
                "arm_sdk_weight": arm_sdk_weight,
                "waist_guard_tripped": waist_guard_tripped,
                "waist_motion_is_diagnostic_only": True,
                "lower_body_peak_speed_rad_s": lower_body_peak_speed_rad_s,
                "lower_body_policy_command_dimensions": 0,
                "regular_mode_owns_lower_body": True,
                "arm_sdk_publish_hz_target": REAL_ARM_SDK_HZ,
                "dex1_publish_hz_target": REAL_DEX1_HZ,
                "camera_timestamp_source": "host_first_observed_approximate",
                "camera_bundle_valid": camera_bundle_valid,
                "camera_skew_ms": camera_skew_ms,
                "camera_stale_roles": list(stale_roles),
                "camera_stream_age_ms": camera_age_ms,
                "camera_stream_poll_errors": poll_errors,
                "image_network_interface": self._image_network_interface,
                "image_network_counters_start": dict(
                    self._image_network_baseline
                ),
                "image_network_counters_current": image_network_current,
                "image_network_counter_delta": image_network_delta,
                "arm_feedforward_torque_nm": applied_arm_torque.tolist(),
                "arm_motion_filter": "official_global_measured_relative_velocity_scale",
                "dex1_motion_filter": "official_feedback_relative_plus_weighted_0_5_0_3_0_2",
                "arm_interlock_reason": arm_interlock_reason,
                "last_applied_command_sequence": last_applied_command_sequence,
                "last_applied_dex1_command_sequence_left_right": list(
                    last_applied_dex1_command_sequence
                ),
                "requested_arm_target_rad": requested_arm.tolist(),
                "requested_arm_feedforward_torque_nm": (
                    requested_arm_torque.tolist()
                ),
                "requested_dex1_opening_left_right": requested_hand.tolist(),
                "dex1_tracking_error_left_right": dex1_tracking_error.tolist(),
                "dex1_feedback_limit_active_left_right": (
                    dex1_feedback_limit_active.tolist()
                ),
                "dex1_state_stale_left_right": dex1_state_stale.tolist(),
                "dex1_thread_error": dex1_thread_error,
                "dds_write_count_arm_left_right": dds_write_count.tolist(),
                "dds_write_failure_count_arm_left_right": (
                    dds_write_failure_count.tolist()
                ),
            },
        )
        evaluation_recorder = getattr(self, "_evaluation_recorder", None)
        if evaluation_recorder is not None:
            evaluation_recorder.record_observation(observation)
        return observation

    def apply(self, target: ArmHandTarget) -> None:
        with self._command_lock:
            if (
                self._latest_command is not None
                and target.sequence < self._latest_command.sequence
            ):
                raise ValueError("real command sequence moved backwards")
            self._latest_command = target
            self._last_command_ns = time.monotonic_ns()
            evaluation_recorder = getattr(self, "_evaluation_recorder", None)
            if evaluation_recorder is not None:
                evaluation_recorder.record_action(target)
            arm_authority_active = target.mode in {
                ControlMode.TRACK,
                ControlMode.HOLD,
            }
            if target.mode is ControlMode.IDLE:
                # IDLE starts/continues the blend-out.  An interlock remains
                # latched until a weight=0 packet has actually been confirmed.
                self._tracking = False
                return
            if not self._tracking and not self._release_complete.is_set():
                self._tracking = False
                return
            if self._arm_interlock_reason is not None:
                # Reaching this branch proves an explicit TRACK/HOLD arrived
                # after the confirmed blend-out, i.e. a deliberate re-arm.
                self._arm_interlock_reason = None
            if arm_authority_active and not self._tracking:
                # A new r/re-anchor starts a fresh, gradual blend and captures
                # a new waist baseline in the servo thread.
                if self._arm_sdk_weight != 0.0:
                    raise RuntimeError(
                        "cannot acquire arm_sdk before confirmed weight=0"
                    )
                self._arm_speed_ramp_started_ns = time.monotonic_ns()
                self._waist_anchor = None
                self._waist_guard_tripped = False
                self._whole_body_hold_q = None
                self._mode_machine_anchor = None
                self._lower_body_peak_speed_rad_s = 0.0
                self._dex1_filter_history = []
                self._release_complete.clear()
                self._release_log_bucket = None
            # HOLD deliberately retains arm_sdk authority at the captured
            # target.  IDLE alone releases it back to the regular controller.
            self._tracking = arm_authority_active

    def close(self) -> None:
        release_required = bool(self._published)
        with self._command_lock:
            self._tracking = False
            self._latest_command = None
        # Never stop the publisher on a wall-clock guess. Wait until the servo
        # has actually published weight=0 through the official-style ramp.
        if hasattr(self, "_servo_thread") and release_required:
            timeout_s = ARM_SDK_BLEND_OUT_S + self.config.safety.command_stop_timeout_s + 0.5
            if not self._release_complete.wait(timeout=timeout_s):
                print(
                    "[shutdown] WARNING: arm_sdk release did not report weight=0 "
                    f"within {timeout_s:.2f}s.",
                    flush=True,
                )
        self._stop.set()
        failed_threads = []
        for name in ("_state_thread", "_servo_thread", "_dex1_thread"):
            thread = getattr(self, name, None)
            if thread is not None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    failed_threads.append(name)
        image_client = getattr(self, "_image_client", None)
        if image_client is not None:
            image_client.close()
        evaluation_recorder = getattr(self, "_evaluation_recorder", None)
        if evaluation_recorder is not None:
            report = evaluation_recorder.close()
            if not report["complete"]:
                failures = [
                    "evaluation recording incomplete: "
                    f"drops={report['queue_drop_count']} errors={report['errors']}"
                ]
            else:
                failures = []
        else:
            failures = []
        if release_required and not self._release_complete.is_set():
            failures.append("arm_sdk weight=0 was not confirmed")
        if failed_threads:
            failures.append("publisher threads did not stop: " + ", ".join(failed_threads))
        if self._dex1_thread_error is not None:
            failures.append(self._dex1_thread_error)
        if failures:
            raise RuntimeError("; ".join(failures))
