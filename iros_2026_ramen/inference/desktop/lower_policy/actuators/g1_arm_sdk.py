"""Official-protocol G1 arm action adapter for Regular Mode.

This is the action sink for Type-B policies.  Its DDS contract deliberately
matches Unitree's ``xr_teleoperate`` G1_29 motion-mode controller:

* publish to ``rt/arm_sdk`` (not debug-only ``rt/lowcmd``);
* copy ``mode_machine`` and the initial whole-body pose from ``rt/lowstate``;
* lock the 29 body joints with the official gains while changing only arm
  joints 15..28; and
* set the official motion-mode enable word at motor slot 29.

It does not command walking.  The G1 motion controller retains responsibility
for the legs and balance while this adapter supplies arm targets.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


G1_ARM_SDK_TOPIC = "rt/arm_sdk"
G1_LOWCMD_MOTOR_ARRAY_LEN = 35
G1_NUM_BODY_JOINTS = 29
G1_NUM_ARM_JOINTS = 14
G1_ARM_JOINT_INDICES: tuple[int, ...] = tuple(range(15, 29))
G1_WRIST_JOINT_INDICES: frozenset[int] = frozenset((19, 20, 21, 26, 27, 28))
G1_WEAK_JOINT_INDICES: frozenset[int] = frozenset(
    (4, 10, 15, 16, 17, 18, 22, 23, 24, 25)
)
G1_MOTION_ENABLE_SLOT = 29


@dataclass(frozen=True)
class ArmControlGains:
    """Gains from Unitree's official G1_29_ArmController."""

    body_kp: float = 300.0
    body_kd: float = 3.0
    weak_kp: float = 80.0
    weak_kd: float = 3.0
    wrist_kp: float = 40.0
    wrist_kd: float = 1.5


@dataclass(frozen=True)
class ArmSafetyLimits:
    """Bound policy targets and their rate of change before DDS publication."""

    joint_max_abs: float = 1.5
    velocity_limit_rad_s: float = 20.0


@dataclass
class _LatestTarget:
    positions: Optional[np.ndarray] = None
    received: bool = False


class G1ArmActuator:
    """Publish a 14-D arm target using Unitree's official Regular-Mode path.

    The runtime constructor waits for a real ``rt/lowstate`` sample before it
    creates the initial command.  This prevents a zero ``mode_machine`` or a
    fabricated whole-body hold pose from being published to a physical robot.
    ``publisher``/``initial_positions`` are dependency-injection hooks used by
    the pure unit tests; they never touch DDS.
    """

    def __init__(
        self,
        *,
        control_freq_hz: float = 250.0,
        gains: Optional[ArmControlGains] = None,
        limits: Optional[ArmSafetyLimits] = None,
        mode_machine: Optional[int] = None,
        initial_positions: Optional[Sequence[float]] = None,
        publisher: Optional[object] = None,
        publish_thread: Optional[object] = None,
        crc_fn: Optional[object] = None,
    ) -> None:
        self._control_freq_hz = float(control_freq_hz)
        self._gains = gains if gains is not None else ArmControlGains()
        self._limits = limits if limits is not None else ArmSafetyLimits()
        self._lock = threading.Lock()
        self._target = _LatestTarget()
        self._running = False
        self._lowstate_subscriber: Optional[object] = None

        if publisher is None:
            from .g1_control_lock import acquire_g1_control_lock

            acquire_g1_control_lock()
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # type: ignore
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # type: ignore
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # type: ignore
            from unitree_sdk2py.utils.crc import CRC  # type: ignore

            # Pull 型 subscriber: handler なしで Init し、publish loop 内から
            # `.Read()` で最新 sample を polling する。rate limiter が毎 tick 現在の
            # arm pose を要求するので event-driven (SetReader(handler=...)) より
            # pull 型が合う (Ros2FrameSource の push 型 pattern とは意図的に非対称)。
            self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._lowstate_subscriber.Init()
            first_state = self._wait_for_lowstate()
            self._mode_machine = int(first_state.mode_machine)
            self._hold_positions = self._positions_from_lowstate(first_state)
            self._latest_arm_positions = self._hold_positions[
                list(G1_ARM_JOINT_INDICES)
            ].copy()

            self._publisher = ChannelPublisher(G1_ARM_SDK_TOPIC, LowCmd_)
            self._publisher.Init()
            self._crc_fn = CRC() if crc_fn is None else crc_fn
            self._lowcmd_factory = unitree_hg_msg_dds__LowCmd_
        else:
            # Dependency-injected unit-test path.  Production cannot use this
            # fallback because it must bootstrap from a real lowstate sample.
            supplied = (
                np.asarray(initial_positions, dtype=np.float64)
                if initial_positions is not None
                else np.zeros(G1_NUM_BODY_JOINTS, dtype=np.float64)
            )
            if supplied.shape != (G1_NUM_BODY_JOINTS,):
                raise ValueError(
                    "initial_positions must have shape "
                    f"({G1_NUM_BODY_JOINTS},), got {supplied.shape}"
                )
            self._mode_machine = int(mode_machine) if mode_machine is not None else 0
            self._hold_positions = supplied.copy()
            self._latest_arm_positions = self._hold_positions[
                list(G1_ARM_JOINT_INDICES)
            ].copy()
            self._publisher = publisher
            self._crc_fn = crc_fn
            self._lowcmd_factory = None

        self._publish_thread = publish_thread

    @staticmethod
    def _positions_from_lowstate(message: object) -> np.ndarray:
        return np.asarray(
            [float(message.motor_state[i].q) for i in range(G1_NUM_BODY_JOINTS)],  # type: ignore[attr-defined]
            dtype=np.float64,
        )

    def _wait_for_lowstate(self, timeout_s: float = 5.0) -> object:
        assert self._lowstate_subscriber is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._lowstate_subscriber.Read()  # type: ignore[attr-defined]
            if message is not None:
                return message
            time.sleep(0.002)
        raise RuntimeError(
            "timed out waiting for rt/lowstate; refusing to publish arm_sdk "
            "without a real mode_machine and whole-body hold pose"
        )

    def _current_arm_positions(self) -> np.ndarray:
        """Use the newest lowstate when available, otherwise retain last target.

        `ChannelSubscriber.Read()` は non-blocking で latest cached sample (or None)
        を返す想定 (SDK 実装の semantics)。250Hz publish loop の hot path で呼ぶが、
        DDS reader cache への O(1) access なので実測 overhead 無視できるレベル。
        SDK 側の Read() semantics が変わった (blocking / new-only 化) 場合は本
        method の再設計が必要 (rate-limit の毎 tick current 要求と衝突するため)。
        """
        if self._lowstate_subscriber is not None:
            message = self._lowstate_subscriber.Read()  # type: ignore[attr-defined]
            if message is not None:
                positions = self._positions_from_lowstate(message)
                self._latest_arm_positions = positions[list(G1_ARM_JOINT_INDICES)]
        return self._latest_arm_positions.copy()

    def read_arm_positions(self) -> np.ndarray:
        """Return the latest physical 14-D arm pose from ``rt/lowstate``.

        This is read-only and is used by the arm smoke tool to prove that an
        accepted command changed the robot, rather than relying on visual
        interpretation of an already-similar pose.
        """
        return self._current_arm_positions()

    def send_action(self, positions: Sequence[float]) -> None:
        arr = np.asarray(positions, dtype=np.float64)
        if arr.shape != (G1_NUM_ARM_JOINTS,):
            raise ValueError(
                f"positions must have shape ({G1_NUM_ARM_JOINTS},), got {arr.shape}"
            )
        clamped = np.clip(arr, -self._limits.joint_max_abs, self._limits.joint_max_abs)
        with self._lock:
            self._target.positions = clamped
            self._target.received = True

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        if self._publish_thread is None:
            from unitree_sdk2py.utils.thread import RecurrentThread  # type: ignore

            self._publish_thread = RecurrentThread(
                interval=1.0 / max(self._control_freq_hz, 1.0),
                target=self._publish_once,
                name="g1_arm_sdk_publish",
            )
        self._publish_thread.Start()  # type: ignore[attr-defined]

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._publish_thread is not None and hasattr(self._publish_thread, "Wait"):
            self._publish_thread.Wait(timeout=1.0)  # type: ignore[attr-defined]

    def _rate_limited_target(self, target: np.ndarray) -> np.ndarray:
        current = self._current_arm_positions()
        max_step = self._limits.velocity_limit_rad_s / max(self._control_freq_hz, 1.0)
        delta = target - current
        scale = max(float(np.max(np.abs(delta))) / max_step, 1.0)
        return current + delta / scale

    def _gains_for_joint(self, joint_index: int) -> tuple[float, float]:
        if joint_index in G1_WRIST_JOINT_INDICES:
            return self._gains.wrist_kp, self._gains.wrist_kd
        if joint_index in G1_WEAK_JOINT_INDICES:
            return self._gains.weak_kp, self._gains.weak_kd
        return self._gains.body_kp, self._gains.body_kd

    def _publish_once(self) -> None:
        with self._lock:
            if not self._running or not self._target.received or self._target.positions is None:
                return
            target = self._target.positions.copy()
        try:
            lowcmd = self._build_lowcmd(self._rate_limited_target(target))
            if self._crc_fn is not None and hasattr(self._crc_fn, "Crc"):
                lowcmd.crc = self._crc_fn.Crc(lowcmd)  # type: ignore[attr-defined]
            self._publisher.Write(lowcmd)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[G1ArmActuator] publish error: {exc!r}", file=sys.stderr)

    def _build_lowcmd(self, arm_positions: np.ndarray) -> object:
        if self._lowcmd_factory is None:
            import types

            motor_cmd = [
                types.SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
                for _ in range(G1_LOWCMD_MOTOR_ARRAY_LEN)
            ]
            lowcmd = types.SimpleNamespace(
                mode_pr=0, mode_machine=self._mode_machine,
                motor_cmd=motor_cmd, reserve=[0, 0, 0, 0], crc=0,
            )
        else:
            lowcmd = self._lowcmd_factory()
            lowcmd.mode_pr = 0  # type: ignore[attr-defined]
            lowcmd.mode_machine = self._mode_machine  # type: ignore[attr-defined]

        # Official G1_29 motion-mode setup: every body joint holds its
        # lowstate pose; then the 14 arm targets overwrite that hold position.
        for joint_index in range(G1_NUM_BODY_JOINTS):
            motor = lowcmd.motor_cmd[joint_index]
            motor.mode = 1
            motor.q = float(self._hold_positions[joint_index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp, motor.kd = self._gains_for_joint(joint_index)
        for slot, joint_index in enumerate(G1_ARM_JOINT_INDICES):
            lowcmd.motor_cmd[joint_index].q = float(arm_positions[slot])

        # Unitree motion-mode handshake used by xr_teleoperate.
        # source: third_party/unitree_sdk2_python/example/g1/high_level/
        #         g1_arm5_sdk_dds_example.py L64, L133
        #   class G1JointIndex: kNotUsedJoint = 29  # NOTE: Weight
        #   self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1  # 1:Enable arm_sdk
        # G1 は SetArmSdkStatus service を持たず (H2 系は `h2_loco_client.py` L106
        # にある)、この slot 29 magic write が公式の enable/disable 経路。firmware
        # 側 protocol 変わった時は上記 SDK example を再確認する。
        lowcmd.motor_cmd[G1_MOTION_ENABLE_SLOT].q = 1.0
        return lowcmd
