"""G1ArmActuator unit test (Issue #75)。

SDK / cyclonedds install 不要な形で書く: publisher / crc_fn / publish_thread を
inject 経路で mock 化 → 実 SDK に触らずに build_lowcmd / send_action / lifecycle
の behavior を検証。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_JOINT_INDICES,
    G1_NUM_ARM_JOINTS,
    ArmControlGains,
    ArmSafetyLimits,
    G1ArmActuator,
)


# =========================================================================
# helpers: fake SDK layer (publisher / CRC / RecurrentThread)
# =========================================================================
class _FakePublisher:
    def __init__(self) -> None:
        self.written: list[Any] = []

    def Write(self, msg: Any) -> None:
        self.written.append(msg)


class _FakeCRC:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def Crc(self, msg: Any) -> int:
        self.calls.append(msg)
        return 0xDEADBEEF


class _FakeRecurrentThread:
    """`Start()` / `Wait()` だけ satisfy する fake、実 thread は起動しない。"""

    def __init__(self) -> None:
        self.started = False
        self.waited = False

    def Start(self) -> None:
        self.started = True

    def Wait(self, timeout: float = 1.0) -> None:
        self.waited = True


def _make_actuator(**overrides: Any) -> tuple[G1ArmActuator, _FakePublisher, _FakeCRC, _FakeRecurrentThread]:
    pub = _FakePublisher()
    crc = _FakeCRC()
    thread = _FakeRecurrentThread()
    actuator = G1ArmActuator(
        publisher=pub,
        crc_fn=crc,
        publish_thread=thread,
        initial_positions=np.linspace(-0.28, 0.28, 29),
        **overrides,
    )
    return actuator, pub, crc, thread


# =========================================================================
# G1_ARM_JOINT_INDICES 定数
# =========================================================================
class TestJointIndexConstants:
    def test_arm_joint_indices_len(self):
        assert G1_NUM_ARM_JOINTS == 14
        assert len(G1_ARM_JOINT_INDICES) == 14

    def test_arm_joint_indices_are_15_to_28(self):
        assert G1_ARM_JOINT_INDICES == tuple(range(15, 29))


# =========================================================================
# send_action shape + clamp
# =========================================================================
class TestSendAction:
    def test_wrong_shape_raises(self):
        actuator, _, _, _ = _make_actuator()
        with pytest.raises(ValueError, match="shape"):
            actuator.send_action(np.zeros(13))
        with pytest.raises(ValueError, match="shape"):
            actuator.send_action(np.zeros(29))

    def test_target_stored_after_send_action(self):
        actuator, _, _, _ = _make_actuator()
        positions = np.linspace(-0.5, 0.5, G1_NUM_ARM_JOINTS)
        actuator.send_action(positions)
        assert actuator._target.received
        assert np.allclose(actuator._target.positions, positions)

    def test_clamp_applied(self):
        """|target| が limits.joint_max_abs を超えたら clamp される。"""
        actuator, _, _, _ = _make_actuator(
            limits=ArmSafetyLimits(joint_max_abs=1.0)
        )
        positions = np.array([2.0] * G1_NUM_ARM_JOINTS)  # 全部 clamp 対象
        actuator.send_action(positions)
        assert actuator._target.positions is not None
        assert np.all(np.abs(actuator._target.positions) <= 1.0)


# =========================================================================
# publish loop (_publish_once) の LowCmd_ 組み立て
# =========================================================================
class TestPublishOnce:
    def test_idle_before_first_send_action(self):
        """未受信なら _publish_once は publisher に何も送らない。"""
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        actuator._publish_once()
        assert pub.written == []

    def test_publish_only_after_start_and_send_action(self):
        actuator, pub, crc, _ = _make_actuator()
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS) * 0.3)
        actuator._publish_once()
        assert len(pub.written) == 1
        # CRC が計算されて lowcmd.crc に注入されてる
        assert crc.calls == pub.written  # same instance passed to CRC then Write

    def test_publish_no_op_if_stopped(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        # start 呼ばず _publish_once するのは、running=False なので何もしない
        actuator._publish_once()
        assert pub.written == []

    def test_lowcmd_matches_official_motion_mode_contract(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        targets = np.linspace(-0.05, 0.05, G1_NUM_ARM_JOINTS)
        actuator.send_action(targets)
        actuator._publish_once()
        assert len(pub.written) == 1
        lowcmd = pub.written[0]
        # Official motion mode locks every physical G1 joint at startup pose.
        for i in range(29):
            assert lowcmd.motor_cmd[i].mode == 1
        # Arms overwrite only their own initial hold target.
        initial_arm = np.linspace(-0.28, 0.28, 29)[15:29]
        max_step = 20.0 / 250.0
        scale = max(float(np.max(np.abs(targets - initial_arm))) / max_step, 1.0)
        expected = initial_arm + (targets - initial_arm) / scale
        for slot, joint_idx in enumerate(G1_ARM_JOINT_INDICES):
            m = lowcmd.motor_cmd[joint_idx]
            assert m.q == pytest.approx(expected[slot])
        assert lowcmd.motor_cmd[0].q == pytest.approx(-0.28)
        assert lowcmd.motor_cmd[29].q == pytest.approx(1.0)
        assert lowcmd.motor_cmd[15].kp == pytest.approx(80.0)
        assert lowcmd.motor_cmd[15].kd == pytest.approx(3.0)
        assert lowcmd.motor_cmd[19].kp == pytest.approx(40.0)
        assert lowcmd.motor_cmd[19].kd == pytest.approx(1.5)

    def test_custom_gains_applied(self):
        actuator, pub, _, _ = _make_actuator(
            gains=ArmControlGains(
                body_kp=25.0, body_kd=0.5,
                weak_kp=26.0, weak_kd=0.6,
                wrist_kp=27.0, wrist_kd=0.7,
            )
        )
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        lowcmd = pub.written[0]
        assert lowcmd.motor_cmd[0].kp == 25.0
        assert lowcmd.motor_cmd[0].kd == 0.5
        assert lowcmd.motor_cmd[15].kp == 26.0
        assert lowcmd.motor_cmd[15].kd == 0.6
        assert lowcmd.motor_cmd[19].kp == 27.0
        assert lowcmd.motor_cmd[19].kd == 0.7

    def test_mode_machine_forwarded(self):
        actuator, pub, _, _ = _make_actuator(mode_machine=5)
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        assert pub.written[0].mode_machine == 5

    def test_large_target_is_velocity_limited(self):
        actuator, pub, _, _ = _make_actuator()
        actuator.start()
        actuator.send_action(np.ones(G1_NUM_ARM_JOINTS))
        actuator._publish_once()
        # The official global scale obeys 20 rad/s at 250 Hz (= 0.08 rad)
        # for the furthest arm joint in each message.
        initial_arm = np.linspace(-0.28, 0.28, 29)[15:29]
        target = np.ones(G1_NUM_ARM_JOINTS)
        scale = np.max(np.abs(target - initial_arm)) / (20.0 / 250.0)
        expected = initial_arm + (target - initial_arm) / scale
        assert pub.written[0].motor_cmd[15].q == pytest.approx(expected[0])

    def test_publish_exception_caught(self):
        """publisher.Write() が raise しても process 死しない (listener thread 保護と同 pattern)。"""

        class _RaisingPublisher:
            def Write(self, msg):
                raise RuntimeError("simulated DDS failure")

        actuator = G1ArmActuator(
            publisher=_RaisingPublisher(),
            crc_fn=_FakeCRC(),
            publish_thread=_FakeRecurrentThread(),
        )
        actuator.start()
        actuator.send_action(np.zeros(G1_NUM_ARM_JOINTS))
        actuator._publish_once()  # 例外が伝播しないことだけ確認 (test fail しなければ OK)


# =========================================================================
# lifecycle (start / stop / idempotent)
# =========================================================================
class TestLifecycle:
    def test_start_starts_publish_thread_once(self):
        actuator, _, _, thread = _make_actuator()
        actuator.start()
        assert thread.started is True
        assert actuator._running is True
        # 二重 start は fake_thread の start が呼ばれない (started は既に True)
        thread.started = False
        actuator.start()
        assert thread.started is False  # 呼ばれない = idempotent

    def test_stop_calls_wait_and_marks_stopped(self):
        actuator, _, _, thread = _make_actuator()
        actuator.start()
        actuator.stop()
        assert thread.waited is True
        assert actuator._running is False
        # 二重 stop は wait 呼ばない (idempotent)
        thread.waited = False
        actuator.stop()
        assert thread.waited is False


# =========================================================================
# thread safety の smoke (競合状態を粗く確認)
# =========================================================================
class TestThreadSafety:
    def test_concurrent_send_action_and_publish(self):
        """send_action と _publish_once が並列でも lock で protect されて壊れない。"""
        actuator, pub, _, _ = _make_actuator()
        actuator.start()

        # send_action を多数回投げる thread と、_publish_once を回す thread を並列
        stop = threading.Event()

        def sender():
            i = 0
            while not stop.is_set():
                actuator.send_action(np.ones(G1_NUM_ARM_JOINTS) * (i % 10) * 0.01)
                i += 1

        def publisher_loop():
            while not stop.is_set():
                actuator._publish_once()

        threads = [threading.Thread(target=sender), threading.Thread(target=publisher_loop)]
        for t in threads:
            t.start()
        threading.Event().wait(0.05)  # 50ms concurrent 実行
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
        # 生きて完走できれば OK (deadlock なし、例外 crash なし)
        assert not any(t.is_alive() for t in threads)
