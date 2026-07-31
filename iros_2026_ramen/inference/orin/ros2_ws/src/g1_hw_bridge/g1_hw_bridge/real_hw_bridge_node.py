"""G1RealHardwareBridgeNode: Unitree SDK 経由で lowstate を subscribe → ROS2 に publish。

Damp state で joint 位置 / IMU / robot_state を Desktop 側の orchestrator に
渡すための I/O layer。**motor 命令は一切発行しない** (lowcmd publisher なし)
= Damp state の安全性を bridge レベルで保証する。

Hand (Dex1-1) は G1 SDK と別 SDK が担当、この bridge の scope 外。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from g1_hw_bridge.joint_mapping import G1_JOINT_NAMES, G1_NUM_MOTOR, extract_positions


def _sdk_lowstate_worker(
    domain_id: int,
    interface: str,
    state_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Receive Unitree DDS in a process that never creates an rclpy node.

    The Unitree SDK and rmw_cyclonedds_cpp cannot create DDS participants in
    the same Python process.  Keeping the SDK subscriber in a spawned child
    avoids that conflict; the parent ROS node receives only plain Python state
    through a bounded local queue.  This worker has no publisher and never
    emits a motor command.
    """
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    def on_lowstate(message: LowState_) -> None:
        state = {
            "positions": extract_positions(message.motor_state),
            "velocities": [float(message.motor_state[i].dq) for i in range(G1_NUM_MOTOR)],
            "efforts": [float(message.motor_state[i].tau_est) for i in range(G1_NUM_MOTOR)],
            "imu_rpy": [float(value) for value in message.imu_state.rpy],
            "mode_machine": int(message.mode_machine),
            "mode_pr": int(message.mode_pr),
            "tick": int(message.tick),
        }
        try:
            state_queue.put_nowait(state)
        except queue.Full:
            try:
                state_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                state_queue.put_nowait(state)
            except queue.Full:
                pass

    ChannelFactoryInitialize(domain_id, interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_lowstate, 10)
    while not stop_event.wait(0.1):
        pass


class G1RealHardwareBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_real_hw_bridge")

        self.declare_parameter("domain_id", 0)
        self.declare_parameter("interface", "eth0")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("joint_names", list(G1_JOINT_NAMES))
        self.declare_parameter("require_first_state_before_publish", True)

        self._domain_id = int(self.get_parameter("domain_id").value)
        self._interface = str(self.get_parameter("interface").value)
        self._publish_rate = float(self.get_parameter("publish_rate_hz").value)
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._require_first = bool(
            self.get_parameter("require_first_state_before_publish").value
        )

        if len(self._joint_names) != G1_NUM_MOTOR:
            raise ValueError(
                f"joint_names must have {G1_NUM_MOTOR} entries, got {len(self._joint_names)}"
            )

        # SDK worker と rclpy timer 側で共有する state
        self._lock = threading.Lock()
        self._positions: list[float] = [0.0] * G1_NUM_MOTOR
        self._velocities: list[float] = [0.0] * G1_NUM_MOTOR
        self._efforts: list[float] = [0.0] * G1_NUM_MOTOR
        self._imu_rpy: list[float] = [0.0, 0.0, 0.0]
        self._mode_machine: int = 0
        self._mode_pr: int = 0
        self._tick: int = 0
        self._received: bool = False

        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._robot_state_pub = self.create_publisher(String, "/g1/robot_state", 10)

        # Unitree SDKとrclpyは同一Python processでCycloneDDS participantを共存
        # できない。spawnしたSDK専用workerがrt/lowstateを購読し、このnodeには最新
        # stateだけをIPCで渡す。workerにはpublisherが無く、lowcmdは発行しない。
        context = mp.get_context("spawn")
        self._sdk_state_queue = context.Queue(maxsize=1)
        self._sdk_stop_event = context.Event()
        self._sdk_worker = context.Process(
            target=_sdk_lowstate_worker,
            args=(
                self._domain_id,
                self._interface,
                self._sdk_state_queue,
                self._sdk_stop_event,
            ),
            daemon=True,
        )
        self._sdk_worker.start()

        self.create_timer(1.0 / max(self._publish_rate, 0.1), self._publish_state)
        self.get_logger().info(
            f"g1_real_hw_bridge ready (domain_id={self._domain_id}, "
            f"interface={self._interface}, publish_rate_hz={self._publish_rate})"
        )

    def _drain_sdk_states(self) -> None:
        """Apply the newest IPC state, dropping stale samples by design."""
        newest = None
        while True:
            try:
                newest = self._sdk_state_queue.get_nowait()
            except queue.Empty:
                break
        if newest is None:
            return
        with self._lock:
            self._positions = newest["positions"]
            self._velocities = newest["velocities"]
            self._efforts = newest["efforts"]
            self._imu_rpy = newest["imu_rpy"]
            self._mode_machine = newest["mode_machine"]
            self._mode_pr = newest["mode_pr"]
            self._tick = newest["tick"]
            self._received = True

    def _publish_state(self) -> None:
        self._drain_sdk_states()
        with self._lock:
            if self._require_first and not self._received:
                return
            positions = list(self._positions)
            velocities = list(self._velocities)
            efforts = list(self._efforts)
            rpy = list(self._imu_rpy)
            mode_machine = self._mode_machine
            mode_pr = self._mode_pr
            tick = self._tick

        stamp = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = list(self._joint_names)
        js.position = positions
        js.velocity = velocities
        js.effort = efforts
        self._joint_state_pub.publish(js)

        rs = String()
        rs.data = json.dumps(
            {
                "mode_machine": mode_machine,
                "mode_pr": mode_pr,
                "tick": tick,
                "imu_rpy": [round(v, 4) for v in rpy],
            },
            sort_keys=True,
        )
        self._robot_state_pub.publish(rs)

    def destroy_node(self) -> bool:
        self._sdk_stop_event.set()
        self._sdk_worker.join(timeout=2.0)
        if self._sdk_worker.is_alive():
            self._sdk_worker.terminate()
            self._sdk_worker.join(timeout=1.0)
        self._sdk_state_queue.close()
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = G1RealHardwareBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
