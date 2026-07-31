"""Process-isolated Apple Vision Pro operator display and IK bridge.

TeleVuer v1.5 creates its own worker processes. The simulator socket must not
share a process tree or inherited descriptors with those workers, so this
module launches the operator with an exec boundary and uses a localhost-only
framed IPC channel for camera frames and G1/Dex1 targets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np

from .config import DEFAULT_TELEOP_CONFIG_PATH, TeleopConfig, load_teleop_config
from .operator_view import (
    compose_head_stereo_view,
    compose_real_desktop_view,
    compose_real_operator_stereo_view,
)
from .transport import FramedSocket
from .xr_runtime import IncoherentBilateralHandFrame, XrInput


@dataclass(frozen=True)
class OperatorTarget:
    """Latest real-compatible AVP target produced by the display process."""

    monotonic_ns: int
    avp_live: bool
    tracking_generation: int
    arm_position_rad: tuple[float, ...] | None
    arm_feedforward_torque_nm: tuple[float, ...] | None
    dex1_opening_fraction: tuple[float, float] | None
    session_age_s: float | None
    hand_age_s: float | None
    hand_tracking_hz: float
    hand_event_count: int
    hand_contiguous_event_count: int
    hand_invalid_event_count: int
    hand_missing_pose_count: int
    hand_invalid_wrist_count: int
    hand_invalid_pinch_count: int
    hand_invalid_unused_skeleton_count: int
    hand_invalid_details: dict[str, int]
    source_sequence: int
    image_processing_ms: float
    ik_processing_ms: float
    total_processing_ms: float


@dataclass
class _TrackingFaultLatch:
    """Keep a tracking fault visible across latest-only IPC coalescing."""

    generation: int = 0

    def trip(self, generation: int) -> None:
        if generation <= 0:
            raise ValueError("only an armed tracking generation can be faulted")
        self.generation = generation

    def blocks(self, generation: int) -> bool:
        return generation > 0 and generation == self.generation


def _decode_jpeg(payload: bytes) -> np.ndarray:
    import cv2

    value = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None or value.shape != (480, 640, 3):
        raise ValueError("camera JPEG does not decode to 640x480 RGB")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def operator_camera_roles(backend: str) -> tuple[str, ...]:
    """Return the camera payload required by each AVP display layout."""

    if backend == "sim":
        return ("head_left", "head_right")
    if backend == "real":
        return ("head_left", "head_right", "left_wrist", "right_wrist")
    raise ValueError(f"unsupported operator backend: {backend!r}")


def _operator_hand_status(
    *,
    avp_live: bool,
    requested_generation: int,
    produced_generation: int,
    had_tracking: bool,
) -> str:
    """Return the operator-facing state without affecting control."""

    if not avp_live:
        return "HANDS WAIT"
    if requested_generation > 0:
        return (
            "TRACKING"
            if produced_generation == requested_generation
            else "ANCHORING"
        )
    return "PRESS R" if had_tracking else "HANDS READY"


def _desktop_preview_enabled() -> bool:
    raw = os.environ.get("FLIP_TABLE_TELEOP_DESKTOP_PREVIEW", "true").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("FLIP_TABLE_TELEOP_DESKTOP_PREVIEW must be boolean")


def _desktop_preview_worker(updates) -> None:
    """Run OpenCV GUI in an isolated process; never participate in control."""

    if not os.environ.get("DISPLAY"):
        print(
            "Desktop camera monitor disabled because DISPLAY is unset.",
            flush=True,
        )
        return
    import cv2

    window = "IROS 2026 RAMEN - Teleoperation Monitor"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1200, 900)
        while True:
            update = updates.get()
            if update is None:
                return
            images = {
                role: _decode_jpeg(payload)
                for role, payload in update["camera_jpeg"].items()
            }
            view = compose_real_desktop_view(
                images["head_left"],
                images["head_right"],
                images["left_wrist"],
                images["right_wrist"],
                np.asarray(update["arm_joint_position_rad"], dtype=np.float64),
                str(update["hand_status"]),
            )
            # Composition uses RGB; OpenCV windows consume BGR.
            cv2.imshow(window, view[..., ::-1])
            cv2.waitKey(1)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return
    except BaseException as exc:  # noqa: BLE001
        # A Desktop/Qt failure must not stop AVP IK or the robot safety loop.
        print(f"Desktop camera monitor stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass


class DesktopPreviewProcess:
    """Non-blocking latest-frame publisher for the real Desktop monitor."""

    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_desktop_preview_worker,
            args=(self._updates,),
            name="flip-table-desktop-monitor",
            daemon=True,
        )
        self._process.start()

    def submit(
        self,
        camera_jpeg: dict[str, bytes],
        arm_joint_position_rad: list[float],
        hand_status: str,
    ) -> None:
        if not self._process.is_alive():
            return
        update = {
            "camera_jpeg": dict(camera_jpeg),
            "arm_joint_position_rad": list(arm_joint_position_rad),
            "hand_status": hand_status,
        }
        try:
            self._updates.put_nowait(update)
        except queue.Full:
            # Never make AVP/control wait for a Desktop paint. The queued item
            # is at most one camera period old and the next free submit wins.
            pass

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._updates.put_nowait(None)
            except queue.Full:
                try:
                    self._updates.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._updates.put_nowait(None)
                except queue.Full:
                    pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._updates.close()


def _worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ipc-port", type=int)
    parser.add_argument("--xr-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    args = parser.parse_args()
    if args.worker and (args.ipc_port is None or args.xr_root is None):
        parser.error("--worker requires --ipc-port and --xr-root")
    return args


def _worker_main(args: argparse.Namespace) -> int:
    config = load_teleop_config(args.config)
    connection = socket.create_connection(("127.0.0.1", args.ipc_port), timeout=30.0)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(None)
    transport = FramedSocket(connection)
    xr: XrInput | None = None
    desktop_preview: DesktopPreviewProcess | None = None
    tracking_fault = _TrackingFaultLatch()
    last_image_processing_ms = 0.0
    had_tracking = False
    try:
        xr = XrInput(args.xr_root, config, real_operator_hud=args.backend == "real")
        if args.backend == "real" and _desktop_preview_enabled():
            desktop_preview = DesktopPreviewProcess()
            print(
                "Desktop camera monitor enabled: head L/R, wrist L/R, and "
                "measured arm angles.",
                flush=True,
            )
        transport.send({"type": "ready", "config_sha256": config.digest})
        while True:
            message = transport.receive()
            kind = message.get("type")
            if kind == "close":
                return 0
            if kind != "update":
                raise ValueError(f"unsupported operator IPC message: {kind!r}")
            source_sequence = message.get("sequence")
            if not isinstance(source_sequence, int) or source_sequence < 0:
                raise ValueError("operator IPC update has an invalid sequence")
            processing_started = time.monotonic()
            # The control path is intentionally completed and returned before
            # JPEG decoding/HUD composition. Official xr_teleoperate solves IK
            # directly from XR + motor state; display work must not add command
            # latency or make a slow Desktop/AVP paint look like stale hands.
            check_ns = time.monotonic_ns()
            liveness = xr.liveness(check_ns)
            avp_live = xr.connected(
                check_ns,
                config.safety.command_hold_timeout_s,
                config.safety.command_hold_timeout_s,
            )
            camera_jpeg = message.get("camera_jpeg")
            if camera_jpeg is not None:
                expected_roles = set(operator_camera_roles(args.backend))
                if not isinstance(camera_jpeg, dict) or set(camera_jpeg) != expected_roles:
                    raise ValueError(
                        "operator IPC camera update has unexpected roles: "
                        f"expected={sorted(expected_roles)} got="
                        f"{sorted(camera_jpeg) if isinstance(camera_jpeg, dict) else camera_jpeg!r}"
                    )
            tracking_generation = message.get("tracking_generation")
            if not isinstance(tracking_generation, int) or tracking_generation < 0:
                raise ValueError("operator IPC tracking_generation must be non-negative")
            response: dict[str, Any] = {
                "type": "target",
                "source_sequence": source_sequence,
                "monotonic_ns": check_ns,
                "avp_live": avp_live,
                "tracking_generation": 0,
                **liveness,
            }
            ik_processing_ms = 0.0
            generation_is_faulted = tracking_fault.blocks(tracking_generation)
            if generation_is_faulted:
                # Latest-only IPC may skip the first fault response. Keep the
                # fault visible until the parent disarms and issues a new,
                # monotonically increasing re-anchor generation.
                xr.disarm()
                response["avp_live"] = False
            elif avp_live and tracking_generation > 0:
                ik_started = time.monotonic()
                try:
                    target = xr.target(
                        np.asarray(message["arm_joint_position_rad"], dtype=np.float64),
                        np.asarray(message["arm_joint_velocity_rad_s"], dtype=np.float64),
                        np.asarray(message["dex1_opening_fraction"], dtype=np.float64),
                        tracking_generation,
                    )
                except IncoherentBilateralHandFrame:
                    # A torn WebXR snapshot is transient, but it is unsafe to
                    # continue a previously anchored command without a known
                    # bilateral pose.  Report it as non-live so the parent
                    # session publishes a HOLD target and asks for a fresh `r`
                    # anchor; never kill the AVP process or the simulator.
                    xr.disarm()
                    tracking_fault.trip(tracking_generation)
                    response["avp_live"] = False
                    target = None
                    print(
                        "AVP bilateral hand frame was incoherent; holding and "
                        "requiring re-anchor.",
                        flush=True,
                    )
                if target is not None:
                    arm, torque, hand = target
                    had_tracking = True
                    # Timestamp freshness immediately after IK. Rendering is
                    # deliberately downstream and cannot age the command.
                    response["monotonic_ns"] = time.monotonic_ns()
                    response["tracking_generation"] = tracking_generation
                    response["arm_position_rad"] = [float(value) for value in arm]
                    response["arm_feedforward_torque_nm"] = [
                        float(value) for value in torque
                    ]
                    response["dex1_opening_fraction"] = [
                        float(hand[0]),
                        float(hand[1]),
                    ]
                ik_processing_ms = (time.monotonic() - ik_started) * 1000.0
            else:
                xr.disarm()
            # Report the previous completed display update because this target
            # is sent before the current display update begins.
            response["image_processing_ms"] = last_image_processing_ms
            response["ik_processing_ms"] = ik_processing_ms
            response["total_processing_ms"] = (
                time.monotonic() - processing_started
            ) * 1000.0
            transport.send(response)

            if camera_jpeg is not None:
                image_started = time.monotonic()
                images = {
                    role: _decode_jpeg(payload)
                    for role, payload in camera_jpeg.items()
                }
                if args.backend == "real":
                    hand_status = _operator_hand_status(
                        avp_live=avp_live,
                        requested_generation=tracking_generation,
                        produced_generation=int(response["tracking_generation"]),
                        had_tracking=had_tracking,
                    )
                    xr.render(
                        compose_real_operator_stereo_view(
                            images["head_left"],
                            images["head_right"],
                            images["left_wrist"],
                            images["right_wrist"],
                            np.asarray(
                                message["arm_joint_position_rad"],
                                dtype=np.float64,
                            ),
                            hand_status,
                        )
                    )
                    if desktop_preview is not None:
                        desktop_preview.submit(
                            camera_jpeg,
                            message["arm_joint_position_rad"],
                            hand_status,
                        )
                else:
                    xr.render(
                        compose_head_stereo_view(
                            images["head_left"], images["head_right"]
                        )
                    )
                last_image_processing_ms = (
                    time.monotonic() - image_started
                ) * 1000.0
    finally:
        if desktop_preview is not None:
            desktop_preview.close()
        if xr is not None:
            xr.close()
        transport.close()


class OperatorProcess:
    """Run AVP UI/IK behind an exec-isolated localhost IPC channel."""

    def __init__(
        self,
        xr_root: str | Path,
        config: TeleopConfig,
        *,
        backend: str = "sim",
    ) -> None:
        self._xr_root = str(Path(xr_root).expanduser().resolve())
        self._config = config
        self._backend = backend
        self._camera_roles = operator_camera_roles(backend)
        self._listener: socket.socket | None = None
        self._transport: FramedSocket | None = None
        self._process: subprocess.Popen[str] | None = None
        self._receiver: threading.Thread | None = None
        self._receiver_error: BaseException | None = None
        self._pending_condition = threading.Condition()
        self._pending_update: dict[str, Any] | None = None
        self._update_inflight = False
        self._sender_stop = False
        self._sender: threading.Thread | None = None
        self._submit_sequence = 0
        self._target_lock = threading.Lock()
        self._latest_target: OperatorTarget | None = None

    def start(self, *, timeout_s: float = 60.0) -> None:
        if self._process is not None:
            raise RuntimeError("operator process is already running")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(timeout_s)
        self._listener = listener
        port = int(listener.getsockname()[1])
        command = [
            sys.executable,
            "-m",
            "data.flip_table_data_augmentation.teleop.operator_process",
            "--worker",
            "--ipc-port",
            str(port),
            "--xr-root",
            self._xr_root,
            "--config",
            str(self._config.path),
            "--backend",
            self._backend,
        ]
        # The parent converts terminal SIGINT into a controlled real-robot
        # release. Keep the WebXR/IK child outside the terminal process group
        # so it cannot die before the parent has sent the release command.
        self._process = subprocess.Popen(
            command,
            close_fds=True,
            text=True,
            start_new_session=True,
        )
        try:
            connection, _address = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(None)
            transport = FramedSocket(connection)
            hello = transport.receive(timeout_s=timeout_s)
            if hello != {"type": "ready", "config_sha256": self._config.digest}:
                raise RuntimeError(f"AVP operator handshake differs from config: {hello!r}")
            self._transport = transport
            self._receiver = threading.Thread(
                target=self._receive_targets,
                name="flip-table-avp-targets",
                daemon=True,
            )
            self._receiver.start()
            self._sender = threading.Thread(
                target=self._send_updates,
                name="flip-table-avp-updates",
                daemon=True,
            )
            self._sender.start()
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()
            self._listener = None

    def _receive_targets(self) -> None:
        assert self._transport is not None
        try:
            while True:
                message = self._transport.receive()
                if message.get("type") != "target":
                    raise ValueError(f"unexpected operator IPC response: {message!r}")
                live = message.get("avp_live")
                timestamp = message.get("monotonic_ns")
                generation = message.get("tracking_generation")
                session_age_s = message.get("session_age_s")
                hand_age_s = message.get("hand_age_s")
                hand_tracking_hz = message.get("hand_tracking_hz")
                hand_event_count = message.get("hand_event_count")
                hand_contiguous_event_count = message.get(
                    "hand_contiguous_event_count"
                )
                hand_invalid_event_count = message.get("hand_invalid_event_count")
                hand_missing_pose_count = message.get("hand_missing_pose_count")
                hand_invalid_wrist_count = message.get("hand_invalid_wrist_count")
                hand_invalid_pinch_count = message.get("hand_invalid_pinch_count")
                hand_invalid_unused_skeleton_count = message.get(
                    "hand_invalid_unused_skeleton_count"
                )
                hand_invalid_details = message.get("hand_invalid_details")
                source_sequence = message.get("source_sequence")
                image_processing_ms = message.get("image_processing_ms")
                ik_processing_ms = message.get("ik_processing_ms")
                total_processing_ms = message.get("total_processing_ms")
                if not isinstance(live, bool) or not isinstance(timestamp, int) or timestamp <= 0:
                    raise ValueError("operator response has invalid heartbeat")
                if not isinstance(generation, int) or generation < 0:
                    raise ValueError("operator response has invalid tracking generation")
                if not isinstance(source_sequence, int) or source_sequence < 0:
                    raise ValueError("operator response has invalid source sequence")
                timings = (
                    image_processing_ms,
                    ik_processing_ms,
                    total_processing_ms,
                )
                if any(
                    not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0.0
                    for value in timings
                ):
                    raise ValueError("operator response has invalid processing timings")
                if session_age_s is not None and not isinstance(session_age_s, (int, float)):
                    raise ValueError("operator response has invalid AVP session age")
                if hand_age_s is not None and not isinstance(hand_age_s, (int, float)):
                    raise ValueError("operator response has invalid AVP hand age")
                if (
                    not isinstance(hand_tracking_hz, (int, float))
                    or not np.isfinite(hand_tracking_hz)
                    or hand_tracking_hz < 0.0
                ):
                    raise ValueError("operator response has invalid hand-tracking rate")
                if not isinstance(hand_event_count, int) or hand_event_count < 0:
                    raise ValueError("operator response has invalid hand-event count")
                if (
                    not isinstance(hand_contiguous_event_count, int)
                    or hand_contiguous_event_count < 0
                ):
                    raise ValueError(
                        "operator response has invalid contiguous hand-event count"
                    )
                diagnostic_counts = (
                    hand_invalid_event_count,
                    hand_missing_pose_count,
                    hand_invalid_wrist_count,
                    hand_invalid_pinch_count,
                    hand_invalid_unused_skeleton_count,
                )
                if any(
                    not isinstance(value, int) or value < 0
                    for value in diagnostic_counts
                ):
                    raise ValueError(
                        "operator response has invalid hand-event diagnostics"
                    )
                if (
                    not isinstance(hand_invalid_details, dict)
                    or any(
                        not isinstance(key, str)
                        or not isinstance(value, int)
                        or value < 0
                        for key, value in hand_invalid_details.items()
                    )
                ):
                    raise ValueError(
                        "operator response has invalid detailed hand diagnostics"
                    )
                arm: tuple[float, ...] | None = None
                torque: tuple[float, ...] | None = None
                hand: tuple[float, float] | None = None
                if generation > 0:
                    if not live:
                        raise ValueError(
                            "operator returned an anchored target without live AVP input"
                        )
                    arm_values = message.get("arm_position_rad")
                    torque_values = message.get("arm_feedforward_torque_nm")
                    hand_values = message.get("dex1_opening_fraction")
                    if not isinstance(arm_values, list) or len(arm_values) != 14:
                        raise ValueError("operator target arm position must be 14-D")
                    if not isinstance(hand_values, list) or len(hand_values) != 2:
                        raise ValueError("operator target hand opening must be 2-D")
                    if not isinstance(torque_values, list) or len(torque_values) != 14:
                        raise ValueError(
                            "operator target arm feedforward torque must be 14-D"
                        )
                    arm = tuple(float(value) for value in arm_values)
                    torque = tuple(float(value) for value in torque_values)
                    hand = (float(hand_values[0]), float(hand_values[1]))
                    if (
                        not np.isfinite(arm).all()
                        or not np.isfinite(torque).all()
                        or not np.isfinite(hand).all()
                    ):
                        raise ValueError("operator target contains NaN or Inf")
                with self._target_lock:
                    self._latest_target = OperatorTarget(
                        monotonic_ns=timestamp,
                        avp_live=live,
                        tracking_generation=generation,
                        arm_position_rad=arm,
                        arm_feedforward_torque_nm=torque,
                        dex1_opening_fraction=hand,
                        session_age_s=(
                            None if session_age_s is None else float(session_age_s)
                        ),
                        hand_age_s=None if hand_age_s is None else float(hand_age_s),
                        hand_tracking_hz=float(hand_tracking_hz),
                        hand_event_count=hand_event_count,
                        hand_contiguous_event_count=hand_contiguous_event_count,
                        hand_invalid_event_count=hand_invalid_event_count,
                        hand_missing_pose_count=hand_missing_pose_count,
                        hand_invalid_wrist_count=hand_invalid_wrist_count,
                        hand_invalid_pinch_count=hand_invalid_pinch_count,
                        hand_invalid_unused_skeleton_count=(
                            hand_invalid_unused_skeleton_count
                        ),
                        hand_invalid_details=dict(hand_invalid_details),
                        source_sequence=source_sequence,
                        image_processing_ms=float(image_processing_ms),
                        ik_processing_ms=float(ik_processing_ms),
                        total_processing_ms=float(total_processing_ms),
                    )
                with self._pending_condition:
                    self._update_inflight = False
                    self._pending_condition.notify_all()
        except (EOFError, OSError) as exc:
            self._receiver_error = exc
        except BaseException as exc:  # noqa: BLE001
            self._receiver_error = exc
            with self._pending_condition:
                self._pending_condition.notify_all()

    def _send_updates(self) -> None:
        """Send at most one request at a time and discard stale pending state."""

        try:
            while True:
                with self._pending_condition:
                    self._pending_condition.wait_for(
                        lambda: self._sender_stop
                        or self._receiver_error is not None
                        or (
                            self._pending_update is not None
                            and not self._update_inflight
                        )
                    )
                    if self._sender_stop or self._receiver_error is not None:
                        return
                    message = self._pending_update
                    self._pending_update = None
                    self._update_inflight = True
                assert message is not None
                transport = self._transport
                if transport is None:
                    raise RuntimeError("AVP operator transport closed while sending")
                transport.send(message)
        except BaseException as exc:  # noqa: BLE001
            self._receiver_error = exc
            with self._pending_condition:
                self._update_inflight = False
                self._pending_condition.notify_all()

    def submit(
        self,
        *,
        camera_jpeg: dict[str, bytes] | None,
        arm_joint_position_rad: tuple[float, ...],
        arm_joint_velocity_rad_s: tuple[float, ...],
        dex1_opening_fraction: tuple[float, float],
        tracking_generation: int,
    ) -> None:
        if self._transport is None:
            raise RuntimeError("AVP operator process is not running")
        if self._receiver_error is not None:
            raise RuntimeError(f"AVP operator channel failed: {self._receiver_error}")
        if camera_jpeg is not None and set(camera_jpeg) != set(self._camera_roles):
            raise ValueError(
                "operator camera update has unexpected roles: "
                f"expected={list(self._camera_roles)} got={sorted(camera_jpeg)}"
            )
        if not isinstance(tracking_generation, int) or tracking_generation < 0:
            raise ValueError("tracking_generation must be non-negative")
        with self._pending_condition:
            if camera_jpeg is None and self._pending_update is not None:
                camera_jpeg = self._pending_update.get("camera_jpeg")
            self._submit_sequence += 1
            self._pending_update = {
                "type": "update",
                "sequence": self._submit_sequence,
                "camera_jpeg": None if camera_jpeg is None else dict(camera_jpeg),
                "arm_joint_position_rad": list(arm_joint_position_rad),
                "arm_joint_velocity_rad_s": list(arm_joint_velocity_rad_s),
                "dex1_opening_fraction": list(dex1_opening_fraction),
                "tracking_generation": tracking_generation,
            }
            self._pending_condition.notify_all()

    def latest_target(self) -> OperatorTarget | None:
        if self._receiver_error is not None:
            raise RuntimeError(f"AVP operator channel failed: {self._receiver_error}")
        with self._target_lock:
            return self._latest_target

    def submitted_sequence(self) -> int:
        with self._pending_condition:
            return self._submit_sequence

    def close(self) -> None:
        with self._pending_condition:
            self._sender_stop = True
            self._pending_update = None
            self._pending_condition.notify_all()
        sender = self._sender
        self._sender = None
        if sender is not None and sender is not threading.current_thread():
            sender.join(timeout=1.0)
        transport = self._transport
        self._transport = None
        if transport is not None:
            try:
                transport.send({"type": "close"})
            except OSError:
                pass
            transport.close()
        receiver = self._receiver
        self._receiver = None
        if receiver is not None and receiver is not threading.current_thread():
            receiver.join(timeout=1.0)
        process = self._process
        self._process = None
        if process is not None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> int:
    args = _worker_args()
    if not args.worker:
        raise SystemExit("operator_process is launched by the teleoperation session")
    return _worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
