"""Backend interface and RoboFinals socket client."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
import socket
import time

from .config import TeleopConfig
from .contracts import ArmHandTarget, MESSAGE_SCHEMA_VERSION, TeleopObservation
from .transport import FramedSocket


class TeleopBackend(ABC):
    @abstractmethod
    def observe(self, timeout_s: float) -> TeleopObservation:
        """Return the newest synchronized camera/proprioception sample."""

    @abstractmethod
    def apply(self, target: ArmHandTarget) -> None:
        """Submit one backend-neutral arm/hand target."""

    @abstractmethod
    def close(self) -> None:
        """Hold safely and release runtime resources."""


class SimSocketBackend(TeleopBackend):
    def __init__(
        self,
        host: str,
        port: int,
        config: TeleopConfig,
        *,
        connect_timeout_s: float = 300.0,
    ) -> None:
        deadline = time.monotonic() + connect_timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            try:
                connection.connect((host, port))
            except OSError as exc:
                last_error = exc
                connection.close()
                time.sleep(0.5)
                continue
            transport = FramedSocket(connection)
            try:
                hello = transport.receive(timeout_s=min(10.0, max(0.1, deadline - time.monotonic())))
            except (EOFError, OSError, TimeoutError) as exc:
                last_error = exc
                transport.close()
                time.sleep(0.5)
                continue
            expected = {
                "schema_version": MESSAGE_SCHEMA_VERSION,
                "type": "hello",
                "backend": "sim",
                "config_sha256": config.digest,
                "runtime_digest": config.runtime.robofinals_digest,
                "servo_hz": config.rates.servo_hz,
                "camera_hz": config.rates.camera_hz,
            }
            if any(hello.get(key) != value for key, value in expected.items()):
                transport.close()
                raise RuntimeError(f"simulator handshake differs from the pinned contract: {hello!r}")
            self.transport = transport
            return
        raise TimeoutError(
            f"could not complete simulator teleop handshake with {host}:{port}"
        ) from last_error

    def observe(self, timeout_s: float) -> TeleopObservation:
        observation = TeleopObservation.from_message(
            self.transport.receive(timeout_s=timeout_s)
        )
        return self._normalize_remote_clock(observation, local_receive_ns=time.monotonic_ns())

    @staticmethod
    def _normalize_remote_clock(
        observation: TeleopObservation,
        *,
        local_receive_ns: int,
    ) -> TeleopObservation:
        """Map a simulator sample into the operator PC monotonic clock domain.

        The simulator and the operator PC run on different hosts. Their
        ``time.monotonic_ns`` values are deliberately unrelated, so comparing a
        simulator capture timestamp directly with a locally generated AVP
        target is invalid. The socket receive time is the common local point
        used for command/image alignment. Camera ages are preserved from the
        simulator's single clock domain, and the original remote timestamps
        remain in diagnostics for offline transport analysis.
        """

        if local_receive_ns <= 0:
            raise ValueError("local receive timestamp must be positive")
        remote_capture_ns = observation.capture_monotonic_ns
        remote_camera_times = dict(observation.camera_capture_monotonic_ns)
        remote_diagnostic_times = dict(observation.diagnostic_camera_capture_monotonic_ns)

        def localize(timestamp: int) -> int:
            age_ns = remote_capture_ns - timestamp
            if age_ns < 0:
                raise ValueError("simulator camera timestamp is newer than its observation")
            return max(1, local_receive_ns - age_ns)

        diagnostics = dict(observation.diagnostics)
        diagnostics["transport_timing"] = {
            "clock_domain": "operator_monotonic_receive_time",
            "remote_observation_monotonic_ns": remote_capture_ns,
            "remote_camera_capture_monotonic_ns": remote_camera_times,
            "remote_diagnostic_camera_capture_monotonic_ns": remote_diagnostic_times,
            "local_receive_monotonic_ns": local_receive_ns,
        }
        return replace(
            observation,
            capture_monotonic_ns=local_receive_ns,
            camera_capture_monotonic_ns={
                role: localize(timestamp) for role, timestamp in remote_camera_times.items()
            },
            diagnostic_camera_capture_monotonic_ns={
                role: localize(timestamp)
                for role, timestamp in remote_diagnostic_times.items()
            },
            diagnostics=diagnostics,
        )

    def apply(self, target: ArmHandTarget) -> None:
        self.transport.send(target.to_message())

    def close(self) -> None:
        self.transport.close()
