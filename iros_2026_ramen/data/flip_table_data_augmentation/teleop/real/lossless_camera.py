"""Lossless physical-camera recording alongside the latest-only AVP preview.

The pinned Unitree TeleImager transport intentionally keeps only the newest
JPEG.  That is the correct contract for a low-latency headset, but it cannot be
the source of a lossless demonstration dataset.  This module adds a
repository-owned MCAP recorder and a small camera-only control protocol.  It
does not import Unitree DDS and cannot send robot commands.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import socketserver
import struct
import threading
import time
from typing import Any, Iterable, Mapping


CAMERA_FRAME_SCHEMA_VERSION = "team_ramen_camera_frame/v1"
CAMERA_RECORDING_SCHEMA_VERSION = "team_ramen_camera_recording/v1"
CAMERA_CONTROL_SCHEMA_VERSION = "team_ramen_camera_recorder_control/v1"
CAMERA_CONTROL_PORT = 60010
PHYSICAL_CAMERA_ROLES = ("head_stereo", "left_wrist", "right_wrist")
_HEADER_LENGTH = struct.Struct("!I")
_MAXIMUM_CONTROL_LINE_BYTES = 64 * 1024
MINIMUM_RECORDING_FREE_BYTES = 10 * 1024**3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class CameraFrameEnvelope:
    role: str
    usb_serial: str
    source_sequence: int
    orin_capture_monotonic_ns: int
    jpeg: bytes
    device_frame_counter: int | None = None
    device_timestamp: float | None = None
    timestamp_domain: str | None = None

    def __post_init__(self) -> None:
        if self.role not in PHYSICAL_CAMERA_ROLES:
            raise ValueError(f"unsupported physical camera role: {self.role!r}")
        if not self.usb_serial:
            raise ValueError("camera USB serial must be non-empty")
        if self.source_sequence <= 0 or self.orin_capture_monotonic_ns <= 0:
            raise ValueError("camera sequence/time must be positive")
        if not isinstance(self.jpeg, bytes) or not self.jpeg:
            raise ValueError("camera payload must be non-empty JPEG bytes")
        if self.device_frame_counter is not None and self.device_frame_counter < 0:
            raise ValueError("device frame counter must be non-negative")
        if self.device_timestamp is not None and not math.isfinite(
            self.device_timestamp
        ):
            raise ValueError("device timestamp must be finite")

    @property
    def jpeg_sha256(self) -> str:
        return hashlib.sha256(self.jpeg).hexdigest()

    @property
    def jpeg_bytes(self) -> bytes:
        """Public V1 contract spelling; ``jpeg`` remains the internal alias."""

        return self.jpeg

    def header(self) -> dict[str, object]:
        return {
            "schema_version": CAMERA_FRAME_SCHEMA_VERSION,
            "role": self.role,
            "usb_serial": self.usb_serial,
            "source_sequence": self.source_sequence,
            "orin_capture_monotonic_ns": self.orin_capture_monotonic_ns,
            "device_frame_counter": self.device_frame_counter,
            "device_timestamp": self.device_timestamp,
            "timestamp_domain": self.timestamp_domain,
            "jpeg_sha256": self.jpeg_sha256,
            "jpeg_size_bytes": len(self.jpeg),
        }

    def encode(self) -> bytes:
        header = json.dumps(
            self.header(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return _HEADER_LENGTH.pack(len(header)) + header + self.jpeg

    @classmethod
    def decode(cls, payload: bytes) -> "CameraFrameEnvelope":
        if len(payload) < _HEADER_LENGTH.size:
            raise ValueError("camera envelope is truncated")
        (header_size,) = _HEADER_LENGTH.unpack_from(payload)
        boundary = _HEADER_LENGTH.size + header_size
        if header_size <= 0 or boundary >= len(payload):
            raise ValueError("camera envelope header length is invalid")
        header = json.loads(payload[_HEADER_LENGTH.size : boundary])
        if (
            not isinstance(header, dict)
            or header.get("schema_version") != CAMERA_FRAME_SCHEMA_VERSION
        ):
            raise ValueError("unsupported camera frame envelope")
        jpeg = bytes(payload[boundary:])
        value = cls(
            role=str(header["role"]),
            usb_serial=str(header["usb_serial"]),
            source_sequence=int(header["source_sequence"]),
            orin_capture_monotonic_ns=int(
                header["orin_capture_monotonic_ns"]
            ),
            device_frame_counter=(
                None
                if header.get("device_frame_counter") is None
                else int(header["device_frame_counter"])
            ),
            device_timestamp=_finite_optional(header.get("device_timestamp")),
            timestamp_domain=(
                None
                if header.get("timestamp_domain") is None
                else str(header["timestamp_domain"])
            ),
            jpeg=jpeg,
        )
        if header.get("jpeg_size_bytes") != len(jpeg):
            raise ValueError("camera envelope JPEG length differs from its header")
        if header.get("jpeg_sha256") != value.jpeg_sha256:
            raise ValueError("camera envelope JPEG digest mismatch")
        return value


# Public contract name used by sidecars and integration documentation. Keep
# the short alias for internal call sites without creating two wire formats.
CameraFrameEnvelopeV1 = CameraFrameEnvelope


@dataclass(frozen=True)
class ClockSyncSample:
    desktop_send_ns: int
    orin_receive_ns: int
    orin_send_ns: int
    desktop_receive_ns: int

    @property
    def round_trip_ns(self) -> int:
        return max(
            0,
            (self.desktop_receive_ns - self.desktop_send_ns)
            - (self.orin_send_ns - self.orin_receive_ns),
        )

    @property
    def desktop_to_orin_offset_ns(self) -> float:
        return (
            (self.orin_receive_ns - self.desktop_send_ns)
            + (self.orin_send_ns - self.desktop_receive_ns)
        ) / 2.0


@dataclass(frozen=True)
class ClockMapping:
    desktop_to_orin_offset_ns: float
    uncertainty_ns: float
    samples: tuple[ClockSyncSample, ...]

    def desktop_to_orin(self, timestamp_ns: int) -> int:
        return int(round(timestamp_ns + self.desktop_to_orin_offset_ns))

    def to_json(self) -> dict[str, object]:
        return {
            "desktop_to_orin_offset_ns": self.desktop_to_orin_offset_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "samples": [
                {
                    "desktop_send_ns": sample.desktop_send_ns,
                    "orin_receive_ns": sample.orin_receive_ns,
                    "orin_send_ns": sample.orin_send_ns,
                    "desktop_receive_ns": sample.desktop_receive_ns,
                    "round_trip_ns": sample.round_trip_ns,
                    "desktop_to_orin_offset_ns": (
                        sample.desktop_to_orin_offset_ns
                    ),
                }
                for sample in self.samples
            ],
        }


def estimate_clock_mapping(
    samples: Iterable[ClockSyncSample], *, best_sample_count: int = 4
) -> ClockMapping:
    values = tuple(samples)
    if len(values) < best_sample_count or best_sample_count <= 0:
        raise ValueError("insufficient clock synchronization samples")
    selected = tuple(
        sorted(values, key=lambda sample: sample.round_trip_ns)[:best_sample_count]
    )
    offsets = sorted(sample.desktop_to_orin_offset_ns for sample in selected)
    midpoint = len(offsets) // 2
    offset = (
        offsets[midpoint]
        if len(offsets) % 2
        else (offsets[midpoint - 1] + offsets[midpoint]) / 2.0
    )
    uncertainty = max(
        max(sample.round_trip_ns / 2.0 for sample in selected),
        max(abs(value - offset) for value in offsets),
    )
    return ClockMapping(offset, uncertainty, selected)


class LosslessCameraRecorder:
    """Thread-safe MCAP recorder fed directly by camera acquisition threads."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        queue_capacity: int = 900,
    ) -> None:
        if queue_capacity < 90:
            raise ValueError("camera recorder queue must hold at least one second")
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[CameraFrameEnvelope] = queue.Queue(
            maxsize=queue_capacity
        )
        self._lock = threading.RLock()
        self._active = False
        self._failed: str | None = None
        self._session_id: str | None = None
        self._partial_path: Path | None = None
        self._final_path: Path | None = None
        self._stream = None
        self._writer = None
        self._channels: dict[str, int] = {}
        self._counts = {role: 0 for role in PHYSICAL_CAMERA_ROLES}
        self._first_capture_ns: dict[str, int] = {}
        self._last_capture_ns: dict[str, int] = {}
        self._last_sequence: dict[str, int] = {}
        self._queue_high_watermark = 0
        self._finalizing = False
        self._last_result: dict[str, object] | None = None
        self._finalize_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(
            target=self._write_loop,
            name="lossless-camera-mcap-writer",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _safe_session_id(value: str) -> str:
        if (
            not value
            or len(value) > 128
            or value[0] in ".-"
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value)
        ):
            raise ValueError("invalid camera recording session_id")
        return value

    def start(self, session_id: str) -> dict[str, object]:
        from mcap.writer import CompressionType, Writer

        normalized = self._safe_session_id(session_id)
        with self._lock:
            if self._active or self._writer is not None or self._finalizing:
                raise RuntimeError("a camera recording is already active/finalizing")
            if not self._queue.empty():
                raise RuntimeError("camera writer queue was not drained")
            free_bytes = shutil.disk_usage(self.output_root).free
            if free_bytes < MINIMUM_RECORDING_FREE_BYTES:
                raise RuntimeError(
                    "insufficient camera recording space: "
                    f"free={free_bytes},required={MINIMUM_RECORDING_FREE_BYTES}"
                )
            final = self.output_root / f"{normalized}.mcap"
            partial = final.with_suffix(".mcap.partial")
            if final.exists() or partial.exists():
                raise FileExistsError(final)
            stream = partial.open("xb")
            writer = Writer(
                stream,
                chunk_size=4 * 1024 * 1024,
                compression=CompressionType.ZSTD,
                enable_crcs=True,
                enable_data_crcs=True,
            )
            writer.start(
                profile="team_ramen_camera_recording",
                library="iros_2026_ramen",
            )
            schema = writer.register_schema(
                name=CAMERA_FRAME_SCHEMA_VERSION,
                encoding="jsonschema",
                data=json.dumps(
                    {
                        "type": "object",
                        "required": [
                            "role",
                            "usb_serial",
                            "source_sequence",
                            "orin_capture_monotonic_ns",
                            "jpeg_sha256",
                        ],
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            )
            channels = {
                role: writer.register_channel(
                    topic=f"/team_ramen/cameras/{role}/jpeg",
                    message_encoding=CAMERA_FRAME_SCHEMA_VERSION,
                    schema_id=schema,
                    metadata={"role": role, "encoding": "jpeg"},
                )
                for role in PHYSICAL_CAMERA_ROLES
            }
            self._session_id = normalized
            self._partial_path = partial
            self._final_path = final
            self._stream = stream
            self._writer = writer
            self._channels = channels
            self._failed = None
            self._counts = {role: 0 for role in PHYSICAL_CAMERA_ROLES}
            self._first_capture_ns.clear()
            self._last_capture_ns.clear()
            self._last_sequence.clear()
            self._queue_high_watermark = 0
            self._last_result = None
            self._active = True
            return self.status()

    def capture(self, frame: CameraFrameEnvelope) -> bool:
        with self._lock:
            if not self._active or self._failed is not None:
                return False
            try:
                # Keep the active-state check and enqueue atomic with respect
                # to stop().  Otherwise stop could observe an empty queue,
                # close the writer, and race with a late enqueue.
                self._queue.put_nowait(frame)
            except queue.Full:
                self._failed = (
                    f"writer_queue_overflow:capacity={self._queue.maxsize}"
                )
                self._active = False
                return False
            self._queue_high_watermark = max(
                self._queue_high_watermark, self._queue.qsize()
            )
        return True

    def _write_loop(self) -> None:
        while not self._worker_stop.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    writer = self._writer
                    channel = self._channels.get(frame.role)
                if writer is None or channel is None:
                    raise RuntimeError("camera frame arrived without an active writer")
                writer.add_message(
                    channel_id=channel,
                    log_time=frame.orin_capture_monotonic_ns,
                    publish_time=frame.orin_capture_monotonic_ns,
                    sequence=frame.source_sequence,
                    data=frame.encode(),
                )
                with self._lock:
                    previous = self._last_sequence.get(frame.role)
                    if previous is not None and frame.source_sequence != previous + 1:
                        self._failed = (
                            f"{frame.role}_source_sequence_gap:"
                            f"{previous}->{frame.source_sequence}"
                        )
                        self._active = False
                    self._last_sequence[frame.role] = frame.source_sequence
                    self._counts[frame.role] += 1
                    self._first_capture_ns.setdefault(
                        frame.role, frame.orin_capture_monotonic_ns
                    )
                    self._last_capture_ns[frame.role] = (
                        frame.orin_capture_monotonic_ns
                    )
            except BaseException as exc:  # noqa: BLE001
                with self._lock:
                    self._failed = f"{type(exc).__name__}: {exc}"
                    self._active = False
            finally:
                self._queue.task_done()

    def stop(self) -> dict[str, object]:
        # Finalizing and hashing a multi-gigabyte MCAP can take longer than the
        # normal control timeout.  Serialize stop calls and make them
        # idempotent so a client may safely retry after a network timeout.
        with self._finalize_lock:
            with self._lock:
                if self._writer is None:
                    if self._last_result is not None:
                        return dict(self._last_result)
                    raise RuntimeError("no camera recording is active")
                self._active = False
                self._finalizing = True
            try:
                self._queue.join()
                with self._lock:
                    writer = self._writer
                    stream = self._stream
                    partial = self._partial_path
                    final = self._final_path
                    failed = self._failed
                    self._writer = None
                    self._stream = None
                    self._channels = {}
                assert writer is not None and stream is not None
                assert partial is not None and final is not None
                try:
                    writer.finish()
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    stream.close()
                os.replace(partial, final)
                with self._lock:
                    self._partial_path = None
                    self._final_path = final
                result = self.status()
                result.update(
                    {
                        "active": False,
                        "finalizing": False,
                        "failed": failed,
                        "path": str(final),
                        "sha256": _sha256_file(final),
                    }
                )
                sidecar = final.with_suffix(".json")
                sidecar.write_text(
                    json.dumps(
                        result, indent=2, sort_keys=True, allow_nan=False
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self._lock:
                    self._session_id = None
                    self._final_path = None
                    self._last_result = dict(result)
                return result
            finally:
                with self._lock:
                    self._finalizing = False

    def status(self) -> dict[str, object]:
        with self._lock:
            rates: dict[str, float] = {}
            for role, count in self._counts.items():
                first = self._first_capture_ns.get(role)
                last = self._last_capture_ns.get(role)
                rates[role] = (
                    0.0
                    if count < 2 or first is None or last is None or last <= first
                    else (count - 1) / ((last - first) / 1.0e9)
                )
            return {
                "schema_version": CAMERA_RECORDING_SCHEMA_VERSION,
                "active": self._active,
                "finalizing": self._finalizing,
                "failed": self._failed,
                "session_id": self._session_id,
                "partial_path": (
                    None if self._partial_path is None else str(self._partial_path)
                ),
                "counts": dict(self._counts),
                "recorded_source_hz": rates,
                "last_sequence": dict(self._last_sequence),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "queue_high_watermark": self._queue_high_watermark,
                "disk_free_bytes": shutil.disk_usage(self.output_root).free,
                "minimum_recording_free_bytes": MINIMUM_RECORDING_FREE_BYTES,
                "last_result": (
                    None
                    if self._last_result is None
                    else dict(self._last_result)
                ),
            }

    def close(self) -> None:
        with self._lock:
            active = self._writer is not None
        if active:
            try:
                self.stop()
            except Exception:
                pass
        self._worker_stop.set()
        self._worker.join(timeout=5.0)


class _RecorderControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        receive_ns = time.monotonic_ns()
        line = self.rfile.readline(_MAXIMUM_CONTROL_LINE_BYTES + 1)
        if not line or len(line) > _MAXIMUM_CONTROL_LINE_BYTES:
            return
        try:
            request = json.loads(line)
            if (
                not isinstance(request, dict)
                or request.get("schema_version") != CAMERA_CONTROL_SCHEMA_VERSION
            ):
                raise ValueError("unsupported recorder control request")
            command = request.get("command")
            recorder = self.server.recorder  # type: ignore[attr-defined]
            if command == "clock_sync":
                result: dict[str, object] = {
                    "orin_receive_ns": receive_ns,
                }
            elif command == "status":
                result = recorder.status()
            elif command == "start":
                result = recorder.start(str(request["session_id"]))
            elif command == "stop":
                result = recorder.stop()
            else:
                raise ValueError(f"unsupported recorder command: {command!r}")
            response = {"ok": True, "result": result}
        except BaseException as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        response["schema_version"] = CAMERA_CONTROL_SCHEMA_VERSION
        response["orin_send_ns"] = time.monotonic_ns()
        self.wfile.write(
            json.dumps(response, sort_keys=True, allow_nan=False).encode("utf-8")
            + b"\n"
        )


class RecorderControlServer:
    def __init__(
        self,
        recorder: LosslessCameraRecorder,
        *,
        host: str = "0.0.0.0",
        port: int = CAMERA_CONTROL_PORT,
    ) -> None:
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((host, port), _RecorderControlHandler)
        self._server.recorder = recorder  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-recorder-control",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3.0)


class RecorderControlClient:
    def __init__(
        self,
        host: str,
        *,
        port: int = CAMERA_CONTROL_PORT,
        timeout_s: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def _request(
        self,
        command: str,
        *,
        response_timeout_s: float | None = None,
        **values: object,
    ) -> tuple[dict[str, Any], int, int]:
        send_ns = time.monotonic_ns()
        timeout_s = (
            self.timeout_s
            if response_timeout_s is None
            else response_timeout_s
        )
        request = {
            "schema_version": CAMERA_CONTROL_SCHEMA_VERSION,
            "command": command,
            **values,
        }
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            connection.settimeout(timeout_s)
            connection.sendall(
                json.dumps(request, sort_keys=True, allow_nan=False).encode("utf-8")
                + b"\n"
            )
            stream = connection.makefile("rb")
            line = stream.readline(_MAXIMUM_CONTROL_LINE_BYTES + 1)
        receive_ns = time.monotonic_ns()
        if not line or len(line) > _MAXIMUM_CONTROL_LINE_BYTES:
            raise RuntimeError("camera recorder control response is missing/oversized")
        response = json.loads(line)
        if (
            not isinstance(response, dict)
            or response.get("schema_version") != CAMERA_CONTROL_SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported camera recorder response")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "recorder request failed")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("camera recorder result must be an object")
        result["_orin_send_ns"] = int(response["orin_send_ns"])
        return result, send_ns, receive_ns

    def status(self) -> dict[str, Any]:
        result, _send, _receive = self._request("status")
        return result

    def start(self, session_id: str) -> dict[str, Any]:
        result, _send, _receive = self._request("start", session_id=session_id)
        return result

    def stop(self) -> dict[str, Any]:
        # MCAP finish/fsync/SHA scales with episode size and runs outside the
        # robot control loop.  Permit a ten-minute recording to finalize.
        result, _send, _receive = self._request(
            "stop", response_timeout_s=300.0
        )
        return result

    def synchronize(self, *, sample_count: int = 12) -> ClockMapping:
        samples: list[ClockSyncSample] = []
        for _ in range(sample_count):
            result, send_ns, receive_ns = self._request("clock_sync")
            samples.append(
                ClockSyncSample(
                    desktop_send_ns=send_ns,
                    orin_receive_ns=int(result["orin_receive_ns"]),
                    orin_send_ns=int(result["_orin_send_ns"]),
                    desktop_receive_ns=receive_ns,
                )
            )
        return estimate_clock_mapping(samples)


def read_camera_mcap(
    path: str | Path,
    *,
    require_contiguous_sequences: bool = True,
) -> dict[str, list[CameraFrameEnvelope]]:
    from mcap.reader import make_reader

    source = Path(path).expanduser().resolve()
    result = {role: [] for role in PHYSICAL_CAMERA_ROLES}
    with source.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, _channel, message in reader.iter_messages():
            frame = CameraFrameEnvelope.decode(message.data)
            if message.log_time != frame.orin_capture_monotonic_ns:
                raise ValueError("MCAP log time differs from camera envelope")
            result[frame.role].append(frame)
    for role, frames in result.items():
        if not frames:
            raise ValueError(f"camera MCAP contains no {role} frames")
        sequences = [frame.source_sequence for frame in frames]
        if require_contiguous_sequences and any(
            second != first + 1
            for first, second in zip(sequences, sequences[1:])
        ):
            raise ValueError(f"camera MCAP {role} source sequence is not contiguous")
    return result


def _match_capture_times(
    reference_ns: list[int],
    candidate_ns: list[int],
    *,
    maximum_error_ns: int,
) -> tuple[set[int], list[int]]:
    """Greedily match ordered timestamps without reusing a source frame."""

    matched_reference_indices: set[int] = set()
    errors_ns: list[int] = []
    next_candidate = 0
    for reference_index, reference_time in enumerate(reference_ns):
        if next_candidate >= len(candidate_ns):
            break
        insertion = bisect.bisect_left(
            candidate_ns, reference_time, lo=next_candidate
        )
        choices = []
        if insertion < len(candidate_ns):
            choices.append(insertion)
        if insertion - 1 >= next_candidate:
            choices.append(insertion - 1)
        if not choices:
            continue
        selected = min(
            choices,
            key=lambda index: (
                abs(candidate_ns[index] - reference_time),
                index,
            ),
        )
        error_ns = abs(candidate_ns[selected] - reference_time)
        if error_ns > maximum_error_ns:
            # A candidate older than the reference can never become useful
            # for a later reference.  A newer candidate must remain available.
            if candidate_ns[selected] < reference_time:
                next_candidate = selected + 1
            continue
        matched_reference_indices.add(reference_index)
        errors_ns.append(error_ns)
        next_candidate = selected + 1
    return matched_reference_indices, errors_ns


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def audit_camera_mcap(
    path: str | Path,
    *,
    maximum_match_error_ms: float = 20.0,
) -> dict[str, object]:
    """Stream-verify a large MCAP without retaining JPEG payloads in RAM."""

    from mcap.reader import make_reader

    source = Path(path).expanduser().resolve()
    counts = {role: 0 for role in PHYSICAL_CAMERA_ROLES}
    sequence_gaps = {role: 0 for role in PHYSICAL_CAMERA_ROLES}
    device_counter_gaps = {role: 0 for role in PHYSICAL_CAMERA_ROLES}
    first_capture_ns: dict[str, int] = {}
    last_capture_ns: dict[str, int] = {}
    last_sequence: dict[str, int] = {}
    last_device_counter: dict[str, int] = {}
    capture_times_ns = {role: [] for role in PHYSICAL_CAMERA_ROLES}
    with source.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, _channel, message in reader.iter_messages():
            frame = CameraFrameEnvelope.decode(message.data)
            if message.log_time != frame.orin_capture_monotonic_ns:
                raise ValueError("MCAP log time differs from camera envelope")
            role = frame.role
            previous_sequence = last_sequence.get(role)
            if (
                previous_sequence is not None
                and frame.source_sequence != previous_sequence + 1
            ):
                sequence_gaps[role] += 1
            if frame.device_frame_counter is not None:
                previous_counter = last_device_counter.get(role)
                if (
                    previous_counter is not None
                    and frame.device_frame_counter != previous_counter + 1
                ):
                    device_counter_gaps[role] += 1
                last_device_counter[role] = frame.device_frame_counter
            counts[role] += 1
            last_sequence[role] = frame.source_sequence
            first_capture_ns.setdefault(
                role, frame.orin_capture_monotonic_ns
            )
            last_capture_ns[role] = frame.orin_capture_monotonic_ns
            capture_times_ns[role].append(frame.orin_capture_monotonic_ns)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"camera MCAP is missing a physical role: {counts}")
    rates = {
        role: (
            0.0
            if counts[role] < 2
            or last_capture_ns[role] <= first_capture_ns[role]
            else (counts[role] - 1)
            / (
                (last_capture_ns[role] - first_capture_ns[role])
                / 1.0e9
            )
        )
        for role in PHYSICAL_CAMERA_ROLES
    }
    maximum_match_error_ns = int(maximum_match_error_ms * 1.0e6)
    match_diagnostics: dict[str, dict[str, float | int | None]] = {}
    matched_head_indices: dict[str, set[int]] = {}
    head_times = capture_times_ns["head_stereo"]
    for role in ("left_wrist", "right_wrist"):
        indices, errors_ns = _match_capture_times(
            head_times,
            capture_times_ns[role],
            maximum_error_ns=maximum_match_error_ns,
        )
        matched_head_indices[role] = indices
        match_diagnostics[role] = {
            "matched_count": len(indices),
            "valid_fraction": len(indices) / len(head_times),
            "maximum_error_ms": (
                None if not errors_ns else max(errors_ns) / 1.0e6
            ),
            "p95_error_ms": (
                None
                if not errors_ns
                else _percentile(errors_ns, 0.95) / 1.0e6
            ),
        }
    bundle_indices = (
        matched_head_indices["left_wrist"]
        & matched_head_indices["right_wrist"]
    )
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
        "counts": counts,
        "recorded_source_hz": rates,
        "source_sequence_gaps": sequence_gaps,
        "device_frame_counter_gaps": device_counter_gaps,
        "last_sequence": last_sequence,
        "camera_match": match_diagnostics,
        "bundle_matched_count": len(bundle_indices),
        "bundle_valid_fraction": len(bundle_indices) / len(head_times),
        "maximum_match_error_ms": maximum_match_error_ms,
    }
