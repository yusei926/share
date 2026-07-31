"""Non-blocking local telemetry capture for physical policy evaluation.

The recorder is enabled only when the sealed model-evaluation launcher sets
``IROS_REAL_EVAL_CAPTURE_DIR``.  It intentionally lives outside the control
loop: robot commands are never delayed or failed because storage is slow.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any


CAPTURE_ENV = "IROS_REAL_EVAL_CAPTURE_DIR"
CAPTURE_SCHEMA = "team_ramen_real_policy_capture/v1"
CAMERA_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True)
class _ObservationItem:
    state: dict[str, Any]
    frames: tuple[tuple[str, int, bytes, dict[str, Any]], ...]


@dataclass(frozen=True)
class _ActionItem:
    value: dict[str, Any]


class RealEvaluationRecorder:
    """Write state, requested targets, and unique JPEG generations off-thread."""

    def __init__(self, root: Path, *, queue_capacity: int = 1024) -> None:
        if queue_capacity < 32:
            raise ValueError("evaluation capture queue must hold at least 32 records")
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._frames_root = self.root / "frames"
        for role in CAMERA_ROLES:
            (self._frames_root / role).mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[_ObservationItem | _ActionItem | None] = queue.Queue(
            maxsize=queue_capacity
        )
        self._lock = threading.Lock()
        self._closed = False
        self._errors: list[str] = []
        self._queue_drops = 0
        self._observation_count = 0
        self._action_count = 0
        self._frame_counts = {role: 0 for role in CAMERA_ROLES}
        self._last_generation: dict[str, int] = {}
        self._first_frame_ns: dict[str, int] = {}
        self._last_frame_ns: dict[str, int] = {}
        self._thread = threading.Thread(
            target=self._writer,
            name="real-evaluation-recorder",
            daemon=False,
        )
        self._thread.start()

    @classmethod
    def from_environment(cls) -> RealEvaluationRecorder | None:
        value = os.environ.get(CAPTURE_ENV)
        if not value:
            return None
        return cls(Path(value))

    def record_observation(self, observation: Any) -> None:
        if self._closed:
            return
        state = {
            "schema_version": CAPTURE_SCHEMA,
            "type": "state",
            "sequence": int(observation.sequence),
            "capture_monotonic_ns": int(observation.capture_monotonic_ns),
            "body_joint_position_rad": list(observation.body_joint_position_rad),
            "body_joint_velocity_rad_s": list(observation.body_joint_velocity_rad_s),
            "dex1_opening_fraction": list(observation.dex1_opening_fraction),
            "applied_arm_target_rad": list(observation.applied_arm_target_rad),
            "applied_dex1_opening_target": list(
                observation.applied_dex1_opening_target
            ),
            "camera_capture_monotonic_ns": dict(
                observation.camera_capture_monotonic_ns
            ),
            "camera_bundle_valid": bool(observation.camera_bundle_valid),
            "camera_skew_ms": float(observation.camera_skew_ms),
            "stale_roles": list(observation.stale_roles),
            "lower_body_command_dimensions": 0,
        }
        frames: list[tuple[str, int, bytes, dict[str, Any]]] = []
        with self._lock:
            for role in CAMERA_ROLES:
                payload = observation.camera_jpeg.get(role)
                metadata = dict(observation.camera_stream_metadata.get(role, {}))
                if not payload:
                    continue
                generation = int(
                    metadata.get(
                        "jpeg_generation",
                        observation.camera_capture_monotonic_ns[role],
                    )
                )
                if self._last_generation.get(role) == generation:
                    continue
                self._last_generation[role] = generation
                frame_index = self._frame_counts[role]
                self._frame_counts[role] += 1
                capture_ns = int(observation.camera_capture_monotonic_ns[role])
                self._first_frame_ns.setdefault(role, capture_ns)
                self._last_frame_ns[role] = capture_ns
                frames.append((role, frame_index, bytes(payload), {
                    "schema_version": CAPTURE_SCHEMA,
                    "type": "camera_frame",
                    "role": role,
                    "frame_index": frame_index,
                    "jpeg_generation": generation,
                    "capture_monotonic_ns": capture_ns,
                    "source_fps": metadata.get("source_fps"),
                    "transition_hz": metadata.get("transition_hz"),
                }))
            self._observation_count += 1
        self._put(_ObservationItem(state=state, frames=tuple(frames)))

    def record_action(self, target: Any) -> None:
        if self._closed:
            return
        value = dict(target.to_message())
        value["recorded_monotonic_ns"] = time.monotonic_ns()
        value["lower_body_command_dimensions"] = 0
        with self._lock:
            self._action_count += 1
        self._put(_ActionItem(value=value))

    def _put(self, item: _ObservationItem | _ActionItem) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._queue_drops += 1

    def _writer(self) -> None:
        try:
            with (
                (self.root / "states.jsonl").open("a", encoding="utf-8") as states,
                (self.root / "backend_actions.jsonl").open(
                    "a", encoding="utf-8"
                ) as actions,
                (self.root / "camera_frames.jsonl").open(
                    "a", encoding="utf-8"
                ) as camera_frames,
            ):
                while True:
                    item = self._queue.get()
                    try:
                        if item is None:
                            return
                        if isinstance(item, _ActionItem):
                            actions.write(_json_line(item.value))
                            continue
                        states.write(_json_line(item.state))
                        for role, frame_index, payload, metadata in item.frames:
                            relative = Path("frames") / role / f"{frame_index:08d}.jpg"
                            (self.root / relative).write_bytes(payload)
                            metadata["relative_path"] = relative.as_posix()
                            camera_frames.write(_json_line(metadata))
                    finally:
                        self._queue.task_done()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.report()
        self._closed = True
        try:
            self._queue.put(None, timeout=2.0)
        except queue.Full:
            with self._lock:
                self._errors.append("capture queue did not accept shutdown sentinel")
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            with self._lock:
                self._errors.append("capture writer did not stop within 30 seconds")
        report = self.report()
        temporary = self.root / "capture_report.json.tmp"
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.root / "capture_report.json")
        return report

    def report(self) -> dict[str, Any]:
        with self._lock:
            rates = {}
            for role, count in self._frame_counts.items():
                first = self._first_frame_ns.get(role)
                last = self._last_frame_ns.get(role)
                span_s = 0.0 if first is None or last is None else (last - first) / 1e9
                rates[role] = (
                    0.0 if count < 2 or span_s <= 0.0 else (count - 1) / span_s
                )
            return {
                "schema_version": CAPTURE_SCHEMA,
                "observation_count": self._observation_count,
                "backend_action_count": self._action_count,
                "camera_frame_count": dict(self._frame_counts),
                "camera_effective_hz": rates,
                "queue_drop_count": self._queue_drops,
                "errors": list(self._errors),
                "complete": (
                    not self._thread.is_alive()
                    and self._queue_drops == 0
                    and not self._errors
                ),
            }
