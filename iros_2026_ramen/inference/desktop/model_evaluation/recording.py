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

from data.flip_table_data_augmentation.teleop.desktop_preview import (
    DesktopPreviewProcess,
    environment_flag,
)


CAPTURE_ENV = "IROS_REAL_EVAL_CAPTURE_DIR"
PREVIEW_ENV = "IROS_REAL_EVAL_DESKTOP_PREVIEW"
CAPTURE_SCHEMA = "team_ramen_real_policy_capture/v1"
CAMERA_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")
STATE_CAPTURE_HZ = 30.0
STATE_CAPTURE_PERIOD_NS = int(1.0e9 / STATE_CAPTURE_HZ)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True)
class _ObservationItem:
    state: dict[str, Any] | None
    frames: tuple[tuple[str, int, bytes, dict[str, Any]], ...]


@dataclass(frozen=True)
class _ActionItem:
    value: dict[str, Any]


class RealEvaluationRecorder:
    """Write state, requested targets, and unique JPEG generations off-thread."""

    def __init__(
        self,
        root: Path | None,
        *,
        queue_capacity: int = 1024,
        preview: Any | None = None,
        capture_initially_active: bool = True,
    ) -> None:
        if queue_capacity < 32:
            raise ValueError("evaluation capture queue must hold at least 32 records")
        self.root = None if root is None else root.expanduser().resolve()
        self._capture_enabled = self.root is not None
        self._frames_root: Path | None = None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._frames_root = self.root / "frames"
            for role in CAMERA_ROLES:
                (self._frames_root / role).mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[_ObservationItem | _ActionItem | None] | None = (
            queue.Queue(maxsize=queue_capacity) if self._capture_enabled else None
        )
        self._lock = threading.Lock()
        self._closed = False
        self._recording_active = bool(
            self._capture_enabled and capture_initially_active
        )
        self._recording_started = self._recording_active
        self._recording_started_ns = (
            time.monotonic_ns() if self._recording_active else None
        )
        self._recording_stopped_ns: int | None = None
        self._errors: list[str] = []
        self._preview_errors: list[str] = []
        self._preview = preview
        self._queue_drops = 0
        self._observation_count = 0
        self._action_count = 0
        self._frame_counts = {role: 0 for role in CAMERA_ROLES}
        self._last_generation: dict[str, int] = {}
        self._last_state_capture_ns: int | None = None
        self._first_frame_ns: dict[str, int] = {}
        self._last_frame_ns: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        if self._capture_enabled:
            self._thread = threading.Thread(
                target=self._writer,
                name="real-evaluation-recorder",
                daemon=False,
            )
            self._thread.start()

    @classmethod
    def from_environment(cls) -> RealEvaluationRecorder | None:
        capture_root = os.environ.get(CAPTURE_ENV)
        preview_enabled = environment_flag(PREVIEW_ENV, default=False)
        if not capture_root and not preview_enabled:
            return None
        # The launcher creates the monitor/capture adapter as soon as the
        # command starts.  Preview is live immediately, while the physical
        # policy runner explicitly opens the single capture interval only
        # after the final operator gate and safety checks have passed.
        recorder = cls(
            None if not capture_root else Path(capture_root),
            capture_initially_active=False,
        )
        if preview_enabled:
            try:
                recorder._preview = DesktopPreviewProcess(
                    window_title="IROS 2026 RAMEN - Real Policy Evaluation"
                )
            except Exception as exc:  # noqa: BLE001
                # A missing/broken GUI must never prevent a sealed policy
                # runner from completing its normal safety checks.
                recorder._preview_errors.append(f"{type(exc).__name__}: {exc}")
        return recorder

    def start_capture(self) -> bool:
        """Start the one policy-inference recording interval.

        Returns ``False`` when file capture is disabled (preview-only mode).
        A recorder deliberately supports one interval: silently appending a
        later policy attempt to the same run would make state/action/video
        timing ambiguous.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("evaluation recorder is already closed")
            if not self._capture_enabled:
                return False
            if self._recording_active:
                return False
            if self._recording_started:
                raise RuntimeError(
                    "evaluation capture interval has already been completed"
                )
            self._recording_active = True
            self._recording_started = True
            self._recording_started_ns = time.monotonic_ns()
            self._recording_stopped_ns = None
            return True

    def stop_capture(self) -> bool:
        """Close the policy-inference interval while keeping preview alive."""

        with self._lock:
            if not self._recording_active:
                return False
            self._recording_active = False
            self._recording_stopped_ns = time.monotonic_ns()
            return True

    def record_observation(
        self,
        observation: Any,
        *,
        preview_status: str = "LIVE CAMERA",
    ) -> None:
        if self._closed:
            return
        if self._preview is not None:
            try:
                camera_jpeg = {
                    role: bytes(observation.camera_jpeg[role])
                    for role in CAMERA_ROLES
                }
                body_position = list(observation.body_joint_position_rad)
                if len(body_position) != 29:
                    raise ValueError("live G1 body state must be 29-D")
                self._preview.submit(
                    camera_jpeg,
                    body_position[15:29],
                    preview_status,
                )
            except Exception as exc:  # noqa: BLE001
                # The monitor is diagnostic-only.  Never make a GUI issue
                # alter robot control, capture completeness, or run outcome.
                with self._lock:
                    if not self._preview_errors:
                        self._preview_errors.append(f"{type(exc).__name__}: {exc}")
        capture_ns = int(observation.capture_monotonic_ns)
        frames: list[tuple[str, int, bytes, dict[str, Any]]] = []
        with self._lock:
            if not self._recording_active:
                return
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
            record_state = (
                self._last_state_capture_ns is None
                or capture_ns - self._last_state_capture_ns
                >= STATE_CAPTURE_PERIOD_NS
            )
            if record_state:
                self._last_state_capture_ns = capture_ns
                self._observation_count += 1
        state = None
        if record_state:
            state = {
                "schema_version": CAPTURE_SCHEMA,
                "type": "state",
                "sequence": int(observation.sequence),
                "capture_monotonic_ns": capture_ns,
                "body_joint_position_rad": list(observation.body_joint_position_rad),
                "body_joint_velocity_rad_s": list(
                    observation.body_joint_velocity_rad_s
                ),
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
        if state is None and not frames:
            return
        self._put(_ObservationItem(state=state, frames=tuple(frames)))

    def record_action(self, target: Any) -> None:
        if self._closed:
            return
        value = dict(target.to_message())
        value["recorded_monotonic_ns"] = time.monotonic_ns()
        value["lower_body_command_dimensions"] = 0
        with self._lock:
            if not self._recording_active:
                return
            self._action_count += 1
        self._put(_ActionItem(value=value))

    def _put(self, item: _ObservationItem | _ActionItem) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._queue_drops += 1

    def _writer(self) -> None:
        if self.root is None or self._queue is None:
            return
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
                        if item.state is not None:
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
        self.stop_capture()
        self._closed = True
        if self._preview is not None:
            try:
                self._preview.close()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._preview_errors.append(f"{type(exc).__name__}: {exc}")
        if self._queue is not None and self._thread is not None:
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
        if self.root is not None:
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
                "capture_enabled": self._capture_enabled,
                "recording_active": self._recording_active,
                "recording_started": self._recording_started,
                "recording_started_monotonic_ns": self._recording_started_ns,
                "recording_stopped_monotonic_ns": self._recording_stopped_ns,
                "recording_duration_s": (
                    None
                    if self._recording_started_ns is None
                    or self._recording_stopped_ns is None
                    else (
                        self._recording_stopped_ns
                        - self._recording_started_ns
                    )
                    / 1.0e9
                ),
                "observation_count": self._observation_count,
                "backend_action_count": self._action_count,
                "camera_frame_count": dict(self._frame_counts),
                "camera_effective_hz": rates,
                "queue_drop_count": self._queue_drops,
                "errors": list(self._errors),
                "desktop_preview_errors": list(self._preview_errors),
                "complete": (
                    (self._thread is None or not self._thread.is_alive())
                    and self._queue_drops == 0
                    and not self._errors
                ),
            }
