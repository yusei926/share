"""Materialize a fixed-30-Hz real episode from Orin-local camera MCAP."""

from __future__ import annotations

from concurrent.futures import Future
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from PIL import Image

from ..contracts import ArmHandTarget, ControlEvent, ControlMode, TeleopObservation
from ..raw_episode import (
    POLICY_CAMERA_ROLE_TO_KEY,
    RAW_EPISODE_SCHEMA_VERSION,
    EpisodeIdentity,
    validate_camera_jpeg,
)
from ..shared.policy_contract import (
    ACTION_DIM,
    ACTION_ORDER,
    STATE_DIM,
    STATE_ORDER,
)
from .lossless_camera import (
    CAMERA_CONTROL_PORT,
    ClockMapping,
    RecorderControlClient,
    read_camera_mcap,
)


LOSSLESS_REAL_CAPTURE_SCHEMA_VERSION = "team_ramen_lossless_real_capture/v1"
MAXIMUM_CAMERA_MATCH_NS = 20_000_000
HARD_MAXIMUM_CAMERA_MATCH_NS = 33_333_334
MINIMUM_VALID_FRACTION = 0.995


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_eye(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, subsampling=0)
    return stream.getvalue()


def _split_head_stereo(payload: bytes) -> tuple[bytes, bytes]:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        if rgb.size != (1280, 480):
            raise ValueError(
                f"head stereo frame must be 1280x480, got {rgb.size}"
            )
        return (
            _encode_eye(rgb.crop((0, 0, 640, 480))),
            _encode_eye(rgb.crop((640, 0, 1280, 480))),
        )


def _nearest_unique_matches(
    anchor_times: np.ndarray,
    candidate_times: np.ndarray,
    *,
    maximum_delta_ns: int = MAXIMUM_CAMERA_MATCH_NS,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match ordered camera streams without reusing a source frame."""

    if anchor_times.ndim != 1 or candidate_times.ndim != 1:
        raise ValueError("camera timestamp arrays must be one-dimensional")
    matches = np.full(len(anchor_times), -1, dtype=np.int64)
    errors = np.full(len(anchor_times), np.inf, dtype=np.float64)
    next_candidate = 0
    for anchor_index, anchor in enumerate(anchor_times):
        if next_candidate >= len(candidate_times):
            break
        insertion = int(
            np.searchsorted(candidate_times, anchor, side="left")
        )
        insertion = max(insertion, next_candidate)
        choices = {
            index
            for index in (insertion - 1, insertion)
            if next_candidate <= index < len(candidate_times)
        }
        if not choices:
            continue
        selected = min(
            choices,
            key=lambda index: (
                abs(int(candidate_times[index]) - int(anchor)),
                index,
            ),
        )
        delta = abs(int(candidate_times[selected]) - int(anchor))
        if delta <= maximum_delta_ns:
            matches[anchor_index] = selected
            errors[anchor_index] = float(delta)
            next_candidate = selected + 1
    return matches, errors


def _load_numeric_rows(path: Path) -> list[dict[str, Any]]:
    vector_widths = {
        "body_joint_position_rad": 29,
        "body_joint_velocity_rad_s": 29,
        "dex1_opening_state": 2,
        "applied_arm_target_rad": 14,
        "applied_dex1_opening_target": 2,
        "root_pose_xyzw": 7,
        "commanded_arm_target_rad": 14,
        "commanded_arm_feedforward_torque_nm": 14,
        "commanded_dex1_opening_target": 2,
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid numeric trace at line {line_number}"
                ) from exc
            if (
                not isinstance(row, dict)
                or int(row.get("sequence", -1)) <= 0
                or int(row.get("orin_estimated_ns", -1)) <= 0
            ):
                raise ValueError("numeric trace row is invalid")
            for field, width in vector_widths.items():
                value = np.asarray(row.get(field), dtype=np.float64)
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError(
                        f"numeric trace {field} must be finite [{width}]"
                    )
            try:
                row["control_mode"] = ControlMode(
                    row.get("control_mode", ControlMode.TRACK.value)
                ).value
                row["control_event"] = ControlEvent(
                    row.get("control_event", ControlEvent.NONE.value)
                ).value
            except ValueError as exc:
                raise ValueError(
                    "numeric trace control mode/event is invalid"
                ) from exc
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("lossless real capture needs at least two numeric rows")
    times = [int(row["orin_estimated_ns"]) for row in rows]
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("numeric trace times must be strictly increasing")
    return rows


def _interpolated_numeric(
    rows: list[dict[str, Any]], timestamp_ns: int
) -> dict[str, Any]:
    times = np.asarray([int(row["orin_estimated_ns"]) for row in rows])
    upper = int(np.searchsorted(times, timestamp_ns, side="left"))
    upper = min(max(upper, 1), len(rows) - 1)
    lower = upper - 1
    t0 = int(times[lower])
    t1 = int(times[upper])
    alpha = 0.0 if t1 == t0 else (timestamp_ns - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    def interpolate(name: str) -> list[float]:
        left = np.asarray(rows[lower][name], dtype=np.float64)
        right = np.asarray(rows[upper][name], dtype=np.float64)
        result = left + alpha * (right - left)
        if not np.isfinite(result).all():
            raise ValueError(f"interpolated {name} contains NaN or Inf")
        return result.tolist()

    command_index = max(
        0, int(np.searchsorted(times, timestamp_ns, side="right")) - 1
    )
    command = rows[command_index]
    return {
        "body_joint_position_rad": interpolate("body_joint_position_rad"),
        "body_joint_velocity_rad_s": interpolate(
            "body_joint_velocity_rad_s"
        ),
        "dex1_opening_state": interpolate("dex1_opening_state"),
        "applied_arm_target_rad": interpolate("applied_arm_target_rad"),
        "applied_dex1_opening_target": interpolate(
            "applied_dex1_opening_target"
        ),
        "root_pose_xyzw": interpolate("root_pose_xyzw"),
        "command_sequence": int(command["sequence"]),
        "command_monotonic_ns": int(command["desktop_monotonic_ns"]),
        "commanded_arm_target_rad": list(command["commanded_arm_target_rad"]),
        "commanded_arm_feedforward_torque_nm": list(
            command["commanded_arm_feedforward_torque_nm"]
        ),
        "commanded_dex1_opening_target": list(
            command["commanded_dex1_opening_target"]
        ),
        "control_mode": str(command["control_mode"]),
        "control_event": str(command["control_event"]),
    }


def materialize_lossless_real_episode(
    pending_root: str | Path,
    *,
    identity: EpisodeIdentity,
    episode_id: str,
    diagnostics: Mapping[str, object],
    operator_success: bool,
) -> Path:
    root = Path(pending_root).expanduser().resolve()
    numeric_rows = _load_numeric_rows(root / "numeric.jsonl")
    streams = read_camera_mcap(
        root / "cameras.mcap",
        require_contiguous_sequences=False,
    )
    head = streams["head_stereo"]
    left = streams["left_wrist"]
    right = streams["right_wrist"]

    numeric_start = int(numeric_rows[0]["orin_estimated_ns"])
    numeric_stop = int(numeric_rows[-1]["orin_estimated_ns"])
    head = [
        frame
        for frame in head
        if numeric_start <= frame.orin_capture_monotonic_ns <= numeric_stop
    ]
    if len(head) < 2:
        raise ValueError("camera capture does not overlap the numeric trace")
    head_times = np.asarray(
        [frame.orin_capture_monotonic_ns for frame in head], dtype=np.int64
    )
    left_times = np.asarray(
        [frame.orin_capture_monotonic_ns for frame in left], dtype=np.int64
    )
    right_times = np.asarray(
        [frame.orin_capture_monotonic_ns for frame in right], dtype=np.int64
    )
    left_match, left_error = _nearest_unique_matches(head_times, left_times)
    right_match, right_error = _nearest_unique_matches(head_times, right_times)
    valid = (left_match >= 0) & (right_match >= 0)
    valid_fraction = float(np.mean(valid))
    invalid_runs: list[int] = []
    run = 0
    for is_valid in valid:
        if is_valid:
            if run:
                invalid_runs.append(run)
            run = 0
        else:
            run += 1
    if run:
        invalid_runs.append(run)
    maximum_invalid_run = max(invalid_runs, default=0)

    temporary = root / "materialized"
    temporary.mkdir()
    for role in POLICY_CAMERA_ROLE_TO_KEY:
        (temporary / "policy_cameras" / role).mkdir(parents=True)
    (temporary / "diagnostics" / "head_right").mkdir(parents=True)
    trace = (temporary / "frames.jsonl").open("x", encoding="utf-8")
    last_left = left[0]
    last_right = right[0]
    maximum_match_ns = 0
    try:
        for index, head_frame in enumerate(head):
            if left_match[index] >= 0:
                last_left = left[int(left_match[index])]
                maximum_match_ns = max(
                    maximum_match_ns, int(left_error[index])
                )
            if right_match[index] >= 0:
                last_right = right[int(right_match[index])]
                maximum_match_ns = max(
                    maximum_match_ns, int(right_error[index])
                )
            head_left, head_right = _split_head_stereo(head_frame.jpeg)
            camera_payloads = {
                "head_left": head_left,
                "head_right": head_right,
                "left_wrist": last_left.jpeg,
                "right_wrist": last_right.jpeg,
            }
            for role, payload in camera_payloads.items():
                validate_camera_jpeg(payload, role)
            numeric = _interpolated_numeric(
                numeric_rows, head_frame.orin_capture_monotonic_ns
            )
            policy_cameras: dict[str, dict[str, object]] = {}
            for role, key in POLICY_CAMERA_ROLE_TO_KEY.items():
                payload = camera_payloads[role]
                relative = (
                    Path("policy_cameras") / role / f"{index:06d}.jpg"
                )
                (temporary / relative).write_bytes(payload)
                source = (
                    head_frame
                    if role == "head_left"
                    else last_left
                    if role == "left_wrist"
                    else last_right
                )
                role_valid = (
                    True
                    if role == "head_left"
                    else bool(left_match[index] >= 0)
                    if role == "left_wrist"
                    else bool(right_match[index] >= 0)
                )
                policy_cameras[key] = {
                    "path": relative.as_posix(),
                    "sha256": _sha256_bytes(payload),
                    "capture_monotonic_ns": source.orin_capture_monotonic_ns,
                    "source_sequence": source.source_sequence,
                    "valid": role_valid,
                }
            right_relative = (
                Path("diagnostics") / "head_right" / f"{index:06d}.jpg"
            )
            (temporary / right_relative).write_bytes(head_right)
            record = {
                "frame_index": index,
                "canonical_timestamp_s": index / 30.0,
                "observation_sequence": index + 1,
                "observation_monotonic_ns": (
                    head_frame.orin_capture_monotonic_ns
                ),
                "camera_bundle_valid": bool(valid[index]),
                "camera_valid": {
                    "head_left": True,
                    "head_right": True,
                    "left_wrist": bool(left_match[index] >= 0),
                    "right_wrist": bool(right_match[index] >= 0),
                },
                "camera_skew_ms": max(
                    0.0
                    if not np.isfinite(left_error[index])
                    else left_error[index] / 1.0e6,
                    0.0
                    if not np.isfinite(right_error[index])
                    else right_error[index] / 1.0e6,
                ),
                "camera_stream_metadata": {
                    "head_left": head_frame.header(),
                    "head_right": head_frame.header(),
                    "left_wrist": last_left.header(),
                    "right_wrist": last_right.header(),
                },
                **numeric,
                "policy_cameras": policy_cameras,
                "diagnostics": {
                    "cameras": {
                        "head_right": {
                            "path": right_relative.as_posix(),
                            "sha256": _sha256_bytes(head_right),
                            "capture_monotonic_ns": (
                                head_frame.orin_capture_monotonic_ns
                            ),
                            "source_sequence": head_frame.source_sequence,
                            "valid": True,
                        }
                    },
                    "real": {
                        "timestamp_source": "orin_capture_monotonic",
                        "placeholder_images_are_training_invalid": True,
                    },
                },
            }
            trace.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )
        trace.flush()
        os.fsync(trace.fileno())
    finally:
        trace.close()

    rates = {}
    sequence_gaps = {}
    device_counter_gaps = {}
    for role, frames in streams.items():
        first = frames[0].orin_capture_monotonic_ns
        last = frames[-1].orin_capture_monotonic_ns
        rates[role] = (
            0.0
            if len(frames) < 2 or last <= first
            else (len(frames) - 1) / ((last - first) / 1.0e9)
        )
        sequence_gaps[role] = sum(
            second.source_sequence != first_sequence.source_sequence + 1
            for first_sequence, second in zip(frames, frames[1:])
        )
        device_counter_gaps[role] = sum(
            later.device_frame_counter != earlier.device_frame_counter + 1
            for earlier, later in zip(frames, frames[1:])
            if earlier.device_frame_counter is not None
            and later.device_frame_counter is not None
        )
    rejection_reasons: list[str] = []
    if not operator_success:
        rejection_reasons.append("operator_did_not_confirm_success")
    recorder_status = diagnostics.get("orin_recorder_status")
    if (
        isinstance(recorder_status, Mapping)
        and recorder_status.get("failed") is not None
    ):
        rejection_reasons.append(
            f"orin_recorder_failed:{recorder_status['failed']}"
        )
    clock_uncertainty_ms = diagnostics.get("clock_uncertainty_ms_p95")
    if (
        not isinstance(clock_uncertainty_ms, (int, float))
        or not math.isfinite(float(clock_uncertainty_ms))
        or float(clock_uncertainty_ms) > 2.0
    ):
        rejection_reasons.append(
            f"clock_uncertainty_above_2ms:{clock_uncertainty_ms}"
        )
    if diagnostics.get("clock_sync_error") is not None:
        rejection_reasons.append(
            f"clock_sync_failed:{diagnostics['clock_sync_error']}"
        )
    for role, hz in rates.items():
        if not 29.5 <= hz <= 30.5:
            rejection_reasons.append(f"{role}_source_hz_outside_29_5_30_5:{hz:.3f}")
    if any(sequence_gaps.values()):
        rejection_reasons.append(f"source_sequence_gaps:{sequence_gaps}")
    for role in ("left_wrist", "right_wrist"):
        if device_counter_gaps[role]:
            rejection_reasons.append(
                f"{role}_device_frame_counter_gaps:"
                f"{device_counter_gaps[role]}"
            )
    if valid_fraction < MINIMUM_VALID_FRACTION:
        rejection_reasons.append(
            f"camera_valid_fraction_below_99_5pct:{valid_fraction:.6f}"
        )
    if maximum_invalid_run > 1:
        rejection_reasons.append(
            f"consecutive_invalid_camera_rows_above_one:{maximum_invalid_run}"
        )
    if maximum_match_ns > HARD_MAXIMUM_CAMERA_MATCH_NS:
        rejection_reasons.append(
            f"camera_match_error_above_33_3ms:{maximum_match_ns / 1.0e6:.3f}"
        )
    accepted = not rejection_reasons
    manifest = {
        "schema_version": RAW_EPISODE_SCHEMA_VERSION,
        "capture_schema_version": LOSSLESS_REAL_CAPTURE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "backend": "real",
        "dr_profile": "real",
        "seed": identity.seed,
        "config_sha256": identity.config_sha256,
        "runtime_digest": identity.runtime_digest,
        "fps": 30,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_order": STATE_ORDER,
        "action_order": ACTION_ORDER,
        "frame_count": len(head),
        "success": accepted,
        "collection_disposition": (
            "accepted" if accepted else "rejected_diagnostic"
        ),
        "policy_camera_keys": list(POLICY_CAMERA_ROLE_TO_KEY.values()),
        "operator_only_cameras": ["head_right"],
        "diagnostic_cameras": [],
        "privileged_policy_features": [],
        "camera_frame_contract": {
            "encoding": "JPEG",
            "width": 640,
            "height": 480,
            "canonical_fps": 30,
            "raw_physical_roles": [
                "head_stereo",
                "left_wrist",
                "right_wrist",
            ],
            "maximum_match_ms": MAXIMUM_CAMERA_MATCH_NS / 1.0e6,
            "placeholder_rows_are_training_invalid": True,
        },
        "diagnostics": {
            **dict(diagnostics),
            "recorded_source_hz": rates,
            "source_sequence_gaps": sequence_gaps,
            "device_frame_counter_gaps": device_counter_gaps,
            "camera_valid_fraction": valid_fraction,
            "maximum_consecutive_invalid_rows": maximum_invalid_run,
            "maximum_camera_match_ms": maximum_match_ns / 1.0e6,
            "training_valid_frame_indices": np.flatnonzero(valid).tolist(),
            "rejection_reasons": rejection_reasons,
            "raw_camera_mcap_sha256": _sha256_file(root / "cameras.mcap"),
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # The episode becomes discoverable only after all immutable raw sources
    # are present. A failed copy therefore leaves only the hidden pending
    # directory, never a partially published accepted episode.
    shutil.copy2(root / "cameras.mcap", temporary / "cameras.mcap")
    shutil.copy2(
        root / "numeric.jsonl", temporary / "numeric_source.jsonl"
    )
    destination = root.parent / episode_id
    if not accepted:
        destination = root.parent / "rejected" / episode_id
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    shutil.rmtree(root)
    return destination


class LosslessRealEpisodeWriter:
    """Coordinate Desktop numeric capture with the Orin-local MCAP recorder."""

    def __init__(
        self,
        output_root: str | Path,
        identity: EpisodeIdentity,
        *,
        recorder_host: str,
        ssh_target: str,
        recorder_port: int = CAMERA_CONTROL_PORT,
    ) -> None:
        if identity.backend != "real":
            raise ValueError("lossless physical recording requires backend=real")
        self.identity = identity
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.episode_id = f"{stamp}_real_real_{uuid4().hex[:8]}"
        self.path = self.output_root / f".{self.episode_id}.recording"
        self.path.mkdir()
        self._numeric = (self.path / "numeric.jsonl").open("x", encoding="utf-8")
        self._client = RecorderControlClient(
            recorder_host, port=recorder_port
        )
        self._ssh_target = ssh_target
        try:
            first_mapping = self._client.synchronize()
        except Exception:
            self._numeric.close()
            shutil.rmtree(self.path)
            raise
        self._clock_mappings: list[ClockMapping] = [first_mapping]
        if self._clock_mappings[-1].uncertainty_ns > 2_000_000:
            self._numeric.close()
            shutil.rmtree(self.path)
            raise RuntimeError("Desktop/Orin clock uncertainty exceeds 2 ms")
        try:
            self._client.start(self.episode_id)
        except Exception:
            self._numeric.close()
            shutil.rmtree(self.path)
            raise
        self._clock_lock = threading.Lock()
        self._clock_stop = threading.Event()
        self._clock_error: str | None = None
        self._last_clock_sync = time.monotonic()
        self._clock_thread = threading.Thread(
            target=self._clock_sync_loop,
            name=f"clock-sync-{self.episode_id}",
            daemon=True,
        )
        self._clock_thread.start()
        self._frame_count = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def source_hz(self) -> float:
        return 0.0

    def _clock_sync_loop(self) -> None:
        while not self._clock_stop.wait(1.0):
            try:
                mapping = self._client.synchronize(sample_count=4)
                status = self._client.status()
            except Exception as exc:  # noqa: BLE001
                with self._clock_lock:
                    self._clock_error = f"{type(exc).__name__}: {exc}"
                continue
            with self._clock_lock:
                if (
                    status.get("active") is not True
                    or status.get("session_id") != self.episode_id
                    or status.get("failed") is not None
                ):
                    self._clock_error = (
                        "Orin recorder session lost or failed: "
                        f"{status}"
                    )
                self._clock_mappings.append(mapping)
                self._last_clock_sync = time.monotonic()

    def _stop_clock_sync(self) -> None:
        self._clock_stop.set()
        self._clock_thread.join(timeout=3.0)
        if self._clock_thread.is_alive():
            with self._clock_lock:
                self._clock_error = (
                    "camera recorder clock-sync thread did not stop"
                )

    def append(self, observation: TeleopObservation, target: ArmHandTarget) -> None:
        if self._closed:
            raise RuntimeError("lossless real writer is closed")
        with self._clock_lock:
            mapping = self._clock_mappings[-1]
            mapping_index = len(self._clock_mappings) - 1
        row = {
            "sequence": target.sequence,
            "desktop_monotonic_ns": target.monotonic_ns,
            "orin_estimated_ns": mapping.desktop_to_orin(target.monotonic_ns),
            "clock_mapping_index": mapping_index,
            "body_joint_position_rad": list(observation.body_joint_position_rad),
            "body_joint_velocity_rad_s": list(
                observation.body_joint_velocity_rad_s
            ),
            "dex1_opening_state": list(observation.dex1_opening_fraction),
            "applied_arm_target_rad": list(observation.applied_arm_target_rad),
            "applied_dex1_opening_target": list(
                observation.applied_dex1_opening_target
            ),
            "root_pose_xyzw": list(observation.root_pose_xyzw),
            "commanded_arm_target_rad": list(target.arm_position_rad),
            "commanded_arm_feedforward_torque_nm": list(
                target.arm_feedforward_torque_nm
            ),
            "commanded_dex1_opening_target": list(
                target.dex1_opening_fraction
            ),
            "control_mode": target.mode.value,
            "control_event": target.event.value,
        }
        self._numeric.write(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        )
        self._frame_count += 1

    def save_async(
        self,
        *,
        diagnostics: Mapping[str, object],
        success: bool | None,
    ) -> Future[Path]:
        if self._closed:
            raise RuntimeError("lossless real writer is closed")
        self._closed = True
        self._stop_clock_sync()
        self._numeric.flush()
        os.fsync(self._numeric.fileno())
        self._numeric.close()
        result: Future[Path] = Future()

        def finalize() -> None:
            try:
                remote = self._client.stop()
                remote_path = Path(str(remote["path"]))
                local_mcap = self.path / "cameras.mcap"
                subprocess.run(
                    [
                        "scp",
                        "-q",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=5",
                        f"{self._ssh_target}:{remote_path}",
                        str(local_mcap),
                    ],
                    check=True,
                    timeout=600,
                )
                if _sha256_file(local_mcap) != remote.get("sha256"):
                    raise RuntimeError("transferred camera MCAP digest mismatch")
                with self._clock_lock:
                    clock_mappings = tuple(self._clock_mappings)
                    clock_error = self._clock_error
                clock_uncertainties = [
                    mapping.uncertainty_ns / 1.0e6
                    for mapping in clock_mappings
                ]
                desktop_points = np.asarray(
                    [
                        (
                            sample.desktop_send_ns
                            + sample.desktop_receive_ns
                        )
                        / 2.0
                        for mapping in clock_mappings
                        for sample in mapping.samples
                    ],
                    dtype=np.float64,
                )
                observed_offsets = np.asarray(
                    [
                        sample.desktop_to_orin_offset_ns
                        for mapping in clock_mappings
                        for sample in mapping.samples
                    ],
                    dtype=np.float64,
                )
                drift = (
                    0.0
                    if len(desktop_points) < 2
                    or float(np.ptp(desktop_points)) <= 0.0
                    else float(
                        np.polyfit(
                            desktop_points - desktop_points[0],
                            observed_offsets,
                            1,
                        )[0]
                    )
                )
                path = materialize_lossless_real_episode(
                    self.path,
                    identity=self.identity,
                    episode_id=self.episode_id,
                    diagnostics={
                        **dict(diagnostics),
                        "clock_mappings": [
                            mapping.to_json()
                            for mapping in clock_mappings
                        ],
                        "clock_uncertainty_ms_p95": float(
                            np.percentile(clock_uncertainties, 95)
                        ),
                        "clock_sync_error": clock_error,
                        "clock_drift_ppm": drift * 1.0e6,
                        "orin_recorder_status": remote,
                    },
                    operator_success=success is True,
                )
                result.set_result(path)
            except BaseException as exc:  # noqa: BLE001
                failure = self.path / "finalization_error.json"
                failure.write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                LOSSLESS_REAL_CAPTURE_SCHEMA_VERSION
                            ),
                            "episode_id": self.episode_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result.set_exception(exc)

        threading.Thread(
            target=finalize,
            name=f"finalize-{self.episode_id}",
            daemon=False,
        ).start()
        return result

    def discard(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_clock_sync()
        self._numeric.close()
        try:
            remote = self._client.stop()
        except Exception as exc:
            remote = {"error": f"{type(exc).__name__}: {exc}"}
        (self.path / "discarded.json").write_text(
            json.dumps(remote, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def discard_async(self) -> Future[None]:
        """Stop and retain a diagnostic capture without blocking servo updates."""

        result: Future[None] = Future()

        def discard() -> None:
            try:
                self.discard()
            except BaseException as exc:  # noqa: BLE001
                result.set_exception(exc)
            else:
                result.set_result(None)

        threading.Thread(
            target=discard,
            name=f"discard-{self.episode_id}",
            daemon=False,
        ).start()
        return result


def start_lossless_real_episode_async(
    output_root: str | Path,
    identity: EpisodeIdentity,
    *,
    recorder_host: str,
    ssh_target: str,
    recorder_port: int = CAMERA_CONTROL_PORT,
) -> Future[LosslessRealEpisodeWriter]:
    """Start clock synchronization and recorder ACK outside the servo loop."""

    result: Future[LosslessRealEpisodeWriter] = Future()

    def start() -> None:
        try:
            writer = LosslessRealEpisodeWriter(
                output_root,
                identity,
                recorder_host=recorder_host,
                ssh_target=ssh_target,
                recorder_port=recorder_port,
            )
        except BaseException as exc:  # noqa: BLE001
            result.set_exception(exc)
        else:
            result.set_result(writer)

    threading.Thread(
        target=start,
        name="start-lossless-real-recording",
        daemon=False,
    ).start()
    return result
