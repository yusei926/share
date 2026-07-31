#!/usr/bin/env python3
"""Run one backend-neutral AVP teleoperation session."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import threading
import time
from contextlib import nullcontext

import numpy as np

from .backend import TeleopBackend, TransientObservationError
from .config import DEFAULT_TELEOP_CONFIG_PATH, load_teleop_config
from .contracts import ArmHandTarget, ControlEvent, ControlMode, TeleopObservation
from .keyboard import FootPedalReader, KeyReader, load_foot_pedal_binding
from .operator_process import OperatorProcess, operator_camera_roles
from .raw_episode import (
    EpisodeIdentity,
    ReplayTrajectoryWriter,
)
from .real.lossless_episode import (
    LosslessRealEpisodeWriter,
    start_lossless_real_episode_async,
)
from .shared.state_machine import (
    TrackingAnchorRequest,
    hold_target_from_observation as _hold_target_from_observation,
)


# The first simulator camera bundle can take noticeably longer than a normal
# 30 Hz sample: Isaac Sim may compile the first RTX render after the TCP
# handshake. This does not relax the in-session watchdog below; it only keeps
# startup deterministic on a cold RTX workstation.
INITIAL_OBSERVATION_TIMEOUT_S = float(
    os.environ.get("FLIP_TABLE_TELEOP_INITIAL_OBSERVATION_TIMEOUT_S", "90.0")
)
WARMUP_OBSERVATION_TIMEOUT_S = float(
    os.environ.get("FLIP_TABLE_TELEOP_WARMUP_OBSERVATION_TIMEOUT_S", "30.0")
)
WARMUP_OBSERVATION_COUNT = 2
OBSERVATION_TIMEOUT_S = 2.0
HAND_TRACKING_REPORT_MIN_HZ = 25.0


def _live_preview_rate_gate_hz() -> tuple[float, float]:
    """Return the configured AVP cadence and its sustained-rate acceptance gate.

    The live display intentionally uses a 24 Hz latest-frame stream for the
    full Isaac scene; the four-camera training episode is rendered separately
    at 30 Hz after collection.  Permit a small scheduler tolerance, but never
    reinterpret a requested 30 Hz stream as passing below its 28 Hz contract.
    """

    raw = os.environ.get("FLIP_TABLE_TELEOP_PREVIEW_HZ", "24")
    try:
        preview_hz = float(raw)
    except ValueError as exc:
        raise ValueError("FLIP_TABLE_TELEOP_PREVIEW_HZ must be a number in [5,30]") from exc
    if not np.isfinite(preview_hz) or not 5.0 <= preview_hz <= 30.0:
        raise ValueError("FLIP_TABLE_TELEOP_PREVIEW_HZ must be in [5,30]")
    return preview_hz, min(28.0, preview_hz * 0.92)


@dataclass(frozen=True)
class ObservationSnapshot:
    observation: TeleopObservation
    generation: int


@dataclass(frozen=True)
class ObservationStreamStats:
    received_count: int
    receive_times: tuple[float, ...]
    capture_times: tuple[float, ...]
    sequences: tuple[int, ...]
    missing_sequences: int


@dataclass(frozen=True)
class ObservationStreamHealth:
    last_receive_age_s: float
    transient_error: str | None


def _camera_alert_roles(
    role_ages_ms: dict[str, float],
    *,
    hold_timeout_s: float,
    outage_timeout_s: float,
) -> tuple[list[str], list[str], list[str]]:
    """Classify independent physical-camera liveness by role."""

    normalized = {
        role: (
            float(age_ms)
            if np.isfinite(age_ms) and float(age_ms) >= 0.0
            else float("inf")
        )
        for role, age_ms in role_ages_ms.items()
    }
    warning_roles = sorted(
        role for role, age_ms in normalized.items() if age_ms > 100.0
    )
    hold_roles = sorted(
        role
        for role, age_ms in normalized.items()
        if age_ms > hold_timeout_s * 1000.0
    )
    outage_roles = sorted(
        role
        for role, age_ms in normalized.items()
        if age_ms > outage_timeout_s * 1000.0
    )
    return warning_roles, hold_roles, outage_roles


def _audit_finalized_lossless_episode(
    audit: dict[str, object], episode_path: Path
) -> None:
    manifest = json.loads(
        (episode_path / "manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = manifest.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ValueError("finalized lossless episode lacks diagnostics")
    entries = audit.setdefault("lossless_recordings", [])
    if not isinstance(entries, list):
        raise RuntimeError("lossless recording audit has an invalid type")
    entries.append(
        {
            "episode_id": manifest.get("episode_id"),
            "path": str(episode_path),
            "collection_disposition": manifest.get(
                "collection_disposition"
            ),
            "frame_count": manifest.get("frame_count"),
            "canonical_fps": manifest.get("fps"),
            "recorded_source_hz": diagnostics.get("recorded_source_hz"),
            "source_sequence_gaps": diagnostics.get(
                "source_sequence_gaps"
            ),
            "device_frame_counter_gaps": diagnostics.get(
                "device_frame_counter_gaps"
            ),
            "camera_valid_fraction": diagnostics.get(
                "camera_valid_fraction"
            ),
            "maximum_camera_match_ms": diagnostics.get(
                "maximum_camera_match_ms"
            ),
            "clock_uncertainty_ms_p95": diagnostics.get(
                "clock_uncertainty_ms_p95"
            ),
            "rejection_reasons": diagnostics.get("rejection_reasons"),
        }
    )


class ObservationStream:
    """Receive cameras asynchronously so they cannot throttle AVP commands."""

    def __init__(
        self,
        backend: TeleopBackend,
        initial_observation: TeleopObservation,
        *,
        max_hz: float | None = None,
    ) -> None:
        if max_hz is not None and max_hz <= 0.0:
            raise ValueError("observation max_hz must be positive when provided")
        self._backend = backend
        self._period_s = None if max_hz is None else 1.0 / max_hz
        self._condition = threading.Condition()
        self._snapshot = ObservationSnapshot(initial_observation, 0)
        initial_receive_time = time.monotonic()
        self._receive_times: deque[float] = deque(
            (initial_receive_time,), maxlen=301
        )
        self._capture_times: deque[float] = deque(
            (_remote_capture_seconds(initial_observation),), maxlen=301
        )
        self._sequences: deque[int] = deque(
            (initial_observation.sequence,), maxlen=301
        )
        self._received_count = 1
        self._missing_sequences = 0
        self._error: BaseException | None = None
        self._transient_error: str | None = None
        self._last_receive_time = initial_receive_time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("observation stream is already running")
        self._thread = threading.Thread(
            target=self._receive,
            name="flip-table-teleop-observations",
            daemon=True,
        )
        self._thread.start()

    def _receive(self) -> None:
        warmup_remaining = WARMUP_OBSERVATION_COUNT
        next_receive_time = time.monotonic()
        try:
            while not self._stop.is_set():
                timeout_s = (
                    WARMUP_OBSERVATION_TIMEOUT_S
                    if warmup_remaining > 0
                    else OBSERVATION_TIMEOUT_S
                )
                try:
                    observation = self._backend.observe(timeout_s=timeout_s)
                except TransientObservationError as exc:
                    with self._condition:
                        self._transient_error = str(exc)
                        self._condition.notify_all()
                    self._stop.wait(0.05)
                    continue
                receive_time = time.monotonic()
                capture_time = _remote_capture_seconds(observation)
                warmup_remaining -= 1
                if self._stop.is_set():
                    return
                with self._condition:
                    previous_sequence = self._sequences[-1]
                    if observation.sequence <= previous_sequence:
                        self._receive_times.clear()
                        self._capture_times.clear()
                        self._sequences.clear()
                    else:
                        self._missing_sequences += max(
                            0, observation.sequence - previous_sequence - 1
                        )
                    self._receive_times.append(receive_time)
                    self._capture_times.append(capture_time)
                    self._sequences.append(observation.sequence)
                    self._received_count += 1
                    self._last_receive_time = receive_time
                    self._transient_error = None
                    self._snapshot = ObservationSnapshot(
                        observation,
                        self._snapshot.generation + 1,
                    )
                    self._condition.notify_all()
                if self._period_s is not None:
                    next_receive_time += self._period_s
                    wait_s = next_receive_time - time.monotonic()
                    if wait_s > 0.0:
                        self._stop.wait(wait_s)
                    else:
                        # A slow receive must not accumulate an unbounded
                        # schedule debt or cause an immediate busy-loop.
                        next_receive_time = time.monotonic()
        except BaseException as exc:  # noqa: BLE001
            if not self._stop.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

    def latest(self) -> ObservationSnapshot:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"teleoperation observation stream failed: {self._error}")
            return self._snapshot

    def stats(self) -> ObservationStreamStats:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"teleoperation observation stream failed: {self._error}")
            return ObservationStreamStats(
                received_count=self._received_count,
                receive_times=tuple(self._receive_times),
                capture_times=tuple(self._capture_times),
                sequences=tuple(self._sequences),
                missing_sequences=self._missing_sequences,
            )

    def health(self, *, now: float | None = None) -> ObservationStreamHealth:
        """Return freshness without turning a recoverable camera gap fatal."""

        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"teleoperation observation stream failed: {self._error}")
            checked = time.monotonic() if now is None else now
            return ObservationStreamHealth(
                last_receive_age_s=max(0.0, checked - self._last_receive_time),
                transient_error=self._transient_error,
            )

    def reset_stats(self) -> None:
        """Start a fresh measurement window from the latest received sample."""

        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"teleoperation observation stream failed: {self._error}")
            observation = self._snapshot.observation
            self._receive_times.clear()
            self._capture_times.clear()
            self._sequences.clear()
            self._receive_times.append(time.monotonic())
            self._capture_times.append(_remote_capture_seconds(observation))
            self._sequences.append(observation.sequence)
            self._received_count = 1
            self._missing_sequences = 0

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.request_stop()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=OBSERVATION_TIMEOUT_S + 0.5)


def parse_args(
    argv: list[str] | None = None,
    *,
    forced_backend: str | None = None,
) -> argparse.Namespace:
    descriptions = {
        None: __doc__,
        "real": "Run physical G1 + Dex1 Apple Vision Pro teleoperation.",
        "sim": "Run Isaac/RoboFinals Apple Vision Pro teleoperation.",
    }
    parser = argparse.ArgumentParser(description=descriptions[forced_backend])
    if forced_backend is None:
        parser.add_argument("backend", choices=("sim", "real"))
    elif forced_backend not in {"sim", "real"}:
        raise ValueError(f"unsupported forced backend: {forced_backend!r}")
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--xr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--foot-pedal-config",
        type=Path,
        default=None,
        help=(
            "calibrated PCsensor foot-pedal mapping. When set, the pedal is "
            "exclusive to this teleop session and emits r/s/q only."
        ),
    )
    parser.add_argument(
        "--session-report",
        type=Path,
        default=None,
        help="write structured AVP/operator acceptance evidence as JSON",
    )
    if forced_backend != "real":
        parser.add_argument("--sim-host", default="127.0.0.1")
        parser.add_argument("--sim-port", type=int, default=None)
        parser.add_argument(
            "--dr-profile",
            choices=("mild", "medium", "full"),
            default="full",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--transport-probe",
            action="store_true",
            help="verify one non-actuating simulator camera packet, then exit",
        )
        parser.add_argument(
            "--control-probe",
            action="store_true",
            help=(
                "verify small upper-body and Dex1 motion through the simulator bridge"
            ),
        )
    if forced_backend != "sim":
        parser.add_argument("--dds-interface")
        parser.add_argument("--image-server-ip")
    args = parser.parse_args(argv)
    if forced_backend is not None:
        args.backend = forced_backend
    if forced_backend == "real":
        args.sim_host = None
        args.sim_port = None
        args.dr_profile = "real"
        args.seed = 42
        args.transport_probe = False
        args.control_probe = False
    elif forced_backend == "sim":
        args.dds_interface = None
        args.image_server_ip = None
    return args


def _decode_jpeg(payload: bytes) -> np.ndarray:
    import cv2

    value = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None or value.shape != (480, 640, 3):
        raise ValueError("camera JPEG does not decode to 640x480 RGB")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _remote_capture_seconds(observation: TeleopObservation) -> float:
    """Return the simulator capture clock retained during clock normalization."""

    timing = observation.diagnostics.get("transport_timing")
    if not isinstance(timing, dict):
        return observation.capture_monotonic_ns / 1.0e9
    value = timing.get("remote_observation_monotonic_ns")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("simulator transport timing omits its remote capture clock")
    return value / 1.0e9


def _make_backend(args: argparse.Namespace, config) -> TeleopBackend:
    if args.backend == "sim":
        from .sim.backend import SimSocketBackend

        return SimSocketBackend(
            args.sim_host,
            args.sim_port or config.workstation.sim_port,
            config,
        )
    from .real.backend import RealDdsBackend

    if not args.dds_interface or not args.image_server_ip:
        raise ValueError("real backend requires --dds-interface and --image-server-ip")
    return RealDdsBackend(args.dds_interface, args.image_server_ip, config)


def _idle_target(sequence: int, event: ControlEvent = ControlEvent.NONE) -> ArmHandTarget:
    """Build a no-motion upper-body command for the simulator bridge."""

    return ArmHandTarget(
        sequence=sequence,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.IDLE,
        event=event,
        arm_position_rad=(0.0,) * 14,
        dex1_opening_fraction=(1.0, 1.0),
    )


def _save_probe_result(
    args: argparse.Namespace,
    config,
    probe_name: str,
    result: dict[str, object],
) -> Path:
    output_root = getattr(args, "output_root", None) or (
        Path(config.collection.output_root) / "raw"
    )
    probe_root = output_root.parent / "probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    path = probe_root / (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{probe_name}_"
        f"{getattr(args, 'dr_profile', 'unknown')}_{getattr(args, 'seed', 0)}.json"
    )
    payload = {
        "schema_version": "team_ramen_flip_table_teleop_probe/v1",
        "probe": probe_name,
        "backend": args.backend,
        "dr_profile": getattr(args, "dr_profile", "unknown"),
        "seed": getattr(args, "seed", 0),
        "config_sha256": config.digest,
        **result,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sample_rate(samples: deque[float]) -> float:
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1] - samples[0]
    return 0.0 if elapsed <= 0.0 else (len(samples) - 1) / elapsed


def _observation_stream_rates(stats: ObservationStreamStats) -> tuple[float, float]:
    """Return remote-source and local-transport rates for the live stream."""

    transport_hz = _sample_rate(deque(stats.receive_times))
    if len(stats.capture_times) < 2:
        return 0.0, transport_hz
    capture_elapsed = stats.capture_times[-1] - stats.capture_times[0]
    if capture_elapsed <= 0.0:
        return 0.0, transport_hz
    source_hz = (stats.sequences[-1] - stats.sequences[0]) / capture_elapsed
    return float(source_hz), transport_hz


def _finite_span(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if lower.shape != upper.shape or not (
        np.isfinite(lower).all() and np.isfinite(upper).all()
    ):
        return np.zeros(lower.shape, dtype=np.float64)
    return np.maximum(upper - lower, 0.0)


def _quantile_or_none(values: deque[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _max_or_none(values: deque[float]) -> float | None:
    return None if not values else float(np.max(np.asarray(values, dtype=np.float64)))


def _assembled_table_mass_kg(randomization: object) -> float | None:
    if not isinstance(randomization, dict):
        return None
    explicit = randomization.get("assembled_table_mass_kg")
    if isinstance(explicit, (int, float)) and np.isfinite(explicit):
        return float(explicit)
    samples = randomization.get("table_part_masses")
    if not isinstance(samples, list):
        return None
    total = 0.0
    found = False
    for sample in samples:
        masses = sample.get("body_mass_kg") if isinstance(sample, dict) else None
        if not isinstance(masses, list):
            continue
        for mass in masses:
            if not isinstance(mass, (int, float)) or not np.isfinite(mass):
                return None
            total += float(mass)
            found = True
    return total if found else None


def _write_session_report(
    path: Path,
    *,
    args: argparse.Namespace,
    config,
    audit: dict[str, object],
    termination_reason: str,
    error: str | None,
) -> None:
    """Persist evidence from one physical AVP/operator acceptance session."""

    camera_receive_times = audit["camera_receive_times"]
    camera_capture_times = audit["camera_capture_times"]
    camera_sequences = audit["camera_sequences"]
    track_command_times = audit["track_command_times"]
    hand_tracking_hz = audit["hand_tracking_hz"]
    latency_ms = audit["command_observation_latency_ms"]
    arm_errors = audit["arm_tracking_errors_rad"]
    hand_errors = audit["hand_tracking_errors"]
    assert isinstance(camera_receive_times, deque)
    assert isinstance(camera_capture_times, deque)
    assert isinstance(camera_sequences, deque)
    assert isinstance(track_command_times, deque)
    assert isinstance(hand_tracking_hz, deque)
    assert isinstance(latency_ms, deque)
    assert isinstance(arm_errors, deque)
    assert isinstance(hand_errors, deque)

    camera_transport_hz = _sample_rate(camera_receive_times)
    camera_source_hz = 0.0
    if len(camera_capture_times) > 1:
        capture_elapsed = camera_capture_times[-1] - camera_capture_times[0]
        if capture_elapsed > 0.0:
            camera_source_hz = (
                camera_sequences[-1] - camera_sequences[0]
            ) / capture_elapsed
    role_rates = {
        str(role): float(rate)
        for role, rate in dict(
            audit.get("camera_role_latest_transition_hz", {})
        ).items()
        if isinstance(rate, (int, float)) and np.isfinite(rate)
    }
    if args.backend == "real" and role_rates:
        camera_source_hz = min(role_rates.values())
        bundle_valid_times = audit.get("camera_bundle_valid_times", deque())
        assert isinstance(bundle_valid_times, deque)
        camera_transport_hz = _sample_rate(bundle_valid_times)
    bundle_skews = audit.get("camera_bundle_skew_ms", deque())
    assert isinstance(bundle_skews, deque)
    bundle_skew_p50 = _quantile_or_none(bundle_skews, 0.5)
    bundle_skew_p95 = _quantile_or_none(bundle_skews, 0.95)
    bundle_skew_max = _max_or_none(bundle_skews)
    command_hz = _sample_rate(track_command_times)
    hand_hz_median = _quantile_or_none(hand_tracking_hz, 0.5)
    latency_median = _quantile_or_none(latency_ms, 0.5)
    latency_p95 = _quantile_or_none(latency_ms, 0.95)
    arm_error_p95 = _quantile_or_none(arm_errors, 0.95)
    hand_error_p95 = _quantile_or_none(hand_errors, 0.95)
    arm_error_max = _max_or_none(arm_errors)
    hand_error_max = _max_or_none(hand_errors)

    commanded_arm_span = _finite_span(
        audit["commanded_arm_min"], audit["commanded_arm_max"]
    )
    observed_arm_span = _finite_span(
        audit["observed_arm_min"], audit["observed_arm_max"]
    )
    commanded_hand_span = _finite_span(
        audit["commanded_hand_min"], audit["commanded_hand_max"]
    )
    observed_hand_span = _finite_span(
        audit["observed_hand_min"], audit["observed_hand_max"]
    )
    observed_waist_span = _finite_span(
        audit.get("observed_waist_min", np.full(3, np.inf, dtype=np.float64)),
        audit.get("observed_waist_max", np.full(3, -np.inf, dtype=np.float64)),
    )
    received = int(audit["camera_received_count"])
    missing = int(audit["missing_camera_sequences"])
    drop_fraction = missing / max(1, received + missing)
    sender_drops = int(audit["sim_sender_drops"])
    arm_command_side_span = (
        float(np.max(commanded_arm_span[:7])),
        float(np.max(commanded_arm_span[7:])),
    )
    arm_observed_side_span = (
        float(np.max(observed_arm_span[:7])),
        float(np.max(observed_arm_span[7:])),
    )
    sim_randomization = audit.get("sim_randomization", {})
    contact_materials = (
        sim_randomization.get("contact_materials")
        if isinstance(sim_randomization, dict)
        else None
    )
    gripper_contact_max = np.asarray(
        audit.get("gripper_contact_force_max_n", (0.0, 0.0)), dtype=np.float64
    )
    dex1_drive_max = np.asarray(
        audit.get("dex1_drive_force_max_n", (0.0, 0.0)), dtype=np.float64
    )
    grasp_force_audit = audit.get("dex1_grasp_force_audit", {})
    if not isinstance(grasp_force_audit, dict):
        grasp_force_audit = {}
    sustained_contact_s = np.asarray(
        grasp_force_audit.get("sustained_contact_max_s_left_right", (0.0, 0.0)),
        dtype=np.float64,
    )
    sustained_load_s = np.asarray(
        grasp_force_audit.get("sustained_drive_load_max_s_left_right", (0.0, 0.0)),
        dtype=np.float64,
    )
    if sustained_contact_s.shape != (2,) or sustained_load_s.shape != (2,):
        sustained_contact_s = np.zeros(2, dtype=np.float64)
        sustained_load_s = np.zeros(2, dtype=np.float64)
    sustained_grasp_verified = (
        (sustained_contact_s >= 0.20) & (sustained_load_s >= 0.20)
    )
    dds_write_count = np.asarray(
        audit.get("dds_write_count_arm_left_right", (0, 0, 0)), dtype=np.int64
    )
    dds_write_failure_count = np.asarray(
        audit.get("dds_write_failure_count_arm_left_right", (0, 0, 0)),
        dtype=np.int64,
    )
    feedback_limit_seen = np.asarray(
        audit.get("dex1_feedback_limit_seen_left_right", (False, False)),
        dtype=bool,
    )
    dex1_state_stale_seen = np.asarray(
        audit.get("dex1_state_stale_seen_left_right", (False, False)),
        dtype=bool,
    )
    camera_outage_events = [
        dict(event)
        for event in audit.get("camera_outage_events", [])
        if isinstance(event, dict)
    ]
    longest_outage_by_role_s: dict[str, float] = {}
    for event in camera_outage_events:
        duration = event.get("duration_s")
        roles = event.get("roles")
        if not isinstance(duration, (int, float)) or not isinstance(roles, list):
            continue
        for role in roles:
            if isinstance(role, str):
                longest_outage_by_role_s[role] = max(
                    longest_outage_by_role_s.get(role, 0.0),
                    float(duration),
                )

    checks = {
        "avp_connected": bool(audit["avp_connected"]),
        "tracking_reanchored_after_reset": int(audit["tracking_enabled_count"])
        >= 2,
        "no_stale_tracking_stop": int(audit["stale_stop_count"]) == 0,
        # Vision Pro hand events are consistently observed around 26--28 Hz.
        # This is a report-only quality gate; command liveness is governed by
        # the monotonic last-valid-event age, not by this aggregate rate.
        "hand_tracking_hz_gte_25": hand_hz_median is not None
        and hand_hz_median >= HAND_TRACKING_REPORT_MIN_HZ,
        "track_command_hz_gte_28": command_hz >= 28.0,
        "camera_source_hz_gte_28": camera_source_hz >= 28.0,
        "camera_transport_hz_gte_28": camera_transport_hz >= 28.0,
        "camera_drop_fraction_lte_5pct": drop_fraction <= 0.05,
        "sim_sender_drop_fraction_lte_1pct": sender_drops
        <= max(1, received // 100),
        "left_arm_commanded": arm_command_side_span[0] >= 0.05,
        "right_arm_commanded": arm_command_side_span[1] >= 0.05,
        "left_arm_observed": arm_observed_side_span[0] >= 0.03,
        "right_arm_observed": arm_observed_side_span[1] >= 0.03,
        "left_dex1_commanded_continuously": bool(
            commanded_hand_span[0] >= 0.25
        ),
        "right_dex1_commanded_continuously": bool(
            commanded_hand_span[1] >= 0.25
        ),
        "left_dex1_observed": bool(observed_hand_span[0] >= 0.20),
        "right_dex1_observed": bool(observed_hand_span[1] >= 0.20),
        "command_observation_latency_p95_lte_150ms": latency_p95 is not None
        and latency_p95 <= 150.0,
        "arm_tracking_error_p95_lte_0_12rad": arm_error_p95 is not None
        and arm_error_p95 <= 0.12,
        "arm_tracking_error_max_lte_0_20rad": arm_error_max is not None
        and arm_error_max <= 0.20,
        "dex1_tracking_error_p95_lte_0_12": hand_error_p95 is not None
        and hand_error_p95 <= 0.12,
        "dex1_tracking_error_max_lte_0_20": hand_error_max is not None
        and hand_error_max <= 0.20,
        "recording_exercised": int(audit["record_started_count"]) > 0
        and (
            int(audit["record_saved_count"]) > 0
            or int(audit["record_discarded_count"]) > 0
        ),
        "reset_exercised": int(audit["reset_count"]) > 0,
        "safe_quit_exercised": bool(audit["safe_quit_requested"]),
        "no_arm_safety_interlock": int(audit.get("arm_interlock_count", 0)) == 0,
        "no_dds_write_failure": bool(np.all(dds_write_failure_count == 0)),
        "no_dex1_feedback_outage": not bool(np.any(dex1_state_stale_seen)),
        "no_camera_outage": int(audit.get("camera_outage_count", 0)) == 0,
        "no_camera_safety_hold": int(audit.get("camera_hold_count", 0)) == 0,
        "no_lossless_recording_failure": int(
            audit.get("record_finalization_failure_count", 0)
        )
        == 0,
        "no_runtime_error": error is None,
    }
    payload = {
        "schema_version": "team_ramen_flip_table_avp_acceptance/v1",
        "backend": args.backend,
        "dr_profile": args.dr_profile if args.backend == "sim" else "real",
        "seed": args.seed,
        "config_sha256": config.digest,
        "started_at_utc": audit["started_at_utc"],
        "duration_s": round(time.monotonic() - float(audit["started_monotonic"]), 3),
        "termination_reason": termination_reason,
        "error": error,
        "rates": {
            # Preview transport is latest-only by design. Lossless physical
            # source rates are reported separately per finalized recording.
            "camera_source_hz": round(camera_source_hz, 3),
            "camera_transport_hz": round(camera_transport_hz, 3),
            "preview_transition_hz_by_role": role_rates,
            "preview_bundle_hz": round(camera_transport_hz, 3),
            "track_command_hz": round(command_hz, 3),
            "hand_tracking_hz_median": (
                None if hand_hz_median is None else round(hand_hz_median, 3)
            ),
        },
        "transport": {
            "camera_frames_received": received,
            "valid_hand_events": int(audit["hand_event_count_max"]),
            "invalid_hand_events": int(audit.get("hand_invalid_event_count_max", 0)),
            "hand_event_rejections": {
                "missing_bilateral_pose": int(
                    audit.get("hand_missing_pose_count_max", 0)
                ),
                "invalid_wrist_matrix": int(
                    audit.get("hand_invalid_wrist_count_max", 0)
                ),
                "invalid_pinch_state": int(
                    audit.get("hand_invalid_pinch_count_max", 0)
                ),
                "invalid_unused_skeleton_diagnostic": int(
                    audit.get(
                        "hand_invalid_unused_skeleton_count_max", 0
                    )
                ),
                "by_side_and_reason": dict(
                    audit.get("hand_invalid_details_max", {})
                ),
            },
            "missing_camera_sequences": missing,
            "camera_drop_fraction": round(drop_fraction, 6),
            "sim_sender_drops": sender_drops,
            "camera_outage_count": int(audit.get("camera_outage_count", 0)),
            "camera_recovery_count": int(
                audit.get("camera_recovery_count", 0)
            ),
            "camera_hold_count": int(audit.get("camera_hold_count", 0)),
            "camera_outage_events": camera_outage_events,
            "camera_longest_outage_by_role_s": longest_outage_by_role_s,
            "camera_roles": {
                "transition_hz": role_rates,
                "maximum_age_ms": dict(
                    audit.get("camera_role_max_age_ms", {})
                ),
                "latest_generation": dict(
                    audit.get("camera_role_latest_generation", {})
                ),
                "latest_metadata": dict(
                    audit.get("camera_role_latest_metadata", {})
                ),
            },
            "camera_bundles": {
                "valid_count": int(
                    audit.get("camera_bundle_valid_count", 0)
                ),
                "invalid_count": int(
                    audit.get("camera_bundle_invalid_count", 0)
                ),
                "skew_ms_p50": (
                    None
                    if bundle_skew_p50 is None
                    else round(bundle_skew_p50, 3)
                ),
                "skew_ms_p95": (
                    None
                    if bundle_skew_p95 is None
                    else round(bundle_skew_p95, 3)
                ),
                "skew_ms_max": (
                    None
                    if bundle_skew_max is None
                    else round(bundle_skew_max, 3)
                ),
            },
            "image_network": {
                "interface": audit.get("image_network_interface"),
                "start": dict(
                    audit.get("image_network_counters_start", {})
                ),
                "end": dict(
                    audit.get("image_network_counters_current", {})
                ),
                "delta": dict(
                    audit.get("image_network_counter_delta", {})
                ),
            },
            "command_observation_latency_ms_median": (
                None if latency_median is None else round(latency_median, 3)
            ),
            "command_observation_latency_ms_p95": (
                None if latency_p95 is None else round(latency_p95, 3)
            ),
        },
        "tracking": {
            "tracking_enabled_count": int(audit["tracking_enabled_count"]),
            "stale_stop_count": int(audit["stale_stop_count"]),
            "track_command_count": int(audit["track_command_count"]),
            "commanded_arm_span_rad": commanded_arm_span.round(6).tolist(),
            "observed_arm_span_rad": observed_arm_span.round(6).tolist(),
            "commanded_dex1_opening_span": commanded_hand_span.round(6).tolist(),
            "observed_dex1_opening_span": observed_hand_span.round(6).tolist(),
            "observed_waist_span_rad": observed_waist_span.round(6).tolist(),
            "waist_guard_tripped": bool(audit.get("waist_guard_tripped", False)),
            "arm_interlock_count": int(audit.get("arm_interlock_count", 0)),
            "arm_interlock_reasons": list(audit.get("arm_interlock_reasons", [])),
            "arm_tracking_error_p95_rad": (
                None if arm_error_p95 is None else round(arm_error_p95, 6)
            ),
            "arm_tracking_error_max_rad": (
                None if arm_error_max is None else round(arm_error_max, 6)
            ),
            "dex1_tracking_error_p95": (
                None if hand_error_p95 is None else round(hand_error_p95, 6)
            ),
            "dex1_tracking_error_max": (
                None if hand_error_max is None else round(hand_error_max, 6)
            ),
            "dds_write_count_arm_left_right": dds_write_count.tolist(),
            "dds_write_failure_count_arm_left_right": (
                dds_write_failure_count.tolist()
            ),
            "dex1_feedback_limit_seen_left_right": feedback_limit_seen.tolist(),
            "dex1_state_stale_seen_left_right": dex1_state_stale_seen.tolist(),
            "lower_body_peak_speed_rad_s": round(
                float(audit.get("lower_body_peak_speed_rad_s", 0.0)), 6
            ),
            "lower_body_policy_command_dimensions": 0,
        },
        "sim_physics": {
            "assembled_table_mass_kg": _assembled_table_mass_kg(
                sim_randomization
            ),
            "dex1_effort_limit_n_per_finger": 20.0,
            "dex1_drive_force_max_n_left_right": dex1_drive_max.round(4).tolist(),
            "gripper_contact_force_max_n_left_right": gripper_contact_max.round(4).tolist(),
            "sustained_grasp_verified_left_right": sustained_grasp_verified.tolist(),
            "grasp_force_audit": grasp_force_audit,
            "contact_materials": contact_materials,
        },
        "operator_controls": {
            "record_started_count": int(audit["record_started_count"]),
            "record_saved_count": int(audit["record_saved_count"]),
            "record_discarded_count": int(audit["record_discarded_count"]),
            "record_sync_drop_count": int(audit.get("record_sync_drop_count", 0)),
            "record_finalization_failure_count": int(
                audit.get("record_finalization_failure_count", 0)
            ),
            "reset_count": int(audit["reset_count"]),
            "safe_quit_requested": bool(audit["safe_quit_requested"]),
        },
        "lossless_recordings": [
            dict(entry)
            for entry in audit.get("lossless_recordings", [])
            if isinstance(entry, dict)
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"AVP acceptance report: {path} (passed={payload['passed']})",
        flush=True,
    )


def _run_transport_probe(args: argparse.Namespace, config) -> int:
    """Validate a single synchronized sim packet without opening AVP control."""

    if args.backend != "sim":
        raise ValueError("--transport-probe is only supported for the sim backend")
    frame_count = int(os.environ.get("FLIP_TABLE_TELEOP_PROBE_FRAMES", "1"))
    if frame_count < 1 or frame_count > 300:
        raise ValueError("FLIP_TABLE_TELEOP_PROBE_FRAMES must be in [1,300]")
    backend = _make_backend(args, config)
    try:
        backend.apply(_idle_target(0))
        received_at = []
        captured_at = []
        sequences = []
        decoded = {}
        observation = None
        for index in range(frame_count):
            observation = backend.observe(
                timeout_s=(INITIAL_OBSERVATION_TIMEOUT_S if index == 0 else 5.0)
            )
            decoded = {
                role: _decode_jpeg(payload)
                for role, payload in observation.camera_jpeg.items()
            }
            if set(decoded) != {"head_left", "head_right"}:
                raise RuntimeError(
                    f"unexpected simulator camera roles: {sorted(decoded)}"
                )
            received_at.append(time.monotonic())
            captured_at.append(_remote_capture_seconds(observation))
            sequences.append(observation.sequence)
        assert observation is not None
        measured_hz = None
        source_hz = None
        if len(received_at) > 1:
            measured_hz = (len(received_at) - 1) / (received_at[-1] - received_at[0])
            source_hz = (sequences[-1] - sequences[0]) / (
                captured_at[-1] - captured_at[0]
            )
        missing_sequences = sum(
            max(0, current - previous - 1)
            for previous, current in zip(sequences, sequences[1:])
        )
        preview_hz, minimum_preview_hz = _live_preview_rate_gate_hz()
        capture_intervals_s = np.diff(np.asarray(captured_at, dtype=np.float64))
        result = {
            "sequence": observation.sequence,
            "camera_roles": sorted(decoded),
            "camera_shapes": [
                list(shape) for shape in sorted({tuple(image.shape) for image in decoded.values()})
            ],
            "frames": frame_count,
            "source_hz": None if source_hz is None else round(source_hz, 3),
            "transport_hz": None if measured_hz is None else round(measured_hz, 3),
            "requested_preview_hz": preview_hz,
            "minimum_preview_hz": round(minimum_preview_hz, 3),
            "capture_interval_ms_p95": (
                None
                if capture_intervals_s.size == 0
                else round(float(np.quantile(capture_intervals_s, 0.95) * 1000.0), 3)
            ),
            "capture_interval_ms_max": (
                None
                if capture_intervals_s.size == 0
                else round(float(np.max(capture_intervals_s) * 1000.0), 3)
            ),
            "missing_sequences": missing_sequences,
            "sim_physics": {
                "assembled_table_mass_kg": _assembled_table_mass_kg(
                    observation.diagnostics.get("randomization")
                ),
                "dex1_effort_limit_n_per_finger": 20.0,
                "dex1_drive_force_n": observation.diagnostics.get(
                    "dex1_drive_force_n"
                ),
                "gripper_contact_force_n": observation.diagnostics.get(
                    "gripper_contact_force_n"
                ),
                "contact_materials": (
                    observation.diagnostics.get("randomization", {}).get(
                        "contact_materials"
                    )
                    if isinstance(
                        observation.diagnostics.get("randomization"), dict
                    )
                    else None
                ),
            },
        }
        report_path = _save_probe_result(args, config, "transport", result)
        print(f"Simulator transport probe passed: {result}, report={report_path}", flush=True)
        if frame_count >= 30:
            if source_hz is None or source_hz < minimum_preview_hz:
                raise RuntimeError(
                    "simulator camera source rate is too low: "
                    f"{source_hz!r} Hz (need >= {minimum_preview_hz:.2f} Hz)"
                )
            if measured_hz is None or measured_hz < minimum_preview_hz:
                raise RuntimeError(
                    "simulator camera transport rate is too low: "
                    f"{measured_hz!r} Hz (need >= {minimum_preview_hz:.2f} Hz)"
                )
        backend.apply(_idle_target(1, ControlEvent.QUIT))
        return 0
    finally:
        backend.close()


def _control_probe_target(
    initial_arm: np.ndarray,
    step: int,
    step_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a collision-conservative bilateral wrist/Dex1 probe target."""

    arm = np.asarray(initial_arm, dtype=np.float64).copy()
    if arm.shape != (14,) or not np.isfinite(arm).all():
        raise ValueError("control-probe initial arm state must be finite [14]")
    if step_count < 2 or not 0 <= step < step_count:
        raise ValueError("control-probe step is outside its trajectory")
    phase = 2.0 * np.pi * step / (step_count - 1)
    arm[4] += 0.12 * np.sin(phase)
    arm[11] -= 0.12 * np.sin(phase)
    hand = np.asarray(
        (0.5 + 0.5 * np.cos(phase), 0.5 - 0.5 * np.cos(phase)),
        dtype=np.float64,
    )
    return arm, hand


def _probe_recording_toggle(
    args: argparse.Namespace,
    config,
    backend: TeleopBackend,
    observation_stream: ObservationStream,
    sequence: int,
) -> tuple[int, dict[str, object]]:
    """Verify that toggling collection does not degrade the live stereo path.

    Simulator collection saves only the command trajectory during AVP control.
    The synchronized four-camera 30 Hz package is rendered offline after the
    session, so requiring wrists in this live socket would reintroduce the
    rendering load that the latest-frame display is designed to avoid.
    """

    del args, config
    expected_roles = {"head_left", "head_right"}
    latest = observation_stream.latest()
    sequence += 1
    backend.apply(
        ArmHandTarget(
            sequence=sequence,
            monotonic_ns=time.monotonic_ns(),
            mode=ControlMode.IDLE,
            event=ControlEvent.RECORD_TOGGLE,
            arm_position_rad=latest.observation.arm_joint_position_rad,
            dex1_opening_fraction=latest.observation.dex1_opening_fraction,
        )
    )
    try:
        deadline = time.monotonic() + 15.0
        live_observation = None
        generation = latest.generation
        while time.monotonic() < deadline:
            snapshot = observation_stream.latest()
            if (
                snapshot.generation > generation
                and set(snapshot.observation.camera_jpeg) == expected_roles
            ):
                live_observation = snapshot.observation
                break
            time.sleep(0.02)
        if live_observation is None:
            raise RuntimeError("recording toggle did not preserve head stereo")

        observation_stream.reset_stats()
        measurement_s = float(
            os.environ.get("FLIP_TABLE_TELEOP_RECORDING_PROBE_SECONDS", "3.0")
        )
        if not np.isfinite(measurement_s) or not 2.0 <= measurement_s <= 10.0:
            raise ValueError(
                "FLIP_TABLE_TELEOP_RECORDING_PROBE_SECONDS must be in [2,10]"
            )
        time.sleep(measurement_s)
        recording_stats = observation_stream.stats()
        if len(recording_stats.receive_times) < 2:
            raise RuntimeError("recording toggle received too few live stereo frames")
        capture_elapsed = (
            recording_stats.capture_times[-1] - recording_stats.capture_times[0]
        )
        receive_elapsed = (
            recording_stats.receive_times[-1] - recording_stats.receive_times[0]
        )
        if capture_elapsed <= 0.0 or receive_elapsed <= 0.0:
            raise RuntimeError("recording probe timestamps did not advance")
        recording_source_hz = (
            recording_stats.sequences[-1] - recording_stats.sequences[0]
        ) / capture_elapsed
        recording_transport_hz = (
            len(recording_stats.receive_times) - 1
        ) / receive_elapsed

        return sequence, {
            "measurement_s": measurement_s,
            "observations": len(recording_stats.receive_times),
            "source_hz": round(float(recording_source_hz), 3),
            "transport_hz": round(float(recording_transport_hz), 3),
            "missing_sequences": recording_stats.missing_sequences,
            "roles": sorted(expected_roles),
            "offline_four_camera_render": True,
        }
    finally:
        latest = observation_stream.latest().observation
        sequence += 1
        backend.apply(
            ArmHandTarget(
                sequence=sequence,
                monotonic_ns=time.monotonic_ns(),
                mode=ControlMode.IDLE,
                event=ControlEvent.RECORD_TOGGLE,
                arm_position_rad=latest.arm_joint_position_rad,
                dex1_opening_fraction=latest.dex1_opening_fraction,
            )
        )


def _run_control_probe(args: argparse.Namespace, config) -> int:
    """Exercise the real teleop command path without requiring an AVP session."""

    if args.backend != "sim":
        raise ValueError("--control-probe is only supported for the sim backend")
    duration_s = float(os.environ.get("FLIP_TABLE_TELEOP_CONTROL_PROBE_SECONDS", "8.0"))
    warmup_s = float(
        os.environ.get("FLIP_TABLE_TELEOP_CONTROL_PROBE_WARMUP_SECONDS", "2.0")
    )
    if not np.isfinite(duration_s) or duration_s < 2.0 or duration_s > 30.0:
        raise ValueError("FLIP_TABLE_TELEOP_CONTROL_PROBE_SECONDS must be in [2,30]")
    if not np.isfinite(warmup_s) or warmup_s < 0.5 or warmup_s > 10.0:
        raise ValueError(
            "FLIP_TABLE_TELEOP_CONTROL_PROBE_WARMUP_SECONDS must be in [0.5,10]"
        )
    step_count = max(2, int(round(duration_s * config.rates.command_hz)))
    period_s = 1.0 / config.rates.command_hz
    backend = _make_backend(args, config)
    cleanup_errors: list[str] = []
    observation_stream: ObservationStream | None = None
    sequence = 0
    latest = None
    try:
        backend.apply(_idle_target(sequence))
        initial = backend.observe(timeout_s=INITIAL_OBSERVATION_TIMEOUT_S)
        latest = initial
        observation_stream = ObservationStream(
            backend,
            initial,
            max_hz=(
                config.rates.camera_poll_hz
                if args.backend == "real"
                else config.rates.camera_hz
            ),
        )
        observation_stream.start()
        initial_generation = observation_stream.latest().generation
        time.sleep(warmup_s)
        warmed = observation_stream.latest()
        if warmed.generation <= initial_generation:
            raise RuntimeError("control probe received no camera frame during warm-up")
        initial = warmed.observation
        latest = initial
        observation_stream.reset_stats()
        initial_arm = np.asarray(initial.arm_joint_position_rad, dtype=np.float64)
        observed_arm = [initial_arm.copy()]
        observed_body = [
            np.asarray(initial.body_joint_position_rad, dtype=np.float64)
        ]
        observed_hand = [np.asarray(initial.dex1_opening_fraction, dtype=np.float64)]
        observed_root = [np.asarray(initial.root_pose_xyzw, dtype=np.float64)]
        tracking_arm_errors: list[np.ndarray] = []
        tracking_hand_errors: list[np.ndarray] = []
        last_generation = -1
        command_times: list[float] = []
        next_command_time = time.monotonic()
        for step in range(step_count):
            arm, hand = _control_probe_target(initial_arm, step, step_count)
            sequence += 1
            backend.apply(
                ArmHandTarget(
                    sequence=sequence,
                    monotonic_ns=time.monotonic_ns(),
                    mode=ControlMode.TRACK,
                    event=ControlEvent.NONE,
                    arm_position_rad=tuple(arm),
                    dex1_opening_fraction=tuple(hand),
                )
            )
            command_times.append(time.monotonic())
            snapshot = observation_stream.latest()
            latest = snapshot.observation
            if snapshot.generation != last_generation:
                last_generation = snapshot.generation
                observed_arm.append(np.asarray(latest.arm_joint_position_rad))
                observed_body.append(
                    np.asarray(latest.body_joint_position_rad, dtype=np.float64)
                )
                observed_hand.append(np.asarray(latest.dex1_opening_fraction))
                observed_root.append(
                    np.asarray(latest.root_pose_xyzw, dtype=np.float64)
                )
                applied_sequence = latest.diagnostics.get(
                    "last_applied_command_sequence", -1
                )
                if isinstance(applied_sequence, int) and applied_sequence > 0:
                    tracking_arm_errors.append(
                        np.abs(
                            np.asarray(latest.applied_arm_target_rad)
                            - np.asarray(latest.arm_joint_position_rad)
                        )
                    )
                    tracking_hand_errors.append(
                        np.abs(
                            np.asarray(latest.applied_dex1_opening_target)
                            - np.asarray(latest.dex1_opening_fraction)
                        )
                    )
            next_command_time += period_s
            remaining_s = next_command_time - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            else:
                next_command_time = time.monotonic()

        time.sleep(0.25)
        snapshot = observation_stream.latest()
        latest = snapshot.observation
        observed_arm.append(np.asarray(latest.arm_joint_position_rad))
        observed_body.append(
            np.asarray(latest.body_joint_position_rad, dtype=np.float64)
        )
        observed_hand.append(np.asarray(latest.dex1_opening_fraction))
        observed_root.append(np.asarray(latest.root_pose_xyzw, dtype=np.float64))
        arm_values = np.stack(observed_arm)
        body_values = np.stack(observed_body)
        hand_values = np.stack(observed_hand)
        root_values = np.stack(observed_root)
        if not tracking_arm_errors or not tracking_hand_errors:
            raise RuntimeError("control probe received no applied TRACK observations")
        command_hz = (len(command_times) - 1) / (
            command_times[-1] - command_times[0]
        )
        arm_span = np.ptp(arm_values, axis=0)
        lower_body_span = np.ptp(body_values[:, :15], axis=0)
        root_xy_drift = np.linalg.norm(
            root_values[:, :2] - root_values[0, :2], axis=1
        )
        root_height_error = np.abs(root_values[:, 2] - 0.74)
        root_quat = root_values[:, 3:7]
        root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)
        qx, qy, qz, qw = root_quat.T
        root_roll = np.arctan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        root_pitch = np.arcsin(
            np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        )
        hand_span = np.ptp(hand_values, axis=0)
        arm_errors = np.stack(tracking_arm_errors)
        hand_errors = np.stack(tracking_hand_errors)
        stream_stats = observation_stream.stats()
        if len(stream_stats.receive_times) < 2:
            raise RuntimeError("control probe received too few camera frames")
        observation_source_hz = (
            (stream_stats.sequences[-1] - stream_stats.sequences[0])
            / (stream_stats.capture_times[-1] - stream_stats.capture_times[0])
        )
        observation_transport_hz = (
            (len(stream_stats.receive_times) - 1)
            / (stream_stats.receive_times[-1] - stream_stats.receive_times[0])
        )
        missing_observation_sequences = stream_stats.missing_sequences
        sequence, recording_toggle = _probe_recording_toggle(
            args,
            config,
            backend,
            observation_stream,
            sequence,
        )
        output = {
            "command_hz": round(float(command_hz), 3),
            "warmup_s": warmup_s,
            "observations": int(len(observed_arm)),
            "observation_source_hz": round(float(observation_source_hz), 3),
            "observation_transport_hz": round(float(observation_transport_hz), 3),
            "missing_observation_sequences": missing_observation_sequences,
            "recording_toggle": recording_toggle,
            "lower_body_span_max_rad": round(float(np.max(lower_body_span)), 4),
            "lower_body_span_rad": [
                round(float(value), 4) for value in lower_body_span
            ],
            "root_xy_drift_max_m": round(float(root_xy_drift.max()), 4),
            "root_height_error_max_m": round(float(root_height_error.max()), 4),
            "root_abs_roll_max_deg": round(float(np.degrees(np.abs(root_roll).max())), 3),
            "root_abs_pitch_max_deg": round(float(np.degrees(np.abs(root_pitch).max())), 3),
            "arm_span_max_rad": round(float(np.max(arm_span)), 4),
            "left_wrist_roll_span_rad": round(float(arm_span[4]), 4),
            "right_wrist_roll_span_rad": round(float(arm_span[11]), 4),
            "arm_span_rad": [round(float(value), 4) for value in arm_span],
            "dex1_opening_span": [round(float(value), 4) for value in hand_span],
            "arm_tracking_error_p95_rad": round(
                float(np.quantile(arm_errors, 0.95)), 4
            ),
            "arm_tracking_error_max_rad": round(float(np.max(arm_errors)), 4),
            "dex1_tracking_error_p95": round(
                float(np.quantile(hand_errors, 0.95)), 4
            ),
            "dex1_tracking_error_max": round(float(np.max(hand_errors)), 4),
        }
        report_path = _save_probe_result(args, config, "control", output)
        print(f"Simulator control probe: {output}, report={report_path}", flush=True)
        if command_hz < 28.0:
            raise RuntimeError(f"control probe command rate is too low: {command_hz:.2f} Hz")
        _preview_hz, minimum_preview_hz = _live_preview_rate_gate_hz()
        if observation_source_hz < minimum_preview_hz:
            raise RuntimeError(
                "control probe camera source rate is too low: "
                f"{observation_source_hz:.2f} Hz (need >= {minimum_preview_hz:.2f} Hz)"
            )
        if observation_transport_hz < minimum_preview_hz:
            raise RuntimeError(
                "control probe camera transport rate is too low: "
                f"{observation_transport_hz:.2f} Hz (need >= {minimum_preview_hz:.2f} Hz)"
            )
        recording_source_hz = float(recording_toggle["source_hz"])
        recording_transport_hz = float(recording_toggle["transport_hz"])
        if (
            recording_source_hz < minimum_preview_hz
            or recording_transport_hz < minimum_preview_hz
        ):
            raise RuntimeError(
                "control probe live-stereo rate after recording toggle is too low: "
                f"source={recording_source_hz:.2f} Hz, "
                f"transport={recording_transport_hz:.2f} Hz"
            )
        if not np.isfinite(lower_body_span).all():
            raise RuntimeError("WBC lower-body state became non-finite")
        if (
            root_xy_drift.max() > 0.20
            or root_height_error.max() > 0.08
            or np.abs(root_roll).max() > np.radians(15.0)
            or np.abs(root_pitch).max() > np.radians(15.0)
        ):
            raise RuntimeError("balanced WBC failed the root stability gate")
        if arm_span[4] < 0.04 or arm_span[11] < 0.04:
            raise RuntimeError("simulator arms did not follow the bilateral wrist probe")
        non_probe_arm_span = np.delete(arm_span, (4, 11))
        if np.max(non_probe_arm_span) > 0.08:
            raise RuntimeError(
                "non-commanded arm joint moved excessively during the control probe"
            )
        if np.any(hand_span < 0.25):
            raise RuntimeError("simulator Dex1 openings did not follow both pinch probes")
        if np.quantile(arm_errors, 0.95) > 0.08 or np.max(arm_errors) > 0.12:
            raise RuntimeError("simulator arm tracking error exceeds the teleop gate")
        if np.quantile(hand_errors, 0.95) > 0.08 or np.max(hand_errors) > 0.12:
            raise RuntimeError("simulator Dex1 tracking error exceeds the teleop gate")
        return 0
    finally:
        if observation_stream is not None:
            observation_stream.request_stop()
        if latest is not None:
            try:
                sequence += 1
                backend.apply(
                    ArmHandTarget(
                        sequence=sequence,
                        monotonic_ns=time.monotonic_ns(),
                        mode=ControlMode.IDLE,
                        event=ControlEvent.QUIT,
                        arm_position_rad=latest.arm_joint_position_rad,
                        dex1_opening_fraction=latest.dex1_opening_fraction,
                    )
                )
            except Exception:
                pass
        if observation_stream is not None:
            try:
                observation_stream.close()
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"observation stream close failed: {exc}")
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"backend close failed: {exc}")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))


def main(
    argv: list[str] | None = None,
    *,
    forced_backend: str | None = None,
) -> int:
    args = parse_args(argv, forced_backend=forced_backend)
    cleanup_errors: list[str] = []
    config = load_teleop_config(args.config)
    foot_pedal_binding = (
        load_foot_pedal_binding(args.foot_pedal_config)
        if args.foot_pedal_config is not None
        else None
    )
    output_root = args.output_root or Path(config.collection.output_root) / "raw"
    if args.transport_probe and args.control_probe:
        raise ValueError("select only one teleoperation probe")
    if args.transport_probe:
        return _run_transport_probe(args, config)
    if args.control_probe:
        return _run_control_probe(args, config)
    # TeleVuer stays in a separate process. The parent exclusively owns the
    # simulator or real-robot connection, so XR worker startup can never alter
    # its socket descriptors.
    # The real operator view adds wrist diagnostics; simulator preview remains
    # intentionally head-only so Isaac rendering stays lightweight.
    operator_roles = operator_camera_roles(args.backend)
    operator = OperatorProcess(args.xr_root, config, backend=args.backend)
    try:
        operator.start()
    except BaseException:
        operator.close()
        raise
    try:
        backend = _make_backend(args, config)
    except BaseException:
        operator.close()
        raise
    tracking = False
    anchor_request = TrackingAnchorRequest()
    # A pause keeps the robot at a target captured from the backend's actual
    # applied state.  It must not be inferred from the newest AVP target: the
    # operator can freely move their hands while paused.
    paused_hold_arm: np.ndarray | None = None
    paused_hold_hand: np.ndarray | None = None
    paused_hold_torque: np.ndarray | None = None
    # Simulator AVP operates with a low-latency stereo preview.  Its four
    # training cameras are rendered later from this 30 Hz command trajectory,
    # so the live operator loop never accumulates camera/render backlog.
    recording: (
        ReplayTrajectoryWriter
        | LosslessRealEpisodeWriter
        | None
    ) = None
    pending_recording_start: Future[LosslessRealEpisodeWriter] | None = None
    cancel_pending_recording_start = False
    pending_recording_discards: list[Future[None]] = []
    pending_recording_finalizers: list[Future[Path]] = []
    sequence = 0
    pending_event = ControlEvent.NONE
    latest = None
    last_sent_target: ArmHandTarget | None = None
    observation_stream: ObservationStream | None = None
    session_error: str | None = None
    termination_reason = "incomplete"
    rearm_notice_pending = False
    camera_warning_active = False
    camera_hold_active = False
    camera_outage_active = False
    camera_outage_started_monotonic: float | None = None
    audit: dict[str, object] = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_monotonic": time.monotonic(),
        "avp_connected": False,
        "hand_tracking_hz": deque(maxlen=301),
        "hand_event_count_max": 0,
        "hand_invalid_event_count_max": 0,
        "hand_missing_pose_count_max": 0,
        "hand_invalid_wrist_count_max": 0,
        "hand_invalid_pinch_count_max": 0,
        "hand_invalid_unused_skeleton_count_max": 0,
        "hand_invalid_details_max": {},
        "track_command_times": deque(maxlen=301),
        "camera_receive_times": deque(maxlen=61),
        "camera_capture_times": deque(maxlen=61),
        "camera_sequences": deque(maxlen=61),
        "camera_received_count": 0,
        "missing_camera_sequences": 0,
        "sim_sender_drops": 0,
        "command_observation_latency_ms": deque(maxlen=301),
        "arm_tracking_errors_rad": deque(maxlen=4214),
        "hand_tracking_errors": deque(maxlen=602),
        "tracking_enabled_count": 0,
        "stale_stop_count": 0,
        "track_command_count": 0,
        "track_command_sequences": set(),
        "commanded_arm_min": np.full(14, np.inf, dtype=np.float64),
        "commanded_arm_max": np.full(14, -np.inf, dtype=np.float64),
        "observed_arm_min": np.full(14, np.inf, dtype=np.float64),
        "observed_arm_max": np.full(14, -np.inf, dtype=np.float64),
        "commanded_hand_min": np.full(2, np.inf, dtype=np.float64),
        "commanded_hand_max": np.full(2, -np.inf, dtype=np.float64),
        "observed_hand_min": np.full(2, np.inf, dtype=np.float64),
        "observed_hand_max": np.full(2, -np.inf, dtype=np.float64),
        "observed_waist_min": np.full(3, np.inf, dtype=np.float64),
        "observed_waist_max": np.full(3, -np.inf, dtype=np.float64),
        "waist_guard_tripped": False,
        "arm_interlock_count": 0,
        "arm_interlock_reasons": [],
        "dds_write_count_arm_left_right": np.zeros(3, dtype=np.int64),
        "dds_write_failure_count_arm_left_right": np.zeros(3, dtype=np.int64),
        "dex1_feedback_limit_seen_left_right": np.zeros(2, dtype=bool),
        "dex1_state_stale_seen_left_right": np.zeros(2, dtype=bool),
        "lower_body_peak_speed_rad_s": 0.0,
        "camera_outage_count": 0,
        "camera_recovery_count": 0,
        "camera_hold_count": 0,
        "camera_outage_events": [],
        "camera_role_max_age_ms": {},
        "camera_role_latest_transition_hz": {},
        "camera_role_latest_generation": {},
        "camera_role_latest_metadata": {},
        "camera_bundle_skew_ms": deque(maxlen=301),
        "camera_bundle_valid_times": deque(maxlen=301),
        "camera_bundle_valid_count": 0,
        "camera_bundle_invalid_count": 0,
        "image_network_interface": None,
        "image_network_counters_start": {},
        "image_network_counters_current": {},
        "image_network_counter_delta": {},
        "record_started_count": 0,
        "record_saved_count": 0,
        "record_discarded_count": 0,
        "record_sync_drop_count": 0,
        "record_finalization_failure_count": 0,
        "lossless_recordings": [],
        "reset_count": 0,
        "safe_quit_requested": False,
        "sim_randomization": {},
        "gripper_contact_force_max_n": np.zeros(2, dtype=np.float64),
        "dex1_drive_force_max_n": np.zeros(2, dtype=np.float64),
        "dex1_grasp_force_audit": {},
        "physics_reported": False,
    }
    if args.backend == "sim":
        # Do not let the simulator render its first high-resolution camera
        # packet until the desktop AVP process is ready to receive it. This is
        # a transport readiness barrier, not a motion command: IDLE makes the
        # simulator hold its measured arm/hand state. The real backend must
        # remain entirely non-actuating until the operator presses ``r``.
        last_sent_target = _idle_target(sequence)
        backend.apply(last_sent_target)
    print(
        "Controls: r=track/pause/resume, s=start/save, d=discard+reset, "
        "q=controlled final release/exit"
        + (
            ", Ctrl-C=controlled final release/exit"
            if args.backend == "real"
            else ""
        )
        + (
            f"; foot pedal active ({foot_pedal_binding.device}: left=q, middle=s, right=r)"
            if foot_pedal_binding is not None
            else ""
        ),
        flush=True,
    )
    interrupt_requested = threading.Event()
    previous_sigint_handler = None
    if args.backend == "real":
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def _request_real_release(_signum, _frame) -> None:
            interrupt_requested.set()

        signal.signal(signal.SIGINT, _request_real_release)
    try:
        # Complete the simulator handshake before spawning the terminal input
        # reader. The first RTX frame is a startup barrier, not an operator
        # interaction, and keeping it separate makes a failed bridge explicit.
        initial_observation = backend.observe(timeout_s=INITIAL_OBSERVATION_TIMEOUT_S)
        latest = initial_observation
        observation_stream = ObservationStream(
            backend,
            initial_observation,
            max_hz=(
                config.rates.camera_poll_hz
                if args.backend == "real"
                else config.rates.camera_hz
            ),
        )
        observation_stream.start()
        last_operator_generation = -1
        last_operator_target_sequence = -1
        operator_target_update_count = 0
        operator_target_times: deque[float] = deque(maxlen=61)
        arm_target_min = np.full(14, np.inf, dtype=np.float64)
        arm_target_max = np.full(14, -np.inf, dtype=np.float64)
        hand_target_min = np.full(2, np.inf, dtype=np.float64)
        hand_target_max = np.full(2, -np.inf, dtype=np.float64)
        command_sent_at: dict[int, float] = {}
        command_observation_latency_ms = audit["command_observation_latency_ms"]
        camera_receive_times = audit["camera_receive_times"]
        camera_capture_times = audit["camera_capture_times"]
        camera_sequences = audit["camera_sequences"]
        assert isinstance(command_observation_latency_ms, deque)
        assert isinstance(camera_receive_times, deque)
        assert isinstance(camera_capture_times, deque)
        assert isinstance(camera_sequences, deque)
        missing_camera_sequences = 0
        previous_camera_sequence: int | None = None
        last_latency_sequence = -1
        command_period_s = 1.0 / config.rates.command_hz
        poll_period_s = 1.0 / (
            2.0 * max(config.rates.command_hz, config.rates.camera_hz)
        )
        next_command_time = time.monotonic()
        next_poll_time = time.monotonic()
        pedal_context = (
            FootPedalReader(foot_pedal_binding)
            if foot_pedal_binding is not None
            else nullcontext(None)
        )
        with KeyReader() as keys, pedal_context as pedal:
            while True:
                save_requested = False
                snapshot = observation_stream.latest()
                observation = snapshot.observation
                latest = observation
                stream_health = observation_stream.health()
                role_ages_ms = {
                    str(role): float(metadata.get("age_ms", 0.0))
                    for role, metadata in observation.camera_stream_metadata.items()
                    if isinstance(metadata, dict)
                    and isinstance(metadata.get("age_ms"), (int, float))
                }
                if args.backend == "real" and role_ages_ms:
                    (
                        warning_roles,
                        hold_roles,
                        outage_roles,
                    ) = _camera_alert_roles(
                        role_ages_ms,
                        hold_timeout_s=(
                            config.safety.command_hold_timeout_s
                        ),
                        outage_timeout_s=(
                            config.safety.command_stop_timeout_s
                        ),
                    )
                    camera_stream_live = not hold_roles
                else:
                    warning_roles = []
                    hold_roles = (
                        ["camera_bundle"]
                        if stream_health.last_receive_age_s
                        > config.safety.command_hold_timeout_s
                        else []
                    )
                    outage_roles = (
                        ["camera_bundle"]
                        if stream_health.last_receive_age_s
                        > config.safety.command_stop_timeout_s
                        else []
                    )
                    camera_stream_live = not hold_roles

                if warning_roles and not camera_warning_active:
                    camera_warning_active = True
                    print(
                        "Camera stream warning: a role has not advanced for "
                        f"more than 100ms (roles={warning_roles}, "
                        f"ages_ms={role_ages_ms}).",
                        flush=True,
                    )
                elif not warning_roles and camera_warning_active:
                    camera_warning_active = False
                    print("Camera stream warning cleared.", flush=True)

                if hold_roles and not camera_hold_active:
                    camera_hold_active = True
                    audit["camera_hold_count"] = int(
                        audit["camera_hold_count"]
                    ) + 1
                    if recording is not None:
                        failed_episode_id = recording.episode_id
                        recording.discard()
                        recording = None
                        audit["record_discarded_count"] = int(
                            audit["record_discarded_count"]
                        ) + 1
                        print(
                            "Recording discarded because a camera stream stopped "
                            f"({failed_episode_id}).",
                            flush=True,
                        )
                    if tracking:
                        tracking = False
                        anchor_request.disarm()
                        (
                            paused_hold_arm,
                            paused_hold_hand,
                            paused_hold_torque,
                        ) = _hold_target_from_observation(observation)
                    print(
                        "Camera stream stale; holding the current arm/Dex1 pose "
                        "after 200ms without a new physical frame "
                        f"(roles={hold_roles}, ages_ms={role_ages_ms}, "
                        f"bundle_skew_ms={observation.camera_skew_ms:.3f}, "
                        f"error={stream_health.transient_error!r}).",
                        flush=True,
                    )
                elif not hold_roles and camera_hold_active:
                    camera_hold_active = False
                    print(
                        "Camera streams recovered; tracking remains paused. "
                        "Press r to re-anchor and resume.",
                        flush=True,
                    )

                if outage_roles and not camera_outage_active:
                    camera_outage_active = True
                    camera_outage_started_monotonic = time.monotonic()
                    audit["camera_outage_count"] = int(
                        audit["camera_outage_count"]
                    ) + 1
                    events = audit["camera_outage_events"]
                    assert isinstance(events, list)
                    events.append(
                        {
                            "roles": outage_roles,
                            "started_monotonic_s": (
                                camera_outage_started_monotonic
                            ),
                            "ages_ms_at_detection": dict(role_ages_ms),
                            "recovered": False,
                        }
                    )
                    print(
                        "Camera outage confirmed after 750ms "
                        f"(roles={outage_roles}, ages_ms={role_ages_ms}).",
                        flush=True,
                    )
                elif not outage_roles and camera_outage_active:
                    camera_outage_active = False
                    events = audit["camera_outage_events"]
                    assert isinstance(events, list)
                    if events and camera_outage_started_monotonic is not None:
                        events[-1]["recovered"] = True
                        events[-1]["ended_monotonic_s"] = time.monotonic()
                        events[-1]["duration_s"] = round(
                            time.monotonic()
                            - camera_outage_started_monotonic,
                            3,
                        )
                        audit["camera_recovery_count"] = int(
                            audit["camera_recovery_count"]
                        ) + 1
                    camera_outage_started_monotonic = None

                interlock_reason = observation.diagnostics.get(
                    "arm_interlock_reason"
                )
                if tracking and isinstance(interlock_reason, str) and interlock_reason:
                    audit["arm_interlock_count"] = int(
                        audit["arm_interlock_count"]
                    ) + 1
                    interlock_reasons = audit["arm_interlock_reasons"]
                    assert isinstance(interlock_reasons, list)
                    interlock_reasons.append(interlock_reason)
                    tracking = False
                    anchor_request.disarm()
                    # A backend safety interlock explicitly releases arm_sdk;
                    # unlike an AVP/camera pause, do not keep HOLD authority.
                    paused_hold_arm = None
                    paused_hold_hand = None
                    paused_hold_torque = None
                    next_command_time = time.monotonic()
                    if recording is not None:
                        failed_episode_id = recording.episode_id
                        recording.discard()
                        recording = None
                        audit["record_discarded_count"] = int(
                            audit["record_discarded_count"]
                        ) + 1
                        print(
                            "Recording discarded after the arm safety interlock "
                            f"({failed_episode_id}).",
                            flush=True,
                        )
                    print(
                        "Arm safety interlock latched; teleoperation released. "
                        f"Reason: {interlock_reason}. Stabilize the robot, then press r "
                        "to re-anchor.",
                        flush=True,
                    )
                fresh_camera = snapshot.generation != last_operator_generation
                if fresh_camera:
                    receive_time = time.monotonic()
                    if observation.camera_bundle_valid:
                        audit["camera_bundle_valid_count"] = int(
                            audit["camera_bundle_valid_count"]
                        ) + 1
                        bundle_times = audit["camera_bundle_valid_times"]
                        assert isinstance(bundle_times, deque)
                        bundle_times.append(receive_time)
                    else:
                        audit["camera_bundle_invalid_count"] = int(
                            audit["camera_bundle_invalid_count"]
                        ) + 1
                    bundle_skews = audit["camera_bundle_skew_ms"]
                    assert isinstance(bundle_skews, deque)
                    bundle_skews.append(float(observation.camera_skew_ms))
                    role_max_ages = audit["camera_role_max_age_ms"]
                    role_rates = audit["camera_role_latest_transition_hz"]
                    role_generations = audit["camera_role_latest_generation"]
                    role_latest_metadata = audit[
                        "camera_role_latest_metadata"
                    ]
                    assert isinstance(role_max_ages, dict)
                    assert isinstance(role_rates, dict)
                    assert isinstance(role_generations, dict)
                    assert isinstance(role_latest_metadata, dict)
                    for role, metadata in observation.camera_stream_metadata.items():
                        if not isinstance(metadata, dict):
                            continue
                        role_latest_metadata[role] = dict(metadata)
                        age = metadata.get("age_ms")
                        rate = metadata.get("transition_hz")
                        generation = metadata.get("jpeg_generation")
                        if isinstance(age, (int, float)) and np.isfinite(age):
                            role_max_ages[role] = max(
                                float(role_max_ages.get(role, 0.0)),
                                float(age),
                            )
                        if isinstance(rate, (int, float)) and np.isfinite(rate):
                            role_rates[role] = float(rate)
                        if isinstance(generation, int) and generation >= 0:
                            role_generations[role] = generation
                    network_interface = observation.diagnostics.get(
                        "image_network_interface"
                    )
                    if isinstance(network_interface, str):
                        audit["image_network_interface"] = network_interface
                    for diagnostic_key in (
                        "image_network_counters_start",
                        "image_network_counters_current",
                        "image_network_counter_delta",
                    ):
                        counters = observation.diagnostics.get(diagnostic_key)
                        if isinstance(counters, dict):
                            audit[diagnostic_key] = {
                                str(name): int(value)
                                for name, value in counters.items()
                                if isinstance(value, int) and value >= 0
                            }
                    if (
                        previous_camera_sequence is not None
                        and observation.sequence <= previous_camera_sequence
                    ):
                        camera_receive_times.clear()
                        camera_capture_times.clear()
                        camera_sequences.clear()
                        previous_camera_sequence = None
                    camera_receive_times.append(receive_time)
                    camera_capture_times.append(_remote_capture_seconds(observation))
                    camera_sequences.append(observation.sequence)
                    audit["camera_received_count"] = int(
                        audit["camera_received_count"]
                    ) + 1
                    if previous_camera_sequence is not None:
                        missing_camera_sequences += max(
                            0, observation.sequence - previous_camera_sequence - 1
                        )
                        audit["missing_camera_sequences"] = missing_camera_sequences
                    previous_camera_sequence = observation.sequence
                    audit["sim_sender_drops"] = max(
                        int(audit["sim_sender_drops"]),
                        int(observation.diagnostics.get("dropped_operator_frames", 0)),
                    )
                    body_position = np.asarray(
                        observation.body_joint_position_rad, dtype=np.float64
                    )
                    if body_position.shape == (29,) and np.isfinite(body_position).all():
                        waist_position = body_position[12:15]
                        audit["observed_waist_min"] = np.minimum(
                            audit["observed_waist_min"], waist_position
                        )
                        audit["observed_waist_max"] = np.maximum(
                            audit["observed_waist_max"], waist_position
                        )
                    audit["waist_guard_tripped"] = bool(
                        audit["waist_guard_tripped"]
                        or observation.diagnostics.get("waist_guard_tripped", False)
                    )
                    for diagnostic_key, audit_key, width in (
                        (
                            "dds_write_count_arm_left_right",
                            "dds_write_count_arm_left_right",
                            3,
                        ),
                        (
                            "dds_write_failure_count_arm_left_right",
                            "dds_write_failure_count_arm_left_right",
                            3,
                        ),
                    ):
                        values = np.asarray(
                            observation.diagnostics.get(
                                diagnostic_key, np.zeros(width, dtype=np.int64)
                            ),
                            dtype=np.int64,
                        )
                        if values.shape == (width,) and np.all(values >= 0):
                            audit[audit_key] = np.maximum(audit[audit_key], values)
                    feedback_limit = np.asarray(
                        observation.diagnostics.get(
                            "dex1_feedback_limit_active_left_right", (False, False)
                        ),
                        dtype=bool,
                    )
                    if feedback_limit.shape == (2,):
                        audit["dex1_feedback_limit_seen_left_right"] = np.logical_or(
                            audit["dex1_feedback_limit_seen_left_right"],
                            feedback_limit,
                        )
                    dex1_stale = np.asarray(
                        observation.diagnostics.get(
                            "dex1_state_stale_left_right", (False, False)
                        ),
                        dtype=bool,
                    )
                    if dex1_stale.shape == (2,):
                        audit["dex1_state_stale_seen_left_right"] = np.logical_or(
                            audit["dex1_state_stale_seen_left_right"], dex1_stale
                        )
                    lower_body_peak = observation.diagnostics.get(
                        "lower_body_peak_speed_rad_s", 0.0
                    )
                    if isinstance(lower_body_peak, (int, float)) and np.isfinite(
                        lower_body_peak
                    ):
                        audit["lower_body_peak_speed_rad_s"] = max(
                            float(audit["lower_body_peak_speed_rad_s"]),
                            float(lower_body_peak),
                        )
                    if args.backend == "sim":
                        randomization = observation.diagnostics.get("randomization")
                        if isinstance(randomization, dict) and randomization:
                            audit["sim_randomization"] = randomization
                        for source_key, audit_key in (
                            (
                                "gripper_contact_force_n",
                                "gripper_contact_force_max_n",
                            ),
                            ("dex1_drive_force_n", "dex1_drive_force_max_n"),
                        ):
                            metrics = observation.diagnostics.get(source_key)
                            if not isinstance(metrics, dict) or not metrics.get("available"):
                                continue
                            sample = np.asarray(
                                (
                                    metrics.get("left_max_n", 0.0),
                                    metrics.get("right_max_n", 0.0),
                                ),
                                dtype=np.float64,
                            )
                            if sample.shape == (2,) and np.isfinite(sample).all():
                                audit[audit_key] = np.maximum(audit[audit_key], sample)
                        grasp_force_audit = observation.diagnostics.get(
                            "dex1_grasp_force_audit"
                        )
                        if isinstance(grasp_force_audit, dict):
                            audit["dex1_grasp_force_audit"] = grasp_force_audit
                        if (
                            not bool(audit["physics_reported"])
                            and _assembled_table_mass_kg(audit["sim_randomization"])
                            is not None
                        ):
                            contact = audit["sim_randomization"].get(
                                "contact_materials", {}
                            )
                            pairs = contact.get("pairs", {}) if isinstance(contact, dict) else {}
                            print(
                                "Simulator task physics: "
                                f"assembled_table_mass_kg={_assembled_table_mass_kg(audit['sim_randomization']):.4f}, "
                                "dex1_effort_limit_n_per_finger=20.0, "
                                f"contact_pairs={pairs}",
                                flush=True,
                            )
                            audit["physics_reported"] = True
                    applied_sequence = observation.diagnostics.get(
                        "last_applied_command_sequence"
                    )
                    if (
                        isinstance(applied_sequence, int)
                        and applied_sequence > last_latency_sequence
                        and applied_sequence in command_sent_at
                    ):
                        command_observation_latency_ms.append(
                            (time.monotonic() - command_sent_at[applied_sequence])
                            * 1000.0
                        )
                        last_latency_sequence = applied_sequence
                    track_sequences = audit["track_command_sequences"]
                    assert isinstance(track_sequences, set)
                    if tracking and applied_sequence in track_sequences:
                        observed_arm = np.asarray(
                            observation.arm_joint_position_rad, dtype=np.float64
                        )
                        observed_hand = np.asarray(
                            observation.dex1_opening_fraction, dtype=np.float64
                        )
                        audit["observed_arm_min"] = np.minimum(
                            audit["observed_arm_min"], observed_arm
                        )
                        audit["observed_arm_max"] = np.maximum(
                            audit["observed_arm_max"], observed_arm
                        )
                        audit["observed_hand_min"] = np.minimum(
                            audit["observed_hand_min"], observed_hand
                        )
                        audit["observed_hand_max"] = np.maximum(
                            audit["observed_hand_max"], observed_hand
                        )
                        arm_errors = audit["arm_tracking_errors_rad"]
                        hand_errors = audit["hand_tracking_errors"]
                        assert isinstance(arm_errors, deque)
                        assert isinstance(hand_errors, deque)
                        arm_errors.extend(
                            np.abs(
                                np.asarray(
                                    observation.applied_arm_target_rad,
                                    dtype=np.float64,
                                )
                                - observed_arm
                            ).tolist()
                        )
                        hand_errors.extend(
                            np.abs(
                                np.asarray(
                                    observation.applied_dex1_opening_target,
                                    dtype=np.float64,
                                )
                                - observed_hand
                            ).tolist()
                        )
                    stale_sequences = [
                        value
                        for value in command_sent_at
                        if value <= last_latency_sequence
                    ]
                    for value in stale_sequences:
                        command_sent_at.pop(value, None)
                operator.submit(
                    camera_jpeg=(
                        {
                            role: observation.camera_jpeg[role]
                            for role in operator_roles
                        }
                        if fresh_camera
                        else None
                    ),
                    arm_joint_position_rad=observation.arm_joint_position_rad,
                    arm_joint_velocity_rad_s=observation.arm_joint_velocity_rad_s,
                    dex1_opening_fraction=observation.dex1_opening_fraction,
                    tracking_generation=anchor_request.generation,
                )
                if fresh_camera:
                    last_operator_generation = snapshot.generation
                # Pedal presses have priority.  The pedal device is grabbed
                # exclusively only inside this context; outside teleop it is
                # ignored by the installed udev/libinput rule.
                key = None if pedal is None else pedal.poll()
                if key is None:
                    key = keys.poll()
                if interrupt_requested.is_set():
                    interrupt_requested.clear()
                    key = "__ctrl_c_release__"
                    print(
                        "Ctrl-C requested; stopping tracking and performing a "
                        "controlled final arm_sdk release.",
                        flush=True,
                    )
                operator_target = operator.latest_target()
                if (
                    operator_target is not None
                    and operator_target.source_sequence
                    != last_operator_target_sequence
                ):
                    last_operator_target_sequence = operator_target.source_sequence
                    operator_target_update_count += 1
                    operator_target_times.append(time.monotonic())
                    audit["hand_event_count_max"] = max(
                        int(audit["hand_event_count_max"]),
                        operator_target.hand_event_count,
                    )
                    audit["hand_invalid_event_count_max"] = max(
                        int(audit["hand_invalid_event_count_max"]),
                        operator_target.hand_invalid_event_count,
                    )
                    audit["hand_missing_pose_count_max"] = max(
                        int(audit["hand_missing_pose_count_max"]),
                        operator_target.hand_missing_pose_count,
                    )
                    audit["hand_invalid_wrist_count_max"] = max(
                        int(audit["hand_invalid_wrist_count_max"]),
                        operator_target.hand_invalid_wrist_count,
                    )
                    audit["hand_invalid_pinch_count_max"] = max(
                        int(audit["hand_invalid_pinch_count_max"]),
                        operator_target.hand_invalid_pinch_count,
                    )
                    audit["hand_invalid_unused_skeleton_count_max"] = max(
                        int(audit["hand_invalid_unused_skeleton_count_max"]),
                        operator_target.hand_invalid_unused_skeleton_count,
                    )
                    hand_detail_max = audit["hand_invalid_details_max"]
                    assert isinstance(hand_detail_max, dict)
                    for name, count in operator_target.hand_invalid_details.items():
                        hand_detail_max[name] = max(
                            int(hand_detail_max.get(name, 0)), count
                        )
                    if operator_target.avp_live:
                        audit["avp_connected"] = True
                        if operator_target.hand_tracking_hz > 0.0:
                            hand_rates = audit["hand_tracking_hz"]
                            assert isinstance(hand_rates, deque)
                            hand_rates.append(operator_target.hand_tracking_hz)
                    if (
                        operator_target.arm_position_rad is not None
                        and operator_target.dex1_opening_fraction is not None
                    ):
                        arm_value = np.asarray(operator_target.arm_position_rad)
                        hand_value = np.asarray(operator_target.dex1_opening_fraction)
                        arm_target_min = np.minimum(arm_target_min, arm_value)
                        arm_target_max = np.maximum(arm_target_max, arm_value)
                        hand_target_min = np.minimum(hand_target_min, hand_value)
                        hand_target_max = np.maximum(hand_target_max, hand_value)
                    if (
                        operator_target_update_count <= 3
                        or operator_target_update_count % 150 == 0
                    ):
                        target_hz = 0.0
                        if len(operator_target_times) > 1:
                            elapsed = operator_target_times[-1] - operator_target_times[0]
                            if elapsed > 0.0:
                                target_hz = (
                                    len(operator_target_times) - 1
                                ) / elapsed
                        have_target_range = bool(
                            np.isfinite(arm_target_min).all()
                            and np.isfinite(hand_target_min).all()
                        )
                        arm_span_max = (
                            float(np.max(arm_target_max - arm_target_min))
                            if have_target_range
                            else 0.0
                        )
                        dex1_range = (
                            (hand_target_max - hand_target_min).round(3).tolist()
                            if have_target_range
                            else [0.0, 0.0]
                        )
                        camera_source_hz = 0.0
                        camera_transport_hz = 0.0
                        stream_stats = observation_stream.stats()
                        if len(stream_stats.capture_times) > 1:
                            capture_elapsed = (
                                stream_stats.capture_times[-1]
                                - stream_stats.capture_times[0]
                            )
                            receive_elapsed = (
                                stream_stats.receive_times[-1]
                                - stream_stats.receive_times[0]
                            )
                            if capture_elapsed > 0.0:
                                camera_source_hz = (
                                    stream_stats.sequences[-1]
                                    - stream_stats.sequences[0]
                                ) / capture_elapsed
                            if receive_elapsed > 0.0:
                                camera_transport_hz = (
                                    len(stream_stats.receive_times) - 1
                                ) / receive_elapsed
                        if args.backend == "real" and observation.camera_stream_metadata:
                            current_role_rates = []
                            role_stream_live = True
                            for metadata in observation.camera_stream_metadata.values():
                                if not isinstance(metadata, dict):
                                    continue
                                age_ms = metadata.get("age_ms")
                                transition_hz = metadata.get("transition_hz")
                                if (
                                    isinstance(age_ms, (int, float))
                                    and float(age_ms) > 100.0
                                ):
                                    role_stream_live = False
                                if isinstance(transition_hz, (int, float)):
                                    current_role_rates.append(float(transition_hz))
                            camera_source_hz = (
                                min(current_role_rates)
                                if role_stream_live and current_role_rates
                                else 0.0
                            )
                            bundle_times = audit["camera_bundle_valid_times"]
                            assert isinstance(bundle_times, deque)
                            camera_transport_hz = _sample_rate(bundle_times)
                        print(
                            "AVP operator timing: "
                            f"target_hz={target_hz:.2f}, "
                            f"image_ms={operator_target.image_processing_ms:.2f}, "
                            f"ik_ms={operator_target.ik_processing_ms:.2f}, "
                            f"total_ms={operator_target.total_processing_ms:.2f}, "
                            f"latest_request_lag={operator.submitted_sequence() - operator_target.source_sequence}, "
                            f"session_age_s={operator_target.session_age_s}, "
                            f"hand_age_s={operator_target.hand_age_s}, "
                            f"hand_tracking_hz={operator_target.hand_tracking_hz:.2f}, "
                            f"hand_event_count={operator_target.hand_event_count}, "
                            f"hand_contiguous_event_count={operator_target.hand_contiguous_event_count}, "
                            f"hand_invalid_event_count={operator_target.hand_invalid_event_count}, "
                            "hand_invalid_reasons="
                            f"[missing_pose={operator_target.hand_missing_pose_count},"
                            f"invalid_wrist={operator_target.hand_invalid_wrist_count},"
                            f"invalid_pinch={operator_target.hand_invalid_pinch_count},"
                            "invalid_unused_skeleton="
                            f"{operator_target.hand_invalid_unused_skeleton_count}], "
                            "camera_roles="
                            f"{observation.camera_stream_metadata}, "
                            f"command_observation_latency_ms={float(np.median(command_observation_latency_ms)) if command_observation_latency_ms else 0.0:.2f}, "
                            f"camera_source_hz={camera_source_hz:.2f}, "
                            f"camera_transport_hz={camera_transport_hz:.2f}, "
                            f"missing_camera_sequences={stream_stats.missing_sequences}, "
                            f"sim_sender_drops={int(observation.diagnostics.get('dropped_operator_frames', 0))}, "
                            f"arm_span_max_rad={arm_span_max:.4f}, "
                            f"dex1_range={dex1_range}",
                            flush=True,
                        )
                now_ns = time.monotonic_ns()
                avp_live = bool(
                    operator_target is not None
                    and operator_target.avp_live
                    and 0
                    <= now_ns - operator_target.monotonic_ns
                    <= int(config.safety.command_stop_timeout_s * 1.0e9)
                )
                if (
                    pending_recording_start is not None
                    and pending_recording_start.done()
                ):
                    try:
                        started_recording = pending_recording_start.result()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            "Recording refused because the recorder did not "
                            f"acknowledge a safe start: {exc}. "
                            "Teleoperation remains active.",
                            flush=True,
                        )
                    else:
                        if cancel_pending_recording_start:
                            pending_recording_discards.append(
                                started_recording.discard_async()
                            )
                            audit["record_discarded_count"] = int(
                                audit["record_discarded_count"]
                            ) + 1
                            print(
                                "Pending recording start was cancelled; Orin "
                                "cleanup is running in the background.",
                                flush=True,
                            )
                        else:
                            recording = started_recording
                            audit["record_started_count"] = int(
                                audit["record_started_count"]
                            ) + 1
                            pending_event = ControlEvent.RECORD_TOGGLE
                            print(
                                f"Recording started: {recording.episode_id}",
                                flush=True,
                            )
                    pending_recording_start = None
                    cancel_pending_recording_start = False
                if key == "r":
                    if tracking or anchor_request.generation > 0:
                        # One r press while tracking (or while an anchor is
                        # pending) is an explicit pause/cancel. Preserve the
                        # latest backend-applied target, not a potentially
                        # moving AVP request, and send HOLD immediately.
                        tracking = False
                        anchor_request.disarm()
                        (
                            paused_hold_arm,
                            paused_hold_hand,
                            paused_hold_torque,
                        ) = _hold_target_from_observation(observation)
                        next_command_time = time.monotonic()
                        print(
                            "Tracking paused; holding the current arm/Dex1 pose. "
                            "Press r again to re-anchor and resume.",
                            flush=True,
                        )
                    else:
                        # Never queue an initial anchor before the headset has
                        # an already-live bilateral hand stream.  Queuing made
                        # a single early pedal press silently arm the robot
                        # later, just as Vision Pro was still acquiring hands.
                        # The real HUD exposes this same READY/WAIT state.
                        if not avp_live or not camera_stream_live:
                            print(
                                "Tracking not armed: wait for HANDS READY and live camera "
                                "streams, then press r.",
                                flush=True,
                            )
                        else:
                            # A second r while paused requests a new anchor.
                            # Keep HOLD active until bilateral hand tracking
                            # has completed that anchor, so the arms never sag
                            # or jump between pause and resume.
                            anchor_request.request()
                            next_command_time = time.monotonic()
                            if paused_hold_arm is not None:
                                message = (
                                    "Tracking resume requested; holding pose until "
                                    "stable bilateral re-anchor completes."
                                )
                            else:
                                message = "Tracking anchor requested; hold both hands steady."
                            print(message, flush=True)
                elif key == "s":
                    if recording is None and pending_recording_start is not None:
                        print(
                            "Recording start is still waiting for the Orin ACK; "
                            "teleoperation remains active.",
                            flush=True,
                        )
                    elif recording is None and not tracking:
                        print("Recording refused: press r after AVP tracking is live.", flush=True)
                    elif recording is None:
                        identity = EpisodeIdentity(
                            backend=args.backend,
                            dr_profile=args.dr_profile if args.backend == "sim" else "real",
                            seed=args.seed,
                            config_sha256=config.digest,
                            runtime_digest=(
                                config.runtime.robofinals_digest
                                if args.backend == "sim"
                                else config.runtime.xr_revision
                            ),
                        )
                        if args.backend == "sim":
                            recording = ReplayTrajectoryWriter(
                                    Path(output_root).parent
                                    / "replay_pending",
                                    identity,
                                )
                            audit["record_started_count"] = int(
                                audit["record_started_count"]
                            ) + 1
                            pending_event = ControlEvent.RECORD_TOGGLE
                            print(
                                "Recording started: "
                                f"{recording.episode_id} (30 Hz command trajectory; "
                                "four cameras will be rendered offline)",
                                flush=True,
                            )
                        else:
                            pending_recording_start = (
                                start_lossless_real_episode_async(
                                    output_root,
                                    identity,
                                    recorder_host=os.environ.get(
                                        "G1_RECORDER_CONTROL_HOST",
                                        args.image_server_ip,
                                    ),
                                    ssh_target=os.environ.get(
                                        "G1_ORIN_SSH_TARGET",
                                        "g1-orin",
                                    ),
                                )
                            )
                            print(
                                "Recording start requested; waiting for the "
                                "Orin recorder ACK in the background.",
                                flush=True,
                            )
                    elif recording.frame_count < 2:
                        print(
                            "Save refused: record at least two synchronized frames.",
                            flush=True,
                        )
                    else:
                        pending_event = ControlEvent.RECORD_TOGGLE
                        save_requested = True
                elif key == "d":
                    if recording is not None:
                        if isinstance(recording, LosslessRealEpisodeWriter):
                            pending_recording_discards.append(
                                recording.discard_async()
                            )
                        else:
                            recording.discard()
                        recording = None
                        audit["record_discarded_count"] = int(
                            audit["record_discarded_count"]
                        ) + 1
                    if pending_recording_start is not None:
                        cancel_pending_recording_start = True
                    audit["reset_count"] = int(audit["reset_count"]) + 1
                    pending_event = ControlEvent.DISCARD_RESET
                    tracking = False
                    anchor_request.disarm()
                    paused_hold_arm = None
                    paused_hold_hand = None
                    paused_hold_torque = None
                    print("Episode discarded; reset requested.", flush=True)
                elif key == "q":
                    audit["safe_quit_requested"] = True
                    termination_reason = (
                        "q_controlled_release"
                        if args.backend == "real"
                        else "safe_quit"
                    )
                    pending_event = ControlEvent.QUIT
                    tracking = False
                    anchor_request.disarm()
                    paused_hold_arm = None
                    paused_hold_hand = None
                    paused_hold_torque = None
                    if args.backend == "real":
                        print(
                            "q requested; stopping tracking and performing a controlled "
                            "final arm_sdk release.",
                            flush=True,
                        )
                elif key == "__ctrl_c_release__":
                    audit["safe_quit_requested"] = True
                    termination_reason = "ctrl_c_controlled_release"
                    pending_event = ControlEvent.QUIT
                    tracking = False
                    anchor_request.disarm()
                    paused_hold_arm = None
                    paused_hold_hand = None
                    paused_hold_torque = None

                if tracking and not avp_live:
                    tracking = False
                    anchor_request.disarm()
                    # Hand occlusion must be a clutch-like pause, not an arm
                    # release. AVP input may recover on its own, but it can
                    # never resume motion until the operator explicitly
                    # presses r and a fresh bilateral anchor is complete.
                    (
                        paused_hold_arm,
                        paused_hold_hand,
                        paused_hold_torque,
                    ) = _hold_target_from_observation(observation)
                    rearm_notice_pending = True
                    audit["stale_stop_count"] = int(audit["stale_stop_count"]) + 1
                    session_age = (
                        None if operator_target is None else operator_target.session_age_s
                    )
                    hand_age = None if operator_target is None else operator_target.hand_age_s
                    print(
                        "AVP hand input stale; pausing at the current arm/Dex1 pose "
                        "(press r after hand tracking returns to re-anchor and resume) "
                        f"(session_age_s={session_age}, hand_age_s={hand_age}).",
                        flush=True,
                    )

                if rearm_notice_pending and avp_live:
                    rearm_notice_pending = False
                    if anchor_request.generation > 0:
                        print(
                            "AVP hand input restored; applying the queued re-anchor.",
                            flush=True,
                        )
                    else:
                        print(
                            "AVP hand input restored; press r to re-anchor and resume.",
                            flush=True,
                        )

                anchored_target_ready = bool(
                    operator_target is not None
                    and avp_live
                    and camera_stream_live
                    and anchor_request.generation > 0
                    and operator_target.tracking_generation
                    == anchor_request.generation
                    and operator_target.arm_position_rad is not None
                    and operator_target.arm_feedforward_torque_nm is not None
                    and operator_target.dex1_opening_fraction is not None
                )
                if (
                    anchor_request.generation > 0
                    and not tracking
                    and anchored_target_ready
                ):
                    tracking = True
                    paused_hold_arm = None
                    paused_hold_hand = None
                    paused_hold_torque = None
                    audit["tracking_enabled_count"] = int(
                        audit["tracking_enabled_count"]
                    ) + 1
                    print(
                        "Tracking enabled after stable bilateral re-anchor.",
                        flush=True,
                    )

                send_command = (
                    time.monotonic() >= next_command_time
                    or pending_event is not ControlEvent.NONE
                )
                sent_event = ControlEvent.NONE
                if send_command:
                    missing_tracking_target = bool(
                        tracking
                        and (
                            operator_target is None
                            or operator_target.arm_position_rad is None
                            or operator_target.arm_feedforward_torque_nm is None
                            or operator_target.dex1_opening_fraction is None
                        )
                    )
                    if missing_tracking_target:
                        # The operator process is asynchronous and latest-only.
                        # A transient bilateral-frame fault may be observed as a
                        # target-less response before its non-live response.
                        # Latch the already applied target and require a fresh
                        # explicit r anchor; never turn an expected clutch event
                        # into process exit and arm_sdk release.
                        tracking = False
                        anchor_request.disarm()
                        (
                            paused_hold_arm,
                            paused_hold_hand,
                            paused_hold_torque,
                        ) = _hold_target_from_observation(observation)
                        rearm_notice_pending = True
                        audit["stale_stop_count"] = int(
                            audit["stale_stop_count"]
                        ) + 1
                        print(
                            "AVP target unavailable; holding the current arm/Dex1 pose "
                            "and requiring r to re-anchor.",
                            flush=True,
                        )
                    if tracking:
                        assert operator_target is not None
                        assert operator_target.arm_position_rad is not None
                        assert operator_target.arm_feedforward_torque_nm is not None
                        assert operator_target.dex1_opening_fraction is not None
                        arm_target = np.asarray(operator_target.arm_position_rad)
                        arm_torque = np.asarray(
                            operator_target.arm_feedforward_torque_nm
                        )
                        hand_target = np.asarray(
                            operator_target.dex1_opening_fraction
                        )
                        mode = ControlMode.TRACK
                    elif (
                        paused_hold_arm is not None
                        and paused_hold_hand is not None
                        and paused_hold_torque is not None
                    ):
                        arm_target = paused_hold_arm
                        arm_torque = paused_hold_torque
                        hand_target = paused_hold_hand
                        mode = ControlMode.HOLD
                    else:
                        arm_target = np.asarray(observation.arm_joint_position_rad)
                        arm_torque = np.zeros(14, dtype=np.float64)
                        hand_target = np.asarray(observation.dex1_opening_fraction)
                        mode = ControlMode.IDLE
                    sequence += 1
                    last_sent_target = ArmHandTarget(
                        sequence=sequence,
                        monotonic_ns=time.monotonic_ns(),
                        mode=mode,
                        event=pending_event,
                        arm_position_rad=tuple(arm_target),
                        dex1_opening_fraction=tuple(hand_target),
                        arm_feedforward_torque_nm=tuple(arm_torque),
                    )
                    backend.apply(last_sent_target)
                    command_sent_time = time.monotonic()
                    command_sent_at[sequence] = command_sent_time
                    if mode is ControlMode.TRACK:
                        audit["track_command_count"] = int(
                            audit["track_command_count"]
                        ) + 1
                        track_command_times = audit["track_command_times"]
                        track_sequences = audit["track_command_sequences"]
                        assert isinstance(track_command_times, deque)
                        assert isinstance(track_sequences, set)
                        track_command_times.append(command_sent_time)
                        track_sequences.add(sequence)
                        audit["commanded_arm_min"] = np.minimum(
                            audit["commanded_arm_min"], arm_target
                        )
                        audit["commanded_arm_max"] = np.maximum(
                            audit["commanded_arm_max"], arm_target
                        )
                        audit["commanded_hand_min"] = np.minimum(
                            audit["commanded_hand_min"], hand_target
                        )
                        audit["commanded_hand_max"] = np.maximum(
                            audit["commanded_hand_max"], hand_target
                        )
                    sent_event = pending_event
                    pending_event = ControlEvent.NONE
                    next_command_time += command_period_s
                    command_now = time.monotonic()
                    if next_command_time <= command_now:
                        next_command_time = command_now + command_period_s

                if (
                    args.backend == "real"
                    and isinstance(recording, LosslessRealEpisodeWriter)
                    and sent_event is ControlEvent.NONE
                    and last_sent_target is not None
                    and send_command
                ):
                    # Numeric state/action is captured on the canonical command
                    # clock, including an intentional HOLD during hand/camera
                    # loss. Physical JPEGs are recorded losslessly on Orin and
                    # joined only after the operator stops the episode.
                    recording.append(observation, last_sent_target)
                if (
                    args.backend == "sim"
                    and isinstance(recording, ReplayTrajectoryWriter)
                    and sent_event is ControlEvent.NONE
                    and last_sent_target is not None
                    and last_sent_target.mode is ControlMode.TRACK
                    and send_command
                ):
                    # Capture the command stream at its configured 30 Hz,
                    # independent of the lower-rate latest-frame AVP preview.
                    recording.append(observation, last_sent_target)
                if save_requested and isinstance(
                    recording, LosslessRealEpisodeWriter
                ):
                    pending_recording_finalizers.append(
                        recording.save_async(
                            diagnostics={
                                "operator_confirmed_save": True,
                                "success_source": "operator_confirmation",
                                "preview_transport_is_latest_only": True,
                                **dict(observation.diagnostics),
                            },
                            success=True,
                        )
                    )
                    queued_episode_id = recording.episode_id
                    recording = None
                    print(
                        "Recording stopped on Orin; transfer and fixed-30-Hz "
                        f"validation queued in the background: {queued_episode_id}",
                        flush=True,
                    )
                elif save_requested and recording is not None:
                    if args.backend != "sim":
                        raise RuntimeError(
                            "real recording bypassed the canonical Orin MCAP writer"
                        )
                    accepted = observation.success is True
                    rejection_reasons: list[str] = []
                    if observation.success is not True:
                        rejection_reasons.append("simulator_success_not_reached")
                    # Preview rate is intentionally not a dataset-quality
                    # signal: the 30 Hz dataset is generated by the
                    # subsequent offline simulator replay. Keep it only as a
                    # latency/UX diagnostic.
                    source_hz, transport_hz = _observation_stream_rates(
                        observation_stream.stats()
                    )
                    stream_rates = {
                        "preview_source_hz": round(source_hz, 3),
                        "preview_transport_hz": round(transport_hz, 3),
                    }
                    if isinstance(recording, ReplayTrajectoryWriter):
                        stream_rates["trajectory_command_hz"] = round(
                            recording.source_hz, 3
                        )
                        if recording.source_hz < 28.0:
                            rejection_reasons.append(
                                "trajectory_command_rate_below_28hz:"
                                f"{recording.source_hz:.3f}"
                            )
                    accepted = accepted and not rejection_reasons
                    path = recording.save(
                        diagnostics={
                            "operator_confirmed_save": True,
                            "simulator_success": observation.success,
                            "success_source": "simulator_validation",
                            "collection_disposition": (
                                "accepted" if accepted else "rejected_diagnostic"
                            ),
                            "rejection_reasons": rejection_reasons,
                            "observation_stream_rates": stream_rates,
                            **dict(observation.diagnostics),
                        },
                        success=accepted,
                    )
                    recording = None
                    audit["record_saved_count"] = int(
                        audit["record_saved_count"]
                    ) + 1
                    if not accepted:
                        print(
                            "Diagnostic recording saved (not eligible for training): "
                            f"{path}; reasons={rejection_reasons}",
                            flush=True,
                        )
                    elif args.backend == "sim":
                        print(
                            "Simulator trajectory saved for offline 30 Hz four-camera rendering: "
                            f"{path}",
                            flush=True,
                        )
                    else:
                        print(f"Episode saved: {path}", flush=True)
                for finalizer in tuple(pending_recording_finalizers):
                    if not finalizer.done():
                        continue
                    pending_recording_finalizers.remove(finalizer)
                    try:
                        finalized_path = finalizer.result()
                    except Exception as exc:  # noqa: BLE001
                        audit["record_finalization_failure_count"] = int(
                            audit["record_finalization_failure_count"]
                        ) + 1
                        audit["record_discarded_count"] = int(
                            audit["record_discarded_count"]
                        ) + 1
                        print(
                            "Lossless recording finalization failed; the "
                            f"pending diagnostic was preserved: {exc}",
                            flush=True,
                        )
                    else:
                        _audit_finalized_lossless_episode(
                            audit, finalized_path
                        )
                        audit["record_saved_count"] = int(
                            audit["record_saved_count"]
                        ) + 1
                        print(
                            f"Fixed-30-Hz episode finalized: {finalized_path}",
                            flush=True,
                        )
                for discard in tuple(pending_recording_discards):
                    if not discard.done():
                        continue
                    pending_recording_discards.remove(discard)
                    try:
                        discard.result()
                    except Exception as exc:  # noqa: BLE001
                        audit["record_finalization_failure_count"] = int(
                            audit["record_finalization_failure_count"]
                        ) + 1
                        print(
                            "Lossless recording discard cleanup failed: "
                            f"{exc}",
                            flush=True,
                        )
                if sent_event is ControlEvent.QUIT:
                    break
                next_poll_time += poll_period_s
                poll_now = time.monotonic()
                if next_poll_time <= poll_now:
                    next_poll_time = poll_now + poll_period_s
                else:
                    time.sleep(next_poll_time - poll_now)
    except BaseException as exc:  # noqa: BLE001
        session_error = f"{type(exc).__name__}: {exc}"
        termination_reason = "error"
        raise
    finally:
        if observation_stream is not None:
            observation_stream.request_stop()
        if latest is not None:
            try:
                backend.apply(
                    ArmHandTarget(
                        sequence=sequence + 1,
                        monotonic_ns=time.monotonic_ns(),
                        mode=ControlMode.IDLE,
                        event=ControlEvent.QUIT,
                        arm_position_rad=latest.arm_joint_position_rad,
                        dex1_opening_fraction=latest.dex1_opening_fraction,
                    )
                )
            except Exception:
                pass
        if observation_stream is not None:
            try:
                observation_stream.close()
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"observation stream close failed: {exc}")
            else:
                try:
                    stream_stats = observation_stream.stats()
                except RuntimeError as exc:
                    audit["observation_stream_error"] = str(exc)
                else:
                    audit["camera_receive_times"] = deque(
                        stream_stats.receive_times, maxlen=301
                    )
                    audit["camera_capture_times"] = deque(
                        stream_stats.capture_times, maxlen=301
                    )
                    audit["camera_sequences"] = deque(
                        stream_stats.sequences, maxlen=301
                    )
                    audit["camera_received_count"] = stream_stats.received_count
                    audit["missing_camera_sequences"] = stream_stats.missing_sequences
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"backend close failed: {exc}")
        try:
            operator.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"operator close failed: {exc}")
        # Remote camera finalization may need disk flush/network I/O. Never
        # put it in front of the real arm-sdk controlled release.
        if pending_recording_start is not None:
            try:
                started_writer = pending_recording_start.result(timeout=10)
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(
                    f"lossless recording start failed: {exc}"
                )
            else:
                if recording is None:
                    recording = started_writer
                else:
                    started_writer.discard()
                    cleanup_errors.append(
                        "duplicate lossless recording writer was discarded"
                    )
        if recording is not None:
            try:
                recording.discard()
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"recording discard failed: {exc}")
            audit["record_discarded_count"] = int(
                audit["record_discarded_count"]
            ) + 1
        for discard in pending_recording_discards:
            try:
                discard.result(timeout=310)
            except Exception as exc:  # noqa: BLE001
                audit["record_finalization_failure_count"] = int(
                    audit["record_finalization_failure_count"]
                ) + 1
                cleanup_errors.append(
                    f"lossless recording discard failed: {exc}"
                )
        # Camera transfer/materialization is deliberately outside the control
        # loop. Wait only after robot and XR resources have been released.
        for finalizer in pending_recording_finalizers:
            try:
                finalized_path = finalizer.result(timeout=660)
            except Exception as exc:  # noqa: BLE001
                audit["record_finalization_failure_count"] = int(
                    audit["record_finalization_failure_count"]
                ) + 1
                cleanup_errors.append(
                    f"lossless recording finalization failed: {exc}"
                )
            else:
                _audit_finalized_lossless_episode(audit, finalized_path)
                audit["record_saved_count"] = int(
                    audit["record_saved_count"]
                ) + 1
                print(
                    f"Fixed-30-Hz episode finalized: {finalized_path}",
                    flush=True,
                )
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if camera_outage_started_monotonic is not None:
            events = audit.get("camera_outage_events", [])
            if isinstance(events, list) and events:
                events[-1]["duration_s"] = round(
                    time.monotonic() - camera_outage_started_monotonic,
                    3,
                )
        if args.session_report is not None:
            _write_session_report(
                args.session_report,
                args=args,
                config=config,
                audit=audit,
                termination_reason=termination_reason,
                error=session_error,
            )
        # Preserve the primary traceback when shutdown follows another error;
        # it is already non-zero and the cleanup failure remains in the report.
        if cleanup_errors and session_error is None:
            raise RuntimeError("; ".join(cleanup_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
