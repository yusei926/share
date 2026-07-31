"""Crash-safe raw teleoperation episode writer.

The raw format keeps true head-right imagery and simulator diagnostics outside
the three-camera policy namespace. A later validated conversion computes the
six immutable numeric LeRobot features and encodes video shards.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Mapping
from uuid import uuid4

import numpy as np

from .contracts import ArmHandTarget, TeleopObservation
from .shared.policy_contract import ACTION_DIM, ACTION_ORDER, STATE_DIM, STATE_ORDER, action_16d, state_19d


RAW_EPISODE_SCHEMA_VERSION = "team_ramen_flip_table_raw_teleop_episode/v2"
POLICY_CAMERA_ROLE_TO_KEY = {
    "head_left": "observation.images.cam_0",
    "left_wrist": "observation.images.cam_2",
    "right_wrist": "observation.images.cam_3",
}
REQUIRED_CAMERA_ROLES = (
    "head_left",
    "head_right",
    "left_wrist",
    "right_wrist",
)
CAMERA_HZ = 30.0
MAXIMUM_SYNTHETIC_CAMERA_DELAY_STEPS = 2
CAMERA_SCHEDULING_MARGIN_STEPS = 1
DEFAULT_MAXIMUM_CAMERA_AGE_S = (
    MAXIMUM_SYNTHETIC_CAMERA_DELAY_STEPS + CAMERA_SCHEDULING_MARGIN_STEPS
) / CAMERA_HZ + 1.0e-3


class FrameSynchronizationError(ValueError):
    """A sample cannot be admitted without violating dataset timing."""


class ReplayTrajectoryWriter:
    """Persist a 30 Hz simulator control trajectory for offline rendering.

    Interactive Isaac rendering cannot sustain four 640x480 RTX cameras at
    30 Hz on this scene.  This writer therefore records the *real-compatible*
    16-D command and 19-D state stream during AVP operation, while the operator sees
    only a latest-frame stereo preview.  A separate offline replay renders the
    four-camera dataset at the requested simulation timestamps.  The pending
    bundle is deliberately outside ``raw/`` so it can never be mistaken for a
    completed training episode.
    """

    _SCHEMA_VERSION = "team_ramen_flip_table_offline_replay_trajectory/v2"

    def __init__(self, output_root: str | Path, identity: EpisodeIdentity) -> None:
        if identity.backend != "sim":
            raise ValueError("offline replay trajectories are simulator-only")
        self.identity = identity
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.episode_id = f"{stamp}_sim_{identity.dr_profile}_{uuid4().hex[:8]}"
        self._root = root
        self.path = root / f".{self.episode_id}.recording.json"
        self.final_path = root / f"{self.episode_id}.json"
        self._rows: list[dict[str, object]] = []
        self._closed = False

    @property
    def frame_count(self) -> int:
        return len(self._rows)

    @property
    def source_hz(self) -> float:
        if self.frame_count < 2:
            return 0.0
        first = int(self._rows[0]["command_monotonic_ns"])
        last = int(self._rows[-1]["command_monotonic_ns"])
        if last <= first:
            return 0.0
        return float(self.frame_count - 1) / ((last - first) / 1.0e9)

    @staticmethod
    def _state_19d(observation: TeleopObservation) -> list[float]:
        result = state_19d(
            observation.body_joint_position_rad,
            observation.dex1_opening_fraction,
        )
        return [float(value) for value in result]

    @staticmethod
    def _action_16d(target: ArmHandTarget) -> list[float]:
        result = action_16d(target.arm_position_rad, target.dex1_opening_fraction)
        return [float(value) for value in result]

    def append(self, observation: TeleopObservation, target: ArmHandTarget) -> None:
        if self._closed:
            raise RuntimeError("replay trajectory writer is closed")
        if observation.backend != "sim" or target.mode.value != "track":
            raise ValueError("replay trajectories require simulator TRACK commands")
        self._rows.append(
            {
                "command_sequence": target.sequence,
                "command_monotonic_ns": target.monotonic_ns,
                "observation_sequence": observation.sequence,
                "observation_monotonic_ns": observation.capture_monotonic_ns,
                "actions_16d": self._action_16d(target),
                "observed_states_19d": self._state_19d(observation),
                "body_joint_position_rad_29d": [
                    float(value) for value in observation.body_joint_position_rad
                ],
                "body_joint_velocity_rad_s_29d": [
                    float(value) for value in observation.body_joint_velocity_rad_s
                ],
                "root_pose_xyzw": [float(value) for value in observation.root_pose_xyzw],
            }
        )

    def save(self, *, diagnostics: Mapping[str, object], success: bool | None) -> Path:
        if self._closed:
            raise RuntimeError("replay trajectory writer is closed")
        if self.frame_count < 2:
            raise ValueError("a replay trajectory must contain at least two 30 Hz commands")
        accepted = success is True
        destination = self.final_path
        if not accepted:
            destination = self._root / "rejected" / self.final_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
        sim_control_raw = diagnostics.get("sim_control_contract")
        if not isinstance(sim_control_raw, Mapping):
            raise ValueError("sim replay requires sim_control_contract diagnostics")
        sim_control = validate_sim_control_contract(sim_control_raw)
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "backend": "sim",
            "dr_profile": self.identity.dr_profile,
            "seed": self.identity.seed,
            "config_sha256": self.identity.config_sha256,
            "runtime_digest": self.identity.runtime_digest,
            "source_hz": 30,
            "measured_live_command_hz": round(self.source_hz, 3),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_order": STATE_ORDER,
            "action_order": ACTION_ORDER,
            "sim_control_contract": dict(sim_control),
            "actions": [row["actions_16d"] for row in self._rows],
            "observed_states_19d": [row["observed_states_19d"] for row in self._rows],
            "initial_state_19d": self._rows[0]["observed_states_19d"],
            "samples": self._rows,
            "success": success,
            "collection_disposition": "pending_offline_render" if accepted else "rejected_diagnostic",
            "diagnostics": dict(diagnostics),
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(self.path, destination)
        self.final_path = destination
        self._closed = True
        return destination

    def discard(self) -> None:
        if self._closed:
            return
        self.path.unlink(missing_ok=True)
        self._closed = True


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_sim_control_contract(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if result.get("body_mode") != "balanced_wbc":
        raise ValueError("training data requires balanced_wbc body mode")
    if float(result.get("physics_hz", 0.0)) != 200.0 or float(
        result.get("control_hz", 0.0)
    ) != 50.0:
        raise ValueError("training data requires 200 Hz physics and 50 Hz WBC control")
    if float(result.get("wbc_base_height_m", 0.0)) != 0.74:
        raise ValueError("WBC base-height contract must remain 0.74 m")
    if result.get("wbc_navigation_velocity_m_s_rad_s") != [0.0, 0.0, 0.0]:
        raise ValueError("WBC navigation command must remain zero")
    if result.get("wbc_torso_rpy_rad") != [0.0, 0.0, 0.0]:
        raise ValueError("WBC torso command must remain zero")
    for key in (
        "wbc_stand_onnx_sha256",
        "wbc_walk_onnx_sha256",
        "team_adapter_sha256",
    ):
        digest = result.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return result


def validate_camera_jpeg(payload: bytes, role: str) -> None:
    """Reject malformed, resized, or non-color frames before writing an episode."""

    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ValueError(f"{role} frame is not JPEG")
            if image.size != (640, 480):
                raise ValueError(f"{role} frame must be 640x480, got {image.size}")
            if image.mode != "RGB":
                raise ValueError(f"{role} frame must be color RGB, got {image.mode}")
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{role} frame is not a valid JPEG") from exc


@dataclass(frozen=True)
class EpisodeIdentity:
    backend: str
    dr_profile: str
    seed: int
    config_sha256: str
    runtime_digest: str


class RawEpisodeWriter:
    def __init__(
        self,
        output_root: str | Path,
        identity: EpisodeIdentity,
        *,
        maximum_camera_skew_s: float = 1.0 / CAMERA_HZ,
        maximum_camera_age_s: float = DEFAULT_MAXIMUM_CAMERA_AGE_S,
    ) -> None:
        if identity.backend not in {"sim", "real"}:
            raise ValueError("episode backend must be sim or real")
        if identity.dr_profile not in {"mild", "medium", "full", "real"}:
            raise ValueError("invalid episode DR profile")
        if maximum_camera_skew_s <= 0.0:
            raise ValueError("maximum camera skew must be positive")
        if maximum_camera_age_s < maximum_camera_skew_s:
            raise ValueError("maximum camera age must not be shorter than maximum skew")
        self.identity = identity
        self.maximum_camera_skew_ns = int(maximum_camera_skew_s * 1.0e9)
        self.maximum_camera_age_ns = int(maximum_camera_age_s * 1.0e9)
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.episode_id = f"{stamp}_{identity.backend}_{identity.dr_profile}_{uuid4().hex[:8]}"
        self.final_path = root / self.episode_id
        self.path = root / f".{self.episode_id}.recording"
        self.path.mkdir()
        for role in POLICY_CAMERA_ROLE_TO_KEY:
            (self.path / "policy_cameras" / role).mkdir(parents=True)
        (self.path / "diagnostics" / "head_right").mkdir(parents=True)
        self._trace = (self.path / "frames.jsonl").open("x", encoding="utf-8")
        self._frame_count = 0
        self._observation_times_ns: list[int] = []
        self._previous_camera_sha256: dict[str, str] = {}
        self._consecutive_camera_duplicates: dict[str, int] = {
            role: 0 for role in REQUIRED_CAMERA_ROLES
        }
        self._diagnostic_camera_roles: set[str] = set()
        self._sim_control_contract: dict[str, object] | None = None
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def source_hz(self) -> float:
        if len(self._observation_times_ns) < 2:
            return 0.0
        elapsed_s = (
            self._observation_times_ns[-1] - self._observation_times_ns[0]
        ) / 1.0e9
        if elapsed_s <= 0.0:
            return 0.0
        return (len(self._observation_times_ns) - 1) / elapsed_s

    @property
    def camera_duplicate_fractions(self) -> dict[str, float]:
        denominator = max(1, self._frame_count - 1)
        return {
            role: count / denominator
            for role, count in self._consecutive_camera_duplicates.items()
        }

    def append(self, observation: TeleopObservation, target: ArmHandTarget) -> None:
        if self._closed:
            raise RuntimeError("episode writer is closed")
        if observation.backend != self.identity.backend:
            raise ValueError("observation backend differs from episode identity")
        required_cameras = set(REQUIRED_CAMERA_ROLES)
        if set(observation.camera_jpeg) != required_cameras:
            raise ValueError("recording requires synchronized head stereo and both wrist cameras")
        if not observation.camera_bundle_valid:
            raise FrameSynchronizationError(
                "camera bundle is not a new synchronized physical sample "
                f"(skew_ms={observation.camera_skew_ms:.3f}, "
                f"stale_roles={list(observation.stale_roles)})"
            )
        for role, payload in observation.camera_jpeg.items():
            validate_camera_jpeg(payload, role)
        camera_times = tuple(observation.camera_capture_monotonic_ns.values())
        if max(camera_times) - min(camera_times) > self.maximum_camera_skew_ns:
            raise FrameSynchronizationError(
                "camera timestamps exceed the 30 Hz synchronization tolerance: "
                f"skew_ms={(max(camera_times) - min(camera_times)) / 1.0e6:.3f}, "
                f"limit_ms={self.maximum_camera_skew_ns / 1.0e6:.3f}"
            )
        camera_ages_ns = {
            role: observation.capture_monotonic_ns - timestamp
            for role, timestamp in observation.camera_capture_monotonic_ns.items()
        }
        if any(
            age_ns < 0 or age_ns > self.maximum_camera_age_ns
            for age_ns in camera_ages_ns.values()
        ):
            ages_ms = {
                role: round(age_ns / 1.0e6, 3)
                for role, age_ns in camera_ages_ns.items()
            }
            raise FrameSynchronizationError(
                "policy camera timestamp is in the future or too old: "
                f"ages_ms={ages_ms}, "
                f"allowed_ms=[0,{self.maximum_camera_age_ns / 1.0e6:.3f}]"
            )
        diagnostic_times = tuple(
            observation.diagnostic_camera_capture_monotonic_ns.values()
        )
        if any(
            timestamp > observation.capture_monotonic_ns
            or observation.capture_monotonic_ns - timestamp > self.maximum_camera_age_ns
            for timestamp in diagnostic_times
        ):
            raise FrameSynchronizationError(
                "diagnostic camera timestamp is in the future or too old"
            )
        if abs(observation.capture_monotonic_ns - target.monotonic_ns) > 2 * self.maximum_camera_skew_ns:
            raise FrameSynchronizationError(
                "command and observation timestamps are not synchronized: "
                f"delta_ms={(target.monotonic_ns - observation.capture_monotonic_ns) / 1.0e6:.3f}, "
                f"limit_ms={2 * self.maximum_camera_skew_ns / 1.0e6:.3f}"
            )
        privileged = observation.diagnostics.get("privileged_policy_features", [])
        if privileged != []:
            raise ValueError("simulator-only values must not enter policy features")
        if observation.backend == "sim":
            control = observation.diagnostics.get("sim_control_contract")
            if not isinstance(control, Mapping):
                raise ValueError("sim observation requires sim_control_contract")
            normalized_control = validate_sim_control_contract(control)
            if self._sim_control_contract is None:
                self._sim_control_contract = normalized_control
            elif normalized_control != self._sim_control_contract:
                raise ValueError("sim_control_contract changed inside one episode")
        if observation.backend == "real":
            dex1_stale = observation.diagnostics.get(
                "dex1_state_stale_left_right", (False, False)
            )
            if (
                not isinstance(dex1_stale, (list, tuple))
                or len(dex1_stale) != 2
                or any(bool(value) for value in dex1_stale)
            ):
                raise FrameSynchronizationError(
                    "real Dex1 feedback was stale during dataset capture"
                )

        index = self._frame_count
        camera_records = {}
        for role, key in POLICY_CAMERA_ROLE_TO_KEY.items():
            payload = observation.camera_jpeg[role]
            digest = _sha256(payload)
            if self._previous_camera_sha256.get(role) == digest:
                self._consecutive_camera_duplicates[role] += 1
            self._previous_camera_sha256[role] = digest
            relative = Path("policy_cameras") / role / f"{index:06d}.jpg"
            (self.path / relative).write_bytes(payload)
            camera_records[key] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "capture_monotonic_ns": observation.camera_capture_monotonic_ns[role],
            }
        head_right = observation.camera_jpeg["head_right"]
        head_right_digest = _sha256(head_right)
        if self._previous_camera_sha256.get("head_right") == head_right_digest:
            self._consecutive_camera_duplicates["head_right"] += 1
        self._previous_camera_sha256["head_right"] = head_right_digest
        right_relative = Path("diagnostics") / "head_right" / f"{index:06d}.jpg"
        (self.path / right_relative).write_bytes(head_right)

        diagnostic_cameras = {
            "head_right": {
                "path": right_relative.as_posix(),
                "sha256": head_right_digest,
                "capture_monotonic_ns": observation.camera_capture_monotonic_ns[
                    "head_right"
                ],
            }
        }
        for role, payload in observation.diagnostic_camera_jpeg.items():
            relative = Path("diagnostics") / role / f"{index:06d}.jpg"
            (self.path / relative.parent).mkdir(parents=True, exist_ok=True)
            (self.path / relative).write_bytes(payload)
            self._diagnostic_camera_roles.add(role)
            diagnostic_cameras[role] = {
                "path": relative.as_posix(),
                "sha256": _sha256(payload),
                "capture_monotonic_ns": (
                    observation.diagnostic_camera_capture_monotonic_ns[role]
                ),
            }

        runtime_diagnostics = {
            key: value
            for key, value in observation.diagnostics.items()
            if key not in {"randomization", "privileged_policy_features"}
        }

        record = {
            "frame_index": index,
            "observation_sequence": observation.sequence,
            "observation_monotonic_ns": observation.capture_monotonic_ns,
            "camera_bundle_valid": observation.camera_bundle_valid,
            "camera_skew_ms": observation.camera_skew_ms,
            "camera_stream_metadata": {
                role: dict(metadata)
                for role, metadata in observation.camera_stream_metadata.items()
            },
            "command_sequence": target.sequence,
            "command_monotonic_ns": target.monotonic_ns,
            "body_joint_position_rad": list(observation.body_joint_position_rad),
            "body_joint_velocity_rad_s": list(observation.body_joint_velocity_rad_s),
            "root_pose_xyzw": list(observation.root_pose_xyzw),
            "dex1_opening_state": list(observation.dex1_opening_fraction),
            "commanded_arm_target_rad": list(target.arm_position_rad),
            "commanded_arm_feedforward_torque_nm": list(
                target.arm_feedforward_torque_nm
            ),
            "commanded_dex1_opening_target": list(target.dex1_opening_fraction),
            "applied_arm_target_rad": list(observation.applied_arm_target_rad),
            "applied_dex1_opening_target": list(
                observation.applied_dex1_opening_target
            ),
            "requested_command": {
                "mode": target.mode.value,
                "event": target.event.value,
                "arm_target_rad": list(target.arm_position_rad),
                "arm_feedforward_torque_nm": list(
                    target.arm_feedforward_torque_nm
                ),
                "dex1_opening_target": list(target.dex1_opening_fraction),
            },
            "policy_cameras": camera_records,
            "diagnostics": {
                "cameras": diagnostic_cameras,
                "sim": runtime_diagnostics if observation.backend == "sim" else {},
                "real": runtime_diagnostics if observation.backend == "real" else {},
            },
        }
        self._trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._trace.flush()
        os.fsync(self._trace.fileno())
        self._observation_times_ns.append(observation.capture_monotonic_ns)
        self._frame_count += 1

    def save(self, *, diagnostics: Mapping[str, object], success: bool | None) -> Path:
        if self._closed:
            raise RuntimeError("episode writer is closed")
        if self._frame_count < 2:
            raise ValueError("an episode must contain at least two synchronized frames")
        self._trace.close()
        accepted = success is True
        final_path = self._root / self.episode_id
        if not accepted:
            # Failed simulator trials are valuable for debugging/reset and
            # curriculum analysis, but they must never be discovered by the
            # successful-demo glob at ``raw/*/manifest.json``.  Keep them in a
            # separate immutable diagnostic namespace.
            final_path = self._root / "rejected" / self.episode_id
            final_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": RAW_EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "backend": self.identity.backend,
            "dr_profile": self.identity.dr_profile,
            "seed": self.identity.seed,
            "config_sha256": self.identity.config_sha256,
            "runtime_digest": self.identity.runtime_digest,
            "fps": 30,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_order": STATE_ORDER,
            "action_order": ACTION_ORDER,
            "frame_count": self._frame_count,
            "camera_frame_contract": {
                "encoding": "JPEG",
                "width": 640,
                "height": 480,
                "color_mode": "RGB",
            },
            "success": success,
            "collection_disposition": "accepted" if accepted else "rejected_diagnostic",
            "policy_camera_keys": list(POLICY_CAMERA_ROLE_TO_KEY.values()),
            "operator_only_cameras": ["head_right"],
            "diagnostic_cameras": sorted(self._diagnostic_camera_roles),
            "privileged_policy_features": [],
            "diagnostics": dict(diagnostics),
        }
        if self.identity.backend == "sim":
            sim_control = diagnostics.get(
                "sim_control_contract", self._sim_control_contract
            )
            if not isinstance(sim_control, Mapping):
                raise ValueError("sim episode requires sim_control_contract diagnostics")
            manifest["sim_control_contract"] = validate_sim_control_contract(
                sim_control
            )
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(self.path, final_path)
        self.final_path = final_path
        self._closed = True
        return self.final_path

    def discard(self) -> None:
        if self._closed:
            return
        self._trace.close()
        shutil.rmtree(self.path, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "RawEpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            self.discard()
