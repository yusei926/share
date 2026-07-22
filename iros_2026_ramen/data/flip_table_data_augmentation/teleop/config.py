"""Strict configuration for the shared real/simulator teleoperation path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "team_ramen_flip_table_teleop/v1"
DEFAULT_TELEOP_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "teleop_v1.json"
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
POLICY_CAMERA_KEYS = (
    "observation.images.cam_0",
    "observation.images.cam_2",
    "observation.images.cam_3",
)
OPERATOR_CAMERA_ROLES = ("head_left", "head_right")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: dict[str, Any], label: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _range(value: Any, label: str, size: int = 2) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    if size == 2 and result[0] > result[1]:
        raise ValueError(f"{label} must be ordered")
    return result


@dataclass(frozen=True)
class RuntimePin:
    xr_repo: str
    xr_revision: str
    televuer_revision: str
    robofinals_image: str
    robofinals_digest: str


@dataclass(frozen=True)
class Rates:
    physics_hz: int
    servo_hz: int
    command_hz: int
    camera_hz: int
    record_hz: int


@dataclass(frozen=True)
class SafetyLimits:
    arm_position_lower_rad: tuple[float, ...]
    arm_position_upper_rad: tuple[float, ...]
    arm_velocity_rad_s: float
    arm_acceleration_rad_s2: float
    hand_velocity_fraction_s: float
    hand_acceleration_fraction_s2: float
    command_hold_timeout_s: float
    command_stop_timeout_s: float


@dataclass(frozen=True)
class CameraRandomization:
    mount_translation_m: float
    mount_rotation_rad: float
    focal_fraction: float
    principal_point_px: float
    distortion_fraction: float
    exposure_ev: float


@dataclass(frozen=True)
class DrProfile:
    name: str
    level: float
    successful_demos: int


@dataclass(frozen=True)
class Collection:
    output_root: str
    dataset_repo_id: str
    private: bool
    target_successful_sim_demos: int
    mimic_pilot_trials: int
    mimic_pilot_success_min: int
    mimic_successful_trajectories_min: int
    appearance_variants_per_trajectory_min: int
    video_shard_size_mb: int
    profiles: tuple[DrProfile, ...]


@dataclass(frozen=True)
class Workstation:
    tailscale_host: str
    ssh_user: str
    remote_repo: str
    sim_port: int


@dataclass(frozen=True)
class TeleopConfig:
    path: Path
    digest: str
    runtime: RuntimePin
    rates: Rates
    safety: SafetyLimits
    camera_randomization: CameraRandomization
    collection: Collection
    workstation: Workstation
    arm_joint_names: tuple[str, ...]
    policy_camera_keys: tuple[str, ...]
    operator_camera_roles: tuple[str, ...]


def _sha256(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def load_teleop_config(path: str | Path = DEFAULT_TELEOP_CONFIG_PATH) -> TeleopConfig:
    config_path = Path(path).expanduser().resolve()
    root = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "config")
    _exact(
        root,
        "config",
        {
            "schema_version",
            "runtime",
            "rates",
            "command_contract",
            "safety",
            "camera_randomization",
            "collection",
            "workstation",
        },
    )
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported teleop schema: {root.get('schema_version')!r}")

    runtime_raw = _mapping(root["runtime"], "runtime")
    _exact(runtime_raw, "runtime", set(RuntimePin.__dataclass_fields__))
    runtime = RuntimePin(**{key: _text(value, f"runtime.{key}") for key, value in runtime_raw.items()})
    if len(runtime.xr_revision) != 40 or len(runtime.televuer_revision) != 40:
        raise ValueError("XR and TeleVuer revisions must be full commit SHAs")
    if not runtime.robofinals_digest.startswith("sha256:") or len(runtime.robofinals_digest) != 71:
        raise ValueError("RoboFinals digest must be a sha256 OCI digest")

    rates_raw = _mapping(root["rates"], "rates")
    _exact(rates_raw, "rates", set(Rates.__dataclass_fields__))
    rates = Rates(**{key: int(_positive(value, f"rates.{key}")) for key, value in rates_raw.items()})
    if rates.physics_hz != 100 or rates.servo_hz != 50:
        raise ValueError("interactive sim physics/servo rates must remain 100/50 Hz")
    if len({rates.command_hz, rates.camera_hz, rates.record_hz}) != 1 or rates.record_hz != 30:
        raise ValueError("AVP commands, cameras, and recording must share the 30 Hz clock")

    command_raw = _mapping(root["command_contract"], "command_contract")
    _exact(command_raw, "command_contract", {"arm_joint_names", "hand_target", "policy_cameras", "operator_cameras"})
    arm_names = tuple(command_raw["arm_joint_names"])
    policy_cameras = tuple(command_raw["policy_cameras"])
    operator_cameras = tuple(command_raw["operator_cameras"])
    if arm_names != ARM_JOINT_NAMES:
        raise ValueError("14-D arm command ordering differs from official G1_29")
    if command_raw["hand_target"] != "dex1_opening_fraction_left_right":
        raise ValueError("Dex1 target must be normalized left/right opening fraction")
    if policy_cameras != POLICY_CAMERA_KEYS:
        raise ValueError("policy cameras must remain cam_0/cam_2/cam_3")
    if operator_cameras != OPERATOR_CAMERA_ROLES:
        raise ValueError("operator display must use true head stereo")

    safety_raw = _mapping(root["safety"], "safety")
    _exact(safety_raw, "safety", set(SafetyLimits.__dataclass_fields__))
    lower = _range(safety_raw["arm_position_lower_rad"], "safety.arm_position_lower_rad", 14)
    upper = _range(safety_raw["arm_position_upper_rad"], "safety.arm_position_upper_rad", 14)
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("arm position limits must have positive intervals")
    safety = SafetyLimits(
        arm_position_lower_rad=lower,
        arm_position_upper_rad=upper,
        **{
            key: _positive(safety_raw[key], f"safety.{key}")
            for key in set(SafetyLimits.__dataclass_fields__) - {"arm_position_lower_rad", "arm_position_upper_rad"}
        },
    )
    if safety.command_hold_timeout_s >= safety.command_stop_timeout_s:
        raise ValueError("command hold timeout must precede stop timeout")

    camera_raw = _mapping(root["camera_randomization"], "camera_randomization")
    _exact(camera_raw, "camera_randomization", set(CameraRandomization.__dataclass_fields__))
    camera_randomization = CameraRandomization(
        **{key: _positive(value, f"camera_randomization.{key}") for key, value in camera_raw.items()}
    )
    if camera_randomization.mount_translation_m > 0.003 + 1e-12:
        raise ValueError("camera mount translation randomization exceeds 3 mm")
    if camera_randomization.mount_rotation_rad > math.radians(1.0) + 1e-12:
        raise ValueError("camera mount rotation randomization exceeds one degree")

    collection_raw = _mapping(root["collection"], "collection")
    expected_collection = set(Collection.__dataclass_fields__) - {"profiles"} | {"dr_profiles"}
    _exact(collection_raw, "collection", expected_collection)
    profile_values = collection_raw["dr_profiles"]
    if not isinstance(profile_values, list):
        raise ValueError("collection.dr_profiles must be an array")
    profiles = []
    for index, value in enumerate(profile_values):
        item = _mapping(value, f"collection.dr_profiles[{index}]")
        _exact(item, f"collection.dr_profiles[{index}]", set(DrProfile.__dataclass_fields__))
        profile = DrProfile(
            name=_text(item["name"], f"profile[{index}].name"),
            level=float(item["level"]),
            successful_demos=int(item["successful_demos"]),
        )
        if not 0.0 <= profile.level <= 1.0 or profile.successful_demos <= 0:
            raise ValueError("DR profile level/count is invalid")
        profiles.append(profile)
    if tuple(profile.name for profile in profiles) != ("mild", "medium", "full"):
        raise ValueError("DR profiles must be mild, medium, full")
    collection = Collection(
        output_root=_text(collection_raw["output_root"], "collection.output_root"),
        dataset_repo_id=_text(collection_raw["dataset_repo_id"], "collection.dataset_repo_id"),
        private=collection_raw["private"],
        target_successful_sim_demos=int(collection_raw["target_successful_sim_demos"]),
        mimic_pilot_trials=int(collection_raw["mimic_pilot_trials"]),
        mimic_pilot_success_min=int(collection_raw["mimic_pilot_success_min"]),
        mimic_successful_trajectories_min=int(collection_raw["mimic_successful_trajectories_min"]),
        appearance_variants_per_trajectory_min=int(collection_raw["appearance_variants_per_trajectory_min"]),
        video_shard_size_mb=int(collection_raw["video_shard_size_mb"]),
        profiles=tuple(profiles),
    )
    if collection.private is not True:
        raise ValueError("augmented dataset must remain private")
    if sum(profile.successful_demos for profile in profiles) != collection.target_successful_sim_demos:
        raise ValueError("DR profile counts must sum to the sim demo target")
    if (
        collection.target_successful_sim_demos != 30
        or collection.mimic_pilot_trials != 100
        or collection.mimic_pilot_success_min < 50
        or collection.mimic_successful_trajectories_min < 2000
        or collection.appearance_variants_per_trajectory_min < 2
        or collection.video_shard_size_mb != 500
    ):
        raise ValueError("collection release gates differ from the approved plan")

    workstation_raw = _mapping(root["workstation"], "workstation")
    _exact(workstation_raw, "workstation", set(Workstation.__dataclass_fields__))
    workstation = Workstation(
        tailscale_host=_text(workstation_raw["tailscale_host"], "workstation.tailscale_host"),
        ssh_user=_text(workstation_raw["ssh_user"], "workstation.ssh_user"),
        remote_repo=_text(workstation_raw["remote_repo"], "workstation.remote_repo"),
        sim_port=int(_positive(workstation_raw["sim_port"], "workstation.sim_port")),
    )
    if not 1024 <= workstation.sim_port <= 65535:
        raise ValueError("sim port must be an unprivileged TCP port")

    return TeleopConfig(
        path=config_path,
        digest=_sha256(root),
        runtime=runtime,
        rates=rates,
        safety=safety,
        camera_randomization=camera_randomization,
        collection=collection,
        workstation=workstation,
        arm_joint_names=arm_names,
        policy_camera_keys=policy_cameras,
        operator_camera_roles=operator_cameras,
    )
