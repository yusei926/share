"""Strict configuration loading for the augmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "team_ramen_flip_table_augmentation_pipeline/v2"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pipeline_v1.json"
EXPECTED_CAMERA_KEYS = (
    "observation.images.cam_0",
    "observation.images.cam_2",
    "observation.images.cam_3",
)
EXPECTED_POLICY_CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
EXPECTED_SUBTASKS = (
    "pre_grasp",
    "grasp",
    "lift",
    "rotate_180",
    "settle",
    "release",
    "retreat",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], label: str, expected: set[str]) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))


def _finite_range(value: Any, label: str, *, minimum: float | None = None) -> tuple[float, float]:
    result = _finite_vector(value, 2, label)
    if result[0] > result[1]:
        raise ValueError(f"{label} must be ordered [minimum, maximum]")
    if minimum is not None and result[0] < minimum:
        raise ValueError(f"{label} cannot contain values below {minimum}")
    return result


@dataclass(frozen=True)
class SourceConfig:
    repo_id: str
    revision: str
    episodes: int
    frames: int
    fps: int


@dataclass(frozen=True)
class RawSourceConfig:
    repo_id: str
    revision: str
    episodes: int
    head_stereo_calibration_sha256: str
    head_stereo_calibration_repo_path: str
    head_stereo_baseline_m: float
    head_stereo_rms_error_px: float
    wrist_calibration_sha256_by_serial: dict[str, str]


@dataclass(frozen=True)
class TargetConfig:
    repo_id: str
    private: bool
    data_shard_size_mb: int
    video_shard_size_mb: int


@dataclass(frozen=True)
class RuntimeConfig:
    container_image: str
    container_digest: str
    image_created_utc: str
    isaac_sim_version: str
    isaac_lab_version: str
    isaac_lab_mimic_version: str
    isaac_lab_mimic_source_sha256: str
    dex1_gripper_python_sha256: str
    replicator_version: str
    robot: str
    physics_backend: str


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    lerobot_version: str
    lerobot_revision: str
    lerobot_wheel_sha256: str
    python_version: str
    pypi_release_utc: str


@dataclass(frozen=True)
class ObjectPoseRuntimeConfig:
    method: str
    foundationpose_repo: str
    foundationpose_revision: str
    foundationpose_license: str
    pytorch3d_repo: str
    pytorch3d_revision: str
    pytorch3d_license: str
    nvdiffrast_repo: str
    nvdiffrast_revision: str
    nvdiffrast_license: str
    fast_stereo_repo: str
    fast_stereo_revision: str
    fast_stereo_license: str
    fast_stereo_model_repo: str
    fast_stereo_model_revision: str
    fast_stereo_model_filename: str
    fast_stereo_model_sha256: str
    fast_stereo_model_size_bytes: int
    fast_stereo_config_filename: str
    fast_stereo_config_sha256: str
    fast_stereo_config_size_bytes: int
    fast_stereo_model_license: str
    timm_version: str
    timm_wheel_sha256: str
    fast_stereo_valid_iterations: int
    fast_stereo_max_disparity_px: int
    fast_stereo_maximum_left_right_error_px: float
    assembled_table_mesh_sha256: str
    assembled_table_mesh_vertices: int
    assembled_table_mesh_triangles: int
    foundationpose_refiner_config_file_id: str
    foundationpose_refiner_config_sha256: str
    foundationpose_refiner_config_size_bytes: int
    foundationpose_refiner_checkpoint_file_id: str
    foundationpose_refiner_checkpoint_sha256: str
    foundationpose_refiner_checkpoint_size_bytes: int
    foundationpose_scorer_config_file_id: str
    foundationpose_scorer_config_sha256: str
    foundationpose_scorer_config_size_bytes: int
    foundationpose_scorer_checkpoint_file_id: str
    foundationpose_scorer_checkpoint_sha256: str
    foundationpose_scorer_checkpoint_size_bytes: int
    detector_repo: str
    detector_revision: str
    detector_checkpoint_filename: str
    detector_checkpoint_sha256: str
    detector_checkpoint_size_bytes: int
    detector_license: str
    segmentation_repo: str
    segmentation_revision: str
    segmentation_checkpoint_filename: str
    segmentation_checkpoint_sha256: str
    segmentation_checkpoint_size_bytes: int
    segmentation_license: str
    transformers_version: str
    transformers_wheel_sha256: str
    robot_visual_urdf_relative_path: str
    robot_visual_urdf_sha256: str
    text_prompt: str
    detector_box_threshold: float
    detector_text_threshold: float
    segmentation_iou_threshold: float
    dense_mask_min_bidirectional_iou: float
    dense_mask_min_area_fraction: float
    dense_mask_max_area_fraction: float
    min_mask_area_fraction: float
    max_mask_area_fraction: float
    minimum_registration_mask_coverage: float
    max_detection_candidates: int
    source_frame_stride: int
    registration_interval_source_frames: int
    registration_refine_iterations: int
    tracking_refine_iterations: int
    temporal_propagation_beam_size: int
    dense_temporal_propagation_beam_size: int
    robot_silhouette_dilation_px: int
    auxiliary_robot_silhouette_dilation_px: int
    auxiliary_view_score_weight: float
    auxiliary_primary_support_saturation_fraction: float
    table_mask_minimum_value: int
    table_mask_minimum_component_area_px: int
    primary_mask_maximum_median_eef_distance_m: float
    primary_mask_minimum_component_valid_depth_fraction: float
    table_prompt_maximum_distance_px: int
    table_prompt_principal_point_weight: float
    mask_sequence_detector_weight: float
    mask_sequence_segmentation_weight: float
    mask_sequence_retention_weight: float
    mask_sequence_transition_iou_weight: float
    mask_sequence_primary_area_weight: float
    mask_sequence_primary_fragmentation_weight: float
    mask_sequence_primary_transition_area_weight: float
    mask_sequence_auxiliary_area_weight: float
    mask_sequence_auxiliary_fragmentation_weight: float
    mask_sequence_auxiliary_transition_area_weight: float
    maximum_terminal_tracking_gap_source_frames: int
    maximum_initial_static_backfill_source_frames: int
    maximum_temporal_evidence_gap_source_frames: int
    min_valid_depth_fraction_in_mask: float
    max_bidirectional_translation_error_m: float
    max_bidirectional_rotation_error_rad: float
    max_registration_correction_translation_m: float
    max_registration_correction_rotation_rad: float
    temporal_candidate_min_mask_precision: float
    temporal_candidate_min_mask_explained_fraction: float
    temporal_candidate_max_depth_error_m: float
    temporal_static_translation_scale_m: float
    temporal_static_rotation_scale_rad: float
    temporal_max_static_translation_step_m: float
    temporal_max_static_rotation_step_rad: float
    temporal_grasp_translation_scale_m: float
    temporal_grasp_rotation_scale_rad: float
    temporal_max_table_speed_m_s: float
    temporal_max_table_angular_speed_rad_s: float
    temporal_max_grasp_relative_translation_step_m: float
    temporal_max_grasp_relative_rotation_step_rad: float
    temporal_grasp_observed_position_max: float
    temporal_blurred_frame_laplacian_variance_max: float
    temporal_carry_min_mask_precision: float
    temporal_carry_min_mask_explained_fraction: float
    temporal_carry_min_auxiliary_explained_fraction: float
    temporal_carry_min_multiview_score: float
    temporal_carry_visual_penalty: float
    temporal_min_stereo_consistent_fraction: float
    max_rendered_depth_median_abs_error_m: float
    min_rendered_depth_overlap_fraction: float
    min_rendered_mask_explained_fraction: float


@dataclass(frozen=True)
class CameraConfig:
    source_key: str
    policy_key: str
    sim_sensor: str
    prim_path: str
    offset_position_m: tuple[float, float, float]
    offset_quaternion_xyzw: tuple[float, float, float, float]
    convention: str
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    intrinsic_matrix_px: tuple[float, ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    intrinsic_calibration_sha256s: tuple[str, ...]
    clipping_range_m: tuple[float, float]
    calibration_basis: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class ContactMaterialConfig:
    static_friction: tuple[float, float]
    dynamic_friction: tuple[float, float]
    restitution: tuple[float, float]


@dataclass(frozen=True)
class PhysicalRandomizationConfig:
    table_long_range_m: float
    table_depth_range_m: float
    table_yaw_range_rad: float
    robot_distance_m: float
    robot_distance_range_m: float
    robot_table_min_distance_m: float
    robot_lateral_range_m: float
    robot_yaw_range_rad: float
    upper_body_joint_noise_rad: float
    dex1_finger_noise_m: float
    table_part_mass_scale: tuple[float, float]
    contact_materials: dict[str, ContactMaterialConfig]


@dataclass(frozen=True)
class AppearanceRandomizationConfig:
    variant_seed_stride: int
    nominal_camera_variant_index: int
    camera_translation_jitter_m_max: float
    camera_rotation_jitter_rad_max: float
    exposure_ev: tuple[float, float]
    color_temperature_k: tuple[float, float]
    distant_light_intensity: tuple[float, float]
    sphere_light_intensity: tuple[float, float]
    floor_materials: tuple[str, ...]
    wall_materials: tuple[str, ...]
    room_props: tuple[str, ...]
    room_prop_visible_probability: float


@dataclass(frozen=True)
class AutomaticPhaseConfig:
    smoothing_window_frames: int
    endpoint_window_frames: int
    minimum_phase_frames: int
    sustained_event_frames: int
    minimum_flip_angle_rad: float
    minimum_table_displacement_m: float
    motion_translation_threshold_m: float
    motion_rotation_threshold_rad: float
    rotate_start_fraction: float
    rotate_end_fraction: float
    grasp_closed_fraction_min: float
    release_open_fraction_min: float
    settled_linear_speed_m_s_max: float
    settled_angular_speed_rad_s_max: float
    retreat_relative_displacement_m_min: float
    hold_relative_position_p95_m_max: float
    rotation_progress_reversal_fraction_max: float


@dataclass(frozen=True)
class SourceAnnotationConfig:
    automatic_phase: AutomaticPhaseConfig


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    raw: dict[str, Any]
    source: SourceConfig
    raw_source: RawSourceConfig
    target: TargetConfig
    runtime: RuntimeConfig
    dataset_runtime: DatasetRuntimeConfig
    object_pose_runtime: ObjectPoseRuntimeConfig
    cameras: tuple[CameraConfig, ...]
    physical_randomization: PhysicalRandomizationConfig
    appearance_randomization: AppearanceRandomizationConfig
    source_annotation: SourceAnnotationConfig
    subtasks: tuple[str, ...]
    digest: str


def canonical_json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    source_path = Path(path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {root.get('schema_version')!r}")
    if "cosmos" in root or "cosmos_transfer" in root:
        raise ValueError("Cosmos is not part of the canonical augmentation pipeline")
    _require_exact_keys(
        root,
        "config",
        {
            "schema_version",
            "source",
            "raw_source",
            "target",
            "runtime",
            "dataset_runtime",
            "object_pose_runtime",
            "cameras",
            "source_contract",
            "source_annotation",
            "subtasks",
            "generation",
            "physical_randomization",
            "appearance_randomization",
            "success",
            "splits",
        },
    )

    source_raw = _mapping(root.get("source"), "source")
    _require_exact_keys(source_raw, "source", set(SourceConfig.__dataclass_fields__))
    source = SourceConfig(
        repo_id=_string(source_raw.get("repo_id"), "source.repo_id"),
        revision=_string(source_raw.get("revision"), "source.revision"),
        episodes=_positive_int(source_raw.get("episodes"), "source.episodes"),
        frames=_positive_int(source_raw.get("frames"), "source.frames"),
        fps=_positive_int(source_raw.get("fps"), "source.fps"),
    )
    if len(source.revision) != 40 or any(ch not in "0123456789abcdef" for ch in source.revision):
        raise ValueError("source.revision must be a full lowercase Git commit SHA")

    raw_source_raw = _mapping(root.get("raw_source"), "raw_source")
    _require_exact_keys(raw_source_raw, "raw_source", set(RawSourceConfig.__dataclass_fields__))
    wrist_hashes_raw = _mapping(
        raw_source_raw.get("wrist_calibration_sha256_by_serial"),
        "raw_source.wrist_calibration_sha256_by_serial",
    )
    wrist_hashes = {
        _string(serial, "raw_source wrist serial"): _string(
            digest, f"raw_source wrist calibration hash for {serial}"
        )
        for serial, digest in wrist_hashes_raw.items()
    }
    raw_source = RawSourceConfig(
        repo_id=_string(raw_source_raw.get("repo_id"), "raw_source.repo_id"),
        revision=_string(raw_source_raw.get("revision"), "raw_source.revision"),
        episodes=_positive_int(raw_source_raw.get("episodes"), "raw_source.episodes"),
        head_stereo_calibration_sha256=_string(
            raw_source_raw.get("head_stereo_calibration_sha256"),
            "raw_source.head_stereo_calibration_sha256",
        ),
        head_stereo_calibration_repo_path=_string(
            raw_source_raw.get("head_stereo_calibration_repo_path"),
            "raw_source.head_stereo_calibration_repo_path",
        ),
        head_stereo_baseline_m=_finite_float(
            raw_source_raw.get("head_stereo_baseline_m"),
            "raw_source.head_stereo_baseline_m",
        ),
        head_stereo_rms_error_px=_finite_float(
            raw_source_raw.get("head_stereo_rms_error_px"),
            "raw_source.head_stereo_rms_error_px",
        ),
        wrist_calibration_sha256_by_serial=wrist_hashes,
    )
    for label, digest, expected_length in (
        ("revision", raw_source.revision, 40),
        ("head_stereo_calibration_sha256", raw_source.head_stereo_calibration_sha256, 64),
    ):
        if len(digest) != expected_length or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"raw_source.{label} must be a full lowercase hexadecimal digest")
    if raw_source.repo_id == source.repo_id:
        raise ValueError("raw_source and LeRobot source repositories must differ")
    calibration_path = Path(raw_source.head_stereo_calibration_repo_path)
    if calibration_path.is_absolute() or ".." in calibration_path.parts:
        raise ValueError("raw_source head stereo calibration path must remain repository-relative")
    if len(raw_source.wrist_calibration_sha256_by_serial) != 2:
        raise ValueError("raw_source must identify exactly two D405 wrist calibrations")
    if raw_source.head_stereo_baseline_m <= 0.0 or raw_source.head_stereo_rms_error_px <= 0.0:
        raise ValueError("raw_source head stereo baseline and RMS must be positive")
    for serial, digest in raw_source.wrist_calibration_sha256_by_serial.items():
        if not serial.isdecimal():
            raise ValueError("raw_source D405 serial numbers must be decimal strings")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"raw_source wrist calibration hash for {serial} must be SHA-256")

    target_raw = _mapping(root.get("target"), "target")
    _require_exact_keys(target_raw, "target", set(TargetConfig.__dataclass_fields__))
    target = TargetConfig(
        repo_id=_string(target_raw.get("repo_id"), "target.repo_id"),
        private=target_raw.get("private"),
        data_shard_size_mb=_positive_int(
            target_raw.get("data_shard_size_mb"), "target.data_shard_size_mb"
        ),
        video_shard_size_mb=_positive_int(
            target_raw.get("video_shard_size_mb"), "target.video_shard_size_mb"
        ),
    )
    if not isinstance(target.private, bool) or not target.private:
        raise ValueError("target.private must be true")
    if source.repo_id == target.repo_id:
        raise ValueError("source and target repositories must differ")
    if (target.data_shard_size_mb, target.video_shard_size_mb) != (100, 500):
        raise ValueError("LeRobot shards must use the agreed 100 MB data / 500 MB video targets")

    runtime_raw = _mapping(root.get("runtime"), "runtime")
    _require_exact_keys(runtime_raw, "runtime", set(RuntimeConfig.__dataclass_fields__))
    runtime = RuntimeConfig(
        **{
            name: _string(runtime_raw.get(name), f"runtime.{name}")
            for name in RuntimeConfig.__dataclass_fields__
        }
    )
    if not runtime.container_digest.startswith("sha256:") or len(runtime.container_digest) != 71:
        raise ValueError("runtime.container_digest must be a complete sha256 digest")
    for name in ("isaac_lab_mimic_source_sha256", "dex1_gripper_python_sha256"):
        digest = getattr(runtime, name)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"runtime.{name} must be a complete lowercase SHA-256")
    if (
        runtime.dex1_gripper_python_sha256
        != "02edd7928913f76b7da74b97ce0e9ff3d9e5cd178e91cbb6599902d01782f269"
    ):
        raise ValueError("runtime.dex1_gripper_python_sha256 differs from organizer V1")
    if runtime.physics_backend != "physx":
        raise ValueError("the V1 acceptance environment must use PhysX")
    if runtime.container_image != "paperc/robofinals:RoboFinals-IKEA-V1":
        raise ValueError("the acceptance runtime must use the organizer's V1 image")
    if runtime.robot != "G1+Dex1-1":
        raise ValueError("the augmentation embodiment must be G1+Dex1-1")

    dataset_runtime_raw = _mapping(root.get("dataset_runtime"), "dataset_runtime")
    _require_exact_keys(
        dataset_runtime_raw,
        "dataset_runtime",
        set(DatasetRuntimeConfig.__dataclass_fields__),
    )
    dataset_runtime = DatasetRuntimeConfig(
        **{
            name: _string(dataset_runtime_raw.get(name), f"dataset_runtime.{name}")
            for name in DatasetRuntimeConfig.__dataclass_fields__
        }
    )
    if dataset_runtime.lerobot_version != "0.6.0":
        raise ValueError("dataset assembly must use LeRobot 0.6.0")
    for name in ("lerobot_revision", "lerobot_wheel_sha256"):
        value = getattr(dataset_runtime, name)
        expected_length = 40 if name == "lerobot_revision" else 64
        if len(value) != expected_length or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"dataset_runtime.{name} must be a full lowercase hexadecimal digest")
    if dataset_runtime.python_version != "3.12":
        raise ValueError("LeRobot 0.6.0 dataset assembly is pinned to Python 3.12")

    object_pose_raw = _mapping(root.get("object_pose_runtime"), "object_pose_runtime")
    string_fields = (
        "method",
        "foundationpose_repo",
        "foundationpose_revision",
        "foundationpose_license",
        "pytorch3d_repo",
        "pytorch3d_revision",
        "pytorch3d_license",
        "nvdiffrast_repo",
        "nvdiffrast_revision",
        "nvdiffrast_license",
        "fast_stereo_repo",
        "fast_stereo_revision",
        "fast_stereo_license",
        "fast_stereo_model_repo",
        "fast_stereo_model_revision",
        "fast_stereo_model_filename",
        "fast_stereo_model_sha256",
        "fast_stereo_config_filename",
        "fast_stereo_config_sha256",
        "fast_stereo_model_license",
        "timm_version",
        "timm_wheel_sha256",
        "assembled_table_mesh_sha256",
        "foundationpose_refiner_config_file_id",
        "foundationpose_refiner_config_sha256",
        "foundationpose_refiner_checkpoint_file_id",
        "foundationpose_refiner_checkpoint_sha256",
        "foundationpose_scorer_config_file_id",
        "foundationpose_scorer_config_sha256",
        "foundationpose_scorer_checkpoint_file_id",
        "foundationpose_scorer_checkpoint_sha256",
        "detector_repo",
        "detector_revision",
        "detector_checkpoint_filename",
        "detector_checkpoint_sha256",
        "detector_license",
        "segmentation_repo",
        "segmentation_revision",
        "segmentation_checkpoint_filename",
        "segmentation_checkpoint_sha256",
        "segmentation_license",
        "transformers_version",
        "transformers_wheel_sha256",
        "robot_visual_urdf_relative_path",
        "robot_visual_urdf_sha256",
        "text_prompt",
    )
    integer_fields = (
        "source_frame_stride",
        "registration_interval_source_frames",
        "registration_refine_iterations",
        "tracking_refine_iterations",
        "temporal_propagation_beam_size",
        "dense_temporal_propagation_beam_size",
        "robot_silhouette_dilation_px",
        "auxiliary_robot_silhouette_dilation_px",
        "table_mask_minimum_value",
        "table_mask_minimum_component_area_px",
        "table_prompt_maximum_distance_px",
        "maximum_terminal_tracking_gap_source_frames",
        "maximum_initial_static_backfill_source_frames",
        "maximum_temporal_evidence_gap_source_frames",
        "max_detection_candidates",
        "assembled_table_mesh_vertices",
        "assembled_table_mesh_triangles",
        "foundationpose_refiner_config_size_bytes",
        "foundationpose_refiner_checkpoint_size_bytes",
        "foundationpose_scorer_config_size_bytes",
        "foundationpose_scorer_checkpoint_size_bytes",
        "detector_checkpoint_size_bytes",
        "segmentation_checkpoint_size_bytes",
        "fast_stereo_model_size_bytes",
        "fast_stereo_config_size_bytes",
        "fast_stereo_valid_iterations",
        "fast_stereo_max_disparity_px",
    )
    float_fields = (
        "min_valid_depth_fraction_in_mask",
        "auxiliary_view_score_weight",
        "auxiliary_primary_support_saturation_fraction",
        "detector_box_threshold",
        "detector_text_threshold",
        "segmentation_iou_threshold",
        "dense_mask_min_bidirectional_iou",
        "dense_mask_min_area_fraction",
        "dense_mask_max_area_fraction",
        "min_mask_area_fraction",
        "max_mask_area_fraction",
        "minimum_registration_mask_coverage",
        "max_bidirectional_translation_error_m",
        "max_bidirectional_rotation_error_rad",
        "max_registration_correction_translation_m",
        "max_registration_correction_rotation_rad",
        "temporal_candidate_min_mask_precision",
        "temporal_candidate_min_mask_explained_fraction",
        "temporal_candidate_max_depth_error_m",
        "temporal_static_translation_scale_m",
        "temporal_static_rotation_scale_rad",
        "temporal_max_static_translation_step_m",
        "temporal_max_static_rotation_step_rad",
        "temporal_grasp_translation_scale_m",
        "temporal_grasp_rotation_scale_rad",
        "temporal_max_table_speed_m_s",
        "temporal_max_table_angular_speed_rad_s",
        "temporal_max_grasp_relative_translation_step_m",
        "temporal_max_grasp_relative_rotation_step_rad",
        "temporal_grasp_observed_position_max",
        "temporal_blurred_frame_laplacian_variance_max",
        "temporal_carry_min_mask_precision",
        "temporal_carry_min_mask_explained_fraction",
        "temporal_carry_min_auxiliary_explained_fraction",
        "temporal_carry_min_multiview_score",
        "temporal_carry_visual_penalty",
        "temporal_min_stereo_consistent_fraction",
        "fast_stereo_maximum_left_right_error_px",
        "table_prompt_principal_point_weight",
        "primary_mask_maximum_median_eef_distance_m",
        "primary_mask_minimum_component_valid_depth_fraction",
        "mask_sequence_detector_weight",
        "mask_sequence_segmentation_weight",
        "mask_sequence_retention_weight",
        "mask_sequence_transition_iou_weight",
        "mask_sequence_primary_area_weight",
        "mask_sequence_primary_fragmentation_weight",
        "mask_sequence_primary_transition_area_weight",
        "mask_sequence_auxiliary_area_weight",
        "mask_sequence_auxiliary_fragmentation_weight",
        "mask_sequence_auxiliary_transition_area_weight",
        "max_rendered_depth_median_abs_error_m",
        "min_rendered_depth_overlap_fraction",
        "min_rendered_mask_explained_fraction",
    )
    _require_exact_keys(
        object_pose_raw,
        "object_pose_runtime",
        set(string_fields + integer_fields + float_fields),
    )
    object_pose_runtime = ObjectPoseRuntimeConfig(
        **{
            name: _string(object_pose_raw.get(name), f"object_pose_runtime.{name}")
            for name in string_fields
        },
        **{
            name: _positive_int(object_pose_raw.get(name), f"object_pose_runtime.{name}")
            for name in integer_fields
        },
        **{
            name: _finite_float(object_pose_raw.get(name), f"object_pose_runtime.{name}")
            for name in float_fields
        },
    )
    expected_pose_strings = {
        "method": (
            "grounded_sam2.1_sparse_keyframes_plus_bidirectional_sam2.1_video_"
            "and_foundationpose_head_rgbd_tracking"
        ),
        "foundationpose_repo": "NVlabs/FoundationPose",
        "foundationpose_revision": "a1b694b83e633c2cb6115b9063d940a687759392",
        "foundationpose_license": "NVIDIA Source Code License (research/evaluation only)",
        "pytorch3d_repo": "facebookresearch/pytorch3d",
        "pytorch3d_revision": "4daa00b41c52455440b938d1b676e00935b204d7",
        "pytorch3d_license": "BSD-3-Clause",
        "nvdiffrast_repo": "NVlabs/nvdiffrast",
        "nvdiffrast_revision": "253ac4fcea7de5f396371124af597e6cc957bfae",
        "nvdiffrast_license": "NVIDIA Source Code License (1-Way Commercial)",
        "fast_stereo_repo": "NVlabs/Fast-FoundationStereo",
        "fast_stereo_revision": "a290ba04c1b3ad1ec41a33974a157b2917b624d4",
        "fast_stereo_license": "NVIDIA Source Code License (research/evaluation only)",
        "fast_stereo_model_repo": "nvidia/c-fast-foundationstereo",
        "fast_stereo_model_revision": "9b446878c81ddb27593036767b29b2859d46103e",
        "fast_stereo_model_filename": "model_best_bp2_serialize.pth",
        "fast_stereo_model_sha256": "7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214",
        "fast_stereo_config_filename": "cfg.yaml",
        "fast_stereo_config_sha256": "d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc",
        "fast_stereo_model_license": "NVIDIA Open Model License Agreement",
        "timm_version": "1.0.27",
        "timm_wheel_sha256": "5ff07c9ddf53cbada88eab1c93ff175c64cab683b5a2fddf863bcee985926f89",
        "assembled_table_mesh_sha256": "93a81f0a6c4b3541973c89ff061a5bb57614471ffa1304dfd4415b292e94e4e4",
        "foundationpose_refiner_config_file_id": "1477-st1s1TxXN6oqfM5ZnsQwd8BCzVg1",
        "foundationpose_refiner_config_sha256": "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
        "foundationpose_refiner_checkpoint_file_id": "1E9FPB5WFIBMLrOJqZLpoVOK4Mjzrrxhv",
        "foundationpose_refiner_checkpoint_sha256": "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
        "foundationpose_scorer_config_file_id": "1kQkQG-q_VvLRozv30hyeLB7P_jiEEqiE",
        "foundationpose_scorer_config_sha256": "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
        "foundationpose_scorer_checkpoint_file_id": "1Zdjnkn4EHOI5_k08apofwRgTjWpai4E4",
        "foundationpose_scorer_checkpoint_sha256": "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
        "detector_repo": "IDEA-Research/grounding-dino-base",
        "detector_revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "detector_checkpoint_filename": "model.safetensors",
        "detector_checkpoint_sha256": "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
        "detector_license": "Apache-2.0",
        "segmentation_repo": "facebook/sam2.1-hiera-large",
        "segmentation_revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        "segmentation_checkpoint_filename": "model.safetensors",
        "segmentation_checkpoint_sha256": "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230",
        "segmentation_license": "Apache-2.0",
        "transformers_version": "5.13.0",
        "transformers_wheel_sha256": "8adbc1d20bd5463cd6876b2eb7cb31971e1065788e7dc6bc12bab597a7c504b7",
        "robot_visual_urdf_relative_path": "robofinals/data/assets/g1_urdf_gripper/g1/g1_29dof_mode_15_with_dex1_1.urdf",
        "robot_visual_urdf_sha256": "b2ce7a8b620dc5511e9189ab19577788a65615eb508f54013b03f8156363ef3f",
    }
    for name, expected in expected_pose_strings.items():
        if getattr(object_pose_runtime, name) != expected:
            raise ValueError(f"object_pose_runtime.{name} differs from the pinned runtime")
    expected_pose_sizes = {
        "foundationpose_refiner_config_size_bytes": 708,
        "foundationpose_refiner_checkpoint_size_bytes": 68220109,
        "foundationpose_scorer_config_size_bytes": 778,
        "foundationpose_scorer_checkpoint_size_bytes": 190229389,
        "detector_checkpoint_size_bytes": 933400872,
        "segmentation_checkpoint_size_bytes": 897897416,
        "fast_stereo_model_size_bytes": 71105659,
        "fast_stereo_config_size_bytes": 182,
        "fast_stereo_valid_iterations": 8,
        "fast_stereo_max_disparity_px": 192,
        "assembled_table_mesh_vertices": 46606,
        "assembled_table_mesh_triangles": 89967,
    }
    for name, expected in expected_pose_sizes.items():
        if getattr(object_pose_runtime, name) != expected:
            raise ValueError(f"object_pose_runtime.{name} differs from the pinned artifact")
    if object_pose_runtime.source_frame_stride != 3:
        raise ValueError("object pose tracking must sample source RGB-D at 10 Hz from 30 Hz input")
    if object_pose_runtime.registration_interval_source_frames % object_pose_runtime.source_frame_stride:
        raise ValueError("FoundationPose registration interval must align with sampled source frames")
    if object_pose_runtime.robot_silhouette_dilation_px % 2:
        raise ValueError("robot silhouette dilation must be an even pixel count")
    if object_pose_runtime.auxiliary_robot_silhouette_dilation_px % 2:
        raise ValueError("auxiliary robot silhouette dilation must be an even pixel count")
    if object_pose_runtime.table_mask_minimum_value > 255:
        raise ValueError("table mask minimum value must not exceed 255")
    if object_pose_runtime.table_prompt_maximum_distance_px > 800:
        raise ValueError("table prompt distance must not exceed the image diagonal")
    if object_pose_runtime.table_prompt_principal_point_weight > 1.0:
        raise ValueError("table prompt principal-point weight must not exceed one")
    if object_pose_runtime.primary_mask_minimum_component_valid_depth_fraction > 1.0:
        raise ValueError(
            "primary-mask component depth fraction must not exceed one"
        )
    if (
        object_pose_runtime.maximum_terminal_tracking_gap_source_frames
        > object_pose_runtime.registration_interval_source_frames
    ):
        raise ValueError("terminal tracking gap must not exceed one registration interval")
    if (
        object_pose_runtime.maximum_temporal_evidence_gap_source_frames
        < object_pose_runtime.registration_interval_source_frames
    ):
        raise ValueError("temporal evidence gap must span at least one registration interval")
    if object_pose_runtime.max_detection_candidates > 20:
        raise ValueError("object-pose detector candidate limit must not exceed 20")
    if object_pose_runtime.fast_stereo_max_disparity_px % 32:
        raise ValueError("Fast FoundationStereo maximum disparity must be divisible by 32")
    if object_pose_runtime.fast_stereo_maximum_left_right_error_px > 5.0:
        raise ValueError("Fast FoundationStereo left-right error limit is implausible")
    if object_pose_runtime.temporal_min_stereo_consistent_fraction > 1.0:
        raise ValueError("temporal stereo consistency fraction must not exceed one")
    if object_pose_runtime.temporal_propagation_beam_size > 256:
        raise ValueError("object-pose temporal propagation beam must not exceed 256")
    if object_pose_runtime.dense_temporal_propagation_beam_size > 256:
        raise ValueError("object-pose dense temporal propagation beam must not exceed 256")
    if not 0.0 < object_pose_runtime.min_valid_depth_fraction_in_mask <= 1.0:
        raise ValueError("object pose minimum mask depth fraction must be in (0, 1]")
    if object_pose_runtime.auxiliary_view_score_weight > 1.0:
        raise ValueError("object pose auxiliary-view score weight must not exceed 1")
    if object_pose_runtime.auxiliary_primary_support_saturation_fraction > 1.0:
        raise ValueError(
            "object pose auxiliary primary-support saturation must not exceed 1"
        )
    for name in (
        "detector_box_threshold",
        "detector_text_threshold",
        "min_mask_area_fraction",
        "max_mask_area_fraction",
        "minimum_registration_mask_coverage",
        "dense_mask_min_bidirectional_iou",
        "dense_mask_min_area_fraction",
        "dense_mask_max_area_fraction",
        "min_rendered_depth_overlap_fraction",
        "min_rendered_mask_explained_fraction",
    ):
        value = getattr(object_pose_runtime, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"object_pose_runtime.{name} must be in (0, 1)")
    if not 0.0 <= object_pose_runtime.segmentation_iou_threshold < 1.0:
        raise ValueError(
            "object_pose_runtime.segmentation_iou_threshold must be in [0, 1)"
        )
    if (
        object_pose_runtime.dense_mask_min_area_fraction
        >= object_pose_runtime.dense_mask_max_area_fraction
    ):
        raise ValueError("object-pose dense mask area bounds must be ordered")
    if object_pose_runtime.min_mask_area_fraction >= object_pose_runtime.max_mask_area_fraction:
        raise ValueError("object-pose mask area bounds must be ordered")
    for name in (
        field for field in float_fields[1:] if field != "segmentation_iou_threshold"
    ):
        if getattr(object_pose_runtime, name) <= 0.0:
            raise ValueError(f"object_pose_runtime.{name} must be positive")

    camera_values = root.get("cameras")
    if not isinstance(camera_values, list) or len(camera_values) != 3:
        raise ValueError("cameras must contain exactly the three deployable RGB cameras")
    for index, item in enumerate(camera_values):
        camera_mapping = _mapping(item, f"cameras[{index}]")
        _require_exact_keys(
            camera_mapping,
            f"cameras[{index}]",
            set(CameraConfig.__dataclass_fields__),
        )
    cameras = tuple(
        CameraConfig(
            source_key=_string(_mapping(item, "camera").get("source_key"), "camera.source_key"),
            policy_key=_string(item.get("policy_key"), "camera.policy_key"),
            sim_sensor=_string(item.get("sim_sensor"), "camera.sim_sensor"),
            prim_path=_string(item.get("prim_path"), "camera.prim_path"),
            offset_position_m=_finite_vector(
                item.get("offset_position_m"), 3, "camera.offset_position_m"
            ),
            offset_quaternion_xyzw=_finite_vector(
                item.get("offset_quaternion_xyzw"), 4, "camera.offset_quaternion_xyzw"
            ),
            convention=_string(item.get("convention"), "camera.convention"),
            focal_length_mm=_finite_float(item.get("focal_length_mm"), "camera.focal_length_mm"),
            horizontal_aperture_mm=_finite_float(
                item.get("horizontal_aperture_mm"), "camera.horizontal_aperture_mm"
            ),
            vertical_aperture_mm=_finite_float(
                item.get("vertical_aperture_mm"), "camera.vertical_aperture_mm"
            ),
            intrinsic_matrix_px=_finite_vector(
                item.get("intrinsic_matrix_px"), 9, "camera.intrinsic_matrix_px"
            ),
            distortion_model=_string(
                item.get("distortion_model"), "camera.distortion_model"
            ),
            distortion_coefficients=_finite_vector(
                item.get("distortion_coefficients"), 5, "camera.distortion_coefficients"
            ),
            intrinsic_calibration_sha256s=_string_tuple(
                item.get("intrinsic_calibration_sha256s"),
                "camera.intrinsic_calibration_sha256s",
            ),
            clipping_range_m=_finite_range(
                item.get("clipping_range_m"), "camera.clipping_range_m", minimum=0.0
            ),
            calibration_basis=_string(item.get("calibration_basis"), "camera.calibration_basis"),
            width=_positive_int(item.get("width"), "camera.width"),
            height=_positive_int(item.get("height"), "camera.height"),
            fps=_positive_int(item.get("fps"), "camera.fps"),
        )
        for item in camera_values
    )
    if tuple(camera.source_key for camera in cameras) != EXPECTED_CAMERA_KEYS:
        raise ValueError(f"camera source order must be {EXPECTED_CAMERA_KEYS}")
    if tuple(camera.policy_key for camera in cameras) != EXPECTED_POLICY_CAMERA_KEYS:
        raise ValueError(f"camera policy order must be {EXPECTED_POLICY_CAMERA_KEYS}")
    expected_sim_sensors = ("first_person_camera", "left_hand_camera", "right_hand_camera")
    if tuple(camera.sim_sensor for camera in cameras) != expected_sim_sensors:
        raise ValueError(f"camera simulator sensor order must be {expected_sim_sensors}")
    expected_camera_frames = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    expected_bases = (
        "organizer_v1_head_rig_center_plus_measured_left_eye_and_raw_intrinsics",
        "dex1_d405_x115_z70_down20_and_two_unit_mean_raw_intrinsics",
        "dex1_d405_x115_z70_down20_and_two_unit_mean_raw_intrinsics",
    )
    if tuple(camera.calibration_basis for camera in cameras) != expected_bases:
        raise ValueError("camera calibration provenance changed")
    for camera, parent_frame in zip(cameras, expected_camera_frames, strict=True):
        if (camera.width, camera.height, camera.fps) != (640, 480, 30):
            raise ValueError(f"{camera.source_key} must be raw 640x480 RGB at 30 Hz")
        if camera.source_key.endswith("_ir") or camera.source_key.endswith("cam_1"):
            raise ValueError("head-right and IR cameras are forbidden synthetic modalities")
        if f"/Robot/{parent_frame}/{camera.sim_sensor}" not in camera.prim_path:
            raise ValueError(f"{camera.sim_sensor} must remain attached to {parent_frame}")
        if camera.convention != "opengl":
            raise ValueError("policy cameras use the pinned OpenGL optical convention")
        if not math.isclose(
            math.sqrt(sum(value * value for value in camera.offset_quaternion_xyzw)),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{camera.sim_sensor} offset quaternion must be unit length")
        if min(
            camera.focal_length_mm,
            camera.horizontal_aperture_mm,
            camera.vertical_aperture_mm,
            camera.clipping_range_m[0],
        ) <= 0.0:
            raise ValueError(f"{camera.sim_sensor} intrinsics must be positive")
        intrinsic = camera.intrinsic_matrix_px
        if (
            intrinsic[1] != 0.0
            or intrinsic[3] != 0.0
            or intrinsic[6] != 0.0
            or intrinsic[7] != 0.0
            or intrinsic[8] != 1.0
            or intrinsic[0] <= 0.0
            or intrinsic[4] <= 0.0
            or not 0.0 <= intrinsic[2] < camera.width
            or not 0.0 <= intrinsic[5] < camera.height
        ):
            raise ValueError(f"{camera.sim_sensor} intrinsic matrix is malformed")
        mean_focal_px = 0.5 * (intrinsic[0] + intrinsic[4])
        usd_focal_x_px = camera.focal_length_mm * camera.width / camera.horizontal_aperture_mm
        usd_focal_y_px = camera.focal_length_mm * camera.height / camera.vertical_aperture_mm
        if not math.isclose(usd_focal_x_px, mean_focal_px, rel_tol=1e-9) or not math.isclose(
            usd_focal_y_px, mean_focal_px, rel_tol=1e-9
        ):
            raise ValueError(
                f"{camera.sim_sensor} USD pinhole must use the mean measured pixel focal length"
            )
        if camera.distortion_model not in {
            "opencv_brown_conrady",
            "realsense_inverse_brown_conrady",
        }:
            raise ValueError(f"{camera.sim_sensor} distortion model is unsupported")
        if not camera.intrinsic_calibration_sha256s:
            raise ValueError(f"{camera.sim_sensor} lacks intrinsic calibration provenance")
        for digest in camera.intrinsic_calibration_sha256s:
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{camera.sim_sensor} intrinsic calibration hash is invalid")
    expected_head_position = (
        0.10209156,
        -0.00937542 + 0.5 * raw_source.head_stereo_baseline_m,
        0.42446595,
    )
    if any(
        not math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(cameras[0].offset_position_m, expected_head_position, strict=True)
    ):
        raise ValueError("head-left mount must use the V1 rig center plus half the measured baseline")
    if cameras[0].intrinsic_calibration_sha256s != (
        raw_source.head_stereo_calibration_sha256,
    ):
        raise ValueError("head-left intrinsic provenance must match the pinned raw calibration")
    expected_wrist_hashes = tuple(raw_source.wrist_calibration_sha256_by_serial.values())
    if any(camera.intrinsic_calibration_sha256s != expected_wrist_hashes for camera in cameras[1:]):
        raise ValueError("wrist intrinsic provenance must contain both pinned D405 calibrations")
    for wrist in cameras[1:]:
        if wrist.offset_position_m != (0.115, 0.0, 0.07):
            raise ValueError("Dex1 D405 nominal mount must remain x115/z70")
        if not math.isclose(wrist.focal_length_mm, 24.0):
            raise ValueError("Dex1 wrist USD focal length changed")

    annotation_raw = _mapping(root.get("source_annotation"), "source_annotation")
    _require_exact_keys(annotation_raw, "source_annotation", {"automatic_phase"})
    automatic_raw = _mapping(
        annotation_raw.get("automatic_phase"),
        "source_annotation.automatic_phase",
    )
    automatic_integer_fields = (
        "smoothing_window_frames",
        "endpoint_window_frames",
        "minimum_phase_frames",
        "sustained_event_frames",
    )
    automatic_float_fields = (
        "minimum_flip_angle_rad",
        "minimum_table_displacement_m",
        "motion_translation_threshold_m",
        "motion_rotation_threshold_rad",
        "rotate_start_fraction",
        "rotate_end_fraction",
        "grasp_closed_fraction_min",
        "release_open_fraction_min",
        "settled_linear_speed_m_s_max",
        "settled_angular_speed_rad_s_max",
        "retreat_relative_displacement_m_min",
        "hold_relative_position_p95_m_max",
        "rotation_progress_reversal_fraction_max",
    )
    expected_automatic_keys = set(automatic_integer_fields + automatic_float_fields)
    if set(automatic_raw) != expected_automatic_keys:
        raise ValueError(
            "source_annotation.automatic_phase keys must be exactly "
            f"{sorted(expected_automatic_keys)}"
        )
    automatic_phase = AutomaticPhaseConfig(
        **{
            name: _positive_int(
                automatic_raw.get(name), f"source_annotation.automatic_phase.{name}"
            )
            for name in automatic_integer_fields
        },
        **{
            name: _finite_float(
                automatic_raw.get(name), f"source_annotation.automatic_phase.{name}"
            )
            for name in automatic_float_fields
        },
    )
    if automatic_phase.smoothing_window_frames % 2 != 1:
        raise ValueError("automatic phase smoothing window must be odd")
    if automatic_phase.sustained_event_frames < automatic_phase.minimum_phase_frames:
        raise ValueError("automatic phase event persistence cannot be shorter than a phase")
    if not 0.0 < automatic_phase.minimum_flip_angle_rad <= math.pi:
        raise ValueError("automatic phase minimum flip angle must be in (0, pi]")
    if not 0.0 < automatic_phase.rotate_start_fraction < automatic_phase.rotate_end_fraction < 1.0:
        raise ValueError("automatic phase rotation fractions must satisfy 0 < start < end < 1")
    for name in (
        "grasp_closed_fraction_min",
        "release_open_fraction_min",
        "rotation_progress_reversal_fraction_max",
    ):
        if not 0.0 < getattr(automatic_phase, name) < 1.0:
            raise ValueError(f"automatic phase {name} must be in (0, 1)")
    for name in (
        "minimum_table_displacement_m",
        "motion_translation_threshold_m",
        "motion_rotation_threshold_rad",
        "settled_linear_speed_m_s_max",
        "settled_angular_speed_rad_s_max",
        "retreat_relative_displacement_m_min",
        "hold_relative_position_p95_m_max",
    ):
        if getattr(automatic_phase, name) <= 0.0:
            raise ValueError(f"automatic phase {name} must be positive")
    source_annotation = SourceAnnotationConfig(automatic_phase=automatic_phase)

    subtasks_raw = root.get("subtasks")
    if not isinstance(subtasks_raw, list) or tuple(subtasks_raw) != EXPECTED_SUBTASKS:
        raise ValueError(f"subtasks must be exactly {EXPECTED_SUBTASKS}")

    source_contract = _mapping(root.get("source_contract"), "source_contract")
    _require_exact_keys(
        source_contract,
        "source_contract",
        {
            "root_pose_dim",
            "body_joint_dim",
            "eef_pose_dim_per_side",
            "eef_pose_order",
            "eef_pose_format",
            "eef_reference_frame",
            "fk_frames",
            "fk_tool_transform_reference",
            "fk_tool_transforms",
            "dex1_hand_command",
            "fk_calibration_episode_modulus",
            "fk_action_validation_position_p95_m_max",
            "fk_action_validation_rotation_p95_rad_max",
            "fk_swapped_assignment_score_ratio_min",
        },
    )
    expected_contract = {
        "root_pose_dim": 7,
        "body_joint_dim": 29,
        "eef_pose_dim_per_side": 6,
        "eef_pose_order": ["left", "right"],
        "eef_pose_format": "xyz_euler_xyz_rad",
        "eef_reference_frame": "robot_root",
        "fk_frames": {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"},
    }
    for key, expected in expected_contract.items():
        if source_contract.get(key) != expected:
            raise ValueError(f"source_contract.{key} must be {expected!r}")
    reference = _mapping(
        source_contract.get("fk_tool_transform_reference"),
        "source_contract.fk_tool_transform_reference",
    )
    _require_exact_keys(
        reference,
        "source_contract.fk_tool_transform_reference",
        {"repo_id", "revision", "path"},
    )
    if reference.get("repo_id") != "unitreerobotics/xr_teleoperate":
        raise ValueError("the FK tool frame must cite Unitree's official xr_teleoperate repository")
    reference_revision = _string(
        reference.get("revision"), "source_contract.fk_tool_transform_reference.revision"
    )
    if len(reference_revision) != 40 or any(
        character not in "0123456789abcdef" for character in reference_revision
    ):
        raise ValueError("FK tool transform reference must use a full lowercase commit SHA")
    _string(reference.get("path"), "source_contract.fk_tool_transform_reference.path")
    tool_transforms = _mapping(
        source_contract.get("fk_tool_transforms"), "source_contract.fk_tool_transforms"
    )
    if tuple(tool_transforms) != ("left", "right"):
        raise ValueError("FK tool transforms must contain ordered left and right entries")
    for side in ("left", "right"):
        transform = _mapping(tool_transforms[side], f"source_contract.fk_tool_transforms.{side}")
        _require_exact_keys(
            transform,
            f"source_contract.fk_tool_transforms.{side}",
            {"translation_m", "quaternion_xyzw"},
        )
        translation = _finite_vector(
            transform.get("translation_m"), 3, f"source_contract.fk_tool_transforms.{side}.translation_m"
        )
        quaternion = _finite_vector(
            transform.get("quaternion_xyzw"),
            4,
            f"source_contract.fk_tool_transforms.{side}.quaternion_xyzw",
        )
        if not math.isclose(math.sqrt(sum(item * item for item in quaternion)), 1.0, abs_tol=1e-6):
            raise ValueError(f"source_contract.fk_tool_transforms.{side} quaternion must be unit length")
        if translation != (0.05, 0.0, 0.0) or quaternion != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("the official G1 EEF tool transform must remain wrist-yaw + [0.05,0,0] m")
    expected_hand_contract = {
        "order": ["left", "right"],
        "closed_motor_position": 0.0,
        "open_motor_position": 4.5,
        "unit": "rad_relative_to_closed_stop",
        "organizer_normalized_open": -1.0,
        "organizer_normalized_closed": 1.0,
        "reference_repo_id": "unitreerobotics/xr_teleoperate",
        "reference_revision": "478248df33681e82069a40c6598a08784a2d69a5",
        "reference_path": "teleop/robot_control/robot_hand_unitree.py",
        "reference_git_blob": "350f7c3139575efceb1a1c12923710d81aecffff",
    }
    if source_contract.get("dex1_hand_command") != expected_hand_contract:
        raise ValueError(
            "source_contract.dex1_hand_command differs from the official Dex1 contract"
        )
    if _positive_int(
        source_contract.get("fk_calibration_episode_modulus"),
        "source_contract.fk_calibration_episode_modulus",
    ) < 2:
        raise ValueError("FK calibration modulus must reserve held-out episodes")
    for key in (
        "fk_action_validation_position_p95_m_max",
        "fk_action_validation_rotation_p95_rad_max",
        "fk_swapped_assignment_score_ratio_min",
    ):
        if _finite_float(source_contract.get(key), f"source_contract.{key}") <= 0.0:
            raise ValueError(f"source_contract.{key} must be positive")

    generation = _mapping(root.get("generation"), "generation")
    _require_exact_keys(
        generation,
        "generation",
        {
            "successful_trajectories_min",
            "appearance_variants_per_trajectory_min",
            "render_accepted_only",
            "cameras_during_physics",
            "keep_failed_episodes",
            "generation_select_src_per_subtask",
            "generation_select_src_per_arm",
            "multi_eef_coordination_scheme",
            "physics_hz",
            "mimic_control_hz",
            "episode_length_s",
            "action_noise_m",
            "action_noise_rad",
        },
    )
    expected_gates = {
        "successful_trajectories_min": 2000,
        "appearance_variants_per_trajectory_min": 2,
    }
    for key, minimum in expected_gates.items():
        value = _positive_int(generation.get(key), f"generation.{key}")
        if value < minimum:
            raise ValueError(f"generation.{key} cannot be below {minimum}")
    if generation.get("render_accepted_only") is not True:
        raise ValueError("only accepted physical trajectories may be rendered")
    if generation.get("cameras_during_physics") is not False:
        raise ValueError("cameras must be disabled during physical trajectory generation")
    if generation.get("keep_failed_episodes") is not False:
        raise ValueError("failed candidates cannot enter the output dataset")
    if generation.get("generation_select_src_per_subtask") is not True:
        raise ValueError("Mimic must be allowed to select source segments per subtask")
    if generation.get("generation_select_src_per_arm") is not False:
        raise ValueError("the two arms cannot select unrelated source trajectories")
    if generation.get("multi_eef_coordination_scheme") != "transform":
        raise ValueError("multi-EEF subtasks must share the object-relative transform")
    physics_hz = _positive_int(generation.get("physics_hz"), "generation.physics_hz")
    mimic_control_hz = _positive_int(
        generation.get("mimic_control_hz"), "generation.mimic_control_hz"
    )
    if (physics_hz, mimic_control_hz) != (200, 50) or physics_hz % mimic_control_hz:
        raise ValueError("V1 generation must run at 200 Hz physics / 50 Hz control")
    if _finite_float(generation.get("episode_length_s"), "generation.episode_length_s") < 32.0:
        raise ValueError("Mimic episode length must leave room for interpolation beyond the source demo")
    _finite_float(generation.get("action_noise_m"), "generation.action_noise_m")
    _finite_float(generation.get("action_noise_rad"), "generation.action_noise_rad")

    physical_raw = _mapping(root.get("physical_randomization"), "physical_randomization")
    _require_exact_keys(
        physical_raw,
        "physical_randomization",
        set(PhysicalRandomizationConfig.__dataclass_fields__),
    )
    physical_scalars = {
        key: _finite_float(physical_raw.get(key), f"physical_randomization.{key}")
        for key in (
            "table_long_range_m",
            "table_depth_range_m",
            "table_yaw_range_rad",
            "robot_distance_m",
            "robot_distance_range_m",
            "robot_table_min_distance_m",
            "robot_lateral_range_m",
            "robot_yaw_range_rad",
            "upper_body_joint_noise_rad",
            "dex1_finger_noise_m",
        )
    }
    if any(value < 0.0 for value in physical_scalars.values()):
        raise ValueError("physical randomization distances and angles cannot be negative")
    mass_scale = _finite_range(
        physical_raw.get("table_part_mass_scale"),
        "physical_randomization.table_part_mass_scale",
        minimum=0.5,
    )
    if mass_scale != (1.0, 1.0):
        raise ValueError("the assembled white table must retain its official 1.596 kg mass")
    contact_raw = _mapping(
        physical_raw.get("contact_materials"), "physical_randomization.contact_materials"
    )
    expected_contacts = ("hand_white_table", "white_table_workbench", "workbench_hand")
    if tuple(contact_raw) != expected_contacts:
        raise ValueError(f"contact_materials must contain ordered keys {expected_contacts}")
    contact_materials = {}
    for name in expected_contacts:
        material = _mapping(contact_raw[name], f"contact_materials.{name}")
        if set(material) != {"static_friction", "dynamic_friction", "restitution"}:
            raise ValueError(f"contact_materials.{name} has unsupported parameters")
        static = _finite_range(material["static_friction"], f"{name}.static_friction", minimum=0.0)
        dynamic = _finite_range(
            material["dynamic_friction"], f"{name}.dynamic_friction", minimum=0.0
        )
        restitution = _finite_range(material["restitution"], f"{name}.restitution", minimum=0.0)
        if dynamic[1] > static[0]:
            raise ValueError(f"{name} can sample dynamic friction above static friction")
        if restitution[1] > 0.2:
            raise ValueError(f"{name} restitution exceeds the realistic task range")
        contact_materials[name] = ContactMaterialConfig(static, dynamic, restitution)
    physical_randomization = PhysicalRandomizationConfig(
        **physical_scalars,
        table_part_mass_scale=mass_scale,
        contact_materials=contact_materials,
    )

    appearance_raw = _mapping(root.get("appearance_randomization"), "appearance_randomization")
    _require_exact_keys(
        appearance_raw,
        "appearance_randomization",
        set(AppearanceRandomizationConfig.__dataclass_fields__),
    )
    floor_materials = appearance_raw.get("floor_materials")
    wall_materials = appearance_raw.get("wall_materials")
    room_props = appearance_raw.get("room_props")
    for name, values in (
        ("floor_materials", floor_materials),
        ("wall_materials", wall_materials),
        ("room_props", room_props),
    ):
        if not isinstance(values, list) or len(values) < 2 or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"appearance_randomization.{name} requires at least two names")
    visible_probability = _finite_float(
        appearance_raw.get("room_prop_visible_probability"),
        "appearance_randomization.room_prop_visible_probability",
    )
    if not 0.0 <= visible_probability <= 1.0:
        raise ValueError("room prop visibility probability must lie in [0,1]")
    appearance_randomization = AppearanceRandomizationConfig(
        variant_seed_stride=_positive_int(
            appearance_raw.get("variant_seed_stride"), "appearance_randomization.variant_seed_stride"
        ),
        nominal_camera_variant_index=int(appearance_raw.get("nominal_camera_variant_index", -1)),
        camera_translation_jitter_m_max=_finite_float(
            appearance_raw.get("camera_translation_jitter_m_max"),
            "appearance_randomization.camera_translation_jitter_m_max",
        ),
        camera_rotation_jitter_rad_max=_finite_float(
            appearance_raw.get("camera_rotation_jitter_rad_max"),
            "appearance_randomization.camera_rotation_jitter_rad_max",
        ),
        exposure_ev=_finite_range(appearance_raw.get("exposure_ev"), "appearance_randomization.exposure_ev"),
        color_temperature_k=_finite_range(
            appearance_raw.get("color_temperature_k"),
            "appearance_randomization.color_temperature_k",
            minimum=1000.0,
        ),
        distant_light_intensity=_finite_range(
            appearance_raw.get("distant_light_intensity"),
            "appearance_randomization.distant_light_intensity",
            minimum=0.0,
        ),
        sphere_light_intensity=_finite_range(
            appearance_raw.get("sphere_light_intensity"),
            "appearance_randomization.sphere_light_intensity",
            minimum=0.0,
        ),
        floor_materials=tuple(floor_materials),
        wall_materials=tuple(wall_materials),
        room_props=tuple(room_props),
        room_prop_visible_probability=visible_probability,
    )
    if appearance_randomization.nominal_camera_variant_index != 0:
        raise ValueError("appearance variant 0 must preserve the nominal camera calibration")
    if not 0.0 <= appearance_randomization.camera_translation_jitter_m_max <= 0.003:
        raise ValueError("camera translation jitter must remain within 3 mm")
    if not 0.0 <= appearance_randomization.camera_rotation_jitter_rad_max <= math.radians(1.0):
        raise ValueError("camera rotation jitter must remain within 1 degree")

    splits = _mapping(root.get("splits"), "splits")
    _require_exact_keys(splits, "splits", {"train", "validation", "test", "group_by"})
    weights = tuple(_finite_float(splits.get(name), f"splits.{name}") for name in ("train", "validation", "test"))
    if any(value <= 0.0 for value in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("train/validation/test split weights must be positive and sum to one")
    if splits.get("group_by") != "source_trajectory_lineage":
        raise ValueError("variants must be split as one source-trajectory lineage group")

    success = _mapping(root.get("success"), "success")
    expected_success_keys = {
        "normal_dot_max",
        "tabletop_lift_m_min",
        "settled_linear_speed_m_s_max",
        "settled_angular_speed_rad_s_max",
        "workbench_edge_margin_m_min",
        "hold_steps_min",
        "finger_contact_force_n_max",
        "reject_root_motion_m_max",
        "reject_root_rotation_rad_max",
        "reject_lower_body_joint_delta_rad_max",
        "reject_joint_limit_violation_rad_max",
    }
    if set(success) != expected_success_keys:
        raise ValueError(f"success keys must be exactly {sorted(expected_success_keys)}")
    success_values = {key: _finite_float(success[key], f"success.{key}") for key in success}
    if not -1.0 <= success_values["normal_dot_max"] < 0.0:
        raise ValueError("success.normal_dot_max must describe an inverted tabletop")
    if success_values["tabletop_lift_m_min"] <= 0.0:
        raise ValueError("success.tabletop_lift_m_min must be positive")
    if _positive_int(success["hold_steps_min"], "success.hold_steps_min") < 10:
        raise ValueError("success.hold_steps_min cannot be below the V1 contract")
    for key in (
        "settled_linear_speed_m_s_max",
        "settled_angular_speed_rad_s_max",
        "finger_contact_force_n_max",
        "reject_root_motion_m_max",
        "reject_root_rotation_rad_max",
        "reject_lower_body_joint_delta_rad_max",
        "reject_joint_limit_violation_rad_max",
    ):
        if success_values[key] <= 0.0:
            raise ValueError(f"success.{key} must be positive")

    return PipelineConfig(
        path=source_path,
        raw=root,
        source=source,
        raw_source=raw_source,
        target=target,
        runtime=runtime,
        dataset_runtime=dataset_runtime,
        object_pose_runtime=object_pose_runtime,
        cameras=cameras,
        physical_randomization=physical_randomization,
        appearance_randomization=appearance_randomization,
        source_annotation=source_annotation,
        subtasks=tuple(subtasks_raw),
        digest=canonical_json_digest(root),
    )
