"""Apply and verify the deployable three-camera contract before scene creation."""

from __future__ import annotations

from typing import Any

from ..config import PipelineConfig


def _apply_sensor(sensor: Any, camera) -> None:
    if sensor is None:
        raise RuntimeError(f"V1 scene omitted required camera sensor {camera.sim_sensor}")
    sensor.prim_path = camera.prim_path
    sensor.offset.pos = camera.offset_position_m
    sensor.offset.rot = camera.offset_quaternion_xyzw
    sensor.offset.convention = camera.convention
    sensor.data_types = ["rgb"]
    sensor.width = camera.width
    sensor.height = camera.height
    # State replay forces a fresh camera update after every selected pose.
    sensor.update_period = 0.0
    sensor.spawn.focal_length = camera.focal_length_mm
    sensor.spawn.horizontal_aperture = camera.horizontal_aperture_mm
    sensor.spawn.vertical_aperture = camera.vertical_aperture_mm
    sensor.spawn.horizontal_aperture_offset = 0.0
    sensor.spawn.vertical_aperture_offset = 0.0
    sensor.spawn.clipping_range = camera.clipping_range_m
    sensor.spawn.lock_camera = True


def apply_camera_contract(env_cfg: Any, config: PipelineConfig) -> None:
    """Mutate both instantiated scene sensors and organizer metadata in lockstep."""

    scene = getattr(env_cfg, "scene", None)
    arena = getattr(env_cfg, "isaaclab_arena_env", None)
    embodiment = getattr(arena, "embodiment", None)
    observation_cameras = getattr(embodiment, "observation_cameras", None)
    if scene is None or not isinstance(observation_cameras, dict):
        raise RuntimeError("unexpected RoboFinals V1 camera configuration structure")

    for camera in config.cameras:
        sensor = getattr(scene, camera.sim_sensor, None)
        _apply_sensor(sensor, camera)
        metadata = observation_cameras.get(camera.sim_sensor)
        if not isinstance(metadata, dict) or "camera_cfg" not in metadata:
            raise RuntimeError(f"V1 metadata omitted required camera {camera.sim_sensor}")
        _apply_sensor(metadata["camera_cfg"], camera)

    active = tuple(getattr(embodiment, "active_observation_camera_names", ()))
    required = tuple(camera.sim_sensor for camera in config.cameras)
    missing = tuple(name for name in required if name not in active)
    if missing:
        raise RuntimeError(f"policy observation omitted required cameras: {missing}")


def verify_runtime_camera_contract(env: Any, config: PipelineConfig) -> dict[str, dict[str, Any]]:
    """Reject a spawned runtime whose camera config differs from the pinned contract."""

    report = {}
    for camera in config.cameras:
        sensor = env.scene[camera.sim_sensor]
        cfg = sensor.cfg
        actual = {
            "prim_path": cfg.prim_path,
            "offset_position_m": tuple(float(value) for value in cfg.offset.pos),
            "offset_quaternion_xyzw": tuple(float(value) for value in cfg.offset.rot),
            "convention": cfg.offset.convention,
            "focal_length_mm": float(cfg.spawn.focal_length),
            "horizontal_aperture_mm": float(cfg.spawn.horizontal_aperture),
            "vertical_aperture_mm": float(cfg.spawn.vertical_aperture),
            "horizontal_aperture_offset_mm": float(cfg.spawn.horizontal_aperture_offset),
            "vertical_aperture_offset_mm": float(cfg.spawn.vertical_aperture_offset),
            "clipping_range_m": tuple(float(value) for value in cfg.spawn.clipping_range),
            "width": int(cfg.width),
            "height": int(cfg.height),
            "update_period_s": float(cfg.update_period),
            "data_types": tuple(cfg.data_types),
        }
        expected = {
            "prim_path": camera.prim_path,
            "offset_position_m": camera.offset_position_m,
            "offset_quaternion_xyzw": camera.offset_quaternion_xyzw,
            "convention": camera.convention,
            "focal_length_mm": camera.focal_length_mm,
            "horizontal_aperture_mm": camera.horizontal_aperture_mm,
            "vertical_aperture_mm": camera.vertical_aperture_mm,
            "horizontal_aperture_offset_mm": 0.0,
            "vertical_aperture_offset_mm": 0.0,
            "clipping_range_m": camera.clipping_range_m,
            "width": camera.width,
            "height": camera.height,
            "update_period_s": 0.0,
            "data_types": ("rgb",),
        }
        if actual != expected:
            raise RuntimeError(
                f"spawned camera {camera.sim_sensor} differs from the pinned contract: "
                f"actual={actual}, expected={expected}"
            )
        report[camera.source_key] = {
            **actual,
            "sim_sensor": camera.sim_sensor,
            "policy_key": camera.policy_key,
            "fps": camera.fps,
            "calibration_basis": camera.calibration_basis,
            "intrinsic_matrix_px": camera.intrinsic_matrix_px,
            "distortion_model": camera.distortion_model,
            "distortion_coefficients": camera.distortion_coefficients,
            "intrinsic_calibration_sha256s": camera.intrinsic_calibration_sha256s,
            "recorded_geometry_postprocess": True,
        }
    return report
