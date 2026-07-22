"""Deterministic V1 room randomization driven through Omniverse Replicator."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..config import PipelineConfig


def _range_text(values: tuple[float, float]) -> str:
    return ",".join(format(value, ".17g") for value in values)


def configure_appearance_environment(config: PipelineConfig, room_asset_root: str | Path) -> None:
    """Set every appearance input before V1 creates task and sensor objects."""

    appearance = config.appearance_randomization
    asset_root = Path(room_asset_root).expanduser().resolve()
    if not (asset_root / "room_props.usda").is_file() or not (asset_root / "textures").is_dir():
        raise FileNotFoundError(f"room randomization assets are incomplete: {asset_root}")
    values = {
        "FLIP_TABLE_RANDOMIZE_ROOM": "true",
        "FLIP_TABLE_RANDOMIZE_ROOM_PROPS": "true",
        "FLIP_TABLE_RANDOMIZE_LIGHTING": "true",
        "FLIP_TABLE_ROOM_ASSET_ROOT": str(asset_root),
        "FLIP_TABLE_ROOM_FLOOR_MATERIALS": ",".join(appearance.floor_materials),
        "FLIP_TABLE_ROOM_WALL_MATERIALS": ",".join(appearance.wall_materials),
        "FLIP_TABLE_ROOM_PROP_ASSETS": ",".join(appearance.room_props),
        "FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY": format(
            appearance.room_prop_visible_probability, ".17g"
        ),
        "FLIP_TABLE_LIGHT_EXPOSURE_RANGE": _range_text(appearance.exposure_ev),
        "FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K": _range_text(
            appearance.color_temperature_k
        ),
        "FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K": _range_text(
            appearance.color_temperature_k
        ),
        "FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE": _range_text(
            appearance.distant_light_intensity
        ),
        "FLIP_TABLE_LIGHT_INTENSITY_RANGE": _range_text(
            appearance.sphere_light_intensity
        ),
        "FLIP_TABLE_RL_RANDOMIZATION_LEVEL": "1.0",
        "FLIP_TABLE_RL_CAMERA_POSITION_JITTER_M": format(
            appearance.camera_translation_jitter_m_max, ".17g"
        ),
        "FLIP_TABLE_RL_CAMERA_ROTATION_JITTER_DEG": format(
            math.degrees(appearance.camera_rotation_jitter_rad_max), ".17g"
        ),
        "FLIP_TABLE_ROOM_PROP_FRONT_AXIS": "+x",
    }
    os.environ.update(values)


def appearance_seed(config: PipelineConfig, trajectory_seed: int, variant_index: int) -> int:
    if trajectory_seed < 0 or variant_index < 0:
        raise ValueError("trajectory and variant seeds must be non-negative")
    return trajectory_seed + variant_index * config.appearance_randomization.variant_seed_stride


def _seed_everything(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AppearanceController:
    """Apply one episode-fixed room/light/camera sample and record its USD result."""

    def __init__(self, env, config: PipelineConfig) -> None:
        self.env = env
        self.config = config
        arena = getattr(env.cfg, "isaaclab_arena_env", None)
        self.task = getattr(arena, "task", None)
        required = (
            "_randomize_room_background",
            "_randomize_lighting",
            "_randomize_policy_camera_mounts",
            "_find_prim_by_suffix",
            "_set_stage_prim_local_pose",
            "_workbench_pose",
            "_robot_root_pose_local",
        )
        missing = [name for name in required if not hasattr(self.task, name)]
        if missing:
            raise RuntimeError(f"V1 flip-table task lacks appearance hooks: {missing}")
        self._replicator_graph_ready = False
        self._event_name = "team_ramen_flip_table_randomize_appearance"

    def _set_nominal_camera_mounts(self) -> None:
        for camera in self.config.cameras:
            suffix = camera.prim_path.replace("{ENV_REGEX_NS}/", "")
            prim = self.task._find_prim_by_suffix(self.env, suffix, env_id=0)
            if prim is None:
                raise RuntimeError(f"cannot find spawned policy camera {camera.sim_sensor}")
            self.task._set_stage_prim_local_pose(
                prim,
                torch.tensor(camera.offset_position_m, dtype=torch.float32),
                torch.tensor(camera.offset_quaternion_xyzw, dtype=torch.float32),
            )

    def _randomize_task_appearance(self, variant_index: int) -> None:
        env_ids = torch.tensor([0], dtype=torch.int64, device=self.env.device)
        self._set_nominal_camera_mounts()
        if variant_index != self.config.appearance_randomization.nominal_camera_variant_index:
            os.environ["FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS"] = "true"
            self.task._randomize_policy_camera_mounts(self.env, env_ids)
        else:
            os.environ["FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS"] = "false"

        workbench_pos_w, workbench_quat = self.task._workbench_pose(self.env)
        workbench_pos_local = None
        if workbench_pos_w is not None and workbench_quat is not None:
            workbench_pos_local = workbench_pos_w - self.env.scene.env_origins
        robot_pos_local, _robot_quat = self.task._robot_root_pose_local(self.env, env_ids)
        self.task._randomize_room_background(
            self.env,
            env_ids,
            None if workbench_pos_local is None else workbench_pos_local[env_ids],
            robot_pos_local,
        )
        self.task._randomize_lighting(self.env, env_ids)

    def _build_replicator_graph(self, seed: int) -> None:
        import omni.replicator.core as rep

        appearance = self.config.appearance_randomization
        rep.set_global_seed(seed)
        sun = rep.get.prims(
            path_pattern=r".*/FlipTableEvalLighting/WindowSun$",
            prim_types=["SphereLight", "DistantLight"],
            cache_result=False,
            return_sorted=True,
        )
        ceiling = rep.get.prims(
            path_pattern=r".*/FlipTableEvalLighting/CeilingLight_.*$",
            prim_types=["SphereLight"],
            cache_result=False,
            return_sorted=True,
        )
        with rep.trigger.on_custom_event(self._event_name):
            with sun:
                rep.modify.attribute(
                    "inputs:intensity",
                    rep.distribution.uniform(*appearance.distant_light_intensity),
                )
                rep.modify.attribute(
                    "inputs:colorTemperature",
                    rep.distribution.uniform(*appearance.color_temperature_k),
                )
                rep.modify.attribute(
                    "inputs:exposure", rep.distribution.uniform(*appearance.exposure_ev)
                )
            with ceiling:
                rep.modify.attribute(
                    "inputs:intensity",
                    rep.distribution.uniform(*appearance.sphere_light_intensity),
                )
                rep.modify.attribute(
                    "inputs:colorTemperature",
                    rep.distribution.uniform(*appearance.color_temperature_k),
                )
                rep.modify.attribute(
                    "inputs:exposure", rep.distribution.uniform(*appearance.exposure_ev)
                )
        self._replicator_graph_ready = True

    def _trigger_replicator(self, seed: int) -> None:
        import omni.replicator.core as rep

        if not self._replicator_graph_ready:
            self._build_replicator_graph(seed)
        rep.utils.send_og_event(self._event_name)
        rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.0)

    def apply(self, trajectory_seed: int, variant_index: int) -> dict[str, Any]:
        seed = appearance_seed(self.config, trajectory_seed, variant_index)
        _seed_everything(seed, self.env.device)
        self._randomize_task_appearance(variant_index)
        self._trigger_replicator(seed)
        self.env.sim.forward()
        self.env.sim.render()
        snapshot = snapshot_appearance(self.env, self.config)
        snapshot.update(
            {
                "appearance_seed": seed,
                "appearance_variant": variant_index,
                "replicator_event": self._event_name,
                "replicator_version": self.config.runtime.replicator_version,
                "nominal_camera_mount": variant_index
                == self.config.appearance_randomization.nominal_camera_variant_index,
            }
        )
        return snapshot


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "__iter__"):
        return [_jsonable(item) for item in value]
    return str(value)


def _prim_local_transform(prim) -> dict[str, Any]:
    from pxr import UsdGeom

    transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    position = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return {
        "position_m": [float(position[index]) for index in range(3)],
        "quaternion_xyzw": [
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(rotation.GetReal()),
        ],
    }


def snapshot_appearance(env, config: PipelineConfig) -> dict[str, Any]:
    """Read back the applied USD state; configuration intent alone is insufficient."""

    from pxr import UsdGeom, UsdShade

    stage = env.sim.stage
    camera_mounts = {}
    for camera in config.cameras:
        suffix = camera.prim_path.replace("{ENV_REGEX_NS}/", "")
        matches = [prim for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one spawned {camera.sim_sensor}, found {len(matches)}")
        camera_mounts[camera.source_key] = _prim_local_transform(matches[0])

    lights = []
    textures = []
    visible_props = []
    room_records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/FlipTableEvalLighting/" in path and prim.GetTypeName() in {
            "SphereLight",
            "DistantLight",
        }:
            attributes = {}
            for name in (
                "inputs:intensity",
                "inputs:colorTemperature",
                "inputs:exposure",
                "inputs:radius",
            ):
                attribute = prim.GetAttribute(name)
                if attribute and attribute.HasAuthoredValueOpinion():
                    attributes[name.removeprefix("inputs:")] = _jsonable(attribute.Get())
            lights.append(
                {
                    "path": path,
                    "visible": str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible",
                    "transform": _prim_local_transform(prim),
                    **attributes,
                }
            )
        if "/FlipTableEvalPropPool/Slot_" in path and prim.GetTypeName() == "Xform":
            imageable = UsdGeom.Imageable(prim)
            if imageable and str(imageable.ComputeVisibility()) != "invisible" and path.count("/") >= 7:
                visible_props.append({"path": path, "transform": _prim_local_transform(prim)})
        if path.startswith("/World/Looks/flip_table_room_env_"):
            for child in prim.GetChildren():
                for attribute in child.GetAttributes():
                    if attribute.GetName() == "inputs:file" and attribute.HasAuthoredValueOpinion():
                        textures.append(str(attribute.Get()))
        if "/FlipTableEvalRoom/" in path and prim.GetTypeName() in {"Cube", "Mesh"}:
            visible = str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible"
            if visible:
                bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                room_records.append(
                    {"path": path, "material": str(bound.GetPath()) if bound else ""}
                )
    room_records.sort(key=lambda value: value["path"])
    encoded_room = json.dumps(room_records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "camera_mounts": camera_mounts,
        "lights": sorted(lights, key=lambda value: value["path"]),
        "room": {
            "visible_surface_count": len(room_records),
            "visible_surface_manifest_sha256": hashlib.sha256(encoded_room).hexdigest(),
            "texture_assets": sorted(set(textures)),
            "visible_props": sorted(visible_props, key=lambda value: value["path"]),
        },
    }
