#!/usr/bin/env python3
"""Verify that resetting selected environments preserves every other task state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--settle_steps", type=int, default=2)
parser.add_argument("--verify_rendered_pixels", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
args_cli.enable_cameras = args_cli.verify_rendered_pixels
args_cli.rl = "FlipTableResidualStateRL"

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.place_utils.env_utils import set_seed


POLICY_CAMERA_NAMES = (
    "first_person_camera",
    "left_hand_camera",
    "right_hand_camera",
)


def _camera_snapshot(env, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Render and clone RGB observations for selected environments."""

    env.sim.render()
    env.cfg.isaaclab_arena_env.task._refresh_camera_sensors(env)
    snapshots = {}
    for sensor_name in POLICY_CAMERA_NAMES:
        sensor = env.scene.sensors.get(sensor_name)
        if sensor is None:
            raise RuntimeError(f"required policy camera is missing: {sensor_name}")
        output = getattr(sensor.data, "output", {})
        if "rgb" not in output:
            raise RuntimeError(f"policy camera has no RGB output: {sensor_name}")
        rgb = as_torch(output["rgb"])[env_ids, ..., :3]
        snapshots[sensor_name] = rgb.to(dtype=torch.float32).clone()
    return snapshots


def _pixel_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    metrics = {}
    if before.keys() != after.keys():
        raise ValueError("camera snapshots contain different sensors")
    for sensor_name in before:
        if before[sensor_name].shape != after[sensor_name].shape:
            raise ValueError(f"camera shape changed for {sensor_name}")
        delta = torch.abs(after[sensor_name] - before[sensor_name])
        metrics[sensor_name] = {
            "mean_abs_delta": float(delta.mean().item()),
            "p99_abs_delta": float(torch.quantile(delta.flatten(), 0.99).item()),
            "max_abs_delta": float(delta.max().item()),
            "fraction_over_1": float((delta > 1.0).float().mean().item()),
        }
    return metrics


def _untouched_pixels_preserved(
    baseline: dict[str, dict[str, float]],
    after_reset: dict[str, dict[str, float]],
) -> bool:
    """Allow renderer jitter while rejecting reset-induced cross-env changes."""

    for sensor_name in POLICY_CAMERA_NAMES:
        baseline_mean = baseline[sensor_name]["mean_abs_delta"]
        baseline_p99 = baseline[sensor_name]["p99_abs_delta"]
        allowed_mean = max(0.5, 3.0 * baseline_mean + 0.25)
        allowed_p99 = max(2.0, 3.0 * baseline_p99 + 1.0)
        if after_reset[sensor_name]["mean_abs_delta"] > allowed_mean:
            return False
        if after_reset[sensor_name]["p99_abs_delta"] > allowed_p99:
            return False
    return True


def _lighting_snapshot(stage, env_ids: torch.Tensor) -> dict[str, dict[str, str]]:
    """Capture only per-environment light state in a JSON-stable form."""

    snapshots: dict[str, dict[str, str]] = {}
    selected_ids = {int(value) for value in env_ids.detach().cpu().tolist()}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/FlipTableEvalLighting/" not in path:
            continue
        env_token = path.split("/World/envs/env_", 1)[-1].split("/", 1)[0]
        if not env_token.isdigit() or int(env_token) not in selected_ids:
            continue
        record = {"type": str(prim.GetTypeName())}
        for attribute_name in (
            "inputs:intensity",
            "inputs:radius",
            "inputs:exposure",
            "inputs:colorTemperature",
            "visibility",
            "xformOp:translate",
        ):
            attribute = prim.GetAttribute(attribute_name)
            if attribute and attribute.HasAuthoredValue():
                record[attribute_name] = repr(attribute.Get())
        for relationship_name in (
            "collection:lightLink:includes",
            "collection:shadowLink:includes",
        ):
            relationship = prim.GetRelationship(relationship_name)
            if relationship:
                record[relationship_name] = repr(
                    [str(target) for target in relationship.GetTargets()]
                )
        snapshots[path] = record
    return snapshots


def _contact_material_snapshot(stage, env_ids: torch.Tensor) -> dict[str, dict[str, str]]:
    snapshots: dict[str, dict[str, str]] = {}
    selected_ids = {int(value) for value in env_ids.detach().cpu().tolist()}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Looks/flip_table_contact_" not in path:
            continue
        env_token = path.split("/World/envs/env_", 1)[-1].split("/", 1)[0]
        if not env_token.isdigit() or int(env_token) not in selected_ids:
            continue
        snapshots[path] = {
            name: repr(prim.GetAttribute(name).Get())
            for name in (
                "physics:staticFriction",
                "physics:dynamicFriction",
                "physics:restitution",
            )
        }
    return snapshots


def _room_snapshot_digest(stage, env_ids: torch.Tensor) -> dict[str, object]:
    selected_ids = {int(value) for value in env_ids.detach().cpu().tolist()}
    records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/FlipTableEvalRoom/" not in path:
            continue
        env_token = path.split("/World/envs/env_", 1)[-1].split("/", 1)[0]
        if not env_token.isdigit() or int(env_token) not in selected_ids:
            continue
        attributes = {
            attribute.GetName(): repr(attribute.Get())
            for attribute in prim.GetAttributes()
            if attribute.HasAuthoredValue()
        }
        records.append((path, str(prim.GetTypeName()), attributes))
    payload = json.dumps(sorted(records), sort_keys=True, separators=(",", ":"))
    return {
        "prim_count": len(records),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _mass_snapshot(env) -> dict[str, torch.Tensor]:
    snapshots = {}
    for entity_name in (
        "Table001_Table001_01",
        "Leg001_Leg001",
        "Leg001_01_Leg001",
        "Leg001_03_Leg001",
        "Leg001_06_Leg001",
    ):
        snapshots[entity_name] = torch.as_tensor(
            env.scene[entity_name].data.body_mass,
            device=env.device,
        ).reshape(env.num_envs, -1).clone()
    return snapshots


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    if args_cli.num_envs < 2:
        raise ValueError("partial-reset audit requires at least two environments")
    if args_cli.settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    if args_cli.seed is None:
        raise ValueError("a deterministic --seed is required")

    env_cfg = parse_env_cfg(
        scene_backend=args_cli.scene_backend,
        task_backend=args_cli.task_backend,
        task_name=args_cli.task,
        robot_name=args_cli.robot,
        scene_name=args_cli.layout,
        rl_name=args_cli.rl,
        robot_scale=args_cli.robot_scale,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        first_person_view=args_cli.first_person_view,
        enable_cameras=args_cli.verify_rendered_pixels,
        execute_mode=ExecuteMode.TRAIN,
        usd_simplify=args_cli.usd_simplify,
        seed=args_cli.seed,
        sources=args_cli.sources,
        object_projects=args_cli.object_projects,
        headless_mode=args_cli.headless,
        enable_full_local_scene=args_cli.enable_full_local_scene,
    )
    env_cfg.terminations.success = None
    env_cfg.seed = int(args_cli.seed)
    task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
    if task_name not in gym.registry:
        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    set_seed(env_cfg.seed, env)
    env.reset()
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        dtype=torch.float32,
        device=env.device,
    )
    for _step in range(args_cli.settle_steps):
        env.step(actions)

    task = env.cfg.isaaclab_arena_env.task
    before_normal = task._initial_table_normal.clone()
    before_position = task._initial_table_pos.clone()
    reset_ids = torch.tensor([env.num_envs - 1], dtype=torch.long, device=env.device)
    untouched_ids = torch.arange(env.num_envs - 1, dtype=torch.long, device=env.device)
    before_untouched_lighting = _lighting_snapshot(env.sim.stage, untouched_ids)
    before_reset_lighting = _lighting_snapshot(env.sim.stage, reset_ids)
    before_untouched_materials = _contact_material_snapshot(env.sim.stage, untouched_ids)
    before_reset_materials = _contact_material_snapshot(env.sim.stage, reset_ids)
    before_untouched_room = _room_snapshot_digest(env.sim.stage, untouched_ids)
    before_reset_room = _room_snapshot_digest(env.sim.stage, reset_ids)
    before_mass = _mass_snapshot(env)
    rendered_pixel_baseline = None
    untouched_rendered_pixel_delta = None
    reset_rendered_pixel_delta = None
    untouched_rendered_pixels_preserved = True
    reset_rendered_pixels_changed = True
    before_pixels = None
    if args_cli.verify_rendered_pixels:
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        baseline_pixels = _camera_snapshot(env, all_ids)
        before_pixels = _camera_snapshot(env, all_ids)
        rendered_pixel_baseline = _pixel_delta(
            {
                name: value[untouched_ids].clone()
                for name, value in baseline_pixels.items()
            },
            {
                name: value[untouched_ids].clone()
                for name, value in before_pixels.items()
            },
        )

    env._reset_idx(reset_ids)

    after_normal = task._initial_table_normal.clone()
    after_position = task._initial_table_pos.clone()
    normal_error = torch.abs(after_normal[untouched_ids] - before_normal[untouched_ids])
    position_error = torch.abs(after_position[untouched_ids] - before_position[untouched_ids])
    reset_normal_norm = torch.linalg.norm(after_normal[reset_ids], dim=1)
    after_untouched_lighting = _lighting_snapshot(env.sim.stage, untouched_ids)
    after_reset_lighting = _lighting_snapshot(env.sim.stage, reset_ids)
    after_untouched_materials = _contact_material_snapshot(env.sim.stage, untouched_ids)
    after_reset_materials = _contact_material_snapshot(env.sim.stage, reset_ids)
    after_untouched_room = _room_snapshot_digest(env.sim.stage, untouched_ids)
    after_reset_room = _room_snapshot_digest(env.sim.stage, reset_ids)
    after_mass = _mass_snapshot(env)
    if args_cli.verify_rendered_pixels:
        if before_pixels is None or rendered_pixel_baseline is None:
            raise RuntimeError("rendered-pixel audit was not initialized")
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        after_pixels = _camera_snapshot(env, all_ids)
        untouched_rendered_pixel_delta = _pixel_delta(
            {
                name: value[untouched_ids].clone()
                for name, value in before_pixels.items()
            },
            {
                name: value[untouched_ids].clone()
                for name, value in after_pixels.items()
            },
        )
        reset_rendered_pixel_delta = _pixel_delta(
            {
                name: value[reset_ids].clone()
                for name, value in before_pixels.items()
            },
            {
                name: value[reset_ids].clone()
                for name, value in after_pixels.items()
            },
        )
        untouched_rendered_pixels_preserved = _untouched_pixels_preserved(
            rendered_pixel_baseline,
            untouched_rendered_pixel_delta,
        )
        reset_rendered_pixels_changed = any(
            metrics["mean_abs_delta"] > 0.5
            for metrics in reset_rendered_pixel_delta.values()
        )
    lighting_randomization_enabled = _enabled("FLIP_TABLE_RANDOMIZE_LIGHTING")
    contact_randomization_enabled = _enabled("FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS")
    room_randomization_enabled = _enabled("FLIP_TABLE_RANDOMIZE_ROOM")
    mass_randomization_enabled = _enabled("FLIP_TABLE_RL_RANDOMIZE_MASS")
    untouched_lighting_preserved = before_untouched_lighting == after_untouched_lighting
    reset_lighting_changed = before_reset_lighting != after_reset_lighting
    untouched_contact_materials_preserved = (
        before_untouched_materials == after_untouched_materials
    )
    reset_contact_materials_changed = before_reset_materials != after_reset_materials
    untouched_room_preserved = before_untouched_room == after_untouched_room
    reset_room_changed = before_reset_room != after_reset_room
    max_untouched_mass_error_kg = max(
        float(
            torch.abs(after_mass[name][untouched_ids] - before_mass[name][untouched_ids])
            .max()
            .item()
        )
        for name in before_mass
    )
    reset_mass_changed = any(
        not torch.equal(after_mass[name][reset_ids], before_mass[name][reset_ids])
        for name in before_mass
    )
    distant_light_paths = sorted(
        path
        for path, record in {
            **after_untouched_lighting,
            **after_reset_lighting,
        }.items()
        if record["type"] == "DistantLight"
    )
    incorrect_light_links = sorted(
        path
        for path, record in {
            **after_untouched_lighting,
            **after_reset_lighting,
        }.items()
        if record.get("collection:lightLink:includes")
        != repr([path.split("/FlipTableEvalLighting/", 1)[0]])
        or record.get("collection:shadowLink:includes")
        != repr([path.split("/FlipTableEvalLighting/", 1)[0]])
    )
    pass_contract = bool(
        torch.equal(after_normal[untouched_ids], before_normal[untouched_ids])
        and torch.equal(after_position[untouched_ids], before_position[untouched_ids])
        and torch.isfinite(after_normal).all()
        and torch.all(reset_normal_norm > 0.99)
        and torch.all(reset_normal_norm < 1.01)
        and untouched_lighting_preserved
        and not distant_light_paths
        and not incorrect_light_links
        and (not lighting_randomization_enabled or reset_lighting_changed)
        and untouched_contact_materials_preserved
        and (not contact_randomization_enabled or reset_contact_materials_changed)
        and untouched_room_preserved
        and (not room_randomization_enabled or reset_room_changed)
        and max_untouched_mass_error_kg == 0.0
        and (not mass_randomization_enabled or reset_mass_changed)
        and untouched_rendered_pixels_preserved
        and reset_rendered_pixels_changed
    )
    report = {
        "method": "manager_based_rl_partial_reset_state_isolation",
        "num_envs": env.num_envs,
        "reset_env_ids": reset_ids.detach().cpu().tolist(),
        "untouched_env_ids": untouched_ids.detach().cpu().tolist(),
        "max_untouched_initial_normal_error": float(normal_error.max().item()),
        "max_untouched_initial_position_error_m": float(position_error.max().item()),
        "reset_initial_normal_norm": reset_normal_norm.detach().cpu().tolist(),
        "before_initial_normal": before_normal.detach().cpu().tolist(),
        "after_initial_normal": after_normal.detach().cpu().tolist(),
        "lighting_randomization_enabled": lighting_randomization_enabled,
        "untouched_lighting_preserved": untouched_lighting_preserved,
        "reset_lighting_changed": reset_lighting_changed,
        "distant_light_paths": distant_light_paths,
        "incorrect_light_links": incorrect_light_links,
        "before_untouched_lighting": before_untouched_lighting,
        "after_untouched_lighting": after_untouched_lighting,
        "contact_randomization_enabled": contact_randomization_enabled,
        "untouched_contact_materials_preserved": untouched_contact_materials_preserved,
        "reset_contact_materials_changed": reset_contact_materials_changed,
        "room_randomization_enabled": room_randomization_enabled,
        "untouched_room_preserved": untouched_room_preserved,
        "reset_room_changed": reset_room_changed,
        "before_untouched_room": before_untouched_room,
        "after_untouched_room": after_untouched_room,
        "mass_randomization_enabled": mass_randomization_enabled,
        "max_untouched_mass_error_kg": max_untouched_mass_error_kg,
        "reset_mass_changed": reset_mass_changed,
        "rendered_pixel_verification_enabled": args_cli.verify_rendered_pixels,
        "rendered_pixel_baseline": rendered_pixel_baseline,
        "untouched_rendered_pixel_delta": untouched_rendered_pixel_delta,
        "reset_rendered_pixel_delta": reset_rendered_pixel_delta,
        "untouched_rendered_pixels_preserved": untouched_rendered_pixels_preserved,
        "reset_rendered_pixels_changed": reset_rendered_pixels_changed,
        "pass": pass_contract,
        "privileged_state_use": "simulation contract audit only; never a policy input",
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    env.close()
    if not pass_contract:
        raise RuntimeError("partial reset corrupted another environment's task state")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
