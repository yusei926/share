#!/usr/bin/env python3
"""Run the real-demo residual baseline and report reward/physics metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import matrix_from_quat

from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.place_utils.env_utils import set_seed
from robofinals_rl.flip_table.flip_table import mdp


def _smoke_action(num_envs: int, action_dim: int, device: str) -> torch.Tensor:
    raw = os.environ.get("FLIP_TABLE_RL_SMOKE_ACTION", "").strip()
    if not raw:
        return torch.zeros((num_envs, action_dim), device=device)
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != action_dim:
        raise ValueError(f"FLIP_TABLE_RL_SMOKE_ACTION must have {action_dim} values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32, device=device).expand(num_envs, -1).clone()


def main() -> None:
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
        enable_cameras=bool(args_cli.enable_cameras),
        execute_mode=ExecuteMode.TRAIN,
        usd_simplify=args_cli.usd_simplify,
        seed=args_cli.seed,
        sources=args_cli.sources,
        object_projects=args_cli.object_projects,
        headless_mode=args_cli.headless,
        enable_full_local_scene=args_cli.enable_full_local_scene,
    )
    env_cfg.observations.policy.concatenate_terms = True
    if args_cli.seed is None:
        raise ValueError("a deterministic --seed is required")
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
    observation, _ = env.reset()
    rollout_started_at = time.perf_counter()
    action = _smoke_action(env.num_envs, env.action_manager.total_action_dim, env.device)
    initial_table_pos, initial_table_quat = env.cfg.isaaclab_arena_env.task._table_body_pose(env)
    initial_table_rotation = matrix_from_quat(initial_table_quat)
    initial_leg_positions = mdp.leg_positions(env)
    initial_leg_offsets_table = torch.matmul(
        initial_table_rotation.transpose(1, 2).unsqueeze(1),
        (initial_leg_positions - initial_table_pos[:, None, :]).unsqueeze(-1),
    ).squeeze(-1)

    samples = []
    success_count = 0
    termination_count = 0
    truncation_count = 0
    maxima = {
        "contact": 0.0,
        "all_surface_finger_force_n": 0.0,
        "white_leg_attributed_force_n": 0.0,
        "grasp": 0.0,
        "lift": 0.0,
        "flip_progress": 0.0,
        "demo_progress": 0,
        "single_hand_grasp": {"left": 0.0, "right": 0.0},
        "assembly_drift_m": 0.0,
        "assembly_drift_per_leg_m": [0.0, 0.0, 0.0, 0.0],
    }
    best_reach = {
        "minimax_distance_m": float("inf"),
        "step": -1,
        "demo_progress": -1,
        "per_hand_distance_m": [],
    }
    best_nearest_leg = {
        side: {"distance_m": float("inf"), "step": -1, "demo_progress": -1}
        for side in ("left", "right")
    }
    best_finger_alignment = {
        side: {"cost_m": float("inf"), "step": -1, "demo_progress": -1}
        for side in ("left", "right")
    }
    for step in range(args_cli.steps):
        observation, reward, terminated, truncated, extras = env.step(action)
        progress_values = torch.stack(
            [
                env.action_manager.get_term(name)._progress
                for name in env.action_manager.active_terms
                if hasattr(env.action_manager.get_term(name), "_progress")
            ],
            dim=1,
        )
        prior_progress = env._flip_table_rl_demo_prior_term._progress
        reach_distances = mdp.bimanual_reach_distances(env)
        nearest_distances = mdp.nearest_leg_distances(env)
        alignment_costs = mdp.finger_leg_alignment_cost(env)
        finger_forces = mdp.finger_contact_forces(env)
        white_leg_forces = mdp.white_table_leg_contact_forces(env)
        contact_forces = mdp.bimanual_contact_forces(env)
        contact = mdp.bimanual_contact(env)
        grasp = mdp.bimanual_grasp(env)
        single_hand_grasps = {
            side: mdp.single_hand_grasp(env, side=side)
            for side in ("left", "right")
        }
        lift = mdp.table_lift_progress(env)
        flip_progress = mdp.table_flip_progress(env)
        table_pos, table_quat = env.cfg.isaaclab_arena_env.task._table_body_pose(env)
        table_rotation = matrix_from_quat(table_quat)
        leg_offsets_table = torch.matmul(
            table_rotation.transpose(1, 2).unsqueeze(1),
            (mdp.leg_positions(env) - table_pos[:, None, :]).unsqueeze(-1),
        ).squeeze(-1)
        assembly_drift = torch.linalg.norm(
            leg_offsets_table - initial_leg_offsets_table,
            dim=-1,
        )
        metrics = {
            "step": step,
            "reward_mean": float(reward.mean().item()),
            "reach_mean": float(mdp.bimanual_reach(env).mean().item()),
            "reach_distance_mean_m": float(reach_distances.mean().item()),
            "reach_distance_per_hand_mean_m": reach_distances.mean(dim=0).detach().cpu().tolist(),
            "finger_alignment_cost_per_hand_mean_m": (
                alignment_costs.mean(dim=0).detach().cpu().tolist()
            ),
            "contact_force_per_hand_max_n": contact_forces.amax(dim=0).detach().cpu().tolist(),
            "contact_force_per_finger_max_n": finger_forces.amax(dim=0).detach().cpu().tolist(),
            "white_leg_attributed_force_per_finger_and_leg_max_n": (
                white_leg_forces.amax(dim=0).detach().cpu().tolist()
            ),
            "single_hand_grasp_mean": {
                side: float(single_hand_grasps[side].mean().item())
                for side in ("left", "right")
            },
            "contact_mean": float(contact.mean().item()),
            "grasp_mean": float(grasp.mean().item()),
            "lift_mean": float(lift.mean().item()),
            "flip_progress_mean": float(flip_progress.mean().item()),
            "assembly_drift_max_m": float(assembly_drift.max().item()),
            "assembly_drift_per_leg_max_m": assembly_drift.amax(dim=0).detach().cpu().tolist(),
            "stage_success_count": int(mdp.table_stage_success(env).sum().item()),
            "stable_success_count": int(mdp.table_stable_success(env).sum().item()),
            "demo_progress_mean": float(progress_values.float().mean().item()),
            "demo_progress_spread": int(
                (progress_values.max(dim=1).values - progress_values.min(dim=1).values).max().item()
            ),
            "observation_prior_progress_mean": float(prior_progress.float().mean().item()),
        }
        minimax = reach_distances.amax(dim=1)
        best_env = int(minimax.argmin().item())
        if float(minimax[best_env].item()) < best_reach["minimax_distance_m"]:
            best_reach = {
                "minimax_distance_m": float(minimax[best_env].item()),
                "step": step,
                "demo_progress": int(progress_values[best_env, 0].item()),
                "per_hand_distance_m": reach_distances[best_env].detach().cpu().tolist(),
            }
        for side_index, side in enumerate(("left", "right")):
            nearest_value, nearest_env = nearest_distances[:, side_index].min(dim=0)
            if float(nearest_value.item()) < best_nearest_leg[side]["distance_m"]:
                best_nearest_leg[side] = {
                    "distance_m": float(nearest_value.item()),
                    "step": step,
                    "demo_progress": int(progress_values[int(nearest_env.item()), 0].item()),
                }
            alignment_value, alignment_env = alignment_costs[:, side_index].min(dim=0)
            if float(alignment_value.item()) < best_finger_alignment[side]["cost_m"]:
                best_finger_alignment[side] = {
                    "cost_m": float(alignment_value.item()),
                    "step": step,
                    "demo_progress": int(progress_values[int(alignment_env.item()), 0].item()),
                }
        maxima["contact"] = max(maxima["contact"], float(contact.max().item()))
        maxima["all_surface_finger_force_n"] = max(
            maxima["all_surface_finger_force_n"],
            float(finger_forces.max().item()),
        )
        maxima["white_leg_attributed_force_n"] = max(
            maxima["white_leg_attributed_force_n"],
            float(white_leg_forces.max().item()),
        )
        maxima["grasp"] = max(maxima["grasp"], float(grasp.max().item()))
        maxima["lift"] = max(maxima["lift"], float(lift.max().item()))
        maxima["flip_progress"] = max(maxima["flip_progress"], float(flip_progress.max().item()))
        maxima["assembly_drift_m"] = max(
            maxima["assembly_drift_m"],
            float(assembly_drift.max().item()),
        )
        maxima["assembly_drift_per_leg_m"] = [
            max(previous, current)
            for previous, current in zip(
                maxima["assembly_drift_per_leg_m"],
                assembly_drift.amax(dim=0).detach().cpu().tolist(),
            )
        ]
        maxima["demo_progress"] = max(maxima["demo_progress"], int(progress_values.max().item()))
        for side in ("left", "right"):
            maxima["single_hand_grasp"][side] = max(
                maxima["single_hand_grasp"][side],
                float(single_hand_grasps[side].max().item()),
            )
        non_timeout_termination = terminated & ~truncated
        termination_count += int(non_timeout_termination.sum().item())
        truncation_count += int(truncated.sum().item())
        success_count += metrics["stage_success_count"]
        if step == 0 or (step + 1) % 10 == 0 or bool(torch.any(terminated | truncated)):
            samples.append(metrics)
            print(json.dumps(metrics), flush=True)
        if bool(torch.any(terminated | truncated)):
            break

    mass_samples = {}
    for entity_name in (
        "Table001_Table001_01",
        "Leg001_Leg001",
        "Leg001_01_Leg001",
        "Leg001_03_Leg001",
        "Leg001_06_Leg001",
    ):
        entity = env.scene[entity_name]
        mass_samples[entity_name] = as_torch(entity.data.body_mass).reshape(env.num_envs, -1)[:, 0].tolist()
    table_pos, _table_quat = env.cfg.isaaclab_arena_env.task._table_body_pose(env)
    rollout_seconds = time.perf_counter() - rollout_started_at
    steps_completed = step + 1
    report = {
        "num_envs": env.num_envs,
        "steps_requested": args_cli.steps,
        "steps_completed": steps_completed,
        "rollout_seconds": rollout_seconds,
        "environment_steps_per_second": steps_completed / rollout_seconds,
        "transitions_per_second": (steps_completed * env.num_envs) / rollout_seconds,
        "action_dim": env.action_manager.total_action_dim,
        "constant_residual_action": action[0].detach().cpu().tolist(),
        "observation_shape": list(observation["policy"].shape),
        "success_observations": success_count,
        "non_timeout_terminations": termination_count,
        "timeout_truncations": truncation_count,
        "randomization": {
            "level": float(os.environ.get("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", "1.0")),
            "action_delay_steps": env._flip_table_rl_action_delay_steps.tolist(),
            "body_mass_kg": mass_samples,
            "table_position_world_m": table_pos.tolist(),
        },
        "maxima": maxima,
        "best_simultaneous_reach": best_reach,
        "best_nearest_leg": best_nearest_leg,
        "best_finger_alignment": best_finger_alignment,
        "samples": samples,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
