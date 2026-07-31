#!/usr/bin/env python3
"""Evaluate a flip-table SKRL checkpoint and save videos plus action/state traces."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    raise ValueError("recorded checkpoint evaluation currently requires --num_envs 1")
if args_cli.episodes <= 0:
    raise ValueError("--episodes must be positive")
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

from robofinals.scripts.rl.env_wrapper import SkrlVecEnvWrapper
from robofinals.utils.env import ExecuteMode, load_cfg_cls_from_registry, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.place_utils.env_utils import set_seed
from robofinals_rl.flip_table import mdp
from robofinals_rl.flip_table.common import dex1_joint_to_command
from robofinals_rl.flip_table.mdp.observations import controller_joint_state_raw


CAMERAS = (
    ("head_left", "first_person_camera"),
    ("left_wrist", "left_hand_camera"),
    ("right_wrist", "right_hand_camera"),
    ("global", "global_camera"),
)

STATE_POLICY_INPUTS = [
    "upper_body_joint_position_and_dex1_command",
    "upper_body_joint_velocity",
    "real_demo_prior",
    "controller_action_prior",
    "controller_action_prior_phase",
    "previous_policy_action",
]
VISUAL_POLICY_INPUTS = [
    "head_left_rgb",
    "left_d405_rgb",
    "right_d405_rgb",
    *STATE_POLICY_INPUTS,
]


def _frame(raw_env, sensor_name: str) -> np.ndarray:
    sensor = raw_env.scene.sensors[sensor_name]
    tensor = as_torch(sensor.data.output["rgb"])
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tuple(tensor.shape[1:3]) != (480, 640):
        raise ValueError(f"{sensor_name} must provide [1,480,640,C] RGB, got {tuple(tensor.shape)}")
    if tensor.shape[-1] < 3 or not torch.isfinite(tensor).all():
        raise ValueError(f"{sensor_name} must provide at least three finite RGB channels")
    value = tensor[0, ..., :3].detach().cpu().numpy()
    if value.dtype != np.uint8:
        scale = 255.0 if float(value.max(initial=0.0)) <= 1.0 else 1.0
        value = np.clip(value * scale, 0.0, 255.0).astype(np.uint8)
    return value


def _open_writers(directory: Path, fps: int):
    directory.mkdir(parents=True, exist_ok=True)
    writers = {
        label: imageio.get_writer(
            directory / f"{label}.mp4",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        for label, _sensor_name in CAMERAS
    }
    writers["composite"] = imageio.get_writer(
        directory / "composite.mp4",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    return writers


def _append_frames(raw_env, writers) -> None:
    frames = {label: _frame(raw_env, sensor_name) for label, sensor_name in CAMERAS}
    for label, frame in frames.items():
        writers[label].append_data(frame)
    top = np.concatenate((frames["head_left"], frames["global"]), axis=1)
    bottom = np.concatenate((frames["left_wrist"], frames["right_wrist"]), axis=1)
    writers["composite"].append_data(np.concatenate((top, bottom), axis=0))


def _close_writers(writers) -> None:
    for writer in writers.values():
        writer.close()


def _mean_action(agent, observation: torch.Tensor) -> torch.Tensor:
    parameters = inspect.signature(agent.act).parameters
    if "observations" in parameters:
        sampled, extras = agent.act(observation, None, timestep=0, timesteps=0)
    else:
        outputs = agent.act(observation, timestep=0, timesteps=0)
        sampled, extras = outputs[0], outputs[-1]
    return extras.get("mean_actions", sampled)


def _constant_action_from_env(action_dim: int, device: str) -> torch.Tensor | None:
    raw = os.environ.get("FLIP_TABLE_RL_EVAL_CONSTANT_ACTION", "").strip()
    if not raw:
        return None
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != action_dim:
        raise ValueError(
            f"FLIP_TABLE_RL_EVAL_CONSTANT_ACTION must have {action_dim} values, got {len(values)}"
        )
    action = torch.tensor(values, dtype=torch.float32, device=device)
    if not torch.isfinite(action).all():
        raise ValueError("FLIP_TABLE_RL_EVAL_CONSTANT_ACTION contains NaN or Inf")
    return action.clamp(-1.0, 1.0).unsqueeze(0)


def _validate_policy_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action.shape != (1, action_dim):
        raise ValueError(f"policy action must have shape (1, {action_dim}), got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("policy action contains NaN or Inf")
    return action


def _processed_controller_target(raw_env) -> torch.Tensor:
    manager = raw_env.action_manager
    try:
        body = manager.get_term("base_action").target_robot_joints_mujoco
    except KeyError:
        # Retained only for fixed_diagnostic replay of historical policies.
        legacy = torch.cat(
            [
                manager.get_term(name)._processed_actions
                for name in ("waist_action", "left_arm_action", "right_arm_action")
            ],
            dim=1,
        )
        body = legacy[:, 3:17]
    if body.shape[1] != 14:
        raise RuntimeError(f"controller arm target must be [B,14], got {tuple(body.shape)}")
    hand_commands = []
    for name in ("left_hand_action", "right_hand_action"):
        joints = manager.get_term(name)._processed_actions
        hand_commands.append(dex1_joint_to_command(joints).mean(dim=1, keepdim=True))
    return torch.cat((body, *hand_commands), dim=1)


def _trace_row(raw_env, policy_action: torch.Tensor, reward: torch.Tensor, step: int) -> dict:
    table_pos, table_quat = raw_env.cfg.isaaclab_arena_env.task._table_body_pose(raw_env)
    distances = mdp.bimanual_reach_distances(raw_env)
    contact_forces = mdp.bimanual_contact_forces(raw_env)
    return {
        "step": step,
        "time_s": step * float(raw_env.step_dt),
        "policy_residual_action": policy_action[0].detach().cpu().tolist(),
        "controller_target": _processed_controller_target(raw_env)[0].detach().cpu().tolist(),
        # Diagnostics must not draw fresh observation noise: doing so would
        # change the RNG stream and therefore the policy trajectory being audited.
        "controller_state": controller_joint_state_raw(raw_env)[0].detach().cpu().tolist(),
        "reward": float(reward[0].item()),
        "privileged_diagnostics": {
            "hand_leg_distance_m": distances[0].detach().cpu().tolist(),
            "hand_contact_force_n": contact_forces[0].detach().cpu().tolist(),
            "bimanual_contact": float(mdp.bimanual_contact(raw_env)[0].item()),
            "bimanual_grasp": float(mdp.bimanual_grasp(raw_env)[0].item()),
            "table_lift_progress": float(mdp.table_lift_progress(raw_env)[0].item()),
            "table_flip_progress": float(mdp.table_flip_progress(raw_env)[0].item()),
            "stage_success": bool(mdp.table_stage_success(raw_env)[0].item()),
            "stable_success": bool(mdp.table_stable_success(raw_env)[0].item()),
            "table_position_world_m": table_pos[0].detach().cpu().tolist(),
            "table_quaternion_xyzw": table_quat[0].detach().cpu().tolist(),
        },
    }


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint)
    output_path = Path(args_cli.output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(
        scene_backend=args_cli.scene_backend,
        task_backend=args_cli.task_backend,
        task_name=args_cli.task,
        robot_name=args_cli.robot,
        scene_name=args_cli.layout,
        rl_name=args_cli.rl,
        robot_scale=args_cli.robot_scale,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        first_person_view=args_cli.first_person_view,
        enable_cameras=True,
        execute_mode=ExecuteMode.EVAL,
        usd_simplify=args_cli.usd_simplify,
        seed=args_cli.seed,
        sources=args_cli.sources,
        object_projects=args_cli.object_projects,
        headless_mode=args_cli.headless,
        enable_full_local_scene=args_cli.enable_full_local_scene,
    )
    # Preserve the successful terminal pose for the final video frame and
    # diagnostics instead of letting ManagerBasedRLEnv auto-reset it in step().
    env_cfg.terminations.success = None
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
    gym_env = gym.make(task_name, cfg=env_cfg)
    raw_env = gym_env.unwrapped
    set_seed(env_cfg.seed, raw_env)
    missing_cameras = [name for _label, name in CAMERAS if name not in raw_env.scene.sensors]
    if missing_cameras:
        raise RuntimeError(f"required evaluation cameras are missing: {missing_cameras}")

    agent_cfg = load_cfg_cls_from_registry("rl", args_cli.rl, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    env = SkrlVecEnvWrapper(gym_env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(checkpoint_path.resolve()))
    agent = runner.agent
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    elif hasattr(agent, "enable_training_mode"):
        agent.enable_training_mode(False, apply_to_models=True)
    constant_action = _constant_action_from_env(raw_env.action_manager.total_action_dim, raw_env.device)

    fps = max(1, round(1.0 / float(raw_env.step_dt)))
    max_steps = max(1, round(float(raw_env.cfg.episode_length_s) / float(raw_env.step_dt)))
    episode_summaries = []
    for episode in range(args_cli.episodes):
        episode_dir = output_path / f"episode_{episode:03d}"
        writers = _open_writers(episode_dir, fps)
        trace_path = episode_dir / "trace.jsonl"
        try:
            observation, _info = env.reset()
            success = False
            _append_frames(raw_env, writers)
            with trace_path.open("w", encoding="utf-8") as trace_file:
                for step in range(max_steps):
                    with torch.inference_mode():
                        action = constant_action if constant_action is not None else _mean_action(agent, observation)
                        action = _validate_policy_action(
                            action,
                            raw_env.action_manager.total_action_dim,
                        )
                        observation, reward, terminated, truncated, _extras = env.step(action)
                    row = _trace_row(raw_env, action, reward, step)
                    row["terminated"] = bool(terminated[0].item())
                    row["truncated"] = bool(truncated[0].item())
                    success = success or row["privileged_diagnostics"]["stage_success"]
                    trace_file.write(json.dumps(row, allow_nan=False) + "\n")
                    _append_frames(raw_env, writers)
                    if success or row["terminated"] or row["truncated"]:
                        break
        finally:
            _close_writers(writers)
        summary = {
            "episode": episode,
            "success": success,
            "steps": step + 1,
            "duration_s": (step + 1) * float(raw_env.step_dt),
            "trace": str(trace_path),
        }
        (episode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        episode_summaries.append(summary)
        print(json.dumps(summary), flush=True)

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "episodes": args_cli.episodes,
        "successes": sum(item["success"] for item in episode_summaries),
        "success_rate": sum(item["success"] for item in episode_summaries) / args_cli.episodes,
        "step_dt_s": float(raw_env.step_dt),
        "camera_resolution": [640, 480],
        "policy_mode": "visual" if "Visual" in args_cli.rl else "state",
        "policy_inputs": VISUAL_POLICY_INPUTS if "Visual" in args_cli.rl else STATE_POLICY_INPUTS,
        "policy_output": "16D arm-and-hand residual converted to real-compatible joint targets",
        "action_source": "constant_scripted_teacher" if constant_action is not None else "checkpoint_mean",
        "constant_residual_action": (
            constant_action[0].detach().cpu().tolist() if constant_action is not None else None
        ),
        "privileged_state_use": "reward, success evaluation, and trace diagnostics only",
        "episode_results": episode_summaries,
    }
    (output_path / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
