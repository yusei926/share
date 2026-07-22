#!/usr/bin/env python3
"""Evaluate a deterministic Flow BC plus residual RLPD curriculum stage."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Combined Flow Matching plus residual RLPD checkpoint.",
)
parser.add_argument(
    "--flow-checkpoint",
    type=Path,
    help=(
        "Standalone Flow Matching checkpoint. This is valid only for zero or "
        "constant residual ablations."
    ),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--policy-hz", type=float, default=30.0)
parser.add_argument("--sim-control-hz", type=float, default=50.0)
parser.add_argument("--max-sim-steps", type=int, default=0)
parser.add_argument("--hard-reset-finger-force-n", type=float, default=15.1)
parser.add_argument("--reset-settle-steps", type=int, default=4)
parser.add_argument(
    "--residual-mode",
    choices=("policy", "zero", "constant", "policy_plus_constant"),
    default="policy",
    help="Use the learned residual, zero ablation, or a fixed deployable residual.",
)
parser.add_argument("--constant-residual", default="")
parser.add_argument("--record-video", action="store_true")
parser.add_argument(
    "--stop-on-curriculum-stage-success",
    action="store_true",
    help=(
        "Stop after the active curriculum gate is reached. This does not mark "
        "the episode as a full flip-table task success."
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--episode-seeds",
    default="",
    help=(
        "Optional comma-separated seeds. The count must match --episodes; "
        "otherwise seeds are generated as --seed + episode index."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
explicit_cli = {
    name: getattr(args_cli, name)
    for name in (
        "checkpoint",
        "flow_checkpoint",
        "output",
        "episodes",
        "num_envs",
        "policy_hz",
        "sim_control_hz",
        "max_sim_steps",
        "hard_reset_finger_force_n",
        "reset_settle_steps",
        "residual_mode",
        "constant_residual",
        "record_video",
        "stop_on_curriculum_stage_success",
        "seed",
        "episode_seeds",
    )
}
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
for name, value in explicit_cli.items():
    setattr(args_cli, name, value)
args_cli.enable_cameras = True


def _parse_constant_residual(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 19:
        raise ValueError("--constant-residual must contain exactly 19 comma-separated values")
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
        raise ValueError("--constant-residual values must be finite and in [-1, 1]")
    return values


def _parse_episode_seeds(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    try:
        seeds = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError("--episode-seeds must contain comma-separated integers") from exc
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("--episode-seeds values must be in [0, 2**32)")
    return seeds


def validate_cli() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("deterministic stage evaluation requires --num-envs 1")
    if args_cli.episodes < 1:
        raise ValueError("--episodes must be positive")
    episode_seeds = _parse_episode_seeds(args_cli.episode_seeds)
    if episode_seeds and len(episode_seeds) != args_cli.episodes:
        raise ValueError("--episode-seeds count must match --episodes")
    if args_cli.policy_hz <= 0 or args_cli.sim_control_hz <= 0:
        raise ValueError("policy and simulator rates must be positive")
    if args_cli.policy_hz > args_cli.sim_control_hz:
        raise ValueError("policy rate cannot exceed simulator control rate")
    if args_cli.max_sim_steps < 0:
        raise ValueError("--max-sim-steps cannot be negative")
    if not 0 <= args_cli.reset_settle_steps <= 10:
        raise ValueError("--reset-settle-steps must be in [0, 10]")
    if args_cli.residual_mode in {"constant", "policy_plus_constant"}:
        _parse_constant_residual(args_cli.constant_residual)
    elif args_cli.constant_residual.strip():
        raise ValueError("--constant-residual requires --residual-mode constant")
    if (
        not math.isfinite(args_cli.hard_reset_finger_force_n)
        or args_cli.hard_reset_finger_force_n <= 15.0
    ):
        raise ValueError("hard reset force must exceed the 15 N success limit")
    if (args_cli.checkpoint is None) == (args_cli.flow_checkpoint is None):
        raise ValueError("provide exactly one of --checkpoint or --flow-checkpoint")
    if (
        args_cli.flow_checkpoint is not None
        and args_cli.residual_mode not in {"zero", "constant"}
    ):
        raise ValueError(
            "--flow-checkpoint supports only zero or constant residual modes"
        )
    if args_cli.checkpoint is not None:
        required = (
            args_cli.checkpoint / "combined_policy.json",
            args_cli.checkpoint / "flow_matching" / "flow_matching_policy.json",
            args_cli.checkpoint / "rlpd" / "rlpd_policy.json",
        )
    else:
        required = (
            args_cli.flow_checkpoint / "flow_matching_policy.json",
            args_cli.flow_checkpoint / "model.safetensors",
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)


validate_cli()
episode_seeds = _parse_episode_seeds(args_cli.episode_seeds) or [
    args_cli.seed + episode for episode in range(args_cli.episodes)
]
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from model.flip_table_reinforcement_learning.rlpd import PolicyControlClock, RLPDAgent
from model.flip_table_reinforcement_learning.rlpd.policy_contract import (
    AbsoluteTargetDelayBuffer,
    BODY_RESIDUAL_SCALE,
    HAND_RESIDUAL_COMMAND_SCALE,
    UpperBodyTargetSafetyFilter,
    apply_residual_to_base,
    residual_observation,
    stochastic_action_mask_for_stage,
    target_safety_from_environment,
)
from model.flip_table_reinforcement_learning.rlpd.sim_runtime import (
    CAMERAS,
    POLICY_INPUTS,
    FlowTargetScheduler,
    camera_batch,
    dataset_joint_state,
    flow_control_ready,
    last_commanded_target,
    settle_after_reset,
    set_flow_control_ready,
)
from model.subtask_policy_training.flow_matching import FlowMatchingPolicy
from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch
from robofinals.utils.place_utils.env_utils import set_seed
from robofinals_rl.flip_table import mdp
from robofinals_rl.flip_table.common import runtime_policy_start_step


VIDEO_CAMERAS = (
    ("head_left", "first_person_camera"),
    ("left_wrist", "left_hand_camera"),
    ("right_wrist", "right_hand_camera"),
    ("global", "global_camera"),
)
EARLY_STAGE_TABLE_LIMITS = {
    "reach": (0.01 / 0.18, 0.03),
    "contact": (0.02 / 0.18, 0.08),
    "grasp": (0.04 / 0.18, 0.25),
}


def _json_value(value: torch.Tensor) -> list[float] | float:
    array = value.detach().cpu().float().numpy()
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()


def _camera_frame(env: Any, sensor_name: str) -> np.ndarray:
    value = as_torch(env.scene.sensors[sensor_name].data.output["rgb"])
    if value.ndim != 4 or value.shape[0] != 1 or tuple(value.shape[1:3]) != (480, 640):
        raise ValueError(
            f"{sensor_name} must provide [1,480,640,C], got {tuple(value.shape)}"
        )
    if value.shape[-1] < 3 or not torch.isfinite(value).all():
        raise ValueError(f"{sensor_name} must provide finite RGB channels")
    frame = value[0, ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        scale = 255.0 if float(frame.max(initial=0.0)) <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
    return frame


def _open_video_writers(directory: Path, fps: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    writers = {
        label: imageio.get_writer(
            directory / f"{label}.mp4",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        for label, _sensor in VIDEO_CAMERAS
    }
    writers["composite"] = imageio.get_writer(
        directory / "composite.mp4",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    return writers


def _append_video_frames(env: Any, writers: dict[str, Any]) -> None:
    frames = {
        label: _camera_frame(env, sensor_name)
        for label, sensor_name in VIDEO_CAMERAS
    }
    for label, frame in frames.items():
        writers[label].append_data(frame)
    top = np.concatenate((frames["head_left"], frames["global"]), axis=1)
    bottom = np.concatenate((frames["left_wrist"], frames["right_wrist"]), axis=1)
    writers["composite"].append_data(np.concatenate((top, bottom), axis=0))


def _close_video_writers(writers: dict[str, Any]) -> None:
    for writer in writers.values():
        writer.close()


def _validate_checkpoint_contract(
    checkpoint: Path,
    target_safety: UpperBodyTargetSafetyFilter,
) -> dict[str, Any]:
    manifest = json.loads((checkpoint / "combined_policy.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "team_ramen_flow_residual_rlpd_v1":
        raise ValueError("unsupported combined Flow Residual RLPD checkpoint")
    if manifest.get("privileged_inputs") != []:
        raise ValueError("combined checkpoint declares privileged policy inputs")
    residual_contract = manifest.get("residual_contract", {})
    if residual_contract.get("body_scale_rad") != list(BODY_RESIDUAL_SCALE):
        raise ValueError("combined checkpoint body residual scale does not match runtime")
    if not math.isclose(
        float(residual_contract.get("hand_normalized_command_scale", float("nan"))),
        HAND_RESIDUAL_COMMAND_SCALE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("combined checkpoint hand residual scale does not match runtime")
    expected_safety = {
        "policy_hz": target_safety.policy_hz,
        "body_velocity_limit_rad_s": target_safety.body_velocity_limit_rad_s,
        "body_acceleration_limit_rad_s2": target_safety.body_acceleration_limit_rad_s2,
        "hand_velocity_limit_command_s": target_safety.hand_velocity_limit_command_s,
        "hand_acceleration_limit_command_s2": (
            target_safety.hand_acceleration_limit_command_s2
        ),
    }
    checkpoint_safety = manifest.get("target_safety", {})
    for key, expected in expected_safety.items():
        if not math.isclose(
            float(checkpoint_safety.get(key, float("nan"))),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"combined checkpoint target safety mismatch for {key}")
    return manifest


def _build_environment() -> tuple[Any, Any]:
    env_cfg = parse_env_cfg(
        scene_backend=args_cli.scene_backend,
        task_backend=args_cli.task_backend,
        task_name=args_cli.task,
        robot_name=args_cli.robot,
        scene_name=args_cli.layout,
        rl_name="FlipTableResidualStateRL",
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
    env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.seed = args_cli.seed
    if args_cli.max_sim_steps:
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s),
            args_cli.max_sim_steps / args_cli.sim_control_hz,
        )
    task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
    if task_name not in gym.registry:
        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )
    gym_env = gym.make(task_name, cfg=env_cfg)
    env = gym_env.unwrapped
    set_seed(args_cli.seed, env)
    required_cameras = list(CAMERAS)
    if args_cli.record_video:
        required_cameras.append("global_camera")
    missing = [name for name in required_cameras if name not in env.scene.sensors]
    if missing:
        raise RuntimeError(f"required evaluation cameras are missing: {missing}")
    expected_step_dt = 1.0 / args_cli.sim_control_hz
    if not math.isclose(float(env.step_dt), expected_step_dt, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            f"simulator step_dt={float(env.step_dt):.9f}s does not match "
            f"--sim-control-hz={args_cli.sim_control_hz:g}"
        )
    return gym_env, env


def main() -> None:
    output = args_cli.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    target_safety = target_safety_from_environment(policy_hz=args_cli.policy_hz)
    combined_manifest: dict[str, Any] | None = None
    agent: RLPDAgent | None = None
    if args_cli.checkpoint is not None:
        checkpoint = args_cli.checkpoint.resolve()
        combined_manifest = _validate_checkpoint_contract(checkpoint, target_safety)
        flow_checkpoint = checkpoint / "flow_matching"
        checkpoint_kind = "flow_matching_plus_residual_rlpd"
        if args_cli.residual_mode in {"policy", "policy_plus_constant"}:
            agent = RLPDAgent.from_pretrained(
                checkpoint / "rlpd",
                device=args_cli.device,
            )
    else:
        checkpoint = args_cli.flow_checkpoint.resolve()
        flow_checkpoint = checkpoint
        checkpoint_kind = "standalone_flow_matching"
    flow = FlowMatchingPolicy.from_pretrained(flow_checkpoint, device=args_cli.device)
    flow.requires_grad_(False)
    constant_residual = (
        torch.tensor(
            _parse_constant_residual(args_cli.constant_residual),
            dtype=torch.float32,
            device=args_cli.device,
        ).reshape(1, 19)
        if args_cli.residual_mode in {"constant", "policy_plus_constant"}
        else None
    )
    expected_observation_dim = flow.config.model_dim + 3 * 19
    if agent is not None and agent.config.observation_dim != expected_observation_dim:
        raise ValueError(
            "RLPD observation dimension does not match Flow checkpoint: "
            f"{agent.config.observation_dim} != {expected_observation_dim}"
        )

    output.mkdir(parents=True, exist_ok=False)
    gym_env, env = _build_environment()
    randomization_level = max(
        0.0,
        min(1.0, float(os.environ.get("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", "1.0"))),
    )
    action_delay_max_steps = max(
        0,
        int(
            os.environ.get(
                "FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS",
                "2" if randomization_level >= 0.8 else "1",
            )
        ),
    )
    target_delay = AbsoluteTargetDelayBuffer(
        num_envs=1,
        max_delay_steps=action_delay_max_steps,
        device=env.device,
    )
    stage = os.environ.get("FLIP_TABLE_RL_STAGE", "reach").strip().lower()
    flow_motion_gain = float(os.environ.get("FLIP_TABLE_RLPD_FLOW_MOTION_GAIN", "1.0"))
    deployable_prefix_controller_steps = runtime_policy_start_step(env)
    max_sim_steps = args_cli.max_sim_steps or max(
        1,
        round(float(env.cfg.episode_length_s) / float(env.step_dt)),
    )
    report = {
        "schema_version": "team_ramen_flow_residual_rlpd_evaluation_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_transitions": (
            combined_manifest.get("transitions")
            if combined_manifest is not None
            else None
        ),
        "residual_mode": args_cli.residual_mode,
        "constant_residual": (
            constant_residual.detach().cpu().reshape(-1).tolist()
            if constant_residual is not None
            else None
        ),
        "stage": stage,
        "evaluation_mode": os.environ.get("FLIP_TABLE_RL_EVAL_MODE", "randomized"),
        "randomization_level": randomization_level,
        "seed": args_cli.seed,
        "episode_seed_strategy": (
            "explicit_episode_seed_list"
            if args_cli.episode_seeds.strip()
            else "base_seed_plus_episode_index"
        ),
        "episode_seeds": episode_seeds,
        "episodes": args_cli.episodes,
        "policy_hz": args_cli.policy_hz,
        "sim_control_hz": args_cli.sim_control_hz,
        "action_delay_max_sim_steps": action_delay_max_steps,
        "flow_motion_gain": flow_motion_gain,
        "deployable_prefix_controller_steps": deployable_prefix_controller_steps,
        "reset_settle_steps": args_cli.reset_settle_steps,
        "policy_inputs": POLICY_INPUTS,
        "policy_output": "19D upper-body absolute joint target",
        "actor_critic_privileged_inputs": [],
        "privileged_use": "success, safety and trace diagnostics only",
        "lower_body_policy_control": False,
        "root_policy_control": False,
        "record_video": args_cli.record_video,
        "success_definition": "stable completed table flip only",
        "curriculum_stage_success_is_task_success": False,
        "stop_on_curriculum_stage_success": args_cli.stop_on_curriculum_stage_success,
        "episode_results": [],
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    try:
        for episode, episode_seed in enumerate(episode_seeds):
            episode_dir = output / f"episode_{episode:03d}"
            episode_dir.mkdir(parents=True)
            trace_path = episode_dir / "trace.jsonl"
            writers = (
                _open_video_writers(episode_dir / "videos", round(args_cli.sim_control_hz))
                if args_cli.record_video
                else {}
            )
            scheduler = FlowTargetScheduler(
                flow,
                motion_gain=flow_motion_gain,
                motion_mask=stochastic_action_mask_for_stage(stage),
            )
            clock = PolicyControlClock(args_cli.policy_hz, args_cli.sim_control_hz)
            target_safety.reset()
            previous_residual = torch.zeros((1, 19), device=env.device)
            episode_safe = True
            ever_curriculum_stage_success = False
            ever_task_success = False
            max_finger_force_n = 0.0
            max_white_leg_attributed_force_n = 0.0
            max_lift_progress = 0.0
            max_flip_progress = 0.0
            max_single_hand_grasp_quality = [0.0, 0.0]
            max_bimanual_grasp_quality = 0.0
            min_hand_leg_distance_m = [float("inf"), float("inf")]
            residual_abs_sum = 0.0
            residual_value_count = 0
            residual_abs_max = 0.0
            safety_clip_count = 0
            terminal_reason = "max_sim_steps"
            step = -1
            try:
                set_seed(episode_seed, env)
                gym_env.reset()
                settle_after_reset(
                    gym_env,
                    env,
                    steps=args_cli.reset_settle_steps,
                )
                images = camera_batch(env)
                state = dataset_joint_state(env)
                base_target = scheduler.current(images, state)
                features = residual_observation(
                    flow,
                    images,
                    state,
                    base_target,
                    previous_residual,
                ).float()
                target_delay.reset(state)
                if writers:
                    _append_video_frames(env, writers)
                residual: torch.Tensor | None = None
                desired_target: torch.Tensor | None = None
                safe_target: torch.Tensor | None = None
                with trace_path.open("w", encoding="utf-8") as trace_file:
                    for step in range(max_sim_steps):
                        if residual is None:
                            if not bool(flow_control_ready(env)[0]):
                                residual = torch.zeros_like(previous_residual)
                            elif args_cli.residual_mode == "zero":
                                residual = torch.zeros_like(previous_residual)
                            elif args_cli.residual_mode == "constant":
                                assert constant_residual is not None
                                residual = constant_residual.expand_as(previous_residual)
                            else:
                                assert agent is not None
                                with torch.inference_mode():
                                    residual = agent.act(features, deterministic=True).float()
                                if args_cli.residual_mode == "policy_plus_constant":
                                    assert constant_residual is not None
                                    residual = (residual + constant_residual).clamp(-1.0, 1.0)
                            desired_target = apply_residual_to_base(base_target, residual)
                            safe_target, clipped = target_safety.filter(desired_target, state)
                            safety_clip_count += clipped
                            residual_abs_sum += float(residual.abs().sum())
                            residual_value_count += residual.numel()
                            residual_abs_max = max(residual_abs_max, float(residual.abs().max()))
                        assert desired_target is not None and safe_target is not None
                        delayed_target = target_delay.apply(safe_target)
                        env._flip_table_rlpd_absolute_target = delayed_target
                        controller_step_before = int(env.episode_length_buf[0])
                        flow_was_ready = bool(flow_control_ready(env)[0])
                        _obs, reward, terminated, truncated, _extras = gym_env.step(residual)
                        policy_interval_complete = clock.advance_sim_step()
                        applied_commanded_target = last_commanded_target(env)

                        with torch.no_grad():
                            forces = mdp.finger_contact_forces(env)
                            white_leg_forces = mdp.white_table_leg_contact_forces(env)
                            force_max = float(forces.max())
                            white_leg_force_max = float(white_leg_forces.max())
                            distances = mdp.nearest_leg_distances(env)
                            grasp_quality = torch.stack(
                                (
                                    mdp.single_hand_grasp(env, side="left"),
                                    mdp.single_hand_grasp(env, side="right"),
                                ),
                                dim=1,
                            )
                            bimanual_grasp_quality = float(mdp.bimanual_grasp(env)[0])
                            lift_progress = float(mdp.table_lift_progress(env)[0])
                            flip_progress = float(mdp.table_flip_progress(env)[0])
                            curriculum_stage_success = bool(
                                mdp.table_stage_success(env)[0]
                            )
                            task_success = bool(mdp.table_stable_success(env)[0])
                            max_finger_force_n = max(max_finger_force_n, force_max)
                            max_white_leg_attributed_force_n = max(
                                max_white_leg_attributed_force_n,
                                white_leg_force_max,
                            )
                            max_lift_progress = max(max_lift_progress, lift_progress)
                            max_flip_progress = max(max_flip_progress, flip_progress)
                            max_bimanual_grasp_quality = max(
                                max_bimanual_grasp_quality,
                                bimanual_grasp_quality,
                            )
                            for side in range(2):
                                max_single_hand_grasp_quality[side] = max(
                                    max_single_hand_grasp_quality[side],
                                    float(grasp_quality[0, side]),
                                )
                                min_hand_leg_distance_m[side] = min(
                                    min_hand_leg_distance_m[side],
                                    float(distances[0, side]),
                                )
                            episode_safe = episode_safe and force_max <= mdp.MAX_SAFE_FINGER_FORCE_N
                            if stage in EARLY_STAGE_TABLE_LIMITS:
                                lift_limit, flip_limit = EARLY_STAGE_TABLE_LIMITS[stage]
                                episode_safe = episode_safe and (
                                    lift_progress <= lift_limit and flip_progress <= flip_limit
                                )
                            ever_curriculum_stage_success = (
                                ever_curriculum_stage_success
                                or curriculum_stage_success
                            )
                            ever_task_success = ever_task_success or task_success
                        measured_after = dataset_joint_state(env)
                        trace_row = {
                            "sim_step": step,
                            "time_s": (step + 1) / args_cli.sim_control_hz,
                            "policy_interval_complete": policy_interval_complete,
                            "deployable": {
                                "teacher_prefix_active": bool(
                                    not flow_was_ready
                                ),
                                "controller_step_before": controller_step_before,
                                "state_before": _json_value(state),
                                "flow_base_target": _json_value(base_target),
                                "residual": _json_value(residual),
                                "desired_target": _json_value(desired_target),
                                "safety_filtered_target": _json_value(safe_target),
                                "delayed_target": _json_value(delayed_target),
                                "applied_commanded_target": _json_value(
                                    applied_commanded_target
                                ),
                                "state_after": _json_value(measured_after),
                            },
                            "reward": float(reward.reshape(-1)[0]),
                            "privileged_diagnostics": {
                                "all_surface_finger_force_n": _json_value(forces),
                                "white_leg_attributed_finger_force_by_leg_n": _json_value(
                                    white_leg_forces
                                ),
                                "hand_leg_distance_m": _json_value(distances),
                                "single_hand_grasp_quality": _json_value(grasp_quality),
                                "bimanual_grasp_quality": bimanual_grasp_quality,
                                "table_lift_progress": lift_progress,
                                "table_flip_progress": flip_progress,
                                # Retained for trace readers written before schema v2.
                                "stage_success": curriculum_stage_success,
                                "curriculum_stage_success": curriculum_stage_success,
                                "task_success": task_success,
                            },
                            "terminated": bool(terminated.reshape(-1)[0]),
                            "truncated": bool(truncated.reshape(-1)[0]),
                        }
                        trace_file.write(json.dumps(trace_row, allow_nan=False) + "\n")
                        if writers:
                            _append_video_frames(env, writers)

                        if task_success:
                            terminal_reason = "task_success"
                            break
                        if (
                            curriculum_stage_success
                            and args_cli.stop_on_curriculum_stage_success
                        ):
                            terminal_reason = "curriculum_stage_success"
                            break
                        if force_max > args_cli.hard_reset_finger_force_n:
                            terminal_reason = "hard_force_limit"
                            break
                        if bool(terminated.reshape(-1)[0]):
                            terminal_reason = "environment_terminated"
                            break
                        if bool(truncated.reshape(-1)[0]):
                            terminal_reason = "environment_truncated"
                            break
                        prefix_finished = (
                            not flow_was_ready
                            and int(env.episode_length_buf[0])
                            >= deployable_prefix_controller_steps
                        )
                        if prefix_finished:
                            scheduler.reset()
                            previous_residual.zero_()
                            clock.reset()
                            images = camera_batch(env)
                            state = measured_after
                            handoff_target = last_commanded_target(env)
                            base_target = scheduler.anchor_to_target(
                                images,
                                state,
                                handoff_target,
                            )
                            features = residual_observation(
                                flow,
                                images,
                                state,
                                base_target,
                                previous_residual,
                            ).float()
                            target_safety.reset(handoff_target)
                            target_delay.reset(handoff_target)
                            set_flow_control_ready(env, True)
                            residual = None
                            desired_target = None
                            safe_target = None
                            continue
                        if not policy_interval_complete:
                            continue
                        scheduler.advance()
                        previous_residual = residual.detach()
                        images = camera_batch(env)
                        state = measured_after
                        base_target = scheduler.current(images, state)
                        features = residual_observation(
                            flow,
                            images,
                            state,
                            base_target,
                            previous_residual,
                        ).float()
                        residual = None
                        desired_target = None
                        safe_target = None
            finally:
                if writers:
                    _close_video_writers(writers)

            curriculum_stage_success = bool(
                ever_curriculum_stage_success and episode_safe
            )
            task_success = bool(ever_task_success and episode_safe)
            episode_result = {
                "episode": episode,
                "episode_seed": episode_seed,
                # `success` is deliberately an alias of the complete task result.
                "success": task_success,
                "task_success": task_success,
                "curriculum_stage_success": curriculum_stage_success,
                "terminal_reason": terminal_reason,
                "sim_steps": step + 1,
                "duration_s": (step + 1) / args_cli.sim_control_hz,
                "episode_safe": episode_safe,
                "max_finger_force_n": max_finger_force_n,
                "max_white_leg_attributed_force_n": max_white_leg_attributed_force_n,
                "max_table_lift_progress": max_lift_progress,
                "max_table_flip_progress": max_flip_progress,
                "max_single_hand_grasp_quality": max_single_hand_grasp_quality,
                "max_bimanual_grasp_quality": max_bimanual_grasp_quality,
                "min_hand_leg_distance_m": min_hand_leg_distance_m,
                "residual_abs_mean": residual_abs_sum / max(residual_value_count, 1),
                "residual_abs_max": residual_abs_max,
                "safety_clip_count": safety_clip_count,
                "trace": str(trace_path),
                "videos": str(episode_dir / "videos") if writers else None,
            }
            (episode_dir / "summary.json").write_text(
                json.dumps(episode_result, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            report["episode_results"].append(episode_result)
            print(json.dumps(episode_result, allow_nan=False), flush=True)
    finally:
        gym_env.close()

    task_successes = sum(
        bool(item["task_success"]) for item in report["episode_results"]
    )
    curriculum_stage_successes = sum(
        bool(item["curriculum_stage_success"])
        for item in report["episode_results"]
    )
    report["successes"] = task_successes
    report["success_rate"] = task_successes / args_cli.episodes
    report["task_successes"] = task_successes
    report["task_success_rate"] = task_successes / args_cli.episodes
    report["curriculum_stage_successes"] = curriculum_stage_successes
    report["curriculum_stage_success_rate"] = (
        curriculum_stage_successes / args_cli.episodes
    )
    report["all_episodes_safe"] = all(
        bool(item["episode_safe"]) for item in report["episode_results"]
    )
    report["max_finger_force_n"] = max(
        float(item["max_finger_force_n"]) for item in report["episode_results"]
    )
    (output / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "successes": task_successes,
                "episodes": args_cli.episodes,
                "success_rate": report["success_rate"],
                "curriculum_stage_successes": curriculum_stage_successes,
                "curriculum_stage_success_rate": report[
                    "curriculum_stage_success_rate"
                ],
                "all_episodes_safe": report["all_episodes_safe"],
                "max_finger_force_n": report["max_finger_force_n"],
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
