#!/usr/bin/env python3
"""Train residual RLPD over a frozen flow-matching flip-table policy in Isaac Lab."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument("--flow-checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--total-transitions", type=int, default=1_000_000)
parser.add_argument("--prior-transitions", type=int, default=50_000)
parser.add_argument("--learning-starts", type=int, default=2_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--update-to-data-ratio", type=float, default=1.0)
parser.add_argument("--online-replay-capacity", type=int, default=1_000_000)
parser.add_argument("--prior-replay-capacity", type=int, default=100_000)
parser.add_argument("--random-residual-std", type=float, default=0.15)
parser.add_argument("--critic-warmup-updates", type=int, default=1000)
parser.add_argument("--prior-bc-weight", type=float, default=10.0)
parser.add_argument("--reference-bc-weight", type=float, default=20.0)
parser.add_argument("--actor-learning-rate", type=float, default=1.0e-4)
parser.add_argument("--actor-q-normalization", type=float, default=1.0)
parser.add_argument("--initial-temperature", type=float, default=1.0e-3)
parser.add_argument(
    "--prior-residual",
    default="",
    help="Successful 16D arm/hand residual used for prior replay and actor initialization.",
)
parser.add_argument("--policy-hz", type=float, default=30.0)
parser.add_argument("--sim-control-hz", type=float, default=50.0)
parser.add_argument("--reset-settle-steps", type=int, default=4)
parser.add_argument(
    "--reuse-prefix-state",
    action="store_true",
    help=(
        "Reuse a captured post-prefix simulator state for fixed-scene curriculum "
        "resets. This never changes policy inputs and is forbidden for randomized runs."
    ),
)
parser.add_argument("--resume", type=Path)
parser.add_argument("--actor-init-checkpoint", type=Path)
parser.add_argument(
    "--prior-action-source",
    choices=("constant", "actor"),
    default="constant",
)
parser.add_argument("--log-every-transitions", type=int, default=10_000)
parser.add_argument("--save-every-transitions", type=int, default=250_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-project", default="iros2026-ramen-flip-table")
parser.add_argument("--wandb-run-name", default="flow-residual-rlpd")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
args_cli.enable_cameras = True


def parse_prior_residual(raw: str) -> list[float]:
    if not raw.strip():
        return [0.0] * 16
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 16:
        raise ValueError("--prior-residual must contain exactly 16 comma-separated values")
    if not all(math.isfinite(value) and -1.0 < value < 1.0 for value in values):
        raise ValueError("--prior-residual values must be finite and strictly inside (-1, 1)")
    return values


def validate_cli() -> None:
    positive = (
        "num_envs",
        "total_transitions",
        "prior_transitions",
        "learning_starts",
        "batch_size",
        "online_replay_capacity",
        "prior_replay_capacity",
        "log_every_transitions",
        "save_every_transitions",
    )
    for name in positive:
        if getattr(args_cli, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args_cli.prior_transitions >= args_cli.total_transitions:
        raise ValueError("prior transitions must be fewer than total transitions")
    if not 0.001 <= args_cli.random_residual_std <= 0.25:
        raise ValueError("random residual standard deviation must be in [0.001, 0.25]")
    if args_cli.critic_warmup_updates < 0:
        raise ValueError("critic warm-up updates cannot be negative")
    if not math.isfinite(args_cli.prior_bc_weight) or args_cli.prior_bc_weight < 0:
        raise ValueError("prior BC weight must be finite and non-negative")
    if not math.isfinite(args_cli.reference_bc_weight) or args_cli.reference_bc_weight < 0:
        raise ValueError("reference BC weight must be finite and non-negative")
    for name in (
        "actor_learning_rate",
        "actor_q_normalization",
        "initial_temperature",
    ):
        if not math.isfinite(getattr(args_cli, name)) or getattr(args_cli, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    parse_prior_residual(args_cli.prior_residual)
    if args_cli.update_to_data_ratio <= 0:
        raise ValueError("update-to-data ratio must be positive")
    if args_cli.policy_hz <= 0 or args_cli.sim_control_hz <= 0:
        raise ValueError("policy and simulator control rates must be positive")
    if args_cli.policy_hz > args_cli.sim_control_hz:
        raise ValueError("policy rate cannot exceed simulator control rate")
    if not 0 <= args_cli.reset_settle_steps <= 10:
        raise ValueError("reset settle steps must be in [0, 10]")
    if not args_cli.flow_checkpoint.is_dir():
        raise FileNotFoundError(args_cli.flow_checkpoint)
    if args_cli.resume is not None and not args_cli.resume.is_dir():
        raise FileNotFoundError(args_cli.resume)
    if (
        args_cli.actor_init_checkpoint is not None
        and not args_cli.actor_init_checkpoint.is_dir()
    ):
        raise FileNotFoundError(args_cli.actor_init_checkpoint)
    if (
        args_cli.actor_init_checkpoint is not None
        and args_cli.prior_action_source != "actor"
    ):
        raise ValueError("actor initialization requires --prior-action-source actor")
    if args_cli.prior_action_source == "actor":
        if args_cli.actor_init_checkpoint is None and args_cli.resume is None:
            raise ValueError("actor prior requires --actor-init-checkpoint or --resume")
        if args_cli.prior_residual.strip():
            raise ValueError("--prior-residual is only valid with a constant prior")


validate_cli()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app


import gymnasium as gym

from model.flip_table_reinforcement_learning.rlpd import (
    PolicyControlClock,
    RLPDAgent,
    RLPDConfig,
    ReplayBuffer,
)
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
from model.flip_table_reinforcement_learning.rlpd.replay import balanced_replay_sample
from model.flip_table_reinforcement_learning.rlpd.sim_runtime import (
    CAMERAS,
    POLICY_INPUTS,
    FlowTargetScheduler,
    camera_batch,
    capture_relative_scene_state,
    dataset_joint_state,
    last_commanded_target,
    restore_relative_scene_state,
    settle_after_reset,
    set_flow_control_ready,
)
from model.subtask_policy_training.flow_matching import FlowMatchingPolicy
from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.place_utils.env_utils import set_seed
from robofinals_rl.flip_table import mdp
from robofinals_rl.flip_table.common import runtime_policy_start_step


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def execute_deployable_teacher_prefix(
    gym_env,
    env,
    *,
    hard_force_limit_n: float,
) -> dict[str, float | int]:
    """Run a clocked real-compatible target prefix before Flow/RLPD control."""

    policy_start_step = runtime_policy_start_step(env)
    if policy_start_step <= 0:
        set_flow_control_ready(env, True)
        return {
            "simulator_steps": 0,
            "max_finger_force_n": 0.0,
            "min_right_grasp_quality": 0.0,
            "min_bimanual_grasp_quality": 0.0,
        }
    set_flow_control_ready(env, False)
    simulator_steps = 0
    max_force_n = 0.0
    while bool((env.episode_length_buf < policy_start_step).any()):
        if not bool((env.episode_length_buf < policy_start_step).all()):
            raise RuntimeError("deployable teacher prefix lost vector-environment synchronization")
        measured_state = dataset_joint_state(env)
        # The action manager executes the clocked teacher while the readiness
        # gate is false. This hold target is only a valid finite fallback.
        env._flip_table_rlpd_absolute_target = measured_state
        zero_residual = torch.zeros_like(measured_state)
        _observation, _reward, terminated, truncated, _extras = gym_env.step(
            zero_residual
        )
        simulator_steps += 1
        if bool(torch.logical_or(terminated, truncated).any()):
            raise RuntimeError("environment ended during deployable teacher prefix")
        forces = mdp.finger_contact_forces(env)
        per_env_force_n = forces.amax(dim=(1, 2))
        force_n = float(per_env_force_n.max())
        max_force_n = max(max_force_n, force_n)
        if force_n > hard_force_limit_n:
            raise RuntimeError(
                "deployable teacher prefix exceeded hard finger-force limit: "
                f"{force_n:.3f} N > {hard_force_limit_n:.3f} N at simulator "
                f"step {simulator_steps}; per-environment maxima "
                f"{per_env_force_n.detach().cpu().tolist()}"
            )
    right_grasp = mdp.single_hand_grasp(env, side="right")
    bimanual_grasp = mdp.bimanual_grasp(env)
    return {
        "simulator_steps": simulator_steps,
        "max_finger_force_n": max_force_n,
        "min_right_grasp_quality": float(right_grasp.min()),
        "min_bimanual_grasp_quality": float(bimanual_grasp.min()),
    }


@torch.no_grad()
def restore_deployable_prefix_state(
    gym_env,
    env,
    state: dict[str, object],
    *,
    reset_settle_steps: int,
    hard_force_limit_n: float,
) -> dict[str, float | int]:
    """Restore a training-only curriculum state and validate physical contact."""

    policy_start_step = runtime_policy_start_step(env)
    if policy_start_step <= 0:
        raise RuntimeError("prefix-state restoration requires a positive policy start step")
    set_flow_control_ready(env, False)
    restore_relative_scene_state(env, state, episode_step=policy_start_step)
    max_force_n = 0.0

    def validate_settle_step(current_env, step: int) -> None:
        nonlocal max_force_n
        forces = mdp.finger_contact_forces(current_env)
        per_env_force_n = forces.amax(dim=(1, 2))
        force_n = float(per_env_force_n.max())
        max_force_n = max(max_force_n, force_n)
        if force_n > hard_force_limit_n:
            raise RuntimeError(
                "restored deployable prefix state exceeded hard finger-force limit: "
                f"{force_n:.3f} N > {hard_force_limit_n:.3f} N at settle "
                f"step {step + 1}; per-environment maxima "
                f"{per_env_force_n.detach().cpu().tolist()}"
            )

    settled = settle_after_reset(
        gym_env,
        env,
        steps=reset_settle_steps,
        post_step=validate_settle_step,
    )
    right_grasp = mdp.single_hand_grasp(env, side="right")
    bimanual_grasp = mdp.bimanual_grasp(env)
    min_bimanual_grasp = float(bimanual_grasp.min())
    if min_bimanual_grasp < mdp.GRASP_SUCCESS_THRESHOLD:
        raise RuntimeError(
            "restored deployable prefix state lost bimanual grasp: "
            f"{min_bimanual_grasp:.6f} < {mdp.GRASP_SUCCESS_THRESHOLD:.6f}"
        )
    return {
        "settle_steps": settled,
        "max_finger_force_n": max_force_n,
        "min_right_grasp_quality": float(right_grasp.min()),
        "min_bimanual_grasp_quality": min_bimanual_grasp,
    }


def save_combined_checkpoint(
    destination: Path,
    *,
    flow_checkpoint: Path,
    agent: RLPDAgent,
    target_safety: UpperBodyTargetSafetyFilter,
    transitions: int,
    prior_residual: torch.Tensor,
    prior_action_source: str,
    actor_init_checkpoint: Path | None,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    flow_destination = temporary / "flow_matching"
    flow_destination.mkdir()
    for name in ("flow_matching_policy.json", "model.safetensors"):
        source = flow_checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(f"Flow Matching checkpoint is missing {source}")
        shutil.copy2(source, flow_destination / name)
    agent.save_pretrained(temporary / "rlpd")
    (temporary / "combined_policy.json").write_text(
        json.dumps(
            {
                "schema_version": "team_ramen_flow_residual_rlpd_v1",
                "transitions": transitions,
                "policy_inputs": POLICY_INPUTS,
                "policy_output": "16D arm/hand absolute joint targets",
                "privileged_inputs": [],
                "sim_privileged_use": "reward, success, curriculum and diagnostics only",
                "training_prior_residual": (
                    prior_residual.detach().cpu().reshape(-1).tolist()
                ),
                "training_prior_action_source": prior_action_source,
                "actor_init_checkpoint": (
                    str(actor_init_checkpoint) if actor_init_checkpoint is not None else None
                ),
                "residual_contract": {
                    "body_scale_rad": list(BODY_RESIDUAL_SCALE),
                    "hand_normalized_command_scale": HAND_RESIDUAL_COMMAND_SCALE,
                },
                "target_safety": {
                    "policy_hz": target_safety.policy_hz,
                    "body_velocity_limit_rad_s": target_safety.body_velocity_limit_rad_s,
                    "body_acceleration_limit_rad_s2": (
                        target_safety.body_acceleration_limit_rad_s2
                    ),
                    "hand_velocity_limit_command_s": (
                        target_safety.hand_velocity_limit_command_s
                    ),
                    "hand_acceleration_limit_command_s2": (
                        target_safety.hand_acceleration_limit_command_s2
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def save_training_resume(
    destination: Path,
    *,
    agent: RLPDAgent,
    prior: ReplayBuffer,
    online: ReplayBuffer,
    rng: np.random.Generator,
    counters: dict[str, int | float],
    flow_checkpoint: Path,
    prior_residual: torch.Tensor,
    prior_action_source: str,
    actor_init_checkpoint: Path | None,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    agent.save_training_state(temporary / "agent_training_state.pt")
    prior.save(temporary / "prior_replay")
    online.save(temporary / "online_replay")
    (temporary / "trainer_state.json").write_text(
        json.dumps(
            {
                "schema_version": "team_ramen_flow_residual_rlpd_training_v1",
                "flow_checkpoint": str(flow_checkpoint),
                "prior_residual": prior_residual.detach().cpu().reshape(-1).tolist(),
                "prior_action_source": prior_action_source,
                "actor_init_checkpoint": (
                    str(actor_init_checkpoint) if actor_init_checkpoint is not None else None
                ),
                "counters": counters,
                "numpy_rng_state": rng.bit_generator.state,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def restore_training_resume(
    source: Path,
    *,
    agent: RLPDAgent,
    prior: ReplayBuffer,
    online: ReplayBuffer,
    rng: np.random.Generator,
    expected_prior_residual: torch.Tensor,
    expected_prior_action_source: str,
    expected_actor_init_checkpoint: Path | None,
) -> dict[str, int | float]:
    payload = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "team_ramen_flow_residual_rlpd_training_v1":
        raise ValueError("unsupported Flow Residual RLPD training checkpoint schema")
    restored_prior = torch.tensor(
        payload.get("prior_residual", [0.0] * 16),
        dtype=expected_prior_residual.dtype,
        device=expected_prior_residual.device,
    ).reshape(-1)
    if not torch.equal(restored_prior, expected_prior_residual.reshape(-1)):
        raise ValueError("resume checkpoint prior residual does not match --prior-residual")
    if payload.get("prior_action_source", "constant") != expected_prior_action_source:
        raise ValueError("resume checkpoint prior action source mismatch")
    expected_actor_init = (
        str(expected_actor_init_checkpoint)
        if expected_actor_init_checkpoint is not None
        else None
    )
    if payload.get("actor_init_checkpoint") != expected_actor_init:
        raise ValueError("resume checkpoint actor initialization mismatch")
    agent.load_training_state(source / "agent_training_state.pt")
    prior.restore(source / "prior_replay")
    online.restore(source / "online_replay")
    rng.bit_generator.state = payload["numpy_rng_state"]
    counters = payload.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("RLPD resume checkpoint is missing counters")
    return counters


def main() -> None:
    seed_everything(args_cli.seed)
    output = args_cli.output.resolve()
    if args_cli.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty RLPD output {output}; "
            "use --resume or a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    flow_checkpoint = args_cli.flow_checkpoint.resolve()

    env_cfg = parse_env_cfg(
        scene_backend=args_cli.scene_backend,
        task_backend=args_cli.task_backend,
        task_name=args_cli.task,
        robot_name=args_cli.robot,
        scene_name=args_cli.layout,
        rl_name="FlipTableResidualStateRL",
        robot_scale=args_cli.robot_scale,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        first_person_view=args_cli.first_person_view,
        enable_cameras=True,
        execute_mode=ExecuteMode.TRAIN,
        usd_simplify=args_cli.usd_simplify,
        seed=args_cli.seed,
        sources=args_cli.sources,
        object_projects=args_cli.object_projects,
        headless_mode=args_cli.headless,
        enable_full_local_scene=args_cli.enable_full_local_scene,
    )
    # Keep all vector environments synchronized. Privileged success remains a
    # reward and diagnostic, never an actor/critic observation.
    env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.seed = args_cli.seed
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
    missing = [name for name in CAMERAS if name not in env.scene.sensors]
    if missing:
        raise RuntimeError(f"required policy cameras are missing: {missing}")

    flow = FlowMatchingPolicy.from_pretrained(flow_checkpoint, device=env.device)
    flow.requires_grad_(False)
    flow_motion_gain = float(os.environ.get("FLIP_TABLE_RLPD_FLOW_MOTION_GAIN", "1.0"))
    observation_dim = flow.config.model_dim + 19 + 2 * 16
    stage = os.environ.get("FLIP_TABLE_RL_STAGE", "reach").strip().lower()
    stochastic_action_mask = stochastic_action_mask_for_stage(stage)
    scheduler = FlowTargetScheduler(
        flow,
        motion_gain=flow_motion_gain,
        motion_mask=stochastic_action_mask,
    )
    stochastic_action_mask_tensor = torch.tensor(
        stochastic_action_mask,
        dtype=torch.float32,
        device=env.device,
    ).reshape(1, 16)
    agent = RLPDAgent(
        RLPDConfig(
            observation_dim=observation_dim,
            min_residual_std=min(0.001, 0.5 * args_cli.random_residual_std),
            initial_residual_std=args_cli.random_residual_std,
            max_residual_std=args_cli.random_residual_std,
            prior_bc_weight=args_cli.prior_bc_weight,
            reference_bc_weight=args_cli.reference_bc_weight,
            actor_learning_rate=args_cli.actor_learning_rate,
            actor_q_normalization=args_cli.actor_q_normalization,
            initial_temperature=args_cli.initial_temperature,
            automatic_entropy_tuning=False,
            stochastic_action_mask=stochastic_action_mask,
        ),
        device=env.device,
    )
    actor_init_checkpoint = (
        args_cli.actor_init_checkpoint.resolve()
        if args_cli.actor_init_checkpoint is not None
        else None
    )
    if actor_init_checkpoint is not None and args_cli.resume is None:
        actor_directory = (
            actor_init_checkpoint / "rlpd"
            if (actor_init_checkpoint / "rlpd" / "rlpd_policy.json").is_file()
            else actor_init_checkpoint
        )
        source_agent = RLPDAgent.from_pretrained(actor_directory, device=env.device)
        compatible = (
            "observation_dim",
            "action_dim",
            "hidden_dim",
            "hidden_layers",
        )
        for name in compatible:
            if getattr(source_agent.config, name) != getattr(agent.config, name):
                raise ValueError(f"actor initialization checkpoint {name} mismatch")
        agent.actor.load_state_dict(source_agent.actor.state_dict(), strict=True)
        agent.set_reference_actor_from_current()
    prior_residual = torch.tensor(
        parse_prior_residual(args_cli.prior_residual),
        dtype=torch.float32,
        device=env.device,
    ).reshape(1, 16)
    prior_residual_reference = prior_residual.clone()
    if args_cli.resume is None and args_cli.prior_residual.strip():
        agent.initialize_actor_residual(prior_residual[0])
    prior = ReplayBuffer(args_cli.prior_replay_capacity, observation_dim)
    online = ReplayBuffer(args_cli.online_replay_capacity, observation_dim)
    rng = np.random.default_rng(args_cli.seed)
    previous_residual = torch.zeros((env.num_envs, 16), device=env.device)
    clock = PolicyControlClock(args_cli.policy_hz, args_cli.sim_control_hz)
    target_safety = target_safety_from_environment(policy_hz=args_cli.policy_hz)
    randomization_level = max(
        0.0, min(1.0, float(os.environ.get("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", "1.0")))
    )
    if args_cli.reuse_prefix_state:
        if env.num_envs != 1:
            raise ValueError("prefix-state reuse is restricted to one simulator environment")
        if not math.isclose(randomization_level, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("prefix-state reuse is forbidden when domain randomization is active")
        if runtime_policy_start_step(env) <= 0:
            raise ValueError("prefix-state reuse requires a deployable teacher prefix")
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
        num_envs=env.num_envs,
        max_delay_steps=action_delay_max_steps,
        device=env.device,
    )
    table_safety_limits = {
        "reach": (0.01 / 0.18, 0.03),
        "contact": (0.02 / 0.18, 0.08),
        "grasp": (0.04 / 0.18, 0.25),
    }
    hard_table_limits = {
        "reach": (0.05 / 0.18, 0.30),
        "contact": (0.08 / 0.18, 0.50),
        "grasp": (0.12 / 0.18, 0.75),
    }
    hard_reset_finger_force_n = float(
        os.environ.get("FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N", "15.1")
    )
    if not math.isfinite(hard_reset_finger_force_n) or (
        hard_reset_finger_force_n <= mdp.MAX_SAFE_FINGER_FORCE_N
    ):
        raise ValueError(
            "FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N must exceed the 15 N safety limit"
        )
    restored_counters: dict[str, int | float] = {}
    if args_cli.resume is not None:
        restored_counters = restore_training_resume(
            args_cli.resume.resolve(),
            agent=agent,
            prior=prior,
            online=online,
            rng=rng,
            expected_prior_residual=prior_residual,
            expected_prior_action_source=args_cli.prior_action_source,
            expected_actor_init_checkpoint=actor_init_checkpoint,
        )
    expected_step_dt = 1.0 / args_cli.sim_control_hz
    if not math.isclose(float(env.step_dt), expected_step_dt, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            f"simulator step_dt={float(env.step_dt):.9f}s does not match "
            f"--sim-control-hz={args_cli.sim_control_hz:g}"
        )

    wandb_run = None
    if args_cli.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            config=vars(args_cli),
        )

    manifest = {
        "schema_version": "team_ramen_flow_residual_rlpd_manifest_v1",
        "flow_checkpoint": str(flow_checkpoint),
        "resume_checkpoint": str(args_cli.resume.resolve()) if args_cli.resume else None,
        "actor_init_checkpoint": (
            str(actor_init_checkpoint) if actor_init_checkpoint is not None else None
        ),
        "prior_action_source": args_cli.prior_action_source,
        "stage": stage,
        "num_envs": env.num_envs,
        "env_spacing_m": float(env.cfg.scene.env_spacing),
        "observation_dim": observation_dim,
        "policy_inputs": POLICY_INPUTS,
        "policy_output": "16D arm/hand absolute joint target",
        "actor_critic_privileged_inputs": [],
        "sim_privileged_use": "reward, success, curriculum and diagnostics only",
        "lower_body_policy_control": False,
        "root_policy_control": False,
        "step_dt_s": float(env.step_dt),
        "policy_hz": args_cli.policy_hz,
        "sim_control_hz": args_cli.sim_control_hz,
        "replay_transition_unit": "one 30 Hz policy interval with accumulated simulator reward",
        "target_total_transitions": args_cli.total_transitions,
        "prior_transitions": args_cli.prior_transitions,
        "learning_starts": args_cli.learning_starts,
        "batch_size": args_cli.batch_size,
        "update_to_data_ratio": args_cli.update_to_data_ratio,
        "random_residual_std": args_cli.random_residual_std,
        "flow_motion_gain": flow_motion_gain,
        "stochastic_action_mask": list(stochastic_action_mask),
        "stochastic_action_indices": [
            index for index, enabled in enumerate(stochastic_action_mask) if enabled
        ],
        "critic_warmup_updates": args_cli.critic_warmup_updates,
        "prior_bc_weight": args_cli.prior_bc_weight,
        "reference_bc_weight": args_cli.reference_bc_weight,
        "actor_learning_rate": args_cli.actor_learning_rate,
        "actor_q_normalization": args_cli.actor_q_normalization,
        "initial_temperature": args_cli.initial_temperature,
        "automatic_entropy_tuning": False,
        "prior_residual": prior_residual.detach().cpu().reshape(-1).tolist(),
        "actor_initialized_from_prior": bool(
            args_cli.resume is None and args_cli.prior_residual.strip()
        ),
        "randomization_level": randomization_level,
        "action_delay_max_sim_steps": action_delay_max_steps,
        "deployable_teacher_prefix": {
            "enabled": runtime_policy_start_step(env) > 0,
            "runtime_controller_steps": runtime_policy_start_step(env),
            "source": os.environ.get("FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY"),
            "handoff": "Flow targets re-anchored to the last commanded 16D target",
            "privileged_inputs": [],
        },
        "training_prefix_state_reuse": {
            "enabled": args_cli.reuse_prefix_state,
            "fixed_scene_only": True,
            "policy_or_critic_input": False,
            "final_evaluation_must_execute_full_prefix": True,
        },
        "max_safe_finger_force_n": mdp.MAX_SAFE_FINGER_FORCE_N,
        "hard_reset_finger_force_n": hard_reset_finger_force_n,
        "reset_settle_steps": args_cli.reset_settle_steps,
        "hand_residual_command_scale": HAND_RESIDUAL_COMMAND_SCALE,
        "hand_velocity_limit_command_s": target_safety.hand_velocity_limit_command_s,
        "hand_acceleration_limit_command_s2": (
            target_safety.hand_acceleration_limit_command_s2
        ),
        "seed": args_cli.seed,
        "rlpd_config": agent.config.to_dict(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = output / "metrics.jsonl"
    append_jsonl(
        output / "training_sessions.jsonl",
        {
            "event": "session_start",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "resume_checkpoint": manifest["resume_checkpoint"],
            "stage": manifest["stage"],
            "target_total_transitions": args_cli.total_transitions,
        },
    )

    gym_env.reset()
    session_reset_settle_simulator_steps = settle_after_reset(
        gym_env,
        env,
        steps=args_cli.reset_settle_steps,
    )
    initial_prefix = execute_deployable_teacher_prefix(
        gym_env,
        env,
        hard_force_limit_n=hard_reset_finger_force_n,
    )
    reusable_prefix_state = (
        capture_relative_scene_state(env) if args_cli.reuse_prefix_state else None
    )
    images = camera_batch(env)
    state = dataset_joint_state(env)
    if runtime_policy_start_step(env) > 0:
        handoff_target = last_commanded_target(env)
        base_target = scheduler.anchor_to_target(images, state, handoff_target)
        target_safety.reset(handoff_target)
        target_delay.reset(handoff_target)
    else:
        base_target = scheduler.current(images, state)
        target_delay.reset(state)
    set_flow_control_ready(env, True)
    features = residual_observation(flow, images, state, base_target, previous_residual).float()
    total_transitions = int(restored_counters.get("total_transitions", 0))
    online_transitions = int(restored_counters.get("online_transitions", 0))
    next_log = (total_transitions // args_cli.log_every_transitions + 1) * args_cli.log_every_transitions
    next_save = (total_transitions // args_cli.save_every_transitions + 1) * args_cli.save_every_transitions
    max_lift = 0.0
    max_flip = 0.0
    max_finger_force_n = 0.0
    unsafe_force_env_steps = int(restored_counters.get("unsafe_force_env_steps", 0))
    successful_env_steps = 0
    completed_episodes = int(restored_counters.get("completed_episodes", 0))
    successful_episodes = int(restored_counters.get("successful_episodes", 0))
    session_completed_episodes = 0
    session_successful_episodes = 0
    last_log_completed_episodes = completed_episodes
    last_log_successful_episodes = successful_episodes
    episode_ever_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    episode_safe = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    episode_return = torch.zeros(env.num_envs, device=env.device)
    window_episode_return_sum = 0.0
    forced_safety_resets = int(restored_counters.get("forced_safety_resets", 0))
    stage_success_resets = int(restored_counters.get("stage_success_resets", 0))
    reused_prefix_state_resets = int(
        restored_counters.get("reused_prefix_state_resets", 0)
    )
    max_reused_prefix_state_force_n = float(
        restored_counters.get("max_reused_prefix_state_force_n", 0.0)
    )
    latest_metrics: dict[str, float] = {}
    accumulated_reward = torch.zeros((env.num_envs, 1), device=env.device)
    residual: torch.Tensor | None = None
    safe_target: torch.Tensor | None = None
    action_is_prior = True
    simulator_steps = int(restored_counters.get("simulator_steps", 0))
    reset_settle_simulator_steps = int(
        restored_counters.get("reset_settle_simulator_steps", 0)
    ) + session_reset_settle_simulator_steps
    deployable_prefix_simulator_steps = int(
        restored_counters.get("deployable_prefix_simulator_steps", 0)
    ) + int(initial_prefix["simulator_steps"])
    max_deployable_prefix_finger_force_n = max(
        float(restored_counters.get("max_deployable_prefix_finger_force_n", 0.0)),
        float(initial_prefix["max_finger_force_n"]),
    )
    min_deployable_prefix_right_grasp_quality = float(
        restored_counters.get(
            "min_deployable_prefix_right_grasp_quality",
            initial_prefix["min_right_grasp_quality"],
        )
    )
    if runtime_policy_start_step(env) > 0:
        min_deployable_prefix_right_grasp_quality = min(
            min_deployable_prefix_right_grasp_quality,
            float(initial_prefix["min_right_grasp_quality"]),
        )
    min_deployable_prefix_bimanual_grasp_quality = float(
        restored_counters.get(
            "min_deployable_prefix_bimanual_grasp_quality",
            initial_prefix["min_bimanual_grasp_quality"],
        )
    )
    if runtime_policy_start_step(env) > 0:
        min_deployable_prefix_bimanual_grasp_quality = min(
            min_deployable_prefix_bimanual_grasp_quality,
            float(initial_prefix["min_bimanual_grasp_quality"]),
        )
    safety_clip_count = int(restored_counters.get("safety_clip_count", 0))
    update_budget = float(restored_counters.get("update_budget", 0.0))
    gradient_updates = int(restored_counters.get("gradient_updates", agent.update_steps))
    actor_updates = int(restored_counters.get("actor_updates", 0))
    session_start_transitions = total_transitions
    started = time.monotonic()

    while total_transitions < args_cli.total_transitions:
        if not torch.equal(prior_residual, prior_residual_reference):
            raise RuntimeError("immutable prior residual was modified during training")
        if residual is None:
            action_is_prior = total_transitions < args_cli.prior_transitions
            if action_is_prior:
                if args_cli.prior_action_source == "actor":
                    residual = agent.act(features, deterministic=True).clone()
                else:
                    # ``previous_residual.zero_()`` runs on episode reset. Clone
                    # the view so reset can never mutate the prior itself.
                    residual = prior_residual.expand_as(previous_residual).clone()
            elif online_transitions < args_cli.learning_starts:
                if args_cli.prior_action_source == "actor":
                    exploration_center = agent.act(features, deterministic=True)
                else:
                    exploration_center = prior_residual
                residual = (
                    exploration_center
                    + torch.randn_like(previous_residual)
                    * args_cli.random_residual_std
                    * stochastic_action_mask_tensor
                ).clamp(-1.0, 1.0)
            else:
                residual = agent.act(features)
            desired_target = apply_residual_to_base(base_target, residual)
            safe_target, clipped = target_safety.filter(desired_target, state)
            safety_clip_count += clipped

        assert safe_target is not None
        env._flip_table_rlpd_absolute_target = target_delay.apply(safe_target)
        _observation, reward, terminated, truncated, _extras = gym_env.step(residual)
        simulator_steps += 1
        accumulated_reward += reward.reshape(-1, 1)
        episode_return += reward
        done_bool = torch.logical_or(terminated, truncated).reshape(-1, 1)
        if bool(done_bool.any().item()) and not bool(done_bool.all().item()):
            raise RuntimeError(
                "RLPD requires synchronized vector resets; received a partial done mask"
            )
        natural_episode_complete = bool(done_bool.all().item())
        policy_interval_complete = clock.advance_sim_step()

        with torch.no_grad():
            lift_progress = mdp.table_lift_progress(env)
            flip_progress = mdp.table_flip_progress(env)
            max_lift = max(max_lift, float(lift_progress.max()))
            max_flip = max(max_flip, float(flip_progress.max()))
            finger_forces = mdp.finger_contact_forces(env)
            per_env_force = finger_forces.amax(dim=(1, 2))
            max_finger_force_n = max(max_finger_force_n, float(per_env_force.max()))
            unsafe_force = per_env_force > mdp.MAX_SAFE_FINGER_FORCE_N
            unsafe_force_env_steps += int(unsafe_force.sum())
            episode_safe &= ~unsafe_force

            if stage in table_safety_limits:
                max_lift_progress, max_flip_progress = table_safety_limits[stage]
                episode_safe &= (lift_progress <= max_lift_progress) & (
                    flip_progress <= max_flip_progress
                )

            stage_success = mdp.table_stage_success(env)
            successful_env_steps += int(stage_success.sum())
            episode_ever_success |= stage_success
            synchronized_stage_success = bool(stage_success.any())

            hard_reset = bool((per_env_force > hard_reset_finger_force_n).any())
            if stage in hard_table_limits:
                hard_lift, hard_flip = hard_table_limits[stage]
                hard_reset = hard_reset or bool(
                    ((lift_progress > hard_lift) | (flip_progress > hard_flip)).any()
                )
        manual_reset = (hard_reset or synchronized_stage_success) and not natural_episode_complete
        if manual_reset:
            done_bool = torch.ones_like(done_bool, dtype=torch.bool)
            if hard_reset:
                forced_safety_resets += 1
            else:
                stage_success_resets += 1

        episode_complete = bool(done_bool.all().item())
        if not policy_interval_complete and not episode_complete:
            continue

        if episode_complete:
            episode_successes = int((episode_ever_success & episode_safe).sum())
            completed_episodes += env.num_envs
            successful_episodes += episode_successes
            session_completed_episodes += env.num_envs
            session_successful_episodes += episode_successes
            window_episode_return_sum += float(episode_return.sum())
            episode_ever_success.zero_()
            episode_safe.fill_(True)
            episode_return.zero_()
            scheduler.reset()
            previous_residual.zero_()
            clock.reset()
            target_safety.reset()
        else:
            scheduler.advance()
            previous_residual = residual.detach()

        if episode_complete:
            if reusable_prefix_state is None:
                gym_env.reset()
                settled = settle_after_reset(
                    gym_env,
                    env,
                    steps=args_cli.reset_settle_steps,
                )
                prefix_result = execute_deployable_teacher_prefix(
                    gym_env,
                    env,
                    hard_force_limit_n=hard_reset_finger_force_n,
                )
                deployable_prefix_simulator_steps += int(
                    prefix_result["simulator_steps"]
                )
                max_deployable_prefix_finger_force_n = max(
                    max_deployable_prefix_finger_force_n,
                    float(prefix_result["max_finger_force_n"]),
                )
            else:
                prefix_result = restore_deployable_prefix_state(
                    gym_env,
                    env,
                    reusable_prefix_state,
                    reset_settle_steps=args_cli.reset_settle_steps,
                    hard_force_limit_n=hard_reset_finger_force_n,
                )
                settled = int(prefix_result["settle_steps"])
                reused_prefix_state_resets += 1
                max_reused_prefix_state_force_n = max(
                    max_reused_prefix_state_force_n,
                    float(prefix_result["max_finger_force_n"]),
                )
            reset_settle_simulator_steps += settled
            session_reset_settle_simulator_steps += settled
            if runtime_policy_start_step(env) > 0:
                min_deployable_prefix_right_grasp_quality = min(
                    min_deployable_prefix_right_grasp_quality,
                    float(prefix_result["min_right_grasp_quality"]),
                )
                min_deployable_prefix_bimanual_grasp_quality = min(
                    min_deployable_prefix_bimanual_grasp_quality,
                    float(prefix_result["min_bimanual_grasp_quality"]),
                )

        next_images = camera_batch(env)
        next_state = dataset_joint_state(env)
        if episode_complete and runtime_policy_start_step(env) > 0:
            handoff_target = last_commanded_target(env)
            next_base_target = scheduler.anchor_to_target(
                next_images,
                next_state,
                handoff_target,
            )
            target_safety.reset(handoff_target)
            target_delay.reset(handoff_target)
        else:
            next_base_target = scheduler.current(next_images, next_state)
        set_flow_control_ready(env, True)
        next_features = residual_observation(
            flow, next_images, next_state, next_base_target, previous_residual
        ).float()
        if episode_complete and runtime_policy_start_step(env) <= 0:
            target_delay.reset(next_state)
        destination = prior if action_is_prior else online
        destination.add(
            features,
            residual,
            accumulated_reward,
            next_features,
            done_bool.float(),
        )
        transition_count = env.num_envs
        total_transitions += transition_count
        if not action_is_prior:
            online_transitions += transition_count

        if (
            not action_is_prior
            and len(prior) >= args_cli.batch_size // 2
            and len(online) >= args_cli.batch_size // 2
            and online_transitions >= args_cli.learning_starts
        ):
            update_budget += transition_count * args_cli.update_to_data_ratio
            scheduled_updates = int(update_budget)
            update_budget -= scheduled_updates
            for _ in range(scheduled_updates):
                batch = balanced_replay_sample(
                    prior,
                    online,
                    args_cli.batch_size,
                    prior_fraction=0.5,
                    rng=rng,
                )
                update_actor = gradient_updates >= args_cli.critic_warmup_updates
                latest_metrics = agent.update(
                    batch,
                    prior_count=args_cli.batch_size // 2,
                    update_actor=update_actor,
                )
                gradient_updates += 1
                actor_updates += int(update_actor)

        images, state, base_target, features = (
            next_images,
            next_state,
            next_base_target,
            next_features,
        )
        accumulated_reward.zero_()
        residual = None
        safe_target = None

        if total_transitions >= next_log:
            elapsed = time.monotonic() - started
            window_completed = completed_episodes - last_log_completed_episodes
            window_successful = successful_episodes - last_log_successful_episodes
            values = {
                "transitions": total_transitions,
                "online_transitions": online_transitions,
                "prior_replay_size": len(prior),
                "online_replay_size": len(online),
                "session_transitions_per_second": (
                    total_transitions - session_start_transitions
                )
                / max(elapsed, 1.0e-6),
                "simulator_steps": simulator_steps,
                "reset_settle_simulator_steps": reset_settle_simulator_steps,
                "deployable_prefix_simulator_steps": deployable_prefix_simulator_steps,
                "reused_prefix_state_resets": reused_prefix_state_resets,
                "max_reused_prefix_state_force_n": max_reused_prefix_state_force_n,
                "max_deployable_prefix_finger_force_n": (
                    max_deployable_prefix_finger_force_n
                ),
                "min_deployable_prefix_right_grasp_quality": (
                    min_deployable_prefix_right_grasp_quality
                ),
                "min_deployable_prefix_bimanual_grasp_quality": (
                    min_deployable_prefix_bimanual_grasp_quality
                ),
                "gradient_updates": gradient_updates,
                "actor_updates": actor_updates,
                "safety_clip_count": safety_clip_count,
                "effective_update_to_data_ratio": gradient_updates
                / max(online_transitions - args_cli.learning_starts, 1),
                "policy_intervals_per_sim_step": total_transitions
                / max(simulator_steps * env.num_envs, 1),
                "max_table_lift_progress": max_lift,
                "max_table_flip_progress": max_flip,
                "max_finger_force_n": max_finger_force_n,
                "unsafe_force_env_steps": unsafe_force_env_steps,
                "forced_safety_resets": forced_safety_resets,
                "stage_success_resets": stage_success_resets,
                "successful_env_steps": successful_env_steps,
                "completed_episodes": completed_episodes,
                "successful_episodes": successful_episodes,
                "episode_success_rate": successful_episodes / max(completed_episodes, 1),
                "session_completed_episodes": session_completed_episodes,
                "session_successful_episodes": session_successful_episodes,
                "session_episode_success_rate": session_successful_episodes
                / max(session_completed_episodes, 1),
                "window_completed_episodes": window_completed,
                "window_successful_episodes": window_successful,
                "window_episode_success_rate": window_successful / max(window_completed, 1),
                "window_mean_episode_return": window_episode_return_sum
                / max(window_completed, 1),
                **latest_metrics,
            }
            print(json.dumps(values), flush=True)
            append_jsonl(metrics_path, values)
            if wandb_run is not None:
                wandb_run.log(values, step=total_transitions)
            last_log_completed_episodes = completed_episodes
            last_log_successful_episodes = successful_episodes
            window_episode_return_sum = 0.0
            next_log += args_cli.log_every_transitions

        if total_transitions >= next_save:
            save_combined_checkpoint(
                output / f"checkpoint_{total_transitions:09d}",
                flow_checkpoint=flow_checkpoint,
                agent=agent,
                target_safety=target_safety,
                transitions=total_transitions,
                prior_residual=prior_residual,
                prior_action_source=args_cli.prior_action_source,
                actor_init_checkpoint=actor_init_checkpoint,
            )
            save_training_resume(
                output / "resume_latest",
                agent=agent,
                prior=prior,
                online=online,
                rng=rng,
                counters={
                    "total_transitions": total_transitions,
                    "online_transitions": online_transitions,
                    "simulator_steps": simulator_steps,
                    "reset_settle_simulator_steps": reset_settle_simulator_steps,
                    "deployable_prefix_simulator_steps": deployable_prefix_simulator_steps,
                    "max_deployable_prefix_finger_force_n": (
                        max_deployable_prefix_finger_force_n
                    ),
                    "min_deployable_prefix_right_grasp_quality": (
                        min_deployable_prefix_right_grasp_quality
                    ),
                    "min_deployable_prefix_bimanual_grasp_quality": (
                        min_deployable_prefix_bimanual_grasp_quality
                    ),
                    "safety_clip_count": safety_clip_count,
                    "update_budget": update_budget,
                    "gradient_updates": gradient_updates,
                    "actor_updates": actor_updates,
                    "completed_episodes": completed_episodes,
                    "successful_episodes": successful_episodes,
                    "unsafe_force_env_steps": unsafe_force_env_steps,
                    "forced_safety_resets": forced_safety_resets,
                    "stage_success_resets": stage_success_resets,
                    "reused_prefix_state_resets": reused_prefix_state_resets,
                    "max_reused_prefix_state_force_n": (
                        max_reused_prefix_state_force_n
                    ),
                },
                flow_checkpoint=flow_checkpoint,
                prior_residual=prior_residual,
                prior_action_source=args_cli.prior_action_source,
                actor_init_checkpoint=actor_init_checkpoint,
            )
            next_save += args_cli.save_every_transitions

    save_combined_checkpoint(
        output / "final",
        flow_checkpoint=flow_checkpoint,
        agent=agent,
        target_safety=target_safety,
        transitions=total_transitions,
        prior_residual=prior_residual,
        prior_action_source=args_cli.prior_action_source,
        actor_init_checkpoint=actor_init_checkpoint,
    )
    save_training_resume(
        output / "resume_latest",
        agent=agent,
        prior=prior,
        online=online,
        rng=rng,
        counters={
            "total_transitions": total_transitions,
            "online_transitions": online_transitions,
            "simulator_steps": simulator_steps,
            "reset_settle_simulator_steps": reset_settle_simulator_steps,
            "deployable_prefix_simulator_steps": deployable_prefix_simulator_steps,
            "max_deployable_prefix_finger_force_n": (
                max_deployable_prefix_finger_force_n
            ),
            "min_deployable_prefix_right_grasp_quality": (
                min_deployable_prefix_right_grasp_quality
            ),
            "min_deployable_prefix_bimanual_grasp_quality": (
                min_deployable_prefix_bimanual_grasp_quality
            ),
            "safety_clip_count": safety_clip_count,
            "update_budget": update_budget,
            "gradient_updates": gradient_updates,
            "actor_updates": actor_updates,
            "completed_episodes": completed_episodes,
            "successful_episodes": successful_episodes,
            "unsafe_force_env_steps": unsafe_force_env_steps,
            "forced_safety_resets": forced_safety_resets,
            "stage_success_resets": stage_success_resets,
            "reused_prefix_state_resets": reused_prefix_state_resets,
            "max_reused_prefix_state_force_n": max_reused_prefix_state_force_n,
        },
        flow_checkpoint=flow_checkpoint,
        prior_residual=prior_residual,
        prior_action_source=args_cli.prior_action_source,
        actor_init_checkpoint=actor_init_checkpoint,
    )
    final_values = {
        "event": "session_end",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "transitions": total_transitions,
        "online_transitions": online_transitions,
        "simulator_steps": simulator_steps,
        "reset_settle_simulator_steps": reset_settle_simulator_steps,
        "session_reset_settle_simulator_steps": session_reset_settle_simulator_steps,
        "deployable_prefix_simulator_steps": deployable_prefix_simulator_steps,
        "reused_prefix_state_resets": reused_prefix_state_resets,
        "max_reused_prefix_state_force_n": max_reused_prefix_state_force_n,
        "max_deployable_prefix_finger_force_n": max_deployable_prefix_finger_force_n,
        "min_deployable_prefix_right_grasp_quality": (
            min_deployable_prefix_right_grasp_quality
        ),
        "min_deployable_prefix_bimanual_grasp_quality": (
            min_deployable_prefix_bimanual_grasp_quality
        ),
        "gradient_updates": gradient_updates,
        "actor_updates": actor_updates,
        "safety_clip_count": safety_clip_count,
        "completed_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
        "episode_success_rate": successful_episodes / max(completed_episodes, 1),
        "session_completed_episodes": session_completed_episodes,
        "session_successful_episodes": session_successful_episodes,
        "session_episode_success_rate": session_successful_episodes
        / max(session_completed_episodes, 1),
        "max_table_lift_progress": max_lift,
        "max_table_flip_progress": max_flip,
        "max_finger_force_n": max_finger_force_n,
        "unsafe_force_env_steps": unsafe_force_env_steps,
        "forced_safety_resets": forced_safety_resets,
        "stage_success_resets": stage_success_resets,
    }
    append_jsonl(metrics_path, final_values)
    append_jsonl(output / "training_sessions.jsonl", final_values)
    if wandb_run is not None:
        wandb_run.finish()
    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
