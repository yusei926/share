"""Real-demonstration residual joint action used by flip-table PPO."""

from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path
from typing import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from robofinals.core.mdp.actions.joint_position_map_action import (
    JointPositionMapAction,
    JointPositionMapActionCfg,
)

from .common import (
    UPPER_BODY_JOINT_NAMES,
    action_prior_at_steps,
    action_prior_schedule,
    demo_hand_to_dex1_command,
    dex1_command_to_demo_hand,
    load_demo_actions,
    phase_demo_targets,
    runtime_controller_steps,
    runtime_policy_start_step,
    teacher_residual_scales,
)


def _phase_settings(env, horizon: int) -> dict[str, object]:
    demo_hz = float(os.environ.get("FLIP_TABLE_RL_DEMO_HZ", "30"))
    return {
        "mode": os.environ.get("FLIP_TABLE_RL_PHASE_MODE", "state").strip().lower(),
        "start_index": int(os.environ.get("FLIP_TABLE_RL_DEMO_START_INDEX", "0")),
        "end_index": int(os.environ.get("FLIP_TABLE_RL_DEMO_END_INDEX", str(horizon - 1))),
        "control_dt": float(env.cfg.sim.dt * env.cfg.decimation),
        "demo_hz": demo_hz,
        "hold_index": int(os.environ.get("FLIP_TABLE_RL_DEMO_HOLD_INDEX", "-1")),
        "hold_steps": runtime_controller_steps(
            env,
            int(os.environ.get("FLIP_TABLE_RL_DEMO_HOLD_STEPS", "0")),
        ),
        "resume_demo_hz": float(os.environ.get("FLIP_TABLE_RL_DEMO_RESUME_HZ", str(demo_hz))),
    }


def _default_action_delay_steps() -> int:
    level = max(0.0, min(1.0, float(os.environ.get("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", "1.0"))))
    return 2 if level >= 0.8 else 1


def _teacher_residual(env) -> tuple[torch.Tensor, float]:
    cached = action_prior_schedule(env)
    correction_scale = float(os.environ.get("FLIP_TABLE_RL_POLICY_CORRECTION_SCALE", "1.0"))
    if not 0.0 < correction_scale <= 1.0:
        raise ValueError("FLIP_TABLE_RL_POLICY_CORRECTION_SCALE must be in (0, 1]")
    return cached, correction_scale


def _policy_correction_gate(env) -> torch.Tensor:
    start_step = runtime_policy_start_step(env)
    clock_ready = env.episode_length_buf >= start_step
    use_flow_base = os.environ.get("FLIP_TABLE_RLPD_USE_FLOW_BASE", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if use_flow_base and start_step > 0:
        runner_ready = getattr(env, "_flip_table_rlpd_flow_ready", None)
        if runner_ready is None:
            runner_ready = torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            )
        clock_ready = clock_ready & torch.as_tensor(
            runner_ready,
            dtype=torch.bool,
            device=env.device,
        )
    return clock_ready.to(torch.float32).unsqueeze(1)


def _teacher_fade_settings() -> dict[str, int]:
    return {
        "fade_start_index": int(os.environ.get("FLIP_TABLE_RL_TEACHER_FADE_START_INDEX", "-1")),
        "fade_end_index": int(os.environ.get("FLIP_TABLE_RL_TEACHER_FADE_END_INDEX", "-1")),
    }


def _policy_residual_range_multiplier() -> float:
    value = float(os.environ.get("FLIP_TABLE_RL_POLICY_RESIDUAL_RANGE_MULTIPLIER", "1.0"))
    if not 0.0 < value <= 3.0:
        raise ValueError("FLIP_TABLE_RL_POLICY_RESIDUAL_RANGE_MULTIPLIER must be in (0, 3]")
    return value


def _enforce_sim_lower_body_lock(env) -> None:
    """Apply the task's fixed-lower-body contract before each RL control tick."""

    arena_cfg = getattr(env.cfg, "isaaclab_arena_env", None)
    task = getattr(arena_cfg, "task", None)
    lock = getattr(task, "_apply_lower_body_lock", None)
    if lock is None:
        raise RuntimeError("flip-table task does not expose the lower-body lock")
    lock(env)


def _rlpd_absolute_target(env, *, num_envs: int) -> torch.Tensor:
    value = getattr(env, "_flip_table_rlpd_absolute_target", None)
    if value is None:
        raise RuntimeError("RLPD runner must set env._flip_table_rlpd_absolute_target before env.step")
    target = torch.as_tensor(value, device=env.device, dtype=torch.float32)
    if target.shape != (num_envs, 19):
        raise ValueError(f"RLPD absolute target must be [{num_envs},19], got {tuple(target.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError("RLPD absolute target contains NaN or Inf")
    return target


def _external_absolute_target_enabled(env) -> bool:
    """Return whether an offline controller currently owns the 19-D target.

    Normal Flow/RLPD execution keeps the default ``True``.  A scripted teacher
    can temporarily replay its recorded residual prefix with this disabled,
    then enable the same adapter's absolute-target path at a controlled
    handoff.  This flag is simulator-teacher infrastructure only.
    """

    return bool(getattr(env, "_flip_table_rlpd_external_target_enabled", True))


def _absolute_target_path_active(env, *, use_flow_base: bool) -> bool:
    """Return whether this tick deliberately replaces the residual action path.

    Flow/RLPD always owns an absolute target.  Offline teachers that replay a
    residual prefix must *not* enable Flow merely to hand off later: doing so
    changes the sampled residual-action delay and invalidates the prefix.  Such
    teachers set this explicit per-environment flag only after their prefix
    has completed, preserving the original residual controller contract.
    """

    return use_flow_base or bool(getattr(env, "_flip_table_rlpd_force_external_target", False))


class _ResidualDelayBuffer:
    """Apply one shared per-episode controller delay to an action term."""

    def _initialize_delay_buffer(self, env, action_dim: int, *, owner: bool) -> None:
        self._delay_env = env
        self._delay_owner = owner
        if os.environ.get("FLIP_TABLE_RLPD_USE_FLOW_BASE", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            self._max_delay_steps = 0
        else:
            self._max_delay_steps = max(
                0,
                int(
                    os.environ.get(
                        "FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS",
                        str(_default_action_delay_steps()),
                    )
                ),
            )
        self._delay_history = torch.zeros(
            (self.num_envs, self._max_delay_steps + 1, action_dim),
            device=self.device,
        )
        if not hasattr(env, "_flip_table_rl_action_delay_steps"):
            env._flip_table_rl_action_delay_steps = torch.zeros(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        if owner and runtime_policy_start_step(env) > 0:
            env._flip_table_rlpd_flow_ready = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if owner:
            env._flip_table_rlpd_last_commanded_target = torch.zeros(
                (self.num_envs, 19),
                dtype=torch.float32,
                device=self.device,
            )

    def _apply_action_delay(self, actions: torch.Tensor) -> torch.Tensor:
        self._delay_history[:, 1:] = self._delay_history[:, :-1].clone()
        self._delay_history[:, 0] = actions
        delays = self._delay_env._flip_table_rl_action_delay_steps.clamp(0, self._max_delay_steps)
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._delay_history[env_ids, delays]

    def _reset_action_delay(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._delay_history[ids] = 0.0
        if self._delay_owner:
            if self._max_delay_steps == 0:
                sampled = torch.zeros(ids.numel(), dtype=torch.long, device=self.device)
            else:
                sampled = torch.randint(
                    0,
                    self._max_delay_steps + 1,
                    (ids.numel(),),
                    device=self.device,
                )
            self._delay_env._flip_table_rl_action_delay_steps[ids] = sampled
            if runtime_policy_start_step(self._delay_env) > 0:
                self._delay_env._flip_table_rlpd_flow_ready[ids] = False


class DemoResidualJointPositionAction(_ResidualDelayBuffer, JointPositionAction):
    """Add bounded PPO residuals to a monotonic real-demonstration prior."""

    cfg: "DemoResidualJointPositionActionCfg"

    def __init__(self, cfg: "DemoResidualJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        demo_path = os.environ.get("FLIP_TABLE_RL_DEMO_ACTION_PATH", "").strip()
        if not demo_path:
            raise ValueError("FLIP_TABLE_RL_DEMO_ACTION_PATH is required for residual RL")
        if not Path(demo_path).is_file():
            raise FileNotFoundError(demo_path)

        self._demo_actions = load_demo_actions(demo_path).to(self.device)
        self._env = env
        self._phase_settings = _phase_settings(env, self._demo_actions.shape[0])
        self._all_joint_ids, resolved_names = self._asset.find_joints(
            list(UPPER_BODY_JOINT_NAMES), preserve_order=True
        )
        if tuple(resolved_names) != UPPER_BODY_JOINT_NAMES:
            raise RuntimeError(f"unexpected upper-body joint order: {resolved_names}")
        demo_lookup = {name: index for index, name in enumerate(UPPER_BODY_JOINT_NAMES)}
        self._demo_columns = torch.tensor(
            [demo_lookup[name] for name in self._joint_names], device=self.device, dtype=torch.long
        )
        self._policy_residual_range_multiplier = _policy_residual_range_multiplier()
        teacher, self._policy_correction_scale = _teacher_residual(env)
        self._teacher_residual = teacher[:, self._demo_columns]
        self._teacher_residual_scale = (
            self._policy_residual_range_multiplier
            if os.environ.get("FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY", "").strip()
            else 1.0
        )
        self._teacher_fade_settings = _teacher_fade_settings()
        self._progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._use_flow_base = os.environ.get("FLIP_TABLE_RLPD_USE_FLOW_BASE", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._initialize_delay_buffer(env, self.action_dim, owner=cfg.sample_shared_delay)

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._delay_owner:
            _enforce_sim_lower_body_lock(self._env)
        self._raw_actions[:] = self._apply_action_delay(actions.clamp(-1.0, 1.0))
        current = self._asset.data.joint_pos[:, self._all_joint_ids]
        targets, self._progress = phase_demo_targets(
            current,
            self._demo_actions,
            self._progress,
            self._env.episode_length_buf,
            **self._phase_settings,
            lookahead=self.cfg.demo_lookahead,
            search_back=self.cfg.demo_search_back,
            search_forward=self.cfg.demo_search_forward,
        )
        prior = targets[:, self._demo_columns]
        teacher_scale = teacher_residual_scales(
            self._progress,
            **self._teacher_fade_settings,
        ).unsqueeze(1)
        teacher_residual = action_prior_at_steps(
            self._teacher_residual,
            self._env.episode_length_buf,
        )
        teacher_component = teacher_scale * self._teacher_residual_scale * teacher_residual
        correction_gate = _policy_correction_gate(self._env)
        residual_limit = max(1.0, self._policy_residual_range_multiplier)
        effective_residual = (
            teacher_component
            + correction_gate
            * self._policy_correction_scale
            * self._policy_residual_range_multiplier
            * self._raw_actions
        ).clamp(-residual_limit, residual_limit)
        self._processed_actions = prior + effective_residual * self._scale
        if _absolute_target_path_active(
            self._env, use_flow_base=self._use_flow_base
        ) and _external_absolute_target_enabled(self._env):
            external_target = _rlpd_absolute_target(
                self._env,
                num_envs=self.num_envs,
            )[:, self._demo_columns]
            self._processed_actions = torch.where(
                correction_gate.to(torch.bool),
                external_target,
                self._processed_actions,
            )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        self._env._flip_table_rlpd_last_commanded_target[:, self._demo_columns] = (
            self._processed_actions.detach()
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._progress[slice(None) if env_ids is None else env_ids] = 0
        self._reset_action_delay(env_ids)


@configclass
class DemoResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = DemoResidualJointPositionAction
    joint_names: list[str] = MISSING
    demo_lookahead: int = 3
    demo_search_back: int = 2
    demo_search_forward: int = 45
    sample_shared_delay: bool = False


class DemoResidualDex1Action(_ResidualDelayBuffer, JointPositionMapAction):
    """Add a bounded residual to one real-demonstration Dex1 command."""

    cfg: "DemoResidualDex1ActionCfg"

    def __init__(self, cfg: "DemoResidualDex1ActionCfg", env) -> None:
        super().__init__(cfg, env)
        if not 0.0 < self.cfg.residual_scale <= 1.0:
            raise ValueError("Dex1 residual_scale must be in (0, 1]")
        demo_path = os.environ.get("FLIP_TABLE_RL_DEMO_ACTION_PATH", "").strip()
        if not demo_path:
            raise ValueError("FLIP_TABLE_RL_DEMO_ACTION_PATH is required for residual RL")
        if not Path(demo_path).is_file():
            raise FileNotFoundError(demo_path)

        self._demo_actions = load_demo_actions(demo_path).to(self.device)
        self._env = env
        self._phase_settings = _phase_settings(env, self._demo_actions.shape[0])
        self._all_joint_ids, resolved_names = self._asset.find_joints(
            list(UPPER_BODY_JOINT_NAMES), preserve_order=True
        )
        if tuple(resolved_names) != UPPER_BODY_JOINT_NAMES:
            raise RuntimeError(f"unexpected upper-body joint order: {resolved_names}")
        self._policy_residual_range_multiplier = _policy_residual_range_multiplier()
        teacher, self._policy_correction_scale = _teacher_residual(env)
        self._teacher_residual = teacher[:, self.cfg.demo_column : self.cfg.demo_column + 1]
        self._teacher_residual_scale = (
            self._policy_residual_range_multiplier
            if os.environ.get("FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY", "").strip()
            else 1.0
        )
        self._teacher_fade_settings = _teacher_fade_settings()
        self._progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._use_flow_base = os.environ.get("FLIP_TABLE_RLPD_USE_FLOW_BASE", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._initialize_delay_buffer(env, self.action_dim, owner=False)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = self._apply_action_delay(actions.clamp(-1.0, 1.0))
        current = self._asset.data.joint_pos[:, self._all_joint_ids]
        targets, self._progress = phase_demo_targets(
            current,
            self._demo_actions,
            self._progress,
            self._env.episode_length_buf,
            **self._phase_settings,
            lookahead=self.cfg.demo_lookahead,
            search_back=self.cfg.demo_search_back,
            search_forward=self.cfg.demo_search_forward,
        )
        prior = demo_hand_to_dex1_command(targets[:, self.cfg.demo_column]).unsqueeze(1)
        teacher_scale = teacher_residual_scales(
            self._progress,
            **self._teacher_fade_settings,
        ).unsqueeze(1)
        teacher_residual = action_prior_at_steps(
            self._teacher_residual,
            self._env.episode_length_buf,
        )
        teacher_component = teacher_scale * self._teacher_residual_scale * teacher_residual
        correction_gate = _policy_correction_gate(self._env)
        residual_limit = max(1.0, self._policy_residual_range_multiplier)
        effective_residual = (
            teacher_component
            + correction_gate
            * self._policy_correction_scale
            * self._policy_residual_range_multiplier
            * self._raw_actions
        ).clamp(-residual_limit, residual_limit)
        command = (prior + self.cfg.residual_scale * effective_residual).clamp(-1.0, 1.0)
        if _absolute_target_path_active(
            self._env, use_flow_base=self._use_flow_base
        ) and _external_absolute_target_enabled(self._env):
            target = _rlpd_absolute_target(self._env, num_envs=self.num_envs)
            external_command = demo_hand_to_dex1_command(
                target[:, self.cfg.demo_column]
            ).unsqueeze(1)
            command = torch.where(
                correction_gate.to(torch.bool),
                external_command,
                command,
            )
        self._processed_actions = self.cfg.post_process_fn(command, self._joint_names)
        self._env._flip_table_rlpd_last_commanded_target[:, self.cfg.demo_column] = (
            dex1_command_to_demo_hand(command.detach().squeeze(1))
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._progress[slice(None) if env_ids is None else env_ids] = 0
        self._reset_action_delay(env_ids)


@configclass
class DemoResidualDex1ActionCfg(JointPositionMapActionCfg):
    class_type: type[ActionTerm] = DemoResidualDex1Action
    joint_names: list[str] = MISSING
    demo_column: int = MISSING
    # Arm residuals stay local, but a one-dimensional gripper command must be
    # able to override a closed demo prior so the hand can open before contact.
    # The final command is still clamped to the real Dex1 controller range.
    residual_scale: float = 1.0
    demo_lookahead: int = 3
    demo_search_back: int = 2
    demo_search_forward: int = 45
