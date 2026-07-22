"""Strict, cumulative V1 acceptance checks for generated trajectories."""

from __future__ import annotations

from typing import Any

import torch

from robofinals.utils.isaac_data_compat import as_torch
from robofinals_rl.flip_table import mdp
from robofinals_rl.flip_table.mdp.rewards import contact_measurements_ready_from_steps


_TRACKED_MAXIMA = (
    "finger_contact_force_n",
    "root_motion_m",
    "root_rotation_rad",
    "lower_body_joint_delta_rad",
    "joint_limit_violation_rad",
)

_CONTACT_SENSOR_WARMUP_STEPS = 4


def _acceptance_config(env) -> dict[str, float]:
    value = getattr(env.cfg, "flip_table_acceptance_config", None)
    if not isinstance(value, dict):
        raise RuntimeError("flip_table_acceptance_config is missing from the environment")
    return value


def _task(env):
    task = env.cfg.isaaclab_arena_env.task
    required = (
        "_stable_flip_success_components",
        "_apply_lower_body_lock",
        "_lower_body_lock_root_pose",
        "_lower_body_lock_joint_pos",
    )
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise RuntimeError(f"V1 flip-table task is missing acceptance hooks: {missing}")
    return task


def _robot_root_pose(env) -> torch.Tensor:
    robot = env.scene["robot"]
    for field in ("root_pose_w", "root_link_pose_w"):
        if hasattr(robot.data, field):
            value = as_torch(getattr(robot.data, field))
            if value.ndim == 2 and value.shape[1] >= 7:
                return value[:, :7]
    raise RuntimeError("robot articulation exposes no root pose for acceptance audit")


def _quaternion_distance_wxyz(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(first * second, dim=-1))
    return 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))


def _lower_body_delta(env, task) -> torch.Tensor:
    robot = env.scene["robot"]
    baseline = task._lower_body_lock_joint_pos
    joint_ids = task._resolve_lower_body_joint_ids(robot, env.device)
    if baseline is None or joint_ids.numel() == 0:
        return torch.full((env.num_envs,), torch.inf, device=env.device)
    current = as_torch(robot.data.joint_pos)[:, joint_ids]
    return torch.amax(torch.abs(current - baseline.to(current)), dim=1)


def _joint_limit_violation(env) -> torch.Tensor:
    robot = env.scene["robot"]
    positions = as_torch(robot.data.joint_pos)
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        return torch.full((env.num_envs,), torch.inf, device=env.device)
    limits = as_torch(limits).to(positions)
    if limits.ndim == 2:
        limits = limits.unsqueeze(0).expand(env.num_envs, -1, -1)
    if limits.shape != (env.num_envs, positions.shape[1], 2):
        raise RuntimeError(
            "soft joint limits have unexpected shape "
            f"{tuple(limits.shape)} for joint positions {tuple(positions.shape)}"
        )
    below = limits[:, :, 0] - positions
    above = positions - limits[:, :, 1]
    return torch.clamp(torch.maximum(below, above), min=0.0).amax(dim=1)


def _current_metrics(env, task) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    root = _robot_root_pose(env)
    root_baseline = task._lower_body_lock_root_pose
    if root_baseline is None:
        root_motion = torch.full((env.num_envs,), torch.inf, device=env.device)
        root_rotation = root_motion.clone()
    else:
        baseline = root_baseline.to(root)
        root_motion = torch.linalg.norm(root[:, :3] - baseline[:, :3], dim=1)
        root_rotation = _quaternion_distance_wxyz(root[:, 3:7], baseline[:, 3:7])

    forces = mdp.finger_contact_forces(env)
    if forces.ndim != 3:
        raise RuntimeError(f"finger contact force tensor must be [B,H,F], got {tuple(forces.shape)}")
    contact_ready = contact_measurements_ready_from_steps(
        as_torch(env.episode_length_buf),
        _CONTACT_SENSOR_WARMUP_STEPS,
    ).to(device=env.device, dtype=torch.bool)
    metrics = {
        "finger_contact_force_n": forces.amax(dim=(1, 2)),
        "root_motion_m": root_motion,
        "root_rotation_rad": root_rotation,
        "lower_body_joint_delta_rad": _lower_body_delta(env, task),
        "joint_limit_violation_rad": _joint_limit_violation(env),
    }
    return metrics, contact_ready


def _update_cumulative_audit(
    env,
    current: dict[str, torch.Tensor],
    contact_ready: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    reset = as_torch(env.episode_length_buf).long() <= 1
    maxima = {}
    for name in _TRACKED_MAXIMA:
        previous = getattr(env, f"_flip_table_mimic_max_{name}", None)
        if previous is None:
            previous = torch.zeros_like(current[name])
        value = torch.where(reset, current[name], torch.maximum(previous, current[name]))
        setattr(env, f"_flip_table_mimic_max_{name}", value)
        maxima[name] = value
    ready_ever = getattr(
        env,
        "_flip_table_mimic_contact_ready_ever",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    ready_ever = torch.where(reset, contact_ready, ready_ever | contact_ready)
    env._flip_table_mimic_contact_ready_ever = ready_ever
    return maxima, ready_ever


def strict_flip_table_success(env) -> torch.Tensor:
    """Return success only after final-state and whole-trajectory safety gates pass."""

    config = _acceptance_config(env)
    task = _task(env)
    components = task._stable_flip_success_components(env)
    current_metrics, contact_ready = _current_metrics(env, task)
    maxima, contact_ready_ever = _update_cumulative_audit(env, current_metrics, contact_ready)
    safe = (
        contact_ready_ever
        & (maxima["finger_contact_force_n"] <= config["finger_contact_force_n_max"])
        & (maxima["root_motion_m"] <= config["reject_root_motion_m_max"])
        & (maxima["root_rotation_rad"] <= config["reject_root_rotation_rad_max"])
        & (
            maxima["lower_body_joint_delta_rad"]
            <= config["reject_lower_body_joint_delta_rad_max"]
        )
        & (
            maxima["joint_limit_violation_rad"]
            <= config["reject_joint_limit_violation_rad_max"]
        )
    )
    candidate = components["candidate"] & safe
    reset = as_torch(env.episode_length_buf).long() <= 1
    streak = getattr(
        env,
        "_flip_table_mimic_success_streak",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    streak = torch.where(
        reset,
        torch.zeros_like(streak),
        torch.where(candidate, streak + 1, torch.zeros_like(streak)),
    )
    env._flip_table_mimic_success_streak = streak
    success = candidate & (streak >= int(config["hold_steps_min"]))
    env._flip_table_mimic_last_report = {
        **components,
        **{f"max_{name}": value for name, value in maxima.items()},
        "contact_sensor_ready": contact_ready_ever,
        "safety_passed": safe,
        "success_streak": streak,
        "passed": success,
    }

    # Audit displacement before restoring the fixed lower-body contract.
    task._apply_lower_body_lock(env)
    return success


def candidate_report(env, env_id: int) -> dict[str, Any]:
    """Return the last-step acceptance decision and explicit rejection reasons."""

    values = getattr(env, "_flip_table_mimic_last_report", None)
    if not isinstance(values, dict):
        raise RuntimeError("no acceptance report was recorded for this attempt")
    config = _acceptance_config(env)

    def scalar(name: str):
        value = values[name][env_id].detach().cpu().item()
        return bool(value) if values[name].dtype == torch.bool else int(value) if name == "success_streak" else float(value)

    report = {
        "passed": scalar("passed"),
        "contact_sensor_ready": scalar("contact_sensor_ready"),
        "normal_dot": scalar("normal_dot"),
        "tabletop_lift_m": scalar("tabletop_lift_m"),
        "linear_speed_m_s": scalar("linear_speed_m_s"),
        "angular_speed_rad_s": scalar("angular_speed_rad_s"),
        "within_workbench": scalar("within_workbench"),
        "gripper_clear": scalar("gripper_clear"),
        "success_streak": scalar("success_streak"),
    }
    for name in _TRACKED_MAXIMA:
        report[f"max_{name}"] = scalar(f"max_{name}")

    reasons = []
    checks = (
        (report["contact_sensor_ready"], "contact_sensor_unavailable"),
        (report["normal_dot"] <= config["normal_dot_max"], "table_not_inverted"),
        (report["tabletop_lift_m"] >= config["tabletop_lift_m_min"], "tabletop_not_lifted"),
        (
            report["linear_speed_m_s"] <= config["settled_linear_speed_m_s_max"],
            "table_linear_speed_too_high",
        ),
        (
            report["angular_speed_rad_s"] <= config["settled_angular_speed_rad_s_max"],
            "table_angular_speed_too_high",
        ),
        (report["within_workbench"], "table_outside_workbench"),
        (report["gripper_clear"], "grippers_not_released"),
        (
            report["max_finger_contact_force_n"] <= config["finger_contact_force_n_max"],
            "finger_contact_force_exceeded",
        ),
        (
            report["max_root_motion_m"] <= config["reject_root_motion_m_max"],
            "root_translation_exceeded",
        ),
        (
            report["max_root_rotation_rad"] <= config["reject_root_rotation_rad_max"],
            "root_rotation_exceeded",
        ),
        (
            report["max_lower_body_joint_delta_rad"]
            <= config["reject_lower_body_joint_delta_rad_max"],
            "lower_body_motion_exceeded",
        ),
        (
            report["max_joint_limit_violation_rad"]
            <= config["reject_joint_limit_violation_rad_max"],
            "joint_limit_violation",
        ),
        (report["success_streak"] >= int(config["hold_steps_min"]), "success_hold_too_short"),
    )
    report["rejection_reasons"] = [reason for passed, reason in checks if not passed]
    report["passed"] = not report["rejection_reasons"]
    return report
