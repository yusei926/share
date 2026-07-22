"""Shared, simulator-independent definitions for flip-table residual RL."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch


UPPER_BODY_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

DEX1_OPEN_POS = 0.0245
DEX1_CLOSE_POS = -0.02
DEMO_HAND_CLOSED = 0.0
DEMO_HAND_OPEN = 4.5

# Residuals are deliberately smaller than the full URDF range. The real
# demonstration supplies the nominal motion and PPO learns contact corrections.
RESIDUAL_SCALE_RAD = {
    "waist_yaw_joint": 0.25,
    "waist_roll_joint": 0.15,
    "waist_pitch_joint": 0.18,
    ".*_shoulder_pitch_joint": 0.40,
    ".*_shoulder_roll_joint": 0.35,
    ".*_shoulder_yaw_joint": 0.35,
    ".*_elbow_joint": 0.40,
    ".*_wrist_roll_joint": 0.45,
    ".*_wrist_pitch_joint": 0.40,
    ".*_wrist_yaw_joint": 0.45,
}

JOINT_POSITION_LIMITS_RAD = {
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.520, 0.520),
    "waist_pitch_joint": (-0.520, 0.520),
    ".*_shoulder_pitch_joint": (-3.089, 2.670),
    "left_shoulder_roll_joint": (-1.588, 2.251),
    "right_shoulder_roll_joint": (-2.251, 1.588),
    ".*_shoulder_yaw_joint": (-2.618, 2.618),
    ".*_elbow_joint": (-1.047, 2.094),
    ".*_wrist_roll_joint": (-1.972, 1.972),
    ".*_wrist_pitch_joint": (-1.614, 1.614),
    ".*_wrist_yaw_joint": (-1.614, 1.614),
}

ACTION_DIM = 19
DEFAULT_STAGE = "reach"


def load_demo_actions(path: str | Path) -> torch.Tensor:
    """Load and validate a 19-D real-robot upper-body target trajectory."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("actions")
    actions = torch.as_tensor(payload, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != 19 or actions.shape[0] < 2:
        raise ValueError(f"demo actions must have shape [N>=2, 19], got {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise ValueError("demo actions contain NaN or Inf")
    return actions


def _trajectory_source_control_hz(payload: object, source: Path) -> float | None:
    if not isinstance(payload, dict):
        return None
    retargeting = payload.get("retargeting", {})
    if not isinstance(retargeting, dict) or retargeting.get("control_hz") is None:
        return None
    value = float(retargeting["control_hz"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"trajectory control_hz must be finite and positive: {source}")
    return value


def resample_controller_schedule(
    schedule: torch.Tensor,
    *,
    source_hz: float,
    target_hz: float,
) -> torch.Tensor:
    """Linearly resample a controller-clock schedule without changing duration."""

    if schedule.ndim != 2 or schedule.shape[0] < 1:
        raise ValueError(f"controller schedule must be [T>=1,D], got {tuple(schedule.shape)}")
    if not torch.isfinite(schedule).all():
        raise ValueError("controller schedule contains NaN or Inf")
    if not all(math.isfinite(value) and value > 0.0 for value in (source_hz, target_hz)):
        raise ValueError("source_hz and target_hz must be finite and positive")
    if math.isclose(source_hz, target_hz, rel_tol=0.0, abs_tol=1.0e-9):
        return schedule.clone()

    target_steps = max(1, int(round(schedule.shape[0] * target_hz / source_hz)))
    # A piecewise trajectory's first sample is the target after one source
    # controller interval. Include its implicit zero target at t=0 so an
    # upsampled trajectory preserves the original ramp instead of jumping.
    padded = torch.cat((torch.zeros_like(schedule[:1]), schedule), dim=0)
    source_positions = (
        torch.arange(1, target_steps + 1, device=schedule.device, dtype=torch.float64)
        * (source_hz / target_hz)
    ).clamp(0.0, float(schedule.shape[0]))
    lower = torch.floor(source_positions).to(torch.long)
    upper = torch.ceil(source_positions).to(torch.long)
    alpha = (source_positions - lower.to(source_positions.dtype)).to(schedule.dtype).unsqueeze(1)
    return padded[lower] + alpha * (padded[upper] - padded[lower])


def load_labeled_residual_action_trajectory(
    path: str | Path,
    *,
    target_control_hz: float | None = None,
    require_source_control_hz: bool = False,
) -> tuple[torch.Tensor, list[str], dict[str, object]]:
    """Expand a trajectory and preserve segment labels across resampling."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"residual trajectory has no segments: {source}")

    previous = torch.zeros(ACTION_DIM, dtype=torch.float32)
    schedule: list[torch.Tensor] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"residual trajectory segment {index} must be an object")
        target = torch.as_tensor(segment.get("action"), dtype=torch.float32)
        if target.shape != (ACTION_DIM,) or not torch.isfinite(target).all():
            raise ValueError(f"residual trajectory segment {index} must contain 19 finite values")
        if torch.any(target.abs() > 1.0):
            raise ValueError(f"residual trajectory segment {index} exceeds [-1, 1]")
        steps = int(segment.get("steps", 0))
        ramp_steps = int(segment.get("ramp_steps", 0))
        if steps <= 0 or not 0 <= ramp_steps <= steps:
            raise ValueError(f"residual trajectory segment {index} has an invalid schedule")
        label = str(segment.get("label", f"segment_{index}"))
        for local_step in range(steps):
            alpha = min(1.0, float(local_step + 1) / ramp_steps) if ramp_steps else 1.0
            schedule.append(previous + (target - previous) * alpha)
            labels.append(label)
        previous = target
    expanded = torch.stack(schedule)
    source_hz = _trajectory_source_control_hz(payload, source)
    if require_source_control_hz and source_hz is None:
        raise ValueError(f"trajectory must declare retargeting.control_hz: {source}")
    if target_control_hz is not None and source_hz is not None:
        resampled = resample_controller_schedule(
            expanded,
            source_hz=source_hz,
            target_hz=target_control_hz,
        )
        if not math.isclose(source_hz, target_control_hz, rel_tol=0.0, abs_tol=1.0e-9):
            resampled_labels = []
            for target_step in range(len(resampled)):
                source_position = (target_step + 1) * source_hz / target_control_hz
                source_index = min(
                    len(labels) - 1,
                    max(0, int(math.ceil(source_position)) - 1),
                )
                resampled_labels.append(labels[source_index])
            labels = resampled_labels
        expanded = resampled
    retargeting = payload.get("retargeting", {}) if isinstance(payload, dict) else {}
    if not isinstance(retargeting, dict):
        raise ValueError(f"trajectory retargeting metadata must be an object: {source}")
    return expanded, labels, dict(retargeting)


def load_residual_action_trajectory(
    path: str | Path,
    *,
    target_control_hz: float | None = None,
    require_source_control_hz: bool = False,
) -> torch.Tensor:
    """Expand and optionally time-resample a deployable 19-D trajectory."""

    schedule, _labels, _metadata = load_labeled_residual_action_trajectory(
        path,
        target_control_hz=target_control_hz,
        require_source_control_hz=require_source_control_hz,
    )
    return schedule


def _environment_control_hz(env) -> float:
    control_dt = float(env.cfg.sim.dt * env.cfg.decimation)
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("environment controller interval must be finite and positive")
    return 1.0 / control_dt


def _action_prior_source_control_hz() -> float | None:
    trajectory_path = os.environ.get("FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY", "").strip()
    if not trajectory_path:
        return None
    source = Path(trajectory_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _trajectory_source_control_hz(payload, source)


def runtime_controller_steps(env, source_steps: int) -> int:
    """Convert source-trajectory steps to the active simulator controller clock."""

    if source_steps < 0:
        raise ValueError("controller steps must be non-negative")
    source_hz = _action_prior_source_control_hz()
    if source_hz is None:
        return source_steps
    return int(round(source_steps * _environment_control_hz(env) / source_hz))


def runtime_policy_start_step(env) -> int:
    """Return the deployable teacher-prefix boundary on the runtime clock."""

    source_step = int(os.environ.get("FLIP_TABLE_RL_POLICY_START_STEP", "0"))
    return runtime_controller_steps(env, source_step)


def action_prior_schedule(env) -> torch.Tensor:
    """Load one controller-clock prior shared by action and observation terms."""

    cached = getattr(env, "_flip_table_rl_action_prior_schedule", None)
    if cached is not None:
        return cached

    trajectory_path = os.environ.get("FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY", "").strip()
    constant_raw = os.environ.get("FLIP_TABLE_RL_TEACHER_RESIDUAL_ACTION", "").strip()
    if trajectory_path and constant_raw:
        raise ValueError(
            "FLIP_TABLE_RL_ACTION_PRIOR_TRAJECTORY and "
            "FLIP_TABLE_RL_TEACHER_RESIDUAL_ACTION are mutually exclusive"
        )
    if trajectory_path:
        cached = load_residual_action_trajectory(
            trajectory_path,
            target_control_hz=_environment_control_hz(env),
        ).to(env.device)
    elif constant_raw:
        values = [float(value.strip()) for value in constant_raw.split(",") if value.strip()]
        if len(values) != ACTION_DIM:
            raise ValueError(
                "FLIP_TABLE_RL_TEACHER_RESIDUAL_ACTION must contain 19 comma-separated values"
            )
        cached = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)
        if not torch.isfinite(cached).all():
            raise ValueError("FLIP_TABLE_RL_TEACHER_RESIDUAL_ACTION contains NaN or Inf")
        cached = cached.clamp(-1.0, 1.0)
    else:
        cached = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=env.device)
    env._flip_table_rl_action_prior_schedule = cached
    return cached


def action_prior_at_steps(schedule: torch.Tensor, episode_steps: torch.Tensor) -> torch.Tensor:
    """Select a full or per-action-term prior using the deployable controller clock."""

    if schedule.ndim != 2 or schedule.shape[0] < 1 or schedule.shape[1] < 1:
        raise ValueError(f"action prior must be [T>=1, D>=1], got {tuple(schedule.shape)}")
    if episode_steps.ndim != 1:
        raise ValueError(f"episode_steps must be [B], got {tuple(episode_steps.shape)}")
    indices = episode_steps.to(device=schedule.device, dtype=torch.long).clamp(
        min=0,
        max=schedule.shape[0] - 1,
    )
    return schedule[indices]


def action_prior_phase(schedule: torch.Tensor, episode_steps: torch.Tensor) -> torch.Tensor:
    """Return normalized controller time for a real-deployable action prior."""

    denominator = max(1, schedule.shape[0] - 1)
    phase = episode_steps.to(device=schedule.device, dtype=torch.float32) / float(denominator)
    return phase.clamp(0.0, 1.0).unsqueeze(1)


def demo_hand_to_dex1_command(values: torch.Tensor) -> torch.Tensor:
    normalized = (values - DEMO_HAND_OPEN) / (DEMO_HAND_CLOSED - DEMO_HAND_OPEN)
    return (2.0 * normalized - 1.0).clamp(-1.0, 1.0)


def dex1_command_to_demo_hand(values: torch.Tensor) -> torch.Tensor:
    command = values.clamp(-1.0, 1.0)
    closed_fraction = 0.5 * (command + 1.0)
    return DEMO_HAND_OPEN + closed_fraction * (DEMO_HAND_CLOSED - DEMO_HAND_OPEN)


def dex1_joint_to_command(values: torch.Tensor) -> torch.Tensor:
    normalized = (values - DEX1_OPEN_POS) / (DEX1_CLOSE_POS - DEX1_OPEN_POS)
    return (2.0 * normalized - 1.0).clamp(-1.0, 1.0)


def dex1_joint_velocity_to_command(values: torch.Tensor) -> torch.Tensor:
    """Map Dex1 prismatic velocity to the derivative of its command scale."""

    return values * (2.0 / (DEX1_CLOSE_POS - DEX1_OPEN_POS))


def demo_actions_in_controller_domain(actions: torch.Tensor) -> torch.Tensor:
    result = actions.clone()
    result[..., 17:19] = demo_hand_to_dex1_command(result[..., 17:19])
    return result


def teacher_residual_scales(
    progress: torch.Tensor,
    *,
    fade_start_index: int = -1,
    fade_end_index: int = -1,
) -> torch.Tensor:
    """Fade a grasp teacher using only real-deployable demo progress."""

    if progress.ndim != 1:
        raise ValueError(f"progress must be [B], got {tuple(progress.shape)}")
    if fade_start_index < 0 and fade_end_index < 0:
        return torch.ones_like(progress, dtype=torch.float32)
    if fade_start_index < 0 or fade_end_index <= fade_start_index:
        raise ValueError("teacher fade requires 0 <= start < end")
    scale = (float(fade_end_index) - progress.to(torch.float32)) / float(
        fade_end_index - fade_start_index
    )
    return scale.clamp(0.0, 1.0)


def nearest_demo_targets(
    current_joint_pos: torch.Tensor,
    demo_actions: torch.Tensor,
    progress: torch.Tensor | None = None,
    *,
    lookahead: int = 3,
    search_back: int = 2,
    search_forward: int = 45,
    min_index: int = 0,
    max_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a monotonic, state-conditioned target from a real demonstration.

    Only the 17 real upper-body joint values are used. This same operation can
    run in simulation and on G1, so it does not introduce privileged state.
    """

    if current_joint_pos.ndim != 2 or current_joint_pos.shape[1] != 17:
        raise ValueError(f"current_joint_pos must be [B, 17], got {tuple(current_joint_pos.shape)}")
    if demo_actions.ndim != 2 or demo_actions.shape[1] != 19:
        raise ValueError(f"demo_actions must be [T, 19], got {tuple(demo_actions.shape)}")

    batch = current_joint_pos.shape[0]
    horizon = demo_actions.shape[0]
    if lookahead < 0 or search_back < 0 or search_forward < 0:
        raise ValueError("lookahead and demo search widths must be non-negative")
    lower = int(min_index)
    upper = horizon - 1 if max_index is None else int(max_index)
    if not 0 <= lower <= upper < horizon:
        raise ValueError(f"demo search bounds must satisfy 0 <= min <= max < {horizon}")
    if progress is None:
        progress = torch.full(
            (batch,), lower, dtype=torch.long, device=current_joint_pos.device
        )
    progress = progress.to(device=current_joint_pos.device, dtype=torch.long).clamp(lower, upper)
    demo = demo_actions.to(device=current_joint_pos.device, dtype=current_joint_pos.dtype)

    offsets = torch.arange(-search_back, search_forward + 1, device=current_joint_pos.device)
    candidates = (progress[:, None] + offsets[None, :]).clamp(lower, upper)
    candidate_states = demo[candidates, :17]
    distances = torch.mean((candidate_states - current_joint_pos[:, None, :]) ** 2, dim=-1)
    nearest = candidates.gather(1, distances.argmin(dim=1, keepdim=True)).squeeze(1)
    monotonic = torch.maximum(progress, nearest)
    target_index = (monotonic + lookahead).clamp(max=upper)
    return demo[target_index], monotonic


def phase_demo_targets(
    current_joint_pos: torch.Tensor,
    demo_actions: torch.Tensor,
    progress: torch.Tensor,
    episode_steps: torch.Tensor,
    *,
    mode: str,
    start_index: int,
    end_index: int,
    control_dt: float,
    demo_hz: float,
    hold_index: int = -1,
    hold_steps: int = 0,
    resume_demo_hz: float | None = None,
    lookahead: int = 3,
    search_back: int = 2,
    search_forward: int = 45,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a demo target using a real-deployable phase source.

    ``hold`` is used to learn initial reach/grasp corrections. ``clock`` uses
    only the monotonic controller step and therefore runs identically in sim
    and on the real robot. ``clock_hold`` pauses at one demonstration frame for
    a fixed number of controller steps before resuming. ``state`` remains
    available for diagnostics.
    """

    horizon = demo_actions.shape[0]
    if horizon < 2:
        raise ValueError("demo_actions must contain at least two frames")
    if mode not in {"hold", "clock", "clock_hold", "state"}:
        raise ValueError(f"unknown demo phase mode: {mode!r}")
    if control_dt <= 0 or demo_hz <= 0:
        raise ValueError("control_dt and demo_hz must be positive")
    if resume_demo_hz is not None and resume_demo_hz <= 0:
        raise ValueError("resume_demo_hz must be positive when specified")
    start = max(0, min(int(start_index), horizon - 1))
    end = max(start, min(int(end_index), horizon - 1))
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative")

    if mode == "state":
        return nearest_demo_targets(
            current_joint_pos,
            demo_actions,
            progress,
            lookahead=lookahead,
            search_back=search_back,
            search_forward=search_forward,
            min_index=start,
            max_index=end,
        )

    if episode_steps.ndim != 1 or episode_steps.shape[0] != current_joint_pos.shape[0]:
        raise ValueError(
            f"episode_steps must be [B], got {tuple(episode_steps.shape)} for B={current_joint_pos.shape[0]}"
        )
    if mode == "hold":
        indices = torch.full_like(progress, start)
        target_indices = indices
    elif mode == "clock_hold":
        hold = max(start, min(int(hold_index), end))
        frames_per_step = float(control_dt * demo_hz)
        resume_frames_per_step = float(control_dt * (resume_demo_hz or demo_hz))
        steps_to_hold = int(math.ceil((hold - start) / frames_per_step))
        steps = episode_steps.to(torch.long)
        before_hold = steps < steps_to_hold
        in_hold = (steps >= steps_to_hold) & (steps < steps_to_hold + int(hold_steps))
        before_indices = (
            torch.floor(steps.to(torch.float32) * frames_per_step).long() + start
        ).clamp(min=start, max=hold)
        after_steps = (steps - steps_to_hold - int(hold_steps)).clamp_min(0)
        after_indices = (
            torch.floor(after_steps.to(torch.float32) * resume_frames_per_step).long() + hold
        ).clamp(min=hold, max=end)
        indices = torch.where(before_hold, before_indices, torch.where(in_hold, hold, after_indices))
        before_targets = (before_indices + max(0, lookahead)).clamp(max=hold)
        after_targets = (after_indices + max(0, lookahead)).clamp(max=end)
        target_indices = torch.where(
            before_hold,
            before_targets,
            torch.where(in_hold, torch.full_like(indices, hold), after_targets),
        )
    else:
        elapsed = torch.floor(episode_steps.to(torch.float32) * float(control_dt * demo_hz)).long()
        indices = (elapsed + start).clamp(min=start, max=end)
        target_indices = (indices + max(0, lookahead)).clamp(max=end)
    return demo_actions.to(current_joint_pos.device)[target_indices], indices
