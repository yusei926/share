"""Shared deployable observation and residual-action contract."""

from __future__ import annotations

import os
from typing import Protocol

import torch


BODY_RESIDUAL_SCALE = (
    0.40,
    0.35,
    0.35,
    0.40,
    0.45,
    0.40,
    0.45,
    0.40,
    0.35,
    0.35,
    0.40,
    0.45,
    0.40,
    0.45,
)
# A residual must be able to replace an incorrect base hand target: moving
# between the normalized open/closed endpoints requires a delta of 2.0. The
# deploy-time velocity and acceleration filter, rather than this range, keeps
# each 30 Hz command update within the demonstrated Dex1 rate envelope.
HAND_RESIDUAL_COMMAND_SCALE = 2.0
POLICY_HAND_CLOSED = 0.0
POLICY_HAND_OPEN = 4.5
POLICY_HAND_MIN = 0.0
POLICY_HAND_MAX = 4.5


def stochastic_action_mask_for_stage(stage: str) -> tuple[float, ...]:
    """Limit early curriculum control to the joints needed by each stage."""

    normalized = stage.strip().lower()
    if normalized in {"reach", "contact", "grasp"}:
        active_indices = set(range(7, 14)) | {15}
        return tuple(
            1.0 if index in active_indices else 0.0 for index in range(16)
        )
    if normalized == "sequential_lift":
        # This curriculum gate deliberately learns the demonstrated right-first
        # lift. Keep the right Dex1 command active so the policy can form and
        # maintain a strict grasp; the subsequent ``lift`` stage unlocks all 16
        # axes so the left arm and Dex1 can join the already lifted table.
        active_indices = {15} | set(range(7, 14))
        return tuple(
            1.0 if index in active_indices else 0.0 for index in range(16)
        )
    if normalized in {
        "lift",
        "rotate",
        "flip",
        "stabilize",
        "full",
    }:
        return (1.0,) * 16
    raise ValueError(f"unknown flip-table curriculum stage: {stage!r}")
BODY_POSITION_LIMITS_RAD = (
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
)
BODY_TARGET_VELOCITY_LIMIT_RAD_S = 3.0
BODY_TARGET_ACCELERATION_LIMIT_RAD_S2 = 75.0
HAND_TARGET_VELOCITY_LIMIT_COMMAND_S = 20.0
HAND_TARGET_ACCELERATION_LIMIT_COMMAND_S2 = 400.0


class FlowPolicyProtocol(Protocol):
    def encode_observation(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor: ...

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor: ...

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor: ...


def residual_observation(
    flow_policy: FlowPolicyProtocol,
    images: torch.Tensor,
    state: torch.Tensor,
    base_target: torch.Tensor,
    previous_residual: torch.Tensor,
) -> torch.Tensor:
    """Build actor/critic features exclusively from real-deployable values."""

    if state.ndim != 2 or state.shape[1] != 19:
        raise ValueError(f"state must be [B,19], got {tuple(state.shape)}")
    for name, value in (("base_target", base_target), ("previous_residual", previous_residual)):
        if value.shape != (state.shape[0], 16):
            raise ValueError(f"{name} must have shape [{state.shape[0]},16], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    context = flow_policy.encode_observation(images, state)
    return torch.cat(
        (
            context,
            flow_policy.normalize_state(state),
            flow_policy.normalize_action(base_target),
            previous_residual,
        ),
        dim=-1,
    )


def apply_residual_to_base(base_target: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Apply bounded residuals in arm-radian and Dex1-command coordinates."""

    if base_target.ndim != 2 or base_target.shape[1] != 16:
        raise ValueError(f"base_target must be [B,16], got {tuple(base_target.shape)}")
    if residual.shape != base_target.shape:
        raise ValueError(f"residual must match base_target, got {tuple(residual.shape)}")
    if not torch.isfinite(base_target).all() or not torch.isfinite(residual).all():
        raise ValueError("base target and residual must be finite")
    bounded = residual.clamp(-1.0, 1.0)
    result = base_target.clone()
    scale = torch.as_tensor(BODY_RESIDUAL_SCALE, device=result.device, dtype=result.dtype)
    result[:, :14] += bounded[:, :14] * scale

    base_command = 2.0 * (
        (base_target[:, 14:16] - POLICY_HAND_OPEN)
        / (POLICY_HAND_CLOSED - POLICY_HAND_OPEN)
    ) - 1.0
    command = (
        base_command + HAND_RESIDUAL_COMMAND_SCALE * bounded[:, 14:16]
    ).clamp(-1.0, 1.0)
    result[:, 14:16] = (
        0.5 * (command + 1.0) * (POLICY_HAND_CLOSED - POLICY_HAND_OPEN)
        + POLICY_HAND_OPEN
    )
    return result


class UpperBodyTargetSafetyFilter:
    """Match the deploy-time G1 target position, velocity and acceleration filter."""

    def __init__(
        self,
        *,
        policy_hz: float = 30.0,
        body_velocity_limit_rad_s: float = BODY_TARGET_VELOCITY_LIMIT_RAD_S,
        body_acceleration_limit_rad_s2: float = BODY_TARGET_ACCELERATION_LIMIT_RAD_S2,
        hand_velocity_limit_command_s: float = HAND_TARGET_VELOCITY_LIMIT_COMMAND_S,
        hand_acceleration_limit_command_s2: float = (
            HAND_TARGET_ACCELERATION_LIMIT_COMMAND_S2
        ),
    ) -> None:
        if min(
            policy_hz,
            body_velocity_limit_rad_s,
            body_acceleration_limit_rad_s2,
            hand_velocity_limit_command_s,
            hand_acceleration_limit_command_s2,
        ) <= 0:
            raise ValueError("target safety rates and limits must be positive")
        self.policy_hz = float(policy_hz)
        self.body_velocity_limit_rad_s = float(body_velocity_limit_rad_s)
        self.body_acceleration_limit_rad_s2 = float(body_acceleration_limit_rad_s2)
        self.hand_velocity_limit_command_s = float(hand_velocity_limit_command_s)
        self.hand_acceleration_limit_command_s2 = float(
            hand_acceleration_limit_command_s2
        )
        self.previous_target: torch.Tensor | None = None
        self.previous_velocity: torch.Tensor | None = None
        self.previous_hand_target: torch.Tensor | None = None
        self.previous_hand_velocity: torch.Tensor | None = None

    def reset(self, initial_target: torch.Tensor | None = None) -> None:
        if initial_target is None:
            self.previous_target = None
            self.previous_velocity = None
            self.previous_hand_target = None
            self.previous_hand_velocity = None
            return
        value = torch.as_tensor(initial_target).detach()
        if value.ndim != 2 or value.shape[1] != 16 or not torch.isfinite(value).all():
            raise ValueError("initial safety target must be finite and shaped [B,16]")
        self.previous_target = value[:, :14].clone()
        self.previous_velocity = torch.zeros_like(self.previous_target)
        self.previous_hand_target = value[:, 14:16].clamp(
            POLICY_HAND_MIN,
            POLICY_HAND_MAX,
        ).clone()
        self.previous_hand_velocity = torch.zeros_like(self.previous_hand_target)

    @torch.no_grad()
    def filter(
        self,
        target: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if target.ndim != 2 or target.shape[1] != 16:
            raise ValueError("target must be [B,16]")
        if current_state.ndim != 2 or current_state.shape != (target.shape[0], 19):
            raise ValueError("current_state must be [B,19]")
        if not torch.isfinite(current_state).all():
            raise ValueError("current_state contains NaN or Inf")

        safe = target.clone()
        current_action = torch.cat((current_state[:, 3:17], current_state[:, 17:19]), dim=1)
        safe[:, :14] = torch.where(
            torch.isfinite(safe[:, :14]), safe[:, :14], current_action[:, :14]
        )
        safe[:, 14:] = torch.nan_to_num(
            safe[:, 14:],
            nan=POLICY_HAND_OPEN,
            posinf=POLICY_HAND_MAX,
            neginf=POLICY_HAND_MIN,
        ).clamp(POLICY_HAND_MIN, POLICY_HAND_MAX)
        limits = torch.as_tensor(BODY_POSITION_LIMITS_RAD, device=safe.device, dtype=safe.dtype)
        safe[:, :14] = torch.maximum(
            torch.minimum(safe[:, :14], limits[:, 1]), limits[:, 0]
        )

        dt = 1.0 / self.policy_hz
        velocity_limit = self.body_velocity_limit_rad_s
        current = current_action[:, :14]
        if self.previous_target is None or self.previous_target.shape != current.shape:
            previous_target = current
            previous_velocity = torch.zeros_like(current)
        else:
            previous_target = self.previous_target.to(current)
            assert self.previous_velocity is not None
            previous_velocity = self.previous_velocity.to(current)
        desired_velocity = (safe[:, :14] - previous_target) / dt
        acceleration_step = self.body_acceleration_limit_rad_s2 * dt
        velocity = previous_velocity + (desired_velocity - previous_velocity).clamp(
            -acceleration_step,
            acceleration_step,
        )
        velocity = velocity.clamp(-velocity_limit, velocity_limit)
        candidate = previous_target + velocity * dt
        clip_count = int(torch.count_nonzero(candidate != safe[:, :14]).item())
        safe[:, :14] = candidate
        self.previous_target = candidate.detach()
        self.previous_velocity = velocity.detach()

        hand_current = current_action[:, 14:16]
        if (
            self.previous_hand_target is None
            or self.previous_hand_target.shape != hand_current.shape
        ):
            previous_hand_target = hand_current
            previous_hand_velocity = torch.zeros_like(hand_current)
        else:
            previous_hand_target = self.previous_hand_target.to(hand_current)
            assert self.previous_hand_velocity is not None
            previous_hand_velocity = self.previous_hand_velocity.to(hand_current)
        desired_hand_velocity = (safe[:, 14:16] - previous_hand_target) / dt
        hand_acceleration_step = self.hand_acceleration_limit_command_s2 * dt
        hand_velocity = previous_hand_velocity + (
            desired_hand_velocity - previous_hand_velocity
        ).clamp(-hand_acceleration_step, hand_acceleration_step)
        hand_velocity = hand_velocity.clamp(
            -self.hand_velocity_limit_command_s,
            self.hand_velocity_limit_command_s,
        )
        hand_candidate = previous_hand_target + hand_velocity * dt
        clip_count += int(
            torch.count_nonzero(hand_candidate != safe[:, 14:16]).item()
        )
        safe[:, 14:16] = hand_candidate.clamp(
            POLICY_HAND_MIN,
            POLICY_HAND_MAX,
        )
        self.previous_hand_target = safe[:, 14:16].detach()
        self.previous_hand_velocity = hand_velocity.detach()
        return safe, clip_count


def target_safety_from_environment(*, policy_hz: float) -> UpperBodyTargetSafetyFilter:
    """Build the deployable target filter from checkpointed runtime settings."""

    return UpperBodyTargetSafetyFilter(
        policy_hz=policy_hz,
        body_velocity_limit_rad_s=float(
            os.environ.get(
                "FLIP_TABLE_RLPD_BODY_TARGET_VELOCITY_LIMIT_RAD_S",
                str(BODY_TARGET_VELOCITY_LIMIT_RAD_S),
            )
        ),
        body_acceleration_limit_rad_s2=float(
            os.environ.get(
                "FLIP_TABLE_RLPD_BODY_TARGET_ACCELERATION_LIMIT_RAD_S2",
                str(BODY_TARGET_ACCELERATION_LIMIT_RAD_S2),
            )
        ),
        hand_velocity_limit_command_s=float(
            os.environ.get(
                "FLIP_TABLE_RLPD_HAND_TARGET_VELOCITY_LIMIT_COMMAND_S",
                str(HAND_TARGET_VELOCITY_LIMIT_COMMAND_S),
            )
        ),
        hand_acceleration_limit_command_s2=float(
            os.environ.get(
                "FLIP_TABLE_RLPD_HAND_TARGET_ACCELERATION_LIMIT_COMMAND_S2",
                str(HAND_TARGET_ACCELERATION_LIMIT_COMMAND_S2),
            )
        ),
    )


class AbsoluteTargetDelayBuffer:
    """Delay complete deployable targets by an episode-randomized sim-step count."""

    def __init__(self, *, num_envs: int, max_delay_steps: int, device: torch.device | str) -> None:
        if num_envs < 1 or max_delay_steps < 0:
            raise ValueError("delay-buffer environment count must be positive and delay non-negative")
        self.num_envs = int(num_envs)
        self.max_delay_steps = int(max_delay_steps)
        self.device = torch.device(device)
        self.history = torch.zeros(
            self.num_envs,
            self.max_delay_steps + 1,
            16,
            device=self.device,
        )
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.initialized = False

    def reset(self, current_state: torch.Tensor) -> None:
        if current_state.shape == (self.num_envs, 19):
            current = torch.cat((current_state[:, 3:17], current_state[:, 17:19]), dim=1).to(self.device)
        elif current_state.shape == (self.num_envs, 16):
            current = current_state.to(self.device)
        else:
            raise ValueError(f"delay reset state must be [{self.num_envs},19] or [{self.num_envs},16]")
        self.history[:] = current[:, None, :]
        if self.max_delay_steps:
            self.delay_steps = torch.randint(
                0,
                self.max_delay_steps + 1,
                (self.num_envs,),
                device=self.device,
            )
        else:
            self.delay_steps.zero_()
        self.initialized = True

    def apply(self, target: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("delay buffer must be reset before use")
        if target.shape != (self.num_envs, 16):
            raise ValueError(f"delayed target must be [{self.num_envs},16]")
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = target
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.history[env_ids, self.delay_steps]
