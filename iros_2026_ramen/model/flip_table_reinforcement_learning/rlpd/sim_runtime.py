"""Shared simulator-side runtime for Flow BC plus residual RLPD."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch


CAMERAS = ("first_person_camera", "left_hand_camera", "right_hand_camera")
POLICY_INPUTS = (
    "head-left RGB 640x480",
    "left D405 RGB 640x480",
    "right D405 RGB 640x480",
    "19D upper-body joint state",
    "19D Flow Matching base target",
    "previous 19D residual",
)


class FlowPolicyProtocol(Protocol):
    config: Any

    def sample_actions(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor: ...


def camera_batch(env: Any) -> torch.Tensor:
    """Read the three real-deployable RGB cameras without resizing."""

    from robofinals.utils.isaac_data_compat import as_torch

    frames = []
    for name in CAMERAS:
        value = as_torch(env.scene.sensors[name].data.output["rgb"])
        if (
            value.ndim != 4
            or value.shape[0] != env.num_envs
            or tuple(value.shape[1:3]) != (480, 640)
        ):
            raise ValueError(
                f"{name} must provide [B,480,640,C], got {tuple(value.shape)}"
            )
        if value.shape[-1] < 3 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must provide finite RGB channels")
        frames.append(value[..., :3].permute(0, 3, 1, 2))
    return torch.stack(frames, dim=1)


def dataset_joint_state(env: Any) -> torch.Tensor:
    """Return the exact 17 body-radian plus 2 Dex1-command training state."""

    from robofinals_rl.flip_table.mdp.observations import controller_joint_state_raw

    state = controller_joint_state_raw(env).clone()
    state[:, 17:19] = 0.5 * (1.0 - state[:, 17:19]) * 4.5
    return state


@torch.no_grad()
def settle_after_reset(
    gym_env: Any,
    env: Any,
    *,
    steps: int = 2,
    state_reader: Callable[[Any], torch.Tensor] = dataset_joint_state,
    post_step: Callable[[Any, int], None] | None = None,
) -> int:
    """Flush reset-boundary sensor history while holding the measured posture.

    Isaac contact sensors can expose the previous episode's final sample for one
    simulator tick after ``reset``. These discarded hold steps prevent that stale
    sample from becoming a transition or triggering another safety reset.
    """

    if steps < 0:
        raise ValueError("reset settle steps cannot be negative")
    for step in range(steps):
        hold_target = state_reader(env).detach().clone()
        if hold_target.shape != (env.num_envs, 19):
            raise ValueError(
                "reset hold state must have shape "
                f"[{env.num_envs},19], got {tuple(hold_target.shape)}"
            )
        if not torch.isfinite(hold_target).all():
            raise ValueError("reset hold state contains NaN or Inf")
        env._flip_table_rlpd_absolute_target = hold_target
        zero_residual = torch.zeros_like(hold_target)
        _observation, _reward, terminated, truncated, _extras = gym_env.step(
            zero_residual
        )
        if bool(torch.logical_or(terminated, truncated).any()):
            raise RuntimeError("environment ended during reset settle steps")
        if post_step is not None:
            post_step(env, step)
    return steps


class FlowTargetScheduler:
    """Execute a fixed prefix of each Flow action chunk at policy frequency."""

    def __init__(
        self,
        flow: FlowPolicyProtocol,
        *,
        motion_gain: float = 1.0,
        motion_mask: tuple[float, ...] | None = None,
    ) -> None:
        if not 0.0 <= motion_gain <= 1.0:
            raise ValueError("Flow motion gain must be in [0, 1]")
        if motion_mask is not None and (
            len(motion_mask) != 19 or any(value not in {0.0, 1.0} for value in motion_mask)
        ):
            raise ValueError("Flow motion mask must contain exactly 19 binary values")
        self.flow = flow
        self.motion_gain = float(motion_gain)
        self.motion_mask = motion_mask
        self.chunk: torch.Tensor | None = None
        self.index = 0
        self.anchor_offset: torch.Tensor | None = None
        self.raw_anchor_target: torch.Tensor | None = None
        self.command_anchor_target: torch.Tensor | None = None

    @torch.no_grad()
    def anchor_to_target(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Re-anchor future Flow motion to a deployable commanded target."""

        self.chunk = None
        self.index = 0
        self.anchor_offset = None
        self.raw_anchor_target = None
        self.command_anchor_target = None
        raw_target = self.current(images, state)
        if raw_target.shape != state.shape or target.shape != state.shape:
            raise ValueError(
                "Flow target and anchor must match state "
                f"{tuple(state.shape)}, got {tuple(raw_target.shape)} and {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("Flow anchor target contains NaN or Inf")
        self.anchor_offset = target - raw_target
        self.raw_anchor_target = raw_target.clone()
        self.command_anchor_target = target.clone()
        return target.clone()

    @torch.no_grad()
    def anchor_to_state(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Re-anchor future Flow targets to a measured deployable joint state."""

        return self.anchor_to_target(images, state, state)

    @torch.no_grad()
    def current(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.chunk is None or self.index >= self.flow.config.n_action_steps:
            with torch.autocast(
                device_type=state.device.type,
                dtype=torch.bfloat16,
                enabled=state.device.type == "cuda",
            ):
                self.chunk = self.flow.sample_actions(images, state).float()
            self.index = 0
        target = self.chunk[:, self.index]
        if self.anchor_offset is not None:
            assert self.raw_anchor_target is not None
            assert self.command_anchor_target is not None
            motion = target - self.raw_anchor_target
            if self.motion_mask is not None:
                mask = torch.as_tensor(
                    self.motion_mask,
                    dtype=motion.dtype,
                    device=motion.device,
                ).reshape(1, 19)
                motion = motion * mask
            target = self.command_anchor_target + self.motion_gain * motion
        return target

    def advance(self) -> None:
        self.index += 1

    def reset(self) -> None:
        self.chunk = None
        self.index = 0
        self.anchor_offset = None
        self.raw_anchor_target = None
        self.command_anchor_target = None


def flow_control_ready(env: Any) -> torch.Tensor:
    """Return the per-environment handoff state for a deployable teacher prefix."""

    value = getattr(env, "_flip_table_rlpd_flow_ready", None)
    if value is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    ready = torch.as_tensor(value, dtype=torch.bool, device=env.device)
    if ready.shape != (env.num_envs,):
        raise ValueError(f"Flow readiness must be [{env.num_envs}], got {tuple(ready.shape)}")
    return ready


def set_flow_control_ready(env: Any, ready: bool) -> None:
    """Set handoff readiness without using simulator-only task state."""

    value = getattr(env, "_flip_table_rlpd_flow_ready", None)
    if value is None:
        if not ready:
            env._flip_table_rlpd_flow_ready = torch.zeros(
                env.num_envs,
                dtype=torch.bool,
                device=env.device,
            )
        return
    value.fill_(ready)


def last_commanded_target(env: Any) -> torch.Tensor:
    """Read the latest 19-D target emitted by the deployable action adapter."""

    value = getattr(env, "_flip_table_rlpd_last_commanded_target", None)
    if value is None:
        raise RuntimeError("action adapter has not published a commanded target")
    target = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if target.shape != (env.num_envs, 19) or not torch.isfinite(target).all():
        raise ValueError(
            "latest commanded target must be finite and shaped "
            f"[{env.num_envs},19], got {tuple(target.shape)}"
        )
    return target.clone()


def _clone_tensor_tree(value: Any, *, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"scene state contains NaN or Inf at {path}")
        return value.detach().clone()
    if isinstance(value, dict):
        return {
            key: _clone_tensor_tree(child, path=f"{path}.{key}")
            for key, child in value.items()
        }
    raise TypeError(
        f"scene state must contain only dictionaries and tensors; "
        f"found {type(value).__name__} at {path}"
    )


@torch.no_grad()
def capture_relative_scene_state(env: Any) -> dict[str, Any]:
    """Capture a simulator reset state without exposing it to the policy."""

    getter = getattr(env.scene, "get_state", None)
    if getter is None:
        raise RuntimeError("simulator scene does not support state capture")
    state = getter(is_relative=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("simulator returned an empty scene state")
    return _clone_tensor_tree(state, path="scene")


@torch.no_grad()
def restore_relative_scene_state(
    env: Any,
    state: dict[str, Any],
    *,
    episode_step: int,
) -> None:
    """Reset scene entities to a captured curriculum state and controller time."""

    if episode_step < 0:
        raise ValueError("restored episode step cannot be negative")
    reset_to = getattr(env, "reset_to", None)
    if reset_to is None:
        raise RuntimeError("simulator environment does not support state restoration")
    reset_to(state, env_ids=None, is_relative=True)
    episode_length = torch.as_tensor(env.episode_length_buf)
    if episode_length.shape != (env.num_envs,):
        raise ValueError(
            "episode length buffer must have shape "
            f"[{env.num_envs}], got {tuple(episode_length.shape)}"
        )
    episode_length.fill_(episode_step)
