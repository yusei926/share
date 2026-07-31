"""Sim-to-Real-compatible observations for flip-table reinforcement learning."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from robofinals.utils.isaac_data_compat import as_torch

from ..common import (
    UPPER_BODY_JOINT_NAMES,
    action_prior_at_steps,
    action_prior_phase,
    action_prior_schedule,
    demo_actions_in_controller_domain,
    dex1_joint_to_command,
    dex1_joint_velocity_to_command,
    load_demo_actions,
    phase_demo_targets,
)


FINGER_JOINT_NAMES = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)


def _randomization_level() -> float:
    value = _env_float("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", 1.0)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"FLIP_TABLE_RL_RANDOMIZATION_LEVEL must be in [0,1], got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    result = float(default if value is None or value == "" else value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _sensor_noise_enabled() -> bool:
    return _env_bool("FLIP_TABLE_RL_ENABLE_SENSOR_NOISE", True)


def _image_geometry_randomization_enabled() -> bool:
    return _env_bool("FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY", True)


def _resolved_joint_ids(robot, names: tuple[str, ...]) -> list[int]:
    joint_ids, resolved = robot.find_joints(list(names), preserve_order=True)
    if tuple(resolved) != names:
        raise RuntimeError(f"unexpected joint order: requested={names}, resolved={resolved}")
    return joint_ids


def controller_joint_state_raw(env) -> torch.Tensor:
    """Return noise-free 17-joint and two-hand state in controller order."""

    robot = env.scene["robot"]
    cache = getattr(env, "_flip_table_rl_joint_ids", None)
    if cache is None:
        cache = (
            _resolved_joint_ids(robot, UPPER_BODY_JOINT_NAMES),
            _resolved_joint_ids(robot, FINGER_JOINT_NAMES),
        )
        env._flip_table_rl_joint_ids = cache
    body_ids, finger_ids = cache
    joint_pos = as_torch(robot.data.joint_pos)
    body = joint_pos[:, body_ids]
    fingers = joint_pos[:, finger_ids]
    hands = torch.stack(
        [dex1_joint_to_command(fingers[:, 0:2]).mean(dim=1), dex1_joint_to_command(fingers[:, 2:4]).mean(dim=1)],
        dim=1,
    )
    return torch.cat([body, hands], dim=1)


def controller_joint_state(env) -> torch.Tensor:
    """Return 17 joint angles plus left/right Dex1 commands in [-1, 1]."""

    state = controller_joint_state_raw(env)
    if _sensor_noise_enabled():
        level = _randomization_level()
        body_std = _env_float("FLIP_TABLE_RL_JOINT_POSITION_NOISE_STD_RAD", 0.0005 + 0.0015 * level)
        hand_std = _env_float("FLIP_TABLE_RL_HAND_STATE_NOISE_STD", 0.002 + 0.008 * level)
        if body_std < 0 or hand_std < 0:
            raise ValueError("joint-position noise standard deviations must be non-negative")
        state = state.clone()
        state[:, :17] += torch.randn_like(state[:, :17]) * body_std
        state[:, 17:19] += torch.randn_like(state[:, 17:19]) * hand_std
    return state


def controller_joint_velocity(env) -> torch.Tensor:
    """Return real-observable upper-body joint velocities in controller order."""

    robot = env.scene["robot"]
    cache = getattr(env, "_flip_table_rl_joint_ids", None)
    if cache is None:
        controller_joint_state_raw(env)
        cache = env._flip_table_rl_joint_ids
    body_ids, finger_ids = cache
    joint_vel = as_torch(robot.data.joint_vel)
    body = joint_vel[:, body_ids]
    finger_vel = joint_vel[:, finger_ids]
    hands = torch.stack(
        [finger_vel[:, 0:2].mean(dim=1), finger_vel[:, 2:4].mean(dim=1)], dim=1
    )
    hands = dex1_joint_velocity_to_command(hands)
    velocity = torch.cat([body, hands], dim=1)
    if _sensor_noise_enabled():
        level = _randomization_level()
        body_std = _env_float("FLIP_TABLE_RL_JOINT_VELOCITY_NOISE_STD_RAD_S", 0.003 + 0.017 * level)
        hand_std = _env_float("FLIP_TABLE_RL_HAND_VELOCITY_NOISE_STD", 0.005 + 0.015 * level)
        if body_std < 0 or hand_std < 0:
            raise ValueError("joint-velocity noise standard deviations must be non-negative")
        velocity = velocity.clone()
        velocity[:, :17] += torch.randn_like(velocity[:, :17]) * body_std
        velocity[:, 17:19] += torch.randn_like(velocity[:, 17:19]) * hand_std
    return velocity


def controller_action_prior(env) -> torch.Tensor:
    """Return the fixed 16-D arm/hand prior available to the real controller."""

    schedule = action_prior_schedule(env)
    return action_prior_at_steps(schedule, as_torch(env.episode_length_buf).long())


def controller_action_prior_phase(env) -> torch.Tensor:
    """Return normalized deployable controller time for the residual prior."""

    schedule = action_prior_schedule(env)
    return action_prior_phase(schedule, as_torch(env.episode_length_buf).long())


class DemoPriorObservation(ManagerTermBase):
    """Expose a real-demo target selected only from the current joint state."""

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        super().__init__(cfg, env)
        path = os.environ.get("FLIP_TABLE_RL_DEMO_ACTION_PATH", "").strip()
        if not path:
            raise ValueError("FLIP_TABLE_RL_DEMO_ACTION_PATH is required")
        self._demo = demo_actions_in_controller_domain(load_demo_actions(path)).to(env.device)
        self._lookahead = int(cfg.params.get("lookahead", 3))
        self._progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._mode = os.environ.get("FLIP_TABLE_RL_PHASE_MODE", "state").strip().lower()
        self._start_index = int(os.environ.get("FLIP_TABLE_RL_DEMO_START_INDEX", "0"))
        self._end_index = int(os.environ.get("FLIP_TABLE_RL_DEMO_END_INDEX", str(self._demo.shape[0] - 1)))
        self._control_dt = float(env.cfg.sim.dt * env.cfg.decimation)
        self._demo_hz = float(os.environ.get("FLIP_TABLE_RL_DEMO_HZ", "30"))
        self._hold_index = int(os.environ.get("FLIP_TABLE_RL_DEMO_HOLD_INDEX", "-1"))
        self._hold_steps = int(os.environ.get("FLIP_TABLE_RL_DEMO_HOLD_STEPS", "0"))
        self._resume_demo_hz = float(
            os.environ.get("FLIP_TABLE_RL_DEMO_RESUME_HZ", str(self._demo_hz))
        )
        env._flip_table_rl_demo_prior_term = self

    def __call__(self, env, lookahead: int = 3) -> torch.Tensor:
        # Demo matching uses the same joint sensors available on G1, without a
        # second synthetic-noise draw that would make observation ordering alter
        # the policy input and evaluation RNG stream.
        current = controller_joint_state_raw(env)
        target, self._progress = phase_demo_targets(
            current[:, 3:17],
            self._demo,
            self._progress,
            env.episode_length_buf,
            mode=self._mode,
            start_index=self._start_index,
            end_index=self._end_index,
            control_dt=self._control_dt,
            demo_hz=self._demo_hz,
            hold_index=self._hold_index,
            hold_steps=self._hold_steps,
            resume_demo_hz=self._resume_demo_hz,
            lookahead=self._lookahead,
        )
        return target

    def reset(self, env_ids=None) -> None:
        self._progress[slice(None) if env_ids is None else env_ids] = 0


class MultiCameraResNetFeatures(ManagerTermBase):
    """Extract one frozen ResNet18 embedding from each real policy camera."""

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        super().__init__(cfg, env)
        from torchvision import models

        self._sensor_names = tuple(cfg.params["sensor_names"])
        expected_sensors = (
            "first_person_camera",
            "left_hand_camera",
            "right_hand_camera",
        )
        if self._sensor_names != expected_sensors:
            raise ValueError(
                f"policy cameras must be exactly {expected_sensors}, got {self._sensor_names}"
            )
        weights_name = os.environ.get("FLIP_TABLE_RL_RESNET_WEIGHTS", "imagenet").lower()
        if weights_name not in {"imagenet", "none"}:
            raise ValueError("FLIP_TABLE_RL_RESNET_WEIGHTS must be 'imagenet' or 'none'")
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if weights_name != "none" else None
        self._encoder = models.resnet18(weights=weights)
        self._encoder.fc = torch.nn.Identity()
        self._encoder.eval().to(env.device)
        self._encoder.requires_grad_(False)
        self._size = int(cfg.params.get("image_size", 224))
        if self._size != 224:
            raise ValueError(f"frozen ResNet18 policy image size must be 224, got {self._size}")
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=env.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=env.device).view(1, 3, 1, 1)
        level = _randomization_level()
        geometry_scale = 1.0 if _image_geometry_randomization_enabled() else 0.0
        self._image_noise_std = _env_float("FLIP_TABLE_RL_IMAGE_NOISE_STD", 0.002 + 0.008 * level)
        self._affine_scale_jitter = _env_float(
            "FLIP_TABLE_RL_IMAGE_SCALE_JITTER", 0.001 + 0.019 * level
        ) * geometry_scale
        self._affine_translation_px = _env_float(
            "FLIP_TABLE_RL_IMAGE_TRANSLATION_JITTER_PX", 0.5 + 3.5 * level
        ) * geometry_scale
        self._affine_rotation_rad = torch.deg2rad(
            torch.tensor(
                _env_float("FLIP_TABLE_RL_IMAGE_ROTATION_JITTER_DEG", 0.05 + 0.45 * level)
                * geometry_scale,
                device=env.device,
            )
        ).item()
        if self._image_noise_std < 0:
            raise ValueError("FLIP_TABLE_RL_IMAGE_NOISE_STD must be non-negative")
        if not 0.0 <= self._affine_scale_jitter < 1.0:
            raise ValueError("FLIP_TABLE_RL_IMAGE_SCALE_JITTER must be in [0,1)")
        if self._affine_translation_px < 0 or self._affine_rotation_rad < 0:
            raise ValueError("image translation and rotation jitter must be non-negative")
        self._affine = torch.zeros(
            (len(self._sensor_names), env.num_envs, 2, 3),
            device=env.device,
        )
        self._last_episode_steps = None

    def _resample_episode_affine(self, env) -> None:
        steps = as_torch(env.episode_length_buf).long()
        if self._last_episode_steps is None:
            reset_ids = torch.arange(env.num_envs, device=env.device)
        else:
            reset_ids = torch.nonzero(steps < self._last_episode_steps, as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            count = len(self._sensor_names) * reset_ids.numel()
            angle = torch.empty(count, device=env.device).uniform_(
                -self._affine_rotation_rad,
                self._affine_rotation_rad,
            )
            scale = torch.empty(count, device=env.device).uniform_(
                1.0 - self._affine_scale_jitter,
                1.0 + self._affine_scale_jitter,
            )
            tx = torch.empty(count, device=env.device).uniform_(
                -2.0 * self._affine_translation_px / 640.0,
                2.0 * self._affine_translation_px / 640.0,
            )
            ty = torch.empty(count, device=env.device).uniform_(
                -2.0 * self._affine_translation_px / 480.0,
                2.0 * self._affine_translation_px / 480.0,
            )
            cosine = torch.cos(angle) * scale
            sine = torch.sin(angle) * scale
            sampled = torch.stack((cosine, -sine, tx, sine, cosine, ty), dim=1).reshape(-1, 2, 3)
            sampled = sampled.reshape(len(self._sensor_names), reset_ids.numel(), 2, 3)
            self._affine[:, reset_ids] = sampled
        self._last_episode_steps = steps.clone()

    def __call__(self, env, sensor_names: tuple[str, ...], image_size: int = 224) -> torch.Tensor:
        if tuple(sensor_names) != self._sensor_names:
            raise ValueError(
                f"camera feature term changed sensor order: configured={self._sensor_names}, called={sensor_names}"
            )
        if int(image_size) != self._size:
            raise ValueError(
                f"camera feature term changed image size: configured={self._size}, called={image_size}"
            )
        images = []
        for name in self._sensor_names:
            sensor = env.scene.sensors[name]
            image = as_torch(sensor.data.output["rgb"])
            if image.ndim != 4 or tuple(image.shape[1:3]) != (480, 640) or image.shape[-1] < 3:
                raise ValueError(f"{name} must provide [B,480,640,C>=3] RGB, got {tuple(image.shape)}")
            image = image[..., :3]
            if not torch.isfinite(image).all():
                raise ValueError(f"{name} contains NaN or Inf")
            images.append(image.permute(0, 3, 1, 2))
        batch = torch.cat(images, dim=0).float()
        minimum = float(batch.min().item())
        maximum = float(batch.max().item())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(f"policy RGB values must be in [0,255], got [{minimum}, {maximum}]")
        if maximum > 1.0:
            batch = batch / 255.0
        batch = F.interpolate(batch, size=(self._size, self._size), mode="bilinear", align_corners=False)
        self._resample_episode_affine(env)
        affine = torch.cat([self._affine[index] for index in range(len(self._sensor_names))], dim=0)
        grid = F.affine_grid(affine, batch.shape, align_corners=False)
        batch = F.grid_sample(batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if _sensor_noise_enabled() and self._image_noise_std > 0.0:
            batch = (batch + torch.randn_like(batch) * self._image_noise_std).clamp(0.0, 1.0)
        batch = (batch - self._mean) / self._std
        device_type = batch.device.type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            features = self._encoder(batch).float()
        num_envs = images[0].shape[0]
        return torch.cat(features.split(num_envs, dim=0), dim=1)
