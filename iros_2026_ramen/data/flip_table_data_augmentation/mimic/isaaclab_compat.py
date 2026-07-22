"""Compatibility for an Isaac Lab Mimic symbol missing from the pinned V1 image."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as PoseUtils


def get_delta_object_pose(
    current_object_pose: torch.Tensor,
    source_object_pose: torch.Tensor,
) -> torch.Tensor:
    """Return the transform that maps a source object pose to the current pose."""

    current = torch.as_tensor(current_object_pose)
    source = torch.as_tensor(source_object_pose)
    if (
        current.shape != source.shape
        or current.shape[-2:] != (4, 4)
        or not torch.isfinite(current).all()
        or not torch.isfinite(source).all()
    ):
        raise ValueError("object poses must be finite matching [...,4,4] tensors")
    return torch.matmul(current, PoseUtils.pose_inv(source))


def add_uniform_noise_to_pose(
    position: torch.Tensor,
    rotation: torch.Tensor,
    position_scale: float,
    rotation_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply bounded uniform translation and axis-angle noise to one pose."""

    position = torch.as_tensor(position)
    rotation = torch.as_tensor(rotation)
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not torch.isfinite(position).all()
        or not torch.isfinite(rotation).all()
        or not math.isfinite(position_scale)
        or not math.isfinite(rotation_scale)
        or position_scale < 0.0
        or rotation_scale < 0.0
    ):
        raise ValueError(
            "pose noise requires a finite [3] position, [3,3] rotation, "
            "and non-negative scales"
        )
    position_noise = torch.empty_like(position).uniform_(
        -position_scale, position_scale
    )
    axis = torch.randn_like(position)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-12)
    angle = torch.empty((), dtype=position.dtype, device=position.device).uniform_(
        -rotation_scale, rotation_scale
    )
    noise_rotation = PoseUtils.matrix_from_quat(
        PoseUtils.quat_from_angle_axis(angle, axis)
    )
    return position + position_noise, noise_rotation @ rotation


def install_missing_mimic_pose_helpers() -> tuple[str, ...]:
    """Install pose helpers referenced but absent in the pinned V1 Isaac Lab."""

    installed = []
    if not hasattr(PoseUtils, "get_delta_object_pose"):
        PoseUtils.get_delta_object_pose = get_delta_object_pose
        installed.append("get_delta_object_pose")
    if not hasattr(PoseUtils, "add_uniform_noise_to_pose"):
        PoseUtils.add_uniform_noise_to_pose = add_uniform_noise_to_pose
        installed.append("add_uniform_noise_to_pose")
    return tuple(installed)
