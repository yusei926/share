from __future__ import annotations

import math

import pytest
import torch

from data.flip_table_data_augmentation.mimic.numeric import (
    dex1_joint_position_to_demo,
    normalized_hand_to_demo,
    pose_matrices_to_xyz_euler,
)
from data.flip_table_data_augmentation.mimic.isaaclab_compat import (
    add_uniform_noise_to_pose,
    get_delta_object_pose,
)


def test_pose_conversion_uses_source_extrinsic_xyz_convention() -> None:
    roll, pitch, yaw = 0.3, -0.4, 0.7
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, :3] = rz @ ry @ rx
    pose[0, :3, 3] = torch.tensor([0.1, -0.2, 0.9])
    actual = pose_matrices_to_xyz_euler(pose)[0]
    expected = torch.tensor([0.1, -0.2, 0.9, roll, pitch, yaw])
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)


def test_dex1_mappings_preserve_real_dataset_scale_and_overshoot() -> None:
    command = normalized_hand_to_demo(torch.tensor([[-1.0, 0.0, 1.0]]))
    torch.testing.assert_close(command, torch.tensor([[4.5, 2.25, 0.0]]))
    measured = dex1_joint_position_to_demo(torch.tensor([[0.0245, -0.02, -0.021]]))
    torch.testing.assert_close(measured[:, :2], torch.tensor([[4.5, 0.0]]), atol=1.0e-6, rtol=0.0)
    assert measured[0, 2] < 0.0


def test_normalized_hand_command_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="within"):
        normalized_hand_to_demo(torch.tensor([[1.01]]))


def test_missing_mimic_delta_pose_maps_source_object_to_current() -> None:
    source = torch.eye(4)
    source[:3, 3] = torch.tensor([0.4, -0.2, 0.7])
    current = torch.eye(4)
    current[:3, 3] = torch.tensor([-0.1, 0.3, 0.9])

    delta = get_delta_object_pose(current, source)

    torch.testing.assert_close(delta @ source, current)


def test_missing_mimic_pose_noise_is_bounded_and_preserves_rotation() -> None:
    torch.manual_seed(7)
    position, rotation = add_uniform_noise_to_pose(
        torch.zeros(3), torch.eye(3), 0.005, 0.02
    )

    assert torch.all(torch.abs(position) <= 0.005)
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1.0e-6, rtol=0.0)
    angle = torch.acos(torch.clamp((torch.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    assert float(angle) <= 0.02 + 1.0e-6
