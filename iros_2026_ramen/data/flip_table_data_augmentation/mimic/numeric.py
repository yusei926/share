"""Tensor conversions shared by Mimic recording and replay export."""

from __future__ import annotations

import torch


DEX1_OPEN_POSITION_M = 0.0245
DEX1_CLOSED_POSITION_M = -0.02
DEMO_HAND_CLOSED = 0.0
DEMO_HAND_OPEN = 4.5


def rotation_matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to SciPy-compatible extrinsic XYZ angles."""

    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError(f"rotation matrix must be [N,3,3], got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("rotation matrix contains NaN or Inf")
    roll = torch.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
    pitch = torch.atan2(
        -matrix[:, 2, 0],
        torch.sqrt(matrix[:, 0, 0].square() + matrix[:, 1, 0].square()),
    )
    yaw = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
    return torch.stack((roll, pitch, yaw), dim=1)


def pose_matrices_to_xyz_euler(pose: torch.Tensor) -> torch.Tensor:
    """Convert homogeneous poses to source-dataset xyz/euler_xyz layout."""

    if pose.ndim != 3 or pose.shape[1:] != (4, 4):
        raise ValueError(f"pose must be [N,4,4], got {tuple(pose.shape)}")
    if not torch.isfinite(pose).all():
        raise ValueError("pose contains NaN or Inf")
    return torch.cat(
        (pose[:, :3, 3], rotation_matrix_to_euler_xyz(pose[:, :3, :3])),
        dim=1,
    )


def normalized_hand_to_demo(command: torch.Tensor) -> torch.Tensor:
    """Map organizer Dex1 commands to the real motor-position convention.

    Organizer V1 uses ``-1=open`` and ``+1=closed``. Unitree's real Dex1
    motor coordinate increases while the jaws open, so the source dataset uses
    ``0=closed`` and approximately ``4.5=open``.
    """

    if not torch.isfinite(command).all() or torch.any((command < -1.0) | (command > 1.0)):
        raise ValueError("normalized Dex1 command must be finite and within [-1,1]")
    open_fraction = 0.5 * (1.0 - command)
    return DEMO_HAND_CLOSED + open_fraction * (DEMO_HAND_OPEN - DEMO_HAND_CLOSED)


def dex1_joint_position_to_demo(position: torch.Tensor) -> torch.Tensor:
    """Map measured Dex1 prismatic position to the real hand-state scale.

    The measured state is intentionally not clipped: the real dataset also
    contains small tracking overshoot outside the nominal [0,4.5] command range.
    """

    if not torch.isfinite(position).all():
        raise ValueError("Dex1 joint position contains NaN or Inf")
    open_fraction = (position - DEX1_CLOSED_POSITION_M) / (
        DEX1_OPEN_POSITION_M - DEX1_CLOSED_POSITION_M
    )
    return DEMO_HAND_CLOSED + open_fraction * (DEMO_HAND_OPEN - DEMO_HAND_CLOSED)
