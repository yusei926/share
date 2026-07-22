"""V1 G1+Dex1 multi-EEF environment adapter for Isaac Lab Mimic."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from robofinals.utils.isaac_data_compat import as_torch

from .acceptance import candidate_report


class FlipTableMimicEnv(ManagerBasedRLMimicEnv):
    """Expose source-compatible EEFs through the organizer V1 PINK controller."""

    ACTION_DIM = 16
    LEFT_WRIST_SLICE = slice(0, 7)
    RIGHT_WRIST_SLICE = slice(7, 14)
    LEFT_GRIPPER_INDEX = 14
    RIGHT_GRIPPER_INDEX = 15
    ACTION_TERM_CONTRACT = (
        ("arms_action", 14),
        ("left_hand_action", 1),
        ("right_hand_action", 1),
    )
    TOOL_OFFSET_M = (0.05, 0.0, 0.0)
    EEF_BODY_NAMES = {
        "left": "left_wrist_yaw_link",
        "right": "right_wrist_yaw_link",
    }

    @staticmethod
    def _single_pose(value: torch.Tensor, label: str) -> torch.Tensor:
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (4, 4) or not torch.isfinite(value).all():
            raise ValueError(f"{label} must be one finite [4,4] pose")
        return value

    @classmethod
    def _tool_offset(cls, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(cls.TOOL_OFFSET_M)

    @classmethod
    def _eef_pose_to_wrist_action(cls, pose: torch.Tensor) -> torch.Tensor:
        pose = cls._single_pose(pose, "target EEF pose")
        eef_position, eef_rotation = PoseUtils.unmake_pose(pose)
        wrist_position = eef_position - eef_rotation @ cls._tool_offset(eef_position)
        wrist_quaternion_xyzw = PoseUtils.quat_from_matrix(eef_rotation)
        wrist_quaternion_wxyz = wrist_quaternion_xyzw[[3, 0, 1, 2]]
        return torch.cat((wrist_position, wrist_quaternion_wxyz), dim=0)

    @classmethod
    def _wrist_action_to_eef_pose(cls, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != 7 or not torch.isfinite(action).all():
            raise ValueError("wrist action must be finite [N,7]")
        wrist_position = action[:, :3]
        wrist_quaternion_wxyz = action[:, 3:7]
        wrist_quaternion_xyzw = wrist_quaternion_wxyz[:, (1, 2, 3, 0)]
        wrist_rotation = PoseUtils.matrix_from_quat(wrist_quaternion_xyzw)
        eef_position = wrist_position + torch.matmul(
            wrist_rotation, cls._tool_offset(wrist_position)
        )
        return PoseUtils.make_pose(eef_position, wrist_rotation)

    def _robot_root_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        for field in ("root_pose_w", "root_link_pose_w"):
            if hasattr(robot.data, field):
                pose = as_torch(getattr(robot.data, field))
                if pose.ndim == 2 and pose.shape[1] >= 7:
                    return pose[:, :3], pose[:, 3:7]
        raise RuntimeError("robot articulation exposes no root pose")

    def _wrist_pose_w(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        if side not in self.EEF_BODY_NAMES:
            raise ValueError(f"unknown EEF {side!r}; expected {tuple(self.EEF_BODY_NAMES)}")
        robot = self.scene["robot"]
        body_name = self.EEF_BODY_NAMES[side]
        try:
            body_index = robot.data.body_names.index(body_name)
        except ValueError as exc:
            raise RuntimeError(f"robot is missing {body_name}") from exc
        for field in ("body_link_pose_w", "body_pose_w"):
            if hasattr(robot.data, field):
                poses = as_torch(getattr(robot.data, field))
                if poses.ndim == 3 and poses.shape[2] >= 7:
                    return poses[:, body_index, :3], poses[:, body_index, 3:7]
        raise RuntimeError("robot articulation exposes no body pose tensor")

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        root_position, root_quaternion = self._robot_root_pose_w()
        wrist_position, wrist_quaternion = self._wrist_pose_w(eef_name)
        wrist_root_position, wrist_root_quaternion = PoseUtils.subtract_frame_transforms(
            root_position, root_quaternion, wrist_position, wrist_quaternion
        )
        tool_position = wrist_root_position.new_tensor(self.TOOL_OFFSET_M).expand_as(
            wrist_root_position
        )
        tool_quaternion = wrist_root_quaternion.new_tensor((0.0, 0.0, 0.0, 1.0)).expand_as(
            wrist_root_quaternion
        )
        eef_position, eef_quaternion = PoseUtils.combine_frame_transforms(
            wrist_root_position,
            wrist_root_quaternion,
            tool_position,
            tool_quaternion,
        )
        if env_ids is not None:
            eef_position = eef_position[env_ids]
            eef_quaternion = eef_quaternion[env_ids]
        return PoseUtils.make_pose(eef_position, PoseUtils.matrix_from_quat(eef_quaternion))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        del env_id
        if tuple(target_eef_pose_dict) != ("left", "right"):
            raise ValueError("target EEF poses must contain ordered left and right entries")
        if tuple(gripper_action_dict) != ("left", "right"):
            raise ValueError("gripper actions must contain ordered left and right entries")
        if action_noise_dict is not None:
            for side, value in action_noise_dict.items():
                noise = torch.as_tensor(value)
                if torch.any(noise != 0):
                    raise ValueError(
                        f"independent {side} action noise violates the bimanual constraint; "
                        "use the shared coordination transform"
                    )
        left = self._eef_pose_to_wrist_action(target_eef_pose_dict["left"])
        right = self._eef_pose_to_wrist_action(target_eef_pose_dict["right"])
        grippers = []
        for side in ("left", "right"):
            gripper = torch.as_tensor(
                gripper_action_dict[side], dtype=left.dtype, device=left.device
            ).reshape(-1)
            if gripper.numel() != 1 or not torch.isfinite(gripper).all():
                raise ValueError(f"{side} Dex1 action must contain one finite scalar")
            if torch.any((gripper < -1.0) | (gripper > 1.0)):
                raise ValueError(f"{side} Dex1 action must be in [-1,1]")
            grippers.append(gripper)
        # The PINK term consumes both wrists first; the two Dex1 terms follow.
        action = torch.cat((left, right, *grippers), dim=0)
        if action.shape != (self.ACTION_DIM,):
            raise RuntimeError(f"constructed action has shape {tuple(action.shape)}")
        return action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if action.ndim != 2 or action.shape[1] != self.ACTION_DIM:
            raise ValueError(f"Mimic action must be [N,{self.ACTION_DIM}]")
        if not torch.isfinite(action).all():
            raise ValueError("Mimic action contains NaN or Inf")
        return {
            "left": self._wrist_action_to_eef_pose(action[:, self.LEFT_WRIST_SLICE]),
            "right": self._wrist_action_to_eef_pose(action[:, self.RIGHT_WRIST_SLICE]),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            actions.ndim not in (2, 3)
            or actions.shape[-1] != self.ACTION_DIM
            or not torch.isfinite(actions).all()
        ):
            raise ValueError(
                f"Mimic actions must be finite [T,{self.ACTION_DIM}] or "
                f"[N,T,{self.ACTION_DIM}]"
            )
        return {
            "left": actions[
                ..., self.LEFT_GRIPPER_INDEX : self.LEFT_GRIPPER_INDEX + 1
            ],
            "right": actions[
                ..., self.RIGHT_GRIPPER_INDEX : self.RIGHT_GRIPPER_INDEX + 1
            ],
        }

    def validate_runtime_action_contract(self) -> None:
        """Fail before generation if V1 changes the action term ordering."""

        manager = self.action_manager
        actual = tuple(zip(manager.active_terms, manager.action_term_dim, strict=True))
        if actual != self.ACTION_TERM_CONTRACT:
            raise RuntimeError(
                "V1 action contract changed: "
                f"expected {self.ACTION_TERM_CONTRACT}, got {actual}"
            )
        if manager.total_action_dim != self.ACTION_DIM:
            raise RuntimeError(
                f"V1 action dimension must be {self.ACTION_DIM}, got {manager.total_action_dim}"
            )

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        task = self.cfg.isaaclab_arena_env.task
        table_position, table_quaternion_xyzw = task._table_body_pose(self)
        if table_position is None or table_quaternion_xyzw is None:
            raise RuntimeError("V1 task exposes no white-table body pose")
        root_position, root_quaternion = self._robot_root_pose_w()
        table_root_position, table_root_quaternion = PoseUtils.subtract_frame_transforms(
            root_position,
            root_quaternion,
            table_position,
            table_quaternion_xyzw,
        )
        if env_ids is not None:
            table_root_position = table_root_position[env_ids]
            table_root_quaternion = table_root_quaternion[env_ids]
        return {
            "white_table": PoseUtils.make_pose(
                table_root_position,
                PoseUtils.matrix_from_quat(table_root_quaternion),
            )
        }

    def get_candidate_acceptance_report(self, env_id: int) -> dict:
        """Expose the strict post-rollout decision to the audited generator."""

        return candidate_report(self, env_id)
