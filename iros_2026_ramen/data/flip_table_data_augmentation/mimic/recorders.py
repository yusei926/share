"""Recorder terms for source-compatible synthetic numeric trajectories."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from robofinals.utils.isaac_data_compat import as_torch

from ..fk_audit import G1_BODY_JOINT_ORDER
from .numeric import (
    dex1_joint_position_to_demo,
    normalized_hand_to_demo,
    pose_matrices_to_xyz_euler,
)


FINGER_JOINT_ORDER = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
NUMERIC_SHAPES = {
    "observation.state.ee_state": 12,
    "observation.state.hand_state": 2,
    "observation.state.robot_q_current": 36,
    "action.ee_action": 12,
    "action.hand_cmd": 2,
    "action.robot_q_desired": 36,
}


class ResumableHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Open an existing pinned-run shard instead of truncating it on resume."""

    def create(self, file_path: str, env_name: str | None = None) -> None:
        path = Path(file_path)
        if path.suffix != ".hdf5":
            path = path.with_suffix(".hdf5")
        if not path.exists():
            super().create(str(path), env_name=env_name)
            return

        super().open(str(path), mode="r+")
        existing_env_name = self.get_env_name()
        if existing_env_name not in {None, "", env_name}:
            self.close()
            raise RuntimeError(
                f"existing recorder shard env_name={existing_env_name!r}, expected {env_name!r}"
            )
        self._demo_count = len(self._hdf5_data_group)

    def write_episode(self, episode, demo_id=None, dataset_compression: bool = True) -> None:
        super().write_episode(episode, demo_id=demo_id, dataset_compression=dataset_compression)
        self._demo_count = len(self._hdf5_data_group)


def _joint_ids(robot, names: tuple[str, ...]) -> list[int]:
    ids, resolved = robot.find_joints(list(names), preserve_order=True)
    if tuple(resolved) != names:
        raise RuntimeError(f"V1 joint contract changed: requested={names}, resolved={resolved}")
    return list(ids)


class DatasetNumericRecorder(RecorderTerm):
    """Record exactly the six numeric features in the immutable source schema."""

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self._body_joint_ids = _joint_ids(robot, G1_BODY_JOINT_ORDER)
        self._arm_joint_ids = _joint_ids(robot, G1_BODY_JOINT_ORDER[-14:])
        self._finger_joint_ids = _joint_ids(robot, FINGER_JOINT_ORDER)

    @staticmethod
    def _check(payload: dict[str, torch.Tensor], batch_size: int) -> None:
        if set(payload) != set(NUMERIC_SHAPES):
            raise RuntimeError("numeric recorder emitted an incomplete source feature set")
        for key, width in NUMERIC_SHAPES.items():
            value = payload[key]
            if value.shape != (batch_size, width) or not torch.isfinite(value).all():
                raise RuntimeError(
                    f"numeric recorder {key} must be finite [{batch_size},{width}], "
                    f"got {tuple(value.shape)}"
                )

    def record_post_step(self):
        env = self._env
        robot = env.scene["robot"]
        root_position, root_quaternion_xyzw = env._robot_root_pose_w()
        root_pose = torch.cat((root_position, root_quaternion_xyzw), dim=1)
        joint_position = as_torch(robot.data.joint_pos)
        body_current = joint_position[:, self._body_joint_ids]

        arms_term = env.action_manager.get_term("arms_action")
        solved_arm_target = as_torch(arms_term.processed_actions)
        articulation_target = as_torch(robot.data.joint_pos_target)[:, self._body_joint_ids]
        applied_arm_target = as_torch(robot.data.joint_pos_target)[:, self._arm_joint_ids]
        target_error = torch.max(torch.abs(solved_arm_target - applied_arm_target))
        if not torch.isfinite(target_error) or float(target_error) > 1.0e-6:
            raise RuntimeError(
                "PINK arm target and articulation target diverged: "
                f"max_abs_error={float(target_error):.9g}"
            )
        body_desired = articulation_target

        action = env.action_manager.action
        target_eef = env.action_to_target_eef_pose(action)
        ee_state = torch.cat(
            [pose_matrices_to_xyz_euler(env.get_robot_eef_pose(side)) for side in ("left", "right")],
            dim=1,
        )
        ee_action = torch.cat(
            [pose_matrices_to_xyz_euler(target_eef[side]) for side in ("left", "right")],
            dim=1,
        )

        fingers = joint_position[:, self._finger_joint_ids]
        hand_state = torch.stack(
            (
                dex1_joint_position_to_demo(fingers[:, 0:2]).mean(dim=1),
                dex1_joint_position_to_demo(fingers[:, 2:4]).mean(dim=1),
            ),
            dim=1,
        )
        hand_command = normalized_hand_to_demo(
            action[:, [env.LEFT_GRIPPER_INDEX, env.RIGHT_GRIPPER_INDEX]]
        )
        payload = {
            "observation.state.ee_state": ee_state,
            "observation.state.hand_state": hand_state,
            "observation.state.robot_q_current": torch.cat((root_pose, body_current), dim=1),
            "action.ee_action": ee_action,
            "action.hand_cmd": hand_command,
            "action.robot_q_desired": torch.cat((root_pose, body_desired), dim=1),
        }
        self._check(payload, env.num_envs)
        return "dataset_numeric", payload


@configclass
class DatasetNumericRecorderCfg(RecorderTermCfg):
    class_type: type[DatasetNumericRecorder] = DatasetNumericRecorder


@configclass
class FlipTableRecorderManagerCfg(ActionStateRecorderManagerCfg):
    dataset_file_handler_class_type: type = ResumableHDF5DatasetFileHandler
    record_dataset_numeric = DatasetNumericRecorderCfg()
