"""Team RAMEN arm-joint adapter for the unmodified organizer G1 WBC.

This module deliberately subclasses the official action term instead of
patching its controller, ONNX policies, observation history, or reset logic.
The high-level action owns arms only; hand terms remain the organizer's native
Dex1 actions and WBC owns floating base, legs and waist.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from robofinals.data import robofinals_DATA_PATH
from robofinals.core.mdp.actions.decoupled_wbc_action import (
    BaseConfig,
    G1DecoupledWBCAction,
    G1WBCUpperbodyController,
    convert_sim_joint_to_wbc_joint,
    get_wbc_policy,
    instantiate_g1_robot_model,
    postprocess_actions,
    prepare_observations,
)


ARM_JOINT_NAMES = (
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


class TeamRamenBalancedWBCAction(G1DecoupledWBCAction):
    """Consume absolute 14-D arm targets and let official WBC balance G1."""

    cfg: "TeamRamenBalancedWBCActionCfg"

    @property
    def action_dim(self) -> int:
        return 14

    def __init__(self, cfg: "TeamRamenBalancedWBCActionCfg", env) -> None:
        # Organizer V1 builds torso_orientation_rpy_cmd with
        # ``array((1, 3)).reshape((num_envs, -1))``. That is equivalent for a
        # single environment but raises before policy construction for every
        # batch size > 1. Keep the organizer source and controller untouched;
        # mirror its pinned constructor here with only that allocation made
        # batch-shaped. The runtime audit verifies the organizer source/ONNX
        # hashes so this compatibility path cannot silently drift.
        ActionTerm.__init__(self, cfg, env)
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)
        self._processed_actions = torch.zeros(
            (self.num_envs, self._num_joints), device=self.device
        )
        self._target_robot_joints_mujoco = None
        self._navigate_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._torso_orientation_rpy_cmd = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._is_navigating = False
        self._navigation_goal_reached = False
        self._default_base_height_cmd = 0.74
        self._wbc_version = "homie_v2"

        config = BaseConfig()
        config.wbc_version = self._wbc_version
        checkpoint_dir = robofinals_DATA_PATH / "ckpts/nv_wbc_v0904/homie_v2"
        config.wbc_model_path = ",".join(
            str(checkpoint_dir / name) for name in ("stand.onnx", "walk.onnx")
        )
        wbc_config = config.load_wbc_yaml()
        waist_location = (
            "lower_and_upper_body" if config.enable_waist else "lower_body"
        )
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)
        self.current_upper_body_pose = self.robot_model.get_initial_upper_body_pose()
        # The organizer factory constructs a single-environment Homie policy
        # (including a single history deque) regardless of ManagerBasedEnv
        # batch size. Keep one untouched policy instance per Isaac environment
        # so histories, gait state and partial resets cannot leak across envs.
        self._wbc_policies = [
            get_wbc_policy(
                "g1", self.robot_model, wbc_config, config.upper_body_joint_speed
            )
            for _ in range(self.num_envs)
        ]
        # Preserve the organizer-facing attribute for diagnostics and callers.
        self.wbc_policy = self._wbc_policies[0]
        self._wbc_goal = {
            "target_upper_body_pose": np.tile(
                self.current_upper_body_pose, (self.num_envs, 1)
            ),
            "navigate_cmd": np.zeros((self.num_envs, 3), dtype=np.float32),
            "base_height_command": np.full(
                (self.num_envs,), self._default_base_height_cmd, dtype=np.float32
            ),
            "toggle_policy_action": False,
            "torso_orientation_rpy_cmd": np.zeros(
                (self.num_envs, 3), dtype=np.float32
            ),
        }
        self.upperbody_controller = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
            control_hands=False,
        )
        arm_ids, arm_names = self._asset.find_joints(
            list(ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(arm_names) != ARM_JOINT_NAMES:
            raise RuntimeError(
                "G1 arm joint order differs from the 14-D policy contract: "
                f"{tuple(arm_names)!r}"
            )
        self._arm_joint_ids = torch.as_tensor(
            arm_ids, dtype=torch.long, device=self.device
        )
        index_to_wbc_name = {
            int(index): str(name) for name, index in self.wbc_g1_joints_order.items()
        }
        wbc_arm_ids = tuple(
            int(index)
            for index in self.robot_model.get_joint_group_indices("upper_body_no_hands")
        )
        wbc_arm_names = tuple(index_to_wbc_name[index] for index in wbc_arm_ids)
        self._wbc_upper_ids = tuple(
            int(index) for index in self.robot_model.get_joint_group_indices("upper_body")
        )
        wbc_upper_names = tuple(
            index_to_wbc_name[index] for index in self._wbc_upper_ids
        )
        if len(wbc_arm_ids) != 14 or set(wbc_arm_names) != set(ARM_JOINT_NAMES):
            raise RuntimeError(
                "organizer WBC upper_body_no_hands must contain exactly the 14 "
                f"arm joints, got {wbc_arm_names!r}"
            )
        if any(name.startswith("waist_") for name in wbc_upper_names):
            raise RuntimeError(
                "organizer WBC upper_body unexpectedly contains waist joints: "
                f"{wbc_upper_names!r}"
            )
        self._requested_arm_target = torch.zeros(
            (self.num_envs, 14), dtype=torch.float32, device=self.device
        )
        self._zero_navigation = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._zero_torso_rpy = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._stand_height = np.full(
            (self.num_envs,), float(cfg.base_height_m), dtype=np.float32
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 14):
            raise ValueError(
                f"balanced WBC arm target must be [{self.num_envs},14], "
                f"got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("balanced WBC arm target contains NaN or Inf")
        self._raw_actions.copy_(actions)
        self._requested_arm_target.copy_(actions)
        self._navigate_cmd.zero_()
        self._torso_orientation_rpy_cmd.zero_()
        self.set_wbc_goal(
            self._zero_navigation,
            self._stand_height,
            self._zero_torso_rpy,
        )
        for env_id, policy in enumerate(self._wbc_policies):
            policy.set_goal(
                {
                    "toggle_policy_action": False,
                    "navigate_cmd": self._zero_navigation[env_id : env_id + 1],
                    "base_height_command": float(self._stand_height[env_id]),
                    "torso_orientation_rpy_cmd": self._zero_torso_rpy[
                        env_id : env_id + 1
                    ],
                }
            )

    def apply_actions(self) -> None:
        # ActionManager has processed every term before apply_actions begins;
        # reading native hand terms here avoids a one-control-tick hand lag.
        left_hand = self._env.action_manager.get_term("left_hand_action")
        right_hand = self._env.action_manager.get_term("right_hand_action")
        sim_targets = torch.zeros(
            (self.num_envs, self._num_joints),
            dtype=self._requested_arm_target.dtype,
            device=self.device,
        )
        sim_targets[:, self._arm_joint_ids] = self._requested_arm_target
        sim_targets[:, left_hand._joint_ids] = left_hand.processed_actions.to(self.device)
        sim_targets[:, right_hand._joint_ids] = right_hand.processed_actions.to(self.device)

        wbc_observation = prepare_observations(
            self.num_envs, self._asset.data, self.wbc_g1_joints_order
        )
        wbc_targets = convert_sim_joint_to_wbc_joint(
            sim_targets,
            self._asset.data.joint_names,
            self.wbc_g1_joints_order,
        )
        processed_rows = []
        for env_id, policy in enumerate(self._wbc_policies):
            observation_row = {
                name: value[env_id : env_id + 1]
                for name, value in wbc_observation.items()
            }
            policy.set_observation(observation_row)
            wbc_action = policy.get_action(
                wbc_targets[env_id : env_id + 1, self._wbc_upper_ids]
            )
            processed_rows.append(
                postprocess_actions(
                    wbc_action,
                    self._asset.data,
                    self.wbc_g1_joints_order,
                    self.device,
                )
            )
        processed = torch.cat(processed_rows, dim=0)
        processed[:, left_hand._joint_ids] = left_hand.processed_actions.to(self.device)
        processed[:, right_hand._joint_ids] = right_hand.processed_actions.to(self.device)
        self._processed_actions = processed
        self._target_robot_joints_mujoco = self._requested_arm_target.detach().clone()
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            if env_ids is None
            else torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            ).reshape(-1)
        )
        self._raw_actions[ids] = 0.0
        self._requested_arm_target[ids] = 0.0
        local_id = torch.zeros(1, dtype=torch.long)
        for env_id in ids.detach().cpu().tolist():
            policy = self._wbc_policies[env_id]
            policy.lower_body_policy.reset(local_id)
            if hasattr(policy.upper_body_policy, "reset"):
                policy.upper_body_policy.reset()


@configclass
class TeamRamenBalancedWBCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TeamRamenBalancedWBCAction
    preserve_order: bool = False
    joint_names: list[str] = [".*"]
    base_height_m: float = 0.74
