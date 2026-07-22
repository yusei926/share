"""Configure the organizer V1 task for bimanual Isaac Lab Mimic generation."""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.envs.mimic_env_cfg import (
    DataGenConfig,
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from robofinals.core.models.grippers.dex1 import Dex1GripperCfg
from robofinals.core.robots.unitree.g1 import G1PinkActionsCfg
from robofinals_rl.flip_table import mdp

from ..config import EXPECTED_SUBTASKS, PipelineConfig
from .acceptance import strict_flip_table_success


@configclass
class MimicPolicyObservationsCfg(ObservationGroupCfg):
    """Minimal recorder state; no simulator-only teacher state is exposed."""

    joint_pos = ObsTerm(func=isaac_mdp.joint_pos)
    joint_vel = ObsTerm(func=isaac_mdp.joint_vel)
    actions = ObsTerm(func=isaac_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MimicRewardsCfg:
    task_progress = RewTerm(func=mdp.table_flip_progress, weight=1.0)


_INTERPOLATION_STEPS = {
    "pre_grasp": 10,
    "grasp": 2,
    "lift": 3,
    "rotate_180": 3,
    "settle": 3,
    "release": 2,
    "retreat": 3,
}


def promote_to_mimic_env_cfg(env_cfg):
    """Copy an organizer Arena config into its official Mimic config type."""

    if isinstance(env_cfg, MimicEnvCfg):
        return env_cfg
    if not isinstance(env_cfg, IsaacLabArenaManagerBasedRLEnvCfg):
        raise TypeError(
            "organizer environment config must be an Isaac Lab Arena RL config, got "
            f"{type(env_cfg).__name__}"
        )
    promoted = IsaacArenaManagerBasedMimicEnvCfg()
    for name, value in vars(env_cfg).items():
        setattr(promoted, name, value)
    return promoted


def _pink_dex1_actions() -> G1PinkActionsCfg:
    gripper = Dex1GripperCfg()
    actions = G1PinkActionsCfg()
    # Mimic controls the two wrists while the robot root remains fixed.  Keep
    # the organizer's PINK arm term, but omit its locomotion/base command.
    actions.base_action = None
    actions.left_hand_action = gripper.left_hand_action_cfg()["handle"]
    actions.right_hand_action = gripper.right_hand_action_cfg()["handle"]
    return actions


def _subtask_configs(source_demo_count: int) -> dict[str, list[SubTaskConfig]]:
    if source_demo_count <= 0:
        raise ValueError("source_demo_count must be positive")
    nearest_neighbors = min(3, source_demo_count)
    output = {}
    for side in ("left", "right"):
        entries = []
        for index, name in enumerate(EXPECTED_SUBTASKS):
            entries.append(
                SubTaskConfig(
                    object_ref="white_table",
                    subtask_term_signal=(
                        None if index == len(EXPECTED_SUBTASKS) - 1 else f"{side}_{name}_done"
                    ),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": nearest_neighbors},
                    first_subtask_start_offset_range=(0, 0),
                    subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                    action_noise=0.0,
                    num_interpolation_steps=_INTERPOLATION_STEPS[name],
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                    description=name,
                    next_subtask_description=(
                        "" if index == len(EXPECTED_SUBTASKS) - 1 else EXPECTED_SUBTASKS[index + 1]
                    ),
                )
            )
        output[side] = entries
    return output


def _bimanual_constraints(config: PipelineConfig) -> list[SubTaskConstraintConfig]:
    generation = config.raw["generation"]
    return [
        SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left", index), ("right", index)],
            constraint_type=SubTaskConstraintType.COORDINATION,
            coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
            coordination_scheme_pos_noise_scale=float(generation["action_noise_m"]),
            coordination_scheme_rot_noise_scale=float(generation["action_noise_rad"]),
            coordination_synchronize_start=True,
        )
        for index in range(len(EXPECTED_SUBTASKS))
    ]


def configure_mimic_env_cfg(
    env_cfg,
    config: PipelineConfig,
    *,
    source_dataset_path: str,
    generation_path: str,
    generation_num_trials: int,
    source_demo_count: int,
    guarantee_success: bool,
    generation_seed: int,
):
    """Configure organizer V1 PINK IK plus coordinated bimanual Mimic."""

    if generation_num_trials <= 0:
        raise ValueError("generation_num_trials must be positive")
    if generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")
    env_cfg.actions = _pink_dex1_actions()
    env_cfg.observations.policy = MimicPolicyObservationsCfg()
    env_cfg.rewards = MimicRewardsCfg()
    physics_hz = int(config.raw["generation"]["physics_hz"])
    control_hz = int(config.raw["generation"]["mimic_control_hz"])
    env_cfg.sim.dt = 1.0 / physics_hz
    env_cfg.decimation = physics_hz // control_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.episode_length_s = float(config.raw["generation"]["episode_length_s"])
    env_cfg.flip_table_acceptance_config = dict(config.raw["success"])
    env_cfg.terminations.success = DoneTerm(func=strict_flip_table_success)

    datagen = DataGenConfig()
    datagen.name = "team_ramen_flip_table_v1"
    datagen.source_dataset_path = source_dataset_path
    datagen.generation_path = generation_path
    datagen.generation_num_trials = generation_num_trials
    datagen.generation_guarantee = guarantee_success
    datagen.generation_keep_failed = False
    datagen.max_num_failures = max(100, generation_num_trials)
    datagen.seed = generation_seed
    datagen.generation_select_src_per_subtask = True
    datagen.generation_select_src_per_arm = False
    datagen.generation_transform_first_robot_pose = False
    datagen.generation_interpolate_from_last_target_pose = True
    datagen.use_skillgen = False
    datagen.use_navigation_controller = False
    env_cfg.datagen_config = datagen
    env_cfg.subtask_configs = _subtask_configs(source_demo_count)
    env_cfg.task_constraint_configs = _bimanual_constraints(config)
    return env_cfg
