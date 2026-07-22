"""RoboFinals RL registration for real-demo residual flip-table PPO."""

from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robofinals.core.rl.base import LwRL, RlBasePolicyObservationCfg
from robofinals.core.robots.unitree.g1 import UnitreeG1GripperControllerDecoupledWBCEnvCfg
from robofinals.utils.decorators import rl_on
from robofinals_tasks.local_auto_tasks.assemble_table_task import AssembleTableTask

from . import mdp
from .actions import DemoResidualDex1ActionCfg, DemoResidualJointPositionActionCfg
from .common import DEFAULT_STAGE, JOINT_POSITION_LIMITS_RAD, RESIDUAL_SCALE_RAD


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = float(default if raw is None or raw == "" else raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


DEX1_FINGER_CONTACT_SENSOR_SPECS = (
    ("left_gripper_contact", "left_dex1_finger_link_1"),
    ("left_gripper_contact_2", "left_dex1_finger_link_2"),
    ("right_gripper_contact", "right_dex1_finger_link_1"),
    ("right_gripper_contact_2", "right_dex1_finger_link_2"),
)
WHITE_LEG_CONTACT_SENSOR_SPECS = (
    ("white_leg_contact_0", "Leg001/Leg001"),
    ("white_leg_contact_1", "Leg001_01/Leg001"),
    ("white_leg_contact_2", "Leg001_03/Leg001"),
    ("white_leg_contact_3", "Leg001_06/Leg001"),
)


def _configure_flip_table_contact_sensors(embodiment) -> None:
    """Install GPU-supported body-level contact attribution.

    The four finger sensors remain unfiltered for safety limits. Each leg-side
    sensor is filtered against four Dex1 finger bodies; reversing and negating
    those forces yields an exact ``finger x leg`` matrix without unsupported
    collision-shape filters.
    """

    scene = embodiment.scene_config
    finger_paths = [
        f"{{ENV_REGEX_NS}}/Robot/{body_name}"
        for _field_name, body_name in DEX1_FINGER_CONTACT_SENSOR_SPECS
    ]
    for field_name, body_name in DEX1_FINGER_CONTACT_SENSOR_SPECS:
        setattr(
            scene,
            field_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=[],
            ),
        )
    for field_name, body_path in WHITE_LEG_CONTACT_SENSOR_SPECS:
        setattr(
            scene,
            field_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Scene/{body_path}",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=list(finger_paths),
            ),
        )


WAIST_RESIDUAL_SCALE = {
    name: RESIDUAL_SCALE_RAD[name]
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
}
WAIST_POSITION_LIMITS = {
    name: JOINT_POSITION_LIMITS_RAD[name]
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
}
LEFT_ARM_RESIDUAL_SCALE = {
    pattern: value
    for pattern, value in RESIDUAL_SCALE_RAD.items()
    if pattern.startswith(".*_")
}
RIGHT_ARM_RESIDUAL_SCALE = dict(LEFT_ARM_RESIDUAL_SCALE)
LEFT_ARM_POSITION_LIMITS = {
    pattern: value
    for pattern, value in JOINT_POSITION_LIMITS_RAD.items()
    if pattern.startswith(".*_") or pattern == "left_shoulder_roll_joint"
}
RIGHT_ARM_POSITION_LIMITS = {
    pattern: value
    for pattern, value in JOINT_POSITION_LIMITS_RAD.items()
    if pattern.startswith(".*_") or pattern == "right_shoulder_roll_joint"
}

@configclass
class FlipTableResidualActionsCfg:
    waist_action: DemoResidualJointPositionActionCfg = DemoResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        scale=WAIST_RESIDUAL_SCALE,
        offset=0.0,
        clip=WAIST_POSITION_LIMITS,
        use_default_offset=False,
        preserve_order=True,
        sample_shared_delay=True,
    )
    left_arm_action: DemoResidualJointPositionActionCfg = DemoResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        scale=LEFT_ARM_RESIDUAL_SCALE,
        offset=0.0,
        clip=LEFT_ARM_POSITION_LIMITS,
        use_default_offset=False,
        preserve_order=True,
    )
    right_arm_action: DemoResidualJointPositionActionCfg = DemoResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=RIGHT_ARM_RESIDUAL_SCALE,
        offset=0.0,
        clip=RIGHT_ARM_POSITION_LIMITS,
        use_default_offset=False,
        preserve_order=True,
    )
    left_hand_action: DemoResidualDex1ActionCfg = DemoResidualDex1ActionCfg(
        asset_name="robot",
        joint_names=["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
        demo_column=17,
    )
    right_hand_action: DemoResidualDex1ActionCfg = DemoResidualDex1ActionCfg(
        asset_name="robot",
        joint_names=["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
        demo_column=18,
    )


@configclass
class FlipTableStatePolicyObsCfg(RlBasePolicyObservationCfg):
    joint_pos = ObsTerm(func=mdp.controller_joint_state, noise=None)
    joint_vel = ObsTerm(func=mdp.controller_joint_velocity, noise=None)
    demo_prior = ObsTerm(func=mdp.DemoPriorObservation, params={"lookahead": 3})
    action_prior = ObsTerm(func=mdp.controller_action_prior, noise=None)
    action_prior_phase = ObsTerm(func=mdp.controller_action_prior_phase, noise=None)
    actions = ObsTerm(func=isaac_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class FlipTableVisualPolicyObsCfg(FlipTableStatePolicyObsCfg):
    camera_features = ObsTerm(
        func=mdp.MultiCameraResNetFeatures,
        params={
            "sensor_names": (
                "first_person_camera",
                "left_hand_camera",
                "right_hand_camera",
            ),
            "image_size": 224,
        },
    )


def _stage_weights() -> dict[str, float]:
    stage = os.environ.get("FLIP_TABLE_RL_STAGE", DEFAULT_STAGE).strip().lower()
    values = {
        "reach": dict(primary_reach=30.0, primary_contact=2.0, primary_grasp=0.0, reach=0.0, gate=0.0, sync=0.0, contact=0.0, grasp=0.0, lift=0.0, grasped_lift=0.0, bimanual_grasped_lift=0.0, rotate=0.0, success=100.0),
        "contact": dict(primary_reach=14.0, primary_contact=30.0, primary_grasp=5.0, reach=0.0, gate=0.0, sync=0.0, contact=0.0, grasp=0.0, lift=1.0, grasped_lift=0.0, bimanual_grasped_lift=0.0, rotate=0.0, success=120.0),
        "grasp": dict(primary_reach=8.0, primary_contact=20.0, primary_grasp=35.0, reach=0.0, gate=0.0, sync=0.0, contact=0.0, grasp=0.0, lift=3.0, grasped_lift=0.0, bimanual_grasped_lift=0.0, rotate=1.0, success=140.0),
        "sequential_lift": dict(primary_reach=2.0, primary_contact=8.0, primary_grasp=30.0, reach=0.0, gate=0.0, sync=0.0, contact=0.0, grasp=0.0, lift=10.0, grasped_lift=80.0, bimanual_grasped_lift=0.0, rotate=5.0, success=180.0),
        "lift": dict(primary_reach=5.0, primary_contact=12.0, primary_grasp=14.0, reach=2.0, gate=4.0, sync=1.0, contact=8.0, grasp=8.0, lift=30.0, grasped_lift=20.0, bimanual_grasped_lift=20.0, rotate=6.0, success=160.0),
        "rotate": dict(primary_reach=3.0, primary_contact=8.0, primary_grasp=8.0, reach=2.0, gate=3.0, sync=1.0, contact=6.0, grasp=6.0, lift=15.0, grasped_lift=12.0, bimanual_grasped_lift=0.0, rotate=40.0, success=180.0),
        "flip": dict(primary_reach=2.0, primary_contact=5.0, primary_grasp=5.0, reach=1.0, gate=2.0, sync=1.0, contact=5.0, grasp=5.0, lift=10.0, grasped_lift=8.0, bimanual_grasped_lift=0.0, rotate=55.0, success=220.0),
        "stabilize": dict(primary_reach=1.0, primary_contact=2.0, primary_grasp=2.0, reach=1.0, gate=1.0, sync=0.5, contact=2.0, grasp=2.0, lift=5.0, grasped_lift=2.0, bimanual_grasped_lift=0.0, rotate=35.0, success=250.0),
        "full": dict(primary_reach=3.0, primary_contact=6.0, primary_grasp=6.0, reach=1.0, gate=2.0, sync=1.0, contact=5.0, grasp=5.0, lift=12.0, grasped_lift=8.0, bimanual_grasped_lift=0.0, rotate=45.0, success=250.0),
    }
    if stage not in values:
        raise ValueError(f"unknown FLIP_TABLE_RL_STAGE={stage!r}; expected one of {tuple(values)}")
    safety_weights = {
        "reach": (-25.0, -20.0, -60.0),
        "contact": (-12.0, -20.0, -60.0),
        "grasp": (-4.0, -20.0, -60.0),
        "sequential_lift": (0.0, -15.0, -50.0),
        "lift": (0.0, -15.0, -50.0),
        "rotate": (0.0, -15.0, -50.0),
        "flip": (0.0, -15.0, -50.0),
        "stabilize": (0.0, -15.0, -50.0),
        "full": (0.0, -15.0, -50.0),
    }
    (
        values[stage]["disturbance"],
        values[stage]["force_margin"],
        values[stage]["unsafe_force"],
    ) = safety_weights[stage]
    return values[stage]


STAGE_WEIGHTS = _stage_weights()
REACH_GATE_THRESHOLD_M = 0.10 if os.environ.get("FLIP_TABLE_RL_STAGE", "reach").strip().lower() == "reach" else 0.08
STAGE_DISTURBANCE_LIMITS = {
    "reach": {"allowed_lift_m": 0.01, "allowed_flip_progress": 0.03},
    "contact": {"allowed_lift_m": 0.02, "allowed_flip_progress": 0.08},
    "grasp": {"allowed_lift_m": 0.04, "allowed_flip_progress": 0.25},
}.get(os.environ.get("FLIP_TABLE_RL_STAGE", "reach").strip().lower(), {})


def _mass_scale_range() -> tuple[float, float]:
    # The organizer USD totals 1.596 kg. Keep that measured task property fixed
    # across RL and evaluation while other visual/contact dynamics are varied.
    return 1.0, 1.0


def _mass_randomization_term(asset_name: str) -> EventTerm:
    return EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "mass_distribution_params": _mass_scale_range(),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )


@configclass
class FlipTableRandomizationEventsCfg:
    table_mass = _mass_randomization_term("Table001_Table001_01")
    leg_0_mass = _mass_randomization_term("Leg001_Leg001")
    leg_1_mass = _mass_randomization_term("Leg001_01_Leg001")
    leg_2_mass = _mass_randomization_term("Leg001_03_Leg001")
    leg_3_mass = _mass_randomization_term("Leg001_06_Leg001")


@configclass
class FlipTableRewardsCfg:
    primary_hand_reach = RewTerm(
        func=mdp.single_hand_reach,
        weight=STAGE_WEIGHTS["primary_reach"],
        params={"side": "right", "std": 0.12},
    )
    primary_hand_contact = RewTerm(
        func=mdp.single_hand_contact,
        weight=STAGE_WEIGHTS["primary_contact"],
        params={"side": "right"},
    )
    primary_hand_grasp = RewTerm(
        func=mdp.single_hand_grasp,
        weight=STAGE_WEIGHTS["primary_grasp"],
        params={"side": "right"},
    )
    bimanual_reach = RewTerm(
        func=mdp.bimanual_reach,
        weight=STAGE_WEIGHTS["reach"],
        params={"std": 0.18},
    )
    bimanual_reach_gate = RewTerm(
        func=mdp.bimanual_reach_gate,
        weight=STAGE_WEIGHTS["gate"],
        params={"threshold": REACH_GATE_THRESHOLD_M, "margin": 0.05},
    )
    bimanual_reach_synchrony = RewTerm(
        func=mdp.bimanual_reach_synchrony,
        weight=STAGE_WEIGHTS["sync"],
        params={"threshold": REACH_GATE_THRESHOLD_M, "margin": 0.05, "balance_std": 0.03},
    )
    bimanual_contact = RewTerm(func=mdp.bimanual_contact, weight=STAGE_WEIGHTS["contact"])
    bimanual_grasp = RewTerm(func=mdp.bimanual_grasp, weight=STAGE_WEIGHTS["grasp"])
    table_lift = RewTerm(func=mdp.table_lift_progress, weight=STAGE_WEIGHTS["lift"])
    grasped_table_lift = RewTerm(
        func=mdp.grasped_table_lift,
        weight=STAGE_WEIGHTS["grasped_lift"],
        params={"side": "right"},
    )
    bimanual_grasped_table_lift = RewTerm(
        func=mdp.bimanual_grasped_table_lift,
        weight=STAGE_WEIGHTS["bimanual_grasped_lift"],
    )
    table_rotation = RewTerm(func=mdp.table_flip_progress, weight=STAGE_WEIGHTS["rotate"])
    table_disturbance = RewTerm(
        func=mdp.table_disturbance_penalty,
        weight=STAGE_WEIGHTS["disturbance"],
        params=STAGE_DISTURBANCE_LIMITS,
    )
    unsafe_finger_force = RewTerm(
        func=mdp.unsafe_finger_force_penalty,
        weight=STAGE_WEIGHTS["unsafe_force"],
    )
    finger_force_margin = RewTerm(
        func=mdp.finger_force_margin_penalty,
        weight=STAGE_WEIGHTS["force_margin"],
    )
    stage_success = RewTerm(func=mdp.stage_success_bonus, weight=STAGE_WEIGHTS["success"])
    demo_residual = RewTerm(func=mdp.demo_residual_l2, weight=-0.12)
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FlipTableCurriculumCfg:
    """The runner advances explicit stages after measured success gates."""

    pass


@rl_on(task=AssembleTableTask)
@rl_on(embodiment=UnitreeG1GripperControllerDecoupledWBCEnvCfg)
class FlipTableResidualStateRL(LwRL):
    def __init__(self):
        super().__init__()
        self.rewards_cfg = FlipTableRewardsCfg()
        self.events_cfg = FlipTableRandomizationEventsCfg()
        self.curriculum_cfg = FlipTableCurriculumCfg()
        self.policy_observation_cfg = FlipTableStatePolicyObsCfg()

    def setup_env_config(self, orchestrator):
        action_cfg = FlipTableResidualActionsCfg()
        embodiment = orchestrator.embodiment
        _configure_flip_table_contact_sensors(embodiment)
        left_hand_cfg = embodiment.gripper_cfg.left_hand_action_cfg()[embodiment.hand_action_mode]
        right_hand_cfg = embodiment.gripper_cfg.right_hand_action_cfg()[embodiment.hand_action_mode]
        action_cfg.left_hand_action.post_process_fn = left_hand_cfg.post_process_fn
        action_cfg.right_hand_action.post_process_fn = right_hand_cfg.post_process_fn
        embodiment.action_config = action_cfg
        super().setup_env_config(orchestrator)
        orchestrator.task.termination_cfg.success = DoneTerm(func=mdp.table_stage_success)

    def modify_env_cfg(self, env_cfg):
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.sim.dt = _env_float("FLIP_TABLE_RL_SIM_DT_S", 0.005)
        control_hz = _env_float("FLIP_TABLE_RL_CONTROL_HZ", 20.0)
        if env_cfg.sim.dt <= 0.0 or control_hz <= 0.0:
            raise ValueError("FLIP_TABLE_RL_SIM_DT_S and FLIP_TABLE_RL_CONTROL_HZ must be positive")
        requested_decimation = 1.0 / (env_cfg.sim.dt * control_hz)
        env_cfg.decimation = int(round(requested_decimation))
        if env_cfg.decimation < 1 or not math.isclose(
            requested_decimation,
            float(env_cfg.decimation),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "FLIP_TABLE_RL_SIM_DT_S and FLIP_TABLE_RL_CONTROL_HZ must produce "
                "an integer simulator decimation"
            )
        env_cfg.sim.render_interval = env_cfg.decimation
        # The organizer default is 100 m, which needlessly loses float32
        # contact precision in parallel environments. A 20 m cell still clears
        # the randomized room's maximum 15 m floor footprint.
        env_spacing_m = _env_float("FLIP_TABLE_RL_ENV_SPACING_M", 20.0)
        if env_spacing_m <= 0.0:
            raise ValueError("FLIP_TABLE_RL_ENV_SPACING_M must be positive")
        env_cfg.scene.env_spacing = env_spacing_m
        # The assembled white table uses four fixed joints. PhysX reports those
        # joints as unsupported by its replication path, so clone complete USD
        # physics for every environment instead.
        env_cfg.scene.replicate_physics = _env_bool("FLIP_TABLE_RL_REPLICATE_PHYSICS", False)
        if not env_cfg.scene.replicate_physics:
            env_cfg.scene.clone_in_fabric = False
        env_cfg.episode_length_s = _env_float("FLIP_TABLE_RL_EPISODE_SECONDS", 32.0)
        if env_cfg.episode_length_s <= 0:
            raise ValueError("FLIP_TABLE_RL_EPISODE_SECONDS must be positive")
        return env_cfg


class FlipTableResidualVisualRL(FlipTableResidualStateRL):
    def __init__(self):
        super().__init__()
        self.policy_observation_cfg = FlipTableVisualPolicyObsCfg()
