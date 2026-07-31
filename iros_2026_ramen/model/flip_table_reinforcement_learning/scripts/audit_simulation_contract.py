#!/usr/bin/env python3
"""Audit the flip-table simulator independently of a learned policy.

The report separates simulator/control defects from policy quality. Simulator
object state is used only for this offline diagnostic and is never exposed to
an actor, critic, planner, or deployment-time branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher
from robofinals.utils.config_loader import config_loader, merge_task_yaml_with_cli


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_config", default="flip_table_rl")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sim-control-hz", type=float, default=50.0)
parser.add_argument("--equilibrium-steps", type=int, default=250)
parser.add_argument(
    "--stability-steps",
    type=int,
    default=3000,
    help="stand-only WBC audit steps (3000 at 50 Hz = 60 s)",
)
parser.add_argument("--joint-response-steps", type=int, default=100)
parser.add_argument("--wrench-steps", type=int, default=8)
parser.add_argument("--upward-force-n", type=float, default=30.0)
parser.add_argument("--torque-nm", type=float, default=0.5)
parser.add_argument("--friction-steps", type=int, default=20)
parser.add_argument("--friction-force-n", type=float, default=5.0)
parser.add_argument("--randomized-reset-trials", type=int, default=16)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("the contract audit currently requires --num-envs=1")
if args_cli.sim_control_hz <= 0.0:
    raise ValueError("--sim-control-hz must be positive")
for name in (
    "equilibrium_steps",
    "stability_steps",
    "joint_response_steps",
    "wrench_steps",
    "friction_steps",
    "randomized_reset_trials",
):
    if getattr(args_cli, name) <= 0:
        raise ValueError(f"--{name.replace('_', '-')} must be positive")
if (
    args_cli.upward_force_n <= 0.0
    or args_cli.torque_nm <= 0.0
    or args_cli.friction_force_n <= 0.0
):
    raise ValueError("wrench magnitudes must be positive")

# Build a deterministic policy-free environment. The organizer WBC still runs
# so this audit exercises the same 16-D arm/hand path used by Flow/RLPD.
os.environ["FLIP_TABLE_SIM_BODY_MODE"] = "balanced_wbc"
os.environ["FLIP_TABLE_LOCK_LOWER_BODY"] = "false"
os.environ["FLIP_TABLE_LOCK_ROBOT_ROOT"] = "false"
os.environ["FLIP_TABLE_FIX_ROOT_LINK"] = "false"
os.environ["FLIP_TABLE_REQUIRE_WAIST_LOCK"] = "false"
os.environ["FLIP_TABLE_RLPD_USE_FLOW_BASE"] = "true"
os.environ["FLIP_TABLE_RL_POLICY_START_STEP"] = "0"
os.environ["FLIP_TABLE_RL_CONTROL_HZ"] = str(args_cli.sim_control_hz)
os.environ["FLIP_TABLE_RL_RANDOMIZATION_LEVEL"] = "0"
os.environ["FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS"] = "0"
os.environ["FLIP_TABLE_TABLE_LONG_RANGE_M"] = "0"
os.environ["FLIP_TABLE_TABLE_DEPTH_RANGE_M"] = "0"
os.environ["FLIP_TABLE_TABLE_YAW_RANGE_RAD"] = "0"
os.environ["FLIP_TABLE_ROBOT_DISTANCE_RANGE_M"] = "0"
os.environ["FLIP_TABLE_ROBOT_LATERAL_RANGE_M"] = "0"
os.environ["FLIP_TABLE_ROBOT_YAW_RANGE_RAD"] = "0"
os.environ["FLIP_TABLE_JOINT_NOISE_RAD"] = "0"
os.environ["FLIP_TABLE_DEX1_FINGER_NOISE_M"] = "0"
os.environ["FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE"] = "false"
os.environ["FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS"] = "false"
os.environ["FLIP_TABLE_RANDOMIZE_ROOM"] = "false"
os.environ["FLIP_TABLE_RANDOMIZE_ROOM_PROPS"] = "false"
os.environ["FLIP_TABLE_RANDOMIZE_LIGHTING"] = "false"
os.environ["FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS"] = "false"
os.environ["FLIP_TABLE_RL_RANDOMIZE_MASS"] = "false"
os.environ["FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY"] = "false"
os.environ["FLIP_TABLE_RL_ENABLE_SENSOR_NOISE"] = "false"
os.environ["FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL"] = "true"

yaml_args = config_loader.load(args_cli.task_config)
merge_task_yaml_with_cli(args_cli, yaml_args)
args_cli.enable_cameras = False
args_cli.rl = "FlipTableResidualStateRL"

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import matrix_from_quat

from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_torch
from robofinals.utils.place_utils.env_utils import set_seed
from robofinals.core.mdp.actions.team_ramen_balanced_wbc_action import (
    TeamRamenBalancedWBCAction,
)
from robofinals_rl.flip_table.common import (
    ARM_JOINT_NAMES,
    LOWER_BODY_JOINT_NAMES,
    UPPER_BODY_JOINT_NAMES,
)
from model.flip_table_reinforcement_learning.rlpd.sim_runtime import (
    dataset_joint_state,
    last_commanded_target,
    set_flow_control_ready,
)


DEX1_JOINT_NAMES = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
WBC_OWNED_JOINT_NAMES = LOWER_BODY_JOINT_NAMES + UPPER_BODY_JOINT_NAMES[:3]
PREPARED_SCENE_RIGID_BODY_PATHS = (
    "/World/Table001/Table001_01",
    "/World/Leg001/Leg001",
    "/World/Leg001_01/Leg001",
    "/World/Leg001_03/Leg001",
    "/World/Leg001_06/Leg001",
    "/World/Table278/Table278",
)
DEX1_FINGER_CONTACT_FILTERS = (
    "{ENV_REGEX_NS}/Robot/left_dex1_finger_link_1",
    "{ENV_REGEX_NS}/Robot/left_dex1_finger_link_2",
    "{ENV_REGEX_NS}/Robot/right_dex1_finger_link_1",
    "{ENV_REGEX_NS}/Robot/right_dex1_finger_link_2",
)
WHITE_LEG_CONTACT_SENSOR_PATHS = (
    ("white_leg_contact_0", "{ENV_REGEX_NS}/Scene/Leg001/Leg001"),
    ("white_leg_contact_1", "{ENV_REGEX_NS}/Scene/Leg001_01/Leg001"),
    ("white_leg_contact_2", "{ENV_REGEX_NS}/Scene/Leg001_03/Leg001"),
    ("white_leg_contact_3", "{ENV_REGEX_NS}/Scene/Leg001_06/Leg001"),
)
FINGER_CONTACT_SENSOR_NAMES = (
    "left_gripper_contact",
    "left_gripper_contact_2",
    "right_gripper_contact",
    "right_gripper_contact_2",
)


def _expected_policy_target_gains(joint_name: str) -> tuple[float, float]:
    if "dex1_finger" in joint_name:
        return 800.0, 3.0
    if "wrist_" in joint_name:
        return 20.0, 1.5
    return 40.0, 3.0


def _tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return record


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_list(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _prepared_scene_velocity_contract(path: Path) -> dict[str, Any]:
    """Verify the derived USD's initial motion and leg contact contract."""

    from pxr import Usd, UsdGeom, UsdPhysics

    result: dict[str, Any] = {
        "scene": _file_identity(path),
        "rigid_bodies": {},
    }
    if not path.is_file():
        result.update({"max_abs_velocity": None, "pass": False})
        return result
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        result.update({"max_abs_velocity": None, "pass": False})
        return result

    max_abs_velocity = 0.0
    valid = True
    leg_contact_reporting_valid = True
    for prim_path in PREPARED_SCENE_RIGID_BODY_PATHS:
        prim = stage.GetPrimAtPath(prim_path)
        rigid_body = UsdPhysics.RigidBodyAPI(prim) if prim.IsValid() else None
        if not rigid_body:
            result["rigid_bodies"][prim_path] = {"valid": False}
            valid = False
            continue
        linear_value = rigid_body.GetVelocityAttr().Get()
        angular_value = rigid_body.GetAngularVelocityAttr().Get()
        linear = [0.0, 0.0, 0.0] if linear_value is None else [float(v) for v in linear_value]
        angular = [0.0, 0.0, 0.0] if angular_value is None else [float(v) for v in angular_value]
        max_abs_velocity = max(
            max_abs_velocity,
            *(abs(value) for value in (*linear, *angular)),
        )
        record = {
            "valid": True,
            "linear_velocity": linear,
            "angular_velocity": angular,
            "kinematic_enabled": rigid_body.GetKinematicEnabledAttr().Get(),
        }
        if prim_path in PREPARED_SCENE_RIGID_BODY_PATHS[1:5]:
            enabled_colliders = [
                str(descendant.GetPath())
                for descendant in Usd.PrimRange(prim)
                if descendant.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get()
                is not False
            ]
            shaft_path = f"{prim_path}/Collisions/Leg001_Collider118"
            shaft_prim = stage.GetPrimAtPath(shaft_path)
            extent = (
                UsdGeom.Mesh(shaft_prim).GetExtentAttr().Get()
                if shaft_prim.IsA(UsdGeom.Mesh)
                else None
            )
            shaft_dimensions = (
                None
                if extent is None
                else [float(extent[1][axis] - extent[0][axis]) for axis in range(3)]
            )
            contact_api = "PhysxContactReportAPI" in prim.GetAppliedSchemas()
            threshold_attr = prim.GetAttribute("physxContactReport:threshold")
            threshold = threshold_attr.Get() if threshold_attr else None
            duplicate_leaf_names = [
                str(descendant.GetPath())
                for descendant in Usd.PrimRange(prim)
                if descendant != prim and descendant.GetName() == prim.GetName()
            ]
            visual_path = f"{prim_path}/Visuals/Leg001_visual"
            legacy_visual_path = f"{prim_path}/Visuals/Leg001"
            contact_sensor_name_unique = (
                not duplicate_leaf_names
                and stage.GetPrimAtPath(visual_path).IsValid()
                and not stage.GetPrimAtPath(legacy_visual_path).IsValid()
            )
            leg_valid = (
                contact_api
                and threshold == 0.0
                and contact_sensor_name_unique
                and len(enabled_colliders) >= 100
                and shaft_path in enabled_colliders
                and shaft_dimensions is not None
                and 0.038 <= shaft_dimensions[0] <= 0.050
                and 0.038 <= shaft_dimensions[1] <= 0.050
                and 0.36 <= shaft_dimensions[2] <= 0.40
            )
            record["contact_reporting"] = {
                "physx_contact_report_api": contact_api,
                "threshold_n": threshold,
                "contact_sensor_leaf_name_unique": contact_sensor_name_unique,
                "duplicate_leaf_name_paths": duplicate_leaf_names,
                "visual_mesh": visual_path,
                "enabled_collision_shape_count": len(enabled_colliders),
                "detailed_collision_geometry_preserved": len(enabled_colliders) >= 100,
                "shaft_collider": shaft_path,
                "shaft_dimensions_m": shaft_dimensions,
                "pass": leg_valid,
            }
            leg_contact_reporting_valid = leg_contact_reporting_valid and leg_valid
        result["rigid_bodies"][prim_path] = record
    result["max_abs_velocity"] = max_abs_velocity
    result["leg_contact_reporting_pass"] = leg_contact_reporting_valid
    result["pass"] = (
        valid and max_abs_velocity <= 1.0e-9 and leg_contact_reporting_valid
    )
    return result


def _urdf_joint_limits(path: Path, joint_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    requested = set(joint_names)
    limits: dict[str, dict[str, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name not in requested:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("effort") is None or limit.get("velocity") is None:
            raise ValueError(f"URDF joint is missing effort/velocity limits: {name}")
        limits[name] = {
            "effort": float(limit.get("effort")),
            "velocity": float(limit.get("velocity")),
        }
    missing = sorted(requested - set(limits))
    if missing:
        raise ValueError(f"URDF is missing controlled joints: {missing}")
    return limits


def _actuator_value_vector(value: Any, count: int) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = as_torch(value).detach().to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 1:
        return tensor.repeat(count)
    if tensor.numel() % count != 0:
        return None
    return tensor.reshape(-1, count)[0]


def _robot_data_joint_vector(robot: Any, field: str, joint_ids: list[int]) -> torch.Tensor | None:
    value = getattr(robot.data, field, None)
    if value is None:
        return None
    tensor = as_torch(value).detach().to(dtype=torch.float64)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] <= max(joint_ids):
        return None
    return tensor[0, joint_ids]


def _wbc_actuator_contract(robot: Any, urdf_path: Path) -> dict[str, Any]:
    """Validate policy-owned drives and organizer-WBC lower-body ownership."""

    # Waist commands are deliberately absent: the organizer WBC owns all three
    # waist joints together with the floating base and legs.  Only arms and
    # Dex1 may be driven by direct policy targets.
    controlled_names = ARM_JOINT_NAMES + DEX1_JOINT_NAMES
    urdf_limits = _urdf_joint_limits(urdf_path, controlled_names)
    runtime_by_joint: dict[str, dict[str, float | None]] = {}
    actuator_groups: dict[str, Any] = {}
    ownership: dict[str, list[str]] = {
        name: [] for name in WBC_OWNED_JOINT_NAMES + ARM_JOINT_NAMES + DEX1_JOINT_NAMES
    }
    for group_name, cfg in robot.cfg.actuators.items():
        actuator = robot.actuators.get(group_name)
        if actuator is None:
            actuator_groups[group_name] = {"present": False}
            continue
        _joint_ids, joint_names = robot.find_joints(
            list(cfg.joint_names_expr), preserve_order=True
        )
        vectors = {
            field: _actuator_value_vector(getattr(actuator, field, None), len(joint_names))
            for field in (
                "effort_limit",
                "velocity_limit",
                "stiffness",
                "damping",
                "armature",
                "friction",
            )
        }
        actuator_groups[group_name] = {
            "present": True,
            "joint_names": list(joint_names),
            "runtime_values": {
                field: None if vector is None else _tensor_list(vector)
                for field, vector in vectors.items()
            },
        }
        for joint_name in joint_names:
            if joint_name in ownership:
                ownership[joint_name].append(group_name)
        for index, joint_name in enumerate(joint_names):
            runtime_by_joint[joint_name] = {
                field: None if vector is None else float(vector[index])
                for field, vector in vectors.items()
            }

    joint_ids = _resolve_joint_ids(robot, controlled_names)
    physx_effort = _robot_data_joint_vector(robot, "joint_effort_limits", joint_ids)
    physx_velocity = _robot_data_joint_vector(robot, "joint_velocity_limits", joint_ids)
    records: dict[str, Any] = {}
    all_match = True
    for index, name in enumerate(controlled_names):
        expected_kp, expected_kd = _expected_policy_target_gains(name)
        runtime = runtime_by_joint.get(name, {})
        expected = {
            **urdf_limits[name],
            "stiffness": expected_kp,
            "damping": expected_kd,
            "armature": 0.01,
            "friction": 0.0,
        }
        actual = {
            **runtime,
            "physx_effort_limit": None
            if physx_effort is None
            else float(physx_effort[index]),
            "physx_velocity_limit": None
            if physx_velocity is None
            else float(physx_velocity[index]),
        }
        comparisons = {
            "actuator_effort_matches_urdf": runtime.get("effort_limit") is not None
            and math.isclose(
                float(runtime["effort_limit"]), expected["effort"], abs_tol=1.0e-5
            ),
            "actuator_velocity_matches_urdf": runtime.get("velocity_limit") is not None
            and math.isclose(
                float(runtime["velocity_limit"]), expected["velocity"], abs_tol=1.0e-5
            ),
            "stiffness_matches_profile": runtime.get("stiffness") is not None
            and math.isclose(
                float(runtime["stiffness"]), expected["stiffness"], abs_tol=1.0e-5
            ),
            "damping_matches_profile": runtime.get("damping") is not None
            and math.isclose(
                float(runtime["damping"]), expected["damping"], abs_tol=1.0e-5
            ),
            "armature_matches_profile": runtime.get("armature") is not None
            and math.isclose(
                float(runtime["armature"]), expected["armature"], abs_tol=1.0e-5
            ),
            "friction_matches_profile": runtime.get("friction") is not None
            and math.isclose(
                float(runtime["friction"]), expected["friction"], abs_tol=1.0e-5
            ),
            "physx_effort_matches_urdf": physx_effort is not None
            and math.isclose(
                float(physx_effort[index]), expected["effort"], abs_tol=1.0e-5
            ),
            "physx_velocity_matches_urdf": physx_velocity is not None
            and math.isclose(
                float(physx_velocity[index]), expected["velocity"], abs_tol=1.0e-5
            ),
        }
        record_pass = all(comparisons.values())
        all_match = all_match and record_pass
        records[name] = {
            "expected": expected,
            "actual": actual,
            "comparisons": comparisons,
            "pass": record_pass,
        }

    arm_ownership = {name: ownership[name] for name in ARM_JOINT_NAMES}
    hand_ownership = {name: ownership[name] for name in DEX1_JOINT_NAMES}
    lower_ownership = {name: ownership[name] for name in WBC_OWNED_JOINT_NAMES}
    ownership_pass = (
        all(groups == ["arms"] for groups in arm_ownership.values())
        and all(groups == ["grippers"] for groups in hand_ownership.values())
        and all(
            len(groups) == 1 and groups[0] not in {"arms", "grippers"}
            for groups in lower_ownership.values()
        )
    )

    rigid_props = robot.cfg.spawn.rigid_props
    articulation_props = robot.cfg.spawn.articulation_props
    solver = {
        "rigid_body_position_iterations": int(
            rigid_props.solver_position_iteration_count
        ),
        "rigid_body_velocity_iterations": int(
            rigid_props.solver_velocity_iteration_count
        ),
        "articulation_position_iterations": int(
            articulation_props.solver_position_iteration_count
        ),
        "articulation_velocity_iterations": int(
            articulation_props.solver_velocity_iteration_count
        ),
    }
    solver["pass"] = (
        solver["rigid_body_position_iterations"] >= 8
        and solver["rigid_body_velocity_iterations"] >= 4
        and solver["articulation_position_iterations"] >= 8
        and solver["articulation_velocity_iterations"] >= 4
    )
    return {
        "profile": "organizer WBC lower body plus direct G1-arm/Dex1 policy targets",
        "urdf": _file_identity(urdf_path),
        "actuator_groups": actuator_groups,
        "policy_owned_joint_names": list(controlled_names),
        "wbc_owned_joint_names": list(WBC_OWNED_JOINT_NAMES),
        "ownership": {
            "arms": arm_ownership,
            "hands": hand_ownership,
            "lower_body_and_waist": lower_ownership,
            "pass": ownership_pass,
        },
        "joints": records,
        "solver": solver,
        "pass": all_match and ownership_pass and bool(solver["pass"]),
    }


def _quat_angle_rad(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    current = torch.nn.functional.normalize(current, dim=-1)
    reference = torch.nn.functional.normalize(reference, dim=-1)
    dot = torch.sum(current * reference, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _resolve_joint_ids(robot: Any, names: tuple[str, ...]) -> list[int]:
    ids, resolved = robot.find_joints(list(names), preserve_order=True)
    if tuple(resolved) != names:
        raise RuntimeError(f"joint order mismatch: expected={names}, resolved={resolved}")
    return ids


def _root_state(robot: Any) -> tuple[torch.Tensor, torch.Tensor]:
    pos = as_torch(robot.data.root_pos_w)[:, :3]
    quat = sim_quat_raw_to_xyzw_torch(as_torch(robot.data.root_quat_w)[:, :4])
    return pos.clone(), quat.clone()


def _xyzw_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return roll/pitch for simulator xyzw root quaternions."""

    q = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _table_state(task: Any, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat = task._table_body_pose(env)
    if pos is None or quat is None:
        raise RuntimeError("white tabletop pose is unavailable")
    return pos.clone(), quat.clone()


def _workbench_state(task: Any, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat = task._workbench_pose(env)
    if pos is None or quat is None:
        raise RuntimeError("workbench pose is unavailable")
    return pos.clone(), quat.clone()


def _assembly_frame(task: Any, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    table_pos, table_quat = _table_state(task, env)
    table_rot = matrix_from_quat(table_quat)
    positions = []
    rotations = []
    for name, prim_path, _site_path in task.leg_reg_int_sites:
        leg_pos, leg_quat = task._extract_object_pose(env, name, prim_path)
        if leg_pos is None or leg_quat is None:
            raise RuntimeError(f"leg pose is unavailable: {name}")
        relative_pos = torch.bmm(
            table_rot.transpose(1, 2),
            (leg_pos[:, :3] - table_pos[:, :3]).unsqueeze(-1),
        ).squeeze(-1)
        relative_rot = torch.bmm(
            table_rot.transpose(1, 2),
            matrix_from_quat(leg_quat),
        )
        positions.append(relative_pos)
        rotations.append(relative_rot)
    return torch.stack(positions, dim=1), torch.stack(rotations, dim=1)


def _rotation_matrix_angle_rad(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    relative = torch.matmul(reference.transpose(-1, -2), current)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return torch.acos(cosine)


def _step_target(env: Any, target: torch.Tensor, zero_action: torch.Tensor) -> None:
    if target.shape != (env.num_envs, 16):
        raise ValueError(f"absolute arm/hand target must be [B,16], got {tuple(target.shape)}")
    env._flip_table_rlpd_absolute_target = target
    _observation, _reward, terminated, truncated, _extras = env.step(zero_action)
    if bool(torch.logical_or(terminated, truncated).any()):
        raise RuntimeError("environment terminated during simulator contract audit")


def _reset_and_hold(
    env: Any,
    zero_action: torch.Tensor,
    steps: int,
    *,
    target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    set_flow_control_ready(env, True)
    if target is None:
        state = dataset_joint_state(env).detach().clone()
        target = torch.cat((state[:, 3:17], state[:, 17:19]), dim=1)
    else:
        target = target.detach().clone().to(device=env.device, dtype=torch.float32)
    if target.shape != (env.num_envs, 16):
        raise ValueError(f"hold target must be [B,16], got {tuple(target.shape)}")
    for _ in range(steps):
        _step_target(env, target, zero_action)
    return target, dataset_joint_state(env).detach().clone()


def _reset_actuator_target_contract(
    env: Any,
    robot: Any,
    zero_action: torch.Tensor,
) -> dict[str, Any]:
    """Prove that an episode reset clears the prior actuator command."""

    env.reset()
    set_flow_control_ready(env, True)
    initial = dataset_joint_state(env).detach().clone()
    stale = torch.cat((initial[:, 3:17], initial[:, 17:19]), dim=1)
    stale[:, 0] += 0.12
    stale[:, 3] += 0.18
    stale[:, 10] -= 0.18
    stale[:, 14:16] = torch.where(
        stale[:, 14:16] < 2.25,
        torch.full_like(stale[:, 14:16], 4.0),
        torch.full_like(stale[:, 14:16], 0.5),
    )
    for _ in range(3):
        _step_target(env, stale, zero_action)
    commanded_before_reset = last_commanded_target(env).detach().clone()

    env.reset()
    state_after_reset = as_torch(robot.data.joint_pos).detach().clone()
    velocity_after_reset = as_torch(robot.data.joint_vel).detach().clone()
    position_target_after_reset = as_torch(robot.data.joint_pos_target).detach().clone()
    velocity_target_after_reset = as_torch(robot.data.joint_vel_target).detach().clone()
    position_error = torch.abs(position_target_after_reset - state_after_reset)
    velocity_target_error = torch.abs(velocity_target_after_reset - velocity_after_reset)
    initial_action = torch.cat((initial[:, 3:17], initial[:, 17:19]), dim=1)
    stale_command_delta = torch.abs(commanded_before_reset - initial_action).max()

    result = {
        "deliberate_pre_reset_command_delta": float(stale_command_delta),
        "max_position_target_state_error_rad": float(position_error.max()),
        "max_velocity_target_state_error_rad_s": float(velocity_target_error.max()),
        "position_target_finite": bool(torch.isfinite(position_target_after_reset).all()),
        "velocity_target_finite": bool(torch.isfinite(velocity_target_after_reset).all()),
    }
    result["pass"] = (
        result["deliberate_pre_reset_command_delta"] >= 0.05
        and result["max_position_target_state_error_rad"] <= 1.0e-6
        and result["max_velocity_target_state_error_rad_s"] <= 1.0e-6
        and result["position_target_finite"]
        and result["velocity_target_finite"]
    )
    return result


def _explicit_effort_saturation_contract(
    env: Any,
    robot: Any,
    body_joint_ids: list[int],
    nominal_target: torch.Tensor,
    zero_action: torch.Tensor,
    urdf_path: Path,
) -> dict[str, Any]:
    """Prove that explicit PD commands are clipped before reaching PhysX."""

    _reset_and_hold(env, zero_action, 10, target=nominal_target)
    current = as_torch(robot.data.joint_pos)[:, body_joint_ids]
    limits = as_torch(robot.data.soft_joint_pos_limits)[:, body_joint_ids]
    farther_upper = torch.abs(limits[..., 1] - current) >= torch.abs(
        current - limits[..., 0]
    )
    saturation_target = nominal_target.clone()
    saturation_target[:, :14] = torch.where(
        farther_upper,
        limits[..., 1],
        limits[..., 0],
    )
    _step_target(env, saturation_target, zero_action)

    computed = as_torch(robot.data.computed_torque)[:, body_joint_ids].detach().clone()
    applied = as_torch(robot.data.applied_torque)[:, body_joint_ids].detach().clone()
    urdf_limits = _urdf_joint_limits(urdf_path, ARM_JOINT_NAMES)
    expected = torch.tensor(
        [urdf_limits[name]["effort"] for name in ARM_JOINT_NAMES],
        device=applied.device,
        dtype=applied.dtype,
    ).unsqueeze(0)
    requested_saturation = torch.abs(computed) >= expected * 1.01
    within_limit = torch.abs(applied) <= expected + 1.0e-4
    saturated_at_limit = torch.abs(torch.abs(applied) - expected) <= 1.0e-3
    saturation_matches = (~requested_saturation) | saturated_at_limit
    sign_matches = (~requested_saturation) | (
        torch.sign(applied) == torch.sign(computed)
    )
    safety_clip_delta = float(
        torch.abs(last_commanded_target(env) - saturation_target).max()
    )
    requested_fraction = float(requested_saturation.float().mean())
    result = {
        "joint_names": list(ARM_JOINT_NAMES),
        "target": _tensor_list(saturation_target[:, :14]),
        "computed_torque_nm": _tensor_list(computed),
        "applied_torque_nm": _tensor_list(applied),
        "urdf_effort_limit_nm": _tensor_list(expected),
        "requested_saturation": _tensor_list(requested_saturation),
        "requested_saturation_fraction": requested_fraction,
        "max_limit_excess_nm": float(
            torch.relu(torch.abs(applied) - expected).max()
        ),
        "max_safety_clip_delta_rad": safety_clip_delta,
    }
    result["pass"] = (
        bool(torch.all(within_limit))
        and bool(torch.all(saturation_matches))
        and bool(torch.all(sign_matches))
        # This is an actuator-clipping proof, not a requirement to overload
        # every arm joint simultaneously.  A quarter of the joints requesting
        # saturation is sufficient to exercise both signs and multiple motor
        # classes; every requested saturation must still clip exactly.
        and requested_fraction >= 0.25
        # The deployable action adapter uses rounded conservative limits that
        # are at most 0.0005 rad inside the exact generated-URDF limits.
        and safety_clip_delta <= 1.0e-3
    )
    return result


def _max_assembly_drift(
    task: Any,
    env: Any,
    reference_pos: torch.Tensor,
    reference_rot: torch.Tensor,
) -> tuple[float, float]:
    current_pos, current_rot = _assembly_frame(task, env)
    pos_drift = torch.linalg.norm(current_pos - reference_pos, dim=-1).max()
    rot_drift = _rotation_matrix_angle_rad(current_rot, reference_rot).max()
    return float(pos_drift), float(rot_drift)


def _prim_bounds(task: Any, env: Any, suffix: str) -> dict[str, Any]:
    """Return local/world aligned bounds for one runtime scene prim."""

    from pxr import Usd, UsdGeom

    prim = task._find_prim_by_suffix(env, suffix, env_id=0)
    if prim is None:
        raise RuntimeError(f"runtime prim is unavailable: {suffix}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )
    local_range = cache.ComputeLocalBound(prim).ComputeAlignedRange()
    world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()

    def range_record(value: Any) -> dict[str, list[float]]:
        minimum = [float(item) for item in value.GetMin()]
        maximum = [float(item) for item in value.GetMax()]
        size = [float(item) for item in value.GetSize()]
        if not all(math.isfinite(item) and item > 0.0 for item in size):
            raise RuntimeError(f"invalid bounds for {prim.GetPath()}: {size}")
        return {"min": minimum, "max": maximum, "size": size}

    return {
        "path": str(prim.GetPath()),
        "local": range_record(local_range),
        "world": range_record(world_range),
    }


def _scene_geometry_contract(task: Any, env: Any, robot: Any) -> dict[str, Any]:
    """Validate the fixed-scene geometry and reset pose used by every policy."""

    from pxr import UsdPhysics

    workbench_bounds = _prim_bounds(task, env, "Scene/Table278/Table278")
    tabletop_bounds = _prim_bounds(task, env, "Scene/Table001/Table001_01")
    workbench_size = workbench_bounds["local"]["size"]
    tabletop_size = tabletop_bounds["local"]["size"]
    configured_workbench_size = [
        2.0 * float(os.environ.get("FLIP_TABLE_WORKBENCH_HALF_LENGTH_M", "0.90")),
        2.0 * float(os.environ.get("FLIP_TABLE_WORKBENCH_HALF_DEPTH_M", "0.375")),
        0.76,
    ]
    configured_tabletop_size = [
        float(os.environ.get("FLIP_TABLE_TOP_LENGTH_M", "0.58")),
        float(os.environ.get("FLIP_TABLE_TOP_DEPTH_M", "0.42")),
        0.04,
    ]
    workbench_dimension_error = [
        abs(actual - expected)
        for actual, expected in zip(workbench_size, configured_workbench_size)
    ]
    tabletop_dimension_error = [
        abs(actual - expected)
        for actual, expected in zip(tabletop_size, configured_tabletop_size)
    ]

    table_pos, table_quat = _table_state(task, env)
    workbench_pos, workbench_quat = _workbench_state(task, env)
    table_rot = matrix_from_quat(table_quat)
    workbench_rot = matrix_from_quat(workbench_quat)
    table_relative_workbench = torch.bmm(
        workbench_rot.transpose(1, 2),
        (table_pos - workbench_pos).unsqueeze(-1),
    ).squeeze(-1)
    table_rotation_workbench = torch.bmm(
        workbench_rot.transpose(1, 2), table_rot
    )
    table_yaw_workbench = torch.atan2(
        table_rotation_workbench[:, 1, 0],
        table_rotation_workbench[:, 0, 0],
    )
    abs_cos_yaw = torch.abs(torch.cos(table_yaw_workbench))
    abs_sin_yaw = torch.abs(torch.sin(table_yaw_workbench))
    projected_half_length = 0.5 * (
        tabletop_size[0] * abs_cos_yaw + tabletop_size[1] * abs_sin_yaw
    )
    projected_half_depth = 0.5 * (
        tabletop_size[0] * abs_sin_yaw + tabletop_size[1] * abs_cos_yaw
    )
    footprint_clearance = torch.stack(
        (
            0.5 * workbench_size[0]
            - torch.abs(table_relative_workbench[:, 0])
            - projected_half_length,
            0.5 * workbench_size[1]
            - torch.abs(table_relative_workbench[:, 1])
            - projected_half_depth,
        ),
        dim=-1,
    )
    support_gap = (
        table_pos[:, 2]
        - 0.5 * tabletop_bounds["world"]["size"][2]
        - workbench_bounds["world"]["max"][2]
    )

    table_normal = table_rot[:, :, 2]
    leg_axis_alignment = []
    leg_center_offset_along_normal = []
    leg_centers_table_frame = []
    for name, prim_path, _site_path in task.leg_reg_int_sites:
        leg_pos, leg_quat = task._extract_object_pose(env, name, prim_path)
        if leg_pos is None or leg_quat is None:
            raise RuntimeError(f"leg pose is unavailable: {name}")
        leg_axis = matrix_from_quat(leg_quat)[:, :, 2]
        table_to_leg = leg_pos[:, :3] - table_pos[:, :3]
        leg_axis_alignment.append(torch.abs(torch.sum(leg_axis * table_normal, dim=-1)))
        leg_center_offset_along_normal.append(
            torch.sum(table_to_leg * table_normal, dim=-1)
        )
        leg_centers_table_frame.append(
            torch.bmm(table_rot.transpose(1, 2), table_to_leg.unsqueeze(-1)).squeeze(-1)
        )
    leg_axis_alignment_tensor = torch.stack(leg_axis_alignment, dim=1)
    leg_center_offset_tensor = torch.stack(leg_center_offset_along_normal, dim=1)
    leg_centers_table_tensor = torch.stack(leg_centers_table_frame, dim=1)

    table_prim = task._find_prim_by_suffix(
        env, "Scene/Table001/Table001_01", env_id=0
    )
    fixed_joint_paths = []
    if table_prim is not None:
        for prim in table_prim.GetChildren():
            if prim.GetName().startswith("FlipTableEvalFixedJoint_") and prim.IsA(
                UsdPhysics.FixedJoint
            ):
                fixed_joint_paths.append(str(prim.GetPath()))

    root_pos, root_quat = _root_state(robot)
    root_forward = matrix_from_quat(root_quat)[:, :, 0]
    root_forward_xy = torch.nn.functional.normalize(root_forward[:, :2], dim=-1)
    root_to_table_xy = table_pos[:, :2] - root_pos[:, :2]
    root_table_distance = torch.linalg.norm(root_to_table_xy, dim=-1)
    root_to_table_direction = torch.nn.functional.normalize(root_to_table_xy, dim=-1)
    robot_facing_cosine = torch.sum(root_forward_xy * root_to_table_direction, dim=-1)
    minimum_root_distance = float(
        os.environ.get("FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M", "0.62")
    )

    result = {
        "workbench_bounds_m": workbench_bounds,
        "tabletop_bounds_m": tabletop_bounds,
        "configured_workbench_size_m": configured_workbench_size,
        "configured_tabletop_size_m": configured_tabletop_size,
        "workbench_dimension_abs_error_m": workbench_dimension_error,
        "tabletop_dimension_abs_error_m": tabletop_dimension_error,
        "table_center_in_workbench_frame_m": _tensor_list(table_relative_workbench),
        "table_yaw_in_workbench_frame_rad": _tensor_list(table_yaw_workbench),
        "table_projected_half_extent_m": _tensor_list(
            torch.stack((projected_half_length, projected_half_depth), dim=-1)
        ),
        "table_footprint_clearance_m": _tensor_list(footprint_clearance),
        "table_support_gap_m": _tensor_list(support_gap),
        "leg_axis_alignment_abs_cosine": _tensor_list(leg_axis_alignment_tensor),
        "leg_center_offset_along_table_normal_m": _tensor_list(
            leg_center_offset_tensor
        ),
        "leg_centers_in_table_frame_m": _tensor_list(leg_centers_table_tensor),
        "assembled_fixed_joint_paths": fixed_joint_paths,
        "robot_root_to_table_distance_m": _tensor_list(root_table_distance),
        "minimum_robot_root_to_table_distance_m": minimum_root_distance,
        "robot_facing_table_cosine": _tensor_list(robot_facing_cosine),
    }
    result["pass"] = (
        max(workbench_dimension_error) <= 0.01
        and max(tabletop_dimension_error) <= 0.01
        and bool(torch.all(footprint_clearance >= 0.03))
        and bool(torch.all(torch.abs(support_gap) <= 0.01))
        and bool(torch.all(leg_axis_alignment_tensor >= math.cos(math.radians(2.0))))
        and bool(torch.all(leg_center_offset_tensor <= -0.15))
        and bool(
            torch.all(
                torch.abs(leg_centers_table_tensor[:, :, 0])
                <= 0.5 * tabletop_size[0] + 0.01
            )
        )
        and bool(
            torch.all(
                torch.abs(leg_centers_table_tensor[:, :, 1])
                <= 0.5 * tabletop_size[1] + 0.01
            )
        )
        and len(fixed_joint_paths) == 4
        and bool(torch.all(root_table_distance >= minimum_root_distance - 0.005))
        and bool(torch.all(robot_facing_cosine >= math.cos(math.radians(10.0))))
    )
    return result


def _randomized_reset_contract(
    task: Any,
    env: Any,
    robot: Any,
    zero_action: torch.Tensor,
) -> dict[str, Any]:
    """Exercise randomized reset invariants, not just one fixed scene."""

    randomized_values = {
        "FLIP_TABLE_RL_RANDOMIZATION_LEVEL": "1.0",
        "FLIP_TABLE_TABLE_LONG_RANGE_M": "0.12",
        "FLIP_TABLE_TABLE_DEPTH_RANGE_M": "0.035",
        "FLIP_TABLE_TABLE_YAW_RANGE_RAD": "0.45",
        "FLIP_TABLE_ROBOT_DISTANCE_RANGE_M": "0.10",
        "FLIP_TABLE_ROBOT_LATERAL_RANGE_M": "0.10",
        "FLIP_TABLE_ROBOT_YAW_RANGE_RAD": "0.08",
        "FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE": "true",
        "FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE": "0.50",
        "FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS": "false",
        "FLIP_TABLE_RANDOMIZE_ROOM": "false",
        "FLIP_TABLE_RANDOMIZE_ROOM_PROPS": "false",
        "FLIP_TABLE_RANDOMIZE_LIGHTING": "false",
    }
    previous = {name: os.environ.get(name) for name in randomized_values}
    for name, value in randomized_values.items():
        os.environ[name] = value

    records = []
    table_centers = []
    table_yaws = []
    robot_roots = []
    upper_states = []
    try:
        for trial in range(args_cli.randomized_reset_trials):
            env.reset()
            set_flow_control_ready(env, True)
            state = dataset_joint_state(env).detach().clone()
            target = torch.cat((state[:, 3:17], state[:, 17:19]), dim=1)
            for _ in range(10):
                _step_target(env, target, zero_action)

            geometry = _scene_geometry_contract(task, env, robot)
            table_pos, table_quat = _table_state(task, env)
            workbench_pos, workbench_quat = _workbench_state(task, env)
            table_rot = matrix_from_quat(table_quat)
            workbench_rot = matrix_from_quat(workbench_quat)
            table_relative = torch.bmm(
                workbench_rot.transpose(1, 2),
                (table_pos - workbench_pos).unsqueeze(-1),
            ).squeeze(-1)
            relative_rot = torch.bmm(workbench_rot.transpose(1, 2), table_rot)
            table_yaw = torch.atan2(relative_rot[:, 1, 0], relative_rot[:, 0, 0])
            root_pos, _root_quat = _root_state(robot)
            upper_state = dataset_joint_state(env)[:, :17]
            success_components = task._stable_flip_success_components(env)

            table_centers.append(table_relative[:, :2].detach().clone())
            table_yaws.append(table_yaw.detach().clone())
            robot_roots.append(root_pos[:, :2].detach().clone())
            upper_states.append(upper_state.detach().clone())
            records.append(
                {
                    "trial": trial,
                    "geometry_pass": bool(geometry["pass"]),
                    "table_center_in_workbench_frame_m": geometry[
                        "table_center_in_workbench_frame_m"
                    ],
                    "table_footprint_clearance_m": geometry[
                        "table_footprint_clearance_m"
                    ],
                    "table_support_gap_m": geometry["table_support_gap_m"],
                    "minimum_leg_axis_alignment_abs_cosine": min(
                        min(row) for row in geometry["leg_axis_alignment_abs_cosine"]
                    ),
                    "robot_root_to_table_distance_m": geometry[
                        "robot_root_to_table_distance_m"
                    ],
                    "robot_facing_table_cosine": geometry[
                        "robot_facing_table_cosine"
                    ],
                    "initial_success_candidate": _tensor_list(
                        success_components["candidate"]
                    ),
                }
            )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    table_centers_tensor = torch.cat(table_centers, dim=0)
    table_yaws_tensor = torch.cat(table_yaws, dim=0)
    robot_roots_tensor = torch.cat(robot_roots, dim=0)
    upper_states_tensor = torch.cat(upper_states, dim=0)
    table_center_span = table_centers_tensor.amax(dim=0) - table_centers_tensor.amin(
        dim=0
    )
    table_yaw_span = table_yaws_tensor.max() - table_yaws_tensor.min()
    robot_root_span = robot_roots_tensor.amax(dim=0) - robot_roots_tensor.amin(dim=0)
    upper_joint_span = upper_states_tensor.amax(dim=0) - upper_states_tensor.amin(dim=0)
    result = {
        "trials": args_cli.randomized_reset_trials,
        "settle_steps_per_trial": 10,
        "configured_randomization": randomized_values,
        "observed_table_center_span_m": _tensor_list(table_center_span),
        "observed_table_yaw_span_rad": float(table_yaw_span),
        "observed_robot_root_xy_span_m": _tensor_list(robot_root_span),
        "observed_max_upper_joint_span_rad": float(upper_joint_span.max()),
        "records": records,
    }
    result["pass"] = (
        all(record["geometry_pass"] for record in records)
        and not any(
            any(record["initial_success_candidate"]) for record in records
        )
        and float(table_center_span[0]) >= 0.03
        and float(table_center_span[1]) >= 0.01
        and float(table_yaw_span) >= 0.10
        and float(robot_root_span.max()) >= 0.05
        and float(upper_joint_span.max()) >= 0.01
    )
    return result


def _contact_material_snapshot(task: Any, env: Any) -> dict[str, Any]:
    """Inspect active collision shapes and inherited physics materials."""

    from pxr import Usd, UsdPhysics, UsdShade

    material_suffixes = {
        "hand": "Robot/Looks/flip_table_contact_hand",
        "white": "Scene/Looks/flip_table_contact_white",
        "workbench": "Scene/Looks/flip_table_contact_workbench",
    }
    materials: dict[str, Any] = {}
    for surface, suffix in material_suffixes.items():
        prim = task._find_prim_by_suffix(env, suffix, env_id=0)
        if prim is None:
            materials[surface] = {"path": None, "attributes": {}}
            continue
        attributes = {}
        for name in (
            "physics:staticFriction",
            "physics:dynamicFriction",
            "physics:restitution",
            "physxMaterial:frictionCombineMode",
            "physxMaterial:restitutionCombineMode",
        ):
            attribute = prim.GetAttribute(name)
            attributes[name] = attribute.Get() if attribute else None
        materials[surface] = {
            "path": str(prim.GetPath()),
            "attributes": _jsonable(attributes),
        }

    counts = {surface: 0 for surface in material_suffixes}
    disabled_counts = {surface: 0 for surface in material_suffixes}
    binding_mismatches: list[dict[str, Any]] = []
    env_prefix = "/World/envs/env_0/"
    hand_tokens = (
        "/Robot/left_wrist_yaw_link/collisions/",
        "/Robot/right_wrist_yaw_link/collisions/",
        "/Robot/left_dex1_finger_link_1/collisions/",
        "/Robot/left_dex1_finger_link_2/collisions/",
        "/Robot/right_dex1_finger_link_1/collisions/",
        "/Robot/right_dex1_finger_link_2/collisions/",
    )
    white_tokens = (
        "/Scene/Table001/Table001_01/",
        "/Scene/Leg001/Leg001/",
        "/Scene/Leg001_01/Leg001/",
        "/Scene/Leg001_03/Leg001/",
        "/Scene/Leg001_06/Leg001/",
    )
    workbench_suffix = "/Scene/Table278/Table278/Collisions/Table278_Collider5"
    for prim in Usd.PrimRange.Stage(env.sim.stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if env_prefix not in path:
            continue
        surface = None
        if any(token in path for token in hand_tokens):
            surface = "hand"
        elif any(token in path for token in white_tokens):
            surface = "white"
        elif path.endswith(workbench_suffix):
            surface = "workbench"
        if surface is None:
            continue
        counts[surface] += 1
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.HasAuthoredValueOpinion() and enabled.Get() is False:
            disabled_counts[surface] += 1
        computed = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        material = computed[0] if computed else None
        material_path = str(material.GetPath()) if material else None
        if material_path is None or not material_path.endswith(material_suffixes[surface]):
            if len(binding_mismatches) < 20:
                binding_mismatches.append(
                    {
                        "collision_path": path,
                        "surface": surface,
                        "material_path": material_path,
                    }
                )

    result = {
        "collision_shape_counts": counts,
        "disabled_collision_shape_counts": disabled_counts,
        "materials": materials,
        "binding_mismatches": binding_mismatches,
    }
    result["pass"] = (
        counts["hand"] >= 8
        and counts["white"] >= 20
        and counts["workbench"] == 1
        and not any(disabled_counts.values())
        and not binding_mismatches
        and all(materials[surface]["path"] is not None for surface in materials)
    )
    return result


def _set_uniform_contact_pair_ranges(
    *, static_friction: float, dynamic_friction: float, restitution: float
) -> None:
    for pair in ("HAND_WHITE", "WHITE_WORKBENCH", "WORKBENCH_HAND"):
        os.environ[f"FLIP_TABLE_CONTACT_{pair}_STATIC_RANGE"] = (
            f"{static_friction},{static_friction}"
        )
        os.environ[f"FLIP_TABLE_CONTACT_{pair}_DYNAMIC_RANGE"] = (
            f"{dynamic_friction},{dynamic_friction}"
        )
        os.environ[f"FLIP_TABLE_CONTACT_{pair}_RESTITUTION_RANGE"] = (
            f"{restitution},{restitution}"
        )


def _friction_trial(
    env: Any,
    task: Any,
    table_entity: Any,
    zero_action: torch.Tensor,
    *,
    static_friction: float,
    dynamic_friction: float,
) -> dict[str, Any]:
    _set_uniform_contact_pair_ranges(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.02,
    )
    os.environ["FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS"] = "true"
    target, _state = _reset_and_hold(env, zero_action, args_cli.equilibrium_steps)
    material_snapshot = _contact_material_snapshot(task, env)
    start_pos, start_quat = _table_state(task, env)
    assembly_pos, assembly_rot = _assembly_frame(task, env)
    workbench_pos, workbench_quat = _workbench_state(task, env)
    forces = torch.zeros((env.num_envs, 1, 3), dtype=torch.float32, device=env.device)
    torques = torch.zeros_like(forces)
    forces[:, :, 0] = args_cli.friction_force_n
    table_entity.permanent_wrench_composer.reset()
    table_entity.permanent_wrench_composer.add_forces_and_torques(
        forces, torques, is_global=True
    )
    max_horizontal_displacement = 0.0
    try:
        for _ in range(args_cli.friction_steps):
            _step_target(env, target, zero_action)
            table_pos, _table_quat = _table_state(task, env)
            horizontal = torch.linalg.norm(
                table_pos[:, :2] - start_pos[:, :2], dim=-1
            )
            max_horizontal_displacement = max(
                max_horizontal_displacement, float(horizontal.max())
            )
    finally:
        table_entity.permanent_wrench_composer.reset()

    final_pos, final_quat = _table_state(task, env)
    assembly_pos_drift, assembly_rot_drift = _max_assembly_drift(
        task, env, assembly_pos, assembly_rot
    )
    final_workbench_pos, final_workbench_quat = _workbench_state(task, env)
    return {
        "requested_static_friction": static_friction,
        "requested_dynamic_friction": dynamic_friction,
        "horizontal_force_n": args_cli.friction_force_n,
        "steps": args_cli.friction_steps,
        "max_horizontal_displacement_m": max_horizontal_displacement,
        "final_translation_m": float(
            torch.linalg.norm(final_pos - start_pos, dim=-1).max()
        ),
        "final_rotation_rad": float(_quat_angle_rad(final_quat, start_quat).max()),
        "assembly_position_drift_m": assembly_pos_drift,
        "assembly_rotation_drift_rad": assembly_rot_drift,
        "workbench_translation_m": float(
            torch.linalg.norm(final_workbench_pos - workbench_pos, dim=-1).max()
        ),
        "workbench_rotation_rad": float(
            _quat_angle_rad(final_workbench_quat, workbench_quat).max()
        ),
        "material_snapshot": material_snapshot,
    }


def _contact_partner_filter_contract(env: Any) -> dict[str, Any]:
    """Verify GPU-supported reverse leg-to-finger contact attribution."""

    finger_records: dict[str, Any] = {}
    leg_records: dict[str, Any] = {}
    passed = True
    resolved_expected_filters = tuple(
        path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        for path in DEX1_FINGER_CONTACT_FILTERS
    )
    for name in FINGER_CONTACT_SENSOR_NAMES:
        sensor = env.scene.sensors[name]
        filters = tuple(sensor.cfg.filter_prim_paths_expr)
        matrix_value = getattr(sensor.data, "force_matrix_w", None)
        matrix = None if matrix_value is None else as_torch(matrix_value)
        num_sensors = int(getattr(sensor, "_num_sensors", -1))
        sensor_pass = (
            not filters
            and matrix is None
            and num_sensors == 1
        )
        finger_records[name] = {
            "prim_path": sensor.cfg.prim_path,
            "filter_prim_paths_expr": list(filters),
            "num_sensor_bodies": num_sensors,
            "force_matrix_shape": None if matrix is None else list(matrix.shape),
            "purpose": "unfiltered all-surface hardware safety",
            "pass": sensor_pass,
        }
        passed = passed and sensor_pass

    for name, expected_prim_path in WHITE_LEG_CONTACT_SENSOR_PATHS:
        sensor = env.scene.sensors[name]
        filters = tuple(sensor.cfg.filter_prim_paths_expr)
        matrix_value = getattr(sensor.data, "force_matrix_w", None)
        matrix = None if matrix_value is None else as_torch(matrix_value)
        num_sensors = int(getattr(sensor, "_num_sensors", -1))
        num_filter_shapes = int(getattr(sensor, "_num_filter_shapes", -1))
        resolved_prim_path = expected_prim_path.replace(
            "{ENV_REGEX_NS}", "/World/envs/env_.*"
        )
        sensor_pass = (
            sensor.cfg.prim_path == resolved_prim_path
            and filters == resolved_expected_filters
            and matrix is not None
            and matrix.shape == (env.num_envs, 1, 4, 3)
            and num_sensors == 1
            and num_filter_shapes == 4
        )
        leg_records[name] = {
            "prim_path": sensor.cfg.prim_path,
            "expected_prim_path": expected_prim_path,
            "filter_prim_paths_expr": list(filters),
            "num_sensor_bodies": num_sensors,
            "num_filter_shapes": num_filter_shapes,
            "force_matrix_shape": None if matrix is None else list(matrix.shape),
            "purpose": "per-leg force from each Dex1 finger body",
            "pass": sensor_pass,
        }
        passed = passed and sensor_pass
    return {
        "strategy": "four leg-body sensors filtered against four finger bodies",
        "unsupported_shape_filters_present": False,
        "expected_filter_prim_paths_expr": list(DEX1_FINGER_CONTACT_FILTERS),
        "expected_resolved_filter_prim_paths_expr": list(resolved_expected_filters),
        "finger_safety_sensors": finger_records,
        "white_leg_attribution_sensors": leg_records,
        "pass": passed,
    }


def main() -> None:
    env_cfg = parse_env_cfg(
        scene_backend=args_cli.scene_backend,
        task_backend=args_cli.task_backend,
        task_name=args_cli.task,
        robot_name=args_cli.robot,
        scene_name=args_cli.layout,
        rl_name=args_cli.rl,
        robot_scale=args_cli.robot_scale,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        first_person_view=args_cli.first_person_view,
        enable_cameras=False,
        execute_mode=ExecuteMode.TRAIN,
        usd_simplify=args_cli.usd_simplify,
        seed=args_cli.seed,
        sources=args_cli.sources,
        object_projects=args_cli.object_projects,
        headless_mode=args_cli.headless,
        enable_full_local_scene=args_cli.enable_full_local_scene,
    )
    env_cfg.terminations.success = None
    required_steps = (
        args_cli.equilibrium_steps
        + args_cli.stability_steps
        + args_cli.joint_response_steps
        + 100
    )
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        required_steps / args_cli.sim_control_hz,
    )
    if args_cli.seed is None:
        raise ValueError("a deterministic --seed is required")
    env_cfg.seed = int(args_cli.seed)

    task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
    if task_name not in gym.registry:
        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )
    env = gym.make(task_name, cfg=env_cfg).unwrapped
    set_seed(env_cfg.seed, env)
    task = env.cfg.isaaclab_arena_env.task
    robot = env.scene["robot"]
    robofinals_root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
    official_root_raw = os.environ.get("FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT", "").strip()
    runtime_paths = {
        "patched_g1_config": robofinals_root
        / "robofinals/core/robots/unitree/g1.py",
        "patched_g1_assets_config": robofinals_root
        / "robofinals/core/robots/unitree/assets_cfg.py",
        "generated_dex1_usd": robofinals_root
        / "robofinals/data/assets/g1_urdf_gripper/G1_GRIPPER.usd",
        "generated_dex1_base_usd": robofinals_root
        / "robofinals/data/assets/g1_urdf_gripper/configuration/usd_base.usd",
        "generated_dex1_urdf": robofinals_root
        / "robofinals/data/assets/g1_urdf_gripper/g1/g1_29dof_mode_15_with_dex1_1.urdf",
        "prepared_scene_usd": Path(str(args_cli.layout)),
        "wbc_stand_onnx": robofinals_root
        / "robofinals/data/ckpts/nv_wbc_v0904/homie_v2/stand.onnx",
        "wbc_walk_onnx": robofinals_root
        / "robofinals/data/ckpts/nv_wbc_v0904/homie_v2/walk.onnx",
        "team_ramen_wbc_adapter": robofinals_root
        / "robofinals/core/mdp/actions/team_ramen_balanced_wbc_action.py",
    }
    if official_root_raw:
        official_root = Path(official_root_raw)
        runtime_paths.update(
            {
                "official_v1_g1_config": official_root
                / "robofinals/core/robots/unitree/g1.py",
                "official_v1_g1_assets_config": official_root
                / "robofinals/core/robots/unitree/assets_cfg.py",
            }
        )
    actuator_configs: dict[str, Any] = {}
    for name, actuator_cfg in robot.cfg.actuators.items():
        actuator_configs[name] = {
            "config_class": type(actuator_cfg).__name__,
            "runtime_class": type(robot.actuators[name]).__name__
            if name in robot.actuators
            else None,
            **{
                field: _jsonable(getattr(actuator_cfg, field))
                for field in (
                    "joint_names_expr",
                    "effort_limit",
                    "effort_limit_sim",
                    "velocity_limit",
                    "velocity_limit_sim",
                    "stiffness",
                    "damping",
                    "armature",
                    "friction",
                )
                if hasattr(actuator_cfg, field)
            },
        }
    file_identities = {
        label: _file_identity(path) for label, path in runtime_paths.items()
    }
    expected_official_hashes = {
        "official_v1_g1_config":
            "da19a18ddff14d7cb0fd8878d7f8c3f55ce44bfba05a44d08c453c23cb7721a0",
        "official_v1_g1_assets_config":
            "a8cec088b01198f105b7c6dac2652ba245e0da197c267512c2dcab9230041919",
    }
    official_hashes_match = bool(official_root_raw) and all(
        file_identities.get(label, {}).get("sha256") == expected
        for label, expected in expected_official_hashes.items()
    )
    expected_runtime_hashes = {
        "wbc_stand_onnx": os.environ.get("FLIP_TABLE_WBC_STAND_ONNX_SHA256", ""),
        "wbc_walk_onnx": os.environ.get("FLIP_TABLE_WBC_WALK_ONNX_SHA256", ""),
        "team_ramen_wbc_adapter": os.environ.get("FLIP_TABLE_WBC_ADAPTER_SHA256", ""),
    }
    runtime_hashes_match = all(
        bool(expected) and file_identities[label].get("sha256") == expected
        for label, expected in expected_runtime_hashes.items()
    )
    base_action = env.action_manager.get_term("base_action")
    action_contract = {
        "total_action_dim": int(env.action_manager.total_action_dim),
        "active_terms": list(env.action_manager.active_terms),
        "base_action_class": type(base_action).__name__,
        "is_team_ramen_balanced_wbc_adapter": isinstance(
            base_action, TeamRamenBalancedWBCAction
        ),
        "base_action_dim": int(base_action.action_dim),
        "base_height_m": float(base_action.cfg.base_height_m),
        "body_mode": os.environ.get("FLIP_TABLE_SIM_BODY_MODE", ""),
    }
    action_contract["pass"] = (
        action_contract["total_action_dim"] == 16
        and action_contract["active_terms"]
        == ["base_action", "left_hand_action", "right_hand_action"]
        and action_contract["is_team_ramen_balanced_wbc_adapter"]
        and action_contract["base_action_dim"] == 14
        and math.isclose(action_contract["base_height_m"], 0.74, abs_tol=1.0e-9)
        and action_contract["body_mode"] == "balanced_wbc"
    )
    runtime_identity = {
        "image_contract": "paperc/robofinals:RoboFinals-IKEA-V1 plus deterministic Dex1 compatibility patch",
        "files": file_identities,
        "expected_official_hashes": expected_official_hashes,
        "official_hashes_match": official_hashes_match,
        "expected_runtime_hashes": expected_runtime_hashes,
        "runtime_hashes_match": runtime_hashes_match,
        "action_contract": action_contract,
        "actuators": actuator_configs,
    }
    runtime_identity["pass"] = (
        official_hashes_match
        and runtime_hashes_match
        and bool(action_contract["pass"])
        and actuator_configs.get("arms", {}).get("config_class") == "IdealPDActuatorCfg"
        and actuator_configs.get("grippers", {}).get("config_class") == "IdealPDActuatorCfg"
        and all(record["exists"] for record in runtime_identity["files"].values())
    )
    prepared_scene_velocity = _prepared_scene_velocity_contract(
        runtime_paths["prepared_scene_usd"]
    )
    wbc_actuators = _wbc_actuator_contract(
        robot, runtime_paths["generated_dex1_urdf"]
    )
    body_joint_ids = _resolve_joint_ids(robot, ARM_JOINT_NAMES)
    lower_joint_ids = _resolve_joint_ids(robot, WBC_OWNED_JOINT_NAMES)
    zero_action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        dtype=torch.float32,
        device=env.device,
    )
    if zero_action.shape[1] != 16:
        raise RuntimeError(f"expected a 16-D WBC action adapter, got {zero_action.shape[1]}")

    expected_step_dt = 1.0 / args_cli.sim_control_hz
    timebase = {
        "sim_dt_s": float(env.cfg.sim.dt),
        "decimation": int(env.cfg.decimation),
        "step_dt_s": float(env.step_dt),
        "requested_control_hz": args_cli.sim_control_hz,
        "pass": math.isclose(
            float(env.step_dt), expected_step_dt, rel_tol=0.0, abs_tol=1.0e-9
        ),
    }

    reset_actuator_targets = _reset_actuator_target_contract(
        env,
        robot,
        zero_action,
    )
    hold_target, hold_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    scene_geometry = _scene_geometry_contract(task, env, robot)
    randomized_resets = _randomized_reset_contract(task, env, robot, zero_action)
    hold_target, hold_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    table_ref_pos, table_ref_quat = _table_state(task, env)
    workbench_ref_pos, workbench_ref_quat = _workbench_state(task, env)
    root_ref_pos, root_ref_quat = _root_state(robot)
    lower_ref = as_torch(robot.data.joint_pos)[:, lower_joint_ids].clone()
    body_ref = hold_state[:, 3:17].clone()
    assembly_ref_pos, assembly_ref_rot = _assembly_frame(task, env)
    stability_max = {
        "table_translation_m": 0.0,
        "table_rotation_rad": 0.0,
        "workbench_translation_m": 0.0,
        "workbench_rotation_rad": 0.0,
        "robot_root_xy_drift_m": 0.0,
        "robot_root_height_error_m": 0.0,
        "robot_root_abs_roll_rad": 0.0,
        "robot_root_abs_pitch_rad": 0.0,
        "lower_body_joint_motion_rad": 0.0,
        "upper_body_joint_drift_rad": 0.0,
        "upper_body_target_error_rad": 0.0,
        "upper_body_joint_velocity_rad_s": 0.0,
        "assembly_position_drift_m": 0.0,
        "assembly_rotation_drift_rad": 0.0,
    }
    for _ in range(args_cli.stability_steps):
        _step_target(env, hold_target, zero_action)
        table_pos, table_quat = _table_state(task, env)
        workbench_pos, workbench_quat = _workbench_state(task, env)
        root_pos, root_quat = _root_state(robot)
        lower = as_torch(robot.data.joint_pos)[:, lower_joint_ids]
        body = as_torch(robot.data.joint_pos)[:, body_joint_ids]
        body_velocity = as_torch(robot.data.joint_vel)[:, body_joint_ids]
        assembly_pos_drift, assembly_rot_drift = _max_assembly_drift(
            task, env, assembly_ref_pos, assembly_ref_rot
        )
        stability_max["table_translation_m"] = max(
            stability_max["table_translation_m"],
            float(torch.linalg.norm(table_pos - table_ref_pos, dim=-1).max()),
        )
        stability_max["table_rotation_rad"] = max(
            stability_max["table_rotation_rad"],
            float(_quat_angle_rad(table_quat, table_ref_quat).max()),
        )
        stability_max["workbench_translation_m"] = max(
            stability_max["workbench_translation_m"],
            float(torch.linalg.norm(workbench_pos - workbench_ref_pos, dim=-1).max()),
        )
        stability_max["workbench_rotation_rad"] = max(
            stability_max["workbench_rotation_rad"],
            float(_quat_angle_rad(workbench_quat, workbench_ref_quat).max()),
        )
        roll, pitch = _xyzw_roll_pitch(root_quat)
        stability_max["robot_root_xy_drift_m"] = max(
            stability_max["robot_root_xy_drift_m"],
            float(torch.linalg.norm(root_pos[:, :2] - root_ref_pos[:, :2], dim=-1).max()),
        )
        stability_max["robot_root_height_error_m"] = max(
            stability_max["robot_root_height_error_m"],
            float(torch.abs(root_pos[:, 2] - 0.74).max()),
        )
        stability_max["robot_root_abs_roll_rad"] = max(
            stability_max["robot_root_abs_roll_rad"], float(torch.abs(roll).max())
        )
        stability_max["robot_root_abs_pitch_rad"] = max(
            stability_max["robot_root_abs_pitch_rad"], float(torch.abs(pitch).max())
        )
        stability_max["lower_body_joint_motion_rad"] = max(
            stability_max["lower_body_joint_motion_rad"],
            float(torch.abs(lower - lower_ref).max()),
        )
        stability_max["upper_body_joint_drift_rad"] = max(
            stability_max["upper_body_joint_drift_rad"],
            float(torch.abs(body - body_ref).max()),
        )
        stability_max["upper_body_target_error_rad"] = max(
            stability_max["upper_body_target_error_rad"],
            float(torch.abs(body - hold_target[:, :14]).max()),
        )
        stability_max["upper_body_joint_velocity_rad_s"] = max(
            stability_max["upper_body_joint_velocity_rad_s"],
            float(torch.abs(body_velocity).max()),
        )
        stability_max["assembly_position_drift_m"] = max(
            stability_max["assembly_position_drift_m"], assembly_pos_drift
        )
        stability_max["assembly_rotation_drift_rad"] = max(
            stability_max["assembly_rotation_drift_rad"], assembly_rot_drift
        )
    stability = {
        "steps": args_cli.stability_steps,
        "duration_s": args_cli.stability_steps * float(env.step_dt),
        "max": stability_max,
        "pass": (
            stability_max["table_translation_m"] <= 0.005
            and stability_max["table_rotation_rad"] <= math.radians(2.0)
            and stability_max["workbench_translation_m"] <= 1.0e-5
            and stability_max["workbench_rotation_rad"] <= 1.0e-5
            and stability_max["robot_root_xy_drift_m"] <= 0.20
            and stability_max["robot_root_height_error_m"] <= 0.08
            and stability_max["robot_root_abs_roll_rad"] <= math.radians(15.0)
            and stability_max["robot_root_abs_pitch_rad"] <= math.radians(15.0)
            and math.isfinite(stability_max["lower_body_joint_motion_rad"])
            and stability_max["upper_body_joint_drift_rad"] <= 0.15
            and stability_max["upper_body_joint_velocity_rad_s"] <= 0.50
            and stability_max["assembly_position_drift_m"] <= 0.002
            and stability_max["assembly_rotation_drift_rad"] <= math.radians(1.0)
        ),
    }
    initial_success_components = task._stable_flip_success_components(env)
    success_contract = {
        "normal_dot_threshold": float(
            os.environ.get("FLIP_TABLE_SUCCESS_DOT_THRESHOLD", "-0.95")
        ),
        "min_tabletop_lift_m": float(
            os.environ.get("FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M", "0.35")
        ),
        "max_linear_speed_m_s": float(
            os.environ.get("FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S", "0.15")
        ),
        "max_angular_speed_rad_s": float(
            os.environ.get("FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S", "0.50")
        ),
        "workbench_edge_margin_m": float(
            os.environ.get("FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M", "0.03")
        ),
        "hold_steps": int(os.environ.get("FLIP_TABLE_SUCCESS_HOLD_STEPS", "20")),
        "initial_pose_components": _jsonable(initial_success_components),
    }
    success_contract["pass"] = (
        success_contract["normal_dot_threshold"] <= -0.90
        and success_contract["min_tabletop_lift_m"] >= 0.30
        and success_contract["max_linear_speed_m_s"] <= 0.20
        and success_contract["max_angular_speed_rad_s"] <= 0.75
        and success_contract["workbench_edge_margin_m"] >= 0.0
        and success_contract["hold_steps"] >= 10
        and not bool(initial_success_components["candidate"].any())
    )

    nominal_target, nominal_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    actuator_effort_saturation = _explicit_effort_saturation_contract(
        env,
        robot,
        body_joint_ids,
        nominal_target,
        zero_action,
        runtime_paths["generated_dex1_urdf"],
    )
    half_delta = torch.tensor(
        [
            0.05,
            -0.05,
            0.05,
            0.08,
            -0.08,
            0.08,
            -0.08,
            0.08,
            -0.08,
            0.08,
            -0.08,
            0.08,
            -0.08,
            0.08,
        ],
        dtype=nominal_target.dtype,
        device=env.device,
    ).unsqueeze(0)
    positive_target = nominal_target.clone()
    negative_target = nominal_target.clone()
    positive_target[:, :14] += half_delta
    negative_target[:, :14] -= half_delta
    soft_limits = as_torch(robot.data.soft_joint_pos_limits)[:, body_joint_ids]
    for target in (positive_target, negative_target):
        target[:, :14] = torch.maximum(
            torch.minimum(target[:, :14], soft_limits[..., 1]),
            soft_limits[..., 0],
        )

    def run_body_trial(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        _reset_and_hold(
            env,
            zero_action,
            args_cli.equilibrium_steps,
            target=nominal_target,
        )
        echo_error = 0.0
        for _ in range(args_cli.joint_response_steps):
            _step_target(env, target, zero_action)
            echo_error = max(
                echo_error,
                float(torch.abs(last_commanded_target(env) - target).max()),
            )
        return (
            dataset_joint_state(env).detach().clone(),
            as_torch(robot.data.joint_vel)[:, body_joint_ids].detach().clone(),
            echo_error,
        )

    positive_state, positive_velocity, positive_echo_error = run_body_trial(
        positive_target
    )
    negative_state, negative_velocity, negative_echo_error = run_body_trial(
        negative_target
    )
    requested_span = positive_target[:, :14] - negative_target[:, :14]
    actual_span = positive_state[:, 3:17] - negative_state[:, 3:17]
    meaningful = torch.abs(requested_span) >= 0.02
    sign_match = torch.sign(actual_span[meaningful]) == torch.sign(
        requested_span[meaningful]
    )
    response_gain = actual_span[meaningful] / requested_span[meaningful]
    positive_tracking_error = positive_state[:, 3:17] - positive_target[:, :14]
    negative_tracking_error = negative_state[:, 3:17] - negative_target[:, :14]
    max_tracking_error = max(
        float(torch.abs(positive_tracking_error).max()),
        float(torch.abs(negative_tracking_error).max()),
    )
    max_command_echo_error = max(positive_echo_error, negative_echo_error)
    body_response = {
        "joint_names": list(ARM_JOINT_NAMES),
        "equilibrium_steps": args_cli.equilibrium_steps,
        "response_steps": args_cli.joint_response_steps,
        "nominal_target": _tensor_list(nominal_target[:, :14]),
        "nominal_equilibrium_state": _tensor_list(nominal_state[:, 3:17]),
        "nominal_equilibrium_error": _tensor_list(
            nominal_state[:, 3:17] - nominal_target[:, :14]
        ),
        "positive_target": _tensor_list(positive_target[:, :14]),
        "positive_state": _tensor_list(positive_state[:, 3:17]),
        "positive_final_velocity_rad_s": _tensor_list(positive_velocity),
        "positive_tracking_error": _tensor_list(positive_tracking_error),
        "negative_target": _tensor_list(negative_target[:, :14]),
        "negative_state": _tensor_list(negative_state[:, 3:17]),
        "negative_final_velocity_rad_s": _tensor_list(negative_velocity),
        "negative_tracking_error": _tensor_list(negative_tracking_error),
        "requested_span": _tensor_list(requested_span),
        "actual_span": _tensor_list(actual_span),
        "response_gain": _tensor_list(response_gain),
        "min_response_gain": float(response_gain.min()),
        "max_response_gain": float(response_gain.max()),
        "max_abs_tracking_error_rad": max_tracking_error,
        "sign_match_fraction": float(sign_match.float().mean()),
        "max_command_echo_error": max_command_echo_error,
        "pass": (
            max_command_echo_error <= 1.0e-5
            and bool(torch.all(sign_match))
            and float(response_gain.min()) >= 0.20
            and max_tracking_error <= 0.25
        ),
    }

    hand_baseline, _hand_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    hand_target_a = hand_baseline.clone()
    hand_target_a[:, 14] = 0.0
    hand_target_a[:, 15] = 4.5
    for _ in range(args_cli.joint_response_steps):
        _step_target(env, hand_target_a, zero_action)
    hand_state_a = dataset_joint_state(env)
    _reset_and_hold(
        env,
        zero_action,
        args_cli.equilibrium_steps,
        target=hand_baseline,
    )
    hand_target_b = hand_baseline.clone()
    hand_target_b[:, 14] = 4.5
    hand_target_b[:, 15] = 0.0
    for _ in range(args_cli.joint_response_steps):
        _step_target(env, hand_target_b, zero_action)
    hand_state_b = dataset_joint_state(env)
    hand_response = {
        "target_open_left_closed_right": _tensor_list(hand_target_a[:, 14:16]),
        "state_open_left_closed_right": _tensor_list(hand_state_a[:, 17:19]),
        "target_closed_left_open_right": _tensor_list(hand_target_b[:, 14:16]),
        "state_closed_left_open_right": _tensor_list(hand_state_b[:, 17:19]),
        "max_command_error": max(
            float(torch.abs(hand_state_a[:, 17:19] - hand_target_a[:, 14:16]).max()),
            float(torch.abs(hand_state_b[:, 17:19] - hand_target_b[:, 14:16]).max()),
        ),
    }
    hand_response["pass"] = hand_response["max_command_error"] <= 0.25

    wrench_target, _wrench_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    table_entity = task._find_scene_entity(env, "Table001_Table001_01")
    if table_entity is None:
        raise RuntimeError("white tabletop scene entity was not found")
    contact_material_bindings = _contact_material_snapshot(task, env)
    force_start_pos, force_start_quat = _table_state(task, env)
    force_assembly_pos, force_assembly_rot = _assembly_frame(task, env)
    force_workbench_pos, force_workbench_quat = _workbench_state(task, env)
    forces = torch.zeros((env.num_envs, 1, 3), dtype=torch.float32, device=env.device)
    torques = torch.zeros_like(forces)
    forces[:, :, 2] = args_cli.upward_force_n
    table_entity.permanent_wrench_composer.reset()
    table_entity.permanent_wrench_composer.add_forces_and_torques(
        forces, torques, is_global=True
    )
    max_height_delta = 0.0
    for _ in range(args_cli.wrench_steps):
        _step_target(env, wrench_target, zero_action)
        table_pos, _table_quat = _table_state(task, env)
        max_height_delta = max(
            max_height_delta, float((table_pos[:, 2] - force_start_pos[:, 2]).max())
        )
    table_entity.permanent_wrench_composer.reset()
    force_final_pos, force_final_quat = _table_state(task, env)
    force_assembly_pos_drift, force_assembly_rot_drift = _max_assembly_drift(
        task, env, force_assembly_pos, force_assembly_rot
    )
    force_final_workbench_pos, force_final_workbench_quat = _workbench_state(task, env)
    force_response = {
        "upward_force_n": args_cli.upward_force_n,
        "steps": args_cli.wrench_steps,
        "max_height_delta_m": max_height_delta,
        "final_translation_m": float(
            torch.linalg.norm(force_final_pos - force_start_pos, dim=-1).max()
        ),
        "final_rotation_rad": float(
            _quat_angle_rad(force_final_quat, force_start_quat).max()
        ),
        "assembly_position_drift_m": force_assembly_pos_drift,
        "assembly_rotation_drift_rad": force_assembly_rot_drift,
        "workbench_translation_m": float(
            torch.linalg.norm(
                force_final_workbench_pos - force_workbench_pos, dim=-1
            ).max()
        ),
        "workbench_rotation_rad": float(
            _quat_angle_rad(force_final_workbench_quat, force_workbench_quat).max()
        ),
    }
    force_response["pass"] = (
        force_response["max_height_delta_m"] >= 0.01
        and force_response["assembly_position_drift_m"] <= 0.002
        and force_response["assembly_rotation_drift_rad"] <= math.radians(1.0)
        and force_response["workbench_translation_m"] <= 1.0e-5
    )

    torque_target, _torque_state = _reset_and_hold(
        env, zero_action, args_cli.equilibrium_steps
    )
    torque_start_pos, torque_start_quat = _table_state(task, env)
    torque_assembly_pos, torque_assembly_rot = _assembly_frame(task, env)
    torque_workbench_pos, torque_workbench_quat = _workbench_state(task, env)
    forces.zero_()
    torques.zero_()
    # Lift the tabletop clear of support friction while checking angular
    # dynamics. A torque-only test on the workbench correctly remains static
    # below the friction moment and therefore cannot validate free rotation.
    forces[:, :, 2] = args_cli.upward_force_n
    torques[:, :, 0] = args_cli.torque_nm
    table_entity.permanent_wrench_composer.reset()
    table_entity.permanent_wrench_composer.add_forces_and_torques(
        forces, torques, is_global=True
    )
    max_rotation = 0.0
    for _ in range(args_cli.wrench_steps):
        _step_target(env, torque_target, zero_action)
        _table_pos, table_quat = _table_state(task, env)
        max_rotation = max(
            max_rotation, float(_quat_angle_rad(table_quat, torque_start_quat).max())
        )
    table_entity.permanent_wrench_composer.reset()
    torque_final_pos, torque_final_quat = _table_state(task, env)
    torque_assembly_pos_drift, torque_assembly_rot_drift = _max_assembly_drift(
        task, env, torque_assembly_pos, torque_assembly_rot
    )
    torque_final_workbench_pos, torque_final_workbench_quat = _workbench_state(task, env)
    torque_response = {
        "torque_nm": args_cli.torque_nm,
        "support_unloading_force_n": args_cli.upward_force_n,
        "axis_world": "x",
        "steps": args_cli.wrench_steps,
        "max_rotation_rad": max_rotation,
        "final_rotation_rad": float(
            _quat_angle_rad(torque_final_quat, torque_start_quat).max()
        ),
        "final_translation_m": float(
            torch.linalg.norm(torque_final_pos - torque_start_pos, dim=-1).max()
        ),
        "assembly_position_drift_m": torque_assembly_pos_drift,
        "assembly_rotation_drift_rad": torque_assembly_rot_drift,
        "workbench_translation_m": float(
            torch.linalg.norm(
                torque_final_workbench_pos - torque_workbench_pos, dim=-1
            ).max()
        ),
        "workbench_rotation_rad": float(
            _quat_angle_rad(torque_final_workbench_quat, torque_workbench_quat).max()
        ),
    }
    torque_response["pass"] = (
        torque_response["max_rotation_rad"] >= math.radians(1.0)
        and torque_response["assembly_position_drift_m"] <= 0.002
        and torque_response["assembly_rotation_drift_rad"] <= math.radians(1.0)
        and torque_response["workbench_translation_m"] <= 1.0e-5
    )

    low_friction = _friction_trial(
        env,
        task,
        table_entity,
        zero_action,
        static_friction=0.20,
        dynamic_friction=0.15,
    )
    high_friction = _friction_trial(
        env,
        task,
        table_entity,
        zero_action,
        static_friction=0.85,
        dynamic_friction=0.70,
    )
    os.environ["FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS"] = "false"
    low_displacement = low_friction["max_horizontal_displacement_m"]
    high_displacement = high_friction["max_horizontal_displacement_m"]
    friction_response = {
        "method": "same horizontal wrench with exact low/high contact coefficients",
        "low_friction": low_friction,
        "high_friction": high_friction,
        "displacement_ratio_low_over_high": low_displacement
        / max(high_displacement, 1.0e-9),
    }
    friction_response["pass"] = (
        low_friction["material_snapshot"]["pass"]
        and high_friction["material_snapshot"]["pass"]
        and low_displacement >= 0.015
        and high_displacement <= 0.005
        and low_displacement >= high_displacement + 0.010
        and low_friction["assembly_position_drift_m"] <= 0.002
        and high_friction["assembly_position_drift_m"] <= 0.002
        and low_friction["workbench_translation_m"] <= 1.0e-5
        and high_friction["workbench_translation_m"] <= 1.0e-5
    )

    masses = {}
    for entity_name in (
        "Table001_Table001_01",
        "Leg001_Leg001",
        "Leg001_01_Leg001",
        "Leg001_03_Leg001",
        "Leg001_06_Leg001",
        "Table278_Table278",
    ):
        try:
            mass = as_torch(env.scene[entity_name].data.body_mass).reshape(env.num_envs, -1)
        except KeyError:
            continue
        masses[entity_name] = _tensor_list(mass[:, 0])

    contact_partner_filters = _contact_partner_filter_contract(env)

    gates = {
        "runtime_identity_and_actuators": bool(runtime_identity["pass"]),
        "prepared_scene_starts_motionless": bool(prepared_scene_velocity["pass"]),
        "reset_clears_prior_actuator_targets": bool(reset_actuator_targets["pass"]),
        "wbc_actuator_ownership_and_solver": bool(wbc_actuators["pass"]),
        "explicit_pd_effort_saturation": bool(actuator_effort_saturation["pass"]),
        "timebase": bool(timebase["pass"]),
        "scene_geometry_and_reset_pose": bool(scene_geometry["pass"]),
        "randomized_reset_invariants": bool(randomized_resets["pass"]),
        "static_stability": bool(stability["pass"]),
        "complete_flip_success_contract": bool(success_contract["pass"]),
        "body_action_mapping_and_tracking": bool(body_response["pass"]),
        "hand_action_mapping_and_tracking": bool(hand_response["pass"]),
        "white_leg_contact_partner_filters": bool(contact_partner_filters["pass"]),
        "contact_material_bindings": bool(contact_material_bindings["pass"]),
        "contact_material_friction_response": bool(friction_response["pass"]),
        "assembled_table_force_response": bool(force_response["pass"]),
        "assembled_table_torque_response": bool(torque_response["pass"]),
    }
    report = {
        "schema_version": "team_ramen_flip_table_simulation_contract_audit_v13",
        "purpose": "policy-independent simulator and control-path validation",
        "privileged_state_use": "diagnostics only; never a policy, critic, planner, or inference input",
        "seed": int(args_cli.seed),
        "action_dim": int(env.action_manager.total_action_dim),
        "action_terms": list(env.action_manager.active_terms),
        "controlled_joint_names": list(ARM_JOINT_NAMES) + ["left_dex1", "right_dex1"],
        "balance_owner": "organizer G1 decoupled WBC (root, legs, waist)",
        "wbc_owned_joint_names": list(WBC_OWNED_JOINT_NAMES),
        "runtime_identity": runtime_identity,
        "prepared_scene_velocity": prepared_scene_velocity,
        "reset_actuator_targets": reset_actuator_targets,
        "wbc_actuators": wbc_actuators,
        "actuator_effort_saturation": actuator_effort_saturation,
        "timebase": timebase,
        "scene_geometry": scene_geometry,
        "randomized_resets": randomized_resets,
        "masses_kg": masses,
        "static_stability": stability,
        "complete_flip_success_contract": success_contract,
        "body_action_response": body_response,
        "hand_action_response": hand_response,
        "contact_partner_filters": contact_partner_filters,
        "contact_material_bindings": contact_material_bindings,
        "contact_material_friction_response": friction_response,
        "force_response": force_response,
        "torque_response": torque_response,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
