#!/usr/bin/env python3
"""Run audited flip-table generation in the pinned RoboFinals V1 runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _validate_source_contract(path: Path) -> None:
    """Reject sim-teleop HDF5 before the GT-dependent Mimic retargeter runs."""

    import h5py

    with h5py.File(path, "r") as stream:
        source_kind = str(stream.attrs.get("source_kind", ""))
        retargeting = str(stream.attrs.get("mimic_retargeting", ""))
        if source_kind != "successful_sim_avp_teleoperation":
            return
        if retargeting != "blocked_pending_rgb_pose_adapter":
            raise ValueError(
                "sim teleop source lacks the required no-GT Mimic contract; "
                "re-export it with export_teleop_mimic_source.py"
            )
        if any(
            "object_pose" in demo.get("obs", {}).get("datagen_info", {})
            for demo in stream["data"].values()
        ):
            raise ValueError("sim teleop HDF5 must not contain simulator object poses")
        raise ValueError(
            "Isaac Lab Mimic retargeting is blocked for sim teleop sources: "
            "the stock generator would consume simulator object pose. Supply "
            "the audited RGB-only pose adapter before generation."
        )


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
config_argument = parser.add_argument("--config", type=Path)
AppLauncher.add_app_launcher_args(parser)
config_argument.required = True
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--output-file", type=Path, required=True)
parser.add_argument("--ledger-root", type=Path, required=True)
parser.add_argument("--runtime-manifest", type=Path, required=True)
parser.add_argument("--run-manifest", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--num-trials", type=int, required=True)
parser.add_argument("--start-attempt-index", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--keep-failed-debug",
    action="store_true",
    help=(
        "Write rejected trajectories to a separate *_failed.hdf5 file for diagnosis. "
        "This never makes a rejected candidate eligible for rendering or export."
    ),
)
parser.add_argument(
    "--scene",
    type=Path,
    default=Path("/workspace/IROS_IKEA_V13_20260702/Scene02_flip_table_assembled.usd"),
)
parser.add_argument(
    "--urdf",
    type=Path,
    default=Path(
        "/workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/"
        "g1_29dof_with_hand.urdf"
    ),
)
args = parser.parse_args()

# Fail before AppLauncher creates an Isaac Sim process. This is intentionally
# before the heavyweight runtime imports below: an unsafe source is a contract
# error, not a GPU-runtime error.
if args.input_file.is_file():
    _validate_source_contract(args.input_file)

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import DatasetExportMode

from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.place_utils.env_utils import set_seed

from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.mimic.env_cfg import (
    configure_mimic_env_cfg,
    promote_to_mimic_env_cfg,
)
from data.flip_table_data_augmentation.mimic.generation_runtime import run_generation
from data.flip_table_data_augmentation.mimic.recorders import FlipTableRecorderManagerCfg
from data.flip_table_data_augmentation.mimic.source_hdf5 import MIMIC_ENV_NAME
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from data.flip_table_data_augmentation.provenance import CandidateLedger
from data.flip_table_data_augmentation.runtime_contract import verify_runtime_manifest


OFFICIAL_V1_G1_ROOT_HEIGHT_M = 0.78


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_success_contract_environment(success: dict) -> None:
    values = {
        "FLIP_TABLE_SUCCESS_DOT_THRESHOLD": success["normal_dot_max"],
        "FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M": success["tabletop_lift_m_min"],
        "FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S": success["settled_linear_speed_m_s_max"],
        "FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S": success["settled_angular_speed_rad_s_max"],
        "FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M": success["workbench_edge_margin_m_min"],
        "FLIP_TABLE_SUCCESS_HOLD_STEPS": success["hold_steps_min"],
        "FLIP_TABLE_LOCK_LOWER_BODY": "true",
        "FLIP_TABLE_LOCK_ROBOT_ROOT": "true",
        "FLIP_TABLE_RL_STAGE": "full",
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def _range_text(values) -> str:
    return ",".join(format(float(value), ".17g") for value in values)


def _set_physical_randomization_environment(config) -> None:
    physical = config.physical_randomization
    scalar_values = {
        # Keep generation explicit about the official RoboFinals V1 root height.
        "FLIP_TABLE_ROBOT_BASE_HEIGHT_M": OFFICIAL_V1_G1_ROOT_HEIGHT_M,
        "FLIP_TABLE_TABLE_LONG_RANGE_M": physical.table_long_range_m,
        "FLIP_TABLE_TABLE_DEPTH_RANGE_M": physical.table_depth_range_m,
        "FLIP_TABLE_TABLE_YAW_RANGE_RAD": physical.table_yaw_range_rad,
        "FLIP_TABLE_ROBOT_DISTANCE_M": physical.robot_distance_m,
        "FLIP_TABLE_ROBOT_DISTANCE_RANGE_M": physical.robot_distance_range_m,
        "FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M": physical.robot_table_min_distance_m,
        "FLIP_TABLE_ROBOT_LATERAL_RANGE_M": physical.robot_lateral_range_m,
        "FLIP_TABLE_ROBOT_YAW_RANGE_RAD": physical.robot_yaw_range_rad,
        "FLIP_TABLE_JOINT_NOISE_RAD": physical.upper_body_joint_noise_rad,
        "FLIP_TABLE_DEX1_FINGER_NOISE_M": physical.dex1_finger_noise_m,
        "FLIP_TABLE_RL_RANDOMIZATION_LEVEL": 1.0,
    }
    for name, value in scalar_values.items():
        os.environ[name] = format(value, ".17g")

    contacts = physical.contact_materials
    contact_prefixes = {
        "hand_white_table": "FLIP_TABLE_CONTACT_HAND_WHITE",
        "white_table_workbench": "FLIP_TABLE_CONTACT_WHITE_WORKBENCH",
        "workbench_hand": "FLIP_TABLE_CONTACT_WORKBENCH_HAND",
    }
    for key, prefix in contact_prefixes.items():
        material = contacts[key]
        os.environ[f"{prefix}_STATIC_RANGE"] = _range_text(material.static_friction)
        os.environ[f"{prefix}_DYNAMIC_RANGE"] = _range_text(material.dynamic_friction)
        os.environ[f"{prefix}_RESTITUTION_RANGE"] = _range_text(material.restitution)

    required_flags = {
        "FLIP_TABLE_PREPARE_ASSEMBLED_SCENE": "true",
        "FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE": "true",
        "FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS": "true",
        "FLIP_TABLE_LOCK_LOWER_BODY": "true",
        "FLIP_TABLE_LOCK_ROBOT_ROOT": "true",
        "FLIP_TABLE_FIX_ROOT_LINK": "true",
        "FLIP_TABLE_RL_RANDOMIZE_MASS": "false",
        "FLIP_TABLE_EVAL_RANDOMIZE_MASS": "false",
        # Appearance is sampled only during accepted-trajectory replay.
        "FLIP_TABLE_RANDOMIZE_ROOM": "false",
        "FLIP_TABLE_RANDOMIZE_ROOM_PROPS": "false",
        "FLIP_TABLE_RANDOMIZE_LIGHTING": "false",
        "FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS": "false",
        "FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY": "false",
    }
    os.environ.update(required_flags)


def _validate_args() -> None:
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.start_attempt_index < 0 or args.seed < 0:
        raise ValueError("attempt index and seed must be non-negative")
    if not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)
    if not args.runtime_manifest.is_file():
        raise FileNotFoundError(args.runtime_manifest)
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if not args.urdf.is_file():
        raise FileNotFoundError(args.urdf)
    if args.output_file.suffix != ".hdf5":
        raise ValueError("--output-file must end in .hdf5")
    if args.output_file.exists() and not args.output_file.is_file():
        raise ValueError(f"--output-file exists but is not a file: {args.output_file}")
    if args.keep_failed_debug and args.num_trials != 1:
        raise ValueError("--keep-failed-debug is restricted to one diagnostic trial")
    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in args.run_id):
        raise ValueError("--run-id must use lowercase letters, digits, dot, underscore, or hyphen")


def _make_env(config):
    env_cfg = parse_env_cfg(
        scene_backend="local",
        task_backend="local",
        task_name="AssembleTableTask",
        robot_name="G1-Gripper-Controller-DecoupledWBC",
        scene_name=str(args.scene.resolve()),
        # Mimic supplies absolute EEF targets. The residual-RL profile uses an
        # incompatible 23-D action term, so this path selects PINK directly.
        rl_name=None,
        robot_scale=1.0,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        first_person_view=True,
        enable_cameras=False,
        execute_mode=ExecuteMode.TRAIN,
        usd_simplify=False,
        seed=args.seed,
        sources=None,
        object_projects=None,
        headless_mode=args.headless,
        enable_full_local_scene=True,
    )
    env_cfg = promote_to_mimic_env_cfg(env_cfg)
    configure_mimic_env_cfg(
        env_cfg,
        config,
        source_dataset_path=str(args.input_file.resolve()),
        generation_path=str(args.output_file.resolve()),
        generation_num_trials=args.num_trials,
        source_demo_count=_source_demo_count(args.input_file),
        guarantee_success=False,
        generation_seed=args.seed,
    )
    env_cfg.seed = args.seed
    env_cfg.env_name = MIMIC_ENV_NAME
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False
    recorder = FlipTableRecorderManagerCfg()
    recorder.dataset_export_dir_path = str(args.output_file.resolve().parent)
    recorder.dataset_filename = args.output_file.stem
    recorder.dataset_compression = True
    recorder.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        if args.keep_failed_debug
        else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    env_cfg.recorders = recorder
    if MIMIC_ENV_NAME not in gym.registry:
        gym.register(
            id=MIMIC_ENV_NAME,
            entry_point="data.flip_table_data_augmentation.mimic.env:FlipTableMimicEnv",
            kwargs={},
            disable_env_checker=True,
        )
    env = gym.make(MIMIC_ENV_NAME, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise TypeError("registered environment is not ManagerBasedRLMimicEnv")
    env.validate_runtime_action_contract()
    return env, success_term


def _source_demo_count(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as stream:
        return len(stream["data"])


def main() -> None:
    _validate_args()
    config = load_pipeline_config(args.config)
    _runtime_audit, runtime_manifest_sha256 = verify_runtime_manifest(
        args.runtime_manifest,
        config,
    )
    _set_success_contract_environment(config.raw["success"])
    _set_physical_randomization_environment(config)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    ledger = CandidateLedger(args.ledger_root)
    manifest = {
        "schema_version": "team_ramen_flip_table_mimic_run/v1",
        "status": "running",
        "run_id": args.run_id,
        "started_at_utc": _utc_now(),
        "config_path": str(args.config.resolve()),
        "config_sha256": config.digest,
        "runtime_digest": runtime_manifest_sha256,
        "container_digest": config.runtime.container_digest,
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "source_hdf5": str(args.input_file.resolve()),
        "source_hdf5_sha256": sha256_file(args.input_file),
        "output_hdf5": str(args.output_file.resolve()),
        "ledger_root": str(args.ledger_root.resolve()),
        "scene": str(args.scene.resolve()),
        "scene_sha256": sha256_file(args.scene),
        "base_seed": args.seed,
        "start_attempt_index": args.start_attempt_index,
        "requested_trials": args.num_trials,
        "num_envs": 1,
        "cameras_enabled_during_physics": False,
        "resumed_existing_hdf5": args.output_file.exists(),
        "failed_debug_capture": args.keep_failed_debug,
        "official_v1_robot_root_height_m": OFFICIAL_V1_G1_ROOT_HEIGHT_M,
        "failed_debug_hdf5": (
            str(args.output_file.with_name(f"{args.output_file.stem}_failed.hdf5").resolve())
            if args.keep_failed_debug
            else None
        ),
    }
    atomic_write_json(args.run_manifest.resolve(), manifest)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    env = None
    try:
        env, success_term = _make_env(config)
        set_seed(args.seed, env)
        env.reset()
        result = run_generation(
            env=env,
            input_file=args.input_file,
            success_term=success_term,
            ledger=ledger,
            run_id=args.run_id,
            output_shard=args.output_file.name,
            start_attempt_index=args.start_attempt_index,
            attempt_count=args.num_trials,
            base_seed=args.seed,
            config_sha256=config.digest,
            runtime_digest=runtime_manifest_sha256,
            urdf_path=args.urdf,
            action_fk_contract=config.raw["source_contract"],
        )
        manifest.update(
            {
                "status": "finished",
                "finished_at_utc": _utc_now(),
                "result": result,
                "ledger_manifest_sha256": ledger.manifest_digest(),
            }
        )
        atomic_write_json(args.run_manifest.resolve(), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(args.run_manifest.resolve(), manifest)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
