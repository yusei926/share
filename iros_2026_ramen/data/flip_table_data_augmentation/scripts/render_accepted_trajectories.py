#!/usr/bin/env python3
"""Replay accepted V1 trajectories and render deterministic Replicator variants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
config_argument = parser.add_argument("--config", type=Path)
AppLauncher.add_app_launcher_args(parser)
config_argument.required = True
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--ledger-root", type=Path, required=True)
parser.add_argument("--runtime-manifest", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--candidate-id", action="append", default=[])
parser.add_argument("--candidate-limit", type=int)
parser.add_argument("--variants", type=int)
parser.add_argument(
    "--scene",
    type=Path,
    default=Path("/workspace/IROS_IKEA_V13_20260702/Scene02_flip_table_assembled.usd"),
)
parser.add_argument(
    "--room-assets",
    type=Path,
    default=Path("/workspace/flip_table_room_assets"),
)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from isaaclab.envs import ManagerBasedRLEnv

from robofinals.utils.env import ExecuteMode, parse_env_cfg
from robofinals.utils.isaac_data_compat import as_torch

from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.export.contracts import TASK, RenderedEpisode
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from data.flip_table_data_augmentation.provenance import (
    AppearanceRecord,
    CandidateLedger,
)
from data.flip_table_data_augmentation.replicator.appearance import (
    AppearanceController,
    appearance_seed,
    configure_appearance_environment,
)
from data.flip_table_data_augmentation.replicator.camera_contract import (
    apply_camera_contract,
    verify_runtime_camera_contract,
)
from data.flip_table_data_augmentation.replicator.camera_image import (
    apply_recorded_camera_geometry,
)
from data.flip_table_data_augmentation.replicator.trajectory import (
    inspect_accepted_trajectory,
    read_numeric_trace,
    read_state_at,
    sample_indices,
    write_numeric_parquet,
)
from data.flip_table_data_augmentation.runtime_contract import verify_runtime_manifest


def _make_env(config):
    env_cfg = parse_env_cfg(
        scene_backend="local",
        task_backend="local",
        task_name="AssembleTableTask",
        robot_name="G1-Gripper-Controller-DecoupledWBC",
        scene_name=str(args.scene.resolve()),
        # State replay does not execute a policy. Loading the residual-RL
        # action config would require an unrelated demo-action file and can
        # overwrite the recorded state during environment setup.
        rl_name=None,
        robot_scale=1.0,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        replay_cfgs={
            "hdf5_path": str(args.input_file.resolve()),
            "ep_meta": {},
            "ep_names": [],
            "add_camera_to_observation": True,
            "render_resolution": [640, 480],
        },
        first_person_view=True,
        enable_cameras=True,
        # Organizer V1 instantiates the G1 policy cameras only in EVAL mode.
        # States are still applied directly below; no policy action is run.
        execute_mode=ExecuteMode.EVAL,
        usd_simplify=False,
        seed=0,
        sources=None,
        object_projects=None,
        headless_mode=args.headless,
        enable_full_local_scene=True,
    )
    env_name = "RoboFinals-FlipTable-Augmentation-Replay-v0"
    env_cfg.env_name = env_name
    env_cfg.terminations = None
    env_cfg.recorders = None
    env_cfg.isaaclab_arena_env.embodiment.active_observation_camera_names = [
        camera.sim_sensor for camera in config.cameras
    ]
    apply_camera_contract(env_cfg, config)
    if env_name in gym.registry:
        gym.registry.pop(env_name)
    gym.register(
        id=env_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={},
        disable_env_checker=True,
    )
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLEnv):
        raise TypeError("replay environment is not ManagerBasedRLEnv")
    env.reset()
    return env


def _state_to_torch(value, device):
    if isinstance(value, dict):
        return {key: _state_to_torch(item, device) for key, item in value.items()}
    array = np.asarray(value)
    return torch.as_tensor(array[None], device=device)


def _apply_state(env, state) -> None:
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
    env.scene.reset_to(_state_to_torch(state, env.device), env_ids, is_relative=False)
    env.sim.forward()


def _capture_rgb(env, camera, output: Path) -> None:
    sensor = env.scene[camera.sim_sensor]
    sensor.update(0.0, force_recompute=True)
    image = as_torch(sensor.data.output["rgb"])[0].detach().cpu().numpy()
    if image.shape == (camera.height, camera.width, 4):
        image = image[:, :, :3]
    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise RuntimeError(
            f"{camera.sim_sensor} returned {image.shape}/{image.dtype}; expected raw uint8 RGB"
        )
    image = apply_recorded_camera_geometry(image, camera)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output, format="PNG", compress_level=3)


def _remove_stale_directory(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents or resolved == root:
        raise ValueError(f"refusing to clear path outside render output root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _existing_variant_is_valid(record, output_root: Path) -> bool:
    relative = record.payload.get("render_manifest")
    expected_hash = record.payload.get("render_manifest_sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        return False
    manifest = (output_root / relative).resolve()
    if output_root.resolve() not in manifest.parents or not manifest.is_file():
        return False
    if sha256_file(manifest) != expected_hash:
        return False
    RenderedEpisode.load(manifest)
    return True


def _render_variant(
    *,
    env,
    controller,
    config,
    ledger,
    trajectory,
    candidate,
    variant_index,
    runtime_manifest_sha256,
    camera_contract,
    output_root,
) -> None:
    seed = appearance_seed(config, trajectory.trajectory_seed, variant_index)
    variant = ledger.ensure_variant_claim(
        AppearanceRecord(
            candidate_id=trajectory.candidate_id,
            variant_index=variant_index,
            status="claimed",
            appearance_seed=seed,
            trajectory_sha256=trajectory.trajectory_sha256,
            config_sha256=config.digest,
            runtime_digest=runtime_manifest_sha256,
            payload={},
        )
    )
    if variant.status in {"rendered", "exported"}:
        if not _existing_variant_is_valid(variant, output_root):
            raise RuntimeError(
                f"ledger says rendered but output validation failed: "
                f"{trajectory.candidate_id}/variant-{variant_index:04d}"
            )
        return
    if variant.status == "rejected":
        raise RuntimeError(
            f"appearance variant is permanently rejected: {trajectory.candidate_id}/{variant_index}"
        )

    final_dir = output_root / trajectory.candidate_id / f"variant-{variant_index:04d}"
    temporary = final_dir.with_name(f".{final_dir.name}.incomplete")
    _remove_stale_directory(temporary, output_root)
    if final_dir.exists():
        manifest_path = final_dir / "manifest.json"
        rendered = RenderedEpisode.load(manifest_path)
        if (
            rendered.candidate_id != trajectory.candidate_id
            or rendered.appearance_variant != variant_index
            or rendered.trajectory_sha256 != trajectory.trajectory_sha256
            or rendered.config_sha256 != config.digest
        ):
            raise RuntimeError(f"existing render output has a different identity: {final_dir}")
        ledger.transition_variant(
            trajectory.candidate_id,
            variant_index,
            "rendered",
            {
                "render_manifest": manifest_path.relative_to(output_root).as_posix(),
                "render_manifest_sha256": sha256_file(manifest_path),
                "frame_count": rendered.frame_count,
            },
        )
        return
    temporary.mkdir(parents=True)

    indices = sample_indices(
        trajectory.source_frame_count,
        source_hz=int(config.raw["generation"]["mimic_control_hz"]),
        target_hz=config.cameras[0].fps,
    )
    _apply_state(env, read_state_at(trajectory, int(indices[0])))
    appearance = controller.apply(trajectory.trajectory_seed, variant_index)
    trace = read_numeric_trace(trajectory, indices)
    numeric_path = temporary / "numeric.parquet"
    numeric_sha256 = write_numeric_parquet(numeric_path, trace)
    camera_dirs = {}
    for camera in config.cameras:
        relative = Path("cameras") / camera.source_key.rsplit(".", 1)[-1]
        camera_dirs[camera.source_key] = relative

    for output_index, source_index in enumerate(indices.tolist()):
        _apply_state(env, read_state_at(trajectory, source_index))
        env.sim.render()
        for camera in config.cameras:
            _capture_rgb(
                env,
                camera,
                temporary / camera_dirs[camera.source_key] / f"frame_{output_index:06d}.png",
            )

    source_kind = (
        "sim_teleop"
        if all(index >= 1_000_000 for index in trajectory.source_episode_indices)
        else "real_demo"
    )
    if source_kind == "sim_teleop":
        source_lineage = "mimic:sim_teleop:" + ",".join(
            str(index) for index in trajectory.source_episode_indices
        )
    else:
        source_lineage = (
            f"mimic:{config.source.repo_id}@{config.source.revision}:"
            + ",".join(str(index) for index in trajectory.source_episode_indices)
        )
    acceptance = dict(candidate.payload["acceptance_report"])
    action_fk_report = candidate.payload.get("action_fk_report")
    if not isinstance(action_fk_report, dict) or action_fk_report.get("pass") is not True:
        raise RuntimeError("validated candidate lacks a passing action FK report")
    success_report = {
        **acceptance,
        "accepted": True,
        "strict_v1_contract": True,
        "rejection_reasons": [],
        "action_fk_report": action_fk_report,
    }
    randomization = {
        "physical": candidate.payload["physical_randomization"],
        "physical_ranges": asdict(config.physical_randomization),
        "appearance": appearance,
        "appearance_ranges": asdict(config.appearance_randomization),
        "camera_contract": camera_contract,
        "trajectory_sampling": {
            "source_hz": int(config.raw["generation"]["mimic_control_hz"]),
            "target_hz": config.cameras[0].fps,
            "source_frame_count": trajectory.source_frame_count,
            "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        },
    }
    manifest = {
        "schema_version": "team_ramen_flip_table_rendered_episode/v1",
        "candidate_id": trajectory.candidate_id,
        "trajectory_kind": "mimic",
        "source_kind": source_kind,
        "appearance_variant": variant_index,
        "source_episode_indices": list(trajectory.source_episode_indices),
        "source_trajectory_lineage": source_lineage,
        "frame_count": len(indices),
        "fps": config.cameras[0].fps,
        "task": TASK,
        "numeric_trace": "numeric.parquet",
        "numeric_trace_sha256": numeric_sha256,
        "cameras": {key: path.as_posix() for key, path in camera_dirs.items()},
        "trajectory_sha256": trajectory.trajectory_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "config_sha256": config.digest,
        "randomization": randomization,
        "success_report": success_report,
    }
    atomic_write_json(temporary / "manifest.json", manifest)
    RenderedEpisode.load(temporary / "manifest.json")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final_dir)
    manifest_path = final_dir / "manifest.json"
    ledger.transition_variant(
        trajectory.candidate_id,
        variant_index,
        "rendered",
        {
            "render_manifest": manifest_path.relative_to(output_root).as_posix(),
            "render_manifest_sha256": sha256_file(manifest_path),
            "frame_count": len(indices),
        },
    )


def main() -> None:
    config = load_pipeline_config(args.config)
    if not args.input_file.is_file() or not args.scene.is_file():
        raise FileNotFoundError("accepted HDF5 or assembled V1 scene is missing")
    if args.candidate_limit is not None and args.candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive")
    variants = args.variants or int(
        config.raw["generation"]["appearance_variants_per_trajectory_min"]
    )
    if variants < int(config.raw["generation"]["appearance_variants_per_trajectory_min"]):
        raise ValueError("--variants cannot weaken the configured appearance gate")
    _runtime_audit, runtime_manifest_sha256 = verify_runtime_manifest(
        args.runtime_manifest,
        config,
    )
    configure_appearance_environment(config, args.room_assets)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = CandidateLedger(args.ledger_root)
    requested = set(args.candidate_id)
    candidates = [
        record
        for record in ledger.list_records()
        if record.status in {"validated", "rendered"}
        and (not requested or record.candidate_id in requested)
    ]
    if requested.difference(record.candidate_id for record in candidates):
        raise ValueError("one or more requested candidates are absent or not validated")
    if args.candidate_limit is not None:
        candidates = candidates[: args.candidate_limit]
    if not candidates:
        raise ValueError("no validated candidates are available to render")

    env = None
    try:
        env = _make_env(config)
        camera_contract = verify_runtime_camera_contract(env, config)
        controller = AppearanceController(env, config)
        for candidate in candidates:
            trajectory = inspect_accepted_trajectory(
                args.input_file, candidate.candidate_id, ledger
            )
            for variant_index in range(variants):
                _render_variant(
                    env=env,
                    controller=controller,
                    config=config,
                    ledger=ledger,
                    trajectory=trajectory,
                    candidate=candidate,
                    variant_index=variant_index,
                    runtime_manifest_sha256=runtime_manifest_sha256,
                    camera_contract=camera_contract,
                    output_root=output_root,
                )
            if ledger.load(candidate.candidate_id).status == "validated":
                ledger.complete_rendering(candidate.candidate_id, variants)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
