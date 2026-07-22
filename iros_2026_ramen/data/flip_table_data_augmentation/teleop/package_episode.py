"""Package one validated direct sim teleop episode for LeRobot assembly."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np

from ..config import EXPECTED_CAMERA_KEYS, PipelineConfig
from ..export.contracts import RENDER_MANIFEST_SCHEMA_VERSION, TASK, RenderedEpisode
from ..fk_audit import synthetic_action_fk_report
from ..io_utils import atomic_write_json, sha256_file
from .numeric import convert_raw_episode, sim_teleop_source_index
from .raw_episode import POLICY_CAMERA_ROLE_TO_KEY, RAW_EPISODE_SCHEMA_VERSION


DIRECT_TELEOP_PACKAGE_SCHEMA_VERSION = "team_ramen_flip_table_direct_teleop_package/v1"


def _read_numeric(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["action.robot_q_desired", "action.ee_action"],
    )
    robot_q = np.asarray(table["action.robot_q_desired"].to_pylist(), dtype=np.float64)
    ee_action = np.asarray(table["action.ee_action"].to_pylist(), dtype=np.float64)
    return robot_q, ee_action


def _write_policy_frames(source_root: Path, target_root: Path, frame_count: int) -> dict[str, str]:
    from PIL import Image

    camera_dirs: dict[str, str] = {}
    for role, key in POLICY_CAMERA_ROLE_TO_KEY.items():
        if key not in EXPECTED_CAMERA_KEYS:
            raise RuntimeError(f"raw policy camera is not in the source contract: {key}")
        output_dir = target_root / "cameras" / role
        output_dir.mkdir(parents=True)
        source_dir = source_root / "policy_cameras" / role
        for index in range(frame_count):
            source = source_dir / f"{index:06d}.jpg"
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                if rgb.size != (640, 480):
                    raise ValueError(f"raw policy frame has invalid size: {source}")
                rgb.save(
                    output_dir / f"frame_{index:06d}.png",
                    format="PNG",
                    compress_level=1,
                )
        camera_dirs[key] = output_dir.relative_to(target_root).as_posix()
    if tuple(camera_dirs) != EXPECTED_CAMERA_KEYS:
        raise RuntimeError("direct teleop package does not contain cam_0/cam_2/cam_3")
    return camera_dirs


def package_direct_sim_teleop_episode(
    episode_root: str | Path,
    *,
    output_root: str | Path,
    urdf_path: str | Path,
    config: PipelineConfig,
) -> dict[str, object]:
    raw_root = Path(episode_root).expanduser().resolve()
    raw_manifest_path = raw_root / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if (
        raw_manifest.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION
        or raw_manifest.get("backend") != "sim"
        or raw_manifest.get("success") is not True
        or raw_manifest.get("privileged_policy_features") != []
    ):
        raise ValueError("only successful non-privileged sim teleop episodes may be packaged")
    episode_id = str(raw_manifest["episode_id"])
    final = Path(output_root).expanduser().resolve() / episode_id / "variant_000"
    if final.exists():
        raise FileExistsError(final)
    temporary = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        numeric_source = raw_root / "numeric.parquet"
        if not numeric_source.is_file():
            convert_raw_episode(raw_root, urdf_path=urdf_path)
        numeric_target = temporary / "numeric.parquet"
        shutil.copy2(numeric_source, numeric_target)
        frame_count = int(raw_manifest["frame_count"])
        camera_dirs = _write_policy_frames(raw_root, temporary, frame_count)

        robot_q, ee_action = _read_numeric(numeric_target)
        source_contract = config.raw["source_contract"]
        action_fk_report = synthetic_action_fk_report(
            robot_q_desired=robot_q,
            ee_action=ee_action,
            urdf_path=Path(urdf_path).expanduser().resolve(),
            frame_names=dict(source_contract["fk_frames"]),
            tool_transforms=dict(source_contract["fk_tool_transforms"]),
            position_p95_max=float(
                source_contract["fk_action_validation_position_p95_m_max"]
            ),
            rotation_p95_max=float(
                source_contract["fk_action_validation_rotation_p95_rad_max"]
            ),
        )
        if action_fk_report["pass"] is not True:
            raise ValueError("direct sim teleop action FK audit failed")

        runtime_manifest = {
            "schema_version": DIRECT_TELEOP_PACKAGE_SCHEMA_VERSION,
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "raw_trace_sha256": sha256_file(raw_root / "frames.jsonl"),
            "teleop_config_sha256": raw_manifest["config_sha256"],
            "robofinals_runtime_digest": raw_manifest["runtime_digest"],
            "pipeline_config_sha256": config.digest,
            "privileged_policy_features": [],
        }
        runtime_path = temporary / "runtime_manifest.json"
        atomic_write_json(runtime_path, runtime_manifest)
        diagnostics = raw_manifest.get("diagnostics", {})
        randomization = diagnostics.get("randomization") if isinstance(diagnostics, dict) else None
        if not isinstance(randomization, dict) or not randomization:
            raise ValueError("direct sim teleop manifest lacks applied randomization")
        source_index = sim_teleop_source_index(episode_id)
        render_manifest = {
            "schema_version": RENDER_MANIFEST_SCHEMA_VERSION,
            "candidate_id": f"direct-sim-{episode_id}",
            "trajectory_kind": "direct_sim_teleop",
            "source_kind": "sim_teleop",
            "appearance_variant": 0,
            "source_episode_indices": [source_index],
            "source_trajectory_lineage": f"sim_teleop:{episode_id}",
            "frame_count": frame_count,
            "fps": 30,
            "task": TASK,
            "numeric_trace": numeric_target.name,
            "numeric_trace_sha256": sha256_file(numeric_target),
            "cameras": camera_dirs,
            "trajectory_sha256": sha256_file(raw_root / "frames.jsonl"),
            "runtime_manifest_sha256": sha256_file(runtime_path),
            "config_sha256": config.digest,
            "randomization": {
                "physical_and_visual": randomization,
                "camera_runtime_at_save": diagnostics.get("camera_runtime"),
                "trajectory_sampling": {
                    "source_hz": 30,
                    "target_hz": 30,
                    "source_frame_count": frame_count,
                },
            },
            "success_report": {
                "accepted": True,
                "strict_v1_contract": True,
                "rejection_reasons": [],
                "sim_success_components": diagnostics.get("success_components"),
                "action_fk_report": action_fk_report,
            },
        }
        manifest_path = temporary / "render_manifest.json"
        atomic_write_json(manifest_path, render_manifest)
        RenderedEpisode.load(manifest_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema_version": DIRECT_TELEOP_PACKAGE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_episode_index": source_index,
        "output": str(final),
        "render_manifest": str(final / "render_manifest.json"),
    }
