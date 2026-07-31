#!/usr/bin/env python3
"""Create a wrist-FK time-offset manifest without shifting recorded RGB.

The source RGB frame remains fixed.  Only the offline association to a
candidate later/earlier ``robot_q_current`` row is changed, so a D405
hand-eye sweep can distinguish timing error from a physical camera-mount
error.  The output is calibration evidence only and is never a policy input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.source_camera_projection import root_from_camera
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex
from data.flip_table_data_augmentation.object_pose.geometry import OPENGL_FROM_OPENCV


SCHEMA_VERSION = "team_ramen_flip_table_wrist_time_offset_manifest/v1"
SIDES = ("left_wrist", "right_wrist")


def _tool_transforms(config: Any) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for side in ("left", "right"):
        raw = config.raw["source_contract"]["fk_tool_transforms"][side]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.asarray(raw["translation_m"], dtype=np.float64)
        from scipy.spatial.transform import Rotation

        transform[:3, :3] = Rotation.from_quat(raw["quaternion_xyzw"]).as_matrix()
        values.append(transform)
    return tuple(values)  # type: ignore[return-value]


def _placement_matrix(placement: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(placement.rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(placement.translation, dtype=np.float64)
    return result


def _wrist_fk(
    *, episode: Any, frames: tuple[int, ...], source_urdf: Path, config: Any
) -> dict[int, dict[str, np.ndarray]]:
    """Compute only mesh-free wrist FK; visual meshes are irrelevant here."""

    import pinocchio as pin
    import pyarrow.parquet as pq

    rows = pq.read_table(
        episode.data_path,
        columns=["frame_index", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode.episode_index)],
    ).to_pylist()
    q_by_frame = {
        int(row["frame_index"]): np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        for row in rows if int(row["frame_index"]) in set(frames)
    }
    if set(q_by_frame) != set(frames):
        raise ValueError("source parquet lacks requested q_current frames")
    model = pin.buildModelFromUrdf(str(source_urdf))
    names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    if any(not model.existFrame(name) for name in names):
        raise ValueError("source FK URDF lacks D405 wrist parent frames")
    missing = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    if missing:
        raise ValueError(f"source FK URDF lacks joints: {missing}")
    joint_indices = np.asarray([model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER], dtype=np.int64)
    frame_ids = tuple(model.getFrameId(name) for name in names)
    data = model.createData()
    tools = _tool_transforms(config)
    result: dict[int, dict[str, np.ndarray]] = {}
    for frame in frames:
        source_q = q_by_frame[frame]
        if source_q.shape != (36,) or not np.isfinite(source_q).all():
            raise ValueError(f"source frame {frame} has invalid robot_q_current")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        wrists = tuple(_placement_matrix(data.oMf[frame_id]) for frame_id in frame_ids)
        result[frame] = {
            "robot_q_current": source_q,
            "eef_current_root_from_fk": np.stack(tuple(wrist @ tool for wrist, tool in zip(wrists, tools, strict=True))),
            "left_wrist_camera": root_from_camera(wrists[0], config.cameras[1]) @ OPENGL_FROM_OPENCV,
            "right_wrist_camera": root_from_camera(wrists[1], config.cameras[2]) @ OPENGL_FROM_OPENCV,
        }
    return result


def _link_evidence(value: str, *, source_directory: Path, output_directory: Path) -> str:
    """Hard-link immutable D405 RGB so the calibration manifest is self-contained."""

    source = (source_directory / value).resolve()
    if not source.is_file() or source_directory.resolve() not in source.parents:
        raise ValueError(f"input evidence path is invalid: {value}")
    destination = output_directory / value
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination.relative_to(output_directory))


def build_manifest(
    *,
    input_manifest: Path,
    source_root: Path,
    source_urdf: Path,
    q_current_offset_frames: int,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Write an immutable-RGB manifest with wrist FK recomputed at one offset."""

    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    source_manifest_path = input_manifest.expanduser().resolve()
    source_directory = source_manifest_path.parent
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("frames"), list):
        raise ValueError("input manifest lacks frame records")
    episode_index = int(manifest["episode_index"])
    config = load_pipeline_config(config_path)
    dataset = SourceDatasetIndex(source_root.expanduser().resolve())
    episode = dataset.episode(episode_index)
    source_pairs = tuple(
        (frame, int(frame["source_frame_index"]), int(frame["source_frame_index"]) + int(q_current_offset_frames))
        for frame in manifest["frames"]
    )
    valid_pairs = tuple(pair for pair in source_pairs if 0 <= pair[2] < episode.frame_count)
    if len(valid_pairs) < 3:
        raise ValueError("q_current offset leaves fewer than three source frames")
    q_frames = tuple(pair[2] for pair in valid_pairs)
    fk = _wrist_fk(episode=episode, frames=q_frames, source_urdf=source_urdf.expanduser().resolve(), config=config)
    result = dict(manifest)
    result["schema_version"] = SCHEMA_VERSION
    result["source_input_manifest"] = str(source_manifest_path)
    result["source_input_manifest_sha256"] = __import__("hashlib").sha256(source_manifest_path.read_bytes()).hexdigest()
    result["wrist_q_current_offset"] = {
        "frames": int(q_current_offset_frames),
        "seconds": float(q_current_offset_frames) / float(config.source.fps),
        "definition": "robot_q_current[source_frame_index + frames] is paired with the unchanged D405 RGB frame",
        "rgb_frames_shifted": False,
    }
    output_dir.mkdir(parents=True)
    result_frames = []
    for original, source_frame, q_frame in valid_pairs:
        frame = json.loads(json.dumps(original))
        frame["q_current_source_frame"] = q_frame
        frame["robot_q_current"] = fk[q_frame]["robot_q_current"].tolist()
        frame["eef_current_root_from_fk"] = fk[q_frame]["eef_current_root_from_fk"].reshape(2, 16).tolist()
        views = frame.get("views")
        if not isinstance(views, dict):
            raise ValueError(f"input manifest frame {source_frame} lacks views")
        for side in SIDES:
            view = views.get(side)
            if not isinstance(view, dict):
                raise ValueError(f"input manifest frame {source_frame} lacks {side}")
            view["robot_root_from_rectified_opencv"] = fk[q_frame][f"{side}_camera"].reshape(-1).tolist()
            if not isinstance(view.get("rgb"), str):
                raise ValueError(f"input manifest frame {source_frame} lacks {side} RGB")
            view["rgb"] = _link_evidence(view["rgb"], source_directory=source_directory, output_directory=output_dir)
        result_frames.append(frame)
    result["frames"] = result_frames
    result["dropped_source_frame_indices"] = [source_frame for _, source_frame, _ in source_pairs if source_frame not in {value[1] for value in valid_pairs}]
    atomic_write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--q-current-offset-frames", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = build_manifest(
        input_manifest=args.input_manifest,
        source_root=args.source_root,
        source_urdf=args.source_urdf,
        q_current_offset_frames=args.q_current_offset_frames,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(json.dumps({"output": str(args.output_dir), "frames": len(report["frames"]), "offset": args.q_current_offset_frames}))


if __name__ == "__main__":
    main()
