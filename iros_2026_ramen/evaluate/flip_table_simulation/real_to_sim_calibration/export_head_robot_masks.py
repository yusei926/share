#!/usr/bin/env python3
"""Export offline G1/Dex1 head-left robot-occupancy masks for RGB scoring.

The masks remove robot pixels from an RGB-only table silhouette comparison.
They are derived from recorded/measured joint encoders and the pinned G1 +
Dex1-1 visual URDF, never from table poses, contacts, segmentation, or a
simulator object render.  They are calibration diagnostics only and must not
be supplied to a policy, planner, reward, or runtime branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import (
    DEFAULT_CONFIG_PATH,
    CameraConfig,
    load_pipeline_config,
)
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.object_pose.robot_silhouette import (
    RobotSilhouetteRenderer,
    robot_silhouette_coverage_is_plausible,
)
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex
from data.flip_table_data_augmentation.source_camera_projection import root_from_camera
from evaluate.flip_table_simulation.real_to_sim_calibration.stereo_geometry import (
    HeadStereoCalibration,
)


SCHEMA_VERSION = "team_ramen_flip_table_head_robot_mask_export/v1"
_SIM_CAMERA_FRAME_PREFIX = "frame_"
_OPENGL_FROM_OPENCV = np.diag((1.0, -1.0, -1.0, 1.0))


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return result


def _world_from_pose(value: Any, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be [x,y,z,qx,qy,qz,qw]")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1.0e-3):
        raise ValueError(f"{label} quaternion is not normalized")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:] / np.linalg.norm(pose[3:])).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"could not write robot mask: {path}")


def _renderer(config_path: Path, robofinals_root: Path) -> RobotSilhouetteRenderer:
    config = load_pipeline_config(config_path)
    runtime = config.object_pose_runtime
    return RobotSilhouetteRenderer(
        robofinals_root / runtime.robot_visual_urdf_relative_path,
        expected_sha256=runtime.robot_visual_urdf_sha256,
        dilation_px=runtime.auxiliary_robot_silhouette_dilation_px,
    )


def _source_head_camera(config_path: Path) -> CameraConfig:
    config = load_pipeline_config(config_path)
    return next(
        camera
        for camera in config.cameras
        if camera.source_key == "observation.images.cam_0"
    )


def _source_head_camera_poses(
    *,
    renderer: RobotSilhouetteRenderer,
    rows: dict[int, dict[str, Any]],
    camera: CameraConfig,
) -> dict[int, np.ndarray]:
    """Recover OpenCV head-left poses from measured body encoders only."""

    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(renderer.urdf_path))
    data = model.createData()
    if not model.existFrame("torso_link"):
        raise ValueError("pinned visual URDF lacks torso_link")
    torso_id = int(model.getFrameId("torso_link"))
    joint_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
        dtype=np.int64,
    )
    result: dict[int, np.ndarray] = {}
    for frame, row in rows.items():
        robot_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        if robot_q.shape != (36,) or not np.isfinite(robot_q).all():
            raise ValueError(f"source frame {frame} has invalid robot_q_current")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = robot_q[7:]
        pin.framesForwardKinematics(model, data, q)
        root_from_torso = np.eye(4, dtype=np.float64)
        root_from_torso[:3, :3] = np.asarray(data.oMf[torso_id].rotation)
        root_from_torso[:3, 3] = np.asarray(data.oMf[torso_id].translation)
        # CameraConfig stores an Isaac/OpenGL camera offset. The source RGB
        # and RobotSilhouetteRenderer both use OpenCV optical coordinates.
        result[frame] = root_from_camera(root_from_torso, camera) @ _OPENGL_FROM_OPENCV
    return result


def _source_rows(source_root: Path, episode_index: int, frames: set[int]) -> dict[int, dict[str, Any]]:
    episode = SourceDatasetIndex(source_root).episode(episode_index)
    rows = pq.read_table(
        episode.data_path,
        columns=[
            "frame_index",
            "observation.state.robot_q_current",
            "observation.state.hand_state",
        ],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    result = {int(row["frame_index"]): row for row in rows if int(row["frame_index"]) in frames}
    missing = sorted(frames - set(result))
    if missing:
        raise ValueError(f"source episode lacks requested frames: {missing}")
    return result


def export_source(
    *,
    source_root: Path,
    source_alignment: Path,
    stereo_calibration: Path,
    robofinals_root: Path,
    config_path: Path,
    output_dir: Path,
    frames: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Export real head-left masks using measured source encoder states."""

    report = json.loads(source_alignment.read_text(encoding="utf-8"))
    if report.get("schema_version") != "team_ramen_flip_table_source_cad_alignment/v1":
        raise ValueError("source alignment schema is unexpected")
    source = report.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("episode_index"), int):
        raise ValueError("source alignment lacks an episode index")
    alignment_frames = []
    for frame in report.get("frames", []):
        if not isinstance(frame, dict) or not isinstance(frame.get("frame_index"), int):
            continue
        cameras = frame.get("root_from_opencv_camera")
        if isinstance(cameras, dict) and cameras.get("head_left") is not None:
            alignment_frames.append(int(frame["frame_index"]))
    if not alignment_frames:
        raise ValueError("source alignment has no head-left camera poses")
    selected_frames = tuple(sorted(set(alignment_frames if frames is None else frames)))
    if not selected_frames or any(frame < 0 for frame in selected_frames):
        raise ValueError("source frames must be non-empty non-negative integers")
    stereo = HeadStereoCalibration.load(stereo_calibration)
    renderer = _renderer(config_path, robofinals_root)
    rows = _source_rows(source_root, int(source["episode_index"]), set(selected_frames))
    root_from_cameras = _source_head_camera_poses(
        renderer=renderer,
        rows=rows,
        camera=_source_head_camera(config_path),
    )
    records = []
    for frame in selected_frames:
        row = rows[frame]
        robot_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        hand_state = np.asarray(row["observation.state.hand_state"], dtype=np.float64)
        if robot_q.shape != (36,) or hand_state.shape != (2,):
            raise ValueError(f"source frame {frame} has invalid robot/hand state")
        mask, metrics = renderer.render(
            robot_q_current=robot_q,
            hand_state=hand_state,
            root_from_camera=root_from_cameras[frame],
            intrinsic_matrix=stereo.left_intrinsic,
            width=640,
            height=480,
        )
        if not robot_silhouette_coverage_is_plausible(metrics.mask_fraction):
            raise ValueError(f"source frame {frame} robot silhouette coverage is implausible")
        relative = Path(f"frame_{frame:04d}") / "head_left_robot_mask.png"
        _write_mask(output_dir / relative, mask)
        records.append({"source_frame": frame, "mask": str(relative), **metrics.to_json()})
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "source_real",
        "policy_use": "forbidden: offline RGB metric self-occlusion exclusion only",
        "source_alignment": str(source_alignment),
        "source_episode_index": int(source["episode_index"]),
        "source_frames_requested": list(selected_frames),
        "camera_pose_method": "recorded_q_current + pinned_torso_link + pinned_head_left_mount",
        "records": records,
    }
    atomic_write_json(output_dir / "head_robot_masks.json", result)
    return result


def _sim_body_state(diagnostics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = diagnostics.get("joint_names")
    position = np.asarray(diagnostics.get("joint_position_rad"), dtype=np.float64)
    if not isinstance(names, list) or position.shape != (len(names),):
        raise ValueError("simulator diagnostics lack joint_names/joint_position_rad")
    values = dict(zip((str(name) for name in names), position, strict=True))
    missing = [name for name in G1_BODY_JOINT_ORDER if name not in values]
    if missing:
        raise ValueError(f"simulator diagnostics lack G1 joints: {missing}")
    root = _world_from_pose(diagnostics.get("root_pose_world_xyzw"), "simulator root pose")
    camera = diagnostics.get("policy_camera_poses", {}).get("head_left")
    if not isinstance(camera, dict):
        raise ValueError("simulator diagnostics lack head_left camera pose")
    world_from_gl_camera = _world_from_pose(
        [*camera.get("position_world_m", ()), *camera.get("quaternion_xyzw", ())],
        "simulator head_left camera pose",
    )
    # V1 reports USD/OpenGL camera poses. RobotSilhouetteRenderer projects in
    # the OpenCV optical convention used by the recorded head-left RGB.
    camera_pose = world_from_gl_camera @ _OPENGL_FROM_OPENCV
    body = np.asarray([values[name] for name in G1_BODY_JOINT_ORDER], dtype=np.float64)
    fingers = np.asarray(
        (
            np.mean([values["left_dex1_finger_joint_1"], values["left_dex1_finger_joint_2"]]),
            np.mean([values["right_dex1_finger_joint_1"], values["right_dex1_finger_joint_2"]]),
        ),
        dtype=np.float64,
    )
    hand_state = (fingers + 0.02) / (0.0245 + 0.02) * 4.5
    robot_q = np.concatenate((
        root[:3, 3],
        Rotation.from_matrix(root[:3, :3]).as_quat()[[3, 0, 1, 2]],
        body,
    ))
    return robot_q, hand_state, np.linalg.inv(root) @ camera_pose


def export_simulation(
    *,
    trace_path: Path,
    replay_actions: Path,
    robofinals_root: Path,
    config_path: Path,
    output_dir: Path,
    simulator_steps: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Export sim masks from logged actual joints and rendered camera poses."""

    replay = json.loads(replay_actions.read_text(encoding="utf-8"))
    mapping = replay.get("camera_frame_map")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("replay actions lack camera_frame_map")
    rows = []
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("step"), int):
                    rows.append(row)
    by_step = {int(row["step"]): row for row in rows}
    requested_steps = None if simulator_steps is None else set(simulator_steps)
    if requested_steps is not None and (
        not requested_steps or any(step < 0 for step in requested_steps)
    ):
        raise ValueError("simulator_steps must contain non-negative steps")
    selected_mapping = [
        entry
        for entry in mapping
        if requested_steps is None or entry.get("simulator_step") in requested_steps
    ]
    if not selected_mapping:
        raise ValueError("no replay camera-frame mapping matched requested simulator steps")
    renderer = _renderer(config_path, robofinals_root)
    records = []
    for entry in selected_mapping:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_frame"), int) or not isinstance(entry.get("simulator_step"), int):
            raise ValueError("camera_frame_map is malformed")
        sim_step = int(entry["simulator_step"])
        row = by_step.get(sim_step)
        if row is None:
            raise ValueError(f"trace has no row for simulator camera step {sim_step}")
        diagnostics = row.get("simulator_scene_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"trace step {sim_step} lacks simulator diagnostics")
        robot_q, hand_state, root_from_camera = _sim_body_state(diagnostics)
        mask, metrics = renderer.render(
            robot_q_current=robot_q,
            hand_state=hand_state,
            root_from_camera=root_from_camera,
            intrinsic_matrix=np.asarray(
                ((337.5311318539417, 0.0, 316.5285046932812),
                 (0.0, 336.61378142923456, 232.50620475777816),
                 (0.0, 0.0, 1.0)),
                dtype=np.float64,
            ),
            width=640,
            height=480,
        )
        if not robot_silhouette_coverage_is_plausible(metrics.mask_fraction):
            raise ValueError(f"simulator step {sim_step} robot silhouette coverage is implausible")
        relative = Path(f"frame_{sim_step:04d}") / "head_left_robot_mask.png"
        _write_mask(output_dir / relative, mask)
        records.append({
            "source_frame": int(entry["source_frame"]),
            "simulator_step": sim_step,
            "mask": str(relative),
            **metrics.to_json(),
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "simulation",
        "policy_use": "forbidden: offline RGB metric self-occlusion exclusion only",
        "trace": str(trace_path),
        "replay_actions": str(replay_actions),
        "simulator_steps_requested": (
            None if simulator_steps is None else sorted(requested_steps)
        ),
        "records": records,
    }
    atomic_write_json(output_dir / "head_robot_masks.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robofinals-root", type=Path, required=True)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    common.add_argument("--output-dir", type=Path, required=True)
    source = subparsers.add_parser("source", parents=[common])
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--source-alignment", type=Path, required=True)
    source.add_argument(
        "--stereo-calibration",
        type=Path,
        required=True,
        help="pinned raw head_camera_params.yaml used by source_cad_alignment.py",
    )
    source.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="source-local frame indices; default: frames preserved by source alignment",
    )
    simulated = subparsers.add_parser("simulation", parents=[common])
    simulated.add_argument("--trace", type=Path, required=True)
    simulated.add_argument("--replay-actions", type=Path, required=True)
    simulated.add_argument(
        "--simulator-step",
        type=int,
        action="append",
        default=None,
        help="export only this replay camera step; repeatable",
    )
    args = parser.parse_args()
    if args.mode == "source":
        report = export_source(
            source_root=args.source_root.expanduser().resolve(),
            source_alignment=args.source_alignment.expanduser().resolve(),
            stereo_calibration=args.stereo_calibration.expanduser().resolve(),
            robofinals_root=args.robofinals_root.expanduser().resolve(),
            config_path=args.config.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            frames=None if args.frames is None else tuple(args.frames),
        )
    else:
        report = export_simulation(
            trace_path=args.trace.expanduser().resolve(),
            replay_actions=args.replay_actions.expanduser().resolve(),
            robofinals_root=args.robofinals_root.expanduser().resolve(),
            config_path=args.config.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            simulator_steps=(
                None if args.simulator_step is None else tuple(args.simulator_step)
            ),
        )
    print(json.dumps({"kind": report["kind"], "masks": len(report["records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
