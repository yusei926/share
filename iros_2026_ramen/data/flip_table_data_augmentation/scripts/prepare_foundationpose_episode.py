#!/usr/bin/env python3
"""Prepare calibrated three-view RGB-D evidence for offline table-pose tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from data.flip_table_data_augmentation.config import (
    DEFAULT_CONFIG_PATH,
    canonical_json_digest,
    load_pipeline_config,
)
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from data.flip_table_data_augmentation.object_pose import INPUT_SCHEMA_VERSION
from data.flip_table_data_augmentation.object_pose.camera_views import (
    POSE_VIEW_NAMES,
    inverse_brown_conrady_rectification_maps,
)
from data.flip_table_data_augmentation.object_pose.geometry import (
    OPENGL_FROM_OPENCV,
    root_from_rectified_opencv_camera,
)
from data.flip_table_data_augmentation.source_camera_projection import root_from_camera
from data.flip_table_data_augmentation.source_contract import (
    download_pinned_source_files,
    snapshot_download_pinned,
)
from data.flip_table_data_augmentation.source_stereo_depth import (
    STEREO_DEPTH_SCHEMA_VERSION,
    FastFoundationStereoDepthEstimator,
    FastFoundationStereoParameters,
    depth_to_uint16_mm,
)
from data.flip_table_data_augmentation.source_video import (
    decode_video_slice_rgb,
    source_frame_indices,
)
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex
from model.flip_table_reinforcement_learning.teacher.source_stereo_calibration import (
    load_stereo_calibration,
)


SCHEMA_VERSION = INPUT_SCHEMA_VERSION
DEFAULT_URDF = Path(
    "/workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/"
    "g1_29dof_with_hand.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FLIP_TABLE_OBJECT_POSE_RUNTIME",
                "~/.cache/team-ramen/flip-table-object-pose",
            )
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--observed-image-digest",
        default=os.environ.get("ROBOFINALS_IMAGE_DIGEST"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"OpenCV could not write {temporary}")
    os.replace(temporary, path)


def _placement_matrix(placement) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = placement.rotation
    value[:3, 3] = placement.translation
    return value


def _laplacian_variance(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _robot_frame_inputs(
    *,
    episode,
    frames: tuple[int, ...],
    urdf: Path,
    cameras,
    rectification_left: np.ndarray,
    eef_tool_transforms: tuple[np.ndarray, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    import pinocchio as pin

    rows = pq.read_table(
        episode.data_path,
        columns=[
            "frame_index",
            "observation.state.robot_q_current",
            "observation.state.ee_state",
            "observation.state.hand_state",
            "action.ee_action",
            "action.hand_cmd",
        ],
        filters=[("episode_index", "=", episode.episode_index)],
    ).to_pylist()
    selected = {int(value) for value in frames}
    state_by_frame = {
        int(row["frame_index"]): (
            np.asarray(row["observation.state.robot_q_current"], dtype=np.float64),
            np.asarray(row["observation.state.ee_state"], dtype=np.float64),
            np.asarray(row["observation.state.hand_state"], dtype=np.float64),
            np.asarray(row["action.ee_action"], dtype=np.float64),
            np.asarray(row["action.hand_cmd"], dtype=np.float64),
        )
        for row in rows
        if int(row["frame_index"]) in selected
    }
    if set(state_by_frame) != selected:
        raise ValueError("source Parquet is missing selected robot state frames")
    wrapper = pin.RobotWrapper.BuildFromURDF(str(urdf), package_dirs=[str(urdf.parent)])
    model, data = wrapper.model, wrapper.data
    camera_parent_frames = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    missing_camera_frames = [name for name in camera_parent_frames if not model.existFrame(name)]
    if missing_camera_frames:
        raise ValueError(f"source FK URDF lacks camera frames: {missing_camera_frames}")
    if len(cameras) != len(POSE_VIEW_NAMES):
        raise ValueError("offline pose annotation requires exactly three configured cameras")
    if any(np.asarray(value).shape != (4, 4) for value in eef_tool_transforms):
        raise ValueError("current EEF FK requires two 4x4 tool transforms")
    missing = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    if missing:
        raise ValueError(f"source FK URDF lacks configured joints: {missing}")
    joint_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
        dtype=np.int64,
    )
    camera_frame_ids = tuple(model.getFrameId(name) for name in camera_parent_frames)
    output = {}
    for frame in frames:
        source_q, ee_state, hand_state, ee_action, hand_cmd = state_by_frame[frame]
        if source_q.shape != (36,) or not np.isfinite(source_q).all():
            raise ValueError(f"frame {frame} robot_q_current is not a finite 36D vector")
        if hand_state.shape != (2,) or not np.isfinite(hand_state).all():
            raise ValueError(f"frame {frame} hand_state is not a finite 2D vector")
        if ee_state.shape != (12,) or not np.isfinite(ee_state).all():
            raise ValueError(f"frame {frame} ee_state is not a finite 12D vector")
        if ee_action.shape != (12,) or not np.isfinite(ee_action).all():
            raise ValueError(f"frame {frame} ee_action is not a finite 12D vector")
        if hand_cmd.shape != (2,) or not np.isfinite(hand_cmd).all():
            raise ValueError(f"frame {frame} hand_cmd is not a finite 2D vector")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        camera_poses = {
            POSE_VIEW_NAMES[0]: root_from_rectified_opencv_camera(
                _placement_matrix(data.oMf[camera_frame_ids[0]]),
                cameras[0],
                rectification_left,
            )
        }
        for view_name, frame_id, camera in zip(
            POSE_VIEW_NAMES[1:], camera_frame_ids[1:], cameras[1:], strict=True
        ):
            camera_poses[view_name] = (
                root_from_camera(_placement_matrix(data.oMf[frame_id]), camera)
                @ OPENGL_FROM_OPENCV
            )
        eef_current_root_from_fk = np.stack(
            [
                _placement_matrix(data.oMf[frame_id]) @ tool_transform
                for frame_id, tool_transform in zip(
                    camera_frame_ids[1:], eef_tool_transforms, strict=True
                )
            ]
        )
        output[frame] = {
            "robot_root_from_cameras_opencv": camera_poses,
            "eef_current_root_from_fk": eef_current_root_from_fk,
            "robot_q_current": source_q,
            "ee_state": ee_state,
            "hand_state": hand_state,
            "ee_action": ee_action,
            "hand_cmd": hand_cmd,
        }
    return output


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    if args.observed_image_digest != config.runtime.container_digest:
        raise ValueError(
            "RGB-D source preparation must run in the pinned organizer V1 image digest"
        )
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else snapshot_download_pinned(config, include_videos=False)
    )
    calibration_path = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else Path(
            hf_hub_download(
                repo_id=config.raw_source.repo_id,
                repo_type="dataset",
                revision=config.raw_source.revision,
                filename=config.raw_source.head_stereo_calibration_repo_path,
            )
        ).resolve()
    )
    if sha256_file(calibration_path) != config.raw_source.head_stereo_calibration_sha256:
        raise ValueError("head-stereo calibration SHA-256 differs from the pinned source")

    index = SourceDatasetIndex(source_root)
    if len(index) != config.source.episodes:
        raise ValueError(f"source contains {len(index)} episodes, expected {config.source.episodes}")
    episode = index.episode(args.episode_index)
    download_pinned_source_files(
        config,
        source_root,
        tuple(
            episode.video_relative_path(source_key)
            for source_key in (
                config.cameras[0].source_key,
                "observation.images.cam_1",
                config.cameras[1].source_key,
                config.cameras[2].source_key,
            )
        ),
    )
    stride = config.object_pose_runtime.source_frame_stride
    frames = source_frame_indices(episode.frame_count, stride)
    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"source FK URDF is missing: {urdf}")
    calibration = load_stereo_calibration(calibration_path)
    pose = config.object_pose_runtime
    runtime_root = args.runtime_root.expanduser().resolve()
    runtime_manifests = {}
    for name in ("runtime-manifest.json", "compiled-runtime-manifest.json"):
        path = runtime_root / name
        if not path.is_file():
            raise FileNotFoundError(f"object-pose runtime manifest is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("config_sha256") != config.digest:
            raise ValueError(f"{name} was prepared for a different pipeline config")
        runtime_manifests[name] = {
            "schema_version": value.get("schema_version"),
            "sha256": sha256_file(path),
        }
    parameters = FastFoundationStereoParameters(
        valid_iterations=pose.fast_stereo_valid_iterations,
        max_disparity_px=pose.fast_stereo_max_disparity_px,
        maximum_left_right_error_px=(
            pose.fast_stereo_maximum_left_right_error_px
        ),
    )
    fast_stereo_source = runtime_root / "Fast-FoundationStereo"
    fast_stereo_model = (
        runtime_root / "hf" / "fast-foundation-stereo" / pose.fast_stereo_model_filename
    )
    fast_stereo_config = (
        runtime_root / "hf" / "fast-foundation-stereo" / pose.fast_stereo_config_filename
    )
    if sha256_file(fast_stereo_model) != pose.fast_stereo_model_sha256:
        raise ValueError("Fast FoundationStereo model SHA-256 differs from the pin")
    if sha256_file(fast_stereo_config) != pose.fast_stereo_config_sha256:
        raise ValueError("Fast FoundationStereo config SHA-256 differs from the pin")
    estimator = FastFoundationStereoDepthEstimator(
        calibration,
        source_root=fast_stereo_source,
        model_path=fast_stereo_model,
        parameters=parameters,
    )
    input_contract_sha256 = canonical_json_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "source_repo_id": config.source.repo_id,
            "source_revision": config.source.revision,
            "episode_index": episode.episode_index,
            "source_frame_count": episode.frame_count,
            "source_fps": config.source.fps,
            "source_frame_stride": stride,
            "head_stereo_calibration_sha256": sha256_file(calibration_path),
            "stereo_depth_schema_version": STEREO_DEPTH_SCHEMA_VERSION,
            "stereo_depth_parameters": asdict(parameters),
            "fast_stereo_repo": pose.fast_stereo_repo,
            "fast_stereo_revision": pose.fast_stereo_revision,
            "fast_stereo_model_repo": pose.fast_stereo_model_repo,
            "fast_stereo_model_revision": pose.fast_stereo_model_revision,
            "fast_stereo_model_sha256": sha256_file(fast_stereo_model),
            "object_pose_runtime_manifests": runtime_manifests,
            "opencv_version": cv2.__version__,
            "urdf_sha256": sha256_file(urdf),
            "joint_order": G1_BODY_JOINT_ORDER,
            "head_camera_offset_position_m": config.cameras[0].offset_position_m,
            "head_camera_offset_quaternion_xyzw": config.cameras[0].offset_quaternion_xyzw,
            "pose_cameras": [
                {
                    "view_name": view_name,
                    "source_key": camera.source_key,
                    "parent_frame": parent_frame,
                    "offset_position_m": camera.offset_position_m,
                    "offset_quaternion_xyzw": camera.offset_quaternion_xyzw,
                    "intrinsic_matrix_px": camera.intrinsic_matrix_px,
                    "distortion_model": camera.distortion_model,
                    "distortion_coefficients": camera.distortion_coefficients,
                    "calibration_basis": camera.calibration_basis,
                }
                for view_name, camera, parent_frame in zip(
                    POSE_VIEW_NAMES,
                    config.cameras,
                    ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
                    strict=True,
                )
            ],
            "eef_tool_transforms": config.raw["source_contract"]["fk_tool_transforms"],
            "runtime_container_digest": config.runtime.container_digest,
        }
    )

    output = args.output_dir.expanduser().resolve()
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError(f"output already exists: {output}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (input_contract_sha256, config.source.revision, episode.episode_index, list(frames))
        actual = (
            previous.get("input_contract_sha256"),
            previous.get("source_revision"),
            previous.get("episode_index"),
            previous.get("source_frame_indices"),
        )
        if actual != expected:
            raise ValueError("existing FoundationPose input was produced from a different contract")
        print(json.dumps({"output": str(output), "resumed": True}, sort_keys=True))
        return

    camera_source_keys = tuple(camera.source_key for camera in config.cameras)
    if camera_source_keys != (
        "observation.images.cam_0",
        "observation.images.cam_2",
        "observation.images.cam_3",
    ):
        raise ValueError("source three-camera pose annotation contract changed")
    head_right = "observation.images.cam_1"
    left_video = episode.video_slice(camera_source_keys[0])
    right_video = episode.video_slice(head_right)
    left_frames = decode_video_slice_rgb(
        left_video, frames, fps=config.source.fps, frame_count=episode.frame_count
    )
    right_frames = decode_video_slice_rgb(
        right_video, frames, fps=config.source.fps, frame_count=episode.frame_count
    )
    wrist_videos = {
        view_name: episode.video_slice(source_key)
        for view_name, source_key in zip(
            POSE_VIEW_NAMES[1:], camera_source_keys[1:], strict=True
        )
    }
    wrist_frames = {
        view_name: decode_video_slice_rgb(
            video, frames, fps=config.source.fps, frame_count=episode.frame_count
        )
        for view_name, video in wrist_videos.items()
    }
    wrist_rectification_maps = {
        view_name: inverse_brown_conrady_rectification_maps(
            np.asarray(camera.intrinsic_matrix_px, dtype=np.float64).reshape(3, 3),
            np.asarray(camera.distortion_coefficients, dtype=np.float64),
            width=camera.width,
            height=camera.height,
        )
        for view_name, camera in zip(
            POSE_VIEW_NAMES[1:], config.cameras[1:], strict=True
        )
    }
    tool_transforms = []
    for side in ("left", "right"):
        raw_transform = config.raw["source_contract"]["fk_tool_transforms"][side]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.asarray(raw_transform["translation_m"], dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(
            raw_transform["quaternion_xyzw"]
        ).as_matrix()
        tool_transforms.append(transform)
    robot_inputs = _robot_frame_inputs(
        episode=episode,
        frames=frames,
        urdf=urdf,
        cameras=config.cameras,
        rectification_left=calibration.rectification_left,
        eef_tool_transforms=tuple(tool_transforms),
    )

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        frame_records = []
        for ordinal, frame_index in enumerate(frames):
            head_rgb, depth_m, diagnostics, stereo_consistency = (
                estimator.estimate_with_confidence(
                    left_frames[frame_index], right_frames[frame_index]
                )
            )
            if diagnostics["valid_fraction"] < 0.10:
                raise ValueError(
                    f"episode {episode.episode_index} frame {frame_index} stereo depth is too sparse: "
                    f"{diagnostics['valid_fraction']:.4f}"
                )
            rgb_by_view = {POSE_VIEW_NAMES[0]: head_rgb}
            for view_name in POSE_VIEW_NAMES[1:]:
                map_x, map_y = wrist_rectification_maps[view_name]
                rgb_by_view[view_name] = cv2.remap(
                    wrist_frames[view_name][frame_index],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            view_records = {}
            for view_name in POSE_VIEW_NAMES:
                rgb_path = temporary / "rgb" / view_name / f"{ordinal:06d}.png"
                _write_png(rgb_path, cv2.cvtColor(rgb_by_view[view_name], cv2.COLOR_RGB2BGR))
                view_records[view_name] = {
                    "rgb": str(rgb_path.relative_to(temporary)),
                    "rgb_sha256": sha256_file(rgb_path),
                    "robot_root_from_rectified_opencv": robot_inputs[frame_index][
                        "robot_root_from_cameras_opencv"
                    ][view_name].reshape(-1).tolist(),
                    "laplacian_variance": _laplacian_variance(rgb_by_view[view_name]),
                }
            depth_path = temporary / "depth" / f"{ordinal:06d}.png"
            _write_png(depth_path, depth_to_uint16_mm(depth_m))
            consistency_path = (
                temporary / "stereo_consistency" / f"{ordinal:06d}.png"
            )
            _write_png(consistency_path, stereo_consistency.astype(np.uint8) * 255)
            view_records[POSE_VIEW_NAMES[0]].update(
                {
                    "depth": str(depth_path.relative_to(temporary)),
                    "depth_sha256": sha256_file(depth_path),
                    "stereo_consistency": str(
                        consistency_path.relative_to(temporary)
                    ),
                    "stereo_consistency_sha256": sha256_file(consistency_path),
                    "stereo": diagnostics,
                }
            )
            frame_records.append(
                {
                    "ordinal": ordinal,
                    "source_frame_index": frame_index,
                    "episode_time_s": frame_index / config.source.fps,
                    "views": view_records,
                    "eef_current_root_from_fk": robot_inputs[frame_index][
                        "eef_current_root_from_fk"
                    ].reshape(2, 16).tolist(),
                    "robot_q_current": robot_inputs[frame_index][
                        "robot_q_current"
                    ].tolist(),
                    "ee_state": robot_inputs[frame_index]["ee_state"].tolist(),
                    "hand_state": robot_inputs[frame_index]["hand_state"].tolist(),
                    "ee_action": robot_inputs[frame_index]["ee_action"].tolist(),
                    "hand_cmd": robot_inputs[frame_index]["hand_cmd"].tolist(),
                }
            )
        atomic_write_text(
            temporary / "K.txt",
            "\n".join(" ".join(f"{value:.12g}" for value in row) for row in estimator.intrinsic_matrix) + "\n",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config.digest,
            "input_contract_sha256": input_contract_sha256,
            "source_repo_id": config.source.repo_id,
            "source_revision": config.source.revision,
            "episode_index": episode.episode_index,
            "source_frame_count": episode.frame_count,
            "source_fps": config.source.fps,
            "source_frame_stride": stride,
            "tracking_fps": config.source.fps / stride,
            "source_frame_indices": list(frames),
            "head_stereo_calibration_repo_id": config.raw_source.repo_id,
            "head_stereo_calibration_revision": config.raw_source.revision,
            "head_stereo_calibration_repo_path": config.raw_source.head_stereo_calibration_repo_path,
            "head_stereo_calibration_sha256": sha256_file(calibration_path),
            "runtime_container_digest": config.runtime.container_digest,
            "opencv_version": cv2.__version__,
            "stereo_depth_schema_version": STEREO_DEPTH_SCHEMA_VERSION,
            "stereo_depth_parameters": asdict(parameters),
            "stereo_runtime": {
                "source_repo": pose.fast_stereo_repo,
                "source_revision": pose.fast_stereo_revision,
                "model_repo": pose.fast_stereo_model_repo,
                "model_revision": pose.fast_stereo_model_revision,
                "model_filename": pose.fast_stereo_model_filename,
                "model_sha256": sha256_file(fast_stereo_model),
                "config_filename": pose.fast_stereo_config_filename,
                "config_sha256": sha256_file(fast_stereo_config),
                "runtime_manifests": runtime_manifests,
            },
            "image_quality_metric": {
                "name": "variance_of_laplacian",
                "opencv_depth": "CV_64F",
                "input": "rectified_uint8_grayscale",
            },
            "source_fk": {
                "urdf": str(urdf),
                "urdf_sha256": sha256_file(urdf),
                "joint_order": list(G1_BODY_JOINT_ORDER),
                "camera_parent_frames": {
                    view_name: parent
                    for view_name, parent in zip(
                        POSE_VIEW_NAMES,
                        ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
                        strict=True,
                    )
                },
                "camera_frames": {
                    view_name: f"rectified_{view_name}_opencv"
                    for view_name in POSE_VIEW_NAMES
                },
            },
            "video_slices": {
                camera_source_keys[0]: {
                    "path": str(left_video.path.relative_to(source_root)),
                    "from_timestamp": left_video.from_timestamp,
                    "to_timestamp": left_video.to_timestamp,
                    "size_bytes": left_video.path.stat().st_size,
                },
                head_right: {
                    "path": str(right_video.path.relative_to(source_root)),
                    "from_timestamp": right_video.from_timestamp,
                    "to_timestamp": right_video.to_timestamp,
                    "size_bytes": right_video.path.stat().st_size,
                },
                **{
                    camera_source_keys[index]: {
                        "path": str(wrist_videos[view_name].path.relative_to(source_root)),
                        "from_timestamp": wrist_videos[view_name].from_timestamp,
                        "to_timestamp": wrist_videos[view_name].to_timestamp,
                        "size_bytes": wrist_videos[view_name].path.stat().st_size,
                    }
                    for index, view_name in enumerate(POSE_VIEW_NAMES[1:], start=1)
                },
            },
            "pose_views": {
                POSE_VIEW_NAMES[0]: {
                    "source_key": camera_source_keys[0],
                    "intrinsic_matrix_px": estimator.intrinsic_matrix.reshape(-1).tolist(),
                    "rectification": "pinned_head_stereo_calibration_left_map",
                    "has_metric_depth": True,
                },
                **{
                    view_name: {
                        "source_key": camera_source_keys[index],
                        "intrinsic_matrix_px": list(config.cameras[index].intrinsic_matrix_px),
                        "rectification": "inverse_brown_conrady_iterative_v1",
                        "raw_distortion_model": config.cameras[index].distortion_model,
                        "raw_distortion_coefficients": list(
                            config.cameras[index].distortion_coefficients
                        ),
                        "has_metric_depth": False,
                    }
                    for index, view_name in enumerate(POSE_VIEW_NAMES[1:], start=1)
                },
            },
            "baseline_m": estimator.baseline_m,
            "frames": frame_records,
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    valid = [
        record["views"][POSE_VIEW_NAMES[0]]["stereo"]["valid_fraction"]
        for record in frame_records
    ]
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": len(frame_records),
                "mean_valid_depth_fraction": float(np.mean(valid)),
                "minimum_valid_depth_fraction": float(np.min(valid)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
