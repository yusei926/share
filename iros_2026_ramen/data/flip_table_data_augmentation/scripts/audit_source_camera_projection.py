#!/usr/bin/env python3
"""Overlay source G1 arm FK on head-left frames to audit camera extrinsics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
import pyarrow.parquet as pq

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_camera_projection import (
    ARM_FRAME_NAMES,
    PROJECTION_AUDIT_SCHEMA_VERSION,
    draw_arm_projection,
    project_root_points,
    projection_json,
    root_from_camera,
    sha256_file,
    visible_mask,
)
from data.flip_table_data_augmentation.source_contract import snapshot_download_pinned
from data.flip_table_data_augmentation.source_dataset import (
    SourceDatasetIndex,
    extract_video_frame,
    select_review_frames,
)


DEFAULT_URDF = Path(
    "/workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/"
    "g1_29dof_with_hand.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _transform(placement) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = placement.rotation
    value[:3, 3] = placement.translation
    return value


def main() -> None:
    import pinocchio as pin

    args = parse_args()
    config = load_pipeline_config(args.config)
    camera = config.cameras[0]
    if camera.source_key != "observation.images.cam_0":
        raise ValueError("the first configured policy camera must be source head-left cam_0")
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else snapshot_download_pinned(config, include_videos=True)
    )
    episode = SourceDatasetIndex(source_root).episode(args.episode_index)
    frames = tuple(args.frames or select_review_frames(episode.frame_count, count=5))
    if tuple(sorted(set(frames))) != frames or frames[0] < 0 or frames[-1] >= episode.frame_count:
        raise ValueError("frames must be sorted, unique, and inside the episode")

    rows = pq.read_table(
        episode.data_path,
        columns=["frame_index", "timestamp", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode.episode_index)],
    ).to_pylist()
    rows_by_frame = {int(row["frame_index"]): row for row in rows}
    if any(frame not in rows_by_frame for frame in frames):
        raise ValueError("selected source data frame is missing")

    urdf = args.urdf.expanduser().resolve()
    # This is a kinematic projection audit.  Loading visual/collision meshes
    # through RobotWrapper would make it depend on an asset bundle that is
    # irrelevant to FK and often intentionally absent from a calibration run.
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    required_frames = ("torso_link",) + tuple(
        name for names in ARM_FRAME_NAMES.values() for name in names
    )
    missing = [name for name in required_frames if not model.existFrame(name)]
    if missing:
        raise ValueError(f"source projection URDF lacks frames: {missing}")
    joint_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
        dtype=np.int64,
    )
    torso_id = model.getFrameId("torso_link")
    arm_ids = {
        side: tuple(model.getFrameId(name) for name in names)
        for side, names in ARM_FRAME_NAMES.items()
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = episode.video_slice(camera.source_key)
    records = []
    visible_distal_counts = {"left": 0, "right": 0}
    for frame in frames:
        row = rows_by_frame[frame]
        source_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        if source_q.shape != (36,) or not np.isfinite(source_q).all():
            raise ValueError(f"frame {frame} robot_q_current is not a finite 36D vector")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        root_camera = root_from_camera(_transform(data.oMf[torso_id]), camera)
        projections = {}
        projection_records = {}
        for side in ("left", "right"):
            points = np.stack([np.asarray(data.oMf[index].translation) for index in arm_ids[side]])
            pixels, depth = project_root_points(points, root_camera, camera)
            projections[side] = (pixels, depth)
            projection_records[side] = projection_json(
                ARM_FRAME_NAMES[side], pixels, depth, camera
            )
            if visible_mask(pixels[3:], depth[3:], camera).any():
                visible_distal_counts[side] += 1

        raw_path = output / "raw" / f"frame_{frame:06d}.png"
        timestamp = video.timestamp_for_frame(frame, config.source.fps, episode.frame_count)
        extract_video_frame(video.path, timestamp, raw_path)
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode {raw_path}")
        overlay = draw_arm_projection(image, projections, camera)
        overlay_path = output / "overlays" / f"frame_{frame:06d}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = overlay_path.with_name(f".{overlay_path.stem}.{os.getpid()}.tmp.png")
        if not cv2.imwrite(str(temporary), overlay):
            raise RuntimeError(f"OpenCV could not write {temporary}")
        os.replace(temporary, overlay_path)
        records.append(
            {
                "frame_index": frame,
                "episode_timestamp_s": float(row["timestamp"]),
                "video_timestamp_s": timestamp,
                "raw_image": str(raw_path.relative_to(output)),
                "raw_image_sha256": sha256_file(raw_path),
                "overlay": str(overlay_path.relative_to(output)),
                "overlay_sha256": sha256_file(overlay_path),
                "projection": projection_records,
            }
        )

    minimum_visible = max(1, len(frames) - 1)
    automated_pass = all(count >= minimum_visible for count in visible_distal_counts.values())
    report = {
        "schema_version": PROJECTION_AUDIT_SCHEMA_VERSION,
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "episode_index": episode.episode_index,
        "config_sha256": config.digest,
        "urdf": str(urdf),
        "urdf_sha256": sha256_file(urdf),
        "camera": {
            "source_key": camera.source_key,
            "sim_sensor": camera.sim_sensor,
            "offset_position_m": list(camera.offset_position_m),
            "offset_quaternion_xyzw": list(camera.offset_quaternion_xyzw),
            "intrinsic_matrix_px": list(camera.intrinsic_matrix_px),
            "distortion_model": camera.distortion_model,
            "distortion_coefficients": list(camera.distortion_coefficients),
            "intrinsic_calibration_sha256s": list(camera.intrinsic_calibration_sha256s),
        },
        "automated_gate": {
            "minimum_frames_with_visible_distal_arm_per_side": minimum_visible,
            "frames_with_visible_distal_arm": visible_distal_counts,
            "pass": automated_pass,
        },
        "visual_review_required": True,
        "frames": records,
    }
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        output / "visual_review.template.json",
        {
            "schema_version": "team_ramen_source_camera_projection_review/v1",
            "report_sha256": sha256_file(output / "report.json"),
            "reviewer": "",
            "decision": "pending",
            "requirements": {
                "left_fk_chain_tracks_visible_left_arm": False,
                "right_fk_chain_tracks_visible_right_arm": False,
                "wrist_points_land_on_their_physical_wrist_links": False,
                "no_systematic_pixel_offset_across_review_frames": False,
            },
            "notes": "",
        },
    )
    print(json.dumps(report["automated_gate"], sort_keys=True))
    if not automated_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
