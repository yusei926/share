#!/usr/bin/env python3
"""Derive one V1 head-mount correction from source FK and a V1 trace.

This is an offline calibration tool.  It compares the source head-left camera
pose reconstructed from real encoders with the V1 camera and torso poses
recorded in an evaluation trace.  Its result is an episode-fixed reset
candidate, never a policy observation, action, reward, or planner input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json

from .source_cad_alignment import OPENGL_FROM_OPENCV


SCHEMA_VERSION = "team_ramen_flip_table_source_head_mount_candidate/v1"


def _transform(position: Any, quaternion_xyzw: Any, label: str) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion_array = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        raise ValueError(f"{label} must contain position[3] and quaternion_xyzw[4]")
    if not np.isfinite([*position_array, *quaternion_array]).all():
        raise ValueError(f"{label} must be finite")
    norm = float(np.linalg.norm(quaternion_array))
    if not np.isclose(norm, 1.0, atol=1.0e-3):
        raise ValueError(f"{label} quaternion is not normalized")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion_array / norm).as_matrix()
    result[:3, 3] = position_array
    return result


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return result


def _trace_diagnostic(trace_path: Path, step: int) -> dict[str, Any]:
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) != step:
                continue
            diagnostic = row.get("simulator_scene_diagnostics")
            if isinstance(diagnostic, dict):
                return diagnostic
    raise ValueError(f"trace has no simulator_scene_diagnostics at step {step}")


def _source_camera(source: dict[str, Any], frame_index: int) -> np.ndarray:
    for frame in source.get("frames", []):
        if int(frame.get("frame_index", -1)) != frame_index:
            continue
        poses = frame.get("root_from_opencv_camera")
        if not isinstance(poses, dict):
            break
        return _matrix(poses.get("head_left"), "source head_left root_from_opencv_camera")
    raise ValueError(f"source alignment has no head-left pose at frame {frame_index}")


def _stereo_rig_translation(
    torso_from_left: np.ndarray,
    torso_from_right: np.ndarray,
) -> np.ndarray:
    """Return the authored stereo-rig centre in torso-local coordinates.

    ``assemble_table_task`` applies a camera rotation about the mean of the
    authored left/right camera translations, then adds the supplied offset.
    Keep that exact convention here.  A plain camera-to-camera translation is
    not interchangeable with this rig-centred offset when a rotation is also
    present.
    """

    return 0.5 * (torso_from_left[:3, 3] + torso_from_right[:3, 3])


def _rig_centred_offset(
    *,
    torso_from_current_left: np.ndarray,
    torso_from_current_right: np.ndarray,
    torso_from_desired_left: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Express a desired left-eye pose in the simulator's rig convention."""

    center = _stereo_rig_translation(torso_from_current_left, torso_from_current_right)
    current_left = torso_from_current_left[:3, 3]
    desired_left = torso_from_desired_left[:3, 3]
    # The task evaluates ``center + offset + R @ (camera - center)``.  Solve
    # that equation for ``offset`` so the offline and runtime conventions are
    # identical.  This is deliberately not a per-frame runtime correction.
    return desired_left - center - rotation @ (current_left - center)


def candidate_from_artifacts(
    source_alignment: dict[str, Any], diagnostics: dict[str, Any], *, source_frame: int
) -> dict[str, Any]:
    """Return the task-compatible stereo-rig correction for one frame pair."""

    root_pose = diagnostics.get("root_pose_world_xyzw")
    torso = diagnostics.get("torso_link")
    cameras = diagnostics.get("policy_camera_poses")
    if not isinstance(root_pose, list) or len(root_pose) != 7:
        raise ValueError("trace root_pose_world_xyzw is unavailable")
    if not isinstance(torso, dict) or not isinstance(cameras, dict):
        raise ValueError("trace must contain torso_link and policy_camera_poses")
    head_left = cameras.get("head_left")
    head_right = cameras.get("head_right")
    if not isinstance(head_left, dict) or not isinstance(head_right, dict):
        raise ValueError("trace must contain both head-stereo camera poses")

    world_from_root = _transform(root_pose[:3], root_pose[3:], "simulator root")
    world_from_torso = _transform(
        torso.get("position_world_m"), torso.get("quaternion_xyzw"), "simulator torso"
    )
    world_from_opengl_left = _transform(
        head_left.get("position_world_m"),
        head_left.get("quaternion_xyzw"),
        "simulator head-left camera",
    )
    world_from_opengl_right = _transform(
        head_right.get("position_world_m"),
        head_right.get("quaternion_xyzw"),
        "simulator head-right camera",
    )
    world_from_opencv_left = world_from_opengl_left @ OPENGL_FROM_OPENCV
    world_from_opencv_right = world_from_opengl_right @ OPENGL_FROM_OPENCV
    desired_world_from_opencv_camera = world_from_root @ _source_camera(
        source_alignment, source_frame
    )

    torso_from_current_left = np.linalg.inv(world_from_torso) @ world_from_opencv_left
    torso_from_current_right = np.linalg.inv(world_from_torso) @ world_from_opencv_right
    torso_from_desired = np.linalg.inv(world_from_torso) @ desired_world_from_opencv_camera
    rotation = torso_from_desired[:3, :3] @ torso_from_current_left[:3, :3].T
    translation = _rig_centred_offset(
        torso_from_current_left=torso_from_current_left,
        torso_from_current_right=torso_from_current_right,
        torso_from_desired_left=torso_from_desired,
        rotation=rotation,
    )
    rpy_deg = Rotation.from_matrix(rotation).as_euler("XYZ", degrees=True)
    return {
        "head_stereo_offset_local_m": [float(value) for value in translation],
        "head_stereo_rotation_rpy_deg": [float(value) for value in rpy_deg],
        "source_to_v1_translation_m": float(
            np.linalg.norm(torso_from_desired[:3, 3] - torso_from_current_left[:3, 3])
        ),
        "source_to_v1_rotation_deg": float(np.degrees(Rotation.from_matrix(rotation).magnitude())),
        "offset_convention": "task_stereo_rig_center_after_rotation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--scene-candidate", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--sim-step", type=int, default=119)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_alignment.expanduser().resolve().read_text(encoding="utf-8"))
    if not source.get("accepted_for_fixed_scene_proposal", False):
        raise ValueError("source CAD alignment did not pass its consistency gate")
    scene = json.loads(args.scene_candidate.expanduser().resolve().read_text(encoding="utf-8"))
    candidates = scene.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("scene candidate report must contain exactly one reset candidate")

    correction = candidate_from_artifacts(
        source,
        _trace_diagnostic(args.sim_trace.expanduser().resolve(), args.sim_step),
        source_frame=args.source_frame,
    )
    candidate = dict(candidates[0])
    candidate.update(
        {
            "head_stereo_offset_local_m": correction["head_stereo_offset_local_m"],
            "head_stereo_rotation_rpy_deg": correction["head_stereo_rotation_rpy_deg"],
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline reset calibration only",
        "source_alignment": str(args.source_alignment.expanduser().resolve()),
        "scene_candidate": str(args.scene_candidate.expanduser().resolve()),
        "sim_trace": str(args.sim_trace.expanduser().resolve()),
        "source_frame": args.source_frame,
        "sim_step": args.sim_step,
        "correction": correction,
        "candidates": [candidate],
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
