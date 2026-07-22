#!/usr/bin/env python3
"""Verify a source-CAD fixed scene against one V1 diagnostic trace.

The source fit is built from partial rim/leg evidence over a stereo sequence;
this verifier intentionally does *not* require all four physical corners to
be visible in any RGB frame. It projects the CAD corners inferred by that fit
into both the source camera and the V1 trace, then compares their cyclic
correspondence. This is offline reset calibration only: no result can enter
an observation, policy, planner, reward, or inference-time branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    CameraCalibration,
    TabletopPoseEstimator,
)

from .source_cad_alignment import OPENGL_FROM_OPENCV, TABLE_YAW_180_SYMMETRY
from .source_head_mount_candidate import _source_camera, _trace_diagnostic, _transform


SCHEMA_VERSION = "team_ramen_flip_table_source_projection_conformance/v1"
_CAMERA_TRANSLATION_GATE_M = 0.005
_CAMERA_ROTATION_GATE_DEG = 0.5
_TABLE_TRANSLATION_GATE_M = 0.006
_TABLE_ROTATION_GATE_DEG = 0.75
_CAD_PROJECTION_GATE_PX = 3.0


def _cad_outer_corners() -> np.ndarray:
    return np.asarray(
        (
            (
                -TabletopPoseEstimator._CAD_OUTER_X_M,
                -TabletopPoseEstimator._CAD_OUTER_Y_M,
                TabletopPoseEstimator._CAD_TABLETOP_Z_M,
            ),
            (
                TabletopPoseEstimator._CAD_OUTER_X_M,
                -TabletopPoseEstimator._CAD_OUTER_Y_M,
                TabletopPoseEstimator._CAD_TABLETOP_Z_M,
            ),
            (
                TabletopPoseEstimator._CAD_OUTER_X_M,
                TabletopPoseEstimator._CAD_OUTER_Y_M,
                TabletopPoseEstimator._CAD_TABLETOP_Z_M,
            ),
            (
                -TabletopPoseEstimator._CAD_OUTER_X_M,
                TabletopPoseEstimator._CAD_OUTER_Y_M,
                TabletopPoseEstimator._CAD_TABLETOP_Z_M,
            ),
        ),
        dtype=np.float64,
    )


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return matrix


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())
    )


def _canonical_table(table: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resolve the physical 180-degree yaw symmetry before scoring a table."""

    return min(
        (table, table @ TABLE_YAW_180_SYMMETRY),
        key=lambda item: _rotation_error_deg(reference, item),
    )


def _project(
    corners: np.ndarray,
    camera_from_table: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
    pixels, _ = cv2.projectPoints(
        corners, rotation_vector, camera_from_table[:3, 3], intrinsic, distortion
    )
    return pixels.reshape(4, 2)


def _cyclic_rmse(first: np.ndarray, second: np.ndarray) -> tuple[float, int, bool]:
    """Return correspondence-free corner RMSE for a physically symmetric table."""

    candidates: list[tuple[float, int, bool]] = []
    for reversed_order in (False, True):
        ordered = second[::-1] if reversed_order else second
        for shift in range(4):
            residual = first - np.roll(ordered, shift, axis=0)
            candidates.append(
                (
                    float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
                    shift,
                    reversed_order,
                )
            )
    return min(candidates, key=lambda item: item[0])


def conformance_from_artifacts(
    source_alignment: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    source_frame: int,
) -> dict[str, Any]:
    """Compute a geometry-only source/V1 conformance report."""

    root_pose = diagnostics.get("root_pose_world_xyzw")
    torso = diagnostics.get("torso_link")
    cameras = diagnostics.get("policy_camera_poses")
    table = diagnostics.get("white_table")
    if not isinstance(root_pose, list) or len(root_pose) != 7:
        raise ValueError("trace root_pose_world_xyzw is unavailable")
    if not isinstance(torso, dict) or not isinstance(cameras, dict) or not isinstance(table, dict):
        raise ValueError("trace lacks torso, policy camera, or white-table diagnostics")
    head_left = cameras.get("head_left")
    if not isinstance(head_left, dict):
        raise ValueError("trace head-left camera pose is unavailable")

    source_root_from_camera = _source_camera(source_alignment, source_frame)
    source_root_from_table = _matrix(
        source_alignment.get("fixed_scene_root_from_table"),
        "source fixed_scene_root_from_table",
    )
    world_from_root = _transform(root_pose[:3], root_pose[3:], "simulator root")
    world_from_camera = _transform(
        head_left.get("position_world_m"),
        head_left.get("quaternion_xyzw"),
        "simulator head-left",
    ) @ OPENGL_FROM_OPENCV
    world_from_table = _transform(
        table.get("position_world_m"),
        table.get("quaternion_xyzw"),
        "simulator white table",
    )
    desired_world_from_camera = world_from_root @ source_root_from_camera
    desired_world_from_table = world_from_root @ source_root_from_table
    world_from_table = _canonical_table(world_from_table, desired_world_from_table)

    intrinsic, distortion = CameraCalibration.g1_head_left_real_raw_intrinsics()
    corners = _cad_outer_corners()
    source_pixels = _project(
        corners,
        np.linalg.inv(source_root_from_camera) @ source_root_from_table,
        intrinsic,
        distortion,
    )
    sim_pixels = _project(
        corners,
        np.linalg.inv(world_from_camera) @ world_from_table,
        intrinsic,
        distortion,
    )
    projection_rmse_px, cyclic_shift, reversed_order = _cyclic_rmse(source_pixels, sim_pixels)
    camera_translation_m = float(np.linalg.norm(desired_world_from_camera[:3, 3] - world_from_camera[:3, 3]))
    table_translation_m = float(np.linalg.norm(desired_world_from_table[:3, 3] - world_from_table[:3, 3]))
    camera_rotation_deg = _rotation_error_deg(desired_world_from_camera, world_from_camera)
    table_rotation_deg = _rotation_error_deg(desired_world_from_table, world_from_table)
    passed = bool(
        camera_translation_m <= _CAMERA_TRANSLATION_GATE_M
        and camera_rotation_deg <= _CAMERA_ROTATION_GATE_DEG
        and table_translation_m <= _TABLE_TRANSLATION_GATE_M
        and table_rotation_deg <= _TABLE_ROTATION_GATE_DEG
        and projection_rmse_px <= _CAD_PROJECTION_GATE_PX
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline reset calibration only",
        "uses_simulator_ground_truth": False,
        "method": "source stereo/FK/CAD partial-feature fit followed by CAD reprojection",
        "source_frame": source_frame,
        "gates": {
            "camera_translation_m_max": _CAMERA_TRANSLATION_GATE_M,
            "camera_rotation_deg_max": _CAMERA_ROTATION_GATE_DEG,
            "table_translation_m_max": _TABLE_TRANSLATION_GATE_M,
            "table_rotation_deg_max": _TABLE_ROTATION_GATE_DEG,
            "cad_projection_rmse_px_max": _CAD_PROJECTION_GATE_PX,
        },
        "metrics": {
            "camera_translation_m": camera_translation_m,
            "camera_rotation_deg": camera_rotation_deg,
            "table_translation_m": table_translation_m,
            "table_rotation_deg": table_rotation_deg,
            "cad_projection_rmse_px": projection_rmse_px,
            "corner_correspondence": {
                "cyclic_shift": cyclic_shift,
                "reversed": reversed_order,
            },
        },
        "passed": passed,
        "notes": [
            "The source fit uses CAD rim and leg-axis evidence across time; no source frame must expose four corners.",
            "Rendered RGB colour/texture similarity is deliberately not a geometric acceptance metric.",
            "Source object pose and trace telemetry are offline diagnostics, never policy features.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--sim-step", type=int, default=119)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_alignment.expanduser().resolve().read_text(encoding="utf-8"))
    if not source.get("accepted_for_fixed_scene_proposal", False):
        raise ValueError("source CAD alignment did not pass its temporal/stereo consistency gate")
    report = conformance_from_artifacts(
        source,
        _trace_diagnostic(args.sim_trace.expanduser().resolve(), args.sim_step),
        source_frame=args.source_frame,
    )
    report.update(
        {
            "source_alignment": str(args.source_alignment.expanduser().resolve()),
            "sim_trace": str(args.sim_trace.expanduser().resolve()),
            "sim_step": args.sim_step,
        }
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
