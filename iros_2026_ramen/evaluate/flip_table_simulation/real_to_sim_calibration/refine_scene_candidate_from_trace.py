#!/usr/bin/env python3
"""Refine one fixed table-reset candidate from an offline realized trace.

The source CAD fit identifies the table relative to the recorded head-left
camera.  A V1 reset can still realize a different pose because the table is a
dynamic PhysX articulation and its authored root differs from the CAD body
frame.  This tool closes that *offline reset-calibration* loop once: it
measures the realized table/camera transform in a saved trace, converts the
residual to a workbench-local reset offset and yaw, and writes a new candidate
for a fresh reset.  It never runs during a policy episode and never exposes
trace, camera pose, or table pose to a policy, planner, or reward.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json

from .source_cad_alignment import OPENGL_FROM_OPENCV, TABLE_YAW_180_SYMMETRY
from .source_head_mount_candidate import _matrix, _source_camera, _trace_diagnostic, _transform
from .source_scene_candidate import _workbench_support_envelope


SCHEMA_VERSION = "team_ramen_flip_table_trace_refined_scene_candidate/v1"
MAXIMUM_RESIDUAL_TILT_DEG = 3.0


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()))


def _canonical_target_table(
    desired_world_from_table: np.ndarray,
    actual_world_from_table: np.ndarray,
) -> np.ndarray:
    """Resolve the physical 180-degree tabletop symmetry before correction."""

    return min(
        (desired_world_from_table, desired_world_from_table @ TABLE_YAW_180_SYMMETRY),
        key=lambda value: _rotation_error_deg(actual_world_from_table, value),
    )


def _yaw_and_tilt_from_relative(rotation: np.ndarray) -> tuple[float, float]:
    """Return the planar reset correction and report unsupported residual tilt."""

    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    planar = Rotation.from_euler("z", yaw).as_matrix()
    tilt = float(np.degrees(Rotation.from_matrix(planar.T @ rotation).magnitude()))
    return yaw, tilt


def _camera_relative_errors(
    reference_camera_from_table: np.ndarray,
    candidate_camera_from_table: np.ndarray,
) -> tuple[float, float]:
    """Score table geometry while respecting the tabletop's 180-degree symmetry."""

    canonical = _canonical_target_table(reference_camera_from_table, candidate_camera_from_table)
    return (
        float(np.linalg.norm(canonical[:3, 3] - candidate_camera_from_table[:3, 3])),
        _rotation_error_deg(canonical, candidate_camera_from_table),
    )


def refine_candidate(
    source_alignment: dict[str, Any],
    candidate: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    source_frame: int,
) -> dict[str, Any]:
    """Return a support-gated candidate corrected from realized camera geometry."""

    source_camera = _source_camera(source_alignment, source_frame)
    source_table = _matrix(
        source_alignment.get("fixed_scene_root_from_table"), "source fixed_scene_root_from_table"
    )
    cameras = diagnostics.get("policy_camera_poses")
    table = diagnostics.get("white_table")
    workbench = diagnostics.get("workbench")
    if not isinstance(cameras, dict) or not isinstance(table, dict) or not isinstance(workbench, dict):
        raise ValueError("trace lacks policy camera, table, or workbench diagnostics")
    head_left = cameras.get("head_left")
    if not isinstance(head_left, dict):
        raise ValueError("trace lacks a head-left camera pose")

    world_from_camera = _transform(
        head_left.get("position_world_m"), head_left.get("quaternion_xyzw"), "simulator head-left"
    ) @ OPENGL_FROM_OPENCV
    actual_world_from_table = _transform(
        table.get("position_world_m"), table.get("quaternion_xyzw"), "simulator table"
    )
    world_from_workbench = _transform(
        workbench.get("position_world_m"), workbench.get("quaternion_xyzw"), "simulator workbench"
    )

    # This relative transform is directly observable in both domains.  It
    # avoids treating the unobserved free-base world pose as a camera/table
    # calibration target.
    camera_from_table = np.linalg.inv(source_camera) @ source_table
    desired_world_from_table = world_from_camera @ camera_from_table
    desired_world_from_table = _canonical_target_table(
        desired_world_from_table, actual_world_from_table
    )
    relative_rotation = desired_world_from_table[:3, :3] @ actual_world_from_table[:3, :3].T
    yaw_delta, residual_tilt_deg = _yaw_and_tilt_from_relative(relative_rotation)
    if residual_tilt_deg > MAXIMUM_RESIDUAL_TILT_DEG:
        raise ValueError(
            "trace residual contains non-planar table rotation "
            f"({residual_tilt_deg:.3f} deg > {MAXIMUM_RESIDUAL_TILT_DEG:.3f} deg)"
        )

    old_offset = np.asarray(candidate.get("offset_local_m"), dtype=np.float64)
    if old_offset.shape != (3,) or not np.isfinite(old_offset).all():
        raise ValueError("candidate offset_local_m must be finite [3]")
    old_yaw = float(candidate.get("yaw_rad"))
    if not math.isfinite(old_yaw):
        raise ValueError("candidate yaw_rad must be finite")
    delta_world = desired_world_from_table[:3, 3] - actual_world_from_table[:3, 3]
    delta_local_raw = world_from_workbench[:3, :3].T @ delta_world
    # The source CAD and V1 articulation origins do not establish a common
    # vertical datum. Applying that residual can put the table through the
    # workbench, so this offline reset loop is intentionally planar.
    unapplied_vertical_offset_m = float(delta_local_raw[2])
    delta_local = delta_local_raw.copy()
    delta_local[2] = 0.0
    applied_delta_world = world_from_workbench[:3, :3] @ delta_local

    refined = dict(candidate)
    refined["offset_local_m"] = (old_offset + delta_local).tolist()
    refined["yaw_rad"] = float(old_yaw + yaw_delta)
    expected_world_from_table = actual_world_from_table.copy()
    expected_world_from_table[:3, 3] += applied_delta_world
    expected_world_from_table[:3, :3] = Rotation.from_euler("z", yaw_delta).as_matrix() @ (
        actual_world_from_table[:3, :3]
    )
    support = _workbench_support_envelope(expected_world_from_table, world_from_workbench)
    if not support["accepted_for_v1_probe"]:
        raise ValueError("refined reset violates the bounded physical workbench support envelope")

    before = np.linalg.inv(world_from_camera) @ actual_world_from_table
    after = np.linalg.inv(world_from_camera) @ expected_world_from_table
    before_translation, before_rotation = _camera_relative_errors(camera_from_table, before)
    after_translation, after_rotation = _camera_relative_errors(camera_from_table, after)
    return {
        "candidate": refined,
        "support": support,
        "method": "single offline camera-relative realized-reset correction",
        "source_camera_from_table": camera_from_table.tolist(),
        "before": {
            "translation_error_m": before_translation,
            "rotation_error_deg": before_rotation,
        },
        "requested_correction": {
            "translation_workbench_local_m": delta_local.tolist(),
            "unapplied_vertical_offset_workbench_local_m": unapplied_vertical_offset_m,
            "vertical_offset_identifiability": (
                "unidentified_between_source_CAD_and_V1_articulation_origins; "
                "fixed_to_zero_for_physical_workbench_support"
            ),
            "yaw_delta_deg": float(np.degrees(yaw_delta)),
            "unapplied_residual_tilt_deg": residual_tilt_deg,
        },
        "expected_after_planar_reset": {
            "translation_error_m": after_translation,
            "rotation_error_deg": after_rotation,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--sim-step", type=int, default=119)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_alignment.expanduser().resolve().read_text(encoding="utf-8"))
    if source.get("accepted_for_fixed_scene_proposal") is not True:
        raise ValueError("source CAD alignment did not pass its consistency gate")
    candidates = json.loads(args.candidate.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("candidate file must contain exactly one candidate")
    result = refine_candidate(
        source,
        candidates[0],
        _trace_diagnostic(args.sim_trace.expanduser().resolve(), args.sim_step),
        source_frame=args.source_frame,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline reset calibration only",
        "source_alignment": str(args.source_alignment.expanduser().resolve()),
        "candidate_input": str(args.candidate.expanduser().resolve()),
        "sim_trace": str(args.sim_trace.expanduser().resolve()),
        "source_frame": args.source_frame,
        "sim_step": args.sim_step,
        "accepted_for_v1_probe": True,
        **result,
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
