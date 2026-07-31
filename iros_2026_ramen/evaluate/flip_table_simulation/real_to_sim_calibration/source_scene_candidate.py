#!/usr/bin/env python3
"""Convert a source CAD fixed-scene fit into V1 reset candidates.

This program bridges two offline calibration artifacts: a source RGB/FK/CAD
alignment and a V1 recorder trace.  It emits workbench-local offsets and yaw
increments consumed once at reset by ``FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON``.
The source pose, simulator pose, and workbench pose are never policy inputs.
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


SCHEMA_VERSION = "team_ramen_flip_table_source_scene_candidate/v1"
TABLE_YAW_180_SYMMETRY = np.diag((-1.0, -1.0, 1.0, 1.0))
WORKBENCH_HALF_LENGTH_M = 0.90
WORKBENCH_HALF_DEPTH_M = 0.375
TABLETOP_HALF_EXTENTS_M = np.asarray((0.29, 0.21), dtype=np.float64)
CALIBRATION_SUPPORT_CENTER_MARGIN_M = 0.05
CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION = 0.70


def _transform(position: Any, quaternion_xyzw: Any, label: str) -> np.ndarray:
    translation = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite([*translation, *quaternion]).all():
        raise ValueError(f"{label} must contain finite position[3] and quaternion_xyzw[4]")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-3):
        raise ValueError(f"{label} quaternion is not normalized")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion / np.linalg.norm(quaternion)).as_matrix()
    result[:3, 3] = translation
    return result


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError(f"{label} has an invalid homogeneous row")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1.0e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    return result


def _reset_robot_pose(diagnostics: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Read the V1 reset root that produced the synchronized baseline trace.

    A calibrated table position must not make the ordinary task placement
    heuristic move G1 by the same offset. This is the trace's one-time local
    reset default, never a source-side root measurement or a dynamic pose.
    """

    randomization = diagnostics.get("randomization")
    robot = randomization.get("robot") if isinstance(randomization, dict) else None
    if not isinstance(robot, dict):
        raise ValueError("trace diagnostics lack the reset randomization.robot record")
    position = np.asarray(robot.get("position_local_m"), dtype=np.float64)
    try:
        yaw = float(robot.get("yaw_rad"))
    except (TypeError, ValueError) as exc:
        raise ValueError("trace reset robot yaw_rad must be numeric") from exc
    if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError("trace reset robot pose must be finite")
    return position, yaw


def _trace_diagnostics_at_step(trace_path: Path, simulator_step: int) -> dict[str, Any]:
    """Read scene diagnostics at the exact RGB comparison control step.

    A replay frame at source index zero is written after the deterministic
    warmup, not at reset step zero.  Deriving a reset candidate from an
    earlier floating-base pose silently folds WBC settling motion into the
    table offset, so this lookup is deliberately exact and fail-closed.
    """

    if simulator_step < 0:
        raise ValueError("simulator_step must be non-negative")
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("step") != simulator_step:
                continue
            diagnostics = row.get("simulator_scene_diagnostics")
            if isinstance(diagnostics, dict):
                return diagnostics
    raise ValueError(
        f"trace contains no simulator_scene_diagnostics at synchronized step {simulator_step}"
    )


def _yaw_delta(base_rotation: np.ndarray, desired_rotation: np.ndarray) -> float:
    """Return the smaller physically equivalent Z-yaw from base to desired."""

    candidates = []
    for symmetry in (np.eye(4), TABLE_YAW_180_SYMMETRY):
        relative = desired_rotation @ symmetry[:3, :3] @ base_rotation.T
        tilt = float(np.linalg.norm(relative[:2, 2])) + abs(float(relative[2, 0])) + abs(float(relative[2, 1]))
        yaw = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
        candidates.append((tilt, abs(yaw), yaw))
    minimum_tilt = min(candidate[0] for candidate in candidates)
    if minimum_tilt > 0.03:
        raise ValueError(
            "source and V1 table orientations differ by non-yaw tilt "
            f"{minimum_tilt:.5f}"
        )
    # Both physical 180-degree table symmetries have the same ideal tilt.
    # Floating-point roundoff can make one tilt smaller by ~1e-17, so choose
    # the smallest yaw only among geometrically equivalent candidates instead
    # of accidentally preferring a near-pi reset.
    tilt_tolerance = max(1.0e-8, minimum_tilt * 1.0e-5)
    _, _, yaw = min(
        (
            candidate
            for candidate in candidates
            if candidate[0] <= minimum_tilt + tilt_tolerance
        ),
        key=lambda candidate: candidate[1],
    )
    return float(yaw)


def _workbench_support_envelope(
    world_from_table: np.ndarray,
    world_from_workbench: np.ndarray,
) -> dict[str, object]:
    """Quantify the physical tabletop overhang of one fixed reset pose.

    This is a geometry gate for offline scene calibration only.  It prevents a
    visually plausible CAD fit from producing a table reset that would be
    insufficiently supported by the 0.75 m workbench. The center safety margin
    and projected support fraction match the simulator reset task.
    """

    workbench_from_table = np.linalg.inv(world_from_workbench) @ world_from_table
    center_local = workbench_from_table[:2, 3]
    projected_half_extents = np.abs(workbench_from_table[:2, :2]) @ TABLETOP_HALF_EXTENTS_M
    workbench_half_extents = np.asarray(
        (WORKBENCH_HALF_LENGTH_M, WORKBENCH_HALF_DEPTH_M), dtype=np.float64
    )
    tabletop_min = center_local - projected_half_extents
    tabletop_max = center_local + projected_half_extents
    overlap_extent = np.maximum(
        np.minimum(tabletop_max, workbench_half_extents)
        - np.maximum(tabletop_min, -workbench_half_extents),
        0.0,
    )
    support_fraction = float(
        np.prod(overlap_extent) / np.prod(2.0 * projected_half_extents)
    )
    overhang_by_axis = np.maximum(
        np.abs(center_local) + projected_half_extents - workbench_half_extents,
        0.0,
    )
    max_overhang = float(np.max(overhang_by_axis))
    center_within_support = bool(
        np.all(np.abs(center_local) <= workbench_half_extents - CALIBRATION_SUPPORT_CENTER_MARGIN_M)
    )
    accepted = center_within_support and support_fraction >= CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION
    return {
        "table_center_workbench_local_m": [float(value) for value in center_local],
        "projected_tabletop_half_extents_m": [float(value) for value in projected_half_extents],
        "overhang_by_workbench_axis_m": [float(value) for value in overhang_by_axis],
        "max_overhang_m": max_overhang,
        "projected_support_fraction": support_fraction,
        "support_center_margin_m": CALIBRATION_SUPPORT_CENTER_MARGIN_M,
        "min_projected_support_fraction": CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION,
        "center_within_support": center_within_support,
        "accepted_for_v1_probe": accepted,
    }


def candidate_from_artifacts(source_alignment: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return a single source-derived workbench-local reset candidate."""

    source_pose = _matrix(source_alignment.get("fixed_scene_root_from_table"), "source fixed_scene_root_from_table")
    root = _transform(
        diagnostics.get("root_pose_world_xyzw", [])[:3],
        diagnostics.get("root_pose_world_xyzw", [])[3:],
        "simulator root",
    )
    reset_robot_position_local, reset_robot_yaw = _reset_robot_pose(diagnostics)
    table = diagnostics.get("white_table")
    workbench = diagnostics.get("workbench")
    if not isinstance(table, dict) or not isinstance(workbench, dict):
        raise ValueError("trace must contain white_table and workbench diagnostic poses")
    world_from_table = _transform(table.get("position_world_m"), table.get("quaternion_xyzw"), "simulator table")
    world_from_workbench = _transform(workbench.get("position_world_m"), workbench.get("quaternion_xyzw"), "simulator workbench")
    desired_world_from_table = root @ source_pose
    offset_world = desired_world_from_table[:3, 3] - world_from_table[:3, 3]
    offset_workbench = world_from_workbench[:3, :3].T @ offset_world
    # Source RGB/FK estimates the table CAD frame, while the V1 assembled
    # articulation is reset at its own rigid-body origin. Their vertical
    # origin offset is not observable from this dataset alone. Applying that
    # difference as a reset translation can start the table inside the
    # workbench and let PhysX drop it. Keep the V1's measured support height
    # fixed; only planar placement and yaw are identified here.
    unapplied_vertical_offset_m = float(offset_workbench[2])
    offset_workbench[2] = 0.0
    yaw = _yaw_delta(world_from_table[:3, :3], desired_world_from_table[:3, :3])
    root_rotation = root[:3, :3]
    root_tilt = float(np.linalg.norm(root_rotation[:2, 2])) + abs(float(root_rotation[2, 0])) + abs(
        float(root_rotation[2, 1])
    )
    if root_tilt > 0.03:
        raise ValueError("fixed-base calibration trace has a non-planar robot root")
    # The synchronized root pose maps source table coordinates to the V1
    # world. The separate reset root below only preserves the V1 baseline
    # placement while this table candidate is applied once at reset.
    root_yaw = math.atan2(float(root_rotation[1, 0]), float(root_rotation[0, 0]))
    temporal = source_alignment.get("temporal_consistency", {})
    support = _workbench_support_envelope(desired_world_from_table, world_from_workbench)
    return {
        "label": f"source_episode_{source_alignment.get('source', {}).get('episode_index', 'unknown')}",
        "offset_local_m": [float(value) for value in offset_workbench],
        "unapplied_vertical_offset_local_m": unapplied_vertical_offset_m,
        "vertical_offset_identifiability": (
            "unidentified_between_source_CAD_and_V1_articulation_origins; "
            "fixed_to_zero_for_physical_workbench_support"
        ),
        "yaw_rad": yaw,
        "synchronized_root_pose_reference": {
            "position_world_m": [float(value) for value in root[:3, 3]],
            "yaw_rad": root_yaw,
            "use": "diagnostic world transform only; never a dynamic root pose",
        },
        "robot_root_pos_local_m": [float(value) for value in reset_robot_position_local],
        "robot_root_yaw_rad": reset_robot_yaw,
        "robot_root_reset": (
            "one reset-time V1 baseline root default; preserves G1 placement while "
            "the table reset is calibrated, never a per-frame root teleport"
        ),
        "source_translation_spread_p95_m": float(temporal.get("translation_spread_p95_m", float("nan"))),
        "source_rotation_spread_p95_deg": float(temporal.get("rotation_spread_p95_deg", float("nan"))),
        "workbench_support": support,
        "accepted_for_v1_probe": support["accepted_for_v1_probe"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument(
        "--sim-step",
        type=int,
        default=119,
        help="trace step for source RGB frame zero; default: terminal 120-step replay warmup",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_alignment.expanduser().resolve().read_text(encoding="utf-8"))
    if not source.get("accepted_for_fixed_scene_proposal", False):
        raise ValueError("source CAD alignment did not pass its internal consistency gate")
    diagnostics = _trace_diagnostics_at_step(
        args.sim_trace.expanduser().resolve(), args.sim_step
    )
    candidate = candidate_from_artifacts(source, diagnostics)
    accepted = bool(candidate["accepted_for_v1_probe"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline reset calibration only",
        "source_alignment": str(args.source_alignment.expanduser().resolve()),
        "sim_trace": str(args.sim_trace.expanduser().resolve()),
        "simulator_step": args.sim_step,
        "candidate": candidate,
        "accepted_for_v1_probe": accepted,
        "rejection_reason": (
            None
            if accepted
            else "source-derived table pose exceeds the bounded physical workbench support envelope"
        ),
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key
                in {
                    "label",
                    "offset_local_m",
                    "yaw_rad",
                    "robot_root_pos_local_m",
                    "robot_root_yaw_rad",
                }
            }
        ] if accepted else [],
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
