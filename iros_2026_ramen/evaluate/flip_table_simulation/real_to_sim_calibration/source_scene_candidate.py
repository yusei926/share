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


def _first_trace_diagnostics(trace_path: Path) -> dict[str, Any]:
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            diagnostics = row.get("simulator_scene_diagnostics")
            if isinstance(diagnostics, dict):
                return diagnostics
    raise ValueError("trace contains no simulator_scene_diagnostics")


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


def candidate_from_artifacts(source_alignment: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return a single source-derived workbench-local reset candidate."""

    source_pose = _matrix(source_alignment.get("fixed_scene_root_from_table"), "source fixed_scene_root_from_table")
    root = _transform(
        diagnostics.get("root_pose_world_xyzw", [])[:3],
        diagnostics.get("root_pose_world_xyzw", [])[3:],
        "simulator root",
    )
    table = diagnostics.get("white_table")
    workbench = diagnostics.get("workbench")
    if not isinstance(table, dict) or not isinstance(workbench, dict):
        raise ValueError("trace must contain white_table and workbench diagnostic poses")
    world_from_table = _transform(table.get("position_world_m"), table.get("quaternion_xyzw"), "simulator table")
    world_from_workbench = _transform(workbench.get("position_world_m"), workbench.get("quaternion_xyzw"), "simulator workbench")
    desired_world_from_table = root @ source_pose
    offset_world = desired_world_from_table[:3, 3] - world_from_table[:3, 3]
    offset_workbench = world_from_workbench[:3, :3].T @ offset_world
    yaw = _yaw_delta(world_from_table[:3, :3], desired_world_from_table[:3, :3])
    root_rotation = root[:3, :3]
    root_tilt = float(np.linalg.norm(root_rotation[:2, 2])) + abs(float(root_rotation[2, 0])) + abs(
        float(root_rotation[2, 1])
    )
    if root_tilt > 0.03:
        raise ValueError("fixed-base calibration trace has a non-planar robot root")
    # Fixed-scene replay is single-environment, so the recorded world root is
    # its local root.  Keeping it in the candidate prevents the regular task
    # placement policy from silently moving the robot when the table reset is
    # changed; otherwise the source initial joint state and camera comparison
    # no longer describe the same physical arrangement.
    root_yaw = math.atan2(float(root_rotation[1, 0]), float(root_rotation[0, 0]))
    temporal = source_alignment.get("temporal_consistency", {})
    return {
        "label": f"source_episode_{source_alignment.get('source', {}).get('episode_index', 'unknown')}",
        "offset_local_m": [float(value) for value in offset_workbench],
        "yaw_rad": yaw,
        "robot_root_pos_local_m": [float(value) for value in root[:3, 3]],
        "robot_root_yaw_rad": root_yaw,
        "source_translation_spread_p95_m": float(temporal.get("translation_spread_p95_m", float("nan"))),
        "source_rotation_spread_p95_deg": float(temporal.get("rotation_spread_p95_deg", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_alignment.expanduser().resolve().read_text(encoding="utf-8"))
    if not source.get("accepted_for_fixed_scene_proposal", False):
        raise ValueError("source CAD alignment did not pass its internal consistency gate")
    diagnostics = _first_trace_diagnostics(args.sim_trace.expanduser().resolve())
    candidate = candidate_from_artifacts(source, diagnostics)
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline reset calibration only",
        "source_alignment": str(args.source_alignment.expanduser().resolve()),
        "sim_trace": str(args.sim_trace.expanduser().resolve()),
        "candidate": candidate,
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
        ],
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
