#!/usr/bin/env python3
"""Audit source encoder/EEF-state timing without changing any recorded data.

The source dataset stores ``robot_q_current`` and ``ee_state`` on each row.
They should describe the same physical instant, but a producer-side encoder or
EEF publication delay can make a camera/FK calibration look like a camera
mount error.  This tool searches a bounded *offline diagnostic* row offset.
It never rewrites labels, shifts videos, changes simulator timing, or exposes
the result to a policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import (
    G1_BODY_JOINT_ORDER,
    _build_fk,
)
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex

from evaluate.flip_table_simulation.real_to_sim_calibration.contracts import (
    SOURCE_EEF_DIM,
    SOURCE_FPS,
    SOURCE_Q_DIM,
    UPPER_BODY_SLICE,
)


SCHEMA_VERSION = "team_ramen_flip_table_state_timing_audit/v1"
SIDES = ("left", "right")
MINIMUM_OVERLAP_FRAMES = 30


@dataclass(frozen=True)
class CandidateMetrics:
    offset_frames: int
    overlap_frames: int
    position_median_m: float
    position_p95_m: float
    rotation_median_deg: float
    rotation_p95_deg: float
    normalized_median_score: float

    def json(self) -> dict[str, float | int]:
        return {
            "q_current_index_minus_ee_state_index_frames": self.offset_frames,
            "offset_s": self.offset_frames / SOURCE_FPS,
            "overlap_frames": self.overlap_frames,
            "position_median_m": self.position_median_m,
            "position_p95_m": self.position_p95_m,
            "rotation_median_deg": self.rotation_median_deg,
            "rotation_p95_deg": self.rotation_p95_deg,
            "normalized_median_score": self.normalized_median_score,
        }


def _tool_transforms(config_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw = load_pipeline_config(config_path).raw["source_contract"]["fk_tool_transforms"]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in SIDES:
        value = raw[side]
        translation = np.asarray(value["translation_m"], dtype=np.float64)
        quaternion = np.asarray(value["quaternion_xyzw"], dtype=np.float64)
        if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(
            np.concatenate((translation, quaternion))
        ).all():
            raise ValueError(f"invalid configured {side} EEF tool transform")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-6):
            raise ValueError(f"configured {side} EEF tool transform is not normalized")
        result[side] = (translation, Rotation.from_quat(quaternion).as_matrix())
    return result


def _eef_targets(values: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if values.ndim != 2 or values.shape[1] != SOURCE_EEF_DIM or not np.isfinite(values).all():
        raise ValueError(f"EEF values must be finite [T,{SOURCE_EEF_DIM}]")
    return {
        side: (
            values[:, start : start + 3],
            Rotation.from_euler("xyz", values[:, start + 3 : start + 6]).as_matrix(),
        )
        for side, start in zip(SIDES, (0, 6), strict=True)
    }


def _fk_wrist_placements(
    robot_q_current: np.ndarray,
    *,
    urdf: Path,
    frame_names: dict[str, str],
    tools: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    q_current = np.asarray(robot_q_current, dtype=np.float64)
    if q_current.ndim != 2 or q_current.shape[1] != SOURCE_Q_DIM or not np.isfinite(q_current).all():
        raise ValueError(f"robot_q_current must be finite [T,{SOURCE_Q_DIM}]")
    pin, model, data, joint_indices, frame_ids = _build_fk(urdf, frame_names)
    missing = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    if missing:
        raise ValueError(f"FK URDF lacks source joints: {missing}")
    positions = {side: [] for side in SIDES}
    rotations = {side: [] for side in SIDES}
    q = np.zeros(model.nq, dtype=np.float64)
    for source_q in q_current:
        q.fill(0.0)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        for side in SIDES:
            frame = data.oMf[frame_ids[side]]
            tool_position, tool_rotation = tools[side]
            positions[side].append(np.asarray(frame.translation) + np.asarray(frame.rotation) @ tool_position)
            rotations[side].append(np.asarray(frame.rotation) @ tool_rotation)
    return {side: (np.stack(positions[side]), np.stack(rotations[side])) for side in SIDES}


def _indices(length: int, offset_frames: int) -> tuple[np.ndarray, np.ndarray]:
    state_indices = np.arange(length, dtype=np.int64)
    current_indices = state_indices + int(offset_frames)
    keep = (current_indices >= 0) & (current_indices < length)
    return state_indices[keep], current_indices[keep]


def candidate_metrics(
    placements: tuple[np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray],
    *,
    offset_frames: int,
) -> CandidateMetrics:
    """Score one row offset; a positive offset means a later q-current row."""

    positions, rotations = placements
    target_positions, target_rotations = targets
    if positions.shape != target_positions.shape or rotations.shape != target_rotations.shape:
        raise ValueError("placement and target shapes must agree")
    state_indices, current_indices = _indices(len(positions), offset_frames)
    if len(state_indices) < MINIMUM_OVERLAP_FRAMES:
        raise ValueError("candidate has too few overlapping frames")
    position_error = np.linalg.norm(
        positions[current_indices] - target_positions[state_indices], axis=1
    )
    rotation_error_deg = np.degrees(
        Rotation.from_matrix(
            rotations[current_indices].transpose(0, 2, 1) @ target_rotations[state_indices]
        ).magnitude()
    )
    position_median = float(np.median(position_error))
    rotation_median = float(np.median(rotation_error_deg))
    return CandidateMetrics(
        offset_frames=int(offset_frames),
        overlap_frames=int(len(state_indices)),
        position_median_m=position_median,
        position_p95_m=float(np.quantile(position_error, 0.95)),
        rotation_median_deg=rotation_median,
        rotation_p95_deg=float(np.quantile(rotation_error_deg, 0.95)),
        # A 10 mm or 3 degree median mismatch each contributes one unit. This
        # is a ranking scale, not an acceptance threshold.
        normalized_median_score=position_median / 0.010 + rotation_median / 3.0,
    )


def _best(candidates: list[CandidateMetrics]) -> CandidateMetrics:
    return min(
        candidates,
        key=lambda value: (
            value.normalized_median_score,
            value.position_p95_m,
            value.rotation_p95_deg,
            abs(value.offset_frames),
            value.offset_frames,
        ),
    )


def timing_report(
    *,
    placements: dict[str, tuple[np.ndarray, np.ndarray]],
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    max_offset_frames: int,
) -> dict[str, Any]:
    if max_offset_frames < 0:
        raise ValueError("max_offset_frames must be non-negative")
    values: dict[str, list[CandidateMetrics]] = {}
    for side in SIDES:
        values[side] = []
        for offset in range(-max_offset_frames, max_offset_frames + 1):
            try:
                values[side].append(candidate_metrics(placements[side], targets[side], offset_frames=offset))
            except ValueError as error:
                if "too few overlapping" not in str(error):
                    raise
    shared: list[dict[str, Any]] = []
    for offset in range(-max_offset_frames, max_offset_frames + 1):
        by_side = {side: next((item for item in values[side] if item.offset_frames == offset), None) for side in SIDES}
        if any(value is None for value in by_side.values()):
            continue
        shared.append(
            {
                "q_current_index_minus_ee_state_index_frames": offset,
                "offset_s": offset / SOURCE_FPS,
                "normalized_median_score": float(sum(value.normalized_median_score for value in by_side.values())),
                "sides": {side: by_side[side].json() for side in SIDES},
            }
        )
    if not shared:
        raise ValueError("no offset candidate has enough overlap for both sides")
    best_shared = min(
        shared,
        key=lambda value: (
            value["normalized_median_score"],
            abs(value["q_current_index_minus_ee_state_index_frames"]),
            value["q_current_index_minus_ee_state_index_frames"],
        ),
    )
    return {
        "search": {
            "max_offset_frames": max_offset_frames,
            "max_offset_s": max_offset_frames / SOURCE_FPS,
            "definition": "positive means robot_q_current[t + offset] is compared with ee_state[t]",
            "minimum_overlap_frames": MINIMUM_OVERLAP_FRAMES,
        },
        "per_side": {
            side: {
                "best": _best(values[side]).json(),
                "zero_offset": next(item for item in values[side] if item.offset_frames == 0).json(),
                "candidates": [item.json() for item in values[side]],
            }
            for side in SIDES
        },
        "shared_offset": {
            "best": best_shared,
            "zero_offset": next(
                value for value in shared if value["q_current_index_minus_ee_state_index_frames"] == 0
            ),
            "candidates": shared,
        },
    }


def joint_command_timing_report(
    robot_q_current: np.ndarray,
    robot_q_desired: np.ndarray,
    *,
    max_offset_frames: int,
) -> dict[str, Any]:
    """Measure raw 17-DoF encoder delay against the desired-joint command.

    Positive means a later encoder row is compared to the command at the
    reference row.  It is a diagnostic convention only; no controller delay,
    source label, or replay timestamp is changed from this result.
    """

    current = np.asarray(robot_q_current, dtype=np.float64)
    desired = np.asarray(robot_q_desired, dtype=np.float64)
    if current.shape != desired.shape or current.ndim != 2 or current.shape[1] != SOURCE_Q_DIM:
        raise ValueError(f"current/desired joints must share finite [T,{SOURCE_Q_DIM}] shape")
    if not np.isfinite(current).all() or not np.isfinite(desired).all():
        raise ValueError("current/desired joints must be finite")
    candidates: list[dict[str, float | int]] = []
    for offset in range(-max_offset_frames, max_offset_frames + 1):
        desired_indices, current_indices = _indices(len(current), offset)
        if len(desired_indices) < MINIMUM_OVERLAP_FRAMES:
            continue
        error = current[current_indices, UPPER_BODY_SLICE] - desired[desired_indices, UPPER_BODY_SLICE]
        candidates.append(
            {
                "q_current_index_minus_robot_q_desired_index_frames": offset,
                "offset_s": offset / SOURCE_FPS,
                "overlap_frames": int(len(desired_indices)),
                "upper_body_rmse_rad": float(np.sqrt(np.mean(error * error))),
                "upper_body_median_abs_error_rad": float(np.median(np.abs(error))),
                "upper_body_p95_abs_error_rad": float(np.quantile(np.abs(error), 0.95)),
            }
        )
    if not candidates:
        raise ValueError("no joint command candidate has enough overlap")
    best = min(
        candidates,
        key=lambda value: (
            value["upper_body_rmse_rad"],
            value["upper_body_p95_abs_error_rad"],
            abs(value["q_current_index_minus_robot_q_desired_index_frames"]),
            value["q_current_index_minus_robot_q_desired_index_frames"],
        ),
    )
    return {
        "definition": "positive means robot_q_current[t + offset] is compared with robot_q_desired[t]",
        "max_offset_frames": max_offset_frames,
        "max_offset_s": max_offset_frames / SOURCE_FPS,
        "minimum_overlap_frames": MINIMUM_OVERLAP_FRAMES,
        "best": best,
        "zero_offset": next(
            value for value in candidates
            if value["q_current_index_minus_robot_q_desired_index_frames"] == 0
        ),
        "candidates": candidates,
    }


def audit_episode(
    *, source_root: Path, episode_index: int, urdf: Path, frame_names: dict[str, str],
    max_offset_frames: int, config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = SourceDatasetIndex(source_root)
    episode = source.episode(episode_index)
    rows = pq.read_table(
        episode.data_path,
        columns=[
            "frame_index",
            "timestamp",
            "observation.state.robot_q_current",
            "action.robot_q_desired",
            "observation.state.ee_state",
        ],
        filters=[("episode_index", "=", int(episode_index))],
    ).to_pylist()
    rows.sort(key=lambda row: int(row["frame_index"]))
    expected_frames = list(range(episode.frame_count))
    if [int(row["frame_index"]) for row in rows] != expected_frames:
        raise ValueError("source rows are not a contiguous complete episode")
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source timestamps are not strictly increasing")
    q_current = np.asarray([row["observation.state.robot_q_current"] for row in rows], dtype=np.float64)
    q_desired = np.asarray([row["action.robot_q_desired"] for row in rows], dtype=np.float64)
    eef_state = np.asarray([row["observation.state.ee_state"] for row in rows], dtype=np.float64)
    tools = _tool_transforms(config_path)
    placements = _fk_wrist_placements(q_current, urdf=urdf, frame_names=frame_names, tools=tools)
    report = timing_report(
        placements=placements,
        targets=_eef_targets(eef_state),
        max_offset_frames=max_offset_frames,
    )
    report["joint_command_vs_encoder"] = joint_command_timing_report(
        q_current,
        q_desired,
        max_offset_frames=max_offset_frames,
    )
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "policy_use": "forbidden: offline timing diagnosis only; no labels, cameras, simulator, or policy inputs are changed",
            "source_episode_index": int(episode_index),
            "source_frames": int(episode.frame_count),
            "source_timestamp_range_s": [float(timestamps[0]), float(timestamps[-1])],
            "source_fps": SOURCE_FPS,
            "urdf": str(urdf.resolve()),
            "frame_names": frame_names,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--left-frame", default="left_wrist_yaw_link")
    parser.add_argument("--right-frame", default="right_wrist_yaw_link")
    parser.add_argument("--max-offset-frames", type=int, default=15)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_index < 0 or args.max_offset_frames < 0 or args.max_offset_frames > 150:
        raise ValueError("episode index and offset bound are invalid")
    report = audit_episode(
        source_root=args.source_root.expanduser().resolve(),
        episode_index=args.episode_index,
        urdf=args.urdf.expanduser().resolve(),
        frame_names={"left": args.left_frame, "right": args.right_frame},
        max_offset_frames=args.max_offset_frames,
        config_path=args.config.expanduser().resolve(),
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"episode": args.episode_index, "shared_offset": report["shared_offset"]["best"]}, indent=2))


if __name__ == "__main__":
    main()
