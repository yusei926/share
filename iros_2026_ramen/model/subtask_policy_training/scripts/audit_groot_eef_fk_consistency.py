"""Audit GR00T EEF teachers against the paired desired G1 joint targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import (
    _build_fk,
    load_stratified_samples,
    run_fk_audit,
)
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from model.subtask_policy_training.gr00t.n17_contract import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    validate_eef_fk_release_audit,
)


DEFAULT_PIPELINE_CONFIG = (
    REPO_ROOT
    / "data"
    / "flip_table_data_augmentation"
    / "configs"
    / "pipeline_v1.json"
)
DEFAULT_URDF = (
    REPO_ROOT
    / "inference"
    / "orin"
    / "ros2_ws"
    / "src"
    / "g1_description"
    / "urdf"
    / "unitree_g1"
    / "g1_29dof_mode_15_with_dex1_1.urdf"
)
EXPECTED_EPISODES = 174
TIMING_OFFSETS = tuple(range(-3, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--samples-per-episode", type=int, default=16)
    return parser.parse_args()


def temporal_alignment_diagnostics(
    source_root: Path,
    *,
    urdf_path: Path,
    frame_names: dict[str, str],
    calibration_episode_modulus: int,
) -> dict:
    """Compare same-row EEF/joint labels against nearby joint-target frames."""
    import numpy as np
    import pyarrow.dataset as pads
    from scipy.spatial.transform import Rotation

    columns = [
        "episode_index",
        "frame_index",
        "action.ee_action",
        "action.robot_q_desired",
    ]
    table = pads.dataset(source_root / "data", format="parquet").to_table(
        columns=columns
    )
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    targets = np.asarray(table["action.ee_action"].to_pylist(), dtype=np.float64)
    desired = np.asarray(
        table["action.robot_q_desired"].to_pylist(), dtype=np.float64
    )
    pin, model, data, joint_indices, frame_ids = _build_fk(
        urdf_path, frame_names
    )
    positions = {
        side: np.empty((len(desired), 3), dtype=np.float64) for side in frame_ids
    }
    rotations = {
        side: np.empty((len(desired), 3, 3), dtype=np.float64)
        for side in frame_ids
    }
    q = np.zeros(model.nq, dtype=np.float64)
    for index, source_q in enumerate(desired):
        q.fill(0.0)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        for side, frame_id in frame_ids.items():
            positions[side][index] = data.oMf[frame_id].translation
            rotations[side][index] = data.oMf[frame_id].rotation

    candidates = []
    for offset in TIMING_OFFSETS:
        source_indices = np.arange(
            max(0, -offset),
            min(len(episodes), len(episodes) - offset),
            dtype=np.int64,
        )
        joint_indices_for_target = source_indices + offset
        same_episode = episodes[source_indices] == episodes[joint_indices_for_target]
        adjacent = (
            frames[joint_indices_for_target] == frames[source_indices] + offset
        )
        source_indices = source_indices[same_episode & adjacent]
        joint_indices_for_target = joint_indices_for_target[same_episode & adjacent]
        calibration = (
            episodes[source_indices] % calibration_episode_modulus != 0
        )
        validation = ~calibration
        sides = {}
        score = 0.0
        for side, target_start in (("left", 0), ("right", 6)):
            target_position = targets[source_indices, target_start : target_start + 3]
            target_rotation = Rotation.from_euler(
                "xyz",
                targets[source_indices, target_start + 3 : target_start + 6],
            ).as_matrix()
            frame_position = positions[side][joint_indices_for_target]
            frame_rotation = rotations[side][joint_indices_for_target]
            local_position = np.einsum(
                "nij,nj->ni",
                frame_rotation[calibration].transpose(0, 2, 1),
                target_position[calibration] - frame_position[calibration],
            )
            local_rotation = (
                frame_rotation[calibration].transpose(0, 2, 1)
                @ target_rotation[calibration]
            )
            tool_position = np.median(local_position, axis=0)
            tool_rotation = Rotation.from_matrix(local_rotation).mean().as_matrix()
            predicted_position = frame_position[validation] + np.einsum(
                "nij,j->ni", frame_rotation[validation], tool_position
            )
            predicted_rotation = frame_rotation[validation] @ tool_rotation
            position_error = np.linalg.norm(
                predicted_position - target_position[validation], axis=1
            )
            rotation_error = Rotation.from_matrix(
                predicted_rotation.transpose(0, 2, 1)
                @ target_rotation[validation]
            ).magnitude()
            position_median = float(np.median(position_error))
            rotation_median = float(np.median(rotation_error))
            score += position_median + rotation_median
            sides[side] = {
                "position_error_median": position_median,
                "position_error_p95": float(np.quantile(position_error, 0.95)),
                "rotation_error_rad_median": rotation_median,
                "rotation_error_rad_p95": float(np.quantile(rotation_error, 0.95)),
                "fitted_tool_translation_m": tool_position.tolist(),
                "fitted_tool_quaternion_xyzw": Rotation.from_matrix(
                    tool_rotation
                ).as_quat().tolist(),
            }
        candidates.append(
            {
                "offset_frames": offset,
                "pair_count": int(len(source_indices)),
                "score": score,
                "sides": sides,
            }
        )
    best = min(candidates, key=lambda item: item["score"])
    zero = next(item for item in candidates if item["offset_frames"] == 0)
    improvement = max(0.0, (zero["score"] - best["score"]) / zero["score"])
    # Small neighboring-frame improvements are expected from the physical IK
    # and servo pipeline. Only a material (>5%) improvement justifies changing
    # the immutable row alignment.
    selected_offset = int(best["offset_frames"]) if improvement > 0.05 else 0
    return {
        "frame_offset_definition": (
            "compare action.ee_action[t] with FK(action.robot_q_desired[t+offset])"
        ),
        "candidates": candidates,
        "raw_best_offset_frames": int(best["offset_frames"]),
        "raw_best_relative_improvement": improvement,
        "material_improvement_threshold": 0.05,
        "selected_offset_frames": selected_offset,
        "pass": selected_offset == 0,
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_episode < 2:
        raise ValueError("--samples-per-episode must be at least two")
    config = load_pipeline_config(args.pipeline_config)
    contract = config.raw["source_contract"]
    samples = load_stratified_samples(
        args.source_root.expanduser().resolve(),
        samples_per_episode=args.samples_per_episode,
    )
    episodes = sorted({sample.episode_index for sample in samples})
    if len(episodes) != EXPECTED_EPISODES or episodes != list(range(EXPECTED_EPISODES)):
        raise ValueError(
            f"EEF-FK audit requires episodes 0..{EXPECTED_EPISODES - 1}, got "
            f"{len(episodes)} episodes"
        )
    report = run_fk_audit(
        samples=samples,
        urdf_path=args.urdf,
        frame_names={key: str(value) for key, value in contract["fk_frames"].items()},
        eef_order=tuple(contract["eef_pose_order"]),
        tool_transforms=contract["fk_tool_transforms"],
        tool_transform_reference=contract["fk_tool_transform_reference"],
        validation_episode_modulus=int(contract["fk_calibration_episode_modulus"]),
        position_p95_max=float(
            contract["fk_action_validation_position_p95_m_max"]
        ),
        rotation_p95_max=float(
            contract["fk_action_validation_rotation_p95_rad_max"]
        ),
        swapped_score_ratio_min=float(
            contract["fk_swapped_assignment_score_ratio_min"]
        ),
        source_repo_id=DATASET_REPO_ID,
        source_revision=DATASET_REVISION,
    )
    timing = temporal_alignment_diagnostics(
        args.source_root.expanduser().resolve(),
        urdf_path=args.urdf,
        frame_names={key: str(value) for key, value in contract["fk_frames"].items()},
        calibration_episode_modulus=int(
            contract["fk_calibration_episode_modulus"]
        ),
    )
    failed_episodes = [
        int(item["episode_index"])
        for item in report["per_episode"]
        if not bool(item["action_fk_residual_pass"])
    ]
    report.update(
        {
            "training_contract": {
                "eef_teacher": "action.ee_action",
                "joint_teacher": "action.robot_q_desired",
                "policy_action_mask_slots": "0:46",
                "contradictory_teacher_policy": (
                    "fail training; never optimize inconsistent EEF and arm targets together"
                ),
                "teacher_pair_status": (
                    "compatible_with_expected_ik_realization_residual"
                    if report["pass"] and timing["pass"]
                    else "incompatible"
                ),
            },
            "coverage": {
                "episode_count": len(episodes),
                "episode_indices": episodes,
                "samples_per_episode": args.samples_per_episode,
                "episode_level_diagnostic_threshold_exceedances": failed_episodes,
                "release_gate_scope": (
                    "pooled held-out episodes; per-episode values expose IK outliers "
                    "but do not redefine the source action contract"
                ),
            },
            "temporal_alignment": timing,
        }
    )
    validate_eef_fk_release_audit(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["pass"] or not timing["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
