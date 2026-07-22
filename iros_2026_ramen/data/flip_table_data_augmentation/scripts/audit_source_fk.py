#!/usr/bin/env python3
"""Audit source EEF labels against held-out robot joint forward kinematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import load_stratified_samples, run_fk_audit
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_contract import snapshot_download_pinned


DEFAULT_URDF = Path(
    "/workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/"
    "g1_29dof_with_hand.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else snapshot_download_pinned(config, include_videos=False)
    )
    contract = config.raw["source_contract"]
    samples = load_stratified_samples(
        source_root,
        samples_per_episode=args.samples_per_episode,
    )
    report = run_fk_audit(
        samples=samples,
        urdf_path=args.urdf,
        frame_names={key: str(value) for key, value in contract["fk_frames"].items()},
        eef_order=tuple(contract["eef_pose_order"]),
        tool_transforms=contract["fk_tool_transforms"],
        tool_transform_reference=contract["fk_tool_transform_reference"],
        validation_episode_modulus=int(contract["fk_calibration_episode_modulus"]),
        position_p95_max=float(contract["fk_action_validation_position_p95_m_max"]),
        rotation_p95_max=float(contract["fk_action_validation_rotation_p95_rad_max"]),
        swapped_score_ratio_min=float(contract["fk_swapped_assignment_score_ratio_min"]),
        source_repo_id=config.source.repo_id,
        source_revision=config.source.revision,
    )
    report["config_sha256"] = config.digest
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_json(args.output, report)
    print(encoded, end="")
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
