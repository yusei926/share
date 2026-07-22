#!/usr/bin/env python3
"""Validate local or re-downloaded flip-table augmented LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.export.validate_dataset import validate_dataset
from data.flip_table_data_augmentation.io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-video-decode", action="store_true")
    parser.add_argument("--require-full-source", action="store_true")
    parser.add_argument("--minimum-synthetic-trajectories", type=int, default=1)
    parser.add_argument("--minimum-appearance-variants", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset_root,
        load_pipeline_config(args.config),
        full_video_decode=args.full_video_decode,
        require_full_source=args.require_full_source,
        minimum_synthetic_trajectories=args.minimum_synthetic_trajectories,
        minimum_appearance_variants=args.minimum_appearance_variants,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        atomic_write_json(args.report, report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
