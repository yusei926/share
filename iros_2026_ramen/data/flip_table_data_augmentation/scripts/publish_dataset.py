#!/usr/bin/env python3
"""Publish a fully validated dataset through private staging and main verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.export.hf_transaction import publish_verified_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-synthetic-trajectories", type=int, default=2000)
    parser.add_argument("--minimum-appearance-variants", type=int, default=2)
    parser.add_argument("--expected-main-revision")
    args = parser.parse_args()
    report = publish_verified_dataset(
        dataset_root=args.dataset_root,
        verification_root=args.verification_root,
        report_path=args.report,
        config=load_pipeline_config(args.config),
        minimum_synthetic_trajectories=args.minimum_synthetic_trajectories,
        minimum_appearance_variants=args.minimum_appearance_variants,
        expected_main_revision=args.expected_main_revision,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
