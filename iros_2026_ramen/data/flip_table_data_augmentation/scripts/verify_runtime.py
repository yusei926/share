#!/usr/bin/env python3
"""Verify pinned V1, Isaac Lab Mimic, and Replicator runtime artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.runtime_contract import verify_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--robofinals-root", type=Path, default=Path("/workspace/robofinals"))
    parser.add_argument("--observed-image-digest", default=os.environ.get("ROBOFINALS_IMAGE_DIGEST"))
    parser.add_argument("--allow-unverified-image-digest", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = verify_runtime(
        load_pipeline_config(args.config),
        robofinals_root=args.robofinals_root,
        observed_image_digest=args.observed_image_digest,
        require_image_digest=not args.allow_unverified_image_digest,
    )
    encoded = json.dumps(audit.to_json(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        atomic_write_json(args.output, audit.to_json())
    print(encoded, end="")


if __name__ == "__main__":
    main()
