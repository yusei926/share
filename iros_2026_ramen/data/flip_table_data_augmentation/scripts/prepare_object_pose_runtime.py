#!/usr/bin/env python3
"""Fetch and verify the pinned Grounded-SAM 2.1 and FoundationPose inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.object_pose.artifacts import prepare_runtime, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runtime-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    root = args.runtime_root.expanduser().resolve()
    manifest = prepare_runtime(root, config)
    path = root / "runtime-manifest.json"
    write_manifest(path, manifest)
    print(json.dumps({"runtime_root": str(root), "manifest": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
