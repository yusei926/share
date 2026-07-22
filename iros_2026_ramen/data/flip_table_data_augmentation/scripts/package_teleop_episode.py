#!/usr/bin/env python3
"""Package one direct sim AVP demo for the augmented LeRobot v3 build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import (
    DEFAULT_CONFIG_PATH,
    load_pipeline_config,
)
from data.flip_table_data_augmentation.teleop.package_episode import (
    package_direct_sim_teleop_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = package_direct_sim_teleop_episode(
        args.episode,
        output_root=args.output_root,
        urdf_path=args.urdf,
        config=load_pipeline_config(args.config),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
