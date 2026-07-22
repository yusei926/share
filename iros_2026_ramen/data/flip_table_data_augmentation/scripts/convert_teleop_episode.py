#!/usr/bin/env python3
"""Convert one successful raw AVP episode to the immutable numeric schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.teleop.numeric import convert_raw_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = convert_raw_episode(
        args.episode,
        urdf_path=args.urdf,
        output_path=args.output,
    )
    print(report["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
