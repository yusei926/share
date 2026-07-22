#!/usr/bin/env python3
"""Export successful sim AVP demos as phase-indexed Mimic staging data."""

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
from data.flip_table_data_augmentation.mimic.teleop_source_hdf5 import (
    export_teleop_mimic_source_hdf5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = sorted(
        path.parent
        for path in args.raw_root.expanduser().resolve().glob("*/manifest.json")
    )
    report = export_teleop_mimic_source_hdf5(
        episode_roots=roots,
        urdf_path=args.urdf,
        output_path=args.output,
        config=load_pipeline_config(args.config),
    )
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
