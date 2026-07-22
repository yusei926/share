#!/usr/bin/env python3
"""Audit the exact Raw-MCAP annotation and calibration behind every source slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.raw_source_contract import (
    audit_raw_source_bindings,
    snapshot_download_raw_contract,
)
from data.flip_table_data_augmentation.source_contract import snapshot_download_pinned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else snapshot_download_pinned(config)
    )
    raw_root = (
        args.raw_root.expanduser().resolve()
        if args.raw_root is not None
        else snapshot_download_raw_contract(config)
    )
    report = audit_raw_source_bindings(raw_root, source_root, config)
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "bindings"}, indent=2))
    print(f"bindings: {len(report['bindings'])}; full report: {args.output}")


if __name__ == "__main__":
    main()
