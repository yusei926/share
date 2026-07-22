#!/usr/bin/env python3
"""Export accepted LeRobot annotations to an Isaac Lab Mimic source HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.mimic.source_hdf5 import export_mimic_source_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--fk-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_mimic_source_hdf5(
        source_root=args.source_root,
        annotations_path=args.annotations,
        fk_audit_path=args.fk_audit,
        output_path=args.output,
        config=load_pipeline_config(args.config),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
