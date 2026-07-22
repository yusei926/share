#!/usr/bin/env python3
"""Merge accepted one-episode annotation artifacts into a Mimic source set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.automatic_annotation import (
    merge_automatic_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, action="append", default=[])
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="also include accepted episode-*/annotation.json files below this directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = list(args.annotation)
    if args.annotation_root is not None:
        paths.extend(sorted(args.annotation_root.glob("episode-*/annotation.json")))
    result = merge_automatic_annotations(paths, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
