#!/usr/bin/env python3
"""Build one audited source annotation from an accepted FoundationPose track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.automatic_annotation import (
    build_automatic_source_annotation,
)
from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.source_contract import snapshot_download_pinned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else snapshot_download_pinned(config, include_videos=False)
    )
    result = build_automatic_source_annotation(
        source_root=source_root,
        track_dir=args.track_dir,
        output_dir=args.output_dir,
        config=config,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["accepted"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
