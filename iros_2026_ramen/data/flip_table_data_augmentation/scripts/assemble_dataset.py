#!/usr/bin/env python3
"""Assemble accepted renders and pinned real episodes as LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.export.build_dataset import assemble_dataset


def _indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source episodes must be comma-separated integers") from exc
    if not result or tuple(sorted(set(result))) != result or result[0] < 0:
        raise argparse.ArgumentTypeError("source episodes must be non-negative, sorted, and unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-episodes", type=_indices)
    parser.add_argument("--min-appearance-variants", type=int)
    args = parser.parse_args()
    result = assemble_dataset(
        source_root=args.source_root,
        render_manifests=args.render_manifest,
        output_root=args.output_root,
        work_root=args.work_root,
        config=load_pipeline_config(args.config),
        source_episode_indices=args.source_episodes,
        min_appearance_variants=args.min_appearance_variants,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
