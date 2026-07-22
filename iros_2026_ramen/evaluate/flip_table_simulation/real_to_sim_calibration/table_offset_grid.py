#!/usr/bin/env python3
"""Emit a bounded workbench-local table-offset grid for offline RGB scoring.

The grid separates per-demonstration initial table placement from the shared
camera mount.  It is reset-only calibration evidence: all generated poses are
fixed for an episode and are forbidden from policy inputs, rewards, planners,
or runtime branches.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_table_offset_grid/v1"


def _axis(value: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated finite floats") from exc
    if not values or not all(np.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite values")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def _candidate(report: dict[str, Any]) -> dict[str, Any]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("base candidate report must contain exactly one candidate")
    base = dict(candidates[0])
    if "robot_root_pos_local_m" not in base or "robot_root_yaw_rad" not in base:
        raise ValueError("base candidate must contain a fixed robot root pose")
    return base


def build_grid(
    base_report: dict[str, Any], *, x_offsets_m: tuple[float, ...], y_offsets_m: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Return candidates with only workbench-local XY table offsets changed."""

    base = _candidate(base_report)
    base_offset = np.asarray(base.get("offset_local_m"), dtype=np.float64)
    if base_offset.shape != (3,) or not np.isfinite(base_offset).all():
        raise ValueError("base candidate offset_local_m must be three finite values")
    result = []
    for x_delta, y_delta in itertools.product(x_offsets_m, y_offsets_m):
        candidate = dict(base)
        candidate["label"] = f"table_dx_{x_delta:+.3f}_dy_{y_delta:+.3f}"
        candidate["offset_local_m"] = [
            float(base_offset[0] + x_delta),
            float(base_offset[1] + y_delta),
            float(base_offset[2]),
        ]
        result.append(candidate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--x-offsets-m", default="-0.12,0,0.12")
    parser.add_argument("--y-offsets-m", default="-0.12,0,0.12")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.base_candidate.expanduser().resolve()
    candidates = build_grid(
        json.loads(source.read_text(encoding="utf-8")),
        x_offsets_m=_axis(args.x_offsets_m, "x offsets"),
        y_offsets_m=_axis(args.y_offsets_m, "y offsets"),
    )
    output_dir = args.output_dir.expanduser().resolve()
    paths: list[str] = []
    for index, candidate in enumerate(candidates):
        path = output_dir / "candidates" / f"candidate_{index:02d}.json"
        atomic_write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "policy_use": "forbidden: offline reset-pose table calibration only",
                "candidate": candidate,
                "candidates": [candidate],
            },
        )
        paths.append(str(path))
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "policy_use": "forbidden: offline reset-pose table calibration only",
            "base_candidate": str(source),
            "candidates": candidates,
            "candidate_files": paths,
        },
    )


if __name__ == "__main__":
    main()
