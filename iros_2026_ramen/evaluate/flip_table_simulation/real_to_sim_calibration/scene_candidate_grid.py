#!/usr/bin/env python3
"""Emit a bounded, source-derived static scene/camera calibration grid.

Each output is an episode-fixed reset candidate for offline RGB scoring only.
The grid is deliberately small and affine between the V1 authored reset and
the source-derived proposal: it is not an unconstrained optimizer and it never
provides simulator state to a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_scene_candidate_grid/v1"


def _candidate(report: dict[str, Any], label: str) -> dict[str, Any]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError(f"{label} must contain exactly one candidate")
    return dict(candidates[0])


def _scales(value: str, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated finite values") from exc
    if not result or not all(np.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{label} values must be finite values in [0,1]")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate values")
    return result


def _scaled(values: Any, scale: float, label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be three finite values")
    return (array * scale).tolist()


def _candidate_payload(
    scene: dict[str, Any],
    head: dict[str, Any] | None,
    *,
    scene_scale: float,
    head_scale: float,
) -> dict[str, Any]:
    result = {
        "label": f"scene_{scene_scale:.2f}_head_{head_scale:.2f}",
        "offset_local_m": _scaled(scene.get("offset_local_m"), scene_scale, "scene offset_local_m"),
        "yaw_rad": float(scene.get("yaw_rad", 0.0)) * scene_scale,
    }
    for key in ("robot_root_pos_local_m", "robot_root_yaw_rad"):
        if key not in scene:
            raise ValueError(f"scene candidate must preserve {key} for a fixed-base probe")
        result[key] = scene[key]
    if head is not None:
        result["head_stereo_offset_local_m"] = _scaled(
            head.get("head_stereo_offset_local_m"), head_scale, "head offset"
        )
        result["head_stereo_rotation_rpy_deg"] = _scaled(
            head.get("head_stereo_rotation_rpy_deg"), head_scale, "head rotation"
        )
    return result


def build_grid(
    scene_report: dict[str, Any],
    head_report: dict[str, Any] | None,
    *,
    scene_scales: tuple[float, ...],
    head_scales: tuple[float, ...],
) -> list[dict[str, Any]]:
    scene = _candidate(scene_report, "scene candidate report")
    head = _candidate(head_report, "head candidate report") if head_report is not None else None
    return [
        _candidate_payload(scene, head, scene_scale=scene_scale, head_scale=head_scale)
        for scene_scale in scene_scales
        for head_scale in head_scales
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-candidate", type=Path, required=True)
    parser.add_argument("--head-candidate", type=Path)
    parser.add_argument("--scene-scales", default="0,0.5,1")
    parser.add_argument("--head-scales", default="0,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scene_path = args.scene_candidate.expanduser().resolve()
    head_path = args.head_candidate.expanduser().resolve() if args.head_candidate else None
    output_dir = args.output_dir.expanduser().resolve()
    candidates = build_grid(
        json.loads(scene_path.read_text(encoding="utf-8")),
        json.loads(head_path.read_text(encoding="utf-8")) if head_path else None,
        scene_scales=_scales(args.scene_scales, "scene scales"),
        head_scales=_scales(args.head_scales, "head scales"),
    )
    paths: list[str] = []
    for index, candidate in enumerate(candidates):
        path = output_dir / "candidates" / f"candidate_{index:02d}.json"
        atomic_write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "policy_use": "forbidden: offline static scene/camera calibration only",
                "candidate": candidate,
                "candidates": [candidate],
            },
        )
        paths.append(str(path))
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "policy_use": "forbidden: offline static scene/camera calibration only",
            "scene_candidate": str(scene_path),
            "head_candidate": str(head_path) if head_path else None,
            "candidates": candidates,
            "candidate_files": paths,
        },
    )


if __name__ == "__main__":
    main()
