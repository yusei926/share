#!/usr/bin/env python3
"""Check flip-table reset joint constants against LeRobot robot_q medians."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = os.environ.get("FLIP_TABLE_DATASET_ROOT")
DEFAULT_OVERLAY = (
    ROOT
    / "evaluate"
    / "flip_table_simulation"
    / "container_overlay"
    / "robofinals_tasks"
    / "local_auto_tasks"
    / "assemble_table_task.py"
)
DEFAULT_MAPPING = ROOT / "model" / "subtask_policy_training" / "gr00t" / "g1_full_body_mapping.py"


def _load_mapping(path: Path):
    spec = importlib.util.spec_from_file_location("g1_full_body_mapping_for_reset_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import mapping from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overlay_dict(text: str, name: str) -> dict[str, float]:
    match = re.search(re.escape(name) + r"\s*=\s*\{(.*?)\n\}", text, flags=re.S)
    if match is None:
        raise RuntimeError(f"Could not find {name} in overlay")
    result: dict[str, float] = {}
    for key, value in re.findall(r'"([^"]+)":\s*([-+0-9.]+)', match.group(1)):
        result[key] = float(value)
    return result


def _upper_body_names(mapping) -> list[str]:
    names = mapping.G1_FULL_BODY_JOINT_NAMES
    slices = mapping.G1_FULL_BODY_STATE_SLICES
    return (
        names[slice(*slices["waist"])]
        + names[slice(*slices["left_arm"])]
        + names[slice(*slices["right_arm"])]
    )


def compute_report(dataset_root: Path, overlay_path: Path, mapping_path: Path, frame_index: int) -> dict[str, object]:
    mapping = _load_mapping(mapping_path)
    source_indices = list(mapping.UPPER_BODY_SOURCE_INDEX_MAP[:17])
    names = _upper_body_names(mapping)

    table = ds.dataset(dataset_root / "data", format="parquet").to_table(
        columns=[
            "frame_index",
            "observation.state.robot_q_current",
            "observation.state.hand_state",
        ]
    )
    frames = np.asarray(table["frame_index"].to_pylist())
    mask = frames == frame_index
    if not mask.any():
        raise RuntimeError(f"No rows found for frame_index={frame_index}")

    robot_q = np.asarray(table["observation.state.robot_q_current"].to_pylist(), dtype=np.float64)
    hand = np.asarray(table["observation.state.hand_state"].to_pylist(), dtype=np.float64)
    medians = np.median(robot_q[mask][:, source_indices], axis=0)
    hand_medians = np.median(hand[mask], axis=0)

    overlay_text = overlay_path.read_text(encoding="utf-8")
    overlay_upper = _overlay_dict(overlay_text, "FLIP_TABLE_DATASET_INITIAL_UPPER_BODY_JOINT_POS")

    joints = []
    max_abs_delta = 0.0
    for source_index, name, median in zip(source_indices, names, medians):
        overlay_value = overlay_upper.get(name)
        delta = None if overlay_value is None else float(overlay_value - median)
        if delta is not None:
            max_abs_delta = max(max_abs_delta, abs(delta))
        joints.append(
            {
                "name": name,
                "source_index": int(source_index),
                "dataset_median": float(median),
                "overlay_value": overlay_value,
                "delta": delta,
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "overlay_path": str(overlay_path),
        "mapping_path": str(mapping_path),
        "frame_index": frame_index,
        "row_count": int(mask.sum()),
        "max_abs_delta": max_abs_delta,
        "hand_state_median": [float(value) for value in hand_medians],
        "joints": joints,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT else None,
        required=DEFAULT_DATASET_ROOT is None,
        help="LeRobot dataset root; defaults to FLIP_TABLE_DATASET_ROOT.",
    )
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compute_report(args.dataset_root, args.overlay, args.mapping, args.frame_index)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        f"frame_index={report['frame_index']} rows={report['row_count']} "
        f"max_abs_delta={report['max_abs_delta']:.6f} hand_state_median={report['hand_state_median']}"
    )


if __name__ == "__main__":
    main()
