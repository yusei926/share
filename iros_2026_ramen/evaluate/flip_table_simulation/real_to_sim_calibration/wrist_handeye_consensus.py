#!/usr/bin/env python3
"""Accept a wrist-camera proposal only when independent episodes agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_wrist_handeye_consensus/v1"
SIDES = ("left", "right")
MAXIMUM_TRANSLATION_DISAGREEMENT_M = 0.015
MAXIMUM_ROTATION_DISAGREEMENT_DEG = 4.0


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite 4x4")
    return matrix


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "team_ramen_flip_table_wrist_handeye_cad_alignment/v1":
        raise ValueError(f"{path} is not a CAD wrist-handeye alignment report")
    return value


def _uses_accepted_source_alignment(document: dict[str, Any]) -> bool:
    source_inputs = document.get("source_inputs")
    return isinstance(source_inputs, dict) and source_inputs.get("source_alignment_accepted") is True


def consensus(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two independent episode reports are required")
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera-calibration acceptance gate only",
        "reports": [str(path) for path, _ in reports],
        "thresholds": {
            "maximum_translation_disagreement_m": MAXIMUM_TRANSLATION_DISAGREEMENT_M,
            "maximum_rotation_disagreement_deg": MAXIMUM_ROTATION_DISAGREEMENT_DEG,
        },
        "sides": {},
    }
    for side in SIDES:
        entries = [document.get("sides", {}).get(side, {}) for _, document in reports]
        if not all(_uses_accepted_source_alignment(document) for _, document in reports):
            output["sides"][side] = {
                "status": "rejected_unaccepted_head_stereo_reference",
                "source_alignment_accepted": [
                    _uses_accepted_source_alignment(document) for _, document in reports
                ],
            }
            continue
        if any(entry.get("status") != "proposal_requires_heldout_validation" for entry in entries):
            output["sides"][side] = {
                "status": "rejected_insufficient_cross_episode_evidence",
                "episode_statuses": [entry.get("status") for entry in entries],
            }
            continue
        transforms = [
            _matrix(entry.get("fitted_wrist_from_rectified_opencv_camera"), f"{side} fitted mount")
            for entry in entries
        ]
        translation_errors = [
            float(np.linalg.norm(transform[:3, 3] - transforms[0][:3, 3]))
            for transform in transforms[1:]
        ]
        rotation_errors = [_rotation_distance_deg(transforms[0], transform) for transform in transforms[1:]]
        passes = (
            max(translation_errors, default=0.0) <= MAXIMUM_TRANSLATION_DISAGREEMENT_M
            and max(rotation_errors, default=0.0) <= MAXIMUM_ROTATION_DISAGREEMENT_DEG
        )
        output["sides"][side] = {
            "status": "accepted_for_heldout_validation" if passes else "rejected_cross_episode_disagreement",
            "translation_disagreement_m": max(translation_errors, default=0.0),
            "rotation_disagreement_deg": max(rotation_errors, default=0.0),
            "candidate": transforms[0].tolist() if passes else None,
        }
    output["accepted_for_heldout_validation"] = all(
        item["status"] == "accepted_for_heldout_validation" for item in output["sides"].values()
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [(path.expanduser().resolve(), _load(path.expanduser().resolve())) for path in args.report]
    result = consensus(reports)
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"accepted_for_heldout_validation": result["accepted_for_heldout_validation"]}))


if __name__ == "__main__":
    main()
