#!/usr/bin/env python3
"""Combine independently derived head-stereo mount proposals conservatively.

The resulting correction is an offline fixed-scene probe candidate.  It is
not a policy input and does not constitute camera acceptance: the shared value
must still pass the unused-episode RGB gate before becoming a simulator
default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_source_head_mount_consensus/v1"
MAXIMUM_TRANSLATION_DISAGREEMENT_M = 0.005
MAXIMUM_ROTATION_DISAGREEMENT_DEG = 1.5


def _vector(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return result


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "team_ramen_flip_table_source_head_mount_candidate/v1":
        raise ValueError(f"{path} is not a source head-mount candidate")
    source = document.get("source_alignment")
    # Candidate reports created before the inter-tool field was restored
    # called this same candidate-relative value ``incremental_correction``.
    # Accept that schema-compatible spelling, normalize it locally, and do
    # not reinterpret it as an absolute camera configuration.
    correction = document.get("correction", document.get("incremental_correction"))
    if not isinstance(source, str) or not isinstance(correction, dict):
        raise ValueError(f"{path} has incomplete source head-mount provenance")
    normalized = dict(document)
    normalized["correction"] = correction
    return normalized


def _rotation_from_rpy_deg(value: object, label: str) -> Rotation:
    return Rotation.from_euler("XYZ", _vector(value, label), degrees=True)


def consensus(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two independent head-mount reports are required")
    source_paths = [document["source_alignment"] for _, document in reports]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("head-mount reports must use distinct source alignments")

    translations = np.stack(
        [
            _vector(document["correction"].get("head_stereo_offset_local_m"), "head offset")
            for _, document in reports
        ]
    )
    rotations = [
        _rotation_from_rpy_deg(
            document["correction"].get("head_stereo_rotation_rpy_deg"), "head rotation"
        )
        for _, document in reports
    ]
    center_translation = np.mean(translations, axis=0)
    center_rotation = Rotation.from_matrix(
        np.stack([rotation.as_matrix() for rotation in rotations])
    ).mean()
    translation_residuals = np.linalg.norm(translations - center_translation, axis=1)
    rotation_residuals = np.asarray(
        [
            np.degrees((center_rotation.inv() * rotation).magnitude())
            for rotation in rotations
        ],
        dtype=np.float64,
    )
    maximum_translation = float(np.max(translation_residuals))
    maximum_rotation = float(np.max(rotation_residuals))
    accepted = bool(
        maximum_translation <= MAXIMUM_TRANSLATION_DISAGREEMENT_M
        and maximum_rotation <= MAXIMUM_ROTATION_DISAGREEMENT_DEG
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline fixed-scene probe candidate only",
        "reports": [str(path) for path, _ in reports],
        "source_alignments": source_paths,
        "thresholds": {
            "maximum_translation_disagreement_m": MAXIMUM_TRANSLATION_DISAGREEMENT_M,
            "maximum_rotation_disagreement_deg": MAXIMUM_ROTATION_DISAGREEMENT_DEG,
        },
        "shared_head_stereo_offset_local_m": center_translation.tolist() if accepted else None,
        "shared_head_stereo_rotation_rpy_deg": (
            center_rotation.as_euler("XYZ", degrees=True).tolist() if accepted else None
        ),
        "residuals": {
            "translation_m_by_report": translation_residuals.tolist(),
            "rotation_deg_by_report": rotation_residuals.tolist(),
            "maximum_translation_m": maximum_translation,
            "maximum_rotation_deg": maximum_rotation,
        },
        "status": "accepted_for_fixed_scene_probe" if accepted else "rejected_cross_episode_disagreement",
        "accepted_for_fixed_scene_probe": accepted,
        "accepted_for_shared_simulator_default": False,
        "remaining_requirement": "unused-episode camera and silhouette acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [(path.expanduser().resolve(), _load(path.expanduser().resolve())) for path in args.report]
    result = consensus(reports)
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"status": result["status"], "residuals": result["residuals"]}, indent=2))


if __name__ == "__main__":
    main()
