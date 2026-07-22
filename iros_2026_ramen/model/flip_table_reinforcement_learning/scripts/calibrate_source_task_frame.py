#!/usr/bin/env python3
"""Build a residual-gated source table-frame calibration from head stereo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    from model.flip_table_reinforcement_learning.teacher.source_stereo_calibration import (
        calibrate_source_task_frame,
        load_stereo_calibration,
    )
except ModuleNotFoundError:  # Direct execution from the repository subdirectory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from model.flip_table_reinforcement_learning.teacher.source_stereo_calibration import (
        calibrate_source_task_frame,
        load_stereo_calibration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-stereo-calibration", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-root-rms-m", type=float, default=0.015)
    parser.add_argument("--max-table-rms-m", type=float, default=0.015)
    args = parser.parse_args()

    correspondences = json.loads(args.correspondences.read_text(encoding="utf-8"))
    if not isinstance(correspondences, dict):
        raise ValueError("correspondences JSON must contain an object")
    calibration_provenance = correspondences.get("head_stereo_calibration")
    if not isinstance(calibration_provenance, dict):
        raise ValueError("correspondences must contain pinned head_stereo_calibration provenance")
    expected_calibration_sha256 = calibration_provenance.get("sha256")
    observed_calibration_sha256 = hashlib.sha256(args.head_stereo_calibration.read_bytes()).hexdigest()
    if expected_calibration_sha256 != observed_calibration_sha256:
        raise ValueError("head-stereo calibration does not match the workspace-pinned SHA-256")
    if correspondences.get("raw_source_repo_id") is None or correspondences.get("raw_source_revision") is None:
        raise ValueError("correspondences must pin the raw source repository and revision")
    result = calibrate_source_task_frame(
        correspondences,
        load_stereo_calibration(args.head_stereo_calibration),
        max_root_rms_m=args.max_root_rms_m,
        max_table_rms_m=args.max_table_rms_m,
    )
    result["inputs"] = {
        "head_stereo_calibration": str(args.head_stereo_calibration),
        "head_stereo_calibration_sha256": observed_calibration_sha256,
        "correspondences": str(args.correspondences),
        "correspondences_sha256": hashlib.sha256(args.correspondences.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "acceptance_eligible": result["acceptance_eligible"],
        "source_task_pose_root": result["source_task_pose_root"],
        "root_rms_m": result["measurements"]["root_camera"]["metrics"]["rms_m"],
        "table_rms_m": result["measurements"]["table"]["metrics"]["rms_m"],
    }))


if __name__ == "__main__":
    main()
