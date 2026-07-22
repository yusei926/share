#!/usr/bin/env python3
"""Build a residual-gated source table frame from D405 IR stereo observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from model.flip_table_reinforcement_learning.teacher.wrist_hand_eye_calibration import (
        calibrate_source_table_from_wrist_ir,
        load_d405_ir_stereo_calibration,
    )
except ModuleNotFoundError:  # Direct execution from the repository subdirectory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from model.flip_table_reinforcement_learning.teacher.wrist_hand_eye_calibration import (
        calibrate_source_table_from_wrist_ir,
        load_d405_ir_stereo_calibration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d405-calibration", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hand-eye-translation-rms-m", type=float, default=0.015)
    parser.add_argument("--max-hand-eye-rotation-rms-rad", type=float, default=0.10)
    parser.add_argument("--max-table-fit-rms-m", type=float, default=0.010)
    parser.add_argument("--max-stereo-epipolar-error-px", type=float, default=1.5)
    args = parser.parse_args()

    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observation JSON must contain an object")
    result = calibrate_source_table_from_wrist_ir(
        payload,
        load_d405_ir_stereo_calibration(args.d405_calibration),
        max_hand_eye_translation_rms_m=args.max_hand_eye_translation_rms_m,
        max_hand_eye_rotation_rms_rad=args.max_hand_eye_rotation_rms_rad,
        max_table_fit_rms_m=args.max_table_fit_rms_m,
        max_stereo_epipolar_error_px=args.max_stereo_epipolar_error_px,
    )
    result["inputs"] = {
        "d405_calibration": str(args.d405_calibration),
        "observations": str(args.observations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "acceptance_eligible": result["acceptance_eligible"],
                "source_task_pose_root": result["source_task_pose_root"],
                "metrics": result["metrics"],
            }
        )
    )


if __name__ == "__main__":
    main()
