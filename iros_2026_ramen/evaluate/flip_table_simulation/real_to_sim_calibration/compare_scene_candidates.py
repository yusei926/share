#!/usr/bin/env python3
"""Compare fixed-scene head-camera candidates without accepting weak PnP fits.

This is an offline calibration utility.  It never provides a pose to a policy;
it only decides whether a proposed simulator scene has enough image evidence to
be carried into the next, multi-view fitting stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    CameraCalibration,
    TabletopPoseEstimator,
)


def _estimate(
    image_path: Path, *, real: bool, sim_recorded_geometry: bool = False
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.shape != (480, 640, 3):
        raise ValueError(f"expected a 640x480 RGB image: {image_path}")
    intrinsic, distortion = (
        CameraCalibration.g1_head_left_real_raw_intrinsics()
        if real or sim_recorded_geometry
        else CameraCalibration._g1_head_left_sim_intrinsics()
    )
    calibration = CameraCalibration(intrinsic, distortion, np.eye(4, dtype=np.float64))
    estimate = TabletopPoseEstimator(calibration).estimate(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )
    return {
        "image": str(image_path),
        "corners_px": estimate.corners_px.tolist(),
        "center_px": estimate.corners_px.mean(axis=0).tolist(),
        "area_fraction": float(estimate.area_fraction),
        "confidence": float(estimate.confidence),
        "reprojection_error_px": float(estimate.reprojection_error_px),
        "camera_from_table": estimate.camera_from_table.tolist(),
    }


def _corner_error(real: np.ndarray, simulated: np.ndarray) -> tuple[float, tuple[bool, int]]:
    """Return correspondence-invariant RMS error for an unlabeled quadrilateral."""

    if real.shape != (4, 2) or simulated.shape != (4, 2):
        raise ValueError("both corner arrays must be [4,2]")
    candidates: list[tuple[float, tuple[bool, int]]] = []
    for reversed_order in (False, True):
        ordered = simulated[::-1] if reversed_order else simulated
        for shift in range(4):
            error = float(np.sqrt(np.mean((real - np.roll(ordered, shift, axis=0)) ** 2)))
            candidates.append((error, (reversed_order, shift)))
    return min(candidates, key=lambda item: item[0])


def compare(
    real_image: Path, candidate_roots: list[Path], *, sim_recorded_geometry: bool = False
) -> dict[str, Any]:
    real = _estimate(real_image, real=True)
    real_corners = np.asarray(real["corners_px"], dtype=np.float64)
    candidates = []
    for root in candidate_roots:
        image = root / "test_0" / "camera_frames" / "frame_0119" / "head_left_rgb.png"
        estimate = _estimate(image, real=False, sim_recorded_geometry=sim_recorded_geometry)
        error, correspondence = _corner_error(
            real_corners, np.asarray(estimate["corners_px"], dtype=np.float64)
        )
        center_error = float(
            np.linalg.norm(
                np.asarray(real["center_px"], dtype=np.float64)
                - np.asarray(estimate["center_px"], dtype=np.float64)
            )
        )
        estimate.update(
            {
                "candidate_root": str(root),
                "corner_rmse_px": error,
                "center_error_px": center_error,
                "corner_correspondence": {
                    "reversed": correspondence[0],
                    "cyclic_shift": correspondence[1],
                },
            }
        )
        candidates.append(estimate)
    candidates.sort(key=lambda item: item["corner_rmse_px"])
    # A single low-confidence RGB PnP estimate is not enough to update a
    # physical scene.  This report must fail closed until temporal/multi-view
    # evidence is added by the next calibration stage.
    reliable_real = real["confidence"] >= 0.10 and real["reprojection_error_px"] <= 8.0
    return {
        "schema_version": "team_ramen_scene_candidate_comparison/v1",
        "real": real,
        "candidates": candidates,
        "accepted_candidate": None,
        "decision": "rejected_pending_multiview_fit",
        "decision_reason": (
            "single-frame real PnP does not satisfy the confidence/reprojection gate"
            if not reliable_real
            else "single-frame candidate scoring is diagnostic only; require multi-view fit"
        ),
        "single_frame_gate": {
            "real_confidence_min": 0.10,
            "real_reprojection_error_px_max": 8.0,
            "real_is_reliable": reliable_real,
        },
        "sim_recorded_geometry": sim_recorded_geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-image", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--sim-recorded-geometry",
        action="store_true",
        help="use raw head intrinsics for a simulator image saved after the recorded-camera remap",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.real_image.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.candidate_root],
        sim_recorded_geometry=args.sim_recorded_geometry,
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"decision": report["decision"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
