#!/usr/bin/env python3
"""Recover auditable head-stereo geometry from the recorded RGB pair.

This utility is deliberately offline-only.  It uses the calibrated physical
head stereo pair to check scene scale and tabletop visibility while fitting the
simulator.  No disparity, mask, or reconstructed 3-D point is exposed to a
policy, planner, reward, or runtime branch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    TabletopPoseEstimator,
)


@dataclass(frozen=True)
class HeadStereoCalibration:
    """Pinned physical stereo calibration expressed in millimetres/pixels."""

    left_intrinsic: np.ndarray
    right_intrinsic: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    rotation_right_from_left: np.ndarray
    translation_right_from_left_mm: np.ndarray
    image_size: tuple[int, int]
    rms_error_px: float

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.translation_right_from_left_mm) / 1000.0)

    @classmethod
    def load(cls, path: Path) -> "HeadStereoCalibration":
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("stereo calibration must be a mapping")

        def matrix(key: str, shape: tuple[int, ...]) -> np.ndarray:
            result = np.asarray(value.get(key), dtype=np.float64)
            if key == "T" and result.shape == (3,):
                result = result.reshape(3, 1)
            if result.shape != shape or not np.isfinite(result).all():
                raise ValueError(f"{key} must be finite with shape {shape}, got {result.shape}")
            return result

        image_size = tuple(int(item) for item in value.get("image_size", ()))
        if image_size != (640, 480):
            raise ValueError(f"head stereo must remain 640x480, got {image_size}")
        result = cls(
            left_intrinsic=matrix("camera_matrix_left", (3, 3)),
            right_intrinsic=matrix("camera_matrix_right", (3, 3)),
            left_distortion=matrix("dist_coeffs_left", (5,)),
            right_distortion=matrix("dist_coeffs_right", (5,)),
            rotation_right_from_left=matrix("R", (3, 3)),
            translation_right_from_left_mm=matrix("T", (3, 1)),
            image_size=image_size,
            rms_error_px=float(value["rms_error"]),
        )
        if not 0.04 <= result.baseline_m <= 0.08:
            raise ValueError(f"physical stereo baseline is implausible: {result.baseline_m:.6f} m")
        if not np.isfinite(result.rms_error_px) or result.rms_error_px <= 0.0:
            raise ValueError("rms_error must be finite and positive")
        return result


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError(f"expected a 640x480 RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def left_right_consistency_mask(
    left: np.ndarray, right: np.ndarray, tolerance_px: float = 1.5
) -> np.ndarray:
    """Return a sign-aware, bilinearly sampled stereo consistency mask."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left/right disparities must have matching [H,W] shapes")
    height, width = left.shape
    columns = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], left.shape)
    corresponding = columns - left
    valid = np.isfinite(left) & (left > 0.0) & (corresponding >= 0.0) & (corresponding <= width - 1)
    x0 = np.clip(np.floor(np.where(np.isfinite(corresponding), corresponding, 0.0)).astype(np.int64), 0, width - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    alpha = corresponding - x0
    rows = np.arange(height, dtype=np.int64)[:, None]
    sample = (1.0 - alpha) * right[rows, x0] + alpha * right[rows, x1]
    return valid & np.isfinite(sample) & (sample > 0.0) & (np.abs(left - sample) <= tolerance_px)


def estimate_pair(left_path: Path, right_path: Path, calibration_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Rectify a recorded pair and return scale diagnostics plus depth in metres."""

    calibration = HeadStereoCalibration.load(calibration_path)
    left = _read_rgb(left_path)
    right = _read_rgb(right_path)
    width, height = calibration.image_size
    rectification = cv2.stereoRectify(
        calibration.left_intrinsic,
        calibration.left_distortion,
        calibration.right_intrinsic,
        calibration.right_distortion,
        (width, height),
        calibration.rotation_right_from_left,
        calibration.translation_right_from_left_mm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    r1, r2, p1, p2 = rectification[:4]
    left_map = cv2.initUndistortRectifyMap(
        calibration.left_intrinsic, calibration.left_distortion, r1, p1, (width, height), cv2.CV_32FC1
    )
    right_map = cv2.initUndistortRectifyMap(
        calibration.right_intrinsic, calibration.right_distortion, r2, p2, (width, height), cv2.CV_32FC1
    )
    left_rectified = cv2.remap(left, *left_map, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, *right_map, interpolation=cv2.INTER_LINEAR)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    left_disparity = matcher.compute(
        cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY),
    ).astype(np.float32) / 16.0
    right_disparity = matcher.compute(
        cv2.flip(cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY), 1),
        cv2.flip(cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY), 1),
    ).astype(np.float32) / 16.0
    right_disparity = cv2.flip(right_disparity, 1)
    consistent = left_right_consistency_mask(left_disparity, right_disparity)
    focal_px = float(p1[0, 0])
    baseline_m = abs(float(p2[0, 3] - p1[0, 3])) / focal_px / 1000.0
    valid = consistent & np.isfinite(left_disparity) & (left_disparity > 0.0)
    depth = np.zeros((height, width), dtype=np.float32)
    depth[valid] = focal_px * baseline_m / left_disparity[valid]
    valid &= (depth >= 0.15) & (depth <= 2.5)
    depth[~valid] = 0.0
    table_mask = TabletopPoseEstimator.segment_table_assembly(left_rectified) > 0
    table_valid = valid & table_mask
    table_depth = depth[table_valid]
    diagnostics = {
        "schema_version": "team_ramen_head_stereo_geometry/v1",
        "policy_use": "forbidden: offline calibration diagnostic only",
        "left_image": str(left_path),
        "right_image": str(right_path),
        "calibration": str(calibration_path),
        "image_size": [width, height],
        "baseline_m": baseline_m,
        "source_calibration_rms_error_px": calibration.rms_error_px,
        "rectified_focal_px": focal_px,
        "valid_depth_fraction": float(np.mean(valid)),
        "table_mask_fraction": float(np.mean(table_mask)),
        "table_valid_depth_fraction": float(np.mean(table_valid)),
        "table_depth_median_m": float(np.median(table_depth)) if table_depth.size else None,
        "table_depth_p05_m": float(np.quantile(table_depth, 0.05)) if table_depth.size else None,
        "table_depth_p95_m": float(np.quantile(table_depth, 0.95)) if table_depth.size else None,
        "accepted_for_metric_scale": bool(table_depth.size >= 800),
    }
    return diagnostics, depth


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    rendered = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        normalized = np.zeros_like(depth, dtype=np.uint8)
        lo, hi = np.quantile(depth[valid], (0.02, 0.98))
        normalized[valid] = np.clip((depth[valid] - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
        rendered = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        rendered[~valid] = 0
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, depth = estimate_pair(
        args.left.expanduser().resolve(),
        args.right.expanduser().resolve(),
        args.calibration.expanduser().resolve(),
    )
    np.save(output_dir / "depth_m.npy", depth)
    cv2.imwrite(str(output_dir / "depth_preview.png"), _depth_visualization(depth))
    atomic_write_json(output_dir / "stereo_diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
