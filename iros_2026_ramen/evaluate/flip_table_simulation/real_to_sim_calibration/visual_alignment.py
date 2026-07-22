#!/usr/bin/env python3
"""Offline, appearance-robust alignment metrics for real/sim RGB pairs.

The real demonstrations frequently occlude the white tabletop rim with hands,
legs, and specular highlights.  A four-corner PnP fit can therefore lock onto
an interior brace and report a numerically plausible but physically wrong
camera pose.  This module deliberately does *not* estimate a pose.  It scores
the visible white-table silhouette and its boundary directly, so it can rank
reset/camera candidates before a geometry fit is trusted.

It is calibration-only.  Masks, edge maps, and scores must never be exposed to
a policy, planner, reward, or inference-time branch.
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
    TabletopPoseEstimator,
)


# This crop retains every table corner/leg root in the head view, but rejects
# the robot forearms and bottom-image hand glare.  It is deliberately a fixed
# image-space measurement, not a learned or simulator-state-dependent mask.
_ROI_Y0, _ROI_Y1 = 70, 365
_ROI_X0, _ROI_X1 = 55, 585


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError(f"expected 640x480 RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def table_silhouette(rgb: np.ndarray) -> np.ndarray:
    """Return a conservative white-table mask for cross-domain comparison."""

    source = TabletopPoseEstimator.segment_table_assembly(rgb)
    mask = np.zeros_like(source, dtype=np.uint8)
    mask[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1] = source[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1]
    # Remove isolated pixels but preserve long, narrow legs.  A small close
    # repairs real-camera JPEG/specular holes without turning braces into a
    # filled tabletop rectangle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8)


def _edge(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))


def _mean_distance(source_edge: np.ndarray, target_edge: np.ndarray) -> float:
    if not np.any(source_edge) or not np.any(target_edge):
        return float("inf")
    distance = cv2.distanceTransform((target_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.mean(distance[source_edge > 0]))


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first > 0, second > 0)
    if not np.any(union):
        return 0.0
    return float(np.logical_and(first > 0, second > 0).sum() / union.sum())


def compare_images(real_path: Path, simulated_path: Path) -> dict[str, Any]:
    """Compare a real/sim head image without assuming a PnP corner fit."""

    real_mask = table_silhouette(_read_rgb(real_path))
    simulated_mask = table_silhouette(_read_rgb(simulated_path))
    real_edge, simulated_edge = _edge(real_mask), _edge(simulated_mask)
    real_to_sim = _mean_distance(real_edge, simulated_edge)
    sim_to_real = _mean_distance(simulated_edge, real_edge)
    return {
        "schema_version": "team_ramen_table_silhouette_alignment/v1",
        "policy_use": "forbidden: offline camera/scene calibration only",
        "real_image": str(real_path),
        "simulated_image": str(simulated_path),
        "roi_xyxy": [_ROI_X0, _ROI_Y0, _ROI_X1, _ROI_Y1],
        "real_mask_fraction": float(real_mask.mean()),
        "simulated_mask_fraction": float(simulated_mask.mean()),
        "mask_iou": _iou(real_mask, simulated_mask),
        "edge_distance_real_to_sim_px": real_to_sim,
        "edge_distance_sim_to_real_px": sim_to_real,
        "edge_distance_symmetric_px": float(0.5 * (real_to_sim + sim_to_real)),
    }


def _debug_overlay(real_path: Path, simulated_path: Path, output: Path) -> None:
    real = _read_rgb(real_path)
    simulated = _read_rgb(simulated_path)
    real_mask, simulated_mask = table_silhouette(real), table_silhouette(simulated)
    overlay = np.zeros_like(real)
    overlay[..., 0] = real_mask * 255
    overlay[..., 1] = simulated_mask * 255
    blended = cv2.addWeighted(real, 0.55, overlay, 0.45, 0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)):
        raise OSError(f"unable to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-overlay", type=Path)
    args = parser.parse_args()
    result = compare_images(args.real.expanduser().resolve(), args.simulated.expanduser().resolve())
    atomic_write_json(args.output.expanduser().resolve(), result)
    if args.debug_overlay is not None:
        _debug_overlay(args.real.expanduser().resolve(), args.simulated.expanduser().resolve(), args.debug_overlay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
