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


def _exclusion_mask(value: np.ndarray | None) -> np.ndarray | None:
    """Validate an optional offline robot-occupancy exclusion mask."""

    if value is None:
        return None
    mask = np.asarray(value)
    if mask.shape != (480, 640):
        raise ValueError("robot exclusion mask must be 480x640")
    if mask.dtype == bool:
        return mask
    if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
        raise ValueError("robot exclusion mask must be finite")
    return mask > 0


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (480, 640):
        raise ValueError(f"expected 640x480 robot exclusion mask: {path}")
    return image > 0


def table_silhouette(
    rgb: np.ndarray,
    *,
    robot_exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a conservative RGB table mask, excluding known robot pixels.

    The exclusion is generated from measured joints and a pinned robot visual
    model outside this module.  It is not a table-CAD render and therefore
    cannot make a candidate camera/table pose look correct by construction.
    """

    source = TabletopPoseEstimator.segment_table_assembly(rgb)
    mask = np.zeros_like(source, dtype=np.uint8)
    mask[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1] = source[_ROI_Y0:_ROI_Y1, _ROI_X0:_ROI_X1]
    # Remove isolated pixels but preserve long, narrow legs.  A small close
    # repairs real-camera JPEG/specular holes without turning braces into a
    # filled tabletop rectangle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    exclusion = _exclusion_mask(robot_exclusion_mask)
    if exclusion is not None:
        mask[exclusion] = 0
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


def compare_images(
    real_path: Path,
    simulated_path: Path,
    *,
    real_robot_mask: Path | None = None,
    simulated_robot_mask: Path | None = None,
) -> dict[str, Any]:
    """Compare a real/sim head image without assuming a PnP corner fit."""

    real_exclusion = None if real_robot_mask is None else _read_mask(real_robot_mask)
    simulated_exclusion = (
        None if simulated_robot_mask is None else _read_mask(simulated_robot_mask)
    )
    real_mask = table_silhouette(
        _read_rgb(real_path), robot_exclusion_mask=real_exclusion
    )
    simulated_mask = table_silhouette(
        _read_rgb(simulated_path), robot_exclusion_mask=simulated_exclusion
    )
    real_edge, simulated_edge = _edge(real_mask), _edge(simulated_mask)
    real_to_sim = _mean_distance(real_edge, simulated_edge)
    sim_to_real = _mean_distance(simulated_edge, real_edge)
    return {
        "schema_version": "team_ramen_table_silhouette_alignment/v2",
        "policy_use": "forbidden: offline camera/scene calibration only",
        "real_image": str(real_path),
        "simulated_image": str(simulated_path),
        "real_robot_exclusion_mask": (
            None if real_robot_mask is None else str(real_robot_mask)
        ),
        "simulated_robot_exclusion_mask": (
            None if simulated_robot_mask is None else str(simulated_robot_mask)
        ),
        "real_robot_exclusion_fraction": (
            0.0 if real_exclusion is None else float(real_exclusion.mean())
        ),
        "simulated_robot_exclusion_fraction": (
            0.0 if simulated_exclusion is None else float(simulated_exclusion.mean())
        ),
        "roi_xyxy": [_ROI_X0, _ROI_Y0, _ROI_X1, _ROI_Y1],
        "real_mask_fraction": float(real_mask.mean()),
        "simulated_mask_fraction": float(simulated_mask.mean()),
        "mask_iou": _iou(real_mask, simulated_mask),
        "edge_distance_real_to_sim_px": real_to_sim,
        "edge_distance_sim_to_real_px": sim_to_real,
        "edge_distance_symmetric_px": float(0.5 * (real_to_sim + sim_to_real)),
    }


def _debug_overlay(
    real_path: Path,
    simulated_path: Path,
    output: Path,
    *,
    real_robot_mask: Path | None = None,
    simulated_robot_mask: Path | None = None,
) -> None:
    real = _read_rgb(real_path)
    simulated = _read_rgb(simulated_path)
    real_mask = table_silhouette(
        real,
        robot_exclusion_mask=(
            None if real_robot_mask is None else _read_mask(real_robot_mask)
        ),
    )
    simulated_mask = table_silhouette(
        simulated,
        robot_exclusion_mask=(
            None if simulated_robot_mask is None else _read_mask(simulated_robot_mask)
        ),
    )
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
    parser.add_argument(
        "--real-robot-mask",
        type=Path,
        help="optional projected real robot-occupancy mask; offline calibration only",
    )
    parser.add_argument(
        "--simulated-robot-mask",
        type=Path,
        help="optional projected simulator robot-occupancy mask; offline calibration only",
    )
    args = parser.parse_args()
    real_robot_mask = (
        None if args.real_robot_mask is None else args.real_robot_mask.expanduser().resolve()
    )
    simulated_robot_mask = (
        None
        if args.simulated_robot_mask is None
        else args.simulated_robot_mask.expanduser().resolve()
    )
    result = compare_images(
        args.real.expanduser().resolve(),
        args.simulated.expanduser().resolve(),
        real_robot_mask=real_robot_mask,
        simulated_robot_mask=simulated_robot_mask,
    )
    atomic_write_json(args.output.expanduser().resolve(), result)
    if args.debug_overlay is not None:
        _debug_overlay(
            args.real.expanduser().resolve(),
            args.simulated.expanduser().resolve(),
            args.debug_overlay,
            real_robot_mask=real_robot_mask,
            simulated_robot_mask=simulated_robot_mask,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
