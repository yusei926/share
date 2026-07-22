#!/usr/bin/env python3
"""Extract auditable 2-D UTTER-leg evidence from the two D405 RGB streams.

This is deliberately an image-measurement tool, not a pose estimator.  D405
intrinsics are pinned, but camera-to-wrist extrinsics still need hand-eye
calibration.  The output can therefore constrain a later multi-view CAD fit
without pretending to provide a table 6-D trajectory on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_wrist_leg_observability/v1"


def detect_leg_axis(path: Path, *, debug_path: Path | None = None) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError(f"expected 640x480 BGR image: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White UTTER plastic is bright and weakly saturated.  The black workbench
    # and Dex1 fingertips therefore provide useful contrast even under D405
    # exposure variation.
    mask = cv2.inRange(hsv, (0, 0, 115), (180, 105, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(mask, 45, 135)
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, threshold=28, minLineLength=42, maxLineGap=18)
    candidates: list[dict[str, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 42.0:
                continue
            angle = float(np.arctan2(y2 - y1, x2 - x1))
            sample_count = max(8, int(length / 4.0))
            xs = np.linspace(x1, x2, sample_count).round().astype(np.int32)
            ys = np.linspace(y1, y2, sample_count).round().astype(np.int32)
            support = float(np.mean(mask[np.clip(ys, 0, 479), np.clip(xs, 0, 639)] > 0))
            candidates.append(
                {
                    "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
                    "length_px": length, "angle_rad": angle, "white_support": support,
                    "score": length * support,
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else None
    interior = bool(
        best is not None
        and all(
            8.0 <= best[key] <= limit
            for key, limit in (("x1_px", 631.0), ("x2_px", 631.0), ("y1_px", 471.0), ("y2_px", 471.0))
        )
    )
    if debug_path is not None:
        overlay = image.copy()
        if best is not None:
            cv2.line(
                overlay,
                (round(best["x1_px"]), round(best["y1_px"])),
                (round(best["x2_px"]), round(best["y2_px"])),
                (0, 255, 0),
                4,
            )
            cv2.putText(overlay, f"score={best['score']:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(debug_path), overlay):
            raise OSError(f"could not write {debug_path}")
    return {
        "image": str(path),
        "white_mask_fraction": float(np.count_nonzero(mask)) / float(mask.size),
        "leg_axis_candidate": best,
        "candidate_count": len(candidates),
        "observable": bool(best is not None and best["white_support"] >= 0.55 and interior),
        "candidate_is_interior": interior,
        "use": "offline D405 image constraint only; not a table pose or policy feature",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--source-episode-index", type=int, required=True)
    parser.add_argument("--source-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path)
    args = parser.parse_args()
    if args.source_episode_index < 0 or args.source_frame < 0:
        raise ValueError("source episode/frame indices must be non-negative")
    debug_dir = args.debug_dir.expanduser().resolve() if args.debug_dir else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_episode_index": args.source_episode_index,
        "source_frame": args.source_frame,
        "left_wrist": detect_leg_axis(
            args.left.expanduser().resolve(),
            debug_path=None if debug_dir is None else debug_dir / "left_wrist_leg_axis.png",
        ),
        "right_wrist": detect_leg_axis(
            args.right.expanduser().resolve(),
            debug_path=None if debug_dir is None else debug_dir / "right_wrist_leg_axis.png",
        ),
        "policy_use": "forbidden: offline camera/table calibration only",
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({key: report[key]["observable"] for key in ("left_wrist", "right_wrist")}, indent=2))


if __name__ == "__main__":
    main()
