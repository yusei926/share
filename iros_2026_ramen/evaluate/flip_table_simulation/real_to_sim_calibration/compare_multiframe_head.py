#!/usr/bin/env python3
"""Audit temporal head-camera table geometry for one real/sim replay pair.

The result is deliberately diagnostic and fail-closed.  It establishes whether
enough independent head frames have reliable RGB table geometry before fitting
shared camera extrinsics or physical contact parameters.
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
from evaluate.flip_table_simulation.real_to_sim_calibration.visual_alignment import (
    compare_images,
)


def parse_frame_map(value: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        try:
            real, simulated = (int(item.strip()) for item in raw.split(":", 1))
        except ValueError as exc:
            raise ValueError("frame map must be real_frame:sim_frame pairs") from exc
        if real < 0 or simulated < 0:
            raise ValueError("frame map indices must be non-negative")
        pairs.append((real, simulated))
    if len(pairs) < 2 or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("frame map must contain at least two distinct real frames")
    return tuple(pairs)


def frame_map_from_replay_actions(path: Path) -> tuple[tuple[int, int], ...]:
    """Read the source-frame/simulator-step map emitted with one replay.

    Camera comparisons must use the exact rounding used by the recorded-target
    policy.  Reading this immutable replay artifact avoids a hand-written map
    accidentally comparing a real RGB frame to a different physical instant.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("camera_frame_map")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("replay actions must contain at least two camera_frame_map entries")
    pairs: list[tuple[int, int]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"camera_frame_map[{index}] must be an object")
        source_frame = entry.get("source_frame")
        simulator_step = entry.get("simulator_step")
        if (
            not isinstance(source_frame, int)
            or not isinstance(simulator_step, int)
            or source_frame < 0
            or simulator_step < 0
        ):
            raise ValueError(
                f"camera_frame_map[{index}] must contain non-negative integer source_frame and simulator_step"
            )
        pairs.append((source_frame, simulator_step))
    if len({source for source, _ in pairs}) != len(pairs):
        raise ValueError("replay camera_frame_map has duplicate source frames")
    if pairs != sorted(pairs):
        raise ValueError("replay camera_frame_map must be sorted by source frame")
    return tuple(pairs)


def _estimate(
    path: Path, *, real: bool, sim_recorded_geometry: bool = False
) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError(f"expected 640x480 RGB image: {path}")
    intrinsic, distortion = (
        CameraCalibration.g1_head_left_real_raw_intrinsics()
        if real or sim_recorded_geometry
        else CameraCalibration._g1_head_left_sim_intrinsics()
    )
    estimate = TabletopPoseEstimator(
        CameraCalibration(intrinsic, distortion, np.eye(4, dtype=np.float64))
    ).estimate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return {
        "image": str(path),
        "corners_px": estimate.corners_px.tolist(),
        "center_px": estimate.corners_px.mean(axis=0).tolist(),
        "confidence": float(estimate.confidence),
        "reprojection_error_px": float(estimate.reprojection_error_px),
        "area_fraction": float(estimate.area_fraction),
        "camera_from_table": estimate.camera_from_table.tolist(),
    }


def _reliable(estimate: dict[str, Any]) -> bool:
    """Apply the same source-PnP quality gate to both images in a pair."""

    return bool(
        estimate["confidence"] >= 0.10
        and estimate["reprojection_error_px"] <= 8.0
    )


def _corner_rmse(real: np.ndarray, simulated: np.ndarray) -> float:
    errors = []
    for reverse in (False, True):
        candidate = simulated[::-1] if reverse else simulated
        for shift in range(4):
            errors.append(float(np.sqrt(np.mean((real - np.roll(candidate, shift, axis=0)) ** 2))))
    return min(errors)


def _sim_head_image_path(
    sim_root: Path, sim_frame: int, environment_index: int | None
) -> Path:
    """Resolve one normal or batched camera-frame export deterministically."""

    frame_dir = sim_root / "test_0" / "camera_frames" / f"frame_{sim_frame:04d}"
    if environment_index is None:
        return frame_dir / "head_left_rgb.png"
    return frame_dir / f"env_{environment_index:03d}" / "head_left_rgb.png"


def _robot_mask_path(root: Path | None, frame: int) -> Path | None:
    """Resolve an optional mask exported for the exact compared frame."""

    if root is None:
        return None
    path = root / f"frame_{frame:04d}" / "head_left_robot_mask.png"
    if not path.is_file():
        raise FileNotFoundError(f"robot exclusion mask is missing for frame {frame}: {path}")
    return path


def compare(
    real_root: Path,
    sim_root: Path,
    mapping: tuple[tuple[int, int], ...],
    *,
    sim_recorded_geometry: bool = False,
    environment_index: int | None = None,
    source_episode_index: int | None = None,
    frame_map_source: str = "explicit",
    real_robot_mask_root: Path | None = None,
    simulated_robot_mask_root: Path | None = None,
) -> dict[str, Any]:
    if (real_robot_mask_root is None) != (simulated_robot_mask_root is None):
        raise ValueError("real and simulated robot mask roots must be supplied together")
    frames = []
    for real_frame, sim_frame in mapping:
        real = _estimate(real_root / f"frame_{real_frame:04d}" / "head_left.png", real=True)
        simulated_path = _sim_head_image_path(sim_root, sim_frame, environment_index)
        simulated = _estimate(
            simulated_path,
            real=False,
            sim_recorded_geometry=sim_recorded_geometry,
        )
        real_path = real_root / f"frame_{real_frame:04d}" / "head_left.png"
        real_robot_mask = _robot_mask_path(real_robot_mask_root, real_frame)
        simulated_robot_mask = _robot_mask_path(simulated_robot_mask_root, sim_frame)
        silhouette = compare_images(
            real_path,
            simulated_path,
            real_robot_mask=real_robot_mask,
            simulated_robot_mask=simulated_robot_mask,
        )
        corner_rmse = _corner_rmse(
            np.asarray(real["corners_px"], dtype=np.float64),
            np.asarray(simulated["corners_px"], dtype=np.float64),
        )
        center_error = float(
            np.linalg.norm(
                np.asarray(real["center_px"], dtype=np.float64)
                - np.asarray(simulated["center_px"], dtype=np.float64)
            )
        )
        real_reliable = _reliable(real)
        sim_reliable = _reliable(simulated)
        frames.append(
            {
                "real_frame": real_frame,
                "sim_frame": sim_frame,
                "real": real,
                "sim": simulated,
                "real_robot_mask": None if real_robot_mask is None else str(real_robot_mask),
                "simulated_robot_mask": (
                    None if simulated_robot_mask is None else str(simulated_robot_mask)
                ),
                "corner_rmse_px": corner_rmse,
                "center_error_px": center_error,
                "silhouette_alignment": silhouette,
                "real_reliable": real_reliable,
                "sim_reliable": sim_reliable,
                "pair_reliable": real_reliable and sim_reliable,
            }
        )
    reliable = [frame for frame in frames if frame["pair_reliable"]]
    metrics: dict[str, float] = {}
    metric_sources: dict[str, str] = {}
    if len(reliable) >= 3:
        metrics = {
            "camera_reprojection_median_px": float(
                np.median([frame["corner_rmse_px"] for frame in reliable])
            ),
            "camera_reprojection_p95_px": float(
                np.quantile([frame["corner_rmse_px"] for frame in reliable], 0.95)
            ),
            "mask_iou": float(
                np.median(
                    [frame["silhouette_alignment"]["mask_iou"] for frame in reliable]
                )
            ),
        }
        metric_sources = {
            "camera_reprojection_median_px": "paired real/sim head-left CAD corner reprojection over reliable frames",
            "camera_reprojection_p95_px": "paired real/sim head-left CAD corner reprojection over reliable frames",
            "mask_iou": "paired real/sim head-left silhouette alignment over reliable frames",
        }
    return {
        "schema_version": "team_ramen_multiframe_head_geometry/v1",
        "source_episode_index": source_episode_index,
        "frame_map": [{"real": real, "sim": simulated} for real, simulated in mapping],
        "frame_map_source": frame_map_source,
        "sim_recorded_geometry": sim_recorded_geometry,
        "environment_index": environment_index,
        "robot_self_occlusion_excluded": real_robot_mask_root is not None,
        "real_robot_mask_root": (
            None if real_robot_mask_root is None else str(real_robot_mask_root)
        ),
        "simulated_robot_mask_root": (
            None if simulated_robot_mask_root is None else str(simulated_robot_mask_root)
        ),
        "frames": frames,
        "summary": {
            "frames": len(frames),
            "reliable_pairs": len(reliable),
            "corner_rmse_median_px": float(np.median([frame["corner_rmse_px"] for frame in frames])),
            "corner_rmse_p95_px": float(np.quantile([frame["corner_rmse_px"] for frame in frames], 0.95)),
            "silhouette_mask_iou_median": float(
                np.median([frame["silhouette_alignment"]["mask_iou"] for frame in frames])
            ),
            "silhouette_edge_distance_median_px": float(
                np.median(
                    [
                        frame["silhouette_alignment"]["edge_distance_symmetric_px"]
                        for frame in frames
                    ]
                )
            ),
            "accepted_for_extrinsic_fit": len(reliable) >= 3,
        },
        "metrics": metrics,
        "metric_sources": metric_sources,
        "decision": (
            "ready_for_multiview_extrinsic_fit" if len(reliable) >= 3 else "insufficient_reliable_rgb_geometry"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    frame_map_group = parser.add_mutually_exclusive_group(required=True)
    frame_map_group.add_argument("--frame-map")
    frame_map_group.add_argument(
        "--replay-actions",
        type=Path,
        help="Read the exact source-frame/simulator-step mapping from replay_actions.json.",
    )
    parser.add_argument(
        "--sim-recorded-geometry",
        action="store_true",
        help="interpret simulator frames as the raw recorded camera model",
    )
    parser.add_argument(
        "--source-episode-index",
        type=int,
        help="record the immutable source episode represented by --real-root",
    )
    parser.add_argument(
        "--environment-index",
        type=int,
        help="read one batched simulator camera export (for example env_000)",
    )
    parser.add_argument(
        "--real-robot-mask-root",
        type=Path,
        help="root written by export_head_robot_masks.py source",
    )
    parser.add_argument(
        "--simulated-robot-mask-root",
        type=Path,
        help="root written by export_head_robot_masks.py simulation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_actions is not None:
        mapping = frame_map_from_replay_actions(args.replay_actions.expanduser().resolve())
        frame_map_source = str(args.replay_actions.expanduser().resolve())
    else:
        mapping = parse_frame_map(args.frame_map)
        frame_map_source = "explicit"
    report = compare(
        args.real_root.expanduser().resolve(),
        args.sim_root.expanduser().resolve(),
        mapping,
        sim_recorded_geometry=args.sim_recorded_geometry,
        environment_index=args.environment_index,
        source_episode_index=args.source_episode_index,
        frame_map_source=frame_map_source,
        real_robot_mask_root=(
            None
            if args.real_robot_mask_root is None
            else args.real_robot_mask_root.expanduser().resolve()
        ),
        simulated_robot_mask_root=(
            None
            if args.simulated_robot_mask_root is None
            else args.simulated_robot_mask_root.expanduser().resolve()
        ),
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
