#!/usr/bin/env python3
"""Recover only observable table poses from calibrated head-stereo RGB.

This is deliberately an offline calibration tool.  It converts the recorded
head-left/head-right RGB pair into a masked stereo point cloud, composes it
with encoder FK, and registers the V1 assembled-table CAD surface with robust
point-to-point ICP.  The tracker never reads a simulator pose, contact, or
segmentation label.

The recorded RGB stream cannot observe the table during every flip frame.  A
failed/occluded frame is explicitly emitted as ``unobserved`` rather than
interpolated, predicted, or copied from the simulator.  Consequently its
output can guide data collection and contact fitting, but it is not by itself
proof of the 20 mm / 3 degree held-out trajectory gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex, extract_video_frame
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    TabletopPoseEstimator,
)

from .source_cad_alignment import (
    _build_torso_fk,
    _load_rows,
    _root_from_head_eyes,
    _transform,
)
from .stereo_geometry import HeadStereoCalibration, left_right_consistency_mask


SCHEMA_VERSION = "team_ramen_flip_table_temporal_stereo_cad_tracker/v1"
MAX_TRACKED_POINTS = 5_000
CAD_SAMPLE_COUNT = 4_000
ICP_ITERATIONS = 18
MIN_CORRESPONDENCES = 260
MAX_CORRESPONDENCE_M = 0.060
MAX_ACCEPTED_RMSE_M = 0.018
MIN_MODEL_SUPPORT = 0.075
CAD_ROI_MAX_PIXEL_DISTANCE = 50.0
CAD_ROI_MAX_DEPTH_ERROR_M = 0.100
TABLE_YAW_180_SYMMETRY = np.diag((-1.0, -1.0, 1.0, 1.0))


@dataclass(frozen=True)
class StereoCloud:
    points_camera_m: np.ndarray
    diagnostics: dict[str, Any]


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite 4x4")
    return matrix


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())
    )


def _canonical_pose(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    alternate = candidate @ TABLE_YAW_180_SYMMETRY
    return min((candidate, alternate), key=lambda value: _rotation_error_deg(reference, value))


def _sample_points(points: np.ndarray, limit: int, *, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    generator = np.random.default_rng(seed)
    return points[generator.choice(len(points), size=limit, replace=False)]


def _stereo_cloud(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    stereo: HeadStereoCalibration,
    *,
    seed: int,
    cad_points: np.ndarray | None = None,
    predicted_root_from_table: np.ndarray | None = None,
    root_from_left_camera: np.ndarray | None = None,
) -> StereoCloud:
    """Return table-only points in the *unrectified* left OpenCV camera frame."""

    cad_roi_requested = (
        cad_points is not None or predicted_root_from_table is not None or root_from_left_camera is not None
    )
    if cad_roi_requested and (
        cad_points is None or predicted_root_from_table is None or root_from_left_camera is None
    ):
        raise ValueError("CAD ROI requires CAD points, predicted table pose, and left-camera pose")
    width, height = stereo.image_size
    rectification = cv2.stereoRectify(
        stereo.left_intrinsic,
        stereo.left_distortion,
        stereo.right_intrinsic,
        stereo.right_distortion,
        (width, height),
        stereo.rotation_right_from_left,
        stereo.translation_right_from_left_mm,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    r1, r2, p1, p2, q = rectification[:5]
    left_map = cv2.initUndistortRectifyMap(
        stereo.left_intrinsic,
        stereo.left_distortion,
        r1,
        p1,
        (width, height),
        cv2.CV_32FC1,
    )
    right_map = cv2.initUndistortRectifyMap(
        stereo.right_intrinsic,
        stereo.right_distortion,
        r2,
        p2,
        (width, height),
        cv2.CV_32FC1,
    )
    left_rectified = cv2.remap(left_rgb, *left_map, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_rgb, *right_map, interpolation=cv2.INTER_LINEAR)
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
    disparity = matcher.compute(
        cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY),
    ).astype(np.float32) / 16.0
    # One-way matching can mistake white arms and highlights for the white
    # table. Keep only pixels with an inverse right-to-left stereo match.
    reverse_disparity = matcher.compute(
        cv2.flip(cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY), 1),
        cv2.flip(cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY), 1),
    ).astype(np.float32) / 16.0
    reverse_disparity = cv2.flip(reverse_disparity, 1)
    stereo_consistent = left_right_consistency_mask(disparity, reverse_disparity)
    points_rectified_mm = cv2.reprojectImageTo3D(disparity, q)
    table_mask = TabletopPoseEstimator.segment_table_assembly(left_rectified) > 0
    table_candidates = (
        table_mask
        & np.isfinite(points_rectified_mm).all(axis=-1)
        & (disparity > 0.0)
        & (points_rectified_mm[..., 2] > 150.0)
        & (points_rectified_mm[..., 2] < 2500.0)
    )
    valid = table_candidates & stereo_consistent
    roi_diagnostics: dict[str, Any] = {"cad_roi_applied": False}
    if cad_roi_requested:
        assert cad_points is not None
        assert predicted_root_from_table is not None
        assert root_from_left_camera is not None
        camera_from_table = np.linalg.inv(root_from_left_camera) @ predicted_root_from_table
        predicted_camera = _apply(cad_points, camera_from_table)
        predicted_rectified = predicted_camera @ np.asarray(r1, dtype=np.float64).T
        predicted_front = predicted_rectified[:, 2] > 0.05
        predicted_rectified = predicted_rectified[predicted_front]
        pixels = np.column_stack(
            (
                p1[0, 0] * predicted_rectified[:, 0] / predicted_rectified[:, 2] + p1[0, 2],
                p1[1, 1] * predicted_rectified[:, 1] / predicted_rectified[:, 2] + p1[1, 2],
            )
        )
        within_image = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < height)
        )
        pixels = pixels[within_image]
        predicted_depth_m = predicted_rectified[within_image, 2]
        if len(pixels) < MIN_CORRESPONDENCES:
            raise RuntimeError("predicted CAD ROI has insufficient in-frame surface samples")
        ys, xs = np.nonzero(valid)
        candidate_pixels = np.column_stack((xs, ys))
        distances, indices = cKDTree(pixels).query(candidate_pixels, k=1, workers=-1)
        candidate_depth_m = points_rectified_mm[ys, xs, 2] / 1000.0
        keep = (distances <= CAD_ROI_MAX_PIXEL_DISTANCE) & (
            np.abs(candidate_depth_m - predicted_depth_m[indices]) <= CAD_ROI_MAX_DEPTH_ERROR_M
        )
        gated = np.zeros_like(valid)
        gated[ys[keep], xs[keep]] = True
        roi_diagnostics = {
            "cad_roi_applied": True,
            "cad_roi_candidate_points": int(len(candidate_pixels)),
            "cad_roi_accepted_points": int(np.count_nonzero(keep)),
            "cad_roi_max_pixel_distance": CAD_ROI_MAX_PIXEL_DISTANCE,
            "cad_roi_max_depth_error_m": CAD_ROI_MAX_DEPTH_ERROR_M,
        }
        valid = gated
    points_rectified_m = points_rectified_mm[valid].astype(np.float64) / 1000.0
    # Stereo rectification maps original left-camera coordinates to R1 @ X.
    # Undo that rotation before composing with encoder-FK head pose.
    points_camera_m = points_rectified_m @ np.asarray(r1, dtype=np.float64)
    points_camera_m = _sample_points(points_camera_m, MAX_TRACKED_POINTS, seed=seed)
    diagnostics = {
        "table_mask_fraction": float(np.mean(table_mask)),
        "table_candidate_fraction": float(np.mean(table_candidates)),
        "table_stereo_consistent_fraction": (
            float(np.count_nonzero(table_candidates & stereo_consistent))
            / float(max(np.count_nonzero(table_candidates), 1))
        ),
        "valid_stereo_fraction": float(np.mean(valid)),
        "point_count": int(len(points_camera_m)),
        "rectified_focal_px": float(p1[0, 0]),
        "baseline_m": abs(float(p2[0, 3] - p1[0, 3])) / float(p1[0, 0]) / 1000.0,
        **roi_diagnostics,
    }
    return StereoCloud(points_camera_m, diagnostics)


def _cad_surface_points(mesh_path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"CAD mesh is not a non-empty triangle mesh: {mesh_path}")
    # ``sample_surface_even`` can legitimately return fewer points on this
    # thin-leg mesh.  Uniform surface-area sampling retains the actual CAD
    # distribution and guarantees the requested fixed evaluation budget.
    points, _ = trimesh.sample.sample_surface(mesh, CAD_SAMPLE_COUNT, seed=0)
    if points.shape != (CAD_SAMPLE_COUNT, 3) or not np.isfinite(points).all():
        raise RuntimeError("CAD surface sampling failed")
    return np.asarray(points, dtype=np.float64)


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper rigid transform mapping paired source points to target."""

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def _icp(cad_points: np.ndarray, observed_root_points: np.ndarray, initial_pose: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Rigid CAD-to-observation ICP with trimming and explicit failure states."""

    if len(observed_root_points) < MIN_CORRESPONDENCES:
        return None, {"reason": "insufficient_stereo_points", "point_count": int(len(observed_root_points))}
    tree = cKDTree(observed_root_points)
    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    residuals = np.empty(0, dtype=np.float64)
    correspondences = 0
    for iteration in range(ICP_ITERATIONS):
        transformed = _apply(cad_points, pose)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        threshold = MAX_CORRESPONDENCE_M if iteration < 4 else 0.030
        keep = distances <= threshold
        if int(np.count_nonzero(keep)) < MIN_CORRESPONDENCES:
            return None, {
                "reason": "insufficient_cad_correspondence",
                "iteration": iteration,
                "correspondences": int(np.count_nonzero(keep)),
            }
        source = transformed[keep]
        target = observed_root_points[indices[keep]]
        delta = _rigid_transform(source, target)
        pose = delta @ pose
        residuals = distances[keep]
        correspondences = len(residuals)
        if float(np.linalg.norm(delta[:3, 3])) < 0.0005 and _rotation_error_deg(np.eye(4), delta) < 0.10:
            break
    support = correspondences / float(len(cad_points))
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) else math.inf
    diagnostics = {
        "iterations": iteration + 1,
        "correspondences": correspondences,
        "model_support": support,
        "trimmed_rmse_m": rmse,
        "trimmed_p95_m": float(np.quantile(residuals, 0.95)) if len(residuals) else None,
    }
    if support < MIN_MODEL_SUPPORT or rmse > MAX_ACCEPTED_RMSE_M:
        diagnostics["reason"] = "weak_or_inaccurate_cad_registration"
        return None, diagnostics
    return pose, diagnostics


def _draw_debug(image_bgr: np.ndarray, root_from_camera: np.ndarray, root_from_table: np.ndarray, stereo: HeadStereoCalibration, cad_points: np.ndarray) -> np.ndarray:
    camera_from_root = np.linalg.inv(root_from_camera)
    points_camera = _apply(_apply(cad_points[::10], root_from_table), camera_from_root)
    valid = points_camera[:, 2] > 0.05
    points = points_camera[valid]
    if not len(points):
        return image_bgr
    pixels, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        stereo.left_intrinsic,
        stereo.left_distortion,
    )
    debug = image_bgr.copy()
    for x, y in np.rint(pixels.reshape(-1, 2)).astype(np.int32):
        if 0 <= x < debug.shape[1] and 0 <= y < debug.shape[0]:
            cv2.circle(debug, (int(x), int(y)), 1, (0, 255, 0), -1)
    return debug


def _frame_indices(start: int, stop: int, stride: int) -> tuple[int, ...]:
    if start < 0 or stop < start or stride < 1:
        raise ValueError("invalid source frame range")
    return tuple(range(start, stop + 1, stride))


def track(
    *,
    source_root: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    stride: int,
    urdf: Path,
    stereo_calibration: Path,
    initial_alignment: Path,
    cad_mesh: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    frames = _frame_indices(start_frame, end_frame, stride)
    source = SourceDatasetIndex(source_root)
    episode = source.episode(episode_index)
    if frames[-1] >= episode.frame_count:
        raise ValueError(f"frame {frames[-1]} exceeds source episode length {episode.frame_count}")
    alignment = json.loads(initial_alignment.read_text(encoding="utf-8"))
    alignment_source = alignment.get("source")
    if not isinstance(alignment_source, dict) or alignment_source.get("episode_index") != episode_index:
        raise ValueError("initial alignment must belong to the tracked source episode")
    initial_pose = _matrix(alignment.get("fixed_scene_root_from_table"), "fixed_scene_root_from_table")
    stereo = HeadStereoCalibration.load(stereo_calibration)
    config = load_pipeline_config(config_path)
    left_camera = next(item for item in config.cameras if item.source_key == "observation.images.cam_0")
    rows = _load_rows(episode.data_path, episode_index, frames)
    pin, model, data, joint_indices, torso_id = _build_torso_fk(urdf)
    cad_points = _cad_surface_points(cad_mesh)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_video = episode.video_slice("observation.images.cam_0")
    right_video = episode.video_slice("observation.images.cam_1")
    last_observed_pose = initial_pose
    records: list[dict[str, Any]] = []

    for frame_index in frames:
        row = rows[frame_index]
        q_source = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        if q_source.shape != (36,) or not np.isfinite(q_source).all():
            raise ValueError(f"source frame {frame_index} has invalid robot_q_current")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = q_source[7:]
        pin.framesForwardKinematics(model, data, q)
        torso = _transform(data.oMf[torso_id].rotation, data.oMf[torso_id].translation)
        root_from_left, _root_from_right = _root_from_head_eyes(torso, left_camera, stereo)
        raw_dir = output_dir / "raw"
        left_path = raw_dir / f"frame_{frame_index:06d}_head_left.png"
        right_path = raw_dir / f"frame_{frame_index:06d}_head_right.png"
        extract_video_frame(left_video.path, left_video.timestamp_for_frame(frame_index, config.source.fps, episode.frame_count), left_path)
        extract_video_frame(right_video.path, right_video.timestamp_for_frame(frame_index, config.source.fps, episode.frame_count), right_path)
        left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None or left_bgr.shape != (480, 640, 3) or right_bgr.shape != (480, 640, 3):
            raise RuntimeError(f"could not decode calibrated stereo RGB for frame {frame_index}")
        cloud = _stereo_cloud(
            cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB),
            stereo,
            seed=episode_index * 1_000_003 + frame_index,
            cad_points=cad_points,
            predicted_root_from_table=last_observed_pose,
            root_from_left_camera=root_from_left,
        )
        points_root = _apply(cloud.points_camera_m, root_from_left)
        pose, fit = _icp(cad_points, points_root, last_observed_pose)
        record: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp_s": float(row["timestamp"]),
            "root_from_opencv_camera": root_from_left.tolist(),
            "stereo": cloud.diagnostics,
            "registration": fit,
        }
        if pose is None:
            record["state"] = "unobserved"
        else:
            pose = _canonical_pose(pose, last_observed_pose)
            record["state"] = "observed"
            record["root_from_table"] = pose.tolist()
            debug = _draw_debug(left_bgr, root_from_left, pose, stereo, cad_points)
            debug_path = output_dir / "debug" / f"frame_{frame_index:06d}_head_left_cad.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(debug_path), debug):
                raise OSError(f"could not write {debug_path}")
            record["debug"] = str(debug_path.relative_to(output_dir))
            last_observed_pose = pose
        records.append(record)

    observed = [record for record in records if record["state"] == "observed"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera/contact calibration evidence only",
        "source": {
            "dataset_root": str(source_root),
            "episode_index": episode_index,
            "frames": list(frames),
            "frame_stride": stride,
        },
        "cad": {"mesh": str(cad_mesh), "sample_count": CAD_SAMPLE_COUNT},
        "initialization": {
            "initial_alignment": str(initial_alignment),
            "initial_root_from_table": initial_pose.tolist(),
        },
        "method": {
            "description": "head-stereo RGB point cloud plus encoder FK, robust CAD-surface ICP",
            "uses_simulator_ground_truth": False,
            "unobserved_frames_are_interpolated": False,
            "mass_or_contact_input": False,
        },
        "records": records,
        "summary": {
            "frames": len(records),
            "observed_frames": len(observed),
            "observed_fraction": len(observed) / float(len(records)),
            "accepted_for_table_motion_metric": False,
            "reason": (
                "stereo-CAD registration is an observation candidate, but no independent "
                "metric uncertainty bound yet proves the 20 mm / 3 degree release gate"
            ),
        },
        "measurement_uncertainty": {
            "independent_metric_bound": {
                "status": "unavailable",
                "passed": False,
                "reason": (
                    "stereo-CAD ICP residuals and resampling agreement are internal fit "
                    "diagnostics, not an independent real-world table-pose reference"
                ),
            }
        },
        "not_identified": [
            "table pose during unobserved frames",
            "contact force, friction, restitution, stiffness, damping",
            "release-gate trajectory precision without independent metric validation",
        ],
    }
    atomic_write_json(output_dir / "temporal_cad_tracker.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--stereo-calibration", type=Path, required=True)
    parser.add_argument("--initial-alignment", type=Path, required=True)
    parser.add_argument("--cad-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = track(
        source_root=args.source_root.expanduser().resolve(),
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        urdf=args.urdf.expanduser().resolve(),
        stereo_calibration=args.stereo_calibration.expanduser().resolve(),
        initial_alignment=args.initial_alignment.expanduser().resolve(),
        cad_mesh=args.cad_mesh.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
    )
    print(json.dumps({"summary": report["summary"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
