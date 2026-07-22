#!/usr/bin/env python3
"""Score a head-mount correction from real arm motion, independent of the table.

The source camera sees both G1 arms.  For an early, pre-manipulation interval,
the observed image difference is therefore independent evidence for the
head-camera pose that does not require a table corner, table pose, simulator
state, or a rendered scene.  This tool is deliberately diagnostic: it may
propose an episode-local correction, but it never writes simulator defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, CameraConfig, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_dataset import (
    SourceDatasetIndex,
    extract_video_frame,
)


SCHEMA_VERSION = "team_ramen_flip_table_head_arm_motion_alignment/v1"
DEFAULT_URDF = Path(
    "/workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/"
    "g1_29dof_with_hand.urdf"
)
_MOTION_DISTANCE_CAP_PX = 32.0
_MOTION_SUPPORT_DISTANCE_PX = 8.0
ARM_FRAME_NAMES = {
    side: tuple(
        f"{side}_{joint}_link"
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
    for side in ("left", "right")
}


@dataclass(frozen=True)
class FrameState:
    index: int
    robot_q_current: np.ndarray
    image_bgr: np.ndarray


def motion_support_mask(before_bgr: np.ndarray, after_bgr: np.ndarray) -> np.ndarray:
    """Return robust, dilated motion evidence from two same-camera RGB frames."""

    before = np.asarray(before_bgr)
    after = np.asarray(after_bgr)
    if before.shape != (480, 640, 3) or after.shape != before.shape:
        raise ValueError("motion images must be matching 640x480 BGR frames")
    difference = cv2.absdiff(
        cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    )
    blurred = cv2.GaussianBlur(difference, (5, 5), 0.0)
    threshold = max(18.0, float(np.quantile(blurred, 0.92)))
    changed = blurred >= threshold
    edges = cv2.Canny(blurred, max(8, int(threshold * 0.4)), max(16, int(threshold))) > 0
    mask = (changed | edges).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8), iterations=1).astype(bool)


def sample_projected_segments(pixels_by_side: dict[str, np.ndarray]) -> np.ndarray:
    """Sample FK arm links densely enough for a distance-transform score."""

    samples: list[np.ndarray] = []
    for pixels in pixels_by_side.values():
        for start, end in zip(pixels[:-1], pixels[1:], strict=True):
            if not np.isfinite(start).all() or not np.isfinite(end).all():
                continue
            count = max(2, int(np.ceil(np.linalg.norm(end - start) / 2.0)))
            samples.append(np.linspace(start, end, count, endpoint=True))
    if not samples:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(samples, axis=0)


def motion_distance_score(mask: np.ndarray, predicted_pixels: np.ndarray) -> tuple[float, float, int]:
    """Return mean edge distance, support fraction, and in-frame FK samples."""

    evidence = np.asarray(mask, dtype=bool)
    points = np.asarray(predicted_pixels, dtype=np.float64)
    if evidence.shape != (480, 640) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("motion mask or projected point shape is invalid")
    in_frame = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < evidence.shape[1])
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < evidence.shape[0])
    )
    if not in_frame.any() or not evidence.any():
        return _MOTION_DISTANCE_CAP_PX, 0.0, int(in_frame.sum())
    distance = cv2.distanceTransform((~evidence).astype(np.uint8), cv2.DIST_L2, 3)
    rounded = np.rint(points[in_frame]).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, evidence.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, evidence.shape[0] - 1)
    values = np.minimum(distance[rounded[:, 1], rounded[:, 0]], _MOTION_DISTANCE_CAP_PX)
    return float(np.mean(values)), float(np.mean(values <= _MOTION_SUPPORT_DISTANCE_PX)), int(len(values))


def corrected_root_from_camera(
    root_from_torso: np.ndarray,
    camera: CameraConfig,
    torso_offset_m: np.ndarray,
    camera_rotation_rpy_deg: np.ndarray,
) -> np.ndarray:
    """Apply an offline correction at the camera centre without changing FK."""

    torso = np.asarray(root_from_torso, dtype=np.float64)
    offset = np.asarray(torso_offset_m, dtype=np.float64)
    rpy = np.asarray(camera_rotation_rpy_deg, dtype=np.float64)
    if torso.shape != (4, 4) or offset.shape != (3,) or rpy.shape != (3,):
        raise ValueError("head correction inputs have incompatible shapes")
    from scipy.spatial.transform import Rotation
    from data.flip_table_data_augmentation.source_camera_projection import root_from_camera

    torso_from_camera = np.linalg.inv(torso) @ root_from_camera(torso, camera)
    torso_from_camera[:3, 3] += offset
    torso_from_camera[:3, :3] = torso_from_camera[:3, :3] @ Rotation.from_euler(
        "XYZ", rpy, degrees=True
    ).as_matrix()
    return torso @ torso_from_camera


class ArmProjector:
    """FK projector for the physical G1 arm links in a source head camera."""

    def __init__(self, urdf: Path, camera: CameraConfig) -> None:
        import pinocchio as pin
        from data.flip_table_data_augmentation.source_camera_projection import project_root_points

        self._pin = pin
        self._project_root_points = project_root_points
        self._camera = camera
        self._model = pin.buildModelFromUrdf(str(urdf))
        self._data = self._model.createData()
        required = ("torso_link",) + tuple(
            frame for names in ARM_FRAME_NAMES.values() for frame in names
        )
        missing = [frame for frame in required if not self._model.existFrame(frame)]
        if missing:
            raise ValueError(f"URDF lacks required G1 links: {missing}")
        self._joint_indices = np.asarray(
            [self._model.joints[self._model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
            dtype=np.int64,
        )
        self._torso_id = self._model.getFrameId("torso_link")
        self._arm_ids = {
            side: tuple(self._model.getFrameId(frame) for frame in names)
            for side, names in ARM_FRAME_NAMES.items()
        }

    def project(
        self,
        robot_q_current: np.ndarray,
        torso_offset_m: np.ndarray,
        camera_rotation_rpy_deg: np.ndarray,
    ) -> dict[str, np.ndarray]:
        robot_q = np.asarray(robot_q_current, dtype=np.float64)
        if robot_q.shape != (36,) or not np.isfinite(robot_q).all():
            raise ValueError("robot_q_current must be a finite 36D vector")
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[self._joint_indices] = robot_q[7:]
        self._pin.framesForwardKinematics(self._model, self._data, q)
        root_from_torso = np.asarray(self._data.oMf[self._torso_id].homogeneous, dtype=np.float64)
        root_from_corrected_camera = corrected_root_from_camera(
            root_from_torso, self._camera, torso_offset_m, camera_rotation_rpy_deg
        )
        output = {}
        for side, frame_ids in self._arm_ids.items():
            points = np.stack(
                [np.asarray(self._data.oMf[frame_id].translation) for frame_id in frame_ids]
            )
            output[side], _ = self._project_root_points(
                points, root_from_corrected_camera, self._camera
            )
        return output


def _score_candidate(
    projector: ArmProjector,
    states: tuple[FrameState, ...],
    correction: np.ndarray,
) -> dict[str, float | int]:
    offset, rpy = correction[:3], correction[3:]
    distances: list[float] = []
    supports: list[float] = []
    projected_count = 0
    for before, after in zip(states[:-1], states[1:], strict=True):
        mask = motion_support_mask(before.image_bgr, after.image_bgr)
        before_pixels = sample_projected_segments(projector.project(before.robot_q_current, offset, rpy))
        after_pixels = sample_projected_segments(projector.project(after.robot_q_current, offset, rpy))
        distance, support, count = motion_distance_score(mask, np.concatenate((before_pixels, after_pixels)))
        distances.append(distance)
        supports.append(support)
        projected_count += count
    if not distances:
        raise ValueError("at least two states are required")
    return {
        "mean_distance_px": float(np.mean(distances)),
        "p95_distance_px": float(np.quantile(distances, 0.95)),
        "support_fraction": float(np.mean(supports)),
        "projected_sample_count": projected_count,
    }


def _load_states(
    source_root: Path,
    episode_index: int,
    frames: tuple[int, ...],
    camera: CameraConfig,
    image_root: Path,
    extracted_image_root: Path | None,
) -> tuple[FrameState, ...]:
    episode = SourceDatasetIndex(source_root).episode(episode_index)
    if frames[0] < 0 or frames[-1] >= episode.frame_count:
        raise ValueError(f"frames must be inside source episode [0, {episode.frame_count})")
    rows = pq.read_table(
        episode.data_path,
        columns=["frame_index", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    q_by_frame = {
        int(row["frame_index"]): np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        for row in rows
    }
    video = episode.video_slice(camera.source_key) if extracted_image_root is None else None
    states = []
    for frame in frames:
        if frame not in q_by_frame:
            raise ValueError(f"source q_current is missing frame {frame}")
        image_path = (
            extracted_image_root / f"frame_{frame:06d}.png"
            if extracted_image_root is not None
            else image_root / f"frame_{frame:06d}.png"
        )
        if extracted_image_root is None:
            assert video is not None
            extract_video_frame(
                video.path,
                video.timestamp_for_frame(frame, 30, episode.frame_count),
                image_path,
            )
        elif not image_path.is_file():
            raise FileNotFoundError(f"pre-extracted source image is missing: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (camera.height, camera.width, 3):
            raise ValueError(f"source image is not {camera.width}x{camera.height}: {image_path}")
        states.append(FrameState(frame, q_by_frame[frame], image))
    return tuple(states)


def _overlay(image: np.ndarray, pixels: dict[str, np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for chain in pixels.values():
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            cv2.line(
                output,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def _parse_frames(values: Iterable[int]) -> tuple[int, ...]:
    frames = tuple(values)
    if len(frames) < 2 or frames != tuple(sorted(set(frames))):
        raise ValueError("--frames must contain at least two sorted, unique frame indices")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="directory containing pre-extracted frame_XXXXXX.png images; avoids a local ffmpeg dependency",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=12)
    args = parser.parse_args()
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive")

    frames = _parse_frames(args.frames)
    config = load_pipeline_config(args.config)
    camera = config.cameras[0]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    states = _load_states(
        args.source_root.expanduser().resolve(),
        args.episode_index,
        frames,
        camera,
        output / "source_frames",
        args.image_root.expanduser().resolve() if args.image_root is not None else None,
    )
    projector = ArmProjector(args.urdf.expanduser().resolve(), camera)
    nominal = np.zeros(6, dtype=np.float64)
    nominal_score = _score_candidate(projector, states, nominal)

    def objective(value: np.ndarray) -> float:
        score = _score_candidate(projector, states, np.asarray(value, dtype=np.float64))
        # A small support regularizer prevents a low score from a mostly off-frame chain.
        return float(score["mean_distance_px"]) + 8.0 * (1.0 - float(score["support_fraction"]))

    from scipy.optimize import differential_evolution

    result = differential_evolution(
        objective,
        bounds=((-0.03, 0.03), (-0.03, 0.03), (-0.03, 0.03), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)),
        maxiter=args.maxiter,
        popsize=8,
        polish=True,
        seed=args.seed,
        updating="deferred",
        workers=1,
    )
    optimized = np.asarray(result.x, dtype=np.float64)
    optimized_score = _score_candidate(projector, states, optimized)
    improvement = 1.0 - float(optimized_score["mean_distance_px"]) / max(
        float(nominal_score["mean_distance_px"]), 1.0e-9
    )
    first = states[0]
    overlay = _overlay(first.image_bgr, projector.project(first.robot_q_current, nominal[:3], nominal[3:]), (0, 255, 255))
    overlay = _overlay(overlay, projector.project(first.robot_q_current, optimized[:3], optimized[3:]), (255, 0, 255))
    cv2.imwrite(str(output / "projection_overlay_frame_000000.png"), overlay)
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera-identifiability diagnostic only",
        "source_episode_index": args.episode_index,
        "source_frames": list(frames),
        "camera_source_key": camera.source_key,
        "optimization": {
            "method": "differential_evolution on real RGB inter-frame motion support",
            "translation_bounds_m": [-0.03, 0.03],
            "rotation_bounds_deg": [-4.0, 4.0],
            "maxiter": args.maxiter,
            "seed": args.seed,
            "success": bool(result.success),
            "message": str(result.message),
        },
        "nominal": {"torso_offset_m": nominal[:3].tolist(), "camera_rotation_rpy_deg": nominal[3:].tolist(), **nominal_score},
        "optimized": {"torso_offset_m": optimized[:3].tolist(), "camera_rotation_rpy_deg": optimized[3:].tolist(), **optimized_score},
        "relative_mean_distance_improvement": improvement,
        "decision": "diagnostic_only_requires_cross_episode_consensus_and_heldout_rgb",
        "limitations": [
            "Motion support can contain table or exposure changes, so this tool cannot alone accept a camera mount.",
            "No table pose, simulator ground truth, or policy-only signal is used in this score.",
            "The output correction is expressed at the source camera centre and is not a RoboFinals rig parameter until separately converted and verified.",
        ],
    }
    atomic_write_json(output / "report.json", report)
    print(json.dumps({"episode": args.episode_index, "nominal": nominal_score, "optimized": optimized_score, "improvement": improvement}, indent=2))


if __name__ == "__main__":
    main()
