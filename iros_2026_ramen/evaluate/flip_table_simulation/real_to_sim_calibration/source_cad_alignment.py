#!/usr/bin/env python3
"""Fit the fixed pre-flip scene to recorded RGB with the V1 table CAD.

This is an *offline calibration* program. It uses the recorded stereo pair,
robot encoders, the audited head mount, and the exact RoboFinals V1 assembled
table geometry. It deliberately does not require four simultaneously visible
tabletop corners: the CAD rim and four leg axes are registered directly to
image edges/white-pixel support. No output of this module is a policy input.

The result is a fixed-scene proposal, not a claim that contact physics has
been identified. It reports left/right disagreement and temporal spread so a
bad visual fit cannot silently become a simulator reset value.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, CameraConfig, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_camera_projection import root_from_camera
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex, extract_video_frame
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    CameraCalibration,
    TabletopPoseEstimator,
)
from evaluate.flip_table_simulation.real_to_sim_calibration.stereo_geometry import HeadStereoCalibration


SCHEMA_VERSION = "team_ramen_flip_table_source_cad_alignment/v1"
OPENGL_FROM_OPENCV = np.diag((1.0, -1.0, -1.0, 1.0))
TABLE_YAW_180_SYMMETRY = np.diag((-1.0, -1.0, 1.0, 1.0))
MINIMUM_EYE_CONFIDENCE = 0.25
MAXIMUM_EYE_CAD_EDGE_ERROR_PX = 12.0
MAXIMUM_STEREO_TRANSLATION_M = 0.050
MAXIMUM_STEREO_ROTATION_DEG = 6.0


@dataclass(frozen=True)
class PoseEstimate:
    frame_index: int
    camera: str
    root_from_table: np.ndarray
    confidence: float
    cad_edge_error_px: float


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _camera_config(intrinsic: np.ndarray, distortion: np.ndarray, root_from_opencv_camera: np.ndarray) -> CameraCalibration:
    return CameraCalibration(
        intrinsic_matrix=np.asarray(intrinsic, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
        root_from_camera=np.asarray(root_from_opencv_camera, dtype=np.float64),
    )


def _right_from_left(stereo: HeadStereoCalibration) -> np.ndarray:
    """Return the calibrated right-from-left OpenCV transform in metres."""

    return _transform(stereo.rotation_right_from_left, stereo.translation_right_from_left_mm.reshape(3) / 1000.0)


def _root_from_head_eyes(root_from_torso: np.ndarray, left_camera: CameraConfig, stereo: HeadStereoCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Compose FK with raw left-eye mount and measured head-stereo geometry."""

    root_from_left_opengl = root_from_camera(root_from_torso, left_camera)
    root_from_left_opencv = root_from_left_opengl @ OPENGL_FROM_OPENCV
    root_from_right_opencv = root_from_left_opencv @ np.linalg.inv(_right_from_left(stereo))
    return root_from_left_opencv, root_from_right_opencv


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(np.asarray(first).T @ np.asarray(second)).magnitude()))


def _canonical_table_pose(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resolve the physical 180-degree yaw symmetry of the UTTER table."""

    direct = np.asarray(candidate, dtype=np.float64)
    rotated = direct @ TABLE_YAW_180_SYMMETRY
    direct_error = _rotation_distance_deg(reference[:3, :3], direct[:3, :3])
    rotated_error = _rotation_distance_deg(reference[:3, :3], rotated[:3, :3])
    return direct if direct_error <= rotated_error else rotated


def robust_fixed_pose(estimates: Iterable[PoseEstimate]) -> tuple[np.ndarray, dict[str, Any]]:
    """Robustly aggregate CAD fits, rejecting only inconsistent whole frames."""

    values = tuple(estimates)
    if len(values) < 3:
        raise ValueError("at least three accepted CAD estimates are required")
    reference = values[0].root_from_table
    canonical = tuple(_canonical_table_pose(item.root_from_table, reference) for item in values)
    translations = np.stack([item[:3, 3] for item in canonical])
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in canonical]))
    center = np.median(translations, axis=0)
    mean_rotation = rotations.mean().as_matrix()
    translation_error = np.linalg.norm(translations - center, axis=1)
    rotation_error = np.asarray([_rotation_distance_deg(mean_rotation, item[:3, :3]) for item in canonical])
    keep = (translation_error <= max(0.035, 2.5 * np.median(translation_error) + 1.0e-6)) & (
        rotation_error <= max(6.0, 2.5 * np.median(rotation_error) + 1.0e-6)
    )
    if int(keep.sum()) < 3:
        raise ValueError("CAD fits have fewer than three temporally consistent frames")
    kept_translations = translations[keep]
    kept_rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item, accepted in zip(canonical, keep, strict=True) if accepted]))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = kept_rotations.mean().as_matrix()
    result[:3, 3] = np.median(kept_translations, axis=0)
    return result, {
        "input_estimates": len(values),
        "accepted_estimates": int(keep.sum()),
        "translation_spread_p95_m": float(np.quantile(np.linalg.norm(kept_translations - result[:3, 3], axis=1), 0.95)),
        "rotation_spread_p95_deg": float(np.quantile([_rotation_distance_deg(result[:3, :3], rotation.as_matrix()) for rotation in kept_rotations], 0.95)),
        "accepted_indices": np.flatnonzero(keep).astype(int).tolist(),
        "yaw_180_symmetry_canonicalized": True,
    }


def _load_rows(episode_path: Path, episode_index: int, frames: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(
        episode_path,
        columns=["frame_index", "timestamp", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    result = {int(row["frame_index"]): row for row in rows if int(row["frame_index"]) in frames}
    missing = sorted(set(frames) - set(result))
    if missing:
        raise ValueError(f"source episode lacks requested frames: {missing}")
    return result


def _build_torso_fk(urdf: Path):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    if not model.existFrame("torso_link"):
        raise ValueError("source FK URDF lacks torso_link")
    missing = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    if missing:
        raise ValueError(f"source FK URDF lacks joints: {missing}")
    joint_indices = np.asarray([model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER], dtype=np.int64)
    return pin, model, model.createData(), joint_indices, int(model.getFrameId("torso_link"))


def _estimate_eye(rgb_bgr: np.ndarray, calibration: CameraCalibration, *, frame_index: int, camera: str) -> tuple[PoseEstimate, np.ndarray]:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    estimate = TabletopPoseEstimator(calibration).estimate(rgb)
    return PoseEstimate(frame_index, camera, estimate.root_from_table, estimate.confidence, estimate.reprojection_error_px), TabletopPoseEstimator.render_debug(rgb, estimate)


def _parse_frames(values: list[int]) -> tuple[int, ...]:
    frames = tuple(sorted(set(int(value) for value in values)))
    if len(frames) < 3 or frames[0] < 0:
        raise ValueError("--frames must contain at least three non-negative frame indices")
    return frames


def align_source_episode(*, source_root: Path, episode_index: int, frames: tuple[int, ...], urdf: Path, stereo_calibration: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Fit a fixed pre-flip table pose using partial CAD edge evidence."""

    config = load_pipeline_config(config_path)
    left_camera = next(item for item in config.cameras if item.source_key == "observation.images.cam_0")
    source = SourceDatasetIndex(source_root)
    episode = source.episode(episode_index)
    if frames[-1] >= episode.frame_count:
        raise ValueError(f"frame {frames[-1]} exceeds episode length {episode.frame_count}")
    stereo = HeadStereoCalibration.load(stereo_calibration)
    rows = _load_rows(episode.data_path, episode_index, frames)
    pin, model, data, joint_indices, torso_id = _build_torso_fk(urdf)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_video = episode.video_slice("observation.images.cam_0")
    right_video = episode.video_slice("observation.images.cam_1")
    accepted: list[PoseEstimate] = []
    stereo_accepted: list[PoseEstimate] = []
    records: list[dict[str, Any]] = []
    pair_translation_errors: list[float] = []
    pair_rotation_errors: list[float] = []

    for frame_index in frames:
        row = rows[frame_index]
        q_source = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        if q_source.shape != (36,) or not np.isfinite(q_source).all():
            raise ValueError(f"frame {frame_index} has invalid robot_q_current")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = q_source[7:]
        pin.framesForwardKinematics(model, data, q)
        torso = _transform(data.oMf[torso_id].rotation, data.oMf[torso_id].translation)
        root_from_left, root_from_right = _root_from_head_eyes(torso, left_camera, stereo)
        paths = {"head_left": output_dir / "raw" / f"frame_{frame_index:06d}_head_left.png", "head_right": output_dir / "raw" / f"frame_{frame_index:06d}_head_right.png"}
        extract_video_frame(left_video.path, left_video.timestamp_for_frame(frame_index, config.source.fps, episode.frame_count), paths["head_left"])
        extract_video_frame(right_video.path, right_video.timestamp_for_frame(frame_index, config.source.fps, episode.frame_count), paths["head_right"])
        images = {name: cv2.imread(str(path), cv2.IMREAD_COLOR) for name, path in paths.items()}
        if any(image is None or image.shape != (480, 640, 3) for image in images.values()):
            raise RuntimeError(f"could not decode stereo RGB for frame {frame_index}")
        eye_specs = (
            ("head_left", images["head_left"], _camera_config(stereo.left_intrinsic, stereo.left_distortion, root_from_left)),
            ("head_right", images["head_right"], _camera_config(stereo.right_intrinsic, stereo.right_distortion, root_from_right)),
        )
        frame_estimates: list[PoseEstimate] = []
        frame_record: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp_s": float(row["timestamp"]),
            # These FK poses are offline calibration evidence. They make the
            # source and V1 head-camera frames directly comparable without
            # deriving a camera correction from a rendered image alone.
            "root_from_opencv_camera": {
                "head_left": root_from_left.tolist(),
                "head_right": root_from_right.tolist(),
            },
            "eyes": {},
        }
        for eye_name, image, calibration in eye_specs:
            try:
                estimate, debug = _estimate_eye(image, calibration, frame_index=frame_index, camera=eye_name)
            except ValueError as exc:
                frame_record["eyes"][eye_name] = {"accepted": False, "reason": str(exc)}
                continue
            debug_path = output_dir / "debug" / f"frame_{frame_index:06d}_{eye_name}.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(debug_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)):
                raise OSError(f"could not write {debug_path}")
            accepted.append(estimate)
            frame_estimates.append(estimate)
            # A monocular local edge minimum can be an arm link or a single
            # visible leg. It becomes an accepted table pose only after the
            # paired-eye geometric check below; retain the raw candidate for
            # audit/debug but do not let it masquerade as an observation.
            frame_record["eyes"][eye_name] = {
                "accepted": False,
                "candidate": True,
                "confidence": estimate.confidence,
                "cad_edge_error_px": estimate.cad_edge_error_px,
                "root_from_table": estimate.root_from_table.tolist(),
                "debug": str(debug_path.relative_to(output_dir)),
            }
        if len(frame_estimates) == 2:
            first, second = frame_estimates
            second_pose = _canonical_table_pose(second.root_from_table, first.root_from_table)
            translation = float(np.linalg.norm(first.root_from_table[:3, 3] - second_pose[:3, 3]))
            rotation = _rotation_distance_deg(first.root_from_table[:3, :3], second_pose[:3, :3])
            pair_translation_errors.append(translation)
            pair_rotation_errors.append(rotation)
            pair_accepted = bool(
                min(first.confidence, second.confidence) >= MINIMUM_EYE_CONFIDENCE
                and max(first.cad_edge_error_px, second.cad_edge_error_px) <= MAXIMUM_EYE_CAD_EDGE_ERROR_PX
                and translation <= MAXIMUM_STEREO_TRANSLATION_M
                and rotation <= MAXIMUM_STEREO_ROTATION_DEG
            )
            if pair_accepted:
                stereo_accepted.extend((first, second))
                frame_record["eyes"]["head_left"]["accepted"] = True
                frame_record["eyes"]["head_right"]["accepted"] = True
            frame_record["stereo_agreement"] = {
                "translation_m": translation,
                "rotation_deg": rotation,
                "accepted": pair_accepted,
            }
        records.append(frame_record)

    # A pair can be geometrically implausible while still producing a local
    # edge minimum. Use only explicit two-eye passes for the fixed scene; the
    # all-eye collection remains in the report for auditability.
    # A single-eye CAD fit may lock onto an inner brace or an arm link. Retain
    # these candidates for audit, but never use them to seed a simulator reset.
    # Three independent stereo-consistent frames are the minimum evidence for
    # a fixed-scene proposal.
    if len(stereo_accepted) < 6:
        raise ValueError("fewer than three stereo-consistent CAD frame pairs")
    fixed_pose, temporal = robust_fixed_pose(stereo_accepted)
    accepted_pairs = [record["stereo_agreement"] for record in records if "stereo_agreement" in record and record["stereo_agreement"]["accepted"]]
    stereo_ok = len(accepted_pairs) >= 3
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline fixed-scene calibration only",
        "source": {"dataset_root": str(source_root), "episode_index": episode_index, "frames": list(frames)},
        "method": {"description": "CAD rim and leg-axis registration to RGB edge/white support; no four-corner PnP", "cad_mesh": "data/flip_table_data_augmentation/outputs/source/v1-table-mesh/Table001_assembled_body_frame.obj", "tabletop_bounds_m": [0.580, 0.420], "yaw_180_symmetry_canonicalized": True, "requires_simulator_ground_truth": False},
        "fixed_scene_root_from_table": fixed_pose.tolist(),
        "temporal_consistency": temporal,
        "stereo_agreement": {
            "paired_frames": len(pair_translation_errors),
            "accepted_paired_frames": len(accepted_pairs),
            "thresholds": {"minimum_eye_confidence": MINIMUM_EYE_CONFIDENCE, "maximum_eye_cad_edge_error_px": MAXIMUM_EYE_CAD_EDGE_ERROR_PX, "maximum_translation_m": MAXIMUM_STEREO_TRANSLATION_M, "maximum_rotation_deg": MAXIMUM_STEREO_ROTATION_DEG},
            "all_translation_median_m": None if not pair_translation_errors else float(np.median(pair_translation_errors)),
            "all_translation_p95_m": None if not pair_translation_errors else float(np.quantile(pair_translation_errors, 0.95)),
            "all_rotation_median_deg": None if not pair_rotation_errors else float(np.median(pair_rotation_errors)),
            "all_rotation_p95_deg": None if not pair_rotation_errors else float(np.quantile(pair_rotation_errors, 0.95)),
            "accepted_translation_p95_m": None if not accepted_pairs else float(np.quantile([item["translation_m"] for item in accepted_pairs], 0.95)),
            "accepted_rotation_p95_deg": None if not accepted_pairs else float(np.quantile([item["rotation_deg"] for item in accepted_pairs], 0.95)),
            "passes_internal_gate": stereo_ok,
        },
        "accepted_for_fixed_scene_proposal": bool(stereo_ok and temporal["translation_spread_p95_m"] <= 0.050 and temporal["rotation_spread_p95_deg"] <= 7.0),
        "not_identified": ["contact friction", "restitution", "Dex1 contact stiffness", "table trajectory during occlusion"],
        "frames": records,
    }
    atomic_write_json(output_dir / "source_cad_alignment.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--stereo-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = align_source_episode(source_root=args.source_root.expanduser().resolve(), episode_index=args.episode_index, frames=_parse_frames(args.frames), urdf=args.urdf.expanduser().resolve(), stereo_calibration=args.stereo_calibration.expanduser().resolve(), output_dir=args.output_dir.expanduser().resolve(), config_path=args.config.expanduser().resolve())
    print(json.dumps({"accepted": report["accepted_for_fixed_scene_proposal"], "stereo": report["stereo_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
