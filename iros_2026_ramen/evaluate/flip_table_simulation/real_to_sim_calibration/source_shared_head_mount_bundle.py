#!/usr/bin/env python3
"""Jointly fit one source head mount and fixed table poses from stereo CAD fits.

This offline diagnostic removes a weakness of per-frame CAD registration: a
camera correction must be shared across every accepted stereo observation,
while each source episode has one independently fixed pre-flip table pose.
It consumes only recorded RGB-derived CAD poses, encoder FK, and the measured
head stereo calibration. It never writes simulator defaults or policy inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.fk_audit import G1_BODY_JOINT_ORDER
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex

from .source_cad_alignment import TABLE_YAW_180_SYMMETRY, _right_from_left
from .stereo_geometry import HeadStereoCalibration


SCHEMA_VERSION = "team_ramen_flip_table_shared_head_mount_bundle/v1"
_TRANSLATION_SCALE_M = 0.020
_ROTATION_SCALE_RAD = np.deg2rad(3.0)
_MAX_MOUNT_TRANSLATION_CORRECTION_M = 0.030
_MAX_MOUNT_ROTATION_CORRECTION_RAD = np.deg2rad(4.0)


@dataclass(frozen=True)
class Observation:
    """One accepted RGB CAD pose expressed in the corresponding eye frame."""

    episode_index: int
    frame_index: int
    eye: str
    root_from_torso: np.ndarray
    left_from_eye: np.ndarray
    nominal_torso_from_left_eye: np.ndarray
    eye_from_table: np.ndarray


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return result


def _transform_from_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("pose vector must be finite [translation xyz, rotation rotvec]")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    result[:3, 3] = vector[:3]
    return result


def _vector_from_transform(value: np.ndarray) -> np.ndarray:
    transform = _matrix(value, "transform")
    return np.concatenate((transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()))


def _mean_transform(transforms: Iterable[np.ndarray]) -> np.ndarray:
    values = tuple(_matrix(item, "transform") for item in transforms)
    if not values:
        raise ValueError("at least one transform is required")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median(np.stack([item[:3, 3] for item in values]), axis=0)
    result[:3, :3] = Rotation.from_matrix(np.stack([item[:3, :3] for item in values])).mean().as_matrix()
    return result


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())


def _canonical_table_pose(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    direct = _matrix(value, "table pose")
    flipped = direct @ TABLE_YAW_180_SYMMETRY
    return direct if _rotation_distance(reference, direct) <= _rotation_distance(reference, flipped) else flipped


def _build_torso_fk(urdf: Path):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    missing = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    if missing or not model.existFrame("torso_link"):
        raise ValueError(f"source FK URDF lacks required links: {missing}")
    joint_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
        dtype=np.int64,
    )
    return pin, model, model.createData(), joint_indices, int(model.getFrameId("torso_link"))


def _torso_poses(source_root: Path, episode_index: int, frames: set[int], urdf: Path) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    episode = SourceDatasetIndex(source_root).episode(episode_index)
    rows = pq.read_table(
        episode.data_path,
        columns=["frame_index", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    pin, model, data, joint_indices, torso_id = _build_torso_fk(urdf)
    result: dict[int, np.ndarray] = {}
    for row in rows:
        frame_index = int(row["frame_index"])
        if frame_index not in frames:
            continue
        source_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        if source_q.shape != (36,) or not np.isfinite(source_q).all():
            raise ValueError(f"episode {episode_index} frame {frame_index} has invalid q_current")
        q = np.zeros(model.nq, dtype=np.float64)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = data.oMf[torso_id].rotation
        pose[:3, 3] = data.oMf[torso_id].translation
        result[frame_index] = pose
    missing = sorted(frames - set(result))
    if missing:
        raise ValueError(f"episode {episode_index} lacks requested FK frames: {missing}")
    return result


def _load_alignment(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "team_ramen_flip_table_source_cad_alignment/v1":
        raise ValueError(f"{path} is not a source CAD alignment report")
    if not document.get("accepted_for_fixed_scene_proposal", False):
        raise ValueError(f"{path} did not pass its internal stereo-CAD gate")
    source = document.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("episode_index"), int):
        raise ValueError(f"{path} has no source episode index")
    return document


def observations_from_alignments(
    reports: Iterable[dict[str, Any]], *, source_root: Path, urdf: Path, stereo: HeadStereoCalibration
) -> tuple[Observation, ...]:
    """Recover camera-frame table observations from immutable CAD reports."""

    documents = tuple(reports)
    frames_by_episode: dict[int, set[int]] = defaultdict(set)
    for document in documents:
        episode = int(document["source"]["episode_index"])
        for frame in document.get("frames", []):
            eyes = frame.get("eyes") if isinstance(frame, dict) else None
            if isinstance(eyes, dict) and all(isinstance(eyes.get(eye), dict) and eyes[eye].get("accepted") for eye in ("head_left", "head_right")):
                frames_by_episode[episode].add(int(frame["frame_index"]))
    torso_by_episode = {
        episode: _torso_poses(source_root, episode, frames, urdf)
        for episode, frames in frames_by_episode.items()
    }
    right_from_left = _right_from_left(stereo)
    observations: list[Observation] = []
    for document in documents:
        episode = int(document["source"]["episode_index"])
        reference = _matrix(document["fixed_scene_root_from_table"], "fixed scene table pose")
        for frame in document["frames"]:
            frame_index = int(frame["frame_index"])
            eyes = frame.get("eyes")
            camera_poses = frame.get("root_from_opencv_camera")
            if not isinstance(eyes, dict) or not isinstance(camera_poses, dict):
                continue
            if not all(isinstance(eyes.get(eye), dict) and eyes[eye].get("accepted") for eye in ("head_left", "head_right")):
                continue
            torso = torso_by_episode[episode][frame_index]
            for eye in ("head_left", "head_right"):
                root_from_eye = _matrix(camera_poses.get(eye), f"{eye} root camera pose")
                root_from_table = _canonical_table_pose(
                    _matrix(eyes[eye].get("root_from_table"), f"{eye} table pose"), reference
                )
                eye_from_table = np.linalg.inv(root_from_eye) @ root_from_table
                left_from_eye = (
                    np.eye(4, dtype=np.float64)
                    if eye == "head_left"
                    else np.linalg.inv(right_from_left)
                )
                nominal_torso_from_left_eye = (
                    np.linalg.inv(torso) @ root_from_eye @ np.linalg.inv(left_from_eye)
                )
                observations.append(
                    Observation(
                        episode,
                        frame_index,
                        eye,
                        torso,
                        left_from_eye,
                        nominal_torso_from_left_eye,
                        eye_from_table,
                    )
                )
    observed_episodes = {item.episode_index for item in observations}
    missing_episodes = sorted(set(frames_by_episode) - observed_episodes)
    if missing_episodes:
        raise ValueError(
            "alignment reports have no accepted stereo observations with recorded "
            f"root_from_opencv_camera for episodes: {missing_episodes}"
        )
    if len(observations) < 6:
        raise ValueError("fewer than three accepted stereo pairs are available")
    return tuple(observations)


def _residuals(
    parameters: np.ndarray,
    observations: tuple[Observation, ...],
    episodes: tuple[int, ...],
    initial_mount: np.ndarray,
) -> np.ndarray:
    mount = initial_mount @ _transform_from_vector(parameters[:6])
    tables = {episode: _transform_from_vector(parameters[6 + 6 * index : 12 + 6 * index]) for index, episode in enumerate(episodes)}
    residuals = []
    for observation in observations:
        predicted = observation.root_from_torso @ mount @ observation.left_from_eye @ observation.eye_from_table
        error = np.linalg.inv(tables[observation.episode_index]) @ predicted
        residuals.extend(error[:3, 3] / _TRANSLATION_SCALE_M)
        residuals.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec() / _ROTATION_SCALE_RAD)
    return np.asarray(residuals, dtype=np.float64)


def fit_shared_mount(observations: Iterable[Observation]) -> dict[str, Any]:
    """Fit a fixed source torso-to-left-eye transform and one table pose per episode."""

    values = tuple(observations)
    episodes = tuple(sorted({item.episode_index for item in values}))
    if len(episodes) < 2:
        raise ValueError("at least two independent episodes are required")
    initial_mount = _mean_transform(item.nominal_torso_from_left_eye for item in values)
    initial_tables = {
        episode: _mean_transform(
            item.root_from_torso @ initial_mount @ item.left_from_eye @ item.eye_from_table
            for item in values
            if item.episode_index == episode
        )
        for episode in episodes
    }
    initial = np.concatenate((np.zeros(6, dtype=np.float64), *(_vector_from_transform(initial_tables[episode]) for episode in episodes)))
    lower = np.concatenate(
        (
            np.asarray(
                (
                    -_MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    -_MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    -_MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    -_MAX_MOUNT_ROTATION_CORRECTION_RAD,
                    -_MAX_MOUNT_ROTATION_CORRECTION_RAD,
                    -_MAX_MOUNT_ROTATION_CORRECTION_RAD,
                ),
                dtype=np.float64,
            ),
            np.full(6 * len(episodes), -np.inf, dtype=np.float64),
        )
    )
    upper = np.concatenate(
        (
            np.asarray(
                (
                    _MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    _MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    _MAX_MOUNT_TRANSLATION_CORRECTION_M,
                    _MAX_MOUNT_ROTATION_CORRECTION_RAD,
                    _MAX_MOUNT_ROTATION_CORRECTION_RAD,
                    _MAX_MOUNT_ROTATION_CORRECTION_RAD,
                ),
                dtype=np.float64,
            ),
            np.full(6 * len(episodes), np.inf, dtype=np.float64),
        )
    )
    result = least_squares(
        _residuals,
        initial,
        args=(values, episodes, initial_mount),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=1000,
    )
    final_mount = initial_mount @ _transform_from_vector(result.x[:6])
    final_tables = {
        episode: _transform_from_vector(result.x[6 + 6 * index : 12 + 6 * index])
        for index, episode in enumerate(episodes)
    }

    def metrics(mount: np.ndarray, tables: dict[int, np.ndarray]) -> dict[str, Any]:
        per_episode: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in values:
            predicted = item.root_from_torso @ mount @ item.left_from_eye @ item.eye_from_table
            error = np.linalg.inv(tables[item.episode_index]) @ predicted
            per_episode[str(item.episode_index)].append((float(np.linalg.norm(error[:3, 3])), float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))))
        return {
            episode: {
                "sample_count": len(errors),
                "translation_median_m": float(np.median([error[0] for error in errors])),
                "translation_p95_m": float(np.quantile([error[0] for error in errors], 0.95)),
                "rotation_median_deg": float(np.median([error[1] for error in errors])),
                "rotation_p95_deg": float(np.quantile([error[1] for error in errors], 0.95)),
            }
            for episode, errors in per_episode.items()
        }

    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    return {
        "episodes": episodes,
        "initial_mount": initial_mount,
        "final_mount": final_mount,
        "initial_tables": initial_tables,
        "final_tables": final_tables,
        "initial_metrics": metrics(initial_mount, initial_tables),
        "final_metrics": metrics(final_mount, final_tables),
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "jacobian_rank": int(np.linalg.matrix_rank(result.jac)),
            "jacobian_singular_values": singular_values.tolist(),
            "mount_correction_bounds": {
                "translation_m": _MAX_MOUNT_TRANSLATION_CORRECTION_M,
                "rotation_deg": float(np.degrees(_MAX_MOUNT_ROTATION_CORRECTION_RAD)),
            },
            "mount_correction_at_bound": bool(
                np.any(np.isclose(result.x[:6], lower[:6], atol=1.0e-6))
                or np.any(np.isclose(result.x[:6], upper[:6], atol=1.0e-6))
            ),
        },
    }


def _pose_payload(value: np.ndarray) -> dict[str, list[float]]:
    return {
        "translation_m": value[:3, 3].tolist(),
        "rotation_rpy_deg": Rotation.from_matrix(value[:3, :3]).as_euler("XYZ", degrees=True).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--stereo-calibration", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [_load_alignment(path.expanduser().resolve()) for path in args.alignment_report]
    episodes = [int(report["source"]["episode_index"]) for report in reports]
    if len(set(episodes)) != len(episodes):
        raise ValueError("alignment reports must use distinct source episodes")
    result = fit_shared_mount(
        observations_from_alignments(
            reports,
            source_root=args.source_root.expanduser().resolve(),
            urdf=args.urdf.expanduser().resolve(),
            stereo=HeadStereoCalibration.load(args.stereo_calibration.expanduser().resolve()),
        )
    )
    initial = result["initial_mount"]
    final = result["final_mount"]
    correction = np.linalg.inv(initial) @ final
    at_bound = bool(result["optimization"]["mount_correction_at_bound"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline source-camera identifiability diagnostic only",
        "alignment_reports": [str(path.expanduser().resolve()) for path in args.alignment_report],
        "source_episode_indices": list(result["episodes"]),
        "shared_source_torso_from_left_eye": _pose_payload(final),
        "source_mount_delta_from_alignment_nominal": _pose_payload(correction),
        "episode_fixed_root_from_table": {str(episode): _pose_payload(pose) for episode, pose in result["final_tables"].items()},
        "metrics": {"initial": result["initial_metrics"], "final": result["final_metrics"]},
        "optimization": result["optimization"],
        "decision": (
            "rejected_mount_correction_at_bound"
            if at_bound
            else "diagnostic_only_requires_source_to_v1_rig_conversion_and_unused_episode_raw_RGB_validation"
        ),
        "accepted_for_source_to_v1_rig_conversion": not at_bound,
        "limitations": [
            "The CAD poses originate from RGB edge/support registration and retain its ambiguity.",
            "This fit does not identify contact parameters or validate a simulator camera.",
            "No result is applied to policy inputs, simulator defaults, or held-out tuning.",
        ],
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"episodes": episodes, "optimization": report["optimization"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
