"""Held-out FK audit for the source EEF state and action labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import sha256_file


FK_AUDIT_SCHEMA_VERSION = "team_ramen_flip_table_fk_audit/v1"
SYNTHETIC_ACTION_FK_SCHEMA_VERSION = "team_ramen_synthetic_action_fk_audit/v1"
G1_BODY_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
REQUIRED_COLUMNS = (
    "episode_index",
    "frame_index",
    "observation.state.ee_state",
    "action.ee_action",
    "observation.state.robot_q_current",
    "action.robot_q_desired",
)


@dataclass(frozen=True)
class Sample:
    episode_index: int
    frame_index: int
    ee_state: tuple[float, ...]
    ee_action: tuple[float, ...]
    robot_q_current: tuple[float, ...]
    robot_q_desired: tuple[float, ...]


def _vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    import math

    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{label} must contain {width} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def select_frame_indices(length: int, samples_per_episode: int) -> tuple[int, ...]:
    if length <= 0 or samples_per_episode <= 0:
        raise ValueError("length and samples_per_episode must be positive")
    count = min(length, samples_per_episode)
    if count == 1:
        return (length // 2,)
    return tuple(sorted({round(index * (length - 1) / (count - 1)) for index in range(count)}))


def load_stratified_samples(
    source_root: Path,
    *,
    samples_per_episode: int,
) -> tuple[Sample, ...]:
    """Read evenly spaced frames without loading all 290k rows into Python."""

    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    root = Path(source_root).resolve()
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not episode_files or not data_files:
        raise FileNotFoundError("source data or episode metadata is missing")
    episode_table = pq.read_table([str(path) for path in episode_files], columns=["episode_index", "length"])
    selected = {
        (int(episode), frame)
        for episode, length in zip(
            episode_table["episode_index"].to_pylist(),
            episode_table["length"].to_pylist(),
            strict=True,
        )
        for frame in select_frame_indices(int(length), samples_per_episode)
    }
    samples: list[Sample] = []
    scanner = pads.dataset([str(path) for path in data_files], format="parquet").scanner(
        columns=list(REQUIRED_COLUMNS),
        batch_size=8192,
    )
    for batch in scanner.to_batches():
        episodes = batch.column("episode_index").to_pylist()
        frames = batch.column("frame_index").to_pylist()
        rows = batch.to_pylist()
        for episode, frame, row in zip(episodes, frames, rows, strict=True):
            key = (int(episode), int(frame))
            if key not in selected:
                continue
            samples.append(
                Sample(
                    episode_index=key[0],
                    frame_index=key[1],
                    ee_state=_vector(row["observation.state.ee_state"], 12, "ee_state"),
                    ee_action=_vector(row["action.ee_action"], 12, "ee_action"),
                    robot_q_current=_vector(row["observation.state.robot_q_current"], 36, "robot_q_current"),
                    robot_q_desired=_vector(row["action.robot_q_desired"], 36, "robot_q_desired"),
                )
            )
    samples.sort(key=lambda sample: (sample.episode_index, sample.frame_index))
    if len(samples) != len(selected):
        found = {(sample.episode_index, sample.frame_index) for sample in samples}
        missing = sorted(selected - found)
        raise ValueError(f"failed to load {len(missing)} selected source frames: {missing[:10]}")
    return tuple(samples)
def _build_fk(urdf_path: Path, frame_names: dict[str, str]):
    import numpy as np
    import pinocchio as pin

    # This audit needs only rigid-body kinematics. Building visual/collision
    # geometry through RobotWrapper makes a valid URDF unusable whenever its
    # cosmetic STL package is not co-located with the file, despite meshes
    # having no effect on frame placements.
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    missing_joints = [name for name in G1_BODY_JOINT_ORDER if not model.existJointName(name)]
    missing_frames = [name for name in frame_names.values() if not model.existFrame(name)]
    if missing_joints or missing_frames:
        raise ValueError(f"URDF contract mismatch: missing_joints={missing_joints}, missing_frames={missing_frames}")
    joint_indices = np.asarray(
        [model.joints[model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
        dtype=np.int64,
    )
    frame_ids = {side: int(model.getFrameId(name)) for side, name in frame_names.items()}
    return pin, model, data, joint_indices, frame_ids


def _fk_placements(samples: Iterable[Sample], urdf_path: Path, frame_names: dict[str, str]):
    import numpy as np

    pin, model, data, joint_indices, frame_ids = _build_fk(urdf_path, frame_names)
    sample_values = tuple(samples)
    output: dict[str, dict[str, tuple[Any, Any]]] = {"state": {}, "action": {}}
    for label, attribute in (("state", "robot_q_current"), ("action", "robot_q_desired")):
        positions = {side: [] for side in frame_ids}
        rotations = {side: [] for side in frame_ids}
        for sample in sample_values:
            source_q = np.asarray(getattr(sample, attribute), dtype=np.float64)
            q = np.zeros(model.nq, dtype=np.float64)
            q[joint_indices] = source_q[7:]
            pin.framesForwardKinematics(model, data, q)
            for side, frame_id in frame_ids.items():
                positions[side].append(np.asarray(data.oMf[frame_id].translation).copy())
                rotations[side].append(np.asarray(data.oMf[frame_id].rotation).copy())
        for side in frame_ids:
            output[label][side] = (np.stack(positions[side]), np.stack(rotations[side]))
    return output


def _eef_targets(samples: tuple[Sample, ...], label: str, assignment: dict[str, int]):
    import numpy as np
    from scipy.spatial.transform import Rotation

    attribute = "ee_state" if label == "state" else "ee_action"
    values = np.asarray([getattr(sample, attribute) for sample in samples], dtype=np.float64)
    result = {}
    for side, start in assignment.items():
        positions = values[:, start : start + 3]
        rotations = Rotation.from_euler("xyz", values[:, start + 3 : start + 6]).as_matrix()
        result[side] = (positions, rotations)
    return result


def _fit_tool_transform(
    placements: dict[str, dict[str, tuple[Any, Any]]],
    targets: dict[str, dict[str, tuple[Any, Any]]],
    indices: Any,
    *,
    labels: tuple[str, ...] = ("action",),
) -> dict[str, tuple[Any, Any]]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    fitted = {}
    for side in ("left", "right"):
        local_positions = []
        local_rotations = []
        for label in labels:
            frame_p, frame_r = placements[label][side]
            target_p, target_r = targets[label][side]
            local_positions.append(
                np.einsum(
                    "nij,nj->ni",
                    frame_r[indices].transpose(0, 2, 1),
                    target_p[indices] - frame_p[indices],
                )
            )
            local_rotations.append(frame_r[indices].transpose(0, 2, 1) @ target_r[indices])
        translation = np.median(np.concatenate(local_positions, axis=0), axis=0)
        rotation = Rotation.from_matrix(np.concatenate(local_rotations, axis=0)).mean().as_matrix()
        fitted[side] = (translation, rotation)
    return fitted


def _percentiles(values: Any) -> dict[str, float]:
    import numpy as np

    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def synthetic_action_fk_report(
    *,
    robot_q_desired: Any,
    ee_action: Any,
    urdf_path: Path,
    frame_names: dict[str, str],
    tool_transforms: dict[str, dict[str, list[float]]],
    position_p95_max: float,
    rotation_p95_max: float,
) -> dict[str, Any]:
    """Audit every synthetic EEF command against its recorded joint target."""

    import numpy as np
    from scipy.spatial.transform import Rotation

    joints = np.asarray(robot_q_desired, dtype=np.float64)
    targets = np.asarray(ee_action, dtype=np.float64)
    if (
        joints.ndim != 2
        or joints.shape[1] != 36
        or targets.shape != (len(joints), 12)
        or len(joints) == 0
        or not np.isfinite(joints).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("synthetic FK inputs must be finite robot_q_desired[T,36] and ee_action[T,12]")
    if position_p95_max <= 0.0 or rotation_p95_max <= 0.0:
        raise ValueError("synthetic FK thresholds must be positive")

    pin, model, data, joint_indices, frame_ids = _build_fk(urdf_path, frame_names)
    configured_tools = {}
    for side in ("left", "right"):
        value = tool_transforms.get(side, {})
        translation = np.asarray(value.get("translation_m"), dtype=np.float64)
        quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=np.float64)
        if (
            translation.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(translation).all()
            or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-6)
        ):
            raise ValueError(f"invalid synthetic FK tool transform for {side}")
        configured_tools[side] = (translation, Rotation.from_quat(quaternion).as_matrix())

    predicted_positions = {side: [] for side in frame_ids}
    predicted_rotations = {side: [] for side in frame_ids}
    q = np.zeros(model.nq, dtype=np.float64)
    for source_q in joints:
        q.fill(0.0)
        q[joint_indices] = source_q[7:]
        pin.framesForwardKinematics(model, data, q)
        for side, frame_id in frame_ids.items():
            frame = data.oMf[frame_id]
            tool_position, tool_rotation = configured_tools[side]
            frame_position = np.asarray(frame.translation)
            frame_rotation = np.asarray(frame.rotation)
            predicted_positions[side].append(
                frame_position + frame_rotation @ tool_position
            )
            predicted_rotations[side].append(frame_rotation @ tool_rotation)

    sides = {}
    passed = True
    for side, offset in (("left", 0), ("right", 6)):
        predicted_position = np.stack(predicted_positions[side])
        predicted_rotation = np.stack(predicted_rotations[side])
        target_position = targets[:, offset : offset + 3]
        target_rotation = Rotation.from_euler(
            "xyz", targets[:, offset + 3 : offset + 6]
        ).as_matrix()
        position_error = np.linalg.norm(predicted_position - target_position, axis=1)
        rotation_error = Rotation.from_matrix(
            predicted_rotation.transpose(0, 2, 1) @ target_rotation
        ).magnitude()
        side_passed = bool(
            np.percentile(position_error, 95) <= position_p95_max
            and np.percentile(rotation_error, 95) <= rotation_p95_max
        )
        passed &= side_passed
        sides[side] = {
            "position_error_m": _percentiles(position_error),
            "rotation_error_rad": _percentiles(rotation_error),
            "passed": side_passed,
        }
    return {
        "schema_version": SYNTHETIC_ACTION_FK_SCHEMA_VERSION,
        "frame_count": len(joints),
        "urdf_sha256": sha256_file(urdf_path),
        "joint_order": list(G1_BODY_JOINT_ORDER),
        "eef_order": ["left", "right"],
        "eef_pose_format": "xyz_euler_xyz_rad",
        "eef_reference_frame": "robot_root",
        "thresholds": {
            "position_p95_m_max": float(position_p95_max),
            "rotation_p95_rad_max": float(rotation_p95_max),
        },
        "sides": sides,
        "pass": bool(passed),
    }


def _metrics(
    placements: dict[str, dict[str, tuple[Any, Any]]],
    targets: dict[str, dict[str, tuple[Any, Any]]],
    tool_transforms: dict[str, tuple[Any, Any]],
    indices: Any,
) -> tuple[dict[str, Any], float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    report: dict[str, Any] = {}
    scores: list[float] = []
    for label in ("state", "action"):
        report[label] = {}
        for side in ("left", "right"):
            frame_p, frame_r = placements[label][side]
            target_p, target_r = targets[label][side]
            tool_p, tool_r = tool_transforms[side]
            predicted_p = frame_p[indices] + np.einsum("nij,j->ni", frame_r[indices], tool_p)
            predicted_r = frame_r[indices] @ tool_r
            position_error = np.linalg.norm(predicted_p - target_p[indices], axis=1)
            rotation_error = Rotation.from_matrix(
                predicted_r.transpose(0, 2, 1) @ target_r[indices]
            ).magnitude()
            report[label][side] = {
                "position_error_m": _percentiles(position_error),
                "rotation_error_rad": _percentiles(rotation_error),
            }
            scores.append(float(np.median(position_error) + np.median(rotation_error)))
    return report, sum(scores)


def run_fk_audit(
    *,
    samples: tuple[Sample, ...],
    urdf_path: Path,
    frame_names: dict[str, str],
    eef_order: tuple[str, str],
    tool_transforms: dict[str, dict[str, list[float]]],
    tool_transform_reference: dict[str, str],
    validation_episode_modulus: int,
    position_p95_max: float,
    rotation_p95_max: float,
    swapped_score_ratio_min: float,
    source_repo_id: str,
    source_revision: str,
) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    if eef_order != ("left", "right"):
        raise ValueError("this audit requires the configured source EEF order left, right")
    if validation_episode_modulus < 2:
        raise ValueError("validation_episode_modulus must be at least two")
    calibration_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.episode_index % validation_episode_modulus != 0],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.episode_index % validation_episode_modulus == 0],
        dtype=np.int64,
    )
    if calibration_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("episode-level calibration/validation split is empty")

    placements = _fk_placements(samples, urdf_path, frame_names)
    configured_assignment = {"left": 0, "right": 6}
    swapped_assignment = {"left": 6, "right": 0}
    configured_targets = {
        label: _eef_targets(samples, label, configured_assignment) for label in ("state", "action")
    }
    swapped_targets = {
        label: _eef_targets(samples, label, swapped_assignment) for label in ("state", "action")
    }
    fitted_tool_transforms = _fit_tool_transform(placements, configured_targets, calibration_indices)
    swapped_transforms = _fit_tool_transform(placements, swapped_targets, calibration_indices)
    configured_tool_transforms = {}
    for side in ("left", "right"):
        value = tool_transforms.get(side, {})
        translation = np.asarray(value.get("translation_m"), dtype=np.float64)
        quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=np.float64)
        if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
            raise ValueError(f"invalid configured tool transform for {side}")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
            raise ValueError(f"configured tool quaternion for {side} is not unit length")
        configured_tool_transforms[side] = (translation, Rotation.from_quat(quaternion).as_matrix())
    calibration_metrics, calibration_score = _metrics(
        placements, configured_targets, configured_tool_transforms, calibration_indices
    )
    validation_metrics, validation_score = _metrics(
        placements, configured_targets, configured_tool_transforms, validation_indices
    )
    _fitted_metrics, configured_assignment_score = _metrics(
        placements, configured_targets, fitted_tool_transforms, validation_indices
    )
    _swapped_metrics, swapped_validation_score = _metrics(
        placements, swapped_targets, swapped_transforms, validation_indices
    )
    assignment_ratio = swapped_validation_score / max(configured_assignment_score, 1e-12)

    # robot_q_desired and ee_action are two labels for the same command and
    # define the conversion gate. robot_q_current and ee_state include controller
    # tracking and sensor timing, so they remain diagnostics and do not redefine
    # the source action contract.
    threshold_passes = []
    for side in ("left", "right"):
        metric = validation_metrics["action"][side]
        threshold_passes.append(metric["position_error_m"]["p95"] <= position_p95_max)
        threshold_passes.append(metric["rotation_error_rad"]["p95"] <= rotation_p95_max)
    assignment_pass = assignment_ratio >= swapped_score_ratio_min
    calibration_episode_count = len({samples[index].episode_index for index in calibration_indices})
    validation_episode_count = len({samples[index].episode_index for index in validation_indices})
    transforms_json = {}
    for side, (translation, rotation) in configured_tool_transforms.items():
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        if quaternion_xyzw[3] < 0:
            quaternion_xyzw *= -1
        transforms_json[side] = {
            "parent_frame": frame_names[side],
            "child_frame": f"{side}_source_eef",
            "translation_m": translation.tolist(),
            "quaternion_xyzw": quaternion_xyzw.tolist(),
        }
    fitted_transforms_json = {}
    for side, (translation, rotation) in fitted_tool_transforms.items():
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        if quaternion_xyzw[3] < 0:
            quaternion_xyzw *= -1
        fitted_transforms_json[side] = {
            "parent_frame": frame_names[side],
            "child_frame": f"{side}_source_eef",
            "translation_m": translation.tolist(),
            "quaternion_xyzw": quaternion_xyzw.tolist(),
        }
    per_episode = []
    for episode_index in sorted({sample.episode_index for sample in samples}):
        indices = np.asarray(
            [
                index
                for index, sample in enumerate(samples)
                if sample.episode_index == episode_index
            ],
            dtype=np.int64,
        )
        metrics, _score = _metrics(
            placements,
            configured_targets,
            configured_tool_transforms,
            indices,
        )
        action_pass = all(
            metrics["action"][side]["position_error_m"]["p95"] <= position_p95_max
            and metrics["action"][side]["rotation_error_rad"]["p95"] <= rotation_p95_max
            for side in ("left", "right")
        )
        per_episode.append(
            {
                "episode_index": episode_index,
                "sample_count": int(indices.size),
                "action_fk_residual_pass": action_pass,
                "action": metrics["action"],
                "state_diagnostic": metrics["state"],
            }
        )
    eligible_episode_indices = [
        value["episode_index"] for value in per_episode if value["action_fk_residual_pass"]
    ]
    return {
        "schema_version": FK_AUDIT_SCHEMA_VERSION,
        "source_repo_id": source_repo_id,
        "source_revision": source_revision,
        "urdf_path": str(Path(urdf_path).resolve()),
        "urdf_sha256": sha256_file(urdf_path),
        "joint_order": list(G1_BODY_JOINT_ORDER),
        "configured_eef_order": list(eef_order),
        "eef_pose_format": "xyz_euler_xyz_rad",
        "eef_reference_frame": "robot_root",
        "sample_count": len(samples),
        "calibration_sample_count": int(calibration_indices.size),
        "validation_sample_count": int(validation_indices.size),
        "calibration_episode_count": calibration_episode_count,
        "validation_episode_count": validation_episode_count,
        "validation_episode_rule": f"episode_index % {validation_episode_modulus} == 0",
        "tool_transforms": transforms_json,
        "tool_transform_reference": tool_transform_reference,
        "fitted_tool_transforms_diagnostic_only": fitted_transforms_json,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "configured_assignment_score": configured_assignment_score,
        "swapped_assignment_score": swapped_validation_score,
        "swapped_to_configured_score_ratio": assignment_ratio,
        "thresholds": {
            "position_p95_m_max": position_p95_max,
            "rotation_p95_rad_max": rotation_p95_max,
            "swapped_score_ratio_min": swapped_score_ratio_min,
        },
        "frame_assignment_pass": assignment_pass,
        "action_fk_residual_pass": all(threshold_passes),
        "mimic_source_episode_gate": {
            "eligible_count": len(eligible_episode_indices),
            "rejected_count": len(per_episode) - len(eligible_episode_indices),
            "eligible_episode_indices": eligible_episode_indices,
            "requires_pose_and_phase_review_in_addition": True,
        },
        "per_episode": per_episode,
        "state_fk_metrics_are_diagnostic_only": True,
        "pass": assignment_pass and all(threshold_passes),
    }
