"""Convert accepted LeRobot annotations to Isaac Lab Mimic source HDF5."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..annotations import SourceEpisodeAnnotation, load_annotations
from ..config import PipelineConfig
from ..fk_audit import FK_AUDIT_SCHEMA_VERSION
from ..io_utils import sha256_file
from ..source_dataset import SourceDatasetIndex


MIMIC_ENV_NAME = "RoboFinals-FlipTable-G1-Dex1-Mimic-v0"
MIMIC_SOURCE_SCHEMA_VERSION = "team_ramen_flip_table_mimic_source/v1"
DEMO_HAND_CLOSED = 0.0
DEMO_HAND_OPEN = 4.5


def _episode_table(episode, columns: list[str]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for Mimic source conversion") from exc

    table = pq.read_table(
        episode.data_path,
        columns=["frame_index", *columns],
        filters=[("episode_index", "=", episode.episode_index)],
    )
    table = table.sort_by([("frame_index", "ascending")])
    indices = [int(value) for value in table["frame_index"].to_pylist()]
    if indices != list(range(episode.frame_count)):
        raise ValueError(f"episode {episode.episode_index} frame_index is not contiguous")
    return table


def _euler_pose_matrices(values: Any, *, width: int = 12) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f"EEF values must be finite [T,{width}]")
    output = {}
    for side, offset in (("left", 0), ("right", 6)):
        matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(array), axis=0)
        matrices[:, :3, 3] = array[:, offset : offset + 3]
        matrices[:, :3, :3] = Rotation.from_euler(
            "xyz", array[:, offset + 3 : offset + 6]
        ).as_matrix()
        output[side] = matrices.astype(np.float32)
    return output


def _table_pose_matrices(annotation: SourceEpisodeAnnotation):
    import numpy as np
    from scipy.spatial.transform import Rotation

    poses = np.asarray(annotation.table_pose_trajectory_robot_root_xyzw, dtype=np.float64)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    return matrices.astype(np.float32)


def _resample_matrices(values: Any, source_hz: int, target_hz: int):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    matrices = np.asarray(values, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) < 2:
        raise ValueError("SE(3) resampling requires at least two [4,4] poses")
    output_count = int(round(len(matrices) * target_hz / source_hz))
    if output_count < 2:
        raise ValueError("resampled trajectory is too short")
    source_times = np.arange(len(matrices), dtype=np.float64) / source_hz
    target_times = np.minimum(
        np.arange(output_count, dtype=np.float64) / target_hz,
        source_times[-1],
    )
    output = np.repeat(np.eye(4, dtype=np.float64)[None], output_count, axis=0)
    for axis in range(3):
        output[:, axis, 3] = np.interp(
            target_times, source_times, matrices[:, axis, 3]
        )
    output[:, :3, :3] = Slerp(
        source_times, Rotation.from_matrix(matrices[:, :3, :3])
    )(target_times).as_matrix()
    return output.astype(np.float32)


def _resample_vectors(values: Any, source_hz: int, target_hz: int):
    import numpy as np

    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 2 or not np.isfinite(vectors).all():
        raise ValueError("vector resampling requires finite [T,D] values")
    output_count = int(round(len(vectors) * target_hz / source_hz))
    source_times = np.arange(len(vectors), dtype=np.float64) / source_hz
    target_times = np.minimum(
        np.arange(output_count, dtype=np.float64) / target_hz,
        source_times[-1],
    )
    return np.stack(
        [np.interp(target_times, source_times, vectors[:, axis]) for axis in range(vectors.shape[1])],
        axis=1,
    ).astype(np.float32)


def _eef_targets_to_pink_actions(targets: dict[str, Any], hand_commands: Any):
    import numpy as np
    from scipy.spatial.transform import Rotation

    count = targets["left"].shape[0]
    action = np.zeros((count, 16), dtype=np.float32)
    tool = np.asarray((0.05, 0.0, 0.0), dtype=np.float64)
    for side, start in (("left", 0), ("right", 7)):
        matrices = np.asarray(targets[side], dtype=np.float64)
        wrist_position = matrices[:, :3, 3] - np.einsum("nij,j->ni", matrices[:, :3, :3], tool)
        quaternion_xyzw = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
        quaternion_wxyz = quaternion_xyzw[:, (3, 0, 1, 2)]
        action[:, start : start + 3] = wrist_position
        action[:, start + 3 : start + 7] = quaternion_wxyz
    hands = np.asarray(hand_commands, dtype=np.float64)
    if hands.shape != (count, 2) or not np.isfinite(hands).all():
        raise ValueError("action.hand_cmd must be finite [T,2]")
    if np.any(hands < DEMO_HAND_CLOSED - 1.0e-6) or np.any(hands > DEMO_HAND_OPEN + 1.0e-6):
        raise ValueError("action.hand_cmd lies outside the measured [0,4.5] range")
    open_fraction = (hands - DEMO_HAND_CLOSED) / (DEMO_HAND_OPEN - DEMO_HAND_CLOSED)
    action[:, 14:16] = np.clip(1.0 - 2.0 * open_fraction, -1.0, 1.0)
    return action


def _pink_actions_to_eef_targets(actions: Any) -> dict[str, Any]:
    """Offline inverse of the simulator adapter, used as an export gate."""

    import numpy as np
    from scipy.spatial.transform import Rotation

    action = np.asarray(actions, dtype=np.float64)
    tool = np.asarray((0.05, 0.0, 0.0), dtype=np.float64)
    output = {}
    if action.ndim != 2 or action.shape[1] != 16 or not np.isfinite(action).all():
        raise ValueError("PINK actions must be finite [T,16]")
    for side, start in (("left", 0), ("right", 7)):
        quaternion_wxyz = action[:, start + 3 : start + 7]
        rotation = Rotation.from_quat(quaternion_wxyz[:, (1, 2, 3, 0)]).as_matrix()
        matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(action), axis=0)
        matrices[:, :3, :3] = rotation
        matrices[:, :3, 3] = action[:, start : start + 3] + np.einsum(
            "nij,j->ni", rotation, tool
        )
        output[side] = matrices
    return output


def _subtask_signals(
    annotation: SourceEpisodeAnnotation,
    *,
    output_count: int,
    source_hz: int,
    target_hz: int,
) -> dict[str, Any]:
    import numpy as np

    signals = {}
    for side, ranges in annotation.subtasks.items():
        for subtask, frame_range in list(ranges.items())[:-1]:
            end = int(round(frame_range.end * target_hz / source_hz))
            end = min(max(end, 1), output_count)
            signal = np.zeros((output_count, 1), dtype=np.uint8)
            signal[end - 1 :] = 1
            signals[f"{side}_{subtask}_done"] = signal
    return signals


def _create_dataset(group, name: str, value: Any) -> None:
    group.create_dataset(name, data=value, compression="gzip", compression_opts=2)


def _load_accepted_fk_episodes(path: Path, config: PipelineConfig) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": FK_AUDIT_SCHEMA_VERSION,
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "config_sha256": config.digest,
        "pass": True,
        "frame_assignment_pass": True,
        "action_fk_residual_pass": True,
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("FK audit does not prove the active source/action contract")
    gate = payload.get("mimic_source_episode_gate")
    indices = gate.get("eligible_episode_indices") if isinstance(gate, dict) else None
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("FK audit lacks a valid episode-level Mimic source gate")
    return set(indices)


def export_mimic_source_hdf5(
    *,
    source_root: str | Path,
    annotations_path: str | Path,
    fk_audit_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Export only residual-gated annotations to an atomic Mimic HDF5 file."""

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and NumPy are required for Mimic source conversion") from exc

    annotations_file = Path(annotations_path).expanduser().resolve()
    annotations = load_annotations(annotations_file)
    fk_audit_file = Path(fk_audit_path).expanduser().resolve()
    eligible_fk_episodes = _load_accepted_fk_episodes(fk_audit_file, config)
    ineligible_annotations = sorted(
        annotation.episode_index
        for annotation in annotations
        if annotation.episode_index not in eligible_fk_episodes
    )
    if ineligible_annotations:
        raise ValueError(
            f"annotations contain episodes rejected by the held-out FK gate: {ineligible_annotations[:10]}"
        )
    source_index = SourceDatasetIndex(source_root)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    total_source_frames = 0
    total_control_steps = 0
    episode_reports = []
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["format_version"] = 1
            stream.attrs["schema_version"] = MIMIC_SOURCE_SCHEMA_VERSION
            stream.attrs["source_repo_id"] = config.source.repo_id
            stream.attrs["source_revision"] = config.source.revision
            stream.attrs["pipeline_config_sha256"] = config.digest
            stream.attrs["annotations_sha256"] = sha256_file(annotations_file)
            stream.attrs["fk_audit_sha256"] = sha256_file(fk_audit_file)
            stream.attrs["source_fps"] = config.source.fps
            stream.attrs["mimic_control_hz"] = int(config.raw["generation"]["mimic_control_hz"])
            data_group = stream.create_group("data")
            data_group.attrs["total"] = 0
            data_group.attrs["env_args"] = json.dumps(
                {
                    "env_name": MIMIC_ENV_NAME,
                    "type": 2,
                    "source_repo_id": config.source.repo_id,
                    "source_revision": config.source.revision,
                },
                sort_keys=True,
            )
            for demo_index, annotation in enumerate(annotations):
                episode = source_index.episode(annotation.episode_index)
                if episode.frame_count != annotation.frame_count:
                    raise ValueError(
                        f"episode {annotation.episode_index} annotation/source frame count mismatch"
                    )
                table = _episode_table(
                    episode,
                    ["observation.state.ee_state", "action.ee_action", "action.hand_cmd"],
                )
                source_hz = config.source.fps
                control_hz = int(config.raw["generation"]["mimic_control_hz"])
                source_eef_pose = _euler_pose_matrices(
                    table["observation.state.ee_state"].to_pylist()
                )
                source_target_eef_pose = _euler_pose_matrices(
                    table["action.ee_action"].to_pylist()
                )
                eef_pose = {
                    side: _resample_matrices(source_eef_pose[side], source_hz, control_hz)
                    for side in ("left", "right")
                }
                target_eef_pose = {
                    side: _resample_matrices(source_target_eef_pose[side], source_hz, control_hz)
                    for side in ("left", "right")
                }
                hand_commands = _resample_vectors(
                    table["action.hand_cmd"].to_pylist(), source_hz, control_hz
                )
                actions = _eef_targets_to_pink_actions(
                    target_eef_pose, hand_commands
                )
                reconstructed = _pink_actions_to_eef_targets(actions)
                inverse_error = max(
                    float(np.max(np.abs(reconstructed[side] - target_eef_pose[side])))
                    for side in ("left", "right")
                )
                if inverse_error > 1.0e-5:
                    raise ValueError(f"PINK/tool inverse gate failed with max error {inverse_error}")

                demo = data_group.create_group(f"demo_{demo_index}")
                control_steps = len(actions)
                demo.attrs["num_samples"] = control_steps
                demo.attrs["success"] = 1
                demo.attrs["source_episode_index"] = annotation.episode_index
                demo.attrs["source_frame_count"] = annotation.frame_count
                demo.attrs["pose_calibration_sha256"] = annotation.pose_evidence.calibration_artifact_sha256
                demo.attrs["subtask_evidence_sha256"] = annotation.subtask_evidence_sha256
                demo.attrs["pink_tool_inverse_max_abs_error"] = inverse_error
                _create_dataset(demo, "actions", actions)
                datagen = demo.create_group("obs").create_group("datagen_info")
                eef_group = datagen.create_group("eef_pose")
                target_group = datagen.create_group("target_eef_pose")
                for side in ("left", "right"):
                    _create_dataset(eef_group, side, eef_pose[side])
                    _create_dataset(target_group, side, target_eef_pose[side])
                object_group = datagen.create_group("object_pose")
                _create_dataset(
                    object_group,
                    "white_table",
                    _resample_matrices(_table_pose_matrices(annotation), source_hz, control_hz),
                )
                signal_group = datagen.create_group("subtask_term_signals")
                for signal_name, signal in _subtask_signals(
                    annotation,
                    output_count=control_steps,
                    source_hz=source_hz,
                    target_hz=control_hz,
                ).items():
                    _create_dataset(signal_group, signal_name, signal)
                total_source_frames += annotation.frame_count
                total_control_steps += control_steps
                data_group.attrs["total"] = total_control_steps
                episode_reports.append(
                    {
                        "demo_index": demo_index,
                        "source_episode_index": annotation.episode_index,
                        "source_frames": annotation.frame_count,
                        "control_steps": control_steps,
                        "pink_tool_inverse_max_abs_error": inverse_error,
                    }
                )
            stream.flush()
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "schema_version": MIMIC_SOURCE_SCHEMA_VERSION,
        "output": str(output),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "episodes": len(annotations),
        "source_frames": total_source_frames,
        "control_steps": total_control_steps,
        "source_revision": config.source.revision,
        "fk_audit_sha256": sha256_file(fk_audit_file),
        "episode_reports": episode_reports,
    }
