"""Build phase-indexed Mimic source records from successful sim teleoperation.

The raw simulator sidecar contains object poses and contact forces so a
successful manual demonstration can be checked and segmented.  Those values
are intentionally consumed in this module only and are never copied into the
resulting HDF5.  The standard Isaac Lab Mimic retargeter uses object pose as a
planning input, so it must not run on this source until an RGB-based pose
adapter is supplied for both the recorded and generated scenes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..automatic_annotation import segment_flip_table_phases
from ..config import PipelineConfig
from ..io_utils import sha256_file
from ..source_contract import NUMERIC_FEATURES
from ..teleop.numeric import (
    NUMERIC_CONVERSION_SCHEMA_VERSION,
    convert_raw_episode,
    sim_teleop_source_index,
)
from ..teleop.raw_episode import RAW_EPISODE_SCHEMA_VERSION
from .source_hdf5 import (
    MIMIC_ENV_NAME,
    _create_dataset,
    _eef_targets_to_pink_actions,
    _euler_pose_matrices,
    _pink_actions_to_eef_targets,
    _resample_matrices,
    _resample_vectors,
)


TELEOP_MIMIC_SOURCE_SCHEMA_VERSION = "team_ramen_flip_table_teleop_mimic_source/v2"
CONTACT_FORCE_THRESHOLD_N = 0.05


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _pose_matrix(pose_xyzw: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pose = np.asarray(pose_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite xyz+xyzw")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def _offline_annotation_teacher_signals(
    episode_root: Path,
    *,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read simulator-only signals for post-recording acceptance annotation.

    ``table_pose`` and ``contact_force`` must not escape this function as
    source HDF5 fields: policy, critic, planner, and inference paths are not
    allowed to consume simulator-only state.
    """
    table_poses = []
    contact_forces = []
    trace_path = episode_root / "frames.jsonl"
    with trace_path.open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            frame = json.loads(line)
            if frame.get("frame_index") != expected_index:
                raise ValueError("raw teleop frame indices are not contiguous")
            sim = frame.get("diagnostics", {}).get("sim", {})
            white_table = sim.get("white_table") if isinstance(sim, dict) else None
            contact = sim.get("gripper_contact_force_n") if isinstance(sim, dict) else None
            if not isinstance(white_table, dict) or not isinstance(contact, dict):
                raise ValueError("sim teleop trace lacks table pose or gripper contact teacher data")
            if contact.get("available") is not True:
                raise ValueError("sim teleop contact sensors were unavailable")
            root_world = _pose_matrix(frame["root_pose_xyzw"])
            table_world = _pose_matrix(
                [
                    *white_table["position_world_m"],
                    *white_table["quaternion_xyzw"],
                ]
            )
            table_poses.append(np.linalg.inv(root_world) @ table_world)
            contact_forces.append(
                [float(contact["left_max_n"]), float(contact["right_max_n"])]
            )
    if len(table_poses) != frame_count:
        raise ValueError("teacher trace frame count differs from manifest")
    table = np.asarray(table_poses, dtype=np.float64)
    forces = np.asarray(contact_forces, dtype=np.float64)
    if table.shape != (frame_count, 4, 4) or forces.shape != (frame_count, 2):
        raise RuntimeError("teacher signal shape is invalid")
    if not np.isfinite(table).all() or not np.isfinite(forces).all() or np.any(forces < 0.0):
        raise ValueError("teacher signals contain invalid values")
    return table, forces


def _numeric_columns(path: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(NUMERIC_FEATURES))
    output = {
        key: np.asarray(table[key].to_pylist(), dtype=np.float64)
        for key in NUMERIC_FEATURES
    }
    count = len(table)
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        if output[key].shape != (count, width) or not np.isfinite(output[key]).all():
            raise ValueError(f"numeric teleop feature is invalid: {key}")
    return output


def _first_sustained(mask: np.ndarray, count: int, stop: int) -> int | None:
    run = 0
    for index, value in enumerate(mask[:stop]):
        run = run + 1 if bool(value) else 0
        if run >= count:
            return index - count + 1
    return None


def _phase_boundaries(
    *,
    table_pose: np.ndarray,
    eef_state: np.ndarray,
    hand_command: np.ndarray,
    contact_force: np.ndarray,
    config: PipelineConfig,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    eef_positions = np.stack((eef_state[:, :3], eef_state[:, 6:9]), axis=1)
    result = segment_flip_table_phases(
        table_poses_root=table_pose,
        eef_positions_root=eef_positions,
        hand_commands=hand_command,
        fps=30,
        config=config.source_annotation.automatic_phase,
    )
    if not result.accepted or result.boundaries is None:
        raise ValueError(
            "sim teleop phase segmentation failed: " + ", ".join(result.rejection_reasons)
        )
    boundaries = list(result.boundaries)
    both_contact = np.all(contact_force >= CONTACT_FORCE_THRESHOLD_N, axis=1)
    contact_start = _first_sustained(
        both_contact,
        config.source_annotation.automatic_phase.sustained_event_frames,
        boundaries[2],
    )
    if contact_start is None:
        raise ValueError("no sustained bimanual simulator contact before lift")
    boundaries[1] = max(boundaries[1], contact_start)
    minimum = config.source_annotation.automatic_phase.minimum_phase_frames
    if boundaries[2] - boundaries[1] < minimum:
        raise ValueError("contact-refined grasp phase is shorter than the release minimum")
    diagnostics = {
        **result.diagnostics,
        "contact_teacher": {
            "threshold_n": CONTACT_FORCE_THRESHOLD_N,
            "sustained_bimanual_start_frame": contact_start,
            "left_max_n": float(np.max(contact_force[:, 0])),
            "right_max_n": float(np.max(contact_force[:, 1])),
        },
        "boundaries_after_contact_refinement": boundaries,
    }
    return tuple(boundaries), diagnostics


def _subtask_signals(
    boundaries: tuple[int, ...],
    *,
    source_hz: int,
    target_hz: int,
    output_count: int,
    subtasks: tuple[str, ...],
) -> dict[str, np.ndarray]:
    signals = {}
    for side in ("left", "right"):
        for index, subtask in enumerate(subtasks[:-1]):
            end = int(round(boundaries[index + 1] * target_hz / source_hz))
            end = min(max(end, 1), output_count)
            signal = np.zeros((output_count, 1), dtype=np.uint8)
            signal[end - 1 :] = 1
            signals[f"{side}_{subtask}_done"] = signal
    return signals


def export_teleop_mimic_source_hdf5(
    *,
    episode_roots: Iterable[str | Path],
    urdf_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Export validated sim demos without copying simulator GT into HDF5.

    The output is deliberately phase-indexed rather than immediately runnable
    by Isaac Lab's object-centric Mimic generator.  That generator would use
    the missing table pose to retarget trajectories, which would violate the
    Sim-to-Real data contract.  A future RGB-only pose adapter may consume this
    source after proving its camera-only contract.
    """

    import h5py

    roots = tuple(sorted(Path(value).expanduser().resolve() for value in episode_roots))
    if not roots:
        raise ValueError("no raw sim teleoperation episodes were supplied")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    reports = []
    total_steps = 0
    try:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["format_version"] = 1
            stream.attrs["schema_version"] = TELEOP_MIMIC_SOURCE_SCHEMA_VERSION
            stream.attrs["source_kind"] = "successful_sim_avp_teleoperation"
            stream.attrs["pipeline_config_sha256"] = config.digest
            stream.attrs["source_fps"] = 30
            stream.attrs["mimic_control_hz"] = int(
                config.raw["generation"]["mimic_control_hz"]
            )
            stream.attrs["sim_gt_usage"] = (
                "offline_phase_boundaries_contact_validation_and_success_only"
            )
            stream.attrs["object_pose_source"] = "not_stored"
            stream.attrs["mimic_retargeting"] = "blocked_pending_rgb_pose_adapter"
            stream.attrs["privileged_policy_features_json"] = "[]"
            data_group = stream.create_group("data")
            data_group.attrs["total"] = 0
            data_group.attrs["env_args"] = json.dumps(
                {"env_name": MIMIC_ENV_NAME, "type": 2, "source_kind": "sim_teleop"},
                sort_keys=True,
            )

            for demo_index, root in enumerate(roots):
                manifest_path = root / "manifest.json"
                manifest = _load_json(manifest_path)
                if (
                    manifest.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION
                    or manifest.get("backend") != "sim"
                    or manifest.get("success") is not True
                    or manifest.get("privileged_policy_features") != []
                ):
                    raise ValueError(f"raw episode is not an accepted sim demo: {root}")
                numeric_path = root / "numeric.parquet"
                numeric_manifest_path = root / "numeric.manifest.json"
                if not numeric_path.is_file() or not numeric_manifest_path.is_file():
                    convert_raw_episode(root, urdf_path=urdf_path)
                numeric_manifest = _load_json(numeric_manifest_path)
                if (
                    numeric_manifest.get("schema_version")
                    != NUMERIC_CONVERSION_SCHEMA_VERSION
                    or numeric_manifest.get("output_sha256") != sha256_file(numeric_path)
                ):
                    raise ValueError("teleop numeric conversion manifest is invalid")
                numeric = _numeric_columns(numeric_path)
                frame_count = int(manifest["frame_count"])
                table_pose, contact_force = _offline_annotation_teacher_signals(
                    root, frame_count=frame_count
                )
                boundaries, phase_diagnostics = _phase_boundaries(
                    table_pose=table_pose,
                    eef_state=numeric["observation.state.ee_state"],
                    hand_command=numeric["action.hand_cmd"],
                    contact_force=contact_force,
                    config=config,
                )

                source_hz = 30
                control_hz = int(config.raw["generation"]["mimic_control_hz"])
                state_pose = _euler_pose_matrices(
                    numeric["observation.state.ee_state"]
                )
                target_pose = _euler_pose_matrices(numeric["action.ee_action"])
                eef_pose = {
                    side: _resample_matrices(state_pose[side], source_hz, control_hz)
                    for side in ("left", "right")
                }
                target_eef_pose = {
                    side: _resample_matrices(target_pose[side], source_hz, control_hz)
                    for side in ("left", "right")
                }
                hand = _resample_vectors(
                    numeric["action.hand_cmd"], source_hz, control_hz
                )
                actions = _eef_targets_to_pink_actions(target_eef_pose, hand)
                reconstructed = _pink_actions_to_eef_targets(actions)
                inverse_error = max(
                    float(np.max(np.abs(reconstructed[side] - target_eef_pose[side])))
                    for side in ("left", "right")
                )
                if inverse_error > 1.0e-5:
                    raise ValueError("teleop PINK/tool inverse gate failed")

                demo = data_group.create_group(f"demo_{demo_index}")
                source_index = sim_teleop_source_index(manifest["episode_id"])
                demo.attrs["num_samples"] = len(actions)
                demo.attrs["success"] = 1
                demo.attrs["source_episode_index"] = source_index
                demo.attrs["source_teleop_episode_id"] = manifest["episode_id"]
                demo.attrs["source_kind"] = "sim_teleop"
                demo.attrs["raw_manifest_sha256"] = sha256_file(manifest_path)
                demo.attrs["numeric_manifest_sha256"] = sha256_file(
                    numeric_manifest_path
                )
                demo.attrs["pink_tool_inverse_max_abs_error"] = inverse_error
                demo.attrs["phase_diagnostics_json"] = json.dumps(
                    phase_diagnostics, sort_keys=True, allow_nan=False
                )
                _create_dataset(demo, "actions", actions)
                datagen = demo.create_group("obs").create_group("datagen_info")
                eef_group = datagen.create_group("eef_pose")
                target_group = datagen.create_group("target_eef_pose")
                for side in ("left", "right"):
                    _create_dataset(eef_group, side, eef_pose[side])
                    _create_dataset(target_group, side, target_eef_pose[side])
                signal_group = datagen.create_group("subtask_term_signals")
                for name, signal in _subtask_signals(
                    boundaries,
                    source_hz=source_hz,
                    target_hz=control_hz,
                    output_count=len(actions),
                    subtasks=config.subtasks,
                ).items():
                    _create_dataset(signal_group, name, signal)
                total_steps += len(actions)
                data_group.attrs["total"] = total_steps
                reports.append(
                    {
                        "demo_index": demo_index,
                        "source_episode_index": source_index,
                        "source_teleop_episode_id": manifest["episode_id"],
                        "frames": frame_count,
                        "control_steps": len(actions),
                        "boundaries": list(boundaries),
                    }
                )
            stream.flush()
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "schema_version": TELEOP_MIMIC_SOURCE_SCHEMA_VERSION,
        "output": str(output),
        "sha256": sha256_file(output),
        "episodes": len(reports),
        "control_steps": total_steps,
        "privileged_policy_features": [],
        "sim_gt_usage": "offline_phase_boundaries_contact_validation_and_success_only",
        "object_pose_source": "not_stored",
        "mimic_retargeting": "blocked_pending_rgb_pose_adapter",
        "episode_reports": reports,
    }
