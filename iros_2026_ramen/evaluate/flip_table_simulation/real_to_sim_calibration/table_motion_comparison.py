#!/usr/bin/env python3
"""Compare an RGB/CAD-recovered real table trajectory with a replay trace.

The source dataset contains no table pose telemetry.  This utility therefore
uses only frame-wise *paired* head-stereo CAD fits recorded by
``source_cad_alignment``.  Frames with insufficient stereo agreement are
omitted rather than filled from simulator state or an interpolated trajectory.

The comparison is intentionally relative to the first mutually observable
table pose.  Absolute reset error belongs to the camera/CAD reprojection gate;
relative motion is the evidence needed for contact-parameter identification.
Neither input is a policy feature.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file

from .contracts import SOURCE_FPS


SCHEMA_VERSION = "team_ramen_flip_table_table_motion_comparison/v1"
TEMPORAL_TRACKER_SCHEMA_VERSION = "team_ramen_flip_table_temporal_stereo_cad_tracker/v1"
FOUNDATIONPOSE_TRACK_SCHEMA_VERSION = "team_ramen_foundationpose_table_track/v48"
MINIMUM_STEREO_PAIRS = 3
TRANSLATION_MOTION_THRESHOLD_M = 0.020
ROTATION_MOTION_THRESHOLD_DEG = 3.0
# A 20 mm trajectory gate cannot be supported by a source stereo fit whose
# own left/right disagreement is already comparable with 20 mm.  Keep the
# observation-noise ceiling materially below the release threshold instead of
# reporting contact timing from camera-fit jitter.
MAX_STEREO_TRANSLATION_P95_M = 0.005
MAX_STEREO_ROTATION_P95_DEG = 0.75
TABLE_YAW_180_SYMMETRY = np.diag((-1.0, -1.0, 1.0, 1.0))


def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4):
        raise ValueError(f"{label} rotation is not orthonormal")
    return matrix


def _world_transform(position: Any, quaternion_xyzw: Any, label: str) -> np.ndarray:
    translation = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(
        np.concatenate((translation, quaternion))
    ).all():
        raise ValueError(f"{label} must have finite position[3] and quaternion_xyzw[4]")
    norm = float(np.linalg.norm(quaternion))
    if not 0.999 <= norm <= 1.001:
        raise ValueError(f"{label} quaternion is not normalized")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _rotation_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _canonical_against(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resolve the physical 180-degree tabletop symmetry deterministically."""

    candidates = (pose, pose @ TABLE_YAW_180_SYMMETRY)
    return min(candidates, key=lambda candidate: _rotation_deg(reference, candidate))


def _mean_stereo_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average two agreeing rigid poses without introducing a scale/shear fit."""

    aligned = _canonical_against(second, first)
    result = first.copy()
    result[:3, 3] = 0.5 * (first[:3, 3] + aligned[:3, 3])
    # The two input rotations passed the stereo gate.  Project their arithmetic
    # mean onto SO(3), which is sufficient for this small-angle averaging.
    u, _, vh = np.linalg.svd(first[:3, :3] + aligned[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    result[:3, :3] = rotation
    return result


def _load_frame_map(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("camera_frame_map")
    if not isinstance(entries, list):
        raise ValueError("replay actions omit camera_frame_map")
    mapping: dict[int, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("camera_frame_map entries must be objects")
        source = entry.get("source_frame")
        simulator = entry.get("simulator_step")
        if not isinstance(source, int) or not isinstance(simulator, int) or source < 0 or simulator < 0:
            raise ValueError("camera_frame_map entries must contain non-negative integer frames")
        if source in mapping:
            raise ValueError(f"duplicate source camera frame {source}")
        mapping[source] = simulator
    if not mapping:
        raise ValueError("camera_frame_map is empty")
    return mapping


def _sim_root_from_table_by_step(trace_path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        diagnostics = row.get("simulator_scene_diagnostics")
        if not isinstance(step, int) or not isinstance(diagnostics, dict):
            raise ValueError(f"trace row {line_number} lacks simulator diagnostics")
        root = diagnostics.get("root_pose_world_xyzw")
        table = diagnostics.get("white_table")
        if not isinstance(root, list) or len(root) != 7 or not isinstance(table, dict):
            raise ValueError(f"trace row {line_number} has invalid root/table pose")
        world_from_root = _world_transform(root[:3], root[3:], f"trace row {line_number} root")
        world_from_table = _world_transform(
            table.get("position_world_m"), table.get("quaternion_xyzw"), f"trace row {line_number} table"
        )
        poses[step] = np.linalg.inv(world_from_root) @ world_from_table
    if not poses:
        raise ValueError("trace contains no simulator table poses")
    return poses


def _source_pairs(source_alignment: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
    source = source_alignment.get("source")
    frames = source_alignment.get("frames")
    if not isinstance(source, dict) or not isinstance(frames, list):
        raise ValueError("source alignment omits source or frames")
    result: list[tuple[int, np.ndarray]] = []
    previous: np.ndarray | None = None
    for entry in sorted(frames, key=lambda value: int(value.get("frame_index", -1)) if isinstance(value, dict) else -1):
        if not isinstance(entry, dict):
            continue
        stereo = entry.get("stereo_agreement")
        eyes = entry.get("eyes")
        frame = entry.get("frame_index")
        if not isinstance(frame, int) or not isinstance(stereo, dict) or stereo.get("accepted") is not True:
            continue
        if not isinstance(eyes, dict):
            continue
        left, right = eyes.get("head_left"), eyes.get("head_right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if left.get("accepted") is not True or right.get("accepted") is not True:
            continue
        pose = _mean_stereo_pose(
            _matrix(left.get("root_from_table"), f"source frame {frame} left"),
            _matrix(right.get("root_from_table"), f"source frame {frame} right"),
        )
        # The square tabletop is physically unchanged by a 180-degree yaw.
        # Resolve that ambiguity against the preceding accepted observation,
        # not independently in every stereo pair, so a detector-side label
        # swap cannot become a fake half-turn in the contact trajectory.
        if previous is not None:
            pose = _canonical_against(pose, previous)
        previous = pose
        result.append((frame, pose))
    return result


def _temporal_source_pairs(report: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
    """Read only directly observed poses from a release-gated temporal tracker."""

    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("temporal tracker omits records")
    result: list[tuple[int, np.ndarray]] = []
    previous: np.ndarray | None = None
    for entry in sorted(records, key=lambda value: int(value.get("frame_index", -1)) if isinstance(value, dict) else -1):
        if not isinstance(entry, dict) or entry.get("state") != "observed":
            continue
        frame = entry.get("frame_index")
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("temporal tracker observed frame has invalid frame_index")
        pose = _matrix(entry.get("root_from_table"), f"temporal frame {frame}")
        if previous is not None:
            pose = _canonical_against(pose, previous)
        previous = pose
        result.append((frame, pose))
    return result


def _temporal_tracker_is_release_validated(report: dict[str, Any]) -> bool:
    summary = report.get("summary")
    uncertainty = report.get("measurement_uncertainty")
    if not isinstance(summary, dict) or not isinstance(uncertainty, dict):
        return False
    bound = uncertainty.get("independent_metric_bound")
    return bool(
        summary.get("accepted_for_table_motion_metric") is True
        and isinstance(bound, dict)
        and bound.get("passed") is True
    )


def _track_artifact_array(
    manifest_path: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    expected_shape_tail: tuple[int, ...],
) -> np.ndarray:
    """Load a hash-pinned FoundationPose artifact without trusting its path."""

    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or not isinstance(arrays.get(name), dict):
        raise ValueError(f"FoundationPose track omits array metadata for {name}")
    metadata = arrays[name]
    relative_path = metadata.get("path")
    expected_sha256 = metadata.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"FoundationPose track has invalid array metadata for {name}")
    root = manifest_path.parent.resolve()
    artifact_path = (root / relative_path).resolve()
    if artifact_path.parent != root or not artifact_path.is_file():
        raise ValueError(f"FoundationPose track array path escapes its output: {name}")
    if sha256_file(artifact_path) != expected_sha256:
        raise ValueError(f"FoundationPose track array digest differs: {name}")
    value = np.load(artifact_path, allow_pickle=False)
    if (
        value.ndim != len(expected_shape_tail) + 1
        or value.shape[1:] != expected_shape_tail
        or len(value) == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"FoundationPose track array has invalid shape: {name}")
    return np.asarray(value, dtype=np.float64)


def _foundationpose_track_is_motion_validated(manifest: dict[str, Any]) -> bool:
    """Check the stricter observation gate required for contact identification.

    The regular tracker acceptance is intentionally looser because it is also
    used for offline annotation.  Table-contact calibration needs a residual
    well below its 20 mm / 3 degree release limit, so it accepts only tracks
    whose *measured*, forward/reverse disagreement stays below this module's
    source-noise ceiling.  This remains source-observation evidence; it is
    never simulator telemetry or a policy input.
    """

    gate = manifest.get("gate")
    if not isinstance(gate, dict):
        return False
    translation_p95 = gate.get("p95_bidirectional_translation_error_m")
    rotation_p95 = gate.get("p95_bidirectional_rotation_error_rad")
    try:
        translation_p95 = float(translation_p95)
        rotation_p95 = float(rotation_p95)
    except (TypeError, ValueError):
        return False
    return bool(
        manifest.get("accepted") is True
        and gate.get("pass") is True
        and gate.get("bidirectional_pass") is True
        and gate.get("rendered_alignment_pass") is True
        and gate.get("pose_evidence_pass") is True
        and math.isfinite(translation_p95)
        and math.isfinite(rotation_p95)
        and translation_p95 <= MAX_STEREO_TRANSLATION_P95_M
        and math.degrees(rotation_p95) <= MAX_STEREO_ROTATION_P95_DEG
    )


def _foundationpose_source_pairs(manifest_path: Path, manifest: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
    """Read only hash-verified, bidirectionally observed CAD poses.

    The complete sampled trajectory includes sparse terminal predictions for
    annotation continuity.  Those are deliberately excluded here: contact
    fitting uses only rows that have a finite forward/reverse residual and a
    per-frame rendered-evidence gate below the strict source-noise ceiling.
    """

    source_indices_raw = _track_artifact_array(
        manifest_path, manifest, "source_frame_indices", expected_shape_tail=()
    )
    if not np.allclose(source_indices_raw, np.rint(source_indices_raw), rtol=0.0, atol=0.0):
        raise ValueError("FoundationPose source frame indices must be integers")
    source_indices = np.rint(source_indices_raw).astype(np.int64)
    if np.any(source_indices < 0) or np.any(np.diff(source_indices) <= 0):
        raise ValueError("FoundationPose source frame indices must be strictly increasing")
    poses = _track_artifact_array(
        manifest_path, manifest, "table_pose_root_sampled", expected_shape_tail=(4, 4)
    )
    translation_errors = _track_artifact_array(
        manifest_path, manifest, "bidirectional_translation_error_m", expected_shape_tail=()
    )
    rotation_errors = _track_artifact_array(
        manifest_path, manifest, "bidirectional_rotation_error_rad", expected_shape_tail=()
    )
    if not (
        len(source_indices)
        == len(poses)
        == len(translation_errors)
        == len(rotation_errors)
    ):
        raise ValueError("FoundationPose track arrays have inconsistent lengths")

    frame_evidence: dict[int, dict[str, Any]] = {}
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("FoundationPose track omits frame evidence")
    for entry in frames:
        if not isinstance(entry, dict):
            continue
        ordinal = entry.get("ordinal")
        if isinstance(ordinal, int):
            frame_evidence[ordinal] = entry

    result: list[tuple[int, np.ndarray]] = []
    previous: np.ndarray | None = None
    for ordinal, (frame, pose, translation_error, rotation_error) in enumerate(
        zip(source_indices, poses, translation_errors, rotation_errors, strict=True)
    ):
        entry = frame_evidence.get(ordinal)
        pose_evidence = entry.get("pose_evidence") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or entry.get("backward_mode") == "unavailable_terminal_forward_only"
            or not isinstance(pose_evidence, dict)
            or pose_evidence.get("passes_gate") is not True
            or translation_error > MAX_STEREO_TRANSLATION_P95_M
            or math.degrees(rotation_error) > MAX_STEREO_ROTATION_P95_DEG
        ):
            continue
        matrix = _matrix(pose, f"FoundationPose source frame {int(frame)}")
        if previous is not None:
            matrix = _canonical_against(matrix, previous)
        previous = matrix
        result.append((int(frame), matrix))
    return result


def _relative(pose: np.ndarray, initial: np.ndarray) -> np.ndarray:
    return np.linalg.inv(initial) @ pose


def _motion_events(items: list[dict[str, Any]], *, time_key: str) -> dict[str, float] | None:
    if not items:
        return None
    active = [
        item
        for item in items
        if item["translation_from_start_m"] >= TRANSLATION_MOTION_THRESHOLD_M
        or item["rotation_from_start_deg"] >= ROTATION_MOTION_THRESHOLD_DEG
    ]
    if not active:
        return None
    peak = max(
        items,
        key=lambda item: item["translation_from_start_m"]
        + math.radians(item["rotation_from_start_deg"]) * 0.20,
    )
    return {"onset_s": float(active[0][time_key]), "peak_s": float(peak[time_key])}


def _source_precision_is_sufficient(source_alignment: dict[str, Any]) -> bool:
    agreement = source_alignment.get("stereo_agreement")
    if not isinstance(agreement, dict):
        return False
    translation = agreement.get("accepted_translation_p95_m")
    rotation = agreement.get("accepted_rotation_p95_deg")
    return bool(
        isinstance(translation, (int, float))
        and isinstance(rotation, (int, float))
        and math.isfinite(float(translation))
        and math.isfinite(float(rotation))
        and float(translation) <= MAX_STEREO_TRANSLATION_P95_M
        and float(rotation) <= MAX_STEREO_ROTATION_P95_DEG
    )


def compare(
    source_alignment_path: Path,
    replay_actions_path: Path,
    sim_trace_path: Path,
) -> dict[str, Any]:
    source_alignment = json.loads(source_alignment_path.read_text(encoding="utf-8"))
    schema_version = source_alignment.get("schema_version")
    is_foundationpose_track = schema_version == FOUNDATIONPOSE_TRACK_SCHEMA_VERSION
    source = source_alignment.get("source")
    source_episode_index = (
        source_alignment.get("episode_index")
        if is_foundationpose_track
        else source.get("episode_index")
        if isinstance(source, dict)
        else None
    )
    if not isinstance(source_episode_index, int):
        raise ValueError("source alignment omits source episode_index")
    is_temporal_tracker = schema_version == TEMPORAL_TRACKER_SCHEMA_VERSION
    if is_temporal_tracker:
        if not _temporal_tracker_is_release_validated(source_alignment):
            return {
                "schema_version": SCHEMA_VERSION,
                "source_episode_index": source_episode_index,
                "source_alignment_path": str(source_alignment_path),
                "replay_actions_path": str(replay_actions_path),
                "sim_trace_path": str(sim_trace_path),
                "metrics": {},
                "metric_sources": {},
                "samples": 0,
                "decision": "temporal_source_uncertainty_not_independently_validated",
                "temporal_tracker_summary": source_alignment.get("summary"),
                "measurement_uncertainty": source_alignment.get("measurement_uncertainty"),
                "policy_use": "forbidden: offline calibration evidence only",
            }
        source_pairs = _temporal_source_pairs(source_alignment)
        source_precision = source_alignment.get("measurement_uncertainty")
    elif is_foundationpose_track:
        if not _foundationpose_track_is_motion_validated(source_alignment):
            return {
                "schema_version": SCHEMA_VERSION,
                "source_episode_index": source_episode_index,
                "source_alignment_path": str(source_alignment_path),
                "replay_actions_path": str(replay_actions_path),
                "sim_trace_path": str(sim_trace_path),
                "metrics": {},
                "metric_sources": {},
                "samples": 0,
                "decision": "foundationpose_source_uncertainty_not_below_motion_gate",
                "foundationpose_gate": source_alignment.get("gate"),
                "policy_use": "forbidden: offline calibration evidence only",
            }
        source_pairs = _foundationpose_source_pairs(source_alignment_path, source_alignment)
        source_precision = {
            "method": "FoundationPose forward/reverse CAD tracking residual",
            "p95_bidirectional_translation_error_m": source_alignment["gate"][
                "p95_bidirectional_translation_error_m"
            ],
            "p95_bidirectional_rotation_error_deg": math.degrees(
                float(source_alignment["gate"]["p95_bidirectional_rotation_error_rad"])
            ),
            "strict_motion_gate": {
                "translation_p95_m": MAX_STEREO_TRANSLATION_P95_M,
                "rotation_p95_deg": MAX_STEREO_ROTATION_P95_DEG,
            },
        }
    else:
        # A static-scene alignment correctly rejects a moving table because its
        # temporal-spread gate is designed for reset-pose fitting.  Motion
        # comparison instead requires the underlying per-frame stereo gate and a
        # stricter measurement-noise ceiling below.  This never turns a dynamic
        # report into a reset candidate.
        stereo_agreement = source_alignment.get("stereo_agreement")
        if not isinstance(stereo_agreement, dict) or stereo_agreement.get("passes_internal_gate") is not True:
            raise ValueError("source CAD alignment did not pass its per-frame stereo gate")
        source_pairs = _source_pairs(source_alignment)
        source_precision = stereo_agreement
        if not _source_precision_is_sufficient(source_alignment):
            return {
                "schema_version": SCHEMA_VERSION,
                "source_episode_index": source_episode_index,
                "source_alignment_path": str(source_alignment_path),
                "replay_actions_path": str(replay_actions_path),
                "sim_trace_path": str(sim_trace_path),
                "metrics": {},
                "metric_sources": {},
                "samples": 0,
                "source_stereo_precision": stereo_agreement,
                "precision_thresholds": {
                    "accepted_translation_p95_m": MAX_STEREO_TRANSLATION_P95_M,
                    "accepted_rotation_p95_deg": MAX_STEREO_ROTATION_P95_DEG,
                },
                "decision": "insufficient_source_stereo_precision",
                "policy_use": "forbidden: offline calibration evidence only",
            }
    frame_map = _load_frame_map(replay_actions_path)
    sim_poses = _sim_root_from_table_by_step(sim_trace_path)
    joined: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for source_frame, source_pose in source_pairs:
        simulator_step = frame_map.get(source_frame)
        if simulator_step is None:
            continue
        sim_pose = sim_poses.get(simulator_step)
        if sim_pose is not None:
            joined.append((source_frame, simulator_step, source_pose, sim_pose))
    if len(joined) < MINIMUM_STEREO_PAIRS:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_episode_index": source_episode_index,
            "source_alignment_path": str(source_alignment_path),
            "replay_actions_path": str(replay_actions_path),
            "sim_trace_path": str(sim_trace_path),
            "metrics": {},
            "metric_sources": {},
            "samples": len(joined),
            "decision": "insufficient_paired_stereo_table_poses",
            "policy_use": "forbidden: offline calibration evidence only",
        }
    source_initial = joined[0][2]
    sim_initial = joined[0][3]
    first_source_time_s = joined[0][0] / SOURCE_FPS
    first_simulator_time_s = joined[0][1] / 50.0
    rows: list[dict[str, Any]] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for source_frame, simulator_step, source_pose, sim_pose in joined:
        source_relative = _relative(source_pose, source_initial)
        sim_relative = _canonical_against(_relative(sim_pose, sim_initial), source_relative)
        translation_error = float(np.linalg.norm(source_relative[:3, 3] - sim_relative[:3, 3]))
        rotation_error = _rotation_deg(source_relative, sim_relative)
        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)
        rows.append(
            {
                "source_frame": source_frame,
                "simulator_step": simulator_step,
                "source_time_s": source_frame / SOURCE_FPS,
                "simulator_time_s": simulator_step / 50.0,
                "source_elapsed_s": source_frame / SOURCE_FPS - first_source_time_s,
                "simulator_elapsed_s": simulator_step / 50.0 - first_simulator_time_s,
                "translation_from_start_m": float(np.linalg.norm(source_relative[:3, 3])),
                "rotation_from_start_deg": _rotation_deg(np.eye(4), source_relative),
                "sim_translation_from_start_m": float(np.linalg.norm(sim_relative[:3, 3])),
                "sim_rotation_from_start_deg": _rotation_deg(np.eye(4), sim_relative),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )
    source_events = _motion_events(rows, time_key="source_elapsed_s")
    sim_event_rows = [
        {
            **item,
            "translation_from_start_m": item["sim_translation_from_start_m"],
            "rotation_from_start_deg": item["sim_rotation_from_start_deg"],
        }
        for item in rows
    ]
    sim_events = _motion_events(sim_event_rows, time_key="simulator_elapsed_s")
    metrics: dict[str, float] = {
        "table_translation_rmse_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "table_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_errors)))),
    }
    if source_events is not None and sim_events is not None:
        metrics["phase_timing_max_error_s"] = max(
            abs(source_events["onset_s"] - sim_events["onset_s"]),
            abs(source_events["peak_s"] - sim_events["peak_s"]),
        )
    if is_foundationpose_track:
        tracker_name = "foundationpose_multiview_rgbd"
        source_description = (
            "hash-verified real multiview RGB/RGB-derived-depth CAD trajectory "
            "with strict forward/reverse residual gating versus fixed-base replay trace"
        )
    elif is_temporal_tracker:
        tracker_name = "temporal_stereo_cad"
        source_description = (
            "paired real head-stereo CAD trajectory versus fixed-base replay trace"
        )
    else:
        tracker_name = "static_stereo_cad"
        source_description = (
            "paired real head-stereo CAD trajectory versus fixed-base replay trace"
        )
    sources = {
        "table_translation_rmse_m": source_description
        + ", relative to first mutually visible pose",
        "table_rotation_rmse_deg": source_description
        + ", relative to first mutually visible pose",
    }
    if "phase_timing_max_error_s" in metrics:
        sources["phase_timing_max_error_s"] = (
            "first observed table-motion onset and peak from "
            + source_description
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_episode_index": source_episode_index,
        "source_alignment_path": str(source_alignment_path),
        "source_alignment_static_gate": bool(
            source_alignment.get("accepted_for_fixed_scene_proposal") is True
        ),
        "source_tracker": tracker_name,
        "source_measurement_precision": source_precision,
        "replay_actions_path": str(replay_actions_path),
        "sim_trace_path": str(sim_trace_path),
        "samples": len(rows),
        "rows": rows,
        "source_motion_events": source_events,
        "sim_motion_events": sim_events,
        "metrics": metrics,
        "metric_sources": sources,
        "decision": "measured" if "phase_timing_max_error_s" in metrics else "measured_phase_unobservable",
        "policy_use": "forbidden: offline calibration evidence only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--replay-actions", type=Path, required=True)
    parser.add_argument("--sim-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.source_alignment.expanduser().resolve(),
        args.replay_actions.expanduser().resolve(),
        args.sim_trace.expanduser().resolve(),
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
