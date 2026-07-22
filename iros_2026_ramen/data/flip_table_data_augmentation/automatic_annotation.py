"""Build audited table-pose and phase annotations from FoundationPose tracks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .annotations import ANNOTATION_SCHEMA_VERSION, SourceEpisodeAnnotation
from .config import AutomaticPhaseConfig, EXPECTED_SUBTASKS, PipelineConfig
from .io_utils import atomic_write_json, read_json_object, sha256_file
from .object_pose import TRACK_SCHEMA_VERSION
from .source_dataset import SourceDatasetIndex


PHASE_EVIDENCE_SCHEMA_VERSION = "team_ramen_flip_table_automatic_phase_evidence/v5"
BUILD_SCHEMA_VERSION = "team_ramen_flip_table_automatic_annotation_build/v7"
POSE_TRAJECTORY_SCHEMA_VERSION = "team_ramen_flip_table_pose_trajectory/v1"
ALGORITHM_ID = "foundationpose_lift_anchored_multi_eef_dex1/v4"
MINIMUM_TERMINAL_REFERENCE_FRAMES = 2
MINIMUM_SUSTAINED_LIFT_HEIGHT_M = 0.03


@dataclass(frozen=True)
class PhaseSegmentationResult:
    accepted: bool
    boundaries: tuple[int, ...] | None
    rejection_reasons: tuple[str, ...]
    diagnostics: dict[str, Any]

    def subtasks(self) -> dict[str, dict[str, list[int]]]:
        if not self.accepted or self.boundaries is None:
            raise ValueError("rejected phase segmentation has no subtask ranges")
        ranges = {
            name: [self.boundaries[index], self.boundaries[index + 1]]
            for index, name in enumerate(EXPECTED_SUBTASKS)
        }
        return {"left": dict(ranges), "right": dict(ranges)}

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "boundaries": None if self.boundaries is None else list(self.boundaries),
            "rejection_reasons": list(self.rejection_reasons),
            "diagnostics": self.diagnostics,
            "subtasks": None if not self.accepted else self.subtasks(),
        }


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if window <= 0 or window % 2 != 1:
        raise ValueError("moving-average window must be a positive odd integer")
    if len(array) < window:
        raise ValueError("trajectory is shorter than the smoothing window")
    radius = window // 2
    padded = np.pad(array, ((radius, radius), *[(0, 0)] * (array.ndim - 1)), mode="edge")
    prefix = np.concatenate(
        (np.zeros((1, *array.shape[1:]), dtype=np.float64), np.cumsum(padded, axis=0)),
        axis=0,
    )
    return (prefix[window:] - prefix[:-window]) / float(window)


def _unit(values: np.ndarray, label: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-9) or not np.isfinite(norms).all():
        raise ValueError(f"{label} cannot be normalized")
    return values / norms


def _smooth_rotations(rotations: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered chordal mean and project every result back to SO(3)."""

    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("rotation smoothing requires [T,3,3] matrices")
    averaged = _moving_average(values, window)
    left, _singular_values, right = np.linalg.svd(averaged)
    smoothed = left @ right
    reflected = np.linalg.det(smoothed) < 0.0
    if np.any(reflected):
        left[reflected, :, -1] *= -1.0
        smoothed[reflected] = left[reflected] @ right[reflected]
    return smoothed


def _validate_transforms(values: Any, frame_count: int) -> np.ndarray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.shape != (frame_count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"table poses must be finite [{frame_count},4,4]")
    if not np.allclose(poses[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError("table poses contain an invalid homogeneous row")
    rotations = poses[:, :3, :3]
    orthogonality = rotations.transpose(0, 2, 1) @ rotations
    if not np.allclose(orthogonality, np.eye(3), atol=1.0e-5):
        raise ValueError("table pose rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1.0e-5):
        raise ValueError("table pose rotations are not proper")
    return poses


def _first_sustained(
    mask: np.ndarray,
    *,
    start: int,
    stop: int,
    count: int,
) -> int | None:
    values = np.asarray(mask, dtype=bool)
    start = max(0, start)
    stop = min(len(values), stop)
    if count <= 0 or stop - start < count:
        return None
    run = 0
    for index in range(start, stop):
        run = run + 1 if values[index] else 0
        if run >= count:
            return index - count + 1
    return None


def _true_runs(mask: np.ndarray, *, stop: int) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask[:stop], dtype=bool)
    padded = np.concatenate(([False], values, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple((int(start), int(end)) for start, end in changes.reshape(-1, 2))


def _early_rejection(reason: str, diagnostics: dict[str, Any]) -> PhaseSegmentationResult:
    return PhaseSegmentationResult(False, None, (reason,), diagnostics)


def _rectify_static_prefix_drift(
    table_poses_root: np.ndarray,
    config: AutomaticPhaseConfig,
) -> tuple[np.ndarray, int | None, dict[str, Any]]:
    """Anchor the pre-lift table pose and preserve subsequent relative motion.

    FoundationPose can report coherent camera-relative drift while the table is
    still resting on the workbench. A flip cannot begin before a sustained
    upward displacement, so that physical event provides an auditable anchor.
    """

    table = np.asarray(table_poses_root, dtype=np.float64)
    frame_count = len(table)
    positions = _moving_average(table[:, :3, 3], config.smoothing_window_frames)
    initial_height = float(np.median(positions[: config.endpoint_window_frames, 2]))
    lifted = positions[:, 2] - initial_height >= MINIMUM_SUSTAINED_LIFT_HEIGHT_M
    lift_start = _first_sustained(
        lifted,
        start=config.minimum_phase_frames * 2,
        stop=frame_count - 5 * config.minimum_phase_frames,
        count=config.sustained_event_frames,
    )
    diagnostics = {
        "lift_height_threshold_m": MINIMUM_SUSTAINED_LIFT_HEIGHT_M,
        "maximum_upward_displacement_m": float(np.max(positions[:, 2] - initial_height)),
        "lift_anchor_frame": lift_start,
    }
    if lift_start is None:
        return table.copy(), None, diagnostics

    anchor = table[lift_start]
    correction = table[0] @ np.linalg.inv(anchor)
    rectified = table.copy()
    rectified[:lift_start] = table[0]
    rectified[lift_start:] = np.einsum("ij,tjk->tik", correction, table[lift_start:])
    translation_correction = correction[:3, 3]
    rotation_correction = math.acos(
        float(np.clip((np.trace(correction[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    )
    diagnostics.update(
        {
            "static_prefix_translation_correction_m": translation_correction.tolist(),
            "static_prefix_translation_correction_norm_m": float(
                np.linalg.norm(translation_correction)
            ),
            "static_prefix_rotation_correction_rad": rotation_correction,
        }
    )
    return rectified, lift_start, diagnostics


def _terminal_reference_start(
    positions: np.ndarray,
    normals: np.ndarray,
    config: AutomaticPhaseConfig,
) -> tuple[int, float, float]:
    """Find the terminal static suffix without averaging a cropped transition."""

    maximum_frames = min(config.endpoint_window_frames, len(positions))
    first_candidate = len(positions) - maximum_frames
    translation_error = np.linalg.norm(positions - positions[-1], axis=1)
    rotation_error = np.arccos(
        np.clip(normals @ normals[-1], -1.0, 1.0)
    )
    within_terminal_pose = (
        (translation_error <= config.motion_translation_threshold_m)
        & (rotation_error <= config.motion_rotation_threshold_rad)
    )
    start = len(positions) - 1
    while start > first_candidate and within_terminal_pose[start - 1]:
        start -= 1
    return (
        start,
        float(np.max(translation_error[start:])),
        float(np.max(rotation_error[start:])),
    )


def segment_flip_table_phases(
    *,
    table_poses_root: np.ndarray,
    eef_positions_root: np.ndarray,
    hand_commands: np.ndarray,
    fps: int,
    config: AutomaticPhaseConfig,
) -> PhaseSegmentationResult:
    """Derive seven ordered phases or return an explicit, auditable rejection."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_count = len(table_poses_root)
    table = _validate_transforms(table_poses_root, frame_count)
    eef = np.asarray(eef_positions_root, dtype=np.float64)
    hands = np.asarray(hand_commands, dtype=np.float64)
    if eef.shape != (frame_count, 2, 3) or not np.isfinite(eef).all():
        raise ValueError(f"EEF positions must be finite [{frame_count},2,3]")
    if hands.shape != (frame_count, 2) or not np.isfinite(hands).all():
        raise ValueError(f"Dex1 commands must be finite [{frame_count},2]")
    if np.any(hands < -1.0e-3) or np.any(hands > 4.5 + 1.0e-3):
        raise ValueError("Dex1 commands lie outside the measured [0,4.5] range")
    minimum = config.minimum_phase_frames
    if frame_count < 7 * minimum + 2 * config.endpoint_window_frames:
        raise ValueError("episode is too short for seven persistent phases")

    table, lift_start, lift_diagnostics = _rectify_static_prefix_drift(table, config)
    positions = _moving_average(table[:, :3, 3], config.smoothing_window_frames)
    smoothed_rotations = _smooth_rotations(
        table[:, :3, :3], config.smoothing_window_frames
    )
    normals = smoothed_rotations[:, :3, 2]
    smooth_eef = _moving_average(eef, config.smoothing_window_frames)
    smooth_hands = _moving_average(hands, config.smoothing_window_frames)
    endpoint = config.endpoint_window_frames
    initial_position = np.median(positions[:endpoint], axis=0)
    initial_normal = _unit(np.mean(normals[:endpoint], axis=0, keepdims=True), "initial normal")[0]
    terminal_start, terminal_translation_error, terminal_rotation_error = (
        _terminal_reference_start(positions, normals, config)
    )
    terminal_reference_frames = frame_count - terminal_start
    final_normal = _unit(
        np.mean(normals[terminal_start:], axis=0, keepdims=True),
        "final normal",
    )[0]
    flip_angle = float(math.acos(float(np.clip(np.dot(initial_normal, final_normal), -1.0, 1.0))))
    normal_angles = np.arccos(np.clip(normals @ initial_normal, -1.0, 1.0))
    displacement = np.linalg.norm(positions - initial_position, axis=1)
    maximum_displacement = float(np.max(displacement))
    progress = np.clip(normal_angles / max(flip_angle, 1.0e-9), 0.0, 1.25)
    dt = 1.0 / float(fps)
    linear_speed = np.linalg.norm(np.gradient(positions, dt, axis=0), axis=1)
    adjacent_dot = np.sum(normals[1:] * normals[:-1], axis=1)
    angular_step = np.arccos(np.clip(adjacent_dot, -1.0, 1.0))
    angular_speed = np.concatenate(([angular_step[0]], angular_step)) / dt
    diagnostics: dict[str, Any] = {
        "frame_count": frame_count,
        "fps": fps,
        "flip_angle_rad": flip_angle,
        "minimum_flip_angle_rad": config.minimum_flip_angle_rad,
        "maximum_table_displacement_m": maximum_displacement,
        "minimum_table_displacement_m": config.minimum_table_displacement_m,
        "terminal_reference_frames": terminal_reference_frames,
        "terminal_reference_frames_max": endpoint,
        "terminal_reference_frames_min": MINIMUM_TERMINAL_REFERENCE_FRAMES,
        "terminal_reference_translation_error_m_max": terminal_translation_error,
        "terminal_reference_rotation_error_rad_max": terminal_rotation_error,
        **lift_diagnostics,
    }
    if terminal_reference_frames < MINIMUM_TERMINAL_REFERENCE_FRAMES:
        return _early_rejection("unstable_terminal_endpoint", diagnostics)
    if flip_angle < config.minimum_flip_angle_rad:
        return _early_rejection("insufficient_table_normal_inversion", diagnostics)
    if maximum_displacement < config.minimum_table_displacement_m:
        return _early_rejection("insufficient_table_displacement", diagnostics)
    if lift_start is None:
        return _early_rejection("no_sustained_upward_lift", diagnostics)

    event = config.sustained_event_frames
    open_fraction = np.clip(smooth_hands / 4.5, 0.0, 1.0)
    both_closed_fraction = 1.0 - np.max(open_fraction, axis=1)
    grasp_active = both_closed_fraction >= config.grasp_closed_fraction_min
    grasp_runs = _true_runs(grasp_active, stop=frame_count - 5 * minimum)
    qualifying = []
    for run in grasp_runs:
        if run[0] < minimum or run[1] - run[0] < event:
            continue
        if (
            run[0] < lift_start
            and run[1] >= lift_start - 2 * event
            and lift_start - run[0] >= minimum
        ):
            qualifying.append(run)
    if not qualifying:
        diagnostics["maximum_bimanual_closed_fraction"] = float(
            np.max(both_closed_fraction)
        )
        return _early_rejection("no_bimanual_grasp_before_table_motion", diagnostics)
    grasp_run = max(qualifying, key=lambda run: (run[1], run[0]))
    grasp_start = grasp_run[0]

    rotate_start = _first_sustained(
        progress >= config.rotate_start_fraction,
        start=lift_start + minimum,
        stop=frame_count - 4 * minimum,
        count=event,
    )
    if rotate_start is None:
        diagnostics.update({"grasp_start": grasp_start, "lift_start": lift_start})
        return _early_rejection("no_rotation_start", diagnostics)
    rotate_end = _first_sustained(
        progress >= config.rotate_end_fraction,
        start=rotate_start + minimum,
        stop=frame_count - 3 * minimum,
        count=event,
    )
    if rotate_end is None:
        diagnostics.update(
            {"grasp_start": grasp_start, "lift_start": lift_start, "rotate_start": rotate_start}
        )
        return _early_rejection("no_rotation_end", diagnostics)

    preliminary_open_fraction = open_fraction[lift_start:rotate_end]
    preliminary_active_coverage = np.any(
        1.0 - preliminary_open_fraction >= config.grasp_closed_fraction_min,
        axis=1,
    )
    if not np.all(preliminary_active_coverage):
        diagnostics.update(
            {
                "grasp_start": grasp_start,
                "lift_start": lift_start,
                "rotate_start": rotate_start,
                "rotate_end": rotate_end,
                "manipulation_active_eef_coverage_fraction": float(
                    np.mean(preliminary_active_coverage)
                ),
            }
        )
        return _early_rejection("manipulation_without_active_eef", diagnostics)

    relative = np.einsum(
        "tji,tsj->tsi",
        smoothed_rotations,
        smooth_eef - positions[:, None, :],
    )
    reference_start = max(rotate_start, rotate_end - event)
    release_reference = np.median(relative[reference_start:rotate_end], axis=0)
    relative_displacement = np.linalg.norm(relative - release_reference[None], axis=2)
    both_open_fraction = np.min(open_fraction, axis=1)
    table_settled = (
        (linear_speed <= config.settled_linear_speed_m_s_max)
        & (angular_speed <= config.settled_angular_speed_rad_s_max)
    )
    release_runs_raw = _true_runs(
        both_open_fraction >= config.release_open_fraction_min,
        stop=frame_count - minimum,
    )
    release_search_start = rotate_end + minimum
    release_runs = tuple(
        (max(run[0], release_search_start), run[1])
        for run in release_runs_raw
        if run[1] - max(run[0], release_search_start) >= event
    )
    if not release_runs:
        diagnostics.update(
            {
                "grasp_start": grasp_start,
                "lift_start": lift_start,
                "rotate_start": rotate_start,
                "rotate_end": rotate_end,
                "maximum_bimanual_open_fraction_after_rotation": float(
                    np.max(both_open_fraction[rotate_end:])
                ),
            }
        )
        return _early_rejection("no_settled_release", diagnostics)
    release_start = release_runs[-1][0]
    settled_start = _first_sustained(
        table_settled,
        start=rotate_end,
        stop=release_start,
        count=minimum,
    )
    if settled_start is None:
        diagnostics.update(
            {
                "grasp_start": grasp_start,
                "lift_start": lift_start,
                "rotate_start": rotate_start,
                "rotate_end": rotate_end,
                "release_start": release_start,
            }
        )
        return _early_rejection("no_settled_interval_before_release", diagnostics)
    retreat_evidence = (
        (both_open_fraction >= config.release_open_fraction_min)
        & (
            np.min(relative_displacement, axis=1)
            >= config.retreat_relative_displacement_m_min
        )
    )
    retreat_start = _first_sustained(
        retreat_evidence,
        start=release_start + minimum,
        stop=frame_count - minimum + 1,
        count=event,
    )
    if retreat_start is None:
        diagnostics.update(
            {
                "grasp_start": grasp_start,
                "lift_start": lift_start,
                "rotate_start": rotate_start,
                "rotate_end": rotate_end,
                "release_start": release_start,
                "maximum_minimum_eef_retreat_m": float(
                    np.max(np.min(relative_displacement[release_start:], axis=1))
                ),
            }
        )
        return _early_rejection("no_bimanual_retreat", diagnostics)

    boundaries = (
        0,
        grasp_start,
        lift_start,
        rotate_start,
        rotate_end,
        release_start,
        retreat_start,
        frame_count,
    )
    durations = np.diff(boundaries)
    if np.any(durations < minimum):
        diagnostics["boundaries"] = list(boundaries)
        diagnostics["phase_durations_frames"] = durations.tolist()
        return _early_rejection("phase_shorter_than_minimum", diagnostics)

    manipulation_slice = slice(lift_start, rotate_end)
    manipulation_open_fraction = open_fraction[manipulation_slice]
    active_hands = (
        1.0 - manipulation_open_fraction >= config.grasp_closed_fraction_min
    )
    rigid_hold_hands = manipulation_open_fraction <= config.release_open_fraction_min
    active_coverage = np.any(active_hands, axis=1)
    bimanual_overlap = np.all(active_hands, axis=1)
    sustained_overlap = _first_sustained(
        bimanual_overlap,
        start=0,
        stop=len(bimanual_overlap),
        count=minimum,
    )
    active_hold_drift_p95: list[float | None] = []
    active_frame_ranges = []
    for side_index in range(2):
        side_active = rigid_hold_hands[:, side_index]
        active_indices = np.flatnonzero(side_active)
        active_frame_ranges.append(
            None
            if len(active_indices) == 0
            else [
                int(lift_start + active_indices[0]),
                int(lift_start + active_indices[-1] + 1),
            ]
        )
        if len(active_indices) < event:
            active_hold_drift_p95.append(None)
            continue
        side_relative = relative[manipulation_slice, side_index][side_active]
        side_reference = np.median(side_relative, axis=0)
        side_drift = np.linalg.norm(side_relative - side_reference[None], axis=1)
        active_hold_drift_p95.append(float(np.quantile(side_drift, 0.95)))
    measured_hold_drift = [
        value for value in active_hold_drift_p95 if value is not None
    ]
    hold_drift_p95 = max(measured_hold_drift) if measured_hold_drift else None
    rotation_delta = np.diff(progress[lift_start : rotate_end + 1])
    rotation_variation = float(np.sum(np.abs(rotation_delta)))
    reversal_fraction = (
        float(np.sum(np.maximum(-rotation_delta, 0.0)) / rotation_variation)
        if rotation_variation > 1.0e-9
        else 1.0
    )
    reasons = []
    if not np.all(active_coverage):
        reasons.append("manipulation_without_active_eef")
    if sustained_overlap is None:
        reasons.append("no_sustained_bimanual_handoff")
    if len(measured_hold_drift) != 2:
        reasons.append("insufficient_active_eef_hold_samples")
    elif hold_drift_p95 is not None and (
        hold_drift_p95 > config.hold_relative_position_p95_m_max
    ):
        reasons.append("active_eef_hold_drift")
    if reversal_fraction > config.rotation_progress_reversal_fraction_max:
        reasons.append("non_monotonic_table_rotation")
    diagnostics.update(
        {
            "algorithm_id": ALGORITHM_ID,
            "boundaries": list(boundaries),
            "phase_durations_frames": durations.tolist(),
            "phase_durations_s": (durations / float(fps)).tolist(),
            "hold_relative_position_p95_m": hold_drift_p95,
            "hold_relative_position_p95_m_max": config.hold_relative_position_p95_m_max,
            "active_hold_relative_position_p95_m_by_eef": {
                "left": active_hold_drift_p95[0],
                "right": active_hold_drift_p95[1],
            },
            "active_hold_frame_range_by_eef": {
                "left": active_frame_ranges[0],
                "right": active_frame_ranges[1],
            },
            "manipulation_active_eef_coverage_fraction": float(np.mean(active_coverage)),
            "bimanual_handoff_overlap_frames": int(np.count_nonzero(bimanual_overlap)),
            "minimum_bimanual_handoff_overlap_frames": minimum,
            "rotation_progress_reversal_fraction": reversal_fraction,
            "rotation_progress_reversal_fraction_max": (
                config.rotation_progress_reversal_fraction_max
            ),
            "maximum_bimanual_closed_fraction": float(np.max(both_closed_fraction)),
            "settled_start": settled_start,
            "final_bimanual_open_fraction": float(
                np.median(both_open_fraction[terminal_start:])
            ),
            "maximum_minimum_eef_retreat_m": float(
                np.max(np.min(relative_displacement[release_start:], axis=1))
            ),
        }
    )
    return PhaseSegmentationResult(not reasons, boundaries if not reasons else None, tuple(reasons), diagnostics)


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-9 or not math.isfinite(norm):
        raise ValueError("rotation matrix produced an invalid quaternion")
    return quaternion / norm


def matrices_to_pose7(transforms: np.ndarray) -> list[list[float]]:
    matrices = _validate_transforms(transforms, len(transforms))
    output = np.empty((len(matrices), 7), dtype=np.float64)
    output[:, :3] = matrices[:, :3, 3]
    for index, matrix in enumerate(matrices):
        output[index, 3:] = _matrix_to_quaternion_xyzw(matrix[:3, :3])
        if index == 0:
            if output[index, 6] < 0.0:
                output[index, 3:] *= -1.0
        elif float(np.dot(output[index - 1, 3:], output[index, 3:])) < 0.0:
            output[index, 3:] *= -1.0
    return output.tolist()


def _load_track(track_root: Path, config: PipelineConfig) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = track_root / "manifest.json"
    manifest = read_json_object(manifest_path, label="FoundationPose track manifest")
    expected = {
        "schema_version": TRACK_SCHEMA_VERSION,
        "config_sha256": config.digest,
        "source_revision": config.source.revision,
        "method": config.object_pose_runtime.method,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("FoundationPose track differs from the active source/config contract")
    gate = manifest.get("gate")
    if not isinstance(gate, dict) or gate.get("pass") is not True:
        raise ValueError("FoundationPose track did not pass all residual gates")
    arrays = manifest.get("arrays")
    record = arrays.get("table_pose_root_30hz") if isinstance(arrays, dict) else None
    if not isinstance(record, dict):
        raise ValueError("FoundationPose track lacks the full-rate table-pose array")
    relative = record.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("FoundationPose table-pose path must be a local basename")
    path = track_root / relative
    if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
        raise ValueError("FoundationPose table-pose array differs from its manifest")
    poses = np.load(path, allow_pickle=False)
    expected_shape = (int(manifest.get("source_frame_count", -1)), 4, 4)
    if list(poses.shape) != record.get("shape") or poses.shape != expected_shape:
        raise ValueError("FoundationPose table-pose array shape differs from its manifest")
    if str(poses.dtype) != record.get("dtype") or poses.dtype != np.float64:
        raise ValueError("FoundationPose table-pose array must be float64")
    return manifest, _validate_transforms(poses, expected_shape[0])


def _load_source_signals(source_root: Path, episode_index: int) -> tuple[Any, np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to build automatic source annotations") from exc

    episode = SourceDatasetIndex(source_root).episode(episode_index)
    table = pq.read_table(
        episode.data_path,
        columns=["frame_index", "observation.state.ee_state", "action.hand_cmd"],
        filters=[("episode_index", "=", episode_index)],
    ).sort_by([("frame_index", "ascending")])
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frames, np.arange(episode.frame_count)):
        raise ValueError("source episode frame indices are not contiguous")
    eef = np.asarray(table["observation.state.ee_state"].to_pylist(), dtype=np.float64)
    hands = np.asarray(table["action.hand_cmd"].to_pylist(), dtype=np.float64)
    if eef.shape != (episode.frame_count, 12) or not np.isfinite(eef).all():
        raise ValueError("source ee_state must be finite [T,12]")
    return episode, np.stack((eef[:, :3], eef[:, 6:9]), axis=1), hands


def _track_quality_metrics(manifest: dict[str, Any], config: PipelineConfig) -> list[dict[str, Any]]:
    gate = manifest["gate"]
    frame_records = manifest.get("frames")
    frame_count = int(manifest["source_frame_count"])
    expected_indices = list(
        range(0, frame_count, config.object_pose_runtime.source_frame_stride)
    )
    if expected_indices[-1] != frame_count - 1:
        expected_indices.append(frame_count - 1)
    if not isinstance(frame_records, list) or len(frame_records) != len(expected_indices):
        raise ValueError("FoundationPose frame diagnostics do not cover sampled input")
    observed_indices = [
        record.get("source_frame_index") if isinstance(record, dict) else None
        for record in frame_records
    ]
    if observed_indices != expected_indices:
        raise ValueError("FoundationPose frame diagnostics are not the expected ordered samples")
    static_depth_errors = []
    required_pose_evidence = 0
    for record in frame_records:
        if not isinstance(record, dict):
            raise ValueError("FoundationPose frame diagnostic is malformed")
        alignment = record.get("rendered_alignment")
        evidence = record.get("pose_evidence")
        if not isinstance(alignment, dict) or not isinstance(evidence, dict):
            raise ValueError("FoundationPose frame lacks pose-evidence diagnostics")
        if evidence.get("required") is True:
            required_pose_evidence += 1
            if evidence.get("passes_gate") is not True:
                raise ValueError("FoundationPose registration anchor did not pass")
            if evidence.get("mode") == "static_rgbd_bidirectional":
                if alignment.get("passes_gate") is not True:
                    raise ValueError("FoundationPose static RGB-D anchor did not pass")
                value = alignment.get("median_absolute_depth_error_m")
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("FoundationPose static rendered depth error is missing")
                static_depth_errors.append(float(value))
        elif (
            evidence.get("required") is not False
            or evidence.get("passes_gate") is not None
            or evidence.get("mode") != "interpolation_between_audited_anchors"
        ):
            raise ValueError("FoundationPose interpolation evidence is malformed")
    if (
        required_pose_evidence != int(gate.get("pose_evidence_required_anchors", -1))
        or required_pose_evidence != int(gate.get("pose_evidence_passed_anchors", -1))
        or len(static_depth_errors)
        != int(gate.get("rendered_alignment_required_frames", -1))
        or len(static_depth_errors)
        != int(gate.get("rendered_alignment_passed_frames", -1))
        or not static_depth_errors
    ):
        raise ValueError("FoundationPose pose-evidence gate counts are inconsistent")
    pose = config.object_pose_runtime
    return [
        {
            "name": "bidirectional_translation_p95",
            "value": float(gate["p95_bidirectional_translation_error_m"]),
            "unit": "m",
            "maximum": pose.max_bidirectional_translation_error_m,
        },
        {
            "name": "bidirectional_rotation_p95",
            "value": float(gate["p95_bidirectional_rotation_error_rad"]),
            "unit": "rad",
            "maximum": pose.max_bidirectional_rotation_error_rad,
        },
        {
            "name": "static_rendered_depth_median_error_p95",
            "value": float(np.quantile(static_depth_errors, 0.95)),
            "unit": "m",
            "maximum": pose.max_rendered_depth_median_abs_error_m,
        },
    ]


def build_automatic_source_annotation(
    *,
    source_root: str | Path,
    track_dir: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    resume: bool = False,
) -> dict[str, Any]:
    """Build one annotation atomically, preserving a rejection artifact on failure."""

    source = Path(source_root).expanduser().resolve()
    track = Path(track_dir).expanduser().resolve()
    track_manifest, table_poses = _load_track(track, config)
    episode_index = int(track_manifest.get("episode_index", -1))
    episode, eef_positions, hand_commands = _load_source_signals(source, episode_index)
    if episode.frame_count != len(table_poses):
        raise ValueError("source and FoundationPose frame counts differ")
    identity = {
        "config_sha256": config.digest,
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "source_episode_index": episode_index,
        "source_data_sha256": sha256_file(episode.data_path),
        "foundationpose_track_manifest_sha256": sha256_file(track / "manifest.json"),
    }
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if output.exists():
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f"annotation output already exists: {output}")
        previous = read_json_object(manifest_path, label="automatic annotation manifest")
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing automatic annotation uses a different immutable contract")
        return previous

    result = segment_flip_table_phases(
        table_poses_root=table_poses,
        eef_positions_root=eef_positions,
        hand_commands=hand_commands,
        fps=config.source.fps,
        config=config.source_annotation.automatic_phase,
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        phase_evidence = {
            "schema_version": PHASE_EVIDENCE_SCHEMA_VERSION,
            **identity,
            "algorithm_id": ALGORITHM_ID,
            "offline_teacher_only": True,
            "frame_count": episode.frame_count,
            **result.to_json(),
        }
        phase_path = temporary / "phase_evidence.json"
        atomic_write_json(phase_path, phase_evidence)
        build_manifest: dict[str, Any] = {
            "schema_version": BUILD_SCHEMA_VERSION,
            **identity,
            "frame_count": episode.frame_count,
            "accepted": result.accepted,
            "rejection_reasons": list(result.rejection_reasons),
            "gate": {"pass": result.accepted},
            "phase_evidence": {
                "path": phase_path.name,
                "sha256": sha256_file(phase_path),
            },
        }
        if result.accepted:
            quality_metrics = _track_quality_metrics(track_manifest, config)
            pose_trajectory = {
                "schema_version": POSE_TRAJECTORY_SCHEMA_VERSION,
                "offline_teacher_only": True,
                "source_repo_id": config.source.repo_id,
                "source_revision": config.source.revision,
                "source_episode_index": episode_index,
                "frame_count": episode.frame_count,
                "subtask_review_sha256": sha256_file(phase_path),
                "method": "pinned_grounded_sam2_foundationpose_bidirectional_rgbd",
                "pose_reviewer": f"automatic:{ALGORITHM_ID}",
                "foundationpose_track_manifest_sha256": identity[
                    "foundationpose_track_manifest_sha256"
                ],
                "quality_metrics": quality_metrics,
                "acceptance_eligible": True,
                "poses_robot_root_xyzw": matrices_to_pose7(
                    _rectify_static_prefix_drift(
                        table_poses,
                        config.source_annotation.automatic_phase,
                    )[0]
                ),
                "limitations": (
                    "Offline Mimic teacher signal only. Table pose, masks, depth, phase "
                    "features, and residuals are forbidden policy observations."
                ),
            }
            trajectory_path = temporary / "pose_trajectory.json"
            atomic_write_json(trajectory_path, pose_trajectory)
            annotation_value = {
                "episode_index": episode_index,
                "frame_count": episode.frame_count,
                "table_pose_trajectory_robot_root_xyzw": pose_trajectory[
                    "poses_robot_root_xyzw"
                ],
                "pose_evidence": {
                    "method": pose_trajectory["method"],
                    "reviewer": pose_trajectory["pose_reviewer"],
                    "calibration_artifact_sha256": sha256_file(trajectory_path),
                    "quality_metrics": quality_metrics,
                },
                "subtask_reviewer": f"automatic:{ALGORITHM_ID}",
                "subtask_evidence_sha256": sha256_file(phase_path),
                "subtasks": result.subtasks(),
            }
            SourceEpisodeAnnotation.from_json(annotation_value)
            annotation = {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "episodes": [annotation_value],
            }
            annotation_path = temporary / "annotation.json"
            atomic_write_json(annotation_path, annotation)
            build_manifest.update(
                {
                    "pose_trajectory": {
                        "path": trajectory_path.name,
                        "sha256": sha256_file(trajectory_path),
                    },
                    "annotation": {
                        "path": annotation_path.name,
                        "sha256": sha256_file(annotation_path),
                    },
                }
            )
        atomic_write_json(temporary / "manifest.json", build_manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return build_manifest


def merge_automatic_annotations(
    annotation_paths: list[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge accepted one-episode artifacts into one sorted immutable annotation set."""

    episodes = []
    sources = []
    for value in annotation_paths:
        path = Path(value).expanduser().resolve()
        payload = read_json_object(path, label="source annotation")
        if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported annotation schema: {path}")
        values = payload.get("episodes")
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"automatic annotation must contain one episode: {path}")
        SourceEpisodeAnnotation.from_json(values[0])
        episodes.append(values[0])
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    episodes.sort(key=lambda item: int(item["episode_index"]))
    indices = [int(item["episode_index"]) for item in episodes]
    if not indices or indices != sorted(set(indices)):
        raise ValueError("automatic annotations must be non-empty and unique by episode")
    payload = {"schema_version": ANNOTATION_SCHEMA_VERSION, "episodes": episodes}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    atomic_write_json(temporary, payload)
    os.replace(temporary, output)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "episodes": len(episodes),
        "episode_indices": indices,
        "sources": sources,
    }
