from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.flip_table_data_augmentation.annotations import load_annotations
from data.flip_table_data_augmentation.automatic_annotation import (
    TRACK_SCHEMA_VERSION,
    build_automatic_source_annotation,
    matrices_to_pose7,
    segment_flip_table_phases,
)
from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.io_utils import sha256_file


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _synthetic_episode(frame_count: int = 210):
    table = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    table[:, :3, 3] = (0.45, 0.0, 0.75)
    for frame in range(50, 80):
        alpha = (frame - 50) / 29.0
        table[frame:, 2, 3] = 0.75 + 0.12 * alpha
    for frame in range(80, 141):
        alpha = (frame - 80) / 60.0
        table[frame, :3, :3] = _rotation_y(np.pi * alpha)
    table[141:, :3, :3] = _rotation_y(np.pi)

    relative = np.asarray([[0.0, 0.22, -0.10], [0.0, -0.22, -0.10]])
    eef = np.empty((frame_count, 2, 3), dtype=np.float64)
    for frame in range(frame_count):
        eef[frame] = table[frame, :3, 3] + np.einsum(
            "ij,sj->si", table[frame, :3, :3], relative
        )
    retreat = np.clip((np.arange(frame_count) - 178) / 20.0, 0.0, 1.0)
    eef[:, 0, 0] -= 0.08 * retreat
    eef[:, 1, 0] -= 0.08 * retreat
    eef[:, 0, 1] += 0.04 * retreat
    eef[:, 1, 1] -= 0.04 * retreat

    hands = np.full((frame_count, 2), 4.5, dtype=np.float64)
    hands[28:46] = np.linspace(4.5, 0.4, 18)[:, None]
    hands[46:158] = 0.4
    hands[158:178] = np.linspace(0.4, 4.5, 20)[:, None]
    return table, eef, hands


def test_phase_segmentation_finds_ordered_persistent_events() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )
    assert result.accepted, result.to_json()
    assert result.boundaries is not None
    assert result.boundaries[0] == 0
    assert result.boundaries[-1] == len(table)
    assert all(left < right for left, right in zip(result.boundaries, result.boundaries[1:]))
    assert result.subtasks()["left"] == result.subtasks()["right"]
    assert 20 <= result.subtasks()["left"]["grasp"][0] <= 40
    assert 75 <= result.subtasks()["left"]["rotate_180"][0] <= 100


def test_phase_segmentation_rejects_non_flip_without_guessing() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    table[:, :3, :3] = np.eye(3)
    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )
    assert not result.accepted
    assert result.rejection_reasons == ("insufficient_table_normal_inversion",)


def test_phase_segmentation_ignores_pregrasp_pose_jitter() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    table[10:20, 0, 3] += 0.03

    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )

    assert result.accepted, result.to_json()
    assert result.boundaries is not None
    assert result.boundaries[2] >= 45


def test_phase_segmentation_anchors_static_prefix_drift_to_upward_lift() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    table[10:50, 0, 3] += np.linspace(0.0, 0.15, 40)
    table[50:, 0, 3] += 0.15

    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )

    assert result.accepted, result.to_json()
    assert result.diagnostics["lift_anchor_frame"] >= 50
    assert result.diagnostics["static_prefix_translation_correction_norm_m"] > 0.1
    assert result.boundaries is not None
    assert result.boundaries[2] == result.diagnostics["lift_anchor_frame"]


def test_phase_segmentation_uses_cropped_terminal_static_suffix() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    transition_start = len(table) - config.endpoint_window_frames
    transition_end = len(table) - 2
    for frame in range(transition_start, transition_end):
        progress = (frame - transition_start) / (transition_end - transition_start)
        table[frame, :3, :3] = _rotation_y(0.5 * np.pi * (1.0 + progress))
    table[transition_end:, :3, :3] = _rotation_y(np.pi)

    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )

    assert result.accepted, result.to_json()
    assert 2 <= result.diagnostics["terminal_reference_frames"] < (
        config.endpoint_window_frames
    )
    assert result.diagnostics["flip_angle_rad"] >= config.minimum_flip_angle_rad


def test_phase_segmentation_accepts_audited_bimanual_handoff() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    hands[100:120, 0] = np.linspace(0.4, 4.5, 20)
    hands[120:, 0] = 4.5
    left_retreat = np.clip((np.arange(len(table)) - 120) / 20.0, 0.0, 1.0)
    eef[:, 0, 0] -= 0.30 * left_retreat

    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )

    assert result.accepted, result.to_json()
    diagnostics = result.diagnostics
    assert diagnostics["manipulation_active_eef_coverage_fraction"] == 1.0
    assert diagnostics["bimanual_handoff_overlap_frames"] >= 6
    assert diagnostics["active_hold_frame_range_by_eef"]["left"][1] < (
        diagnostics["active_hold_frame_range_by_eef"]["right"][1]
    )


def test_phase_segmentation_rejects_unheld_rotation_interval() -> None:
    config = load_pipeline_config().source_annotation.automatic_phase
    table, eef, hands = _synthetic_episode()
    hands[105:] = 4.5

    result = segment_flip_table_phases(
        table_poses_root=table,
        eef_positions_root=eef,
        hand_commands=hands,
        fps=30,
        config=config,
    )

    assert not result.accepted
    assert "manipulation_without_active_eef" in result.rejection_reasons


def test_pose7_quaternion_is_continuous_and_xyzw() -> None:
    table, _eef, _hands = _synthetic_episode(210)
    poses = np.asarray(matrices_to_pose7(table))
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1.0e-8)
    assert np.all(np.sum(poses[:-1, 3:] * poses[1:, 3:], axis=1) >= 0.0)
    np.testing.assert_allclose(poses[0, 3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8)


def test_automatic_builder_binds_track_source_and_phase_hashes() -> None:
    config = load_pipeline_config()
    table, eef, hands = _synthetic_episode()
    frame_count = len(table)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        track = root / "track"
        output = root / "annotation"
        (source / "meta/episodes/chunk-000").mkdir(parents=True)
        (source / "data/chunk-000").mkdir(parents=True)
        track.mkdir()
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0],
                    "length": [frame_count],
                    "data/chunk_index": [0],
                    "data/file_index": [0],
                }
            ),
            source / "meta/episodes/chunk-000/file-000.parquet",
        )
        eef_rows = np.concatenate(
            (
                eef[:, 0],
                np.zeros((frame_count, 3)),
                eef[:, 1],
                np.zeros((frame_count, 3)),
            ),
            axis=1,
        ).tolist()
        pq.write_table(
            pa.table(
                {
                    "episode_index": [0] * frame_count,
                    "frame_index": list(range(frame_count)),
                    "observation.state.ee_state": eef_rows,
                    "action.hand_cmd": hands.tolist(),
                }
            ),
            source / "data/chunk-000/file-000.parquet",
        )
        pose_path = track / "table_pose_root_30hz.npy"
        np.save(pose_path, table, allow_pickle=False)
        sampled = tuple(range(0, frame_count, config.object_pose_runtime.source_frame_stride))
        if sampled[-1] != frame_count - 1:
            sampled = (*sampled, frame_count - 1)
        frame_records = [
            {
                "ordinal": ordinal,
                "source_frame_index": frame,
                "rendered_alignment": {
                    "passes_gate": True,
                    "median_absolute_depth_error_m": 0.005,
                },
                "pose_evidence": {
                    "required": True,
                    "passes_gate": True,
                    "mode": "static_rgbd_bidirectional",
                },
            }
            for ordinal, frame in enumerate(sampled)
        ]
        track_manifest = {
            "schema_version": TRACK_SCHEMA_VERSION,
            "config_sha256": config.digest,
            "source_revision": config.source.revision,
            "episode_index": 0,
            "source_frame_count": frame_count,
            "method": config.object_pose_runtime.method,
            "gate": {
                "pass": True,
                "p95_bidirectional_translation_error_m": 0.005,
                "p95_bidirectional_rotation_error_rad": 0.01,
                "pose_evidence_required_anchors": len(sampled),
                "pose_evidence_passed_anchors": len(sampled),
                "rendered_alignment_required_frames": len(sampled),
                "rendered_alignment_passed_frames": len(sampled),
            },
            "arrays": {
                "table_pose_root_30hz": {
                    "path": pose_path.name,
                    "shape": list(table.shape),
                    "dtype": str(table.dtype),
                    "size_bytes": pose_path.stat().st_size,
                    "sha256": sha256_file(pose_path),
                }
            },
            "frames": frame_records,
        }
        (track / "manifest.json").write_text(
            json.dumps(track_manifest), encoding="utf-8"
        )
        report = build_automatic_source_annotation(
            source_root=source,
            track_dir=track,
            output_dir=output,
            config=config,
        )
        assert report["accepted"] is True
        annotation = load_annotations(output / "annotation.json")[0]
        assert annotation.episode_index == 0
        assert annotation.subtask_evidence_sha256 == report["phase_evidence"]["sha256"]
        assert annotation.pose_evidence.calibration_artifact_sha256 == report["pose_trajectory"]["sha256"]

        track_manifest["frames"][1]["source_frame_index"] += 1
        (track / "manifest.json").write_text(
            json.dumps(track_manifest), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="expected ordered samples"):
            build_automatic_source_annotation(
                source_root=source,
                track_dir=track,
                output_dir=root / "invalid-annotation",
                config=config,
            )
