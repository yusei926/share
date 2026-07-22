from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate.flip_table_simulation.real_to_sim_calibration.wrist_handeye_cad_alignment import (
    Observation,
    _mask_metrics,
    accepted_root_from_table,
)
from evaluate.flip_table_simulation.real_to_sim_calibration.wrist_handeye_consensus import consensus
from evaluate.flip_table_simulation.real_to_sim_calibration.wrist_handeye_calibration import (
    _camera_from_table_candidates,
    _robust_mean,
    _rotation_distance_deg,
)
from evaluate.flip_table_simulation.real_to_sim_calibration.state_timing_audit import (
    joint_command_timing_report,
    timing_report,
)


def _transform(translation: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    value[:3, 3] = translation
    return value


def test_wrist_handeye_pnp_keeps_the_physical_rectangle_solution() -> None:
    intrinsic = np.asarray(((435.0, 0.0, 320.0), (0.0, 435.0, 240.0), (0.0, 0.0, 1.0)))
    expected = _transform((0.03, -0.02, 0.72), (172.0, 4.0, 11.0))
    table_corners = np.asarray(
        ((-0.29, -0.21, 0.0), (0.29, -0.21, 0.0), (0.29, 0.21, 0.0), (-0.29, 0.21, 0.0)),
        dtype=np.float64,
    )
    rotation_vector, _ = cv2.Rodrigues(expected[:3, :3])
    pixels, _ = cv2.projectPoints(
        table_corners,
        rotation_vector,
        expected[:3, 3],
        intrinsic,
        np.zeros(5, dtype=np.float64),
    )
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 255)

    candidates = _camera_from_table_candidates(mask, intrinsic, np.zeros(5))

    assert any(
        np.linalg.norm(candidate[:3, 3] - expected[:3, 3]) < 0.002
        and _rotation_distance_deg(expected, candidate) < 2.0
        for candidate in candidates
    )


def test_wrist_handeye_robust_mean_rejects_a_single_outlier() -> None:
    expected = _transform((0.115, 0.0, 0.070), (-135.0, 1.0, -87.0))
    candidates = [
        expected,
        _transform((0.116, -0.001, 0.071), (-134.5, 1.2, -86.8)),
        _transform((0.114, 0.001, 0.069), (-135.4, 0.8, -87.2)),
        _transform((0.42, -0.30, 0.33), (35.0, -42.0, 90.0)),
    ]

    fitted, inliers = _robust_mean(candidates)

    assert inliers.tolist() == [True, True, True, False]
    assert np.linalg.norm(fitted[:3, 3] - expected[:3, 3]) < 0.003
    assert _rotation_distance_deg(expected, fitted) < 1.0


def test_wrist_handeye_cad_rejects_unaccepted_head_stereo_reference() -> None:
    try:
        accepted_root_from_table(
            {
                "schema_version": "team_ramen_flip_table_source_cad_alignment/v1",
                "accepted_for_fixed_scene_proposal": False,
                "fixed_scene_root_from_table": np.eye(4).tolist(),
            }
        )
    except ValueError as error:
        assert "stereo-CAD consistency gate" in str(error)
    else:
        raise AssertionError("unaccepted head-stereo alignment was not rejected")


def test_wrist_handeye_cad_rejects_a_weak_stereo_reference() -> None:
    alignment = {
        "schema_version": "team_ramen_flip_table_source_cad_alignment/v1",
        "accepted_for_fixed_scene_proposal": True,
        "fixed_scene_root_from_table": np.eye(4).tolist(),
        "stereo_agreement": {
            "accepted_paired_frames": 3,
            "accepted_translation_p95_m": 0.021,
            "accepted_rotation_p95_deg": 1.0,
        },
        "temporal_consistency": {
            "translation_spread_p95_m": 0.005,
            "rotation_spread_p95_deg": 1.0,
        },
    }

    try:
        accepted_root_from_table(alignment)
    except ValueError as error:
        assert "too uncertain" in str(error)
        assert "accepted_translation_p95_m" in str(error)
    else:
        raise AssertionError("weak head-stereo alignment was not rejected")


def test_wrist_handeye_consensus_rejects_reports_without_accepted_reference() -> None:
    transform = np.eye(4).tolist()
    report = {
        "schema_version": "team_ramen_flip_table_wrist_handeye_cad_alignment/v1",
        "source_inputs": {"source_alignment_accepted": False},
        "sides": {
            side: {
                "status": "proposal_requires_heldout_validation",
                "fitted_wrist_from_rectified_opencv_camera": transform,
            }
            for side in ("left", "right")
        },
    }

    result = consensus([(Path("first.json"), report), (Path("second.json"), report)])

    assert result["accepted_for_heldout_validation"] is False
    assert result["sides"]["left"]["status"] == "rejected_unaccepted_head_stereo_reference"


def test_wrist_handeye_reports_robot_overlap_relative_to_the_observed_table() -> None:
    observed = np.zeros((480, 640), dtype=bool)
    observed[100:200, 100:200] = True
    robot = np.zeros_like(observed)
    robot[100:150, 100:200] = True
    depth = np.zeros((1, 480, 640), dtype=np.float32)
    depth[0, 100:200, 100:200] = 1.0
    observation = Observation(
        ordinal=0,
        source_frame_index=0,
        rgb_path=Path("unused.png"),
        mask=observed,
        root_from_wrist=np.eye(4),
        robot_q_current=np.zeros(36),
        hand_state=np.zeros(2),
    )

    _, records = _mask_metrics(depth, [observation], [robot])

    assert records[0]["observed_robot_overlap_pixels"] == 5000.0
    assert records[0]["observed_robot_overlap_fraction"] == 0.5


def test_state_timing_audit_recovers_a_shared_encoder_offset() -> None:
    frames = 80
    offset = 3
    time = np.arange(frames, dtype=np.float64)
    positions = np.column_stack((0.01 * time, 0.002 * time * time, 0.005 * np.sin(time / 7.0)))
    rotations = Rotation.from_euler("xyz", np.column_stack((0.02 * time, np.zeros(frames), -0.01 * time))).as_matrix()
    targets = {}
    placements = {}
    for side, x_bias in (("left", 0.0), ("right", 0.04)):
        shifted_positions = positions.copy()
        shifted_positions[:, 0] += x_bias
        target_positions = np.empty_like(shifted_positions)
        target_rotations = np.empty_like(rotations)
        target_positions[:-offset] = shifted_positions[offset:]
        target_positions[-offset:] = shifted_positions[-1]
        target_rotations[:-offset] = rotations[offset:]
        target_rotations[-offset:] = rotations[-1]
        placements[side] = (shifted_positions, rotations)
        targets[side] = (target_positions, target_rotations)

    report = timing_report(placements=placements, targets=targets, max_offset_frames=5)

    assert report["per_side"]["left"]["best"]["q_current_index_minus_ee_state_index_frames"] == offset
    assert report["per_side"]["right"]["best"]["q_current_index_minus_ee_state_index_frames"] == offset
    assert report["shared_offset"]["best"]["q_current_index_minus_ee_state_index_frames"] == offset


def test_joint_command_timing_audit_recovers_encoder_lag() -> None:
    frames = 80
    offset = 4
    desired = np.zeros((frames, 36), dtype=np.float64)
    desired[:, 19:] = np.column_stack(
        tuple(0.03 * (index + 1) * np.arange(frames) for index in range(17))
    )
    current = desired.copy()
    current[offset:, 19:] = desired[:-offset, 19:]
    current[:offset, 19:] = desired[0, 19:]

    report = joint_command_timing_report(current, desired, max_offset_frames=6)

    assert report["best"]["q_current_index_minus_robot_q_desired_index_frames"] == offset
