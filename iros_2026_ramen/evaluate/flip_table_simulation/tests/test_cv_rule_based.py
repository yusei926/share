from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "container_overlay" / "policy"
sys.path.insert(0, str(POLICY_ROOT))

from cv_rule_based.motion import (
    GRASP_RETRY_OFFSETS_TOOL_M,
    GeometricFlipPlanner,
    Phase,
    _interpolate_rotation,
    _rotation_from_quaternion_wxyz,
    apply_tool_position_offset,
    blend_table_frames,
    dex1_enclosure_from_joint_positions,
    grasp_retry_action,
    grasp_retry_total_steps,
    limit_cartesian_action_rate,
    update_bounded_integral_offsets,
    validate_cartesian_action,
    validate_static_table_redetection,
)
from cv_rule_based.vision import (
    CameraCalibration,
    LegDetection,
    TableLegDetector,
    TabletopEstimate,
    TabletopPoseEstimator,
    WristShaftDetector,
    WristTabletopEdgeDetector,
    _quadrilateral_from_mask,
)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)), dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)), dtype=np.float64)


def test_rgb_estimator_recovers_synthetic_preflip_tabletop() -> None:
    calibration = CameraCalibration.g1_head_left()
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(math.pi / 2 + 0.12) @ _rotation_x(math.pi)
    root_from_table[:3, 3] = (0.52, 0.025, -0.14)
    camera_from_table = np.linalg.inv(calibration.root_from_camera) @ root_from_table
    rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
    object_points = np.asarray(
        ((-0.29, -0.21, 0), (0.29, -0.21, 0), (0.29, 0.21, 0), (-0.29, 0.21, 0)),
        dtype=np.float64,
    )
    corners, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        camera_from_table[:3, 3],
        calibration.intrinsic_matrix,
        calibration.distortion,
    )
    image = np.full((480, 640, 3), 24, dtype=np.uint8)
    cv2.fillConvexPoly(image, np.rint(corners.reshape(-1, 2)).astype(np.int32), (238, 238, 238))

    estimate = TabletopPoseEstimator(calibration).estimate(image)

    assert estimate.confidence >= 0.20
    assert estimate.reprojection_error_px < 8.0
    assert np.linalg.norm(estimate.center_root_m - root_from_table[:3, 3]) < 0.08
    assert float(np.dot(estimate.root_from_table[:3, 2], (0.0, 0.0, -1.0))) > 0.95
    assert abs(float(np.dot(estimate.root_from_table[:2, 0], root_from_table[:2, 0]))) > 0.95


def test_rgb_estimator_does_not_alias_long_edge_presentation_to_short_edge() -> None:
    calibration = CameraCalibration.g1_head_left()
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(0.08) @ _rotation_x(math.pi)
    root_from_table[:3, 3] = (0.58, -0.01, -0.14)
    camera_from_table = np.linalg.inv(calibration.root_from_camera) @ root_from_table
    rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
    object_points = np.asarray(
        ((-0.29, -0.21, 0), (0.29, -0.21, 0), (0.29, 0.21, 0), (-0.29, 0.21, 0)),
        dtype=np.float64,
    )
    corners, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        camera_from_table[:3, 3],
        calibration.intrinsic_matrix,
        calibration.distortion,
    )
    image = np.full((480, 640, 3), 24, dtype=np.uint8)
    cv2.fillConvexPoly(
        image, np.rint(corners.reshape(-1, 2)).astype(np.int32), (238, 238, 238)
    )

    estimate = TabletopPoseEstimator(calibration).estimate(image)

    assert abs(float(np.dot(estimate.root_from_table[:2, 0], root_from_table[:2, 0]))) > 0.95


def test_rgb_estimator_registers_outer_rim_and_legs_not_inner_braces() -> None:
    calibration = CameraCalibration.g1_head_left()
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(0.58) @ _rotation_x(math.pi)
    root_from_table[:3, 3] = (0.72, -0.06, -0.14)
    camera_from_table = np.linalg.inv(calibration.root_from_camera) @ root_from_table
    rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
    outer_corners = np.asarray(
        ((-0.29, -0.21, 0), (0.29, -0.21, 0), (0.29, 0.21, 0), (-0.29, 0.21, 0)),
        dtype=np.float64,
    )
    image = np.full((480, 640, 3), 24, dtype=np.uint8)
    projected_outer, _ = cv2.projectPoints(
        outer_corners,
        rotation_vector,
        camera_from_table[:3, 3],
        calibration.intrinsic_matrix,
        calibration.distortion,
    )
    projected_outer = np.rint(projected_outer.reshape(-1, 2)).astype(np.int32)
    for start, end in zip(projected_outer, np.roll(projected_outer, -1, axis=0)):
        cv2.line(image, tuple(start), tuple(end), (238, 238, 238), 14)
    for x in (-0.255, 0.255):
        for y in (-0.175, 0.175):
            leg = np.asarray(((x, y, 0.0), (x, y, -0.45)), dtype=np.float64)
            projected_leg, _ = cv2.projectPoints(
                leg,
                rotation_vector,
                camera_from_table[:3, 3],
                calibration.intrinsic_matrix,
                calibration.distortion,
            )
            start, end = np.rint(projected_leg.reshape(-1, 2)).astype(np.int32)
            cv2.line(image, tuple(start), tuple(end), (238, 238, 238), 16)
    # An inner brace is deliberately brighter and thicker than the rim. A
    # contour-only detector tends to lock to it; the CAD wireframe fit must not.
    cv2.line(image, tuple(projected_outer[0]), tuple(projected_outer[2]), (238, 238, 238), 24)

    estimate = TabletopPoseEstimator(calibration).estimate(image)

    assert estimate.confidence >= 0.35
    assert estimate.reprojection_error_px < 14.0
    assert np.linalg.norm(estimate.center_root_m - root_from_table[:3, 3]) < 0.09
    assert abs(float(np.dot(estimate.root_from_table[:2, 0], root_from_table[:2, 0]))) > 0.90


def test_tabletop_debug_center_is_projective_diagonal_intersection() -> None:
    corners = np.asarray(
        ((120.0, 90.0), (500.0, 145.0), (420.0, 360.0), (155.0, 285.0)),
        dtype=np.float64,
    )
    estimate = TabletopEstimate(
        root_from_table=np.eye(4, dtype=np.float64),
        camera_from_table=np.eye(4, dtype=np.float64),
        corners_px=corners,
        mask=np.zeros((480, 640), dtype=np.uint8),
        confidence=1.0,
        reprojection_error_px=0.0,
        area_fraction=0.1,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    debug = TabletopPoseEstimator.render_debug(image, estimate)

    first = np.cross(np.append(corners[0], 1.0), np.append(corners[2], 1.0))
    second = np.cross(np.append(corners[1], 1.0), np.append(corners[3], 1.0))
    expected = np.cross(first, second)
    expected = np.rint(expected[:2] / expected[2]).astype(np.int32)
    mean = np.rint(corners.mean(axis=0)).astype(np.int32)

    assert np.array_equal(debug[expected[1], expected[0]], np.asarray((255, 0, 0)))
    assert not np.array_equal(debug[mean[1], mean[0]], np.asarray((255, 0, 0)))


def test_tabletop_segmentation_rejects_bright_floor_outside_black_workbench() -> None:
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.fillConvexPoly(
        image,
        np.asarray(((30, 150), (610, 170), (590, 450), (20, 430)), dtype=np.int32),
        (20, 20, 20),
    )
    target = np.asarray(((235, 180), (405, 185), (465, 330), (285, 350)), dtype=np.int32)
    cv2.fillConvexPoly(image, target, (238, 238, 238))

    mask = TabletopPoseEstimator.segment_tabletop(image)

    assert mask[240, 320] == 255
    assert mask[130, 320] == 0
    assert mask[440, 610] == 0


def test_tabletop_quadrilateral_uses_outer_panel_not_legs_or_bright_floor() -> None:
    image = np.full((480, 640, 3), (194, 181, 164), dtype=np.uint8)
    cv2.fillConvexPoly(
        image,
        np.asarray(((20, 105), (620, 110), (610, 402), (28, 400)), dtype=np.int32),
        (23, 23, 23),
    )
    outer = np.asarray(((188, 137), (452, 126), (526, 258), (155, 247)), dtype=np.int32)
    cv2.fillConvexPoly(image, outer, (212, 210, 208))
    # These are visual distractors that previously pulled the hull inward or
    # made a bright floor strip dominate the selected component.
    cv2.line(image, (191, 140), (157, 74), (212, 210, 208), 20)
    cv2.line(image, (450, 130), (520, 67), (212, 210, 208), 20)
    cv2.line(image, (160, 245), (105, 316), (212, 210, 208), 18)
    cv2.line(image, (525, 257), (576, 319), (212, 210, 208), 18)
    cv2.line(image, (230, 170), (432, 226), (170, 170, 170), 18)
    image[406:480, :, :] = (224, 220, 215)

    mask = TabletopPoseEstimator.segment_tabletop(image)
    corners, _ = _quadrilateral_from_mask(mask)

    estimated_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    expected_area = abs(float(cv2.contourArea(outer.astype(np.float32))))
    np.testing.assert_allclose(corners.mean(axis=0), outer.mean(axis=0), atol=18.0)
    assert 0.75 <= estimated_area / expected_area <= 1.25


def test_leg_segmentation_preserves_narrow_white_shafts() -> None:
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (20, 120), (620, 460), (20, 20, 20), -1)
    cv2.rectangle(image, (230, 250), (410, 390), (238, 238, 238), -1)
    cv2.line(image, (240, 260), (180, 130), (238, 238, 238), 8)

    assembly = TabletopPoseEstimator.segment_table_assembly(image)
    tabletop = TabletopPoseEstimator.segment_tabletop(image)

    assert assembly[160, 194] == 255
    assert tabletop[160, 194] == 0
    assert tabletop[320, 320] == 255


def test_wrist_shaft_detector_requires_a_centered_leg_through_finger_corridor() -> None:
    detector = WristShaftDetector()
    valid = np.full((480, 640, 3), 18, dtype=np.uint8)
    cv2.rectangle(valid, (292, 120), (348, 438), (238, 238, 238), -1)
    outside = np.full_like(valid, 18)
    cv2.rectangle(outside, (390, 120), (470, 438), (238, 238, 238), -1)
    short = np.full_like(valid, 18)
    cv2.rectangle(short, (292, 210), (348, 275), (238, 238, 238), -1)

    observation = detector.detect(valid)

    assert observation.detected
    assert observation.center_px is not None
    assert abs(observation.center_px[0] - 320.0) < 2.0
    assert observation.confidence >= 0.30
    assert not detector.detect(outside).detected
    assert detector.detect(outside, require_centered=False).detected
    assert not detector.detect(short).detected


def test_wrist_shaft_detector_rejects_broad_tabletop_underside() -> None:
    detector = WristShaftDetector()
    image = np.full((480, 640, 3), 18, dtype=np.uint8)
    cv2.rectangle(image, (224, 202), (415, 430), (238, 238, 238), -1)

    observation = detector.detect(image)

    assert not observation.detected


def test_wrist_tabletop_edge_detector_tracks_edge_depth() -> None:
    detector = WristTabletopEdgeDetector()
    shallow = np.full((480, 640, 3), 18, dtype=np.uint8)
    deep = np.full_like(shallow, 18)
    cv2.rectangle(shallow, (0, 0), (639, 220), (238, 238, 238), -1)
    cv2.rectangle(deep, (0, 0), (639, 360), (238, 238, 238), -1)

    shallow_observation = detector.detect(shallow)
    deep_observation = detector.detect(deep)

    assert shallow_observation.detected
    assert deep_observation.detected
    assert 215.0 <= shallow_observation.edge_y_px <= 225.0
    assert 355.0 <= deep_observation.edge_y_px <= 365.0
    assert not detector.detect(np.full_like(shallow, 18)).detected


def test_head_camera_calibration_composes_dynamic_torso_fk() -> None:
    root_from_torso = np.eye(4, dtype=np.float64)
    root_from_torso[:3, :3] = _rotation_z(0.2) @ _rotation_x(-0.1)
    root_from_torso[:3, 3] = (0.01, -0.02, 0.05)

    calibration = CameraCalibration.g1_head_left_from_torso(root_from_torso)

    np.testing.assert_allclose(
        calibration.root_from_camera,
        root_from_torso @ CameraCalibration.g1_head_left_torso_from_camera(),
        atol=1.0e-12,
    )


def test_head_camera_calibration_matches_official_v1_camera_authoring() -> None:
    intrinsic, distortion = CameraCalibration._g1_head_left_sim_intrinsics()
    torso_from_camera = CameraCalibration.g1_head_left_torso_from_camera()

    expected_focal_px = 24.0 * 640.0 / 45.56883749280177
    np.testing.assert_allclose(
        intrinsic,
        ((expected_focal_px, 0.0, 320.0), (0.0, expected_focal_px, 240.0), (0.0, 0.0, 1.0)),
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(distortion, np.zeros(5, dtype=np.float64))
    np.testing.assert_allclose(
        torso_from_camera[:3, 3],
        (0.10209156, 0.02077481159355057, 0.42446595),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torso_from_camera[:3, :3].T @ torso_from_camera[:3, :3],
        np.eye(3),
        atol=5.0e-9,
    )
    np.testing.assert_allclose(
        CameraCalibration.g1_head_left().root_from_camera[:3, 3],
        (0.09812806, 0.02077481159355057, 0.46846595),
        atol=1.0e-12,
    )


def test_geometry_planner_has_ordered_phases_and_valid_actions() -> None:
    root_from_table = np.asarray(
        ((0, 1, 0, 0.62), (1, 0, 0, 0.0), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    planner = GeometricFlipPlanner(root_from_table, 50.0)
    phases = [planner.phase_at(step) for step in range(planner.total_steps)]

    assert planner.total_steps > 500
    assert phases[0] is Phase.CLEARANCE_STAGING
    assert Phase.ALIGN_APPROACH in phases
    assert Phase.ALIGN_SHORT_EDGE in phases
    assert Phase.LEFT_LEG_FLIP_90 in phases
    assert Phase.RIGHT_TOP_FLIP_90 in phases
    assert phases[-1] is Phase.SETTLE_AND_RETREAT
    assert abs(planner.aligned_long_axis[1]) < 1.0e-6
    assert abs(planner.aligned_long_axis[0] + 1.0) < 1.0e-6
    for step in range(planner.total_steps):
        action = planner.action_at(step)
        assert action.shape == (16,)
        assert np.isfinite(action).all()
        assert abs(np.linalg.norm(action[3:7]) - 1.0) < 1.0e-5
        assert abs(np.linalg.norm(action[10:14]) - 1.0) < 1.0e-5
        assert np.all((-1.0 <= action[14:16]) & (action[14:16] <= 1.0))


def test_long_edge_alignment_uses_push_recenter_pull_compass_sequence() -> None:
    root_from_table = np.asarray(
        ((0, 1, 0, 0.68), (1, 0, 0, -0.08), (0, 0, -1, -0.21), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    attachments = TableLegDetector.estimate_near_leg_centers(root_from_table)

    planner = GeometricFlipPlanner(root_from_table, 50.0, attachments)

    assert planner.alignment_mode == "long_edge_push_recenter_pull"
    assert planner.alignment_pivot_side in {"left", "right"}
    assert planner.alignment_moving_side != planner.alignment_pivot_side
    assert abs(planner.alignment_yaw_delta_rad) > math.pi / 4
    np.testing.assert_allclose(
        planner.aligned_long_axis[:2],
        (-0.9931506, 0.11684125),
        atol=1.0e-6,
    )

    first_roll_step = planner.phase_start_step(Phase.LEFT_LEG_FLIP_90)
    released = planner.action_at(first_roll_step - 1)
    assert np.all(released[14:16] < -0.99)


def test_aligned_scene_planner_does_not_apply_another_yaw_rotation() -> None:
    yaw = math.radians(17.0)
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(yaw)
    root_from_table[:3, 2] = (0.0, 0.0, -1.0)
    root_from_table[:3, 3] = (0.56, -0.03, -0.17)

    attachments = TableLegDetector.select_arm_reachable_leg_centers(root_from_table)
    planner = GeometricFlipPlanner(
        root_from_table,
        50.0,
        attachments,
        table_is_aligned=True,
    )

    assert planner.alignment_yaw_delta_rad == 0.0
    reachable_left = attachments["left"].copy()
    reachable_left[2] = (
        planner.table_root_z_m + GeometricFlipPlanner.LEG_GRASP_HEIGHT_M
    )
    np.testing.assert_allclose(
        planner.first_roll_grasp_center, reachable_left, atol=1.0e-9
    )


def test_post_alignment_leg_assignment_avoids_cross_body_left_grasp() -> None:
    yaw = math.radians(-145.0)
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(yaw)
    root_from_table[:3, 2] = (0.0, 0.0, -1.0)
    root_from_table[:3, 3] = (0.56, -0.20, -0.17)

    near = TableLegDetector.estimate_near_leg_centers(root_from_table)
    reachable = TableLegDetector.select_arm_reachable_leg_centers(root_from_table)

    assert near["left"][1] < 0.0
    assert reachable["left"][1] > near["left"][1] + 0.20
    assert reachable["left"][1] > reachable["right"][1]


def test_geometry_planner_reports_exact_phase_bounds() -> None:
    planner = GeometricFlipPlanner(np.eye(4, dtype=np.float64), 50.0)

    for phase in Phase:
        start = planner.phase_start_step(phase)
        end = planner.phase_end_step(phase)
        assert 0 <= start < end <= planner.total_steps
        assert planner.phase_at(start) is phase
        assert planner.phase_at(end - 1) is phase
        if start:
            assert planner.phase_at(start - 1) is not phase


def test_grasp_retry_backs_off_approaches_closes_and_holds() -> None:
    aligned = GeometricFlipPlanner.neutral_action()
    aligned[0:3] = (0.25, 0.18, 0.04)
    offset = GRASP_RETRY_OFFSETS_TOOL_M[1]
    total = grasp_retry_total_steps(50.0)

    first, first_stage, first_ready = grasp_retry_action(aligned, offset, 0, 50.0)
    middle, middle_stage, middle_ready = grasp_retry_action(aligned, offset, 40, 50.0)
    closing, closing_stage, closing_ready = grasp_retry_action(
        aligned, offset, 80, 50.0
    )
    last, last_stage, last_ready = grasp_retry_action(aligned, offset, total - 1, 50.0)

    assert first_stage == "open_backoff"
    assert first[0] < aligned[0] + offset[0]
    assert first[14] == -1.0
    assert not first_ready
    assert middle_stage == "approach"
    assert middle[14] == -1.0
    assert not middle_ready
    assert closing_stage == "close"
    assert not closing_ready
    assert last_stage == "verify"
    np.testing.assert_allclose(
        last[0:3], aligned[0:3] + offset + (0.035, 0.0, 0.0), atol=1.0e-7
    )
    assert last[14] == 0.75
    assert last_ready


def test_regrasp_offset_rotates_with_left_tool_without_a_position_jump() -> None:
    action = GeometricFlipPlanner.neutral_action()
    action[0:3] = (0.27, 0.22, -0.01)
    action[3:7] = (
        math.cos(math.pi / 4),
        math.sin(math.pi / 4),
        0.0,
        0.0,
    )

    adjusted = apply_tool_position_offset(action, (0.075, 0.0, 0.0), "left")

    np.testing.assert_allclose(adjusted[0:3], (0.345, 0.22, -0.01), atol=1.0e-7)
    np.testing.assert_allclose(adjusted[3:7], action[3:7], atol=1.0e-7)


def test_regrasp_offset_can_target_right_tool_without_moving_left_tool() -> None:
    action = GeometricFlipPlanner.neutral_action()
    action[0:3] = (0.22, 0.20, 0.01)
    action[7:10] = (0.29, -0.18, -0.02)
    action[10:14] = (
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    )

    adjusted = apply_tool_position_offset(action, (0.04, 0.0, 0.0), "right")

    np.testing.assert_allclose(adjusted[0:7], action[0:7], atol=1.0e-7)
    np.testing.assert_allclose(adjusted[7:10], (0.29, -0.14, -0.02), atol=1.0e-7)
    np.testing.assert_allclose(adjusted[10:14], action[10:14], atol=1.0e-7)


def test_grasp_retry_can_close_right_hand_without_changing_left_hand() -> None:
    aligned = GeometricFlipPlanner.neutral_action()
    aligned[0:3] = (0.21, 0.20, 0.03)
    aligned[7:10] = (0.27, -0.18, 0.01)
    total = grasp_retry_total_steps(50.0)

    action, stage, ready = grasp_retry_action(
        aligned,
        (0.03, 0.0, 0.0),
        total - 1,
        50.0,
        side="right",
    )

    assert stage == "verify"
    assert ready
    np.testing.assert_allclose(action[0:7], aligned[0:7], atol=1.0e-7)
    np.testing.assert_allclose(action[7:10], (0.335, -0.18, 0.01), atol=1.0e-7)
    assert action[14] == aligned[14]
    assert action[15] == 0.75


def test_tabletop_frame_reconstructs_all_four_legs_and_selects_near_pair() -> None:
    root_from_table = np.asarray(
        ((0, 1, 0, 0.62), (1, 0, 0, 0.03), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )

    all_legs = TableLegDetector.estimate_all_leg_centers(root_from_table)
    near = TableLegDetector.estimate_near_leg_centers(root_from_table)

    assert len(all_legs) == 4
    assert near["left"][0] < root_from_table[0, 3]
    assert near["right"][0] < root_from_table[0, 3]
    assert near["left"][1] > near["right"][1]


def test_dex1_enclosure_requires_both_fingers_to_be_obstructed() -> None:
    threshold = -0.017

    assert dex1_enclosure_from_joint_positions(
        np.asarray(((0.004, -0.010), (0.003, -0.012))), threshold
    ) == (True, True)
    assert dex1_enclosure_from_joint_positions(
        np.asarray(((0.020, -0.020), (-0.020, -0.020))), threshold
    ) == (False, False)
    assert dex1_enclosure_from_joint_positions(
        np.asarray(((0.008, -0.016), (0.002, -0.006))), threshold
    ) == (False, True)
    assert dex1_enclosure_from_joint_positions(
        np.asarray(((0.0245, 0.0245), (0.019, 0.019))), threshold
    ) == (False, True)


def test_rotation_interpolation_handles_exact_half_turn() -> None:
    start = np.eye(3, dtype=np.float64)
    end = _rotation_x(math.pi)

    midpoint = _interpolate_rotation(start, end, 0.5)

    np.testing.assert_allclose(midpoint @ midpoint.T, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(midpoint, _rotation_x(math.pi / 2), atol=1.0e-12)
    assert math.isclose(float(np.linalg.det(midpoint)), 1.0, abs_tol=1.0e-12)


def test_cartesian_rate_limiter_bounds_translation_rotation_and_hand_motion() -> None:
    previous = GeometricFlipPlanner.neutral_action()
    requested = previous.copy()
    requested[0:3] += (0.20, -0.10, 0.05)
    requested[3:7] = (0.0, 1.0, 0.0, 0.0)
    requested[7:10] += (-0.15, 0.08, -0.04)
    requested[10:14] = (0.0, 1.0, 0.0, 0.0)
    requested[14:16] = 1.0

    limited = limit_cartesian_action_rate(
        previous,
        requested,
        50.0,
        max_linear_speed_m_s=0.25,
        max_angular_speed_rad_s=0.75,
        max_hand_speed_s=3.5,
    )

    assert np.linalg.norm(limited[0:3] - previous[0:3]) <= 0.005 + 1.0e-7
    assert np.linalg.norm(limited[7:10] - previous[7:10]) <= 0.005 + 1.0e-7
    angle = 2.0 * math.acos(abs(float(np.dot(limited[3:7], previous[3:7]))))
    assert angle <= 0.015 + 5.0e-6
    np.testing.assert_allclose(limited[14:16], -0.93, atol=1.0e-6)


def test_geometry_planner_actions_are_valid_across_randomized_table_yaws() -> None:
    for yaw in np.linspace(-math.pi, math.pi, 37):
        root_from_table = np.eye(4, dtype=np.float64)
        root_from_table[:3, :3] = _rotation_z(float(yaw)) @ _rotation_x(math.pi)
        root_from_table[:3, 3] = (0.62, 0.02, -0.17)
        planner = GeometricFlipPlanner(root_from_table, 50.0)
        for step in range(planner.total_steps):
            validate_cartesian_action(planner.action_at(step))


def test_geometry_planner_uses_detected_tabletop_height() -> None:
    root_from_table = np.eye(4, dtype=np.float64)
    root_from_table[:3, :3] = _rotation_z(math.pi / 2) @ _rotation_x(math.pi)
    root_from_table[:3, 3] = (0.62, 0.02, 0.015)

    planner = GeometricFlipPlanner(root_from_table, 50.0)
    assert planner.table_root_z_m == 0.015

    grasp_actions = [
        planner.action_at(step)
        for step in range(planner.total_steps)
        if planner.phase_at(step) is Phase.ALIGN_GRASP
        and np.all(planner.action_at(step)[14:16] <= -0.99)
    ]
    assert grasp_actions
    for action in grasp_actions:
        for position, quaternion in ((action[0:3], action[3:7]), (action[7:10], action[10:14])):
            grasp = position + _rotation_from_quaternion_wxyz(quaternion)[:, 0] * planner.WRIST_TO_GRASP_M
            assert grasp[2] >= planner.table_root_z_m + 0.045


def test_geometry_planner_approaches_from_robot_side_for_every_table_yaw() -> None:
    for yaw in np.linspace(-math.pi, math.pi, 17):
        root_from_table = np.eye(4, dtype=np.float64)
        root_from_table[:3, :3] = _rotation_z(float(yaw)) @ _rotation_x(math.pi)
        root_from_table[:3, 3] = (0.62, 0.02, -0.17)
        planner = GeometricFlipPlanner(root_from_table, 50.0)
        step = planner.phase_start_step(Phase.ALIGN_GRASP)
        action = planner.action_at(step)
        robot_to_table = root_from_table[:3, 3].copy()
        robot_to_table[2] = 0.0
        robot_to_table /= np.linalg.norm(robot_to_table)

        for quaternion in (action[3:7], action[10:14]):
            tool_x = _rotation_from_quaternion_wxyz(quaternion)[:, 0]
            horizontal = tool_x.copy()
            horizontal[2] = 0.0
            horizontal /= np.linalg.norm(horizontal)
            assert float(np.dot(horizontal, robot_to_table)) > 0.999
            assert -0.501 <= float(tool_x[2]) <= 1.0e-6


def test_geometry_planner_grasps_nearest_leg_and_opposite_tabletop_edge() -> None:
    root_from_table = np.asarray(
        ((1, 0, 0, 0.62), (0, -1, 0, 0.0), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    planner = GeometricFlipPlanner(root_from_table, 50.0)

    contact_action = [
        planner.action_at(step)
        for step in range(planner.total_steps)
        if planner.phase_at(step) is Phase.ALIGN_GRASP
        and np.all(planner.action_at(step)[14:16] <= -0.99)
    ][-1]

    left_rotation = _rotation_from_quaternion_wxyz(contact_action[3:7])
    right_rotation = _rotation_from_quaternion_wxyz(contact_action[10:14])
    left_grasp = contact_action[0:3] + left_rotation[:, 0] * planner.WRIST_TO_GRASP_M
    right_grasp = contact_action[7:10] + right_rotation[:, 0] * planner.WRIST_TO_GRASP_M
    pivot_grasp = left_grasp if planner.alignment_pivot_side == "left" else right_grasp
    edge_grasp = right_grasp if planner.alignment_pivot_side == "left" else left_grasp
    np.testing.assert_allclose(
        pivot_grasp, planner.alignment_pivot_grasp_center, atol=1.0e-6
    )
    np.testing.assert_allclose(
        edge_grasp, planner.alignment_edge_grasp_center, atol=1.0e-6
    )
    assert math.isclose(
        float(pivot_grasp[2]),
        planner.table_root_z_m + planner.LEG_GRASP_HEIGHT_M,
        abs_tol=1.0e-6,
    )
    assert math.isclose(
        float(edge_grasp[2]), planner.table_root_z_m + 0.055, abs_tol=1.0e-6
    )
    assert planner.alignment_mode == "short_edge_pull"

    first_flip_actions = [
        planner.action_at(step)
        for step in range(planner.total_steps)
        if planner.phase_at(step) is Phase.LEFT_LEG_FLIP_90
    ]
    assert min(float(action[2]) for action in first_flip_actions) < (
        planner.table_root_z_m + planner.LEG_GRASP_HEIGHT_M - 0.05
    )
    assert min(float(action[1]) for action in first_flip_actions) > 0.17
    np.testing.assert_allclose(
        first_flip_actions[-1][1],
        0.5 * planner.TABLE_DEPTH_M + planner.LEG_GRASP_HEIGHT_M,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        first_flip_actions[-1][2],
        planner.table_root_z_m + planner.LEG_INSET_M,
        atol=1.0e-6,
    )
    for action in first_flip_actions[1:]:
        np.testing.assert_allclose(action[7:14], first_flip_actions[0][7:14], atol=1.0e-6)
        assert action[15] == -1.0

    right_pregrasp_actions = [
        planner.action_at(step)
        for step in range(planner.total_steps)
        if planner.phase_at(step) is Phase.RIGHT_PREGRASP
    ]
    assert right_pregrasp_actions[-1][9] > 0.15

    right_push_actions = [
        planner.action_at(step)
        for step in range(planner.total_steps)
        if planner.phase_at(step) is Phase.RIGHT_TOP_FLIP_90
    ]
    assert right_push_actions[-1][8] > right_push_actions[0][8] + 0.10


def test_aligned_planner_preserves_rgb_workspace_leg_assignment() -> None:
    root_from_table = np.asarray(
        ((1, 0, 0, 0.62), (0, -1, 0, 0.0), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    attachments = {
        "left": np.asarray((0.51, 0.13, -0.17)),
        "right": np.asarray((0.49, -0.12, -0.17)),
    }
    planner = GeometricFlipPlanner(
        root_from_table, 50.0, attachments, table_is_aligned=True
    )

    expected = attachments["left"].copy()
    expected[2] = planner.table_root_z_m + planner.LEG_GRASP_HEIGHT_M
    np.testing.assert_allclose(planner.first_roll_grasp_center, expected, atol=1.0e-9)
    assert planner.alignment_mode == "already_aligned"


def test_closed_loop_table_frame_filter_smooths_and_rejects_outliers() -> None:
    current = np.asarray(
        ((0, 1, 0, 0.62), (1, 0, 0, 0.0), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    candidate = current.copy()
    candidate[0, 3] += 0.04
    candidate[1, 3] -= 0.02
    angle = 0.10
    candidate[:3, 0] = (-math.sin(angle), math.cos(angle), 0.0)
    candidate[:3, 1] = (math.cos(angle), math.sin(angle), 0.0)

    blended = blend_table_frames(
        current, candidate, 0.25, max_translation_m=0.12, max_yaw_rad=0.35
    )

    np.testing.assert_allclose(blended[:2, 3], (0.63, -0.005), atol=1.0e-8)
    assert abs(math.atan2(blended[1, 0], blended[0, 0]) - (math.pi / 2 + 0.025)) < 1.0e-8

    candidate[0, 3] += 0.20
    with np.testing.assert_raises_regex(ValueError, "translation jump"):
        blend_table_frames(
            current, candidate, 0.25, max_translation_m=0.12, max_yaw_rad=0.35
        )


def test_static_table_redetection_rejects_hand_occlusion_drift() -> None:
    initial = np.eye(4, dtype=np.float64)
    initial[:2, 3] = (0.61, -0.06)
    attachments = {
        "left": np.asarray((0.32, 0.06, -0.214565)),
        "right": np.asarray((0.46, -0.30, -0.214565)),
    }
    candidate = initial.copy()
    candidate[:2, 3] += (0.01, -0.01)
    nearby = {
        "left": attachments["left"] + (0.01, -0.01, 0.0),
        "right": attachments["right"] + (-0.01, 0.01, 0.0),
    }

    validate_static_table_redetection(
        initial,
        candidate,
        attachments,
        nearby,
        max_center_drift_m=0.03,
        max_attachment_drift_m=0.03,
    )

    occluded = dict(nearby)
    occluded["left"] = attachments["left"] + (0.0, 0.05, 0.0)
    with np.testing.assert_raises_regex(ValueError, "left leg-attachment drift"):
        validate_static_table_redetection(
            initial,
            candidate,
            attachments,
            occluded,
            max_center_drift_m=0.03,
            max_attachment_drift_m=0.03,
        )


def test_integral_wrist_servo_converges_and_clears_disabled_side() -> None:
    offsets = np.zeros((2, 3), dtype=np.float64)
    errors = np.asarray(((0.05, -0.04, 0.02), (-0.08, 0.0, 0.0)))
    for _ in range(20):
        offsets = update_bounded_integral_offsets(
            offsets,
            errors,
            (True, True),
            gain=0.08,
            max_step_m=0.004,
            max_norm_m=0.06,
        )

    assert 0.04 < np.linalg.norm(offsets[0]) <= 0.06 + 1.0e-12
    assert np.linalg.norm(offsets[1]) <= 0.06 + 1.0e-12

    cleared = update_bounded_integral_offsets(
        offsets,
        errors,
        (True, False),
        gain=0.08,
        max_step_m=0.004,
        max_norm_m=0.06,
    )
    np.testing.assert_allclose(cleared[1], 0.0)


def test_rgb_leg_detector_associates_shafts_with_front_tabletop_corners(monkeypatch) -> None:
    detector = TableLegDetector()
    root_from_table = np.asarray(
        ((0, 1, 0, 0.62), (1, 0, 0, 0.02), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    expected = detector.estimate_near_leg_centers(root_from_table)
    left_endpoints = np.asarray(((205, 330), (170, 180)), dtype=np.float64)
    right_endpoints = np.asarray(((440, 335), (485, 185)), dtype=np.float64)
    candidates = [
        SimpleNamespace(
            attachment_root_m=expected["left"], endpoints_px=left_endpoints,
            confidence=0.8, vertical_alignment=1.0, score=3.0,
        ),
        SimpleNamespace(
            attachment_root_m=expected["right"], endpoints_px=right_endpoints,
            confidence=0.9, vertical_alignment=1.0, score=3.1,
        ),
        # A rear-left shaft may be visually cleaner but must never replace the
        # front-left shaft selected by tabletop-corner topology.
        SimpleNamespace(
            attachment_root_m=expected["left"] + np.asarray((0.50, 0.0, 0.0)),
            endpoints_px=np.asarray(((250, 160), (270, 275)), dtype=np.float64),
            confidence=1.0, vertical_alignment=1.0, score=5.0,
        ),
    ]
    monkeypatch.setattr(detector, "_axis_candidates", lambda *args, **kwargs: candidates)
    estimate = TabletopEstimate(
        root_from_table=root_from_table,
        camera_from_table=np.eye(4),
        corners_px=np.asarray(((200, 200), (440, 200), (440, 360), (200, 360))),
        mask=np.zeros((480, 640), dtype=np.uint8),
        confidence=1.0,
        reprojection_error_px=0.0,
        area_fraction=0.1,
    )

    calibration = CameraCalibration.g1_head_left()
    detections = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8),
        estimate,
        calibration,
        -0.17,
    )

    assert set(detections) == {"left", "right"}
    np.testing.assert_array_equal(detections["left"].endpoints_px, left_endpoints)
    np.testing.assert_array_equal(detections["right"].endpoints_px, right_endpoints)
    assert not detections["left"].inferred_from_tabletop
    assert not detections["right"].inferred_from_tabletop
    assert detections["left"].confidence >= 0.20
    assert detections["right"].confidence >= 0.20

    table_frame = TableLegDetector.estimate_table_frame(
        estimate,
        detections,
        calibration,
        tabletop_z_m=-0.17,
    )
    assert table_frame.shape == (4, 4)
    assert np.isfinite(table_frame).all()
    assert table_frame[2, 3] == -0.17
    assert np.linalg.det(table_frame[:3, :3]) > 0.99
    near_legs = TableLegDetector.estimate_near_leg_centers(table_frame)
    assert near_legs["left"][1] > near_legs["right"][1]
    np.testing.assert_allclose(
        np.linalg.norm(near_legs["left"] - near_legs["right"]),
        0.51,
        atol=1.0e-12,
    )
    assert abs(float(table_frame[1, 0])) > 0.99
    assert abs(float(table_frame[0, 1])) > 0.99
    assert 0.5 * (near_legs["left"][0] + near_legs["right"][0]) < table_frame[0, 3]


def test_near_leg_targets_remain_cad_consistent_despite_rgb_shaft_bias(monkeypatch) -> None:
    detector = TableLegDetector()
    table_frame = np.eye(4, dtype=np.float64)
    table_frame[:3, 3] = (0.50, -0.15, -0.214565)
    expected = detector.estimate_near_leg_centers(table_frame)
    monkeypatch.setattr(
        detector,
        "_axis_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct shaft correction must not move a CAD leg target")
        ),
    )

    tabletop = TabletopEstimate(
        root_from_table=np.eye(4),
        camera_from_table=np.eye(4),
        corners_px=np.zeros((4, 2)),
        mask=np.zeros((480, 640), dtype=np.uint8),
        confidence=1.0,
        reprojection_error_px=0.0,
        area_fraction=0.1,
    )
    actual = detector.detect_near_leg_attachment_points(
        np.zeros((480, 640, 3), dtype=np.uint8),
        tabletop,
        CameraCalibration.g1_head_left(),
        table_frame,
        tabletop_z_m=-0.214565,
    )

    np.testing.assert_allclose(actual["left"], expected["left"])
    np.testing.assert_allclose(actual["right"], expected["right"])


def test_leg_axis_attachment_uses_tabletop_intersection() -> None:
    endpoints = np.asarray(((202.0, 238.0), (255.0, 386.0)))
    corners = np.asarray(
        ((242.0, 355.0), (250.0, 192.0), (374.0, 174.0), (443.0, 354.0))
    )

    attachment = TableLegDetector._project_nearest_corner_to_axis(
        endpoints, corners
    )

    assert np.linalg.norm(attachment - endpoints[1]) > 10.0
    assert np.linalg.norm(attachment - corners[0]) < 8.0


def test_table_frame_center_uses_pnp_pose_and_supplied_tabletop_height() -> None:
    calibration = CameraCalibration.g1_head_left()
    expected_center = np.asarray((0.82, -0.08, -0.17), dtype=np.float64)
    long_axis = np.asarray((1.0, 0.0, 0.0))
    short_axis = np.asarray((0.0, 1.0, 0.0))
    rendered_center = np.asarray((0.60, 0.02, -0.17), dtype=np.float64)
    corners_root = np.stack(
        [
            rendered_center - 0.20 * long_axis - 0.14 * short_axis,
            rendered_center + 0.20 * long_axis - 0.14 * short_axis,
            rendered_center + 0.20 * long_axis + 0.14 * short_axis,
            rendered_center - 0.20 * long_axis + 0.14 * short_axis,
        ]
    )
    corners_px = np.stack(
        [TableLegDetector._project_root_point(point, calibration) for point in corners_root]
    )
    biased_pnp_pose = np.eye(4, dtype=np.float64)
    biased_pnp_pose[:3, 3] = (0.82, -0.08, -0.30)
    tabletop = TabletopEstimate(
        root_from_table=biased_pnp_pose,
        camera_from_table=np.eye(4),
        corners_px=corners_px,
        mask=np.zeros((480, 640), dtype=np.uint8),
        confidence=1.0,
        reprojection_error_px=0.0,
        area_fraction=0.1,
    )
    image_center_x = float(corners_px[:, 0].mean())
    detections = {}
    for side in ("left", "right"):
        side_indices = np.flatnonzero(
            corners_px[:, 0] < image_center_x
            if side == "left"
            else corners_px[:, 0] >= image_center_x
        )
        attachment_index = int(max(side_indices, key=lambda index: corners_px[index, 1]))
        attachment_root = corners_root[attachment_index]
        vertical_tip_px = TableLegDetector._project_root_point(
            attachment_root + np.asarray((0.0, 0.0, 0.40)), calibration
        )
        detections[side] = LegDetection(
            side,
            np.asarray((corners_px[attachment_index], vertical_tip_px)),
            1.0,
        )

    table_frame = TableLegDetector.estimate_table_frame(
        tabletop,
        detections,
        calibration,
        tabletop_z_m=-0.17,
    )

    np.testing.assert_allclose(table_frame[:3, 3], expected_center, atol=1.0e-8)
    assert abs(float(table_frame[0, 0])) > 0.99
    assert abs(float(table_frame[1, 1])) > 0.99


def test_rgb_leg_detector_infers_an_occluded_front_leg_from_tabletop(monkeypatch) -> None:
    detector = TableLegDetector()
    root_from_table = np.asarray(
        ((0, 1, 0, 0.62), (1, 0, 0, 0.02), (0, 0, -1, -0.17), (0, 0, 0, 1)),
        dtype=np.float64,
    )
    expected = detector.estimate_near_leg_centers(root_from_table)
    estimate = TabletopEstimate(
        root_from_table=root_from_table,
        camera_from_table=np.eye(4),
        corners_px=np.asarray(((200, 200), (440, 200), (440, 360), (200, 360))),
        mask=np.zeros((480, 640), dtype=np.uint8),
        confidence=1.0,
        reprojection_error_px=0.0,
        area_fraction=0.1,
    )
    right_endpoints = np.asarray(((440, 335), (485, 185)), dtype=np.float64)
    monkeypatch.setattr(
        detector,
        "_axis_candidates",
        lambda *args, **kwargs: [
            SimpleNamespace(
                attachment_root_m=expected["right"], endpoints_px=right_endpoints,
                confidence=0.9, vertical_alignment=1.0, score=3.1,
            )
        ],
    )

    detections = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8),
        estimate,
        CameraCalibration.g1_head_left(),
        -0.17,
    )

    assert detections["left"].inferred_from_tabletop
    assert not detections["right"].inferred_from_tabletop
    np.testing.assert_allclose(detections["left"].attachment_root_m, expected["left"])
    np.testing.assert_array_equal(detections["right"].endpoints_px, right_endpoints)
