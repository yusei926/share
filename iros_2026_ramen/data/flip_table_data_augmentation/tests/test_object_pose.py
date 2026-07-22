from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.object_pose.camera_views import (
    inverse_brown_conrady_rectification_maps,
)
from data.flip_table_data_augmentation.object_pose.geometry import (
    OPENGL_FROM_OPENCV,
    evaluate_rendered_alignment,
    fuse_bidirectional_poses,
    interpolate_pose_trajectory,
    make_pose_continuous,
    pose_errors,
    project_to_rigid_transform,
    root_from_rectified_opencv_camera,
)
from data.flip_table_data_augmentation.object_pose.segmentation import (
    MaskSequenceParameters,
    MaskRefinementMetrics,
    evaluate_mask_candidates,
    filter_reachable_table_components,
    fuse_bidirectional_masks,
    mask_iou,
    refine_table_mask,
    select_mask_candidate_sequence,
    registration_ordinals,
    target_point_prompt,
)
from data.flip_table_data_augmentation.scripts.run_grounded_sam2_masks import (
    _mask_observation_ordinals,
    _mask_rejection_reasons,
    _terminal_source_gap,
)
from data.flip_table_data_augmentation.object_pose.robot_silhouette import (
    demo_hand_to_dex1_joint_position,
    projected_convex_hull_mask,
    robot_silhouette_coverage_is_plausible,
)
from data.flip_table_data_augmentation.object_pose.temporal_selection import (
    TemporalAnchor,
    TemporalSelectionError,
    TemporalSelectionParameters,
    TemporalSelectionResult,
    _ALLOWED_PREVIOUS_STATES,
    _candidate_visual_evidence,
    _causal_expanded_anchor,
    _expanded_anchor,
    audit_temporal_evidence_gaps,
    select_causally_constrained_poses,
    select_temporally_consistent_poses,
    temporal_static_visual_costs,
    temporal_visual_costs,
)
from data.flip_table_data_augmentation.scripts.track_foundationpose_episode import (
    _attach_terminal_reverse_lineage_consensus,
    _append_causal_current_frame_candidate,
    _bidirectional_consensus_records,
    _current_eef_poses_from_record,
    _propagated_root_candidates,
    _retain_manipulation_and_final_static_anchors,
    _select_initial_static_anchor,
    _select_terminal_static_anchor_ordinals,
    _temporal_solver_candidate_indices,
    _select_with_optional_dense_anchors,
    _with_terminal_confirmation_registration,
    _track_rejection_reasons,
    _track_direction,
    contact_lineage_priority_indices,
    evaluate_unsegmented_multiview_depths,
    lineage_preserving_candidate_indices,
    load_initial_root_from_table,
    propagation_candidate_indices,
    rank_registration_depths,
)


def test_source_cad_seed_requires_accepted_matching_rgb_only_evidence() -> None:
    document = {
        "schema_version": "team_ramen_flip_table_source_cad_alignment/v1",
        "accepted_for_fixed_scene_proposal": True,
        "source": {"episode_index": 250},
        "method": {"requires_simulator_ground_truth": False},
        "fixed_scene_root_from_table": np.eye(4).tolist(),
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "source_cad_alignment.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        pose, evidence = load_initial_root_from_table(path, episode_index=250)
        np.testing.assert_allclose(pose, np.eye(4))
        assert evidence is not None
        assert evidence["use"] == "initial_registration_candidate_only"

        document["source"]["episode_index"] = 499
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="different source episode"):
            load_initial_root_from_table(path, episode_index=250)

        document["source"]["episode_index"] = 250
        document["method"]["requires_simulator_ground_truth"] = True
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="without simulator ground truth"):
            load_initial_root_from_table(path, episode_index=250)


def test_initial_source_cad_seed_requires_static_visual_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "source_cad_seed",
        "mask_precision": 0.9,
        "mask_explained_fraction": 0.9,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.9,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.asarray((4.5, 3.0)),
    )

    assert _select_initial_static_anchor(
        [(anchor, ("source_cad_seed",))], parameters
    ) == (0, "source_cad_seed_static_rgbd")

    assert _select_initial_static_anchor(
        [(anchor, ("global_registration",))], parameters
    ) is None


def test_track_gate_requires_bidirectional_and_rendered_alignment() -> None:
    assert _track_rejection_reasons(
        bidirectional_pass=True, rendered_alignment_pass=True
    ) == []
    assert _track_rejection_reasons(
        bidirectional_pass=False, rendered_alignment_pass=True
    ) == ["bidirectional_pose_gate"]
    assert _track_rejection_reasons(
        bidirectional_pass=True, rendered_alignment_pass=False
    ) == ["rendered_alignment_gate"]
    assert _track_rejection_reasons(
        bidirectional_pass=False, rendered_alignment_pass=False
    ) == ["bidirectional_pose_gate", "rendered_alignment_gate"]


def test_mask_gate_rejection_reasons_identify_each_failed_condition() -> None:
    passing = {
        "tracking_eligible_frames": 9,
        "minimum_tracking_eligible_frames": 8,
        "first_selected": True,
        "terminal_anchor_pass": True,
    }
    assert _mask_rejection_reasons(passing) == []
    assert _mask_rejection_reasons(
        {
            **passing,
            "tracking_eligible_frames": 7,
            "first_selected": False,
            "terminal_anchor_pass": False,
        }
    ) == [
        "insufficient_registration_mask_coverage",
        "missing_initial_registration_mask",
        "missing_terminal_registration_mask",
    ]


def test_mask_observation_ordinals_keep_wrist_bridges_separate() -> None:
    def frame(ordinal: int, head: bool, left: bool, right: bool):
        present = {"head_left": head, "left_wrist": left, "right_wrist": right}
        return {
            "ordinal": ordinal,
            "views": {
                name: {"selected_candidate_index": 0 if value else None}
                for name, value in present.items()
            },
        }

    primary, wrist_bridges, eligible = _mask_observation_ordinals(
        [
            frame(0, True, False, False),
            frame(1, False, True, True),
            frame(2, False, True, False),
            frame(3, True, True, True),
        ]
    )

    assert primary == [0, 3]
    assert wrist_bridges == [1]
    assert eligible == [0, 1, 3]


def test_terminal_source_gap_does_not_promote_wrist_bridge_to_primary() -> None:
    source_frames = np.asarray((0, 15, 30, 45), dtype=np.int64)

    assert _terminal_source_gap(source_frames, [0, 1]) == 30
    assert _terminal_source_gap(source_frames, [0, 1, 3]) == 0
    assert _terminal_source_gap(source_frames, []) is None


class _FakeFoundationPose:
    def get_tf_to_centered_mesh(self) -> np.ndarray:
        return np.eye(4)

    def _pose(self, rgb: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[0, 3] = float(rgb[0, 0, 0])
        return pose

    def register(self, *, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        return self._pose(rgb)

    def track_one(self, *, rgb: np.ndarray, **_kwargs) -> np.ndarray:
        return self._pose(rgb)


class _SeedCapturingSelector:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray | None, tuple[str, ...] | None]] = []

    def __call__(
        self,
        *,
        root_from_camera: np.ndarray,
        additional_root_candidates: np.ndarray | None,
        additional_sources: tuple[str, ...] | None,
        **_kwargs,
    ) -> tuple[np.ndarray, str, dict[str, object]]:
        self.calls.append((additional_root_candidates, additional_sources))
        return np.linalg.inv(root_from_camera), "register_geometry_test", {}


def test_current_eef_poses_are_loaded_from_observed_fk_record() -> None:
    left = np.eye(4)
    left[:3, 3] = (0.2, 0.3, 0.4)
    right = np.eye(4)
    right[:3, :3] = Rotation.from_euler("xyz", (0.1, -0.2, 0.3)).as_matrix()
    right[:3, 3] = (-0.2, 0.5, 0.7)
    record = {
        "eef_current_root_from_fk": np.stack((left, right)).reshape(2, 16).tolist(),
        "ee_action": [99.0] * 12,
    }

    actual = _current_eef_poses_from_record(record)

    np.testing.assert_allclose(actual, np.stack((left, right)), atol=1.0e-12)


def test_current_eef_poses_reject_missing_observed_fk_record() -> None:
    with pytest.raises(ValueError, match="robot_q_current"):
        _current_eef_poses_from_record({"ee_action": [0.0] * 12})


def test_bidirectional_consensus_resolves_symmetry_and_rejects_outlier() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "global_registration",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
    }
    reverse = np.eye(4)
    reverse[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    outlier = np.eye(4)
    outlier[0, 3] = 0.2
    common = {
        "ordinal": 0,
        "source_frame_index": 0,
        "eef_poses_root": np.stack((np.eye(4), np.eye(4))),
        "hand_observed_position": np.full(2, 4.5),
    }
    forward_anchor = TemporalAnchor(
        candidate_poses_root=np.stack((np.eye(4), outlier)),
        candidate_metrics=(metric, metric),
        **common,
    )
    backward_anchor = TemporalAnchor(
        candidate_poses_root=reverse[None],
        candidate_metrics=(metric,),
        **common,
    )

    records = _bidirectional_consensus_records(
        forward_anchor=forward_anchor,
        backward_anchor=backward_anchor,
        parameters=parameters,
        maximum_translation_error_m=0.03,
        maximum_rotation_error_rad=0.15,
    )

    assert records[0]["passes_gate"] is True
    assert records[0]["reverse_symmetry_index"] == 1
    assert records[1]["passes_gate"] is False


def test_bidirectional_consensus_records_missing_reverse_visual_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    valid = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
    }
    invalid = {
        **valid,
        "mask_precision": 0.0,
        "mask_explained_fraction": 0.0,
        "median_absolute_depth_error_m": None,
    }
    common = {
        "ordinal": 0,
        "source_frame_index": 0,
        "eef_poses_root": np.stack((np.eye(4), np.eye(4))),
        "hand_observed_position": np.full(2, 4.5),
    }

    records = _bidirectional_consensus_records(
        forward_anchor=TemporalAnchor(
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=(valid,),
            **common,
        ),
        backward_anchor=TemporalAnchor(
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=(invalid,),
            **common,
        ),
        parameters=parameters,
        maximum_translation_error_m=0.03,
        maximum_rotation_error_rad=0.15,
    )

    assert records == [
        {
            "passes_gate": False,
            "translation_error_m": None,
            "rotation_error_rad": None,
            "maximum_translation_error_m": 0.03,
            "maximum_rotation_error_rad": 0.15,
            "validation_mode": "reverse_registration_no_visual_candidate",
            "reverse_candidate_index": None,
            "reverse_candidate_source": None,
            "reverse_symmetry_index": None,
        }
    ]


def test_terminal_reverse_consensus_requires_static_verified_parent() -> None:
    evidence = {
        "temporal_candidate_metrics": [{}, {}, {}],
        "temporal_candidate_sources": [
            "static_carry_reverse",
            "left_eef_carry_reverse",
            "static_carry_reverse",
        ],
        "temporal_candidate_lineage_ids": [0, 0, 1],
    }
    terminal = {
        "temporal_candidate_metrics": [
            {"bidirectional_consensus": {"passes_gate": True}},
            {"bidirectional_consensus": {"passes_gate": False}},
        ],
        "temporal_candidate_sources": [
            "global_registration",
            "continued_tracking",
        ],
    }

    _attach_terminal_reverse_lineage_consensus(
        evidence,
        terminal,
        terminal_ordinal=10,
        maximum_translation_error_m=0.03,
        maximum_rotation_error_rad=0.15,
    )

    consensus = [
        metric["bidirectional_consensus"]
        for metric in evidence["temporal_candidate_metrics"]
    ]
    assert [value["passes_gate"] for value in consensus] == [True, False, False]
    assert consensus[0]["validation_mode"] == "terminal_reverse_static_lineage"
    assert consensus[0]["reverse_candidate_index"] == 0
    assert consensus[0]["reverse_candidate_source"] == "global_registration"
    assert consensus[0]["terminal_registration_ordinal"] == 10


def test_inverse_brown_rectification_map_is_identity_without_distortion() -> None:
    intrinsic = np.array([[400.0, 0.0, 2.0], [0.0, 410.0, 1.0], [0.0, 0.0, 1.0]])
    map_x, map_y = inverse_brown_conrady_rectification_maps(
        intrinsic,
        np.zeros(5),
        width=5,
        height=3,
    )
    expected_x, expected_y = np.meshgrid(np.arange(5), np.arange(3))
    np.testing.assert_allclose(map_x, expected_x)
    np.testing.assert_allclose(map_y, expected_y)


def test_inverse_brown_rectification_map_inverts_calibrated_model() -> None:
    intrinsic = np.array([[435.0, 0.0, 3.0], [0.0, 434.0, 2.0], [0.0, 0.0, 1.0]])
    coefficients = np.array([-0.05, 0.06, 0.001, 0.0011, -0.02])
    map_x, map_y = inverse_brown_conrady_rectification_maps(
        intrinsic,
        coefficients,
        width=7,
        height=5,
    )
    distorted_x = (map_x - intrinsic[0, 2]) / intrinsic[0, 0]
    distorted_y = (map_y - intrinsic[1, 2]) / intrinsic[1, 1]
    radius2 = distorted_x**2 + distorted_y**2
    k1, k2, p1, p2, k3 = coefficients
    radial = 1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
    undistorted_x = (
        distorted_x * radial
        + 2.0 * p1 * distorted_x * distorted_y
        + p2 * (radius2 + 2.0 * distorted_x**2)
    )
    undistorted_y = (
        distorted_y * radial
        + p1 * (radius2 + 2.0 * distorted_y**2)
        + 2.0 * p2 * distorted_x * distorted_y
    )
    pixel_x, pixel_y = np.meshgrid(np.arange(7), np.arange(5))
    np.testing.assert_allclose(
        undistorted_x,
        (pixel_x - intrinsic[0, 2]) / intrinsic[0, 0],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        undistorted_y,
        (pixel_y - intrinsic[1, 2]) / intrinsic[1, 1],
        atol=1.0e-7,
    )


def test_registration_ordinals_include_unaligned_endpoint() -> None:
    frames = tuple(range(0, 744, 3)) + (743,)
    selected = registration_ordinals(frames, 30)
    assert selected[0] == 0
    assert selected[-1] == len(frames) - 1
    assert all(frames[index] % 30 == 0 for index in selected[1:-1])


def test_temporal_candidates_propagate_with_static_and_each_eef() -> None:
    candidates = np.repeat(np.eye(4)[None], 2, axis=0)
    candidates[1, 2, 3] = 0.4
    previous_eef = np.repeat(np.eye(4)[None], 2, axis=0)
    current_eef = previous_eef.copy()
    current_eef[0, 0, 3] = 0.2
    current_eef[1, 1, 3] = -0.3

    propagated, sources = _propagated_root_candidates(
        candidates, previous_eef, current_eef, np.ones(2, dtype=bool)
    )

    np.testing.assert_allclose(propagated[:2], candidates)
    np.testing.assert_allclose(propagated[2:4, 0, 3], 0.2)
    np.testing.assert_allclose(propagated[4:6, 1, 3], -0.3)
    np.testing.assert_allclose(propagated[6:8, 0, 3], 0.1)
    np.testing.assert_allclose(propagated[6:8, 1, 3], -0.15)
    assert sources == (
        "static_carry_forward",
        "static_carry_forward",
        "left_eef_carry_forward",
        "left_eef_carry_forward",
        "right_eef_carry_forward",
        "right_eef_carry_forward",
        "bimanual_eef_carry_forward",
        "bimanual_eef_carry_forward",
    )


def test_bimanual_candidate_averages_propagated_table_poses() -> None:
    candidate = np.eye(4)[None]
    candidate[0, 0, 3] = 1.0
    previous_eef = np.repeat(np.eye(4)[None], 2, axis=0)
    current_eef = previous_eef.copy()
    current_eef[0, :3, :3] = Rotation.from_euler("z", 0.5).as_matrix()

    propagated, _ = _propagated_root_candidates(
        candidate, previous_eef, current_eef, np.ones(2, dtype=bool)
    )

    expected_translation = 0.5 * (
        propagated[1, :3, 3] + propagated[2, :3, 3]
    )
    np.testing.assert_allclose(propagated[3, :3, 3], expected_translation)


def test_temporal_candidates_exclude_open_hand_carry_models() -> None:
    candidates = np.eye(4)[None]
    previous_eef = np.repeat(np.eye(4)[None], 2, axis=0)
    current_eef = previous_eef.copy()
    current_eef[0, 0, 3] = 0.2
    current_eef[1, 1, 3] = -0.3

    propagated, sources = _propagated_root_candidates(
        candidates,
        previous_eef,
        current_eef,
        np.asarray((False, True)),
    )

    assert propagated.shape == (2, 4, 4)
    assert sources == ("static_carry_forward", "right_eef_carry_forward")
    np.testing.assert_allclose(propagated[1, 1, 3], -0.3)


def test_temporal_candidate_sources_record_reverse_propagation() -> None:
    eef = np.repeat(np.eye(4)[None], 2, axis=0)

    _, sources = _propagated_root_candidates(
        np.eye(4)[None],
        eef,
        eef,
        np.asarray((True, False)),
        direction="reverse",
    )

    assert sources == ("static_carry_reverse", "left_eef_carry_reverse")


def test_optional_dense_anchor_is_pruned_after_physical_path_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = tuple(
        TemporalAnchor(
            ordinal=ordinal,
            source_frame_index=ordinal * 3,
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=({},),
            eef_poses_root=np.repeat(np.eye(4)[None], 2, axis=0),
            hand_observed_position=np.ones(2),
        )
        for ordinal in range(4)
    )
    calls = []
    selected = object()

    def fake_select(values, **_kwargs):
        values = tuple(values)
        calls.append(tuple(anchor.ordinal for anchor in values))
        if any(anchor.ordinal == 1 for anchor in values):
            raise TemporalSelectionError(
                "dense anchor blocks the path", {"anchor_index": 1}
            )
        return selected

    monkeypatch.setattr(
        "data.flip_table_data_augmentation.scripts.track_foundationpose_episode."
        "select_temporally_consistent_poses",
        fake_select,
    )

    result, retained, _, pruned = _select_with_optional_dense_anchors(
        anchors,
        tuple(("candidate",) for _ in anchors),
        optional_ordinals={1},
        source_fps=30.0,
        parameters=object(),
    )

    assert result is selected
    assert calls == [(0, 1, 2, 3), (0, 2, 3)]
    assert tuple(anchor.ordinal for anchor in retained) == (0, 2, 3)
    assert pruned[0]["ordinal"] == 1


def test_mask_candidate_gate_and_selection() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[100:220, 200:320] = 220
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[100:220, 200:320] = 0.6
    valid = np.zeros((480, 640), dtype=bool)
    valid[100:220, 200:320] = True
    too_small = np.zeros_like(valid)
    too_small[10:20, 10:20] = True
    metrics, selected = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=depth,
        masks=[valid, too_small],
        detector_scores=[0.9, 0.99],
        segmentation_iou_scores=[0.92, 0.95],
        detector_boxes_xyxy=[[195, 95, 325, 225], [9, 9, 21, 21]],
        minimum_segmentation_iou=0.6,
        minimum_mask_area_fraction=0.02,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.25,
    )
    assert selected == 0
    assert metrics[0].passes_gate
    assert metrics[0].valid_depth_fraction_in_mask == 1.0
    assert metrics[0].neutral_bright_fraction == 1.0
    assert not metrics[1].passes_gate
    assert "mask_too_small" in metrics[1].rejection_reasons
    assert metrics[0].to_json()["candidate_index"] == 0


def test_mask_iou_empty_and_overlap() -> None:
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    assert mask_iou(first, second) == 0.0
    first[:2, :2] = True
    second[1:3, 1:3] = True
    assert mask_iou(first, second) == 1.0 / 7.0


def test_bidirectional_mask_fusion_returns_conservative_intersection() -> None:
    forward = np.zeros((10, 10), dtype=bool)
    backward = np.zeros((10, 10), dtype=bool)
    forward[1:7, 1:7] = True
    backward[2:8, 1:7] = True

    fused, metrics = fuse_bidirectional_masks(
        forward,
        backward,
        minimum_iou=0.5,
        minimum_area_fraction=0.05,
        maximum_area_fraction=0.8,
    )

    assert fused is not None
    np.testing.assert_array_equal(fused, forward & backward)
    assert metrics.passes_gate
    assert metrics.rejection_reasons == ()
    assert metrics.bidirectional_iou == 30.0 / 42.0


def test_bidirectional_mask_fusion_rejects_disagreement_and_small_intersection() -> None:
    forward = np.zeros((10, 10), dtype=bool)
    backward = np.zeros((10, 10), dtype=bool)
    forward[:4, :4] = True
    backward[6:, 6:] = True

    fused, metrics = fuse_bidirectional_masks(
        forward,
        backward,
        minimum_iou=0.5,
        minimum_area_fraction=0.05,
        maximum_area_fraction=0.8,
    )

    assert fused is None
    assert not metrics.passes_gate
    assert metrics.rejection_reasons == (
        "bidirectional_iou_below_threshold",
        "fused_area_out_of_bounds",
    )


def test_mask_candidate_gate_supports_calibrated_rgb_only_view() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:260, 160:420] = True
    metrics, selected = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=None,
        masks=[mask],
        detector_scores=[0.9],
        segmentation_iou_scores=[0.9],
        detector_boxes_xyxy=[[150, 90, 430, 270]],
        minimum_segmentation_iou=0.6,
        minimum_mask_area_fraction=0.02,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.0,
    )
    assert selected == 0
    assert metrics[0].passes_gate
    assert metrics[0].valid_depth_fraction_in_mask is None
    assert metrics[0].median_depth_m is None


def test_mask_candidate_keeps_low_sam_score_as_geometry_proposal() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:260, 160:420] = True
    metrics, selected = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=None,
        masks=[mask],
        detector_scores=[0.9],
        segmentation_iou_scores=[0.01],
        detector_boxes_xyxy=[[150, 90, 430, 270]],
        minimum_segmentation_iou=0.0,
        minimum_mask_area_fraction=0.02,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.0,
    )

    assert selected == 0
    assert metrics[0].passes_gate
    assert metrics[0].segmentation_iou_score == 0.01


def test_mask_candidate_ranking_penalizes_robot_overlap_and_fragmentation() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    depth = np.full((480, 640), 0.6, dtype=np.float32)
    fragmented = np.zeros((480, 640), dtype=bool)
    fragmented[50:250, 50:350] = True
    clean = np.zeros_like(fragmented)
    clean[70:230, 90:330] = True
    refinements = [
        MaskRefinementMetrics(100_000, 30_000, 65_000, 5, 4, int(fragmented.sum())),
        MaskRefinementMetrics(42_000, 2_000, 39_000, 1, 1, int(clean.sum())),
    ]
    metrics, selected = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=depth,
        masks=[fragmented, clean],
        detector_scores=[0.40, 0.30],
        segmentation_iou_scores=[0.90, 0.95],
        detector_boxes_xyxy=[[40, 40, 360, 260], [80, 60, 340, 240]],
        refinement_metrics=refinements,
        minimum_segmentation_iou=0.6,
        minimum_mask_area_fraction=0.02,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.25,
    )
    assert selected == 1
    assert metrics[1].refinement_retention_fraction > metrics[0].refinement_retention_fraction
    assert metrics[0].retained_component_count == 4


def test_global_mask_sequence_prefers_complete_temporally_consistent_table() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    complete = np.zeros((480, 640), dtype=bool)
    complete[80:300, 100:500] = True
    fragment = np.zeros_like(complete)
    fragment[130:250, 210:390] = True
    masks_by_frame = []
    metrics_by_frame = []
    for detector_scores in ([0.75, 0.95], [0.80, 0.96], [0.78, 0.94]):
        metrics, _ = evaluate_mask_candidates(
            rgb=rgb,
            depth_m=None,
            masks=[complete, fragment],
            detector_scores=detector_scores,
            segmentation_iou_scores=[0.9, 0.9],
            detector_boxes_xyxy=[[90, 70, 510, 310], [200, 120, 400, 260]],
            minimum_segmentation_iou=0.0,
            minimum_mask_area_fraction=0.02,
            maximum_mask_area_fraction=0.75,
            minimum_valid_depth_fraction=0.0,
        )
        masks_by_frame.append((complete, fragment))
        metrics_by_frame.append(metrics)

    selected = select_mask_candidate_sequence(
        masks_by_frame=masks_by_frame,
        metrics_by_frame=metrics_by_frame,
        parameters=MaskSequenceParameters(),
    )

    assert selected == (0, 0, 0)


def test_primary_mask_sequence_recovers_disconnected_assembled_table() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    fragment = np.zeros((480, 640), dtype=bool)
    fragment[80:160, 80:160] = True
    complete = fragment.copy()
    complete[80:160, 240:320] = True
    complete[240:320, 80:160] = True
    complete[240:320, 240:320] = True

    def metrics_for(masks: tuple[np.ndarray, ...]) -> tuple[object, ...]:
        refinements = tuple(
            MaskRefinementMetrics(
                input_pixels=int(mask.sum()),
                robot_overlap_pixels=0,
                bright_non_robot_pixels=int(mask.sum()),
                connected_components=index,
                retained_components=index,
                output_pixels=int(mask.sum()),
            )
            for mask, index in zip(masks, (1, 4), strict=True)
        )
        metrics, _ = evaluate_mask_candidates(
            rgb=rgb,
            depth_m=None,
            masks=masks,
            detector_scores=[0.9] * len(masks),
            segmentation_iou_scores=[0.95, 0.4][: len(masks)],
            detector_boxes_xyxy=[[0, 0, 640, 480]] * len(masks),
            refinement_metrics=refinements,
            minimum_segmentation_iou=0.0,
            minimum_mask_area_fraction=0.01,
            maximum_mask_area_fraction=0.75,
            minimum_valid_depth_fraction=0.0,
        )
        return metrics

    first_metrics, _ = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=None,
        masks=[fragment],
        detector_scores=[0.9],
        segmentation_iou_scores=[0.95],
        detector_boxes_xyxy=[[0, 0, 640, 480]],
        refinement_metrics=[
            MaskRefinementMetrics(
                int(fragment.sum()), 0, int(fragment.sum()), 1, 1, int(fragment.sum())
            )
        ],
        minimum_segmentation_iou=0.0,
        minimum_mask_area_fraction=0.01,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.0,
    )
    terminal_metrics = metrics_for((fragment, complete))
    masks = [(fragment,)] + [(fragment, complete)] * 4
    metrics = [first_metrics] + [terminal_metrics] * 4

    selected = select_mask_candidate_sequence(
        masks_by_frame=masks,
        metrics_by_frame=metrics,
        parameters=MaskSequenceParameters(
            area_weight=1.5,
            fragmentation_weight=0.05,
            transition_area_weight=0.1,
            fragmentation_mode="logarithmic",
        ),
    )

    assert selected == (0, 1, 1, 1, 1)


def test_auxiliary_mask_sequence_rejects_unrelated_large_region() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    target = np.zeros((480, 640), dtype=bool)
    target[80:180, 80:180] = True
    unrelated = np.zeros_like(target)
    unrelated[220:470, 300:630] = True

    def metrics_for(masks: tuple[np.ndarray, ...]) -> tuple[object, ...]:
        values, _ = evaluate_mask_candidates(
            rgb=rgb,
            depth_m=None,
            masks=masks,
            detector_scores=[0.8] * len(masks),
            segmentation_iou_scores=[0.8] * len(masks),
            detector_boxes_xyxy=[[0, 0, 640, 480]] * len(masks),
            minimum_segmentation_iou=0.0,
            minimum_mask_area_fraction=0.01,
            maximum_mask_area_fraction=0.75,
            minimum_valid_depth_fraction=0.0,
        )
        return values

    first = metrics_for((target,))
    later = metrics_for((target, unrelated))
    assert select_mask_candidate_sequence(
        masks_by_frame=[(target,), (target, unrelated), (target, unrelated)],
        metrics_by_frame=[first, later, later],
        parameters=MaskSequenceParameters(),
    ) == (0, 0, 0)


def test_global_mask_sequence_keeps_missing_frame_explicit() -> None:
    rgb = np.full((480, 640, 3), 220, dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:260, 160:420] = True
    metrics, _ = evaluate_mask_candidates(
        rgb=rgb,
        depth_m=None,
        masks=[mask],
        detector_scores=[0.9],
        segmentation_iou_scores=[0.9],
        detector_boxes_xyxy=[[150, 90, 430, 270]],
        minimum_segmentation_iou=0.0,
        minimum_mask_area_fraction=0.02,
        maximum_mask_area_fraction=0.75,
        minimum_valid_depth_fraction=0.0,
    )

    assert select_mask_candidate_sequence(
        masks_by_frame=[(mask,), (), (mask,)],
        metrics_by_frame=[metrics, (), metrics],
    ) == (0, None, 0)


def test_table_mask_refinement_removes_robot_dark_pixels_and_speckles() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[80:200, 100:300] = 220
    rgb[250:450, 100:500] = 30
    candidate = np.zeros((480, 640), dtype=bool)
    candidate[80:200, 100:300] = True
    candidate[250:450, 100:500] = True
    candidate[20:25, 20:25] = True
    robot = np.zeros_like(candidate)
    robot[80:200, 100:150] = True
    refined, metrics = refine_table_mask(
        rgb=rgb,
        candidate_mask=candidate,
        robot_silhouette=robot,
        minimum_value=100,
        minimum_component_area_px=256,
    )
    assert not refined[:, :150].any()
    assert refined[100:180, 160:280].all()
    assert not refined[250:450].any()
    assert metrics.robot_overlap_pixels == 120 * 50
    assert metrics.retained_components == 1


def test_reachable_component_filter_removes_distant_background_object() -> None:
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:260, 300:340] = True
    mask[200:260, 520:560] = True
    depth = np.ones((480, 640), dtype=np.float32)
    intrinsic = np.array(
        [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    filtered, metrics = filter_reachable_table_components(
        candidate_mask=mask,
        depth_m=depth,
        intrinsic_matrix=intrinsic,
        root_from_camera=np.eye(4),
        eef_positions_root=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        maximum_median_eef_distance_m=0.85,
        minimum_component_valid_depth_fraction=0.25,
    )

    assert filtered[200:260, 300:340].all()
    assert not filtered[200:260, 520:560].any()
    assert metrics.input_component_count == 2
    assert metrics.retained_component_count == 1
    assert metrics.components[0].retained
    assert metrics.components[1].rejection_reason == "outside_reachable_workspace"


def test_dex1_hand_conversion_and_convex_hull_projection() -> None:
    np.testing.assert_allclose(
        demo_hand_to_dex1_joint_position(np.array([0.0, 4.5])),
        [-0.02, -0.02, 0.0245, 0.0245],
    )
    points = np.array(
        [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0]]
    )
    mask = projected_convex_hull_mask(
        points,
        root_from_camera=np.eye(4),
        intrinsic_matrix=np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0, 0, 1]]),
        width=200,
        height=200,
    )
    assert mask[100, 100]
    assert not mask[10, 10]


def test_robot_silhouette_coverage_accepts_offscreen_robot() -> None:
    assert robot_silhouette_coverage_is_plausible(0.0)
    assert robot_silhouette_coverage_is_plausible(0.75)
    assert not robot_silhouette_coverage_is_plausible(0.751)
    assert not robot_silhouette_coverage_is_plausible(float("nan"))


def test_target_point_prompt_uses_nearest_white_non_robot_pixel_to_eefs() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[220:281, 260:381] = 220
    robot = np.zeros((480, 640), dtype=bool)
    robot[330:480, 180:460] = True
    eef = np.repeat(np.eye(4)[None], 2, axis=0)
    eef[0, :3, 3] = (-0.1, 1.5, 1.0)
    eef[1, :3, 3] = (0.1, 1.5, 1.0)
    result = target_point_prompt(
        rgb=rgb,
        robot_silhouette=robot,
        eef_poses_root=eef,
        root_from_camera=np.eye(4),
        intrinsic_matrix=np.array(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        detector_box_xyxy=(200.0, 150.0, 450.0, 320.0),
        minimum_value=100,
        maximum_point_distance_px=240,
        principal_point_weight=0.75,
    )

    assert result.passes_gate
    assert result.visible_eef_count == 2
    assert result.point_xy is not None
    assert 260 <= result.point_xy[0] < 381
    assert 220 <= result.point_xy[1] < 281


def test_target_point_prompt_uses_calibrated_principal_point_without_visible_eef() -> None:
    eef = np.repeat(np.eye(4)[None], 2, axis=0)
    eef[:, 2, 3] = -1.0
    result = target_point_prompt(
        rgb=np.full((480, 640, 3), 220, dtype=np.uint8),
        robot_silhouette=np.zeros((480, 640), dtype=bool),
        eef_poses_root=eef,
        root_from_camera=np.eye(4),
        intrinsic_matrix=np.array(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        detector_box_xyxy=(0.0, 0.0, 640.0, 480.0),
        minimum_value=100,
        maximum_point_distance_px=240,
        principal_point_weight=0.75,
    )

    assert result.passes_gate
    assert result.visible_eef_count == 0
    assert result.reference_source == "calibrated_principal_point"
    assert result.point_xy == (320, 240)


def test_rectified_camera_transform_axes() -> None:
    camera = load_pipeline_config().cameras[0]
    parent = np.eye(4)
    parent[:3, :3] = Rotation.from_quat(camera.offset_quaternion_xyzw).inv().as_matrix()
    parent[:3, 3] = -parent[:3, :3] @ np.asarray(camera.offset_position_m)
    result = root_from_rectified_opencv_camera(parent, camera, np.eye(3))
    np.testing.assert_allclose(result, OPENGL_FROM_OPENCV, atol=1.0e-7)


def test_temporal_selection_prefers_bimanual_rigid_path() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    translations = (0.0, 0.0, 0.18, 0.36, 0.36, 0.36, 0.36)
    rotations = (0.0, 0.0, 0.45, 0.9, 0.9, 0.9, 0.9)
    anchors = []
    for index, (translation, angle) in enumerate(zip(translations, rotations, strict=True)):
        table = np.eye(4)
        table[:3, :3] = Rotation.from_euler("y", angle).as_matrix()
        table[0, 3] = translation
        distractor = table.copy()
        distractor[1, 3] += 0.35 if index % 2 else -0.35
        eef = np.stack((table.copy(), table.copy()))
        eef[0, :3, 3] += table[:3, :3] @ np.array([0.0, 0.18, 0.05])
        eef[1, :3, 3] += table[:3, :3] @ np.array([0.0, -0.18, 0.05])
        true_metrics = {
            "mask_precision": 0.8,
            "mask_explained_fraction": 0.75,
            "depth_overlap_fraction": 1.0,
            "depth_consistent_union_fraction": 0.18,
            "median_absolute_depth_error_m": 0.015,
            "bidirectional_consensus": {"passes_gate": True},
        }
        distractor_metrics = {
            "mask_precision": 0.9,
            "mask_explained_fraction": 0.85,
            "depth_overlap_fraction": 1.0,
            "depth_consistent_union_fraction": 0.21,
            "median_absolute_depth_error_m": 0.012,
            "bidirectional_consensus": {"passes_gate": True},
        }
        anchors.append(
            TemporalAnchor(
                ordinal=index * 10,
                source_frame_index=index * 30,
                candidate_poses_root=np.stack((table, distractor)),
                candidate_metrics=(true_metrics, distractor_metrics),
                eef_poses_root=eef,
                hand_observed_position=(
                    np.array([2.0, 2.0])
                    if index in (2, 3)
                    else np.array([2.0, 4.5])
                    if index == 4
                    else np.array([4.5, 4.5])
                ),
            )
        )
    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )
    np.testing.assert_array_equal(result.selected_candidate_indices, 0)
    assert np.count_nonzero(result.phase_indices == 2) >= 2
    assert result.phase_indices[0] == 0
    assert result.phase_indices[-1] == 4


def test_causal_temporal_selection_does_not_bind_initially_closed_idle_hand() -> None:
    """A closed Dex1 reading alone must not create a false initial grasp."""

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.03,
        maximum_grasp_relative_rotation_step_rad=0.2,
        grasp_observed_position_max=3.5,
    )
    true_metric = {
        "mask_precision": 0.85,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    distractor_metric = {
        "mask_precision": 0.5,
        "mask_explained_fraction": 0.5,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.4,
        "median_absolute_depth_error_m": 0.02,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchors = []
    for index in range(4):
        table = np.eye(4)
        table[0, 3] = 0.02 * index
        distractor = table.copy()
        distractor[1, 3] += 0.25
        eef = np.repeat(np.eye(4)[None], 2, axis=0)
        # The right hand is already closed at the first source frame but is
        # visually unbound until the first causal RGB-D update is accepted.
        eef[1, :3, 3] = table[:3, 3] + np.array([0.08, -0.12, 0.0])
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 3,
                candidate_poses_root=np.stack((table, distractor)),
                candidate_metrics=(true_metric, distractor_metric),
                eef_poses_root=eef,
                hand_observed_position=np.array([4.5, 3.0]),
            )
        )

    result = select_causally_constrained_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    np.testing.assert_array_equal(result.selected_candidate_indices, 0)
    assert result.selected_evidence_modes[1] == "causal_attachment_rgbd"
    assert result.selected_evidence_modes[2] == "causal_attachment_rgbd"


def test_causal_temporal_selection_releases_open_hand_constraint() -> None:
    """A released hand must not constrain later table motion."""

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.03,
        maximum_grasp_relative_rotation_step_rad=0.2,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.85,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchors = []
    for index, hand_position in enumerate((4.5, 3.0, 4.5, 4.5)):
        table = np.eye(4)
        table[0, 3] = 0.02 * index
        eef = np.repeat(np.eye(4)[None], 2, axis=0)
        # The hand moves incompatibly after release.  The final two frames
        # remain valid only when that stale attachment is cleared.
        eef[0, 0, 3] = 0.02 if index < 2 else 1.0
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 3,
                candidate_poses_root=table[None],
                candidate_metrics=(metric,),
                eef_poses_root=eef,
                hand_observed_position=np.array([hand_position, 4.5]),
            )
        )

    result = select_causally_constrained_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    assert int(result.phase_indices[-1]) == 4


def test_causal_temporal_selection_keeps_hand_compatible_visual_hypothesis() -> None:
    """The causal beam must not lose a valid symmetric start pose greedily."""

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    strong_metric = {
        "mask_precision": 0.9,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    weaker_metric = {**strong_metric, "mask_precision": 0.6}
    positions = ((0.0, 0.1), (0.05, 0.15), (0.25,), (0.25,))
    anchors = []
    for index, values in enumerate(positions):
        poses = []
        for translation in values:
            pose = np.eye(4)
            pose[0, 3] = translation
            poses.append(pose)
        metrics = (
            (strong_metric, weaker_metric)
            if len(poses) == 2
            else (weaker_metric,)
        )
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 3,
                candidate_poses_root=np.asarray(poses),
                candidate_metrics=metrics,
                eef_poses_root=np.repeat(np.eye(4)[None], 2, axis=0),
                hand_observed_position=np.array([4.5, 4.5]),
            )
        )

    result = select_causally_constrained_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    # Candidate zero has stronger initial visual evidence, but only the
    # alternative start hypothesis reaches the later observed table pose.
    assert result.selected_candidate_indices[0] == 1
    np.testing.assert_allclose(result.selected_poses_root[-1][0, 3], 0.25)


def test_causal_selection_accepts_current_frame_static_rgbd_bridge() -> None:
    """A bounded static carry may bridge an unpaired reverse registration."""

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    strict = {
        "source": "global_registration",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.6,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    causal_bridge = {
        **strict,
        "source": "static_carry_forward",
        "bidirectional_consensus": {
            "passes_gate": False,
            "validation_mode": "causal_current_frame_rgbd_propagation",
        },
    }
    anchors = []
    for index in range(4):
        pose = np.eye(4)
        pose[0, 3] = 0.01 * index
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 3,
                candidate_poses_root=pose[None],
                candidate_metrics=(causal_bridge if index == 1 else strict,),
                eef_poses_root=np.repeat(np.eye(4)[None], 2, axis=0),
                hand_observed_position=np.array([4.5, 4.5]),
            )
        )

    result = select_causally_constrained_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    assert result.selected_evidence_modes[1] == "causal_static_bridge_rgbd"


def test_causal_current_frame_candidate_is_rendered_before_use() -> None:
    """A held hand adds an FK prediction only after current-view scoring."""

    class Selector:
        depth_consistency_m = 0.08
        auxiliary_view_score_weight = 0.5
        auxiliary_primary_support_saturation_fraction = 0.05

        @staticmethod
        def render_root_poses_depths(poses, root_from_cameras):
            assert set(root_from_cameras) == {
                "head_left",
                "left_wrist",
                "right_wrist",
            }
            return {
                name: np.ones((len(poses), 2, 2), dtype=np.float32)
                for name in root_from_cameras
            }

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    base_metric = {
        "source": "global_registration",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.6,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    eef = np.repeat(np.eye(4)[None], 2, axis=0)
    previous_pose = np.eye(4)
    current_eef = eef.copy()
    current_eef[1, 0, 3] = 0.01
    anchors = (
        TemporalAnchor(
            ordinal=0,
            source_frame_index=0,
            candidate_poses_root=previous_pose[None],
            candidate_metrics=(base_metric,),
            eef_poses_root=eef,
            hand_observed_position=np.array([4.5, 2.0]),
        ),
        TemporalAnchor(
            ordinal=1,
            source_frame_index=3,
            candidate_poses_root=previous_pose[None],
            candidate_metrics=(base_metric,),
            eef_poses_root=current_eef,
            hand_observed_position=np.array([4.5, 2.0]),
        ),
    )
    error = TemporalSelectionError(
        "missing candidate",
        {
            "anchor_index": 1,
            "causal_prefix": {
                "selected_poses_root": [previous_pose.tolist()],
                "attached_hands": [False, True],
            },
        },
    )
    masks = {
        name: {1: np.ones((2, 2), dtype=bool)}
        for name in ("head_left", "left_wrist", "right_wrist")
    }
    cameras = {
        name: np.repeat(np.eye(4)[None], 2, axis=0) for name in masks
    }
    result = _append_causal_current_frame_candidate(
        anchors,
        error,
        registration_selector=Selector(),
        frames=[
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
        ],
        dense_masks_by_view=masks,
        root_from_cameras=cameras,
        stereo_consistency_fractions=np.ones(2, dtype=np.float64),
        parameters=parameters,
    )

    assert result is not None
    assert len(result[1].candidate_metrics) == 2
    assert result[1].candidate_metrics[-1]["source"] == "right_eef_carry_forward"
    assert (
        result[1].candidate_metrics[-1]["bidirectional_consensus"]["validation_mode"]
        == "causal_current_frame_rgbd_propagation"
    )


def test_causal_current_frame_candidate_keeps_novel_pose_after_duplicate() -> None:
    """One existing carry must not suppress another current-frame prediction."""

    class Selector:
        depth_consistency_m = 0.08
        auxiliary_view_score_weight = 0.5
        auxiliary_primary_support_saturation_fraction = 0.05

        @staticmethod
        def render_root_poses_depths(poses, root_from_cameras):
            assert len(poses) == 1
            return {
                name: np.ones((len(poses), 2, 2), dtype=np.float32)
                for name in root_from_cameras
            }

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    base_metric = {
        "source": "global_registration",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.6,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    duplicate_pose = np.eye(4)
    novel_pose = np.eye(4)
    novel_pose[0, 3] = 0.02
    current_eef = np.repeat(np.eye(4)[None], 2, axis=0)
    anchors = (
        TemporalAnchor(
            ordinal=0,
            source_frame_index=0,
            candidate_poses_root=duplicate_pose[None],
            candidate_metrics=(base_metric,),
            eef_poses_root=np.repeat(np.eye(4)[None], 2, axis=0),
            hand_observed_position=np.array([2.0, 4.5]),
        ),
        TemporalAnchor(
            ordinal=1,
            source_frame_index=3,
            candidate_poses_root=np.stack((duplicate_pose, novel_pose)),
            candidate_metrics=(
                {**base_metric, "source": "left_eef_carry_forward"},
                base_metric,
            ),
            eef_poses_root=current_eef,
            hand_observed_position=np.array([2.0, 4.5]),
        ),
    )
    error = TemporalSelectionError(
        "missing candidate",
        {
            "anchor_index": 1,
            "causal_prefix": {
                "selected_poses_root": [duplicate_pose.tolist()],
                "attached_hands": [True, False],
            },
        },
    )
    masks = {
        name: {1: np.ones((2, 2), dtype=bool)}
        for name in ("head_left", "left_wrist", "right_wrist")
    }
    cameras = {
        name: np.repeat(np.eye(4)[None], 2, axis=0) for name in masks
    }

    with patch(
        "data.flip_table_data_augmentation.scripts.track_foundationpose_episode._propagated_root_candidates",
        return_value=(
            np.stack((duplicate_pose, novel_pose)),
            ("left_eef_carry_forward", "left_eef_carry_forward"),
        ),
    ):
        result = _append_causal_current_frame_candidate(
            anchors,
            error,
            registration_selector=Selector(),
            frames=[
                (
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.ones((2, 2), dtype=np.float32),
                ),
                (
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.ones((2, 2), dtype=np.float32),
                ),
            ],
            dense_masks_by_view=masks,
            root_from_cameras=cameras,
            stereo_consistency_fractions=np.ones(2, dtype=np.float64),
            parameters=parameters,
        )

    assert result is not None
    assert len(result[1].candidate_metrics) == 3
    assert result[1].candidate_metrics[-1]["source"] == "left_eef_carry_forward"


def test_causal_current_frame_candidate_revalidates_registered_prediction() -> None:
    """A matching dense proposal needs current-frame evidence before reuse."""

    class Selector:
        depth_consistency_m = 0.08
        auxiliary_view_score_weight = 0.5
        auxiliary_primary_support_saturation_fraction = 0.05

        @staticmethod
        def render_root_poses_depths(poses, root_from_cameras):
            assert len(poses) == 1
            return {
                name: np.ones((len(poses), 2, 2), dtype=np.float32)
                for name in root_from_cameras
            }

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "left_eef_carry_forward",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.6,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    pose = np.eye(4)
    eef = np.repeat(np.eye(4)[None], 2, axis=0)
    anchors = (
        TemporalAnchor(
            ordinal=0,
            source_frame_index=0,
            candidate_poses_root=pose[None],
            candidate_metrics=(metric,),
            eef_poses_root=eef,
            hand_observed_position=np.array([2.0, 4.5]),
        ),
        TemporalAnchor(
            ordinal=1,
            source_frame_index=3,
            candidate_poses_root=pose[None],
            candidate_metrics=(metric,),
            eef_poses_root=eef,
            hand_observed_position=np.array([2.0, 4.5]),
        ),
    )
    error = TemporalSelectionError(
        "missing candidate",
        {
            "anchor_index": 1,
            "causal_prefix": {
                "selected_poses_root": [pose.tolist()],
                "attached_hands": [True, False],
            },
        },
    )
    masks = {
        name: {1: np.ones((2, 2), dtype=bool)}
        for name in ("head_left", "left_wrist", "right_wrist")
    }
    cameras = {
        name: np.repeat(np.eye(4)[None], 2, axis=0) for name in masks
    }

    result = _append_causal_current_frame_candidate(
        anchors,
        error,
        registration_selector=Selector(),
        frames=[
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
        ],
        dense_masks_by_view=masks,
        root_from_cameras=cameras,
        stereo_consistency_fractions=np.ones(2, dtype=np.float64),
        parameters=parameters,
    )

    assert result is not None
    assert len(result[1].candidate_metrics) == 2
    assert result[1].candidate_metrics[-1]["source"] == "left_eef_carry_forward"
    assert (
        result[1].candidate_metrics[-1]["bidirectional_consensus"]["validation_mode"]
        == "causal_current_frame_rgbd_propagation"
    )


def test_causal_current_frame_candidate_skips_cached_rejection() -> None:
    """A retry must not render the same rejected FK pose again."""

    class Selector:
        depth_consistency_m = 0.08
        auxiliary_view_score_weight = 0.5
        auxiliary_primary_support_saturation_fraction = 0.05

        @staticmethod
        def render_root_poses_depths(poses, root_from_cameras):
            raise AssertionError("cached rejection must not reach the renderer")

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "global_registration",
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.6,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    pose = np.eye(4)
    eef = np.repeat(pose[None], 2, axis=0)
    anchors = (
        TemporalAnchor(
            ordinal=0,
            source_frame_index=0,
            candidate_poses_root=pose[None],
            candidate_metrics=(metric,),
            eef_poses_root=eef,
            hand_observed_position=np.array([4.5, 2.0]),
        ),
        TemporalAnchor(
            ordinal=1,
            source_frame_index=3,
            candidate_poses_root=pose[None],
            candidate_metrics=(metric,),
            eef_poses_root=eef,
            hand_observed_position=np.array([4.5, 2.0]),
        ),
    )
    error = TemporalSelectionError(
        "missing candidate",
        {
            "anchor_index": 1,
            "causal_prefix": {
                "selected_poses_root": [pose.tolist()],
                "attached_hands": [False, True],
            },
        },
    )
    masks = {
        name: {1: np.ones((2, 2), dtype=bool)}
        for name in ("head_left", "left_wrist", "right_wrist")
    }
    cameras = {name: np.repeat(pose[None], 2, axis=0) for name in masks}
    result = _append_causal_current_frame_candidate(
        anchors,
        error,
        registration_selector=Selector(),
        frames=[
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
            (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.float32)),
        ],
        dense_masks_by_view=masks,
        root_from_cameras=cameras,
        stereo_consistency_fractions=np.ones(2, dtype=np.float64),
        parameters=parameters,
        rejected_candidate_keys={(1, "right_eef_carry_forward", pose.tobytes())},
    )

    assert result is None


def test_temporal_selection_requires_strict_static_endpoint_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    strict = {
        "mask_precision": 0.3,
        "mask_explained_fraction": 0.65,
        "depth_overlap_fraction": 0.9,
        "depth_consistent_union_fraction": 0.1,
        "median_absolute_depth_error_m": 0.02,
        "bidirectional_consensus": {"passes_gate": True},
    }
    motion_only = {
        "mask_precision": 0.9,
        "mask_explained_fraction": 0.5,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.8,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    hand_states = (
        (4.5, 4.5),
        (4.5, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = tuple(
        TemporalAnchor(
            ordinal=index,
            source_frame_index=index,
            candidate_poses_root=np.stack((np.eye(4), np.eye(4))),
            candidate_metrics=(strict, motion_only),
            eef_poses_root=np.stack((np.eye(4), np.eye(4))),
            hand_observed_position=np.asarray(hand_state),
        )
        for index, hand_state in enumerate(hand_states)
    )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    assert result.selected_candidate_indices[0] == 0
    assert result.selected_candidate_indices[-1] == 0
    assert result.selected_evidence_modes[0] == "static_rgbd_bidirectional"
    assert result.selected_evidence_modes[-1] == "static_rgbd_bidirectional"


def test_temporal_selection_enforces_flip_endpoint_gravity_orientation() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
        minimum_endpoint_normal_vertical_component=0.7,
    )
    strict = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    preferred = {**strict, "median_absolute_depth_error_m": 0.005}
    angles = (np.pi, np.pi, 2.4, 1.5, 0.5, 0.0, 0.0)
    hands = (
        (4.5, 4.5),
        (2.0, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, (angle, hand) in enumerate(zip(angles, hands, strict=True)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("y", angle).as_matrix()
        distractor = pose.copy()
        if index == 0:
            distractor[:3, :3] = np.eye(3)
        elif index >= 5:
            distractor[:3, :3] = Rotation.from_euler("y", np.pi / 2).as_matrix()
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=30 * index,
                candidate_poses_root=np.stack((pose, distractor)),
                candidate_metrics=(strict, preferred),
                eef_poses_root=np.stack((pose, pose)),
                hand_observed_position=np.asarray(hand),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    assert result.selected_poses_root[0, 2, 2] <= -0.7
    assert result.selected_poses_root[-1, 2, 2] >= 0.7
    assert result.phase_indices[-1] == 4


def test_temporal_selection_supports_release_and_regrasp_cycles() -> None:
    assert 4 in _ALLOWED_PREVIOUS_STATES[1]
    assert 4 in _ALLOWED_PREVIOUS_STATES[2]
    assert 1 not in _ALLOWED_PREVIOUS_STATES[4]
    assert _ALLOWED_PREVIOUS_STATES[5] == (4,)


def test_temporal_selection_initializes_bimanual_attachment_from_continuing_hand() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    hand_states = (
        (4.5, 4.5),
        (2.0, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (4.5, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    table_translations = (
        0.0,
        0.0,
        0.15,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
        0.30,
    )
    anchors = []
    for index, (hand_state, table_translation) in enumerate(
        zip(hand_states, table_translations, strict=True)
    ):
        table = np.eye(4)
        table[0, 3] = table_translation
        eef = np.stack((table.copy(), table.copy()))
        eef[0, 1, 3] = 0.18
        eef[1, 1, 3] = -0.18
        if index in (1, 5):
            eef[1 if index == 1 else 0, 0, 3] = 0.5
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 15,
                candidate_poses_root=table[None],
                candidate_metrics=(metric,),
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    np.testing.assert_array_equal(result.phase_indices[5:8], (1, 2, 2))
    assert result.phase_indices[-1] == 4


def test_temporal_selection_reinitializes_attachment_when_second_hand_closes() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    hand_states = (
        (4.5, 4.5),
        (4.5, 4.5),
        (2.0, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, hand_state in enumerate(hand_states):
        table = np.eye(4)
        table[0, 3] = 0.15 * max(0, min(index - 2, 2))
        eef = np.stack((table.copy(), table.copy()))
        eef[0, 1, 3] = 0.18
        eef[1, 1, 3] = -0.18
        if index == 2:
            # The first hand slides before the second hand establishes the new
            # bimanual grasp. Its prior table-relative transform must not be
            # imposed on the visually verified pose at the next anchor.
            eef[0, 0, 3] -= 0.25
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 15,
                candidate_poses_root=table[None],
                candidate_metrics=(metric,),
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    np.testing.assert_array_equal(result.phase_indices[2:5], (1, 2, 2))
    assert result.phase_indices[-1] == 4


def test_temporal_selection_treats_fast_opening_hand_as_release() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    hand_states = (
        (4.5, 4.5),
        (4.5, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 3.0),
        (4.5, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, hand_state in enumerate(hand_states):
        table = np.eye(4)
        table[0, 3] = 0.15 * max(0, min(index - 1, 2))
        eef = np.stack((table.copy(), table.copy()))
        eef[0, 1, 3] = 0.18
        eef[1, 1, 3] = -0.18
        if index == 4:
            eef[0, 0, 3] -= 0.25
            eef[1, 0, 3] += 0.25
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index * 15,
                candidate_poses_root=table[None],
                candidate_metrics=(metric,),
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    assert result.phase_indices[4] == 3
    assert result.phase_indices[-1] == 4


def test_static_pose_uses_raw_silhouette_when_stereo_marks_false_occlusion() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "raw_mask_precision": 0.75,
        "raw_mask_explained_fraction": 0.95,
        "mask_precision": 0.70,
        "mask_explained_fraction": 0.50,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.2,
        "median_absolute_depth_error_m": 0.015,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.full(2, 4.5),
    )

    costs = temporal_static_visual_costs(anchor, parameters)

    assert np.isfinite(costs[0])


def test_static_pose_accepts_visible_cad_when_depth_occludes_the_raw_render() -> None:
    """A hand occlusion must not invalidate an otherwise verified final pose."""

    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "raw_mask_precision": 0.10,
        "raw_mask_explained_fraction": 0.95,
        "mask_precision": 0.30,
        "mask_explained_fraction": 0.90,
        "raw_rendered_pixels": 1000,
        "rendered_pixels": 400,
        "occluded_rendered_pixels": 600,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.015,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.full(2, 4.5),
    )

    costs = temporal_static_visual_costs(anchor, parameters)

    assert np.isfinite(costs[0])


def test_low_confidence_static_silhouette_requires_two_views_and_consensus() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "raw_mask_precision": 0.1,
        "raw_mask_explained_fraction": 0.9,
        "mask_precision": 0.1,
        "mask_explained_fraction": 0.5,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.02,
        "median_absolute_depth_error_m": 0.02,
        "auxiliary_mask_precisions": {
            "left_wrist": 0.1,
            "right_wrist": 0.1,
        },
        "auxiliary_mask_explained_fractions": {
            "left_wrist": 0.9,
            "right_wrist": 0.9,
        },
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.full(2, 4.5),
        primary_stereo_consistent_fraction=0.1,
    )

    costs = temporal_visual_costs(
        anchor, parameters, require_bidirectional_consensus=True
    )
    assert np.isfinite(costs[0])
    with pytest.raises(ValueError, match="no static RGB-D candidates"):
        temporal_static_visual_costs(anchor, parameters)

    without_consensus = TemporalAnchor(
        **{
            **anchor.__dict__,
            "candidate_metrics": (
                {**metric, "bidirectional_consensus": {"passes_gate": False}},
            ),
        }
    )
    with pytest.raises(ValueError, match="no geometrically eligible"):
        temporal_visual_costs(
            without_consensus,
            parameters,
            require_bidirectional_consensus=True,
        )


def test_expanded_anchor_prunes_only_visually_ineligible_candidates() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = (0.0, 0.1, 0.2)
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=poses,
        candidate_metrics=(
            metric,
            {**metric, "bidirectional_consensus": {"passes_gate": False}},
            metric,
        ),
        eef_poses_root=np.repeat(np.eye(4)[None], 2, axis=0),
        hand_observed_position=np.full(2, 4.5),
    )

    expanded, visual, _, candidate_indices, _ = _expanded_anchor(
        anchor, parameters
    )

    assert len(expanded) == 4
    assert np.isfinite(visual).all()
    np.testing.assert_array_equal(candidate_indices, (0, 0, 2, 2))


def test_unsegmented_bridge_requires_head_rgbd_two_wrists_and_consensus() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "bidirectional_tracking_bridge",
        "evidence_kind": "unsegmented_multiview_rgbd",
        "raw_mask_precision": 0.0,
        "raw_mask_explained_fraction": 0.0,
        "mask_precision": 0.0,
        "mask_explained_fraction": 0.0,
        "depth_overlap_fraction": 0.9,
        "depth_consistent_union_fraction": 0.5,
        "multiview_score": 0.8,
        "median_absolute_depth_error_m": 0.02,
        "primary_stereo_consistent_fraction": 0.8,
        "auxiliary_mask_precisions": {
            "left_wrist": 0.2,
            "right_wrist": 0.2,
        },
        "auxiliary_mask_explained_fractions": {
            "left_wrist": 0.8,
            "right_wrist": 0.8,
        },
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=10,
        source_frame_index=30,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.asarray((2.0, 2.0)),
        primary_stereo_consistent_fraction=0.0,
    )

    evidence = _candidate_visual_evidence(
        anchor,
        parameters,
        metric,
        require_bidirectional_consensus=True,
    )
    assert evidence is not None
    assert evidence[1] == "unsegmented_multiview_rgbd_bidirectional"
    with pytest.raises(ValueError, match="no static RGB-D candidates"):
        temporal_static_visual_costs(anchor, parameters)

    no_consensus = {
        **metric,
        "bidirectional_consensus": {"passes_gate": False},
    }
    assert (
        _candidate_visual_evidence(
            anchor,
            parameters,
            no_consensus,
            require_bidirectional_consensus=True,
        )
        is None
    )
    missing_wrist = {
        **metric,
        "auxiliary_mask_explained_fractions": {"left_wrist": 0.8},
    }
    assert (
        _candidate_visual_evidence(
            anchor,
            parameters,
            missing_wrist,
            require_bidirectional_consensus=True,
        )
        is None
    )


def test_null_candidate_stereo_fraction_uses_anchor_measurement() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "primary_stereo_consistent_fraction": None,
        "bidirectional_consensus": {"passes_gate": True},
    }
    anchor = TemporalAnchor(
        ordinal=0,
        source_frame_index=0,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(metric,),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.full(2, 4.5),
        primary_stereo_consistent_fraction=0.9,
    )

    assert np.isfinite(temporal_visual_costs(anchor, parameters)[0])


def test_temporal_gap_audit_requires_dense_or_physical_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )

    def anchors_and_selection(phases, second_pose, second_eef):
        first_eef = np.stack((np.eye(4), np.eye(4)))
        anchors = (
            TemporalAnchor(
                ordinal=0,
                source_frame_index=0,
                candidate_poses_root=np.eye(4)[None],
                candidate_metrics=({},),
                eef_poses_root=first_eef,
                hand_observed_position=np.full(2, 4.5),
            ),
            TemporalAnchor(
                ordinal=4,
                source_frame_index=60,
                candidate_poses_root=second_pose[None],
                candidate_metrics=({},),
                eef_poses_root=second_eef,
                hand_observed_position=np.full(2, 4.5),
            ),
        )
        selection = TemporalSelectionResult(
            selected_candidate_indices=np.zeros(2, dtype=np.int64),
            selected_symmetry_indices=np.zeros(2, dtype=np.int64),
            phase_indices=np.asarray(phases, dtype=np.int64),
            selected_poses_root=np.stack((np.eye(4), second_pose)),
            transition_translation_m=np.zeros(1),
            transition_rotation_rad=np.zeros(1),
            selected_evidence_modes=("rgbd_bidirectional",) * 2,
            total_cost=0.0,
        )
        return anchors, selection

    static_anchors, static_selection = anchors_and_selection(
        (4, 4), np.eye(4), np.stack((np.eye(4), np.eye(4)))
    )
    static_audit = audit_temporal_evidence_gaps(
        static_anchors,
        static_selection,
        sampled_hand_observed_positions=np.full((5, 2), 4.5),
        maximum_dense_gap_source_frames=30,
        source_fps=30.0,
        parameters=parameters,
    )
    assert static_audit[0]["mode"] == "open_hand_static_endpoint_bridge"

    moved_pose = np.eye(4)
    moved_pose[0, 3] = 0.1
    settle_anchors, settle_selection = anchors_and_selection(
        (3, 4), moved_pose, np.stack((np.eye(4), np.eye(4)))
    )
    settle_audit = audit_temporal_evidence_gaps(
        settle_anchors,
        settle_selection,
        sampled_hand_observed_positions=np.full((5, 2), 4.5),
        maximum_dense_gap_source_frames=10,
        source_fps=30.0,
        parameters=parameters,
    )
    assert (
        settle_audit[0]["mode"]
        == "open_hand_settle_to_verified_final_static"
    )

    moved_eef = np.stack((moved_pose.copy(), np.eye(4)))
    held_anchors, held_selection = anchors_and_selection(
        (3, 3), moved_pose, moved_eef
    )
    held_audit = audit_temporal_evidence_gaps(
        held_anchors,
        held_selection,
        sampled_hand_observed_positions=np.tile((2.0, 4.5), (5, 1)),
        maximum_dense_gap_source_frames=30,
        source_fps=30.0,
        parameters=parameters,
    )
    assert held_audit[0]["mode"] == "continuous_hand_rigid_constraint_bridge"

    released_anchors, released_selection = anchors_and_selection(
        (3, 3), moved_pose, np.stack((np.eye(4), np.eye(4)))
    )
    released_audit = audit_temporal_evidence_gaps(
        released_anchors,
        released_selection,
        sampled_hand_observed_positions=np.asarray(
            ((2.0, 2.2), (2.6, 2.8), (3.2, 3.4), (3.8, 4.0), (4.5, 4.5))
        ),
        maximum_dense_gap_source_frames=30,
        source_fps=30.0,
        parameters=parameters,
    )
    assert (
        released_audit[0]["mode"]
        == "release_regrasp_visual_endpoint_bridge"
    )

    release_with_one_closing_peak = audit_temporal_evidence_gaps(
        released_anchors,
        released_selection,
        sampled_hand_observed_positions=np.asarray(
            ((2.6, 3.8), (2.2, 4.0), (2.4, 4.2), (3.8, 4.4), (4.5, 4.5))
        ),
        maximum_dense_gap_source_frames=30,
        source_fps=30.0,
        parameters=parameters,
    )
    assert release_with_one_closing_peak[0]["release_to_open_without_regrasp"]

    with pytest.raises(TemporalSelectionError) as exc_info:
        audit_temporal_evidence_gaps(
            held_anchors,
            held_selection,
            sampled_hand_observed_positions=np.asarray(
                ((2.0, 4.5), (4.5, 4.5), (2.0, 4.5), (4.5, 4.5), (2.0, 4.5))
            ),
            maximum_dense_gap_source_frames=30,
            source_fps=30.0,
            parameters=parameters,
        )
    assert (
        exc_info.value.diagnostics["rejection_reason"]
        == "temporal_evidence_gap_unconstrained"
    )


def test_eef_carry_requires_matching_active_hand_and_wrist_support() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "source": "right_eef_carry_forward",
        "mask_precision": 0.04,
        "mask_explained_fraction": 0.1,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.12,
        "multiview_score": 0.16,
        "median_absolute_depth_error_m": 0.015,
        "bidirectional_consensus": {"passes_gate": True},
        "auxiliary_mask_explained_fractions": {
            "left_wrist": 0.8,
            "right_wrist": 0.9,
        },
    }

    def anchor(hand_state, *, source="right_eef_carry_forward", sharpness=80.0):
        value = {**metric, "source": source}
        return TemporalAnchor(
            ordinal=10,
            source_frame_index=30,
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=(value,),
            eef_poses_root=np.stack((np.eye(4), np.eye(4))),
            hand_observed_position=np.asarray(hand_state, dtype=np.float64),
            primary_laplacian_variance=sharpness,
        )

    assert np.isfinite(temporal_visual_costs(anchor((4.5, 2.0)), parameters)).all()
    with pytest.raises(ValueError, match="no geometrically eligible"):
        temporal_visual_costs(anchor((4.5, 4.5)), parameters)
    with pytest.raises(ValueError, match="no geometrically eligible"):
        temporal_visual_costs(anchor((4.5, 2.0), source="global_registration"), parameters)
    assert np.isfinite(
        temporal_visual_costs(anchor((4.5, 2.0), sharpness=200.0), parameters)
    ).all()
    missing_right_support = {
        **metric,
        "mask_explained_fraction": 0.03,
        "auxiliary_mask_explained_fractions": {
            "left_wrist": 0.9,
            "right_wrist": 0.0,
        },
    }
    with pytest.raises(ValueError, match="no geometrically eligible"):
        temporal_visual_costs(
            TemporalAnchor(
                ordinal=10,
                source_frame_index=30,
                candidate_poses_root=np.eye(4)[None],
                candidate_metrics=(missing_right_support,),
                eef_poses_root=np.stack((np.eye(4), np.eye(4))),
                hand_observed_position=np.asarray((4.5, 2.0)),
                primary_laplacian_variance=80.0,
            ),
            parameters,
        )


def test_causal_selector_keeps_rendered_eef_carry_without_bidirectional_vote() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    anchor = TemporalAnchor(
        ordinal=1,
        source_frame_index=3,
        candidate_poses_root=np.eye(4)[None],
        candidate_metrics=(
            {
                "source": "right_eef_carry_forward",
                "mask_precision": 0.08,
                "mask_explained_fraction": 0.10,
                "depth_overlap_fraction": 1.0,
                "depth_consistent_union_fraction": 0.15,
                "multiview_score": 0.20,
                "median_absolute_depth_error_m": 0.01,
                "auxiliary_mask_explained_fractions": {"right_wrist": 0.9},
                "bidirectional_consensus": {
                    "passes_gate": False,
                    "validation_mode": "causal_current_frame_rgbd_propagation",
                },
            },
        ),
        eef_poses_root=np.stack((np.eye(4), np.eye(4))),
        hand_observed_position=np.asarray((4.5, 2.0)),
    )

    _, visual, _, static_bridge, candidate_indices, _ = _causal_expanded_anchor(
        anchor, parameters
    )

    np.testing.assert_array_equal(candidate_indices, (0, 0))
    assert np.isfinite(visual).all()
    assert not static_bridge.any()

    arbitrary_forward = TemporalAnchor(
        ordinal=anchor.ordinal,
        source_frame_index=anchor.source_frame_index,
        candidate_poses_root=anchor.candidate_poses_root,
        candidate_metrics=(
            {
                **anchor.candidate_metrics[0],
                "bidirectional_consensus": {"passes_gate": False},
            },
        ),
        eef_poses_root=anchor.eef_poses_root,
        hand_observed_position=anchor.hand_observed_position,
    )
    with pytest.raises(ValueError, match="no geometrically eligible candidates"):
        _causal_expanded_anchor(arbitrary_forward, parameters)

    initial_metric = {
        "source": "source_cad_seed",
        "mask_precision": 0.9,
        "mask_explained_fraction": 0.9,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.9,
        "multiview_score": 0.9,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    trajectory = (
        TemporalAnchor(
            ordinal=0,
            source_frame_index=0,
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=(initial_metric,),
            eef_poses_root=np.stack((np.eye(4), np.eye(4))),
            hand_observed_position=np.asarray((4.5, 4.5)),
        ),
        *(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=3 * index,
                candidate_poses_root=np.eye(4)[None],
                candidate_metrics=(anchor.candidate_metrics[0],),
                eef_poses_root=np.stack((np.eye(4), np.eye(4))),
                hand_observed_position=np.asarray((4.5, 2.0)),
            )
            for index in (1, 2, 3)
        ),
    )
    result = select_causally_constrained_poses(
        trajectory, source_fps=30.0, parameters=parameters
    )
    assert result.selected_evidence_modes[1:] == (
        "causal_attachment_rgbd",
        "causal_eef_rigid_rgbd",
        "causal_eef_rigid_rgbd",
    )


def test_temporal_selection_allows_multi_anchor_grasp_transition() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    secondary_metrics = ({
        "mask_precision": 0.3,
        "mask_explained_fraction": 0.1,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.02,
        "median_absolute_depth_error_m": 0.1,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    translations = (
        0.0,
        0.0,
        0.0,
        0.18,
        0.36,
        0.54,
        0.72,
        0.72,
        0.72,
        0.72,
    )
    hand_states = (
        (4.5, 4.5),
        (4.5, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, (translation, hand_state) in enumerate(
        zip(translations, hand_states, strict=True)
    ):
        table = np.eye(4)
        table[0, 3] = translation
        eef = np.stack((table.copy(), table.copy()))
        eef[0, 1, 3] += 0.18
        eef[1, 1, 3] -= 0.18
        anchors.append(
            TemporalAnchor(
                ordinal=index * 10,
                source_frame_index=index * 30,
                candidate_poses_root=(
                    table[None] if index % 2 else np.stack((table, table))
                ),
                candidate_metrics=(metrics if index % 2 else metrics + secondary_metrics),
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    np.testing.assert_array_equal(
        result.phase_indices, (0, 0, 0, 1, 1, 2, 2, 3, 4, 4)
    )
    np.testing.assert_array_equal(result.selected_candidate_indices, 0)


def test_temporal_selection_rejects_table_motion_during_open_hand_approach() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    anchors = []
    for index, translation in enumerate((0.0, 0.18, 0.18, 0.18)):
        table = np.eye(4)
        table[0, 3] = translation
        anchors.append(
            TemporalAnchor(
                ordinal=index * 10,
                source_frame_index=index * 30,
                candidate_poses_root=table[None],
                candidate_metrics=metrics,
                eef_poses_root=np.stack((np.eye(4), np.eye(4))),
                hand_observed_position=np.asarray((4.5, 4.5)),
            )
        )

    with pytest.raises(TemporalSelectionError, match="no physically coherent"):
        select_temporally_consistent_poses(
            anchors, source_fps=30.0, parameters=parameters
        )


def test_temporal_selection_requires_two_static_final_anchors() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    hand_states = (
        (4.5, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, hand_state in enumerate(hand_states):
        table = np.eye(4)
        if index == len(hand_states) - 1:
            table[0, 3] = 0.18
        eef = np.stack((table.copy(), table.copy()))
        anchors.append(
            TemporalAnchor(
                ordinal=index * 10,
                source_frame_index=index * 30,
                candidate_poses_root=table[None],
                candidate_metrics=metrics,
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    with pytest.raises(TemporalSelectionError, match="final-static"):
        select_temporally_consistent_poses(
            anchors, source_fps=30.0, parameters=parameters
        )


def test_temporal_selection_allows_release_regrasp_before_final_static() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    translations = (0.0, 0.0, 0.18, 0.36, 0.36, 0.41, 0.46, 0.46, 0.46)
    hand_states = (
        (4.5, 4.5),
        (4.5, 4.5),
        (2.0, 2.0),
        (2.0, 2.0),
        (4.5, 4.5),
        (2.0, 4.5),
        (2.0, 4.5),
        (4.5, 4.5),
        (4.5, 4.5),
    )
    anchors = []
    for index, (translation, hand_state) in enumerate(
        zip(translations, hand_states, strict=True)
    ):
        table = np.eye(4)
        table[0, 3] = translation
        eef = np.stack((table.copy(), table.copy()))
        eef[0, 1, 3] += 0.18
        eef[1, 1, 3] -= 0.18
        anchors.append(
            TemporalAnchor(
                ordinal=index * 10,
                source_frame_index=index * 30,
                candidate_poses_root=table[None],
                candidate_metrics=metrics,
                eef_poses_root=eef,
                hand_observed_position=np.asarray(hand_state),
            )
        )

    result = select_temporally_consistent_poses(
        anchors, source_fps=30.0, parameters=parameters
    )

    np.testing.assert_array_equal(
        result.phase_indices, (0, 1, 2, 2, 3, 3, 3, 4, 4)
    )


def test_temporal_selection_reports_physical_failure_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=0.1,
        maximum_table_angular_speed_rad_s=0.1,
        maximum_grasp_relative_translation_step_m=0.01,
        maximum_grasp_relative_rotation_step_rad=0.1,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.8,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.5,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    anchors = []
    for index in range(4):
        pose = np.eye(4)
        pose[0, 3] = float(index)
        anchors.append(
            TemporalAnchor(
                ordinal=index,
                source_frame_index=index,
                candidate_poses_root=pose[None],
                candidate_metrics=metrics,
                eef_poses_root=np.stack((np.eye(4), np.eye(4))),
                hand_observed_position=np.zeros(2),
            )
        )
    with pytest.raises(TemporalSelectionError) as caught:
        select_temporally_consistent_poses(
            anchors, source_fps=30.0, parameters=parameters
        )
    diagnostics = caught.value.diagnostics
    assert diagnostics["anchor_index"] == 1
    assert diagnostics["physically_possible_pairs"] == 0
    assert diagnostics["reachable_transition_pairs_by_internal_state"] == {
        str(index): 0 for index in range(7)
    }


def test_temporal_selection_reports_ineligible_anchor_evidence() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.03,
        maximum_candidate_depth_error_m=0.45,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metrics = ({
        "mask_precision": 0.1,
        "mask_explained_fraction": 0.02,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.01,
        "median_absolute_depth_error_m": 0.5,
        "bidirectional_consensus": {"passes_gate": True},
    },)
    anchors = [
        TemporalAnchor(
            ordinal=index,
            source_frame_index=index,
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=metrics,
            eef_poses_root=np.stack((np.eye(4), np.eye(4))),
            hand_observed_position=np.zeros(2),
        )
        for index in range(4)
    ]
    with pytest.raises(TemporalSelectionError) as caught:
        select_temporally_consistent_poses(
            anchors, source_fps=30.0, parameters=parameters
        )
    assert caught.value.diagnostics["anchor_index"] == 0
    assert caught.value.diagnostics["candidate_metrics"]["candidate_count"] == 1
    assert caught.value.diagnostics["candidate_metrics"]["mask_precision"] == {
        "minimum": 0.1,
        "maximum": 0.1,
    }
def test_table_symmetry_continuity_and_pose_errors() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    poses[2, :3, 3] = [0.1, 0.0, 0.0]
    continuous, selected = make_pose_continuous(poses)
    assert selected == (0, 1, 0)
    np.testing.assert_allclose(continuous[0, :3, :3], continuous[1, :3, :3], atol=1.0e-7)
    translation, rotation, symmetry = pose_errors(poses[0], poses[1])
    assert translation == 0.0
    assert rotation < 1.0e-7
    assert symmetry == 1


def test_rigid_projection_repairs_only_small_positive_drift() -> None:
    pose = np.eye(4)
    pose[0, 0] += 2.0e-6
    projected, correction = project_to_rigid_transform(pose, "drifted pose")
    np.testing.assert_allclose(projected[:3, :3].T @ projected[:3, :3], np.eye(3))
    assert 0.0 < correction < 1.0e-4

    reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
    with np.testing.assert_raises_regex(ValueError, "orientation-preserving"):
        project_to_rigid_transform(reflection, "reflection")

    large_drift = np.eye(4)
    large_drift[0, 0] = 0.9
    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        project_to_rigid_transform(large_drift, "large drift")


def test_pose_interpolation_covers_every_source_frame() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, 3] = [0.4, 0.0, 0.0]
    poses[2, :3, 3] = [0.8, 0.0, 0.0]
    poses[2, :3, :3] = Rotation.from_euler("y", np.pi / 2).as_matrix()
    output = interpolate_pose_trajectory(np.array([0, 2, 4]), poses, 5)
    assert output.shape == (5, 4, 4)
    np.testing.assert_allclose(output[:, 0, 3], np.linspace(0.0, 0.8, 5))
    np.testing.assert_allclose(output[-1], poses[-1], atol=1.0e-7)


def test_bidirectional_fusion_resolves_table_symmetry() -> None:
    forward = np.repeat(np.eye(4)[None], 3, axis=0)
    forward[:, 0, 3] = [0.0, 0.1, 0.2]
    backward = forward.copy()
    backward[:, :3, :3] = (
        backward[:, :3, :3] @ Rotation.from_euler("z", np.pi).as_matrix()
    )
    backward[:, 0, 3] += 0.02
    fused, translation, rotation, symmetry = fuse_bidirectional_poses(forward, backward)
    np.testing.assert_allclose(fused[:, 0, 3], [0.01, 0.11, 0.21], atol=1.0e-8)
    np.testing.assert_allclose(translation, 0.02, atol=1.0e-8)
    np.testing.assert_allclose(rotation, 0.0, atol=1.0e-7)
    np.testing.assert_array_equal(symmetry, 1)


def test_rendered_alignment_metrics_and_gate() -> None:
    observed = np.zeros((4, 5), dtype=np.float32)
    rendered = np.zeros_like(observed)
    mask = np.zeros_like(observed, dtype=bool)
    observed[1:3, 1:4] = 0.60
    rendered[1:3, 1:4] = 0.61
    mask[1:3, 1:4] = True
    metrics = evaluate_rendered_alignment(
        observed_depth_m=observed,
        rendered_depth_m=rendered,
        observed_mask=mask,
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.6,
    )
    assert metrics.passes_gate
    assert metrics.depth_overlap_fraction == 1.0
    assert metrics.rendered_mask_explained_fraction == 1.0
    assert abs(metrics.median_absolute_depth_error_m - 0.01) < 1.0e-6
    assert metrics.to_json()["rejection_reasons"] == []


def test_rendered_alignment_rejects_empty_render() -> None:
    depth = np.ones((3, 3), dtype=np.float32)
    metrics = evaluate_rendered_alignment(
        observed_depth_m=depth,
        rendered_depth_m=np.zeros_like(depth),
        observed_mask=np.ones_like(depth, dtype=bool),
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.6,
    )
    assert not metrics.passes_gate
    assert metrics.median_absolute_depth_error_m is None
    assert "empty_render" in metrics.rejection_reasons


def test_rendered_alignment_ignores_background_depth_outside_object_mask() -> None:
    observed = np.full((4, 5), 0.2, dtype=np.float32)
    rendered = np.full((4, 5), 1.5, dtype=np.float32)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    observed[mask] = 0.60
    rendered[mask] = 0.61
    metrics = evaluate_rendered_alignment(
        observed_depth_m=observed,
        rendered_depth_m=rendered,
        observed_mask=mask,
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.25,
    )
    assert metrics.passes_gate
    assert abs(metrics.median_absolute_depth_error_m - 0.01) < 1.0e-6


def test_rendered_alignment_excludes_pixels_hidden_by_foreground_geometry() -> None:
    observed = np.full((4, 4), 0.3, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    observed[mask] = 0.60
    rendered = np.full((4, 4), 0.61, dtype=np.float32)
    metrics = evaluate_rendered_alignment(
        observed_depth_m=observed,
        rendered_depth_m=rendered,
        observed_mask=mask,
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.6,
    )
    assert metrics.passes_gate
    assert metrics.raw_rendered_pixels == 16
    assert metrics.occluded_rendered_pixels == 12
    assert metrics.rendered_pixels == 4
    assert metrics.rendered_mask_explained_fraction == 1.0


def test_rendered_alignment_separates_raw_silhouette_from_stereo_occlusion() -> None:
    observed = np.zeros((3, 4), dtype=np.float32)
    rendered = np.zeros_like(observed)
    mask = np.zeros_like(observed, dtype=bool)
    mask[1, :] = True
    rendered[mask] = 0.61
    observed[mask] = 0.60
    observed[1, 2:] = 0.30

    metrics = evaluate_rendered_alignment(
        observed_depth_m=observed,
        rendered_depth_m=rendered,
        observed_mask=mask,
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.8,
    )

    assert metrics.passes_gate
    assert metrics.raw_rendered_mask_precision == 1.0
    assert metrics.raw_rendered_mask_explained_fraction == 1.0
    assert metrics.rendered_mask_explained_fraction == 0.5


def test_rendered_alignment_does_not_mutate_visible_render_count_with_mask() -> None:
    observed = np.full((4, 4), 0.60, dtype=np.float32)
    rendered = np.full((4, 4), 0.61, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    metrics = evaluate_rendered_alignment(
        observed_depth_m=observed,
        rendered_depth_m=rendered,
        observed_mask=mask,
        maximum_occlusion_depth_error_m=0.025,
        maximum_median_absolute_depth_error_m=0.025,
        minimum_depth_overlap_fraction=0.25,
        minimum_rendered_mask_explained_fraction=0.6,
    )

    assert metrics.raw_rendered_pixels == 16
    assert metrics.occluded_rendered_pixels == 0
    assert metrics.rendered_pixels == 16


def test_registration_depth_ranking_prefers_masked_depth_consistency() -> None:
    observed = np.zeros((6, 8), dtype=np.float32)
    observed[1:5, 2:6] = 0.6
    mask = observed > 0.0
    candidates = np.zeros((3, 6, 8), dtype=np.float32)
    candidates[0, 1:5, 2:6] = 0.61
    candidates[1, 1:5, 2:6] = 0.8
    candidates[2, 1:5, 3:7] = 0.61
    metrics, selected = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=mask,
        sources=("global_registration", "global_registration", "continued_tracking"),
        maximum_consistent_depth_error_m=0.025,
    )
    assert selected == 0
    assert metrics[0].depth_consistent_pixels == 16
    assert metrics[1].depth_consistent_pixels == 0
    assert metrics[2].source == "continued_tracking"


def test_registration_depth_ranking_excludes_depth_occluded_cad_pixels() -> None:
    observed = np.full((4, 4), 0.3, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    observed[mask] = 0.60
    rendered = np.full((1, 4, 4), 0.61, dtype=np.float32)
    metrics, selected = rank_registration_depths(
        rendered_depths_m=rendered,
        observed_depth_m=observed,
        observed_mask=mask,
        sources=("global_registration",),
        maximum_consistent_depth_error_m=0.025,
    )
    assert selected == 0
    assert metrics[0].raw_rendered_pixels == 16
    assert metrics[0].occluded_rendered_pixels == 12
    assert metrics[0].rendered_pixels == 4
    assert metrics[0].mask_precision == 1.0
    assert metrics[0].mask_explained_fraction == 1.0


def test_registration_depth_ranking_uses_real_auxiliary_view_to_break_tie() -> None:
    observed = np.zeros((4, 4), dtype=np.float32)
    observed[1:3, 1:3] = 0.6
    primary_mask = observed > 0.0
    candidates = np.repeat(observed[None], 2, axis=0)
    auxiliary = np.zeros((2, 4, 4), dtype=np.float32)
    auxiliary[0, :2, :2] = 0.5
    auxiliary[1, 2:, 2:] = 0.5
    auxiliary_mask = np.zeros((4, 4), dtype=bool)
    auxiliary_mask[2:, 2:] = True
    metrics, selected = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=primary_mask,
        sources=("first", "second"),
        maximum_consistent_depth_error_m=0.025,
        auxiliary_rendered_depths_m={"left_wrist": auxiliary},
        auxiliary_observed_masks={"left_wrist": auxiliary_mask},
        auxiliary_view_score_weight=0.25,
    )
    assert selected == 1
    assert metrics[0].auxiliary_mask_precisions == {"left_wrist": 0.0}
    assert metrics[1].auxiliary_mask_precisions == {"left_wrist": 1.0}
    assert metrics[0].auxiliary_mask_explained_fractions == {"left_wrist": 0.0}
    assert metrics[1].auxiliary_mask_explained_fractions == {"left_wrist": 1.0}
    assert metrics[1].multiview_score > metrics[0].multiview_score

    beam = propagation_candidate_indices(
        metrics,
        ("global", "global"),
        selected_candidate_index=selected,
        limit=2,
    )
    assert beam == [1, 0]


def test_unsegmented_multiview_depths_preserve_head_and_wrist_evidence() -> None:
    observed = np.full((4, 4), 0.6, dtype=np.float32)
    rendered = np.zeros((4, 4), dtype=np.float32)
    rendered[1:3, 1:3] = 0.61
    consistency = np.zeros((4, 4), dtype=bool)
    consistency[1:3, 1:3] = True
    wrist_rendered = {
        "left_wrist": rendered.copy(),
        "right_wrist": rendered.copy(),
    }
    wrist_masks = {
        "left_wrist": rendered > 0.0,
        "right_wrist": rendered > 0.0,
    }

    metric = evaluate_unsegmented_multiview_depths(
        rendered_depth_m=rendered,
        observed_depth_m=observed,
        stereo_consistency_mask=consistency,
        auxiliary_rendered_depths_m=wrist_rendered,
        auxiliary_observed_masks=wrist_masks,
        source="bidirectional_tracking_bridge",
        maximum_consistent_depth_error_m=0.025,
        auxiliary_view_score_weight=0.25,
    )

    assert metric.evidence_kind == "unsegmented_multiview_rgbd"
    assert metric.observed_mask_pixels == 0
    assert metric.mask_precision == 0.0
    assert metric.depth_overlap_fraction == 1.0
    assert metric.median_absolute_depth_error_m == pytest.approx(0.01)
    assert metric.primary_stereo_consistent_fraction == 1.0
    assert metric.auxiliary_mask_precisions == {
        "left_wrist": 1.0,
        "right_wrist": 1.0,
    }


def test_propagation_beam_reserves_source_diversity_before_priority_fill() -> None:
    observed = np.ones((2, 2), dtype=np.float32)
    candidates = np.repeat(observed[None], 4, axis=0)
    metrics, _ = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=np.ones((2, 2), dtype=bool),
        sources=("global", "global", "left", "right"),
        maximum_consistent_depth_error_m=0.025,
    )

    beam = propagation_candidate_indices(
        metrics,
        ("global", "global", "left", "right"),
        selected_candidate_index=0,
        limit=3,
        priority_indices=(0, 1, 2, 3),
    )

    assert beam == [0, 2, 3]


def test_propagation_beam_keeps_required_physical_candidates_first() -> None:
    observed = np.ones((2, 2), dtype=np.float32)
    candidates = np.repeat(observed[None], 4, axis=0)
    metrics, selected = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=np.ones((2, 2), dtype=bool),
        sources=("global", "static", "left", "left"),
        maximum_consistent_depth_error_m=0.025,
    )

    beam = propagation_candidate_indices(
        metrics,
        ("global", "static", "left", "left"),
        selected_candidate_index=selected,
        limit=2,
        priority_indices=(0, 1, 2, 3),
        required_indices=(3, 2),
    )

    assert beam == [3, 2]


def test_contact_lineage_priority_uses_available_rigid_hand_model() -> None:
    sources = (
        "static_carry_forward",
        "static_carry_forward",
        "left_eef_carry_forward",
        "left_eef_carry_forward",
    )
    lineages = (0, 1, 0, 1)

    entering_bimanual = contact_lineage_priority_indices(
        sources,
        lineages,
        np.asarray((2.0, 3.4)),
        grasp_observed_position_max=3.5,
    )
    assert entering_bimanual == (2, 3)

    with_bimanual = contact_lineage_priority_indices(
        (*sources, "bimanual_eef_carry_forward", "bimanual_eef_carry_forward"),
        (*lineages, 0, 1),
        np.asarray((2.0, 2.0)),
        grasp_observed_position_max=3.5,
    )
    assert with_bimanual == (4, 5)


def test_dense_propagation_beam_preserves_each_physical_lineage() -> None:
    observed = np.ones((2, 2), dtype=np.float32)
    candidates = np.repeat(observed[None], 6, axis=0)
    candidates[0, 0, 0] = 0.0
    candidates[2, 0, 0] = 0.0
    candidates[4, 0, 0] = 0.0
    metrics, selected = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=np.ones((2, 2), dtype=bool),
        sources=("static", "static", "left", "left", "right", "right"),
        maximum_consistent_depth_error_m=0.025,
    )

    beam = lineage_preserving_candidate_indices(
        metrics,
        ("static", "static", "left", "left", "right", "right"),
        (0, 1, 0, 1, 0, 1),
        selected_candidate_index=selected,
        limit=2,
        priority_indices=(1, 3, 5, 0, 2, 4),
    )

    assert len(beam) == 2
    assert {beam_index % 2 for beam_index in beam} == {0, 1}


def test_auxiliary_view_cannot_override_absent_primary_depth_support() -> None:
    observed = np.zeros((4, 4), dtype=np.float32)
    observed[1:3, 1:3] = 0.6
    primary_mask = observed > 0.0
    candidates = np.zeros((2, 4, 4), dtype=np.float32)
    candidates[0, 1:3, 1:3] = 0.61
    candidates[1, :1, :1] = 0.61
    auxiliary = np.zeros((2, 4, 4), dtype=np.float32)
    auxiliary[1, 2:, 2:] = 0.5
    auxiliary_mask = np.zeros((4, 4), dtype=bool)
    auxiliary_mask[2:, 2:] = True

    metrics, selected = rank_registration_depths(
        rendered_depths_m=candidates,
        observed_depth_m=observed,
        observed_mask=primary_mask,
        sources=("primary", "auxiliary_only"),
        maximum_consistent_depth_error_m=0.025,
        auxiliary_rendered_depths_m={"left_wrist": auxiliary},
        auxiliary_observed_masks={"left_wrist": auxiliary_mask},
        auxiliary_view_score_weight=1.0,
        auxiliary_primary_support_saturation_fraction=0.05,
    )

    assert selected == 0
    assert metrics[1].depth_consistent_union_fraction == 0.0
    assert metrics[1].multiview_score == 0.0


def test_tracking_direction_registers_at_anchors_and_covers_frames() -> None:
    frames = []
    for index in range(4):
        rgb = np.full((2, 2, 3), index, dtype=np.uint8)
        frames.append((rgb, np.ones((2, 2), dtype=np.float32)))
    root_from_cameras = np.repeat(np.eye(4)[None], 4, axis=0)
    root_from_cameras[:, 1, 3] = 0.5
    masks = {0: np.ones((2, 2), dtype=bool), 2: np.ones((2, 2), dtype=bool)}
    poses, modes, corrections = _track_direction(
        estimator=_FakeFoundationPose(),
        frames=frames,
        intrinsic_matrix=np.eye(3),
        root_from_cameras=root_from_cameras,
        registration_masks=masks,
        order=range(4),
        registration_iterations=5,
        tracking_iterations=2,
    )
    np.testing.assert_allclose(poses[:, 0, 3], np.arange(4))
    np.testing.assert_allclose(poses[:, 1, 3], 0.5)
    assert modes == ("register", "track", "register", "track")
    np.testing.assert_allclose(corrections, 0.0)


def test_tracking_direction_evaluates_source_cad_seed_as_first_registration_candidate() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    selector = _SeedCapturingSelector()
    seed = np.eye(4)
    seed[:3, 3] = (0.4, -0.1, 0.2)

    _track_direction(
        estimator=_FakeFoundationPose(),
        frames=[(rgb, np.ones((2, 2), dtype=np.float32))],
        intrinsic_matrix=np.eye(3),
        root_from_cameras=np.eye(4)[None],
        registration_masks={0: np.ones((2, 2), dtype=bool)},
        order=range(1),
        registration_iterations=5,
        tracking_iterations=2,
        registration_selector=selector,
        initial_root_from_object=seed,
    )

    assert len(selector.calls) == 1
    candidates, sources = selector.calls[0]
    np.testing.assert_allclose(candidates, seed[None])
    assert sources == ("source_cad_seed",)


def test_terminal_confirmation_promotes_nearest_audited_dense_mask() -> None:
    masks = {
        0: np.zeros((2, 2), dtype=bool),
        4: np.ones((2, 2), dtype=bool),
    }
    dense = {
        **masks,
        1: np.ones((2, 2), dtype=bool),
        2: np.eye(2, dtype=bool),
        3: np.fliplr(np.eye(2, dtype=bool)),
    }

    registrations, promoted, confirmation = _with_terminal_confirmation_registration(
        masks,
        dense,
        np.asarray((0, 3, 6, 9, 12)),
        maximum_evidence_source_frame_gap=5,
        maximum_confirmation_source_frame_gap=5,
    )

    assert confirmation == 3
    assert promoted == (1, 2, 3)
    assert sorted(registrations) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(registrations[3], dense[3])


def test_post_release_filter_keeps_release_and_two_final_static_anchors() -> None:
    values = []
    for ordinal in range(6):
        anchor = TemporalAnchor(
            ordinal=ordinal,
            source_frame_index=ordinal * 3,
            candidate_poses_root=np.eye(4)[None],
            candidate_metrics=({},),
            eef_poses_root=np.stack((np.eye(4), np.eye(4))),
            hand_observed_position=np.asarray(
                (2.0, 4.5) if ordinal <= 1 else (4.5, 4.5)
            ),
        )
        values.append((anchor, ("test",)))

    retained, excluded, release = _retain_manipulation_and_final_static_anchors(
        values,
        last_active_hand_ordinal=1,
        final_static_ordinals=(4, 5),
        grasp_observed_position_max=3.5,
    )

    assert release == 2
    assert [value[0].ordinal for value in retained] == [0, 1, 2, 4, 5]
    assert [value["ordinal"] for value in excluded] == [3]


def test_terminal_static_selection_uses_recent_verified_anchors_not_endpoint() -> None:
    parameters = TemporalSelectionParameters(
        minimum_mask_precision=0.2,
        minimum_mask_explained_fraction=0.04,
        maximum_candidate_depth_error_m=0.08,
        static_translation_scale_m=0.05,
        static_rotation_scale_rad=0.25,
        maximum_static_translation_step_m=0.08,
        maximum_static_rotation_step_rad=0.35,
        grasp_translation_scale_m=0.05,
        grasp_rotation_scale_rad=0.4,
        maximum_table_speed_m_s=1.0,
        maximum_table_angular_speed_rad_s=6.0,
        maximum_grasp_relative_translation_step_m=0.12,
        maximum_grasp_relative_rotation_step_rad=1.0,
        grasp_observed_position_max=3.5,
    )
    metric = {
        "mask_precision": 0.8,
        "mask_explained_fraction": 0.9,
        "depth_overlap_fraction": 1.0,
        "depth_consistent_union_fraction": 0.8,
        "median_absolute_depth_error_m": 0.01,
        "bidirectional_consensus": {"passes_gate": True},
    }
    values = []
    for ordinal in range(6):
        values.append(
            (
                TemporalAnchor(
                    ordinal=ordinal,
                    source_frame_index=ordinal * 3,
                    candidate_poses_root=np.eye(4)[None],
                    candidate_metrics=(metric,) if ordinal in {3, 4} else ({},),
                    eef_poses_root=np.stack((np.eye(4), np.eye(4))),
                    hand_observed_position=np.asarray(
                        (2.0, 4.5) if ordinal <= 1 else (4.5, 4.5)
                    ),
                ),
                ("test",),
            )
        )

    selected = _select_terminal_static_anchor_ordinals(
        values,
        parameters,
        last_active_hand_ordinal=1,
        terminal_source_frame_index=15,
        maximum_terminal_gap_source_frames=7,
    )

    assert selected == (3, 4)


def test_temporal_solver_uses_only_the_propagated_pose_beam() -> None:
    evidence = {
        "temporal_candidate_poses_root": ["pose-0", "pose-1", "pose-2"],
        "temporal_candidate_sources": ["static", "left", "right"],
        "propagation_candidate_indices": [2, 0],
    }

    assert _temporal_solver_candidate_indices(evidence) == (2, 0)

    evidence.pop("propagation_candidate_indices")
    assert _temporal_solver_candidate_indices(evidence) == (0, 1, 2)

    evidence["propagation_candidate_indices"] = [0, 0]
    with pytest.raises(ValueError, match="propagation beam is invalid"):
        _temporal_solver_candidate_indices(evidence)


def test_tracking_direction_requires_endpoint_registration() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = [(rgb, np.ones((2, 2), dtype=np.float32))] * 2
    with np.testing.assert_raises_regex(ValueError, "does not begin"):
        _track_direction(
            estimator=_FakeFoundationPose(),
            frames=frames,
            intrinsic_matrix=np.eye(3),
            root_from_cameras=np.repeat(np.eye(4)[None], 2, axis=0),
            registration_masks={0: np.ones((2, 2), dtype=bool)},
            order=range(1, -1, -1),
            registration_iterations=5,
            tracking_iterations=2,
        )
