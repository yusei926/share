"""Select a physically coherent table-pose hypothesis sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .geometry import table_symmetry_transforms


TEMPORAL_SELECTION_SCHEMA_VERSION = "team_ramen_table_pose_temporal_selection/v24"
PHASE_NAMES = (
    "initial_static",
    "grasp_transition",
    "bimanual_hold",
    "release_or_settle",
    "final_static",
)

# Internal states 2 and 3 confirm bimanual attachment over two consecutive
# anchors. States 5 and 6 likewise require two consecutive final-static
# anchors. State 4 may return to approach/grasp for an observed recovery or
# regrasp; final-static states remain terminal.
_ALLOWED_PREVIOUS_STATES = (
    (0,),
    (0, 1, 4),
    (1, 4),
    (2, 3),
    (3, 4),
    (4,),
    (5, 6),
)


class TemporalSelectionError(ValueError):
    """A source trajectory has no table-pose path satisfying the physical contract."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class TemporalSelectionParameters:
    minimum_mask_precision: float
    minimum_mask_explained_fraction: float
    maximum_candidate_depth_error_m: float
    static_translation_scale_m: float
    static_rotation_scale_rad: float
    maximum_static_translation_step_m: float
    maximum_static_rotation_step_rad: float
    grasp_translation_scale_m: float
    grasp_rotation_scale_rad: float
    maximum_table_speed_m_s: float
    maximum_table_angular_speed_rad_s: float
    maximum_grasp_relative_translation_step_m: float
    maximum_grasp_relative_rotation_step_rad: float
    grasp_observed_position_max: float
    static_minimum_mask_precision: float = 0.15
    static_minimum_mask_explained_fraction: float = 0.6
    static_minimum_depth_overlap_fraction: float = 0.25
    static_maximum_depth_error_m: float = 0.025
    blurred_frame_laplacian_variance_max: float = 120.0
    carry_minimum_mask_precision: float = 0.03
    carry_minimum_mask_explained_fraction: float = 0.03
    carry_minimum_auxiliary_explained_fraction: float = 0.5
    carry_minimum_multiview_score: float = 0.1
    carry_visual_penalty: float = 2.0
    minimum_stereo_consistent_fraction: float = 0.6
    maximum_grasp_opening_speed_units_s: float = 0.5
    minimum_endpoint_normal_vertical_component: float | None = None

    def validate(self) -> None:
        values = asdict(self)
        if not all(
            value is None
            or (math.isfinite(float(value)) and float(value) > 0.0)
            for value in values.values()
        ):
            raise ValueError("temporal-selection parameters must be finite and positive")
        if (
            self.minimum_endpoint_normal_vertical_component is not None
            and self.minimum_endpoint_normal_vertical_component >= 1.0
        ):
            raise ValueError(
                "minimum_endpoint_normal_vertical_component must be below 1"
            )
        if not 0.0 < self.minimum_mask_precision < 1.0:
            raise ValueError("minimum_mask_precision must be in (0, 1)")
        if not 0.0 < self.minimum_mask_explained_fraction < 1.0:
            raise ValueError("minimum_mask_explained_fraction must be in (0, 1)")
        for name in (
            "static_minimum_mask_precision",
            "static_minimum_mask_explained_fraction",
            "static_minimum_depth_overlap_fraction",
        ):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.grasp_observed_position_max > 4.5:
            raise ValueError(
                "grasp_observed_position_max exceeds the measured Dex1 range"
            )
        for name in (
            "carry_minimum_mask_precision",
            "carry_minimum_mask_explained_fraction",
            "carry_minimum_auxiliary_explained_fraction",
            "minimum_stereo_consistent_fraction",
        ):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")


@dataclass(frozen=True)
class TemporalAnchor:
    ordinal: int
    source_frame_index: int
    candidate_poses_root: np.ndarray
    candidate_metrics: tuple[Mapping[str, object], ...]
    eef_poses_root: np.ndarray
    hand_observed_position: np.ndarray
    primary_laplacian_variance: float = 1000.0
    primary_stereo_consistent_fraction: float = 1.0


@dataclass(frozen=True)
class TemporalSelectionResult:
    selected_candidate_indices: np.ndarray
    selected_symmetry_indices: np.ndarray
    phase_indices: np.ndarray
    selected_poses_root: np.ndarray
    transition_translation_m: np.ndarray
    transition_rotation_rad: np.ndarray
    selected_evidence_modes: tuple[str, ...]
    total_cost: float

    def to_json(self, anchors: tuple[TemporalAnchor, ...]) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_SELECTION_SCHEMA_VERSION,
            "total_cost": self.total_cost,
            "anchors": [
                {
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "candidate_index": int(self.selected_candidate_indices[index]),
                    "candidate_source": str(
                        anchor.candidate_metrics[
                            int(self.selected_candidate_indices[index])
                        ].get("source", "global_registration")
                    ),
                    "symmetry_index": int(self.selected_symmetry_indices[index]),
                    "phase": PHASE_NAMES[int(self.phase_indices[index])],
                    "evidence_mode": self.selected_evidence_modes[index],
                    "primary_laplacian_variance": anchor.primary_laplacian_variance,
                    "primary_stereo_consistent_fraction": (
                        anchor.primary_stereo_consistent_fraction
                    ),
                }
                for index, anchor in enumerate(anchors)
            ],
            "transition_translation_m": self.transition_translation_m.tolist(),
            "transition_rotation_rad": self.transition_rotation_rad.tolist(),
        }


def audit_temporal_evidence_gaps(
    anchors: tuple[TemporalAnchor, ...],
    selection: TemporalSelectionResult,
    *,
    sampled_hand_observed_positions: np.ndarray,
    maximum_dense_gap_source_frames: int,
    source_fps: float,
    parameters: TemporalSelectionParameters,
) -> tuple[dict[str, object], ...]:
    """Require dense vision or an explicit physical constraint for every gap."""

    hands = np.asarray(sampled_hand_observed_positions, dtype=np.float64)
    if (
        len(anchors) != len(selection.selected_poses_root)
        or hands.ndim != 2
        or hands.shape[1:] != (2,)
        or not np.isfinite(hands).all()
        or maximum_dense_gap_source_frames <= 0
        or not math.isfinite(source_fps)
        or source_fps <= 0.0
    ):
        raise ValueError("temporal evidence-gap audit inputs are invalid")
    audits = []
    failures = []
    maximum_constrained_gap = 3 * maximum_dense_gap_source_frames
    for index in range(1, len(anchors)):
        previous = anchors[index - 1]
        current = anchors[index]
        source_gap = current.source_frame_index - previous.source_frame_index
        if source_gap <= maximum_dense_gap_source_frames:
            audits.append(
                {
                    "previous_source_frame_index": previous.source_frame_index,
                    "current_source_frame_index": current.source_frame_index,
                    "source_frame_gap": source_gap,
                    "mode": "dense_visual_evidence",
                    "passes_gate": True,
                }
            )
            continue
        if (
            previous.ordinal < 0
            or current.ordinal >= len(hands)
            or previous.ordinal >= current.ordinal
        ):
            raise ValueError("temporal evidence-gap ordinals are invalid")
        interval_hands = hands[previous.ordinal : current.ordinal + 1]
        active_throughout = np.all(
            interval_hands <= parameters.grasp_observed_position_max, axis=0
        )
        open_throughout = bool(
            np.all(interval_hands > parameters.grasp_observed_position_max)
        )
        hand_steps = np.diff(interval_hands, axis=0)
        single_attachment_transition = bool(
            np.all(
                np.all(hand_steps >= -0.1, axis=0)
                | np.all(hand_steps <= 0.1, axis=0)
            )
        )
        active_samples = (
            interval_hands <= parameters.grasp_observed_position_max
        )
        active_run_starts = active_samples & np.concatenate(
            (np.ones((1, 2), dtype=bool), ~active_samples[:-1]), axis=0
        )
        release_to_open_without_regrasp = bool(
            np.all(~active_samples[-1])
            and np.all(np.count_nonzero(active_run_starts, axis=0) <= 1)
        )
        previous_phase = int(selection.phase_indices[index - 1])
        current_phase = int(selection.phase_indices[index])
        previous_pose = selection.selected_poses_root[index - 1]
        current_pose = selection.selected_poses_root[index]
        translation, rotation = _pair_errors(
            previous_pose[None], current_pose[None]
        )
        table_translation = float(translation[0, 0])
        table_rotation = float(rotation[0, 0])
        elapsed_s = source_gap / source_fps
        bounded_free_motion = bool(
            table_translation
            <= parameters.maximum_table_speed_m_s * elapsed_s + 0.02
            and table_rotation
            <= parameters.maximum_table_angular_speed_rad_s * elapsed_s + 0.1
        )
        mode = None
        constraint = {}
        if (
            source_gap <= maximum_constrained_gap
            and open_throughout
            and previous_phase == current_phase
            and previous_phase in (0, 4)
            and table_translation <= parameters.maximum_static_translation_step_m
            and table_rotation <= parameters.maximum_static_rotation_step_rad
        ):
            mode = "open_hand_static_endpoint_bridge"
        else:
            required_active_sides = np.flatnonzero(active_throughout)
            if previous_phase == 2 or current_phase == 2:
                required_active_sides = (
                    np.asarray((0, 1), dtype=np.int64)
                    if np.all(active_throughout)
                    else np.empty(0, dtype=np.int64)
                )
            residuals = []
            for side in required_active_sides:
                previous_relative = _table_from_eef(
                    previous_pose[None], previous.eef_poses_root[int(side)]
                )
                current_relative = _table_from_eef(
                    current_pose[None], current.eef_poses_root[int(side)]
                )
                residual_translation, residual_rotation = _pair_errors(
                    previous_relative, current_relative
                )
                residuals.append(
                    {
                        "side": "left" if side == 0 else "right",
                        "translation_m": float(residual_translation[0, 0]),
                        "rotation_rad": float(residual_rotation[0, 0]),
                    }
                )
            if (
                source_gap <= maximum_constrained_gap
                and len(residuals) > 0
                and all(
                    value["translation_m"]
                    <= parameters.maximum_grasp_relative_translation_step_m
                    and value["rotation_rad"]
                    <= parameters.maximum_grasp_relative_rotation_step_rad
                    for value in residuals
                )
                and previous_phase in (1, 2, 3)
                and current_phase in (1, 2, 3)
            ):
                mode = "continuous_hand_rigid_constraint_bridge"
                constraint["active_hand_residuals"] = residuals
            released_hand = bool(
                np.any(
                    (interval_hands[0] <= parameters.grasp_observed_position_max)
                    & (interval_hands[-1] > parameters.grasp_observed_position_max)
                )
            )
            if (
                mode is None
                and source_gap <= maximum_constrained_gap
                and released_hand
                and previous_phase in (2, 3)
                and current_phase == 4
            ):
                mode = "release_to_verified_final_static_bridge"
            if (
                mode is None
                and source_gap <= maximum_constrained_gap
                and not np.any(active_throughout)
                and previous_phase == current_phase == 3
                and bounded_free_motion
                and (
                    open_throughout
                    or single_attachment_transition
                    or release_to_open_without_regrasp
                )
            ):
                # No stale attachment spans this interval. Each hand is either
                # open or changes in only one direction, so this covers one
                # release/regrasp transition without admitting oscillatory
                # contact. V1 replay remains the downstream physical authority.
                mode = "release_regrasp_visual_endpoint_bridge"
            if (
                mode is None
                and open_throughout
                and release_to_open_without_regrasp
                and previous_phase == 3
                and current_phase == 4
                and bounded_free_motion
            ):
                # Once both hands have released the object, no EEF-relative
                # transform is used by Mimic. Independently verified release
                # and final-static endpoints may therefore bound an occluded
                # free-settle interval without inventing a contact constraint.
                mode = "open_hand_settle_to_verified_final_static"
        audit = {
            "previous_source_frame_index": previous.source_frame_index,
            "current_source_frame_index": current.source_frame_index,
            "source_frame_gap": source_gap,
            "maximum_dense_gap_source_frames": maximum_dense_gap_source_frames,
            "maximum_constrained_gap_source_frames": maximum_constrained_gap,
            "previous_phase": PHASE_NAMES[previous_phase],
            "current_phase": PHASE_NAMES[current_phase],
            "table_translation_m": table_translation,
            "table_rotation_rad": table_rotation,
            "average_table_speed_m_s": table_translation / elapsed_s,
            "average_table_angular_speed_rad_s": table_rotation / elapsed_s,
            "active_hands_throughout": active_throughout.tolist(),
            "all_hands_open_throughout": open_throughout,
            "single_attachment_transition": single_attachment_transition,
            "release_to_open_without_regrasp": (
                release_to_open_without_regrasp
            ),
            "mode": mode or "unconstrained_gap",
            "passes_gate": mode is not None,
            **constraint,
        }
        audits.append(audit)
        if mode is None:
            failures.append(audit)
    if failures:
        raise TemporalSelectionError(
            "visual evidence gaps lack a continuous physical constraint",
            {
                "stage": "evidence_coverage",
                "rejection_reason": "temporal_evidence_gap_unconstrained",
                "failed_gaps": failures,
                "gap_audits": audits,
            },
        )
    return tuple(audits)


def _transforms(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (4, 4) or len(result) == 0:
        raise ValueError(f"{label} must be non-empty [N,4,4]")
    if not np.isfinite(result).all() or not np.allclose(
        result[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7
    ):
        raise ValueError(f"{label} contains invalid transforms")
    rotations = result[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{label} rotations are not orthonormal")
    if np.any(np.linalg.det(rotations) < 0.99999):
        raise ValueError(f"{label} rotations are not proper")
    return result


def _pair_errors(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(
        first[:, None, :3, 3] - second[None, :, :3, 3], axis=2
    )
    dot = np.einsum("nij,mij->nm", first[:, :3, :3], second[:, :3, :3])
    rotation = np.arccos(np.clip((dot - 1.0) * 0.5, -1.0, 1.0))
    return translation, rotation


def _table_from_eef(tables: np.ndarray, eef: np.ndarray) -> np.ndarray:
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(tables), axis=0)
    table_rotation_t = tables[:, :3, :3].transpose(0, 2, 1)
    result[:, :3, :3] = table_rotation_t @ eef[:3, :3]
    result[:, :3, 3] = np.einsum(
        "nij,nj->ni", table_rotation_t, eef[:3, 3] - tables[:, :3, 3]
    )
    return result


def _carry_source_matches_observed_hand(
    source: str,
    observed_hand_position: np.ndarray,
    threshold: float,
) -> bool:
    active = observed_hand_position <= threshold
    if source in {"left_eef_carry_forward", "left_eef_carry_reverse"}:
        return bool(active[0])
    if source in {"right_eef_carry_forward", "right_eef_carry_reverse"}:
        return bool(active[1])
    if source in {"bimanual_eef_carry_forward", "bimanual_eef_carry_reverse"}:
        return bool(np.all(active))
    return False


def _carry_auxiliary_support(
    source: str, auxiliary: Mapping[str, object]
) -> tuple[float, ...]:
    required_views = {
        "left_eef_carry_forward": ("left_wrist",),
        "left_eef_carry_reverse": ("left_wrist",),
        "right_eef_carry_forward": ("right_wrist",),
        "right_eef_carry_reverse": ("right_wrist",),
        "bimanual_eef_carry_forward": ("left_wrist", "right_wrist"),
        "bimanual_eef_carry_reverse": ("left_wrist", "right_wrist"),
    }.get(source, ())
    try:
        return tuple(float(auxiliary[name]) for name in required_views)
    except (KeyError, TypeError, ValueError):
        return ()


def _candidate_visual_evidence(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
    metric: Mapping[str, object],
    *,
    require_bidirectional_consensus: bool,
) -> tuple[float, str] | None:
    try:
        precision = float(metric["mask_precision"])
        explained = float(metric["mask_explained_fraction"])
        raw_precision = float(metric.get("raw_mask_precision", precision))
        raw_explained = float(
            metric.get("raw_mask_explained_fraction", explained)
        )
        consistent = float(metric["depth_consistent_union_fraction"])
        multiview = float(metric.get("multiview_score", consistent))
        depth_overlap = float(metric.get("depth_overlap_fraction", 0.0))
        depth_value = metric["median_absolute_depth_error_m"]
        depth_error = float(depth_value) if depth_value is not None else math.inf
        source = str(metric.get("source", "global_registration"))
        evidence_kind = str(metric.get("evidence_kind", "primary_mask_rgbd"))
        auxiliary = metric.get("auxiliary_mask_explained_fractions", {})
        auxiliary_precision = metric.get("auxiliary_mask_precisions", {})
        metric_stereo_fraction = metric.get("primary_stereo_consistent_fraction")
        stereo_consistent_fraction = float(
            anchor.primary_stereo_consistent_fraction
            if metric_stereo_fraction is None
            else metric_stereo_fraction
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value)
        for value in (
            precision,
            explained,
            raw_precision,
            raw_explained,
            consistent,
            multiview,
            depth_overlap,
            stereo_consistent_fraction,
        )
    ):
        return None
    rgbd_visual = (
        stereo_consistent_fraction
        >= parameters.minimum_stereo_consistent_fraction
        and precision >= parameters.minimum_mask_precision
        and explained >= parameters.minimum_mask_explained_fraction
        and depth_error <= parameters.maximum_candidate_depth_error_m
    )
    auxiliary_values = (
        _carry_auxiliary_support(source, auxiliary)
        if isinstance(auxiliary, Mapping)
        else ()
    )
    auxiliary_carry_visual = (
        bool(auxiliary_values)
        and all(math.isfinite(value) for value in auxiliary_values)
        and min(auxiliary_values)
        >= parameters.carry_minimum_auxiliary_explained_fraction
    )
    primary_carry_visual = (
        precision >= parameters.carry_minimum_mask_precision
        and explained >= parameters.carry_minimum_mask_explained_fraction
        and math.isfinite(depth_error)
        and depth_error <= parameters.maximum_candidate_depth_error_m
        and max((multiview, *auxiliary_values), default=multiview)
        >= parameters.carry_minimum_multiview_score
    )
    primary_view_unobserved = (
        precision == 0.0
        and explained == 0.0
        and not math.isfinite(depth_error)
    )
    carry_visual = (
        _carry_source_matches_observed_hand(
            source,
            anchor.hand_observed_position,
            parameters.grasp_observed_position_max,
        )
        and auxiliary_carry_visual
        and (
            primary_carry_visual
            or (
                primary_view_unobserved
                and (
                    anchor.primary_laplacian_variance
                    <= parameters.blurred_frame_laplacian_variance_max
                    or stereo_consistent_fraction
                    < parameters.minimum_stereo_consistent_fraction
                )
            )
        )
    )
    expected_auxiliary_views = {"left_wrist", "right_wrist"}
    static_auxiliary_visual = False
    if (
        isinstance(auxiliary, Mapping)
        and isinstance(auxiliary_precision, Mapping)
        and set(auxiliary) == expected_auxiliary_views
        and set(auxiliary_precision) == expected_auxiliary_views
    ):
        try:
            static_auxiliary_visual = bool(
                min(float(auxiliary[name]) for name in expected_auxiliary_views)
                >= parameters.carry_minimum_auxiliary_explained_fraction
                and min(
                    float(auxiliary_precision[name])
                    for name in expected_auxiliary_views
                )
                >= parameters.carry_minimum_mask_precision
            )
        except (TypeError, ValueError):
            static_auxiliary_visual = False
    static_silhouette_visual = (
        not rgbd_visual
        and (
            anchor.primary_laplacian_variance
            <= parameters.blurred_frame_laplacian_variance_max
            or anchor.primary_stereo_consistent_fraction
            < parameters.minimum_stereo_consistent_fraction
        )
        and np.all(
            anchor.hand_observed_position
            > parameters.grasp_observed_position_max
        )
        and raw_precision >= parameters.carry_minimum_mask_precision
        and raw_explained >= parameters.static_minimum_mask_explained_fraction
        and math.isfinite(depth_error)
        and depth_error <= parameters.maximum_candidate_depth_error_m
        and static_auxiliary_visual
    )
    consensus = metric.get("bidirectional_consensus")
    unsegmented_multiview_visual = (
        evidence_kind == "unsegmented_multiview_rgbd"
        and stereo_consistent_fraction
        >= parameters.minimum_stereo_consistent_fraction
        and depth_overlap >= parameters.static_minimum_depth_overlap_fraction
        and math.isfinite(depth_error)
        and depth_error <= parameters.maximum_candidate_depth_error_m
        and static_auxiliary_visual
        and isinstance(consensus, Mapping)
        and consensus.get("passes_gate") is True
    )
    if (
        not rgbd_visual
        and not carry_visual
        and not static_silhouette_visual
        and not unsegmented_multiview_visual
    ):
        return None
    quality = (
        8.0 * multiview
        + 3.0 * explained
        + 2.0 * precision
        - 3.0 * min(depth_error if math.isfinite(depth_error) else 0.3, 0.3)
    )
    if carry_visual:
        causal_current_frame_carry = bool(
            isinstance(consensus, Mapping)
            and consensus.get("validation_mode")
            == "causal_current_frame_rgbd_propagation"
        )
        if require_bidirectional_consensus and not (
            causal_current_frame_carry
            or (
                isinstance(consensus, Mapping)
                and consensus.get("passes_gate") is True
            )
        ):
            return None
        return -quality + parameters.carry_visual_penalty, "eef_wrist_rigid"
    if require_bidirectional_consensus:
        if not isinstance(consensus, Mapping) or consensus.get("passes_gate") is not True:
            return None
    if unsegmented_multiview_visual:
        bridge_quality = (
            8.0 * multiview
            + 3.0 * depth_overlap
            + sum(float(auxiliary[name]) for name in expected_auxiliary_views)
            + sum(
                float(auxiliary_precision[name])
                for name in expected_auxiliary_views
            )
        )
        return (
            -bridge_quality + parameters.carry_visual_penalty,
            "unsegmented_multiview_rgbd_bidirectional",
        )
    if static_silhouette_visual:
        silhouette_quality = (
            3.0 * raw_explained
            + 2.0 * raw_precision
            + sum(float(auxiliary[name]) for name in expected_auxiliary_views)
            + sum(
                float(auxiliary_precision[name])
                for name in expected_auxiliary_views
            )
        )
        return (
            -silhouette_quality + parameters.carry_visual_penalty,
            "static_multiview_silhouette_bidirectional",
        )
    return -quality, "rgbd_bidirectional"


def temporal_visual_costs(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
    *,
    require_bidirectional_consensus: bool = False,
) -> np.ndarray:
    metrics = tuple(anchor.candidate_metrics)
    costs = np.full(len(metrics), np.inf, dtype=np.float64)
    for index, metric in enumerate(metrics):
        evidence = _candidate_visual_evidence(
            anchor,
            parameters,
            metric,
            require_bidirectional_consensus=require_bidirectional_consensus,
        )
        if evidence is not None:
            costs[index] = evidence[0]
    if not np.isfinite(costs).any():
        raise ValueError("registration anchor has no geometrically eligible candidates")
    return costs


def temporal_static_visual_costs(
    anchor: TemporalAnchor, parameters: TemporalSelectionParameters
) -> np.ndarray:
    costs = temporal_visual_costs(
        anchor,
        parameters,
        require_bidirectional_consensus=True,
    )
    for index, metric in enumerate(anchor.candidate_metrics):
        try:
            raw_precision = float(
                metric.get("raw_mask_precision", metric["mask_precision"])
            )
            raw_explained = float(
                metric.get(
                    "raw_mask_explained_fraction",
                    metric["mask_explained_fraction"],
                )
            )
            visible_precision = float(metric["mask_precision"])
            visible_explained = float(metric["mask_explained_fraction"])
            overlap = float(metric["depth_overlap_fraction"])
            depth_value = metric["median_absolute_depth_error_m"]
            depth_error = float(depth_value) if depth_value is not None else math.inf
            consensus = metric["bidirectional_consensus"]
            raw_rendered_pixels = int(metric.get("raw_rendered_pixels", 0))
            rendered_pixels = int(metric.get("rendered_pixels", 0))
            occluded_rendered_pixels = int(metric.get("occluded_rendered_pixels", 0))
        except (KeyError, TypeError, ValueError):
            costs[index] = math.inf
            continue
        common_static_evidence = (
            math.isfinite(raw_precision)
            and math.isfinite(raw_explained)
            and math.isfinite(visible_precision)
            and math.isfinite(visible_explained)
            and math.isfinite(overlap)
            and math.isfinite(depth_error)
            and anchor.primary_stereo_consistent_fraction
            >= parameters.minimum_stereo_consistent_fraction
            and overlap >= parameters.static_minimum_depth_overlap_fraction
            and depth_error <= parameters.static_maximum_depth_error_m
            and isinstance(consensus, Mapping)
            and consensus.get("passes_gate") is True
        )
        raw_static_silhouette = (
            raw_precision >= parameters.static_minimum_mask_precision
            and raw_explained >= parameters.static_minimum_mask_explained_fraction
        )
        occlusion_aware_static_silhouette = (
            visible_precision >= parameters.static_minimum_mask_precision
            and visible_explained >= parameters.static_minimum_mask_explained_fraction
            and raw_explained >= parameters.static_minimum_mask_explained_fraction
            and raw_rendered_pixels > rendered_pixels > 0
            and occluded_rendered_pixels > 0
        )
        if not (
            common_static_evidence
            and (raw_static_silhouette or occlusion_aware_static_silhouette)
        ):
            costs[index] = math.inf
    if not np.isfinite(costs).any():
        raise ValueError("registration anchor has no static RGB-D candidates")
    return costs


def _causal_static_bridge_visual_costs(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
) -> np.ndarray:
    """Return forward static-carry candidates usable before a confirmed grasp.

    A bidirectional registration can disagree slightly while the table is
    static because the reverse pass starts from an occluded manipulation
    frame.  This bridge therefore accepts only a *forward static carry* with
    strong current RGB-D support and disagreement bounded by the existing
    static-motion contract.  It is unavailable once a hand has a confirmed
    closure transition, where the EEF-rigid constraint is stronger evidence.
    """

    costs = np.full(len(anchor.candidate_metrics), np.inf, dtype=np.float64)
    for index, metric in enumerate(anchor.candidate_metrics):
        try:
            source = str(metric["source"])
            precision = float(metric["mask_precision"])
            explained = float(metric["mask_explained_fraction"])
            overlap = float(metric["depth_overlap_fraction"])
            depth_error = float(metric["median_absolute_depth_error_m"])
            multiview = float(metric["multiview_score"])
            consensus = metric.get("bidirectional_consensus", {})
        except (KeyError, TypeError, ValueError):
            continue
        direct_causal_rgbd = bool(
            isinstance(consensus, Mapping)
            and consensus.get("validation_mode")
            == "causal_current_frame_rgbd_propagation"
        )
        if direct_causal_rgbd:
            reverse_translation = 0.0
            reverse_rotation = 0.0
        else:
            try:
                reverse_translation = float(consensus["translation_error_m"])
                reverse_rotation = float(consensus["rotation_error_rad"])
            except (KeyError, TypeError, ValueError):
                continue
        if (
            source != "static_carry_forward"
            or anchor.primary_stereo_consistent_fraction
            < parameters.minimum_stereo_consistent_fraction
            or precision < parameters.static_minimum_mask_precision
            or explained < parameters.static_minimum_mask_explained_fraction
            or overlap < parameters.static_minimum_depth_overlap_fraction
            or depth_error > parameters.static_maximum_depth_error_m
            or reverse_translation > parameters.maximum_static_translation_step_m
            or reverse_rotation > parameters.maximum_static_rotation_step_rad
        ):
            continue
        quality = (
            8.0 * multiview
            + 3.0 * explained
            + 2.0 * precision
            - 3.0 * min(depth_error, 0.3)
        )
        costs[index] = -quality + parameters.carry_visual_penalty
    return costs


def _causal_eef_carry_visual_costs(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
) -> np.ndarray:
    """Admit only current-frame-rendered EEF carries into the causal tracker.

    The candidate is constructed from the previously selected table pose and
    the measured change in a continuously closed wrist FK.  Its render is then
    checked against the *current* head/wrist RGB-D masks.  That is different
    from accepting an arbitrary forward registration: it has an explicit
    causal kinematic parent and current visual support.  The normal tracker
    still requires bidirectional evidence for any terminal/static conclusion.
    """

    try:
        visual_costs = temporal_visual_costs(
            anchor, parameters, require_bidirectional_consensus=False
        )
    except ValueError:
        return np.full(len(anchor.candidate_metrics), np.inf, dtype=np.float64)
    costs = np.full_like(visual_costs, np.inf)
    accepted_sources = {
        "left_eef_carry_forward",
        "right_eef_carry_forward",
        "bimanual_eef_carry_forward",
    }
    for index, metric in enumerate(anchor.candidate_metrics):
        consensus = metric.get("bidirectional_consensus")
        if (
            str(metric.get("source", "")) not in accepted_sources
            or not isinstance(consensus, Mapping)
            or consensus.get("validation_mode")
            != "causal_current_frame_rgbd_propagation"
            or not np.isfinite(visual_costs[index])
        ):
            continue
        costs[index] = visual_costs[index]
    return costs


def _causal_expanded_anchor(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand strict and pre-grasp static-bridge visual candidates."""

    poses = _transforms(anchor.candidate_poses_root, "candidate_poses_root")
    metrics = tuple(anchor.candidate_metrics)
    if len(metrics) != len(poses):
        raise ValueError("candidate pose and metric counts differ")
    try:
        strict_costs = temporal_visual_costs(
            anchor, parameters, require_bidirectional_consensus=True
        )
    except ValueError:
        strict_costs = np.full(len(poses), np.inf, dtype=np.float64)
    carry_costs = _causal_eef_carry_visual_costs(anchor, parameters)
    bridge_costs = _causal_static_bridge_visual_costs(anchor, parameters)
    combined_costs = np.where(
        np.isfinite(strict_costs),
        strict_costs,
        np.where(np.isfinite(carry_costs), carry_costs, bridge_costs),
    )
    eligible_indices = np.flatnonzero(np.isfinite(combined_costs))
    if len(eligible_indices) == 0:
        raise ValueError("causal anchor has no geometrically eligible candidates")
    symmetries = np.stack(table_symmetry_transforms())
    expanded = (poses[eligible_indices, None] @ symmetries[None]).reshape(-1, 4, 4)
    candidate_indices = np.repeat(eligible_indices, len(symmetries))
    symmetry_indices = np.tile(
        np.arange(len(symmetries), dtype=np.int64), len(poses[eligible_indices])
    )
    visual = np.repeat(combined_costs[eligible_indices], len(symmetries))
    # Only a static carry is a pre-grasp bridge. A causally propagated EEF
    # candidate remains an attached-motion candidate and must stay eligible
    # when the selector applies the wrist-rigidity constraint.
    bridge = np.repeat(
        ~np.isfinite(strict_costs[eligible_indices])
        & ~np.isfinite(carry_costs[eligible_indices]),
        len(symmetries),
    )
    try:
        static_costs = temporal_static_visual_costs(anchor, parameters)
        static_visual = np.repeat(static_costs[eligible_indices], len(symmetries))
    except ValueError:
        static_visual = np.full(len(expanded), np.inf, dtype=np.float64)
    return expanded, visual, static_visual, bridge, candidate_indices, symmetry_indices


def _causal_trace(state: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Recover a retained causal beam path without copying it at every step."""

    trace = []
    current: Mapping[str, object] | None = state
    while current is not None:
        trace.append(current)
        parent = current.get("parent")
        if parent is not None and not isinstance(parent, Mapping):
            raise ValueError("causal beam parent is invalid")
        current = parent
    trace.reverse()
    return tuple(trace)


def _expanded_anchor(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poses = _transforms(anchor.candidate_poses_root, "candidate_poses_root")
    metrics = tuple(anchor.candidate_metrics)
    if len(metrics) != len(poses):
        raise ValueError("candidate pose and metric counts differ")
    visual_costs = temporal_visual_costs(
        anchor,
        parameters,
        require_bidirectional_consensus=True,
    )
    eligible_indices = np.flatnonzero(np.isfinite(visual_costs))
    poses = poses[eligible_indices]
    symmetries = np.stack(table_symmetry_transforms())
    expanded = (poses[:, None] @ symmetries[None]).reshape(-1, 4, 4)
    candidate_indices = np.repeat(eligible_indices, len(symmetries))
    symmetry_indices = np.tile(np.arange(len(symmetries), dtype=np.int64), len(poses))
    visual = np.repeat(visual_costs[eligible_indices], len(symmetries))
    try:
        static_costs = temporal_static_visual_costs(anchor, parameters)
        static_visual = np.repeat(
            static_costs[eligible_indices], len(symmetries)
        )
    except ValueError:
        static_visual = np.full(len(expanded), np.inf, dtype=np.float64)
    return expanded, visual, static_visual, candidate_indices, symmetry_indices


def select_causally_constrained_poses(
    anchors: Iterable[TemporalAnchor],
    *,
    source_fps: float,
    parameters: TemporalSelectionParameters,
) -> TemporalSelectionResult:
    """Select an RGB-D pose path using only the preceding accepted pose.

    This fallback deliberately avoids independent per-interval beam lineages.
    During a continuous observed grasp, every accepted pose must preserve the
    table-to-EEF transform from the immediately preceding pose.  At a grasp
    transition, only bounded visual motion is allowed.  It is therefore a
    causal CAD/RGB-D/FK tracker, not a simulator-state shortcut.
    """

    parameters.validate()
    values = tuple(anchors)
    if len(values) < 4 or not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("causal temporal selection needs four anchors and positive FPS")
    expanded = [_causal_expanded_anchor(anchor, parameters) for anchor in values]
    # Keep the complete configured sparse registration beam for each of the
    # two table symmetries.  A smaller visual top-k can discard the sole
    # hand-compatible orientation before the first observed grasp.
    initial_beam_size = 128
    attached_beam_size = 8
    first_static = expanded[0][2]
    first_indices = np.flatnonzero(np.isfinite(first_static))
    if len(first_indices) == 0:
        raise TemporalSelectionError(
            "initial causal tracker anchor lacks static RGB-D evidence",
            {"anchor_index": 0, "ordinal": values[0].ordinal},
        )
    first_indices = first_indices[np.argsort(first_static[first_indices], kind="stable")]
    states = [
        {
            "choice": int(choice),
            "cost": float(first_static[choice]),
            "attached": np.zeros(2, dtype=bool),
            "parent": None,
            "phase": 0,
            "mode": "static_rgbd_bidirectional",
        }
        for choice in first_indices[:initial_beam_size]
    ]
    had_grasp = False
    for index, anchor in enumerate(values[1:], start=1):
        poses, visual, static_visual, bridge, _, _ = expanded[index]
        previous = values[index - 1]
        active = anchor.hand_observed_position <= parameters.grasp_observed_position_max
        elapsed = (anchor.source_frame_index - previous.source_frame_index) / source_fps
        max_translation = parameters.maximum_table_speed_m_s * elapsed + 0.02
        max_rotation = parameters.maximum_table_angular_speed_rad_s * elapsed + 0.1
        next_states = []
        diagnostics = {
            "visual_eligible_candidate_count": int(
                np.count_nonzero(np.isfinite(visual))
            ),
            "bounded_candidate_count": 0,
            "constrained_candidate_count": 0,
            "translation_eligible_candidate_count": 0,
            "rotation_eligible_candidate_count": 0,
            "minimum_table_translation_m": math.inf,
            "minimum_table_rotation_rad": math.inf,
        }
        for state in states:
            previous_pose = expanded[index - 1][0][state["choice"]]
            translation = np.linalg.norm(
                poses[:, :3, 3] - previous_pose[:3, 3], axis=1
            )
            dot = np.einsum("nij,ij->n", poses[:, :3, :3], previous_pose[:3, :3])
            rotation = np.arccos(np.clip((dot - 1.0) * 0.5, -1.0, 1.0))
            translation_eligible = translation <= max_translation
            rotation_eligible = rotation <= max_rotation
            bounded = translation_eligible & rotation_eligible
            previous_active = (
                previous.hand_observed_position
                <= parameters.grasp_observed_position_max
            )
            engaging = active & ~previous_active
            continuous = active & previous_active & state["attached"]
            confirmed_active = active & state["attached"]
            phase = (
                2
                if np.all(confirmed_active)
                else 1
                if np.any(confirmed_active | engaging)
                else 4
                if had_grasp
                else 0
            )
            allow_static_bridge = not np.any(state["attached"]) and not np.any(engaging)
            candidate_visual = (
                np.where(np.isfinite(static_visual), static_visual, visual)
                if allow_static_bridge
                else static_visual
                if phase == 4
                else visual
            )
            constrained = bounded.copy()
            for side in np.flatnonzero(continuous):
                previous_relative = _table_from_eef(
                    previous_pose[None], previous.eef_poses_root[int(side)]
                )[0]
                current_relative = _table_from_eef(poses, anchor.eef_poses_root[int(side)])
                relative_translation = np.linalg.norm(
                    current_relative[:, :3, 3] - previous_relative[:3, 3], axis=1
                )
                relative_dot = np.einsum(
                    "nij,ij->n", current_relative[:, :3, :3], previous_relative[:3, :3]
                )
                relative_rotation = np.arccos(
                    np.clip((relative_dot - 1.0) * 0.5, -1.0, 1.0)
                )
                constrained &= (
                    relative_translation
                    <= parameters.maximum_grasp_relative_translation_step_m
                ) & (
                    relative_rotation
                    <= parameters.maximum_grasp_relative_rotation_step_rad
                )
            eligible = (
                constrained
                & np.isfinite(candidate_visual)
                & (allow_static_bridge | ~bridge)
            )
            diagnostics["bounded_candidate_count"] += int(np.count_nonzero(bounded))
            diagnostics["constrained_candidate_count"] += int(np.count_nonzero(constrained))
            diagnostics["translation_eligible_candidate_count"] += int(
                np.count_nonzero(translation_eligible)
            )
            diagnostics["rotation_eligible_candidate_count"] += int(
                np.count_nonzero(rotation_eligible)
            )
            diagnostics["minimum_table_translation_m"] = min(
                diagnostics["minimum_table_translation_m"], float(np.min(translation))
            )
            diagnostics["minimum_table_rotation_rad"] = min(
                diagnostics["minimum_table_rotation_rad"], float(np.min(rotation))
            )
            # A command/state reading above the measured closed threshold is a
            # release, not merely absence of a new grasp.  Keeping the old
            # constraint here would force a released hand to remain rigidly
            # attached to the table in later RGB-D/FK updates.
            attached = state["attached"].copy()
            attached &= active
            attached |= engaging
            for choice in np.flatnonzero(eligible):
                mode = (
                    "causal_eef_rigid_rgbd"
                    if np.any(continuous)
                    else "causal_static_bridge_rgbd"
                    if bridge[choice]
                    else "causal_attachment_rgbd"
                )
                next_states.append(
                    {
                        "choice": int(choice),
                        "cost": float(
                            state["cost"]
                            + candidate_visual[choice]
                            + translation[choice] / 0.2
                            + rotation[choice]
                        ),
                        "attached": attached,
                        "parent": state,
                        "phase": phase,
                        "mode": mode,
                    }
                )
        if not next_states:
            prefix = min(states, key=lambda state: (state["cost"], state["choice"]))
            prefix_trace = _causal_trace(prefix)
            diagnostics.update(
                {
                    "anchor_index": index,
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "active_hands": active.tolist(),
                    "elapsed_s": float(elapsed),
                    "maximum_table_translation_m": float(max_translation),
                    "maximum_table_rotation_rad": float(max_rotation),
                    # The tracker may render an additional causal candidate
                    # from this accepted prefix.  It is diagnostic evidence,
                    # not an unobserved object-state shortcut.
                    "causal_prefix": {
                        "selected_poses_root": [
                            expanded[anchor_index][0][int(state["choice"])].tolist()
                            for anchor_index, state in enumerate(prefix_trace)
                        ],
                        "attached_hands": prefix["attached"].tolist(),
                    },
                }
            )
            raise TemporalSelectionError(
                "causal RGB-D/FK tracker has no physically constrained candidate",
                diagnostics,
            )
        next_states.sort(key=lambda state: (state["cost"], state["choice"]))
        # Preserve every sparse orientation candidate until a real open-to-
        # closed transition confirms a grasp.  Once attached, current wrist
        # RGB-D plus the rigid EEF constraint remove the initial 180-degree
        # ambiguity, so a smaller beam is sufficient and avoids quadratic
        # work across the remaining dense anchors.
        states = next_states[
            : attached_beam_size if had_grasp else initial_beam_size
        ]
        had_grasp = had_grasp or bool(
            np.any(
                active
                & (
                    previous.hand_observed_position
                    > parameters.grasp_observed_position_max
                )
            )
        )
    winner = min(states, key=lambda state: (state["cost"], state["choice"]))
    winner_trace = _causal_trace(winner)
    selected_expanded = tuple(int(state["choice"]) for state in winner_trace)
    phases = tuple(int(state["phase"]) for state in winner_trace)
    evidence_modes = tuple(str(state["mode"]) for state in winner_trace)
    raw_indices = np.asarray(
        [expanded[index][4][choice] for index, choice in enumerate(selected_expanded)],
        dtype=np.int64,
    )
    symmetry_indices = np.asarray(
        [expanded[index][5][choice] for index, choice in enumerate(selected_expanded)],
        dtype=np.int64,
    )
    poses = np.asarray(
        [expanded[index][0][choice] for index, choice in enumerate(selected_expanded)],
        dtype=np.float64,
    )
    translation, rotation = _pair_errors(poses[:-1], poses[1:])
    return TemporalSelectionResult(
        selected_candidate_indices=raw_indices,
        selected_symmetry_indices=symmetry_indices,
        phase_indices=np.asarray(phases, dtype=np.int64),
        selected_poses_root=poses,
        transition_translation_m=np.diag(translation),
        transition_rotation_rad=np.diag(rotation),
        selected_evidence_modes=tuple(evidence_modes),
        total_cost=float(winner["cost"]),
    )


def select_temporally_consistent_poses(
    anchors: Iterable[TemporalAnchor],
    *,
    source_fps: float,
    parameters: TemporalSelectionParameters,
) -> TemporalSelectionResult:
    """Run a monotonic static/hold/static Viterbi selection over pose hypotheses."""

    parameters.validate()
    values = tuple(anchors)
    if len(values) < 4 or not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("temporal selection needs at least four anchors and positive FPS")
    if any(
        anchor.ordinal < 0
        or anchor.source_frame_index < 0
        or anchor.eef_poses_root.shape != (2, 4, 4)
        or anchor.hand_observed_position.shape != (2,)
        or not np.isfinite(anchor.eef_poses_root).all()
        or not np.isfinite(anchor.hand_observed_position).all()
        or not math.isfinite(anchor.primary_laplacian_variance)
        or anchor.primary_laplacian_variance <= 0.0
        or not math.isfinite(anchor.primary_stereo_consistent_fraction)
        or not 0.0 <= anchor.primary_stereo_consistent_fraction <= 1.0
        for anchor in values
    ):
        raise ValueError("temporal anchor metadata is invalid")
    frame_indices = np.asarray([anchor.source_frame_index for anchor in values], dtype=np.int64)
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("temporal anchors must use increasing source frames")

    expanded = []
    for anchor in values:
        try:
            expanded.append(_expanded_anchor(anchor, parameters))
        except ValueError as exc:
            metric_summary: dict[str, dict[str, float] | int] = {
                "candidate_count": len(anchor.candidate_metrics)
            }
            for key in (
                "mask_precision",
                "mask_explained_fraction",
                "depth_consistent_union_fraction",
                "median_absolute_depth_error_m",
            ):
                finite = []
                for metric in anchor.candidate_metrics:
                    try:
                        value = float(metric[key])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        finite.append(value)
                if finite:
                    metric_summary[key] = {
                        "minimum": min(finite),
                        "maximum": max(finite),
                    }
            raise TemporalSelectionError(
                f"temporal anchor {anchor.ordinal} (source frame "
                f"{anchor.source_frame_index}) is invalid: {exc}",
                {
                    "anchor_index": len(expanded),
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "candidate_metrics": metric_summary,
                },
            ) from exc
    # The duplicated bimanual and final-static states require two consecutive
    # constrained anchors. A separate release/settle state models asynchronous
    # finger release and the table's physically plausible residual motion.
    internal_phase = np.asarray((0, 1, 2, 2, 3, 4, 4), dtype=np.int64)
    state_count = len(internal_phase)
    final_state = state_count - 1
    costs = np.full((state_count, len(expanded[0][0])), np.inf, dtype=np.float64)
    costs[0] = expanded[0][2]
    endpoint_vertical = parameters.minimum_endpoint_normal_vertical_component
    if endpoint_vertical is not None:
        initial_orientation = expanded[0][0][:, 2, 2] <= -endpoint_vertical
        costs[0] = np.where(initial_orientation, costs[0], np.inf)
    if not np.isfinite(costs[0]).any():
        raise TemporalSelectionError(
            "the initial anchor has no bidirectionally verified static RGB-D pose",
            {
                "anchor_index": 0,
                "ordinal": values[0].ordinal,
                "source_frame_index": values[0].source_frame_index,
            },
        )
    back_pointers: list[np.ndarray] = []
    anchor_trace: list[dict[str, object]] = [
        {
            "anchor_index": 0,
            "ordinal": values[0].ordinal,
            "source_frame_index": values[0].source_frame_index,
            "hand_observed_position": values[0].hand_observed_position.tolist(),
            "reachable_by_internal_state": [
                int(np.count_nonzero(np.isfinite(costs[state])))
                for state in range(state_count)
            ],
        }
    ]

    for anchor_index in range(1, len(values)):
        previous_poses = expanded[anchor_index - 1][0]
        current_poses, visual, static_visual, _, _ = expanded[anchor_index]
        if endpoint_vertical is None:
            initial_orientation_possible = np.ones(len(current_poses), dtype=bool)
            final_orientation_possible = np.ones(len(current_poses), dtype=bool)
        else:
            current_normal_z = current_poses[:, 2, 2]
            initial_orientation_possible = current_normal_z <= -endpoint_vertical
            final_orientation_possible = current_normal_z >= endpoint_vertical
        table_translation, table_rotation = _pair_errors(previous_poses, current_poses)
        elapsed_s = (frame_indices[anchor_index] - frame_indices[anchor_index - 1]) / source_fps
        physically_possible = (
            table_translation <= parameters.maximum_table_speed_m_s * elapsed_s + 0.02
        ) & (
            table_rotation
            <= parameters.maximum_table_angular_speed_rad_s * elapsed_s + 0.1
        )
        static_cost = (
            table_translation / parameters.static_translation_scale_m
            + table_rotation / parameters.static_rotation_scale_rad
        )
        static_possible = physically_possible & (
            table_translation <= parameters.maximum_static_translation_step_m
        ) & (
            table_rotation <= parameters.maximum_static_rotation_step_rad
        )
        motion_cost = table_translation / 0.2 + table_rotation / 1.0

        grasp_translation = np.zeros_like(table_translation)
        grasp_rotation = np.zeros_like(table_rotation)
        bimanual_geometry_possible = physically_possible.copy()
        single_hand_grasp_possible = np.zeros_like(physically_possible)
        release_grasp_cost = np.full_like(table_translation, np.inf)
        any_continuous_hand_active = False
        continuous_hand_active = np.zeros(2, dtype=bool)
        hand_opening_speed = (
            values[anchor_index].hand_observed_position
            - values[anchor_index - 1].hand_observed_position
        ) / elapsed_s
        for side in range(2):
            previous_relative = _table_from_eef(
                previous_poses, values[anchor_index - 1].eef_poses_root[side]
            )
            current_relative = _table_from_eef(
                current_poses, values[anchor_index].eef_poses_root[side]
            )
            side_translation, side_rotation = _pair_errors(previous_relative, current_relative)
            grasp_translation += side_translation
            grasp_rotation += side_rotation
            side_possible = (
                side_translation <= parameters.maximum_grasp_relative_translation_step_m
            ) & (
                side_rotation <= parameters.maximum_grasp_relative_rotation_step_rad
            )
            side_possible &= physically_possible
            side_active = bool(
                values[anchor_index - 1].hand_observed_position[side]
                <= parameters.grasp_observed_position_max
                and values[anchor_index].hand_observed_position[side]
                <= parameters.grasp_observed_position_max
                and hand_opening_speed[side]
                <= parameters.maximum_grasp_opening_speed_units_s
            )
            continuous_hand_active[side] = side_active
            if side_active:
                any_continuous_hand_active = True
                single_hand_grasp_possible |= side_possible
                release_grasp_cost = np.minimum(
                    release_grasp_cost,
                    side_translation / parameters.grasp_translation_scale_m
                    + side_rotation / parameters.grasp_rotation_scale_rad,
                )
            bimanual_geometry_possible &= side_possible
        current_both_hands_active = bool(
            np.all(
                values[anchor_index].hand_observed_position
                <= parameters.grasp_observed_position_max
            )
        )
        previous_both_hands_active = bool(
            np.all(
                values[anchor_index - 1].hand_observed_position
                <= parameters.grasp_observed_position_max
            )
        )
        continuous_both_hands_active = bool(
            current_both_hands_active
            and previous_both_hands_active
            and np.all(continuous_hand_active)
        )
        bimanual_continue_possible = (
            bimanual_geometry_possible & continuous_both_hands_active
        )
        entering_bimanual_attachment = bool(
            current_both_hands_active and not previous_both_hands_active
        )
        # Closing the second hand may deliberately slide or regrasp the object,
        # so the prior single-hand transform is no longer a valid rigid constraint.
        # Reinitialize both attachments from the independently verified visual
        # pose while retaining bounded object motion. State 3 confirms both new
        # attachments at the following anchor.
        if entering_bimanual_attachment:
            attachment_enter_possible = physically_possible
            attachment_enter_cost = motion_cost
        elif any_continuous_hand_active:
            attachment_enter_possible = (
                single_hand_grasp_possible & current_both_hands_active
            )
            attachment_enter_cost = release_grasp_cost
        else:
            attachment_enter_possible = (
                physically_possible & current_both_hands_active
            )
            attachment_enter_cost = motion_cost
        both_hands_open = bool(
            np.all(
                values[anchor_index].hand_observed_position
                > parameters.grasp_observed_position_max
            )
        )
        grasp_cost = (
            grasp_translation / parameters.grasp_translation_scale_m
            + grasp_rotation / parameters.grasp_rotation_scale_rad
        )
        opening_transition = bool(
            np.any(
                hand_opening_speed
                > parameters.maximum_grasp_opening_speed_units_s
            )
        )
        regrasp_enter = not both_hands_open and not any_continuous_hand_active
        if both_hands_open:
            release_possible = physically_possible
            release_cost = motion_cost + 0.5
        elif opening_transition:
            # During asynchronous release the object may slide against the
            # remaining hand or settle onto the workbench. Keep the bounded
            # motion and visual gates, but do not preserve a stale attachment.
            release_possible = physically_possible
            release_cost = motion_cost + 0.5
        elif regrasp_enter:
            release_possible = physically_possible
            release_cost = motion_cost + 1.0
        else:
            release_possible = single_hand_grasp_possible
            release_cost = release_grasp_cost

        next_costs = np.full(
            (state_count, len(current_poses)), np.inf, dtype=np.float64
        )
        pointers = np.full(
            (state_count, len(current_poses), 2), -1, dtype=np.int64
        )
        for current_state, previous_states in enumerate(_ALLOWED_PREVIOUS_STATES):
            current_phase = int(internal_phase[current_state])
            state_visual = (
                static_visual if current_state in (0, 5, 6) else visual
            )
            for previous_state in previous_states:
                # Most candidate/state pairs become unreachable long before
                # the end of a dense episode.  Restricting the reduction to
                # finite predecessors is algebraically identical to retaining
                # their +inf rows, while avoiding the dominant dense
                # [previous_candidates, current_candidates] allocation.
                reachable_previous = np.flatnonzero(
                    np.isfinite(costs[previous_state])
                )
                if len(reachable_previous) == 0:
                    continue
                if current_state == 1:
                    if any_continuous_hand_active:
                        transition = release_grasp_cost
                        possible = single_hand_grasp_possible
                    else:
                        transition = static_cost
                        possible = static_possible
                elif current_state == 2:
                    transition = attachment_enter_cost
                    possible = attachment_enter_possible
                elif current_state == 3:
                    transition = grasp_cost
                    possible = bimanual_continue_possible
                elif current_state == 4:
                    transition = release_cost
                    possible = release_possible
                elif current_state == 5:
                    transition = motion_cost
                    possible = (
                        physically_possible
                        & both_hands_open
                        & final_orientation_possible[None, :]
                    )
                elif current_state == 6:
                    transition = static_cost
                    possible = (
                        static_possible
                        & both_hands_open
                        & final_orientation_possible[None, :]
                    )
                else:
                    transition = static_cost
                    possible = (
                        static_possible & initial_orientation_possible[None, :]
                    )
                if (
                    previous_state == 4
                    and current_state in (1, 2)
                    and both_hands_open
                ):
                    possible = np.zeros_like(possible)
                candidate_costs = (
                    costs[previous_state, reachable_previous][:, None]
                    + transition[reachable_previous]
                    + state_visual[None, :]
                    + (
                        0.5
                        if internal_phase[previous_state] != current_phase
                        else 0.0
                    )
                )
                candidate_costs = np.where(
                    possible[reachable_previous], candidate_costs, np.inf
                )
                best_previous_local = np.argmin(candidate_costs, axis=0)
                best_previous = reachable_previous[best_previous_local]
                best_values = candidate_costs[
                    best_previous_local, np.arange(len(current_poses))
                ]
                improve = best_values < next_costs[current_state]
                next_costs[current_state, improve] = best_values[improve]
                pointers[current_state, improve, 0] = previous_state
                pointers[current_state, improve, 1] = best_previous[improve]
        anchor_trace.append(
            {
                "anchor_index": anchor_index,
                "ordinal": values[anchor_index].ordinal,
                "source_frame_index": values[anchor_index].source_frame_index,
                "hand_observed_position": values[
                    anchor_index
                ].hand_observed_position.tolist(),
                "hand_opening_speed_units_s": hand_opening_speed.tolist(),
                "continuous_hand_active": continuous_hand_active.tolist(),
                "opening_transition": opening_transition,
                "initial_orientation_eligible_candidates": int(
                    np.count_nonzero(initial_orientation_possible)
                ),
                "final_orientation_eligible_candidates": int(
                    np.count_nonzero(final_orientation_possible)
                ),
                "visual_eligible_candidates": int(np.count_nonzero(np.isfinite(visual))),
                "static_visual_eligible_candidates": int(
                    np.count_nonzero(np.isfinite(static_visual))
                ),
                "physically_possible_pairs": int(np.count_nonzero(physically_possible)),
                "static_possible_pairs": int(np.count_nonzero(static_possible)),
                "attachment_initialization_possible_pairs": int(
                    np.count_nonzero(attachment_enter_possible)
                ),
                "entering_bimanual_attachment": entering_bimanual_attachment,
                "bimanual_grasp_continue_possible_pairs": int(
                    np.count_nonzero(bimanual_continue_possible)
                ),
                "single_hand_grasp_possible_pairs": int(
                    np.count_nonzero(single_hand_grasp_possible)
                ),
                "release_regrasp_enter_possible_pairs": int(
                    np.count_nonzero(physically_possible) if regrasp_enter else 0
                ),
                "reachable_by_internal_state": [
                    int(np.count_nonzero(np.isfinite(next_costs[state])))
                    for state in range(state_count)
                ],
                "reachable_candidate_indices_by_internal_state": [
                    sorted(
                        {
                            int(expanded[anchor_index][3][expanded_index])
                            for expanded_index in np.flatnonzero(
                                np.isfinite(next_costs[state])
                            )
                        }
                    )
                    for state in range(state_count)
                ],
            }
        )
        if not np.isfinite(next_costs).any():
            previous_reachable = np.isfinite(costs)
            current_eligible = np.isfinite(visual)
            reachable_transition_pairs: dict[str, int] = {}
            for current_state, previous_states in enumerate(_ALLOWED_PREVIOUS_STATES):
                count = 0
                for previous_state in previous_states:
                    if current_state == 1:
                        possible = (
                            single_hand_grasp_possible
                            if any_continuous_hand_active
                            else static_possible
                        )
                    elif current_state == 2:
                        possible = attachment_enter_possible
                    elif current_state == 3:
                        possible = bimanual_continue_possible
                    elif current_state == 4:
                        possible = release_possible
                    elif current_state == 5:
                        possible = (
                            physically_possible
                            & both_hands_open
                            & final_orientation_possible[None, :]
                        )
                    elif current_state == 6:
                        possible = (
                            static_possible
                            & both_hands_open
                            & final_orientation_possible[None, :]
                        )
                    else:
                        possible = (
                            static_possible
                            & initial_orientation_possible[None, :]
                        )
                    if (
                        previous_state == 4
                        and current_state in (1, 2)
                        and both_hands_open
                    ):
                        possible = np.zeros_like(possible)
                    count += int(
                        np.count_nonzero(
                            possible
                            & previous_reachable[previous_state, :, None]
                            & current_eligible[None, :]
                        )
                    )
                reachable_transition_pairs[str(current_state)] = count
            diagnostics = {
                "anchor_index": anchor_index,
                "ordinal": values[anchor_index].ordinal,
                "source_frame_index": values[anchor_index].source_frame_index,
                "previous_source_frame_index": values[anchor_index - 1].source_frame_index,
                "elapsed_s": elapsed_s,
                "current_hand_observed_position": values[
                    anchor_index
                ].hand_observed_position.tolist(),
                "hand_opening_speed_units_s": hand_opening_speed.tolist(),
                "continuous_hand_active": continuous_hand_active.tolist(),
                "opening_transition": opening_transition,
                "initial_orientation_eligible_candidates": int(
                    np.count_nonzero(initial_orientation_possible)
                ),
                "final_orientation_eligible_candidates": int(
                    np.count_nonzero(final_orientation_possible)
                ),
                "previous_reachable_by_internal_state": [
                    int(np.count_nonzero(previous_reachable[state]))
                    for state in range(state_count)
                ],
                "current_visual_eligible_candidates": int(
                    np.count_nonzero(current_eligible)
                ),
                "expanded_current_candidates": len(current_poses),
                "physically_possible_pairs": int(np.count_nonzero(physically_possible)),
                "static_possible_pairs": int(np.count_nonzero(static_possible)),
                "attachment_initialization_possible_pairs": int(
                    np.count_nonzero(attachment_enter_possible)
                ),
                "entering_bimanual_attachment": entering_bimanual_attachment,
                "bimanual_grasp_continue_possible_pairs": int(
                    np.count_nonzero(bimanual_continue_possible)
                ),
                "single_hand_grasp_possible_pairs": int(
                    np.count_nonzero(single_hand_grasp_possible)
                ),
                "release_regrasp_enter_possible_pairs": int(
                    np.count_nonzero(physically_possible) if regrasp_enter else 0
                ),
                "current_both_hands_active": current_both_hands_active,
                "continuous_both_hands_active": continuous_both_hands_active,
                "both_hands_open": both_hands_open,
                "reachable_transition_pairs_by_internal_state": reachable_transition_pairs,
                "anchor_trace": anchor_trace,
            }
            raise TemporalSelectionError(
                f"no physically coherent temporal path reaches anchor {anchor_index}",
                diagnostics,
            )
        costs = next_costs
        back_pointers.append(pointers)

    state = final_state
    candidate = int(np.argmin(costs[state]))
    if not np.isfinite(costs[state, candidate]):
        raise TemporalSelectionError(
            "no temporal path reaches the required final-static phase",
            {
                "anchor_index": len(values) - 1,
                "ordinal": values[-1].ordinal,
                "source_frame_index": values[-1].source_frame_index,
                "reachable_by_internal_state": [
                    int(np.count_nonzero(np.isfinite(costs[index])))
                    for index in range(state_count)
                ],
                "anchor_trace": anchor_trace,
            },
        )
    internal_states = np.empty(len(values), dtype=np.int64)
    expanded_indices = np.empty(len(values), dtype=np.int64)
    internal_states[-1] = state
    expanded_indices[-1] = candidate
    for anchor_index in range(len(values) - 1, 0, -1):
        state, candidate = back_pointers[anchor_index - 1][state, candidate]
        if state < 0 or candidate < 0:
            raise RuntimeError("temporal-selection back pointer is incomplete")
        internal_states[anchor_index - 1] = state
        expanded_indices[anchor_index - 1] = candidate
    states = internal_phase[internal_states]

    selected_poses = np.stack(
        [expanded[index][0][expanded_indices[index]] for index in range(len(values))]
    )
    transition_translation = np.empty(len(values) - 1, dtype=np.float64)
    transition_rotation = np.empty(len(values) - 1, dtype=np.float64)
    for index in range(1, len(values)):
        if states[index] == 2:
            residuals = []
            for side in range(2):
                previous_relative = _table_from_eef(
                    selected_poses[index - 1 : index], values[index - 1].eef_poses_root[side]
                )[0]
                current_relative = _table_from_eef(
                    selected_poses[index : index + 1], values[index].eef_poses_root[side]
                )[0]
                residuals.append(
                    _pair_errors(previous_relative[None], current_relative[None])
                )
            transition_translation[index - 1] = sum(float(value[0][0, 0]) for value in residuals)
            transition_rotation[index - 1] = sum(float(value[1][0, 0]) for value in residuals)
        elif states[index] == 3 and np.any(
            values[index].hand_observed_position
            <= parameters.grasp_observed_position_max
        ):
            active_residuals = []
            for side in range(2):
                if not (
                    values[index - 1].hand_observed_position[side]
                    <= parameters.grasp_observed_position_max
                    and values[index].hand_observed_position[side]
                    <= parameters.grasp_observed_position_max
                ):
                    continue
                previous_relative = _table_from_eef(
                    selected_poses[index - 1 : index], values[index - 1].eef_poses_root[side]
                )[0]
                current_relative = _table_from_eef(
                    selected_poses[index : index + 1], values[index].eef_poses_root[side]
                )[0]
                active_residuals.append(
                    _pair_errors(previous_relative[None], current_relative[None])
                )
                if active_residuals:
                    transition_translation[index - 1] = min(
                        float(value[0][0, 0]) for value in active_residuals
                    )
                    transition_rotation[index - 1] = min(
                        float(value[1][0, 0]) for value in active_residuals
                    )
                else:
                    translation, rotation = _pair_errors(
                        selected_poses[index - 1 : index],
                        selected_poses[index : index + 1],
                    )
                    transition_translation[index - 1] = float(translation[0, 0])
                    transition_rotation[index - 1] = float(rotation[0, 0])
        else:
            translation, rotation = _pair_errors(
                selected_poses[index - 1 : index], selected_poses[index : index + 1]
            )
            transition_translation[index - 1] = float(translation[0, 0])
            transition_rotation[index - 1] = float(rotation[0, 0])

    original_candidates = np.asarray(
        [expanded[index][3][expanded_indices[index]] for index in range(len(values))],
        dtype=np.int64,
    )
    symmetry_indices = np.asarray(
        [expanded[index][4][expanded_indices[index]] for index in range(len(values))],
        dtype=np.int64,
    )
    evidence_modes = []
    for index, candidate_index in enumerate(original_candidates):
        if states[index] in (0, 4):
            evidence_modes.append("static_rgbd_bidirectional")
            continue
        evidence = _candidate_visual_evidence(
            values[index],
            parameters,
            values[index].candidate_metrics[int(candidate_index)],
            require_bidirectional_consensus=True,
        )
        if evidence is None:
            raise RuntimeError("selected temporal candidate lacks visual evidence")
        evidence_modes.append(evidence[1])
    return TemporalSelectionResult(
        selected_candidate_indices=original_candidates,
        selected_symmetry_indices=symmetry_indices,
        phase_indices=states,
        selected_poses_root=selected_poses,
        transition_translation_m=transition_translation,
        transition_rotation_rad=transition_rotation,
        selected_evidence_modes=tuple(evidence_modes),
        total_cost=float(costs[5, expanded_indices[-1]]),
    )
