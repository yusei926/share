#!/usr/bin/env python3
"""Track the assembled white table and emit an audited 30 Hz root-frame pose trajectory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import (
    atomic_write_json,
    read_json_object,
    sha256_file,
)
from data.flip_table_data_augmentation.object_pose import (
    INPUT_SCHEMA_VERSION,
    TRACK_SCHEMA_VERSION,
)
from data.flip_table_data_augmentation.object_pose.camera_views import (
    POSE_VIEW_NAMES,
    PRIMARY_POSE_VIEW,
)
from data.flip_table_data_augmentation.object_pose.artifacts import (
    MANIFEST_SCHEMA_VERSION as SOURCE_RUNTIME_SCHEMA_VERSION,
)
from data.flip_table_data_augmentation.object_pose.geometry import (
    evaluate_rendered_alignment,
    fuse_bidirectional_poses,
    interpolate_pose_trajectory,
    pose_errors,
    project_to_rigid_transform,
    root_from_object_pose,
    table_symmetry_transforms,
    visible_rendered_mask,
)
from data.flip_table_data_augmentation.object_pose.segmentation import (
    SEGMENTATION_SCHEMA_VERSION,
)
from data.flip_table_data_augmentation.object_pose.temporal_selection import (
    PHASE_NAMES,
    TemporalAnchor,
    TemporalSelectionError,
    TemporalSelectionParameters,
    TemporalSelectionResult,
    audit_temporal_evidence_gaps,
    select_causally_constrained_poses,
    select_temporally_consistent_poses,
    temporal_static_visual_costs,
    temporal_visual_costs,
)
from data.flip_table_data_augmentation.scripts.verify_object_pose_runtime import (
    SCHEMA_VERSION as COMPILED_RUNTIME_SCHEMA_VERSION,
)


SCHEMA_VERSION = TRACK_SCHEMA_VERSION


@contextmanager
def _foundationpose_debug_output() -> object:
    """Keep FoundationPose's per-candidate debug arrays out of normal run logs."""

    if os.environ.get("FLIP_TABLE_FOUNDATIONPOSE_VERBOSE") == "1":
        yield
        return
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
        ):
            yield
    finally:
        logging.disable(previous_disable_level)


def _track_trace(event: str, **fields: object) -> None:
    """Emit bounded timing diagnostics for expensive offline track stages."""

    if os.environ.get("FLIP_TABLE_TRACK_TRACE") != "1":
        return
    print(
        "[track-trace] " + json.dumps({"event": event, **fields}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _track_rejection_reasons(
    *, bidirectional_pass: bool, rendered_alignment_pass: bool
) -> list[str]:
    reasons = []
    if not bidirectional_pass:
        reasons.append("bidirectional_pose_gate")
    if not rendered_alignment_pass:
        reasons.append("rendered_alignment_gate")
    return reasons


@dataclass(frozen=True)
class RegistrationGeometryMetrics:
    candidate_index: int
    source: str
    raw_rendered_pixels: int
    occluded_rendered_pixels: int
    rendered_pixels: int
    observed_mask_pixels: int
    mask_intersection_pixels: int
    depth_consistent_pixels: int
    raw_mask_precision: float
    raw_mask_explained_fraction: float
    mask_precision: float
    mask_explained_fraction: float
    depth_overlap_fraction: float
    depth_consistent_union_fraction: float
    median_absolute_depth_error_m: float | None
    auxiliary_mask_precisions: dict[str, float]
    auxiliary_mask_explained_fractions: dict[str, float]
    multiview_score: float
    evidence_kind: str = "primary_mask_rgbd"
    primary_stereo_consistent_fraction: float | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def rank_registration_depths(
    *,
    rendered_depths_m: np.ndarray,
    observed_depth_m: np.ndarray,
    observed_mask: np.ndarray,
    sources: tuple[str, ...],
    maximum_consistent_depth_error_m: float,
    auxiliary_rendered_depths_m: dict[str, np.ndarray] | None = None,
    auxiliary_observed_masks: dict[str, np.ndarray] | None = None,
    auxiliary_view_score_weight: float = 0.0,
    auxiliary_primary_support_saturation_fraction: float = 0.05,
) -> tuple[tuple[RegistrationGeometryMetrics, ...], int]:
    """Rank pose hypotheses by CAD silhouette and measured stereo depth."""

    rendered = np.asarray(rendered_depths_m, dtype=np.float32)
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    mask = np.asarray(observed_mask, dtype=bool)
    if (
        rendered.ndim != 3
        or observed.shape != rendered.shape[1:]
        or mask.shape != observed.shape
        or len(rendered) != len(sources)
        or len(rendered) == 0
        or not np.isfinite(rendered).all()
        or not np.isfinite(observed).all()
        or maximum_consistent_depth_error_m <= 0.0
        or not 0.0 <= auxiliary_view_score_weight <= 1.0
        or not 0.0 < auxiliary_primary_support_saturation_fraction <= 1.0
    ):
        raise ValueError("registration depth-ranking inputs are invalid")
    auxiliary_rendered = auxiliary_rendered_depths_m or {}
    auxiliary_masks = auxiliary_observed_masks or {}
    if set(auxiliary_rendered) != set(auxiliary_masks):
        raise ValueError("auxiliary rendered and observed view names differ")
    for view_name in auxiliary_rendered:
        rendered_values = np.asarray(auxiliary_rendered[view_name], dtype=np.float32)
        observed_values = np.asarray(auxiliary_masks[view_name], dtype=bool)
        if (
            rendered_values.shape != rendered.shape
            or observed_values.shape != observed.shape
            or not np.isfinite(rendered_values).all()
        ):
            raise ValueError(f"invalid auxiliary registration evidence: {view_name}")
        auxiliary_rendered[view_name] = rendered_values
        auxiliary_masks[view_name] = observed_values
    observed_pixels = int(mask.sum())
    if observed_pixels == 0:
        raise ValueError("registration mask is empty")
    valid_observed = observed > 0.0
    axes = (1, 2)
    raw_rendered = rendered > 0.0
    occluded = (
        raw_rendered
        & valid_observed[None]
        & (rendered > observed[None] + maximum_consistent_depth_error_m)
    )
    visible = raw_rendered & ~occluded
    intersection = visible & mask[None]
    depth_overlap = intersection & valid_observed[None]
    errors = np.abs(rendered - observed[None])

    raw_rendered_pixels = np.count_nonzero(raw_rendered, axis=axes)
    occluded_pixels = np.count_nonzero(occluded, axis=axes)
    raw_intersection_pixels = np.count_nonzero(raw_rendered & mask[None], axis=axes)
    rendered_pixels = np.count_nonzero(visible, axis=axes)
    intersection_pixels = np.count_nonzero(intersection, axis=axes)
    depth_overlap_pixels = np.count_nonzero(depth_overlap, axis=axes)
    consistent_pixels = np.count_nonzero(
        depth_overlap & (errors <= maximum_consistent_depth_error_m), axis=axes
    )
    union_pixels = np.count_nonzero(visible | mask[None], axis=axes)
    median_errors = []
    for index in range(len(rendered)):
        candidate_errors = errors[index][depth_overlap[index]]
        median_errors.append(
            float(np.median(candidate_errors)) if candidate_errors.size else None
        )
    del raw_rendered, occluded, visible, intersection, depth_overlap, errors

    auxiliary_precision_by_view = {}
    auxiliary_explained_by_view = {}
    for view_name in auxiliary_rendered:
        auxiliary_mask = auxiliary_masks[view_name]
        auxiliary_pixels = int(np.count_nonzero(auxiliary_mask))
        candidate_masks = auxiliary_rendered[view_name] > 0.0
        candidate_pixels = np.count_nonzero(candidate_masks, axis=axes)
        auxiliary_intersections = np.count_nonzero(
            candidate_masks & auxiliary_mask[None], axis=axes
        )
        auxiliary_precision_by_view[view_name] = np.divide(
            auxiliary_intersections,
            candidate_pixels,
            out=np.zeros(len(rendered), dtype=np.float64),
            where=candidate_pixels > 0,
        )
        auxiliary_explained_by_view[view_name] = (
            auxiliary_intersections / auxiliary_pixels
            if auxiliary_pixels
            else np.zeros(len(rendered), dtype=np.float64)
        )

    metrics = []
    for index, source in enumerate(sources):
        auxiliary_precision = {
            view_name: float(values[index])
            for view_name, values in auxiliary_precision_by_view.items()
        }
        auxiliary_explained = {
            view_name: float(values[index])
            for view_name, values in auxiliary_explained_by_view.items()
        }
        depth_consistent_union_fraction = (
            consistent_pixels[index] / union_pixels[index]
            if union_pixels[index]
            else 0.0
        )
        auxiliary_mean = (
            float(np.mean(tuple(auxiliary_explained.values())))
            if auxiliary_explained
            else 0.0
        )
        primary_support = min(
            1.0,
            depth_consistent_union_fraction
            / auxiliary_primary_support_saturation_fraction,
        )
        metrics.append(
            RegistrationGeometryMetrics(
                candidate_index=index,
                source=source,
                raw_rendered_pixels=int(raw_rendered_pixels[index]),
                occluded_rendered_pixels=int(occluded_pixels[index]),
                rendered_pixels=int(rendered_pixels[index]),
                observed_mask_pixels=observed_pixels,
                mask_intersection_pixels=int(intersection_pixels[index]),
                depth_consistent_pixels=int(consistent_pixels[index]),
                raw_mask_precision=(
                    raw_intersection_pixels[index] / raw_rendered_pixels[index]
                    if raw_rendered_pixels[index]
                    else 0.0
                ),
                raw_mask_explained_fraction=(
                    raw_intersection_pixels[index] / observed_pixels
                ),
                mask_precision=(
                    intersection_pixels[index] / rendered_pixels[index]
                    if rendered_pixels[index]
                    else 0.0
                ),
                mask_explained_fraction=intersection_pixels[index] / observed_pixels,
                depth_overlap_fraction=(
                    depth_overlap_pixels[index] / intersection_pixels[index]
                    if intersection_pixels[index]
                    else 0.0
                ),
                depth_consistent_union_fraction=depth_consistent_union_fraction,
                median_absolute_depth_error_m=median_errors[index],
                auxiliary_mask_precisions=auxiliary_precision,
                auxiliary_mask_explained_fractions=auxiliary_explained,
                multiview_score=(
                    depth_consistent_union_fraction
                    + auxiliary_view_score_weight * auxiliary_mean * primary_support
                ),
            )
        )
    selected = max(
        range(len(metrics)),
        key=lambda index: (
            metrics[index].multiview_score,
            metrics[index].depth_consistent_union_fraction,
            metrics[index].mask_explained_fraction,
            metrics[index].mask_precision,
            -index,
        ),
    )
    return tuple(metrics), selected


def evaluate_unsegmented_multiview_depths(
    *,
    rendered_depth_m: np.ndarray,
    observed_depth_m: np.ndarray,
    stereo_consistency_mask: np.ndarray,
    auxiliary_rendered_depths_m: dict[str, np.ndarray],
    auxiliary_observed_masks: dict[str, np.ndarray],
    source: str,
    maximum_consistent_depth_error_m: float,
    auxiliary_view_score_weight: float,
) -> RegistrationGeometryMetrics:
    """Audit a tracked pose when head segmentation is absent.

    This does not synthesize a head mask.  It measures CAD-vs-head RGB-D
    support and requires independent masks from both wrist views downstream.
    """

    rendered = np.asarray(rendered_depth_m, dtype=np.float32)
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    consistency = np.asarray(stereo_consistency_mask, dtype=bool)
    required_views = set(POSE_VIEW_NAMES[1:])
    if (
        rendered.ndim != 2
        or observed.shape != rendered.shape
        or consistency.shape != rendered.shape
        or not np.isfinite(rendered).all()
        or not np.isfinite(observed).all()
        or set(auxiliary_rendered_depths_m) != required_views
        or set(auxiliary_observed_masks) != required_views
        or maximum_consistent_depth_error_m <= 0.0
        or not 0.0 <= auxiliary_view_score_weight <= 1.0
    ):
        raise ValueError("unsegmented multi-view evidence is invalid")

    visible, occluded = visible_rendered_mask(
        observed,
        rendered,
        maximum_occlusion_depth_error_m=maximum_consistent_depth_error_m,
    )
    raw_rendered = visible | occluded
    raw_rendered_pixels = int(np.count_nonzero(raw_rendered))
    rendered_pixels = int(np.count_nonzero(visible))
    valid_overlap = visible & (observed > 0.0)
    overlap_pixels = int(np.count_nonzero(valid_overlap))
    errors = np.abs(rendered[valid_overlap] - observed[valid_overlap])
    consistent_pixels = int(
        np.count_nonzero(errors <= maximum_consistent_depth_error_m)
    )
    auxiliary_precision = {}
    auxiliary_explained = {}
    for view_name in POSE_VIEW_NAMES[1:]:
        candidate_depth = np.asarray(
            auxiliary_rendered_depths_m[view_name], dtype=np.float32
        )
        observed_mask = np.asarray(auxiliary_observed_masks[view_name], dtype=bool)
        if (
            candidate_depth.shape != rendered.shape
            or observed_mask.shape != rendered.shape
            or not np.isfinite(candidate_depth).all()
        ):
            raise ValueError(f"invalid unsegmented auxiliary view: {view_name}")
        candidate_mask = candidate_depth > 0.0
        candidate_pixels = int(np.count_nonzero(candidate_mask))
        observed_pixels = int(np.count_nonzero(observed_mask))
        intersection = int(np.count_nonzero(candidate_mask & observed_mask))
        auxiliary_precision[view_name] = (
            intersection / candidate_pixels if candidate_pixels else 0.0
        )
        auxiliary_explained[view_name] = (
            intersection / observed_pixels if observed_pixels else 0.0
        )
    depth_support = (
        consistent_pixels / raw_rendered_pixels if raw_rendered_pixels else 0.0
    )
    auxiliary_mean = float(np.mean(tuple(auxiliary_explained.values())))
    stereo_fraction = (
        float(np.count_nonzero(consistency & raw_rendered) / raw_rendered_pixels)
        if raw_rendered_pixels
        else 0.0
    )
    return RegistrationGeometryMetrics(
        candidate_index=0,
        source=source,
        raw_rendered_pixels=raw_rendered_pixels,
        occluded_rendered_pixels=int(np.count_nonzero(occluded)),
        rendered_pixels=rendered_pixels,
        observed_mask_pixels=0,
        mask_intersection_pixels=0,
        depth_consistent_pixels=consistent_pixels,
        raw_mask_precision=0.0,
        raw_mask_explained_fraction=0.0,
        mask_precision=0.0,
        mask_explained_fraction=0.0,
        depth_overlap_fraction=(
            overlap_pixels / rendered_pixels if rendered_pixels else 0.0
        ),
        depth_consistent_union_fraction=depth_support,
        median_absolute_depth_error_m=(
            float(np.median(errors)) if errors.size else None
        ),
        auxiliary_mask_precisions=auxiliary_precision,
        auxiliary_mask_explained_fractions=auxiliary_explained,
        multiview_score=(
            depth_support + auxiliary_view_score_weight * auxiliary_mean
        ),
        evidence_kind="unsegmented_multiview_rgbd",
        primary_stereo_consistent_fraction=stereo_fraction,
    )


def propagation_candidate_indices(
    metrics: tuple[RegistrationGeometryMetrics, ...],
    sources: tuple[str, ...],
    *,
    selected_candidate_index: int,
    limit: int,
    priority_indices: tuple[int, ...] = (),
    required_indices: tuple[int, ...] = (),
) -> list[int]:
    """Keep required physical hypotheses before source-diverse visual ranks."""

    if (
        len(metrics) != len(sources)
        or not metrics
        or selected_candidate_index not in range(len(metrics))
        or limit <= 0
        or any(index not in range(len(metrics)) for index in priority_indices)
        or any(index not in range(len(metrics)) for index in required_indices)
        or len(set(required_indices)) != len(required_indices)
        or len(required_indices) > limit
    ):
        raise ValueError("propagation candidate beam inputs are invalid")
    multiview = sorted(
        range(len(metrics)),
        key=lambda index: (
            metrics[index].multiview_score,
            metrics[index].depth_consistent_union_fraction,
            metrics[index].mask_explained_fraction,
            metrics[index].mask_precision,
            -index,
        ),
        reverse=True,
    )
    primary = sorted(
        range(len(metrics)),
        key=lambda index: (
            metrics[index].depth_consistent_union_fraction,
            metrics[index].mask_explained_fraction,
            metrics[index].mask_precision,
            -index,
        ),
        reverse=True,
    )
    ordered = [*required_indices, selected_candidate_index]
    for source in dict.fromkeys(sources):
        members = (index for index, value in enumerate(sources) if value == source)
        ordered.append(
            max(
                members,
                key=lambda index: (
                    metrics[index].multiview_score,
                    metrics[index].depth_consistent_union_fraction,
                    -index,
                ),
            )
        )
    ordered.extend(priority_indices)
    for pair in zip(multiview, primary, strict=True):
        ordered.extend(pair)
    unique = []
    seen = set()
    for index in ordered:
        if index in seen:
            continue
        unique.append(index)
        seen.add(index)
        if len(unique) == min(limit, len(metrics)):
            break
    return unique


def contact_lineage_priority_indices(
    sources: tuple[str, ...],
    lineage_ids: tuple[int, ...],
    hand_observed_position: np.ndarray,
    *,
    grasp_observed_position_max: float,
) -> tuple[int, ...]:
    """Retain one physically constrained child for every incoming pose lineage."""

    hands = np.asarray(hand_observed_position, dtype=np.float64)
    if (
        len(sources) != len(lineage_ids)
        or not sources
        or hands.shape != (2,)
        or not np.isfinite(hands).all()
        or not math.isfinite(grasp_observed_position_max)
        or grasp_observed_position_max <= 0.0
        or any(not isinstance(value, int) or value < 0 for value in lineage_ids)
    ):
        raise ValueError("contact-lineage priority inputs are invalid")
    active = hands <= grasp_observed_position_max
    if not np.any(active):
        return ()

    directions = tuple(
        direction
        for direction in ("forward", "reverse")
        if any(source.endswith(f"_{direction}") for source in sources)
    )
    if len(directions) != 1:
        raise ValueError("contact-lineage sources must have one propagation direction")
    direction = directions[0]
    preferred = []
    if np.all(active):
        preferred.append(f"bimanual_eef_carry_{direction}")
    if active[0]:
        preferred.append(f"left_eef_carry_{direction}")
    if active[1]:
        preferred.append(f"right_eef_carry_{direction}")

    expected_lineages = set(lineage_ids)
    for preferred_source in preferred:
        by_lineage = {
            lineage: index
            for index, (source, lineage) in enumerate(
                zip(sources, lineage_ids, strict=True)
            )
            if source == preferred_source
        }
        if set(by_lineage) == expected_lineages:
            return tuple(by_lineage[lineage] for lineage in sorted(by_lineage))
    return ()


def lineage_preserving_candidate_indices(
    metrics: tuple[RegistrationGeometryMetrics, ...],
    sources: tuple[str, ...],
    lineage_ids: tuple[int, ...],
    *,
    selected_candidate_index: int,
    limit: int,
    priority_indices: tuple[int, ...] = (),
) -> list[int]:
    """Retain the best child of every incoming physical pose lineage."""

    if (
        not metrics
        or len(sources) != len(metrics)
        or len(lineage_ids) != len(metrics)
        or selected_candidate_index not in range(len(metrics))
        or limit <= 0
        or any(index not in range(len(metrics)) for index in priority_indices)
        or any(not isinstance(value, int) or value < 0 for value in lineage_ids)
        or len(set(lineage_ids)) > limit
    ):
        raise ValueError("pose-lineage beam inputs are invalid")
    multiview = sorted(
        range(len(metrics)),
        key=lambda index: (
            metrics[index].multiview_score,
            metrics[index].depth_consistent_union_fraction,
            metrics[index].mask_explained_fraction,
            metrics[index].mask_precision,
            -index,
        ),
        reverse=True,
    )
    primary = sorted(
        range(len(metrics)),
        key=lambda index: (
            metrics[index].depth_consistent_union_fraction,
            metrics[index].mask_explained_fraction,
            metrics[index].mask_precision,
            -index,
        ),
        reverse=True,
    )
    rank_order = [selected_candidate_index, *priority_indices]
    for pair in zip(multiview, primary, strict=True):
        rank_order.extend(pair)

    winners = []
    retained_lineages = set()
    for index in rank_order:
        lineage = lineage_ids[index]
        if lineage in retained_lineages:
            continue
        winners.append(index)
        retained_lineages.add(lineage)
    if retained_lineages != set(lineage_ids):
        raise RuntimeError("pose-lineage ranking did not cover every incoming lineage")

    fill_order = propagation_candidate_indices(
        metrics,
        sources,
        selected_candidate_index=selected_candidate_index,
        limit=len(metrics),
        priority_indices=priority_indices,
    )
    selected = list(winners)
    target_count = min(limit, len(metrics))
    if len(selected) == target_count:
        return selected
    selected_set = set(selected)
    for index in fill_order:
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        if len(selected) == target_count:
            break
    return selected


def _temporal_solver_candidate_indices(evidence: dict[str, object]) -> tuple[int, ...]:
    """Return only hypotheses retained by the forward propagation beam.

    Dense RGB-D propagation deliberately keeps a bounded set of pose lineages.
    Letting the temporal solver select a visually plausible candidate outside
    that set creates a dead end at the next active-hand anchor: no child pose
    was ever generated for it.  Restricting the solver to the retained beam
    makes every selected contact hypothesis eligible for the next physical
    transition without changing a visual or motion threshold.
    """

    poses = evidence.get("temporal_candidate_poses_root")
    sources = evidence.get("temporal_candidate_sources")
    indices = evidence.get("propagation_candidate_indices")
    if not isinstance(poses, list) or not isinstance(sources, list):
        raise ValueError("temporal solver evidence is incomplete")
    if len(poses) == 0 or len(sources) != len(poses):
        raise ValueError("temporal solver evidence has invalid candidate lengths")
    if indices is None:
        return tuple(range(len(poses)))
    if (
        not isinstance(indices, list)
        or not indices
        or any(not isinstance(index, int) or index not in range(len(poses)) for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("temporal solver propagation beam is invalid")
    return tuple(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-root-from-table",
        type=Path,
        help=(
            "accepted source_cad_alignment.json used only as an initial CAD "
            "hypothesis at source frame zero; it remains subject to RGB-D, "
            "bidirectional, and temporal gates"
        ),
    )
    parser.add_argument(
        "--dense-propagation-beam-size",
        type=int,
        help=(
            "tracking-only dense pose-lineage beam override; it does not "
            "change prepared RGB-D, masks, or the pinned runtime contract"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_initial_root_from_table(
    path: Path | None, *, episode_index: int
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    """Load an auditable source-CAD seed without making it a tracked result.

    The fixed-scene fitter works from recorded RGB, calibrated stereo and
    robot FK.  Its output may initialize the first registration candidate, but
    never bypasses the registration selector or any temporal gate below.
    """

    if path is None:
        return None, None
    document = read_json_object(path.expanduser().resolve())
    if document.get("schema_version") != "team_ramen_flip_table_source_cad_alignment/v1":
        raise ValueError("initial table seed has an unsupported schema")
    if document.get("accepted_for_fixed_scene_proposal") is not True:
        raise ValueError("initial table seed did not pass its fixed-scene gate")
    source = document.get("source")
    if not isinstance(source, dict) or int(source.get("episode_index", -1)) != episode_index:
        raise ValueError("initial table seed belongs to a different source episode")
    method = document.get("method")
    if not isinstance(method, dict) or method.get("requires_simulator_ground_truth") is not False:
        raise ValueError("initial table seed must be derived without simulator ground truth")
    root_from_table = _as_transform(
        document.get("fixed_scene_root_from_table"), "initial fixed_scene_root_from_table"
    )
    return root_from_table, {
        "path": str(path.expanduser().resolve()),
        "schema_version": document["schema_version"],
        "episode_index": episode_index,
        "accepted_for_fixed_scene_proposal": True,
        "requires_simulator_ground_truth": False,
        "use": "initial_registration_candidate_only",
    }


def _temporal_parameters(pose_config) -> TemporalSelectionParameters:
    return TemporalSelectionParameters(
        minimum_mask_precision=pose_config.temporal_candidate_min_mask_precision,
        minimum_mask_explained_fraction=(
            pose_config.temporal_candidate_min_mask_explained_fraction
        ),
        maximum_candidate_depth_error_m=(
            pose_config.temporal_candidate_max_depth_error_m
        ),
        static_translation_scale_m=pose_config.temporal_static_translation_scale_m,
        static_rotation_scale_rad=pose_config.temporal_static_rotation_scale_rad,
        maximum_static_translation_step_m=(
            pose_config.temporal_max_static_translation_step_m
        ),
        maximum_static_rotation_step_rad=(
            pose_config.temporal_max_static_rotation_step_rad
        ),
        grasp_translation_scale_m=pose_config.temporal_grasp_translation_scale_m,
        grasp_rotation_scale_rad=pose_config.temporal_grasp_rotation_scale_rad,
        maximum_table_speed_m_s=pose_config.temporal_max_table_speed_m_s,
        maximum_table_angular_speed_rad_s=(
            pose_config.temporal_max_table_angular_speed_rad_s
        ),
        maximum_grasp_relative_translation_step_m=(
            pose_config.temporal_max_grasp_relative_translation_step_m
        ),
        maximum_grasp_relative_rotation_step_rad=(
            pose_config.temporal_max_grasp_relative_rotation_step_rad
        ),
        grasp_observed_position_max=(
            pose_config.temporal_grasp_observed_position_max
        ),
        static_minimum_mask_precision=(
            pose_config.temporal_candidate_min_mask_precision
        ),
        static_minimum_mask_explained_fraction=(
            pose_config.min_rendered_mask_explained_fraction
        ),
        static_minimum_depth_overlap_fraction=(
            pose_config.min_rendered_depth_overlap_fraction
        ),
        static_maximum_depth_error_m=(
            pose_config.max_rendered_depth_median_abs_error_m
        ),
        blurred_frame_laplacian_variance_max=(
            pose_config.temporal_blurred_frame_laplacian_variance_max
        ),
        carry_minimum_mask_precision=pose_config.temporal_carry_min_mask_precision,
        carry_minimum_mask_explained_fraction=(
            pose_config.temporal_carry_min_mask_explained_fraction
        ),
        carry_minimum_auxiliary_explained_fraction=(
            pose_config.temporal_carry_min_auxiliary_explained_fraction
        ),
        carry_minimum_multiview_score=(
            pose_config.temporal_carry_min_multiview_score
        ),
        carry_visual_penalty=pose_config.temporal_carry_visual_penalty,
        minimum_stereo_consistent_fraction=(
            pose_config.temporal_min_stereo_consistent_fraction
        ),
        # RoboFinals flip_table starts upside-down and ends upright. Robot-root
        # Z is gravity-aligned in the audited source contract.
        minimum_endpoint_normal_vertical_component=0.7,
    )


def _as_transform(value: object, label: str) -> np.ndarray:
    result, _ = project_to_rigid_transform(
        np.asarray(value, dtype=np.float64).reshape(4, 4), label
    )
    return result


def _current_eef_poses_from_record(record: dict[str, object]) -> np.ndarray:
    values = np.asarray(record.get("eef_current_root_from_fk"), dtype=np.float64)
    if values.shape != (2, 16) or not np.isfinite(values).all():
        raise ValueError(
            "prepared frame lacks two finite current EEF poses derived from robot_q_current"
        )
    return np.stack(
        [
            _as_transform(values[side].reshape(4, 4), f"current EEF {side}")
            for side in range(2)
        ]
    )


def _propagated_root_candidates(
    previous_candidates_root: np.ndarray,
    previous_eef_poses_root: np.ndarray,
    current_eef_poses_root: np.ndarray,
    active_hand_sides: np.ndarray,
    *,
    direction: str = "forward",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Propagate a pose beam with static, single-EEF, and bimanual rigid models."""

    candidates = np.asarray(previous_candidates_root, dtype=np.float64)
    previous_eef = np.asarray(previous_eef_poses_root, dtype=np.float64)
    current_eef = np.asarray(current_eef_poses_root, dtype=np.float64)
    active_sides = np.asarray(active_hand_sides, dtype=bool)
    if candidates.ndim != 3 or candidates.shape[1:] != (4, 4) or len(candidates) == 0:
        raise ValueError("previous temporal candidates must be non-empty [N,4,4]")
    if direction not in {"forward", "reverse"}:
        raise ValueError("candidate propagation direction must be forward or reverse")
    if previous_eef.shape != (2, 4, 4) or current_eef.shape != (2, 4, 4):
        raise ValueError("EEF propagation requires previous/current [2,4,4] poses")
    if active_sides.shape != (2,):
        raise ValueError("EEF propagation requires two observed hand-contact states")
    if not all(np.isfinite(value).all() for value in (candidates, previous_eef, current_eef)):
        raise ValueError("temporal propagation inputs must be finite")
    propagated = [candidates]
    labels = [f"static_carry_{direction}"] * len(candidates)
    eef_propagated = []
    for side, name in enumerate(("left", "right")):
        root_delta = current_eef[side] @ np.linalg.inv(previous_eef[side])
        side_propagated = root_delta[None] @ candidates
        eef_propagated.append(side_propagated)
        if not active_sides[side]:
            continue
        propagated.append(side_propagated)
        labels.extend([f"{name}_eef_carry_{direction}"] * len(candidates))
    if np.all(active_sides):
        bimanual = np.repeat(
            np.eye(4, dtype=np.float64)[None], len(candidates), axis=0
        )
        bimanual[:, :3, 3] = 0.5 * (
            eef_propagated[0][:, :3, 3] + eef_propagated[1][:, :3, 3]
        )
        for index in range(len(candidates)):
            bimanual[index, :3, :3] = Rotation.from_matrix(
                np.stack(
                    (
                        eef_propagated[0][index, :3, :3],
                        eef_propagated[1][index, :3, :3],
                    )
                )
            ).mean().as_matrix()
        propagated.append(bimanual)
        labels.extend([f"bimanual_eef_carry_{direction}"] * len(candidates))
    return np.concatenate(propagated), tuple(labels)


def _bidirectional_consensus_records(
    *,
    forward_anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
    maximum_translation_error_m: float,
    maximum_rotation_error_rad: float,
    backward_anchor: TemporalAnchor | None = None,
    backward_reference_pose_root: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Match every forward hypothesis to independent reverse-direction evidence."""

    if (backward_anchor is None) == (backward_reference_pose_root is None):
        raise ValueError("exactly one reverse-consensus source is required")
    if maximum_translation_error_m <= 0.0 or maximum_rotation_error_rad <= 0.0:
        raise ValueError("bidirectional consensus thresholds must be positive")
    forward = np.asarray(forward_anchor.candidate_poses_root, dtype=np.float64)
    if backward_anchor is not None:
        try:
            reverse_costs = temporal_visual_costs(
                backward_anchor,
                parameters,
                require_bidirectional_consensus=False,
            )
        except ValueError:
            return [
                {
                    "passes_gate": False,
                    "translation_error_m": None,
                    "rotation_error_rad": None,
                    "maximum_translation_error_m": maximum_translation_error_m,
                    "maximum_rotation_error_rad": maximum_rotation_error_rad,
                    "validation_mode": "reverse_registration_no_visual_candidate",
                    "reverse_candidate_index": None,
                    "reverse_candidate_source": None,
                    "reverse_symmetry_index": None,
                }
                for _ in forward
            ]
        reverse_indices = np.flatnonzero(np.isfinite(reverse_costs))
        if len(reverse_indices) == 0:
            raise ValueError("reverse registration has no visual candidate")
        backward = np.asarray(
            backward_anchor.candidate_poses_root[reverse_indices], dtype=np.float64
        )
        reverse_metrics = tuple(
            backward_anchor.candidate_metrics[int(index)] for index in reverse_indices
        )
        validation_mode = "reverse_registration_candidate"
    else:
        backward = np.asarray(backward_reference_pose_root, dtype=np.float64).reshape(
            1, 4, 4
        )
        reverse_indices = np.asarray((-1,), dtype=np.int64)
        reverse_metrics = ({"source": "seeded_reverse_validation"},)
        validation_mode = "terminal_seeded_reverse_validation"
    if (
        forward.ndim != 3
        or forward.shape[1:] != (4, 4)
        or backward.ndim != 3
        or backward.shape[1:] != (4, 4)
        or len(forward) == 0
        or not np.isfinite(forward).all()
        or not np.isfinite(backward).all()
    ):
        raise ValueError("bidirectional consensus poses are invalid")

    symmetries = np.stack(table_symmetry_transforms())
    reverse_equivalents = (backward[:, None] @ symmetries[None]).reshape(-1, 4, 4)
    reverse_candidate_indices = np.repeat(reverse_indices, len(symmetries))
    reverse_symmetry_indices = np.tile(
        np.arange(len(symmetries), dtype=np.int64), len(backward)
    )
    translation = np.linalg.norm(
        forward[:, None, :3, 3] - reverse_equivalents[None, :, :3, 3], axis=2
    )
    rotation_dot = np.einsum(
        "fij,bij->fb",
        forward[:, :3, :3],
        reverse_equivalents[:, :3, :3],
    )
    rotation = np.arccos(np.clip((rotation_dot - 1.0) * 0.5, -1.0, 1.0))
    normalized_error = (
        translation / maximum_translation_error_m
        + rotation / maximum_rotation_error_rad
    )
    within_gate = (
        (translation <= maximum_translation_error_m)
        & (rotation <= maximum_rotation_error_rad)
    )

    records = []
    for forward_index in range(len(forward)):
        eligible = np.flatnonzero(within_gate[forward_index])
        candidate_pool = eligible if len(eligible) else np.arange(len(reverse_equivalents))
        selected = int(
            candidate_pool[
                np.argmin(normalized_error[forward_index, candidate_pool])
            ]
        )
        reverse_original_index = int(reverse_candidate_indices[selected])
        reverse_metric_index = (
            int(np.flatnonzero(reverse_indices == reverse_original_index)[0])
            if reverse_original_index >= 0
            else 0
        )
        records.append(
            {
                "passes_gate": bool(within_gate[forward_index, selected]),
                "translation_error_m": float(translation[forward_index, selected]),
                "rotation_error_rad": float(rotation[forward_index, selected]),
                "maximum_translation_error_m": maximum_translation_error_m,
                "maximum_rotation_error_rad": maximum_rotation_error_rad,
                "validation_mode": validation_mode,
                "reverse_candidate_index": (
                    reverse_original_index if reverse_original_index >= 0 else None
                ),
                "reverse_candidate_source": str(
                    reverse_metrics[reverse_metric_index].get(
                        "source", "global_registration"
                    )
                ),
                "reverse_symmetry_index": int(reverse_symmetry_indices[selected]),
            }
        )
    return records


def _set_estimator_original_pose(estimator, camera_from_object: np.ndarray) -> None:
    centered_from_original = estimator.get_tf_to_centered_mesh()
    if hasattr(centered_from_original, "detach"):
        centered_numpy = centered_from_original.detach().cpu().numpy()
        estimator.pose_last = centered_from_original.new_tensor(
            camera_from_object @ np.linalg.inv(centered_numpy)
        )
    else:
        estimator.pose_last = camera_from_object @ np.linalg.inv(
            np.asarray(centered_from_original, dtype=np.float64)
        )


def _load_frame(
    input_root: Path, record: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = record.get("views")
    if not isinstance(views, dict) or not isinstance(views.get(PRIMARY_POSE_VIEW), dict):
        raise ValueError("prepared frame lacks the primary pose view")
    view = views[PRIMARY_POSE_VIEW]
    rgb_bgr = cv2.imread(str(input_root / str(view["rgb"])), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(input_root / str(view["depth"])), cv2.IMREAD_UNCHANGED)
    consistency_path = input_root / str(view["stereo_consistency"])
    consistency_image = cv2.imread(str(consistency_path), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or rgb_bgr.shape != (480, 640, 3) or rgb_bgr.dtype != np.uint8:
        raise ValueError(f"invalid prepared RGB frame: {view['rgb']}")
    if depth_mm is None or depth_mm.shape != (480, 640) or depth_mm.dtype != np.uint16:
        raise ValueError(f"invalid prepared depth frame: {view['depth']}")
    if (
        consistency_image is None
        or consistency_image.shape != (480, 640)
        or consistency_image.dtype != np.uint8
        or not np.isin(consistency_image, (0, 255)).all()
    ):
        raise ValueError(f"invalid stereo consistency image: {consistency_path}")
    if sha256_file(input_root / str(view["rgb"])) != view.get("rgb_sha256"):
        raise ValueError(f"prepared RGB hash differs: {view['rgb']}")
    if sha256_file(input_root / str(view["depth"])) != view.get("depth_sha256"):
        raise ValueError(f"prepared depth hash differs: {view['depth']}")
    if sha256_file(consistency_path) != view.get("stereo_consistency_sha256"):
        raise ValueError(f"stereo consistency hash differs: {consistency_path}")
    return (
        cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth_mm.astype(np.float32) / 1000.0,
        consistency_image > 0,
    )


def _load_masks(
    mask_root: Path,
    mask_manifest: dict[str, object],
) -> dict[str, dict[int, np.ndarray]]:
    masks = {view_name: {} for view_name in POSE_VIEW_NAMES}
    records = mask_manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("mask manifest frames must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("mask frame record must be an object")
        ordinal = int(record["ordinal"])
        views = record.get("views")
        if (
            not isinstance(views, dict)
            or len(views) != len(POSE_VIEW_NAMES)
            or set(views) != set(POSE_VIEW_NAMES)
        ):
            raise ValueError("mask frame lacks the ordered three-view contract")
        for view_name in POSE_VIEW_NAMES:
            view = views[view_name]
            relative = view.get("selected_mask")
            if relative is None:
                continue
            path = mask_root / str(relative)
            expected_hash = view.get("selected_mask_sha256")
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ValueError(f"selected mask hash differs: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (480, 640) or mask.dtype != np.uint8:
                raise ValueError(f"invalid selected mask: {path}")
            if ordinal in masks[view_name]:
                raise ValueError(f"duplicate selected mask ordinal: {view_name}/{ordinal}")
            masks[view_name][ordinal] = mask > 0
    return masks


def _load_dense_masks(
    mask_root: Path,
    mask_manifest: dict[str, object],
    sparse_masks_by_view: dict[str, dict[int, np.ndarray]],
    *,
    frame_count: int,
    minimum_bidirectional_iou: float,
    minimum_area_fraction: float,
    maximum_area_fraction: float,
) -> dict[str, dict[int, np.ndarray]]:
    """Validate the complete dense-mask audit before loading any tracking evidence."""

    dense = mask_manifest.get("dense_tracking_masks")
    if not isinstance(dense, dict) or dense.get("method") != (
        "per_interval_forward_backward_sam2_video_mask_intersection"
    ):
        raise ValueError("mask manifest lacks the bidirectional dense-mask contract")
    thresholds = (
        ("minimum_bidirectional_iou", minimum_bidirectional_iou),
        ("minimum_area_fraction", minimum_area_fraction),
        ("maximum_area_fraction", maximum_area_fraction),
    )
    for name, expected in thresholds:
        value = dense.get(name)
        if not isinstance(value, (int, float)) or not np.isclose(float(value), expected):
            raise ValueError(f"dense-mask {name} differs from the pipeline config")
    views = dense.get("views")
    if not isinstance(views, dict) or set(views) != set(POSE_VIEW_NAMES):
        raise ValueError("dense-mask manifest lacks the ordered three-view contract")

    result = {view_name: {} for view_name in POSE_VIEW_NAMES}
    for view_name in POSE_VIEW_NAMES:
        view = views[view_name]
        if not isinstance(view, dict):
            raise ValueError(f"dense-mask view is malformed: {view_name}")
        keyframes = sorted(sparse_masks_by_view[view_name])
        if view.get("keyframe_ordinals") != keyframes:
            raise ValueError(f"dense-mask keyframes differ from sparse masks: {view_name}")
        frame_records = view.get("frames")
        intervals = view.get("intervals")
        accepted_ordinals = view.get("accepted_ordinals")
        if (
            not isinstance(frame_records, list)
            or not isinstance(intervals, list)
            or not isinstance(accepted_ordinals, list)
            or accepted_ordinals != sorted(set(int(value) for value in accepted_ordinals))
            or int(view.get("accepted_frame_count", -1)) != len(accepted_ordinals)
        ):
            raise ValueError(f"dense-mask view index is malformed: {view_name}")
        record_by_ordinal = {}
        for record in frame_records:
            if not isinstance(record, dict):
                raise ValueError(f"dense-mask frame is malformed: {view_name}")
            ordinal = int(record.get("ordinal", -1))
            if ordinal not in range(frame_count) or ordinal in record_by_ordinal:
                raise ValueError(f"dense-mask ordinal is invalid: {view_name}/{ordinal}")
            path = mask_root / str(record.get("path"))
            expected_hash = record.get("sha256")
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ValueError(f"dense mask hash differs: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (480, 640) or mask.dtype != np.uint8:
                raise ValueError(f"invalid dense mask: {path}")
            mask = mask > 0
            source = record.get("source")
            if source == "selected_sparse_keyframe":
                if ordinal not in sparse_masks_by_view[view_name] or not np.array_equal(
                    mask, sparse_masks_by_view[view_name][ordinal]
                ):
                    raise ValueError(f"dense keyframe differs from sparse mask: {view_name}/{ordinal}")
                if record.get("fusion") is not None:
                    raise ValueError("dense keyframes must not claim propagated-mask fusion")
            elif source == "bidirectional_sam2_video_intersection":
                fusion = record.get("fusion")
                interval = record.get("interval_keyframe_ordinals")
                if (
                    not isinstance(fusion, dict)
                    or fusion.get("passes_gate") is not True
                    or float(fusion.get("bidirectional_iou", -1.0))
                    < minimum_bidirectional_iou
                    or not minimum_area_fraction
                    <= float(fusion.get("fused_area_fraction", -1.0))
                    <= maximum_area_fraction
                    or not isinstance(interval, list)
                    or len(interval) != 2
                    or not int(interval[0]) < ordinal < int(interval[1])
                ):
                    raise ValueError(f"dense propagated-mask audit failed: {view_name}/{ordinal}")
            else:
                raise ValueError(f"unknown dense-mask source: {source}")
            record_by_ordinal[ordinal] = record
            result[view_name][ordinal] = mask
        if sorted(record_by_ordinal) != accepted_ordinals:
            raise ValueError(f"dense-mask frame index differs: {view_name}")

        expected_pairs = list(zip(keyframes, keyframes[1:]))
        if len(intervals) != len(expected_pairs):
            raise ValueError(f"dense-mask interval count differs: {view_name}")
        interval_accepted = set(keyframes)
        for interval, (left, right) in zip(intervals, expected_pairs, strict=True):
            if not isinstance(interval, dict) or (
                int(interval.get("left_keyframe_ordinal", -1)),
                int(interval.get("right_keyframe_ordinal", -1)),
            ) != (left, right):
                raise ValueError(f"dense-mask interval endpoints differ: {view_name}")
            accepted = [int(value) for value in interval.get("accepted_ordinals", [])]
            rejected = [int(value) for value in interval.get("rejected_ordinals", [])]
            expected_intermediate = list(range(left + 1, right))
            if (
                sorted(accepted + rejected) != expected_intermediate
                or set(accepted) & set(rejected)
                or int(interval.get("intermediate_frame_count", -1))
                != len(expected_intermediate)
            ):
                raise ValueError(f"dense-mask interval coverage differs: {view_name}")
            frame_audit = interval.get("frames")
            if (
                not isinstance(frame_audit, list)
                or [int(value.get("ordinal", -1)) for value in frame_audit]
                != expected_intermediate
            ):
                raise ValueError(f"dense-mask interval audit differs: {view_name}")
            for audit in frame_audit:
                fusion = audit.get("fusion")
                ordinal = int(audit["ordinal"])
                if not isinstance(fusion, dict) or (
                    fusion.get("passes_gate") is True
                ) != (ordinal in accepted):
                    raise ValueError(f"dense-mask fusion decision differs: {view_name}/{ordinal}")
            interval_accepted.update(accepted)
        if interval_accepted != set(accepted_ordinals):
            raise ValueError(f"dense-mask accepted set differs: {view_name}")
    return result


def _with_terminal_confirmation_registration(
    sparse_masks: dict[int, np.ndarray],
    dense_masks: dict[int, np.ndarray],
    source_frame_indices: np.ndarray,
    *,
    maximum_evidence_source_frame_gap: int,
    maximum_confirmation_source_frame_gap: int,
) -> tuple[dict[int, np.ndarray], tuple[int, ...], int]:
    """Fill the terminal sparse interval and add a pre-terminal confirmation."""

    if (
        len(sparse_masks) < 2
        or maximum_evidence_source_frame_gap <= 0
        or maximum_confirmation_source_frame_gap <= 0
    ):
        raise ValueError("terminal confirmation requires two masks and positive gaps")
    source_frames = np.asarray(source_frame_indices, dtype=np.int64)
    ordered_sparse = sorted(sparse_masks)
    interval_start, terminal = ordered_sparse[-2:]
    if terminal >= len(source_frames):
        raise ValueError("terminal registration ordinal is outside the source-frame index")
    dense_interval = sorted(
        ordinal
        for ordinal in dense_masks
        if interval_start < ordinal < terminal
        and ordinal < len(source_frames)
    )
    registrations = dict(sparse_masks)
    promoted = []
    cursor = interval_start
    while (
        source_frames[terminal] - source_frames[cursor]
        > maximum_evidence_source_frame_gap
    ):
        eligible = [
            ordinal
            for ordinal in dense_interval
            if ordinal > cursor
            and source_frames[ordinal] - source_frames[cursor]
            <= maximum_evidence_source_frame_gap
        ]
        if not eligible:
            raise ValueError("audited terminal masks cannot cover the evidence gap")
        cursor = max(eligible)
        registrations[cursor] = dense_masks[cursor]
        promoted.append(cursor)
    confirmation_eligible = [
        ordinal
        for ordinal in dense_interval
        if source_frames[terminal] - source_frames[ordinal]
        <= maximum_confirmation_source_frame_gap
    ]
    if not confirmation_eligible:
        raise ValueError("no audited pre-terminal mask is close enough for confirmation")
    confirmation = max(confirmation_eligible)
    registrations[confirmation] = dense_masks[confirmation]
    if confirmation not in promoted:
        promoted.append(confirmation)
    return registrations, tuple(promoted), confirmation


def _retain_manipulation_and_final_static_anchors(
    values: list[tuple[TemporalAnchor, tuple[str, ...]]],
    *,
    last_active_hand_ordinal: int,
    final_static_ordinals: tuple[int, int],
    grasp_observed_position_max: float,
) -> tuple[
    list[tuple[TemporalAnchor, tuple[str, ...]]],
    list[dict[str, object]],
    int,
]:
    """Exclude noisy free-settle poses that are not used by Mimic."""

    if (
        not values
        or last_active_hand_ordinal < 0
        or grasp_observed_position_max <= 0.0
    ):
        raise ValueError("post-release filtering requires an observed grasp")
    release = next(
        (
            anchor
            for anchor, _ in values
            if anchor.ordinal > last_active_hand_ordinal
            and np.all(
                anchor.hand_observed_position > grasp_observed_position_max
            )
        ),
        None,
    )
    required = set(final_static_ordinals)
    available = {anchor.ordinal for anchor, _ in values}
    if (
        release is None
        or len(required) != 2
        or not required.issubset(available)
        or release.ordinal >= min(required)
    ):
        raise ValueError("release and two final-static anchors are not independently observed")
    retained = []
    excluded = []
    for value in values:
        anchor = value[0]
        if anchor.ordinal <= release.ordinal or anchor.ordinal in required:
            retained.append(value)
        else:
            excluded.append(
                {
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "reason": "open_hand_free_settle_pose_not_used_by_mimic",
                }
            )
    return retained, excluded, release.ordinal


def _select_terminal_static_anchor_ordinals(
    values: list[tuple[TemporalAnchor, tuple[str, ...]]],
    parameters: TemporalSelectionParameters,
    *,
    last_active_hand_ordinal: int,
    terminal_source_frame_index: int,
    maximum_terminal_gap_source_frames: int,
) -> tuple[int, int]:
    """Return the last two visually verified open-hand static anchors.

    The final source frame may legitimately have insufficient table pixels for
    registration after the operator releases the table.  We therefore require
    two static RGB-D confirmations inside a bounded terminal interval, then
    hold that verified final pose through the short unobserved tail.  This is
    not a simulator-state fallback: both confirmations retain the ordinary
    CAD, mask, stereo-depth, and bidirectional-pose gates.
    """

    eligible: list[TemporalAnchor] = []
    for anchor, _ in values:
        if (
            anchor.ordinal <= last_active_hand_ordinal
            or terminal_source_frame_index - anchor.source_frame_index
            > maximum_terminal_gap_source_frames
            or not np.all(
                anchor.hand_observed_position > parameters.grasp_observed_position_max
            )
        ):
            continue
        try:
            static_costs = temporal_static_visual_costs(anchor, parameters)
        except ValueError:
            continue
        if np.isfinite(static_costs).any():
            eligible.append(anchor)
    if len(eligible) < 2:
        raise ValueError(
            "two visually verified final-static anchors are required within "
            "the terminal tracking interval"
        )
    selected = eligible[-2:]
    return (selected[0].ordinal, selected[1].ordinal)


def _select_initial_static_anchor(
    values: list[tuple[TemporalAnchor, tuple[str, ...]]],
    parameters: TemporalSelectionParameters,
) -> tuple[int, str] | None:
    """Select the first independently verified initial-static anchor.

    A partially closed hand is not by itself evidence that it is attached to
    the table.  The source CAD alignment is eligible at the initial frame only
    when its own registration candidate passes the normal static RGB-D and
    bidirectional gates.  This avoids discarding a visually verified table
    pose solely because an idle Dex1 hand starts closed.
    """

    for index, (anchor, sources) in enumerate(values):
        try:
            static_costs = temporal_static_visual_costs(anchor, parameters)
        except ValueError:
            continue
        if np.all(
            anchor.hand_observed_position > parameters.grasp_observed_position_max
        ):
            return index, "open_hand_static_rgbd"
        if index != 0:
            continue
        source_seed_indices = [
            candidate_index
            for candidate_index, source in enumerate(sources)
            if source == "source_cad_seed"
        ]
        if any(
            np.isfinite(static_costs[candidate_index])
            for candidate_index in source_seed_indices
        ):
            return index, "source_cad_seed_static_rgbd"
    return None


def _track_direction(
    *,
    estimator,
    frames: list[tuple[np.ndarray, np.ndarray]],
    intrinsic_matrix: np.ndarray,
    root_from_cameras: np.ndarray,
    registration_masks: dict[int, np.ndarray],
    order: range,
    registration_iterations: int,
    tracking_iterations: int,
    registration_selector=None,
    registration_evidence: dict[int, dict[str, object]] | None = None,
    initial_camera_from_object: np.ndarray | None = None,
    initial_root_from_object: np.ndarray | None = None,
    eef_poses_root: np.ndarray | None = None,
    hand_observed_positions: np.ndarray | None = None,
    image_laplacian_variances: np.ndarray | None = None,
    stereo_consistency_fractions: np.ndarray | None = None,
    temporal_parameters: TemporalSelectionParameters | None = None,
    grasp_observed_position_max: float | None = None,
    auxiliary_root_from_cameras: dict[str, np.ndarray] | None = None,
    auxiliary_registration_masks: dict[str, dict[int, np.ndarray]] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    root_poses = np.empty((len(frames), 4, 4), dtype=np.float64)
    modes = [""] * len(frames)
    rotation_projection_corrections = np.empty(len(frames), dtype=np.float64)
    initialized = False
    previous_registration_candidates_root = None
    previous_registration_eef_poses_root = None
    previous_registration_hand_observed_position = None
    if eef_poses_root is not None:
        eef_poses_root = np.asarray(eef_poses_root, dtype=np.float64)
        if eef_poses_root.shape != (len(frames), 2, 4, 4) or not np.isfinite(
            eef_poses_root
        ).all():
            raise ValueError("tracking EEF poses must be finite [T,2,4,4]")
        hand_observed_positions = np.asarray(
            hand_observed_positions, dtype=np.float64
        )
        if (
            hand_observed_positions.shape != (len(frames), 2)
            or not np.isfinite(hand_observed_positions).all()
            or grasp_observed_position_max is None
            or not np.isfinite(grasp_observed_position_max)
            or grasp_observed_position_max <= 0.0
            or image_laplacian_variances is None
            or stereo_consistency_fractions is None
            or temporal_parameters is None
        ):
            raise ValueError(
                "EEF propagation requires finite observed hand positions and threshold"
            )
        image_laplacian_variances = np.asarray(
            image_laplacian_variances, dtype=np.float64
        )
        if (
            image_laplacian_variances.shape != (len(frames),)
            or not np.isfinite(image_laplacian_variances).all()
            or np.any(image_laplacian_variances <= 0.0)
        ):
            raise ValueError("EEF propagation requires finite positive image sharpness")
        stereo_consistency_fractions = np.asarray(
            stereo_consistency_fractions, dtype=np.float64
        )
        if (
            stereo_consistency_fractions.shape != (len(frames),)
            or not np.isfinite(stereo_consistency_fractions).all()
            or np.any((stereo_consistency_fractions < 0.0) | (stereo_consistency_fractions > 1.0))
        ):
            raise ValueError("EEF propagation requires stereo consistency fractions")
    elif hand_observed_positions is not None or grasp_observed_position_max is not None:
        raise ValueError("observed hand positions require EEF propagation poses")
    auxiliary_roots = auxiliary_root_from_cameras or {}
    auxiliary_masks = auxiliary_registration_masks or {}
    if set(auxiliary_roots) != set(auxiliary_masks):
        raise ValueError("tracking auxiliary camera roots and masks differ")
    for view_name, values in auxiliary_roots.items():
        roots = np.asarray(values, dtype=np.float64)
        if roots.shape != (len(frames), 4, 4) or not np.isfinite(roots).all():
            raise ValueError(f"tracking auxiliary camera poses are invalid: {view_name}")
        auxiliary_roots[view_name] = roots
    first_ordinal = order.start
    if initial_root_from_object is not None:
        initial_root_from_object = _as_transform(
            initial_root_from_object, "initial_root_from_object"
        )
    if initial_camera_from_object is not None:
        seed = _as_transform(initial_camera_from_object, "initial_camera_from_object")
        _set_estimator_original_pose(estimator, seed)
    for ordinal in order:
        rgb, depth = frames[ordinal]
        if ordinal == first_ordinal and initial_camera_from_object is not None:
            camera_from_object = initial_camera_from_object
            initialized = True
            modes[ordinal] = "seeded_reverse_validation"
        elif ordinal in registration_masks:
            tracked_centered_pose = None
            if initialized and registration_selector is not None:
                with _foundationpose_debug_output():
                    estimator.track_one(
                        rgb=rgb,
                        depth=depth,
                        K=intrinsic_matrix,
                        iteration=tracking_iterations,
                    )
                tracked_centered_pose = estimator.pose_last.detach().clone()
            with _foundationpose_debug_output():
                camera_from_object = estimator.register(
                    K=intrinsic_matrix,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=registration_masks[ordinal],
                    iteration=registration_iterations,
                )
            mode = "register"
            if registration_selector is not None:
                additional_root_candidates = None
                additional_sources = None
                if previous_registration_candidates_root is not None:
                    additional_root_candidates, additional_sources = (
                        _propagated_root_candidates(
                            previous_registration_candidates_root,
                            previous_registration_eef_poses_root,
                            eef_poses_root[ordinal],
                            (
                                previous_registration_hand_observed_position
                                <= grasp_observed_position_max
                            )
                            & (
                                hand_observed_positions[ordinal]
                                <= grasp_observed_position_max
                            ),
                        )
                    )
                if ordinal == first_ordinal and initial_root_from_object is not None:
                    seed_candidates = initial_root_from_object[None]
                    if additional_root_candidates is None:
                        additional_root_candidates = seed_candidates
                        additional_sources = ("source_cad_seed",)
                    else:
                        additional_root_candidates = np.concatenate(
                            (additional_root_candidates, seed_candidates), axis=0
                        )
                        additional_sources = (*additional_sources, "source_cad_seed")
                camera_from_object, mode, evidence = registration_selector(
                    estimator=estimator,
                    rgb=rgb,
                    depth=depth,
                    observed_mask=registration_masks[ordinal],
                    tracked_centered_pose=tracked_centered_pose,
                    root_from_camera=root_from_cameras[ordinal],
                    additional_root_candidates=additional_root_candidates,
                    additional_sources=additional_sources,
                    auxiliary_root_from_cameras={
                        view_name: values[ordinal]
                        for view_name, values in auxiliary_roots.items()
                        if ordinal in auxiliary_masks[view_name]
                    },
                    auxiliary_observed_masks={
                        view_name: auxiliary_masks[view_name][ordinal]
                        for view_name in auxiliary_roots
                        if ordinal in auxiliary_masks[view_name]
                    },
                )
                if registration_evidence is not None:
                    registration_evidence[ordinal] = evidence
                if eef_poses_root is not None:
                    candidate_poses = np.asarray(
                        evidence["temporal_candidate_poses_root"], dtype=np.float64
                    )
                    candidate_metrics = tuple(
                        RegistrationGeometryMetrics(**value)
                        for value in evidence["temporal_candidate_metrics"]
                    )
                    candidate_sources = tuple(
                        str(value) for value in evidence["temporal_candidate_sources"]
                    )
                    anchor = TemporalAnchor(
                        ordinal=ordinal,
                        source_frame_index=ordinal,
                        candidate_poses_root=candidate_poses,
                        candidate_metrics=tuple(
                            value.to_json() for value in candidate_metrics
                        ),
                        eef_poses_root=eef_poses_root[ordinal],
                        hand_observed_position=hand_observed_positions[ordinal],
                        primary_laplacian_variance=float(
                            image_laplacian_variances[ordinal]
                        ),
                        primary_stereo_consistent_fraction=float(
                            stereo_consistency_fractions[ordinal]
                        ),
                    )
                    try:
                        visual_costs = temporal_visual_costs(
                            anchor,
                            temporal_parameters,
                            require_bidirectional_consensus=False,
                        )
                        priority_indices = tuple(
                            int(index)
                            for index in np.argsort(visual_costs, kind="stable")
                            if np.isfinite(visual_costs[index])
                        )
                    except ValueError:
                        priority_indices = ()
                    propagation_indices = propagation_candidate_indices(
                        candidate_metrics,
                        candidate_sources,
                        selected_candidate_index=int(
                            evidence["selected_candidate_index"]
                        ),
                        limit=registration_selector.propagation_beam_size,
                        priority_indices=priority_indices,
                    )
                    evidence["temporal_visual_eligible_candidate_indices"] = list(
                        priority_indices
                    )
                    evidence["propagation_candidate_indices"] = propagation_indices
                    propagation_indices = np.asarray(
                        propagation_indices, dtype=np.int64
                    )
                    previous_registration_candidates_root = candidate_poses[
                        propagation_indices
                    ]
                    previous_registration_eef_poses_root = eef_poses_root[ordinal]
                    previous_registration_hand_observed_position = (
                        hand_observed_positions[ordinal]
                    )
            initialized = True
            modes[ordinal] = mode
        elif initialized:
            with _foundationpose_debug_output():
                camera_from_object = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=intrinsic_matrix,
                    iteration=tracking_iterations,
                )
            modes[ordinal] = "track"
        else:
            raise ValueError("tracking direction does not begin at a valid registration frame")
        camera_from_object, rotation_projection_corrections[ordinal] = (
            project_to_rigid_transform(
                np.asarray(camera_from_object, dtype=np.float64).reshape(4, 4),
                f"camera_from_object[{ordinal}]",
            )
        )
        _set_estimator_original_pose(estimator, camera_from_object)
        root_poses[ordinal] = root_from_object_pose(
            root_from_cameras[ordinal], camera_from_object
        )
    if any(not value for value in modes):
        raise ValueError("tracking did not cover every sampled frame")
    return root_poses, tuple(modes), rotation_projection_corrections


class _RegistrationGeometrySelector:
    def __init__(
        self,
        *,
        intrinsic_matrix,
        mesh_tensors,
        glctx,
        render_function,
        torch_module,
        depth_consistency_m: float,
        minimum_mask_explained_fraction: float,
        max_correction_translation_m: float,
        max_correction_rotation_rad: float,
        propagation_beam_size: int,
        auxiliary_intrinsics: dict[str, np.ndarray],
        auxiliary_view_score_weight: float,
        auxiliary_primary_support_saturation_fraction: float,
    ):
        self.intrinsic_matrix = intrinsic_matrix
        self.mesh_tensors = mesh_tensors
        self.glctx = glctx
        self.render_function = render_function
        self.torch = torch_module
        self.depth_consistency_m = depth_consistency_m
        self.minimum_mask_explained_fraction = minimum_mask_explained_fraction
        self.max_correction_translation_m = max_correction_translation_m
        self.max_correction_rotation_rad = max_correction_rotation_rad
        self.propagation_beam_size = propagation_beam_size
        self.auxiliary_intrinsics = {
            name: np.asarray(value, dtype=np.float64).reshape(3, 3)
            for name, value in auxiliary_intrinsics.items()
        }
        self.auxiliary_view_score_weight = auxiliary_view_score_weight
        self.auxiliary_primary_support_saturation_fraction = (
            auxiliary_primary_support_saturation_fraction
        )

    def _render_depths(self, camera_from_objects, intrinsic_matrix: np.ndarray) -> np.ndarray:
        rendered_chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(camera_from_objects), 32):
                _, rendered_depth, _ = self.render_function(
                    K=intrinsic_matrix,
                    H=480,
                    W=640,
                    ob_in_cams=camera_from_objects[start : start + 32],
                    glctx=self.glctx,
                    context="cuda",
                    get_normal=False,
                    mesh_tensors=self.mesh_tensors,
                )
                rendered_chunks.append(rendered_depth.detach().cpu())
        return self.torch.cat(rendered_chunks, dim=0).numpy().astype(np.float32)

    def render_root_pose_depths(
        self,
        root_from_object: np.ndarray,
        root_from_cameras: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Render one object pose into each calibrated annotation camera."""

        rendered = self.render_root_poses_depths(
            np.asarray(root_from_object, dtype=np.float64).reshape(1, 4, 4),
            root_from_cameras,
        )
        return {view_name: values[0] for view_name, values in rendered.items()}

    def render_root_poses_depths(
        self,
        root_from_objects: np.ndarray,
        root_from_cameras: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Render a root-frame pose beam into each calibrated annotation camera."""

        poses = np.asarray(root_from_objects, dtype=np.float64)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or len(poses) == 0
            or not np.isfinite(poses).all()
        ):
            raise ValueError("pose-beam rendering requires finite [N,4,4] transforms")
        if set(root_from_cameras) != set(POSE_VIEW_NAMES):
            raise ValueError("pose-beam rendering requires the ordered three-view contract")
        rendered = {}
        for view_name in POSE_VIEW_NAMES:
            root_from_camera = _as_transform(
                root_from_cameras[view_name], f"pose beam {view_name} root_from_camera"
            )
            camera_from_objects = np.linalg.inv(root_from_camera)[None] @ poses
            camera_from_object_tensor = self.torch.as_tensor(
                camera_from_objects,
                dtype=self.torch.float32,
                device="cuda",
            )
            intrinsic = (
                self.intrinsic_matrix
                if view_name == PRIMARY_POSE_VIEW
                else self.auxiliary_intrinsics[view_name]
            )
            rendered[view_name] = self._render_depths(
                camera_from_object_tensor, intrinsic
            )
        return rendered

    def _select(self, metrics, indices) -> tuple[int, str]:
        values = tuple(indices)
        strict = tuple(
            index
            for index in values
            if metrics[index].median_absolute_depth_error_m is not None
            and metrics[index].median_absolute_depth_error_m <= self.depth_consistency_m
            and metrics[index].mask_explained_fraction
            >= self.minimum_mask_explained_fraction
        )
        pool = strict or values
        selected = max(
            pool,
            key=lambda index: (
                metrics[index].multiview_score,
                metrics[index].depth_consistent_union_fraction,
                metrics[index].mask_explained_fraction,
                metrics[index].mask_precision,
                -index,
            ),
        )
        return selected, "strict_geometry_gate" if strict else "best_available_geometry"

    def __call__(
        self,
        *,
        estimator,
        rgb: np.ndarray,
        depth: np.ndarray,
        observed_mask: np.ndarray,
        tracked_centered_pose,
        root_from_camera: np.ndarray,
        additional_root_candidates: np.ndarray | None = None,
        additional_sources: tuple[str, ...] | None = None,
        auxiliary_root_from_cameras: dict[str, np.ndarray] | None = None,
        auxiliary_observed_masks: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, str, dict[str, object]]:
        del rgb
        centered_poses = estimator.poses.detach().clone()
        sources = ["global_registration"] * len(centered_poses)
        global_candidate_count = len(centered_poses)
        if tracked_centered_pose is not None:
            centered_poses = self.torch.cat(
                (centered_poses, tracked_centered_pose.reshape(1, 4, 4)), dim=0
            )
            sources.append("continued_tracking")
        original_from_centered = estimator.get_tf_to_centered_mesh()
        root_from_camera = _as_transform(root_from_camera, "registration root_from_camera")
        if additional_root_candidates is not None:
            additional_root = np.asarray(additional_root_candidates, dtype=np.float64)
            if (
                additional_root.ndim != 3
                or additional_root.shape[1:] != (4, 4)
                or additional_sources is None
                or len(additional_sources) != len(additional_root)
                or not np.isfinite(additional_root).all()
            ):
                raise ValueError("additional temporal registration candidates are invalid")
            original_from_centered_numpy = (
                original_from_centered.detach().cpu().numpy()
                if hasattr(original_from_centered, "detach")
                else np.asarray(original_from_centered, dtype=np.float64)
            )
            camera_from_root = np.linalg.inv(root_from_camera)
            additional_centered = (
                camera_from_root[None]
                @ additional_root
                @ np.linalg.inv(original_from_centered_numpy)[None]
            )
            centered_poses = self.torch.cat(
                (centered_poses, centered_poses.new_tensor(additional_centered)), dim=0
            )
            sources.extend(additional_sources)
        camera_from_objects = centered_poses @ original_from_centered
        rendered = self._render_depths(camera_from_objects, self.intrinsic_matrix)
        auxiliary_roots = auxiliary_root_from_cameras or {}
        auxiliary_masks = auxiliary_observed_masks or {}
        if set(auxiliary_roots) != set(auxiliary_masks) or not set(
            auxiliary_roots
        ).issubset(self.auxiliary_intrinsics):
            raise ValueError("auxiliary registration views are inconsistent")
        root_from_objects = (
            root_from_camera[None]
            @ camera_from_objects.detach().cpu().numpy()
        )
        auxiliary_rendered = {}
        for view_name, root_from_auxiliary_camera in auxiliary_roots.items():
            camera_from_root = np.linalg.inv(
                _as_transform(root_from_auxiliary_camera, f"{view_name} root_from_camera")
            )
            auxiliary_camera_from_objects = camera_from_root[None] @ root_from_objects
            auxiliary_rendered[view_name] = self._render_depths(
                camera_from_objects.new_tensor(auxiliary_camera_from_objects),
                self.auxiliary_intrinsics[view_name],
            )
        metrics, selected = rank_registration_depths(
            rendered_depths_m=rendered,
            observed_depth_m=depth,
            observed_mask=observed_mask,
            sources=tuple(sources),
            maximum_consistent_depth_error_m=self.depth_consistency_m,
            auxiliary_rendered_depths_m=auxiliary_rendered,
            auxiliary_observed_masks=auxiliary_masks,
            auxiliary_view_score_weight=self.auxiliary_view_score_weight,
            auxiliary_primary_support_saturation_fraction=(
                self.auxiliary_primary_support_saturation_fraction
            ),
        )
        correction_metrics: list[tuple[float | None, float | None]] = [
            (None, None) for _ in metrics
        ]
        eligible = list(range(len(metrics)))
        selection_reason = "best_global_registration"
        tracked_reliable = None
        if tracked_centered_pose is not None:
            tracked_index = sources.index("continued_tracking")
            tracked_original = camera_from_objects[tracked_index].detach().cpu().numpy()
            eligible = []
            for index, pose in enumerate(camera_from_objects.detach().cpu().numpy()):
                translation, rotation, _ = pose_errors(tracked_original, pose)
                correction_metrics[index] = (translation, rotation)
                if (
                    translation <= self.max_correction_translation_m
                    and rotation <= self.max_correction_rotation_rad
                ):
                    eligible.append(index)
            if tracked_index not in eligible:
                raise RuntimeError("continued-tracking candidate failed its own continuity gate")
            tracked_metrics = metrics[tracked_index]
            tracked_reliable = bool(
                tracked_metrics.median_absolute_depth_error_m is not None
                and tracked_metrics.median_absolute_depth_error_m
                <= self.depth_consistency_m
                and tracked_metrics.mask_explained_fraction
                >= self.minimum_mask_explained_fraction
            )
            if tracked_reliable:
                selected, tier = self._select(metrics, eligible)
                selection_reason = f"continuity_gated_{tier}"
            else:
                selected, tier = self._select(
                    metrics, (index for index in range(len(metrics)) if index != tracked_index)
                )
                selection_reason = f"unreliable_track_global_reinitialization_{tier}"
        else:
            selected, tier = self._select(metrics, range(len(metrics)))
            selection_reason = tier
        estimator.pose_last = centered_poses[selected].detach().clone()
        selected_pose = camera_from_objects[selected].detach().cpu().numpy()
        ranked = sorted(
            metrics,
            key=lambda value: (
                value.multiview_score,
                value.depth_consistent_union_fraction,
                value.mask_explained_fraction,
                value.mask_precision,
                -value.candidate_index,
            ),
            reverse=True,
        )
        propagation_indices = propagation_candidate_indices(
            tuple(metrics),
            tuple(sources),
            selected_candidate_index=selected,
            limit=self.propagation_beam_size,
        )
        evidence = {
            "candidate_count": len(metrics),
            "selected_candidate_index": selected,
            "selected_source": sources[selected],
            "selection_reason": selection_reason,
            "continued_tracking_reliable": tracked_reliable,
            "eligible_candidate_count": len(eligible),
            "max_correction_translation_m": self.max_correction_translation_m,
            "max_correction_rotation_rad": self.max_correction_rotation_rad,
            "selected_correction_translation_m": correction_metrics[selected][0],
            "selected_correction_rotation_rad": correction_metrics[selected][1],
            "selected": metrics[selected].to_json(),
            "top_candidates": [value.to_json() for value in ranked[:5]],
            "propagation_candidate_indices": propagation_indices,
            # The temporal solver needs the continuity hypothesis as well as
            # independent registrations. Its visual and physical gates decide
            # whether that hypothesis is usable.
            "temporal_candidate_poses_root": [
                (root_from_camera @ pose).tolist()
                for pose in camera_from_objects.detach().cpu().numpy()
            ],
            "temporal_candidate_metrics": [value.to_json() for value in metrics],
            "temporal_candidate_sources": sources,
            "global_candidate_poses_root": [
                (root_from_camera @ pose).tolist()
                for pose in camera_from_objects[:global_candidate_count].detach().cpu().numpy()
            ],
            "global_candidate_metrics": [
                value.to_json() for value in metrics[:global_candidate_count]
            ],
        }
        return selected_pose, f"register_geometry_{sources[selected]}", evidence


def _registration_metrics_from_json(
    values: object,
) -> tuple[RegistrationGeometryMetrics, ...]:
    if not isinstance(values, list):
        raise ValueError("registration metric records must be a list")
    fields = tuple(RegistrationGeometryMetrics.__dataclass_fields__)
    metrics = []
    for value in values:
        if not isinstance(value, dict) or any(name not in value for name in fields):
            raise ValueError("registration metric record is incomplete")
        metrics.append(
            RegistrationGeometryMetrics(**{name: value[name] for name in fields})
        )
    return tuple(metrics)


def _dense_directional_candidate_evidence(
    *,
    start_ordinal: int,
    target_ordinals: tuple[int, ...],
    start_evidence: dict[str, object],
    direction: str,
    registration_selector: _RegistrationGeometrySelector,
    frames: list[tuple[np.ndarray, np.ndarray]],
    dense_masks_by_view: dict[str, dict[int, np.ndarray]],
    root_from_cameras: dict[str, np.ndarray],
    eef_poses_root: np.ndarray,
    hand_observed_positions: np.ndarray,
    image_laplacian_variances: np.ndarray,
    stereo_consistency_fractions: np.ndarray,
    source_frame_indices: np.ndarray,
    temporal_parameters: TemporalSelectionParameters,
    propagation_beam_size: int,
) -> dict[int, dict[str, object]]:
    """Propagate and RGB-D-rank a bounded pose beam through dense mask anchors."""

    if direction not in {"forward", "reverse"}:
        raise ValueError("dense candidate direction must be forward or reverse")
    expected = tuple(sorted(target_ordinals, reverse=direction == "reverse"))
    if target_ordinals != expected or any(
        ordinal not in dense_masks_by_view[PRIMARY_POSE_VIEW]
        for ordinal in target_ordinals
    ):
        raise ValueError("dense directional targets are unordered or unmasked")
    poses = np.asarray(
        start_evidence.get("temporal_candidate_poses_root"), dtype=np.float64
    )
    sources = start_evidence.get("temporal_candidate_sources")
    start_metrics = _registration_metrics_from_json(
        start_evidence.get("temporal_candidate_metrics")
    )
    priority_indices = tuple(
        int(index)
        for index in start_evidence.get(
            "temporal_visual_eligible_candidate_indices", []
        )
    )
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not isinstance(sources, list)
        or len(sources) != len(poses)
        or len(start_metrics) != len(poses)
        or propagation_beam_size <= 0
    ):
        raise ValueError("dense propagation start evidence has no bounded pose beam")
    stored_beam = start_evidence.get("propagation_candidate_indices")
    expected_beam_size = min(propagation_beam_size, len(poses))
    if (
        isinstance(stored_beam, list)
        and len(stored_beam) == expected_beam_size
        and all(
            isinstance(index, int) and index in range(len(poses))
            for index in stored_beam
        )
        and len(set(stored_beam)) == len(stored_beam)
    ):
        beam_indices = list(stored_beam)
    else:
        beam_indices = propagation_candidate_indices(
            start_metrics,
            tuple(str(value) for value in sources),
            selected_candidate_index=int(
                start_evidence["selected_candidate_index"]
            ),
            limit=propagation_beam_size,
            priority_indices=priority_indices,
        )
    beam = poses[np.asarray(beam_indices, dtype=np.int64)]
    beam_lineage_ids = np.asarray(beam_indices, dtype=np.int64)
    previous_ordinal = start_ordinal
    result = {}
    started_at = time.monotonic()
    _track_trace(
        "dense_direction_start",
        direction=direction,
        start_ordinal=start_ordinal,
        target_count=len(target_ordinals),
        beam_size=len(beam),
    )
    for ordinal in target_ordinals:
        anchor_started_at = time.monotonic()
        active_hands = (
            hand_observed_positions[previous_ordinal]
            <= temporal_parameters.grasp_observed_position_max
        ) & (
            hand_observed_positions[ordinal]
            <= temporal_parameters.grasp_observed_position_max
        )
        candidates, candidate_sources = _propagated_root_candidates(
            beam,
            eef_poses_root[previous_ordinal],
            eef_poses_root[ordinal],
            active_hands,
            direction=direction,
        )
        if len(candidates) % len(beam) != 0:
            raise RuntimeError("dense pose propagation broke lineage grouping")
        candidate_lineage_ids = np.tile(
            beam_lineage_ids, len(candidates) // len(beam)
        )
        rendered_by_view = registration_selector.render_root_poses_depths(
            candidates,
            {
                view_name: root_from_cameras[view_name][ordinal]
                for view_name in POSE_VIEW_NAMES
            },
        )
        auxiliary_names = tuple(
            view_name
            for view_name in POSE_VIEW_NAMES[1:]
            if ordinal in dense_masks_by_view[view_name]
        )
        metrics, selected = rank_registration_depths(
            rendered_depths_m=rendered_by_view[PRIMARY_POSE_VIEW],
            observed_depth_m=frames[ordinal][1],
            observed_mask=dense_masks_by_view[PRIMARY_POSE_VIEW][ordinal],
            sources=candidate_sources,
            maximum_consistent_depth_error_m=(
                registration_selector.depth_consistency_m
            ),
            auxiliary_rendered_depths_m={
                view_name: rendered_by_view[view_name]
                for view_name in auxiliary_names
            },
            auxiliary_observed_masks={
                view_name: dense_masks_by_view[view_name][ordinal]
                for view_name in auxiliary_names
            },
            auxiliary_view_score_weight=(
                registration_selector.auxiliary_view_score_weight
            ),
            auxiliary_primary_support_saturation_fraction=(
                registration_selector.auxiliary_primary_support_saturation_fraction
            ),
        )
        metric_json = []
        for metric in metrics:
            value = metric.to_json()
            value["primary_stereo_consistent_fraction"] = float(
                stereo_consistency_fractions[ordinal]
            )
            metric_json.append(value)
        anchor = TemporalAnchor(
            ordinal=ordinal,
            source_frame_index=int(source_frame_indices[ordinal]),
            candidate_poses_root=candidates,
            candidate_metrics=tuple(metric_json),
            eef_poses_root=eef_poses_root[ordinal],
            hand_observed_position=hand_observed_positions[ordinal],
            primary_laplacian_variance=float(image_laplacian_variances[ordinal]),
            primary_stereo_consistent_fraction=float(
                stereo_consistency_fractions[ordinal]
            ),
        )
        try:
            costs = temporal_visual_costs(
                anchor,
                temporal_parameters,
                require_bidirectional_consensus=False,
            )
            priority_indices = tuple(
                int(index)
                for index in np.argsort(costs, kind="stable")
                if np.isfinite(costs[index])
            )
        except ValueError:
            priority_indices = ()
        lineage_ids = tuple(int(value) for value in candidate_lineage_ids)
        contact_priority = contact_lineage_priority_indices(
            candidate_sources,
            lineage_ids,
            hand_observed_positions[ordinal],
            grasp_observed_position_max=(
                temporal_parameters.grasp_observed_position_max
            ),
        )
        if contact_priority:
            propagation_indices = propagation_candidate_indices(
                metrics,
                candidate_sources,
                selected_candidate_index=selected,
                limit=propagation_beam_size,
                priority_indices=priority_indices,
                required_indices=contact_priority,
            )
        else:
            propagation_indices = lineage_preserving_candidate_indices(
                metrics,
                candidate_sources,
                lineage_ids,
                selected_candidate_index=selected,
                limit=propagation_beam_size,
                priority_indices=priority_indices,
            )
        result[ordinal] = {
            "candidate_count": len(candidates),
            "selected_candidate_index": selected,
            "selected_source": candidate_sources[selected],
            "selection_reason": f"dense_{direction}_eef_pose_beam_rgbd_rank",
            "temporal_candidate_poses_root": candidates.tolist(),
            "temporal_candidate_metrics": metric_json,
            "temporal_candidate_sources": list(candidate_sources),
            "temporal_candidate_lineage_ids": candidate_lineage_ids.tolist(),
            "temporal_visual_eligible_candidate_indices": list(priority_indices),
            "propagation_candidate_indices": propagation_indices,
            "contact_lineage_priority_candidate_indices": list(
                contact_priority
            ),
            "propagation_lineage_ids": candidate_lineage_ids[
                np.asarray(propagation_indices, dtype=np.int64)
            ].tolist(),
        }
        _track_trace(
            "dense_anchor_complete",
            direction=direction,
            ordinal=ordinal,
            candidate_count=len(candidates),
            selected_candidate_index=selected,
            elapsed_s=round(time.monotonic() - anchor_started_at, 3),
        )
        propagation_indices_array = np.asarray(propagation_indices, dtype=np.int64)
        beam = candidates[propagation_indices_array]
        beam_lineage_ids = candidate_lineage_ids[propagation_indices_array]
        previous_ordinal = ordinal
    _track_trace(
        "dense_direction_complete",
        direction=direction,
        start_ordinal=start_ordinal,
        target_count=len(target_ordinals),
        elapsed_s=round(time.monotonic() - started_at, 3),
    )
    return result


def _registration_evidence_anchor(
    evidence: dict[str, object],
    *,
    ordinal: int,
    source_frame_indices: np.ndarray,
    eef_poses_root: np.ndarray,
    hand_observed_positions: np.ndarray,
    image_laplacian_variances: np.ndarray,
    stereo_consistency_fractions: np.ndarray,
) -> TemporalAnchor:
    return TemporalAnchor(
        ordinal=ordinal,
        source_frame_index=int(source_frame_indices[ordinal]),
        candidate_poses_root=np.asarray(
            evidence["temporal_candidate_poses_root"], dtype=np.float64
        ),
        candidate_metrics=tuple(evidence["temporal_candidate_metrics"]),
        eef_poses_root=eef_poses_root[ordinal],
        hand_observed_position=hand_observed_positions[ordinal],
        primary_laplacian_variance=float(image_laplacian_variances[ordinal]),
        primary_stereo_consistent_fraction=float(
            stereo_consistency_fractions[ordinal]
        ),
    )


def _attach_dense_bidirectional_consensus(
    forward_evidence: dict[str, object],
    backward_evidence: dict[str, object],
    *,
    ordinal: int,
    source_frame_indices: np.ndarray,
    eef_poses_root: np.ndarray,
    hand_observed_positions: np.ndarray,
    image_laplacian_variances: np.ndarray,
    stereo_consistency_fractions: np.ndarray,
    parameters: TemporalSelectionParameters,
    maximum_translation_error_m: float,
    maximum_rotation_error_rad: float,
) -> None:
    forward_anchor = _registration_evidence_anchor(
        forward_evidence,
        ordinal=ordinal,
        source_frame_indices=source_frame_indices,
        eef_poses_root=eef_poses_root,
        hand_observed_positions=hand_observed_positions,
        image_laplacian_variances=image_laplacian_variances,
        stereo_consistency_fractions=stereo_consistency_fractions,
    )
    backward_anchor = _registration_evidence_anchor(
        backward_evidence,
        ordinal=ordinal,
        source_frame_indices=source_frame_indices,
        eef_poses_root=eef_poses_root,
        hand_observed_positions=hand_observed_positions,
        image_laplacian_variances=image_laplacian_variances,
        stereo_consistency_fractions=stereo_consistency_fractions,
    )
    consensus = _bidirectional_consensus_records(
        forward_anchor=forward_anchor,
        backward_anchor=backward_anchor,
        parameters=parameters,
        maximum_translation_error_m=maximum_translation_error_m,
        maximum_rotation_error_rad=maximum_rotation_error_rad,
    )
    metrics = forward_evidence.get("temporal_candidate_metrics")
    if not isinstance(metrics, list):
        raise ValueError("dense forward metrics are malformed")
    for metric, consensus_record in zip(metrics, consensus, strict=True):
        if not isinstance(metric, dict):
            raise ValueError("dense forward metric is malformed")
        metric["bidirectional_consensus"] = consensus_record
    forward_evidence["reverse_candidate_count"] = int(
        backward_evidence["candidate_count"]
    )


def _attach_terminal_reverse_lineage_consensus(
    evidence: dict[str, object],
    terminal_evidence: dict[str, object],
    *,
    terminal_ordinal: int,
    maximum_translation_error_m: float,
    maximum_rotation_error_rad: float,
) -> None:
    """Validate reverse static carries from an independently registered endpoint."""

    metrics = evidence.get("temporal_candidate_metrics")
    sources = evidence.get("temporal_candidate_sources")
    lineages = evidence.get("temporal_candidate_lineage_ids")
    terminal_metrics = terminal_evidence.get("temporal_candidate_metrics")
    terminal_sources = terminal_evidence.get("temporal_candidate_sources")
    if not (
        isinstance(metrics, list)
        and isinstance(sources, list)
        and isinstance(lineages, list)
        and len(metrics) == len(sources) == len(lineages)
        and isinstance(terminal_metrics, list)
        and isinstance(terminal_sources, list)
        and len(terminal_metrics) == len(terminal_sources)
        and terminal_ordinal >= 0
        and maximum_translation_error_m > 0.0
        and maximum_rotation_error_rad > 0.0
    ):
        raise ValueError("terminal reverse-lineage evidence is malformed")
    for metric, source, lineage in zip(metrics, sources, lineages, strict=True):
        if (
            not isinstance(metric, dict)
            or not isinstance(source, str)
            or not isinstance(lineage, int)
            or lineage not in range(len(terminal_metrics))
        ):
            raise ValueError("terminal reverse-lineage candidate is malformed")
        parent_consensus = terminal_metrics[lineage].get("bidirectional_consensus")
        passes = bool(
            source == "static_carry_reverse"
            and isinstance(parent_consensus, dict)
            and parent_consensus.get("passes_gate") is True
        )
        metric["bidirectional_consensus"] = {
            "passes_gate": passes,
            "translation_error_m": 0.0 if passes else None,
            "rotation_error_rad": 0.0 if passes else None,
            "maximum_translation_error_m": maximum_translation_error_m,
            "maximum_rotation_error_rad": maximum_rotation_error_rad,
            "validation_mode": "terminal_reverse_static_lineage",
            "reverse_candidate_index": lineage,
            "reverse_candidate_source": terminal_sources[lineage],
            "reverse_symmetry_index": 0 if passes else None,
            "terminal_registration_ordinal": terminal_ordinal,
        }


def _merge_dense_endpoint_evidence(
    base: dict[str, object],
    additional: dict[str, object],
    *,
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
    propagation_beam_size: int,
) -> None:
    base_poses = list(base["temporal_candidate_poses_root"])
    base_metrics = list(base["temporal_candidate_metrics"])
    base_sources = list(base["temporal_candidate_sources"])
    additional_poses = list(additional["temporal_candidate_poses_root"])
    additional_metrics = list(additional["temporal_candidate_metrics"])
    additional_sources = list(additional["temporal_candidate_sources"])
    additional_lineage_ids = additional.get("temporal_candidate_lineage_ids")
    if not (
        len(base_poses) == len(base_metrics) == len(base_sources)
        and len(additional_poses)
        == len(additional_metrics)
        == len(additional_sources)
        and isinstance(additional_lineage_ids, list)
        and len(additional_lineage_ids) == len(additional_poses)
    ):
        raise ValueError("endpoint registration evidence lengths differ")
    offset = len(base_poses)
    for index, metric in enumerate(additional_metrics):
        if not isinstance(metric, dict):
            raise ValueError("endpoint dense metric is malformed")
        metric["candidate_index"] = offset + index
    base["temporal_candidate_poses_root"] = base_poses + additional_poses
    base["temporal_candidate_metrics"] = base_metrics + additional_metrics
    base["temporal_candidate_sources"] = base_sources + additional_sources
    base["candidate_count"] = len(base_poses) + len(additional_poses)
    base["dense_endpoint_candidate_count"] = len(additional_poses)

    combined_anchor = TemporalAnchor(
        ordinal=anchor.ordinal,
        source_frame_index=anchor.source_frame_index,
        candidate_poses_root=np.asarray(
            base["temporal_candidate_poses_root"], dtype=np.float64
        ),
        candidate_metrics=tuple(base["temporal_candidate_metrics"]),
        eef_poses_root=anchor.eef_poses_root,
        hand_observed_position=anchor.hand_observed_position,
        primary_laplacian_variance=anchor.primary_laplacian_variance,
        primary_stereo_consistent_fraction=(
            anchor.primary_stereo_consistent_fraction
        ),
    )
    try:
        costs = temporal_visual_costs(
            combined_anchor,
            parameters,
            require_bidirectional_consensus=True,
        )
        priority_indices = tuple(
            int(index)
            for index in np.argsort(costs, kind="stable")
            if np.isfinite(costs[index])
        )
    except ValueError:
        priority_indices = ()
    metrics = _registration_metrics_from_json(base["temporal_candidate_metrics"])
    base["temporal_visual_eligible_candidate_indices"] = list(priority_indices)
    contact_priority = contact_lineage_priority_indices(
        tuple(str(value) for value in additional_sources),
        tuple(int(value) for value in additional_lineage_ids),
        anchor.hand_observed_position,
        grasp_observed_position_max=parameters.grasp_observed_position_max,
    )
    required_indices = tuple(offset + index for index in contact_priority)
    base["propagation_candidate_indices"] = propagation_candidate_indices(
        metrics,
        tuple(str(value) for value in base["temporal_candidate_sources"]),
        selected_candidate_index=int(base["selected_candidate_index"]),
        limit=propagation_beam_size,
        priority_indices=priority_indices,
        required_indices=required_indices,
    )
    base["contact_lineage_priority_candidate_indices"] = list(required_indices)


def _append_causal_current_frame_candidate(
    anchors: tuple[TemporalAnchor, ...],
    error: TemporalSelectionError,
    *,
    registration_selector: _RegistrationGeometrySelector,
    frames: list[tuple[np.ndarray, np.ndarray]],
    dense_masks_by_view: dict[str, dict[int, np.ndarray]],
    root_from_cameras: dict[str, np.ndarray],
    stereo_consistency_fractions: np.ndarray,
    parameters: TemporalSelectionParameters,
    rejected_candidate_keys: set[tuple[int, str, bytes]] | None = None,
) -> tuple[TemporalAnchor, ...] | None:
    """Render a missing causal candidate from the accepted predecessor.

    The ordinary dense tracker propagates a registration beam.  Once that beam
    has drifted during a grasp, it can omit the pose implied by the *selected*
    predecessor and current robot FK.  This recovery adds only that physical
    prediction, then scores it against the current head and wrist observations.
    It never reads simulator state or accepts an unrendered pose.
    """

    failure_index = error.diagnostics.get("anchor_index")
    prefix = error.diagnostics.get("causal_prefix")
    if (
        not isinstance(failure_index, int)
        or failure_index not in range(1, len(anchors))
        or not isinstance(prefix, dict)
    ):
        return None
    selected_prefix = np.asarray(prefix.get("selected_poses_root"), dtype=np.float64)
    attached = np.asarray(prefix.get("attached_hands"), dtype=bool)
    if (
        selected_prefix.shape != (failure_index, 4, 4)
        or attached.shape != (2,)
        or not np.isfinite(selected_prefix).all()
    ):
        return None
    previous = anchors[failure_index - 1]
    current = anchors[failure_index]
    if current.ordinal not in dense_masks_by_view[PRIMARY_POSE_VIEW]:
        return None
    previous_closed = (
        previous.hand_observed_position <= parameters.grasp_observed_position_max
    )
    current_closed = (
        current.hand_observed_position <= parameters.grasp_observed_position_max
    )
    continuous_attachment = attached & previous_closed & current_closed
    if np.any(continuous_attachment):
        propagated, sources = _propagated_root_candidates(
            selected_prefix[-1:],
            previous.eef_poses_root,
            current.eef_poses_root,
            continuous_attachment,
        )
        keep = np.asarray(
            [source != "static_carry_forward" for source in sources], dtype=bool
        )
        candidates = propagated[keep]
        candidate_sources = tuple(
            source for source, retain in zip(sources, keep, strict=True) if retain
        )
    elif not np.any(attached):
        candidates = selected_prefix[-1:]
        candidate_sources = ("static_carry_forward",)
    else:
        # A released grasp needs independently bidirectional static evidence;
        # rigidly carrying the prior attachment through release would be false.
        return None
    if len(candidates) == 0:
        _track_trace(
            "causal_candidate_unavailable",
            anchor_ordinal=current.ordinal,
            source_frame_index=current.source_frame_index,
            attached_hands=attached.tolist(),
            reason="no_propagated_candidate_for_current_attachment",
        )
        return None
    existing_sources = tuple(
        str(metric.get("source", "")) for metric in current.candidate_metrics
    )
    novel_pairs = tuple(
        (candidate, source)
        for candidate, source in zip(candidates, candidate_sources, strict=True)
        if not any(
            source == existing_source
            and np.allclose(existing_pose, candidate, rtol=0.0, atol=1.0e-7)
            for existing_pose, existing_source in zip(
                current.candidate_poses_root, existing_sources, strict=True
            )
        )
    )
    if not novel_pairs:
        # A dense forward pass can already contain exactly the FK prediction,
        # but its metric is only a proposal from that pass.  Re-score it in the
        # current RGB-D frame before treating it as a causal carry.  Do not add
        # the same evidence twice on a retry of this recovery.
        revalidation_pairs = []
        for candidate, source in zip(candidates, candidate_sources, strict=True):
            equivalent_metrics = [
                metric
                for existing_pose, existing_source, metric in zip(
                    current.candidate_poses_root,
                    existing_sources,
                    current.candidate_metrics,
                    strict=True,
                )
                if (
                    source == existing_source
                    and np.allclose(existing_pose, candidate, rtol=0.0, atol=1.0e-7)
                )
            ]
            already_revalidated = any(
                isinstance(metric.get("bidirectional_consensus"), dict)
                and metric["bidirectional_consensus"].get("validation_mode")
                == "causal_current_frame_rgbd_propagation"
                for metric in equivalent_metrics
            )
            if not already_revalidated:
                revalidation_pairs.append((candidate, source))
        if not revalidation_pairs:
            _track_trace(
                "causal_candidate_unavailable",
                anchor_ordinal=current.ordinal,
                source_frame_index=current.source_frame_index,
                attached_hands=attached.tolist(),
                reason="equivalent_candidate_already_causally_validated",
            )
            return None
        novel_pairs = tuple(revalidation_pairs)
        _track_trace(
            "causal_candidate_revalidation",
            anchor_ordinal=current.ordinal,
            source_frame_index=current.source_frame_index,
            attached_hands=attached.tolist(),
            candidate_count=len(novel_pairs),
        )
    def candidate_key(candidate: np.ndarray, source: str) -> tuple[int, str, bytes]:
        pose = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64))
        return (current.ordinal, source, pose.tobytes())

    if rejected_candidate_keys is not None:
        novel_pairs = tuple(
            (candidate, source)
            for candidate, source in novel_pairs
            if candidate_key(candidate, source) not in rejected_candidate_keys
        )
        if not novel_pairs:
            _track_trace(
                "causal_candidate_unavailable",
                anchor_ordinal=current.ordinal,
                source_frame_index=current.source_frame_index,
                attached_hands=attached.tolist(),
                reason="current_frame_candidate_previously_rejected",
            )
            return None
    candidates = np.stack([candidate for candidate, _ in novel_pairs])
    candidate_sources = tuple(source for _, source in novel_pairs)
    rendered_by_view = registration_selector.render_root_poses_depths(
        candidates,
        {
            view_name: root_from_cameras[view_name][current.ordinal]
            for view_name in POSE_VIEW_NAMES
        },
    )
    auxiliary_names = tuple(
        view_name
        for view_name in POSE_VIEW_NAMES[1:]
        if current.ordinal in dense_masks_by_view[view_name]
    )
    metrics, _ = rank_registration_depths(
        rendered_depths_m=rendered_by_view[PRIMARY_POSE_VIEW],
        observed_depth_m=frames[current.ordinal][1],
        observed_mask=dense_masks_by_view[PRIMARY_POSE_VIEW][current.ordinal],
        sources=candidate_sources,
        maximum_consistent_depth_error_m=registration_selector.depth_consistency_m,
        auxiliary_rendered_depths_m={
            view_name: rendered_by_view[view_name] for view_name in auxiliary_names
        },
        auxiliary_observed_masks={
            view_name: dense_masks_by_view[view_name][current.ordinal]
            for view_name in auxiliary_names
        },
        auxiliary_view_score_weight=registration_selector.auxiliary_view_score_weight,
        auxiliary_primary_support_saturation_fraction=(
            registration_selector.auxiliary_primary_support_saturation_fraction
        ),
    )
    provisional_metrics = []
    for metric in metrics:
        value = metric.to_json()
        value["primary_stereo_consistent_fraction"] = float(
            stereo_consistency_fractions[current.ordinal]
        )
        value["bidirectional_consensus"] = {
            "passes_gate": False,
            "translation_error_m": None,
            "rotation_error_rad": None,
            "validation_mode": "causal_current_frame_rgbd_propagation",
        }
        provisional_metrics.append(value)
    provisional_anchor = TemporalAnchor(
        ordinal=current.ordinal,
        source_frame_index=current.source_frame_index,
        candidate_poses_root=np.concatenate(
            (current.candidate_poses_root, candidates), axis=0
        ),
        candidate_metrics=tuple((*current.candidate_metrics, *provisional_metrics)),
        eef_poses_root=current.eef_poses_root,
        hand_observed_position=current.hand_observed_position,
        primary_laplacian_variance=current.primary_laplacian_variance,
        primary_stereo_consistent_fraction=(
            current.primary_stereo_consistent_fraction
        ),
    )
    try:
        provisional_costs = temporal_visual_costs(
            provisional_anchor,
            parameters,
            require_bidirectional_consensus=False,
        )
    except ValueError:
        provisional_costs = np.full(
            len(provisional_anchor.candidate_metrics), np.inf, dtype=np.float64
        )
    retained_offsets = np.flatnonzero(
        np.isfinite(provisional_costs[len(current.candidate_metrics) :])
    )
    if len(retained_offsets) == 0:
        if rejected_candidate_keys is not None:
            rejected_candidate_keys.update(
                candidate_key(candidate, source)
                for candidate, source in zip(candidates, candidate_sources, strict=True)
            )
        _track_trace(
            "causal_candidate_rejected",
            anchor_ordinal=current.ordinal,
            source_frame_index=current.source_frame_index,
            reason="no_new_candidate_meets_visual_evidence_gate",
        )
        return None
    candidates = candidates[retained_offsets]
    candidate_sources = tuple(candidate_sources[index] for index in retained_offsets)
    augmented_metrics = list(current.candidate_metrics)
    for offset in retained_offsets:
        value = dict(provisional_metrics[int(offset)])
        value["candidate_index"] = len(augmented_metrics)
        augmented_metrics.append(value)
    augmented_anchor = TemporalAnchor(
        ordinal=current.ordinal,
        source_frame_index=current.source_frame_index,
        candidate_poses_root=np.concatenate(
            (current.candidate_poses_root, candidates), axis=0
        ),
        candidate_metrics=tuple(augmented_metrics),
        eef_poses_root=current.eef_poses_root,
        hand_observed_position=current.hand_observed_position,
        primary_laplacian_variance=current.primary_laplacian_variance,
        primary_stereo_consistent_fraction=(
            current.primary_stereo_consistent_fraction
        ),
    )
    _track_trace(
        "causal_candidate_rendered",
        anchor_ordinal=current.ordinal,
        source_frame_index=current.source_frame_index,
        attached_hands=attached.tolist(),
        candidate_sources=list(candidate_sources),
        generated_candidate_count=len(candidates),
        auditable_new_candidate_count=len(retained_offsets),
        primary_mask_precision=[
            round(float(metric.mask_precision), 5) for metric in metrics
        ],
        primary_mask_explained_fraction=[
            round(float(metric.mask_explained_fraction), 5) for metric in metrics
        ],
        depth_error_m=[
            None
            if metric.median_absolute_depth_error_m is None
            else round(float(metric.median_absolute_depth_error_m), 5)
            for metric in metrics
        ],
        auxiliary_explained_fraction=[
            {
                name: round(float(value), 5)
                for name, value in metric.auxiliary_mask_explained_fractions.items()
            }
            for metric in metrics
        ],
    )
    result = list(anchors)
    result[failure_index] = augmented_anchor
    return tuple(result)


def _append_causal_current_frame_chain(
    anchors: tuple[TemporalAnchor, ...],
    error: TemporalSelectionError,
    **kwargs,
) -> tuple[TemporalAnchor, ...] | None:
    """Extend one accepted EEF/FK prefix through its continuous grasp run.

    Re-running the full causal beam after adding a single dense frame makes a
    long occlusion quadratic in episode length.  Each step below is still
    rendered and scored by `_append_causal_current_frame_candidate`; this
    helper merely uses its best newly scored physical prediction as the next
    predecessor while the same grasp remains continuously observed.
    """

    result = anchors
    current_error = error
    changed = False
    for _ in range(len(anchors)):
        failure_index = current_error.diagnostics.get("anchor_index")
        prefix = current_error.diagnostics.get("causal_prefix")
        if not isinstance(failure_index, int) or not isinstance(prefix, dict):
            break
        augmented = _append_causal_current_frame_candidate(
            result, current_error, **kwargs
        )
        if augmented is None:
            break
        changed = True
        old_count = len(result[failure_index].candidate_metrics)
        current = augmented[failure_index]
        try:
            costs = temporal_visual_costs(
                current, kwargs["parameters"], require_bidirectional_consensus=False
            )
        except ValueError:
            break
        new_indices = np.arange(old_count, len(costs), dtype=np.int64)
        new_indices = new_indices[np.isfinite(costs[new_indices])]
        if len(new_indices) == 0:
            _track_trace(
                "causal_candidate_rejected",
                anchor_ordinal=current.ordinal,
                source_frame_index=current.source_frame_index,
                reason="no_new_candidate_meets_visual_evidence_gate",
            )
            break
        choice = int(new_indices[np.argmin(costs[new_indices])])
        selected_poses = list(prefix.get("selected_poses_root", ()))
        attached = np.asarray(prefix.get("attached_hands"), dtype=bool)
        if attached.shape != (2,):
            break
        selected_poses.append(current.candidate_poses_root[choice].tolist())
        previous = result[failure_index - 1]
        engaging = (
            current.hand_observed_position
            <= kwargs["parameters"].grasp_observed_position_max
        ) & (
            previous.hand_observed_position
            > kwargs["parameters"].grasp_observed_position_max
        )
        # Mirror the causal selector's release semantics.  A hand that is now
        # open cannot be used to propagate a rigid table-to-EEF constraint.
        attached &= (
            current.hand_observed_position
            <= kwargs["parameters"].grasp_observed_position_max
        )
        attached |= engaging
        result = augmented
        next_index = failure_index + 1
        if next_index >= len(result):
            break
        current_error = TemporalSelectionError(
            "continue causal current-frame RGB-D chain",
            {
                "anchor_index": next_index,
                "causal_prefix": {
                    "selected_poses_root": selected_poses,
                    "attached_hands": attached.tolist(),
                },
            },
        )
    return result if changed else None


def _select_with_optional_dense_anchors(
    anchors: tuple[TemporalAnchor, ...],
    candidate_sources: tuple[tuple[str, ...], ...],
    *,
    optional_ordinals: set[int],
    source_fps: float,
    parameters: TemporalSelectionParameters,
    causal_candidate_augmenter=None,
) -> tuple[
    TemporalSelectionResult,
    tuple[TemporalAnchor, ...],
    tuple[tuple[str, ...], ...],
    list[dict[str, object]],
]:
    """Drop only designated observations that block an otherwise valid path."""

    if len(anchors) != len(candidate_sources):
        raise ValueError("optional temporal anchor sources differ from anchors")

    trace_enabled = os.environ.get("FLIP_TABLE_TEMPORAL_SELECTION_TRACE") == "1"

    def trace(event: str, **fields: object) -> None:
        if not trace_enabled:
            return
        print(
            "[temporal-selection] "
            + json.dumps({"event": event, **fields}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )

    working = list(zip(anchors, candidate_sources, strict=True))
    pruned = []
    attempt = 0
    while True:
        attempt += 1
        current_anchors = tuple(value[0] for value in working)
        trace(
            "attempt_start",
            attempt=attempt,
            anchor_count=len(current_anchors),
            candidate_count=sum(
                len(anchor.candidate_poses_root) for anchor in current_anchors
            ),
            pruned_count=len(pruned),
        )
        failure: TemporalSelectionError | None = None
        if causal_candidate_augmenter is not None:
            # The causal path is deliberately attempted first when it can ask
            # the renderer for a current-frame EEF prediction.  The Viterbi
            # path cannot express that recovery and is substantially more
            # expensive on the full dense anchor set.
            try:
                causal_started_at = time.monotonic()
                selection = select_causally_constrained_poses(
                    current_anchors,
                    source_fps=source_fps,
                    parameters=parameters,
                )
            except TemporalSelectionError as causal_error:
                trace(
                    "causal_failure",
                    attempt=attempt,
                    elapsed_s=round(time.monotonic() - causal_started_at, 3),
                    anchor_index=causal_error.diagnostics.get("anchor_index"),
                    message=str(causal_error),
                )
                augmentation_started_at = time.monotonic()
                augmented = causal_candidate_augmenter(
                    current_anchors, causal_error
                )
                trace(
                    "causal_augmentation",
                    attempt=attempt,
                    elapsed_s=round(time.monotonic() - augmentation_started_at, 3),
                    augmented=augmented is not None,
                )
                if augmented is not None:
                    working = [
                        (
                            anchor,
                            tuple(
                                str(metric.get("source", ""))
                                for metric in anchor.candidate_metrics
                            ),
                        )
                        for anchor in augmented
                    ]
                    continue
                try:
                    viterbi_started_at = time.monotonic()
                    trace("viterbi_start", attempt=attempt)
                    selection = select_temporally_consistent_poses(
                        current_anchors,
                        source_fps=source_fps,
                        parameters=parameters,
                    )
                except TemporalSelectionError as exc:
                    trace(
                        "viterbi_failure",
                        attempt=attempt,
                        elapsed_s=round(time.monotonic() - viterbi_started_at, 3),
                        anchor_index=exc.diagnostics.get("anchor_index"),
                        message=str(exc),
                    )
                    # Both paths remain in the manifest.  Causal failure is
                    # the primary failure here; Viterbi is retained only as a
                    # conservative fallback for legacy evidence.
                    exc.diagnostics["causal_first_failure"] = {
                        "message": str(causal_error),
                        "diagnostics": causal_error.diagnostics,
                    }
                    failure = exc
                else:
                    trace(
                        "viterbi_success",
                        attempt=attempt,
                        elapsed_s=round(time.monotonic() - viterbi_started_at, 3),
                    )
                    return (
                        selection,
                        current_anchors,
                        tuple(value[1] for value in working),
                        pruned,
                    )
            else:
                trace(
                    "causal_success",
                    attempt=attempt,
                    elapsed_s=round(time.monotonic() - causal_started_at, 3),
                )
                return (
                    selection,
                    current_anchors,
                    tuple(value[1] for value in working),
                    pruned,
                )
        else:
            try:
                selection = select_temporally_consistent_poses(
                    current_anchors,
                    source_fps=source_fps,
                    parameters=parameters,
                )
            except TemporalSelectionError as exc:
                try:
                    selection = select_causally_constrained_poses(
                        current_anchors,
                        source_fps=source_fps,
                        parameters=parameters,
                    )
                except TemporalSelectionError as causal_error:
                    # Retain both rejection paths in the manifest.  The causal
                    # fallback is an evidence-preserving recovery, so a failure
                    # must remain auditable rather than silently disappearing.
                    exc.diagnostics["causal_fallback_failure"] = {
                        "message": str(causal_error),
                        "diagnostics": causal_error.diagnostics,
                    }
                    failure = exc
                else:
                    return (
                        selection,
                        current_anchors,
                        tuple(value[1] for value in working),
                        pruned,
                    )
            else:
                return (
                    selection,
                    current_anchors,
                    tuple(value[1] for value in working),
                    pruned,
                )
        if failure is None:
            raise RuntimeError("temporal selection failed without diagnostics")
        anchor_index = failure.diagnostics.get("anchor_index")
        if not isinstance(anchor_index, int) or anchor_index not in range(
            len(working)
        ):
            raise failure
        removal_index = None
        if working[anchor_index][0].ordinal in optional_ordinals:
            removal_index = anchor_index
        elif (
            anchor_index > 0
            and working[anchor_index - 1][0].ordinal in optional_ordinals
        ):
            removal_index = anchor_index - 1
        if removal_index is None:
            trace(
                "terminal_failure",
                attempt=attempt,
                anchor_index=anchor_index,
                message=str(failure),
            )
            raise failure
        failed_anchor_ordinal = working[anchor_index][0].ordinal
        removed_anchor, _ = working.pop(removal_index)
        trace(
            "optional_anchor_pruned",
            attempt=attempt,
            failed_anchor_ordinal=failed_anchor_ordinal,
            removed_anchor_ordinal=removed_anchor.ordinal,
        )
        pruned.append(
            {
                "ordinal": removed_anchor.ordinal,
                "source_frame_index": removed_anchor.source_frame_index,
                "reason": "optional_dense_anchor_blocks_physical_path",
                "failed_anchor_ordinal": failed_anchor_ordinal,
                "detail": str(failure),
            }
        )


def _write_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finite_metric_values(
    metrics: tuple[object, ...], key: str
) -> list[float]:
    values = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        value = metric.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _temporal_anchor_diagnostic(
    anchor: TemporalAnchor,
    parameters: TemporalSelectionParameters,
) -> dict[str, object]:
    metrics = tuple(anchor.candidate_metrics)
    try:
        visual_candidate_count = int(
            np.count_nonzero(
                np.isfinite(
                    temporal_visual_costs(
                        anchor,
                        parameters,
                        require_bidirectional_consensus=True,
                    )
                )
            )
        )
    except ValueError:
        visual_candidate_count = 0
    try:
        static_candidate_count = int(
            np.count_nonzero(
                np.isfinite(temporal_static_visual_costs(anchor, parameters))
            )
        )
    except ValueError:
        static_candidate_count = 0
    precision = _finite_metric_values(metrics, "mask_precision")
    explained = _finite_metric_values(metrics, "mask_explained_fraction")
    raw_precision = _finite_metric_values(metrics, "raw_mask_precision")
    raw_explained = _finite_metric_values(metrics, "raw_mask_explained_fraction")
    overlap = _finite_metric_values(metrics, "depth_overlap_fraction")
    depth_error = _finite_metric_values(metrics, "median_absolute_depth_error_m")
    bidirectional_count = sum(
        isinstance(metric, dict)
        and isinstance(metric.get("bidirectional_consensus"), dict)
        and metric["bidirectional_consensus"].get("passes_gate") is True
        for metric in metrics
    )
    return {
        "ordinal": anchor.ordinal,
        "source_frame_index": anchor.source_frame_index,
        "hand_observed_position": anchor.hand_observed_position.tolist(),
        "hands_open": bool(
            np.all(
                anchor.hand_observed_position
                > parameters.grasp_observed_position_max
            )
        ),
        "primary_laplacian_variance": anchor.primary_laplacian_variance,
        "primary_stereo_consistent_fraction": (
            anchor.primary_stereo_consistent_fraction
        ),
        "candidate_count": len(metrics),
        "bidirectional_candidate_count": bidirectional_count,
        "auditable_visual_candidate_count": visual_candidate_count,
        "static_rgbd_candidate_count": static_candidate_count,
        "best_mask_precision": max(precision, default=None),
        "best_mask_explained_fraction": max(explained, default=None),
        "best_raw_mask_precision": max(raw_precision, default=None),
        "best_raw_mask_explained_fraction": max(raw_explained, default=None),
        "best_depth_overlap_fraction": max(overlap, default=None),
        "best_median_absolute_depth_error_m": min(depth_error, default=None),
    }


def _write_temporal_rejection(
    *,
    output: Path,
    identity: dict[str, object],
    episode_index: int,
    source_revision: str,
    rejection_reason: str,
    error: TemporalSelectionError,
    parameters: TemporalSelectionParameters,
    anchors: tuple[TemporalAnchor, ...],
    candidate_sources: tuple[tuple[str, ...], ...],
    excluded_anchors: list[dict[str, object]],
) -> None:
    if len(candidate_sources) != len(anchors):
        raise ValueError("temporal rejection evidence lengths differ")
    candidate_counts = np.asarray(
        [len(anchor.candidate_poses_root) for anchor in anchors], dtype=np.int64
    )
    candidate_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(candidate_counts))
    )
    candidate_poses = (
        np.concatenate([anchor.candidate_poses_root for anchor in anchors])
        if anchors
        else np.empty((0, 4, 4), dtype=np.float64)
    )
    eef_poses = (
        np.stack([anchor.eef_poses_root for anchor in anchors])
        if anchors
        else np.empty((0, 2, 4, 4), dtype=np.float64)
    )
    rejection = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "episode_index": episode_index,
        "source_revision": source_revision,
        "accepted": False,
        "rejection_reasons": [rejection_reason],
        "gate": {
            "temporal_selection_pass": False,
            "pass": False,
        },
        "temporal_selection_error": {
            "message": str(error),
            "diagnostics": error.diagnostics,
            "parameters": asdict(parameters),
        },
        "temporal_anchor_diagnostics": [
            _temporal_anchor_diagnostic(anchor, parameters) for anchor in anchors
        ],
        "temporal_candidate_evidence": [
            {
                "ordinal": anchor.ordinal,
                "source_frame_index": anchor.source_frame_index,
                "hand_observed_position": anchor.hand_observed_position.tolist(),
                "primary_laplacian_variance": anchor.primary_laplacian_variance,
                "primary_stereo_consistent_fraction": (
                    anchor.primary_stereo_consistent_fraction
                ),
                "candidate_sources": list(candidate_sources[index]),
                "candidate_metrics": [dict(value) for value in anchor.candidate_metrics],
            }
            for index, anchor in enumerate(anchors)
        ],
        "excluded_temporal_anchors": sorted(
            excluded_anchors,
            key=lambda value: int(value["source_frame_index"]),
        ),
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.rejected")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    rejection["arrays"] = {
        "temporal_candidate_poses_root": _write_npy(
            temporary / "temporal_candidate_poses_root.npy", candidate_poses
        ),
        "temporal_candidate_offsets": _write_npy(
            temporary / "temporal_candidate_offsets.npy", candidate_offsets
        ),
        "temporal_eef_poses_root": _write_npy(
            temporary / "temporal_eef_poses_root.npy", eef_poses
        ),
    }
    atomic_write_json(temporary / "manifest.json", rejection)
    os.replace(temporary, output)
    shutil.rmtree(output.with_name(f".{output.name}.debug"), ignore_errors=True)
    print(json.dumps({"output": str(output), "gate": rejection["gate"]}, sort_keys=True))


def _review_image(
    rgb: np.ndarray,
    observed_mask: np.ndarray | None,
    rendered_depth: np.ndarray,
    observed_depth: np.ndarray,
) -> np.ndarray:
    rendered_mask = rendered_depth > 0.0
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tint = overlay.copy()
    tint[rendered_mask] = (40, 210, 40)
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0.0)
    contours, _ = cv2.findContours(
        rendered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (40, 255, 40), 2)
    if observed_mask is not None:
        contours, _ = cv2.findContours(
            observed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 80, 220), 2)

    valid = (rendered_depth > 0.0) & (observed_depth > 0.0)
    error = np.zeros_like(rendered_depth, dtype=np.float32)
    error[valid] = np.abs(rendered_depth[valid] - observed_depth[valid])
    heat = cv2.applyColorMap(
        np.clip(error / 0.05 * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heat[~valid] = 0
    return np.concatenate((overlay, heat), axis=1)


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    pose_config = config.object_pose_runtime
    input_root = args.input_dir.expanduser().resolve()
    mask_root = args.mask_dir.expanduser().resolve()
    runtime_root = args.runtime_root.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    input_manifest_path = input_root / "manifest.json"
    mask_manifest_path = mask_root / "manifest.json"
    source_runtime_path = runtime_root / "runtime-manifest.json"
    compiled_runtime_path = runtime_root / "compiled-runtime-manifest.json"
    input_manifest = read_json_object(input_manifest_path)
    mask_manifest = read_json_object(mask_manifest_path)
    source_runtime = read_json_object(source_runtime_path)
    compiled_runtime = read_json_object(compiled_runtime_path)
    if input_manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported prepared RGB-D input schema")
    initial_root_from_table, initial_seed_evidence = load_initial_root_from_table(
        args.initial_root_from_table,
        episode_index=int(input_manifest.get("episode_index", -1)),
    )
    if mask_manifest.get("schema_version") != SEGMENTATION_SCHEMA_VERSION:
        raise ValueError("unsupported table-mask schema")
    if source_runtime.get("schema_version") != SOURCE_RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported object-pose source runtime schema")
    if compiled_runtime.get("schema_version") != COMPILED_RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported compiled object-pose runtime schema")
    if source_runtime.get("config_sha256") != config.digest:
        raise ValueError("object-pose source runtime uses a different config")
    if compiled_runtime.get("config_sha256") != config.digest:
        raise ValueError("compiled object-pose runtime uses a different config")
    if compiled_runtime.get("container_digest") != config.runtime.container_digest:
        raise ValueError("compiled object-pose runtime uses a different V1 image")
    if compiled_runtime.get("source_runtime_manifest_sha256") != sha256_file(
        source_runtime_path
    ):
        raise ValueError("compiled runtime does not match the object-pose source runtime")
    if mask_manifest.get("config_sha256") != config.digest:
        raise ValueError("table masks use a different config")
    if mask_manifest.get("input_manifest_sha256") != sha256_file(input_manifest_path):
        raise ValueError("table masks use different prepared RGB-D input")
    if mask_manifest.get("runtime_manifest_sha256") != sha256_file(source_runtime_path):
        raise ValueError("table masks use a different object-pose source runtime")
    if not bool(dict(mask_manifest.get("gate", {})).get("pass")):
        raise ValueError("table-mask quality gate did not pass")
    if args.dense_propagation_beam_size is not None:
        if args.dense_propagation_beam_size <= 0:
            raise ValueError("dense propagation beam size must be positive")
        pose_config = replace(
            pose_config,
            dense_temporal_propagation_beam_size=(
                args.dense_propagation_beam_size
            ),
        )
    if sha256_file(mesh_path) != pose_config.assembled_table_mesh_sha256:
        raise ValueError("assembled V1 table mesh hash differs from the pinned mesh")

    identity = {
        "config_sha256": config.digest,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "mask_manifest_sha256": sha256_file(mask_manifest_path),
        "source_runtime_manifest_sha256": sha256_file(source_runtime_path),
        "compiled_runtime_manifest_sha256": sha256_file(compiled_runtime_path),
        "assembled_table_mesh_sha256": sha256_file(mesh_path),
        "initial_root_from_table_seed": initial_seed_evidence,
        "tracking_overrides": {
            "dense_propagation_beam_size": (
                args.dense_propagation_beam_size
            )
        },
    }
    output_manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume or not output_manifest_path.is_file():
            raise FileExistsError(f"output already exists: {output}")
        previous = read_json_object(output_manifest_path)
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("existing table track uses a different schema")
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing table track was generated from a different contract")
        print(
            json.dumps(
                {"output": str(output), "resumed": True, "gate": previous["gate"]},
                sort_keys=True,
            )
        )
        if not bool(dict(previous["gate"])["pass"]):
            raise SystemExit(2)
        return

    records = input_manifest.get("frames")
    source_frame_indices = np.asarray(input_manifest.get("source_frame_indices"), dtype=np.int64)
    if not isinstance(records, list) or len(records) != len(source_frame_indices):
        raise ValueError("prepared RGB-D records differ from sampled frame indices")
    if len(records) < 2 or source_frame_indices[0] != 0:
        raise ValueError("prepared RGB-D track must include the first frame")
    source_frame_count = int(input_manifest["source_frame_count"])
    if source_frame_indices[-1] != source_frame_count - 1:
        raise ValueError("prepared RGB-D track must include the last frame")
    pose_views = input_manifest.get("pose_views")
    if (
        not isinstance(pose_views, dict)
        or len(pose_views) != len(POSE_VIEW_NAMES)
        or set(pose_views) != set(POSE_VIEW_NAMES)
    ):
        raise ValueError("prepared input lacks the ordered three-view contract")
    intrinsic_matrices = {
        view_name: np.asarray(
            pose_views[view_name]["intrinsic_matrix_px"], dtype=np.float64
        ).reshape(3, 3)
        for view_name in POSE_VIEW_NAMES
    }
    if any(
        not np.isfinite(value).all() or not np.allclose(value[2], (0.0, 0.0, 1.0))
        for value in intrinsic_matrices.values()
    ):
        raise ValueError("invalid rectified camera intrinsic matrix")
    intrinsic_matrix = intrinsic_matrices[PRIMARY_POSE_VIEW]
    frames = []
    stereo_consistency_masks = []
    root_from_cameras = {view_name: [] for view_name in POSE_VIEW_NAMES}
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict) or int(record.get("ordinal", -1)) != ordinal:
            raise ValueError("prepared RGB-D ordinals must be contiguous")
        if int(record.get("source_frame_index", -1)) != int(source_frame_indices[ordinal]):
            raise ValueError("prepared RGB-D frame index differs from manifest index")
        rgb, depth, stereo_consistency = _load_frame(input_root, record)
        frames.append((rgb, depth))
        stereo_consistency_masks.append(stereo_consistency)
        views = record.get("views")
        if (
            not isinstance(views, dict)
            or len(views) != len(POSE_VIEW_NAMES)
            or set(views) != set(POSE_VIEW_NAMES)
        ):
            raise ValueError(f"prepared frame {ordinal} lacks the three-view contract")
        for view_name in POSE_VIEW_NAMES:
            root_from_cameras[view_name].append(
                _as_transform(
                    views[view_name]["robot_root_from_rectified_opencv"],
                    f"root_from_camera[{view_name}][{ordinal}]",
                )
            )
    root_from_cameras_array = {
        view_name: np.asarray(values, dtype=np.float64)
        for view_name, values in root_from_cameras.items()
    }
    masks_by_view = _load_masks(mask_root, mask_manifest)
    dense_masks_by_view = _load_dense_masks(
        mask_root,
        mask_manifest,
        masks_by_view,
        frame_count=len(records),
        minimum_bidirectional_iou=(
            pose_config.dense_mask_min_bidirectional_iou
        ),
        minimum_area_fraction=pose_config.dense_mask_min_area_fraction,
        maximum_area_fraction=pose_config.dense_mask_max_area_fraction,
    )
    sparse_registration_masks = masks_by_view[PRIMARY_POSE_VIEW]
    if 0 not in sparse_registration_masks:
        raise ValueError("the first tracking frame requires an audited registration mask")
    registration_masks, promoted_terminal_ordinals, terminal_confirmation_ordinal = (
        _with_terminal_confirmation_registration(
            sparse_registration_masks,
            dense_masks_by_view[PRIMARY_POSE_VIEW],
            source_frame_indices,
            maximum_evidence_source_frame_gap=(
                pose_config.maximum_temporal_evidence_gap_source_frames
            ),
            maximum_confirmation_source_frame_gap=(
                pose_config.maximum_terminal_tracking_gap_source_frames
            ),
        )
    )
    wrist_bridge_ordinals = sorted(
        (
            set(masks_by_view["left_wrist"])
            & set(masks_by_view["right_wrist"])
        )
        - set(sparse_registration_masks)
    )
    tracking_eligible_ordinals = sorted(
        set(sparse_registration_masks) | set(wrist_bridge_ordinals)
    )
    last_registration_ordinal = max(registration_masks)
    primary_terminal_gap_source_frames = int(
        source_frame_indices[-1] - source_frame_indices[last_registration_ordinal]
    )
    if (
        primary_terminal_gap_source_frames
        > pose_config.maximum_terminal_tracking_gap_source_frames
    ):
        raise ValueError("the final registration anchor is too far from the episode endpoint")
    last_tracking_eligible_ordinal = tracking_eligible_ordinals[-1]
    terminal_gap_source_frames = int(
        source_frame_indices[-1]
        - source_frame_indices[last_tracking_eligible_ordinal]
    )
    mask_gate = dict(mask_manifest["gate"])
    if (
        mask_gate.get("primary_selected_ordinals")
        != sorted(sparse_registration_masks)
        or mask_gate.get("bimanual_wrist_bridge_ordinals")
        != wrist_bridge_ordinals
        or mask_gate.get("tracking_eligible_ordinals")
        != tracking_eligible_ordinals
        or int(mask_gate.get("last_primary_selected_ordinal", -1))
        != last_registration_ordinal
        or int(mask_gate.get("primary_terminal_gap_source_frames", -1))
        != primary_terminal_gap_source_frames
        or int(mask_gate.get("last_selected_ordinal", -1))
        != last_tracking_eligible_ordinal
        or int(mask_gate.get("terminal_tracking_gap_source_frames", -1))
        != terminal_gap_source_frames
    ):
        raise ValueError("table-mask tracking eligibility evidence is inconsistent")

    foundationpose_root = runtime_root / "FoundationPose"
    sys.path.insert(0, str(foundationpose_root / "mycpp" / "build"))
    sys.path.insert(0, str(foundationpose_root))
    try:
        import nvdiffrast.torch as dr
        import torch
        import trimesh
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        from Utils import make_mesh_tensors, nvdiffrast_render, set_seed
    except ImportError as exc:
        raise RuntimeError("compiled FoundationPose runtime is incomplete") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose tracking requires CUDA")
    set_seed(0)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if len(mesh.vertices) != pose_config.assembled_table_mesh_vertices:
        raise ValueError("assembled table mesh vertex count differs")
    if len(mesh.faces) != pose_config.assembled_table_mesh_triangles:
        raise ValueError("assembled table mesh triangle count differs")
    symmetry_transforms = np.stack(table_symmetry_transforms())
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        symmetry_tfs=symmetry_transforms,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=glctx,
        debug=0,
        debug_dir=str(output.with_name(f".{output.name}.debug")),
    )
    render_mesh_tensors = make_mesh_tensors(mesh)
    registration_selector = _RegistrationGeometrySelector(
        intrinsic_matrix=intrinsic_matrix,
        mesh_tensors=render_mesh_tensors,
        glctx=glctx,
        render_function=nvdiffrast_render,
        torch_module=torch,
        depth_consistency_m=pose_config.max_rendered_depth_median_abs_error_m,
        minimum_mask_explained_fraction=(
            pose_config.min_rendered_mask_explained_fraction
        ),
        max_correction_translation_m=(
            pose_config.max_registration_correction_translation_m
        ),
        max_correction_rotation_rad=pose_config.max_registration_correction_rotation_rad,
        propagation_beam_size=pose_config.temporal_propagation_beam_size,
        auxiliary_intrinsics={
            view_name: intrinsic_matrices[view_name]
            for view_name in POSE_VIEW_NAMES[1:]
        },
        auxiliary_view_score_weight=pose_config.auxiliary_view_score_weight,
        auxiliary_primary_support_saturation_fraction=(
            pose_config.auxiliary_primary_support_saturation_fraction
        ),
    )

    eef_poses_root_array = np.stack(
        [_current_eef_poses_from_record(record) for record in records]
    )
    hand_observed_positions_array = np.stack(
        [np.asarray(record.get("hand_state"), dtype=np.float64) for record in records]
    )
    if (
        hand_observed_positions_array.shape != (len(records), 2)
        or not np.isfinite(hand_observed_positions_array).all()
    ):
        raise ValueError("prepared frames lack finite observed Dex1 positions")
    image_laplacian_variances = np.asarray(
        [
            record["views"][PRIMARY_POSE_VIEW]["laplacian_variance"]
            for record in records
        ],
        dtype=np.float64,
    )
    if (
        image_laplacian_variances.shape != (len(records),)
        or not np.isfinite(image_laplacian_variances).all()
        or np.any(image_laplacian_variances <= 0.0)
    ):
        raise ValueError("prepared frames lack finite positive image sharpness")
    temporal_parameters = _temporal_parameters(pose_config)
    stereo_consistency_fractions = np.ones(len(frames), dtype=np.float64)
    for ordinal, observed_mask in dense_masks_by_view[PRIMARY_POSE_VIEW].items():
        valid_mask_depth = observed_mask & (frames[ordinal][1] > 0.0)
        valid_count = int(np.count_nonzero(valid_mask_depth))
        stereo_consistency_fractions[ordinal] = (
            float(
                np.count_nonzero(
                    stereo_consistency_masks[ordinal] & valid_mask_depth
                )
                / valid_count
            )
            if valid_count
            else 0.0
        )

    forward_registration_evidence: dict[int, dict[str, object]] = {}
    _track_trace("forward_tracking_start", frame_count=len(frames))
    forward_started_at = time.monotonic()
    forward_root, forward_modes, forward_rotation_corrections = _track_direction(
        estimator=estimator,
        frames=frames,
        intrinsic_matrix=intrinsic_matrix,
        root_from_cameras=root_from_cameras_array[PRIMARY_POSE_VIEW],
        registration_masks=registration_masks,
        order=range(len(frames)),
        registration_iterations=pose_config.registration_refine_iterations,
        tracking_iterations=pose_config.tracking_refine_iterations,
        registration_selector=registration_selector,
        registration_evidence=forward_registration_evidence,
        initial_root_from_object=initial_root_from_table,
        eef_poses_root=eef_poses_root_array,
        hand_observed_positions=hand_observed_positions_array,
        image_laplacian_variances=image_laplacian_variances,
        stereo_consistency_fractions=stereo_consistency_fractions,
        temporal_parameters=temporal_parameters,
        grasp_observed_position_max=(
            pose_config.temporal_grasp_observed_position_max
        ),
        auxiliary_root_from_cameras={
            view_name: root_from_cameras_array[view_name]
            for view_name in POSE_VIEW_NAMES[1:]
        },
        auxiliary_registration_masks={
            view_name: masks_by_view[view_name] for view_name in POSE_VIEW_NAMES[1:]
        },
    )
    _track_trace(
        "forward_tracking_complete",
        elapsed_s=round(time.monotonic() - forward_started_at, 3),
        registration_count=len(forward_registration_evidence),
    )
    backward_registration_evidence: dict[int, dict[str, object]] = {}
    _track_trace(
        "backward_tracking_start", frame_count=last_registration_ordinal + 1
    )
    backward_started_at = time.monotonic()
    backward_root, backward_modes, backward_rotation_corrections = _track_direction(
        estimator=estimator,
        frames=frames[: last_registration_ordinal + 1],
        intrinsic_matrix=intrinsic_matrix,
        root_from_cameras=root_from_cameras_array[PRIMARY_POSE_VIEW][
            : last_registration_ordinal + 1
        ],
        registration_masks={
            ordinal: mask
            for ordinal, mask in registration_masks.items()
            if ordinal <= last_registration_ordinal
        },
        order=range(last_registration_ordinal, -1, -1),
        registration_iterations=pose_config.registration_refine_iterations,
        tracking_iterations=pose_config.tracking_refine_iterations,
        registration_selector=registration_selector,
        registration_evidence=backward_registration_evidence,
        # Register the terminal frame independently. Seeding the reverse pass
        # with the forward pass's selected pose makes terminal consensus
        # circular and can suppress a physically correct endpoint hypothesis.
        eef_poses_root=eef_poses_root_array[: last_registration_ordinal + 1],
        hand_observed_positions=hand_observed_positions_array[
            : last_registration_ordinal + 1
        ],
        image_laplacian_variances=image_laplacian_variances[
            : last_registration_ordinal + 1
        ],
        stereo_consistency_fractions=stereo_consistency_fractions[
            : last_registration_ordinal + 1
        ],
        temporal_parameters=temporal_parameters,
        grasp_observed_position_max=(
            pose_config.temporal_grasp_observed_position_max
        ),
        auxiliary_root_from_cameras={
            view_name: root_from_cameras_array[view_name][
                : last_registration_ordinal + 1
            ]
            for view_name in POSE_VIEW_NAMES[1:]
        },
        auxiliary_registration_masks={
            view_name: {
                ordinal: mask
                for ordinal, mask in masks_by_view[view_name].items()
                if ordinal <= last_registration_ordinal
            }
            for view_name in POSE_VIEW_NAMES[1:]
        },
    )
    _track_trace(
        "backward_tracking_complete",
        elapsed_s=round(time.monotonic() - backward_started_at, 3),
        registration_count=len(backward_registration_evidence),
    )
    consensus_started_at = time.monotonic()
    _track_trace(
        "bidirectional_consensus_start",
        anchor_count=len(forward_registration_evidence),
    )
    for consensus_index, (ordinal, forward_evidence) in enumerate(
        sorted(forward_registration_evidence.items())
    ):
        if consensus_index % 25 == 0:
            _track_trace(
                "bidirectional_consensus_progress",
                completed=consensus_index,
                total=len(forward_registration_evidence),
                ordinal=ordinal,
            )
        forward_metrics = forward_evidence.get("temporal_candidate_metrics")
        forward_poses = np.asarray(
            forward_evidence.get("temporal_candidate_poses_root"), dtype=np.float64
        )
        if not isinstance(forward_metrics, list) or len(forward_metrics) != len(
            forward_poses
        ):
            raise ValueError(
                f"forward registration evidence is incomplete at ordinal {ordinal}"
            )
        record = records[ordinal]
        common_anchor = {
            "ordinal": ordinal,
            "source_frame_index": int(source_frame_indices[ordinal]),
            "eef_poses_root": _current_eef_poses_from_record(record),
            "hand_observed_position": np.asarray(
                record.get("hand_state"), dtype=np.float64
            ),
            "primary_laplacian_variance": float(
                record["views"][PRIMARY_POSE_VIEW]["laplacian_variance"]
            ),
            "primary_stereo_consistent_fraction": float(
                stereo_consistency_fractions[ordinal]
            ),
        }
        forward_anchor = TemporalAnchor(
            candidate_poses_root=forward_poses,
            candidate_metrics=tuple(forward_metrics),
            **common_anchor,
        )
        backward_evidence = backward_registration_evidence.get(ordinal)
        if backward_evidence is None:
            if ordinal != last_registration_ordinal:
                raise ValueError(
                    f"reverse registration evidence is missing at ordinal {ordinal}"
                )
            backward_anchor = None
            backward_reference = backward_root[ordinal]
        else:
            backward_metrics = backward_evidence.get("temporal_candidate_metrics")
            backward_poses = np.asarray(
                backward_evidence.get("temporal_candidate_poses_root"),
                dtype=np.float64,
            )
            if not isinstance(backward_metrics, list) or len(backward_metrics) != len(
                backward_poses
            ):
                raise ValueError(
                    f"reverse registration evidence is incomplete at ordinal {ordinal}"
                )
            backward_anchor = TemporalAnchor(
                candidate_poses_root=backward_poses,
                candidate_metrics=tuple(backward_metrics),
                **common_anchor,
            )
            backward_reference = None
        consensus = _bidirectional_consensus_records(
            forward_anchor=forward_anchor,
            backward_anchor=backward_anchor,
            backward_reference_pose_root=backward_reference,
            parameters=temporal_parameters,
            maximum_translation_error_m=(
                pose_config.max_bidirectional_translation_error_m
            ),
            maximum_rotation_error_rad=(
                pose_config.max_bidirectional_rotation_error_rad
            ),
        )
        for metric, consensus_record in zip(
            forward_metrics, consensus, strict=True
        ):
            if not isinstance(metric, dict):
                raise ValueError("forward temporal candidate metric is malformed")
            metric["bidirectional_consensus"] = consensus_record
    _track_trace(
        "bidirectional_consensus_complete",
        elapsed_s=round(time.monotonic() - consensus_started_at, 3),
    )
    fusion_started_at = time.monotonic()
    _track_trace("bidirectional_fusion_start")
    bidirectional_fused_root, translation_errors, rotation_errors, selected_symmetries = (
        fuse_bidirectional_poses(
            forward_root[: last_registration_ordinal + 1], backward_root
        )
    )
    _track_trace(
        "bidirectional_fusion_complete",
        elapsed_s=round(time.monotonic() - fusion_started_at, 3),
    )

    dense_forward_evidence = {}
    dense_backward_evidence = {}
    dense_propagation_beam_size = pose_config.dense_temporal_propagation_beam_size
    sparse_primary_ordinals = sorted(registration_masks)
    _track_trace(
        "dense_propagation_start",
        interval_count=max(0, len(sparse_primary_ordinals) - 1),
        beam_size=dense_propagation_beam_size,
    )
    for left, right in zip(sparse_primary_ordinals, sparse_primary_ordinals[1:]):
        targets = tuple(
            ordinal
            for ordinal in sorted(dense_masks_by_view[PRIMARY_POSE_VIEW])
            if left < ordinal < right
        )
        if not targets or right not in backward_registration_evidence:
            continue
        _track_trace(
            "dense_interval_start",
            left_ordinal=left,
            right_ordinal=right,
            target_count=len(targets),
        )
        forward_interval = _dense_directional_candidate_evidence(
            start_ordinal=left,
            target_ordinals=(*targets, right),
            start_evidence=forward_registration_evidence[left],
            direction="forward",
            registration_selector=registration_selector,
            frames=frames,
            dense_masks_by_view=dense_masks_by_view,
            root_from_cameras=root_from_cameras_array,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
            source_frame_indices=source_frame_indices,
            temporal_parameters=temporal_parameters,
            propagation_beam_size=dense_propagation_beam_size,
        )
        backward_interval = _dense_directional_candidate_evidence(
            start_ordinal=right,
            target_ordinals=tuple(reversed(targets)),
            start_evidence=backward_registration_evidence[right],
            direction="reverse",
            registration_selector=registration_selector,
            frames=frames,
            dense_masks_by_view=dense_masks_by_view,
            root_from_cameras=root_from_cameras_array,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
            source_frame_indices=source_frame_indices,
            temporal_parameters=temporal_parameters,
            propagation_beam_size=dense_propagation_beam_size,
        )
        if set(forward_interval) != set(targets) | {right} or set(
            backward_interval
        ) != set(targets):
            raise RuntimeError("dense interval pose-beam coverage differs")
        for ordinal in targets:
            _attach_dense_bidirectional_consensus(
                forward_interval[ordinal],
                backward_interval[ordinal],
                ordinal=ordinal,
                source_frame_indices=source_frame_indices,
                eef_poses_root=eef_poses_root_array,
                hand_observed_positions=hand_observed_positions_array,
                image_laplacian_variances=image_laplacian_variances,
                stereo_consistency_fractions=stereo_consistency_fractions,
                parameters=temporal_parameters,
                maximum_translation_error_m=(
                    pose_config.max_bidirectional_translation_error_m
                ),
                maximum_rotation_error_rad=(
                    pose_config.max_bidirectional_rotation_error_rad
                ),
            )
            dense_forward_evidence[ordinal] = forward_interval[ordinal]
            dense_backward_evidence[ordinal] = backward_interval[ordinal]
            forward_registration_evidence[ordinal] = forward_interval[ordinal]
            backward_registration_evidence[ordinal] = backward_interval[ordinal]

        endpoint_evidence = forward_interval[right]
        _attach_dense_bidirectional_consensus(
            endpoint_evidence,
            backward_registration_evidence[right],
            ordinal=right,
            source_frame_indices=source_frame_indices,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
            parameters=temporal_parameters,
            maximum_translation_error_m=(
                pose_config.max_bidirectional_translation_error_m
            ),
            maximum_rotation_error_rad=(
                pose_config.max_bidirectional_rotation_error_rad
            ),
        )
        endpoint_anchor = _registration_evidence_anchor(
            forward_registration_evidence[right],
            ordinal=right,
            source_frame_indices=source_frame_indices,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
        )
        _merge_dense_endpoint_evidence(
            forward_registration_evidence[right],
            endpoint_evidence,
            anchor=endpoint_anchor,
            parameters=temporal_parameters,
            propagation_beam_size=dense_propagation_beam_size,
        )
    terminal_reverse_candidate_count = 0
    if terminal_confirmation_ordinal < last_registration_ordinal:
        terminal_backward_evidence = backward_registration_evidence[
            last_registration_ordinal
        ]
        _attach_dense_bidirectional_consensus(
            terminal_backward_evidence,
            forward_registration_evidence[last_registration_ordinal],
            ordinal=last_registration_ordinal,
            source_frame_indices=source_frame_indices,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
            parameters=temporal_parameters,
            maximum_translation_error_m=(
                pose_config.max_bidirectional_translation_error_m
            ),
            maximum_rotation_error_rad=(
                pose_config.max_bidirectional_rotation_error_rad
            ),
        )
        terminal_reverse = _dense_directional_candidate_evidence(
            start_ordinal=last_registration_ordinal,
            target_ordinals=(terminal_confirmation_ordinal,),
            start_evidence=terminal_backward_evidence,
            direction="reverse",
            registration_selector=registration_selector,
            frames=frames,
            dense_masks_by_view=dense_masks_by_view,
            root_from_cameras=root_from_cameras_array,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
            source_frame_indices=source_frame_indices,
            temporal_parameters=temporal_parameters,
            propagation_beam_size=dense_propagation_beam_size,
        )[terminal_confirmation_ordinal]
        _attach_terminal_reverse_lineage_consensus(
            terminal_reverse,
            terminal_backward_evidence,
            terminal_ordinal=last_registration_ordinal,
            maximum_translation_error_m=(
                pose_config.max_bidirectional_translation_error_m
            ),
            maximum_rotation_error_rad=(
                pose_config.max_bidirectional_rotation_error_rad
            ),
        )
        terminal_reverse_candidate_count = int(
            terminal_reverse["candidate_count"]
        )
        terminal_anchor = _registration_evidence_anchor(
            forward_registration_evidence[terminal_confirmation_ordinal],
            ordinal=terminal_confirmation_ordinal,
            source_frame_indices=source_frame_indices,
            eef_poses_root=eef_poses_root_array,
            hand_observed_positions=hand_observed_positions_array,
            image_laplacian_variances=image_laplacian_variances,
            stereo_consistency_fractions=stereo_consistency_fractions,
        )
        _merge_dense_endpoint_evidence(
            forward_registration_evidence[terminal_confirmation_ordinal],
            terminal_reverse,
            anchor=terminal_anchor,
            parameters=temporal_parameters,
            propagation_beam_size=dense_propagation_beam_size,
        )
        forward_registration_evidence[terminal_confirmation_ordinal][
            "terminal_reverse_static_candidate_count"
        ] = terminal_reverse_candidate_count
    dense_primary_ordinals = sorted(dense_forward_evidence)

    for ordinal in wrist_bridge_ordinals:
        if ordinal > last_registration_ordinal or ordinal in forward_registration_evidence:
            continue
        rendered_by_view = registration_selector.render_root_pose_depths(
            bidirectional_fused_root[ordinal],
            {
                view_name: root_from_cameras_array[view_name][ordinal]
                for view_name in POSE_VIEW_NAMES
            },
        )
        metric = evaluate_unsegmented_multiview_depths(
            rendered_depth_m=rendered_by_view[PRIMARY_POSE_VIEW],
            observed_depth_m=frames[ordinal][1],
            stereo_consistency_mask=stereo_consistency_masks[ordinal],
            auxiliary_rendered_depths_m={
                view_name: rendered_by_view[view_name]
                for view_name in POSE_VIEW_NAMES[1:]
            },
            auxiliary_observed_masks={
                view_name: masks_by_view[view_name][ordinal]
                for view_name in POSE_VIEW_NAMES[1:]
            },
            source="bidirectional_tracking_bridge",
            maximum_consistent_depth_error_m=(
                pose_config.max_rendered_depth_median_abs_error_m
            ),
            auxiliary_view_score_weight=pose_config.auxiliary_view_score_weight,
        )
        metric_json = metric.to_json()
        metric_json["bidirectional_consensus"] = {
            "passes_gate": bool(
                translation_errors[ordinal]
                <= pose_config.max_bidirectional_translation_error_m
                and rotation_errors[ordinal]
                <= pose_config.max_bidirectional_rotation_error_rad
            ),
            "translation_error_m": float(translation_errors[ordinal]),
            "rotation_error_rad": float(rotation_errors[ordinal]),
            "maximum_translation_error_m": (
                pose_config.max_bidirectional_translation_error_m
            ),
            "maximum_rotation_error_rad": (
                pose_config.max_bidirectional_rotation_error_rad
            ),
            "validation_mode": "bidirectional_tracking_bridge",
            "reverse_candidate_index": None,
            "reverse_candidate_source": "reverse_tracking",
            "reverse_symmetry_index": int(selected_symmetries[ordinal]),
        }
        forward_registration_evidence[ordinal] = {
            "candidate_count": 1,
            "selected_candidate_index": 0,
            "selected_source": "bidirectional_tracking_bridge",
            "selection_reason": "missing_head_mask_multiview_audit",
            "temporal_candidate_poses_root": [
                bidirectional_fused_root[ordinal].tolist()
            ],
            "temporal_candidate_metrics": [metric_json],
            "temporal_candidate_sources": ["bidirectional_tracking_bridge"],
        }
        stereo_consistency_fractions[ordinal] = float(
            metric.primary_stereo_consistent_fraction
        )

    candidate_temporal_anchors = []
    candidate_temporal_sources = []
    for ordinal in sorted(forward_registration_evidence):
        evidence = forward_registration_evidence[ordinal]
        all_candidate_poses = np.asarray(
            evidence.get("temporal_candidate_poses_root"), dtype=np.float64
        )
        all_candidate_metrics = evidence.get("temporal_candidate_metrics")
        all_candidate_sources = evidence.get("temporal_candidate_sources")
        solver_indices = _temporal_solver_candidate_indices(evidence)
        hand_observed_position = np.asarray(
            records[ordinal].get("hand_state"), dtype=np.float64
        )
        if (
            all_candidate_poses.ndim != 3
            or all_candidate_poses.shape[1:] != (4, 4)
            or not isinstance(all_candidate_metrics, list)
            or not isinstance(all_candidate_sources, list)
            or len(all_candidate_metrics) != len(all_candidate_poses)
            or len(all_candidate_sources) != len(all_candidate_poses)
            or any(not isinstance(metric, dict) for metric in all_candidate_metrics)
            or hand_observed_position.shape != (2,)
            or not np.isfinite(hand_observed_position).all()
        ):
            raise ValueError(f"registration evidence is incomplete at ordinal {ordinal}")
        candidate_poses = all_candidate_poses[np.asarray(solver_indices, dtype=np.int64)]
        candidate_metrics = tuple(
            {
                **dict(all_candidate_metrics[index]),
                "temporal_solver_original_candidate_index": index,
            }
            for index in solver_indices
        )
        candidate_sources = tuple(
            str(all_candidate_sources[index]) for index in solver_indices
        )
        candidate_temporal_anchors.append(
            TemporalAnchor(
                ordinal=ordinal,
                source_frame_index=int(source_frame_indices[ordinal]),
                candidate_poses_root=candidate_poses,
                candidate_metrics=candidate_metrics,
                eef_poses_root=_current_eef_poses_from_record(records[ordinal]),
                hand_observed_position=hand_observed_position,
                primary_laplacian_variance=float(
                    records[ordinal]["views"][PRIMARY_POSE_VIEW]["laplacian_variance"]
                ),
                primary_stereo_consistent_fraction=float(
                    stereo_consistency_fractions[ordinal]
                ),
            )
        )
        candidate_temporal_sources.append(
            candidate_sources
        )

    all_temporal_anchors = tuple(candidate_temporal_anchors)
    all_temporal_candidate_sources = tuple(candidate_temporal_sources)
    visual_eligible = []
    excluded_temporal_anchors = []
    for anchor, sources in zip(
        candidate_temporal_anchors, candidate_temporal_sources, strict=True
    ):
        try:
            temporal_visual_costs(
                anchor,
                temporal_parameters,
                require_bidirectional_consensus=True,
            )
        except ValueError as exc:
            excluded_temporal_anchors.append(
                {
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "reason": "no_auditable_visual_candidate",
                    "detail": str(exc),
                    "primary_stereo_consistent_fraction": (
                        anchor.primary_stereo_consistent_fraction
                    ),
                }
            )
        else:
            visual_eligible.append((anchor, sources))
    try:
        initial_static_selection = _select_initial_static_anchor(
            visual_eligible, temporal_parameters
        )
        if initial_static_selection is None:
            raise TemporalSelectionError(
                "no open-hand static RGB-D anchor can initialize the table pose",
                {
                    "stage": "initialization",
                    "rejection_reason": "temporal_initial_static_seed_missing",
                    "visual_eligible_anchor_count": len(visual_eligible),
                },
            )
        initial_index, initial_static_selection_mode = initial_static_selection
        initial_anchor = visual_eligible[initial_index][0]
        if (
            initial_anchor.source_frame_index
            > pose_config.maximum_initial_static_backfill_source_frames
        ):
            raise TemporalSelectionError(
                "the first reliable static table pose is too late to backfill",
                {
                    "stage": "initialization",
                    "rejection_reason": "temporal_initial_static_seed_too_late",
                    "first_reliable_source_frame_index": (
                        initial_anchor.source_frame_index
                    ),
                    "maximum_allowed_source_frame_index": (
                        pose_config.maximum_initial_static_backfill_source_frames
                    ),
                },
            )
        preceding_closed_frames = [
            int(source_frame_indices[index])
            for index, record in enumerate(records[: initial_anchor.ordinal])
            if np.any(
                np.asarray(record["hand_state"], dtype=np.float64)
                <= temporal_parameters.grasp_observed_position_max
            )
        ]
        if preceding_closed_frames:
            raise TemporalSelectionError(
                "initial static backfill crosses an observed hand closure",
                {
                    "stage": "initialization",
                    "rejection_reason": "temporal_initial_backfill_crosses_grasp",
                    "first_reliable_source_frame_index": (
                        initial_anchor.source_frame_index
                    ),
                    "preceding_closed_source_frame_indices": preceding_closed_frames,
                },
            )
        for anchor, _ in visual_eligible[:initial_index]:
            excluded_temporal_anchors.append(
                {
                    "ordinal": anchor.ordinal,
                    "source_frame_index": anchor.source_frame_index,
                    "reason": "before_first_reliable_static_seed",
                    "primary_stereo_consistent_fraction": (
                        anchor.primary_stereo_consistent_fraction
                    ),
                }
            )
        selected_temporal = visual_eligible[initial_index:]
        active_hand_ordinals = np.flatnonzero(
            np.any(
                hand_observed_positions_array
                <= temporal_parameters.grasp_observed_position_max,
                axis=1,
            )
        )
        if len(active_hand_ordinals) == 0:
            raise TemporalSelectionError(
                "source trajectory has no observed table grasp",
                {
                    "stage": "lifecycle",
                    "rejection_reason": "observed_table_grasp_missing",
                },
        )
        last_active_hand_ordinal = int(active_hand_ordinals[-1])
        try:
            final_static_ordinals = _select_terminal_static_anchor_ordinals(
                selected_temporal,
                temporal_parameters,
                last_active_hand_ordinal=last_active_hand_ordinal,
                terminal_source_frame_index=int(source_frame_indices[-1]),
                maximum_terminal_gap_source_frames=(
                    pose_config.maximum_terminal_tracking_gap_source_frames
                ),
            )
            (
                selected_temporal,
                free_settle_exclusions,
                release_anchor_ordinal,
            ) = _retain_manipulation_and_final_static_anchors(
                selected_temporal,
                last_active_hand_ordinal=last_active_hand_ordinal,
                final_static_ordinals=final_static_ordinals,
                grasp_observed_position_max=(
                    temporal_parameters.grasp_observed_position_max
                ),
            )
        except ValueError as exc:
            raise TemporalSelectionError(
                str(exc),
                {
                    "stage": "lifecycle",
                    "rejection_reason": "release_final_static_evidence_missing",
                    "last_active_hand_ordinal": last_active_hand_ordinal,
                    "terminal_confirmation_ordinal": (
                        terminal_confirmation_ordinal
                    ),
                    "last_registration_ordinal": last_registration_ordinal,
                    "maximum_terminal_tracking_gap_source_frames": (
                        pose_config.maximum_terminal_tracking_gap_source_frames
                    ),
                },
            ) from exc
        excluded_temporal_anchors.extend(free_settle_exclusions)
        initial_temporal_anchors = tuple(value[0] for value in selected_temporal)
        initial_temporal_sources = tuple(value[1] for value in selected_temporal)
        if len(initial_temporal_anchors) < 4:
            raise TemporalSelectionError(
                "fewer than four auditable temporal anchors remain",
                {
                    "stage": "evidence_coverage",
                    "rejection_reason": "temporal_anchor_count_insufficient",
                    "anchor_count": len(initial_temporal_anchors),
                    "required_anchor_count": 4,
                    "source_frame_indices": [
                        anchor.source_frame_index
                        for anchor in initial_temporal_anchors
                    ],
                },
            )
        optional_ordinals = set(dense_primary_ordinals) | set(
            wrist_bridge_ordinals
        )
        # An optional-anchor retry can rediscover the same FK propagation at a
        # frame whose current RGB-D evidence already rejected it.  Cache only
        # that exact frame/source/pose triple for this selection invocation;
        # a changed pose hypothesis is still rendered and evaluated normally.
        rejected_causal_candidate_keys: set[tuple[int, str, bytes]] = set()

        def augment_causal_candidate(
            anchors: tuple[TemporalAnchor, ...], error: TemporalSelectionError
        ) -> tuple[TemporalAnchor, ...] | None:
            return _append_causal_current_frame_chain(
                anchors,
                error,
                registration_selector=registration_selector,
                frames=frames,
                dense_masks_by_view=dense_masks_by_view,
                root_from_cameras=root_from_cameras_array,
                stereo_consistency_fractions=stereo_consistency_fractions,
                parameters=temporal_parameters,
                rejected_candidate_keys=rejected_causal_candidate_keys,
            )

        (
            temporal_selection,
            temporal_anchor_tuple,
            temporal_candidate_sources,
            optional_pruned_anchors,
        ) = _select_with_optional_dense_anchors(
            initial_temporal_anchors,
            initial_temporal_sources,
            optional_ordinals=optional_ordinals,
            source_fps=float(input_manifest["source_fps"]),
            parameters=temporal_parameters,
            causal_candidate_augmenter=augment_causal_candidate,
        )
        excluded_temporal_anchors.extend(optional_pruned_anchors)
        temporal_frames = np.asarray(
            [anchor.source_frame_index for anchor in temporal_anchor_tuple],
            dtype=np.int64,
        )
        temporal_gap_audit = audit_temporal_evidence_gaps(
            temporal_anchor_tuple,
            temporal_selection,
            sampled_hand_observed_positions=hand_observed_positions_array,
            maximum_dense_gap_source_frames=(
                pose_config.maximum_temporal_evidence_gap_source_frames
            ),
            source_fps=float(input_manifest["source_fps"]),
            parameters=temporal_parameters,
        )
    except TemporalSelectionError as exc:
        rejection_reason = str(
            exc.diagnostics.get(
                "rejection_reason", "temporal_selection_no_physical_path"
            )
        )
        _write_temporal_rejection(
            output=output,
            identity=identity,
            episode_index=int(input_manifest["episode_index"]),
            source_revision=config.source.revision,
            rejection_reason=rejection_reason,
            error=exc,
            parameters=temporal_parameters,
            anchors=all_temporal_anchors,
            candidate_sources=all_temporal_candidate_sources,
            excluded_anchors=excluded_temporal_anchors,
        )
        raise SystemExit(2) from exc
    temporal_ordinals = np.asarray(
        [anchor.ordinal for anchor in temporal_anchor_tuple], dtype=np.int64
    )
    temporal_source_frames = np.asarray(
        [anchor.source_frame_index for anchor in temporal_anchor_tuple], dtype=np.int64
    )
    temporal_poses = temporal_selection.selected_poses_root
    initial_static_backfill = {
        "enabled": bool(temporal_ordinals[0] > 0),
        "seed_ordinal": int(temporal_ordinals[0]),
        "seed_source_frame_index": int(temporal_source_frames[0]),
        "backfilled_source_frame_range": (
            [0, int(temporal_source_frames[0] - 1)]
            if temporal_source_frames[0] > 0
            else None
        ),
        "contract": (
            "all preceding sampled hand states open plus raw CAD silhouette agreement "
            "at every preceding registration mask"
        ),
    }
    initial_static_seed_selection = {
        "mode": initial_static_selection_mode,
        "ordinal": int(initial_anchor.ordinal),
        "source_frame_index": int(initial_anchor.source_frame_index),
        "source_cad_seed_used": (
            initial_static_selection_mode == "source_cad_seed_static_rgbd"
        ),
    }
    interpolation_ordinals = temporal_ordinals.copy()
    interpolation_source_frames = temporal_source_frames.copy()
    interpolation_poses = temporal_poses.copy()
    if interpolation_ordinals[0] != 0:
        interpolation_ordinals = np.insert(interpolation_ordinals, 0, 0)
        interpolation_source_frames = np.insert(interpolation_source_frames, 0, 0)
        interpolation_poses = np.concatenate(
            (interpolation_poses[:1], interpolation_poses), axis=0
        )
    if temporal_ordinals[-1] != len(frames) - 1:
        interpolation_ordinals = np.append(
            interpolation_ordinals, len(frames) - 1
        )
        interpolation_source_frames = np.append(
            interpolation_source_frames, source_frame_count - 1
        )
        interpolation_poses = np.concatenate(
            (interpolation_poses, interpolation_poses[-1:]), axis=0
        )
    fused_root = interpolate_pose_trajectory(
        interpolation_ordinals, interpolation_poses, len(frames)
    )
    fused_camera = np.asarray(
        [
            np.linalg.inv(root_from_cameras_array[PRIMARY_POSE_VIEW][index])
            @ fused_root[index]
            for index in range(len(frames))
        ],
        dtype=np.float64,
    )
    full_rate_root = interpolate_pose_trajectory(
        interpolation_source_frames, interpolation_poses, source_frame_count
    )
    temporal_anchor_by_ordinal = {
        anchor.ordinal: index for index, anchor in enumerate(temporal_anchor_tuple)
    }
    selected_consensus = []
    required_bidirectional_consensus = []
    for anchor_index, anchor in enumerate(temporal_anchor_tuple):
        candidate_index = int(
            temporal_selection.selected_candidate_indices[anchor_index]
        )
        metric = anchor.candidate_metrics[candidate_index]
        consensus = metric.get("bidirectional_consensus")
        if not isinstance(consensus, dict):
            raise RuntimeError("selected temporal pose lacks consensus diagnostics")
        evidence_mode = temporal_selection.selected_evidence_modes[anchor_index]
        consensus_required = evidence_mode in {
            "rgbd_bidirectional",
            "static_rgbd_bidirectional",
            "static_multiview_silhouette_bidirectional",
            "unsegmented_multiview_rgbd_bidirectional",
        }
        if consensus_required and consensus.get("passes_gate") is not True:
            raise RuntimeError("selected RGB-D pose lacks bidirectional consensus")
        selected_consensus.append(consensus)
        required_bidirectional_consensus.append(consensus_required)
    required_bidirectional_consensus_array = np.asarray(
        required_bidirectional_consensus, dtype=bool
    )
    selected_consensus_translation = np.asarray(
        [
            float(value["translation_error_m"])
            for value, required in zip(
                selected_consensus,
                required_bidirectional_consensus_array,
                strict=True,
            )
            if required
        ],
        dtype=np.float64,
    )
    selected_consensus_rotation = np.asarray(
        [
            float(value["rotation_error_rad"])
            for value, required in zip(
                selected_consensus,
                required_bidirectional_consensus_array,
                strict=True,
            )
            if required
        ],
        dtype=np.float64,
    )
    if len(selected_consensus_translation) == 0:
        raise RuntimeError("temporal path has no bidirectionally verified RGB-D anchor")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    debug_directory = output.with_name(f".{output.name}.debug")
    try:
        frame_results = []
        rendered_diagnostic_pass_count = 0
        rendered_diagnostic_count = 0
        static_rendered_pass_count = 0
        static_rendered_required_count = 0
        initial_backfill_pass_count = 0
        initial_backfill_required_count = 0
        pose_evidence_pass_count = 0
        pose_evidence_required_count = 0
        for ordinal, ((rgb, observed_depth), camera_from_object) in enumerate(
            zip(frames, fused_camera, strict=True)
        ):
            with torch.inference_mode():
                _, rendered_depth_tensor, _ = nvdiffrast_render(
                    K=intrinsic_matrix,
                    H=480,
                    W=640,
                    ob_in_cams=torch.as_tensor(
                        camera_from_object[None], device="cuda", dtype=torch.float32
                    ),
                    glctx=glctx,
                    context="cuda",
                    get_normal=False,
                    mesh_tensors=render_mesh_tensors,
                )
            rendered_depth = rendered_depth_tensor[0].detach().cpu().numpy().astype(np.float32)
            observed_mask = dense_masks_by_view[PRIMARY_POSE_VIEW].get(ordinal)
            metrics = evaluate_rendered_alignment(
                observed_depth_m=observed_depth,
                rendered_depth_m=rendered_depth,
                observed_mask=observed_mask,
                maximum_occlusion_depth_error_m=(
                    pose_config.max_rendered_depth_median_abs_error_m
                ),
                maximum_median_absolute_depth_error_m=(
                    pose_config.max_rendered_depth_median_abs_error_m
                ),
                minimum_depth_overlap_fraction=(
                    pose_config.min_rendered_depth_overlap_fraction
                ),
                minimum_rendered_mask_explained_fraction=(
                    pose_config.min_rendered_mask_explained_fraction
                ),
            )
            if observed_mask is not None:
                rendered_diagnostic_count += 1
                rendered_diagnostic_pass_count += int(metrics.passes_gate)
            backfill_alignment = None
            if observed_mask is not None and ordinal < temporal_ordinals[0]:
                raw_rendered_mask = rendered_depth > 0.0
                rendered_pixels = int(np.count_nonzero(raw_rendered_mask))
                observed_pixels = int(np.count_nonzero(observed_mask))
                intersection = int(
                    np.count_nonzero(raw_rendered_mask & observed_mask)
                )
                precision = intersection / rendered_pixels if rendered_pixels else 0.0
                explained = intersection / observed_pixels if observed_pixels else 0.0
                backfill_pass = bool(
                    precision >= temporal_parameters.static_minimum_mask_precision
                    and explained
                    >= temporal_parameters.static_minimum_mask_explained_fraction
                )
                backfill_alignment = {
                    "rendered_pixels": rendered_pixels,
                    "observed_mask_pixels": observed_pixels,
                    "intersection_pixels": intersection,
                    "mask_precision": precision,
                    "mask_explained_fraction": explained,
                    "minimum_mask_precision": (
                        temporal_parameters.static_minimum_mask_precision
                    ),
                    "minimum_mask_explained_fraction": (
                        temporal_parameters.static_minimum_mask_explained_fraction
                    ),
                    "passes_gate": backfill_pass,
                }
                initial_backfill_required_count += 1
                initial_backfill_pass_count += int(backfill_pass)
                static_rendered_required_count += 1
                static_rendered_pass_count += int(backfill_pass)
            anchor_index = temporal_anchor_by_ordinal.get(ordinal)
            if backfill_alignment is not None:
                pose_evidence_required_count += 1
                pose_evidence_pass_count += int(
                    backfill_alignment["passes_gate"]
                )
                pose_evidence = {
                    "required": True,
                    "passes_gate": backfill_alignment["passes_gate"],
                    "mode": "initial_static_backfill",
                    "phase": PHASE_NAMES[0],
                    "raw_silhouette_alignment": backfill_alignment,
                }
            elif anchor_index is None:
                pose_evidence = {
                    "required": False,
                    "passes_gate": None,
                    "mode": "interpolation_between_audited_anchors",
                }
            else:
                evidence_mode = temporal_selection.selected_evidence_modes[
                    anchor_index
                ]
                static_required = evidence_mode == "static_rgbd_bidirectional"
                bidirectional_required = bool(
                    required_bidirectional_consensus_array[anchor_index]
                )
                evidence_pass = bool(
                    (
                        not bidirectional_required
                        or selected_consensus[anchor_index]["passes_gate"]
                    )
                    and (not static_required or metrics.passes_gate)
                )
                pose_evidence_required_count += 1
                pose_evidence_pass_count += int(evidence_pass)
                if static_required:
                    static_rendered_required_count += 1
                    static_rendered_pass_count += int(metrics.passes_gate)
                pose_evidence = {
                    "required": True,
                    "passes_gate": evidence_pass,
                    "mode": evidence_mode,
                    "phase": PHASE_NAMES[
                        int(temporal_selection.phase_indices[anchor_index])
                    ],
                    "selected_candidate_index": int(
                        temporal_selection.selected_candidate_indices[anchor_index]
                    ),
                    "selected_candidate_source": str(
                        temporal_candidate_sources[anchor_index][
                            int(
                                temporal_selection.selected_candidate_indices[
                                    anchor_index
                                ]
                            )
                        ]
                    ),
                    "bidirectional_consensus": selected_consensus[anchor_index],
                    "bidirectional_consensus_required": bidirectional_required,
                    "static_rendered_alignment_required": static_required,
                }
            review_record = None
            if observed_mask is not None:
                review_path = temporary / "review" / f"frame-{ordinal:06d}.png"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(review_path),
                    _review_image(rgb, observed_mask, rendered_depth, observed_depth),
                ):
                    raise RuntimeError(f"OpenCV could not write {review_path}")
                review_record = {
                    "path": str(review_path.relative_to(temporary)),
                    "sha256": sha256_file(review_path),
                }
            frame_results.append(
                {
                    "ordinal": ordinal,
                    "source_frame_index": int(source_frame_indices[ordinal]),
                    "forward_mode": forward_modes[ordinal],
                    "forward_rotation_projection_correction_frobenius": float(
                        forward_rotation_corrections[ordinal]
                    ),
                    "backward_mode": (
                        backward_modes[ordinal]
                        if ordinal <= last_registration_ordinal
                        else "unavailable_terminal_forward_only"
                    ),
                    "backward_rotation_projection_correction_frobenius": (
                        float(backward_rotation_corrections[ordinal])
                        if ordinal <= last_registration_ordinal
                        else None
                    ),
                    "bidirectional_translation_error_m": (
                        float(translation_errors[ordinal])
                        if ordinal <= last_registration_ordinal
                        else None
                    ),
                    "bidirectional_rotation_error_rad": (
                        float(rotation_errors[ordinal])
                        if ordinal <= last_registration_ordinal
                        else None
                    ),
                    "backward_symmetry_index": (
                        int(selected_symmetries[ordinal])
                        if ordinal <= last_registration_ordinal
                        else None
                    ),
                    "forward_registration_geometry": forward_registration_evidence.get(
                        ordinal
                    ),
                    "backward_registration_geometry": backward_registration_evidence.get(
                        ordinal
                    ),
                    "rendered_alignment": metrics.to_json(),
                    "pose_evidence": pose_evidence,
                    "review": review_record,
                }
            )

        bidirectional_pass = bool(
            np.all(
                selected_consensus_translation
                <= pose_config.max_bidirectional_translation_error_m
            )
            and np.all(
                selected_consensus_rotation
                <= pose_config.max_bidirectional_rotation_error_rad
            )
        )
        rendered_pass = (
            static_rendered_required_count > 0
            and static_rendered_pass_count == static_rendered_required_count
        )
        pose_evidence_pass = (
            pose_evidence_required_count
            == len(temporal_anchor_tuple) + initial_backfill_required_count
            and pose_evidence_pass_count == pose_evidence_required_count
        )
        rejection_reasons = _track_rejection_reasons(
            bidirectional_pass=bidirectional_pass,
            rendered_alignment_pass=(rendered_pass and pose_evidence_pass),
        )
        gate = {
            "sampled_frames": len(frames),
            "bidirectional_sampled_frames": int(
                np.count_nonzero(required_bidirectional_consensus_array)
            ),
            "bidirectional_scope": "selected_temporal_registration_anchors",
            "raw_tracker_bidirectional_sampled_frames": (
                last_registration_ordinal + 1
            ),
            "raw_tracker_max_bidirectional_translation_error_m": float(
                np.max(translation_errors)
            ),
            "raw_tracker_p95_bidirectional_translation_error_m": float(
                np.quantile(translation_errors, 0.95)
            ),
            "raw_tracker_max_bidirectional_rotation_error_rad": float(
                np.max(rotation_errors)
            ),
            "raw_tracker_p95_bidirectional_rotation_error_rad": float(
                np.quantile(rotation_errors, 0.95)
            ),
            "last_registration_ordinal": last_registration_ordinal,
            "last_tracking_eligible_ordinal": last_tracking_eligible_ordinal,
            "terminal_forward_only_sampled_frames": (
                len(frames) - last_registration_ordinal - 1
            ),
            "primary_terminal_tracking_gap_source_frames": (
                primary_terminal_gap_source_frames
            ),
            "tracking_eligible_terminal_gap_source_frames": (
                terminal_gap_source_frames
            ),
            "maximum_terminal_tracking_gap_source_frames": (
                pose_config.maximum_terminal_tracking_gap_source_frames
            ),
            "registration_frames": len(registration_masks),
            "last_active_hand_ordinal": last_active_hand_ordinal,
            "release_anchor_ordinal": release_anchor_ordinal,
            "final_static_anchor_ordinals": list(final_static_ordinals),
            "terminal_static_tail": {
                "mode": "verified_static_hold",
                "anchor_ordinals": list(final_static_ordinals),
                "final_anchor_source_frame_index": int(
                    source_frame_indices[final_static_ordinals[-1]]
                ),
                "terminal_source_frame_index": int(source_frame_indices[-1]),
                "tail_source_frame_gap": int(
                    source_frame_indices[-1]
                    - source_frame_indices[final_static_ordinals[-1]]
                ),
                "maximum_gap_source_frames": (
                    pose_config.maximum_terminal_tracking_gap_source_frames
                ),
            },
            "terminal_confirmation_registration_ordinal": (
                terminal_confirmation_ordinal
            ),
            "terminal_reverse_static_candidate_count": (
                terminal_reverse_candidate_count
            ),
            "promoted_terminal_registration_ordinals": list(
                promoted_terminal_ordinals
            ),
            "dense_head_mask_frames": len(
                dense_masks_by_view[PRIMARY_POSE_VIEW]
            ),
            "dense_head_tracking_anchor_candidates": len(
                dense_primary_ordinals
            ),
            "dense_mask_frames_by_view": {
                view_name: len(dense_masks_by_view[view_name])
                for view_name in POSE_VIEW_NAMES
            },
            "bimanual_wrist_bridge_frames": len(wrist_bridge_ordinals),
            "max_forward_rotation_projection_correction_frobenius": float(
                np.max(forward_rotation_corrections)
            ),
            "max_backward_rotation_projection_correction_frobenius": float(
                np.max(backward_rotation_corrections)
            ),
            "bidirectional_pass": bidirectional_pass,
            "max_bidirectional_translation_error_m": float(
                np.max(selected_consensus_translation)
            ),
            "p95_bidirectional_translation_error_m": float(
                np.quantile(selected_consensus_translation, 0.95)
            ),
            "maximum_allowed_bidirectional_translation_error_m": (
                pose_config.max_bidirectional_translation_error_m
            ),
            "max_bidirectional_rotation_error_rad": float(
                np.max(selected_consensus_rotation)
            ),
            "p95_bidirectional_rotation_error_rad": float(
                np.quantile(selected_consensus_rotation, 0.95)
            ),
            "maximum_allowed_bidirectional_rotation_error_rad": (
                pose_config.max_bidirectional_rotation_error_rad
            ),
            "rendered_alignment_pass": rendered_pass,
            "rendered_alignment_passed_frames": static_rendered_pass_count,
            "rendered_alignment_required_frames": static_rendered_required_count,
            "rendered_alignment_scope": "all_selected_static_rgbd_anchors",
            "initial_static_backfill_pass": (
                initial_backfill_required_count == 0
                or initial_backfill_pass_count == initial_backfill_required_count
            ),
            "initial_static_backfill_passed_frames": initial_backfill_pass_count,
            "initial_static_backfill_required_frames": (
                initial_backfill_required_count
            ),
            "rendered_alignment_diagnostic_passed_frames": (
                rendered_diagnostic_pass_count
            ),
            "rendered_alignment_diagnostic_frames": rendered_diagnostic_count,
            "pose_evidence_pass": pose_evidence_pass,
            "pose_evidence_passed_anchors": pose_evidence_pass_count,
            "pose_evidence_required_anchors": pose_evidence_required_count,
            "temporal_selection_pass": True,
            "temporal_anchor_count": len(temporal_anchor_tuple),
            "temporal_bimanual_hold_anchor_count": int(
                np.count_nonzero(temporal_selection.phase_indices == 2)
            ),
            "pass": not rejection_reasons,
        }
        arrays = {
            "source_frame_indices": _write_npy(
                temporary / "source_frame_indices.npy", source_frame_indices
            ),
            "forward_table_pose_root_sampled": _write_npy(
                temporary / "forward_table_pose_root_sampled.npy", forward_root
            ),
            "backward_table_pose_root_sampled": _write_npy(
                temporary / "backward_table_pose_root_sampled.npy", backward_root
            ),
            "table_pose_root_sampled": _write_npy(
                temporary / "table_pose_root_sampled.npy", fused_root
            ),
            "table_pose_rectified_camera_sampled": _write_npy(
                temporary / "table_pose_rectified_camera_sampled.npy", fused_camera
            ),
            "table_pose_root_30hz": _write_npy(
                temporary / "table_pose_root_30hz.npy", full_rate_root
            ),
            "bidirectional_translation_error_m": _write_npy(
                temporary / "bidirectional_translation_error_m.npy", translation_errors
            ),
            "bidirectional_rotation_error_rad": _write_npy(
                temporary / "bidirectional_rotation_error_rad.npy", rotation_errors
            ),
            "selected_bidirectional_translation_error_m": _write_npy(
                temporary / "selected_bidirectional_translation_error_m.npy",
                selected_consensus_translation,
            ),
            "selected_bidirectional_rotation_error_rad": _write_npy(
                temporary / "selected_bidirectional_rotation_error_rad.npy",
                selected_consensus_rotation,
            ),
            "selected_bidirectional_anchor_ordinals": _write_npy(
                temporary / "selected_bidirectional_anchor_ordinals.npy",
                np.asarray(
                    [anchor.ordinal for anchor in temporal_anchor_tuple],
                    dtype=np.int64,
                )[required_bidirectional_consensus_array],
            ),
            "backward_symmetry_index": _write_npy(
                temporary / "backward_symmetry_index.npy", selected_symmetries
            ),
            "temporal_anchor_ordinals": _write_npy(
                temporary / "temporal_anchor_ordinals.npy",
                np.asarray([anchor.ordinal for anchor in temporal_anchor_tuple], dtype=np.int64),
            ),
            "temporal_selected_candidate_index": _write_npy(
                temporary / "temporal_selected_candidate_index.npy",
                temporal_selection.selected_candidate_indices,
            ),
            "temporal_selected_symmetry_index": _write_npy(
                temporary / "temporal_selected_symmetry_index.npy",
                temporal_selection.selected_symmetry_indices,
            ),
            "temporal_phase_index": _write_npy(
                temporary / "temporal_phase_index.npy", temporal_selection.phase_indices
            ),
            "temporal_table_pose_root_anchors": _write_npy(
                temporary / "temporal_table_pose_root_anchors.npy",
                temporal_selection.selected_poses_root,
            ),
            "forward_rotation_projection_correction_frobenius": _write_npy(
                temporary / "forward_rotation_projection_correction_frobenius.npy",
                forward_rotation_corrections,
            ),
            "backward_rotation_projection_correction_frobenius": _write_npy(
                temporary / "backward_rotation_projection_correction_frobenius.npy",
                backward_rotation_corrections,
            ),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "episode_index": int(input_manifest["episode_index"]),
            "source_revision": config.source.revision,
            "accepted": gate["pass"],
            "rejection_reasons": rejection_reasons,
            "source_frame_count": source_frame_count,
            "source_fps": float(input_manifest["source_fps"]),
            "tracking_fps": float(input_manifest["tracking_fps"]),
            "method": pose_config.method,
            "mesh": {
                "path": str(mesh_path),
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.faces),
            },
            "thresholds": {
                "max_bidirectional_translation_error_m": (
                    pose_config.max_bidirectional_translation_error_m
                ),
                "max_bidirectional_rotation_error_rad": (
                    pose_config.max_bidirectional_rotation_error_rad
                ),
                "max_rendered_depth_median_abs_error_m": (
                    pose_config.max_rendered_depth_median_abs_error_m
                ),
                "min_rendered_depth_overlap_fraction": (
                    pose_config.min_rendered_depth_overlap_fraction
                ),
                "min_rendered_mask_explained_fraction": (
                    pose_config.min_rendered_mask_explained_fraction
                ),
                "temporal_selection": asdict(temporal_parameters),
            },
            "temporal_selection": temporal_selection.to_json(temporal_anchor_tuple),
            "temporal_evidence_gap_audit": list(temporal_gap_audit),
            "initial_static_backfill": initial_static_backfill,
            "initial_static_seed_selection": initial_static_seed_selection,
            "excluded_temporal_anchors": sorted(
                excluded_temporal_anchors,
                key=lambda value: int(value["source_frame_index"]),
            ),
            "gate": gate,
            "arrays": arrays,
            "frames": frame_results,
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(debug_directory, ignore_errors=True)
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
