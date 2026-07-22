"""Deterministic Grounded-SAM mask gates for the source white table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import cv2
import numpy as np


SEGMENTATION_SCHEMA_VERSION = "team_ramen_grounded_sam2_table_masks/v15"

MASK_SEQUENCE_FRAGMENTATION_MODES = frozenset({"linear", "logarithmic"})


@dataclass(frozen=True)
class TargetPointPrompt:
    projected_eef_pixels_xy: tuple[tuple[float, float], ...]
    visible_eef_count: int
    reference_source: str
    reference_pixel_xy: tuple[float, float] | None
    point_xy: tuple[int, int] | None
    point_distance_px: float | None
    passes_gate: bool
    rejection_reason: str | None

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["projected_eef_pixels_xy"] = [
            list(point) for point in self.projected_eef_pixels_xy
        ]
        value["reference_pixel_xy"] = (
            None if self.reference_pixel_xy is None else list(self.reference_pixel_xy)
        )
        value["point_xy"] = None if self.point_xy is None else list(self.point_xy)
        return value


@dataclass(frozen=True)
class MaskRefinementMetrics:
    input_pixels: int
    robot_overlap_pixels: int
    bright_non_robot_pixels: int
    connected_components: int
    retained_components: int
    output_pixels: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReachableComponentMetrics:
    component_index: int
    pixel_count: int
    valid_depth_fraction: float
    median_nearest_eef_distance_m: float | None
    centroid_root_m: tuple[float, float, float] | None
    retained: bool
    rejection_reason: str | None

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["centroid_root_m"] = (
            None if self.centroid_root_m is None else list(self.centroid_root_m)
        )
        return value


@dataclass(frozen=True)
class ReachabilityRefinementMetrics:
    input_component_count: int
    retained_component_count: int
    output_pixels: int
    components: tuple[ReachableComponentMetrics, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "input_component_count": self.input_component_count,
            "retained_component_count": self.retained_component_count,
            "output_pixels": self.output_pixels,
            "components": [component.to_json() for component in self.components],
        }


@dataclass(frozen=True)
class MaskCandidateMetrics:
    candidate_index: int
    detector_score: float
    segmentation_iou_score: float
    detector_box_xyxy: tuple[float, float, float, float]
    mask_box_xyxy: tuple[int, int, int, int]
    mask_area_fraction: float
    valid_depth_fraction_in_mask: float | None
    median_depth_m: float | None
    neutral_bright_fraction: float
    refinement_retention_fraction: float
    retained_component_count: int
    temporal_iou: float | None
    ranking_score: float
    passes_gate: bool
    rejection_reasons: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["detector_box_xyxy"] = list(self.detector_box_xyxy)
        value["mask_box_xyxy"] = list(self.mask_box_xyxy)
        value["rejection_reasons"] = list(self.rejection_reasons)
        return value


@dataclass(frozen=True)
class BidirectionalMaskFusionMetrics:
    forward_pixels: int
    backward_pixels: int
    intersection_pixels: int
    union_pixels: int
    forward_area_fraction: float
    backward_area_fraction: float
    fused_area_fraction: float
    bidirectional_iou: float
    passes_gate: bool
    rejection_reasons: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["rejection_reasons"] = list(self.rejection_reasons)
        return value


@dataclass(frozen=True)
class MaskSequenceParameters:
    detector_weight: float = 1.0
    segmentation_weight: float = 0.35
    retention_weight: float = 0.25
    area_weight: float = 1.0
    fragmentation_weight: float = 0.1
    transition_iou_weight: float = 2.0
    transition_area_weight: float = 1.5
    probability_floor: float = 0.05
    fragmentation_mode: str = "linear"

    def validate(self) -> None:
        values = {
            key: value
            for key, value in asdict(self).items()
            if key != "fragmentation_mode"
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("mask sequence weights must be finite and positive")
        if self.probability_floor >= 1.0:
            raise ValueError("mask sequence probability floor must be below one")
        if self.fragmentation_mode not in MASK_SEQUENCE_FRAGMENTATION_MODES:
            raise ValueError(
                "mask sequence fragmentation mode must be linear or logarithmic"
            )


def _mask_sequence_unary(
    metrics: MaskCandidateMetrics, parameters: MaskSequenceParameters
) -> float:
    floor = parameters.probability_floor
    component_count = max(1, metrics.retained_component_count)
    fragmentation = (
        float(component_count - 1)
        if parameters.fragmentation_mode == "linear"
        else math.log(float(component_count))
    )
    return (
        parameters.detector_weight * math.log(metrics.detector_score + floor)
        + parameters.segmentation_weight
        * math.log(metrics.segmentation_iou_score + floor)
        + parameters.retention_weight
        * math.log(metrics.refinement_retention_fraction + floor)
        + parameters.area_weight * math.log(metrics.mask_area_fraction + floor)
        - parameters.fragmentation_weight * fragmentation
    )


def select_mask_candidate_sequence(
    *,
    masks_by_frame: Iterable[Iterable[np.ndarray]],
    metrics_by_frame: Iterable[Iterable[MaskCandidateMetrics]],
    parameters: MaskSequenceParameters = MaskSequenceParameters(),
) -> tuple[int | None, ...]:
    """Select one existing candidate per valid frame with global temporal consistency."""

    parameters.validate()
    frame_masks = tuple(tuple(np.asarray(mask, dtype=bool) for mask in masks) for masks in masks_by_frame)
    frame_metrics = tuple(tuple(values) for values in metrics_by_frame)
    if len(frame_masks) != len(frame_metrics) or not frame_masks:
        raise ValueError("mask sequence inputs must contain the same non-zero frame count")
    valid_frames: list[tuple[int, tuple[int, ...]]] = []
    expected_shape = None
    for frame_index, (masks, metrics) in enumerate(zip(frame_masks, frame_metrics, strict=True)):
        if len(masks) != len(metrics):
            raise ValueError(f"mask and metric counts differ at frame {frame_index}")
        for candidate_index, (mask, metric) in enumerate(zip(masks, metrics, strict=True)):
            if expected_shape is None:
                expected_shape = mask.shape
            if mask.ndim != 2 or mask.shape != expected_shape:
                raise ValueError("all sequence masks must share one 2D shape")
            if metric.candidate_index != candidate_index:
                raise ValueError("candidate metrics are not in candidate-index order")
        passing = tuple(metric.candidate_index for metric in metrics if metric.passes_gate)
        if passing:
            valid_frames.append((frame_index, passing))
    selected: list[int | None] = [None] * len(frame_masks)
    if not valid_frames:
        return tuple(selected)

    scores: dict[int, float] = {}
    backpointers: list[dict[int, int | None]] = []
    previous_frame = None
    for frame_index, candidates in valid_frames:
        current_scores: dict[int, float] = {}
        current_backpointers: dict[int, int | None] = {}
        for candidate_index in candidates:
            unary = _mask_sequence_unary(
                frame_metrics[frame_index][candidate_index], parameters
            )
            if previous_frame is None:
                current_scores[candidate_index] = unary
                current_backpointers[candidate_index] = None
                continue
            transitions = []
            current_mask = frame_masks[frame_index][candidate_index]
            current_area = float(current_mask.sum())
            for previous_index, previous_score in scores.items():
                previous_mask = frame_masks[previous_frame][previous_index]
                previous_area = float(previous_mask.sum())
                area_ratio = min(current_area, previous_area) / max(
                    current_area, previous_area
                )
                transition = (
                    parameters.transition_iou_weight
                    * mask_iou(previous_mask, current_mask)
                    + parameters.transition_area_weight * math.log(area_ratio)
                )
                transitions.append((previous_score + transition, -previous_index, previous_index))
            best_score, _, best_previous = max(transitions)
            current_scores[candidate_index] = unary + best_score
            current_backpointers[candidate_index] = best_previous
        scores = current_scores
        backpointers.append(current_backpointers)
        previous_frame = frame_index

    candidate = max((score, -index, index) for index, score in scores.items())[2]
    for (frame_index, _), pointers in zip(
        reversed(valid_frames), reversed(backpointers), strict=True
    ):
        selected[frame_index] = candidate
        previous = pointers[candidate]
        if previous is None:
            break
        candidate = previous
    return tuple(selected)


def registration_ordinals(
    source_frame_indices: Iterable[int],
    interval_source_frames: int,
) -> tuple[int, ...]:
    frames = tuple(int(value) for value in source_frame_indices)
    if (
        not frames
        or interval_source_frames <= 0
        or tuple(sorted(set(frames))) != frames
        or frames[0] < 0
    ):
        raise ValueError("source frames must be sorted and unique and interval must be positive")
    selected = [
        ordinal
        for ordinal, frame in enumerate(frames)
        if frame % interval_source_frames == 0
    ]
    selected.extend((0, len(frames) - 1))
    return tuple(sorted(set(selected)))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("masks must be same-sized 2D arrays")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def fuse_bidirectional_masks(
    forward_mask: np.ndarray,
    backward_mask: np.ndarray,
    *,
    minimum_iou: float,
    minimum_area_fraction: float,
    maximum_area_fraction: float,
) -> tuple[np.ndarray | None, BidirectionalMaskFusionMetrics]:
    """Intersect independently propagated masks after deterministic quality gates."""

    forward = np.asarray(forward_mask, dtype=bool)
    backward = np.asarray(backward_mask, dtype=bool)
    if forward.ndim != 2 or forward.shape != backward.shape or forward.size == 0:
        raise ValueError("bidirectional masks must be same-sized non-empty 2D arrays")
    if not (
        0.0 < minimum_iou < 1.0
        and 0.0 < minimum_area_fraction < maximum_area_fraction < 1.0
    ):
        raise ValueError("bidirectional mask thresholds are invalid")

    intersection = forward & backward
    union = forward | backward
    forward_pixels = int(np.count_nonzero(forward))
    backward_pixels = int(np.count_nonzero(backward))
    intersection_pixels = int(np.count_nonzero(intersection))
    union_pixels = int(np.count_nonzero(union))
    denominator = float(forward.size)
    forward_area = forward_pixels / denominator
    backward_area = backward_pixels / denominator
    fused_area = intersection_pixels / denominator
    iou = intersection_pixels / union_pixels if union_pixels else 0.0
    reasons = []
    if iou < minimum_iou:
        reasons.append("bidirectional_iou_below_threshold")
    if not minimum_area_fraction <= forward_area <= maximum_area_fraction:
        reasons.append("forward_area_out_of_bounds")
    if not minimum_area_fraction <= backward_area <= maximum_area_fraction:
        reasons.append("backward_area_out_of_bounds")
    if not minimum_area_fraction <= fused_area <= maximum_area_fraction:
        reasons.append("fused_area_out_of_bounds")
    metrics = BidirectionalMaskFusionMetrics(
        forward_pixels=forward_pixels,
        backward_pixels=backward_pixels,
        intersection_pixels=intersection_pixels,
        union_pixels=union_pixels,
        forward_area_fraction=forward_area,
        backward_area_fraction=backward_area,
        fused_area_fraction=fused_area,
        bidirectional_iou=iou,
        passes_gate=not reasons,
        rejection_reasons=tuple(reasons),
    )
    return (intersection if metrics.passes_gate else None), metrics


def target_point_prompt(
    *,
    rgb: np.ndarray,
    robot_silhouette: np.ndarray,
    eef_poses_root: np.ndarray,
    root_from_camera: np.ndarray,
    intrinsic_matrix: np.ndarray,
    detector_box_xyxy: Iterable[float],
    minimum_value: int,
    maximum_point_distance_px: int,
    principal_point_weight: float,
) -> TargetPointPrompt:
    """Locate the target near the calibrated optical axis with optional EEF correction."""

    image = np.asarray(rgb)
    robot = np.asarray(robot_silhouette, dtype=bool)
    eef = np.asarray(eef_poses_root, dtype=np.float64)
    transform = np.asarray(root_from_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    box = tuple(float(value) for value in detector_box_xyxy)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 (480, 640, 3)")
    if robot.shape != (480, 640):
        raise ValueError("robot_silhouette must be (480, 640)")
    if eef.shape != (2, 4, 4) or not np.isfinite(eef).all():
        raise ValueError("eef_poses_root must be finite (2, 4, 4)")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("root_from_camera must be a finite 4x4 transform")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or not (0.0 <= box[0] < box[2] <= 640.0 and 0.0 <= box[1] < box[3] <= 480.0)
    ):
        raise ValueError("detector_box_xyxy is invalid")
    if (
        not 0 < minimum_value <= 255
        or maximum_point_distance_px <= 0
        or not 0.0 < principal_point_weight <= 1.0
    ):
        raise ValueError("target point thresholds are invalid")

    camera_from_root = np.linalg.inv(transform)
    points_root = eef[:, :3, 3]
    points_camera = (
        camera_from_root[:3, :3] @ points_root.T
    ).T + camera_from_root[:3, 3]
    positive_depth = points_camera[:, 2] > 0.05
    pixels_h = (intrinsic @ points_camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    in_frame = (
        positive_depth
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < 640.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < 480.0)
    )
    projected = tuple(tuple(float(value) for value in point) for point in pixels[in_frame])
    principal_point = np.asarray((intrinsic[0, 2], intrinsic[1, 2]), dtype=np.float64)
    if not (
        0.0 <= principal_point[0] < 640.0 and 0.0 <= principal_point[1] < 480.0
    ):
        raise ValueError("camera principal point lies outside the image")
    if projected:
        eef_midpoint = np.mean(np.asarray(projected, dtype=np.float64), axis=0)
        reference = (
            principal_point_weight * principal_point
            + (1.0 - principal_point_weight) * eef_midpoint
        )
        reference_source = "calibrated_principal_point_with_visible_eef_correction"
    else:
        reference = principal_point
        reference_source = "calibrated_principal_point"
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    candidates = (hsv[..., 1] <= 80) & (hsv[..., 2] >= minimum_value) & ~robot
    x0, y0 = int(math.floor(box[0])), int(math.floor(box[1]))
    x1, y1 = int(math.ceil(box[2])), int(math.ceil(box[3]))
    within_box = np.zeros_like(candidates)
    within_box[y0:y1, x0:x1] = True
    candidates &= within_box
    if not candidates.any():
        return TargetPointPrompt(
            projected,
            len(projected),
            reference_source,
            tuple(float(value) for value in reference),
            None,
            None,
            False,
            "no_white_non_robot_pixel_in_detection",
        )

    # Prefer a point inside a white region instead of an antialiased edge pixel.
    distance_inside = cv2.distanceTransform(
        candidates.astype(np.uint8), cv2.DIST_L2, 3
    )
    interior = candidates & (distance_inside >= 2.0)
    search = interior if interior.any() else candidates
    rows, columns = np.nonzero(search)
    squared_distance = (columns - reference[0]) ** 2 + (rows - reference[1]) ** 2
    selected = int(np.argmin(squared_distance))
    distance = float(math.sqrt(float(squared_distance[selected])))
    point = (int(columns[selected]), int(rows[selected]))
    if distance > maximum_point_distance_px:
        return TargetPointPrompt(
            projected,
            len(projected),
            reference_source,
            tuple(float(value) for value in reference),
            point,
            distance,
            False,
            "white_non_robot_pixel_too_far_from_target_reference",
        )
    return TargetPointPrompt(
        projected,
        len(projected),
        reference_source,
        tuple(float(value) for value in reference),
        point,
        distance,
        True,
        None,
    )


def refine_table_mask(
    *,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    robot_silhouette: np.ndarray,
    minimum_value: int,
    minimum_component_area_px: int,
) -> tuple[np.ndarray, MaskRefinementMetrics]:
    """Remove projected robot pixels and dark workbench regions from a SAM mask."""

    image = np.asarray(rgb)
    candidate = np.asarray(candidate_mask, dtype=bool)
    robot = np.asarray(robot_silhouette, dtype=bool)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 (480, 640, 3)")
    if candidate.shape != (480, 640) or robot.shape != candidate.shape:
        raise ValueError("candidate and robot masks must be (480, 640)")
    if not 0 < minimum_value <= 255 or minimum_component_area_px <= 0:
        raise ValueError("table-mask refinement thresholds are invalid")

    value = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 2]
    robot_overlap = candidate & robot
    bright_non_robot = candidate & ~robot & (value >= minimum_value)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_non_robot.astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(candidate)
    retained_components = 0
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) < minimum_component_area_px:
            continue
        retained |= labels == component
        retained_components += 1
    metrics = MaskRefinementMetrics(
        input_pixels=int(candidate.sum()),
        robot_overlap_pixels=int(robot_overlap.sum()),
        bright_non_robot_pixels=int(bright_non_robot.sum()),
        connected_components=max(0, count - 1),
        retained_components=retained_components,
        output_pixels=int(retained.sum()),
    )
    return retained, metrics


def filter_reachable_table_components(
    *,
    candidate_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    root_from_camera: np.ndarray,
    eef_positions_root: np.ndarray,
    maximum_median_eef_distance_m: float,
    minimum_component_valid_depth_fraction: float,
) -> tuple[np.ndarray, ReachabilityRefinementMetrics]:
    """Remove head-view components that are outside the robot's reachable workspace."""

    candidate = np.asarray(candidate_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    transform = np.asarray(root_from_camera, dtype=np.float64)
    eef = np.asarray(eef_positions_root, dtype=np.float64)
    if candidate.shape != (480, 640) or depth.shape != candidate.shape:
        raise ValueError("candidate mask and depth must be (480, 640)")
    if not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError("depth must be finite and non-negative")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("intrinsic matrix is invalid")
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8)
        or eef.shape != (2, 3)
        or not np.isfinite(eef).all()
    ):
        raise ValueError("root-frame geometry is invalid")
    if (
        not math.isfinite(maximum_median_eef_distance_m)
        or maximum_median_eef_distance_m <= 0.0
        or not 0.0 < minimum_component_valid_depth_fraction <= 1.0
    ):
        raise ValueError("reachable-component thresholds are invalid")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(candidate)
    component_records = []
    for component in range(1, count):
        component_mask = labels == component
        pixel_count = int(stats[component, cv2.CC_STAT_AREA])
        valid = component_mask & (depth > 0.0)
        valid_fraction = float(valid.sum() / pixel_count) if pixel_count else 0.0
        median_distance = None
        centroid = None
        if valid_fraction < minimum_component_valid_depth_fraction:
            rejection_reason = "insufficient_valid_depth"
        else:
            rows, columns = np.nonzero(valid)
            z = depth[rows, columns]
            camera_points = np.stack(
                (
                    (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
                    (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
                    z,
                    np.ones_like(z),
                ),
                axis=1,
            )
            root_points = (camera_points @ transform.T)[:, :3]
            nearest_eef_distance = np.linalg.norm(
                root_points[:, None, :] - eef[None, :, :], axis=2
            ).min(axis=1)
            median_distance = float(np.median(nearest_eef_distance))
            centroid = tuple(float(value) for value in np.median(root_points, axis=0))
            rejection_reason = (
                None
                if median_distance <= maximum_median_eef_distance_m
                else "outside_reachable_workspace"
            )
        is_retained = rejection_reason is None
        if is_retained:
            retained |= component_mask
        component_records.append(
            ReachableComponentMetrics(
                component_index=component,
                pixel_count=pixel_count,
                valid_depth_fraction=valid_fraction,
                median_nearest_eef_distance_m=median_distance,
                centroid_root_m=centroid,
                retained=is_retained,
                rejection_reason=rejection_reason,
            )
        )
    metrics = ReachabilityRefinementMetrics(
        input_component_count=max(0, count - 1),
        retained_component_count=sum(record.retained for record in component_records),
        output_pixels=int(retained.sum()),
        components=tuple(component_records),
    )
    return retained, metrics


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return (0, 0, 0, 0)
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)


def evaluate_mask_candidates(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    masks: Iterable[np.ndarray],
    detector_scores: Iterable[float],
    segmentation_iou_scores: Iterable[float],
    detector_boxes_xyxy: Iterable[Iterable[float]],
    refinement_metrics: Iterable[MaskRefinementMetrics] | None = None,
    minimum_segmentation_iou: float,
    minimum_mask_area_fraction: float,
    maximum_mask_area_fraction: float,
    minimum_valid_depth_fraction: float,
    previous_mask: np.ndarray | None = None,
) -> tuple[tuple[MaskCandidateMetrics, ...], int | None]:
    image = np.asarray(rgb)
    depth = None if depth_m is None else np.asarray(depth_m, dtype=np.float32)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 (480, 640, 3)")
    if depth is not None and (
        depth.shape != (480, 640)
        or not np.isfinite(depth).all()
        or np.any(depth < 0.0)
    ):
        raise ValueError("depth_m must be a finite non-negative (480, 640) image")
    if not (
        0.0 <= minimum_segmentation_iou < 1.0
        and 0.0 < minimum_mask_area_fraction < maximum_mask_area_fraction < 1.0
        and 0.0 <= minimum_valid_depth_fraction <= 1.0
    ):
        raise ValueError("mask gate thresholds are invalid")
    if depth is None and minimum_valid_depth_fraction != 0.0:
        raise ValueError("depth-free mask evaluation requires a zero depth threshold")
    mask_values = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    detector_values = tuple(float(value) for value in detector_scores)
    segmentation_values = tuple(float(value) for value in segmentation_iou_scores)
    boxes = tuple(tuple(float(value) for value in box) for box in detector_boxes_xyxy)
    refinements = (
        tuple(refinement_metrics)
        if refinement_metrics is not None
        else tuple(
            MaskRefinementMetrics(
                input_pixels=int(mask.sum()),
                robot_overlap_pixels=0,
                bright_non_robot_pixels=int(mask.sum()),
                connected_components=1 if mask.any() else 0,
                retained_components=1 if mask.any() else 0,
                output_pixels=int(mask.sum()),
            )
            for mask in mask_values
        )
    )
    if not (
        len(mask_values)
        == len(detector_values)
        == len(segmentation_values)
        == len(boxes)
        == len(refinements)
    ):
        raise ValueError("candidate mask, score, and box counts differ")
    previous = None if previous_mask is None else np.asarray(previous_mask, dtype=bool)
    if previous is not None and previous.shape != image.shape[:2]:
        raise ValueError("previous_mask shape differs from current masks")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    neutral_bright = (hsv[..., 1] <= 80) & (hsv[..., 2] >= 100)
    records = []
    for index, (mask, detector_score, segmentation_score, box, refinement) in enumerate(
        zip(
            mask_values,
            detector_values,
            segmentation_values,
            boxes,
            refinements,
            strict=True,
        )
    ):
        if mask.shape != image.shape[:2]:
            raise ValueError(f"candidate mask {index} shape differs from the source image")
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or not (0.0 <= box[0] < box[2] <= 640.0 and 0.0 <= box[1] < box[3] <= 480.0)
        ):
            raise ValueError(f"candidate detector box {index} is invalid")
        if not (0.0 <= detector_score <= 1.0 and 0.0 <= segmentation_score <= 1.0):
            raise ValueError(f"candidate scores {index} must be probabilities")
        area = int(mask.sum())
        area_fraction = area / mask.size
        valid_depth = None if depth is None else mask & (depth > 0.0)
        valid_fraction = (
            None
            if valid_depth is None
            else float(valid_depth.sum() / area) if area else 0.0
        )
        median_depth = (
            float(np.median(depth[valid_depth]))
            if valid_depth is not None and valid_depth.any()
            else None
        )
        neutral_fraction = float((neutral_bright & mask).sum() / area) if area else 0.0
        if refinement.input_pixels < 0 or refinement.output_pixels != area:
            raise ValueError(f"candidate refinement metrics {index} are inconsistent")
        retention_fraction = (
            refinement.output_pixels / refinement.input_pixels
            if refinement.input_pixels
            else 0.0
        )
        component_factor = 1.0 / (1.0 + 0.2 * max(0, refinement.retained_components - 1))
        temporal_iou = None if previous is None else mask_iou(mask, previous)
        reasons = []
        if segmentation_score < minimum_segmentation_iou:
            reasons.append("segmentation_iou_below_threshold")
        if area_fraction < minimum_mask_area_fraction:
            reasons.append("mask_too_small")
        if area_fraction > maximum_mask_area_fraction:
            reasons.append("mask_too_large")
        if valid_fraction is not None and valid_fraction < minimum_valid_depth_fraction:
            reasons.append("valid_depth_fraction_below_threshold")
        temporal_factor = 1.0 if temporal_iou is None else 0.5 + 0.5 * temporal_iou
        ranking = (
            detector_score
            * segmentation_score
            * (0.5 + 0.5 * neutral_fraction)
            * retention_fraction
            * component_factor
            * temporal_factor
        )
        records.append(
            MaskCandidateMetrics(
                candidate_index=index,
                detector_score=detector_score,
                segmentation_iou_score=segmentation_score,
                detector_box_xyxy=box,
                mask_box_xyxy=_mask_box(mask),
                mask_area_fraction=area_fraction,
                valid_depth_fraction_in_mask=valid_fraction,
                median_depth_m=median_depth,
                neutral_bright_fraction=neutral_fraction,
                refinement_retention_fraction=retention_fraction,
                retained_component_count=refinement.retained_components,
                temporal_iou=temporal_iou,
                ranking_score=ranking,
                passes_gate=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )
    passing = [record for record in records if record.passes_gate]
    selected = (
        max(passing, key=lambda record: (record.ranking_score, -record.candidate_index)).candidate_index
        if passing
        else None
    )
    return tuple(records), selected
