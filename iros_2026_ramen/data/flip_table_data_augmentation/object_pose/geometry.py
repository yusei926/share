"""Coordinate transforms and trajectory gates for offline table-pose tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..config import CameraConfig
from ..source_camera_projection import root_from_camera


OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class RenderedAlignmentMetrics:
    raw_rendered_pixels: int
    occluded_rendered_pixels: int
    rendered_pixels: int
    valid_observed_pixels: int
    depth_overlap_pixels: int
    depth_overlap_fraction: float
    median_absolute_depth_error_m: float | None
    observed_mask_pixels: int | None
    raw_rendered_mask_precision: float | None
    raw_rendered_mask_explained_fraction: float | None
    rendered_mask_explained_fraction: float | None
    passes_gate: bool
    rejection_reasons: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "raw_rendered_pixels": self.raw_rendered_pixels,
            "occluded_rendered_pixels": self.occluded_rendered_pixels,
            "rendered_pixels": self.rendered_pixels,
            "valid_observed_pixels": self.valid_observed_pixels,
            "depth_overlap_pixels": self.depth_overlap_pixels,
            "depth_overlap_fraction": self.depth_overlap_fraction,
            "median_absolute_depth_error_m": self.median_absolute_depth_error_m,
            "observed_mask_pixels": self.observed_mask_pixels,
            "raw_rendered_mask_precision": self.raw_rendered_mask_precision,
            "raw_rendered_mask_explained_fraction": (
                self.raw_rendered_mask_explained_fraction
            ),
            "rendered_mask_explained_fraction": self.rendered_mask_explained_fraction,
            "passes_gate": self.passes_gate,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _transform(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation is not orthonormal")
    if np.linalg.det(result[:3, :3]) < 0.999999:
        raise ValueError(f"{label} rotation is not proper")
    return result


def project_to_rigid_transform(
    value: np.ndarray,
    label: str,
    *,
    maximum_rotation_correction_frobenius: float = 1.0e-4,
) -> tuple[np.ndarray, float]:
    """Project small floating-point rotation drift onto SO(3).

    FoundationPose refines transforms in float32. Repeated tracking can therefore
    move a valid rotation a few ulps outside the strict SE(3) contract. Reflections
    and corrections larger than the explicit numerical tolerance remain errors.
    """

    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError(f"{label} has an invalid homogeneous row")
    if maximum_rotation_correction_frobenius <= 0.0:
        raise ValueError("maximum rotation correction must be positive")
    rotation = result[:3, :3]
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"{label} rotation is not orientation-preserving")
    left, _, right = np.linalg.svd(rotation)
    projected = left @ right
    correction = float(np.linalg.norm(projected - rotation, ord="fro"))
    if correction > maximum_rotation_correction_frobenius:
        raise ValueError(
            f"{label} rotation correction {correction:.9g} exceeds "
            f"{maximum_rotation_correction_frobenius:.9g}"
        )
    result[:3, :3] = projected
    return _transform(result, label), correction


def root_from_rectified_opencv_camera(
    root_from_camera_parent: np.ndarray,
    camera: CameraConfig,
    rectified_from_raw_opencv_rotation: np.ndarray,
) -> np.ndarray:
    """Compose G1 FK with raw optical axes and stereo left-image rectification."""

    rectification = np.asarray(rectified_from_raw_opencv_rotation, dtype=np.float64)
    if rectification.shape != (3, 3) or not np.isfinite(rectification).all():
        raise ValueError("left stereo rectification must be a finite 3x3 rotation")
    if not np.allclose(rectification.T @ rectification, np.eye(3), atol=1.0e-6):
        raise ValueError("left stereo rectification is not orthonormal")
    raw_opencv_from_rectified = np.eye(4, dtype=np.float64)
    raw_opencv_from_rectified[:3, :3] = rectification.T
    root_from_raw_opengl = root_from_camera(root_from_camera_parent, camera)
    result = root_from_raw_opengl @ OPENGL_FROM_OPENCV @ raw_opencv_from_rectified
    return _transform(result, "root_from_rectified_opencv_camera")


def root_from_object_pose(
    root_from_rectified_camera: np.ndarray,
    rectified_camera_from_object: np.ndarray,
) -> np.ndarray:
    return _transform(root_from_rectified_camera, "root_from_rectified_camera") @ _transform(
        rectified_camera_from_object, "rectified_camera_from_object"
    )


def visible_rendered_mask(
    observed_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
    *,
    maximum_occlusion_depth_error_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible and occluded CAD pixels under measured z-buffer evidence."""

    observed = np.asarray(observed_depth_m, dtype=np.float64)
    rendered = np.asarray(rendered_depth_m, dtype=np.float64)
    if observed.shape != rendered.shape or observed.ndim != 2:
        raise ValueError("observed and rendered depth must be matching 2D arrays")
    if maximum_occlusion_depth_error_m <= 0.0:
        raise ValueError("occlusion depth tolerance must be positive")
    valid_observed = np.isfinite(observed) & (observed > 0.0)
    raw_rendered = np.isfinite(rendered) & (rendered > 0.0)
    occluded = (
        raw_rendered
        & valid_observed
        & (rendered > observed + maximum_occlusion_depth_error_m)
    )
    return raw_rendered & ~occluded, occluded


def table_symmetry_transforms() -> tuple[np.ndarray, ...]:
    identity = np.eye(4, dtype=np.float64)
    rotate_180 = np.eye(4, dtype=np.float64)
    rotate_180[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    return identity, rotate_180


def pose_errors(
    first: np.ndarray,
    second: np.ndarray,
    *,
    symmetries: tuple[np.ndarray, ...] = table_symmetry_transforms(),
) -> tuple[float, float, int]:
    left = _transform(first, "first pose")
    right = _transform(second, "second pose")
    candidates = []
    for index, symmetry in enumerate(symmetries):
        equivalent = right @ _transform(symmetry, f"symmetry {index}")
        translation = float(np.linalg.norm(left[:3, 3] - equivalent[:3, 3]))
        rotation = float(Rotation.from_matrix(left[:3, :3].T @ equivalent[:3, :3]).magnitude())
        candidates.append((translation + rotation, translation, rotation, index))
    _, translation, rotation, index = min(candidates)
    return translation, rotation, index


def fuse_bidirectional_poses(
    forward_poses: np.ndarray,
    backward_poses: np.ndarray,
    *,
    symmetries: tuple[np.ndarray, ...] = table_symmetry_transforms(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse two root-frame tracks after resolving the table's 180-degree symmetry."""

    forward = np.asarray(forward_poses, dtype=np.float64)
    backward = np.asarray(backward_poses, dtype=np.float64)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[1:] != (4, 4):
        raise ValueError("forward and backward poses must be matching Nx4x4 arrays")
    if len(forward) == 0:
        raise ValueError("bidirectional pose tracks must not be empty")
    forward, _ = make_pose_continuous(forward, symmetries=symmetries)
    backward, _ = make_pose_continuous(backward[::-1], symmetries=symmetries)
    backward = backward[::-1]

    fused = np.repeat(np.eye(4, dtype=np.float64)[None], len(forward), axis=0)
    translation_errors = np.empty(len(forward), dtype=np.float64)
    rotation_errors = np.empty(len(forward), dtype=np.float64)
    selected_symmetries = np.empty(len(forward), dtype=np.int64)
    for index, (left, right) in enumerate(zip(forward, backward, strict=True)):
        translation, rotation, symmetry_index = pose_errors(
            left, right, symmetries=symmetries
        )
        equivalent = right @ _transform(
            symmetries[symmetry_index], f"symmetry {symmetry_index}"
        )
        rotations = Rotation.from_matrix(np.stack((left[:3, :3], equivalent[:3, :3])))
        fused[index, :3, :3] = Slerp([0.0, 1.0], rotations)([0.5]).as_matrix()[0]
        fused[index, :3, 3] = 0.5 * (left[:3, 3] + equivalent[:3, 3])
        translation_errors[index] = translation
        rotation_errors[index] = rotation
        selected_symmetries[index] = symmetry_index
        _transform(fused[index], f"fused pose {index}")
    fused, _ = make_pose_continuous(fused, symmetries=symmetries)
    return fused, translation_errors, rotation_errors, selected_symmetries


def evaluate_rendered_alignment(
    *,
    observed_depth_m: np.ndarray,
    rendered_depth_m: np.ndarray,
    observed_mask: np.ndarray | None,
    maximum_occlusion_depth_error_m: float,
    maximum_median_absolute_depth_error_m: float,
    minimum_depth_overlap_fraction: float,
    minimum_rendered_mask_explained_fraction: float,
) -> RenderedAlignmentMetrics:
    """Compare rendered mesh depth with stereo depth and an optional audited mask."""

    observed = np.asarray(observed_depth_m, dtype=np.float64)
    rendered = np.asarray(rendered_depth_m, dtype=np.float64)
    if observed.shape != rendered.shape or observed.ndim != 2:
        raise ValueError("observed and rendered depth must be matching 2D arrays")
    if (
        maximum_occlusion_depth_error_m <= 0.0
        or maximum_median_absolute_depth_error_m <= 0.0
    ):
        raise ValueError("depth tolerances must be positive")
    for name, value in (
        ("minimum_depth_overlap_fraction", minimum_depth_overlap_fraction),
        (
            "minimum_rendered_mask_explained_fraction",
            minimum_rendered_mask_explained_fraction,
        ),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")

    valid_observed = np.isfinite(observed) & (observed > 0.0)
    rendered_mask, occluded_rendered_mask = visible_rendered_mask(
        observed,
        rendered,
        maximum_occlusion_depth_error_m=maximum_occlusion_depth_error_m,
    )
    raw_rendered_mask = rendered_mask | occluded_rendered_mask
    comparison_region = rendered_mask.copy()
    if observed_mask is not None:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError("observed mask must match depth dimensions")
        comparison_region &= mask
    overlap = valid_observed & comparison_region
    raw_rendered_pixels = int(np.count_nonzero(raw_rendered_mask))
    occluded_rendered_pixels = int(np.count_nonzero(occluded_rendered_mask))
    rendered_pixels = int(np.count_nonzero(rendered_mask))
    overlap_pixels = int(np.count_nonzero(overlap))
    comparison_pixels = int(np.count_nonzero(comparison_region))
    overlap_fraction = overlap_pixels / comparison_pixels if comparison_pixels else 0.0
    median_error: float | None = (
        float(np.median(np.abs(observed[overlap] - rendered[overlap])))
        if overlap_pixels
        else None
    )

    observed_mask_pixels = None
    raw_precision = None
    raw_explained_fraction = None
    explained_fraction = None
    if observed_mask is not None:
        observed_mask_pixels = int(np.count_nonzero(mask))
        raw_intersection_pixels = int(np.count_nonzero(mask & raw_rendered_mask))
        raw_precision = (
            raw_intersection_pixels / raw_rendered_pixels
            if raw_rendered_pixels
            else 0.0
        )
        raw_explained_fraction = (
            raw_intersection_pixels / observed_mask_pixels
            if observed_mask_pixels
            else 0.0
        )
        explained_fraction = (
            float(np.count_nonzero(mask & rendered_mask)) / observed_mask_pixels
            if observed_mask_pixels
            else 0.0
        )

    reasons = []
    if raw_rendered_pixels == 0:
        reasons.append("empty_render")
    elif rendered_pixels == 0:
        reasons.append("fully_occluded_render")
    if overlap_fraction < minimum_depth_overlap_fraction:
        reasons.append("insufficient_depth_overlap")
    if median_error is None or median_error > maximum_median_absolute_depth_error_m:
        reasons.append("rendered_depth_error")
    if raw_explained_fraction is not None and (
        raw_explained_fraction < minimum_rendered_mask_explained_fraction
    ):
        reasons.append("raw_rendered_mask_does_not_explain_observation")
    return RenderedAlignmentMetrics(
        raw_rendered_pixels=raw_rendered_pixels,
        occluded_rendered_pixels=occluded_rendered_pixels,
        rendered_pixels=rendered_pixels,
        valid_observed_pixels=int(np.count_nonzero(valid_observed)),
        depth_overlap_pixels=overlap_pixels,
        depth_overlap_fraction=float(overlap_fraction),
        median_absolute_depth_error_m=median_error,
        observed_mask_pixels=observed_mask_pixels,
        raw_rendered_mask_precision=raw_precision,
        raw_rendered_mask_explained_fraction=raw_explained_fraction,
        rendered_mask_explained_fraction=explained_fraction,
        passes_gate=not reasons,
        rejection_reasons=tuple(reasons),
    )


def make_pose_continuous(
    poses: np.ndarray,
    *,
    symmetries: tuple[np.ndarray, ...] = table_symmetry_transforms(),
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError("poses must be a non-empty Nx4x4 array")
    output = np.empty_like(values)
    output[0] = _transform(values[0], "pose 0")
    selected = [0]
    for frame in range(1, len(values)):
        candidates = []
        current = _transform(values[frame], f"pose {frame}")
        for index, symmetry in enumerate(symmetries):
            equivalent = current @ _transform(symmetry, f"symmetry {index}")
            translation = np.linalg.norm(output[frame - 1, :3, 3] - equivalent[:3, 3])
            rotation = Rotation.from_matrix(
                output[frame - 1, :3, :3].T @ equivalent[:3, :3]
            ).magnitude()
            candidates.append((float(translation + rotation), index, equivalent))
        _, index, output[frame] = min(candidates, key=lambda value: (value[0], value[1]))
        selected.append(index)
    return output, tuple(selected)


def interpolate_pose_trajectory(
    sampled_frame_indices: np.ndarray,
    sampled_poses: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    frames = np.asarray(sampled_frame_indices, dtype=np.int64)
    poses = np.asarray(sampled_poses, dtype=np.float64)
    if (
        frame_count <= 0
        or frames.ndim != 1
        or len(frames) < 2
        or poses.shape != (len(frames), 4, 4)
        or frames[0] != 0
        or frames[-1] != frame_count - 1
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError("sampled poses must cover sorted unique endpoints of the episode")
    continuous, _ = make_pose_continuous(poses)
    targets = np.arange(frame_count, dtype=np.float64)
    translation = np.column_stack(
        [np.interp(targets, frames, continuous[:, axis, 3]) for axis in range(3)]
    )
    rotation = Slerp(frames.astype(np.float64), Rotation.from_matrix(continuous[:, :3, :3]))(
        targets
    ).as_matrix()
    output = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    output[:, :3, :3] = rotation
    output[:, :3, 3] = translation
    return output
