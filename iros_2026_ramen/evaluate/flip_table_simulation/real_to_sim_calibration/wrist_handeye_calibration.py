#!/usr/bin/env python3
"""Propose a D405-to-wrist-link extrinsic from static real table observations.

This program is deliberately offline-only.  It combines a fixed table pose
accepted from head-stereo CAD alignment, recorded G1 joint FK, and a selected
D405 table mask.  It never reads simulator state and does not modify the
runtime camera configuration.  The output is a proposal that still requires a
held-out image gate before a simulator mount change is considered.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from evaluate.flip_table_simulation.container_overlay.policy.cv_rule_based.vision import (
    _quadrilateral_from_mask,
)


SCHEMA_VERSION = "team_ramen_flip_table_wrist_handeye_proposal/v1"
SIDES = ("left", "right")
TABLE_LENGTH_M = 0.580
TABLE_DEPTH_M = 0.420
MINIMUM_INLIERS = 3
MAXIMUM_CANDIDATE_TRANSLATION_FROM_NOMINAL_M = 0.180
MAXIMUM_CANDIDATE_ROTATION_FROM_NOMINAL_DEG = 35.0
MAXIMUM_INLIER_TRANSLATION_M = 0.030
MAXIMUM_INLIER_ROTATION_DEG = 8.0


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(f"{label} has an invalid homogeneous row")
    return matrix


def _transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude())
    )


def _tool_transform(config: PipelineConfig, side: str) -> np.ndarray:
    transforms = config.raw["source_contract"].get("fk_tool_transforms")
    if not isinstance(transforms, dict) or not isinstance(transforms.get(side), dict):
        raise ValueError(f"pipeline config lacks eef tool transform for {side}")
    value = transforms[side]
    translation = np.asarray(value.get("translation_m"), dtype=np.float64)
    quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(
        np.concatenate((translation, quaternion))
    ).all():
        raise ValueError(f"pipeline config has invalid {side} eef tool transform")
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1.0e-5):
        raise ValueError(f"pipeline config {side} eef tool quaternion is not unit length")
    return _transform(translation, Rotation.from_quat(quaternion).as_matrix())


def _camera_intrinsic(manifest: dict[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    camera_name = f"{side}_wrist"
    camera = manifest.get("pose_views", {}).get(camera_name)
    if not isinstance(camera, dict):
        raise ValueError(f"prepared manifest lacks {camera_name} calibration")
    intrinsic = np.asarray(camera.get("intrinsic_matrix_px"), dtype=np.float64).reshape(3, 3)
    # The prepared wrist PNG has already been inverse-Brown rectified. Applying
    # raw D405 coefficients here would distort it a second time.
    distortion = np.zeros(5, dtype=np.float64)
    if not np.isfinite(intrinsic).all() or not np.isfinite(distortion).all():
        raise ValueError(f"prepared manifest {camera_name} intrinsics are non-finite")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError(f"prepared manifest {camera_name} focal length is invalid")
    return intrinsic, distortion


def _camera_from_table_candidates(
    mask: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return all positive-depth physical PnP corner assignments.

    A rectangle has cyclic and mirror ambiguity.  The later FK/nominal-mount
    gate resolves it globally, instead of assuming a camera-specific image
    order.  The mask is selected offline evidence, never a policy feature.
    """

    if mask.ndim != 2 or mask.shape != (480, 640):
        raise ValueError("selected D405 mask must be 640x480")
    corners, _ = _quadrilateral_from_mask(mask.astype(np.uint8))
    object_corners = np.asarray(
        (
            (-0.5 * TABLE_LENGTH_M, -0.5 * TABLE_DEPTH_M, 0.0),
            (0.5 * TABLE_LENGTH_M, -0.5 * TABLE_DEPTH_M, 0.0),
            (0.5 * TABLE_LENGTH_M, 0.5 * TABLE_DEPTH_M, 0.0),
            (-0.5 * TABLE_LENGTH_M, 0.5 * TABLE_DEPTH_M, 0.0),
        ),
        dtype=np.float64,
    )
    candidates: list[np.ndarray] = []
    for reverse in (False, True):
        sequence = corners[::-1] if reverse else corners
        for shift in range(4):
            image_corners = np.roll(sequence, shift, axis=0)
            ok, rotation_vector, translation_vector = cv2.solvePnP(
                object_corners,
                image_corners,
                intrinsic,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok or float(translation_vector[2, 0]) <= 0.10:
                continue
            rotation, _ = cv2.Rodrigues(rotation_vector)
            candidate = _transform(translation_vector.reshape(3), rotation)
            if not any(np.allclose(candidate, existing, atol=1.0e-8) for existing in candidates):
                candidates.append(candidate)
    if not candidates:
        raise ValueError("D405 table-mask PnP has no positive-depth solution")
    return tuple(candidates)


def _robust_mean(candidates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(candidates) < MINIMUM_INLIERS:
        raise ValueError("fewer than three candidate hand-eye transforms")
    translations = np.stack([candidate[:3, 3] for candidate in candidates])
    # Do not average rotations before rejecting gross PnP ambiguity. A single
    # 90/180-degree rectangle assignment can move that mean far enough to
    # reject every otherwise coherent static-frame observation. Select the
    # transform with the smallest median normalized pairwise residual first.
    pairwise_scores = []
    for candidate in candidates:
        translation_residuals = np.linalg.norm(translations - candidate[:3, 3], axis=1)
        rotation_residuals = np.asarray(
            [_rotation_distance_deg(candidate, other) for other in candidates], dtype=np.float64
        )
        pairwise_scores.append(
            float(
                np.median(
                    translation_residuals / MAXIMUM_INLIER_TRANSLATION_M
                    + rotation_residuals / MAXIMUM_INLIER_ROTATION_DEG
                )
            )
        )
    center = candidates[int(np.argmin(pairwise_scores))]
    translation_error = np.linalg.norm(translations - center[:3, 3], axis=1)
    rotation_error = np.asarray(
        [_rotation_distance_deg(center, candidate) for candidate in candidates], dtype=np.float64
    )
    inliers = (translation_error <= MAXIMUM_INLIER_TRANSLATION_M) & (
        rotation_error <= MAXIMUM_INLIER_ROTATION_DEG
    )
    if int(np.count_nonzero(inliers)) < MINIMUM_INLIERS:
        raise ValueError("no stable hand-eye consensus across static table frames")
    kept_translations = translations[inliers]
    kept_rotations = Rotation.from_matrix(
        np.stack([candidate[:3, :3] for candidate, keep in zip(candidates, inliers, strict=True) if keep])
    )
    result = _transform(np.median(kept_translations, axis=0), kept_rotations.mean().as_matrix())
    return result, inliers


def _mask_path(mask_root: Path, record: dict[str, Any], side: str) -> Path | None:
    view = record.get("views", {}).get(f"{side}_wrist")
    if not isinstance(view, dict):
        return None
    value = view.get("selected_mask")
    if not isinstance(value, str) or not value:
        return None
    path = (mask_root / value).resolve()
    if mask_root.resolve() not in path.parents or not path.is_file():
        return None
    return path


def _by_ordinal(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        ordinal = record.get("ordinal")
        if not isinstance(ordinal, int):
            raise ValueError("mask manifest frame lacks integer ordinal")
        result[ordinal] = record
    return result


def _nominal_wrist_from_camera(frame: dict[str, Any], side: str, wrist_root: np.ndarray) -> np.ndarray:
    view = frame.get("views", {}).get(f"{side}_wrist")
    if not isinstance(view, dict):
        raise ValueError(f"prepared frame lacks {side} wrist view")
    root_from_camera = _matrix(
        view.get("robot_root_from_rectified_opencv"), f"{side} nominal root_from_camera"
    )
    return np.linalg.inv(wrist_root) @ root_from_camera


def _candidate_record(
    *,
    frame: dict[str, Any],
    mask_record: dict[str, Any],
    side: str,
    root_from_table: np.ndarray,
    tool_from_eef: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    mask_root: Path,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    path = _mask_path(mask_root, mask_record, side)
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    eef = np.asarray(frame.get("eef_current_root_from_fk"), dtype=np.float64)
    if eef.shape != (2, 16):
        raise ValueError("prepared frame has invalid two-side EEF FK")
    wrist_root = eef[SIDES.index(side)].reshape(4, 4) @ tool_from_eef
    # ``tool_from_eef`` is the inverse of the configured wrist-to-tool frame.
    nominal = _nominal_wrist_from_camera(frame, side, wrist_root)
    try:
        camera_candidates = _camera_from_table_candidates(mask, intrinsic, distortion)
    except ValueError:
        # A selected segmentation mask can still have an occluded rim or no
        # usable quadrilateral in one wrist view.  It is one rejected source
        # observation, not a reason to discard all other static evidence.
        return None
    wrist_candidates = tuple(
        np.linalg.inv(wrist_root) @ root_from_table @ np.linalg.inv(camera_from_table)
        for camera_from_table in camera_candidates
    )
    ranked = sorted(
        (
            (
                float(np.linalg.norm(candidate[:3, 3] - nominal[:3, 3])),
                _rotation_distance_deg(nominal, candidate),
                candidate,
            )
            for candidate in wrist_candidates
        ),
        key=lambda value: (value[0], value[1]),
    )
    translation_delta, rotation_delta, selected = ranked[0]
    if (
        translation_delta > MAXIMUM_CANDIDATE_TRANSLATION_FROM_NOMINAL_M
        or rotation_delta > MAXIMUM_CANDIDATE_ROTATION_FROM_NOMINAL_DEG
    ):
        return None
    return selected, {
        "ordinal": int(frame["ordinal"]),
        "source_frame_index": int(frame["source_frame_index"]),
        "mask": str(path),
        "pnp_solution_count": len(camera_candidates),
        "translation_from_nominal_m": translation_delta,
        "rotation_from_nominal_deg": rotation_delta,
    }


def calibrate(
    *, input_manifest_path: Path, mask_manifest_path: Path, source_alignment_path: Path,
    output_path: Path, maximum_source_frame: int, config: PipelineConfig,
) -> dict[str, Any]:
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(source_alignment_path.read_text(encoding="utf-8"))
    if alignment.get("schema_version") != "team_ramen_flip_table_source_cad_alignment/v1":
        raise ValueError("source alignment has an unexpected schema")
    root_from_table = _matrix(
        alignment.get("fixed_scene_root_from_table"), "fixed source root_from_table"
    )
    input_frames = input_manifest.get("frames")
    if not isinstance(input_frames, list):
        raise ValueError("prepared input manifest lacks frames")
    masks_by_ordinal = _by_ordinal(mask_manifest.get("frames", []))
    mask_root = mask_manifest_path.parent
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline calibration proposal only",
        "source_inputs": {
            "prepared_input_manifest": str(input_manifest_path),
            "selected_mask_manifest": str(mask_manifest_path),
            "source_cad_alignment": str(source_alignment_path),
            "maximum_source_frame": maximum_source_frame,
        },
        "fixed_source_root_from_table": root_from_table.tolist(),
        "sides": {},
        "limitations": [
            "Uses only frames in the static initial table interval accepted by head-stereo CAD alignment.",
            "Does not use simulator state, simulator rendering, contact labels, or policy inputs.",
            "A proposal requires independent held-out real-image validation before changing the simulator mount.",
        ],
    }
    for side in SIDES:
        intrinsic, distortion = _camera_intrinsic(input_manifest, side)
        tool_from_eef = np.linalg.inv(_tool_transform(config, side))
        candidates: list[np.ndarray] = []
        evidence: list[dict[str, Any]] = []
        skipped = 0
        for frame in input_frames:
            if not isinstance(frame, dict):
                raise ValueError("prepared input frame must be an object")
            if int(frame.get("source_frame_index", -1)) > maximum_source_frame:
                continue
            ordinal = frame.get("ordinal")
            if not isinstance(ordinal, int) or ordinal not in masks_by_ordinal:
                skipped += 1
                continue
            candidate = _candidate_record(
                frame=frame,
                mask_record=masks_by_ordinal[ordinal],
                side=side,
                root_from_table=root_from_table,
                tool_from_eef=tool_from_eef,
                intrinsic=intrinsic,
                distortion=distortion,
                mask_root=mask_root,
            )
            if candidate is None:
                skipped += 1
                continue
            transform, item = candidate
            candidates.append(transform)
            evidence.append(item)
        side_report: dict[str, Any] = {
            "candidate_count": len(candidates),
            "skipped_frame_count": skipped,
            "evidence": evidence,
            "status": "insufficient_evidence",
        }
        if len(candidates) >= MINIMUM_INLIERS:
            try:
                fitted, inliers = _robust_mean(candidates)
            except ValueError as error:
                side_report["rejection_reason"] = str(error)
            else:
                inlier_candidates = [candidate for candidate, keep in zip(candidates, inliers, strict=True) if keep]
                translation_residuals = [
                    float(np.linalg.norm(candidate[:3, 3] - fitted[:3, 3]))
                    for candidate in inlier_candidates
                ]
                rotation_residuals = [
                    _rotation_distance_deg(fitted, candidate) for candidate in inlier_candidates
                ]
                side_report.update(
                    {
                        "status": "proposal_requires_heldout_validation",
                        "wrist_from_rectified_opencv_camera": fitted.tolist(),
                        "inlier_count": int(np.count_nonzero(inliers)),
                        "inlier_ordinals": [
                            evidence[index]["ordinal"] for index, keep in enumerate(inliers) if keep
                        ],
                        "inlier_translation_residual_p95_m": float(np.quantile(translation_residuals, 0.95)),
                        "inlier_rotation_residual_p95_deg": float(np.quantile(rotation_residuals, 0.95)),
                    }
                )
        report["sides"][side] = side_report
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--maximum-source-frame", type=int, default=50)
    args = parser.parse_args()
    if args.maximum_source_frame < 0:
        raise ValueError("--maximum-source-frame must be non-negative")
    report = calibrate(
        input_manifest_path=args.input_manifest.expanduser().resolve(),
        mask_manifest_path=args.mask_manifest.expanduser().resolve(),
        source_alignment_path=args.source_alignment.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        maximum_source_frame=args.maximum_source_frame,
        config=load_pipeline_config(args.config.expanduser().resolve()),
    )
    print(json.dumps({side: report["sides"][side]["status"] for side in SIDES}, sort_keys=True))


if __name__ == "__main__":
    main()
