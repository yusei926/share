#!/usr/bin/env python3
"""Fit a D405 camera mount against real masks by rendering the exact V1 CAD.

The fit is an offline calibration proposal.  It needs only the real RGB mask,
recorded encoder FK, the source head-stereo table pose, camera intrinsics, and
the immutable assembled-table CAD.  Simulator state, contacts, and rendered
scene images are never inputs.  A fitted mount is not applied automatically:
it must pass a separate held-out real-image gate first.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from data.flip_table_data_augmentation.object_pose.robot_silhouette import RobotSilhouetteRenderer


SCHEMA_VERSION = "team_ramen_flip_table_wrist_handeye_cad_alignment/v1"
SIDES = ("left", "right")
MINIMUM_FRAMES = 3
MAXIMUM_SOURCE_FRAME_DEFAULT = 50
MAX_TRANSLATION_M = 0.060
MAX_ROTATION_DEG = 20.0
MINIMUM_STEREO_PAIRS_FOR_HAND_EYE = 3
MAXIMUM_STEREO_TRANSLATION_P95_M_FOR_HAND_EYE = 0.010
MAXIMUM_STEREO_ROTATION_P95_DEG_FOR_HAND_EYE = 3.0
MAXIMUM_TEMPORAL_TRANSLATION_P95_M_FOR_HAND_EYE = 0.010
MAXIMUM_TEMPORAL_ROTATION_P95_DEG_FOR_HAND_EYE = 2.0


@dataclass(frozen=True)
class Observation:
    ordinal: int
    source_frame_index: int
    rgb_path: Path
    mask: np.ndarray
    root_from_wrist: np.ndarray
    robot_q_current: np.ndarray
    hand_state: np.ndarray


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(f"{label} has an invalid homogeneous row")
    return matrix


def accepted_root_from_table(alignment: dict[str, Any]) -> np.ndarray:
    """Return a table pose only after the upstream stereo-CAD gate accepted it."""

    if alignment.get("schema_version") != "team_ramen_flip_table_source_cad_alignment/v1":
        raise ValueError("source alignment schema is unexpected")
    if alignment.get("accepted_for_fixed_scene_proposal") is not True:
        raise ValueError(
            "source alignment was not accepted by its stereo-CAD consistency gate; "
            "it cannot anchor wrist hand-eye calibration"
        )
    stereo = alignment.get("stereo_agreement")
    temporal = alignment.get("temporal_consistency")
    if not isinstance(stereo, dict) or not isinstance(temporal, dict):
        raise ValueError("source alignment lacks stereo or temporal quality evidence")
    quality = {
        "accepted_paired_frames": (
            stereo.get("accepted_paired_frames"), MINIMUM_STEREO_PAIRS_FOR_HAND_EYE
        ),
        "accepted_translation_p95_m": (
            stereo.get("accepted_translation_p95_m"), MAXIMUM_STEREO_TRANSLATION_P95_M_FOR_HAND_EYE
        ),
        "accepted_rotation_p95_deg": (
            stereo.get("accepted_rotation_p95_deg"), MAXIMUM_STEREO_ROTATION_P95_DEG_FOR_HAND_EYE
        ),
        "temporal_translation_p95_m": (
            temporal.get("translation_spread_p95_m"), MAXIMUM_TEMPORAL_TRANSLATION_P95_M_FOR_HAND_EYE
        ),
        "temporal_rotation_p95_deg": (
            temporal.get("rotation_spread_p95_deg"), MAXIMUM_TEMPORAL_ROTATION_P95_DEG_FOR_HAND_EYE
        ),
    }
    failed = [
        name
        for name, (value, threshold) in quality.items()
        if not isinstance(value, (int, float))
        or not np.isfinite(value)
        or (value < threshold if name == "accepted_paired_frames" else value > threshold)
    ]
    if failed:
        raise ValueError(
            "source alignment is too uncertain for wrist hand-eye calibration: "
            + ", ".join(failed)
        )
    return _matrix(alignment.get("fixed_scene_root_from_table"), "fixed source table pose")


def _transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()))


def _tool_transform(config: PipelineConfig, side: str) -> np.ndarray:
    raw = config.raw["source_contract"]["fk_tool_transforms"][side]
    translation = np.asarray(raw["translation_m"], dtype=np.float64)
    quaternion = np.asarray(raw["quaternion_xyzw"], dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(
        np.concatenate((translation, quaternion))
    ).all():
        raise ValueError(f"pipeline config {side} tool transform is invalid")
    return _transform(translation, Rotation.from_quat(quaternion).as_matrix())


def _intrinsic(manifest: dict[str, Any], side: str) -> np.ndarray:
    view = manifest.get("pose_views", {}).get(f"{side}_wrist")
    if not isinstance(view, dict):
        raise ValueError(f"prepared manifest lacks {side} wrist calibration")
    intrinsic = np.asarray(view.get("intrinsic_matrix_px"), dtype=np.float64)
    if intrinsic.size != 9:
        raise ValueError(f"prepared manifest {side} wrist intrinsics are malformed")
    intrinsic = intrinsic.reshape(3, 3)
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0 or not np.isfinite(intrinsic).all():
        raise ValueError(f"prepared manifest {side} wrist intrinsics are invalid")
    return intrinsic


def _mask_records(mask_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = mask_manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("mask manifest lacks frame records")
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("ordinal"), int):
            raise ValueError("mask frame record is malformed")
        result[int(record["ordinal"])] = record
    return result


def _nominal_wrist_from_camera(frame: dict[str, Any], side: str, root_from_wrist: np.ndarray) -> np.ndarray:
    view = frame.get("views", {}).get(f"{side}_wrist")
    if not isinstance(view, dict):
        raise ValueError(f"prepared frame lacks {side} wrist view")
    root_from_camera = _matrix(view.get("robot_root_from_rectified_opencv"), "nominal root_from_camera")
    return np.linalg.inv(root_from_wrist) @ root_from_camera


def _observations(
    *, input_manifest: dict[str, Any], mask_manifest: dict[str, Any], input_root: Path,
    mask_root: Path, side: str, maximum_source_frame: int, config: PipelineConfig,
) -> tuple[list[Observation], np.ndarray]:
    records = input_manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("prepared input manifest lacks frames")
    by_ordinal = _mask_records(mask_manifest)
    tool_from_eef = np.linalg.inv(_tool_transform(config, side))
    values: list[Observation] = []
    nominal_values: list[np.ndarray] = []
    side_index = SIDES.index(side)
    for frame in records:
        if not isinstance(frame, dict):
            raise ValueError("prepared input frame is malformed")
        ordinal = frame.get("ordinal")
        source_frame = frame.get("source_frame_index")
        if not isinstance(ordinal, int) or not isinstance(source_frame, int) or source_frame > maximum_source_frame:
            continue
        mask_view = by_ordinal.get(ordinal, {}).get("views", {}).get(f"{side}_wrist")
        input_view = frame.get("views", {}).get(f"{side}_wrist")
        if not isinstance(mask_view, dict) or not isinstance(input_view, dict):
            continue
        relative_mask, relative_rgb = mask_view.get("selected_mask"), input_view.get("rgb")
        if not isinstance(relative_mask, str) or not isinstance(relative_rgb, str):
            continue
        mask_path = (mask_root / relative_mask).resolve()
        rgb_path = (input_root / relative_rgb).resolve()
        if mask_root.resolve() not in mask_path.parents or input_root.resolve() not in rgb_path.parents:
            raise ValueError("manifest path escapes its evidence directory")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (480, 640):
            continue
        eef = np.asarray(frame.get("eef_current_root_from_fk"), dtype=np.float64)
        if eef.shape != (2, 16) or not np.isfinite(eef).all():
            raise ValueError("prepared frame lacks finite two-side EEF FK")
        root_from_wrist = eef[side_index].reshape(4, 4) @ tool_from_eef
        nominal_values.append(_nominal_wrist_from_camera(frame, side, root_from_wrist))
        robot_q_current = np.asarray(frame.get("robot_q_current"), dtype=np.float64)
        hand_state = np.asarray(frame.get("hand_state"), dtype=np.float64)
        if robot_q_current.shape != (36,) or hand_state.shape != (2,):
            raise ValueError("prepared frame lacks robot_q_current or hand_state")
        values.append(
            Observation(
                ordinal,
                source_frame,
                rgb_path,
                mask > 0,
                root_from_wrist,
                robot_q_current,
                hand_state,
            )
        )
    if len(values) < MINIMUM_FRAMES:
        raise ValueError(f"{side} has fewer than {MINIMUM_FRAMES} static D405 mask observations")
    # The fixed mount is independent of pose. Keep the medoid nominal so a
    # corrupted input frame cannot define the search origin.
    scores = [
        float(np.median([
            np.linalg.norm(item[:3, 3] - candidate[:3, 3]) / 0.01
            + _rotation_distance_deg(item, candidate) / 2.0
            for item in nominal_values
        ]))
        for candidate in nominal_values
    ]
    return values, nominal_values[int(np.argmin(scores))]


def _delta_transform(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("hand-eye delta must be six finite values")
    return _transform(values[:3], Rotation.from_rotvec(values[3:]).as_matrix())


def _candidate_mount(nominal: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    # Left multiplication expresses translation and rotation in wrist-link
    # coordinates, matching the simulator camera parent convention.
    return _delta_transform(parameters) @ nominal


def _mask_metrics(
    rendered_depths: np.ndarray,
    observations: list[Observation],
    robot_occlusions: list[np.ndarray],
) -> tuple[float, list[dict[str, float]]]:
    if rendered_depths.shape != (len(observations), 480, 640):
        raise ValueError("rendered depth batch has an unexpected shape")
    records: list[dict[str, float]] = []
    scores = []
    for rendered_depth, observation, robot_occlusion in zip(
        rendered_depths, observations, robot_occlusions, strict=True
    ):
        rendered = np.asarray(rendered_depth > 0.0, dtype=bool) & ~robot_occlusion
        observed = observation.mask
        intersection = int(np.count_nonzero(rendered & observed))
        union = int(np.count_nonzero(rendered | observed))
        rendered_pixels = int(np.count_nonzero(rendered))
        observed_pixels = int(np.count_nonzero(observed))
        iou = intersection / union if union else 0.0
        precision = intersection / rendered_pixels if rendered_pixels else 0.0
        explained = intersection / observed_pixels if observed_pixels else 0.0
        observed_robot_overlap_pixels = int(np.count_nonzero(observed & robot_occlusion))
        observed_robot_overlap_fraction = (
            observed_robot_overlap_pixels / observed_pixels if observed_pixels else 0.0
        )
        # IoU prevents a camera candidate from winning merely by rendering a
        # tiny CAD fragment. Precision/recall are retained for diagnosis.
        score = 0.70 * iou + 0.15 * precision + 0.15 * explained
        scores.append(score)
        records.append({
            "ordinal": float(observation.ordinal), "source_frame_index": float(observation.source_frame_index),
            "iou": iou, "precision": precision, "explained_fraction": explained,
            "rendered_pixels": float(rendered_pixels), "observed_pixels": float(observed_pixels),
            "robot_occlusion_fraction": float(robot_occlusion.mean()),
            "observed_robot_overlap_pixels": float(observed_robot_overlap_pixels),
            "observed_robot_overlap_fraction": observed_robot_overlap_fraction,
        })
    return float(np.mean(scores)), records


def coordinate_descent(
    evaluate: Callable[[np.ndarray], float], *, translation_steps_m: tuple[float, ...],
    rotation_steps_deg: tuple[float, ...], passes_per_scale: int = 3,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Bounded, deterministic six-DoF coordinate search around the mount."""

    if len(translation_steps_m) != len(rotation_steps_deg) or passes_per_scale < 1:
        raise ValueError("coordinate-descent schedule is invalid")
    current = np.zeros(6, dtype=np.float64)
    best_score = float(evaluate(current))
    trace = [{"parameters": current.tolist(), "score": best_score, "stage": "initial"}]
    for translation_step, rotation_step_deg in zip(translation_steps_m, rotation_steps_deg, strict=True):
        for _ in range(passes_per_scale):
            candidates = [current.copy()]
            for index in range(6):
                step = translation_step if index < 3 else math.radians(rotation_step_deg)
                for direction in (-1.0, 1.0):
                    proposal = current.copy()
                    proposal[index] += direction * step
                    if (
                        np.max(np.abs(proposal[:3])) <= MAX_TRANSLATION_M
                        and np.max(np.abs(proposal[3:])) <= math.radians(MAX_ROTATION_DEG)
                    ):
                        candidates.append(proposal)
            scored = [(float(evaluate(candidate)), candidate) for candidate in candidates]
            score, proposal = max(scored, key=lambda value: value[0])
            if score <= best_score + 1.0e-8:
                break
            current, best_score = proposal, score
            trace.append({"parameters": current.tolist(), "score": best_score, "stage": "improve"})
    return current, best_score, trace


def _render_depths(camera_from_tables: np.ndarray, intrinsic: np.ndarray, *, mesh_tensors: Any, glctx: Any, render_function: Callable[..., Any], torch_module: Any) -> np.ndarray:
    output = []
    with torch_module.inference_mode():
        for start in range(0, len(camera_from_tables), 32):
            poses = torch_module.as_tensor(camera_from_tables[start:start + 32], dtype=torch_module.float32, device="cuda")
            _, depth, _ = render_function(K=intrinsic, H=480, W=640, ob_in_cams=poses, glctx=glctx, context="cuda", get_normal=False, mesh_tensors=mesh_tensors)
            output.append(depth.detach().cpu())
    return torch_module.cat(output, dim=0).numpy().astype(np.float32)


def _overlay(path: Path, rgb_path: Path, observed: np.ndarray, rendered: np.ndarray) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    overlay = image.copy()
    overlay[observed] = (0.55 * overlay[observed] + np.asarray((0, 180, 0))).astype(np.uint8)
    only_rendered = rendered & ~observed
    overlay[only_rendered] = (0.55 * overlay[only_rendered] + np.asarray((0, 0, 220))).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def align_side(
    *, side: str, observations: list[Observation], nominal: np.ndarray, root_from_table: np.ndarray,
    intrinsic: np.ndarray, mesh_tensors: Any, glctx: Any, render_function: Callable[..., Any], torch_module: Any,
    robot_renderer: RobotSilhouetteRenderer,
) -> dict[str, Any]:
    latest_depths: np.ndarray | None = None
    latest_records: list[dict[str, float]] = []

    def evaluate(parameters: np.ndarray) -> float:
        nonlocal latest_depths, latest_records
        mount = _candidate_mount(nominal, parameters)
        cameras = np.stack([observation.root_from_wrist @ mount for observation in observations])
        camera_from_tables = np.linalg.inv(cameras) @ root_from_table
        depths = _render_depths(camera_from_tables, intrinsic, mesh_tensors=mesh_tensors, glctx=glctx, render_function=render_function, torch_module=torch_module)
        robot_occlusions = [
            robot_renderer.render(
                robot_q_current=observation.robot_q_current,
                hand_state=observation.hand_state,
                root_from_camera=camera,
                intrinsic_matrix=intrinsic,
                width=640,
                height=480,
            )[0]
            for observation, camera in zip(observations, cameras, strict=True)
        ]
        score, records = _mask_metrics(depths, observations, robot_occlusions)
        latest_depths, latest_records = depths, records
        return score

    parameters, score, trace = coordinate_descent(
        evaluate,
        translation_steps_m=(0.030, 0.012, 0.005),
        rotation_steps_deg=(8.0, 3.0, 1.0),
    )
    fitted = _candidate_mount(nominal, parameters)
    # Re-render after the final call in case the last evaluated coordinate did
    # not equal the selected optimum.
    final_score = evaluate(parameters)
    assert latest_depths is not None
    assert latest_records
    return {
        "status": "proposal_requires_heldout_validation",
        "observation_count": len(observations),
        "mask_score": final_score,
        "nominal_wrist_from_rectified_opencv_camera": nominal.tolist(),
        "fitted_wrist_from_rectified_opencv_camera": fitted.tolist(),
        "wrist_frame_delta_translation_m": parameters[:3].tolist(),
        "wrist_frame_delta_rotation_deg": np.degrees(parameters[3:]).tolist(),
        "search_trace": trace,
        "per_frame": latest_records,
        "rendered_depths": latest_depths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--source-alignment", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--robofinals-root",
        type=Path,
        default=Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals")),
    )
    parser.add_argument("--maximum-source-frame", type=int, default=MAXIMUM_SOURCE_FRAME_DEFAULT)
    args = parser.parse_args()
    if args.maximum_source_frame < 0:
        raise ValueError("--maximum-source-frame must be non-negative")
    input_manifest_path = args.input_manifest.expanduser().resolve()
    mask_manifest_path = args.mask_manifest.expanduser().resolve()
    alignment_path = args.source_alignment.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = load_pipeline_config(args.config.expanduser().resolve())
    if sha256_file(mesh_path) != config.object_pose_runtime.assembled_table_mesh_sha256:
        raise ValueError("assembled table mesh digest differs from the pinned V1 CAD")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    root_from_table = accepted_root_from_table(alignment)
    try:
        import nvdiffrast.torch as dr
        import torch
        from Utils import make_mesh_tensors, nvdiffrast_render
    except ImportError as error:
        raise RuntimeError("run this tool through run_object_pose_runtime.sh inside the verified V1 runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CAD wrist hand-eye alignment requires CUDA")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("assembled table CAD is not a usable triangle mesh")
    mesh_tensors = make_mesh_tensors(mesh)
    glctx = dr.RasterizeCudaContext()
    robot_urdf = (
        args.robofinals_root.expanduser().resolve()
        / config.object_pose_runtime.robot_visual_urdf_relative_path
    )
    robot_renderer = RobotSilhouetteRenderer(
        robot_urdf,
        expected_sha256=config.object_pose_runtime.robot_visual_urdf_sha256,
        dilation_px=config.object_pose_runtime.auxiliary_robot_silhouette_dilation_px,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera-calibration proposal only",
        "source_inputs": {
            "input_manifest": str(input_manifest_path), "mask_manifest": str(mask_manifest_path),
            "source_alignment": str(alignment_path), "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "source_alignment_sha256": sha256_file(alignment_path),
            "source_alignment_accepted": True,
            "maximum_source_frame": args.maximum_source_frame,
        },
        "fixed_source_root_from_table": root_from_table.tolist(),
        "robot_occlusion_model": {
            "urdf": str(robot_urdf),
            "urdf_sha256": config.object_pose_runtime.robot_visual_urdf_sha256,
            "dilation_px": config.object_pose_runtime.auxiliary_robot_silhouette_dilation_px,
            "use": "offline D405 self-occlusion exclusion only",
        },
        "sides": {},
        "limitations": [
            "The fit uses only the static source interval already accepted by head-stereo CAD alignment.",
            "No simulator pose, contact, rendering, or policy input is used in the objective.",
            "The result is a proposal and must pass held-out real D405 image validation before a runtime mount update.",
        ],
    }
    for side in SIDES:
        try:
            observations, nominal = _observations(
                input_manifest=input_manifest, mask_manifest=mask_manifest, input_root=input_manifest_path.parent,
                mask_root=mask_manifest_path.parent, side=side, maximum_source_frame=args.maximum_source_frame, config=config,
            )
        except ValueError as error:
            report["sides"][side] = {
                "status": "insufficient_evidence",
                "rejection_reason": str(error),
            }
            continue
        result = align_side(side=side, observations=observations, nominal=nominal, root_from_table=root_from_table, intrinsic=_intrinsic(input_manifest, side), mesh_tensors=mesh_tensors, glctx=glctx, render_function=nvdiffrast_render, torch_module=torch, robot_renderer=robot_renderer)
        depths = result.pop("rendered_depths")
        for observation, depth in zip(observations, depths, strict=True):
            _overlay(output_dir / "debug" / side / f"frame-{observation.source_frame_index:06d}.png", observation.rgb_path, observation.mask, depth > 0.0)
        report["sides"][side] = result
    atomic_write_json(output_dir / "wrist_handeye_cad_alignment.json", report)
    print(
        json.dumps(
            {
                side: {
                    "status": report["sides"][side]["status"],
                    "mask_score": report["sides"][side].get("mask_score"),
                }
                for side in SIDES
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
