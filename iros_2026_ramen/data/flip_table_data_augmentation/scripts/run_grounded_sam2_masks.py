#!/usr/bin/env python3
"""Generate audited white-table masks at FoundationPose registration frames."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
from PIL import Image

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from data.flip_table_data_augmentation.object_pose.artifacts import (
    MANIFEST_SCHEMA_VERSION as RUNTIME_MANIFEST_SCHEMA_VERSION,
)
from data.flip_table_data_augmentation.object_pose.camera_views import (
    POSE_VIEW_NAMES,
    PRIMARY_POSE_VIEW,
)
from data.flip_table_data_augmentation.object_pose.segmentation import (
    MaskSequenceParameters,
    SEGMENTATION_SCHEMA_VERSION,
    evaluate_mask_candidates,
    filter_reachable_table_components,
    fuse_bidirectional_masks,
    refine_table_mask,
    registration_ordinals,
    select_mask_candidate_sequence,
    target_point_prompt,
)
from data.flip_table_data_augmentation.object_pose import INPUT_SCHEMA_VERSION
from data.flip_table_data_augmentation.object_pose.robot_silhouette import (
    RobotSilhouetteRenderer,
    robot_silhouette_coverage_is_plausible,
)


DEFAULT_ROBOFINALS_ROOT = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))


def _mask_rejection_reasons(gate: dict[str, object]) -> list[str]:
    reasons = []
    if int(gate["tracking_eligible_frames"]) < int(
        gate["minimum_tracking_eligible_frames"]
    ):
        reasons.append("insufficient_registration_mask_coverage")
    if gate["first_selected"] is not True:
        reasons.append("missing_initial_registration_mask")
    if gate["terminal_anchor_pass"] is not True:
        reasons.append("missing_terminal_registration_mask")
    return reasons


def _mask_observation_ordinals(
    frame_results: list[dict[str, object]],
) -> tuple[list[int], list[int], list[int]]:
    """Separate primary masks from auditable bimanual-wrist bridge frames."""

    primary = []
    wrist_bridges = []
    eligible = []
    auxiliary_views = POSE_VIEW_NAMES[1:]
    for record in frame_results:
        ordinal = int(record["ordinal"])
        views = record.get("views")
        if not isinstance(views, dict) or set(views) != set(POSE_VIEW_NAMES):
            raise ValueError("mask result lacks the ordered three-view contract")
        primary_present = (
            views[PRIMARY_POSE_VIEW].get("selected_candidate_index") is not None
        )
        wrist_pair_present = all(
            views[view_name].get("selected_candidate_index") is not None
            for view_name in auxiliary_views
        )
        if primary_present:
            primary.append(ordinal)
        elif wrist_pair_present:
            wrist_bridges.append(ordinal)
        if primary_present or wrist_pair_present:
            eligible.append(ordinal)
    return primary, wrist_bridges, eligible


def _terminal_source_gap(
    source_frame_indices: np.ndarray, ordinals: list[int]
) -> int | None:
    """Return the source-frame gap from the last observation to the endpoint."""

    if not ordinals:
        return None
    return int(source_frame_indices[-1] - source_frame_indices[ordinals[-1]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--robofinals-root",
        type=Path,
        default=DEFAULT_ROBOFINALS_ROOT,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"OpenCV could not write {temporary}")
    os.replace(temporary, path)


def _image_and_depth(
    input_root: Path,
    record: dict[str, object],
    view_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    views = record.get("views")
    if not isinstance(views, dict) or not isinstance(views.get(view_name), dict):
        raise ValueError(f"prepared frame lacks camera view {view_name}")
    view = views[view_name]
    rgb_path = input_root / str(view["rgb"])
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None or rgb_bgr.shape != (480, 640, 3) or rgb_bgr.dtype != np.uint8:
        raise ValueError(f"invalid prepared RGB frame: {view['rgb']}")
    if sha256_file(rgb_path) != view.get("rgb_sha256"):
        raise ValueError(f"prepared RGB hash differs: {rgb_path}")
    depth = None
    if "depth" in view:
        depth_path = input_root / str(view["depth"])
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.shape != (480, 640) or depth_mm.dtype != np.uint16:
            raise ValueError(f"invalid prepared depth frame: {view['depth']}")
        if sha256_file(depth_path) != view.get("depth_sha256"):
            raise ValueError(f"prepared depth hash differs: {depth_path}")
        depth = depth_mm.astype(np.float32) / 1000.0
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), depth


def _to_device(batch, device: str):
    return batch.to(device)


def _disable_unused_torchaudio() -> None:
    """Keep vision-only Transformers from loading V1's incompatible audio wheel."""

    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    def unavailable() -> bool:
        return False

    transformers_utils.is_torchaudio_available = unavailable
    import_utils.is_torchaudio_available = unavailable


def _detector_results(
    *,
    frames: list[tuple[int, Image.Image]],
    model_root: Path,
    prompt: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    candidate_limit: int,
) -> dict[int, dict[str, object]]:
    import torch

    _disable_unused_torchaudio()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    output = {}
    with torch.inference_mode():
        for ordinal, image in frames:
            inputs = _to_device(
                processor(images=image, text=prompt, return_tensors="pt"), device
            )
            predictions = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                predictions,
                input_ids=inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            scores = result["scores"].detach().cpu().numpy().astype(np.float64)
            boxes = result["boxes"].detach().cpu().numpy().astype(np.float64)
            labels = [str(value) for value in result["text_labels"]]
            order = np.argsort(-scores, kind="stable")[:candidate_limit]
            clipped = []
            for index in order:
                box = boxes[index].copy()
                box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(image.width))
                box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(image.height))
                if box[2] - box[0] >= 2.0 and box[3] - box[1] >= 2.0:
                    clipped.append(
                        {
                            "score": float(scores[index]),
                            "box_xyxy": box.tolist(),
                            "text_label": labels[index],
                        }
                    )
            output[ordinal] = {"detections": clipped}
    model.to("cpu")
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _sam_masks(
    *,
    frames: list[tuple[int, Image.Image]],
    detections: dict[int, dict[str, object]],
    point_prompts: dict[int, list[dict[str, object]]],
    model_root: Path,
    device: str,
) -> dict[int, list[dict[str, object]]]:
    import torch

    _disable_unused_torchaudio()
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = Sam2Model.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    output = {}
    with torch.inference_mode():
        for ordinal, image in frames:
            values = detections[ordinal]["detections"]
            prompts = point_prompts[ordinal]
            if any(
                not isinstance(prompt.get("detection_index"), int)
                or not 0 <= int(prompt["detection_index"]) < len(values)
                for prompt in prompts
            ):
                raise ValueError("target-point prompt references an invalid detection")
            usable = [index for index, prompt in enumerate(prompts) if prompt["passes_gate"]]
            if not usable:
                output[ordinal] = []
                continue
            points = [[[prompts[index]["point_xy"]] for index in usable]]
            labels = [[[1] for _ in usable]]
            inputs = processor(
                images=image,
                input_points=points,
                input_labels=labels,
                return_tensors="pt",
            )
            original_sizes = inputs["original_sizes"].clone()
            predictions = model(**_to_device(inputs, device), multimask_output=True)
            masks = processor.post_process_masks(
                predictions.pred_masks.detach().cpu(), original_sizes
            )[0]
            scores = predictions.iou_scores.detach().cpu()[0]
            candidates = []
            if masks.ndim != 4 or masks.shape[0] != len(usable) or masks.shape[1] != 3:
                raise RuntimeError("SAM2 did not return three multimask candidates per prompt")
            for object_index, prompt_index in enumerate(usable):
                for mode in range(3):
                    candidates.append(
                        {
                            "mask": masks[object_index, mode].numpy().astype(bool),
                            "segmentation_iou_score": float(scores[object_index, mode].item()),
                            "detection_index": int(
                                prompts[prompt_index]["detection_index"]
                            ),
                            "prompt_index": prompt_index,
                            "multimask_index": mode,
                        }
                    )
            output[ordinal] = candidates
    model.to("cpu")
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _propagate_sam2_video_interval(
    *,
    model,
    processor,
    frames: list[np.ndarray],
    seed_index: int,
    seed_mask: np.ndarray,
    reverse: bool,
    device: str,
) -> dict[int, np.ndarray]:
    """Propagate one independently seeded object through a short frame interval."""

    import torch

    if (
        not frames
        or seed_index not in range(len(frames))
        or np.asarray(seed_mask).shape != frames[0].shape[:2]
        or any(frame.shape != frames[0].shape for frame in frames)
    ):
        raise ValueError("SAM2 video interval inputs are inconsistent")
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
    )
    processor.add_inputs_to_inference_session(
        session,
        frame_idx=seed_index,
        obj_ids=1,
        input_masks=np.asarray(seed_mask, dtype=bool)[None],
    )
    propagated = {}
    with torch.inference_mode():
        for prediction in model.propagate_in_video_iterator(
            session,
            start_frame_idx=seed_index,
            max_frame_num_to_track=len(frames) - 1,
            reverse=reverse,
        ):
            frame_index = int(prediction.frame_idx)
            if prediction.object_ids != [1]:
                raise RuntimeError("SAM2 video propagation changed the table object ID")
            masks = processor.post_process_masks(
                [prediction.pred_masks.detach().cpu()],
                [(frames[0].shape[0], frames[0].shape[1])],
            )[0]
            if masks.shape != (1, 1, *frames[0].shape[:2]):
                raise RuntimeError("SAM2 video returned an unexpected mask shape")
            propagated[frame_index] = masks[0, 0].numpy().astype(bool)
    expected = set(range(len(frames)))
    if set(propagated) != expected:
        raise RuntimeError("SAM2 video propagation did not cover the complete interval")
    del session
    return propagated


def _generate_dense_video_masks(
    *,
    input_root: Path,
    records: list[dict[str, object]],
    sparse_masks_by_view: dict[str, dict[int, np.ndarray]],
    model_root: Path,
    output_root: Path,
    device: str,
    minimum_bidirectional_iou: float,
    minimum_area_fraction: float,
    maximum_area_fraction: float,
) -> dict[str, object]:
    """Create dense masks only where two endpoint-seeded SAM2 tracks agree."""

    import torch

    _disable_unused_torchaudio()
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    processor = Sam2VideoProcessor.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model = Sam2VideoModel.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    view_results = {}
    try:
        for view_name in POSE_VIEW_NAMES:
            sparse_masks = sparse_masks_by_view[view_name]
            keyframes = sorted(sparse_masks)
            dense_records: dict[int, dict[str, object]] = {}
            for ordinal in keyframes:
                path = (
                    output_root
                    / "dense_masks"
                    / view_name
                    / f"frame-{ordinal:06d}.png"
                )
                mask = sparse_masks[ordinal]
                _write_png(path, mask.astype(np.uint8) * 255)
                dense_records[ordinal] = {
                    "ordinal": ordinal,
                    "source": "selected_sparse_keyframe",
                    "path": str(path.relative_to(output_root)),
                    "sha256": sha256_file(path),
                    "area_fraction": float(np.count_nonzero(mask) / mask.size),
                    "interval_keyframe_ordinals": [ordinal, ordinal],
                    "fusion": None,
                }

            view_frames = [
                _image_and_depth(input_root, record, view_name)[0]
                for record in records
            ]
            interval_results = []
            for left, right in zip(keyframes, keyframes[1:]):
                interval_frames = view_frames[left : right + 1]
                forward = _propagate_sam2_video_interval(
                    model=model,
                    processor=processor,
                    frames=interval_frames,
                    seed_index=0,
                    seed_mask=sparse_masks[left],
                    reverse=False,
                    device=device,
                )
                backward = _propagate_sam2_video_interval(
                    model=model,
                    processor=processor,
                    frames=interval_frames,
                    seed_index=len(interval_frames) - 1,
                    seed_mask=sparse_masks[right],
                    reverse=True,
                    device=device,
                )
                accepted = []
                rejected = []
                frame_audit = []
                for local_index in range(1, len(interval_frames) - 1):
                    ordinal = left + local_index
                    fused, metrics = fuse_bidirectional_masks(
                        forward[local_index],
                        backward[local_index],
                        minimum_iou=minimum_bidirectional_iou,
                        minimum_area_fraction=minimum_area_fraction,
                        maximum_area_fraction=maximum_area_fraction,
                    )
                    audit = {
                        "ordinal": ordinal,
                        "fusion": metrics.to_json(),
                    }
                    if fused is None:
                        rejected.append(ordinal)
                    else:
                        path = (
                            output_root
                            / "dense_masks"
                            / view_name
                            / f"frame-{ordinal:06d}.png"
                        )
                        _write_png(path, fused.astype(np.uint8) * 255)
                        dense_records[ordinal] = {
                            "ordinal": ordinal,
                            "source": "bidirectional_sam2_video_intersection",
                            "path": str(path.relative_to(output_root)),
                            "sha256": sha256_file(path),
                            "area_fraction": metrics.fused_area_fraction,
                            "interval_keyframe_ordinals": [left, right],
                            "fusion": metrics.to_json(),
                        }
                        accepted.append(ordinal)
                    frame_audit.append(audit)
                interval_results.append(
                    {
                        "left_keyframe_ordinal": left,
                        "right_keyframe_ordinal": right,
                        "intermediate_frame_count": max(0, right - left - 1),
                        "accepted_ordinals": accepted,
                        "rejected_ordinals": rejected,
                        "frames": frame_audit,
                    }
                )
            ordered_records = [dense_records[index] for index in sorted(dense_records)]
            view_results[view_name] = {
                "keyframe_ordinals": keyframes,
                "accepted_ordinals": sorted(dense_records),
                "accepted_frame_count": len(dense_records),
                "intervals": interval_results,
                "frames": ordered_records,
            }
            del view_frames
    finally:
        model.to("cpu")
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "method": "per_interval_forward_backward_sam2_video_mask_intersection",
        "minimum_bidirectional_iou": minimum_bidirectional_iou,
        "minimum_area_fraction": minimum_area_fraction,
        "maximum_area_fraction": maximum_area_fraction,
        "views": view_results,
    }


def _overlay(
    rgb: np.ndarray,
    masks: list[np.ndarray],
    selected_mask: np.ndarray | None,
) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for mask in masks:
        color = np.array((30, 140, 230), dtype=np.float32)
        pixels = image[mask].astype(np.float32)
        image[mask] = np.clip(0.8 * pixels + 0.2 * color, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, tuple(int(x) for x in color), 2)
    if selected_mask is not None:
        color = np.array((60, 220, 60), dtype=np.float32)
        pixels = image[selected_mask].astype(np.float32)
        image[selected_mask] = np.clip(
            0.65 * pixels + 0.35 * color, 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, contours, -1, tuple(int(x) for x in color), 2)
    return image


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    pose = config.object_pose_runtime
    input_root = args.input_dir.expanduser().resolve()
    runtime_root = args.runtime_root.expanduser().resolve()
    input_manifest_path = input_root / "manifest.json"
    runtime_manifest_path = runtime_root / "runtime-manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported prepared RGB-D input schema")
    if runtime_manifest.get("schema_version") != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported object-pose runtime manifest schema")
    if runtime_manifest.get("config_sha256") != config.digest:
        raise ValueError("object-pose runtime was prepared from a different pipeline config")
    if input_manifest.get("source_revision") != config.source.revision:
        raise ValueError("prepared RGB-D input uses a different source revision")
    if input_manifest.get("head_stereo_calibration_sha256") != config.raw_source.head_stereo_calibration_sha256:
        raise ValueError("prepared RGB-D input uses a different stereo calibration")
    source_frames = tuple(int(value) for value in input_manifest["source_frame_indices"])
    ordinals = registration_ordinals(
        source_frames, pose.registration_interval_source_frames
    )
    records = input_manifest.get("frames")
    if not isinstance(records, list) or len(records) != len(source_frames):
        raise ValueError("prepared RGB-D frame records differ from source frame indices")
    pose_views = input_manifest.get("pose_views")
    if (
        not isinstance(pose_views, dict)
        or len(pose_views) != len(POSE_VIEW_NAMES)
        or set(pose_views) != set(POSE_VIEW_NAMES)
    ):
        raise ValueError("prepared input does not contain the ordered three-view contract")
    intrinsics = {
        view_name: np.asarray(
            pose_views[view_name]["intrinsic_matrix_px"], dtype=np.float64
        ).reshape(3, 3)
        for view_name in POSE_VIEW_NAMES
    }
    robot_urdf = (
        args.robofinals_root.expanduser().resolve()
        / pose.robot_visual_urdf_relative_path
    )
    robot_renderers = {
        view_name: RobotSilhouetteRenderer(
            robot_urdf,
            expected_sha256=pose.robot_visual_urdf_sha256,
            dilation_px=(
                pose.robot_silhouette_dilation_px
                if view_name == PRIMARY_POSE_VIEW
                else pose.auxiliary_robot_silhouette_dilation_px
            ),
        )
        for view_name in POSE_VIEW_NAMES
    }

    output = args.output_dir.expanduser().resolve()
    output_manifest_path = output / "manifest.json"
    identity = {
        "config_sha256": config.digest,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
    }
    if output.exists():
        if not args.resume or not output_manifest_path.is_file():
            raise FileExistsError(f"output already exists: {output}")
        previous = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing masks were generated from a different contract")
        print(json.dumps({"output": str(output), "resumed": True, "gate": previous["gate"]}, sort_keys=True))
        if not previous["gate"]["pass"]:
            raise SystemExit(2)
        return

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            f"install the audited transformers=={pose.transformers_version} runtime first"
        ) from exc
    if transformers.__version__ != pose.transformers_version:
        raise ValueError(
            f"transformers {transformers.__version__} differs from pinned {pose.transformers_version}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        frame_images = []
        token_to_location = {}
        arrays = {}
        depths = {}
        robot_masks = {}
        robot_metrics = {}
        token = 0
        for ordinal in ordinals:
            record = records[ordinal]
            views = record.get("views")
            if not isinstance(views, dict):
                raise ValueError(f"prepared frame {ordinal} lacks views")
            for view_name in POSE_VIEW_NAMES:
                rgb, depth = _image_and_depth(input_root, record, view_name)
                location = (ordinal, view_name)
                arrays[location] = rgb
                depths[location] = depth
                view = views[view_name]
                root_from_camera = np.asarray(
                    view["robot_root_from_rectified_opencv"], dtype=np.float64
                ).reshape(4, 4)
                robot_mask, silhouette_metrics = robot_renderers[view_name].render(
                    robot_q_current=np.asarray(record["robot_q_current"], dtype=np.float64),
                    hand_state=np.asarray(record["hand_state"], dtype=np.float64),
                    root_from_camera=root_from_camera,
                    intrinsic_matrix=intrinsics[view_name],
                    width=640,
                    height=480,
                )
                if not robot_silhouette_coverage_is_plausible(
                    silhouette_metrics.mask_fraction
                ):
                    raise ValueError(
                        f"robot silhouette coverage is implausible at ordinal {ordinal} "
                        f"view {view_name}: {silhouette_metrics.mask_fraction:.6f}"
                    )
                robot_masks[location] = robot_mask
                robot_metrics[location] = silhouette_metrics
                token_to_location[token] = location
                frame_images.append((token, Image.fromarray(rgb)))
                token += 1
        detections = _detector_results(
            frames=frame_images,
            model_root=runtime_root / "hf" / "grounding-dino-base",
            prompt=pose.text_prompt,
            device=args.device,
            box_threshold=pose.detector_box_threshold,
            text_threshold=pose.detector_text_threshold,
            candidate_limit=pose.max_detection_candidates,
        )
        point_prompts = {}
        for frame_token, (ordinal, view_name) in token_to_location.items():
            record = records[ordinal]
            root_from_camera = np.asarray(
                record["views"][view_name]["robot_root_from_rectified_opencv"],
                dtype=np.float64,
            ).reshape(4, 4)
            eef_poses_root = np.asarray(
                record["eef_current_root_from_fk"], dtype=np.float64
            ).reshape(2, 4, 4)
            point_prompts[frame_token] = []
            prompt_weights = tuple(
                dict.fromkeys((pose.table_prompt_principal_point_weight, 1.0))
            )
            for detection_index, detection in enumerate(
                detections[frame_token]["detections"]
            ):
                for principal_point_weight in prompt_weights:
                    prompt = target_point_prompt(
                        rgb=arrays[(ordinal, view_name)],
                        robot_silhouette=robot_masks[(ordinal, view_name)],
                        eef_poses_root=eef_poses_root,
                        root_from_camera=root_from_camera,
                        intrinsic_matrix=intrinsics[view_name],
                        detector_box_xyxy=detection["box_xyxy"],
                        minimum_value=pose.table_mask_minimum_value,
                        maximum_point_distance_px=pose.table_prompt_maximum_distance_px,
                        principal_point_weight=principal_point_weight,
                    ).to_json()
                    prompt.update(
                        {
                            "detection_index": detection_index,
                            "principal_point_weight": principal_point_weight,
                        }
                    )
                    point_prompts[frame_token].append(prompt)
        masks_by_token = _sam_masks(
            frames=frame_images,
            detections=detections,
            point_prompts=point_prompts,
            model_root=runtime_root / "hf" / "sam2.1-hiera-large",
            device=args.device,
        )
        frame_results = []
        sequence_masks = {view_name: [] for view_name in POSE_VIEW_NAMES}
        sequence_metrics = {view_name: [] for view_name in POSE_VIEW_NAMES}
        location_to_token = {location: value for value, location in token_to_location.items()}
        for ordinal in ordinals:
            view_results = {}
            for view_name in POSE_VIEW_NAMES:
                location = (ordinal, view_name)
                frame_token = location_to_token[location]
                sam_candidates = masks_by_token[frame_token]
                raw_masks = [value["mask"] for value in sam_candidates]
                mask_scores = [
                    float(value["segmentation_iou_score"]) for value in sam_candidates
                ]
                masks = []
                refinement_metrics = []
                reachability_metrics = []
                record = records[ordinal]
                root_from_camera = np.asarray(
                    record["views"][view_name]["robot_root_from_rectified_opencv"],
                    dtype=np.float64,
                ).reshape(4, 4)
                eef_positions_root = np.asarray(
                    record["eef_current_root_from_fk"], dtype=np.float64
                ).reshape(2, 4, 4)[:, :3, 3]
                for raw_mask in raw_masks:
                    refined, metrics = refine_table_mask(
                        rgb=arrays[location],
                        candidate_mask=raw_mask,
                        robot_silhouette=robot_masks[location],
                        minimum_value=pose.table_mask_minimum_value,
                        minimum_component_area_px=pose.table_mask_minimum_component_area_px,
                    )
                    reachable_metrics = None
                    if view_name == PRIMARY_POSE_VIEW:
                        refined, reachable_metrics = filter_reachable_table_components(
                            candidate_mask=refined,
                            depth_m=depths[location],
                            intrinsic_matrix=intrinsics[view_name],
                            root_from_camera=root_from_camera,
                            eef_positions_root=eef_positions_root,
                            maximum_median_eef_distance_m=(
                                pose.primary_mask_maximum_median_eef_distance_m
                            ),
                            minimum_component_valid_depth_fraction=(
                                pose.primary_mask_minimum_component_valid_depth_fraction
                            ),
                        )
                        metrics = replace(
                            metrics,
                            retained_components=(
                                reachable_metrics.retained_component_count
                            ),
                            output_pixels=reachable_metrics.output_pixels,
                        )
                    masks.append(refined)
                    refinement_metrics.append(metrics)
                    reachability_metrics.append(reachable_metrics)
                detector_values = detections[frame_token]["detections"]
                values = [
                    {
                        **detector_values[int(candidate["detection_index"])],
                        "detection_index": int(candidate["detection_index"]),
                        "prompt_index": int(candidate["prompt_index"]),
                        "sam_multimask_index": int(candidate["multimask_index"]),
                        "target_point_prompt": point_prompts[frame_token][
                            int(candidate["prompt_index"])
                        ],
                    }
                    for candidate in sam_candidates
                ]
                metrics, selected = evaluate_mask_candidates(
                    rgb=arrays[location],
                    depth_m=depths[location],
                    masks=masks,
                    detector_scores=[value["score"] for value in values],
                    segmentation_iou_scores=mask_scores,
                    detector_boxes_xyxy=[value["box_xyxy"] for value in values],
                    refinement_metrics=refinement_metrics,
                    minimum_segmentation_iou=pose.segmentation_iou_threshold,
                    minimum_mask_area_fraction=pose.min_mask_area_fraction,
                    maximum_mask_area_fraction=pose.max_mask_area_fraction,
                    minimum_valid_depth_fraction=(
                        pose.min_valid_depth_fraction_in_mask
                        if view_name == PRIMARY_POSE_VIEW
                        else 0.0
                    ),
                    previous_mask=None,
                )
                sequence_masks[view_name].append(tuple(masks))
                sequence_metrics[view_name].append(metrics)
                selected_mask = None if selected is None else masks[selected]
                selected_components = () if selected is None else (selected,)
                selected_operation = (
                    None if selected is None else "ranked_single_target_point_candidate"
                )
                raw_mask_paths = []
                raw_mask_sha256s = []
                mask_paths = []
                mask_sha256s = []
                for index, (raw_mask, mask) in enumerate(
                    zip(raw_masks, masks, strict=True)
                ):
                    raw_path = (
                        temporary
                        / "raw_masks"
                        / view_name
                        / f"frame-{ordinal:06d}-candidate-{index:02d}.png"
                    )
                    _write_png(raw_path, raw_mask.astype(np.uint8) * 255)
                    raw_mask_paths.append(str(raw_path.relative_to(temporary)))
                    raw_mask_sha256s.append(sha256_file(raw_path))
                    path = (
                        temporary
                        / "masks"
                        / view_name
                        / f"frame-{ordinal:06d}-candidate-{index:02d}.png"
                    )
                    _write_png(path, mask.astype(np.uint8) * 255)
                    mask_paths.append(str(path.relative_to(temporary)))
                    mask_sha256s.append(sha256_file(path))
                review = temporary / "review" / view_name / f"frame-{ordinal:06d}.png"
                review_image = _overlay(arrays[location], masks, selected_mask)
                robot_contours, _ = cv2.findContours(
                    robot_masks[location].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(review_image, robot_contours, -1, (220, 40, 220), 2)
                _write_png(review, review_image)
                robot_mask_path = (
                    temporary / "robot_masks" / view_name / f"frame-{ordinal:06d}.png"
                )
                _write_png(robot_mask_path, robot_masks[location].astype(np.uint8) * 255)
                view_results[view_name] = {
                    "target_point_prompts": point_prompts[frame_token],
                    "detections": values,
                    "raw_candidate_masks": raw_mask_paths,
                    "raw_candidate_mask_sha256s": raw_mask_sha256s,
                    "candidate_masks": mask_paths,
                    "candidate_mask_sha256s": mask_sha256s,
                    "candidates": [value.to_json() for value in metrics],
                    "refinement": [value.to_json() for value in refinement_metrics],
                    "reachability_refinement": [
                        None if value is None else value.to_json()
                        for value in reachability_metrics
                    ],
                    "robot_silhouette": {
                        **robot_metrics[location].to_json(),
                        "path": str(robot_mask_path.relative_to(temporary)),
                        "sha256": sha256_file(robot_mask_path),
                    },
                    "selected_candidate_index": selected,
                    "selected_candidate_indices": list(selected_components),
                    "selected_mask_operation": selected_operation,
                    "selected_mask": None,
                    "selected_mask_sha256": None,
                    "review": str(review.relative_to(temporary)),
                    "review_sha256": sha256_file(review),
                }
            frame_results.append(
                {
                    "ordinal": ordinal,
                    "source_frame_index": source_frames[ordinal],
                    "views": view_results,
                }
            )
        common_sequence_parameters = {
            "detector_weight": pose.mask_sequence_detector_weight,
            "segmentation_weight": pose.mask_sequence_segmentation_weight,
            "retention_weight": pose.mask_sequence_retention_weight,
            "transition_iou_weight": pose.mask_sequence_transition_iou_weight,
        }
        primary_sequence_parameters = MaskSequenceParameters(
            **common_sequence_parameters,
            area_weight=pose.mask_sequence_primary_area_weight,
            fragmentation_weight=pose.mask_sequence_primary_fragmentation_weight,
            transition_area_weight=pose.mask_sequence_primary_transition_area_weight,
            fragmentation_mode="logarithmic",
        )
        auxiliary_sequence_parameters = MaskSequenceParameters(
            **common_sequence_parameters,
            area_weight=pose.mask_sequence_auxiliary_area_weight,
            fragmentation_weight=pose.mask_sequence_auxiliary_fragmentation_weight,
            transition_area_weight=pose.mask_sequence_auxiliary_transition_area_weight,
            fragmentation_mode="linear",
        )
        sequence_parameters_by_view = {
            view_name: (
                primary_sequence_parameters
                if view_name == PRIMARY_POSE_VIEW
                else auxiliary_sequence_parameters
            )
            for view_name in POSE_VIEW_NAMES
        }
        sequence_selection = {
            view_name: select_mask_candidate_sequence(
                masks_by_frame=sequence_masks[view_name],
                metrics_by_frame=sequence_metrics[view_name],
                parameters=sequence_parameters_by_view[view_name],
            )
            for view_name in POSE_VIEW_NAMES
        }
        selected_counts = {view_name: 0 for view_name in POSE_VIEW_NAMES}
        sparse_masks_by_view = {view_name: {} for view_name in POSE_VIEW_NAMES}
        for frame_position, ordinal in enumerate(ordinals):
            for view_name in POSE_VIEW_NAMES:
                location = (ordinal, view_name)
                selected = sequence_selection[view_name][frame_position]
                view_result = frame_results[frame_position]["views"][view_name]
                selected_mask = (
                    None
                    if selected is None
                    else sequence_masks[view_name][frame_position][selected]
                )
                selected_path = None
                selected_sha256 = None
                if selected_mask is not None:
                    selected_output = (
                        temporary
                        / "selected_masks"
                        / view_name
                        / f"frame-{ordinal:06d}.png"
                    )
                    _write_png(selected_output, selected_mask.astype(np.uint8) * 255)
                    selected_path = str(selected_output.relative_to(temporary))
                    selected_sha256 = sha256_file(selected_output)
                    selected_counts[view_name] += 1
                    sparse_masks_by_view[view_name][ordinal] = selected_mask
                review = temporary / "review" / view_name / f"frame-{ordinal:06d}.png"
                review_image = _overlay(
                    arrays[location], sequence_masks[view_name][frame_position], selected_mask
                )
                robot_contours, _ = cv2.findContours(
                    robot_masks[location].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(review_image, robot_contours, -1, (220, 40, 220), 2)
                _write_png(review, review_image)
                view_result.update(
                    {
                        "selected_candidate_index": selected,
                        "selected_candidate_indices": [] if selected is None else [selected],
                        "selected_mask_operation": (
                            None
                            if selected is None
                            else "global_viterbi_single_target_point_candidate"
                        ),
                        "selected_mask": selected_path,
                        "selected_mask_sha256": selected_sha256,
                        "review_sha256": sha256_file(review),
                    }
                )
        dense_tracking_masks = _generate_dense_video_masks(
            input_root=input_root,
            records=records,
            sparse_masks_by_view=sparse_masks_by_view,
            model_root=runtime_root / "hf" / "sam2.1-hiera-large",
            output_root=temporary,
            device=args.device,
            minimum_bidirectional_iou=pose.dense_mask_min_bidirectional_iou,
            minimum_area_fraction=pose.dense_mask_min_area_fraction,
            maximum_area_fraction=pose.dense_mask_max_area_fraction,
        )
        minimum_count = math.ceil(len(ordinals) * pose.minimum_registration_mask_coverage)
        primary_ordinals, wrist_bridge_ordinals, eligible_ordinals = (
            _mask_observation_ordinals(frame_results)
        )
        first_selected = bool(primary_ordinals and primary_ordinals[0] == 0)
        last_selected_ordinal = eligible_ordinals[-1] if eligible_ordinals else None
        primary_terminal_gap = _terminal_source_gap(source_frames, primary_ordinals)
        terminal_gap = _terminal_source_gap(source_frames, eligible_ordinals)
        terminal_anchor_pass = (
            primary_terminal_gap is not None
            and primary_terminal_gap
            <= pose.maximum_terminal_tracking_gap_source_frames
        )
        gate = {
            "registration_frames": len(ordinals),
            "selected_masks_by_view": selected_counts,
            "primary_selected_frames": len(primary_ordinals),
            "bimanual_wrist_bridge_frames": len(wrist_bridge_ordinals),
            "tracking_eligible_frames": len(eligible_ordinals),
            "minimum_tracking_eligible_frames": minimum_count,
            "primary_selected_ordinals": primary_ordinals,
            "bimanual_wrist_bridge_ordinals": wrist_bridge_ordinals,
            "tracking_eligible_ordinals": eligible_ordinals,
            "first_selected": first_selected,
            "last_selected_ordinal": last_selected_ordinal,
            "last_primary_selected_ordinal": (
                primary_ordinals[-1] if primary_ordinals else None
            ),
            "terminal_tracking_gap_source_frames": terminal_gap,
            "primary_terminal_gap_source_frames": primary_terminal_gap,
            "maximum_terminal_tracking_gap_source_frames": (
                pose.maximum_terminal_tracking_gap_source_frames
            ),
            "terminal_anchor_pass": terminal_anchor_pass,
            "pass": (
                len(eligible_ordinals) >= minimum_count
                and first_selected
                and terminal_anchor_pass
            ),
        }
        rejection_reasons = _mask_rejection_reasons(gate)
        if gate["pass"] != (not rejection_reasons):
            raise RuntimeError("mask gate and rejection reasons disagree")
        manifest = {
            "schema_version": SEGMENTATION_SCHEMA_VERSION,
            **identity,
            "episode_index": input_manifest["episode_index"],
            "source_revision": config.source.revision,
            "accepted": gate["pass"],
            "rejection_reasons": rejection_reasons,
            "method": pose.method,
            "text_prompt": pose.text_prompt,
            "detector": {
                "repo": pose.detector_repo,
                "revision": pose.detector_revision,
                "box_threshold": pose.detector_box_threshold,
                "text_threshold": pose.detector_text_threshold,
            },
            "segmentation": {
                "repo": pose.segmentation_repo,
                "revision": pose.segmentation_revision,
                "iou_threshold": pose.segmentation_iou_threshold,
                "video_propagation": {
                    "method": dense_tracking_masks["method"],
                    "minimum_bidirectional_iou": (
                        pose.dense_mask_min_bidirectional_iou
                    ),
                    "minimum_area_fraction": pose.dense_mask_min_area_fraction,
                    "maximum_area_fraction": pose.dense_mask_max_area_fraction,
                },
            },
            "robot_silhouette": {
                "urdf": str(robot_urdf),
                "urdf_sha256": pose.robot_visual_urdf_sha256,
                "dilation_px_by_view": {
                    view_name: (
                        pose.robot_silhouette_dilation_px
                        if view_name == PRIMARY_POSE_VIEW
                        else pose.auxiliary_robot_silhouette_dilation_px
                    )
                    for view_name in POSE_VIEW_NAMES
                },
            },
            "table_mask_refinement": {
                "minimum_value": pose.table_mask_minimum_value,
                "minimum_component_area_px": pose.table_mask_minimum_component_area_px,
                "primary_view_reachable_components": {
                    "view": PRIMARY_POSE_VIEW,
                    "maximum_median_eef_distance_m": (
                        pose.primary_mask_maximum_median_eef_distance_m
                    ),
                    "minimum_component_valid_depth_fraction": (
                        pose.primary_mask_minimum_component_valid_depth_fraction
                    ),
                    "geometry_source": (
                        "head_stereo_depth_plus_calibrated_camera_plus_robot_q_current_fk"
                    ),
                    "uses_policy_input": False,
                    "offline_annotation_only": True,
                },
            },
            "target_point_prompt": {
                "source": "robot_q_current_fk_projected_eef_midpoint",
                "maximum_point_distance_px": pose.table_prompt_maximum_distance_px,
                "principal_point_weight": pose.table_prompt_principal_point_weight,
                "uses_policy_input": False,
                "offline_annotation_only": True,
            },
            "candidate_sequence_selection": {
                "method": "global_viterbi_view_role_single_candidate_no_mask_union",
                "primary_view": PRIMARY_POSE_VIEW,
                "profile_by_view": {
                    view_name: (
                        "complete_assembled_object"
                        if view_name == PRIMARY_POSE_VIEW
                        else "local_contact_support"
                    )
                    for view_name in POSE_VIEW_NAMES
                },
                "parameters_by_view": {
                    view_name: parameters.__dict__
                    for view_name, parameters in sequence_parameters_by_view.items()
                },
            },
            "transformers_version": transformers.__version__,
            "device": args.device,
            "registration_ordinals": list(ordinals),
            "dense_tracking_masks": dense_tracking_masks,
            "gate": gate,
            "frames": frame_results,
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "gate": gate}, sort_keys=True))
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
