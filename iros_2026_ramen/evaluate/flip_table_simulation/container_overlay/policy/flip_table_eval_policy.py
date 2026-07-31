"""Real-compatible policies and diagnostics for flip-table simulation evaluation.

The organizer image does not provide adapters for the Team RAMEN ACT and GR00T
checkpoints. This module adds those adapters together with narrowly scoped replay
and motion probes. Learned policies receive only observations available on the
real G1; simulator-only state is used for success accounting and traces.
"""

from __future__ import annotations

import io
import json
import math
import os
import queue
import socket
import struct
import sys
import threading
import time
import traceback
import types
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import mediapy as media
import numpy as np
import torch

from policy.base import BasePolicy

try:
    from policy.team_ramen_groot.dex1_hand_synergy import dex1_to_hand
    from policy.team_ramen_groot.n17_contract import (
        validate_finalized_furniture_checkpoint,
    )
    from policy.team_ramen_groot.temporal_ensemble import (
        PhysicalTargetTemporalEnsembler,
        logical_chunk_to_physical_targets,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.subtask_policy_training.gr00t.dex1_hand_synergy import dex1_to_hand
    from model.subtask_policy_training.gr00t.n17_contract import (
        validate_finalized_furniture_checkpoint,
    )
    from model.subtask_policy_training.gr00t.temporal_ensemble import (
        PhysicalTargetTemporalEnsembler,
        logical_chunk_to_physical_targets,
    )

try:
    from .cv_rule_based import (
        GRASP_RETRY_OFFSETS_TOOL_M,
        GeometricFlipPlanner,
        CameraCalibration,
        Phase,
        TableLegDetector,
        TabletopPoseEstimator,
        WristShaftDetector,
        WristTabletopEdgeDetector,
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
except ImportError:
    # The repository's unit tests load this file directly rather than as the
    # ``policy`` package used by RoboFinals.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cv_rule_based import (
        GRASP_RETRY_OFFSETS_TOOL_M,
        GeometricFlipPlanner,
        CameraCalibration,
        Phase,
        TableLegDetector,
        TabletopPoseEstimator,
        WristShaftDetector,
        WristTabletopEdgeDetector,
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


CAMERA_SAVE_ROLES = {
    "first_person_camera": "head_left",
    "head_right_camera": "head_right",
    "left_hand_camera": "left_wrist",
    "right_hand_camera": "right_wrist",
    # Older RoboFinals IKEA images exposed the right wrist camera as
    # ``hand_camera``.  Keep it as a compatibility alias, but prefer the V1 name.
    "hand_camera": "right_wrist",
    "global_camera": "global",
}

_G1_UPPER_BODY_JOINT_INDICES = (
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
)
_G1_ARM_JOINT_INDICES = _G1_UPPER_BODY_JOINT_INDICES[3:]
_G1_LEFT_DEX1_JOINT_INDICES = (29, 30)
_G1_RIGHT_DEX1_JOINT_INDICES = (31, 32)
_DEX1_OPEN_POS = 0.0245
_DEX1_CLOSE_POS = -0.02


# The simulator renders an ideal, centered pinhole image.  The recorded policy
# streams are the unrectified RGB streams from the head stereo camera and the
# two D405 color cameras.  This table pins the image formation model used by
# the dataset contract.  It is deliberately a rendering adapter, not a policy
# feature: no intrinsic, distortion, or episode-randomization value is exposed
# to a learned policy.
_RECORDED_CAMERA_GEOMETRY = {
    "first_person_camera": {
        "ideal_focal_px": 337.0724566415881,
        "intrinsic": (
            337.5311318539417,
            336.61378142923456,
            316.5285046932812,
            232.50620475777816,
        ),
        "distortion_model": "opencv_brown_conrady",
        "distortion": (
            0.06635329597971165,
            -0.07841619072258442,
            -0.0032837567734969727,
            -0.0010816865229956933,
            0.021030073866954904,
        ),
    },
    "head_right_camera": {
        # Raw right-eye calibration from the same pinned head-stereo YAML as
        # first_person_camera.  This remains diagnostic/operator imagery, not
        # a learned-policy feature, but must not silently inherit left-eye
        # intrinsics while stereo and camera calibration are being audited.
        "ideal_focal_px": 335.88671031702785,
        "intrinsic": (
            336.30012498108425,
            335.47329565297144,
            321.60051380995424,
            231.69425545320323,
        ),
        "distortion_model": "opencv_brown_conrady",
        "distortion": (
            0.06366431884731834,
            -0.08229830690155956,
            -0.0031845859537499963,
            0.0017675102141209843,
            0.027381390668112876,
        ),
    },
    "left_hand_camera": {
        "ideal_focal_px": 435.0029373168945,
        "intrinsic": (
            435.36712646484375,
            434.6387481689453,
            317.3426818847656,
            244.7300567626953,
        ),
        "distortion_model": "realsense_inverse_brown_conrady",
        "distortion": (
            -0.05092783644795418,
            0.059635864570736885,
            0.0010625082795741037,
            0.0011093204957433045,
            -0.020096530206501484,
        ),
    },
    "right_hand_camera": {
        "ideal_focal_px": 435.0029373168945,
        "intrinsic": (
            435.36712646484375,
            434.6387481689453,
            317.3426818847656,
            244.7300567626953,
        ),
        "distortion_model": "realsense_inverse_brown_conrady",
        "distortion": (
            -0.05092783644795418,
            0.059635864570736885,
            0.0010625082795741037,
            0.0011093204957433045,
            -0.020096530206501484,
        ),
    },
}
_RECORDED_CAMERA_TORCH_GRIDS: dict[tuple[str, str], torch.Tensor] = {}


def _recorded_camera_geometry_enabled() -> bool:
    """Return whether learned-policy images emulate the recorded raw RGB geometry."""

    return _env_bool("FLIP_TABLE_APPLY_RECORDED_CAMERA_GEOMETRY", True)


def _recorded_camera_remap(camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Map target raw pixels to source ideal-pinhole pixels for one RGB sensor."""

    spec = _RECORDED_CAMERA_GEOMETRY.get(camera_name)
    if spec is None:
        raise ValueError(f"camera {camera_name!r} has no recorded geometry contract")
    import cv2

    fx, fy, cx, cy = spec["intrinsic"]
    intrinsic = np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)
    focal = float(spec["ideal_focal_px"])
    ideal = np.asarray(((focal, 0.0, 320.0), (0.0, focal, 240.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    columns, rows = np.meshgrid(np.arange(640, dtype=np.float64), np.arange(480, dtype=np.float64))
    output_pixels = np.stack((columns.reshape(-1), rows.reshape(-1)), axis=1)
    coefficients = np.asarray(spec["distortion"], dtype=np.float64)
    if spec["distortion_model"] == "opencv_brown_conrady":
        ideal_pixels = cv2.undistortPoints(
            output_pixels.reshape(-1, 1, 2), intrinsic, coefficients, P=ideal
        ).reshape(-1, 2)
    elif spec["distortion_model"] == "realsense_inverse_brown_conrady":
        # librealsense defines this model in the distorted-to-undistorted
        # direction, unlike OpenCV Brown-Conrady.  Do not use cv2.undistortPoints
        # for D405 RGB frames.
        x = (output_pixels[:, 0] - cx) / fx
        y = (output_pixels[:, 1] - cy) / fy
        r2 = x * x + y * y
        radial = 1.0 + r2 * (
            coefficients[0] + r2 * (coefficients[1] + r2 * coefficients[4])
        )
        ideal_x = x * radial + 2.0 * coefficients[2] * x * y + coefficients[3] * (
            r2 + 2.0 * x * x
        )
        ideal_y = y * radial + 2.0 * coefficients[3] * x * y + coefficients[2] * (
            r2 + 2.0 * y * y
        )
        ideal_pixels = np.stack((ideal_x * focal + 320.0, ideal_y * focal + 240.0), axis=1)
    else:  # pragma: no cover - static contract validation above covers this.
        raise RuntimeError(f"unsupported recorded camera model: {spec['distortion_model']}")
    if not np.isfinite(ideal_pixels).all():
        raise RuntimeError(f"non-finite remap for {camera_name}")
    return (
        ideal_pixels[:, 0].reshape(480, 640).astype(np.float32),
        ideal_pixels[:, 1].reshape(480, 640).astype(np.float32),
    )


def _apply_recorded_camera_geometry_numpy(image: np.ndarray, camera_name: str) -> np.ndarray:
    """Return a 640x480 raw-camera RGB image from a simulator pinhole render."""

    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 [480,640,3] image, got {image.shape}/{image.dtype}")
    if not _recorded_camera_geometry_enabled():
        return image
    import cv2

    map_x, map_y = _recorded_camera_remap(camera_name)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _apply_recorded_camera_geometry_tensor(tensor: torch.Tensor, camera_name: str) -> torch.Tensor:
    """Apply the same raw-camera map on the policy device without a CPU round trip."""

    if tuple(tensor.shape) != (3, 480, 640):
        raise ValueError(f"expected CHW [3,480,640] image, got {tuple(tensor.shape)}")
    if not _recorded_camera_geometry_enabled():
        return tensor
    cache_key = (camera_name, str(tensor.device))
    grid = _RECORDED_CAMERA_TORCH_GRIDS.get(cache_key)
    if grid is None:
        map_x, map_y = _recorded_camera_remap(camera_name)
        grid_array = np.stack(
            (map_x / 639.0 * 2.0 - 1.0, map_y / 479.0 * 2.0 - 1.0), axis=-1
        )
        grid = torch.as_tensor(grid_array, dtype=torch.float32, device=tensor.device).unsqueeze(0)
        _RECORDED_CAMERA_TORCH_GRIDS[cache_key] = grid
    values = tensor.to(dtype=torch.float32).unsqueeze(0)
    return torch.nn.functional.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]


def _num_envs_from_obs(observation: dict[str, Any], fallback: int = 1) -> int:
    for group_name in ("policy", "embodiment_general_obs"):
        group = observation.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for value in group.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return max(1, int(value.shape[0]))
    return fallback


def _success_mask(extras: dict[str, Any], terminated: Any, num_envs: int) -> np.ndarray:
    """Return successful environments under the overlay's success-only termination contract."""

    value = extras.get("is_success") if isinstance(extras, dict) else None
    # AutoConditionSuccessTask defines no failure termination. The organizer
    # policy API does not currently expose ``is_success``, so its raw termination
    # signal is the authoritative fallback for this overlay.
    if value is None:
        value = terminated
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value, dtype=bool).reshape(-1)
    if mask.size == 1 and num_envs > 1:
        mask = np.repeat(mask, num_envs)
    if mask.size < num_envs:
        padded = np.zeros(num_envs, dtype=bool)
        padded[: mask.size] = mask
        return padded
    return mask[:num_envs]


def _terminated_any(terminated: Any) -> bool:
    if torch.is_tensor(terminated):
        return bool(torch.any(terminated).item())
    return bool(np.any(np.asarray(terminated, dtype=bool)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _camera_names(usr_args: dict[str, Any]) -> list[str]:
    value = os.environ.get("FLIP_TABLE_SAVE_CAMERA_NAMES", "")
    if value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(name) for name in usr_args.get("record_camera", [])]


def _camera_output_name(camera_name: str, key: str) -> str:
    if not _env_bool("FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES", True):
        return f"{key}.png"
    role = CAMERA_SAVE_ROLES.get(camera_name, camera_name)
    suffix = "_rgb" if key.endswith("_rgb") else ""
    return f"{role}{suffix}.png"


def _is_rgb_observation(value: Any) -> bool:
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) == 4:
        if shape[0] != 1 and not _env_bool("FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT", False):
            return False
        shape = shape[1:]
    if len(shape) != 3:
        return False
    return shape[0] in (3, 4) or shape[-1] in (3, 4)


def _resolve_camera_rgb_key(camera_obs: dict[str, Any], camera_name: str) -> str:
    preferred = (camera_name, f"{camera_name}_rgb")
    for key in preferred:
        if key in camera_obs and _is_rgb_observation(camera_obs[key]):
            return key
    matches = sorted(
        key
        for key, value in camera_obs.items()
        if key.startswith(camera_name) and key.endswith("_rgb") and _is_rgb_observation(value)
    )
    if len(matches) != 1:
        raise ValueError(
            f"camera {camera_name!r} must resolve to exactly one RGB observation; "
            f"matches={matches}, available={sorted(camera_obs)}"
        )
    return matches[0]


def _camera_image_uint8(value: Any, *, batch_index: int = 0) -> np.ndarray:
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1 and not _env_bool(
            "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT", False
        ):
            raise ValueError(
                "camera image batch size 1 is required unless "
                "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT=true"
            )
        if not 0 <= batch_index < tensor.shape[0]:
            raise ValueError(
                f"camera batch index {batch_index} is outside shape {tuple(tensor.shape)}"
            )
        tensor = tensor[batch_index]
    array = tensor.numpy()
    if array.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {array.shape}")
    channels_first = array.shape[0] in (3, 4)
    channels_last = array.shape[-1] in (3, 4)
    if channels_first == channels_last:
        raise ValueError(f"expected unambiguous RGB/RGBA image layout, got shape {array.shape}")
    if channels_first:
        array = np.moveaxis(array, 0, -1)
    if not np.isfinite(array).all():
        raise ValueError("camera image contains NaN or Inf")
    array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _maybe_save_camera_frames(policy: BasePolicy, observation: dict[str, Any], usr_args: dict[str, Any], step: int) -> None:
    if not _env_bool("FLIP_TABLE_SAVE_CAMERA_FRAMES", False):
        return
    raw_indices = os.environ.get("FLIP_TABLE_CAMERA_FRAME_INDICES", "").strip()
    if raw_indices:
        try:
            frame_indices = tuple(sorted({int(value.strip()) for value in raw_indices.split(",")}))
        except ValueError as exc:
            raise ValueError("FLIP_TABLE_CAMERA_FRAME_INDICES must be comma-separated integers") from exc
        if not frame_indices or frame_indices[0] < 0:
            raise ValueError("FLIP_TABLE_CAMERA_FRAME_INDICES must contain non-negative indices")
    else:
        frame_indices = (int(os.environ.get("FLIP_TABLE_CAMERA_FRAME_INDEX", "10")),)
    if step not in frame_indices:
        return

    camera_obs: dict[str, Any] = {}
    for group_name in ("policy", "embodiment_general_obs"):
        group = observation.get(group_name, {})
        if isinstance(group, dict):
            camera_obs.update(group)
    if not camera_obs:
        return
    output_root = os.environ.get("FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR", "")
    frame_root = Path(output_root) if output_root else Path(str(usr_args.get("save_path", "."))) / "camera_frames"
    save_key = (str(frame_root), step)
    saved_keys = getattr(policy, "_flip_table_camera_frames_saved_keys", set())
    if save_key in saved_keys:
        return

    camera_names = _camera_names(usr_args)
    keys = {camera_name: _resolve_camera_rgb_key(camera_obs, camera_name) for camera_name in camera_names}
    batch_sizes = {
        int(getattr(camera_obs[key], "shape", (1,))[0])
        if len(tuple(getattr(camera_obs[key], "shape", ()))) == 4
        else 1
        for key in keys.values()
    }
    if len(batch_sizes) != 1:
        raise ValueError(f"camera export batch sizes disagree: {sorted(batch_sizes)}")
    batch_size = batch_sizes.pop()
    batch_export = _env_bool("FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT", False)
    if batch_size != 1 and not batch_export:
        raise ValueError(
            "camera export received multiple environments; set "
            "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT=true for calibration output"
        )
    if batch_size > 64:
        raise ValueError(f"camera export supports at most 64 environments, got {batch_size}")

    frame_dir = frame_root / f"frame_{step:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    save_recorded_geometry = _env_bool("FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY", False)
    frame_metadata = {
        "frame_index": step,
        "batch_size": batch_size,
        "recorded_camera_geometry_applied": save_recorded_geometry,
        "environments": [],
    }
    for batch_index in range(batch_size):
        env_dir = frame_dir / f"env_{batch_index:03d}" if batch_size > 1 else frame_dir
        env_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "frame_index": step,
            "environment_index": batch_index,
            "recorded_camera_geometry_applied": save_recorded_geometry,
            "saved": [],
        }
        for camera_name, key in keys.items():
            image = _camera_image_uint8(camera_obs[key], batch_index=batch_index)
            if save_recorded_geometry and camera_name in _RECORDED_CAMERA_GEOMETRY:
                image = _apply_recorded_camera_geometry_numpy(image, camera_name)
            path = env_dir / _camera_output_name(camera_name, key)
            media.write_image(path, image)
            metadata["saved"].append(
                {
                    "camera": camera_name,
                    "role": CAMERA_SAVE_ROLES.get(camera_name, camera_name),
                    "key": key,
                    "path": path.name,
                    "shape": list(image.shape),
                }
            )
        (env_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        frame_metadata["environments"].append(
            {"environment_index": batch_index, "path": str(env_dir.relative_to(frame_dir))}
        )
    (frame_dir / "metadata.json").write_text(json.dumps(frame_metadata, indent=2), encoding="utf-8")
    saved_keys.add(save_key)
    policy._flip_table_camera_frames_saved_keys = saved_keys
    policy._flip_table_camera_frames_saved = True


def _trace_value(value: Any) -> Any:
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
        if array.ndim > 1:
            array = array[0]
        return array.reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=False)
        if array.ndim > 1:
            array = array[0]
        return array.reshape(-1).tolist()
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _trace_array(value: Any) -> Any:
    """Serialize an array without dropping its time/chunk dimension."""
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).tolist()
    return _trace_value(value)


def _write_action_state_trace(policy: BasePolicy, usr_args: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not _env_bool("FLIP_TABLE_SAVE_ACTION_STATE_TRACE", True):
        return
    output_root = usr_args.get("save_path")
    if not output_root:
        return
    path = Path(str(output_root)) / "action_state_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as trace_file:
        for row in rows:
            trace_file.write(json.dumps(row, allow_nan=False) + "\n")
    policy._flip_table_action_state_trace_path = str(path)


def _append_action_state_trace(policy: BasePolicy, usr_args: dict[str, Any], row: dict[str, Any]) -> None:
    """Durably append one row so simulator failures do not erase the rollout."""

    if not _env_bool("FLIP_TABLE_SAVE_ACTION_STATE_TRACE", True):
        return
    output_root = usr_args.get("save_path")
    if not output_root:
        return
    path = Path(str(output_root)) / "action_state_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(row, allow_nan=False) + "\n")
    policy._flip_table_action_state_trace_path = str(path)


def _calibration_scene_diagnostics(task_env: Any) -> dict[str, Any] | None:
    """Return simulator-only scene facts for offline replay diagnosis.

    These values intentionally never enter an observation, action, policy
    branch, or success decision.  They make it possible to tell whether a
    recorded action replay diverged because of scene dynamics or camera/vision
    geometry without granting a policy privileged simulator state.
    """

    if not _env_bool("FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE", False):
        return None
    callback = getattr(task_env, "flip_table_teleop_diagnostics", None)
    if callback is None:
        raise RuntimeError("calibration scene diagnostics are unavailable")
    diagnostics = callback()
    if not isinstance(diagnostics, dict):
        raise RuntimeError("calibration scene diagnostics must be a mapping")
    return diagnostics


def _joint_pos_from_observation(observation: dict[str, Any], device: torch.device) -> torch.Tensor:
    group = observation.get("embodiment_general_obs", {})
    joint_pos = group.get("joint_pos") if isinstance(group, dict) else None
    if joint_pos is None:
        raise ValueError("embodiment_general_obs.joint_pos is required for 16-D arm/hand control")
    tensor = joint_pos.detach() if torch.is_tensor(joint_pos) else torch.as_tensor(joint_pos)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != 33:
        raise ValueError(f"expected named G1 + Dex1 joint_pos shape [N, 33], got {tuple(tensor.shape)}")
    tensor = tensor.to(device=device, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("G1 + Dex1 joint_pos contains NaN or Inf")
    return tensor


def _dex1_joint_pos_to_command(joint_pos: torch.Tensor) -> torch.Tensor:
    normalized = (joint_pos - _DEX1_OPEN_POS) / (_DEX1_CLOSE_POS - _DEX1_OPEN_POS)
    return (2.0 * normalized - 1.0).clamp(-1.0, 1.0)


def _joint_position_hold_action(observation: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Build the simulator's 16-D arm/hand target from measured state."""
    joint_pos = _joint_pos_from_observation(observation, device)
    body_indices = torch.tensor(_G1_ARM_JOINT_INDICES, dtype=torch.long, device=device)
    body = joint_pos.index_select(1, body_indices)
    left_hand = joint_pos[:, _G1_LEFT_DEX1_JOINT_INDICES].mean(dim=1)
    right_hand = joint_pos[:, _G1_RIGHT_DEX1_JOINT_INDICES].mean(dim=1)
    hands = _dex1_joint_pos_to_command(torch.stack((left_hand, right_hand), dim=1))
    return torch.cat((body, hands), dim=1)


def _teleop_action_dim(usr_args: dict[str, Any]) -> int:
    """Resolve the organizer CLI's omitted action dimension to the V1 contract."""

    explicit = usr_args.get("actions_dim", usr_args.get("action_dim"))
    if explicit is None:
        return 16
    try:
        action_dim = int(explicit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid teleoperation action dimension: {explicit!r}") from exc
    if action_dim != 16:
        raise ValueError(
            "AvpTeleopPolicy requires the 16-D arm/hand joint action, "
            f"got explicit action dimension {action_dim}"
        )
    return action_dim


def _runtime_control_hz(task_env: Any) -> float:
    """Read the actual simulator action rate instead of trusting a launcher default."""

    env = getattr(task_env, "unwrapped", task_env)
    service = getattr(env, "_svc", None)
    try:
        if service is not None and hasattr(service, "getattr_value"):
            # RemoteEnv must return only primitive values. Fetching cfg.sim would
            # require unpickling Isaac Sim classes in the policy process.
            dt = service.getattr_value("unwrapped.cfg.sim.dt")
            decimation = service.getattr_value("unwrapped.cfg.decimation")
        else:
            cfg = getattr(env, "cfg", None)
            sim = getattr(cfg, "sim", None)
            dt = getattr(sim, "dt", None)
            decimation = getattr(cfg, "decimation", None)
    except Exception as exc:
        raise RuntimeError("cannot read simulator control-rate scalars") from exc
    if not isinstance(dt, (int, float)) or not isinstance(decimation, (int, float)):
        raise RuntimeError("cannot determine simulator control rate from task_env.cfg.sim.dt and cfg.decimation")
    control_period = float(dt) * float(decimation)
    if not math.isfinite(control_period) or control_period <= 0.0:
        raise RuntimeError(f"invalid simulator control period: dt={dt!r}, decimation={decimation!r}")
    return 1.0 / control_period


def _synchronize_policy_control_rate(policy: Any, task_env: Any, *, attribute: str) -> float:
    """Make a policy's chunk/replay clock match the live V1 environment rate."""

    configured_hz = float(getattr(policy, attribute))
    try:
        actual_hz = _runtime_control_hz(task_env)
    except RuntimeError:
        # Lightweight unit-test environments and third-party wrappers may not
        # expose IsaacLab's cfg. The organizer V1 environment does, so only
        # that runtime path performs automatic synchronization.
        return configured_hz
    if not math.isclose(configured_hz, actual_hz, rel_tol=0.0, abs_tol=1.0e-6):
        print(
            f"[{type(policy).__name__}] overriding configured simulator rate "
            f"{configured_hz:.3f} Hz with live V1 rate {actual_hz:.3f} Hz"
        )
        setattr(policy, attribute, actual_hz)
    return actual_hz


class NoOpPolicy(BasePolicy):
    """Hold the measured reset pose while recording randomized resets."""

    def get_model(self, usr_args: dict[str, Any]) -> None:
        self.action_dim = int(usr_args.get("actions_dim", usr_args.get("action_dim", 1)))
        self.device = torch.device(usr_args.get("env_cfg", {}).get("device", "cuda:0"))

    def get_action(self) -> torch.Tensor:
        return self._blank_action(1, self.action_dim)

    def _blank_action(self, num_envs: int, action_dim: int) -> torch.Tensor:
        action = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=self.device)
        if action_dim == 23:
            # G1-Gripper-Controller-DecoupledWBC action layout:
            # left/right hand scalars, then 21-D WBC action.  The WBC action
            # contains two wxyz quaternions at [5:9] and [12:16].  All-zero
            # quaternions are invalid, so initialize both to the identity.
            action[:, 5] = 1.0
            action[:, 12] = 1.0
        return action

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self.action_dim))
        ever_success = np.zeros(num_envs, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        hold_action = (
            _joint_position_hold_action(observation, self.device)
            if action_dim == 16
            else self._blank_action(num_envs, action_dim)
        )

        for step in range(int(usr_args["time_out_limit"])):
            state_before = self._merged_trace_state(observation)
            action = hold_action
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            scene_diagnostics = _calibration_scene_diagnostics(task_env)
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": True,
                    "action": _trace_value(action),
                    "state_before": _trace_value(state_before),
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                    "simulator_scene_diagnostics": scene_diagnostics,
                }
            )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success

    @staticmethod
    def _merged_trace_state(observation: dict[str, Any]) -> Any:
        group = observation.get("embodiment_general_obs", {})
        if isinstance(group, dict):
            return group.get("joint_pos")
        return None

    def reset_model(self) -> None:
        pass


class TeleopPerformanceBenchmarkPolicy(NoOpPolicy):
    """Measure simulator throughput with head rendering enabled and disabled."""

    _HEAD_CAMERAS = ("first_person_camera", "head_right_camera")
    _OTHER_CAMERAS = (
        "left_hand_camera",
        "right_hand_camera",
        "global_camera",
    )
    _INACTIVE_UPDATE_PERIOD_S = 3600.0

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    @classmethod
    def _set_camera_mode(cls, task_env: Any, *, head_enabled: bool) -> None:
        env = getattr(task_env, "unwrapped", task_env)
        sensors = getattr(getattr(env, "scene", None), "sensors", None)
        if not isinstance(sensors, dict):
            raise RuntimeError("performance benchmark requires direct camera sensor access")
        names = set(cls._HEAD_CAMERAS + cls._OTHER_CAMERAS)
        missing = sorted(names - set(sensors))
        if missing:
            raise RuntimeError(f"performance benchmark cameras are missing: {missing}")
        for name in cls._HEAD_CAMERAS:
            sensors[name].cfg.update_period = (
                1.0 / 30.0 if head_enabled else cls._INACTIVE_UPDATE_PERIOD_S
            )
        for name in cls._OTHER_CAMERAS:
            sensors[name].cfg.update_period = cls._INACTIVE_UPDATE_PERIOD_S
        # ManagerBasedEnv uses this switch to skip the Kit app/render pump while
        # retaining the same physics, actions, observations, and scene.
        env.render_enabled = head_enabled

    @staticmethod
    def _active_white_collision_stats() -> dict[str, int]:
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        white_tokens = (
            "/Scene/Table001/Table001_01/",
            "/Scene/Leg001/Leg001/",
            "/Scene/Leg001_01/Leg001/",
            "/Scene/Leg001_03/Leg001/",
            "/Scene/Leg001_06/Leg001/",
        )
        total = 0
        enabled = 0
        enabled_threads = 0
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not any(token in path for token in white_tokens):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            total += 1
            is_enabled = (
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                is not False
            )
            enabled += int(is_enabled)
            enabled_threads += int(is_enabled and "/ThreadColliders/" in path)
        return {
            "authored": total,
            "enabled": enabled,
            "enabled_thread_colliders": enabled_threads,
        }

    def _warmup(
        self,
        task_env: Any,
        observation: dict[str, Any],
        action: torch.Tensor,
        steps: int,
    ) -> dict[str, Any]:
        for _ in range(steps):
            observation, _, _, _, _ = task_env.step(action)
        self._synchronize(self.device)
        return observation

    def _measure_block(
        self,
        task_env: Any,
        observation: dict[str, Any],
        action: torch.Tensor,
        *,
        mode: str,
        block: int,
        steps: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        durations = []
        self._synchronize(self.device)
        block_started = time.perf_counter()
        for _ in range(steps):
            step_started = time.perf_counter()
            observation, _, _, _, _ = task_env.step(action)
            durations.append(time.perf_counter() - step_started)
        self._synchronize(self.device)
        elapsed_s = time.perf_counter() - block_started
        values = np.asarray(durations, dtype=np.float64)
        report = {
            "mode": mode,
            "block": block,
            "steps": steps,
            "elapsed_s": elapsed_s,
            "throughput_hz": steps / elapsed_s,
            "step_mean_s": float(values.mean()),
            "step_p50_s": float(np.quantile(values, 0.50)),
            "step_p95_s": float(np.quantile(values, 0.95)),
            "step_max_s": float(values.max()),
        }
        print(f"[TeleopPerformanceBenchmarkPolicy] {json.dumps(report)}", flush=True)
        return observation, report

    def eval(
        self,
        task_env: Any,
        observation: dict[str, Any],
        usr_args: dict[str, Any],
        video_writer: Any,
    ):
        if int(usr_args.get("actions_dim", self.action_dim)) != 16:
            raise RuntimeError("teleop performance benchmark requires the 16-D action")
        warmup_steps = int(os.environ.get("FLIP_TABLE_BENCHMARK_WARMUP_STEPS", "40"))
        measure_steps = int(os.environ.get("FLIP_TABLE_BENCHMARK_MEASURE_STEPS", "180"))
        if warmup_steps < 1 or measure_steps < 10:
            raise ValueError("performance benchmark step counts are too small")
        hold_action = _joint_position_hold_action(observation, self.device)
        blocks = []
        for block, mode in enumerate(("head_on", "all_cameras_off") * 2):
            head_enabled = mode == "head_on"
            self._set_camera_mode(task_env, head_enabled=head_enabled)
            observation = self._warmup(
                task_env,
                observation,
                hold_action,
                warmup_steps,
            )
            observation, report = self._measure_block(
                task_env,
                observation,
                hold_action,
                mode=mode,
                block=block,
                steps=measure_steps,
            )
            blocks.append(report)

        aggregate = {}
        for mode in ("head_on", "all_cameras_off"):
            selected = [item for item in blocks if item["mode"] == mode]
            total_steps = sum(item["steps"] for item in selected)
            total_elapsed = sum(item["elapsed_s"] for item in selected)
            aggregate[mode] = {
                "steps": total_steps,
                "elapsed_s": total_elapsed,
                "throughput_hz": total_steps / total_elapsed,
            }
        output = {
            "schema_version": "team_ramen_teleop_performance_benchmark/v1",
            "seed": int(usr_args.get("seed", 0)),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "physics_hz": round(1.0 / float(task_env.unwrapped.cfg.sim.dt)),
            "control_hz_target": _runtime_control_hz(task_env),
            "collision": self._active_white_collision_stats(),
            "camera_modes": {
                "head_on": list(self._HEAD_CAMERAS),
                "all_cameras_off": [],
                "always_inactive": list(self._OTHER_CAMERAS),
            },
            "warmup_steps_per_block": warmup_steps,
            "measure_steps_per_block": measure_steps,
            "blocks": blocks,
            "aggregate": aggregate,
        }
        output_path = Path(str(usr_args["save_path"])) / "teleop_performance_benchmark.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[TeleopPerformanceBenchmarkPolicy] report={output_path} "
            f"aggregate={json.dumps(aggregate)}",
            flush=True,
        )

        return np.zeros(1, dtype=bool)


class Dex1ForceCalibrationPolicy(NoOpPolicy):
    """Verify the simulated 20 N Dex1 drives against known static blockers.

    This diagnostic creates one 40 mm-thick fixture per open gripper and
    closes both hands. Fixture centers use bounds measured from the official
    Dex1 collision STL meshes, which are composed as V1 instance proxies and
    therefore do not expose usable runtime USD bounds. The fixtures and
    simulator force values are never policy observations or training data;
    they only prove the actuator/collider/sensor path.
    """

    _COMPONENTS = tuple(
        (side, finger)
        for side in ("left", "right")
        for finger in (1, 2)
    )

    # Centers measured from the official ``dex1_col_*.stl`` collision meshes
    # named by the shipped G1 URDF. The collision origins are identity. Keep
    # this explicit CAD evidence rather than querying USD BBoxCache: V1
    # composes the mesh as an instance proxy and returns an empty bound.
    _FINGER_COLLISION_CAD_CENTERS_M = {
        "finger_1": np.asarray((0.110152014, -0.030937371, 0.0), dtype=np.float64),
        "finger_2": np.asarray((0.111173369, 0.030937371, 0.0), dtype=np.float64),
    }

    @staticmethod
    def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = quaternion[:3]
        twice_cross = 2.0 * np.cross(xyz, vector)
        return vector + quaternion[3] * twice_cross + np.cross(xyz, twice_cross)

    @classmethod
    def _gripper_probe_geometry(
        cls, task_env: Any
    ) -> dict[str, dict[str, np.ndarray]]:
        from robofinals.utils.isaac_data_compat import (
            as_torch,
            sim_quat_raw_to_xyzw_torch,
        )

        env = task_env.unwrapped
        robot = env.scene["robot"]
        names = [str(name) for name in robot.data.body_names]
        positions = as_torch(robot.data.body_pos_w)[0].detach().float().cpu().numpy()
        quaternions = sim_quat_raw_to_xyzw_torch(
            as_torch(robot.data.body_quat_w)[0].detach().float()
        ).cpu().numpy()
        result: dict[str, dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            collision_centers = []
            body_positions = []
            body_quaternions = []
            for finger_index in (1, 2):
                name = f"{side}_dex1_finger_link_{finger_index}"
                if name not in names:
                    raise RuntimeError(f"Dex1 force probe body is missing: {name}")
                body_index = names.index(name)
                offset = cls._FINGER_COLLISION_CAD_CENTERS_M[
                    f"finger_{finger_index}"
                ]
                collision_centers.append(
                    positions[body_index]
                    + cls._rotate_xyzw(quaternions[body_index], offset)
                )
                body_positions.append(positions[body_index])
                body_quaternions.append(quaternions[body_index])
            collision_centers_array = np.asarray(collision_centers)
            closing_axis = collision_centers_array[1] - collision_centers_array[0]
            closing_axis /= np.linalg.norm(closing_axis)
            longitudinal_axes = [
                cls._rotate_xyzw(quaternion, np.asarray((1.0, 0.0, 0.0)))
                for quaternion in body_quaternions
            ]
            if np.dot(longitudinal_axes[0], longitudinal_axes[1]) < 0.0:
                longitudinal_axes[1] = -longitudinal_axes[1]
            longitudinal_axis = longitudinal_axes[0] + longitudinal_axes[1]
            longitudinal_axis /= np.linalg.norm(longitudinal_axis)
            result[side] = {
                "center": np.mean(collision_centers_array, axis=0),
                "finger_collision_centers": collision_centers_array,
                "finger_body_positions": np.asarray(body_positions),
                "finger_body_quaternions_xyzw": np.asarray(body_quaternions),
                "closing_axis": closing_axis,
                "longitudinal_axis": longitudinal_axis,
            }
        return result

    @classmethod
    def _gripper_probe_centers(cls, task_env: Any) -> dict[str, np.ndarray]:
        return {
            side: values["center"]
            for side, values in cls._gripper_probe_geometry(task_env).items()
        }

    @staticmethod
    def _verify_probe_blockers(
        task_env: Any, centers: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        from pxr import UsdGeom

        stage = task_env.unwrapped.sim.stage
        cache = UsdGeom.XformCache()
        blocker_centers = {}
        for side, center in centers.items():
            path = (
                "/World/envs/env_0/Scene/FlipTableDex1ForceCalibration/"
                f"{side}_blocker"
            )
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
                raise RuntimeError(f"pre-start Dex1 calibration blocker is missing: {path}")
            blocker_center = np.asarray(
                cache.GetLocalToWorldTransform(prim).ExtractTranslation(),
                dtype=np.float64,
            )
            error_m = float(np.linalg.norm(blocker_center - center))
            if error_m > 0.005:
                raise RuntimeError(
                    f"{side} Dex1 calibration blocker is misaligned by {error_m:.6f} m"
                )
            blocker_centers[side] = blocker_center
        return blocker_centers

    @staticmethod
    def _force_pair(metrics: Any) -> np.ndarray:
        if not isinstance(metrics, dict) or not metrics.get("available"):
            raise RuntimeError("Dex1 force calibration metrics are unavailable")
        result = np.asarray(
            (metrics.get("left_max_n"), metrics.get("right_max_n")),
            dtype=np.float64,
        )
        if result.shape != (2,) or not np.isfinite(result).all():
            raise RuntimeError("Dex1 force calibration metrics are invalid")
        return result

    @staticmethod
    def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
        """Return a finite unit vector or fail the diagnostic explicitly."""

        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 1.0e-9:
            raise RuntimeError(f"invalid Dex1 force-calibration {name} axis")
        return vector / length

    @classmethod
    def _blocker_frame(
        cls, geometry: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Build a fixed fixture frame from the pre-contact open-hand pose.

        The fixture is static in the world, so all later finger positions must
        be expressed in this initial frame.  Recording this frame prevents a
        moving WBC hand from being mistaken for a missing collider.
        """

        closing_axis = cls._normalized(geometry["closing_axis"], name="closing")
        longitudinal_axis = cls._normalized(
            geometry["longitudinal_axis"], name="longitudinal"
        )
        normal_axis = cls._normalized(
            np.cross(longitudinal_axis, closing_axis), name="normal"
        )
        # Re-orthogonalize the longitudinal axis to make the reported fixture
        # coordinates numerically stable even when the two finger links have a
        # small relative orientation error after reset.
        longitudinal_axis = cls._normalized(
            np.cross(closing_axis, normal_axis), name="orthogonal longitudinal"
        )
        return {
            "longitudinal_axis": longitudinal_axis,
            "closing_axis": closing_axis,
            "normal_axis": normal_axis,
        }

    @staticmethod
    def _fixture_coordinates(
        point_world_m: np.ndarray,
        blocker_center_world_m: np.ndarray,
        frame: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return ``[longitudinal, closing, normal]`` coordinates in metres."""

        delta = point_world_m - blocker_center_world_m
        return np.asarray(
            (
                float(np.dot(delta, frame["longitudinal_axis"])),
                float(np.dot(delta, frame["closing_axis"])),
                float(np.dot(delta, frame["normal_axis"])),
            ),
            dtype=np.float64,
        )

    def eval(
        self,
        task_env: Any,
        observation: dict[str, Any],
        usr_args: dict[str, Any],
        video_writer: Any,
    ):
        if int(usr_args.get("actions_dim", self.action_dim)) != 16:
            raise RuntimeError("Dex1 force calibration requires the 16-D action")
        env = task_env.unwrapped
        # Rendering is irrelevant to this physics-only probe.  Keep contact
        # sensors at their configured 0 s period; disabling every scene sensor
        # here would leave their force buffers permanently at zero.
        for sensor_name, sensor in env.scene.sensors.items():
            if "camera" not in str(sensor_name).lower():
                continue
            if hasattr(sensor, "cfg") and hasattr(sensor.cfg, "update_period"):
                sensor.cfg.update_period = 3600.0
        env.render_enabled = False

        probe_geometry = self._gripper_probe_geometry(task_env)
        blocker_centers = self._verify_probe_blockers(
            task_env, {side: values["center"] for side, values in probe_geometry.items()}
        )
        blocker_frames = {
            side: self._blocker_frame(values) for side, values in probe_geometry.items()
        }
        hold = _joint_position_hold_action(observation, self.device)
        drive_max = np.zeros(2, dtype=np.float64)
        contact_max = np.zeros(2, dtype=np.float64)
        drive_joint_max = {component: 0.0 for component in self._COMPONENTS}
        contact_sensor_max = {component: 0.0 for component in self._COMPONENTS}
        contact_streak = {component: 0 for component in self._COMPONENTS}
        contact_streak_max = {component: 0 for component in self._COMPONENTS}
        finger_position_min_m = {component: math.inf for component in self._COMPONENTS}
        finger_position_max_m = {component: -math.inf for component in self._COMPONENTS}
        fixture_coordinate_min_m = {
            component: np.full(3, math.inf, dtype=np.float64)
            for component in self._COMPONENTS
        }
        fixture_coordinate_max_m = {
            component: np.full(3, -math.inf, dtype=np.float64)
            for component in self._COMPONENTS
        }
        fixture_center_displacement_max_m = {component: 0.0 for component in self._COMPONENTS}
        # A fixture is 160 x 40 x 300 mm in [longitudinal, closing, normal].
        # This is a geometric audit only: the collision result remains the
        # contact sensor and joint-position evidence below.
        fixture_half_extents_m = np.asarray((0.080, 0.020, 0.150), dtype=np.float64)
        fixture_center_overlap_max_m = {component: 0.0 for component in self._COMPONENTS}
        opening_min = np.ones(2, dtype=np.float64)
        rows = []
        open_steps = 50
        close_steps = 150
        for step in range(open_steps + close_steps):
            action = hold.clone()
            action[:, 14:16] = -1.0 if step < open_steps else 1.0
            observation, _, _, _, _ = task_env.step(action)
            diagnostics = env.flip_table_teleop_force_diagnostics()
            drive_metrics = diagnostics.get("dex1_drive_force_n")
            contact_metrics = diagnostics.get("gripper_contact_force_n")
            drive = self._force_pair(drive_metrics)
            contact = self._force_pair(contact_metrics)
            drive_max = np.maximum(drive_max, drive)
            contact_max = np.maximum(contact_max, contact)
            joint_values = (
                _joint_pos_from_observation(observation, self.device)[0]
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            opening = AvpTeleopPolicy._dex1_opening(joint_values)
            opening_min = np.minimum(opening_min, opening)
            current_geometry = self._gripper_probe_geometry(task_env)
            drive_by_joint = drive_metrics.get("joint_abs_n", {})
            contact_by_sensor = contact_metrics.get("sensor_max_n", {})
            for side_index, (side, finger_index) in enumerate(self._COMPONENTS):
                joint_name = f"{side}_dex1_finger_joint_{finger_index}"
                sensor_name = (
                    f"{side}_gripper_contact"
                    + ("" if finger_index == 1 else "_2")
                )
                if joint_name not in drive_by_joint or sensor_name not in contact_by_sensor:
                    raise RuntimeError(
                        f"Dex1 calibration component is unavailable: {joint_name}/{sensor_name}"
                    )
                component = (side, finger_index)
                joint_force = float(drive_by_joint[joint_name])
                sensor_force = float(contact_by_sensor[sensor_name])
                joint_index = (
                    _G1_LEFT_DEX1_JOINT_INDICES[finger_index - 1]
                    if side == "left"
                    else _G1_RIGHT_DEX1_JOINT_INDICES[finger_index - 1]
                )
                joint_position = float(joint_values[joint_index])
                finger_position_min_m[component] = min(
                    finger_position_min_m[component], joint_position
                )
                finger_position_max_m[component] = max(
                    finger_position_max_m[component], joint_position
                )
                finger_center = current_geometry[side]["finger_collision_centers"][
                    finger_index - 1
                ]
                fixture_coordinates = self._fixture_coordinates(
                    finger_center,
                    blocker_centers[side],
                    blocker_frames[side],
                )
                fixture_coordinate_min_m[component] = np.minimum(
                    fixture_coordinate_min_m[component], fixture_coordinates
                )
                fixture_coordinate_max_m[component] = np.maximum(
                    fixture_coordinate_max_m[component], fixture_coordinates
                )
                fixture_center_displacement_max_m[component] = max(
                    fixture_center_displacement_max_m[component],
                    float(
                        np.linalg.norm(
                            finger_center
                            - probe_geometry[side]["finger_collision_centers"][finger_index - 1]
                        )
                    ),
                )
                fixture_center_overlap_max_m[component] = max(
                    fixture_center_overlap_max_m[component],
                    float(
                        np.min(fixture_half_extents_m - np.abs(fixture_coordinates))
                    ),
                )
                drive_joint_max[component] = max(
                    drive_joint_max[component], joint_force
                )
                contact_sensor_max[component] = max(
                    contact_sensor_max[component], sensor_force
                )
                side_opening = opening[0 if side == "left" else 1]
                sustained_contact = sensor_force >= 1.0 and side_opening >= 0.20
                contact_streak[component] = (
                    contact_streak[component] + 1 if sustained_contact else 0
                )
                contact_streak_max[component] = max(
                    contact_streak_max[component], contact_streak[component]
                )
            if step % 5 == 0 or step == open_steps + close_steps - 1:
                rows.append(
                    {
                        "step": step,
                        "command": "open" if step < open_steps else "closed",
                        "opening_fraction_left_right": opening.round(6).tolist(),
                        "drive_force_n_left_right": drive.round(6).tolist(),
                        "contact_force_n_left_right": contact.round(6).tolist(),
                    }
                )

        sustained_contact_by_finger_s = {
            component: streak / 50.0
            for component, streak in contact_streak_max.items()
        }
        passed_by_side = np.asarray(
            [
                opening_min[side_index] >= 0.20
                and all(
                    drive_joint_max[(side, finger)] >= 18.0
                    and contact_sensor_max[(side, finger)] >= 2.0
                    and sustained_contact_by_finger_s[(side, finger)] >= 0.50
                    for finger in (1, 2)
                )
                for side_index, side in enumerate(("left", "right"))
            ],
            dtype=bool,
        )
        sustained_contact_s = np.asarray(
            [
                min(
                    sustained_contact_by_finger_s[(side, finger)]
                    for finger in (1, 2)
                )
                for side in ("left", "right")
            ]
        )
        report = {
            "schema_version": "team_ramen_dex1_force_calibration/v4",
            "diagnostic_only": True,
            "fixture_alignment": {
                "time": "authored before Isaac starts; static throughout the diagnostic",
                "center_source": "official Dex1 collision-STL bounds plus runtime link FK",
            },
            "blocker_size_m": [0.160, 0.040, 0.300],
            "blocker_leg_thickness_m": 0.040,
            "configured_effort_limit_n_per_finger": 20.0,
            "blocker_center_world_m": {
                side: center.round(6).tolist()
                for side, center in blocker_centers.items()
            },
            "probe_geometry": {
                side: {
                    name: value.round(6).tolist()
                    for name, value in values.items()
                }
                for side, values in probe_geometry.items()
            },
            "final_probe_center_world_m": {
                side: values["center"].round(6).tolist()
                for side, values in self._gripper_probe_geometry(task_env).items()
            },
            "drive_force_max_n_left_right": drive_max.round(6).tolist(),
            "contact_force_max_n_left_right": contact_max.round(6).tolist(),
            "sustained_contact_max_s_left_right": sustained_contact_s.round(4).tolist(),
            "drive_force_max_n_by_joint": {
                f"{side}_finger_{finger}": round(value, 6)
                for (side, finger), value in drive_joint_max.items()
            },
            "contact_force_max_n_by_sensor": {
                f"{side}_finger_{finger}": round(value, 6)
                for (side, finger), value in contact_sensor_max.items()
            },
            "sustained_contact_s_by_finger": {
                f"{side}_finger_{finger}": round(value, 4)
                for (side, finger), value in sustained_contact_by_finger_s.items()
            },
            "finger_prismatic_position_range_m": {
                f"{side}_finger_{finger}": [
                    round(finger_position_min_m[(side, finger)], 6),
                    round(finger_position_max_m[(side, finger)], 6),
                ]
                for side, finger in self._COMPONENTS
            },
            "fixture_frame_world_axes": {
                side: {
                    name: value.round(6).tolist()
                    for name, value in frame.items()
                }
                for side, frame in blocker_frames.items()
            },
            "finger_fixture_coordinate_range_m": {
                f"{side}_finger_{finger}": {
                    "minimum": fixture_coordinate_min_m[(side, finger)].round(6).tolist(),
                    "maximum": fixture_coordinate_max_m[(side, finger)].round(6).tolist(),
                }
                for side, finger in self._COMPONENTS
            },
            "finger_fixture_center_displacement_max_m": {
                f"{side}_finger_{finger}": round(value, 6)
                for (side, finger), value in fixture_center_displacement_max_m.items()
            },
            "finger_fixture_center_overlap_max_m": {
                f"{side}_finger_{finger}": round(value, 6)
                for (side, finger), value in fixture_center_overlap_max_m.items()
            },
            "minimum_opening_fraction_left_right": opening_min.round(6).tolist(),
            "passed_left_right": passed_by_side.tolist(),
            "passed": bool(np.all(passed_by_side)),
            "samples": rows,
        }
        output_path = Path(str(usr_args["save_path"])) / "dex1_force_calibration.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"[Dex1ForceCalibrationPolicy] {json.dumps(report)}", flush=True)
        if not report["passed"]:
            raise RuntimeError(
                "Dex1 actuator/collider calibration failed; see " + str(output_path)
            )
        return np.zeros(1, dtype=bool)


class ScriptedJointPolicy(NoOpPolicy):
    """A conservative absolute joint-target tracking probe.

    This is not a trained flip policy. It is useful for checking that the reset,
    action, recording, and success-accounting pipeline runs end to end.
    """

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self.action_dim))
        amplitude = float(os.environ.get("FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE", usr_args.get("scripted_action_amplitude", 0.12)))
        period = max(1, int(os.environ.get("FLIP_TABLE_SCRIPTED_ACTION_PERIOD", usr_args.get("scripted_action_period", 80))))
        debug_every = int(os.environ.get("FLIP_TABLE_SCRIPTED_DEBUG_EVERY", "0"))
        upper_body_indices = [2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28, 29, 31]
        ever_success = np.zeros(num_envs, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        base_action = (
            _joint_position_hold_action(observation, self.device)
            if action_dim == 16
            else self._blank_action(num_envs, action_dim)
        )

        for step in range(int(usr_args["time_out_limit"])):
            phase = math.sin(2.0 * math.pi * float(step) / float(period))
            action = base_action.clone()
            if action_dim >= 28:
                # Small mirrored shoulder/elbow probe in the common G1 eval order.
                for index, sign in ((6, -1.0), (8, 1.0), (20, -1.0), (22, 1.0)):
                    action[:, index] = sign * amplitude * phase
            elif action_dim == 23:
                # Keep the WBC identity quaternions from _blank_action intact.
                for index, sign in ((2, -1.0), (9, 1.0)):
                    action[:, index] = sign * amplitude * phase
            elif action_dim == 16:
                # Absolute target layout: left arm(7), right arm(7), Dex1(2).
                # Keep every unprobed joint at its
                # measured reset value and move only mirrored shoulder pitch.
                action[:, 0] = base_action[:, 0] + amplitude * phase
                action[:, 7] = base_action[:, 7] - amplitude * phase
            elif action_dim > 0:
                raise ValueError(f"ScriptedJointPolicy does not support action_dim={action_dim}")

            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            actual = (
                _joint_position_hold_action(observation, self.device)[:, :14]
                if action_dim == 16
                else None
            )
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": True,
                    "phase": phase,
                    "target": _trace_value(action),
                    "actual_joint_position": _trace_value(actual),
                    "tracking_error": _trace_value(action[:, :14] - actual) if actual is not None else None,
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                }
            )
            if debug_every > 0 and step % debug_every == 0:
                merged = observation.get("embodiment_general_obs", {}) if isinstance(observation, dict) else {}
                joint_pos = merged.get("joint_pos") if isinstance(merged, dict) else None
                if torch.is_tensor(joint_pos):
                    joints = joint_pos[0, upper_body_indices].detach().float().cpu()
                    print(
                        "[ScriptedJointPolicy][debug] "
                        f"step={step} phase={phase:+.4f} "
                        f"action_min={action.min().item():+.4f} action_max={action.max().item():+.4f} "
                        f"upper_body_min={joints.min().item():+.4f} upper_body_max={joints.max().item():+.4f} "
                        f"upper_body_values={[round(float(value), 4) for value in joints.tolist()]}"
                    )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success


class RecordedJointTargetPolicy(NoOpPolicy):
    """Replay recorded real-robot upper-body targets for simulator diagnosis.

    The replay file contains only the same 19-D absolute joint targets that can
    be sent to the real upper-body controller. It is intentionally open-loop
    and is not a production policy; its purpose is to separate scene/physics
    failures from the learned policy's camera-domain failure.
    """

    _SOURCE_HAND_MAX = 4.5

    @staticmethod
    def _review_video_stride(control_hz: float) -> int:
        """Return a deterministic diagnostic-video stride for recorded replays.

        Physics, control, action/state traces, and requested RGB evidence remain
        at their native rates.  Only the human-review MP4 is sampled, which
        avoids making long calibration replays encoder-bound.  The effective
        rate is recorded in the trace so it cannot be mistaken for the control
        or source-data clock.
        """

        review_hz = float(os.environ.get("FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ", "10"))
        if not math.isfinite(review_hz) or review_hz <= 0.0 or review_hz > control_hz:
            raise ValueError(
                "FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ must be finite and in "
                f"(0, {control_hz}], got {review_hz}"
            )
        return max(1, int(round(control_hz / review_hz)))

    def get_model(self, usr_args: dict[str, Any]) -> None:
        super().get_model(usr_args)
        replay_path = os.environ.get("FLIP_TABLE_REPLAY_ACTION_PATH", "").strip()
        if not replay_path:
            raise ValueError("FLIP_TABLE_REPLAY_ACTION_PATH is required for RecordedJointTargetPolicy")
        path = Path(replay_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            initial_state = payload.get("initial_state_19d")
            observed_states = payload.get("observed_states_19d")
            payload = payload.get("actions")
        else:
            initial_state = None
            observed_states = None
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"replay file must contain a non-empty action list: {path}")
        actions = torch.as_tensor(payload, dtype=torch.float32, device=self.device)
        if actions.ndim != 2 or actions.shape[-1] not in {16, 19}:
            raise ValueError(f"replay actions must have shape [N,16] (or legacy [N,19]), got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("replay actions contain NaN or Inf")
        if actions.shape[-1] == 19:
            actions = torch.cat((actions[:, 3:17], actions[:, 17:19]), dim=1)
            print("[RecordedJointTargetPolicy] deterministically dropped legacy waist actions")
        self._replay_actions = actions
        if observed_states is None:
            raise ValueError(
                "replay file must contain observed_states_19d so real observed-state matching "
                "cannot be confused with command tracking"
            )
        self._replay_observed_states = torch.as_tensor(
            observed_states, dtype=torch.float32, device=self.device
        )
        if self._replay_observed_states.shape != (actions.shape[0], 19) or not torch.isfinite(
            self._replay_observed_states
        ).all():
            raise ValueError("replay observed_states_19d must be finite [N,19] matching actions")
        if initial_state is None:
            self._replay_initial_state = self._replay_observed_states[0].clone()
        else:
            self._replay_initial_state = torch.as_tensor(
                initial_state, dtype=torch.float32, device=self.device
            )
            if self._replay_initial_state.shape != (19,) or not torch.isfinite(self._replay_initial_state).all():
                raise ValueError("replay initial_state_19d must be finite [19]")
        self._replay_hz = float(os.environ.get("FLIP_TABLE_REPLAY_HZ", "30"))
        self._sim_control_hz = float(os.environ.get("FLIP_TABLE_ACT_SIM_CONTROL_HZ", "50"))
        self._review_video_stride_steps = self._review_video_stride(self._sim_control_hz)
        self._replay_warmup_steps = int(os.environ.get("FLIP_TABLE_REPLAY_WARMUP_STEPS", "0"))
        self._replay_command_delay_steps = int(
            os.environ.get("FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS", "0")
        )
        raw_hold_index = os.environ.get("FLIP_TABLE_REPLAY_HOLD_INDEX", "").strip()
        self._replay_hold_index = int(raw_hold_index) if raw_hold_index else None
        if (
            self._replay_hz <= 0
            or self._sim_control_hz <= 0
            or self._replay_warmup_steps < 0
            or not 0 <= self._replay_command_delay_steps <= 25
        ):
            raise ValueError(
                "replay rates must be positive, warmup must be non-negative, and command delay "
                "must be an integer in [0,25]"
            )
        print(
            f"[RecordedJointTargetPolicy] loaded {actions.shape[0]} actions from {path}; "
            f"replay_hz={self._replay_hz:.3f}, sim_control_hz={self._sim_control_hz:.3f}, "
            f"warmup_steps={self._replay_warmup_steps}, "
            f"command_delay_steps={self._replay_command_delay_steps}, "
            f"review_video_stride_steps={self._review_video_stride_steps}"
        )

    def _observed_state_at_control_step(self, replay_step: int) -> torch.Tensor:
        """Interpolate the 30 Hz encoder stream at the post-step sim time.

        This affects trace evidence only. Actions retain their recorded 30 Hz
        zero-order hold semantics; a state after a 50 Hz control step is
        compared with the encoder state at the same elapsed time.
        """

        source_position = min(
            float(self._replay_observed_states.shape[0] - 1),
            (float(replay_step) + 1.0) * self._replay_hz / self._sim_control_hz,
        )
        lower = int(math.floor(source_position))
        upper = min(lower + 1, self._replay_observed_states.shape[0] - 1)
        fraction = source_position - float(lower)
        return (
            (1.0 - fraction) * self._replay_observed_states[lower]
            + fraction * self._replay_observed_states[upper]
        )

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        _synchronize_policy_control_rate(self, task_env, attribute="_sim_control_hz")
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self.action_dim))
        if action_dim != 16:
            raise ValueError(f"RecordedJointTargetPolicy requires the 16-D arm/hand action space, got {action_dim}")
        ever_success = np.zeros(num_envs, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        for step in range(int(usr_args["time_out_limit"])):
            replay_step = max(0, step - self._replay_warmup_steps)
            command_step = max(0, replay_step - self._replay_command_delay_steps)
            replay_index = min(
                self._replay_actions.shape[0] - 1,
                int(round(float(command_step) * self._replay_hz / self._sim_control_hz)),
            )
            if self._replay_hold_index is not None:
                replay_index = max(0, min(self._replay_actions.shape[0] - 1, self._replay_hold_index))
            source_action = (
                torch.cat((self._replay_initial_state[3:17], self._replay_initial_state[17:19]))
                if step < self._replay_warmup_steps
                else self._replay_actions[replay_index]
            )
            source_observed_state = (
                self._replay_initial_state
                if step < self._replay_warmup_steps
                else self._observed_state_at_control_step(replay_step)
            )
            action = source_action.unsqueeze(0).expand(num_envs, -1).clone()
            # Dataset hand_cmd uses the production policy convention:
            # 0.0 is closed and 4.5 is open.  The environment's Dex1 action
            # convention is the inverse: -1.0 is open and +1.0 is closed.
            # Keep this conversion here rather than treating a recorded
            # diagnostic replay as a special dataset format.
            action[:, 14:16] = (
                1.0
                - 2.0
                * action[:, 14:16].clamp(0.0, self._SOURCE_HAND_MAX)
                / self._SOURCE_HAND_MAX
            )
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            scene_diagnostics = _calibration_scene_diagnostics(task_env)
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": True,
                    "replay_warmup": step < self._replay_warmup_steps,
                    "replay_index": replay_index,
                    "source_observed_frame": min(
                        float(self._replay_observed_states.shape[0] - 1),
                        (float(replay_step) + 1.0)
                        * self._replay_hz
                        / self._sim_control_hz,
                    ),
                    "command_delay_steps": self._replay_command_delay_steps,
                    "review_video_stride_steps": self._review_video_stride_steps,
                    "source_action_16d": _trace_value(source_action),
                    "source_observed_state_19d": _trace_value(source_observed_state),
                    "action": _trace_value(action),
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                    "simulator_scene_diagnostics": scene_diagnostics,
                }
            )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            if step % self._review_video_stride_steps == 0:
                self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success


class RecordedFullBodyTargetPolicy(NoOpPolicy):
    """Replay body29 + Dex1 records for offline V1 controller identification.

    This path is intentionally not a manipulation policy and is never used by
    real deployment.  It applies recorded joint targets through the simulator
    action manager once per control tick, while leaving the floating root
    dynamic.  Its purpose is to expose a lower-body/controller mismatch rather
    than concealing it with a root lock or per-frame state write.
    """

    _SOURCE_HAND_MAX = 4.5

    def get_model(self, usr_args: dict[str, Any]) -> None:
        super().get_model(usr_args)
        replay_path = os.environ.get("FLIP_TABLE_REPLAY_ACTION_PATH", "").strip()
        if not replay_path:
            raise ValueError("FLIP_TABLE_REPLAY_ACTION_PATH is required for RecordedFullBodyTargetPolicy")
        path = Path(replay_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("full-body replay file must be a JSON object")
        actions = torch.as_tensor(payload.get("actions"), dtype=torch.float32, device=self.device)
        observed = torch.as_tensor(
            payload.get("observed_states_31d"), dtype=torch.float32, device=self.device
        )
        initial = torch.as_tensor(payload.get("initial_state_31d"), dtype=torch.float32, device=self.device)
        if actions.ndim != 2 or actions.shape[-1] != 31 or actions.shape[0] == 0:
            raise ValueError("full-body replay actions must be finite [N,31]")
        if observed.shape != actions.shape or initial.shape != (31,):
            raise ValueError("full-body replay observed_states_31d/initial_state_31d shape mismatch")
        if not torch.isfinite(actions).all() or not torch.isfinite(observed).all() or not torch.isfinite(initial).all():
            raise ValueError("full-body replay contains NaN or Inf")
        self._replay_actions = actions
        self._replay_observed_states = observed
        self._replay_initial_state = initial
        self._replay_hz = float(os.environ.get("FLIP_TABLE_REPLAY_HZ", "30"))
        self._sim_control_hz = float(os.environ.get("FLIP_TABLE_ACT_SIM_CONTROL_HZ", "50"))
        self._replay_warmup_steps = int(os.environ.get("FLIP_TABLE_REPLAY_WARMUP_STEPS", "0"))
        raw_hold_index = os.environ.get("FLIP_TABLE_REPLAY_HOLD_INDEX", "").strip()
        self._replay_hold_index = int(raw_hold_index) if raw_hold_index else None
        if self._replay_hz <= 0.0 or self._sim_control_hz <= 0.0 or self._replay_warmup_steps < 0:
            raise ValueError("full-body replay rates must be positive and warmup non-negative")
        print(
            f"[RecordedFullBodyTargetPolicy] loaded {actions.shape[0]} actions from {path}; "
            "root_replay=forbidden_per_frame",
            flush=True,
        )

    def _observed_state_at_control_step(self, replay_step: int) -> torch.Tensor:
        source_position = min(
            float(self._replay_observed_states.shape[0] - 1),
            (float(replay_step) + 1.0) * self._replay_hz / self._sim_control_hz,
        )
        lower = int(math.floor(source_position))
        upper = min(lower + 1, self._replay_observed_states.shape[0] - 1)
        return (
            (1.0 - (source_position - lower)) * self._replay_observed_states[lower]
            + (source_position - lower) * self._replay_observed_states[upper]
        )

    @staticmethod
    def _actual_state_31d(observation: dict[str, Any]) -> torch.Tensor:
        joint_pos = _joint_pos_from_observation(observation, torch.device("cpu"))
        values = joint_pos[0].detach().cpu().double().numpy()
        body, _ = AvpTeleopPolicy._body_state_from_joint_vector(values)
        fingers = np.asarray(
            (
                values[list(_G1_LEFT_DEX1_JOINT_INDICES)].mean(),
                values[list(_G1_RIGHT_DEX1_JOINT_INDICES)].mean(),
            ),
            dtype=np.float64,
        )
        hands = (fingers - _DEX1_CLOSE_POS) / (_DEX1_OPEN_POS - _DEX1_CLOSE_POS) * 4.5
        return torch.as_tensor(np.concatenate((body, hands)), dtype=torch.float32)

    @staticmethod
    def _verify_direct_action_layout(task_env: Any) -> None:
        """Fail closed if V1 changes the body/Dex1 action concatenation."""

        manager = getattr(getattr(task_env, "unwrapped", task_env), "action_manager", None)
        expected = (
            ("left_leg_action", 6),
            ("right_leg_action", 6),
            ("waist_action", 3),
            ("left_arm_action", 7),
            ("right_arm_action", 7),
            ("left_hand_action", 1),
            ("right_hand_action", 1),
        )
        if manager is None:
            raise RuntimeError("full-body diagnostic requires an Isaac action manager")
        active = tuple(str(name) for name in getattr(manager, "active_terms", ()))
        if active and active != tuple(name for name, _ in expected):
            raise RuntimeError(
                "full-body action term order differs from the recorded dataset contract: "
                f"expected={[name for name, _ in expected]!r}, actual={active!r}"
            )
        observed = []
        for name, dimension in expected:
            term = manager.get_term(name)
            actual_dimension = int(getattr(term, "action_dim", -1))
            if actual_dimension != dimension:
                raise RuntimeError(
                    f"full-body action term {name} must have dimension {dimension}, "
                    f"got {actual_dimension}"
                )
            observed.append(actual_dimension)
        if sum(observed) != 31:
            raise RuntimeError(f"full-body action dimensions must sum to 31, got {observed}")

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        _synchronize_policy_control_rate(self, task_env, attribute="_sim_control_hz")
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self.action_dim))
        if action_dim != 31:
            raise ValueError(f"RecordedFullBodyTargetPolicy requires 31-D body/Dex1 actions, got {action_dim}")
        self._verify_direct_action_layout(task_env)
        ever_success = np.zeros(num_envs, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        for step in range(int(usr_args["time_out_limit"])):
            replay_step = max(0, step - self._replay_warmup_steps)
            replay_index = min(
                self._replay_actions.shape[0] - 1,
                int(round(float(replay_step) * self._replay_hz / self._sim_control_hz)),
            )
            if self._replay_hold_index is not None:
                replay_index = max(0, min(self._replay_actions.shape[0] - 1, self._replay_hold_index))
            source_action = (
                self._replay_initial_state if step < self._replay_warmup_steps else self._replay_actions[replay_index]
            )
            source_observed = (
                self._replay_initial_state
                if step < self._replay_warmup_steps
                else self._observed_state_at_control_step(replay_step)
            )
            action = source_action.unsqueeze(0).expand(num_envs, -1).clone()
            # Source: 0=closed, 4.5=open. Native V1 Dex1 term: +1=closed, -1=open.
            action[:, 29:31] = 1.0 - 2.0 * action[:, 29:31].clamp(0.0, self._SOURCE_HAND_MAX) / self._SOURCE_HAND_MAX
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            actual = self._actual_state_31d(observation)
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": True,
                    "replay_warmup": step < self._replay_warmup_steps,
                    "replay_index": replay_index,
                    "source_action_31d": _trace_value(source_action),
                    "source_observed_state_31d": _trace_value(source_observed),
                    "action": _trace_value(action),
                    "actual_state_31d": _trace_value(actual),
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                    "simulator_scene_diagnostics": _calibration_scene_diagnostics(task_env),
                }
            )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success


class CvRuleBasedPolicy(NoOpPolicy):
    """RGB leg detection followed by geometry-only grasp and flip rules."""

    _ACTION_DIM = 16
    _MIN_CV_CONFIDENCE = 0.015
    _MIN_LEG_CONFIDENCE = 0.20
    # The RTX camera pipeline can still expose a pre-reset frame for roughly
    # one second after Isaac Sim resets the assembled-table scene.  Keep the
    # neutral hold for two seconds, then select only from its stable tail.
    _CV_WARMUP_STEPS = 100
    _CV_SETTLED_SELECTION_STEPS = 30
    _WRIST_SERVO_GAIN = 0.80
    _WRIST_SERVO_MAX_CORRECTION_M = 0.12
    _FIRST_ROLL_WRIST_SERVO_GAIN = 0.35
    _FIRST_ROLL_WRIST_SERVO_MAX_CORRECTION_M = 0.05
    _FIRST_ROLL_RECOVERY_SERVO_GAIN = 0.80
    _FIRST_ROLL_RECOVERY_SERVO_MAX_CORRECTION_M = 0.08
    _FIRST_ROLL_MAX_TRACKING_ERROR_M = 0.09
    _FIRST_ROLL_MAX_STALL_STEPS = 500
    _WRIST_VISUAL_SERVO_M_PER_PX = 0.0002
    _WRIST_VISUAL_SERVO_MAX_M = 0.08
    _WRIST_VISUAL_SERVO_MAX_STEP_M = 0.012
    _WRIST_VISUAL_DEPTH_GAIN_M_PER_PX = 0.0002
    _WRIST_VISUAL_DEPTH_MAX_STEP_M = 0.004
    _WRIST_VISUAL_DEPTH_MAX_M = 0.025
    _WRIST_EDGE_VISUAL_DEPTH_GAIN_M_PER_PX = 0.00025
    _WRIST_EDGE_VISUAL_DEPTH_MAX_STEP_M = 0.004
    _WRIST_EDGE_VISUAL_DEPTH_MAX_M = 0.05
    _WRIST_SHAFT_MIN_CONFIDENCE = 0.75
    _WRIST_SHAFT_MAX_CENTER_ERROR_PX = 8.0
    _WRIST_SHAFT_MIN_BOTTOM_FRACTION = 0.78
    _WRIST_EDGE_MIN_BOTTOM_FRACTION = 0.60
    _DEX1_GRASP_BLOCK_THRESHOLD_RAD = -0.017
    _GRASP_LOSS_LIMIT_STEPS = 8
    _GRASP_GATE_STABLE_STEPS = 2
    _FIRST_ROLL_ADVANCE_INTERVAL = 5
    _MAX_CARTESIAN_LINEAR_SPEED_M_S = 0.25
    _MAX_CARTESIAN_ANGULAR_SPEED_RAD_S = 0.75
    _MAX_DEX1_COMMAND_SPEED_S = 3.5
    _DEX1_COLLISION_CENTER_OFFSETS = {
        "finger_1": np.asarray((0.1102, -0.0309, 0.0), dtype=np.float64),
        "finger_2": np.asarray((0.1112, 0.0309, 0.0), dtype=np.float64),
    }
    _PINOCCHIO_MODEL_RELATIVE = Path(
        "robofinals/data/assets/g1_urdf_gripper/g1/g1_29dof_mode_15_with_dex1_1.urdf"
    )
    _G1_GRIPPER_33_JOINT_ORDER = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
    )

    def get_model(self, usr_args: dict[str, Any]) -> None:
        super().get_model(usr_args)
        self.estimator = TabletopPoseEstimator()
        self.leg_detector = TableLegDetector()
        self.wrist_shaft_detector = WristShaftDetector()
        self.wrist_tabletop_edge_detector = WristTabletopEdgeDetector()
        self._ensure_camera_fk_model()
        self._wrist_from_camera_rotation = self._rotation_from_wxyz(
            np.asarray((0.66900606, 0.28059008, -0.25108559, -0.64082457))
        )
        self.sim_control_hz = float(os.environ.get("FLIP_TABLE_CV_SIM_CONTROL_HZ", "50"))
        self.minimum_confidence = float(
            os.environ.get("FLIP_TABLE_CV_MIN_CONFIDENCE", str(self._MIN_CV_CONFIDENCE))
        )
        self.minimum_leg_confidence = float(
            os.environ.get("FLIP_TABLE_CV_MIN_LEG_CONFIDENCE", str(self._MIN_LEG_CONFIDENCE))
        )
        self.warmup_steps = int(os.environ.get("FLIP_TABLE_CV_WARMUP_STEPS", str(self._CV_WARMUP_STEPS)))
        self.settled_selection_steps = int(
            os.environ.get(
                "FLIP_TABLE_CV_SETTLED_SELECTION_STEPS",
                str(self._CV_SETTLED_SELECTION_STEPS),
            )
        )
        self.redetect_interval_steps = int(
            os.environ.get("FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS", "10")
        )
        self.redetect_alpha = float(os.environ.get("FLIP_TABLE_CV_REDETECT_ALPHA", "0.30"))
        self.redetect_max_translation_m = float(
            os.environ.get("FLIP_TABLE_CV_REDETECT_MAX_TRANSLATION_M", "0.12")
        )
        self.redetect_max_yaw_rad = float(
            os.environ.get("FLIP_TABLE_CV_REDETECT_MAX_YAW_RAD", "0.35")
        )
        self.redetect_max_center_drift_m = float(
            os.environ.get("FLIP_TABLE_CV_REDETECT_MAX_CENTER_DRIFT_M", "0.03")
        )
        self.redetect_max_attachment_drift_m = float(
            os.environ.get("FLIP_TABLE_CV_REDETECT_MAX_ATTACHMENT_DRIFT_M", "0.03")
        )
        self.wrist_servo_gain = float(
            os.environ.get("FLIP_TABLE_CV_WRIST_SERVO_GAIN", str(self._WRIST_SERVO_GAIN))
        )
        self.wrist_servo_max_correction_m = float(
            os.environ.get(
                "FLIP_TABLE_CV_WRIST_SERVO_MAX_CORRECTION_M",
                str(self._WRIST_SERVO_MAX_CORRECTION_M),
            )
        )
        self.first_roll_wrist_servo_gain = float(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_WRIST_SERVO_GAIN",
                str(self._FIRST_ROLL_WRIST_SERVO_GAIN),
            )
        )
        self.first_roll_wrist_servo_max_correction_m = float(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_WRIST_SERVO_MAX_CORRECTION_M",
                str(self._FIRST_ROLL_WRIST_SERVO_MAX_CORRECTION_M),
            )
        )
        self.first_roll_recovery_servo_gain = float(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_RECOVERY_SERVO_GAIN",
                str(self._FIRST_ROLL_RECOVERY_SERVO_GAIN),
            )
        )
        self.first_roll_recovery_servo_max_correction_m = float(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_RECOVERY_SERVO_MAX_CORRECTION_M",
                str(self._FIRST_ROLL_RECOVERY_SERVO_MAX_CORRECTION_M),
            )
        )
        self.first_roll_max_tracking_error_m = float(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_MAX_TRACKING_ERROR_M",
                str(self._FIRST_ROLL_MAX_TRACKING_ERROR_M),
            )
        )
        self.first_roll_max_stall_steps = int(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_MAX_STALL_STEPS",
                str(self._FIRST_ROLL_MAX_STALL_STEPS),
            )
        )
        self.wrist_visual_servo_m_per_px = float(
            os.environ.get(
                "FLIP_TABLE_CV_WRIST_VISUAL_SERVO_M_PER_PX",
                str(self._WRIST_VISUAL_SERVO_M_PER_PX),
            )
        )
        self.wrist_visual_servo_max_m = float(
            os.environ.get(
                "FLIP_TABLE_CV_WRIST_VISUAL_SERVO_MAX_M",
                str(self._WRIST_VISUAL_SERVO_MAX_M),
            )
        )
        self.wrist_shaft_min_confidence = float(
            os.environ.get(
                "FLIP_TABLE_CV_WRIST_SHAFT_MIN_CONFIDENCE",
                str(self._WRIST_SHAFT_MIN_CONFIDENCE),
            )
        )
        self.wrist_shaft_max_center_error_px = float(
            os.environ.get(
                "FLIP_TABLE_CV_WRIST_SHAFT_MAX_CENTER_ERROR_PX",
                str(self._WRIST_SHAFT_MAX_CENTER_ERROR_PX),
            )
        )
        self.wrist_servo_integral_gain = float(
            os.environ.get("FLIP_TABLE_CV_WRIST_SERVO_INTEGRAL_GAIN", "0.0")
        )
        self.wrist_servo_integral_max_step_m = float(
            os.environ.get("FLIP_TABLE_CV_WRIST_SERVO_INTEGRAL_MAX_STEP_M", "0.004")
        )
        self.wrist_servo_integral_max_norm_m = float(
            os.environ.get("FLIP_TABLE_CV_WRIST_SERVO_INTEGRAL_MAX_NORM_M", "0.02")
        )
        self.dex1_grasp_block_threshold_rad = float(
            os.environ.get(
                "FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD",
                str(self._DEX1_GRASP_BLOCK_THRESHOLD_RAD),
            )
        )
        self.grasp_loss_limit_steps = int(
            os.environ.get(
                "FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS",
                str(self._GRASP_LOSS_LIMIT_STEPS),
            )
        )
        self.grasp_gate_stable_steps = int(
            os.environ.get(
                "FLIP_TABLE_CV_GRASP_GATE_STABLE_STEPS",
                str(self._GRASP_GATE_STABLE_STEPS),
            )
        )
        self.grasp_retry_attempts = int(
            os.environ.get(
                "FLIP_TABLE_CV_GRASP_RETRY_ATTEMPTS",
                str(len(GRASP_RETRY_OFFSETS_TOOL_M)),
            )
        )
        self.first_roll_advance_interval = int(
            os.environ.get(
                "FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL",
                str(self._FIRST_ROLL_ADVANCE_INTERVAL),
            )
        )
        self.max_cartesian_linear_speed_m_s = float(
            os.environ.get(
                "FLIP_TABLE_CV_MAX_CARTESIAN_LINEAR_SPEED_M_S",
                str(self._MAX_CARTESIAN_LINEAR_SPEED_M_S),
            )
        )
        self.max_cartesian_angular_speed_rad_s = float(
            os.environ.get(
                "FLIP_TABLE_CV_MAX_CARTESIAN_ANGULAR_SPEED_RAD_S",
                str(self._MAX_CARTESIAN_ANGULAR_SPEED_RAD_S),
            )
        )
        self.max_dex1_command_speed_s = float(
            os.environ.get(
                "FLIP_TABLE_CV_MAX_DEX1_COMMAND_SPEED_S",
                str(self._MAX_DEX1_COMMAND_SPEED_S),
            )
        )
        if (
            self.sim_control_hz <= 0.0
            or not 0.0 <= self.minimum_confidence <= 1.0
            or not 0.0 <= self.minimum_leg_confidence <= 1.0
            or self.redetect_interval_steps < 1
            or not 0.0 < self.redetect_alpha <= 1.0
            or self.redetect_max_translation_m <= 0.0
            or self.redetect_max_yaw_rad <= 0.0
            or self.redetect_max_center_drift_m <= 0.0
            or self.redetect_max_attachment_drift_m <= 0.0
            or not 0.0 <= self.wrist_servo_gain <= 2.0
            or self.wrist_servo_max_correction_m <= 0.0
            or not 0.0 <= self.first_roll_wrist_servo_gain <= 2.0
            or self.first_roll_wrist_servo_max_correction_m <= 0.0
            or not 0.0 <= self.first_roll_recovery_servo_gain <= 2.0
            or self.first_roll_recovery_servo_max_correction_m <= 0.0
            or self.first_roll_max_tracking_error_m <= 0.0
            or self.first_roll_max_stall_steps < 1
            or self.wrist_visual_servo_m_per_px <= 0.0
            or self.wrist_visual_servo_max_m <= 0.0
            or not 0.0 <= self.wrist_shaft_min_confidence <= 1.0
            or self.wrist_shaft_max_center_error_px <= 0.0
            or self.wrist_servo_integral_gain < 0.0
            or self.wrist_servo_integral_max_step_m <= 0.0
            or self.wrist_servo_integral_max_norm_m <= 0.0
            or not -0.02 < self.dex1_grasp_block_threshold_rad < 0.0245
            or self.grasp_loss_limit_steps < 1
            or self.grasp_gate_stable_steps < 1
            or not 1 <= self.grasp_retry_attempts <= len(GRASP_RETRY_OFFSETS_TOOL_M)
            or self.first_roll_advance_interval < 1
            or self.max_cartesian_linear_speed_m_s <= 0.0
            or self.max_cartesian_angular_speed_rad_s <= 0.0
            or self.max_dex1_command_speed_s <= 0.0
        ):
            raise ValueError("invalid CV controller or redetection setting")
        if self.warmup_steps < 1 or not 1 <= self.settled_selection_steps <= self.warmup_steps:
            raise ValueError(
                "CV warmup must be positive and settled selection must fit inside warmup"
            )

    def _ensure_camera_fk_model(self) -> None:
        import pinocchio as pin

        robofinals_root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
        urdf_path = robofinals_root / self._PINOCCHIO_MODEL_RELATIVE
        if not urdf_path.exists():
            local_root = Path(__file__).resolve().parents[1]
            urdf_path = local_root / self._PINOCCHIO_MODEL_RELATIVE
        if not urdf_path.exists():
            raise FileNotFoundError(f"G1 camera FK URDF not found: {urdf_path}")
        wrapper = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path), package_dirs=[str(urdf_path.parent)]
        )
        model = wrapper.model
        observed_joint_names = self._G1_GRIPPER_33_JOINT_ORDER
        missing_joints = [name for name in observed_joint_names if not model.existJointName(name)]
        if missing_joints or not model.existFrame("torso_link"):
            raise RuntimeError(
                "G1 camera FK contract mismatch: "
                f"missing_joints={missing_joints}, torso_link={model.existFrame('torso_link')}"
            )
        self._camera_pin = pin
        self._camera_pin_model = model
        self._camera_pin_data = wrapper.data
        self._camera_pin_q = np.zeros(model.nq, dtype=np.float64)
        self._camera_pin_joint_indices = {
            name: int(model.joints[model.getJointId(name)].idx_q)
            for name in observed_joint_names
        }
        self._camera_pin_torso_frame_id = int(model.getFrameId("torso_link"))
        self._camera_pin_wrist_frame_ids = {
            side: int(model.getFrameId(f"{side}_wrist_yaw_link"))
            for side in ("left", "right")
        }
        finger_frame_names = {
            side: {
                finger: f"{side}_dex1_finger_link_{finger_index}"
                for finger_index, finger in enumerate(("finger_1", "finger_2"), start=1)
            }
            for side in ("left", "right")
        }
        missing_finger_frames = [
            name
            for names in finger_frame_names.values()
            for name in names.values()
            if not model.existFrame(name)
        ]
        if missing_finger_frames:
            raise RuntimeError(
                f"G1 camera FK contract is missing Dex1 frames: {missing_finger_frames}"
            )
        self._camera_pin_finger_frame_ids = {
            side: {
                finger: int(model.getFrameId(name))
                for finger, name in names.items()
            }
            for side, names in finger_frame_names.items()
        }

    def _update_head_camera_calibration(self, observation: dict[str, Any]) -> None:
        merged = self._merged_observation(observation)
        joint_pos = merged.get("joint_pos")
        if joint_pos is None:
            raise KeyError("CV policy requires measured joint_pos for head-camera FK")
        if torch.is_tensor(joint_pos):
            values = joint_pos.detach().cpu().numpy()
        else:
            values = np.asarray(joint_pos)
        values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        if values.shape != (1, len(self._G1_GRIPPER_33_JOINT_ORDER)):
            raise ValueError(f"CV joint_pos must be [1,33], got {values.shape}")
        self._camera_pin_q.fill(0.0)
        for obs_index, name in enumerate(self._G1_GRIPPER_33_JOINT_ORDER):
            self._camera_pin_q[self._camera_pin_joint_indices[name]] = values[0, obs_index]
        self._last_cv_joint_pos = values[0].copy()
        self._camera_pin.framesForwardKinematics(
            self._camera_pin_model, self._camera_pin_data, self._camera_pin_q
        )
        placement = self._camera_pin_data.oMf[self._camera_pin_torso_frame_id]
        root_from_torso = np.eye(4, dtype=np.float64)
        root_from_torso[:3, :3] = np.asarray(placement.rotation)
        root_from_torso[:3, 3] = np.asarray(placement.translation).reshape(3)
        camera_fk_mode = os.environ.get(
            "FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE", "pinocchio"
        ).strip().lower()
        if camera_fk_mode == "robofinals_v1_fixed":
            # This is only valid when all waist joints are fixed. It remains a
            # useful diagnostic mode for comparing the authored V1 rest pose.
            self.estimator.calibration = CameraCalibration.g1_head_left()
        elif camera_fk_mode == "pinocchio":
            # The official V1 rest pose matches this URDF's torso transform.
            # Recompute it from encoder positions because domain randomization
            # and real operation can both move the waist joints.
            self.estimator.calibration = CameraCalibration.g1_head_left_from_torso(
                root_from_torso
            )
        else:
            raise ValueError(
                "FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE must be "
                "robofinals_v1_fixed or pinocchio"
            )

    def _measured_wrist_positions(self, observation: dict[str, Any]) -> np.ndarray:
        """Return encoder-FK wrist origins in the pelvis frame."""

        self._update_head_camera_calibration(observation)
        return np.stack(
            [
                np.asarray(
                    self._camera_pin_data.oMf[self._camera_pin_wrist_frame_ids[side]].translation
                ).reshape(3)
                for side in ("left", "right")
            ]
        )

    def _measured_grasp_centers(self) -> np.ndarray:
        """Return Dex1 collision-center midpoints from encoder FK."""

        centers = []
        for side in ("left", "right"):
            finger_centers = []
            for finger, frame_id in self._camera_pin_finger_frame_ids[side].items():
                placement = self._camera_pin_data.oMf[frame_id]
                finger_centers.append(
                    np.asarray(placement.translation).reshape(3)
                    + np.asarray(placement.rotation)
                    @ self._DEX1_COLLISION_CENTER_OFFSETS[finger]
                )
            centers.append(np.mean(finger_centers, axis=0))
        return np.stack(centers)

    def _dex1_grasp_observation(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, tuple[bool, bool]]:
        """Infer an object between the fingers from real-compatible encoders.

        A fully closed, unobstructed Dex1 converges to about -0.02 rad on both
        finger joints. A valid cylindrical-leg enclosure blocks both joints
        above that limit; one blocked finger alone can be an exterior contact.
        This deliberately uses measured joints, not simulator contact reports.
        """

        merged = self._merged_observation(observation)
        joint_pos = merged.get("joint_pos")
        if joint_pos is None:
            raise KeyError("CV grasp verification requires measured joint_pos")
        values = (
            joint_pos.detach().cpu().numpy()
            if torch.is_tensor(joint_pos)
            else np.asarray(joint_pos)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        if values.shape != (1, len(self._G1_GRIPPER_33_JOINT_ORDER)):
            raise ValueError(f"CV joint_pos must be [1,33], got {values.shape}")
        fingers = values[0, 29:33].copy().reshape(2, 2)
        blocked = dex1_enclosure_from_joint_positions(
            fingers,
            self.dex1_grasp_block_threshold_rad,
        )
        return fingers, blocked

    @staticmethod
    def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1.0e-9:
            raise ValueError("wrist quaternion must be non-zero")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    @classmethod
    def _desired_grasp_centers(cls, nominal_action: np.ndarray) -> np.ndarray:
        action = np.asarray(nominal_action, dtype=np.float64)
        if action.shape != (16,):
            raise ValueError("grasp-center target requires a 16-D action")
        centers = []
        for position_slice, quaternion_slice in ((slice(0, 3), slice(3, 7)), (slice(7, 10), slice(10, 14))):
            rotation = cls._rotation_from_wxyz(action[quaternion_slice])
            centers.append(
                action[position_slice]
                + rotation[:, 0] * GeometricFlipPlanner.WRIST_TO_GRASP_M
            )
        return np.stack(centers)

    @classmethod
    def _right_handover_wrist_target(
        cls,
        left_grasp_center: np.ndarray,
        aligned_short_axis: np.ndarray,
        right_tool_rotation: np.ndarray,
        *,
        pregrasp: bool,
    ) -> np.ndarray:
        """Place the right gripper at the tabletop edge currently held by the left hand."""

        left_center = np.asarray(left_grasp_center, dtype=np.float64).reshape(3)
        short_axis = np.asarray(aligned_short_axis, dtype=np.float64).reshape(3)
        short_axis /= np.linalg.norm(short_axis)
        tool = np.asarray(right_tool_rotation, dtype=np.float64).reshape(3, 3)
        # The left gripper holds a leg above the tabletop edge. Move toward the
        # table center and down to the adjacent edge, then stage 80 mm back
        # along tool +X before the final approach. All terms come from encoder
        # FK, initial RGB table axes, and known table/hand geometry.
        grasp_center = left_center + short_axis * 0.07
        grasp_center[2] -= 0.12
        if pregrasp:
            grasp_center -= tool[:, 0] * 0.08
        return grasp_center - tool[:, 0] * GeometricFlipPlanner.WRIST_TO_GRASP_M

    @staticmethod
    def _servo_wrist_positions(
        nominal_action: np.ndarray,
        measured_positions: np.ndarray,
        gain: float,
        max_correction_m: float,
        *,
        integral_offsets: np.ndarray | None = None,
        measured_grasp_centers: np.ndarray | None = None,
        grasp_center_mask: tuple[bool, bool] = (False, False),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(nominal_action, dtype=np.float32).copy()
        measured = np.asarray(measured_positions, dtype=np.float64)
        if action.shape != (16,) or measured.shape != (2, 3):
            raise ValueError("wrist servo requires a 16-D action and two measured 3-D wrists")
        desired = np.stack((action[0:3], action[7:10])).astype(np.float64)
        feedback = measured.copy()
        if any(grasp_center_mask):
            grasp_centers = np.asarray(measured_grasp_centers, dtype=np.float64)
            if grasp_centers.shape != (2, 3):
                raise ValueError("grasp-center servo requires two measured 3-D centers")
            desired_grasps = CvRuleBasedPolicy._desired_grasp_centers(action)
            for side_index, enabled in enumerate(grasp_center_mask):
                if enabled:
                    desired[side_index] = desired_grasps[side_index]
                    feedback[side_index] = grasp_centers[side_index]
        tracking_error = desired - feedback
        integral = (
            np.zeros((2, 3), dtype=np.float64)
            if integral_offsets is None
            else np.asarray(integral_offsets, dtype=np.float64)
        )
        if integral.shape != (2, 3):
            raise ValueError("integral wrist offsets must contain two 3-D vectors")
        correction = integral + gain * tracking_error
        norms = np.linalg.norm(correction, axis=1)
        for side_index, norm in enumerate(norms):
            if norm > max_correction_m:
                correction[side_index] *= max_correction_m / norm
        action[0:3] += correction[0].astype(np.float32)
        action[7:10] += correction[1].astype(np.float32)
        return action, correction, tracking_error

    @staticmethod
    def _merged_observation(observation: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for group_name in ("policy", "embodiment_general_obs"):
            group = observation.get(group_name, {})
            if isinstance(group, dict):
                merged.update(group)
        return merged

    def _head_left_rgb(self, observation: dict[str, Any]) -> np.ndarray:
        merged = self._merged_observation(observation)
        key = _resolve_camera_rgb_key(merged, "first_person_camera")
        return _camera_image_uint8(merged[key])

    def _wrist_rgb(self, observation: dict[str, Any], side: str) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported wrist camera side: {side}")
        merged = self._merged_observation(observation)
        key = _resolve_camera_rgb_key(merged, f"{side}_hand_camera")
        return _camera_image_uint8(merged[key])

    def _localize_table_frame(
        self,
        rgb: np.ndarray,
        previous_legs: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        estimate = self.estimator.estimate(rgb)
        if estimate.confidence < self.minimum_confidence:
            raise ValueError(
                f"confidence {estimate.confidence:.3f} < {self.minimum_confidence:.3f}"
            )
        tabletop_z_m = float(estimate.center_root_m[2])
        if not -0.45 <= tabletop_z_m <= 0.20:
            raise ValueError(
                "RGB tabletop PnP height is outside the reachable root-frame range: "
                f"z={tabletop_z_m:.3f} m"
            )
        legs = self.leg_detector.detect(
            rgb,
            estimate,
            self.estimator.calibration,
            tabletop_z_m,
            previous=previous_legs,
        )
        leg_confidences = {side: legs[side].confidence for side in ("left", "right")}
        if any(
            confidence < self.minimum_leg_confidence
            for confidence in leg_confidences.values()
        ):
            raise ValueError(
                "leg confidence below threshold: "
                f"detected={leg_confidences}, minimum={self.minimum_leg_confidence:.3f}"
            )
        frame = self.leg_detector.estimate_table_frame(
            estimate,
            legs,
            self.estimator.calibration,
            tabletop_z_m=tabletop_z_m,
        )
        if not (0.35 <= frame[0, 3] <= 0.95 and abs(frame[1, 3]) <= 0.35):
            raise ValueError(
                "RGB leg-derived table frame is outside the manipulation workspace: "
                f"center={frame[:3, 3].tolist()}"
            )
        leg_attachment_points = self.leg_detector.detect_near_leg_attachment_points(
            rgb,
            estimate,
            self.estimator.calibration,
            frame,
            tabletop_z_m=tabletop_z_m,
        )
        return estimate, legs, frame, leg_attachment_points

    def _save_cv_diagnostic(
        self,
        usr_args: dict[str, Any],
        rgb: np.ndarray,
        estimate: Any | None,
        legs: dict[str, Any] | None = None,
        error: str = "",
        control_root_from_table: np.ndarray | None = None,
        leg_attachment_points: dict[str, np.ndarray] | None = None,
        label: str = "initial",
    ) -> None:
        output = usr_args.get("save_path")
        if not output:
            return
        root = Path(str(output)) / "cv_rule_based"
        root.mkdir(parents=True, exist_ok=True)
        # Preserve the exact, unannotated policy image beside every overlay.
        # It makes RGB-only localization regressions reviewable without adding
        # simulator state to the policy or to its recorded observation schema.
        media.write_image(root / f"{label}_head_left_rgb.png", rgb)
        debug = self.estimator.render_debug(rgb, estimate, error)
        if estimate is not None and legs:
            debug = self.leg_detector.render_debug(rgb, estimate, legs)
        media.write_image(root / f"{label}_tabletop_estimate.png", debug)
        payload = {
            "error": error,
            "policy_inputs": [
                "head_left_rgb",
                "measured_joint_pos",
                "g1_urdf",
                "head_left_mount_from_joint_fk",
                "camera_intrinsics",
            ],
            "root_from_camera": self.estimator.calibration.root_from_camera.tolist(),
            "measured_joint_pos_order": list(self._G1_GRIPPER_33_JOINT_ORDER),
            "measured_joint_pos": getattr(
                self, "_last_cv_joint_pos", np.empty(0, dtype=np.float64)
            ).tolist(),
            "motion_source": "geometry_rules_only",
            "simulator_object_pose_used": False,
        }
        if estimate is not None:
            media.write_image(root / f"{label}_tabletop_mask.png", estimate.mask)
            payload.update({
                "confidence": estimate.confidence,
                "reprojection_error_px": estimate.reprojection_error_px,
                "area_fraction": estimate.area_fraction,
                "center_root_m": estimate.center_root_m.tolist(),
                "yaw_root_rad": estimate.yaw_root_rad,
                "root_from_table": estimate.root_from_table.tolist(),
                "control_root_from_table": (
                    control_root_from_table.tolist() if control_root_from_table is not None else None
                ),
                "detected_legs": {
                    side: {
                        "endpoints_px": detection.endpoints_px.tolist(),
                        "confidence": detection.confidence,
                        "tracked": detection.tracked,
                        "inferred_from_tabletop": detection.inferred_from_tabletop,
                        "cad_projected_axis": detection.cad_projected_axis,
                        "vertical_alignment": detection.vertical_alignment,
                        "attachment_root_m": (
                            None
                            if detection.attachment_root_m is None
                            else detection.attachment_root_m.tolist()
                        ),
                    }
                    for side, detection in (legs or {}).items()
                },
                "leg_attachment_points_root_m": {
                    side: point.tolist()
                    for side, point in (leg_attachment_points or {}).items()
                },
                "all_cad_leg_attachment_points_root_m": {
                    label: point.tolist()
                    for label, point in self.leg_detector.estimate_all_leg_centers(
                        control_root_from_table
                        if control_root_from_table is not None
                        else estimate.root_from_table
                    ).items()
                },
            })
        (root / f"{label}_tabletop_estimate.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        _synchronize_policy_control_rate(self, task_env, attribute="sim_control_hz")
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self.action_dim))
        if num_envs != 1:
            raise ValueError("CvRuleBasedPolicy requires num_envs=1")
        if action_dim != self._ACTION_DIM:
            raise ValueError(f"CvRuleBasedPolicy requires 16-D Pink+Dex1 actions, got {action_dim}")

        estimate = None
        legs = None
        control_root_from_table = None
        leg_attachment_points = None
        best_score = -math.inf
        best_rgb = None
        last_rgb = None
        errors: list[str] = []
        # Let reset physics and camera exposure settle. Every localization uses
        # only the real-compatible head-left stream, encoder FK, and calibration.
        for warmup_step in range(self.warmup_steps):
            last_rgb = self._head_left_rgb(observation)
            try:
                self._update_head_camera_calibration(observation)
                candidate, candidate_legs, candidate_frame, candidate_attachments = (
                    self._localize_table_frame(last_rgb)
                )
                score = candidate.confidence + 0.10 * sum(
                    detection.confidence for detection in candidate_legs.values()
                )
                settled_selection_start = self.warmup_steps - self.settled_selection_steps
                if warmup_step >= settled_selection_start and score > best_score:
                    estimate = candidate
                    legs = candidate_legs
                    control_root_from_table = candidate_frame
                    leg_attachment_points = candidate_attachments
                    best_score = score
                    best_rgb = last_rgb.copy()
            except ValueError as exc:
                errors.append(str(exc))
            hold = GeometricFlipPlanner.neutral_action()
            observation, _, terminated, _, _ = task_env.step(
                torch.as_tensor(hold, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                break
        assert last_rgb is not None
        if (
            estimate is None
            or legs is None
            or control_root_from_table is None
            or leg_attachment_points is None
        ):
            reason = errors[-1] if errors else "tabletop and both front legs were not detected"
            self._save_cv_diagnostic(usr_args, last_rgb, None, None, reason)
            print(f"[CvRuleBasedPolicy] episode rejected by CV: {reason}", flush=True)
            return np.zeros(1, dtype=bool)
        assert best_rgb is not None
        planner = GeometricFlipPlanner(
            control_root_from_table,
            self.sim_control_hz,
            leg_attachment_points,
        )
        initial_alignment_mode = planner.alignment_mode
        initial_alignment_pivot_side = planner.alignment_pivot_side
        initial_alignment_moving_side = planner.alignment_moving_side
        self._save_cv_diagnostic(
            usr_args,
            best_rgb,
            estimate,
            legs,
            control_root_from_table=control_root_from_table,
            leg_attachment_points=leg_attachment_points,
        )

        ever_success = np.zeros(1, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        previous_phase: Phase | None = None
        redetection_attempts = 0
        redetection_updates = 0
        right_handover_target: np.ndarray | None = None
        right_push_offset: np.ndarray | None = None
        latest_detection_rgb = best_rgb
        wrist_servo_integral_offsets = np.zeros((2, 3), dtype=np.float64)
        settled_control_root_from_table = control_root_from_table.copy()
        settled_leg_attachment_points = {
            side: point.copy() for side, point in leg_attachment_points.items()
        }
        rollout_limit = max(0, int(usr_args["time_out_limit"]) - self.warmup_steps)
        trajectory_step = 0
        left_grasp_loss_streak = 0
        right_grasp_loss_streak = 0
        first_roll_stall_steps = 0
        first_roll_error_baseline: np.ndarray | None = None
        grasp_gate_passed = False
        grasp_gate_streak = 0
        grasp_retry_index = 0
        grasp_retry_step: int | None = None
        alignment_grasp_gate_passed = False
        alignment_pivot_grasp_passed = False
        alignment_pivot_loss_streak = 0
        alignment_grasp_gate_streak = 0
        alignment_grasp_retry_index = 0
        alignment_grasp_retry_step: int | None = None
        grasp_retry_steps = grasp_retry_total_steps(self.sim_control_hz)
        grasp_visual_retry_index = -1
        grasp_visual_gate = ""
        grasp_visual_offset = np.zeros(3, dtype=np.float64)
        grasp_visual_depth_offset_m = 0.0
        grasp_visual_ready_streak = 0
        grasp_close_authorized = False
        accepted_alignment_grasp_offsets_tool_m = np.zeros((2, 3), dtype=np.float64)
        accepted_left_grasp_offset_tool_m = np.zeros(3, dtype=np.float64)
        post_alignment_relocalized = False
        post_alignment_redetection_status = "not_due"
        last_grasp_gate_rgb: np.ndarray | None = None
        last_wrist_alignment_rgb: np.ndarray | None = None
        previous_limited_action = GeometricFlipPlanner.neutral_action()
        abort_reason: str | None = None
        _write_action_state_trace(self, usr_args, [])
        for control_step in range(rollout_limit):
            if trajectory_step >= planner.total_steps:
                break
            phase = planner.phase_at(trajectory_step)
            if phase != previous_phase:
                print(
                    f"[CvRuleBasedPolicy] phase={phase.value} step={control_step} "
                    f"cv_confidence={estimate.confidence:.3f}",
                    flush=True,
                )
                previous_phase = phase
            _, grasp_blocked_before = self._dex1_grasp_observation(observation)
            alignment_start = planner.phase_start_step(Phase.ALIGN_SHORT_EDGE)
            alignment_grasp_gate_active = (
                phase is Phase.ALIGN_SHORT_EDGE
                and trajectory_step == alignment_start
                and not alignment_grasp_gate_passed
            )
            if phase is Phase.LEFT_LEG_FLIP_90 and not post_alignment_relocalized:
                try:
                    self._update_head_camera_calibration(observation)
                    post_rgb = self._head_left_rgb(observation)
                    (
                        post_estimate,
                        post_legs,
                        post_frame,
                        post_attachments,
                    ) = self._localize_table_frame(post_rgb)
                    post_attachments = (
                        self.leg_detector.select_arm_reachable_leg_centers(post_frame)
                    )
                    planner = GeometricFlipPlanner(
                        post_frame,
                        self.sim_control_hz,
                        post_attachments,
                        table_is_aligned=True,
                    )
                    trajectory_step = planner.phase_start_step(
                        Phase.LEFT_LEG_FLIP_90
                    )
                    estimate = post_estimate
                    legs = post_legs
                    control_root_from_table = post_frame
                    leg_attachment_points = post_attachments
                    post_alignment_relocalized = True
                    post_alignment_redetection_status = "accepted"
                    self._save_cv_diagnostic(
                        usr_args,
                        post_rgb,
                        post_estimate,
                        post_legs,
                        control_root_from_table=post_frame,
                        leg_attachment_points=post_attachments,
                        label="post_alignment",
                    )
                    print(
                        "[CvRuleBasedPolicy] post-alignment RGB relocalization "
                        f"accepted center={post_frame[:2, 3].round(3).tolist()}",
                        flush=True,
                    )
                except ValueError as exc:
                    post_alignment_redetection_status = f"rejected: {exc}"
                    print(
                        "[CvRuleBasedPolicy] post-alignment RGB relocalization "
                        f"rejected: {exc}",
                        flush=True,
                    )
            first_roll_start = planner.phase_start_step(Phase.LEFT_LEG_FLIP_90)
            grasp_gate_active = (
                phase is Phase.LEFT_LEG_FLIP_90
                and trajectory_step == first_roll_start
                and not grasp_gate_passed
            )
            first_roll_active = phase is Phase.LEFT_LEG_FLIP_90 and grasp_gate_passed
            grasp_retry_stage = "inactive"
            grasp_gate_verification_ready = False
            grasp_command_offset_tool_m = np.zeros(3, dtype=np.float64)
            active_grasp_rotation = np.eye(3, dtype=np.float64)
            active_grasp_retry_index = grasp_retry_index
            alignment_grasp_role = "inactive"
            grasp_camera_side = "left"
            grasp_requires_wrist_shaft = False
            grasp_requires_wrist_edge = False
            active_grasp_side_index = 0
            active_grasp_hand_index = 14
            if alignment_grasp_gate_active:
                alignment_action = planner.action_at(alignment_start - 1)
                if alignment_pivot_grasp_passed:
                    alignment_grasp_role = "tabletop_edge"
                    grasp_camera_side = initial_alignment_moving_side
                    grasp_requires_wrist_edge = True
                    pivot_index = 0 if initial_alignment_pivot_side == "left" else 1
                    alignment_action = apply_tool_position_offset(
                        alignment_action,
                        accepted_alignment_grasp_offsets_tool_m[pivot_index],
                        initial_alignment_pivot_side,
                    )
                    alignment_action[14 + pivot_index] = 1.0
                else:
                    alignment_grasp_role = "pivot_leg"
                    grasp_camera_side = initial_alignment_pivot_side
                    moving_index = 0 if initial_alignment_moving_side == "left" else 1
                    alignment_action[14 + moving_index] = -1.0
                    grasp_requires_wrist_shaft = True
                active_grasp_side_index = 0 if grasp_camera_side == "left" else 1
                active_grasp_hand_index = 14 + active_grasp_side_index
                active_grasp_retry_index = alignment_grasp_retry_index
                if alignment_grasp_retry_step is None:
                    alignment_grasp_retry_step = 0
                nominal_action_np, grasp_retry_stage, grasp_gate_verification_ready = (
                    grasp_retry_action(
                        alignment_action,
                        GRASP_RETRY_OFFSETS_TOOL_M[alignment_grasp_retry_index],
                        alignment_grasp_retry_step,
                        self.sim_control_hz,
                        side=grasp_camera_side,
                    )
                )
                position_slice = (
                    slice(0, 3) if grasp_camera_side == "left" else slice(7, 10)
                )
                quaternion_slice = (
                    slice(3, 7) if grasp_camera_side == "left" else slice(10, 14)
                )
                alignment_rotation = self._rotation_from_wxyz(
                    alignment_action[quaternion_slice]
                )
                active_grasp_rotation = alignment_rotation
                grasp_command_offset_tool_m = alignment_rotation.T @ (
                    nominal_action_np[position_slice] - alignment_action[position_slice]
                )
            elif grasp_gate_active:
                alignment_grasp_role = "first_roll_leg"
                grasp_requires_wrist_shaft = True
                aligned_action = planner.action_at(first_roll_start - 1)
                if grasp_retry_step is None:
                    grasp_retry_step = 0
                nominal_action_np, grasp_retry_stage, grasp_gate_verification_ready = (
                    grasp_retry_action(
                        aligned_action,
                        GRASP_RETRY_OFFSETS_TOOL_M[grasp_retry_index],
                        grasp_retry_step,
                        self.sim_control_hz,
                    )
                )
                aligned_rotation = self._rotation_from_wxyz(aligned_action[3:7])
                active_grasp_rotation = aligned_rotation
                grasp_command_offset_tool_m = aligned_rotation.T @ (
                    nominal_action_np[0:3] - aligned_action[0:3]
                )
            else:
                nominal_action_np = planner.action_at(trajectory_step)
                if (
                    alignment_grasp_gate_passed
                    and phase is Phase.ALIGN_SHORT_EDGE
                ):
                    for side_index, side in enumerate(("left", "right")):
                        if float(nominal_action_np[14 + side_index]) > -0.50:
                            nominal_action_np = apply_tool_position_offset(
                                nominal_action_np,
                                accepted_alignment_grasp_offsets_tool_m[side_index],
                                side,
                            )
                if (
                    grasp_gate_passed
                    and phase in (
                        Phase.LEFT_LEG_FLIP_90,
                        Phase.RIGHT_PREGRASP,
                        Phase.RIGHT_GRASP,
                        Phase.HANDOVER,
                    )
                    and float(nominal_action_np[14]) > -0.50
                ):
                    nominal_action_np = apply_tool_position_offset(
                        nominal_action_np,
                        accepted_left_grasp_offset_tool_m,
                        "left",
                    )
            measured_wrist_positions = self._measured_wrist_positions(observation)
            measured_grasp_centers = self._measured_grasp_centers()
            grasp_verification_active = alignment_grasp_gate_active or grasp_gate_active
            grasp_wrist_rgb_before = (
                self._wrist_rgb(observation, grasp_camera_side)
                if (
                    grasp_requires_wrist_shaft
                    or grasp_requires_wrist_edge
                    or first_roll_active
                )
                else None
            )
            grasp_wrist_shaft_before = (
                self.wrist_shaft_detector.detect(
                    grasp_wrist_rgb_before,
                    require_centered=False,
                )
                if grasp_wrist_rgb_before is not None
                else None
            )
            grasp_wrist_edge_before = (
                self.wrist_tabletop_edge_detector.detect(grasp_wrist_rgb_before)
                if grasp_wrist_rgb_before is not None and grasp_requires_wrist_edge
                else None
            )
            current_grasp_visual_gate = ":".join(
                (alignment_grasp_role, grasp_camera_side)
            )
            if grasp_verification_active and (
                grasp_visual_gate != current_grasp_visual_gate
                or grasp_visual_retry_index != active_grasp_retry_index
            ):
                grasp_visual_gate = current_grasp_visual_gate
                grasp_visual_retry_index = active_grasp_retry_index
                grasp_visual_offset.fill(0.0)
                grasp_visual_depth_offset_m = 0.0
                grasp_visual_ready_streak = 0
                grasp_close_authorized = False
                last_grasp_gate_rgb = None
                last_wrist_alignment_rgb = None
            wrist_center_frame_fresh = bool(
                grasp_verification_active
                and grasp_wrist_rgb_before is not None
                and (
                    last_wrist_alignment_rgb is None
                    or not np.array_equal(
                        grasp_wrist_rgb_before, last_wrist_alignment_rgb
                    )
                )
            )
            if wrist_center_frame_fresh:
                last_wrist_alignment_rgb = grasp_wrist_rgb_before.copy()
            wrist_shaft_centered_before = bool(
                grasp_wrist_shaft_before is not None
                and grasp_wrist_shaft_before.detected
                and grasp_wrist_shaft_before.confidence
                >= self.wrist_shaft_min_confidence
                and grasp_wrist_shaft_before.center_px is not None
                and abs(grasp_wrist_shaft_before.center_px[0] - 320.0)
                <= self.wrist_shaft_max_center_error_px
            )
            wrist_shaft_bottom_px = (
                None
                if grasp_wrist_shaft_before is None
                or grasp_wrist_shaft_before.bounding_box_px is None
                else float(
                    grasp_wrist_shaft_before.bounding_box_px[1]
                    + grasp_wrist_shaft_before.bounding_box_px[3]
                )
            )
            wrist_shaft_deep_before = bool(
                wrist_shaft_bottom_px is not None
                and grasp_wrist_rgb_before is not None
                and wrist_shaft_bottom_px
                >= self._WRIST_SHAFT_MIN_BOTTOM_FRACTION
                * grasp_wrist_rgb_before.shape[0]
            )
            wrist_edge_y_px = (
                None
                if grasp_wrist_edge_before is None
                else grasp_wrist_edge_before.edge_y_px
            )
            wrist_edge_deep_before = bool(
                grasp_wrist_edge_before is not None
                and grasp_wrist_edge_before.detected
                and grasp_wrist_edge_before.confidence >= 0.65
                and wrist_edge_y_px is not None
                and grasp_wrist_rgb_before is not None
                and wrist_edge_y_px
                >= self._WRIST_EDGE_MIN_BOTTOM_FRACTION
                * grasp_wrist_rgb_before.shape[0]
            )
            if (
                (grasp_requires_wrist_shaft or grasp_requires_wrist_edge)
                and wrist_center_frame_fresh
                and (
                    grasp_retry_stage in {"open_backoff", "approach"}
                    or not grasp_close_authorized
                )
            ):
                grasp_target_ready = (
                    wrist_shaft_centered_before and wrist_shaft_deep_before
                    if grasp_requires_wrist_shaft
                    else wrist_edge_deep_before
                )
                if grasp_target_ready:
                    grasp_visual_ready_streak += 1
                else:
                    grasp_visual_ready_streak = 0
                grasp_close_authorized = (
                    grasp_visual_ready_streak >= self.grasp_gate_stable_steps
                )
            if (
                grasp_wrist_shaft_before is not None
                and grasp_wrist_shaft_before.detected
                and grasp_wrist_shaft_before.center_px is not None
                and wrist_center_frame_fresh
            ):
                image_error_px = grasp_wrist_shaft_before.center_px[0] - 320.0
                correction_m = float(
                    np.clip(
                        image_error_px * self.wrist_visual_servo_m_per_px,
                        -self._WRIST_VISUAL_SERVO_MAX_STEP_M,
                        self._WRIST_VISUAL_SERVO_MAX_STEP_M,
                    )
                )
                root_from_wrist = np.asarray(
                    self._camera_pin_data.oMf[
                        self._camera_pin_wrist_frame_ids[grasp_camera_side]
                    ].rotation
                )
                correction_delta = (
                    root_from_wrist @ self._wrist_from_camera_rotation[:, 0]
                ) * correction_m
                grasp_visual_offset += correction_delta
                correction_norm = float(np.linalg.norm(grasp_visual_offset))
                if correction_norm > self.wrist_visual_servo_max_m:
                    grasp_visual_offset *= (
                        self.wrist_visual_servo_max_m / correction_norm
                    )
                assert wrist_shaft_bottom_px is not None
                target_bottom_px = (
                    self._WRIST_SHAFT_MIN_BOTTOM_FRACTION
                    * grasp_wrist_rgb_before.shape[0]
                )
                depth_delta_m = float(
                    np.clip(
                        (target_bottom_px - wrist_shaft_bottom_px)
                        * self._WRIST_VISUAL_DEPTH_GAIN_M_PER_PX,
                        0.0,
                        self._WRIST_VISUAL_DEPTH_MAX_STEP_M,
                    )
                )
                grasp_visual_depth_offset_m = float(
                    np.clip(
                        grasp_visual_depth_offset_m + depth_delta_m,
                        0.0,
                        self._WRIST_VISUAL_DEPTH_MAX_M,
                    )
                )
            if (
                grasp_wrist_edge_before is not None
                and grasp_wrist_edge_before.detected
                and wrist_edge_y_px is not None
                and grasp_wrist_rgb_before is not None
                and wrist_center_frame_fresh
            ):
                target_edge_y_px = (
                    self._WRIST_EDGE_MIN_BOTTOM_FRACTION
                    * grasp_wrist_rgb_before.shape[0]
                )
                edge_depth_delta_m = float(
                    np.clip(
                        (target_edge_y_px - wrist_edge_y_px)
                        * self._WRIST_EDGE_VISUAL_DEPTH_GAIN_M_PER_PX,
                        0.0,
                        self._WRIST_EDGE_VISUAL_DEPTH_MAX_STEP_M,
                    )
                )
                grasp_visual_depth_offset_m = float(
                    np.clip(
                        grasp_visual_depth_offset_m + edge_depth_delta_m,
                        0.0,
                        self._WRIST_EDGE_VISUAL_DEPTH_MAX_M,
                    )
                )
            wrist_visual_correction = np.zeros(3, dtype=np.float64)
            if (
                grasp_requires_wrist_shaft or grasp_requires_wrist_edge
            ) and grasp_retry_stage in {
                "approach",
                "close",
                "verify",
            }:
                wrist_visual_correction = (
                    grasp_visual_offset
                    + active_grasp_rotation[:, 0] * grasp_visual_depth_offset_m
                )
                position_slice = (
                    slice(0, 3) if grasp_camera_side == "left" else slice(7, 10)
                )
                nominal_action_np[position_slice] += wrist_visual_correction.astype(
                    np.float32
                )
                grasp_command_offset_tool_m += (
                    active_grasp_rotation.T @ wrist_visual_correction
                )
            if (
                (grasp_requires_wrist_shaft or grasp_requires_wrist_edge)
                and grasp_retry_stage in {"close", "verify"}
                and not grasp_close_authorized
            ):
                nominal_action_np[active_grasp_hand_index] = -1.0
                grasp_gate_verification_ready = False
            update_handover_target = phase in (
                Phase.RIGHT_PREGRASP,
                Phase.RIGHT_GRASP,
                Phase.HANDOVER,
            ) and (
                phase is not Phase.HANDOVER or float(nominal_action_np[14]) > -0.50
            )
            if update_handover_target:
                right_rotation = self._rotation_from_wxyz(nominal_action_np[10:14])
                target = self._right_handover_wrist_target(
                    measured_grasp_centers[0],
                    planner.aligned_short_axis,
                    right_rotation,
                    pregrasp=phase is Phase.RIGHT_PREGRASP,
                )
                if right_handover_target is None:
                    right_handover_target = target
                else:
                    right_handover_target = (
                        0.85 * right_handover_target + 0.15 * target
                    )
                nominal_action_np[7:10] = right_handover_target.astype(np.float32)
            elif right_handover_target is not None and phase is Phase.RIGHT_TOP_FLIP_90:
                if right_push_offset is None:
                    right_push_offset = right_handover_target - nominal_action_np[7:10]
                nominal_action_np[7:10] += right_push_offset.astype(np.float32)
            elif (
                right_push_offset is not None
                and phase is Phase.SETTLE_AND_RETREAT
                and float(nominal_action_np[15]) > -0.50
            ):
                nominal_action_np[7:10] += right_push_offset.astype(np.float32)
            if phase in (Phase.ALIGN_APPROACH, Phase.ALIGN_GRASP, Phase.RIGHT_GRASP):
                grasp_center_mask = (True, True)
            elif phase in (Phase.LEFT_LEG_FLIP_90, Phase.RIGHT_PREGRASP):
                grasp_center_mask = (True, phase is Phase.RIGHT_PREGRASP)
            elif phase in (Phase.ALIGN_SHORT_EDGE, Phase.HANDOVER):
                grasp_center_mask = (
                    float(nominal_action_np[14]) > -0.50,
                    float(nominal_action_np[15]) > -0.50,
                )
            elif phase is Phase.RIGHT_TOP_FLIP_90:
                grasp_center_mask = (False, True)
            else:
                grasp_center_mask = (False, False)
            first_roll_tracking = phase is Phase.LEFT_LEG_FLIP_90 and not grasp_gate_active
            first_roll_recovery = first_roll_tracking and first_roll_stall_steps >= 5
            if first_roll_recovery:
                servo_gain = self.first_roll_recovery_servo_gain
                servo_max_correction_m = self.first_roll_recovery_servo_max_correction_m
            elif first_roll_tracking:
                servo_gain = self.first_roll_wrist_servo_gain
                servo_max_correction_m = self.first_roll_wrist_servo_max_correction_m
            else:
                servo_gain = self.wrist_servo_gain
                servo_max_correction_m = self.wrist_servo_max_correction_m
            if phase is Phase.LEFT_LEG_FLIP_90:
                wrist_servo_integral_offsets[0] = 0.0
            action_np, wrist_servo_correction, wrist_servo_error = self._servo_wrist_positions(
                nominal_action_np,
                measured_wrist_positions,
                servo_gain,
                servo_max_correction_m,
                integral_offsets=wrist_servo_integral_offsets,
                measured_grasp_centers=measured_grasp_centers,
                grasp_center_mask=grasp_center_mask,
            )
            try:
                requested_action_np = validate_cartesian_action(action_np).copy()
                action_np = limit_cartesian_action_rate(
                    previous_limited_action,
                    requested_action_np,
                    self.sim_control_hz,
                    max_linear_speed_m_s=self.max_cartesian_linear_speed_m_s,
                    max_angular_speed_rad_s=self.max_cartesian_angular_speed_rad_s,
                    max_hand_speed_s=self.max_dex1_command_speed_s,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid CV controller action at step={control_step}, phase={phase.value}: {exc}"
                ) from exc
            previous_limited_action = action_np.copy()
            wrist_servo_integral_offsets = update_bounded_integral_offsets(
                wrist_servo_integral_offsets,
                wrist_servo_error,
                grasp_center_mask,
                gain=self.wrist_servo_integral_gain,
                max_step_m=self.wrist_servo_integral_max_step_m,
                max_norm_m=self.wrist_servo_integral_max_norm_m,
            )
            action = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, 1)
            dex1_finger_positions, encoder_grasp_blocked = self._dex1_grasp_observation(
                observation
            )
            grasp_wrist_rgb_after = (
                self._wrist_rgb(observation, grasp_camera_side)
                if grasp_verification_active or first_roll_active
                else None
            )
            grasp_wrist_shaft_after = (
                self.wrist_shaft_detector.detect(
                    grasp_wrist_rgb_after,
                    require_centered=False,
                )
                if grasp_wrist_rgb_after is not None
                and (grasp_requires_wrist_shaft or first_roll_active)
                else None
            )
            grasp_gate_frame_fresh = bool(
                grasp_verification_active
                and grasp_wrist_rgb_after is not None
                and (
                    last_grasp_gate_rgb is None
                    or not np.array_equal(grasp_wrist_rgb_after, last_grasp_gate_rgb)
                )
            )
            if grasp_gate_frame_fresh:
                last_grasp_gate_rgb = grasp_wrist_rgb_after.copy()
            wrist_shaft_gate_ok = bool(
                grasp_wrist_shaft_after is not None
                and grasp_wrist_shaft_after.detected
                and grasp_wrist_shaft_after.confidence
                >= self.wrist_shaft_min_confidence
                and grasp_wrist_shaft_after.center_px is not None
                and abs(grasp_wrist_shaft_after.center_px[0] - 320.0)
                <= self.wrist_shaft_max_center_error_px
            )
            first_roll_tracking_error_m = float(np.linalg.norm(wrist_servo_error[0]))
            if alignment_grasp_gate_active:
                pivot_index = 0 if initial_alignment_pivot_side == "left" else 1
                if alignment_grasp_role == "tabletop_edge":
                    alignment_pivot_loss_streak = (
                        0
                        if encoder_grasp_blocked[pivot_index]
                        else alignment_pivot_loss_streak + 1
                    )
                    if alignment_pivot_loss_streak >= self.grasp_loss_limit_steps:
                        alignment_pivot_grasp_passed = False
                        alignment_pivot_loss_streak = 0
                        alignment_grasp_retry_index = 0
                        alignment_grasp_retry_step = None
                        alignment_grasp_gate_streak = 0
                        accepted_alignment_grasp_offsets_tool_m[pivot_index] = 0.0
                        grasp_visual_retry_index = -1
                        grasp_visual_gate = ""
                        grasp_visual_offset.fill(0.0)
                        grasp_visual_depth_offset_m = 0.0
                        grasp_visual_ready_streak = 0
                        grasp_close_authorized = False
                        last_grasp_gate_rgb = None
                        last_wrist_alignment_rgb = None
                        print(
                            "[CvRuleBasedPolicy] alignment pivot grasp lost; "
                            f"returning to wrist-RGB reacquisition side="
                            f"{initial_alignment_pivot_side}",
                            flush=True,
                        )
                else:
                    alignment_pivot_loss_streak = 0
                if grasp_gate_frame_fresh:
                    visual_gate_ok = (
                        grasp_close_authorized
                        if alignment_grasp_role in {"pivot_leg", "tabletop_edge"}
                        else True
                    )
                    enclosure_gate_ok = encoder_grasp_blocked[
                        active_grasp_side_index
                    ]
                    if alignment_grasp_role == "tabletop_edge":
                        enclosure_gate_ok = bool(
                            enclosure_gate_ok
                            and encoder_grasp_blocked[pivot_index]
                            and alignment_pivot_grasp_passed
                        )
                    if (
                        grasp_gate_verification_ready
                        and visual_gate_ok
                        and enclosure_gate_ok
                    ):
                        alignment_grasp_gate_streak += 1
                    else:
                        alignment_grasp_gate_streak = 0
                if alignment_grasp_gate_streak >= self.grasp_gate_stable_steps:
                    accepted_alignment_grasp_offsets_tool_m[
                        active_grasp_side_index
                    ] = grasp_command_offset_tool_m.copy()
                    alignment_grasp_retry_step = None
                    last_grasp_gate_rgb = None
                    last_wrist_alignment_rgb = None
                    alignment_grasp_gate_streak = 0
                    alignment_grasp_retry_index = 0
                    grasp_visual_retry_index = -1
                    grasp_visual_gate = ""
                    grasp_visual_offset.fill(0.0)
                    grasp_visual_depth_offset_m = 0.0
                    grasp_visual_ready_streak = 0
                    grasp_close_authorized = False
                    if alignment_grasp_role == "pivot_leg":
                        alignment_pivot_grasp_passed = True
                        print(
                            "[CvRuleBasedPolicy] alignment pivot-leg wrist-RGB "
                            f"grasp passed side={grasp_camera_side}",
                            flush=True,
                        )
                    else:
                        alignment_grasp_gate_passed = True
                        print(
                            "[CvRuleBasedPolicy] alignment tabletop-edge grasp "
                            f"passed side={grasp_camera_side}",
                            flush=True,
                        )
                elif alignment_grasp_retry_step is None:
                    alignment_grasp_retry_step = 0
                else:
                    alignment_grasp_retry_step += 1
                    if alignment_grasp_retry_step >= grasp_retry_steps:
                        alignment_grasp_retry_index += 1
                        alignment_grasp_retry_step = 0
                        alignment_grasp_gate_streak = 0
                        grasp_visual_ready_streak = 0
                        grasp_close_authorized = False
                        grasp_visual_depth_offset_m = 0.0
                        last_wrist_alignment_rgb = None
                        if alignment_grasp_retry_index >= self.grasp_retry_attempts:
                            target_name = (
                                "pivot leg was not centered in the wrist camera and "
                                "captured"
                                if alignment_grasp_role == "pivot_leg"
                                else "tabletop edge was not captured"
                            )
                            abort_reason = (
                                f"alignment {target_name} by the {grasp_camera_side} "
                                "hand after all RGB-frame regrasp candidates"
                            )
                        else:
                            print(
                                "[CvRuleBasedPolicy] alignment grasp retry "
                                f"role={alignment_grasp_role} "
                                f"side={grasp_camera_side} "
                                f"attempt={alignment_grasp_retry_index} offset_tool_m="
                                f"{GRASP_RETRY_OFFSETS_TOOL_M[alignment_grasp_retry_index]}",
                                flush=True,
                            )
            elif grasp_gate_active:
                if grasp_gate_frame_fresh:
                    if (
                        grasp_gate_verification_ready
                        and grasp_close_authorized
                        and encoder_grasp_blocked[0]
                    ):
                        grasp_gate_streak += 1
                    else:
                        grasp_gate_streak = 0
                if grasp_gate_streak >= self.grasp_gate_stable_steps:
                    grasp_gate_passed = True
                    accepted_left_grasp_offset_tool_m = (
                        grasp_command_offset_tool_m.copy()
                    )
                    first_roll_error_baseline = wrist_servo_error[0].copy()
                    grasp_retry_step = None
                    grasp_visual_depth_offset_m = 0.0
                    print(
                        "[CvRuleBasedPolicy] left grasp gate passed "
                        f"attempt={grasp_retry_index} streak={grasp_gate_streak}",
                        flush=True,
                    )
                elif grasp_retry_step is None:
                    grasp_retry_step = 0
                else:
                    grasp_retry_step += 1
                    if grasp_retry_step >= grasp_retry_steps:
                        grasp_retry_index += 1
                        grasp_retry_step = 0
                        grasp_gate_streak = 0
                        grasp_visual_ready_streak = 0
                        grasp_close_authorized = False
                        grasp_visual_depth_offset_m = 0.0
                        last_wrist_alignment_rgb = None
                        if grasp_retry_index >= self.grasp_retry_attempts:
                            abort_reason = (
                                "left leg was not centered in the wrist camera and "
                                "captured after all RGB-frame regrasp candidates"
                            )
                        else:
                            print(
                                "[CvRuleBasedPolicy] left grasp retry "
                                f"attempt={grasp_retry_index} offset_tool_m="
                                f"{GRASP_RETRY_OFFSETS_TOOL_M[grasp_retry_index]}",
                                flush=True,
                            )
            first_roll_tracking_error_delta_m = float(
                np.linalg.norm(
                    wrist_servo_error[0]
                    - (
                        wrist_servo_error[0]
                        if first_roll_error_baseline is None
                        else first_roll_error_baseline
                    )
                )
            )
            if phase is Phase.LEFT_LEG_FLIP_90 and grasp_gate_passed:
                left_grasp_loss_streak = (
                    0 if encoder_grasp_blocked[0] else left_grasp_loss_streak + 1
                )
            else:
                left_grasp_loss_streak = 0
            if phase in (Phase.HANDOVER, Phase.RIGHT_TOP_FLIP_90):
                right_grasp_loss_streak = (
                    0 if encoder_grasp_blocked[1] else right_grasp_loss_streak + 1
                )
            else:
                right_grasp_loss_streak = 0
            redetection_status = "not_due"
            approaching_with_open_hands = (
                phase in (
                    Phase.CLEARANCE_STAGING,
                    Phase.ALIGN_APPROACH,
                    Phase.ALIGN_GRASP,
                )
                and float(np.max(action_np[14:16])) <= -0.95
            )
            if (
                approaching_with_open_hands
                and (control_step + 1) % self.redetect_interval_steps == 0
            ):
                redetection_attempts += 1
                refresh_rgb = self._head_left_rgb(observation)
                try:
                    self._update_head_camera_calibration(observation)
                    (
                        refresh_estimate,
                        refresh_legs,
                        refresh_frame,
                        refresh_attachments,
                    ) = self._localize_table_frame(refresh_rgb, previous_legs=legs)
                    if any(detection.tracked for detection in refresh_legs.values()):
                        raise ValueError("both front legs must be freshly detected")
                    validate_static_table_redetection(
                        settled_control_root_from_table,
                        refresh_frame,
                        settled_leg_attachment_points,
                        refresh_attachments,
                        max_center_drift_m=self.redetect_max_center_drift_m,
                        max_attachment_drift_m=self.redetect_max_attachment_drift_m,
                    )
                    control_root_from_table = blend_table_frames(
                        control_root_from_table,
                        refresh_frame,
                        self.redetect_alpha,
                        max_translation_m=self.redetect_max_translation_m,
                        max_yaw_rad=self.redetect_max_yaw_rad,
                    )
                    leg_attachment_points = {
                        side: (
                            (1.0 - self.redetect_alpha) * leg_attachment_points[side]
                            + self.redetect_alpha * refresh_attachments[side]
                        )
                        for side in ("left", "right")
                    }
                    planner = GeometricFlipPlanner(
                        control_root_from_table,
                        self.sim_control_hz,
                        leg_attachment_points,
                    )
                    estimate = refresh_estimate
                    legs = refresh_legs
                    latest_detection_rgb = refresh_rgb.copy()
                    redetection_updates += 1
                    redetection_status = "accepted"
                    print(
                        "[CvRuleBasedPolicy] closed-loop update "
                        f"step={control_step + 1} center="
                        f"{np.round(control_root_from_table[:2, 3], 3).tolist()} "
                        f"confidence={estimate.confidence:.3f}",
                        flush=True,
                    )
                except ValueError as exc:
                    redetection_status = f"rejected: {exc}"
            trace_row = {
                    "step": control_step + self.warmup_steps,
                    "trajectory_step": trajectory_step,
                    "phase": phase.value,
                    "initial_alignment_mode": initial_alignment_mode,
                    "initial_alignment_pivot_side": initial_alignment_pivot_side,
                    "initial_alignment_moving_side": initial_alignment_moving_side,
                    "action": action_np.tolist(),
                    "pre_rate_limit_action": requested_action_np.tolist(),
                    "cartesian_rate_limited": bool(
                        not np.allclose(action_np, requested_action_np, atol=1.0e-7)
                    ),
                    "nominal_action": nominal_action_np.tolist(),
                    "measured_wrist_positions_before": measured_wrist_positions.tolist(),
                    "measured_grasp_centers_before": measured_grasp_centers.tolist(),
                    "grasp_center_servo_enabled": list(grasp_center_mask),
                    "wrist_servo_correction": wrist_servo_correction.tolist(),
                    "wrist_servo_error": wrist_servo_error.tolist(),
                    "wrist_servo_integral_offsets": wrist_servo_integral_offsets.tolist(),
                    "wrist_servo_gain": servo_gain,
                    "wrist_servo_max_correction_m": servo_max_correction_m,
                    "first_roll_tracking_error_m": first_roll_tracking_error_m,
                    "first_roll_tracking_error_baseline": (
                        None
                        if first_roll_error_baseline is None
                        else first_roll_error_baseline.tolist()
                    ),
                    "first_roll_tracking_error_delta_m": (
                        first_roll_tracking_error_delta_m
                    ),
                    "first_roll_recovery_servo": first_roll_recovery,
                    "first_roll_stall_steps": first_roll_stall_steps,
                    "grasp_gate_active": grasp_gate_active,
                    "grasp_gate_passed": grasp_gate_passed,
                    "grasp_gate_streak": grasp_gate_streak,
                    "alignment_grasp_gate_active": alignment_grasp_gate_active,
                    "alignment_grasp_gate_passed": alignment_grasp_gate_passed,
                    "alignment_pivot_grasp_passed": alignment_pivot_grasp_passed,
                    "alignment_pivot_loss_streak": alignment_pivot_loss_streak,
                    "alignment_grasp_role": alignment_grasp_role,
                    "grasp_camera_side": grasp_camera_side,
                    "alignment_grasp_gate_streak": alignment_grasp_gate_streak,
                    "alignment_grasp_retry_index": alignment_grasp_retry_index,
                    "alignment_grasp_retry_step": alignment_grasp_retry_step,
                    "grasp_gate_frame_fresh": grasp_gate_frame_fresh,
                    "wrist_center_frame_fresh": wrist_center_frame_fresh,
                    "grasp_retry_index": grasp_retry_index,
                    "grasp_retry_step": grasp_retry_step,
                    "grasp_retry_stage": grasp_retry_stage,
                    "grasp_retry_offset_tool_m": (
                        None
                        if active_grasp_retry_index >= len(GRASP_RETRY_OFFSETS_TOOL_M)
                        else list(GRASP_RETRY_OFFSETS_TOOL_M[active_grasp_retry_index])
                    ),
                    "accepted_left_grasp_offset_tool_m": (
                        accepted_left_grasp_offset_tool_m.tolist()
                    ),
                    "accepted_alignment_grasp_offsets_tool_m": (
                        accepted_alignment_grasp_offsets_tool_m.tolist()
                    ),
                    "wrist_visual_correction_m": wrist_visual_correction.tolist(),
                    "wrist_shaft_centered_before": wrist_shaft_centered_before,
                    "wrist_shaft_bottom_px": wrist_shaft_bottom_px,
                    "wrist_shaft_deep_before": wrist_shaft_deep_before,
                    "wrist_edge_y_px": wrist_edge_y_px,
                    "wrist_edge_deep_before": wrist_edge_deep_before,
                    "grasp_wrist_edge_before": (
                        None
                        if grasp_wrist_edge_before is None
                        else {
                            "detected": grasp_wrist_edge_before.detected,
                            "edge_y_px": grasp_wrist_edge_before.edge_y_px,
                            "confidence": grasp_wrist_edge_before.confidence,
                            "white_fraction": grasp_wrist_edge_before.white_fraction,
                        }
                    ),
                    "wrist_shaft_gate_ok": wrist_shaft_gate_ok,
                    "grasp_visual_ready_streak": grasp_visual_ready_streak,
                    "grasp_close_authorized": grasp_close_authorized,
                    "grasp_visual_depth_offset_m": grasp_visual_depth_offset_m,
                    "grasp_wrist_shaft_before": (
                        None
                        if grasp_wrist_shaft_before is None
                        else {
                            "detected": grasp_wrist_shaft_before.detected,
                            "center_px": grasp_wrist_shaft_before.center_px,
                            "confidence": grasp_wrist_shaft_before.confidence,
                            "white_fraction": grasp_wrist_shaft_before.white_fraction,
                            "vertical_support": grasp_wrist_shaft_before.vertical_support,
                            "bounding_box_px": grasp_wrist_shaft_before.bounding_box_px,
                        }
                    ),
                    "grasp_wrist_shaft_after": (
                        None
                        if grasp_wrist_shaft_after is None
                        else {
                            "detected": grasp_wrist_shaft_after.detected,
                            "center_px": grasp_wrist_shaft_after.center_px,
                            "confidence": grasp_wrist_shaft_after.confidence,
                            "white_fraction": grasp_wrist_shaft_after.white_fraction,
                            "vertical_support": grasp_wrist_shaft_after.vertical_support,
                            "bounding_box_px": grasp_wrist_shaft_after.bounding_box_px,
                        }
                    ),
                    "right_handover_target": (
                        None
                        if right_handover_target is None
                        else right_handover_target.tolist()
                    ),
                    "right_push_offset": (
                        None if right_push_offset is None else right_push_offset.tolist()
                    ),
                    "dex1_finger_positions_after": dex1_finger_positions.tolist(),
                    "encoder_grasp_blocked": list(encoder_grasp_blocked),
                    "left_grasp_loss_streak": left_grasp_loss_streak,
                    "right_grasp_loss_streak": right_grasp_loss_streak,
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "cv_confidence": estimate.confidence,
                    "closed_loop_redetection": redetection_status,
                    "closed_loop_update_count": redetection_updates,
                    "post_alignment_relocalized": post_alignment_relocalized,
                    "post_alignment_redetection": post_alignment_redetection_status,
                    "control_root_from_table": control_root_from_table.tolist(),
                    "leg_attachment_points_root_m": {
                        side: point.tolist() for side, point in leg_attachment_points.items()
                    },
                    "simulator_object_pose_used": False,
                    "success": ever_success.tolist(),
                }
            trace_rows.append(trace_row)
            _append_action_state_trace(self, usr_args, trace_row)
            _maybe_save_camera_frames(self, observation, usr_args, control_step + self.warmup_steps)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if left_grasp_loss_streak >= self.grasp_loss_limit_steps:
                abort_reason = (
                    "left grasp lost during first roll according to Dex1 encoders"
                )
            elif right_grasp_loss_streak >= self.grasp_loss_limit_steps:
                abort_reason = (
                    "right grasp missing during handover according to Dex1 encoders"
                )
            if abort_reason is not None:
                print(f"[CvRuleBasedPolicy] safe abort: {abort_reason}", flush=True)
                break
            if _terminated_any(terminated):
                break
            if alignment_grasp_gate_active:
                first_roll_stall_steps = 0
            elif phase is Phase.LEFT_LEG_FLIP_90:
                first_roll_progress_allowed = (
                    grasp_gate_passed
                    and encoder_grasp_blocked[0]
                    and first_roll_tracking_error_delta_m
                    <= self.first_roll_max_tracking_error_m
                )
                if first_roll_progress_allowed:
                    first_roll_stall_steps = 0
                elif grasp_gate_passed:
                    first_roll_stall_steps += 1
                if first_roll_stall_steps >= self.first_roll_max_stall_steps:
                    abort_reason = (
                        "first roll did not converge within the real-compatible "
                        "joint-FK tracking-error increase limit"
                    )
                    print(f"[CvRuleBasedPolicy] safe abort: {abort_reason}", flush=True)
                    break
                if (
                    first_roll_progress_allowed
                    and (control_step + 1) % self.first_roll_advance_interval == 0
                ):
                    trajectory_step += 1
            else:
                first_roll_stall_steps = 0
                trajectory_step += 1
        if redetection_updates:
            self._save_cv_diagnostic(
                usr_args,
                latest_detection_rgb,
                estimate,
                legs,
                control_root_from_table=control_root_from_table,
                leg_attachment_points=leg_attachment_points,
                label="closed_loop_last",
            )
        print(
            "[CvRuleBasedPolicy] closed-loop summary: "
            f"attempts={redetection_attempts}, accepted={redetection_updates}, "
            f"trajectory_step={trajectory_step}/{planner.total_steps}, "
            f"abort={abort_reason or 'none'}",
            flush=True,
        )
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success

    def reset_model(self) -> None:
        pass


class AvpTeleopPolicy(NoOpPolicy):
    """Real-time AVP bridge using only real-compatible cameras and joint state."""

    _BODY_29_JOINT_ORDER = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    _CAMERA_BY_ROLE = {
        "head_left": "first_person_camera",
        "head_right": "head_right_camera",
        "left_wrist": "left_hand_camera",
        "right_wrist": "right_hand_camera",
    }
    _CAMERA_RIG_BY_ROLE = {
        "head_left": "head_stereo",
        "head_right": "head_stereo",
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
        "global": "global",
    }
    _HEAD_CAMERA_ROLES = ("head_left", "head_right")
    _WRIST_CAMERA_ROLES = ("left_wrist", "right_wrist")
    _INACTIVE_CAMERA_UPDATE_PERIOD_S = 3600.0
    _CAMERA_CALIBRATION = {
        "head_left": {
            "intrinsics": (337.5311318539417, 336.61378142923456, 316.5285046932812, 232.50620475777816),
            "distortion": (0.06635329597971165, -0.07841619072258442, -0.0032837567734969727, -0.0010816865229956933, 0.021030073866954904),
        },
        "head_right": {
            "intrinsics": (336.30012498108425, 335.47329565297144, 321.60051380995424, 231.69425545320323),
            "distortion": (0.06366431884731834, -0.08229830690155956, -0.0031845859537499963, 0.0017675102141209843, 0.027381390668112876),
        },
        "left_wrist": {
            "intrinsics": (435.36712646484375, 434.6387481689453, 317.3426818847656, 244.7300567626953),
            "distortion": (-0.05092783644795418, 0.059635864570736885, 0.0010625082795741037, 0.0011093204957433045, -0.020096530206501484),
        },
        "right_wrist": {
            "intrinsics": (435.36712646484375, 434.6387481689453, 317.3426818847656, 244.7300567626953),
            "distortion": (-0.05092783644795418, 0.059635864570736885, 0.0010625082795741037, 0.0011093204957433045, -0.020096530206501484),
        },
    }

    @staticmethod
    def _persistent_sessions_enabled() -> bool:
        """Keep Isaac alive between operator sessions when explicitly enabled."""

        value = os.environ.get("FLIP_TABLE_TELEOP_PERSISTENT", "false").strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        raise ValueError("FLIP_TABLE_TELEOP_PERSISTENT must be boolean")

    @staticmethod
    def _resolve_review_video_hz() -> float:
        """Return the bounded native-RGB review-video rate.

        Review footage is deliberately lower rate than the 30 Hz policy camera
        stream. Rendering and encoding all five views at the servo rate made
        the control/recording transport miss its 30 Hz contract.
        """

        raw = os.environ.get("FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ", "5.0")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                "FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ must be a number in [1,10]"
            ) from exc
        if not math.isfinite(value) or not 1.0 <= value <= 10.0:
            raise ValueError(
                "FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ must be in [1,10]"
            )
        return value

    def _resolve_preview_hz(self) -> float:
        """Return the bounded AVP display cadence, independently of recording.

        The desktop side keeps only one unsent observation.  A lower display
        cadence therefore reduces RTX rendering work without adding a queue of
        stale images: every transmitted frame is the newest completed stereo
        pair.  The dataset clock remains 30 Hz in the offline replay path.
        """

        raw = os.environ.get("FLIP_TABLE_TELEOP_PREVIEW_HZ", "24")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                "FLIP_TABLE_TELEOP_PREVIEW_HZ must be a number in [5,30]"
            ) from exc
        if not math.isfinite(value) or not 5.0 <= value <= 30.0:
            raise ValueError("FLIP_TABLE_TELEOP_PREVIEW_HZ must be in [5,30]")
        return value

    def get_model(self, usr_args: dict[str, Any]) -> None:
        super().get_model(usr_args)
        # RoboFinals V1's evaluation CLI does not populate ``actions_dim``.
        # BasePolicy therefore falls back to 1 even though the configured
        # ActionManager is the 16-D arm/hand action; WBC owns waist and legs.
        self.action_dim = _teleop_action_dim(usr_args)
        try:
            from .teleop.config import load_teleop_config
            from .teleop.contracts import ControlEvent
            from .teleop.sim.safety import CommandSafetyFilter, WatchdogState
            from .teleop.transport import FramedSocket
        except ImportError:
            from teleop.config import load_teleop_config
            from teleop.contracts import ControlEvent
            from teleop.sim.safety import CommandSafetyFilter, WatchdogState
            from teleop.transport import FramedSocket

        config_path = os.environ.get(
            "FLIP_TABLE_TELEOP_CONFIG",
            str(Path(__file__).resolve().parent / "teleop" / "configs" / "teleop_v1.json"),
        )
        self.teleop_config = load_teleop_config(config_path)
        self._ControlEvent = ControlEvent
        self._WatchdogState = WatchdogState
        self._FramedSocket = FramedSocket
        self._safety = CommandSafetyFilter(
            self.teleop_config.safety,
            servo_hz=self.teleop_config.rates.servo_hz,
        )
        self._command_lock = threading.Lock()
        self._latest_command = None
        self._last_command_received_ns: int | None = None
        self._last_event_sequence = -1
        self._transport = None
        self._receiver_thread = None
        self._receiver_error: BaseException | None = None
        self._previous_body_position: np.ndarray | None = None
        self._sim_recording = False
        self._review_video_hz = self._resolve_review_video_hz()
        self._preview_hz = self._resolve_preview_hz()
        self._next_review_video_time = 0.0
        self._episode_terminated = False
        self._current_episode_success = False
        self._applied_arm = np.zeros(14, dtype=np.float64)
        self._applied_hand = np.ones(2, dtype=np.float64)
        self._action_delay_queue: deque[
            tuple[int, np.ndarray, np.ndarray]
        ] = deque()
        self._camera_delay_queue: deque[
            tuple[
                int,
                dict[str, np.ndarray],
                dict[str, np.ndarray],
                dict[str, float],
                dict[str, int],
            ]
        ] = deque()
        self._camera_map_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._camera_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._camera_host_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}
        self._sensor_rng = np.random.default_rng(int(usr_args.get("seed", 42)))
        self._latest_sim_diagnostics: dict[str, Any] = {}
        self._observation_sequence = 0
        self._observation_send_times: deque[float] = deque(maxlen=61)
        self._control_step_times: deque[float] = deque(maxlen=101)
        self._command_receive_times: deque[float] = deque(maxlen=61)
        self._last_watchdog_state = "stop"
        self._last_arm_tracking_error_rad = 0.0
        self._last_hand_tracking_error_fraction = 0.0
        self._last_applied_command_sequence = -1
        self._sim_hold_arm: np.ndarray | None = None
        self._sim_hold_hand: np.ndarray | None = None
        self._force_audit = self._new_force_audit()
        self._observation_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=1
        )
        self._observation_sender_thread: threading.Thread | None = None
        self._observation_sender_error: BaseException | None = None
        self._dropped_operator_frames = 0
        self._server = None
        self._open_server()

    def _open_server(self) -> None:
        """Create the AVP listener once per policy process.

        RoboFinals invokes ``reset_model`` after the first environment reset and
        immediately before ``eval``. The listener is a process resource, not an
        episode resource, so it must survive that reset.
        """

        if self._server is not None:
            return
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = os.environ.get("FLIP_TABLE_TELEOP_BIND_HOST", "0.0.0.0")
        port = int(
            os.environ.get(
                "FLIP_TABLE_TELEOP_PORT", str(self.teleop_config.workstation.sim_port)
            )
        )
        self._server.bind((host, port))
        self._server.listen(1)
        print(f"[AvpTeleopPolicy] waiting for AVP bridge on {host}:{port}", flush=True)

    def _accept_client(self, *, timeout_s: float) -> None:
        if self._transport is not None:
            return
        self._open_server()
        assert self._server is not None
        self._server.settimeout(timeout_s)
        connection, address = self._server.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._transport = self._FramedSocket(connection)
        self._transport.send(
            {
                "schema_version": "team_ramen_flip_table_teleop_message/v1",
                "type": "hello",
                "backend": "sim",
                "config_sha256": self.teleop_config.digest,
                "runtime_digest": self.teleop_config.runtime.robofinals_digest,
                "servo_hz": self.teleop_config.rates.servo_hz,
                "camera_hz": self.teleop_config.rates.camera_hz,
                "preview_hz": self._preview_hz,
            }
        )
        self._receiver_error = None
        self._receiver_thread = threading.Thread(
            target=self._receive_commands,
            args=(self._transport,),
            name="avp-sim-command-receiver",
            daemon=True,
        )
        self._receiver_thread.start()
        print(f"[AvpTeleopPolicy] AVP bridge connected from {address}", flush=True)

    def _drop_unready_client(self) -> None:
        """Release a pre-ready bridge so the listener can accept a clean retry.

        An SSH tunnel can occasionally establish a TCP channel while the V1
        process is still coming up, then close it before the desktop has read
        the hello frame.  This method only applies before a control command is
        accepted, so reconnecting cannot create or extend robot motion.
        """

        transport = self._transport
        self._transport = None
        if transport is not None:
            transport.close()
        self._receiver_thread = None
        self._receiver_error = None
        with self._command_lock:
            self._latest_command = None
            self._last_command_received_ns = None

    def _release_client(self) -> None:
        """Safely release one desktop session while retaining the Isaac process.

        The TCP bridge and its sender threads are episode resources.  The
        listener, Isaac application, and environment remain process resources
        so a later AVP session can reconnect without a cold simulator start.
        """

        self._stop_observation_sender()
        transport = self._transport
        self._transport = None
        if transport is not None:
            transport.close()
        receiver = self._receiver_thread
        self._receiver_thread = None
        if receiver is not None and receiver.is_alive():
            receiver.join(timeout=2.0)
        if receiver is not None and receiver.is_alive():
            raise RuntimeError("AVP command receiver did not stop")
        self._receiver_error = None
        self._observation_sender_error = None
        self._observation_queue = queue.Queue(maxsize=1)
        with self._command_lock:
            self._latest_command = None
            self._last_command_received_ns = None
        self._last_event_sequence = -1
        self._previous_body_position = None
        self._action_delay_queue.clear()
        self._camera_delay_queue.clear()

    def close(self) -> None:
        """Release all process resources when the simulator is explicitly stopped."""

        self._release_client()
        server = self._server
        self._server = None
        if server is not None:
            server.close()

    def _receive_commands(self, transport) -> None:
        try:
            try:
                from .teleop.contracts import ArmHandTarget
            except ImportError:
                from teleop.contracts import ArmHandTarget
            while True:
                command = ArmHandTarget.from_message(transport.receive())
                received_ns = time.monotonic_ns()
                with self._command_lock:
                    if (
                        self._latest_command is not None
                        and command.sequence < self._latest_command.sequence
                    ):
                        raise ValueError("AVP command sequence moved backwards")
                    self._latest_command = command
                    self._last_command_received_ns = received_ns
                received_s = received_ns / 1.0e9
                self._command_receive_times.append(received_s)
                if command.sequence <= 3 or command.sequence % 150 == 0:
                    receive_hz = None
                    if len(self._command_receive_times) > 1:
                        elapsed = (
                            self._command_receive_times[-1]
                            - self._command_receive_times[0]
                        )
                        if elapsed > 0.0:
                            receive_hz = (
                                len(self._command_receive_times) - 1
                            ) / elapsed
                    print(
                        "[AvpTeleopPolicy] command received: "
                        f"sequence={command.sequence}, mode={command.mode.value}, "
                        f"receive_hz={None if receive_hz is None else round(receive_hz, 2)}, "
                        f"dex1_opening={[round(value, 3) for value in command.dex1_opening_fraction]}",
                        flush=True,
                    )
        except (EOFError, OSError) as exc:
            if self._transport is transport:
                self._receiver_error = exc
        except BaseException as exc:  # noqa: BLE001
            if self._transport is transport:
                self._receiver_error = exc

    def _command_snapshot(self):
        with self._command_lock:
            return self._latest_command, self._last_command_received_ns

    def _wait_for_client_ready(self, *, timeout_s: float) -> None:
        """Wait for the desktop's non-actuating readiness command.

        The client creates its AVP renderer after the TCP hello exchange.
        Waiting for its explicit IDLE message prevents a cold simulator from
        filling a tunnel before the operator process has entered its receive
        loop. No environment step or joint target is issued while waiting.
        """

        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("FLIP_TABLE_TELEOP_READY_TIMEOUT_S must be positive")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            command, _received_ns = self._command_snapshot()
            if command is not None:
                if command.mode.value != "idle" or command.event is not self._ControlEvent.NONE:
                    raise ValueError("first AVP bridge message must be an IDLE readiness command")
                print("[AvpTeleopPolicy] desktop AVP bridge ready", flush=True)
                return
            if self._receiver_error is not None:
                raise RuntimeError(
                    f"AVP bridge closed before readiness: {self._receiver_error}"
                )
            time.sleep(0.01)
        raise TimeoutError("AVP desktop did not send its readiness command")

    def _accept_ready_client(self) -> None:
        """Accept a desktop bridge, retrying only before its IDLE readiness.

        The desktop still has to send a valid IDLE command before the simulator
        takes a step.  Once that command is received, normal watchdog behavior
        remains unchanged and any later disconnect ends the evaluation loop.
        """

        total_timeout_s = float(
            os.environ.get("FLIP_TABLE_TELEOP_ACCEPT_TIMEOUT_S", "900")
        )
        pre_ready_timeout_s = float(
            os.environ.get("FLIP_TABLE_TELEOP_PRE_READY_TIMEOUT_S", "20")
        )
        if (
            not math.isfinite(total_timeout_s)
            or not math.isfinite(pre_ready_timeout_s)
            or total_timeout_s <= 0.0
            or pre_ready_timeout_s <= 0.0
        ):
            raise ValueError("teleop connection timeouts must be positive")
        deadline = time.monotonic() + total_timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            remaining_s = deadline - time.monotonic()
            self._accept_client(timeout_s=max(0.1, remaining_s))
            try:
                self._wait_for_client_ready(
                    timeout_s=min(pre_ready_timeout_s, remaining_s)
                )
                return
            except (RuntimeError, TimeoutError) as exc:
                last_error = exc
                print(
                    "[AvpTeleopPolicy] pre-ready bridge failed; "
                    f"waiting for a clean reconnect: {exc}",
                    flush=True,
                )
                self._drop_unready_client()
        raise TimeoutError(
            "no AVP bridge completed the non-actuating readiness handshake"
        ) from last_error

    @staticmethod
    def _merged_observation(observation: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for group_name in ("policy", "embodiment_general_obs"):
            group = observation.get(group_name, {})
            if isinstance(group, dict):
                merged.update(group)
        return merged

    @classmethod
    def _body_state_from_joint_vector(
        cls, joint_values: np.ndarray, joint_velocities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(joint_values, dtype=np.float64).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError(f"simulator joint position must be finite [33], got {values.shape}")
        source_order = CvRuleBasedPolicy._G1_GRIPPER_33_JOINT_ORDER
        lookup = {name: index for index, name in enumerate(source_order)}
        body = np.asarray([values[lookup[name]] for name in cls._BODY_29_JOINT_ORDER])
        if joint_velocities is None:
            velocity = np.zeros(29, dtype=np.float64)
        else:
            source_velocity = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
            if source_velocity.shape != (33,) or not np.isfinite(source_velocity).all():
                raise ValueError("simulator joint velocity must be finite [33]")
            velocity = np.asarray(
                [source_velocity[lookup[name]] for name in cls._BODY_29_JOINT_ORDER]
            )
        return body, velocity

    @staticmethod
    def _dex1_opening(joint_values: np.ndarray) -> np.ndarray:
        values = np.asarray(joint_values, dtype=np.float64)
        fingers = np.asarray(
            (
                values[list(_G1_LEFT_DEX1_JOINT_INDICES)].mean(),
                values[list(_G1_RIGHT_DEX1_JOINT_INDICES)].mean(),
            )
        )
        return np.clip(
            (fingers - _DEX1_CLOSE_POS) / (_DEX1_OPEN_POS - _DEX1_CLOSE_POS),
            0.0,
            1.0,
        )

    @staticmethod
    def _jpeg(image: np.ndarray, *, recording: bool) -> bytes:
        # The data contract requires 4:4:4 JPEG for a recording.  OpenCV's
        # native encoder produces the same sampling mode much faster than the
        # Pillow path previously used here; that matters when four 640x480
        # streams are active at 30 Hz.
        import cv2

        quality = 95 if recording else 88
        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        if recording:
            sampling_key = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
            sampling_444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
            if sampling_key is None or sampling_444 is None:
                raise RuntimeError("OpenCV lacks the required JPEG 4:4:4 encoder option")
            parameters.extend((int(sampling_key), int(sampling_444)))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, parameters)
        if not ok:
            raise RuntimeError("failed to encode AVP camera frame")
        return encoded.tobytes()

    def _diagnostics(self, task_env: Any) -> dict[str, Any]:
        try:
            result = task_env.unwrapped.flip_table_teleop_diagnostics()
        except (AttributeError, RuntimeError):
            return {}
        if not isinstance(result, dict):
            raise TypeError("flip_table_teleop_diagnostics must return a dictionary")
        return result

    @staticmethod
    def _new_force_audit() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "contact_max_n": np.zeros(2, dtype=np.float64),
            "drive_max_n": np.zeros(2, dtype=np.float64),
            "closed_sample_count": np.zeros(2, dtype=np.int64),
            "closed_without_load_count": np.zeros(2, dtype=np.int64),
            "contact_streak": np.zeros(2, dtype=np.int64),
            "contact_streak_max": np.zeros(2, dtype=np.int64),
            "loaded_streak": np.zeros(2, dtype=np.int64),
            "loaded_streak_max": np.zeros(2, dtype=np.int64),
        }

    @staticmethod
    def _force_pair(metrics: Any) -> np.ndarray | None:
        if not isinstance(metrics, dict) or not metrics.get("available"):
            return None
        values = np.asarray(
            (metrics.get("left_max_n", 0.0), metrics.get("right_max_n", 0.0)),
            dtype=np.float64,
        )
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError("simulator Dex1 force diagnostics are invalid")
        return values

    def _sample_force_diagnostics(self, task_env: Any) -> None:
        """Accumulate every 50 Hz force sample, independently of camera timing."""

        try:
            result = task_env.unwrapped.flip_table_teleop_force_diagnostics()
        except AttributeError as exc:
            raise RuntimeError("servo-rate Dex1 force diagnostics are unavailable") from exc
        if not isinstance(result, dict):
            raise TypeError("flip_table_teleop_force_diagnostics must return a dictionary")
        contact = self._force_pair(result.get("gripper_contact_force_n"))
        drive = self._force_pair(result.get("dex1_drive_force_n"))
        if contact is None or drive is None:
            raise RuntimeError("all four Dex1 force sensors and drives must be available")

        audit = self._force_audit
        audit["sample_count"] += 1
        audit["contact_max_n"] = np.maximum(audit["contact_max_n"], contact)
        audit["drive_max_n"] = np.maximum(audit["drive_max_n"], drive)

        closed = self._applied_hand <= 0.15
        in_contact = contact >= 0.5
        loaded = drive >= 2.0
        audit["closed_sample_count"] += closed.astype(np.int64)
        audit["closed_without_load_count"] += (closed & ~in_contact & ~loaded).astype(
            np.int64
        )
        audit["contact_streak"] = np.where(
            closed & in_contact, audit["contact_streak"] + 1, 0
        )
        audit["loaded_streak"] = np.where(
            closed & loaded, audit["loaded_streak"] + 1, 0
        )
        audit["contact_streak_max"] = np.maximum(
            audit["contact_streak_max"], audit["contact_streak"]
        )
        audit["loaded_streak_max"] = np.maximum(
            audit["loaded_streak_max"], audit["loaded_streak"]
        )

    def _force_audit_diagnostics(self) -> dict[str, Any]:
        audit = self._force_audit
        servo_hz = self.teleop_config.rates.servo_hz
        return {
            "gripper_contact_force_n": {
                "available": audit["sample_count"] > 0,
                "aggregation": "servo_rate_session_max",
                "left_max_n": float(audit["contact_max_n"][0]),
                "right_max_n": float(audit["contact_max_n"][1]),
            },
            "dex1_drive_force_n": {
                "available": audit["sample_count"] > 0,
                "aggregation": "servo_rate_session_max",
                "effort_limit_n": 20.0,
                "left_max_n": float(audit["drive_max_n"][0]),
                "right_max_n": float(audit["drive_max_n"][1]),
            },
            "dex1_grasp_force_audit": {
                "sample_hz": servo_hz,
                "sample_count": int(audit["sample_count"]),
                "closed_sample_count_left_right": audit[
                    "closed_sample_count"
                ].tolist(),
                "closed_without_load_count_left_right": audit[
                    "closed_without_load_count"
                ].tolist(),
                "sustained_contact_max_s_left_right": (
                    audit["contact_streak_max"] / servo_hz
                ).round(4).tolist(),
                "sustained_drive_load_max_s_left_right": (
                    audit["loaded_streak_max"] / servo_hz
                ).round(4).tolist(),
                "contact_threshold_n": 0.5,
                "drive_load_threshold_n": 2.0,
                "closed_command_threshold": 0.15,
            },
        }

    @staticmethod
    def _sensor_rgb_snapshot(task_env: Any, camera_name: str) -> torch.Tensor:
        """Snapshot one due RGB frame on-device without synchronizing to CPU."""

        env = getattr(task_env, "unwrapped", task_env)
        sensors = getattr(getattr(env, "scene", None), "sensors", None)
        if not isinstance(sensors, dict) or camera_name not in sensors:
            raise RuntimeError(f"AVP camera sensor is missing: {camera_name}")
        output = getattr(getattr(sensors[camera_name], "data", None), "output", None)
        if not isinstance(output, dict) or "rgb" not in output:
            raise RuntimeError(f"AVP camera has no initialized RGB buffer: {camera_name}")
        value = output["rgb"]
        tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
        if tuple(tensor.shape) not in {(1, 480, 640, 3), (1, 480, 640, 4)}:
            raise ValueError(
                f"{camera_name} must remain batched RGB(A) 640x480, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.clone()

    @staticmethod
    def _camera_randomization(diagnostics: dict[str, Any]) -> dict[str, Any]:
        randomization = diagnostics.get("randomization")
        camera = randomization.get("camera_image") if isinstance(randomization, dict) else None
        if not isinstance(camera, dict) or not isinstance(camera.get("rigs"), dict):
            raise RuntimeError("sim teleoperation requires episode camera randomization metadata")
        return camera

    def _camera_remap(
        self,
        role: str,
        profile: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration = self._CAMERA_CALIBRATION[role]
        focal_scale = float(profile["focal_scale"])
        principal_delta = np.asarray(
            profile["principal_point_delta_px"], dtype=np.float64
        )
        distortion_scale = float(profile["distortion_scale"])
        if (
            not math.isfinite(focal_scale)
            or not math.isfinite(distortion_scale)
            or principal_delta.shape != (2,)
            or not np.isfinite(principal_delta).all()
        ):
            raise ValueError(f"invalid camera randomization profile for {role}")
        cache_key = (
            role,
            round(focal_scale, 9),
            round(float(principal_delta[0]), 9),
            round(float(principal_delta[1]), 9),
            round(distortion_scale, 9),
        )
        cached = self._camera_map_cache.get(cache_key)
        if cached is not None:
            return cached

        import cv2

        fx, fy, cx, cy = calibration["intrinsics"]
        nominal = np.asarray(
            ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        randomized = nominal.copy()
        randomized[0, 0] *= focal_scale
        randomized[1, 1] *= focal_scale
        randomized[0, 2] += principal_delta[0]
        randomized[1, 2] += principal_delta[1]
        distortion = np.asarray(calibration["distortion"], dtype=np.float64)
        distortion *= distortion_scale
        x, y = np.meshgrid(
            np.arange(640, dtype=np.float32),
            np.arange(480, dtype=np.float32),
        )
        output_pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
        source_pixels = cv2.undistortPoints(
            output_pixels,
            randomized,
            distortion,
            P=nominal,
        ).reshape(480, 640, 2)
        maps = (
            source_pixels[..., 0].astype(np.float32),
            source_pixels[..., 1].astype(np.float32),
        )
        self._camera_map_cache[cache_key] = maps
        return maps

    def _apply_camera_randomization(
        self,
        image: np.ndarray,
        role: str,
        profile: dict[str, Any],
        noise_std_fraction: float,
        noise_seed: int,
    ) -> np.ndarray:
        import cv2

        map_x, map_y = self._camera_remap(role, profile)
        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)
        exposure_ev = float(profile["exposure_ev"])
        if not math.isfinite(exposure_ev) or not 0.0 <= noise_std_fraction <= 0.1:
            raise ValueError(f"invalid camera exposure/noise for {role}")
        warped *= 2.0**exposure_ev
        if noise_std_fraction:
            # Seed every frame explicitly so replaying one episode remains
            # deterministic across runs.
            noise = np.empty_like(warped)
            cv2.setRNGSeed(noise_seed)
            sigma = 255.0 * noise_std_fraction
            cv2.randn(noise, (0.0, 0.0, 0.0), (sigma, sigma, sigma))
            cv2.add(warped, noise, dst=warped)
        return np.clip(np.rint(warped), 0.0, 255.0).astype(np.uint8)

    def _camera_packet(
        self,
        images: dict[str, np.ndarray],
        diagnostic_images: dict[str, np.ndarray],
        diagnostics: dict[str, Any],
        capture_ns: int,
    ) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        camera = self._camera_randomization(diagnostics)
        rigs = camera["rigs"]
        noise_max = float(camera.get("noise_std_fraction_max", 0.0))
        latency_max = int(camera.get("latency_max_steps", 0))
        if not 0.0 <= noise_max <= 0.1 or not 0 <= latency_max <= 2:
            raise ValueError("camera sensor randomization exceeds the release limits")
        noise_samples = {}
        noise_seeds = {}
        randomized_images = {}
        for role, image in images.items():
            rig = self._CAMERA_RIG_BY_ROLE[role]
            profile = rigs.get(rig)
            if not isinstance(profile, dict):
                raise RuntimeError(f"camera randomization profile is missing: {rig}")
            noise = float(self._sensor_rng.uniform(0.0, noise_max))
            noise_seed = int(self._sensor_rng.integers(0, 2**31 - 1))
            noise_samples[role] = noise
            noise_seeds[role] = noise_seed
            randomized_images[role] = self._apply_camera_randomization(
                image,
                role,
                profile,
                noise,
                noise_seed,
            )

        self._camera_delay_queue.append(
            (
                capture_ns,
                randomized_images,
                diagnostic_images,
                noise_samples,
                noise_seeds,
            )
        )
        while len(self._camera_delay_queue) > 3:
            self._camera_delay_queue.popleft()
        requested_delay = int(self._sensor_rng.integers(0, latency_max + 1))
        try:
            from .teleop.timing import bounded_delay_steps
        except ImportError:
            from teleop.timing import bounded_delay_steps

        nominal_latency_ns = int(
            round(latency_max / self.teleop_config.rates.camera_hz * 1.0e9)
        )
        actual_delay = bounded_delay_steps(
            [sample[0] for sample in self._camera_delay_queue],
            requested_delay,
            maximum_age_ns=nominal_latency_ns,
        )
        selected = self._camera_delay_queue[-1 - actual_delay]
        return (
            selected[0],
            selected[1],
            selected[2],
            {
                "latency_steps": actual_delay,
                "latency_s": (capture_ns - selected[0]) / 1.0e9,
                "requested_latency_steps": requested_delay,
                "noise_std_fraction": selected[3],
                "noise_seed": selected[4],
            },
        )

    def _send_observation(
        self,
        task_env: Any,
        observation: dict[str, Any],
        joint_values: np.ndarray,
    ) -> None:
        assert self._transport is not None
        started = time.monotonic()
        diagnostics = (
            self._diagnostics(task_env)
            if self._sim_recording
            else self._latest_sim_diagnostics
        )
        if not diagnostics:
            raise RuntimeError("AVP reset diagnostics were not initialized")
        # The AVP is an operator display, not the dataset recorder.  Keeping
        # it head-stereo-only avoids serially rendering four extra RTX cameras
        # during hand control.  The 30 Hz four-camera package is rendered from
        # the saved command trajectory after a successful session.
        active_roles = self._HEAD_CAMERA_ROLES
        # The world-fixed global view belongs to simulator diagnostics and
        # offline success accounting, not the AVP operator loop. Sending it
        # at 30 Hz would consume tunnel bandwidth without contributing to a
        # real-compatible control decision. The organizer recorder continues
        # to save the same camera in its local diagnostic video.
        diagnostic_images: dict[str, np.ndarray] = {}

        merged = self._merged_observation(observation)
        velocity_value = merged.get("joint_vel")
        velocity_33 = (
            None
            if velocity_value is None
            else np.asarray(
                velocity_value[0].detach().cpu()
                if torch.is_tensor(velocity_value)
                else velocity_value
            )
        )
        if velocity_33 is None:
            if self._previous_body_position is None:
                body, body_velocity = self._body_state_from_joint_vector(joint_values)
            else:
                body, _ = self._body_state_from_joint_vector(joint_values)
                body_velocity = (
                    body - self._previous_body_position
                ) * self.teleop_config.rates.servo_hz
        else:
            body, body_velocity = self._body_state_from_joint_vector(
                joint_values, np.asarray(velocity_33)
            )
        self._previous_body_position = body.copy()
        root_pose = diagnostics.get("root_pose_world_xyzw", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        now_ns = time.monotonic_ns()
        message_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key not in {"joint_names", "joint_position_rad", "joint_velocity_rad_s"}
        }
        message_diagnostics.update(self._force_audit_diagnostics())
        physics_hz = round(1.0 / float(task_env.unwrapped.cfg.sim.dt), 6)
        control_hz = round(_runtime_control_hz(task_env), 6)
        message_diagnostics["sim_control_contract"] = {
            "body_mode": os.environ.get(
                "FLIP_TABLE_SIM_BODY_MODE", "balanced_wbc"
            ).strip().lower(),
            "physics_hz": physics_hz,
            "control_hz": control_hz,
            "wbc_navigation_velocity_m_s_rad_s": [0.0, 0.0, 0.0],
            "wbc_base_height_m": 0.74,
            "wbc_torso_rpy_rad": [0.0, 0.0, 0.0],
            "wbc_stand_onnx_sha256": os.environ.get(
                "FLIP_TABLE_WBC_STAND_ONNX_SHA256", ""
            ),
            "wbc_walk_onnx_sha256": os.environ.get(
                "FLIP_TABLE_WBC_WALK_ONNX_SHA256", ""
            ),
            "team_adapter_sha256": os.environ.get(
                "FLIP_TABLE_WBC_ADAPTER_SHA256", ""
            ),
        }
        command, _received_ns = self._command_snapshot()
        message_diagnostics["last_received_command_sequence"] = (
            -1 if command is None else command.sequence
        )
        message_diagnostics["last_applied_command_sequence"] = (
            self._last_applied_command_sequence
        )
        message_diagnostics["dropped_operator_frames"] = (
            self._dropped_operator_frames
        )
        # Take the GPU snapshots only after every small state tensor has been
        # copied. A later synchronous state read on the default CUDA stream
        # would otherwise wait for these image clones and defeat the dedicated
        # asynchronous camera-copy stream.
        gpu_snapshot_started = time.monotonic()
        images = {
            role: self._sensor_rgb_snapshot(
                task_env, self._CAMERA_BY_ROLE[role]
            )
            for role in active_roles
        }
        gpu_ready_event = self._camera_ready_event(images)
        gpu_snapshot_ms = (time.monotonic() - gpu_snapshot_started) * 1000.0
        self._observation_sequence += 1
        self._queue_observation(
            {
                "started": started,
                "sequence": self._observation_sequence,
                "capture_ns": now_ns,
                "images": images,
                "gpu_ready_event": gpu_ready_event,
                "gpu_snapshot_ms": gpu_snapshot_ms,
                "diagnostic_images": diagnostic_images,
                "diagnostics": diagnostics,
                "message_diagnostics": message_diagnostics,
                "recording": self._sim_recording,
                "body": tuple(body),
                "body_velocity": tuple(body_velocity),
                "dex1_opening": tuple(self._dex1_opening(joint_values)),
                "applied_arm": tuple(self._applied_arm.copy()),
                "applied_hand": tuple(self._applied_hand.copy()),
                "root_pose": tuple(root_pose),
                "success": diagnostics.get("success"),
            }
        )

    def _camera_ready_event(
        self, images: dict[str, torch.Tensor]
    ) -> torch.cuda.Event | None:
        if self._camera_copy_stream is None or not images:
            return None
        event = torch.cuda.Event(blocking=False)
        event.record(torch.cuda.current_stream(device=self.device))
        return event

    def _camera_snapshots_to_cpu(
        self,
        images: dict[str, torch.Tensor],
        ready_event: torch.cuda.Event | None,
    ) -> dict[str, np.ndarray]:
        stream = self._camera_copy_stream
        if stream is None:
            return {
                role: _camera_image_uint8(image) for role, image in images.items()
            }
        if ready_event is None:
            raise RuntimeError("CUDA camera snapshots require a readiness event")
        host_tensors: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            for role, image in images.items():
                key = (role, tuple(image.shape), image.dtype)
                host = self._camera_host_buffers.get(key)
                if host is None:
                    host = torch.empty(
                        tuple(image.shape),
                        dtype=image.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self._camera_host_buffers[key] = host
                host.copy_(image, non_blocking=True)
                host_tensors[role] = host
            copied = torch.cuda.Event(blocking=False)
            copied.record(stream)
        copied.synchronize()
        return {
            role: _camera_image_uint8(image)
            for role, image in host_tensors.items()
        }

    def _warm_live_camera_pipeline(self, task_env: Any) -> None:
        """Allocate GPU/host/JPEG resources before the first operator frame."""

        started = time.monotonic()
        images = {
            role: self._sensor_rgb_snapshot(task_env, self._CAMERA_BY_ROLE[role])
            for role in self._HEAD_CAMERA_ROLES
        }
        host_images = self._camera_snapshots_to_cpu(
            images,
            self._camera_ready_event(images),
        )
        for image in host_images.values():
            self._jpeg(image, recording=False)
        print(
            "[AvpTeleopPolicy] live camera pipeline warmed: "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )

    def _deliver_observation(self, payload: dict[str, Any]) -> None:
        try:
            from .teleop.contracts import TeleopObservation
        except ImportError:
            from teleop.contracts import TeleopObservation

        copy_started = time.monotonic()
        images = self._camera_snapshots_to_cpu(
            payload["images"], payload["gpu_ready_event"]
        )
        gpu_to_cpu_ms = (time.monotonic() - copy_started) * 1000.0
        for role, image in images.items():
            if image.shape != (480, 640, 3):
                raise ValueError(
                    f"{role} must remain RGB 640x480, got {image.shape}"
                )
        # The live AVP stream is deliberately independent of the collection
        # state.  A saved simulator trajectory is replayed after the session
        # to generate the synchronized 30 Hz head/wrist/global dataset.  Do
        # not enable recording-only latency/noise/high-quality JPEG work here:
        # it adds simulator load precisely when the operator presses ``s`` and
        # turns a low-latency latest-frame display into a visibly choppy one.
        # Episode-fixed mount and scene randomization are already rendered.
        camera_capture_ns = payload["capture_ns"]
        selected_images = images
        selected_diagnostic_images: dict[str, np.ndarray] = {}
        camera_runtime = {
            "latency_steps": 0,
            "latency_s": 0.0,
            "noise_std_fraction": {role: 0.0 for role in selected_images},
            "noise_seed": {},
        }
        jpeg_started = time.monotonic()
        # Dataset frames are produced by offline replay. Starting an episode
        # must never switch the latency-sensitive AVP stream to the expensive
        # archival JPEG profile.
        encoded_images = {
            role: self._jpeg(image, recording=False)
            for role, image in selected_images.items()
        }
        encoded_diagnostics = {
            role: self._jpeg(image, recording=False)
            for role, image in selected_diagnostic_images.items()
        }
        jpeg_ms = (time.monotonic() - jpeg_started) * 1000.0
        message_diagnostics = dict(payload["message_diagnostics"])
        message_diagnostics["camera_runtime"] = camera_runtime
        message = TeleopObservation(
            sequence=payload["sequence"],
            capture_monotonic_ns=payload["capture_ns"],
            backend="sim",
            body_joint_position_rad=payload["body"],
            body_joint_velocity_rad_s=payload["body_velocity"],
            dex1_opening_fraction=payload["dex1_opening"],
            applied_arm_target_rad=payload["applied_arm"],
            applied_dex1_opening_target=payload["applied_hand"],
            root_pose_xyzw=payload["root_pose"],
            camera_capture_monotonic_ns={
                role: camera_capture_ns for role in encoded_images
            },
            camera_jpeg=encoded_images,
            diagnostic_camera_capture_monotonic_ns={
                role: camera_capture_ns for role in encoded_diagnostics
            },
            diagnostic_camera_jpeg=encoded_diagnostics,
            success=payload["success"],
            diagnostics=message_diagnostics,
        )
        transport = self._transport
        if transport is None:
            raise RuntimeError("AVP observation transport closed while sending")
        transport_started = time.monotonic()
        transport.send(message.to_message())
        transport_ms = (time.monotonic() - transport_started) * 1000.0
        self._observation_send_times.append(time.monotonic())
        sequence = int(payload["sequence"])
        report_rate = sequence <= 3 or sequence % 150 == 0
        if report_rate:
            payload_bytes = sum(len(value) for value in encoded_images.values())
            stream_hz = None
            if len(self._observation_send_times) > 1:
                duration = self._observation_send_times[-1] - self._observation_send_times[0]
                if duration > 0.0:
                    stream_hz = (len(self._observation_send_times) - 1) / duration
            print(
                "[AvpTeleopPolicy] observation delivered: "
                f"sequence={sequence}, roles={sorted(encoded_images)}, "
                f"jpeg_bytes={payload_bytes}, "
                f"elapsed_s={time.monotonic() - payload['started']:.3f}, "
                f"gpu_snapshot_ms={payload['gpu_snapshot_ms']:.2f}, "
                f"gpu_to_cpu_ms={gpu_to_cpu_ms:.2f}, jpeg_ms={jpeg_ms:.2f}, "
                f"transport_ms={transport_ms:.2f}, "
                f"stream_hz={None if stream_hz is None else round(stream_hz, 2)}, "
                f"dropped_operator_frames={self._dropped_operator_frames}",
                flush=True,
            )

    def _observation_sender_loop(self) -> None:
        while True:
            payload = self._observation_queue.get()
            try:
                if payload is None:
                    return
                self._deliver_observation(payload)
            except BaseException as exc:  # noqa: BLE001
                self._observation_sender_error = exc
                print("[AvpTeleopPolicy] observation sender failed:", flush=True)
                traceback.print_exc()
                return
            finally:
                self._observation_queue.task_done()

    def _start_observation_sender(self) -> None:
        if self._observation_sender_thread is not None:
            return
        self._observation_sender_error = None
        self._observation_sender_thread = threading.Thread(
            target=self._observation_sender_loop,
            name="avp-sim-observation-sender",
            daemon=True,
        )
        self._observation_sender_thread.start()

    def _queue_observation(self, payload: dict[str, Any]) -> None:
        if self._observation_sender_error is not None:
            raise RuntimeError(
                f"AVP observation sender failed: {self._observation_sender_error}"
            )
        if self._observation_sender_thread is None:
            raise RuntimeError("AVP observation sender is not running")
        # Recording stores the 30 Hz command trajectory, not this lossy AVP
        # display stream.  Keep latest-only behavior in both modes: blocking
        # here would turn a slow JPEG/WebXR consumer into visible hand-control
        # latency.
        try:
            self._observation_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._observation_queue.get_nowait()
                self._observation_queue.task_done()
                self._dropped_operator_frames += 1
            except queue.Empty:
                pass
            self._observation_queue.put_nowait(payload)

    def _flush_observation_sender(self) -> None:
        if self._observation_sender_thread is None:
            return
        self._observation_queue.join()
        if self._observation_sender_error is not None:
            raise RuntimeError(
                f"AVP observation sender failed: {self._observation_sender_error}"
            )

    def _stop_observation_sender(self) -> None:
        thread = self._observation_sender_thread
        self._observation_sender_thread = None
        if thread is None:
            return
        if thread.is_alive():
            self._observation_queue.put(None)
            thread.join(timeout=2.0)
        if thread.is_alive():
            raise RuntimeError("AVP observation sender did not stop")

    def _set_camera_capture_mode(self, task_env: Any, *, recording: bool) -> None:
        """Keep the AVP path to latest head stereo in every collection state.

        Four policy cameras and the global review image are rendered by the
        offline trajectory materializer after a successful session.  Rendering
        them during AVP control is both unnecessary and a direct source of
        display jitter.
        """

        self._flush_observation_sender()
        scene = getattr(task_env, "scene", None)
        sensors = getattr(scene, "sensors", None)
        if not isinstance(sensors, dict):
            raise RuntimeError("AVP teleoperation requires direct access to Isaac Lab sensors")
        # Camera rendering is the limiting part of the simulator loop.  The
        # display cadence is intentionally independent of the 30 Hz dataset
        # cadence.  The sender queue has length one and drops replaced frames,
        # so this trades FPS for fresh images rather than accumulating delay.
        active_period = 1.0 / self._preview_hz
        periods = {
            "first_person_camera": active_period,
            "head_right_camera": active_period,
            "left_hand_camera": self._INACTIVE_CAMERA_UPDATE_PERIOD_S,
            "right_hand_camera": self._INACTIVE_CAMERA_UPDATE_PERIOD_S,
            "global_camera": self._INACTIVE_CAMERA_UPDATE_PERIOD_S,
        }
        missing = sorted(set(periods) - set(sensors))
        if missing:
            raise RuntimeError(f"AVP camera sensors are missing: {missing}")
        for name, period in periods.items():
            sensors[name].cfg.update_period = period
        self._camera_delay_queue.clear()
        print(
            "[AvpTeleopPolicy] camera capture mode: "
            f"recording={recording}, preview_hz={self._preview_hz}, "
            "transport_queue=latest_only, wrist_enabled=false, "
            "global_review_hz=0.0 (offline replay)",
            flush=True,
        )

    def _safe_action(
        self, observation: dict[str, Any], joint_values: np.ndarray
    ) -> torch.Tensor:
        command, received_ns = self._command_snapshot()
        body, _velocity = self._body_state_from_joint_vector(joint_values)
        measured_arm = body[15:29]
        measured_hand = self._dex1_opening(joint_values)
        tracking = command is not None and command.mode.value in {"track", "hold"}
        safe = self._safety.apply(
            command,
            measured_arm_position_rad=measured_arm,
            measured_dex1_opening_fraction=measured_hand,
            now_ns=time.monotonic_ns(),
            last_command_ns=received_ns,
            tracking=tracking,
        )
        requested_arm = np.asarray(safe.arm_position_rad, dtype=np.float64)
        requested_hand = np.asarray(safe.dex1_opening_fraction, dtype=np.float64)
        if safe.watchdog is self._WatchdogState.ACTIVE:
            self._sim_hold_arm = requested_arm.copy()
            self._sim_hold_hand = requested_hand.copy()
        else:
            # A measured-position target refreshed every tick follows gravity
            # instead of holding the robot. Latch once on STOP, while HOLD
            # retains the last safe target, so a stale stream cannot continue
            # motion or let the arms sag before tracking resumes.
            latch_current = (
                self._sim_hold_arm is None
                or self._sim_hold_hand is None
                or (
                    safe.watchdog is self._WatchdogState.STOP
                    and self._last_watchdog_state != self._WatchdogState.STOP.value
                )
            )
            if latch_current:
                self._sim_hold_arm = measured_arm.copy()
                self._sim_hold_hand = measured_hand.copy()
            requested_arm = self._sim_hold_arm.copy()
            requested_hand = self._sim_hold_hand.copy()
            self._safety.reset(requested_arm, requested_hand)
        if safe.watchdog is self._WatchdogState.ACTIVE:
            randomization = self._latest_sim_diagnostics.get("randomization", {})
            control = randomization.get("control", {}) if isinstance(randomization, dict) else {}
            delay_steps = int(control.get("action_delay_steps", 0))
            if not 0 <= delay_steps <= 2:
                raise ValueError("sim teleoperation action delay must be in [0,2] steps")
            self._action_delay_queue.append(
                (command.sequence, requested_arm.copy(), requested_hand.copy())
            )
            while len(self._action_delay_queue) > 3:
                self._action_delay_queue.popleft()
            selected_delay = min(delay_steps, len(self._action_delay_queue) - 1)
            selected_sequence, selected_arm, selected_hand = (
                self._action_delay_queue[-1 - selected_delay]
            )
            self._last_applied_command_sequence = selected_sequence
            self._applied_arm = selected_arm.copy()
            self._applied_hand = selected_hand.copy()
        else:
            self._action_delay_queue.clear()
            self._last_applied_command_sequence = (
                -1 if command is None else command.sequence
            )
            self._applied_arm = requested_arm
            self._applied_hand = requested_hand
        self._last_watchdog_state = safe.watchdog.value
        self._last_arm_tracking_error_rad = float(
            np.max(np.abs(self._applied_arm - measured_arm))
        )
        self._last_hand_tracking_error_fraction = float(
            np.max(np.abs(self._applied_hand - measured_hand))
        )
        action = _joint_position_hold_action(observation, self.device)
        action[:, :14] = torch.as_tensor(
            self._applied_arm, dtype=action.dtype, device=action.device
        )
        action[:, 14:16] = torch.as_tensor(
            1.0 - 2.0 * self._applied_hand,
            dtype=action.dtype,
            device=action.device,
        )
        return action

    def _consume_event(self):
        command, _received_ns = self._command_snapshot()
        if command is None or command.sequence <= self._last_event_sequence:
            return self._ControlEvent.NONE
        self._last_event_sequence = command.sequence
        return command.event

    def _reset_sim_episode(self, task_env: Any):
        self._flush_observation_sender()
        reset_result = task_env.reset()
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self._previous_body_position = None
        self._episode_terminated = False
        self._sim_recording = False
        self._next_review_video_time = 0.0
        self._current_episode_success = False
        self._action_delay_queue.clear()
        self._camera_delay_queue.clear()
        self._latest_sim_diagnostics = {}
        self._sim_hold_arm = None
        self._sim_hold_hand = None
        self._set_camera_capture_mode(task_env, recording=False)
        with self._command_lock:
            self._latest_command = None
            self._last_command_received_ns = None
        return observation

    @staticmethod
    def _initial_table_motion(diagnostics: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        table = diagnostics.get("white_table")
        if not isinstance(table, dict):
            raise RuntimeError("teleoperation diagnostics omit the white table")
        position = np.asarray(table.get("position_world_m"), dtype=np.float64)
        linear_velocity = np.asarray(table.get("linear_velocity_m_s"), dtype=np.float64)
        angular_velocity = np.asarray(table.get("angular_velocity_rad_s"), dtype=np.float64)
        if (
            position.shape != (3,)
            or linear_velocity.shape != (3,)
            or angular_velocity.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(linear_velocity).all()
            or not np.isfinite(angular_velocity).all()
        ):
            raise RuntimeError("white-table preflight diagnostics are invalid")
        return (
            position,
            float(np.linalg.norm(linear_velocity)),
            float(np.linalg.norm(angular_velocity)),
        )

    def _preflight_initial_scene(self, task_env: Any, observation: dict[str, Any]):
        """Reject a reset that moves before the operator receives control."""

        settle_steps = int(os.environ.get("FLIP_TABLE_TELEOP_PREFLIGHT_STEPS", "50"))
        max_attempts = int(os.environ.get("FLIP_TABLE_TELEOP_PREFLIGHT_ATTEMPTS", "8"))
        max_displacement_m = float(
            os.environ.get("FLIP_TABLE_TELEOP_PREFLIGHT_MAX_DISPLACEMENT_M", "0.03")
        )
        max_linear_speed_m_s = float(
            # The assembled table is constrained to the kinematic workbench.
            # Its measured endpoint displacement during a 50-step settle is
            # only a few millimetres, but PhysX reports short constraint
            # correction velocities around 0.11 m/s.  Keep this gate below
            # the task's final success stability limit (0.15 m/s), while
            # retaining the independent 3 cm displacement and 0.20 rad/s
            # angular-motion checks that reject a genuinely moving reset.
            os.environ.get("FLIP_TABLE_TELEOP_PREFLIGHT_MAX_LINEAR_SPEED_M_S", "0.12")
        )
        max_angular_speed_rad_s = float(
            os.environ.get("FLIP_TABLE_TELEOP_PREFLIGHT_MAX_ANGULAR_SPEED_RAD_S", "0.20")
        )
        if settle_steps < 1 or max_attempts < 1:
            raise ValueError("teleoperation preflight steps and attempts must be positive")
        thresholds = (
            max_displacement_m,
            max_linear_speed_m_s,
            max_angular_speed_rad_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("teleoperation preflight thresholds must be finite and non-negative")

        for attempt in range(1, max_attempts + 1):
            initial_position, _, _ = self._initial_table_motion(
                self._diagnostics(task_env)
            )
            fixed_hold_action = _joint_position_hold_action(observation, self.device)
            terminated_early = False
            for _ in range(settle_steps):
                observation, _, terminated, _, _ = task_env.step(fixed_hold_action)
                if _terminated_any(terminated):
                    terminated_early = True
                    break
            final_diagnostics = self._diagnostics(task_env)
            final_position, linear_speed, angular_speed = self._initial_table_motion(
                final_diagnostics
            )
            displacement = float(np.linalg.norm(final_position - initial_position))
            stable = (
                not terminated_early
                and displacement <= max_displacement_m
                and linear_speed <= max_linear_speed_m_s
                and angular_speed <= max_angular_speed_rad_s
            )
            print(
                "[AvpTeleopPolicy] initial-scene preflight: "
                f"attempt={attempt}/{max_attempts}, stable={stable}, "
                f"displacement_m={displacement:.4f}, "
                f"linear_speed_m_s={linear_speed:.4f}, "
                f"angular_speed_rad_s={angular_speed:.4f}",
                flush=True,
            )
            if stable:
                self._latest_sim_diagnostics = final_diagnostics
                return observation
            if attempt < max_attempts:
                observation = self._reset_sim_episode(task_env)
        raise RuntimeError(
            "could not obtain a stationary assembled-table reset for AVP teleoperation"
        )

    def _reset_stable_sim_episode(self, task_env: Any):
        return self._preflight_initial_scene(
            task_env, self._reset_sim_episode(task_env)
        )

    def eval(
        self,
        task_env: Any,
        observation: dict[str, Any],
        usr_args: dict[str, Any],
        video_writer: Any,
    ):
        persistent = self._persistent_sessions_enabled()
        while True:
            self._accept_ready_client()
            self._start_observation_sender()
            self._set_camera_capture_mode(task_env, recording=False)
            observation = self._preflight_initial_scene(task_env, observation)
            self._warm_live_camera_pipeline(task_env)
            initial_hold_action = _joint_position_hold_action(observation, self.device)
            if initial_hold_action.shape[-1] != self.action_dim:
                raise RuntimeError(
                    "teleoperation action contract mismatch: "
                    f"expected {self.action_dim}, constructed {initial_hold_action.shape[-1]}"
                )
            control_hz = _runtime_control_hz(task_env)
            if not math.isclose(
                control_hz, self.teleop_config.rates.servo_hz, abs_tol=1.0e-6
            ):
                raise RuntimeError(
                    f"teleop requires 50 Hz simulator control, got {control_hz:.3f} Hz"
                )
            period_s = 1.0 / control_hz
            camera_period_s = 1.0 / self._preview_hz
            next_camera_time = time.monotonic()
            ever_success = np.zeros(1, dtype=bool)
            timeout_steps = int(usr_args["time_out_limit"])
            for step in range(timeout_steps):
                started = time.monotonic()
                joint_tensor = _joint_pos_from_observation(observation, self.device)
                joint_values = joint_tensor[0].detach().cpu().double().numpy()
                now = time.monotonic()
                camera_due = now >= next_camera_time
                if camera_due:
                    self._send_observation(task_env, observation, joint_values)
                    next_camera_time += camera_period_s
                    if next_camera_time <= now:
                        next_camera_time = now + camera_period_s

                event = self._consume_event()
                if event is self._ControlEvent.QUIT:
                    break
                if event is self._ControlEvent.DISCARD_RESET:
                    observation = self._reset_stable_sim_episode(task_env)
                    continue
                if event is self._ControlEvent.RECORD_TOGGLE:
                    if self._sim_recording:
                        self._sim_recording = False
                        self._next_review_video_time = 0.0
                        self._set_camera_capture_mode(task_env, recording=False)
                        if self._current_episode_success:
                            observation = self._reset_stable_sim_episode(task_env)
                            continue
                    else:
                        self._sim_recording = True
                        self._next_review_video_time = time.monotonic()
                        self._set_camera_capture_mode(task_env, recording=True)

                if self._episode_terminated:
                    time.sleep(max(0.0, period_s - (time.monotonic() - started)))
                    continue

                action = self._safe_action(observation, joint_values)
                step_started = time.monotonic()
                observation, _, terminated, _, extras = task_env.step(action)
                self._sample_force_diagnostics(task_env)
                task_env_step_elapsed_s = time.monotonic() - step_started
                self._control_step_times.append(time.monotonic())
                step_success = _success_mask(extras, terminated, 1)
                ever_success |= step_success
                self._current_episode_success |= bool(step_success[0])
                self._episode_terminated = _terminated_any(terminated)
                _maybe_save_camera_frames(self, observation, usr_args, step)
                review_now = time.monotonic()
                if self._sim_recording and review_now >= self._next_review_video_time:
                    # Keep a clean simulator-side review video of native RGB.
                    # It is intentionally separate from RawEpisodeWriter:
                    # policy images may be randomized, review video is not.
                    self.add_video_frame(
                        video_writer,
                        observation,
                        usr_args.get("record_camera", []),
                    )
                    self._next_review_video_time += 1.0 / self._review_video_hz
                    if self._next_review_video_time <= review_now:
                        self._next_review_video_time = (
                            review_now + 1.0 / self._review_video_hz
                        )
                if self._receiver_error is not None:
                    print(
                        f"[AvpTeleopPolicy] command connection closed: {self._receiver_error}",
                        flush=True,
                    )
                    break
                if self._observation_sender_error is not None:
                    raise RuntimeError(
                        f"AVP observation sender failed: {self._observation_sender_error}"
                    )
                if step > 0 and step % 100 == 0 and len(self._control_step_times) > 1:
                    control_window_s = (
                        self._control_step_times[-1] - self._control_step_times[0]
                    )
                    measured_hz = (
                        (len(self._control_step_times) - 1) / control_window_s
                        if control_window_s > 0.0
                        else 0.0
                    )
                    print(
                        "[AvpTeleopPolicy] control timing: "
                        f"step={step}, task_env_step_s={task_env_step_elapsed_s:.3f}, "
                        f"measured_hz={measured_hz:.2f}, "
                        f"watchdog={self._last_watchdog_state}, "
                        f"arm_error_max_rad={self._last_arm_tracking_error_rad:.4f}, "
                        f"hand_error_max_fraction={self._last_hand_tracking_error_fraction:.4f}, "
                        "dex1_drive_max_n="
                        f"{self._force_audit['drive_max_n'].round(3).tolist()}, "
                        "gripper_contact_max_n="
                        f"{self._force_audit['contact_max_n'].round(3).tolist()}",
                        flush=True,
                    )
                time.sleep(max(0.0, period_s - (time.monotonic() - started)))
            self._flush_observation_sender()
            if not persistent:
                return ever_success
            self._release_client()
            print(
                "[AvpTeleopPolicy] session ended safely; returning the queued job "
                "so the persistent Isaac worker can accept the next job.",
                flush=True,
            )
            # A persistent *worker* reuses only SimulationApp.  Keeping this
            # completed job inside its own accept loop leaves ready.json in
            # ``running`` state and blocks every later queued evaluation.  The
            # worker performs the next job's fresh environment reset itself.
            return ever_success

    def reset_model(self) -> None:
        self._release_client()
        with self._command_lock:
            self._latest_command = None
            self._last_command_received_ns = None
        self._previous_body_position = None
        self._sim_recording = False
        self._next_review_video_time = 0.0
        self._episode_terminated = False
        self._current_episode_success = False
        self._action_delay_queue.clear()
        self._camera_delay_queue.clear()
        self._sim_hold_arm = None
        self._sim_hold_hand = None
        self._latest_sim_diagnostics = {}
        self._force_audit = self._new_force_audit()


class LeRobotACTPolicy(BasePolicy):
    """LeRobot ACT adapter for the Team RAMEN upper-body joint policy.

    New checkpoints observe waist3+arms14+hands2 (19-D), but output only
    arms14+hands2 (16-D).  The Team RAMEN action adapter gives these targets to
    the unmodified organizer WBC, which exclusively owns waist, legs and base.
    """

    _ACT_STATE_DIM = 19
    _ACT_ACTION_DIM = 16
    _WBC_ACTION_DIM = 16
    _DEX1_OPEN_POS = 0.0245
    _DEX1_CLOSE_POS = -0.02
    _POLICY_HAND_CLOSED = 0.0
    _POLICY_HAND_OPEN = 4.5
    _POLICY_HAND_MIN = 0.0
    _POLICY_HAND_MAX = 4.5
    _DEFAULT_N_ACTION_STEPS = 10
    _DEFAULT_POLICY_HZ = 30.0
    _DEFAULT_SIM_CONTROL_HZ = 50.0
    # Across all contiguous 30 Hz flip-table demonstrations, the largest-joint
    # per-frame velocity is 2.76 rad/s at p99 and acceleration is 78.3 rad/s^2
    # at p99. Keep learned targets inside that demonstrated controller envelope
    # instead of using the much looser motor-side 20 rad/s ceiling.
    _REAL_ARM_TARGET_VELOCITY_RAD_S = 3.0
    _DEFAULT_TARGET_ACCELERATION_RAD_S2 = 75.0
    # Measured from all contiguous 30 Hz flip-table demonstration frames. The
    # largest 0..4.5 Dex1 command change is about 0.707 per frame; these limits
    # keep deployment inside that demonstrated target-rate envelope.
    _REAL_HAND_TARGET_VELOCITY_COMMAND_S = 20.0
    _DEFAULT_HAND_TARGET_ACCELERATION_COMMAND_S2 = 400.0

    _UPPER_BODY_JOINT_NAMES = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    _LEFT_ARM_JOINT_NAMES = _UPPER_BODY_JOINT_NAMES[3:10]
    _RIGHT_ARM_JOINT_NAMES = _UPPER_BODY_JOINT_NAMES[10:17]
    _PINOCCHIO_MODEL_RELATIVE = Path(
        "robofinals/core/mdp/actions/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf"
    )
    _G1_GRIPPER_33_JOINT_ORDER = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
    )
    _G1_BODY_29_JOINT_ORDER = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    def get_model(self, usr_args: dict[str, Any]) -> None:
        self.device = torch.device(
            os.environ.get("FLIP_TABLE_ACT_DEVICE")
            or usr_args.get("device")
            or usr_args.get("env_cfg", {}).get("device", "cuda:0")
        )
        self.checkpoint = usr_args["checkpoint"]
        self.camera_mapping = usr_args.get(
            "act_camera_mapping",
            {
                "observation.images.head_left": "first_person_camera",
                "observation.images.left_wrist": "left_hand_camera",
                "observation.images.right_wrist": "right_hand_camera",
            },
        )
        expected_camera_keys = {
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        }
        if not isinstance(self.camera_mapping, dict) or set(self.camera_mapping) != expected_camera_keys:
            raise ValueError(
                "act_camera_mapping must map exactly head_left, left_wrist, and right_wrist; "
                f"got {self.camera_mapping!r}"
            )
        if not all(isinstance(value, str) and value for value in self.camera_mapping.values()):
            raise ValueError(f"act_camera_mapping values must be non-empty camera names: {self.camera_mapping!r}")
        if len(set(self.camera_mapping.values())) != len(self.camera_mapping):
            raise ValueError(f"act_camera_mapping contains aliased simulator cameras: {self.camera_mapping!r}")
        self.state_source = usr_args.get("act_state_source", "joint_pos")
        self.state_indices = usr_args.get("act_state_indices", [])
        self.gripper_state_source = usr_args.get("act_gripper_state_source", "gripper_pos")
        self.n_action_steps = int(
            os.environ.get("FLIP_TABLE_ACT_N_ACTION_STEPS", str(self._DEFAULT_N_ACTION_STEPS))
        )
        self.policy_hz = float(os.environ.get("FLIP_TABLE_ACT_POLICY_HZ", str(self._DEFAULT_POLICY_HZ)))
        self.sim_control_hz = float(
            os.environ.get("FLIP_TABLE_ACT_SIM_CONTROL_HZ", str(self._DEFAULT_SIM_CONTROL_HZ))
        )
        self.target_velocity_scale = float(
            os.environ.get("FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE", "1.0")
        )
        self.target_acceleration_rad_s2 = float(
            os.environ.get(
                "FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2",
                str(self._DEFAULT_TARGET_ACCELERATION_RAD_S2),
            )
        )
        if self.n_action_steps < 1:
            raise ValueError("FLIP_TABLE_ACT_N_ACTION_STEPS must be positive")
        if self.policy_hz <= 0 or self.sim_control_hz <= 0:
            raise ValueError("ACT policy and simulator control rates must be positive")
        if self.target_velocity_scale <= 0 or self.target_acceleration_rad_s2 <= 0:
            raise ValueError(
                "ACT target velocity scale and acceleration limit must be positive"
            )
        self.convert_dex1_hand = _env_bool("FLIP_TABLE_ACT_CONVERT_DEX1_HAND", True)
        self.debug_steps = int(os.environ.get("FLIP_TABLE_ACT_DEBUG_STEPS", "0"))
        self.debug_every = max(1, int(os.environ.get("FLIP_TABLE_ACT_DEBUG_EVERY", "1")))
        self.debug_action_terms = _env_bool("FLIP_TABLE_ACT_DEBUG_ACTION_TERMS", False)
        self._debug_logged_obs_keys = False
        self._last_normalized_action: torch.Tensor | None = None
        self._last_raw_action: torch.Tensor | None = None
        self._last_wbc_action: torch.Tensor | None = None
        self._last_env_action: torch.Tensor | None = None
        self._last_safe_target: torch.Tensor | None = None
        self._last_safe_velocity: torch.Tensor | None = None
        self._last_safe_hand_target: torch.Tensor | None = None
        self._last_safe_hand_velocity: torch.Tensor | None = None
        self._safety_clip_count = 0
        self.log_safety_clips = _env_bool("FLIP_TABLE_ACT_LOG_SAFETY_CLIPS", False)
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_model_inference = False
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._pin_joint_indices: dict[str, int] = {}
        self._pin_frame_ids: dict[str, int] = {}
        self.input_state_dim = self._ACT_STATE_DIM
        self.output_action_dim = self._ACT_ACTION_DIM
        self._load_checkpoint_feature_dims()

        ACTPolicy = self._import_act_policy_class()
        self.model = self._load_act_model(ACTPolicy)
        self._configure_action_schedule()
        self._load_processor_stats()
        self._ensure_fk_model()

    def _load_checkpoint_feature_dims(self) -> None:
        config_path = Path(str(self.checkpoint)) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"ACT checkpoint config is missing: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ACT checkpoint config: {config_path}") from exc

        if config.get("type") != "act":
            raise ValueError(f"checkpoint policy type must be 'act', got {config.get('type')!r}")

        def feature_dim(group: str, key: str) -> int:
            shape = config.get(group, {}).get(key, {}).get("shape")
            if not isinstance(shape, (list, tuple)) or not shape:
                raise ValueError(f"invalid {group}.{key}.shape in {config_path}: {shape!r}")
            dimension = int(shape[-1])
            if dimension <= 0:
                raise ValueError(f"invalid {group}.{key} dimension in {config_path}: {dimension}")
            return dimension

        self.input_state_dim = feature_dim("input_features", "observation.state")
        self.output_action_dim = feature_dim("output_features", "action")
        if self.input_state_dim != self._ACT_STATE_DIM or self.output_action_dim != self._ACT_ACTION_DIM:
            raise ValueError(
                "LeRobotACTPolicy requires state=19 and arm/hand action=16; "
                f"checkpoint has state={self.input_state_dim}, action={self.output_action_dim}"
            )
        checkpoint_cameras = {
            key
            for key in config.get("input_features", {})
            if key.startswith("observation.images.")
        }
        expected_cameras = set(self.camera_mapping)
        if checkpoint_cameras != expected_cameras:
            raise ValueError(
                "ACT checkpoint cameras must be exactly head_left, left_wrist, and right_wrist; "
                f"got {sorted(checkpoint_cameras)}"
            )
        for camera_key in expected_cameras:
            shape = config["input_features"][camera_key].get("shape")
            if shape != [3, 480, 640]:
                raise ValueError(f"ACT checkpoint {camera_key} must have shape [3,480,640], got {shape}")
        normalization = config.get("normalization_mapping")
        expected_normalization = {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
        if normalization != expected_normalization:
            raise ValueError(
                "ACT simulator adapter supports the training contract's MEAN_STD normalization only; "
                f"got {normalization!r}"
            )

    def _import_act_policy_class(self) -> type:
        if _env_bool("FLIP_TABLE_ACT_USE_STANDARD_LEROBOT_IMPORT", False):
            from lerobot.policies.act.modeling_act import ACTPolicy

            return ACTPolicy
        return self._import_act_policy_without_policy_init()

    def _import_act_policy_without_policy_init(self) -> type:
        """Load ACT without importing lerobot.policies.__init__.

        Some RoboFinals V1 images bundle a LeRobot install where
        ``lerobot.policies.__init__`` imports GR00T, which imports transformers,
        which then fails if the environment has a newer ``huggingface-hub`` than
        that transformers build accepts.  ACT itself does not need transformers,
        so load the required policy submodules through a lightweight package
        shim instead of mutating the conda environment.
        """
        import importlib
        import lerobot

        lerobot_root = Path(next(iter(lerobot.__path__)))
        policies_path = lerobot_root / "policies"
        act_path = policies_path / "act"
        for module_name in list(sys.modules):
            if module_name == "lerobot.policies" or module_name.startswith("lerobot.policies."):
                sys.modules.pop(module_name, None)

        policies_pkg = types.ModuleType("lerobot.policies")
        policies_pkg.__path__ = [str(policies_path)]
        policies_pkg.__package__ = "lerobot"
        sys.modules["lerobot.policies"] = policies_pkg

        act_pkg = types.ModuleType("lerobot.policies.act")
        act_pkg.__path__ = [str(act_path)]
        act_pkg.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies.act"] = act_pkg

        module = importlib.import_module("lerobot.policies.act.modeling_act")
        return module.ACTPolicy

    def _load_act_model(self, policy_class: type) -> Any:
        import dataclasses
        import draccus
        from safetensors.torch import load_model as load_model_as_safetensor

        checkpoint = Path(str(self.checkpoint))
        config_file_path = checkpoint / "config.json"
        model_file = checkpoint / "model.safetensors"
        if not config_file_path.is_file() or not model_file.is_file():
            raise FileNotFoundError(
                f"ACT checkpoint requires config.json and model.safetensors: {checkpoint}"
            )
        raw_config = json.loads(config_file_path.read_text(encoding="utf-8"))
        config_class = policy_class.config_class
        valid_fields = {field.name for field in dataclasses.fields(config_class)}
        filtered_config = {key: value for key, value in raw_config.items() if key in valid_fields}
        if "device" in valid_fields:
            filtered_config["device"] = str(self.device)

        with NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as config_file:
            json.dump(filtered_config, config_file)
            config_path = config_file.name
        try:
            config = draccus.parse(config_class, config_path, args=[])
        finally:
            Path(config_path).unlink(missing_ok=True)

        model = policy_class(config)
        load_kwargs = {"strict": True}
        try:
            result = load_model_as_safetensor(
                model,
                str(model_file),
                device=str(self.device),
                **load_kwargs,
            )
        except TypeError:
            result = load_model_as_safetensor(model, str(model_file), **load_kwargs)
        missing, unexpected = result if result is not None else ((), ())
        if missing or unexpected:
            raise RuntimeError(
                "ACT checkpoint does not exactly match its config: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.to(self.device)
        model.eval()
        return model

    def _configure_action_schedule(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None or not hasattr(config, "n_action_steps"):
            raise AttributeError("loaded ACT model does not expose config.n_action_steps")
        chunk_size = int(getattr(config, "chunk_size", self.n_action_steps))
        if self.n_action_steps > chunk_size:
            raise ValueError(
                f"FLIP_TABLE_ACT_N_ACTION_STEPS={self.n_action_steps} exceeds checkpoint chunk_size={chunk_size}"
            )
        if getattr(config, "temporal_ensemble_coeff", None) is not None and self.n_action_steps != 1:
            raise ValueError("temporal ensembling requires FLIP_TABLE_ACT_N_ACTION_STEPS=1")
        checkpoint_steps = int(config.n_action_steps)
        config.n_action_steps = self.n_action_steps
        self.model.reset()
        print(
            "[LeRobotACTPolicy] runtime schedule: "
            f"checkpoint_n_action_steps={checkpoint_steps}, "
            f"runtime_n_action_steps={self.n_action_steps}, "
            f"policy_hz={self.policy_hz:.3f}, sim_control_hz={self.sim_control_hz:.3f}"
        )

    def _policy_velocity_limits(self) -> torch.Tensor:
        limits = torch.full(
            (14,),
            self._REAL_ARM_TARGET_VELOCITY_RAD_S * self.target_velocity_scale,
            dtype=torch.float32,
            device=self.device,
        )
        if self._pin_model is None:
            return limits
        urdf_limits = getattr(self._pin_model, "velocityLimit", None)
        if urdf_limits is None:
            return limits
        for offset, name in enumerate(self._UPPER_BODY_JOINT_NAMES[3:17]):
            index = self._pin_joint_indices.get(name)
            if index is None or index >= len(urdf_limits):
                continue
            urdf_limit = float(urdf_limits[index])
            if math.isfinite(urdf_limit) and urdf_limit > 0:
                limits[offset] = min(float(limits[offset]), urdf_limit * self.target_velocity_scale)
        return limits

    def _clip_policy_joint_targets(
        self,
        action: torch.Tensor,
        current_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] < 14:
            return action
        clipped = action.clone()
        if current_state is not None:
            if current_state.ndim == 1:
                current_state = current_state.unsqueeze(0)
            current_state = current_state.to(device=action.device, dtype=action.dtype)
            if current_state.shape[0] == 1 and action.shape[0] > 1:
                current_state = current_state.expand(action.shape[0], -1)

        finite_action = torch.isfinite(clipped)
        if current_state is not None and current_state.shape[-1] >= 19:
            current_arms = current_state[:, 3:17]
            clipped[:, :14] = torch.where(
                finite_action[:, :14],
                clipped[:, :14],
                current_arms,
            )
        else:
            clipped[:, :14] = torch.nan_to_num(
                clipped[:, :14], nan=0.0, posinf=0.0, neginf=0.0
            )
        if clipped.shape[-1] >= 16:
            clipped[:, 14:16] = torch.nan_to_num(
                clipped[:, 14:16],
                nan=self._POLICY_HAND_OPEN,
                posinf=self._POLICY_HAND_MAX,
                neginf=self._POLICY_HAND_MIN,
            ).clamp(self._POLICY_HAND_MIN, self._POLICY_HAND_MAX)

        clipped_count = 0
        if self._pin_model is not None:
            lower = self._pin_model.lowerPositionLimit
            upper = self._pin_model.upperPositionLimit
            for offset, name in enumerate(self._UPPER_BODY_JOINT_NAMES[3:17]):
                index = self._pin_joint_indices.get(name)
                if index is None:
                    continue
                low = float(lower[index])
                high = float(upper[index])
                if not (math.isfinite(low) and math.isfinite(high) and low < high):
                    continue
                before = clipped[:, offset].clone()
                clipped[:, offset] = clipped[:, offset].clamp(low, high)
                clipped_count += int(torch.count_nonzero(before != clipped[:, offset]).item())

        # Absolute targets are filtered at the same rate at which the policy
        # is refreshed. This is the software-side counterpart of Unitree's
        # low-level target clipping and prevents a bad chunk from becoming a
        # large one-frame command at the real robot interface.
        if current_state is not None and current_state.shape[-1] >= 19:
            dt = 1.0 / self.policy_hz
            velocity_limits = self._policy_velocity_limits().to(dtype=clipped.dtype)
            current = current_state[:, 3:17]
            candidate = current + (clipped[:, :14] - current).clamp(
                -velocity_limits * dt, velocity_limits * dt
            )
            if self._last_safe_target is not None and self._last_safe_target.shape == candidate.shape:
                previous_target = self._last_safe_target.to(device=clipped.device, dtype=clipped.dtype)
                previous_velocity = self._last_safe_velocity
                if previous_velocity is None or previous_velocity.shape != candidate.shape:
                    previous_velocity = torch.zeros_like(candidate)
                else:
                    previous_velocity = previous_velocity.to(device=clipped.device, dtype=clipped.dtype)
            else:
                # Start from the measured state with zero command velocity so
                # the very first policy target is acceleration-limited too.
                previous_target = current
                previous_velocity = torch.zeros_like(candidate)
            desired_velocity = (candidate - previous_target) / dt
            acceleration_step = self.target_acceleration_rad_s2 * dt
            velocity = previous_velocity + (
                desired_velocity - previous_velocity
            ).clamp(-acceleration_step, acceleration_step)
            velocity = velocity.clamp(-velocity_limits, velocity_limits)
            candidate = previous_target + velocity * dt
            self._last_safe_velocity = velocity.detach()
            clipped_count += int(torch.count_nonzero(candidate != clipped[:, :14]).item())
            clipped[:, :14] = candidate
            self._last_safe_target = candidate.detach()

        if (
            clipped.shape[-1] >= 16
            and current_state is not None
            and current_state.shape[-1] >= 19
        ):
            dt = 1.0 / self.policy_hz
            hand_velocity_limit = self._REAL_HAND_TARGET_VELOCITY_COMMAND_S
            hand_acceleration_step = (
                self._DEFAULT_HAND_TARGET_ACCELERATION_COMMAND_S2 * dt
            )
            hand_current = current_state[:, 17:19]
            hand_candidate = hand_current + (
                clipped[:, 14:16] - hand_current
            ).clamp(-hand_velocity_limit * dt, hand_velocity_limit * dt)
            previous_hand_target = getattr(self, "_last_safe_hand_target", None)
            previous_hand_velocity = getattr(self, "_last_safe_hand_velocity", None)
            if previous_hand_target is None or previous_hand_target.shape != hand_candidate.shape:
                previous_hand_target = hand_current
                previous_hand_velocity = torch.zeros_like(hand_candidate)
            else:
                previous_hand_target = previous_hand_target.to(
                    device=clipped.device,
                    dtype=clipped.dtype,
                )
                if previous_hand_velocity is None or previous_hand_velocity.shape != hand_candidate.shape:
                    previous_hand_velocity = torch.zeros_like(hand_candidate)
                else:
                    previous_hand_velocity = previous_hand_velocity.to(
                        device=clipped.device,
                        dtype=clipped.dtype,
                    )
            desired_hand_velocity = (hand_candidate - previous_hand_target) / dt
            hand_velocity = previous_hand_velocity + (
                desired_hand_velocity - previous_hand_velocity
            ).clamp(-hand_acceleration_step, hand_acceleration_step)
            hand_velocity = hand_velocity.clamp(
                -hand_velocity_limit,
                hand_velocity_limit,
            )
            hand_candidate = previous_hand_target + hand_velocity * dt
            clipped_count += int(
                torch.count_nonzero(hand_candidate != clipped[:, 14:16]).item()
            )
            clipped[:, 14:16] = hand_candidate.clamp(
                self._POLICY_HAND_MIN,
                self._POLICY_HAND_MAX,
            )
            self._last_safe_hand_target = clipped[:, 14:16].detach()
            self._last_safe_hand_velocity = hand_velocity.detach()

        if clipped_count:
            self._safety_clip_count += clipped_count
            if getattr(self, "log_safety_clips", False):
                print(
                    f"[{type(self).__name__}] safety-limited "
                    f"{clipped_count} target values (cumulative={self._safety_clip_count})"
                )
        return clipped

    def _load_processor_stats(self) -> None:
        from safetensors.torch import load_file

        checkpoint = Path(str(self.checkpoint))
        pre_path, pre_eps = self._resolve_processor_state(
            checkpoint,
            "policy_preprocessor.json",
            "normalizer_processor",
        )
        post_path, post_eps = self._resolve_processor_state(
            checkpoint,
            "policy_postprocessor.json",
            "unnormalizer_processor",
        )
        if pre_eps != post_eps:
            raise ValueError(
                "ACT preprocessor and postprocessor normalization eps values differ: "
                f"{pre_eps} != {post_eps}"
            )
        self._pre_stats = load_file(str(pre_path), device=str(self.device))
        self._post_stats = load_file(str(post_path), device=str(self.device))

        expected_pre_shapes = {
            "observation.state.mean": (self._ACT_STATE_DIM,),
            "observation.state.std": (self._ACT_STATE_DIM,),
            **{
                f"{camera_key}.{stat_name}": (3, 1, 1)
                for camera_key in self.camera_mapping
                for stat_name in ("mean", "std")
            },
        }
        expected_post_shapes = {
            "action.mean": (self._ACT_ACTION_DIM,),
            "action.std": (self._ACT_ACTION_DIM,),
        }
        self._validate_stats(self._pre_stats, expected_pre_shapes, "preprocessor")
        self._validate_stats(self._post_stats, expected_post_shapes, "postprocessor")
        self._normalizer_eps = pre_eps

    @staticmethod
    def _resolve_processor_state(
        checkpoint: Path,
        manifest_name: str,
        registry_name: str,
    ) -> tuple[Path, float]:
        manifest_path = checkpoint / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ACT checkpoint is missing processor manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse ACT processor manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("steps"), list):
            raise ValueError(f"ACT processor manifest {manifest_path} must contain a steps list")

        matching_steps = [
            step
            for step in manifest["steps"]
            if isinstance(step, dict) and step.get("registry_name") == registry_name
        ]
        if len(matching_steps) != 1:
            raise ValueError(
                f"ACT processor manifest {manifest_path} must contain exactly one "
                f"{registry_name!r} step, got {len(matching_steps)}"
            )
        step = matching_steps[0]
        state_file = step.get("state_file")
        if not isinstance(state_file, str) or not state_file.strip():
            raise ValueError(
                f"ACT processor step {registry_name!r} in {manifest_path} has no state_file"
            )
        checkpoint_root = checkpoint.resolve()
        state_path = (checkpoint / state_file).resolve()
        try:
            state_path.relative_to(checkpoint_root)
        except ValueError as exc:
            raise ValueError(
                f"ACT processor state_file must remain inside the checkpoint: {state_file!r}"
            ) from exc
        if not state_path.is_file():
            raise FileNotFoundError(f"ACT checkpoint is missing processor state: {state_path}")

        config = step.get("config")
        eps_value = config.get("eps") if isinstance(config, dict) else None
        if isinstance(eps_value, bool) or not isinstance(eps_value, (int, float)):
            raise ValueError(
                f"ACT processor step {registry_name!r} in {manifest_path} has invalid eps"
            )
        eps = float(eps_value)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                f"ACT processor step {registry_name!r} in {manifest_path} eps must be finite and > 0"
            )
        return state_path, eps

    @staticmethod
    def _validate_stats(
        stats: dict[str, torch.Tensor],
        expected_shapes: dict[str, tuple[int, ...]],
        label: str,
    ) -> None:
        missing = sorted(set(expected_shapes) - set(stats))
        if missing:
            raise ValueError(f"ACT {label} statistics are missing keys: {missing}")
        for key, expected_shape in expected_shapes.items():
            value = stats[key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"ACT {label} statistic {key!r} must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"ACT {label} statistic {key!r} contains NaN or Inf")
            if key.endswith(".std") and torch.any(value < 0):
                raise ValueError(f"ACT {label} statistic {key!r} contains a negative standard deviation")

    def _normalize_feature(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        mean_key = f"{key}.mean"
        std_key = f"{key}.std"
        if mean_key not in self._pre_stats or std_key not in self._pre_stats:
            raise KeyError(f"ACT preprocessor statistics are missing {mean_key!r} or {std_key!r}")
        mean = self._pre_stats[mean_key].to(device=tensor.device, dtype=tensor.dtype)
        std = self._pre_stats[std_key].to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / (std + self._normalizer_eps)

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self._post_stats["action.mean"].to(device=action.device, dtype=action.dtype)
        std = self._post_stats["action.std"].to(device=action.device, dtype=action.dtype)
        return action * (std + self._normalizer_eps) + mean

    def _dex1_joint_pos_to_command(self, joint_pos: torch.Tensor) -> torch.Tensor:
        normalized = (joint_pos - self._DEX1_OPEN_POS) / (self._DEX1_CLOSE_POS - self._DEX1_OPEN_POS)
        return (2.0 * normalized - 1.0).clamp(-1.0, 1.0)

    def _dex1_command_to_policy_hand(self, command: torch.Tensor) -> torch.Tensor:
        normalized = (command.clamp(-1.0, 1.0) + 1.0) * 0.5
        return self._POLICY_HAND_OPEN + normalized * (self._POLICY_HAND_CLOSED - self._POLICY_HAND_OPEN)

    def _dex1_joint_pos_to_policy_hand(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return self._dex1_command_to_policy_hand(self._dex1_joint_pos_to_command(joint_pos))

    def _policy_hand_to_dex1_command(self, hand: torch.Tensor) -> torch.Tensor:
        normalized = (hand - self._POLICY_HAND_OPEN) / (self._POLICY_HAND_CLOSED - self._POLICY_HAND_OPEN)
        return (2.0 * normalized - 1.0).clamp(-1.0, 1.0)

    def _ensure_fk_model(self) -> None:
        if self._pin_model is not None:
            return
        import pinocchio as pin

        robofinals_root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
        urdf_path = robofinals_root / self._PINOCCHIO_MODEL_RELATIVE
        if not urdf_path.exists():
            # The overlay policy is copied into /workspace/robofinals/policy.
            local_root = Path(__file__).resolve().parents[1]
            urdf_path = local_root / self._PINOCCHIO_MODEL_RELATIVE
        if not urdf_path.exists():
            raise FileNotFoundError(f"G1 FK URDF not found: {urdf_path}")

        asset_path = urdf_path.parent
        wrapper = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dirs=[str(asset_path)])
        self._pin = pin
        self._pin_model = wrapper.model
        self._pin_data = wrapper.data
        self._pin_q = np.zeros(self._pin_model.nq, dtype=np.float64)
        missing_joints = [
            name for name in self._UPPER_BODY_JOINT_NAMES if not self._pin_model.existJointName(name)
        ]
        if missing_joints:
            raise RuntimeError(f"G1 FK URDF is missing upper-body joints: {missing_joints}")
        self._pin_joint_indices = {
            name: int(self._pin_model.joints[self._pin_model.getJointId(name)].idx_q)
            for name in self._UPPER_BODY_JOINT_NAMES
        }
        frame_names = {
            "left": "left_wrist_yaw_link",
            "right": "right_wrist_yaw_link",
        }
        missing_frames = [name for name in frame_names.values() if not self._pin_model.existFrame(name)]
        if missing_frames:
            raise RuntimeError(f"G1 FK URDF is missing wrist frames: {missing_frames}")
        self._pin_frame_ids = {
            side: int(self._pin_model.getFrameId(name)) for side, name in frame_names.items()
        }

    def _merged_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for group_name in ("policy", "embodiment_general_obs"):
            group = observation.get(group_name, {})
            if isinstance(group, dict):
                merged.update(group)
        return merged

    @staticmethod
    def _merged_trace_state(observation: dict[str, Any]) -> Any:
        group = observation.get("embodiment_general_obs", {})
        if isinstance(group, dict):
            return group.get("joint_pos")
        return None

    def _image_chw(self, value: Any, *, camera_name: str) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value)
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"ACT evaluation requires one environment, got image shape {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
        channels_first = tensor.shape[0] in (3, 4)
        channels_last = tensor.shape[-1] in (3, 4)
        if channels_first == channels_last:
            raise ValueError(f"expected unambiguous RGB/RGBA image layout, got {tuple(tensor.shape)}")
        if channels_last:
            tensor = tensor[..., :3].permute(2, 0, 1)
        else:
            tensor = tensor[:3]
        if tuple(tensor.shape) != (3, 480, 640):
            raise ValueError(f"ACT camera must be unmodified 640x480 RGB, got {tuple(tensor.shape)}")
        tensor = _apply_recorded_camera_geometry_tensor(tensor, camera_name)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("ACT camera observation contains NaN or Inf")
        minimum = float(tensor.detach().min().item())
        maximum = float(tensor.detach().max().item())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(f"ACT camera values must be in [0,255], got [{minimum}, {maximum}]")
        if maximum > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _ordered_joint_indices(self, source_dim: int) -> list[int]:
        if self.state_indices:
            indices = [int(index) for index in self.state_indices]
            if len(indices) != len(self._UPPER_BODY_JOINT_NAMES):
                raise ValueError(
                    f"act_state_indices must contain exactly {len(self._UPPER_BODY_JOINT_NAMES)} entries"
                )
            if len(indices) != len(set(indices)) or any(index < 0 or index >= source_dim for index in indices):
                raise ValueError(f"act_state_indices are invalid for {source_dim}-D state: {indices}")
            return indices
        if source_dim == len(self._G1_GRIPPER_33_JOINT_ORDER):
            order = self._G1_GRIPPER_33_JOINT_ORDER
        elif source_dim == len(self._G1_BODY_29_JOINT_ORDER):
            order = self._G1_BODY_29_JOINT_ORDER
        else:
            raise ValueError(
                "joint_pos must use the named 33-D G1+Dex1 or 29-D G1 body contract; "
                f"got {source_dim} values without act_state_indices"
            )
        lookup = {name: index for index, name in enumerate(order)}
        missing = [name for name in self._UPPER_BODY_JOINT_NAMES if name not in lookup]
        if missing:
            raise RuntimeError(f"joint observation contract is missing joints: {missing}")
        return [lookup[name] for name in self._UPPER_BODY_JOINT_NAMES]

    def _state_tensor(self, merged: dict[str, Any]) -> torch.Tensor:
        source = merged.get(self.state_source)
        if source is None:
            raise ValueError(
                f"state observation {self.state_source!r} is missing; available={sorted(merged)}"
            )
        source_tensor = source.detach() if torch.is_tensor(source) else torch.as_tensor(source)
        if source_tensor.ndim == 2:
            if source_tensor.shape[0] != 1:
                raise ValueError(
                    f"ACT evaluation requires one environment, got state shape {tuple(source_tensor.shape)}"
                )
            source_tensor = source_tensor[0]
        elif source_tensor.ndim != 1:
            raise ValueError(f"ACT joint state must be [D] or [1,D], got {tuple(source_tensor.shape)}")
        source_tensor = source_tensor.to(device=self.device, dtype=torch.float32).flatten()
        if not torch.isfinite(source_tensor).all():
            raise ValueError(f"state observation {self.state_source!r} contains NaN or Inf")

        state = torch.zeros((self.input_state_dim,), dtype=torch.float32, device=self.device)
        indices = self._ordered_joint_indices(int(source_tensor.numel()))
        state[:17] = source_tensor[indices]

        hand_state = None
        if source_tensor.numel() >= 33 and self.input_state_dim >= 19:
            # Use named left/right Dex1 fingers from the full joint vector. The
            # vendor gripper_pos helper returns two right-finger-derived values.
            hand_state = torch.stack([
                source_tensor[29:31].mean(),
                source_tensor[31:33].mean(),
            ])
        else:
            gripper_source = merged.get(self.gripper_state_source)
            if gripper_source is not None and self.input_state_dim >= 19:
                gripper = gripper_source.detach() if torch.is_tensor(gripper_source) else torch.as_tensor(gripper_source)
                if gripper.ndim == 2:
                    if gripper.shape[0] != 1:
                        raise ValueError(
                            f"ACT evaluation requires one environment, got gripper shape {tuple(gripper.shape)}"
                        )
                    gripper = gripper[0]
                elif gripper.ndim != 1:
                    raise ValueError(
                        f"gripper observation must be [2] or [1,2], got {tuple(gripper.shape)}"
                    )
                gripper = gripper.to(device=self.device, dtype=torch.float32).flatten()
                if gripper.numel() != 2 or not torch.isfinite(gripper).all():
                    raise ValueError(
                        f"gripper observation must contain two finite Dex1 values, got {tuple(gripper.shape)}"
                    )
                hand_state = gripper
        if hand_state is None:
            raise ValueError(
                "19-D ACT state requires left/right Dex1 state in 33-D joint_pos "
                f"or observation {self.gripper_state_source!r}"
            )
        if self.convert_dex1_hand:
            hand_state = self._dex1_joint_pos_to_policy_hand(hand_state)
        state[17:19] = hand_state
        return state

    def encode_obs(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        merged = self._merged_obs(observation)
        batch: dict[str, torch.Tensor] = {
            "observation.state": self._normalize_feature("observation.state", self._state_tensor(merged)).unsqueeze(0),
        }
        for target_key, source_key in self.camera_mapping.items():
            camera_key = _resolve_camera_rgb_key(merged, source_key)
            image = self._normalize_feature(
                target_key,
                self._image_chw(merged[camera_key], camera_name=source_key),
            )
            batch[target_key] = image.unsqueeze(0)
        return batch

    def _to_wbc_action(self, action: torch.Tensor, env_action_dim: int) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(device=self.device, dtype=torch.float32)
        if action.shape[-1] != self.output_action_dim:
            raise ValueError(
                f"ACT output has {action.shape[-1]} values, expected {self.output_action_dim}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("ACT output contains NaN or Inf")
        if env_action_dim != self._ACT_ACTION_DIM:
            raise ValueError(
                "balanced_wbc environment must expose the canonical 16-D action; "
                f"got {env_action_dim}"
            )
        env_action = action.clone()
        if self.convert_dex1_hand:
            env_action[:, 14:16] = self._policy_hand_to_dex1_command(
                env_action[:, 14:16]
            )
        return env_action

    def get_action(
        self,
        observation: dict[str, torch.Tensor],
        env_action_dim: int,
        current_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_model_inference = self._act_model_will_infer()
        normalized_action = self.model.select_action(observation)
        action = self._unnormalize_action(normalized_action)
        action = self._clip_policy_joint_targets(action, current_state)
        wbc_action = self._to_wbc_action(action, env_action_dim)
        self._last_normalized_action = normalized_action.detach()
        self._last_raw_action = action.detach()
        self._last_wbc_action = wbc_action.detach()
        return wbc_action

    def _act_model_will_infer(self) -> bool:
        """Return whether the next ACT selection will run the neural network."""

        model = getattr(self, "model", None)
        config = getattr(model, "config", None)
        if config is None:
            return True
        if getattr(config, "temporal_ensemble_coeff", None) is not None:
            return True
        queue = getattr(model, "_action_queue", None)
        return queue is None or len(queue) == 0

    def _debug_first_row(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        value = tensor.detach().float()
        if value.ndim >= 2:
            value = value[0]
        return value.flatten()

    def _debug_stats(self, name: str, tensor: torch.Tensor | None) -> str:
        value = self._debug_first_row(tensor)
        if value is None or value.numel() == 0:
            return f"{name}=missing"
        return (
            f"{name}: shape={tuple(tensor.shape)} "
            f"min={value.min().item():+.4f} max={value.max().item():+.4f} "
            f"mean={value.mean().item():+.4f} std={value.std(unbiased=False).item():+.4f} "
            f"norm={value.norm().item():.4f}"
        )

    def _debug_values(self, tensor: torch.Tensor | None, limit: int = 23) -> list[float]:
        value = self._debug_first_row(tensor)
        if value is None:
            return []
        return [round(float(item), 4) for item in value[:limit].detach().cpu().tolist()]

    def _debug_get_action_term(self, task_env: Any, name: str) -> Any | None:
        env = task_env
        manager = getattr(env, "action_manager", None)
        if manager is None and hasattr(env, "unwrapped"):
            manager = getattr(env.unwrapped, "action_manager", None)
        if manager is None:
            return None
        try:
            return manager.get_term(name)
        except Exception:
            return None

    def _debug_log_step(
        self,
        task_env: Any,
        step: int,
        observation_before: dict[str, Any],
        observation_after: dict[str, Any],
    ) -> None:
        if self.debug_steps <= 0 or step >= self.debug_steps or step % self.debug_every != 0:
            return
        try:
            merged_before = self._merged_obs(observation_before)
            merged_after = self._merged_obs(observation_after)
            if not self._debug_logged_obs_keys:
                shapes = {
                    key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
                    for key, value in sorted(merged_before.items())
                }
                print(f"[LeRobotACTPolicy][debug] observation keys/shapes: {shapes}")
                self._debug_logged_obs_keys = True

            state_before = self._state_tensor(merged_before)
            state_after = self._state_tensor(merged_after)
            raw = self._debug_first_row(self._last_raw_action)
            wbc = self._debug_first_row(self._last_wbc_action)
            norm = self._debug_first_row(self._last_normalized_action)
            state_delta = state_after - state_before
            current_action_state = torch.cat((state_before[3:17], state_before[17:19]))
            action_delta = raw - current_action_state if raw is not None else None

            print(f"[LeRobotACTPolicy][debug] step={step}")
            print(f"  {self._debug_stats('state_before', state_before)}")
            print(f"  {self._debug_stats('normalized_action', norm)} values={self._debug_values(norm, 16)}")
            print(f"  {self._debug_stats('raw_action', raw)} values={self._debug_values(raw, 16)}")
            if action_delta is not None:
                print(
                    "  raw_action-state_before: "
                    f"max_abs={action_delta.abs().max().item():.6f} "
                    f"norm={action_delta.norm().item():.6f} "
                    f"values={self._debug_values(action_delta, 16)}"
                )
            print(f"  {self._debug_stats('env_action', wbc)} values={self._debug_values(wbc, 16)}")
            print(
                "  state_after-state_before: "
                f"max_abs={state_delta.abs().max().item():.6f} "
                f"norm={state_delta.norm().item():.6f} "
                f"values={self._debug_values(state_delta, 19)}"
            )

            if not self.debug_action_terms:
                return
            for term_name in ("base_action", "left_hand_action", "right_hand_action"):
                term = self._debug_get_action_term(task_env, term_name)
                if term is None:
                    continue
                parts = []
                for attr_name in ("raw_actions", "processed_actions", "target_robot_joints_mujoco"):
                    value = getattr(term, attr_name, None)
                    if value is None:
                        continue
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value, device=self.device)
                    parts.append(self._debug_stats(f"{term_name}.{attr_name}", value))
                if parts:
                    print("  " + " | ".join(parts))
        except Exception as exc:
            print(f"[LeRobotACTPolicy][debug] failed to log step {step}: {exc}")

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        if num_envs != 1:
            raise ValueError("LeRobotACTPolicy currently requires num_envs=1")
        _synchronize_policy_control_rate(self, task_env, attribute="sim_control_hz")
        action_dim = int(usr_args.get("actions_dim", self._WBC_ACTION_DIM))
        ever_success = np.zeros(num_envs, dtype=bool)
        self._last_env_action = None
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_model_inference = False
        trace_rows: list[dict[str, Any]] = []
        for step in range(int(usr_args["time_out_limit"])):
            observation_before = observation
            should_advance = self._last_env_action is None or self._policy_clock >= self.sim_control_hz
            current_state = self._state_tensor(self._merged_obs(observation)).unsqueeze(0)
            if should_advance:
                if self._last_env_action is None:
                    self._policy_clock = 0.0
                else:
                    self._policy_clock -= self.sim_control_hz
                encoded = self.encode_obs(observation)
                self._last_env_action = self.get_action(encoded, action_dim, current_state)
                self._action_advance_count += 1
                if self._last_model_inference:
                    self._policy_inference_count += 1
            action = self._last_env_action
            assert action is not None
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": should_advance and self._last_model_inference,
                    "policy_inference_index": self._policy_inference_count,
                    "action_advanced": should_advance,
                    "action_advance_index": self._action_advance_count,
                    "state_before": _trace_value(current_state),
                    "normalized_action": _trace_value(self._last_normalized_action),
                    "raw_target": _trace_value(self._last_raw_action),
                    "sent_action": _trace_value(action),
                    "safety_clip_count": int(getattr(self, "_safety_clip_count", 0)),
                    "state_after": _trace_value(self._state_tensor(self._merged_obs(observation))),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                }
            )
            self._debug_log_step(task_env, step, observation_before, observation)
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            self._policy_clock += self.policy_hz
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                print(
                    f"[LeRobotACTPolicy] model_inferences={self._policy_inference_count} "
                    f"action_advances={self._action_advance_count} sim_steps={step + 1}"
                )
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        print(
            f"[LeRobotACTPolicy] model_inferences={self._policy_inference_count} "
            f"action_advances={self._action_advance_count} "
            f"sim_steps={int(usr_args['time_out_limit'])}"
        )
        return ever_success

    def reset_model(self) -> None:
        if self.model is not None:
            self.model.reset()
        self._last_env_action = None
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_model_inference = False
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_hand_target = None
        self._last_safe_hand_velocity = None
        self._safety_clip_count = 0


class FlowMatchingBCPolicy(LeRobotACTPolicy):
    """Three-camera flow-matching BC with the shared 19-D G1 safety adapter."""

    _DEFAULT_N_ACTION_STEPS = 6

    def get_model(self, usr_args: dict[str, Any]) -> None:
        self.device = torch.device(
            os.environ.get("FLIP_TABLE_FLOW_DEVICE")
            or usr_args.get("device")
            or usr_args.get("env_cfg", {}).get("device", "cuda:0")
        )
        self.checkpoint = usr_args["checkpoint"]
        self.camera_mapping = usr_args.get(
            "act_camera_mapping",
            {
                "observation.images.head_left": "first_person_camera",
                "observation.images.left_wrist": "left_hand_camera",
                "observation.images.right_wrist": "right_hand_camera",
            },
        )
        expected = {
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        }
        if not isinstance(self.camera_mapping, dict) or set(self.camera_mapping) != expected:
            raise ValueError("Flow Matching cameras must be exactly head_left, left_wrist, right_wrist")
        if len(set(self.camera_mapping.values())) != len(self.camera_mapping):
            raise ValueError("Flow Matching simulator cameras must not be aliased")
        self.state_source = usr_args.get("act_state_source", "joint_pos")
        self.state_indices = usr_args.get("act_state_indices", [])
        self.gripper_state_source = usr_args.get("act_gripper_state_source", "gripper_pos")
        self.n_action_steps = int(
            os.environ.get("FLIP_TABLE_FLOW_N_ACTION_STEPS", str(self._DEFAULT_N_ACTION_STEPS))
        )
        self.policy_hz = float(os.environ.get("FLIP_TABLE_FLOW_POLICY_HZ", "30"))
        self.sim_control_hz = float(os.environ.get("FLIP_TABLE_FLOW_SIM_CONTROL_HZ", "50"))
        self.target_velocity_scale = float(
            os.environ.get("FLIP_TABLE_FLOW_TARGET_VELOCITY_SCALE", "1.0")
        )
        self.target_acceleration_rad_s2 = float(
            os.environ.get(
                "FLIP_TABLE_FLOW_TARGET_ACCELERATION_RAD_S2",
                str(self._DEFAULT_TARGET_ACCELERATION_RAD_S2),
            )
        )
        if self.n_action_steps < 1 or self.policy_hz <= 0 or self.sim_control_hz <= 0:
            raise ValueError("Flow Matching action count and control rates must be positive")
        if self.target_velocity_scale <= 0 or self.target_acceleration_rad_s2 <= 0:
            raise ValueError("Flow Matching target velocity and acceleration limits must be positive")

        self.convert_dex1_hand = _env_bool("FLIP_TABLE_ACT_CONVERT_DEX1_HAND", True)
        self.debug_steps = int(os.environ.get("FLIP_TABLE_FLOW_DEBUG_STEPS", "0"))
        self.debug_every = max(1, int(os.environ.get("FLIP_TABLE_FLOW_DEBUG_EVERY", "1")))
        self.debug_action_terms = _env_bool("FLIP_TABLE_ACT_DEBUG_ACTION_TERMS", False)
        self._debug_logged_obs_keys = False
        self._last_normalized_action = None
        self._last_raw_action = None
        self._last_wbc_action = None
        self._last_env_action = None
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_hand_target = None
        self._last_safe_hand_velocity = None
        self._safety_clip_count = 0
        self.log_safety_clips = _env_bool("FLIP_TABLE_FLOW_LOG_SAFETY_CLIPS", False)
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_model_inference = False
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._pin_joint_indices = {}
        self._pin_frame_ids = {}
        self.input_state_dim = self._ACT_STATE_DIM
        self.output_action_dim = self._ACT_ACTION_DIM
        self._action_queue: deque[torch.Tensor] = deque()

        from .flow_matching import FlowMatchingPolicy

        self.model = FlowMatchingPolicy.from_pretrained(self.checkpoint, device=self.device)
        if self.model.config.state_dim != 19 or self.model.config.action_dim != 16:
            raise ValueError("Flow Matching checkpoint must use state=19, action=16")
        if self.n_action_steps > self.model.config.action_horizon:
            raise ValueError(
                f"FLIP_TABLE_FLOW_N_ACTION_STEPS={self.n_action_steps} exceeds "
                f"checkpoint horizon={self.model.config.action_horizon}"
            )
        # Residual deployment can move beyond the BC dataset envelope. Load
        # the organizer WBC URDF so the shared adapter enforces its exact joint
        # position limits as well as velocity and acceleration limits.
        self._ensure_fk_model()

    def encode_obs(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        merged = self._merged_obs(observation)
        images = torch.stack(
            [
                self._image_chw(
                    merged[_resolve_camera_rgb_key(merged, self.camera_mapping[key])],
                    camera_name=self.camera_mapping[key],
                )
                for key in self.camera_mapping
            ],
            dim=0,
        ).unsqueeze(0)
        return {"images": images, "state": self._state_tensor(merged).unsqueeze(0)}

    def _act_model_will_infer(self) -> bool:
        return not self._action_queue

    def get_action(
        self,
        observation: dict[str, torch.Tensor],
        env_action_dim: int,
        current_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_model_inference = not self._action_queue
        if not self._action_queue:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                chunk = self.model.sample_actions(observation["images"], observation["state"])
            for action in chunk[0, : self.n_action_steps].float():
                self._action_queue.append(action.unsqueeze(0))
        action = self._action_queue.popleft()
        normalized = self.model.normalize_action(action)
        action = self._clip_policy_joint_targets(action, current_state)
        env_action = self._to_wbc_action(action, env_action_dim)
        self._last_normalized_action = normalized.detach()
        self._last_raw_action = action.detach()
        self._last_wbc_action = env_action.detach()
        return env_action

    def reset_model(self) -> None:
        self._action_queue.clear()
        self._last_env_action = None
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_model_inference = False
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_hand_target = None
        self._last_safe_hand_velocity = None
        self._safety_clip_count = 0


class FlowMatchingRLPDPolicy(FlowMatchingBCPolicy):
    """Deploy the frozen Flow BC policy with its deterministic RLPD residual."""

    def get_model(self, usr_args: dict[str, Any]) -> None:
        combined_checkpoint = Path(str(usr_args["checkpoint"]))
        flow_checkpoint = combined_checkpoint / "flow_matching"
        rlpd_checkpoint = combined_checkpoint / "rlpd"
        if not (combined_checkpoint / "combined_policy.json").is_file():
            raise FileNotFoundError(f"combined policy manifest is missing: {combined_checkpoint}")
        if not flow_checkpoint.is_dir() or not rlpd_checkpoint.is_dir():
            raise FileNotFoundError("combined checkpoint must contain flow_matching/ and rlpd/")
        flow_args = dict(usr_args)
        flow_args["checkpoint"] = str(flow_checkpoint)
        super().get_model(flow_args)

        from .rlpd import RLPDAgent

        self.residual_agent = RLPDAgent.from_pretrained(rlpd_checkpoint, device=self.device)
        expected_observation_dim = self.model.config.model_dim + 19 + 2 * 16
        if self.residual_agent.config.observation_dim != expected_observation_dim:
            raise ValueError(
                "RLPD observation contract does not match Flow checkpoint: "
                f"{self.residual_agent.config.observation_dim} != {expected_observation_dim}"
            )
        self._previous_residual = torch.zeros((1, 16), device=self.device)
        self._last_base_target: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None

    def get_action(
        self,
        observation: dict[str, torch.Tensor],
        env_action_dim: int,
        current_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .rlpd.policy_contract import apply_residual_to_base, residual_observation

        self._last_model_inference = not self._action_queue
        if not self._action_queue:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                chunk = self.model.sample_actions(observation["images"], observation["state"])
            for action in chunk[0, : self.n_action_steps].float():
                self._action_queue.append(action.unsqueeze(0))
        base_target = self._action_queue.popleft()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            feature = residual_observation(
                self.model,
                observation["images"],
                observation["state"],
                base_target,
                self._previous_residual,
            ).float()
            residual = self.residual_agent.act(feature, deterministic=True).float()
        target = apply_residual_to_base(base_target, residual)
        normalized = self.model.normalize_action(target)
        target = self._clip_policy_joint_targets(target, current_state)
        env_action = self._to_wbc_action(target, env_action_dim)
        self._previous_residual = residual.detach()
        self._last_base_target = base_target.detach()
        self._last_residual = residual.detach()
        self._last_normalized_action = normalized.detach()
        self._last_raw_action = target.detach()
        self._last_wbc_action = env_action.detach()
        return env_action

    def reset_model(self) -> None:
        super().reset_model()
        self._previous_residual.zero_()
        self._last_base_target = None
        self._last_residual = None


class _GrootInferenceClient:
    """Length-prefixed NumPy protocol for the isolated LeRobot 0.6 runtime."""

    _MAX_MESSAGE_BYTES = 64 * 1024 * 1024

    def __init__(self, socket_path: Path, timeout_seconds: float = 600.0) -> None:
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self.connection: socket.socket | None = None

    def _connect(self) -> socket.socket:
        if self.connection is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            self.connection = connection
        return self.connection

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise ConnectionError("GR00T inference server closed the socket")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, **arrays: Any) -> dict[str, np.ndarray]:
        output = io.BytesIO()
        np.savez(output, **arrays)
        payload = output.getvalue()
        if len(payload) > self._MAX_MESSAGE_BYTES:
            raise ValueError(f"GR00T inference request is too large: {len(payload)} bytes")
        connection = self._connect()
        try:
            connection.sendall(struct.pack("!Q", len(payload)) + payload)
            size = struct.unpack("!Q", self._recv_exact(connection, 8))[0]
            if size <= 0 or size > self._MAX_MESSAGE_BYTES:
                raise ValueError(f"invalid GR00T inference response size: {size}")
            response_payload = self._recv_exact(connection, size)
        except Exception:
            self.close()
            raise
        with np.load(io.BytesIO(response_payload), allow_pickle=False) as archive:
            response = {key: archive[key] for key in archive.files}
        ok = response.get("ok")
        if ok is None or ok.size != 1 or int(ok.reshape(-1)[0]) != 1:
            error = response.get("error")
            message = "unknown inference-server error" if error is None else str(error.reshape(-1)[0])
            raise RuntimeError(f"GR00T inference failed: {message}")
        return response

    def ping(self) -> None:
        self._request(kind=np.asarray("ping"))

    def reset(self, seed: int | None = None) -> None:
        payload: dict[str, np.ndarray] = {"kind": np.asarray("reset")}
        if seed is not None:
            if seed < 0 or seed > np.iinfo(np.uint32).max:
                raise ValueError(f"GR00T inference seed must fit uint32, got {seed}")
            payload["seed"] = np.asarray([seed], dtype=np.uint64)
        response = self._request(**payload)
        if seed is not None:
            returned_seed = int(np.asarray(response["seed"]).reshape(-1)[0])
            if returned_seed != seed:
                raise RuntimeError(
                    f"GR00T inference server reset to seed {returned_seed}, expected {seed}"
                )

    def predict(self, payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        response = self._request(kind=np.asarray("predict"), **payload)
        action = np.asarray(response["action"], dtype=np.float32)
        normalized = np.asarray(response["normalized_action"], dtype=np.float32)
        elapsed = float(np.asarray(response["inference_seconds"]).reshape(-1)[0])
        return action, normalized, elapsed

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class LeRobotGrootN17Policy(LeRobotACTPolicy):
    """LeRobot 0.6 GR00T N1.7 adapter for the real-compatible G1 contract.

    The policy sees only head-left RGB, both D405 RGB streams, and proprioception
    reproducible on the real G1. End-effector state is FK from measured joints in
    the root frame; no simulator link pose, object pose, contact, or global camera
    enters inference. The full relative 53-D chunk is decoded by LeRobot before
    this adapter queues its absolute upper-body joint targets.
    """

    _GROOT_STATE_DIM = 49
    _GROOT_ACTION_DIM = 53
    _GROOT_EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
    _GROOT_DEFAULT_N_ACTION_STEPS = 10
    _GROOT_ACTION_HORIZON = 40
    _GROOT_VIDEO_LAG_SECONDS = 20.0 / 30.0
    _GROOT_EEF_FORWARD_OFFSET_M = 0.05
    _GROOT_CAMERA_MAPPING = {
        "head_left": "first_person_camera",
        "left_wrist": "left_hand_camera",
        "right_wrist": "right_hand_camera",
    }
    _GROOT_REQUIRED_CAMERA_KEYS = {
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    _GROOT_STATE_SLICES = {
        "left_wrist_eef_9d": (0, 9),
        "right_wrist_eef_9d": (9, 18),
        "left_hand": (18, 25),
        "right_hand": (25, 32),
        "left_arm": (32, 39),
        "right_arm": (39, 46),
        "waist": (46, 49),
    }
    _GROOT_ACTION_SLICES = {
        **_GROOT_STATE_SLICES,
        "base_height_command": (49, 50),
        "navigate_command": (50, 53),
    }

    def get_model(self, usr_args: dict[str, Any]) -> None:
        self.device = torch.device(
            os.environ.get("FLIP_TABLE_GROOT_DEVICE")
            or usr_args.get("device")
            or usr_args.get("env_cfg", {}).get("device", "cuda:0")
        )
        self.checkpoint = str(usr_args["checkpoint"])
        self.instruction = str(
            usr_args.get("instruction", "flip table")
        )
        if self.instruction != "flip table":
            raise ValueError("Furniture-GR00T instruction must be exactly 'flip table'")
        self.state_source = "joint_pos"
        self.gripper_state_source = "gripper_pos"
        self.state_indices: list[int] = []
        self.input_state_dim = self._GROOT_STATE_DIM
        self.output_action_dim = self._GROOT_ACTION_DIM
        self.n_action_steps = int(
            os.environ.get(
                "FLIP_TABLE_GROOT_N_ACTION_STEPS",
                str(self._GROOT_DEFAULT_N_ACTION_STEPS),
            )
        )
        self.policy_hz = float(
            os.environ.get("FLIP_TABLE_GROOT_POLICY_HZ", str(self._DEFAULT_POLICY_HZ))
        )
        self.sim_control_hz = float(
            os.environ.get(
                "FLIP_TABLE_GROOT_SIM_CONTROL_HZ",
                str(self._DEFAULT_SIM_CONTROL_HZ),
            )
        )
        self.target_velocity_scale = float(
            os.environ.get("FLIP_TABLE_GROOT_TARGET_VELOCITY_SCALE", "1.0")
        )
        self.target_acceleration_rad_s2 = float(
            os.environ.get(
                "FLIP_TABLE_GROOT_TARGET_ACCELERATION_RAD_S2",
                str(self._DEFAULT_TARGET_ACCELERATION_RAD_S2),
            )
        )
        if self.n_action_steps < 1:
            raise ValueError("FLIP_TABLE_GROOT_N_ACTION_STEPS must be positive")
        if self.policy_hz <= 0 or self.sim_control_hz <= 0:
            raise ValueError("GR00T policy and simulator control rates must be positive")
        if self.target_velocity_scale <= 0 or self.target_acceleration_rad_s2 <= 0:
            raise ValueError("GR00T target velocity and acceleration limits must be positive")

        self.convert_dex1_hand = True
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._pin_joint_indices = {}
        self._pin_frame_ids = {}
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_hand_target = None
        self._last_safe_hand_velocity = None
        self._safety_clip_count = 0
        self._last_env_action = None
        self._last_raw_action = None
        self._last_wbc_action = None
        self._last_normalized_action = None
        self._last_decoded_action = None
        self._last_safe_joint_target = None
        self._last_groot_state = None
        self._last_normalized_chunk = None
        self._last_decoded_chunk = None
        self._last_inference_seconds = None
        temporal_lambda_value = os.environ.get("FLIP_TABLE_GROOT_TEMPORAL_LAMBDA", "-0.1")
        temporal_lambda = (
            None
            if temporal_lambda_value.strip().lower() in {"none", "off", "disabled"}
            else float(temporal_lambda_value)
        )
        self._temporal_ensemble = PhysicalTargetTemporalEnsembler(
            decay_lambda=temporal_lambda
        )
        self._camera_history: deque[tuple[float, dict[str, np.ndarray]]] = deque(
            maxlen=max(4, int(math.ceil(self.sim_control_hz * 1.0)))
        )
        self._current_sim_time = 0.0
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._inference_seed_base = int(
            os.environ.get(
                "FLIP_TABLE_GROOT_INFERENCE_SEED",
                os.environ.get("FLIP_TABLE_EVAL_SEED", "42"),
            )
        )
        if self._inference_seed_base < 0 or self._inference_seed_base > np.iinfo(np.uint32).max:
            raise ValueError("FLIP_TABLE_GROOT_INFERENCE_SEED must fit uint32")
        self._inference_episode_index = 0

        checkpoint_config = self._validate_checkpoint_contract(Path(self.checkpoint))
        chunk_size = int(checkpoint_config["chunk_size"])
        if self.n_action_steps > chunk_size:
            raise ValueError(
                f"FLIP_TABLE_GROOT_N_ACTION_STEPS={self.n_action_steps} exceeds "
                f"checkpoint chunk_size={chunk_size}"
            )
        self._ensure_fk_model()
        socket_path = Path(
            os.environ.get("FLIP_TABLE_GROOT_SOCKET", "/tmp/flip_table_groot_n17.sock")
        )
        self._client = _GrootInferenceClient(socket_path)
        self._client.ping()
        self.model = self._client
        print(
            f"[{type(self).__name__}] runtime schedule: "
            f"n_action_steps={self.n_action_steps}, policy_hz={self.policy_hz:.3f}, "
            f"sim_control_hz={self.sim_control_hz:.3f}, socket={socket_path}, "
            f"inference_seed_base={self._inference_seed_base}"
        )

    def _next_inference_episode_seed(self) -> int:
        seed = self._inference_seed_base + self._inference_episode_index
        if seed > np.iinfo(np.uint32).max:
            raise ValueError("GR00T per-episode inference seed exceeds uint32")
        self._inference_episode_index += 1
        return seed

    @classmethod
    def _validate_checkpoint_contract(cls, checkpoint: Path) -> dict[str, Any]:
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete GR00T checkpoint {checkpoint}: missing {missing}")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))

        def feature_dim(group: str, key: str) -> int | None:
            shape = config.get(group, {}).get(key, {}).get("shape")
            return int(shape[-1]) if isinstance(shape, list) and shape else None

        if config.get("type") not in {"groot", "furniture_groot"}:
            raise ValueError("GR00T checkpoint config.type must be 'groot' or 'furniture_groot'")
        if config.get("model_version", "n1.7") != "n1.7":
            raise ValueError("GR00T checkpoint model_version must be 'n1.7'")
        if config.get("embodiment_tag") != cls._GROOT_EMBODIMENT_TAG:
            raise ValueError(
                f"GR00T checkpoint embodiment_tag must be {cls._GROOT_EMBODIMENT_TAG!r}"
            )
        if config.get("use_relative_actions") is not True:
            raise ValueError("GR00T checkpoint must set use_relative_actions=true")
        if feature_dim("input_features", "observation.state") != cls._GROOT_STATE_DIM:
            raise ValueError("GR00T checkpoint observation.state must be 49-D")
        if feature_dim("output_features", "action") != cls._GROOT_ACTION_DIM:
            raise ValueError("GR00T checkpoint action must be 53-D")
        camera_keys = {
            key
            for key in config.get("input_features", {})
            if key.startswith("observation.images.")
        }
        if camera_keys != cls._GROOT_REQUIRED_CAMERA_KEYS:
            raise ValueError(
                "GR00T checkpoint cameras must be exactly head_left, left_wrist, right_wrist; "
                f"got {sorted(camera_keys)}"
            )
        for camera_key in cls._GROOT_REQUIRED_CAMERA_KEYS:
            shape = config["input_features"][camera_key].get("shape")
            if shape != [3, 480, 640]:
                raise ValueError(f"GR00T checkpoint {camera_key} must have shape [3,480,640], got {shape}")
        excluded = config.get("relative_exclude_joints")
        if not isinstance(excluded, list) or set(excluded) != {"hand", "waist", "base_height", "navigate"}:
            raise ValueError(
                "GR00T checkpoint relative_exclude_joints must preserve absolute "
                "hand, waist, base_height, and navigate groups"
            )
        chunk_size = int(config.get("chunk_size", 0))
        if chunk_size != cls._GROOT_ACTION_HORIZON:
            raise ValueError(
                f"GR00T N1.7 checkpoint chunk_size must be 40, got {chunk_size}"
            )
        if int(config.get("max_state_dim", 0)) != 132:
            raise ValueError("GR00T N1.7 checkpoint max_state_dim must be 132")
        if int(config.get("max_action_dim", 0)) != 132:
            raise ValueError("GR00T N1.7 checkpoint max_action_dim must be 132")
        if config.get("type") == "furniture_groot":
            if int(config.get("valid_action_dim", 0)) != 46:
                raise ValueError("Furniture-GR00T checkpoint valid_action_dim must be 46")
            if config.get("base_model_revision") != (
                "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
            ):
                raise ValueError(
                    "Furniture-GR00T checkpoint base_model_revision must be pinned"
                )
            validate_finalized_furniture_checkpoint(checkpoint)
        return config

    def _current_joint_target_state(self, merged: dict[str, Any]) -> torch.Tensor:
        source = merged.get(self.state_source)
        if source is None:
            raise ValueError("embodiment_general_obs.joint_pos is required for GR00T proprioception")
        source_tensor = source.detach() if torch.is_tensor(source) else torch.as_tensor(source)
        if source_tensor.ndim == 2:
            if source_tensor.shape[0] != 1:
                raise ValueError(
                    f"GR00T evaluation requires one environment, got joint_pos {tuple(source_tensor.shape)}"
                )
            source_tensor = source_tensor[0]
        elif source_tensor.ndim != 1:
            raise ValueError(
                f"GR00T joint_pos must be [33] or [1,33], got {tuple(source_tensor.shape)}"
            )
        source_tensor = source_tensor.to(device=self.device, dtype=torch.float32).flatten()
        if source_tensor.numel() != len(self._G1_GRIPPER_33_JOINT_ORDER):
            raise ValueError(
                "GR00T evaluation requires the named 33-D G1 + Dex1 joint vector; "
                f"got {source_tensor.numel()} values"
            )
        lookup = {name: index for index, name in enumerate(self._G1_GRIPPER_33_JOINT_ORDER)}
        body = torch.stack([source_tensor[lookup[name]] for name in self._UPPER_BODY_JOINT_NAMES])
        fingers = torch.stack(
            (
                source_tensor[list(_G1_LEFT_DEX1_JOINT_INDICES)].mean(),
                source_tensor[list(_G1_RIGHT_DEX1_JOINT_INDICES)].mean(),
            )
        )
        hands = self._dex1_joint_pos_to_policy_hand(fingers)
        state = torch.cat((body, hands), dim=0)
        if not torch.isfinite(state).all():
            raise ValueError("simulator upper-body proprioception contains non-finite values")
        return state

    def _fk_eef_xyz_rot6d(self, joint_state: torch.Tensor, side: str) -> torch.Tensor:
        self._ensure_fk_model()
        assert self._pin is not None
        assert self._pin_model is not None
        assert self._pin_data is not None
        assert self._pin_q is not None

        values = joint_state[:17].detach().cpu().double().numpy()
        q = self._pin_q.copy()
        for offset, name in enumerate(self._UPPER_BODY_JOINT_NAMES):
            index = self._pin_joint_indices.get(name)
            if index is not None:
                q[index] = float(values[offset])
        self._pin.framesForwardKinematics(self._pin_model, self._pin_data, q)
        placement = self._pin_data.oMf[self._pin_frame_ids[side]]
        rotation = np.asarray(placement.rotation, dtype=np.float64)
        translation = np.asarray(placement.translation, dtype=np.float64) + rotation @ np.asarray(
            [self._GROOT_EEF_FORWARD_OFFSET_M, 0.0, 0.0], dtype=np.float64
        )
        pose = np.concatenate((translation, rotation[0], rotation[1])).astype(np.float32)
        if not np.isfinite(pose).all():
            raise ValueError(f"{side} end-effector FK produced non-finite values")
        return torch.as_tensor(pose, dtype=torch.float32, device=self.device)

    def _groot_state_tensor(self, merged: dict[str, Any]) -> torch.Tensor:
        joints = self._current_joint_target_state(merged)
        state = torch.zeros(self._GROOT_STATE_DIM, dtype=torch.float32, device=self.device)
        state[0:9] = self._fk_eef_xyz_rot6d(joints, "left")
        state[9:18] = self._fk_eef_xyz_rot6d(joints, "right")
        state[18:25] = torch.as_tensor(
            dex1_to_hand(float(joints[17]), side="left", kind="state"),
            dtype=torch.float32,
            device=self.device,
        )
        state[25:32] = torch.as_tensor(
            dex1_to_hand(float(joints[18]), side="right", kind="state"),
            dtype=torch.float32,
            device=self.device,
        )
        state[32:39] = joints[3:10]
        state[39:46] = joints[10:17]
        state[46:49] = joints[0:3]
        return state

    @staticmethod
    def _camera_value(merged: dict[str, Any], source_key: str) -> Any:
        return merged[_resolve_camera_rgb_key(merged, source_key)]

    def encode_obs(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        merged = self._merged_obs(observation)
        state = self._groot_state_tensor(merged)
        self._last_groot_state = state.detach()
        payload: dict[str, np.ndarray] = {
            "state": state.detach().cpu().numpy().astype(np.float32),
            "task": np.asarray(self.instruction),
        }
        if not self._camera_history:
            self._capture_camera_history(observation, timestamp=self._current_sim_time)
        target_time = self._current_sim_time - self._GROOT_VIDEO_LAG_SECONDS
        history_entry = min(self._camera_history, key=lambda item: abs(item[0] - target_time))
        current_entry = self._camera_history[-1]
        for target_key in self._GROOT_CAMERA_MAPPING:
            payload[target_key] = np.stack(
                (history_entry[1][target_key], current_entry[1][target_key]),
                axis=0,
            )
        return payload

    def _capture_camera_history(
        self,
        observation: dict[str, Any],
        *,
        timestamp: float,
    ) -> None:
        merged = self._merged_obs(observation)
        images: dict[str, np.ndarray] = {}
        for target_key, source_key in self._GROOT_CAMERA_MAPPING.items():
            image = _camera_image_uint8(self._camera_value(merged, source_key))
            image = _apply_recorded_camera_geometry_numpy(image, source_key)
            if image.shape != (480, 640, 3):
                raise ValueError(
                    f"policy camera {target_key} must be raw 640x480 RGB, got {image.shape}"
                )
            images[target_key] = np.ascontiguousarray(image)
        self._camera_history.append((float(timestamp), images))

    def _decoded_chunk_to_joint_targets(self, decoded: Any) -> torch.Tensor:
        chunk = decoded.detach() if torch.is_tensor(decoded) else torch.as_tensor(decoded)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[-1] != self._GROOT_ACTION_DIM:
            raise ValueError(
                f"decoded GR00T action must have shape [T,{self._GROOT_ACTION_DIM}], "
                f"got {tuple(chunk.shape)}"
            )
        chunk = chunk.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(chunk).all():
            raise ValueError("decoded GR00T action contains non-finite values")
        physical = logical_chunk_to_physical_targets(chunk.detach().cpu().numpy())
        return torch.as_tensor(physical, dtype=torch.float32, device=self.device)

    def _predict_and_add_chunk(self, observation: dict[str, Any], *, origin_step: int) -> None:
        payload = self.encode_obs(observation)
        decoded_np, normalized_np, elapsed = self._client.predict(payload)
        decoded = torch.as_tensor(decoded_np, dtype=torch.float32, device=self.device)
        normalized = torch.as_tensor(normalized_np, dtype=torch.float32, device=self.device)
        targets = self._decoded_chunk_to_joint_targets(decoded)
        if targets.shape[0] != self._GROOT_ACTION_HORIZON:
            raise RuntimeError(
                f"GR00T inference must return H40, got {targets.shape[0]} actions"
            )
        if targets.shape[0] < 1:
            raise RuntimeError("GR00T inference returned an empty action chunk")
        self._temporal_ensemble.add_chunk(
            origin_step=origin_step,
            absolute_targets=targets.detach().cpu().numpy(),
        )
        self._last_decoded_chunk = decoded.detach()
        self._last_normalized_chunk = normalized.detach()
        self._last_inference_seconds = elapsed
        self._policy_inference_count += 1
        print(
            f"[{type(self).__name__}] inference={self._policy_inference_count} "
            f"chunk={targets.shape[0]} origin_step={origin_step} latency={elapsed:.3f}s"
        )

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        _synchronize_policy_control_rate(self, task_env, attribute="sim_control_hz")
        num_envs = max(
            1,
            int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))),
        )
        if num_envs != 1:
            raise ValueError("LeRobotGrootN17Policy currently requires num_envs=1")
        action_dim = int(usr_args.get("actions_dim", self._ACT_ACTION_DIM))
        if action_dim != self._ACT_ACTION_DIM:
            raise ValueError(
                "LeRobotGrootN17Policy requires the 16-D arm/hand action space; "
                f"got {action_dim}"
            )

        ever_success = np.zeros(1, dtype=bool)
        inference_seed = self._next_inference_episode_seed()
        self._client.reset(inference_seed)
        self._temporal_ensemble.reset()
        self._camera_history.clear()
        self._last_env_action = None
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_hand_target = None
        self._last_safe_hand_velocity = None
        self._last_safe_joint_target = None
        self._last_decoded_action = None
        self._last_normalized_chunk = None
        self._last_decoded_chunk = None
        self._last_inference_seconds = None
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._current_sim_time = 0.0
        trace_rows: list[dict[str, Any]] = []

        for step in range(int(usr_args["time_out_limit"])):
            self._current_sim_time = float(step) / self.sim_control_hz
            self._capture_camera_history(
                observation,
                timestamp=self._current_sim_time,
            )
            current_state = self._current_joint_target_state(self._merged_obs(observation)).unsqueeze(0)
            should_advance = self._last_env_action is None or self._policy_clock >= self.sim_control_hz
            inferred = False
            ensemble_candidates = 0
            if should_advance:
                if self._last_env_action is None:
                    self._policy_clock = 0.0
                else:
                    self._policy_clock -= self.sim_control_hz
                execution_step = self._action_advance_count
                if execution_step % self.n_action_steps == 0:
                    self._predict_and_add_chunk(
                        observation,
                        origin_step=execution_step,
                    )
                    inferred = True
                ensemble_candidates = self._temporal_ensemble.candidate_count(execution_step)
                raw_target = torch.as_tensor(
                    self._temporal_ensemble.target(execution_step),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                safe_target = self._clip_policy_joint_targets(raw_target, current_state)
                self._last_env_action = self._to_wbc_action(safe_target, action_dim)
                self._last_raw_action = raw_target.detach()
                self._last_safe_joint_target = safe_target.detach()
                self._last_decoded_action = (
                    self._last_decoded_chunk[0].detach() if inferred else None
                )
                self._last_wbc_action = self._last_env_action.detach()
                self._last_normalized_action = None
                self._action_advance_count += 1

            action = self._last_env_action
            assert action is not None
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, 1)
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference_seed": inference_seed,
                    "evaluation_mode": os.environ.get(
                        "FLIP_TABLE_EVAL_MODE", "randomized"
                    ),
                    "domain_randomization_profile": os.environ.get(
                        "FLIP_TABLE_GROOT_DR_PROFILE", "generic_v1"
                    ),
                    "policy_inference": inferred,
                    "policy_inference_index": self._policy_inference_count,
                    "action_advanced": should_advance,
                    "action_advance_index": self._action_advance_count,
                    "temporal_ensemble_candidates": ensemble_candidates,
                    "inference_seconds": self._last_inference_seconds if inferred else None,
                    "groot_state_before": _trace_value(
                        self._last_groot_state if inferred else None
                    ),
                    "normalized_chunk": _trace_array(
                        self._last_normalized_chunk if inferred else None
                    ),
                    "decoded_chunk": _trace_array(self._last_decoded_chunk if inferred else None),
                    "decoded_action_53d": _trace_value(self._last_decoded_action),
                    "raw_joint_target_16d": _trace_value(self._last_raw_action),
                    "safe_joint_target_16d": _trace_value(self._last_safe_joint_target),
                    "sent_action_16d": _trace_value(action),
                    "joint_state_before_19d": _trace_value(current_state),
                    "joint_state_after_19d": _trace_value(
                        self._current_joint_target_state(self._merged_obs(observation))
                    ),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                }
            )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            self._policy_clock += self.policy_hz
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success

        _write_action_state_trace(self, usr_args, trace_rows)
        print(
            f"[{type(self).__name__}] policy_inferences={self._policy_inference_count} "
            f"action_advances={self._action_advance_count} "
            f"sim_steps={int(usr_args['time_out_limit'])}"
        )
        return ever_success

    def reset_model(self) -> None:
        if hasattr(self, "_client"):
            self._client.reset()
        self._temporal_ensemble.reset()
        self._camera_history.clear()
        self._last_env_action = None
        self._policy_clock = self.sim_control_hz
        self._policy_inference_count = 0
        self._action_advance_count = 0
        self._last_safe_target = None
        self._last_safe_velocity = None
        self._last_safe_joint_target = None
        self._last_decoded_action = None
        self._last_normalized_chunk = None
        self._last_decoded_chunk = None
        self._last_inference_seconds = None
        self._safety_clip_count = 0
        self._current_sim_time = 0.0


class RecordedWBCPolicy(LeRobotACTPolicy):
    """Replay real joint targets through the official V1 WBC action path."""

    def get_model(self, usr_args: dict[str, Any]) -> None:
        self.device = torch.device(usr_args.get("env_cfg", {}).get("device", "cuda:0"))
        self.model = None
        self.output_action_dim = self._ACT_ACTION_DIM
        self.convert_dex1_hand = True
        self._pin = None
        self._pin_model = None
        self._pin_data = None
        self._pin_q = None
        self._pin_joint_indices = {}
        self._pin_frame_ids = {}
        self._ensure_fk_model()

        replay_path = os.environ.get("FLIP_TABLE_REPLAY_ACTION_PATH", "").strip()
        if not replay_path:
            raise ValueError("FLIP_TABLE_REPLAY_ACTION_PATH is required for RecordedWBCPolicy")
        path = Path(replay_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("actions")
        actions = torch.as_tensor(payload, dtype=torch.float32, device=self.device)
        if actions.ndim != 2 or actions.shape[-1] != self._ACT_ACTION_DIM:
            raise ValueError(f"replay actions must have shape [N,16], got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("replay actions contain NaN or Inf")
        self._replay_actions = actions
        self._replay_hz = float(os.environ.get("FLIP_TABLE_REPLAY_HZ", "30"))
        self._sim_control_hz = float(os.environ.get("FLIP_TABLE_ACT_SIM_CONTROL_HZ", "50"))
        raw_hold_index = os.environ.get("FLIP_TABLE_REPLAY_HOLD_INDEX", "").strip()
        self._replay_hold_index = int(raw_hold_index) if raw_hold_index else None
        print(
            f"[RecordedWBCPolicy] loaded {actions.shape[0]} actions from {path}; "
            f"replay_hz={self._replay_hz:.3f}, sim_control_hz={self._sim_control_hz:.3f}"
        )

    def eval(self, task_env: Any, observation: dict[str, Any], usr_args: dict[str, Any], video_writer: Any):
        _synchronize_policy_control_rate(self, task_env, attribute="_sim_control_hz")
        num_envs = max(1, int(usr_args.get("env_cfg", {}).get("num_envs", _num_envs_from_obs(observation))))
        action_dim = int(usr_args.get("actions_dim", self._ACT_ACTION_DIM))
        if action_dim != self._ACT_ACTION_DIM:
            raise ValueError(f"RecordedWBCPolicy requires the 16-D balanced WBC action space, got {action_dim}")
        ever_success = np.zeros(num_envs, dtype=bool)
        trace_rows: list[dict[str, Any]] = []
        debug_every = int(os.environ.get("FLIP_TABLE_WBC_DEBUG_EVERY", "0"))
        for step in range(int(usr_args["time_out_limit"])):
            replay_index = min(
                self._replay_actions.shape[0] - 1,
                int(round(float(step) * self._replay_hz / self._sim_control_hz)),
            )
            if self._replay_hold_index is not None:
                replay_index = max(0, min(self._replay_actions.shape[0] - 1, self._replay_hold_index))
            source_action = self._replay_actions[replay_index].unsqueeze(0)
            action = self._to_wbc_action(source_action, self._ACT_ACTION_DIM).expand(num_envs, -1).clone()
            observation, _, terminated, _, extras = task_env.step(action)
            ever_success |= _success_mask(extras, terminated, num_envs)
            if debug_every > 0 and step % debug_every == 0:
                try:
                    manager = getattr(task_env, "action_manager", None)
                except Exception:
                    manager = None
                if manager is None:
                    try:
                        manager = getattr(task_env.unwrapped, "action_manager", None)
                    except Exception:
                        manager = None
                term = None
                if manager is not None:
                    try:
                        term = manager.get_term("base_action")
                    except Exception:
                        term = None
                print(
                    "[RecordedWBCPolicy][debug] "
                    f"step={step}, replay_index={replay_index}, "
                    f"sent_arm_hand={_trace_value(action)}, "
                    f"target_robot_joints={_trace_value(getattr(term, 'target_robot_joints_mujoco', None))}, "
                    f"processed_actions={_trace_value(getattr(term, 'processed_actions', None))}",
                    flush=True,
                )
            trace_rows.append(
                {
                    "step": step,
                    "policy_inference": True,
                    "replay_index": replay_index,
                    "source_joint_target": _trace_value(source_action),
                    "sent_wbc_action": _trace_value(action),
                    "state_after": _trace_value(self._merged_trace_state(observation)),
                    "terminated": _trace_value(terminated),
                    "success": ever_success.tolist(),
                }
            )
            _maybe_save_camera_frames(self, observation, usr_args, step)
            self.add_video_frame(video_writer, observation, usr_args.get("record_camera", []))
            if _terminated_any(terminated):
                _write_action_state_trace(self, usr_args, trace_rows)
                return ever_success
        _write_action_state_trace(self, usr_args, trace_rows)
        return ever_success

    def reset_model(self) -> None:
        pass
