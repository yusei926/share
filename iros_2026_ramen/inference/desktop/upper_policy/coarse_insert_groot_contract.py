"""Exact physical contract for the pinned coarse-insert GR00T N1.7 model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    source_euler_xyz_pose_to_xyz_rot6d,
)


MODEL_REPO_ID = (
    "Team-RAMEN/IROS2026_RAMEN_takada_groot_n17_coarse_insert_100k_dex1_v2"
)
MODEL_REVISION = "55bff52e3849323d1e37d03376ea6f4520440236"
MODEL_SHA256 = "f2f4e035657c0a6ca13b8d87dc17f692a83be2f1186fd20bf8cfba5569dfa2f2"
TASK_TEXT = "coarse align table leg to table base"
LEROBOT_VERSION = "0.6.0"
EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
MODEL_STATE_DIM = 49
MODEL_ACTION_DIM = 53
MODEL_ACTION_HORIZON = 16
EXECUTABLE_ACTION_DIM = 16
DEX1_DATASET_OPEN_VALUE = 4.5
CAMERA_ROLE_TO_KEY = {
    "head_left": "observation.images.head_left",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
CAMERA_ROLES = tuple(CAMERA_ROLE_TO_KEY)
CAMERA_KEYS = tuple(CAMERA_ROLE_TO_KEY.values())


def compose_model_state(
    body_joint_position_rad: Sequence[float],
    dex1_opening_fraction: Sequence[float],
    eef_xyz_euler: Sequence[float],
) -> np.ndarray:
    body = _vector(body_joint_position_rad, 29, "G1 body state")
    hands = _vector(dex1_opening_fraction, 2, "Dex1 opening fraction")
    eef = _vector(eef_xyz_euler, 12, "root-frame EEF XYZ+Euler")
    if np.any((hands < 0.0) | (hands > 1.0)):
        raise ValueError("Dex1 opening fraction must lie in [0,1]")
    result = np.zeros(MODEL_STATE_DIM, dtype=np.float32)
    result[0:9] = source_euler_xyz_pose_to_xyz_rot6d(eef[0:6].tolist())
    result[9:18] = source_euler_xyz_pose_to_xyz_rot6d(eef[6:12].tolist())
    physical_hands = hands * DEX1_DATASET_OPEN_VALUE
    result[18] = -physical_hands[0] / 3.0
    result[25] = physical_hands[1] / 3.0
    result[32:39] = body[15:22]
    result[39:46] = body[22:29]
    result[46:49] = body[12:15]
    if result.shape != (MODEL_STATE_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("coarse-insert state assembly violated 49-D contract")
    return result


def extract_executable_action(action_chunk: Any) -> np.ndarray:
    """Discard EEF, waist, base and nav; return arms14 + physical Dex1 two."""
    values = np.asarray(action_chunk, dtype=np.float64)
    expected = (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"decoded coarse-insert action must be finite {expected}")
    result = np.empty((MODEL_ACTION_HORIZON, EXECUTABLE_ACTION_DIM))
    result[:, :7] = values[:, 32:39]
    result[:, 7:14] = values[:, 39:46]
    # This checkpoint's serialized processor uses one meaningful hand slot per
    # side. This is not the later seven-joint Dex1 synergy representation.
    result[:, 14] = np.clip(-3.0 * values[:, 18], 0.0, DEX1_DATASET_OPEN_VALUE)
    result[:, 15] = np.clip(3.0 * values[:, 25], 0.0, DEX1_DATASET_OPEN_VALUE)
    return result


def camera_payloads(camera_jpeg: Mapping[str, bytes]) -> dict[str, bytes]:
    missing = set(CAMERA_ROLES) - set(camera_jpeg)
    if missing:
        raise ValueError(f"coarse-insert is missing cameras: {sorted(missing)}")
    result = {
        key: bytes(camera_jpeg[role]) for role, key in CAMERA_ROLE_TO_KEY.items()
    }
    if any(not value for value in result.values()):
        raise ValueError("coarse-insert camera JPEGs must be non-empty")
    return result


def validate_checkpoint_metadata(
    checkpoint: str | Path,
    *,
    model_repo_id: str = MODEL_REPO_ID,
    model_revision: str = MODEL_REVISION,
    task: str = TASK_TEXT,
    expected_model_sha256: str = MODEL_SHA256,
) -> dict[str, Any]:
    root = Path(checkpoint).expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete coarse-insert checkpoint: {missing}")
    config = _json(root / "config.json")
    if (
        config.get("type") != "groot"
        or config.get("embodiment_tag") != EMBODIMENT_TAG
        or config.get("chunk_size") != MODEL_ACTION_HORIZON
        or config.get("n_action_steps") != MODEL_ACTION_HORIZON
        or config.get("use_relative_actions") is not True
    ):
        raise ValueError("coarse-insert GR00T scalar contract changed")
    if set(config.get("relative_exclude_joints") or ()) != {
        "hand",
        "waist",
        "base_height",
        "navigate",
    }:
        raise ValueError("coarse-insert relative action exclusions changed")
    inputs = config.get("input_features") or {}
    if (inputs.get("observation.state") or {}).get("shape") != [MODEL_STATE_DIM]:
        raise ValueError("coarse-insert state must be 49-D")
    if {
        key for key in inputs if key.startswith("observation.images.")
    } != set(CAMERA_KEYS):
        raise ValueError("coarse-insert camera roles changed")
    if (config.get("output_features", {}).get("action") or {}).get("shape") != [
        MODEL_ACTION_DIM
    ]:
        raise ValueError("coarse-insert action must be 53-D")
    _validate_serialized_processors(root)
    digest = _sha256(root / "model.safetensors")
    if digest != expected_model_sha256:
        raise ValueError(f"coarse-insert model SHA-256 changed: {digest}")
    return {
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "weights_sha256": digest,
        "task": task,
        "state_dim": MODEL_STATE_DIM,
        "decoded_action_dim": MODEL_ACTION_DIM,
        "executable_action_dim": EXECUTABLE_ACTION_DIM,
        "action_horizon": MODEL_ACTION_HORIZON,
        "camera_roles": list(CAMERA_ROLES),
        "lower_body_command_dimensions": 0,
    }


def _validate_serialized_processors(root: Path) -> None:
    pre = _json(root / "policy_preprocessor.json").get("steps")
    post = _json(root / "policy_postprocessor.json").get("steps")
    if not isinstance(pre, list) or not isinstance(post, list):
        raise ValueError("coarse-insert serialized processor steps are missing")
    pack = next(
        (step for step in pre if step.get("registry_name") == "groot_n1_7_pack_inputs_v1"),
        None,
    )
    decode = next(
        (step for step in post if step.get("registry_name") == "groot_n1_7_action_decode_v1"),
        None,
    )
    if not isinstance(pack, dict) or not isinstance(decode, dict):
        raise ValueError("coarse-insert pack/decode processors are missing")
    pack_config = pack.get("config") or {}
    decode_config = decode.get("config") or {}
    if (
        pack_config.get("state_horizon") != 1
        or pack_config.get("video_horizon") != 1
        or pack_config.get("action_horizon") != 40
        or pack_config.get("valid_action_horizon") != 40
        or pack_config.get("video_modality_keys")
        != ["head_left", "left_wrist", "right_wrist"]
        or decode_config.get("env_action_dim") != MODEL_ACTION_DIM
        or decode_config.get("use_relative_action") is not True
    ):
        raise ValueError("coarse-insert serialized processor contract changed")


def _vector(values: Sequence[float], width: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}], got {result.shape}")
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
