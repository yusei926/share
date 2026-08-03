"""Physical G1 contract for the flip-table Furniture-GR00T N1.7 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from model.subtask_policy_training.gr00t.dex1_hand_synergy import DEX1_OPEN
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    GROOT_N17_NATIVE_ACTION_HORIZON,
    GROOT_N17_PACKED_ACTION_DIM,
    GROOT_N17_PACKED_STATE_DIM,
    GROOT_N17_VALID_ACTION_DIM,
    REAL_G1_RELATIVE_EEF_ACTION_DIM,
    REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
    REAL_G1_RELATIVE_EEF_STATE_DIM,
    SOURCE_ROBOT_Q_DIM,
    map_source_state_to_real_g1_relative_eef,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    BASE_MODEL_REVISION,
    EXPECTED_TUNING_SCOPE,
    validate_finalized_furniture_checkpoint,
)
from model.subtask_policy_training.gr00t.temporal_ensemble import (
    PHYSICAL_ACTION_DIM,
    logical_chunk_to_physical_targets,
)


TASK_TEXT = "flip table"
LEROBOT_VERSION = "0.6.0"
CAMERA_ROLE_TO_KEY = {
    "head_left": "observation.images.head_left",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
CAMERA_ROLES = tuple(CAMERA_ROLE_TO_KEY)
CAMERA_KEYS = tuple(CAMERA_ROLE_TO_KEY.values())
VIDEO_DELTA_INDICES = (-20, 0)
VIDEO_HORIZON = len(VIDEO_DELTA_INDICES)
DATASET_FPS = 30.0
MODEL_ACTION_HORIZON = GROOT_N17_NATIVE_ACTION_HORIZON
DEX1_DATASET_OPEN_VALUE = DEX1_OPEN
LEGACY_V2_DATASET_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
LEGACY_V2_BASE_MODEL_PATH = "/dev/shm/iros_2026_ramen_groot_n17/groot_overlay"


def compose_model_state(
    body_joint_position_rad: Sequence[float],
    dex1_opening_fraction: Sequence[float],
    eef_xyz_euler_xyz: Sequence[float],
) -> np.ndarray:
    """Build the exact 49-D state used by training from live observations."""
    body = _finite_vector(body_joint_position_rad, 29, "G1 body state")
    dex1_fraction = _finite_vector(
        dex1_opening_fraction, 2, "Dex1 opening fraction"
    )
    if np.any((dex1_fraction < 0.0) | (dex1_fraction > 1.0)):
        raise ValueError("Dex1 opening fraction must lie in [0,1]")
    eef = _finite_vector(eef_xyz_euler_xyz, 12, "root-frame EEF state")
    # Root coordinates are ignored by this mapping; only the following 29
    # joints are read. Keeping a complete 36-D source vector preserves the
    # exact dataset conversion path.
    source_robot_q = np.concatenate((np.zeros(7, dtype=np.float64), body))
    state = map_source_state_to_real_g1_relative_eef(
        ee_state=eef.tolist(),
        robot_q_current=source_robot_q.tolist(),
        hand_state=(dex1_fraction * DEX1_DATASET_OPEN_VALUE).tolist(),
    )
    result = np.asarray(state, dtype=np.float32)
    if result.shape != (REAL_G1_RELATIVE_EEF_STATE_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("Furniture-GR00T state assembly violated the 49-D contract")
    return result


def extract_executable_action(action_chunk: Any) -> np.ndarray:
    """Return only absolute arm14 and Dex1-1 left/right targets."""
    values = np.asarray(action_chunk, dtype=np.float64)
    if values.shape != (
        GROOT_N17_NATIVE_ACTION_HORIZON,
        REAL_G1_RELATIVE_EEF_ACTION_DIM,
    ):
        raise ValueError(
            "decoded Furniture-GR00T action must be "
            f"[{GROOT_N17_NATIVE_ACTION_HORIZON},{REAL_G1_RELATIVE_EEF_ACTION_DIM}], "
            f"got {values.shape}"
        )
    result = logical_chunk_to_physical_targets(values)
    if result.shape != (GROOT_N17_NATIVE_ACTION_HORIZON, PHYSICAL_ACTION_DIM):
        raise RuntimeError("physical action extraction changed dimension")
    return result


def camera_payload_history(
    observations: Sequence[Any],
) -> dict[str, list[bytes]]:
    if len(observations) != VIDEO_HORIZON:
        raise ValueError(f"camera history requires exactly {VIDEO_HORIZON} observations")
    result: dict[str, list[bytes]] = {}
    for role, key in CAMERA_ROLE_TO_KEY.items():
        payloads = [bytes(observation.camera_jpeg[role]) for observation in observations]
        if any(not payload for payload in payloads):
            raise ValueError(f"{role} camera history contains an empty JPEG")
        result[key] = payloads
    return result


def validate_checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    """Fail closed unless a finalized checkpoint preserves every N1.7 contract."""
    root = Path(checkpoint).expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "training_manifest.json",
        "training_run_record.json",
        "dex1_g1_synergy.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Furniture-GR00T checkpoint: missing {missing}")

    config = _read_json(root / "config.json")
    expected_scalars = {
        "type": "furniture_groot",
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "base_model_revision": BASE_MODEL_REVISION,
        "embodiment_tag": REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
        "chunk_size": GROOT_N17_NATIVE_ACTION_HORIZON,
        "max_state_dim": GROOT_N17_PACKED_STATE_DIM,
        "max_action_dim": GROOT_N17_PACKED_ACTION_DIM,
        "valid_action_dim": GROOT_N17_VALID_ACTION_DIM,
        "use_relative_actions": True,
    }
    actual_scalars = {key: config.get(key) for key in expected_scalars}
    if actual_scalars != expected_scalars:
        raise ValueError(
            "checkpoint violates the pinned Furniture-GR00T contract: "
            f"{actual_scalars}"
        )
    if set(config.get("relative_exclude_joints") or ()) != {
        "hand",
        "waist",
        "base_height",
        "navigate",
    }:
        raise ValueError("checkpoint relative-action exclusion groups changed")
    if config.get("action_decode_transform") is not None:
        raise ValueError("physical checkpoint must not use a simulator action transform")
    _validate_features(config)

    manifest = _read_json(root / "training_manifest.json")
    contract = manifest.get("contract")
    expected_manifest = {
        "logical_state_dim": REAL_G1_RELATIVE_EEF_STATE_DIM,
        "logical_action_dim": REAL_G1_RELATIVE_EEF_ACTION_DIM,
        "packed_state_dim": GROOT_N17_PACKED_STATE_DIM,
        "packed_action_dim": GROOT_N17_PACKED_ACTION_DIM,
        "valid_action_dim": GROOT_N17_VALID_ACTION_DIM,
        "action_horizon": GROOT_N17_NATIVE_ACTION_HORIZON,
        "policy_cameras": list(CAMERA_ROLES),
        "head_right_used": False,
        "progress_in_action": False,
        "task_instruction": TASK_TEXT,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in expected_manifest.items()
    ):
        raise ValueError("training manifest does not match the physical policy contract")
    expected_progress_shape = (
        [GROOT_N17_NATIVE_ACTION_HORIZON, 1]
        if bool(config.get("progress_enabled", False))
        else None
    )
    if contract.get("progress_head_shape") != expected_progress_shape:
        raise ValueError("training manifest auxiliary progress shape changed")

    release = validate_finalized_furniture_checkpoint(root)
    model_hash = release["model_safetensors_sha256"]

    return {
        "task": TASK_TEXT,
        "state_dim": REAL_G1_RELATIVE_EEF_STATE_DIM,
        "logical_action_dim": REAL_G1_RELATIVE_EEF_ACTION_DIM,
        "executable_action_dim": PHYSICAL_ACTION_DIM,
        "action_horizon": GROOT_N17_NATIVE_ACTION_HORIZON,
        "execution_steps": release["execution_steps"],
        "temporal_lambda": release["temporal_lambda"],
        "temporal_lambda_label": release["temporal_lambda_label"],
        "camera_roles": list(CAMERA_ROLES),
        "video_delta_indices": list(VIDEO_DELTA_INDICES),
        "lower_body_command_dimensions": 0,
        "weights_sha256": model_hash,
        "progress_enabled": bool(config.get("progress_enabled", False)),
        "release_certified": True,
    }


def validate_legacy_v2_candidate_checkpoint(
    checkpoint: str | Path,
    *,
    expected_model_sha256: str,
    verify_model_hash: bool = True,
) -> dict[str, Any]:
    """Validate the immutable 20k v2 training candidate without promoting it.

    This intentionally does not call the finalized-release validator.  The
    checkpoint repository contains resumable training snapshots, not the sim
    selection and release evidence required by :func:`validate_checkpoint_metadata`.
    Keeping this separate prevents an intermediate candidate from being
    mistaken for a release-certified policy.
    """

    root = Path(checkpoint).expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_groot_n1_7_pack_inputs_v1.safetensors",
        "train_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete legacy Furniture-GR00T candidate: {missing}")
    actual_model_sha256 = (
        _sha256(root / "model.safetensors")
        if verify_model_hash
        else expected_model_sha256
    )
    if actual_model_sha256 != expected_model_sha256:
        raise ValueError(
            "legacy Furniture-GR00T model hash changed: "
            f"expected={expected_model_sha256}, actual={actual_model_sha256}"
        )

    config = _read_json(root / "config.json")
    expected_scalars = {
        "type": "furniture_groot",
        "base_model_path": LEGACY_V2_BASE_MODEL_PATH,
        "base_model_revision": BASE_MODEL_REVISION,
        "embodiment_tag": REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
        "chunk_size": MODEL_ACTION_HORIZON,
        "n_action_steps": 10,
        "max_state_dim": GROOT_N17_PACKED_STATE_DIM,
        "max_action_dim": GROOT_N17_PACKED_ACTION_DIM,
        "valid_action_dim": GROOT_N17_VALID_ACTION_DIM,
        "use_relative_actions": True,
        "progress_enabled": False,
        "consistent_gpu_augmentation": True,
    }
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in expected_scalars.items()
        if config.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"legacy Furniture-GR00T config changed: {mismatch}")
    if any(config.get(key) != value for key, value in EXPECTED_TUNING_SCOPE.items()):
        raise ValueError("legacy Furniture-GR00T tuning scope changed")
    if set(config.get("relative_exclude_joints") or ()) != {
        "hand",
        "waist",
        "base_height",
        "navigate",
    }:
        raise ValueError("legacy Furniture-GR00T relative exclusions changed")
    if config.get("action_decode_transform") is not None:
        raise ValueError("legacy candidate contains a simulator-only action transform")
    _validate_features(config)

    training = _read_json(root / "train_config.json")
    dataset = training.get("dataset") or {}
    episodes = dataset.get("episodes")
    if (
        dataset.get("repo_id") != LEGACY_V2_DATASET_REPO_ID
        or dataset.get("revision") is not None
        or episodes != list(range(156))
    ):
        raise ValueError("legacy candidate training dataset provenance changed")

    pre = _read_json(root / "policy_preprocessor.json")
    post = _read_json(root / "policy_postprocessor.json")
    pre_steps = [step.get("registry_name") for step in pre.get("steps", [])]
    if pre_steps != [
        "rename_observations_processor",
        "to_batch_processor",
        "furniture_groot_temporal_progress_v1",
        "groot_n1_7_pack_inputs_v1",
        "furniture_groot_consistent_gpu_augmentation_v1",
        "groot_n1_7_vlm_encode_v1",
        "device_processor",
    ]:
        raise ValueError("legacy Furniture-GR00T preprocessor pipeline changed")
    post_steps = [step.get("registry_name") for step in post.get("steps", [])]
    if post_steps != ["groot_n1_7_action_decode_v1", "device_processor"]:
        raise ValueError("legacy Furniture-GR00T postprocessor pipeline changed")
    pack = pre["steps"][3].get("config") or {}
    if any(
        pack.get(key) != value
        for key, value in {
            "state_horizon": 1,
            "action_horizon": 40,
            "valid_action_horizon": 40,
            "video_horizon": 2,
            "max_state_dim": 132,
            "max_action_dim": 132,
            "embodiment_tag": REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
            "video_modality_keys": list(CAMERA_ROLES),
        }.items()
    ):
        raise ValueError("legacy Furniture-GR00T serialized pack contract changed")
    decode = post["steps"][0].get("config") or {}
    if decode.get("env_action_dim") != 53 or decode.get("use_relative_action") is not True:
        raise ValueError("legacy Furniture-GR00T action decoder changed")

    return {
        "task": TASK_TEXT,
        "state_dim": REAL_G1_RELATIVE_EEF_STATE_DIM,
        "logical_action_dim": REAL_G1_RELATIVE_EEF_ACTION_DIM,
        "executable_action_dim": PHYSICAL_ACTION_DIM,
        "action_horizon": MODEL_ACTION_HORIZON,
        "execution_steps": 10,
        "temporal_lambda": None,
        "temporal_lambda_label": "none",
        "camera_roles": list(CAMERA_ROLES),
        "video_delta_indices": list(VIDEO_DELTA_INDICES),
        "lower_body_command_dimensions": 0,
        "weights_sha256": actual_model_sha256,
        "progress_enabled": False,
        "release_certified": False,
        "candidate_status": "intermediate_unselected_20k",
        "training_dataset_repo_id": LEGACY_V2_DATASET_REPO_ID,
        "training_dataset_revision": None,
    }


def _validate_features(config: Mapping[str, Any]) -> None:
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    if (inputs.get("observation.state") or {}).get("shape") != [
        REAL_G1_RELATIVE_EEF_STATE_DIM
    ]:
        raise ValueError("checkpoint observation.state must be 49-D")
    camera_keys = {
        key for key in inputs if str(key).startswith("observation.images.")
    }
    if camera_keys != set(CAMERA_KEYS):
        raise ValueError(f"checkpoint camera keys changed: {sorted(camera_keys)}")
    for key in CAMERA_KEYS:
        if (inputs.get(key) or {}).get("shape") != [3, 480, 640]:
            raise ValueError(f"checkpoint {key} must be RGB [3,480,640]")
    if (outputs.get("action") or {}).get("shape") != [
        REAL_G1_RELATIVE_EEF_ACTION_DIM
    ]:
        raise ValueError("checkpoint logical action must be 53-D")


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}], got {result.shape}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
