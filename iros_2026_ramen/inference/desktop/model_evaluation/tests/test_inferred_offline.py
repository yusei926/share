from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.desktop.model_evaluation.artifacts import load_prepared_spec
from inference.desktop.model_evaluation.inferred import (
    OFFLINE_CONTRACT_SCHEMA,
    InferredOfflineContract,
    _artifact_paths,
    _select_config,
)
from inference.desktop.model_evaluation.inferred_artifacts import (
    OFFLINE_LOCK_FILENAME,
    download_plan,
    seal,
    validate,
)


def _contract(*, required_files: tuple[str, ...], hashes: dict[str, str]):
    return InferredOfflineContract(
        schema_version=OFFLINE_CONTRACT_SCHEMA,
        repo_id="Team-RAMEN/test-act",
        revision="1" * 40,
        category="lerobot_offline",
        config_type="act",
        loader_kind="lerobot",
        checkpoint_subdir="pretrained_model",
        config_path="config.json",
        state_dim=16,
        action_dim=16,
        observation_horizon=1,
        action_horizon=30,
        execution_steps=30,
        camera_keys=("observation.images.cam_0",),
        image_shapes={"observation.images.cam_0": (3, 480, 640)},
        normalization={"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        required_files=required_files,
        file_sha256=hashes,
        total_download_bytes=1,
        confidence="high",
        issues=(),
    )


def test_config_selection_prefers_canonical_release_not_training_snapshots() -> None:
    config = {
        "type": "act",
        "input_features": {"observation.state": {"shape": [16]}},
        "output_features": {"action": {"shape": [16]}},
    }
    candidates = [
        ("checkpoints/010000/config.json", config),
        ("pretrained_model/config.json", config),
    ]
    names = [
        "checkpoints/010000/config.json",
        "checkpoints/010000/model.safetensors",
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
    ]
    assert _select_config(candidates, names)[0] == "pretrained_model/config.json"


def test_multiple_checkpoint_snapshots_without_release_are_ambiguous() -> None:
    config = {"type": "act"}
    candidates = [
        ("checkpoints/010000/config.json", config),
        ("checkpoints/020000/config.json", config),
    ]
    names = [
        "checkpoints/010000/model.safetensors",
        "checkpoints/020000/model.safetensors",
    ]
    assert _select_config(candidates, names) is None


def test_artifact_selection_is_exact_and_excludes_training_state() -> None:
    names = [
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/policy_preprocessor.json",
        "pretrained_model/optimizer.pt",
        "pretrained_model/subdir/unexpected.bin",
        "checkpoints/020000/model.safetensors",
    ]
    assert _artifact_paths(
        names, checkpoint_subdir="pretrained_model", config_type="act"
    ) == (
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/policy_preprocessor.json",
    )


def test_inferred_lock_is_tamper_evident_and_rejected_by_physical_loader(
    tmp_path: Path,
) -> None:
    payload = b"safe tensor bytes"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"type": "act"}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(payload)
    config_hash = hashlib.sha256(
        (checkpoint / "config.json").read_bytes()
    ).hexdigest()
    contract = _contract(
        required_files=("config.json", "model.safetensors"),
        hashes={"config.json": config_hash, "model.safetensors": digest},
    )
    report = seal(contract, tmp_path)
    assert report["actuation_allowed"] is False
    assert (tmp_path / OFFLINE_LOCK_FILENAME).is_file()
    assert validate(tmp_path)["robot_command_sent"] is False
    plan = download_plan(contract, tmp_path)
    assert plan["allow_patterns"] == [
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
    ]
    with pytest.raises(FileNotFoundError):
        load_prepared_spec(
            tmp_path,
            reference=contract.repo_id,
            revision=contract.revision,
        )
    (checkpoint / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed after sealing"):
        validate(tmp_path)


def test_inferred_contract_cannot_claim_physical_mapping() -> None:
    raw = _contract(required_files=(), hashes={}).to_mapping()
    raw["physical_mapping_verified"] = True
    with pytest.raises(ValueError, match="cannot verify physical"):
        InferredOfflineContract.from_mapping(raw)
    raw["physical_mapping_verified"] = False
    raw["actuation_allowed"] = True
    with pytest.raises(ValueError, match="must never allow actuation"):
        InferredOfflineContract.from_mapping(raw)
