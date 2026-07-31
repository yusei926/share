"""Pinned artifact download, validation, and local tamper-evident sealing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .registry import (
    ModelSpec,
    model_spec_from_manifest,
    normalize_model_reference,
)


LOCK_FILENAME = ".iros_ramen_model_lock.json"
LOCK_SCHEMA = "team_ramen_local_model_lock/v1"


def checkpoint_path(download_root: Path, spec: ModelSpec) -> Path:
    root = download_root.expanduser().resolve()
    checkpoint = (
        root / spec.artifact.checkpoint_subdir
        if spec.artifact.checkpoint_subdir
        else root
    )
    resolved = checkpoint.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("checkpoint_subdir escapes the local model directory")
    return resolved


def validate_artifacts(download_root: Path, spec: ModelSpec) -> dict[str, Any]:
    checkpoint = checkpoint_path(download_root, spec)
    missing = [
        name for name in spec.artifact.required_files if not (checkpoint / name).is_file()
    ]
    empty = [
        name
        for name in spec.artifact.required_files
        if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == 0
    ]
    if missing or empty:
        raise FileNotFoundError(
            f"incomplete checkpoint for {spec.model_id}: missing={missing}, empty={empty}"
        )
    report: dict[str, Any] = {
        "model_id": spec.model_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "checkpoint": str(checkpoint),
        "required_files": list(spec.artifact.required_files),
        "complete": True,
    }
    verified_hashes: dict[str, str] = {}
    for name, expected in spec.artifact.file_sha256.items():
        path = _required_path(checkpoint, name)
        if not path.is_file():
            raise FileNotFoundError(f"{spec.model_id}: hashed artifact is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{spec.model_id}: {name} hash mismatch "
                f"(expected={expected}, actual={actual})"
            )
        verified_hashes[name] = actual
    report["verified_weight_sha256"] = verified_hashes
    if spec.expected_model_sha256 is not None:
        model = checkpoint / "model.safetensors"
        if not model.is_file():
            raise FileNotFoundError("expected model.safetensors is missing")
        actual = verified_hashes.get("model.safetensors") or sha256_file(model)
        if actual != spec.expected_model_sha256:
            raise ValueError(
                f"{spec.model_id}: model.safetensors hash mismatch "
                f"(expected={spec.expected_model_sha256}, actual={actual})"
            )
    _validate_static_contract(checkpoint, spec)
    return report


def download_plan(spec: ModelSpec, local_dir: Path) -> dict[str, Any]:
    """Return a commit-pinned minimal snapshot_download plan."""
    return {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "local_dir": str(local_dir.expanduser().resolve()),
        "allow_patterns": list(spec.artifact.allow_patterns),
        "ignore_patterns": [
            "**/optimizer.pt",
            "**/optimizer_state.safetensors",
            "**/rng_state.pth",
            "**/rng_state.safetensors",
            "**/scheduler.pt",
            "**/training_args.bin",
        ],
    }


def seal_local_artifacts(download_root: Path, spec: ModelSpec) -> dict[str, Any]:
    """Validate, hash every required file, and atomically write a local lock."""
    report = validate_artifacts(download_root, spec)
    checkpoint = checkpoint_path(download_root, spec)
    all_hashes = {
        name: sha256_file(_required_path(checkpoint, name))
        for name in spec.artifact.required_files
    }
    document = {
        "schema_version": LOCK_SCHEMA,
        "model": spec.to_lock_mapping(),
        "required_file_sha256": all_hashes,
    }
    root = download_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_FILENAME
    temporary = root / f"{LOCK_FILENAME}.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, lock)
    report["lock_file"] = str(lock)
    report["lock_sha256"] = sha256_file(lock)
    return report


def validate_prepared_artifacts(download_root: Path, spec: ModelSpec) -> dict[str, Any]:
    """Require a matching seal and detect any post-prepare local mutation."""
    root = download_root.expanduser().resolve()
    lock_path = root / LOCK_FILENAME
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"{LOCK_FILENAME} is missing; run `prepare` or `seal` before launch"
        )
    document = _read_json(lock_path)
    if document.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("local model lock schema changed")
    locked_model = document.get("model")
    if not isinstance(locked_model, Mapping):
        raise ValueError("local model lock has no model contract")
    expected_model = spec.to_lock_mapping()
    if dict(locked_model) != expected_model:
        raise ValueError("local model lock does not match the resolved HF contract")
    locked_hashes = document.get("required_file_sha256")
    if not isinstance(locked_hashes, Mapping):
        raise ValueError("local model lock has no required-file hashes")
    if set(locked_hashes) != set(spec.artifact.required_files):
        raise ValueError("local model lock required-file set changed")
    checkpoint = checkpoint_path(root, spec)
    for name, expected in locked_hashes.items():
        actual = sha256_file(_required_path(checkpoint, str(name)))
        if actual != expected:
            raise ValueError(
                f"prepared artifact changed after validation: {name} "
                f"(expected={expected}, actual={actual})"
            )
    report = validate_artifacts(root, spec)
    report["lock_file"] = str(lock_path)
    report["lock_sha256"] = sha256_file(lock_path)
    report["tamper_check"] = "passed"
    return report


def load_prepared_spec(
    download_root: Path,
    *,
    reference: str,
    revision: str,
) -> ModelSpec:
    """Load the sealed contract without contacting Hugging Face."""
    root = download_root.expanduser().resolve()
    document = _read_json(root / LOCK_FILENAME)
    if document.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("local model lock schema changed")
    locked = document.get("model")
    if not isinstance(locked, Mapping):
        raise ValueError("local model lock has no model contract")
    repo_id = str(locked.get("repo_id", ""))
    locked_revision = str(locked.get("revision", ""))
    if normalize_model_reference(reference) != repo_id:
        raise ValueError(
            f"prepared repo mismatch: requested={reference}, sealed={repo_id}"
        )
    if revision != locked_revision:
        raise ValueError(
            f"prepared revision mismatch: requested={revision}, sealed={locked_revision}"
        )
    return model_spec_from_manifest(
        locked,
        repo_id=repo_id,
        revision=locked_revision,
        source=str(locked.get("manifest_source", "local_lock")),
        allow_local_lock_metadata=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_path(checkpoint: Path, name: str) -> Path:
    path = checkpoint / name
    resolved = path.resolve()
    if checkpoint != resolved and checkpoint not in resolved.parents:
        raise ValueError(f"artifact path escapes checkpoint: {name}")
    return path


def _validate_static_contract(checkpoint: Path, spec: ModelSpec) -> None:
    config = _read_json(checkpoint / "config.json")
    if spec.family == "act_absolute_joint16_v1":
        expected_camera_features = {
            f"observation.images.cam_{index}": {
                "type": "VISUAL",
                "shape": [3, 480, 640],
            }
            for index in range(4)
        }
        expected = {
            "type": "act",
            "n_obs_steps": 1,
            "chunk_size": 30,
            "n_action_steps": 30,
            "input_features": {
                **expected_camera_features,
                "observation.state": {"type": "STATE", "shape": [16]},
            },
            "output_features": {
                "action": {"type": "ACTION", "shape": [16]}
            },
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        }
        mismatch = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatch:
            raise ValueError(f"ACT joint16 contract changed: {mismatch}")
        pre = _read_json(checkpoint / "policy_preprocessor.json")
        post = _read_json(checkpoint / "policy_postprocessor.json")
        if [step.get("registry_name") for step in pre.get("steps", [])] != [
            "rename_observations_processor",
            "to_batch_processor",
            "device_processor",
            "normalizer_processor",
        ]:
            raise ValueError("ACT preprocessor pipeline changed")
        if [step.get("registry_name") for step in post.get("steps", [])] != [
            "unnormalizer_processor",
            "device_processor",
        ]:
            raise ValueError("ACT postprocessor pipeline changed")
    elif spec.family == "groot_absolute_joint_v1":
        if config.get("model_type") != "Gr00tN1d7" or config.get("action_horizon") != 40:
            raise ValueError("raw GR00T absolute-joint config changed")
        processor = _read_json(checkpoint / "processor_config.json")
        modality = (
            processor.get("processor_kwargs", {})
            .get("modality_configs", {})
            .get("new_embodiment", {})
        )
        if (modality.get("video") or {}).get("modality_keys") != [
            "cam_0",
            "cam_1",
            "cam_2",
            "cam_3",
        ]:
            raise ValueError("pick-leg camera order changed")
        if (modality.get("state") or {}).get("modality_keys") != ["robot_q", "hand"]:
            raise ValueError("pick-leg state groups changed")
        if (modality.get("action") or {}).get("modality_keys") != [
            "robot_q",
            "hand",
        ]:
            raise ValueError("pick-leg action groups changed")
    elif spec.family == "groot_relative_eef_v1":
        if (
            config.get("type") != "groot"
            or config.get("embodiment_tag")
            != "real_g1_relative_eef_relative_joints"
            or config.get("use_relative_actions") is not True
            or (config.get("input_features", {}).get("observation.state") or {}).get(
                "shape"
            )
            != [49]
            or (config.get("output_features", {}).get("action") or {}).get("shape")
            != [53]
        ):
            raise ValueError("coarse-insert relative-EEF contract changed")
        if set(config.get("relative_exclude_joints") or ()) != {
            "hand",
            "waist",
            "base_height",
            "navigate",
        }:
            raise ValueError("coarse-insert relative action exclusions changed")
    elif spec.family == "diffusion_chunk_relative_v1":
        expected = {
            "type": "flip_table_native_diffusion_chunk_relative",
            "observation_horizon": 2,
            "action_horizon": 16,
            "action_execution_steps": 8,
            "state_dim": 19,
            "action_dim": 16,
            "clip_sample": False,
        }
        mismatch = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatch:
            raise ValueError(f"Diffusion contract changed: {mismatch}")
        normalization = _read_json(checkpoint / "normalization.json")
        if len((normalization.get("observation.state") or {}).get("mean", ())) != 19:
            raise ValueError("Diffusion state normalization is not 19-D")
        if len((normalization.get("action") or {}).get("mean", ())) != 16:
            raise ValueError("Diffusion action normalization is not 16-D")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
