"""Exact-file, commit-pinned artifacts for inferred offline-only contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .inferred import InferredOfflineContract


OFFLINE_LOCK_FILENAME = ".iros_ramen_inferred_offline_lock.json"
OFFLINE_LOCK_SCHEMA = "team_ramen_inferred_offline_lock/v1"


def download_plan(
    contract: InferredOfflineContract, local_dir: Path
) -> dict[str, Any]:
    if not contract.weight_load_supported:
        raise ValueError(
            f"{contract.repo_id} has no supported, unambiguous offline weight contract"
        )
    prefix = (
        f"{contract.checkpoint_subdir}/" if contract.checkpoint_subdir else ""
    )
    return {
        "repo_id": contract.repo_id,
        "revision": contract.revision,
        "local_dir": str(local_dir.expanduser().resolve()),
        "allow_patterns": [prefix + name for name in contract.required_files],
    }


def prepare(
    contract: InferredOfflineContract, local_dir: Path
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot_download(**download_plan(contract, local_dir))
    return seal(contract, local_dir)


def checkpoint_path(
    contract: InferredOfflineContract, local_dir: Path
) -> Path:
    root = local_dir.expanduser().resolve()
    path = (
        root / contract.checkpoint_subdir
        if contract.checkpoint_subdir
        else root
    ).resolve()
    if path != root and root not in path.parents:
        raise ValueError("inferred checkpoint path escapes local directory")
    return path


def seal(
    contract: InferredOfflineContract, local_dir: Path
) -> dict[str, Any]:
    root = local_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(contract, root)
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for name in contract.required_files:
        path = _safe_path(checkpoint, name)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
            continue
        actual = _sha256(path)
        expected = contract.file_sha256.get(name)
        if expected is not None and actual != expected:
            raise ValueError(
                f"{contract.repo_id}: LFS SHA-256 mismatch for {name}: "
                f"expected={expected}, actual={actual}"
            )
        hashes[name] = actual
    if missing:
        raise FileNotFoundError(f"incomplete inferred checkpoint: {missing}")
    document = {
        "schema_version": OFFLINE_LOCK_SCHEMA,
        "contract": contract.to_mapping(),
        "required_file_sha256": hashes,
        "physical_launcher_compatible": False,
        "actuation_allowed": False,
    }
    target = root / OFFLINE_LOCK_FILENAME
    temporary = root / f"{OFFLINE_LOCK_FILENAME}.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return validate(root, expected=contract)


def validate(
    local_dir: Path,
    *,
    expected: InferredOfflineContract | None = None,
) -> dict[str, Any]:
    root = local_dir.expanduser().resolve()
    path = root / OFFLINE_LOCK_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != OFFLINE_LOCK_SCHEMA:
        raise ValueError("inferred offline lock schema changed")
    if payload.get("physical_launcher_compatible") is not False:
        raise ValueError("inferred lock must not be physical-launcher compatible")
    if payload.get("actuation_allowed") is not False:
        raise ValueError("inferred lock unexpectedly allows actuation")
    contract = InferredOfflineContract.from_mapping(payload["contract"])
    if expected is not None and contract.to_mapping() != expected.to_mapping():
        raise ValueError("local inferred lock does not match current HF contract")
    hashes = payload.get("required_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(contract.required_files):
        raise ValueError("inferred lock required-file set changed")
    checkpoint = checkpoint_path(contract, root)
    for name, digest in hashes.items():
        actual = _sha256(_safe_path(checkpoint, str(name)))
        if actual != digest:
            raise ValueError(f"inferred artifact changed after sealing: {name}")
    return {
        "repo_id": contract.repo_id,
        "revision": contract.revision,
        "checkpoint": str(checkpoint),
        "required_file_count": len(contract.required_files),
        "lock_file": str(path),
        "tamper_check": "passed",
        "actuation_allowed": False,
        "robot_command_sent": False,
    }


def load_contract(local_dir: Path) -> InferredOfflineContract:
    root = local_dir.expanduser().resolve()
    payload = json.loads(
        (root / OFFLINE_LOCK_FILENAME).read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != OFFLINE_LOCK_SCHEMA:
        raise ValueError("inferred offline lock schema changed")
    return InferredOfflineContract.from_mapping(payload["contract"])


def _safe_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact escapes checkpoint: {name}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
