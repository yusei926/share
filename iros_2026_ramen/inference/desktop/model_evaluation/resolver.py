"""Resolve a local model id or Hugging Face path to a pinned safe contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse

from .registry import (
    CANONICAL_OUTPUT,
    DEPLOYMENT_SCHEMA,
    FAMILY_CONTRACTS,
    LOWER_BODY_OWNER,
    ModelSpec,
    get_model_spec,
    model_spec_from_manifest,
    normalize_model_reference,
)


DEPLOYMENT_MANIFEST = "iros_ramen_deployment.json"
METADATA_BASENAMES = frozenset(
    {
        "config.json",
        "processor_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
)
_METADATA_CONTRACT_KEYS = frozenset(
    {
        "type",
        "model_type",
        "n_obs_steps",
        "input_features",
        "output_features",
        "chunk_size",
        "n_action_steps",
        "normalization_mapping",
        "use_relative_actions",
        "state_dim",
        "action_dim",
        "action_horizon",
        "prediction_horizon",
        "observation_horizon",
        "execution_steps",
        "clip_sample",
        "name",
        "steps",
    }
)
_TRAINING_ONLY_PATH_MARKERS = frozenset(
    {
        "optimizer",
        "scheduler",
        "rng",
        "trainer_state",
        "training_state",
        "scaler",
    }
)


class UnsupportedModelError(RuntimeError):
    """The repo exists, but no safe physical-G1 contract can be established."""


@dataclass(frozen=True)
class ResolvedModel:
    requested: str
    spec: ModelSpec
    resolution_source: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "repo_id": self.spec.repo_id,
            "revision": self.spec.revision,
            "model_id": self.spec.model_id,
            "family": self.spec.family,
            "resolution_source": self.resolution_source,
            "canonical_output": CANONICAL_OUTPUT,
            "lower_body_owner": LOWER_BODY_OWNER,
        }


def parse_hf_reference(reference: str) -> tuple[str, str | None]:
    text = str(reference).strip().rstrip("/")
    embedded_revision: str | None = None
    if text.startswith(("https://", "http://")):
        parsed = urlparse(text)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise ValueError(f"unsupported model URL host: {parsed.netloc!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("Hugging Face URL must contain owner/repository")
        if len(parts) >= 4 and parts[2] in {"tree", "commit"}:
            embedded_revision = parts[3]
        return f"{parts[0]}/{parts[1]}", embedded_revision
    return normalize_model_reference(text), None


def resolve_model(
    reference: str,
    *,
    revision: str | None = None,
    allow_network: bool = True,
) -> ResolvedModel:
    repo_or_alias, embedded_revision = parse_hf_reference(reference)
    if revision is not None and embedded_revision is not None and revision != embedded_revision:
        raise ValueError("URL revision and --revision disagree")
    requested_revision = revision or embedded_revision
    local: ModelSpec | None
    try:
        local = get_model_spec(repo_or_alias)
    except KeyError:
        local = None

    is_repo_reference = "/" in repo_or_alias
    if local is not None and not is_repo_reference:
        if requested_revision is not None and requested_revision != local.revision:
            raise ValueError(
                f"registered model is pinned to {local.revision}, "
                f"not {requested_revision}"
            )
        return ResolvedModel(reference, local, "legacy_catalog")
    if local is not None and requested_revision == local.revision:
        return ResolvedModel(reference, local, "legacy_catalog")
    if not allow_network:
        if local is not None and requested_revision is None:
            return ResolvedModel(reference, local, "legacy_catalog")
        raise UnsupportedModelError(
            f"{reference!r} requires Hugging Face resolution but network is disabled"
        )
    if not is_repo_reference:
        if local is None:
            raise KeyError(f"unknown local model id/alias {reference!r}")
        repo_id = local.repo_id
    else:
        repo_id = repo_or_alias

    api, hf_hub_download = _hub()
    info = api.model_info(
        repo_id,
        revision=requested_revision,
        files_metadata=True,
    )
    resolved_sha = str(info.sha or "")
    if len(resolved_sha) != 40:
        raise RuntimeError(f"HF did not resolve a full commit SHA for {repo_id}")
    if local is not None and resolved_sha == local.revision:
        return ResolvedModel(reference, local, "legacy_catalog")

    try:
        manifest_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=DEPLOYMENT_MANIFEST,
                revision=resolved_sha,
            )
        )
    except Exception as exc:
        drift = (
            ""
            if local is None
            else f" Catalog SHA is {local.revision}, current HF SHA is {resolved_sha}."
        )
        raise UnsupportedModelError(
            f"{repo_id}@{resolved_sha} has no valid {DEPLOYMENT_MANIFEST}."
            f"{drift} Run the onboarding command; actuation is refused."
        ) from exc
    raw_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_bytes)
    if not isinstance(manifest, Mapping):
        raise ValueError("deployment manifest must contain a JSON object")
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    spec = model_spec_from_manifest(
        manifest,
        repo_id=repo_id,
        revision=resolved_sha,
        source=f"hf_manifest_sha256:{manifest_hash}",
    )
    return ResolvedModel(reference, spec, "hf_manifest")


def inspect_hf_model(
    reference: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read only small metadata and LFS pointers; never download model weights."""
    repo_id, embedded_revision = parse_hf_reference(reference)
    requested_revision = revision or embedded_revision
    api, hf_hub_download = _hub()
    info = api.model_info(
        repo_id,
        revision=requested_revision,
        files_metadata=True,
    )
    resolved_sha = str(info.sha or "")
    files = list(info.siblings or ())
    metadata: dict[str, Any] = {}
    manifest_present = False
    manifest_validation: dict[str, Any] = {
        "valid": False,
        "error": "deployment manifest is absent",
    }
    weight_files: list[dict[str, Any]] = []
    for item in files:
        name = str(item.rfilename)
        if name == DEPLOYMENT_MANIFEST:
            manifest_present = True
        if (
            name.endswith((".safetensors", ".pt", ".pth", ".onnx"))
            and _is_inference_artifact(name)
        ):
            lfs = getattr(item, "lfs", None)
            weight_files.append(
                {
                    "path": name,
                    "size": getattr(item, "size", None),
                    "sha256": _lfs_sha256(lfs),
                }
            )
        if Path(name).name not in METADATA_BASENAMES:
            continue
        size = getattr(item, "size", None)
        if size is not None and int(size) > 5_000_000:
            continue
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                revision=resolved_sha,
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            metadata[name] = (
                _contract_metadata_summary(value)
                if isinstance(value, Mapping)
                else {
                "json_type": type(value).__name__
                }
            )
        except Exception as exc:
            metadata[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    if manifest_present:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=DEPLOYMENT_MANIFEST,
                    revision=resolved_sha,
                )
            )
            raw_bytes = path.read_bytes()
            manifest = json.loads(raw_bytes)
            if not isinstance(manifest, Mapping):
                raise ValueError("deployment manifest must contain a JSON object")
            spec = model_spec_from_manifest(
                manifest,
                repo_id=repo_id,
                revision=resolved_sha,
                source=f"hf_manifest_sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
            )
            manifest_validation = {
                "valid": True,
                "model_id": spec.model_id,
                "family": spec.family,
            }
        except Exception as exc:
            manifest_validation = {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    candidates = _candidate_families(metadata)
    unresolved = [] if manifest_present else [
        "exact camera role and modality mapping",
        "state dimension ordering, units, and physical source",
        "action ordering, units, absolute/relative reference",
        "Dex1 representation and scale",
        "trusted local loader family and environment",
        "default checkpoint when multiple checkpoints exist",
    ]
    if not weight_files:
        unresolved.append("no model weight artifact found")
    if any(not item["sha256"] for item in weight_files):
        unresolved.append("one or more weight files have no Hub LFS SHA-256")
    return {
        "repo_id": repo_id,
        "revision": resolved_sha,
        "manifest_present": manifest_present,
        "manifest_validation": manifest_validation,
        "candidate_families": candidates,
        "metadata_files": metadata,
        "weight_files": weight_files,
        "unresolved_contract_fields": unresolved,
        "safe_for_actuation": bool(manifest_validation["valid"]),
    }


def onboarding_manifest_draft(
    reference: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    audit = inspect_hf_model(reference, revision=revision)
    candidates = audit["candidate_families"]
    family = candidates[0] if len(candidates) == 1 else "REQUIRED"
    contract = FAMILY_CONTRACTS.get(family)
    weights = {
        item["path"]: item["sha256"]
        for item in audit["weight_files"]
        if item.get("sha256")
    }
    required = sorted(weights)
    return {
        "schema_version": DEPLOYMENT_SCHEMA,
        "model_id": audit["repo_id"].replace("/", "__"),
        "family": family,
        "task": "REQUIRED",
        "camera_roles": (
            sorted(contract.required_camera_roles) if contract else ["REQUIRED"]
        ),
        "observation_horizon": (
            contract.observation_horizon if contract else "REQUIRED"
        ),
        "model_state_dim": contract.state_dim if contract else "REQUIRED",
        "model_action_dim": contract.action_dim if contract else "REQUIRED",
        "model_action_horizon": contract.action_horizon if contract else "REQUIRED",
        "execution_steps": "REQUIRED",
        "state_semantics": (
            contract.state_semantics
            if contract
            else "REQUIRED_EXACT_ORDER_UNITS_AND_SOURCE"
        ),
        "action_semantics": (
            contract.action_semantics
            if contract
            else "REQUIRED_EXACT_ORDER_UNITS_AND_REFERENCE"
        ),
        "canonical_output": CANONICAL_OUTPUT,
        "lower_body_owner": LOWER_BODY_OWNER,
        "artifact": {
            "checkpoint_subdir": "",
            "required_files": required,
            "allow_patterns": required,
            "file_sha256": weights,
        },
        "_onboarding": {
            "repo_id": audit["repo_id"],
            "resolved_revision": audit["revision"],
            "candidate_families": candidates,
            "unresolved_contract_fields": audit["unresolved_contract_fields"],
            "instruction": (
                "Resolve every REQUIRED field, add this file to the HF repo as "
                f"{DEPLOYMENT_MANIFEST}, and commit it. Do not actuate from this draft."
            ),
        },
    }


def _candidate_families(metadata: Mapping[str, Any]) -> list[str]:
    result: set[str] = set()
    for name, value in metadata.items():
        if not isinstance(value, Mapping) or Path(name).name != "config.json":
            continue
        kind = value.get("type") or value.get("model_type")
        if kind == "Gr00tN1d7":
            result.add("groot_absolute_joint_v1")
        if (
            kind == "groot"
            and value.get("use_relative_actions") is True
            and (value.get("input_features", {}).get("observation.state") or {}).get(
                "shape"
            )
            == [49]
            and (value.get("output_features", {}).get("action") or {}).get("shape")
            == [53]
        ):
            result.add("groot_relative_eef_v1")
        if (
            kind == "flip_table_native_diffusion_chunk_relative"
            and value.get("state_dim") == 19
            and value.get("action_dim") == 16
            and value.get("clip_sample") is False
        ):
            result.add("diffusion_chunk_relative_v1")
    return sorted(result)


def _lfs_sha256(lfs: Any) -> str | None:
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    return value if isinstance(value, str) and len(value) == 64 else None


def _is_inference_artifact(name: str) -> bool:
    parts = {
        part.lower().replace("-", "_")
        for part in PurePosixPath(name).parts
    }
    return not any(
        marker in part
        for part in parts
        for marker in _TRAINING_ONLY_PATH_MARKERS
    )


def _contract_metadata_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only deployment-contract fields; never expose training paths/lists."""
    summary: dict[str, Any] = {}
    for key in _METADATA_CONTRACT_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key == "steps" and isinstance(item, list):
            summary[key] = [
                {
                    "registry_name": step.get("registry_name"),
                    "has_state_file": bool(step.get("state_file")),
                }
                for step in item
                if isinstance(step, Mapping)
            ]
        else:
            summary[key] = item
    return summary


def _hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is unavailable; use `pixi run -e model-eval ...`"
        ) from exc
    return HfApi(), hf_hub_download
