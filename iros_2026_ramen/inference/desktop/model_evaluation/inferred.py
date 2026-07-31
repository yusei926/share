"""Read-only inference contracts inferred from Hugging Face metadata.

These contracts deliberately cannot be converted to :class:`ModelSpec` and
cannot be consumed by the physical-G1 launcher.  They are useful for answering
"can this checkpoint be loaded and produce a finite tensor?" before a model
author has supplied the exact joint/camera semantics required for actuation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from .registry import load_registry, normalize_model_reference


OFFLINE_CONTRACT_SCHEMA = "team_ramen_inferred_offline_contract/v1"
SAFE_CONFIG_TYPES = frozenset(
    {
        "act",
        "pi05",
        "flip_table_native_act_chunk_relative",
        "flip_table_native_diffusion_chunk_relative",
        "furniture_groot",
    }
)
NATIVE_TYPES = frozenset(
    {
        "flip_table_native_act_chunk_relative",
        "flip_table_native_diffusion_chunk_relative",
    }
)
PERCEPTION_MARKERS = ("yolo", "vit_phase")
_IGNORED_ARTIFACT_PARTS = frozenset(
    {
        "optimizer.pt",
        "optimizer_state.safetensors",
        "scheduler.pt",
        "training_args.bin",
        "rng_state.pth",
        "rng_state.safetensors",
    }
)


@dataclass(frozen=True)
class InferredOfflineContract:
    schema_version: str
    repo_id: str
    revision: str
    category: str
    config_type: str | None
    loader_kind: str | None
    checkpoint_subdir: str
    config_path: str | None
    state_dim: int | None
    action_dim: int | None
    observation_horizon: int | None
    action_horizon: int | None
    execution_steps: int | None
    camera_keys: tuple[str, ...]
    image_shapes: Mapping[str, tuple[int, ...]]
    normalization: Mapping[str, str]
    required_files: tuple[str, ...]
    file_sha256: Mapping[str, str]
    total_download_bytes: int
    confidence: str
    issues: tuple[str, ...]
    actuation_allowed: bool = False
    physical_mapping_verified: bool = False

    @property
    def weight_load_supported(self) -> bool:
        return (
            self.loader_kind is not None
            and self.config_type in SAFE_CONFIG_TYPES
            and any(name.endswith(".safetensors") for name in self.required_files)
        )

    @property
    def offline_test_supported(self) -> bool:
        return self.category == "registered_physical" or self.weight_load_supported

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["weight_load_supported"] = self.weight_load_supported
        result["offline_test_supported"] = self.offline_test_supported
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferredOfflineContract":
        if value.get("schema_version") != OFFLINE_CONTRACT_SCHEMA:
            raise ValueError("inferred offline contract schema changed")
        if value.get("actuation_allowed") is not False:
            raise ValueError("inferred contracts must never allow actuation")
        if value.get("physical_mapping_verified") is not False:
            raise ValueError("inferred contracts cannot verify physical mappings")
        return cls(
            schema_version=OFFLINE_CONTRACT_SCHEMA,
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            category=str(value["category"]),
            config_type=(
                None if value.get("config_type") is None else str(value["config_type"])
            ),
            loader_kind=(
                None if value.get("loader_kind") is None else str(value["loader_kind"])
            ),
            checkpoint_subdir=str(value.get("checkpoint_subdir", "")),
            config_path=(
                None if value.get("config_path") is None else str(value["config_path"])
            ),
            state_dim=_optional_int(value.get("state_dim")),
            action_dim=_optional_int(value.get("action_dim")),
            observation_horizon=_optional_int(value.get("observation_horizon")),
            action_horizon=_optional_int(value.get("action_horizon")),
            execution_steps=_optional_int(value.get("execution_steps")),
            camera_keys=tuple(str(item) for item in value.get("camera_keys", ())),
            image_shapes={
                str(key): tuple(int(item) for item in shape)
                for key, shape in dict(value.get("image_shapes", {})).items()
            },
            normalization={
                str(key): str(item)
                for key, item in dict(value.get("normalization", {})).items()
            },
            required_files=tuple(
                str(item) for item in value.get("required_files", ())
            ),
            file_sha256={
                str(key): str(item)
                for key, item in dict(value.get("file_sha256", {})).items()
            },
            total_download_bytes=int(value.get("total_download_bytes", 0)),
            confidence=str(value.get("confidence", "none")),
            issues=tuple(str(item) for item in value.get("issues", ())),
        )


def list_namespace_models(namespace: str = "Team-RAMEN") -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    return sorted(model.id for model in api.list_models(author=namespace))


def infer_offline_contract(
    reference: str,
    *,
    revision: str | None = None,
    api: Any | None = None,
) -> InferredOfflineContract:
    """Infer a non-actuating contract from small metadata files only."""
    repo_id = normalize_model_reference(reference)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    from huggingface_hub import hf_hub_download
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    resolved_revision = str(info.sha)
    siblings = {item.rfilename: item for item in info.siblings}
    names = sorted(siblings)
    registered = {
        spec.repo_id: spec for spec in load_registry().values()
    }.get(repo_id)
    if registered is not None and resolved_revision == registered.revision:
        return InferredOfflineContract(
            schema_version=OFFLINE_CONTRACT_SCHEMA,
            repo_id=repo_id,
            revision=registered.revision,
            category="registered_physical",
            config_type=registered.family,
            loader_kind="registered_worker",
            checkpoint_subdir=registered.artifact.checkpoint_subdir,
            config_path="config.json",
            state_dim=registered.model_state_dim,
            action_dim=registered.model_action_dim,
            observation_horizon=registered.observation_horizon,
            action_horizon=registered.model_action_horizon,
            execution_steps=registered.execution_steps,
            camera_keys=tuple(registered.camera_roles),
            image_shapes={},
            normalization={},
            required_files=registered.artifact.required_files,
            file_sha256=registered.artifact.file_sha256,
            total_download_bytes=0,
            confidence="verified",
            issues=(
                "use adapter-dry-run/offline-model-dry-run for this reviewed model; "
                "the inferred-model launcher remains disabled",
            ),
        )

    config_candidates: list[tuple[str, dict[str, Any]]] = []
    for name in names:
        if not name.endswith("config.json"):
            continue
        if PurePosixPath(name).name in {
            "processor_config.json",
            "train_config.json",
        }:
            continue
        item = siblings[name]
        size = int(getattr(item, "size", 0) or 0)
        if size > 2 * 1024 * 1024:
            continue
        try:
            path = hf_hub_download(
                repo_id,
                name,
                revision=resolved_revision,
            )
            payload = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            config_candidates.append((name, payload))

    selected = _select_config(config_candidates, names)
    issues: list[str] = []
    if selected is None:
        weight_names = [
            name for name in names if name.endswith((".pt", ".pth", ".safetensors"))
        ]
        if not weight_names:
            category = "incomplete"
            issues.append("repository contains no recognized config or model weight")
        elif any(marker in repo_id.lower() for marker in PERCEPTION_MARKERS):
            category = "perception_structure"
            issues.append(
                "perception checkpoint is not an upper-body policy and has no safe "
                "generic loader contract"
            )
        else:
            category = "opaque_structure"
            issues.append(
                "opaque or ambiguous checkpoint; automatic pickle loading is disabled"
            )
        if len(weight_names) > 1:
            issues.append(
                f"{len(weight_names)} weight candidates exist and no release checkpoint "
                "can be selected safely"
            )
        return InferredOfflineContract(
            schema_version=OFFLINE_CONTRACT_SCHEMA,
            repo_id=repo_id,
            revision=resolved_revision,
            category=category,
            config_type=None,
            loader_kind=None,
            checkpoint_subdir="",
            config_path=None,
            state_dim=None,
            action_dim=None,
            observation_horizon=None,
            action_horizon=None,
            execution_steps=None,
            camera_keys=(),
            image_shapes={},
            normalization={},
            required_files=(),
            file_sha256={},
            total_download_bytes=0,
            confidence="none",
            issues=tuple(issues),
        )

    config_path, config = selected
    config_type = str(config.get("type") or config.get("model_type") or "")
    checkpoint_subdir = str(PurePosixPath(config_path).parent)
    if checkpoint_subdir == ".":
        checkpoint_subdir = ""
    state_dim = _feature_dim(config, "input_features", "observation.state")
    action_dim = _feature_dim(config, "output_features", "action")
    camera_keys, image_shapes = _camera_contract(config)
    if config_type in NATIVE_TYPES and not camera_keys:
        camera_keys = (
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        )
        image_shapes = {key: (3, 480, 640) for key in camera_keys}
    normalization = {
        str(key): str(value)
        for key, value in dict(config.get("normalization_mapping") or {}).items()
    }
    required_repo_paths = _artifact_paths(
        names,
        checkpoint_subdir=checkpoint_subdir,
        config_type=config_type,
    )
    relative_required = tuple(
        _relative_to_checkpoint(name, checkpoint_subdir)
        for name in required_repo_paths
    )
    file_sha256 = {
        _relative_to_checkpoint(name, checkpoint_subdir): digest
        for name in required_repo_paths
        if (digest := _lfs_sha256(siblings[name])) is not None
    }
    total_bytes = sum(
        int(getattr(siblings[name], "size", 0) or 0) for name in required_repo_paths
    )

    if config_type in {"act", "pi05"}:
        loader_kind = "lerobot"
        category = "lerobot_offline"
        confidence = "high"
    elif config_type in NATIVE_TYPES:
        loader_kind = "native"
        category = "native_offline"
        confidence = "high"
    elif config_type == "furniture_groot":
        loader_kind = "furniture_groot"
        category = "lerobot_offline"
        confidence = "medium"
    else:
        loader_kind = None
        category = "opaque_structure"
        confidence = "low"
        issues.append(f"unsupported config type {config_type!r}")

    if not any(name.endswith(".safetensors") for name in relative_required):
        loader_kind = None
        issues.append("no safetensors release weight selected")
    if config_type == "flip_table_native_diffusion_chunk_relative":
        if config.get("clip_sample") is True:
            issues.append(
                "clip_sample=true: offline inference is testable, but this checkpoint "
                "must not be promoted to the z-score physical contract"
            )
    if state_dim is None or action_dim is None:
        if config_type in NATIVE_TYPES:
            state_dim = _optional_int(config.get("state_dim"))
            action_dim = _optional_int(config.get("action_dim"))
        if state_dim is None or action_dim is None:
            issues.append("state/action dimension could not be inferred")

    return InferredOfflineContract(
        schema_version=OFFLINE_CONTRACT_SCHEMA,
        repo_id=repo_id,
        revision=resolved_revision,
        category=category,
        config_type=config_type or None,
        loader_kind=loader_kind,
        checkpoint_subdir=checkpoint_subdir,
        config_path=_relative_to_checkpoint(config_path, checkpoint_subdir),
        state_dim=state_dim,
        action_dim=action_dim,
        observation_horizon=_optional_int(
            config.get("n_obs_steps", config.get("observation_horizon"))
        ),
        action_horizon=_optional_int(
            config.get("chunk_size", config.get("action_horizon"))
        ),
        execution_steps=_optional_int(
            config.get("n_action_steps", config.get("action_execution_steps"))
        ),
        camera_keys=camera_keys,
        image_shapes=image_shapes,
        normalization=normalization,
        required_files=relative_required,
        file_sha256=file_sha256,
        total_download_bytes=total_bytes,
        confidence=confidence,
        issues=tuple(issues),
    )


def audit_namespace(namespace: str = "Team-RAMEN") -> dict[str, Any]:
    models = []
    for repo_id in list_namespace_models(namespace):
        try:
            models.append(infer_offline_contract(repo_id).to_mapping())
        except Exception as exc:  # one malformed repo must not hide the remaining org
            models.append(
                {
                    "repo_id": repo_id,
                    "category": "audit_error",
                    "actuation_allowed": False,
                    "physical_mapping_verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    counts: dict[str, int] = {}
    for model in models:
        category = str(model["category"])
        counts[category] = counts.get(category, 0) + 1
    return {
        "schema_version": "team_ramen_namespace_model_audit/v1",
        "namespace": namespace,
        "model_count": len(models),
        "category_counts": counts,
        "weight_load_supported_count": sum(
            bool(model.get("weight_load_supported")) for model in models
        ),
        "offline_test_supported_count": sum(
            bool(model.get("offline_test_supported")) for model in models
        ),
        "registered_physical_count": sum(
            model.get("category") == "registered_physical" for model in models
        ),
        "actuation_allowed_count": 0,
        "models": models,
    }


def _select_config(
    candidates: Iterable[tuple[str, dict[str, Any]]],
    names: Iterable[str],
) -> tuple[str, dict[str, Any]] | None:
    recognized = [
        item
        for item in candidates
        if str(item[1].get("type") or item[1].get("model_type") or "")
        in SAFE_CONFIG_TYPES
    ]
    if not recognized:
        return None
    names_set = set(names)

    def usable(item: tuple[str, dict[str, Any]]) -> bool:
        parent = str(PurePosixPath(item[0]).parent)
        parent = "" if parent == "." else parent
        prefix = f"{parent}/" if parent else ""
        return any(
            name.startswith(prefix) and name.endswith(".safetensors")
            for name in names_set
        )

    recognized = [item for item in recognized if usable(item)]
    if not recognized:
        return None
    for preferred in ("config.json", "pretrained_model/config.json"):
        for item in recognized:
            if item[0] == preferred:
                return item
    if len(recognized) == 1:
        return recognized[0]
    pretrained = [item for item in recognized if item[0].endswith("/pretrained_model/config.json")]
    if len(pretrained) == 1:
        return pretrained[0]
    # Multiple checkpoint snapshots without a canonical release are ambiguous.
    return None


def _artifact_paths(
    names: Iterable[str],
    *,
    checkpoint_subdir: str,
    config_type: str,
) -> tuple[str, ...]:
    prefix = f"{checkpoint_subdir}/" if checkpoint_subdir else ""
    selected: list[str] = []
    for name in names:
        if prefix and not name.startswith(prefix):
            continue
        relative = name[len(prefix) :] if prefix else name
        if "/" in relative and not (
            config_type == "flip_table_native_diffusion_chunk_relative"
            and relative == "_diffusion_config/config.json"
        ):
            continue
        basename = PurePosixPath(name).name
        if basename in _IGNORED_ARTIFACT_PARTS:
            continue
        if (
            basename in {
                "config.json",
                "train_config.json",
                "normalization.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "processor_config.json",
                "model.safetensors.index.json",
            }
            or basename.endswith(".safetensors")
            or relative == "_diffusion_config/config.json"
        ):
            selected.append(name)
    return tuple(sorted(selected))


def _relative_to_checkpoint(path: str, checkpoint_subdir: str) -> str:
    prefix = f"{checkpoint_subdir}/" if checkpoint_subdir else ""
    if prefix and not path.startswith(prefix):
        raise ValueError(f"{path} is outside selected checkpoint {checkpoint_subdir}")
    return path[len(prefix) :] if prefix else path


def _feature_dim(config: Mapping[str, Any], group: str, key: str) -> int | None:
    raw = config.get(group)
    if not isinstance(raw, Mapping):
        return None
    feature = raw.get(key)
    if not isinstance(feature, Mapping):
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def _camera_contract(
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    raw = config.get("input_features")
    if not isinstance(raw, Mapping):
        return (), {}
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in raw.items():
        if not str(key).startswith("observation.images.") or not isinstance(value, Mapping):
            continue
        shape = value.get("shape")
        if isinstance(shape, list):
            shapes[str(key)] = tuple(int(item) for item in shape)
    return tuple(shapes), shapes


def _lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if not value:
        return None
    value = str(value).removeprefix("sha256:")
    return value if len(value) == 64 else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
