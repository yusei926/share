"""Fail-closed model registry for heterogeneous physical-G1 policies.

Only declarative data lives in a model manifest. Executable runner/worker
paths and the canonical physical output are owned by trusted local family
plugins. This prevents a remote Hugging Face repository from selecting code
to execute on the robot workstation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping
from urllib.parse import urlparse


REGISTRY_PATH = Path(__file__).with_name("models.json")
DEPLOYMENT_SCHEMA = "team_ramen_g1_deployment/v1"
CANONICAL_OUTPUT = "arm14_absolute_rad+dex1_opening_fraction2"
LOWER_BODY_OWNER = "unitree_regular_mode"
HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class FamilyContract:
    family: str
    runner: str
    worker: str
    state_dim: int
    action_dim: int
    action_horizon: int
    observation_horizon: int
    state_semantics: str
    action_semantics: str
    allowed_camera_roles: frozenset[str]
    required_camera_roles: frozenset[str]


FAMILY_CONTRACTS: Mapping[str, FamilyContract] = {
    "act_absolute_joint16_v1": FamilyContract(
        family="act_absolute_joint16_v1",
        runner="inference.desktop.upper_policy.run_pick_leg_act",
        worker=(
            "model/subtask_policy_training/deployment/"
            "real_act_joint16_worker.py"
        ),
        state_dim=16,
        action_dim=16,
        action_horizon=30,
        observation_horizon=1,
        state_semantics="g1_sdk_arms14+dex1_physical2",
        action_semantics=(
            "absolute_arms14_rad+dex1_physical2;"
            "clamp_to_serialized_training_support"
        ),
        allowed_camera_roles=frozenset(
            {"head_left", "head_right", "left_wrist", "right_wrist"}
        ),
        required_camera_roles=frozenset(
            {"head_left", "head_right", "left_wrist", "right_wrist"}
        ),
    ),
    "groot_absolute_joint_v1": FamilyContract(
        family="groot_absolute_joint_v1",
        runner="inference.desktop.upper_policy.run_pick_leg_groot",
        worker="model/subtask_policy_training/deployment/real_groot_n17_worker.py",
        state_dim=38,
        action_dim=38,
        action_horizon=16,
        observation_horizon=1,
        state_semantics="root_xyz_wxyz7+g1_sdk_body29+dex1_physical2",
        action_semantics=(
            "absolute_root7+body29+dex1_physical2;"
            "execute_body[15:29]+hands_only"
        ),
        allowed_camera_roles=frozenset(
            {"head_left", "head_right", "left_wrist", "right_wrist"}
        ),
        required_camera_roles=frozenset(
            {"head_left", "head_right", "left_wrist", "right_wrist"}
        ),
    ),
    "groot_relative_eef_v1": FamilyContract(
        family="groot_relative_eef_v1",
        runner="inference.desktop.upper_policy.run_coarse_insert_groot",
        worker=(
            "model/subtask_policy_training/deployment/"
            "real_coarse_insert_groot_n17_worker.py"
        ),
        state_dim=49,
        action_dim=53,
        action_horizon=16,
        observation_horizon=1,
        state_semantics="eef_xyz_rot6d18+hand7x2+arms14+waist3",
        action_semantics=(
            "decoded_absolute53;execute_arms[32:46]+"
            "dex1(left18,right25);discard_eef,waist,base,navigation"
        ),
        allowed_camera_roles=frozenset(
            {"head_left", "head_right", "left_wrist", "right_wrist"}
        ),
        required_camera_roles=frozenset(
            {"head_left", "left_wrist", "right_wrist"}
        ),
    ),
    "diffusion_chunk_relative_v1": FamilyContract(
        family="diffusion_chunk_relative_v1",
        runner="inference.desktop.upper_policy.run_flip_table_diffusion",
        worker="model/subtask_policy_training/deployment/real_diffusion_worker.py",
        state_dim=19,
        action_dim=16,
        action_horizon=16,
        observation_horizon=2,
        state_semantics="waist3+arms14+dex1_physical2",
        action_semantics=(
            "arm14_relative_to_measured_chunk_start+"
            "dex1_physical2_absolute;zscore;clip_sample_false"
        ),
        allowed_camera_roles=frozenset(
            {"head_left", "left_wrist", "right_wrist"}
        ),
        required_camera_roles=frozenset(
            {"head_left", "left_wrist", "right_wrist"}
        ),
    ),
}
SUPPORTED_FAMILIES = frozenset(FAMILY_CONTRACTS)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "aliases",
        "family",
        "task",
        "camera_roles",
        "observation_horizon",
        "model_state_dim",
        "model_action_dim",
        "model_action_horizon",
        "execution_steps",
        "state_semantics",
        "action_semantics",
        "canonical_output",
        "lower_body_owner",
        "artifact",
        "expected_model_sha256",
    }
)
_LOCAL_LOCK_ONLY_FIELDS = frozenset(
    {"repo_id", "revision", "manifest_source"}
)
_ARTIFACT_FIELDS = frozenset(
    {"checkpoint_subdir", "required_files", "allow_patterns", "file_sha256"}
)


@dataclass(frozen=True)
class ArtifactSpec:
    checkpoint_subdir: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...]
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    aliases: tuple[str, ...]
    repo_id: str
    revision: str
    family: str
    task: str
    camera_roles: tuple[str, ...]
    observation_horizon: int
    model_state_dim: int
    model_action_dim: int
    model_action_horizon: int
    execution_steps: int
    state_semantics: str
    action_semantics: str
    runner: str
    worker: str
    artifact: ArtifactSpec
    expected_model_sha256: str | None = None
    manifest_source: str = "legacy_catalog"

    @property
    def canonical_output(self) -> str:
        return CANONICAL_OUTPUT

    @property
    def lower_body_command_dimensions(self) -> int:
        return 0

    @classmethod
    def from_mapping(
        cls,
        model_id: str,
        raw: Mapping[str, Any],
        *,
        repo_id_override: str | None = None,
        revision_override: str | None = None,
        manifest_source: str = "legacy_catalog",
    ) -> "ModelSpec":
        family = _nonempty(raw.get("family"), f"{model_id}.family")
        try:
            contract = FAMILY_CONTRACTS[family]
        except KeyError as exc:
            raise ValueError(f"{model_id}: unsupported family {family!r}") from exc
        artifact_raw = _mapping(raw.get("artifact"), f"{model_id}.artifact")
        artifact = ArtifactSpec(
            checkpoint_subdir=_safe_relative_dir(
                artifact_raw.get("checkpoint_subdir", ""),
                f"{model_id}.checkpoint_subdir",
            ),
            required_files=tuple(
                _safe_relative_file(name, f"{model_id}.required_files")
                for name in _strings(
                    artifact_raw.get("required_files"), f"{model_id}.required_files"
                )
            ),
            allow_patterns=tuple(
                _safe_pattern(name, f"{model_id}.allow_patterns")
                for name in _strings(
                    artifact_raw.get("allow_patterns"), f"{model_id}.allow_patterns"
                )
            ),
            file_sha256={
                _safe_relative_file(str(name), f"{model_id}.file_sha256"): _sha256(
                    value, f"{model_id}.file_sha256.{name}"
                )
                for name, value in _mapping(
                    artifact_raw.get("file_sha256", {}),
                    f"{model_id}.file_sha256",
                ).items()
            },
        )
        raw_runner = raw.get("runner", contract.runner)
        raw_worker = raw.get("worker", contract.worker)
        if raw_runner != contract.runner or raw_worker != contract.worker:
            raise ValueError(
                f"{model_id}: runner/worker must be the trusted local family plugin"
            )
        spec = cls(
            model_id=_model_id(model_id),
            aliases=_strings(raw.get("aliases", ()), f"{model_id}.aliases"),
            repo_id=_repo_id(
                repo_id_override or raw.get("repo_id"), f"{model_id}.repo_id"
            ),
            revision=_git_sha(
                revision_override or raw.get("revision"), f"{model_id}.revision"
            ),
            family=family,
            task=_nonempty(raw.get("task"), f"{model_id}.task"),
            camera_roles=_strings(
                raw.get("camera_roles"), f"{model_id}.camera_roles"
            ),
            observation_horizon=_positive_int(
                raw.get("observation_horizon"), f"{model_id}.observation_horizon"
            ),
            model_state_dim=_positive_int(
                raw.get("model_state_dim"), f"{model_id}.model_state_dim"
            ),
            model_action_dim=_positive_int(
                raw.get("model_action_dim"), f"{model_id}.model_action_dim"
            ),
            model_action_horizon=_positive_int(
                raw.get("model_action_horizon"), f"{model_id}.model_action_horizon"
            ),
            execution_steps=_positive_int(
                raw.get("execution_steps"), f"{model_id}.execution_steps"
            ),
            state_semantics=_nonempty(
                raw.get("state_semantics"), f"{model_id}.state_semantics"
            ),
            action_semantics=_nonempty(
                raw.get("action_semantics"), f"{model_id}.action_semantics"
            ),
            runner=contract.runner,
            worker=contract.worker,
            artifact=artifact,
            expected_model_sha256=(
                None
                if raw.get("expected_model_sha256") is None
                else _sha256(
                    raw.get("expected_model_sha256"),
                    f"{model_id}.expected_model_sha256",
                )
            ),
            manifest_source=manifest_source,
        )
        _validate_spec(spec, contract)
        return spec

    def to_lock_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DEPLOYMENT_SCHEMA,
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "family": self.family,
            "task": self.task,
            "camera_roles": list(self.camera_roles),
            "observation_horizon": self.observation_horizon,
            "model_state_dim": self.model_state_dim,
            "model_action_dim": self.model_action_dim,
            "model_action_horizon": self.model_action_horizon,
            "execution_steps": self.execution_steps,
            "state_semantics": self.state_semantics,
            "action_semantics": self.action_semantics,
            "canonical_output": self.canonical_output,
            "lower_body_owner": LOWER_BODY_OWNER,
            "artifact": {
                "checkpoint_subdir": self.artifact.checkpoint_subdir,
                "required_files": list(self.artifact.required_files),
                "allow_patterns": list(self.artifact.allow_patterns),
                "file_sha256": dict(self.artifact.file_sha256),
            },
            "expected_model_sha256": self.expected_model_sha256,
            "manifest_source": self.manifest_source,
        }


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, ModelSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("model registry schema_version must be 1")
    models = _mapping(raw.get("models"), "models")
    result = {
        str(model_id): ModelSpec.from_mapping(str(model_id), _mapping(value, model_id))
        for model_id, value in models.items()
    }
    identities: dict[str, str] = {}
    for model_id, spec in result.items():
        for name in (model_id, spec.repo_id, *spec.aliases):
            normalized = normalize_model_reference(name)
            if normalized in identities:
                raise ValueError(f"duplicate model id/alias/repo {name!r}")
            identities[normalized] = model_id
    return result


def get_model_spec(name: str, path: Path = REGISTRY_PATH) -> ModelSpec:
    needle = normalize_model_reference(name)
    matches = [
        spec
        for model_id, spec in load_registry(path).items()
        if needle
        in {
            normalize_model_reference(value)
            for value in (model_id, spec.repo_id, *spec.aliases)
        }
    ]
    if len(matches) != 1:
        available = sorted(spec.repo_id for spec in load_registry(path).values())
        raise KeyError(f"unknown model {name!r}; registered_repos={available}")
    return matches[0]


def model_spec_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    source: str,
    allow_local_lock_metadata: bool = False,
) -> ModelSpec:
    allowed = (
        _MANIFEST_FIELDS | _LOCAL_LOCK_ONLY_FIELDS
        if allow_local_lock_metadata
        else _MANIFEST_FIELDS
    )
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(
            f"deployment manifest contains unsupported fields: {sorted(unknown)}"
        )
    artifact = manifest.get("artifact")
    if isinstance(artifact, Mapping):
        artifact_unknown = set(artifact) - _ARTIFACT_FIELDS
        if artifact_unknown:
            raise ValueError(
                "deployment manifest artifact contains unsupported fields: "
                f"{sorted(artifact_unknown)}"
            )
    if manifest.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError(
            f"deployment manifest schema must be {DEPLOYMENT_SCHEMA!r}"
        )
    if manifest.get("canonical_output") != CANONICAL_OUTPUT:
        raise ValueError("deployment manifest canonical_output is not physical-G1 safe")
    if manifest.get("lower_body_owner") != LOWER_BODY_OWNER:
        raise ValueError("deployment manifest must leave lower body to Regular Mode")
    model_id = _model_id(
        manifest.get("model_id") or repo_id.replace("/", "__")
    )
    return ModelSpec.from_mapping(
        model_id,
        manifest,
        repo_id_override=repo_id,
        revision_override=revision,
        manifest_source=source,
    )


def normalize_model_reference(value: str) -> str:
    text = str(value).strip().rstrip("/")
    if text.startswith(("https://", "http://")):
        parsed = urlparse(text)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise ValueError(f"unsupported model URL host: {parsed.netloc!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("Hugging Face model URL must contain owner/repository")
        text = f"{parts[0]}/{parts[1]}"
    return text


def _validate_spec(spec: ModelSpec, contract: FamilyContract) -> None:
    if (
        spec.model_state_dim != contract.state_dim
        or spec.model_action_dim != contract.action_dim
        or spec.model_action_horizon != contract.action_horizon
        or spec.observation_horizon != contract.observation_horizon
    ):
        raise ValueError(
            f"{spec.model_id}: {spec.family} dimensions must be "
            f"state={contract.state_dim}, action={contract.action_dim}, "
            f"observation_horizon={contract.observation_horizon}, "
            f"action_horizon={contract.action_horizon}"
        )
    if (
        spec.state_semantics != contract.state_semantics
        or spec.action_semantics != contract.action_semantics
    ):
        raise ValueError(
            f"{spec.model_id}: state/action semantics must exactly match the "
            f"trusted local {spec.family} adapter"
        )
    roles = set(spec.camera_roles)
    if len(roles) != len(spec.camera_roles):
        raise ValueError(f"{spec.model_id}: camera roles must be unique")
    if not contract.required_camera_roles <= roles <= contract.allowed_camera_roles:
        raise ValueError(
            f"{spec.model_id}: camera roles {sorted(roles)} violate "
            f"{spec.family} contract"
        )
    if spec.execution_steps > spec.model_action_horizon:
        raise ValueError(f"{spec.model_id}: execution_steps exceeds action horizon")
    if not spec.artifact.required_files or not spec.artifact.allow_patterns:
        raise ValueError(f"{spec.model_id}: artifact file lists cannot be empty")
    weights = {
        name
        for name in spec.artifact.required_files
        if name.endswith((".safetensors", ".pt", ".pth", ".onnx"))
    }
    if not weights:
        raise ValueError(f"{spec.model_id}: no required model weights are declared")
    unhashed = weights - set(spec.artifact.file_sha256)
    if unhashed:
        raise ValueError(
            f"{spec.model_id}: every required weight must be SHA-256 pinned: "
            f"{sorted(unhashed)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a string list")
    return tuple(item.strip() for item in value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _git_sha(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != 40:
        raise ValueError(f"{label} must be a full 40-character commit SHA")
    return _hex(text, label)


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return _hex(text, label)


def _hex(text: str, label: str) -> str:
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return text.lower()


def _repo_id(value: Any, label: str) -> str:
    text = normalize_model_reference(_nonempty(value, label))
    if not HF_REPO_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be owner/repository")
    return text


def _model_id(value: Any) -> str:
    text = _nonempty(value, "model_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError("model_id must contain only letters, digits, dot, dash, underscore")
    return text


def _safe_relative_dir(value: Any, label: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must be a safe relative directory")
    return path.as_posix()


def _safe_relative_file(value: str, label: str) -> str:
    text = _nonempty(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{label} contains an unsafe relative file")
    return path.as_posix()


def _safe_pattern(value: str, label: str) -> str:
    text = _nonempty(value, label)
    if text.startswith("/") or ".." in PurePosixPath(text).parts:
        raise ValueError(f"{label} contains an unsafe pattern")
    return text
