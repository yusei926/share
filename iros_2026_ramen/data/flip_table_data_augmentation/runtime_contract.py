"""Verify the pinned RoboFinals V1 simulator runtime before generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import site
from typing import Any

from .config import PipelineConfig
from .io_utils import read_json_object, sha256_file


DEFAULT_ROBOFINALS_ROOT = Path(
    os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals")
)
DEFAULT_AUGMENTATION_ROOT = Path(
    os.environ.get(
        "FLIP_TABLE_AUGMENTATION_ROOT", "/repo/data/flip_table_data_augmentation"
    )
)
DEFAULT_RL_OVERLAY_ROOT = Path(
    os.environ.get(
        "FLIP_TABLE_AUG_RL_OVERLAY_ROOT",
        "/workspace/robofinals/robofinals_rl/flip_table",
    )
)
DEFAULT_ROOM_ASSET_ROOT = Path(
    os.environ.get("FLIP_TABLE_AUG_ROOM_ASSET_ROOT", "/workspace/flip_table_room_assets")
)
MIMIC_SOURCE_RELATIVE = Path(
    "third_party/IsaacLab-Arena/submodules/isaaclab-Newton/source/isaaclab_mimic"
)
RUNTIME_ARTIFACTS = {
    "task_overlay": Path("robofinals_tasks/local_auto_tasks/assemble_table_task.py"),
    "recorder_monkey_patch": Path("robofinals/utils/monkey_patch.py"),
    "g1_python": Path("robofinals/core/robots/unitree/g1.py"),
    "dex1_gripper_python": Path("robofinals/core/models/grippers/dex1.py"),
    "g1_assets_config": Path("robofinals/core/robots/unitree/assets_cfg.py"),
    "g1_gripper_usd": Path("robofinals/data/assets/g1_urdf_gripper/G1_GRIPPER.usd"),
    "g1_gripper_usd_base": Path(
        "robofinals/data/assets/g1_urdf_gripper/configuration/usd_base.usd"
    ),
}


def stable_tree_sha256(
    root: Path,
    *,
    excluded_directory_names: frozenset[str] = frozenset(),
) -> tuple[str, int]:
    """Hash source relative paths and bytes, excluding interpreter artifacts."""

    source = Path(root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and not excluded_directory_names.intersection(path.relative_to(source).parts)
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _replicator_extension_version() -> str:
    matches: list[str] = []
    for package_root in site.getsitepackages():
        cache = Path(package_root) / "isaacsim" / "extscache"
        matches.extend(path.name.removeprefix("omni.replicator.core-") for path in cache.glob("omni.replicator.core-*"))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one omni.replicator.core extension, found {unique}")
    return unique[0]


@dataclass(frozen=True)
class RuntimeAudit:
    configured_image: str
    configured_image_digest: str
    observed_image_digest: str | None
    image_digest_verified: bool
    isaac_sim_version: str
    isaac_lab_version: str
    isaac_lab_mimic_version: str
    isaac_lab_mimic_source_sha256: str
    isaac_lab_mimic_source_files: int
    replicator_version: str
    config_sha256: str
    augmentation_source_sha256: str
    augmentation_source_files: int
    rl_overlay_sha256: str
    rl_overlay_files: int
    room_assets_sha256: str
    room_asset_files: int
    runtime_artifact_sha256: dict[str, str]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def verify_runtime(
    config: PipelineConfig,
    *,
    robofinals_root: str | Path = DEFAULT_ROBOFINALS_ROOT,
    observed_image_digest: str | None = None,
    require_image_digest: bool = True,
    augmentation_root: str | Path = DEFAULT_AUGMENTATION_ROOT,
    rl_overlay_root: str | Path = DEFAULT_RL_OVERLAY_ROOT,
    room_asset_root: str | Path = DEFAULT_ROOM_ASSET_ROOT,
) -> RuntimeAudit:
    versions = {
        "isaac_sim_version": importlib.metadata.version("isaacsim"),
        "isaac_lab_version": importlib.metadata.version("isaaclab"),
        "isaac_lab_mimic_version": importlib.metadata.version("isaaclab_mimic"),
    }
    for key, actual in versions.items():
        expected = getattr(config.runtime, key)
        if actual != expected:
            raise RuntimeError(f"runtime {key} mismatch: expected {expected!r}, got {actual!r}")

    mimic_root = Path(robofinals_root).resolve() / MIMIC_SOURCE_RELATIVE
    source_hash, source_files = stable_tree_sha256(mimic_root)
    if source_hash != config.runtime.isaac_lab_mimic_source_sha256:
        raise RuntimeError(
            "Isaac Lab Mimic source mismatch: "
            f"expected {config.runtime.isaac_lab_mimic_source_sha256}, got {source_hash}"
        )
    replicator_version = _replicator_extension_version()
    if replicator_version != config.runtime.replicator_version:
        raise RuntimeError(
            f"Replicator mismatch: expected {config.runtime.replicator_version!r}, "
            f"got {replicator_version!r}"
        )

    verified = observed_image_digest == config.runtime.container_digest
    if observed_image_digest is not None and not verified:
        raise RuntimeError(
            f"container digest mismatch: expected {config.runtime.container_digest}, "
            f"got {observed_image_digest}"
        )
    if require_image_digest and not verified:
        raise RuntimeError(
            "the container digest was not independently observed; pass the digest from "
            "`docker image inspect` or the Vast instance manifest"
        )
    augmentation_hash, augmentation_files = stable_tree_sha256(
        Path(augmentation_root),
        excluded_directory_names=frozenset({"outputs"}),
    )
    rl_hash, rl_files = stable_tree_sha256(Path(rl_overlay_root))
    room_hash, room_files = stable_tree_sha256(Path(room_asset_root))
    root = Path(robofinals_root).resolve()
    artifact_hashes = {}
    for name, relative_path in RUNTIME_ARTIFACTS.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_hashes[name] = sha256_file(path)
    dex1_hash = artifact_hashes["dex1_gripper_python"]
    if dex1_hash != config.runtime.dex1_gripper_python_sha256:
        raise RuntimeError(
            "organizer V1 Dex1 implementation mismatch: "
            f"expected {config.runtime.dex1_gripper_python_sha256}, got {dex1_hash}"
        )
    return RuntimeAudit(
        configured_image=config.runtime.container_image,
        configured_image_digest=config.runtime.container_digest,
        observed_image_digest=observed_image_digest,
        image_digest_verified=verified,
        isaac_sim_version=versions["isaac_sim_version"],
        isaac_lab_version=versions["isaac_lab_version"],
        isaac_lab_mimic_version=versions["isaac_lab_mimic_version"],
        isaac_lab_mimic_source_sha256=source_hash,
        isaac_lab_mimic_source_files=source_files,
        replicator_version=replicator_version,
        config_sha256=config.digest,
        augmentation_source_sha256=augmentation_hash,
        augmentation_source_files=augmentation_files,
        rl_overlay_sha256=rl_hash,
        rl_overlay_files=rl_files,
        room_assets_sha256=room_hash,
        room_asset_files=room_files,
        runtime_artifact_sha256=artifact_hashes,
    )


def verify_runtime_manifest(
    path: str | Path,
    config: PipelineConfig,
    *,
    robofinals_root: str | Path = DEFAULT_ROBOFINALS_ROOT,
) -> tuple[dict[str, Any], str]:
    """Verify a saved audit against the currently mounted runtime and return its hash."""

    manifest = Path(path).expanduser().resolve()
    payload = read_json_object(manifest, label="runtime manifest")
    observed_digest = payload.get("observed_image_digest")
    if not isinstance(observed_digest, str):
        raise ValueError("runtime manifest lacks an observed container digest")
    current = verify_runtime(
        config,
        robofinals_root=robofinals_root,
        observed_image_digest=observed_digest,
        require_image_digest=True,
    ).to_json()
    if payload != current:
        differing = sorted(
            key for key in set(payload).union(current) if payload.get(key) != current.get(key)
        )
        raise RuntimeError(f"runtime changed since the audit: {differing}")
    return payload, sha256_file(manifest)
