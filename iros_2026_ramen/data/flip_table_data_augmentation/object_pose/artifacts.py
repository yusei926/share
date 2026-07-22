"""Pinned stereo, segmentation, and pose-estimation runtime artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import BinaryIO
from urllib.request import Request, urlopen

from ..config import ObjectPoseRuntimeConfig, PipelineConfig
from ..io_utils import atomic_write_json, sha256_file


MANIFEST_SCHEMA_VERSION = "team_ramen_object_pose_runtime/v2"
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relative_path: str
    source: str
    size_bytes: int
    sha256: str


def artifact_specs(config: ObjectPoseRuntimeConfig) -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec(
            name="foundationpose_refiner_config",
            relative_path="FoundationPose/weights/2023-10-28-18-33-37/config.yml",
            source=(
                "https://drive.usercontent.google.com/download?id="
                f"{config.foundationpose_refiner_config_file_id}&export=download&confirm=t"
            ),
            size_bytes=config.foundationpose_refiner_config_size_bytes,
            sha256=config.foundationpose_refiner_config_sha256,
        ),
        ArtifactSpec(
            name="foundationpose_refiner_checkpoint",
            relative_path="FoundationPose/weights/2023-10-28-18-33-37/model_best.pth",
            source=(
                "https://drive.usercontent.google.com/download?id="
                f"{config.foundationpose_refiner_checkpoint_file_id}&export=download&confirm=t"
            ),
            size_bytes=config.foundationpose_refiner_checkpoint_size_bytes,
            sha256=config.foundationpose_refiner_checkpoint_sha256,
        ),
        ArtifactSpec(
            name="foundationpose_scorer_config",
            relative_path="FoundationPose/weights/2024-01-11-20-02-45/config.yml",
            source=(
                "https://drive.usercontent.google.com/download?id="
                f"{config.foundationpose_scorer_config_file_id}&export=download&confirm=t"
            ),
            size_bytes=config.foundationpose_scorer_config_size_bytes,
            sha256=config.foundationpose_scorer_config_sha256,
        ),
        ArtifactSpec(
            name="foundationpose_scorer_checkpoint",
            relative_path="FoundationPose/weights/2024-01-11-20-02-45/model_best.pth",
            source=(
                "https://drive.usercontent.google.com/download?id="
                f"{config.foundationpose_scorer_checkpoint_file_id}&export=download&confirm=t"
            ),
            size_bytes=config.foundationpose_scorer_checkpoint_size_bytes,
            sha256=config.foundationpose_scorer_checkpoint_sha256,
        ),
        ArtifactSpec(
            name="grounding_dino_checkpoint",
            relative_path=f"hf/grounding-dino-base/{config.detector_checkpoint_filename}",
            source=(
                f"hf://{config.detector_repo}@{config.detector_revision}/"
                f"{config.detector_checkpoint_filename}"
            ),
            size_bytes=config.detector_checkpoint_size_bytes,
            sha256=config.detector_checkpoint_sha256,
        ),
        ArtifactSpec(
            name="sam2.1_checkpoint",
            relative_path=f"hf/sam2.1-hiera-large/{config.segmentation_checkpoint_filename}",
            source=(
                f"hf://{config.segmentation_repo}@{config.segmentation_revision}/"
                f"{config.segmentation_checkpoint_filename}"
            ),
            size_bytes=config.segmentation_checkpoint_size_bytes,
            sha256=config.segmentation_checkpoint_sha256,
        ),
        ArtifactSpec(
            name="fast_foundationstereo_checkpoint",
            relative_path=(
                f"hf/fast-foundation-stereo/{config.fast_stereo_model_filename}"
            ),
            source=(
                f"hf://{config.fast_stereo_model_repo}@"
                f"{config.fast_stereo_model_revision}/"
                f"{config.fast_stereo_model_filename}"
            ),
            size_bytes=config.fast_stereo_model_size_bytes,
            sha256=config.fast_stereo_model_sha256,
        ),
        ArtifactSpec(
            name="fast_foundationstereo_config",
            relative_path=(
                f"hf/fast-foundation-stereo/{config.fast_stereo_config_filename}"
            ),
            source=(
                f"hf://{config.fast_stereo_model_repo}@"
                f"{config.fast_stereo_model_revision}/"
                f"{config.fast_stereo_config_filename}"
            ),
            size_bytes=config.fast_stereo_config_size_bytes,
            sha256=config.fast_stereo_config_sha256,
        ),
    )


def verify_artifact(path: Path, spec: ArtifactSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec.name}: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise ValueError(f"{spec.name} size differs: expected {spec.size_bytes}, got {size}")
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise ValueError(f"{spec.name} SHA-256 differs: expected {spec.sha256}, got {digest}")


def _copy_stream(source: BinaryIO, target: BinaryIO) -> None:
    while block := source.read(_DOWNLOAD_CHUNK_BYTES):
        target.write(block)


def download_verified(spec: ArtifactSpec, target: Path) -> None:
    """Download one non-HF artifact atomically, or verify the existing file."""

    if target.exists():
        verify_artifact(target, spec)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    if temporary.exists():
        temporary.unlink()
    request = Request(spec.source, headers={"User-Agent": "Team-RAMEN-Issue-70/1"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
            _copy_stream(response, stream)
            stream.flush()
            os.fsync(stream.fileno())
        verify_artifact(temporary, spec)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def ensure_git_checkout(destination: Path, repo: str, revision: str) -> dict[str, str]:
    """Create or verify a clean detached checkout at an immutable commit."""

    expected_url = f"https://github.com/{repo}.git"
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise ValueError(f"runtime checkout is not a Git repository: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            _run_git(["clone", "--filter=blob:none", "--no-checkout", expected_url, str(temporary)])
            _run_git(["checkout", "--detach", revision], cwd=temporary)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    observed_url = _run_git(["remote", "get-url", "origin"], cwd=destination)
    observed_revision = _run_git(["rev-parse", "HEAD"], cwd=destination)
    if observed_url not in {expected_url, expected_url.removesuffix(".git")}:
        raise ValueError(f"unexpected origin for {repo}: {observed_url}")
    if observed_revision != revision:
        raise ValueError(f"{repo} is at {observed_revision}, expected {revision}")
    dirty = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd=destination)
    if dirty:
        raise ValueError(f"tracked files in {repo} checkout are modified")
    return {
        "repo": repo,
        "origin": observed_url,
        "revision": observed_revision,
        "tree": _run_git(["rev-parse", "HEAD^{tree}"], cwd=destination),
    }


def _link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.is_symlink():
        target.unlink()
    if target.exists():
        if target.is_file() and sha256_file(target) == sha256_file(source):
            return
        raise ValueError(f"existing HF runtime file differs from the pinned snapshot: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _materialize_hf_snapshot(
    *,
    repo: str,
    revision: str,
    required_files: tuple[str, ...],
    target: Path,
    checkpoint: ArtifactSpec,
) -> dict[str, object]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for object-pose checkpoints") from exc
    cached_root = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            revision=revision,
            allow_patterns=list(required_files),
        )
    ).resolve()
    records = []
    for relative in required_files:
        source = cached_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"{repo}@{revision} is missing {relative}")
        destination = target / relative
        _link_or_copy(source, destination)
        records.append(
            {
                "path": relative,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    actual = {
        str(path.relative_to(target))
        for path in target.rglob("*")
        if path.is_file()
    }
    if actual != set(required_files):
        raise ValueError(f"HF runtime snapshot {target} has unexpected files: {sorted(actual)}")
    verify_artifact(target / Path(checkpoint.relative_path).name, checkpoint)
    return {"repo": repo, "revision": revision, "files": records}


def prepare_runtime(root: Path, config: PipelineConfig) -> dict[str, object]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pose = config.object_pose_runtime
    repositories = [
        ensure_git_checkout(root / "FoundationPose", pose.foundationpose_repo, pose.foundationpose_revision),
        ensure_git_checkout(root / "pytorch3d", pose.pytorch3d_repo, pose.pytorch3d_revision),
        ensure_git_checkout(root / "nvdiffrast", pose.nvdiffrast_repo, pose.nvdiffrast_revision),
        ensure_git_checkout(
            root / "Fast-FoundationStereo",
            pose.fast_stereo_repo,
            pose.fast_stereo_revision,
        ),
    ]
    records: list[dict[str, object]] = []
    specs = artifact_specs(pose)
    spec_by_name = {spec.name: spec for spec in specs}
    for spec in specs[:4]:
        target = root / spec.relative_path
        download_verified(spec, target)
        verify_artifact(target, spec)
        records.append({**asdict(spec), "path": str(target.relative_to(root))})
    detector_files = (
        "config.json",
        pose.detector_checkpoint_filename,
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    )
    segmentation_files = (
        "config.json",
        pose.segmentation_checkpoint_filename,
        "preprocessor_config.json",
        "processor_config.json",
        "sam2.1_hiera_l.yaml",
        "video_preprocessor_config.json",
    )
    hf_snapshots = [
        _materialize_hf_snapshot(
            repo=pose.detector_repo,
            revision=pose.detector_revision,
            required_files=detector_files,
            target=root / "hf" / "grounding-dino-base",
            checkpoint=specs[4],
        ),
        _materialize_hf_snapshot(
            repo=pose.segmentation_repo,
            revision=pose.segmentation_revision,
            required_files=segmentation_files,
            target=root / "hf" / "sam2.1-hiera-large",
            checkpoint=specs[5],
        ),
        _materialize_hf_snapshot(
            repo=pose.fast_stereo_model_repo,
            revision=pose.fast_stereo_model_revision,
            required_files=(
                pose.fast_stereo_config_filename,
                pose.fast_stereo_model_filename,
            ),
            target=root / "hf" / "fast-foundation-stereo",
            checkpoint=spec_by_name["fast_foundationstereo_checkpoint"],
        ),
    ]
    for spec in specs[4:]:
        target = root / spec.relative_path
        verify_artifact(target, spec)
        records.append({**asdict(spec), "path": str(target.relative_to(root))})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_sha256": config.digest,
        "repositories": repositories,
        "hf_snapshots": hf_snapshots,
        "artifacts": records,
    }


def write_manifest(path: Path, value: dict[str, object]) -> None:
    atomic_write_json(path, value)
