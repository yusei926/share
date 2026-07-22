"""Transactional private-Hub staging, verification, and atomic promotion."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from .file_manifest import DEFAULT_MANIFEST_NAME, verify_file_manifest
from .validate_dataset import validate_dataset, validate_release_thresholds
from ..config import PipelineConfig
from ..io_utils import atomic_write_json, sha256_file


UPLOAD_REPORT_SCHEMA_VERSION = "team_ramen_flip_table_hf_upload/v1"
UPLOAD_WORKSPACE_SCHEMA_VERSION = "team_ramen_flip_table_hf_upload_workspace/v1"


def _local_files(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()))


def _prepare_upload_workspace(
    verification_root: Path,
    *,
    dataset_root: Path,
    repo_id: str,
    config_sha256: str,
    manifest_sha256: str,
    local_files: tuple[str, ...],
) -> Path:
    """Create a resumable hard-linked upload tree without mutating the dataset."""

    identity = {
        "schema_version": UPLOAD_WORKSPACE_SCHEMA_VERSION,
        "repo_id": repo_id,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
    }
    marker = verification_root / "upload-workspace.json"
    if verification_root.exists():
        if not marker.is_file():
            raise FileExistsError(
                "verification_root exists without a resumable upload-workspace marker"
            )
        observed = json.loads(marker.read_text(encoding="utf-8"))
        if observed != identity:
            raise ValueError("verification_root belongs to a different dataset upload")
    else:
        verification_root.mkdir(parents=True)
        atomic_write_json(marker, identity)

    upload_root = verification_root / "upload-source"
    upload_root.mkdir(exist_ok=True)
    expected = set(local_files)
    actual = {
        path.relative_to(upload_root).as_posix()
        for path in upload_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(upload_root).parts
    }
    extras = sorted(actual - expected)
    if extras:
        raise ValueError(f"upload worktree contains files outside the dataset manifest: {extras[:10]}")
    for relative in local_files:
        source = dataset_root / relative
        destination = upload_root / relative
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                raise ValueError(f"resumable upload file differs from dataset: {relative}")
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"resumable upload file hash differs from dataset: {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return upload_root


def _remote_manifest_matches(api, repo_id: str, revision: str, manifest_sha256: str) -> bool:
    try:
        path = Path(
            api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=DEFAULT_MANIFEST_NAME,
                force_download=True,
            )
        )
    except Exception as exc:
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError

        if isinstance(exc, (EntryNotFoundError, RevisionNotFoundError)):
            return False
        raise
    return sha256_file(path) == manifest_sha256


def _delete_remote_extras(api, repo_id: str, revision: str, expected: set[str]) -> None:
    from huggingface_hub import CommitOperationDelete

    current = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))
    extras = sorted(current - expected)
    if extras:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            operations=[CommitOperationDelete(path_in_repo=path) for path in extras],
            commit_message="Remove files outside the staged augmentation manifest",
        )


def _download_and_validate(
    api,
    *,
    repo_id: str,
    revision: str,
    cache_root: Path,
    config: PipelineConfig,
    minimum_synthetic_trajectories: int,
    minimum_appearance_variants: int,
) -> tuple[Path, dict[str, Any]]:
    snapshot = Path(
        api.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_root,
            force_download=True,
        )
    ).resolve()
    verify_file_manifest(snapshot)
    report = validate_dataset(
        snapshot,
        config,
        full_video_decode=True,
        require_full_source=True,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )
    return snapshot, report


def publish_verified_dataset(
    *,
    dataset_root: str | Path,
    verification_root: str | Path,
    report_path: str | Path,
    config: PipelineConfig,
    minimum_synthetic_trajectories: int,
    minimum_appearance_variants: int,
    expected_main_revision: str | None = None,
) -> dict[str, Any]:
    """Upload to staging, re-download, atomically promote, and verify main."""

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be present in the environment")
    validate_release_thresholds(
        config,
        require_full_source=True,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for dataset publication") from exc

    root = Path(dataset_root).expanduser().resolve()
    verify_root = Path(verification_root).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    if report_file.is_relative_to(root):
        raise ValueError("upload report must be outside the immutable dataset root")
    verify_file_manifest(root)
    local_validation = validate_dataset(
        root,
        config,
        full_video_decode=True,
        require_full_source=True,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )
    manifest_sha = sha256_file(root / DEFAULT_MANIFEST_NAME)
    stage_branch = f"staging-{manifest_sha[:16]}"
    local_files = _local_files(root)
    local_file_set = set(local_files)
    upload_root = _prepare_upload_workspace(
        verify_root,
        dataset_root=root,
        repo_id=config.target.repo_id,
        config_sha256=config.digest,
        manifest_sha256=manifest_sha,
        local_files=local_files,
    )

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    repository = api.repo_info(repo_id=config.target.repo_id, repo_type="dataset")
    if repository.private is not True:
        raise RuntimeError("target Hugging Face dataset repository must be private")
    try:
        main_info = api.repo_info(repo_id=config.target.repo_id, repo_type="dataset", revision="main")
        main_revision_before = main_info.sha
        main_files_before = set(
            api.list_repo_files(repo_id=config.target.repo_id, repo_type="dataset", revision="main")
        )
    except (RevisionNotFoundError, EntryNotFoundError):
        seed = root.parent / f".{root.name}.staging-init"
        seed.write_text("private augmentation staging repository\n", encoding="utf-8")
        try:
            commit = api.create_commit(
                repo_id=config.target.repo_id,
                repo_type="dataset",
                operations=[CommitOperationAdd(path_in_repo=".staging-init", path_or_fileobj=seed)],
                commit_message="Initialize private augmentation staging repository",
            )
        finally:
            seed.unlink(missing_ok=True)
        main_revision_before = commit.oid
        main_files_before = {".staging-init"}
    main_already_matches = (
        main_files_before == local_file_set
        and _remote_manifest_matches(
            api,
            config.target.repo_id,
            main_revision_before,
            manifest_sha,
        )
    )
    if (
        expected_main_revision is not None
        and main_revision_before != expected_main_revision
        and not main_already_matches
    ):
        raise RuntimeError(
            f"main changed: expected {expected_main_revision}, found {main_revision_before}"
        )
    dataset_files_before = main_files_before - {".staging-init"}
    if dataset_files_before and expected_main_revision is None and not main_already_matches:
        raise RuntimeError(
            "target main already contains files; pass its reviewed SHA as expected_main_revision"
        )

    api.create_branch(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        branch=stage_branch,
        revision=main_revision_before,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        revision=stage_branch,
        folder_path=upload_root,
        private=True,
        ignore_patterns=[".cache/**"],
        print_report=True,
    )
    _delete_remote_extras(api, config.target.repo_id, stage_branch, local_file_set)
    stage_info = api.repo_info(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        revision=stage_branch,
    )
    stage_files = set(
        api.list_repo_files(
            repo_id=config.target.repo_id,
            repo_type="dataset",
            revision=stage_info.sha,
        )
    )
    if stage_files != local_file_set:
        raise RuntimeError("staging file list differs from the local upload manifest")

    stage_snapshot, stage_validation = _download_and_validate(
        api,
        repo_id=config.target.repo_id,
        revision=stage_info.sha,
        cache_root=verify_root / "staging-cache",
        config=config,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )

    main_now = api.repo_info(repo_id=config.target.repo_id, repo_type="dataset", revision="main").sha
    if main_now != main_revision_before:
        raise RuntimeError(f"main changed during staging verification: {main_revision_before} -> {main_now}")
    reused_existing_main = main_already_matches
    if reused_existing_main:
        final_commit_sha = main_revision_before
    else:
        operations = [
            CommitOperationDelete(path_in_repo=path)
            for path in sorted(main_files_before - local_file_set)
        ]
        operations.extend(
            CommitOperationCopy(
                src_path_in_repo=path,
                path_in_repo=path,
                src_revision=stage_info.sha,
            )
            for path in local_files
        )
        final_commit = api.create_commit(
            repo_id=config.target.repo_id,
            repo_type="dataset",
            revision="main",
            parent_commit=main_revision_before,
            operations=operations,
            commit_message="Publish verified flip-table augmented LeRobot v3 dataset",
        )
        final_commit_sha = final_commit.oid
    published_main_before_download = api.repo_info(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        revision="main",
    ).sha
    if published_main_before_download != final_commit_sha:
        raise RuntimeError(
            "main does not point to the verified promotion commit: "
            f"{published_main_before_download} != {final_commit_sha}"
        )
    final_files = set(
        api.list_repo_files(
            repo_id=config.target.repo_id,
            repo_type="dataset",
            revision="main",
        )
    )
    if final_files != local_file_set:
        raise RuntimeError("final main file list differs from the staged upload manifest")
    final_snapshot, final_validation = _download_and_validate(
        api,
        repo_id=config.target.repo_id,
        revision="main",
        cache_root=verify_root / "main-cache",
        config=config,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )
    published_main_after_download = api.repo_info(
        repo_id=config.target.repo_id,
        repo_type="dataset",
        revision="main",
    ).sha
    if published_main_after_download != final_commit_sha:
        raise RuntimeError(
            "main changed during final verification: "
            f"{final_commit_sha} -> {published_main_after_download}"
        )

    report = {
        "schema_version": UPLOAD_REPORT_SCHEMA_VERSION,
        "repo_id": config.target.repo_id,
        "url": f"https://huggingface.co/datasets/{config.target.repo_id}",
        "private": True,
        "manifest_sha256": manifest_sha,
        "staging_branch": stage_branch,
        "staging_commit_sha": stage_info.sha,
        "main_commit_sha": final_commit_sha,
        "main_parent_sha": main_revision_before,
        "reused_existing_main": reused_existing_main,
        "local_validation": local_validation,
        "staging_validation": stage_validation,
        "main_validation": final_validation,
        "staging_snapshot": str(stage_snapshot),
        "main_snapshot": str(final_snapshot),
    }
    atomic_write_json(report_file, report)
    return report
