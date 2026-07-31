from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
    snapshot_download,
)
from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

from .build import MANIFEST_NAME, OWNED_MARKER, dataset_root
from .config import CurationConfig
from .util import atomic_write_json, sha256_file
from .validate import validate_dataset_root, validate_local


def _files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != OWNED_MARKER
    )


def publish(config: CurationConfig) -> dict:
    local_validation = validate_local(config)
    root = dataset_root(config)
    manifest_sha = sha256_file(root / MANIFEST_NAME)
    files = _files(root)
    expected = set(files)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id=config.target_repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    repo = api.repo_info(config.target_repo_id, repo_type="dataset")
    if repo.private is not True:
        raise RuntimeError("target dataset must remain private")
    try:
        main = api.repo_info(
            config.target_repo_id, repo_type="dataset", revision="main"
        )
        main_sha = main.sha
        main_files = set(
            api.list_repo_files(
                config.target_repo_id, repo_type="dataset", revision=main_sha
            )
        )
    except (RevisionNotFoundError, EntryNotFoundError):
        seed = config.workspace / "publish" / ".staging-init"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_text("private curated dataset staging\n", encoding="utf-8")
        commit = api.create_commit(
            repo_id=config.target_repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=".staging-init", path_or_fileobj=seed
                )
            ],
            commit_message="Initialize private curated dataset",
        )
        main_sha = commit.oid
        main_files = {".staging-init"}
    initial_files = {".staging-init", ".gitattributes"}
    if main_files - initial_files:
        try:
            remote_manifest = hf_hub_download(
                config.target_repo_id,
                MANIFEST_NAME,
                repo_type="dataset",
                revision=main_sha,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as error:
            raise RuntimeError("target main already contains an unrecognized dataset") from error
        if sha256_file(Path(remote_manifest)) != manifest_sha:
            raise RuntimeError(
                "target main contains a different manifest; refusing to overwrite"
            )
        return {
            "repo_id": config.target_repo_id,
            "main_commit_sha": main_sha,
            "reused_existing_main": True,
            "local_validation": local_validation,
        }
    stage = f"staging-{manifest_sha[:16]}"
    api.create_branch(
        repo_id=config.target_repo_id,
        repo_type="dataset",
        branch=stage,
        revision=main_sha,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=config.target_repo_id,
        repo_type="dataset",
        revision=stage,
        folder_path=root,
        ignore_patterns=[OWNED_MARKER],
        private=True,
        print_report=True,
    )
    stage_info = api.repo_info(
        config.target_repo_id, repo_type="dataset", revision=stage
    )
    stage_files = set(
        api.list_repo_files(
            config.target_repo_id, repo_type="dataset", revision=stage_info.sha
        )
    )
    extras = stage_files - expected
    if extras:
        api.create_commit(
            repo_id=config.target_repo_id,
            repo_type="dataset",
            revision=stage,
            operations=[
                CommitOperationDelete(path_in_repo=path) for path in sorted(extras)
            ],
            commit_message="Remove staging files outside curated manifest",
        )
        stage_info = api.repo_info(
            config.target_repo_id, repo_type="dataset", revision=stage
        )
        stage_files = set(
            api.list_repo_files(
                config.target_repo_id, repo_type="dataset", revision=stage_info.sha
            )
        )
    if stage_files != expected:
        raise RuntimeError("staging inventory differs from local dataset")
    staging_root = Path(
        snapshot_download(
            config.target_repo_id,
            repo_type="dataset",
            revision=stage_info.sha,
            cache_dir=config.workspace / "publish" / "verify_cache",
        )
    )
    staging_validation = validate_dataset_root(
        staging_root, config, require_source_comparison=True
    )
    current_main = api.repo_info(
        config.target_repo_id, repo_type="dataset", revision="main"
    ).sha
    if current_main != main_sha:
        raise RuntimeError("remote main changed during staging validation")
    operations = [
        CommitOperationDelete(path_in_repo=path)
        for path in sorted(main_files - expected)
    ]
    operations.extend(
        CommitOperationCopy(
            src_path_in_repo=path,
            path_in_repo=path,
            src_revision=stage_info.sha,
        )
        for path in files
    )
    final = api.create_commit(
        repo_id=config.target_repo_id,
        repo_type="dataset",
        revision="main",
        parent_commit=main_sha,
        operations=operations,
        commit_message="Publish verified flip_table_2 curated LeRobot v3 dataset",
    )
    final_files = set(
        api.list_repo_files(
            config.target_repo_id, repo_type="dataset", revision=final.oid
        )
    )
    if final_files != expected:
        raise RuntimeError("published main inventory differs from staging")
    report = {
        "schema_version": "team_ramen_flip_table_curation_publish/v1",
        "repo_id": config.target_repo_id,
        "private": True,
        "manifest_sha256": manifest_sha,
        "staging_branch": stage,
        "staging_commit_sha": stage_info.sha,
        "main_parent_sha": main_sha,
        "main_commit_sha": final.oid,
        "reused_existing_main": False,
        "local_validation": local_validation,
        "staging_validation": staging_validation,
    }
    output = config.workspace / "publish" / "publish_report.json"
    atomic_write_json(output, report)
    print(f"[publish] {config.target_repo_id}@{final.oid}")
    return report
