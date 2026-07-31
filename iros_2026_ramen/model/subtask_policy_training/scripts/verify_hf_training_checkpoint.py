"""Verify that a resumable LeRobot checkpoint is durably stored on the Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


REQUIRED_FILES = (
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_name(step: int) -> str:
    if step <= 0:
        raise ValueError("--step must be positive")
    return f"{step:06d}"


def remote_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        sha = getattr(lfs, "sha256", None)
        if sha:
            return str(sha)
    blob_id = getattr(sibling, "blob_id", None)
    return str(blob_id) if blob_id else None


def main() -> None:
    args = parse_args()
    name = checkpoint_name(args.step)
    local = args.local_checkpoint.resolve()
    if local.name != name or not local.is_dir():
        raise ValueError(
            f"local checkpoint must be the exact step directory {name}: {local}"
        )

    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)
    if not info.private:
        raise RuntimeError(f"checkpoint repository must be private: {args.repo_id}")
    siblings = {item.rfilename: item for item in info.siblings}
    prefix = f"checkpoints/{name}/"
    missing = [relative for relative in REQUIRED_FILES if prefix + relative not in siblings]
    if missing:
        raise FileNotFoundError(
            f"Hub checkpoint {args.repo_id}@{name} is incomplete: {missing}"
        )

    refs = api.list_repo_refs(args.repo_id)
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    if name not in tags:
        raise ValueError(f"Hub checkpoint has no immutable {name} tag")

    hashes: dict[str, dict[str, str | None]] = {}
    for relative in REQUIRED_FILES:
        local_path = local / relative
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        local_hash = sha256(local_path)
        remote_hash = remote_sha256(siblings[prefix + relative])
        if remote_hash is not None and len(remote_hash) == 64 and remote_hash != local_hash:
            raise ValueError(f"Hub checkpoint hash differs for {relative}")
        hashes[relative] = {
            "local_sha256": local_hash,
            "remote_sha256": remote_hash,
        }

    receipt = {
        "schema_version": "groot_n17_resumable_checkpoint_backup_v1",
        "repo_id": args.repo_id,
        "private": True,
        "checkpoint_step": args.step,
        "checkpoint_tag": name,
        "repository_head": info.sha,
        "tag_commit": tags[name],
        "required_files": list(REQUIRED_FILES),
        "hashes": hashes,
        "resumable": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
