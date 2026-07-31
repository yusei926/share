"""Validate and synchronize a trained policy with Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.gr00t.n17_contract import (  # noqa: E402
    validate_finalized_furniture_checkpoint,
)


REQUIRED_POLICY_FILES = (
    "config.json",
    "evaluation_report.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
    "train_config.json",
    "training_manifest.json",
    "training_run_record.json",
)
PRESERVED_REMOTE_FILES = {".gitattributes", "README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a trained LeRobot policy checkpoint to Hugging Face.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload LeRobot policy checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    missing = [name for name in REQUIRED_POLICY_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_dir} is missing required policy files: {missing}")
    for metadata_name in ("config.json", "train_config.json"):
        metadata_path = model_dir / metadata_name
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint metadata: {metadata_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint metadata must be a JSON object: {metadata_path}")
        if metadata_name == "config.json" and payload.get("type") == "furniture_groot":
            required_groot_files = (
                "candidate_selection.json",
                "dex1_g1_synergy.json",
                "eef_fk_audit.json",
                "groot_contract.json",
                "orientation_contact_sheet.approved",
                "orientation_contact_sheet.jpg",
                "progress_manifest.json",
                "progress.jsonl",
                "sim_candidate_selection.json",
                "sim_evaluation_manifest.json",
                "sim_release_evaluation.json",
                "source_snapshot_manifest.json",
                "visual_rotation_manifest.json",
                "visual_rotation.jsonl",
            )
            missing_groot = [
                name for name in required_groot_files if not (model_dir / name).is_file()
            ]
            if missing_groot:
                raise FileNotFoundError(
                    f"{model_dir} is missing Furniture-GR00T provenance files: "
                    f"{missing_groot}"
                )
            if not (model_dir / "sim_evaluation").is_dir():
                raise FileNotFoundError(
                    f"{model_dir} is missing the simulator evaluation bundle"
                )
            validate_finalized_furniture_checkpoint(model_dir)
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_path = model_dir / processor_name
        try:
            processor = json.loads(processor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid serialized processor: {processor_path}") from exc
        steps = processor.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"serialized processor has no step list: {processor_path}")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"serialized processor contains a non-object step: {processor_path}")
            registry_name = step.get("registry_name")
            state_file = step.get("state_file")
            requires_state = registry_name in {
                "normalizer_processor",
                "unnormalizer_processor",
            }
            if state_file is None:
                if requires_state:
                    raise ValueError(
                        f"{processor_path.name} {registry_name!r} step has no state_file"
                    )
                continue
            if not isinstance(state_file, str) or not state_file.strip():
                raise ValueError(
                    f"{processor_path.name} has an invalid state_file: {state_file!r}"
                )
            state_path = (model_dir / state_file).resolve()
            try:
                state_path.relative_to(model_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"{processor_path.name} state_file escapes model directory: {state_file!r}"
                ) from exc
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"{processor_path.name} references missing state file {state_file!r}"
                )


def resolve_model_dir(*, model_dir: Path | None, output_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir.resolve()
    if output_dir is None:
        raise ValueError("Either --model-dir or --output-dir is required")

    checkpoints_dir = output_dir / "checkpoints"
    candidates = [checkpoints_dir / "last" / "pretrained_model"]
    if checkpoints_dir.is_dir():
        def checkpoint_key(path: Path) -> tuple[int, int | str]:
            return (1, int(path.name)) if path.name.isdigit() else (0, path.name)

        for checkpoint in sorted(checkpoints_dir.iterdir(), key=checkpoint_key, reverse=True):
            if checkpoint.name == "last":
                continue
            if checkpoint.is_dir():
                candidates.append(checkpoint / "pretrained_model")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"no pretrained_model checkpoint found under {checkpoints_dir}")


def main() -> int:
    args = parse_args()
    model_dir = resolve_model_dir(model_dir=args.model_dir, output_dir=args.output_dir)
    validate_model_dir(model_dir)

    print(f"repo_id: {args.repo_id}")
    print(f"model_dir: {model_dir}")
    print(f"private: {args.private}")
    print(f"revision: {args.revision}")
    print(f"dry_run: {args.dry_run}")
    if args.dry_run:
        return 0

    api = HfApi()
    repo_url = api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    repo_info = api.repo_info(args.repo_id, repo_type="model")
    if args.private and not repo_info.private:
        raise RuntimeError(
            f"refusing to upload a policy to public repository {args.repo_id}; make it private first"
        )
    local_files = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    remote_files = set(
        api.list_repo_files(args.repo_id, repo_type="model", revision=args.revision)
    )
    stale_files = sorted(remote_files - local_files - PRESERVED_REMOTE_FILES)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        path_in_repo=".",
        revision=args.revision,
        commit_message=args.commit_message,
        delete_patterns=stale_files or None,
    )
    print(f"repo_url: {repo_url}")
    print(f"commit: {commit.oid}")
    print(f"commit_url: {commit.commit_url}")
    print(f"deleted_stale_files: {len(stale_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
