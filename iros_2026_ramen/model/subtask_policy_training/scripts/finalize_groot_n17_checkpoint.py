"""Attach the complete GR00T N1.7 provenance and held-out evaluation record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from model.subtask_policy_training.gr00t.n17_contract import (
    SIM_FIXED_DR_PROFILE,
    SIM_UNSEEN_DR_PROFILE,
    valid_sim_candidate_evidence,
    validate_eef_fk_release_audit,
    validate_temporal_selection_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED_BASE_MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"
PINNED_BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
PINNED_DATASET_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
PINNED_DATASET_REVISION = "0dc47877dfb2efbea796a059c81290c649bc773c"
MILESTONE_NAMES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
EXPECTED_TUNING_SCOPE = {
    "tune_llm": False,
    "tune_visual": False,
    "tune_projector": True,
    "tune_diffusion_model": True,
    "tune_vlln": True,
    "tune_top_llm_layers": 0,
}
SYNERGY_ASSET = (
    REPO_ROOT
    / "model"
    / "subtask_policy_training"
    / "gr00t"
    / "assets"
    / "dex1_g1_synergy.json"
)
SOURCE_SNAPSHOT_ROOTS = (
    Path("model/subtask_policy_training"),
    Path("inference/desktop/upper_policy"),
    Path("evaluate/flip_table_simulation"),
)
SOURCE_SNAPSHOT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
SOURCE_SNAPSHOT_EXCLUDED_PARTS = {
    ".venv",
    ".venv_lerobot060",
    "__pycache__",
    "logs",
    "outputs",
}
SOURCE_SNAPSHOT_BUNDLE = "source_snapshot"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_task_contract(tasks_path: Path) -> list[dict[str, Any]]:
    """Read the LeRobot tasks table without relying on a pandas index layout."""
    import pyarrow.parquet as pq

    table = pq.read_table(tasks_path)
    names = set(table.schema.names)
    text_column = (
        "task"
        if "task" in names
        else "__index_level_0__"
        if "__index_level_0__" in names
        else None
    )
    if "task_index" not in names or text_column is None:
        raise ValueError(f"unexpected tasks.parquet schema: {table.schema.names}")
    result = sorted(
        (
            {
                "task_index": int(row["task_index"]),
                "task": str(row[text_column]),
            }
            for row in table.select(["task_index", text_column]).to_pylist()
        ),
        key=lambda row: row["task_index"],
    )
    if result != [{"task_index": 0, "task": "flip table"}]:
        raise ValueError(
            "Furniture-GR00T training requires exactly task_index 0 = 'flip table'; "
            f"got {result}"
        )
    return result


def validate_evaluation_checkpoint(
    evaluation: dict[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    if evaluation.get("model_safetensors_sha256") != sha256(checkpoint_path):
        raise ValueError(
            "final evaluation checkpoint hash differs from the selected model"
        )


def validate_progress_sidecar_bundle(
    progress_manifest_path: Path,
    visual_manifest_path: Path,
) -> dict[str, Path]:
    """Verify all 174 annotations and their human-reviewed visual provenance."""

    progress_manifest_path = progress_manifest_path.resolve()
    visual_manifest_path = visual_manifest_path.resolve()
    progress_manifest = read_json(progress_manifest_path)
    visual_manifest = read_json(visual_manifest_path)
    if (
        progress_manifest.get("schema_version")
        != "flip_table_progress_sidecar_manifest_v1"
        or progress_manifest.get("dataset_repo_id") != PINNED_DATASET_REPO_ID
        or progress_manifest.get("dataset_revision") != PINNED_DATASET_REVISION
        or progress_manifest.get("milestones") != list(MILESTONE_NAMES)
    ):
        raise ValueError("progress sidecar manifest violates the pinned contract")
    summary = progress_manifest.get("summary") or {}
    orientation_groups = summary.get("orientation_groups") or {}
    if (
        int(summary.get("episode_count", -1)) != 174
        or summary.get("fixed_phase_segmentation") is not False
        or set(orientation_groups) != {"0", "1", "2", "3"}
        or sum(int(value) for value in orientation_groups.values()) != 174
    ):
        raise ValueError("progress sidecar summary is incomplete")
    expected_exclusions = {
        "milestone labels",
        "progress labels",
        "future images",
        "sim ground truth",
        "object pose",
        "contact ground truth",
    }
    if set(progress_manifest.get("policy_input_exclusions") or ()) != expected_exclusions:
        raise ValueError("progress labels are not isolated from policy inputs")

    progress_path = _sibling_artifact(
        progress_manifest_path,
        progress_manifest.get("annotation_file"),
        expected_name="progress.jsonl",
    )
    if sha256(progress_path) != progress_manifest.get("annotation_sha256"):
        raise ValueError("progress sidecar hash differs from its manifest")
    progress_lengths, review_count, valid_counts = _validate_progress_annotations(
        progress_path
    )
    if (
        int(summary.get("review_required_count", -1)) != review_count
        or summary.get("valid_by_milestone") != valid_counts
    ):
        raise ValueError("progress manifest summary differs from its annotations")

    if (
        visual_manifest.get("schema_version")
        != "flip_table_visual_rotation_manifest_v1"
        or visual_manifest.get("dataset_repo_id") != PINNED_DATASET_REPO_ID
        or visual_manifest.get("dataset_revision") != PINNED_DATASET_REVISION
        or visual_manifest.get("video_key") != "observation.images.cam_0"
        or int(visual_manifest.get("episode_count", -1)) != 174
        or visual_manifest.get("contact_sheet_human_review_required") is not True
        or visual_manifest.get("policy_input") is not False
    ):
        raise ValueError("visual-rotation manifest violates the pinned contract")
    visual_path = _sibling_artifact(
        visual_manifest_path,
        "visual_rotation.jsonl",
        expected_name="visual_rotation.jsonl",
    )
    if sha256(visual_path) != visual_manifest.get("sidecar_sha256"):
        raise ValueError("visual-rotation sidecar hash differs from its manifest")
    _validate_visual_annotations(visual_path, progress_lengths)

    visual_reference = progress_manifest.get("visual_rotation_sidecar") or {}
    if visual_reference.get("sha256") != sha256(visual_path):
        raise ValueError("progress annotations reference a different visual sidecar")
    contact_sheet = _sibling_artifact(
        visual_manifest_path,
        visual_manifest.get("contact_sheet"),
        expected_name="orientation_contact_sheet.jpg",
    )
    contact_sheet_hash = sha256(contact_sheet)
    if contact_sheet_hash != visual_manifest.get("contact_sheet_sha256"):
        raise ValueError("orientation contact-sheet hash differs from its manifest")
    approval = _sibling_artifact(
        visual_manifest_path,
        "orientation_contact_sheet.approved",
        expected_name="orientation_contact_sheet.approved",
    )
    if approval.read_text(encoding="utf-8").strip() != contact_sheet_hash:
        raise ValueError("orientation contact sheet lacks matching human approval")
    return {
        "progress.jsonl": progress_path,
        "visual_rotation.jsonl": visual_path,
        "orientation_contact_sheet.jpg": contact_sheet,
        "orientation_contact_sheet.approved": approval,
    }


def _sibling_artifact(
    manifest_path: Path,
    name: Any,
    *,
    expected_name: str,
) -> Path:
    if not isinstance(name, str) or name != expected_name:
        raise ValueError(f"sidecar artifact must be named {expected_name}")
    path = manifest_path.parent / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _validate_progress_annotations(
    path: Path,
) -> tuple[dict[int, int], int, dict[str, int]]:
    records = _jsonl_records(path)
    if [int(item.get("episode_index", -1)) for item in records] != list(range(174)):
        raise ValueError("progress sidecar must contain exactly episodes 0..173")
    lengths: dict[int, int] = {}
    valid_counts = {name: 0 for name in MILESTONE_NAMES}
    review_count = 0
    for item in records:
        episode = int(item["episode_index"])
        length = int(item.get("length", -1))
        progress = item.get("progress")
        mask = item.get("progress_mask")
        milestones = item.get("milestones") or {}
        if (
            item.get("schema_version") != "flip_table_event_progress_v1"
            or length < 2
            or not isinstance(progress, list)
            or not isinstance(mask, list)
            or len(progress) != length
            or len(mask) != length
            or set(milestones) != set(MILESTONE_NAMES)
        ):
            raise ValueError(f"invalid progress annotation for episode {episode}")
        values = [float(value) for value in progress]
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
            or not all(isinstance(value, bool) for value in mask)
        ):
            raise ValueError(f"invalid progress values for episode {episode}")
        masked_values = [value for value, valid in zip(values, mask, strict=True) if valid]
        if any(
            current + 1.0e-7 < previous
            for previous, current in zip(masked_values, masked_values[1:])
        ):
            raise ValueError(f"non-monotonic progress for episode {episode}")
        previous_frame = -1
        for name in MILESTONE_NAMES:
            milestone = milestones[name]
            valid = bool(milestone.get("valid"))
            frame = milestone.get("frame")
            confidence = float(milestone.get("confidence", 0.0))
            expected_valid = (
                isinstance(frame, int)
                and 0 <= frame < length
                and confidence >= 0.5
            )
            if valid is not expected_valid:
                raise ValueError(
                    f"inconsistent {name} validity for episode {episode}"
                )
            if valid:
                if int(frame) <= previous_frame:
                    raise ValueError(
                        f"unordered milestones for episode {episode}"
                    )
                previous_frame = int(frame)
                valid_counts[name] += 1
        review_count += int(bool(item.get("review_required")))
        lengths[episode] = length
    return lengths, review_count, valid_counts


def _validate_visual_annotations(
    path: Path,
    progress_lengths: dict[int, int],
) -> None:
    records = _jsonl_records(path)
    if [int(item.get("episode_index", -1)) for item in records] != list(range(174)):
        raise ValueError("visual sidecar must contain exactly episodes 0..173")
    for item in records:
        episode = int(item["episode_index"])
        length = int(item.get("length", -1))
        rotation = item.get("rotation_rad")
        confidence = item.get("confidence")
        if (
            item.get("schema_version") != "flip_table_visual_rotation_v1"
            or length != progress_lengths[episode]
            or not isinstance(rotation, list)
            or not isinstance(confidence, list)
            or len(rotation) != length
            or len(confidence) != length
            or not all(math.isfinite(float(value)) for value in rotation)
            or not all(
                math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
                for value in confidence
            )
        ):
            raise ValueError(f"invalid visual annotation for episode {episode}")


def command_output(command: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_directory_manifest(root: Path, *, schema_version: str) -> dict[str, Any]:
    """Hash every regular file in a release artifact directory."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError(f"release artifact directory is empty: {root}")
    return {
        "schema_version": schema_version,
        "root": root.name,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "files": files,
    }


def build_source_snapshot_manifest(
    *, bundle_root: Path | None = None
) -> dict[str, Any]:
    """Bundle and hash the complete release-time source tree."""
    if bundle_root is not None:
        bundle_root = bundle_root.resolve()
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        bundle_root.mkdir(parents=True)
    files: dict[str, str] = {}
    for relative_root in SOURCE_SNAPSHOT_ROOTS:
        root = REPO_ROOT / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(REPO_ROOT)
            if (
                not path.is_file()
                or path.suffix not in SOURCE_SNAPSHOT_SUFFIXES
                or SOURCE_SNAPSHOT_EXCLUDED_PARTS.intersection(relative.parts)
            ):
                continue
            files[relative.as_posix()] = sha256(path)
            if bundle_root is not None:
                destination = bundle_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    required = {
        "model/subtask_policy_training/gr00t/g1_full_body_mapping.py",
        "model/subtask_policy_training/gr00t/n17_contract.py",
        (
            "model/subtask_policy_training/lerobot_policy_furniture_groot/"
            "lerobot_policy_furniture_groot/modeling_furniture_groot.py"
        ),
        "model/subtask_policy_training/scripts/materialize_lerobot_training_view.py",
        "model/subtask_policy_training/scripts/evaluate_groot_n17_offline.py",
        "model/subtask_policy_training/scripts/run_h100_flip_table_groot_n17.sh",
        "model/subtask_policy_training/scripts/select_groot_n17_candidate.py",
        "model/subtask_policy_training/scripts/validate_groot_n17_candidate.py",
        "model/subtask_policy_training/scripts/finalize_groot_n17_checkpoint.py",
        "model/subtask_policy_training/scripts/upload_policy.py",
        "model/subtask_policy_training/scripts/verify_policy_hub_roundtrip.py",
        (
            "model/subtask_policy_training/deployment/"
            "real_furniture_groot_n17_worker.py"
        ),
        "inference/desktop/upper_policy/furniture_groot_contract.py",
        "inference/desktop/upper_policy/run_flip_table_furniture_groot.py",
        "inference/desktop/upper_policy/run_real_flip_table_furniture_groot.sh",
        (
            "evaluate/flip_table_simulation/container_overlay/policy/"
            "flip_table_eval_policy.py"
        ),
        (
            "evaluate/flip_table_simulation/groot_runtime/"
            "groot_inference_server.py"
        ),
        "evaluate/flip_table_simulation/run_eval.sh",
        "evaluate/flip_table_simulation/run_eval_in_container.sh",
        "evaluate/flip_table_simulation/run_groot_candidate_comparison.sh",
        "evaluate/flip_table_simulation/run_groot_release_evaluation.sh",
        (
            "evaluate/flip_table_simulation/"
            "summarize_groot_candidate_comparison.py"
        ),
        (
            "evaluate/flip_table_simulation/"
            "summarize_groot_release_evaluation.py"
        ),
    }
    missing = sorted(required - files.keys())
    if missing:
        raise FileNotFoundError(f"source snapshot is missing release files: {missing}")
    return {
        "schema_version": "groot_n17_source_snapshot_v2",
        "scope": "release_time_training_inference_and_evaluation_source",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": command_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "git_status_porcelain": command_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT
        ),
        "roots": [path.as_posix() for path in SOURCE_SNAPSHOT_ROOTS],
        "bundle_root": SOURCE_SNAPSHOT_BUNDLE,
        "file_count": len(files),
        "files": files,
    }


def discover_wandb_url(training_output: Path) -> str:
    run_dirs = sorted((training_output / "wandb").glob("run-*-*"))
    run_dirs = [path for path in run_dirs if path.is_dir()]
    if len(run_dirs) != 1:
        raise ValueError(
            f"expected one W&B run under {training_output}, found {len(run_dirs)}"
        )
    run_id = run_dirs[0].name.rsplit("-", 1)[-1]
    project = os.environ.get("WANDB_PROJECT", "").strip()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if not project:
        raise ValueError("WANDB_PROJECT is required to finalize the checkpoint")
    if not entity:
        try:
            import wandb

            entity = str(wandb.Api().default_entity or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("failed to resolve the W&B entity") from exc
    if not entity:
        raise ValueError("WANDB_ENTITY or a W&B default entity is required")
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--progress-manifest", type=Path, required=True)
    parser.add_argument("--visual-rotation-manifest", type=Path, required=True)
    parser.add_argument("--video-cache-manifest", type=Path)
    parser.add_argument("--eef-fk-audit", type=Path)
    parser.add_argument("--sim-comparison-report", type=Path)
    parser.add_argument("--sim-release-report", type=Path)
    parser.add_argument("--sim-evaluation-bundle", type=Path)
    parser.add_argument("--wandb-url", default="")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    training_output = args.training_output.resolve()
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = read_json(model_dir / "config.json")
    if config.get("type") != "furniture_groot":
        raise ValueError("only a contract-preserving furniture_groot checkpoint can be finalized")
    if config.get("base_model_path") != PINNED_BASE_MODEL_REPO_ID:
        raise ValueError("checkpoint must use the canonical pinned GR00T N1.7 repository")
    revision = config.get("base_model_revision")
    if revision not in (None, PINNED_BASE_MODEL_REVISION):
        raise ValueError(f"unexpected GR00T base revision: {revision!r}")
    actual_tuning_scope = {
        key: config.get(key) for key in EXPECTED_TUNING_SCOPE
    }
    if actual_tuning_scope != EXPECTED_TUNING_SCOPE:
        raise ValueError(
            "checkpoint tuning scope must freeze vision/LLM and update only the "
            "reviewed N1.7 post-training modules: "
            f"{actual_tuning_scope}"
        )
    config["base_model_revision"] = PINNED_BASE_MODEL_REVISION
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = read_json(args.training_view / "meta" / "team_ramen_training_view.json")
    split = read_json(args.training_view / "meta" / "team_ramen_episode_split.json")
    task_contract = read_training_task_contract(
        args.training_view / "meta" / "tasks.parquet"
    )
    evaluation = read_json(args.evaluation_report)
    selection = read_json(args.selection_report)
    contract = read_json(args.contract_report)
    progress_manifest = read_json(args.progress_manifest)
    visual_manifest = read_json(args.visual_rotation_manifest)
    sidecar_artifacts = validate_progress_sidecar_bundle(
        args.progress_manifest,
        args.visual_rotation_manifest,
    )
    if (
        marker.get("source_repo_id") != PINNED_DATASET_REPO_ID
        or marker.get("source_revision") != PINNED_DATASET_REVISION
    ):
        raise ValueError("training view does not use the pinned flip-table dataset")
    video_cache_manifest = (
        read_json(args.video_cache_manifest)
        if args.video_cache_manifest is not None
        else None
    )
    eef_fk_audit_path = (
        args.eef_fk_audit.resolve()
        if args.eef_fk_audit is not None
        else training_output.parent / "eef_fk_audit.json"
    )
    eef_fk_audit = read_json(eef_fk_audit_path)
    eef_fk_validation = validate_eef_fk_release_audit(eef_fk_audit)
    if (
        eef_fk_audit["source_repo_id"] != marker["source_repo_id"]
        or eef_fk_audit["source_revision"] != marker["source_revision"]
    ):
        raise ValueError("EEF-FK audit source does not match the training dataset")
    wandb_url = args.wandb_url.strip() or discover_wandb_url(training_output)
    if video_cache_manifest is not None:
        if video_cache_manifest.get("schema_version") != (
            "flip_table_lossless_intra_video_cache_v1"
        ):
            raise ValueError("unexpected lossless video-cache schema")
        if video_cache_manifest.get("pixel_exact") is not True:
            raise ValueError("training video cache must be pixel-exact")
        cache_source = video_cache_manifest.get("source", {})
        if cache_source.get("revision") != marker["source_revision"]:
            raise ValueError("training video cache revision does not match the dataset")
        if (
            cache_source.get("fingerprint_sha256")
            != marker["source_fingerprint_sha256"]
        ):
            raise ValueError("training video cache fingerprint does not match the dataset")
    hand_audit = contract.get("dex1_official_q01_q99_audit")
    if not isinstance(hand_audit, dict):
        raise ValueError("GR00T contract report is missing the Dex1 official-range audit")
    for name, group in hand_audit.get("groups", {}).items():
        if float(group.get("maximum_outside_fraction", 1.0)) > 0.05:
            raise ValueError(f"Dex1 official-range audit failed for {name}")

    test_episodes = split["splits"]["test"]["episode_indices"]
    validate_evaluation_checkpoint(evaluation, checkpoint_path=checkpoint)
    evaluation_randomness = evaluation.get("randomness") or {}
    if (
        evaluation.get("schema_version") != "groot_n17_offline_chunk_reset_v2"
        or int(evaluation_randomness.get("base_seed", -1)) != 42
        or int(evaluation_randomness.get("episode_stride", 0)) <= 0
        or int(evaluation_randomness.get("uint32_modulus", 0)) != 2**32
    ):
        raise ValueError(
            "final evaluation lacks deterministic per-episode/chunk inference seeds"
        )
    if evaluation.get("episodes") != test_episodes:
        raise ValueError("final evaluation must cover exactly the immutable test split")
    if evaluation.get("declared_split") != "test":
        raise ValueError("final evaluation must be labeled as the declared test split")
    if (
        selection.get("selection_data")
        != "offline_validation_plus_same_seed_sim_validation"
    ):
        raise ValueError(
            "candidate selection must use offline and same-seed simulator validation"
        )

    sim_comparison_path = (
        args.sim_comparison_report.resolve()
        if args.sim_comparison_report is not None
        else training_output.parent / "sim_candidate_selection.json"
    )
    sim_release_path = (
        args.sim_release_report.resolve()
        if args.sim_release_report is not None
        else training_output.parent / "sim_release_evaluation.json"
    )
    sim_bundle = (
        args.sim_evaluation_bundle.resolve()
        if args.sim_evaluation_bundle is not None
        else training_output.parent / "sim_evaluation_bundle"
    )
    sim_comparison = read_json(sim_comparison_path)
    sim_release = read_json(sim_release_path)
    temporal_selection_path = sim_bundle / "release" / "temporal_selection.json"
    temporal_selection = read_json(temporal_selection_path)
    validated_temporal = validate_temporal_selection_report(
        temporal_selection
    )
    fixed_scene = sim_release.get("fixed_scene") or {}
    unseen_dr = sim_release.get("unseen_dr") or {}
    selected_temporal = sim_release.get("selected_temporal_setting") or {}
    selected_temporal_lambda = str(selected_temporal.get("temporal_lambda"))
    selected_execution_steps = int(selected_temporal.get("execution_steps", -1))
    temporal_setting_evidence = (
        selected_temporal_lambda in {"none", "-0.25", "-0.1", "0"}
        and selected_execution_steps in {5, 10, 20}
        and validated_temporal["temporal_lambda"]
        == selected_temporal_lambda
        and validated_temporal["execution_steps"]
        == selected_execution_steps
        and sim_release.get("temporal_validation") == temporal_selection
        and sim_release.get("temporal_validation_sha256")
        == sha256(temporal_selection_path)
        and sim_release.get("scripted_controller_tracking")
        == temporal_selection.get("scripted_controller_tracking")
        and str(fixed_scene.get("temporal_lambda")) == selected_temporal_lambda
        and int(fixed_scene.get("execution_steps", -1))
        == selected_execution_steps
        and str(unseen_dr.get("temporal_lambda")) == selected_temporal_lambda
        and int(unseen_dr.get("execution_steps", -1))
        == selected_execution_steps
    )
    fixed_seed_evidence = (
        fixed_scene.get("seed") == 93001
        and fixed_scene.get("policy_inference_seed") == 93001
        and fixed_scene.get("episode_inference_seeds") == [93001, 93002, 93003]
        and fixed_scene.get("episode_ids")
        == [f"93001:{index}" for index in range(3)]
        and fixed_scene.get("mode") == "nominal"
        and fixed_scene.get("domain_randomization_profile")
        == SIM_FIXED_DR_PROFILE
        and fixed_scene.get("runtime_evaluation_mode") == "nominal"
    )
    unseen_seed_evidence = (
        unseen_dr.get("seed") == 94001
        and unseen_dr.get("policy_inference_seed") == 94001
        and unseen_dr.get("episode_inference_seeds")
        == [94001 + index for index in range(50)]
        and unseen_dr.get("episode_ids")
        == [f"94001:{index}" for index in range(50)]
        and unseen_dr.get("mode") == "unseen_dr"
        and unseen_dr.get("domain_randomization_profile")
        == SIM_UNSEEN_DR_PROFILE
        and unseen_dr.get("runtime_evaluation_mode") == "unseen_dr"
    )
    evidence = selection.get("evidence") or {}
    candidate_hashes = selection.get("candidate_hashes") or {}
    selected_candidate = selection.get("selected")
    strict_candidate_evidence = (
        set(candidate_hashes) == {"baseline", "auxiliary_progress"}
        and valid_sim_candidate_evidence(
            sim_comparison,
            candidate_hashes={
                str(name): str(value) for name, value in candidate_hashes.items()
            },
        )
        and sim_comparison.get("selected") == selected_candidate
        and candidate_hashes.get(selected_candidate) == sha256(checkpoint)
        and config.get("progress_enabled")
        is (selected_candidate == "auxiliary_progress")
    )
    if (
        evidence.get("sim_candidate_comparison_sha256")
        != sha256(sim_comparison_path)
        or evidence.get("sim_release_evaluation_sha256")
        != sha256(sim_release_path)
    ):
        raise ValueError("simulator evidence differs from candidate selection")
    if (
        sim_comparison.get("schema_version")
        != "groot_n17_sim_candidate_comparison_v1"
        or sim_release.get("schema_version")
        != "team_ramen_groot_n17_release_evaluation/v1"
        or not strict_candidate_evidence
        or sim_comparison.get("selected") != selection.get("selected")
        or sim_release.get("candidate_name") != selected_candidate
        or sim_release.get("model_safetensors_sha256") != sha256(checkpoint)
        or int(fixed_scene.get("test_count", -1)) != 3
        or int(fixed_scene.get("success_count", -1)) != 3
        or not fixed_seed_evidence
        or int(unseen_dr.get("test_count", -1)) != 50
        or int(unseen_dr.get("success_count", -1)) < 40
        or not unseen_seed_evidence
        or not temporal_setting_evidence
        or (sim_release.get("release_goal") or {}).get("unseen_dr_passed")
        is not True
    ):
        raise ValueError("simulator comparison or release gate did not pass")
    if (
        not sim_bundle.is_dir()
        or sha256(sim_bundle / "sim_candidate_selection.json")
        != sha256(sim_comparison_path)
        or sha256(sim_bundle / "sim_release_evaluation.json")
        != sha256(sim_release_path)
    ):
        raise ValueError("complete simulator evaluation bundle is unavailable")
    destination_bundle = model_dir / "sim_evaluation"
    if destination_bundle.exists():
        shutil.rmtree(destination_bundle)
    shutil.copytree(
        sim_bundle,
        destination_bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.sock"),
    )
    sim_bundle_manifest = build_directory_manifest(
        destination_bundle,
        schema_version="groot_n17_sim_evaluation_bundle_v1",
    )
    sim_bundle_manifest_path = model_dir / "sim_evaluation_manifest.json"
    sim_bundle_manifest_path.write_text(
        json.dumps(sim_bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_snapshot = build_source_snapshot_manifest(
        bundle_root=model_dir / SOURCE_SNAPSHOT_BUNDLE
    )
    source_snapshot_path = model_dir / "source_snapshot_manifest.json"
    source_snapshot_path.write_text(
        json.dumps(source_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record = {
        "schema_version": "groot_n17_training_run_record_v1",
        "source": {
            "snapshot_scope": source_snapshot["scope"],
            "snapshot_captured_at_utc": source_snapshot["captured_at_utc"],
            "git_head": source_snapshot["git_head"],
            "git_status_porcelain": source_snapshot["git_status_porcelain"],
            "snapshot_manifest": source_snapshot_path.name,
            "snapshot_manifest_sha256": sha256(source_snapshot_path),
            "snapshot_file_count": source_snapshot["file_count"],
        },
        "hardware": {
            "nvidia_smi_at_completion": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                cwd=REPO_ROOT,
            )
        },
        "dataset": {
            "repo_id": marker["source_repo_id"],
            "revision": marker["source_revision"],
            "task_instruction": task_contract[0]["task"],
            "task_index": task_contract[0]["task_index"],
            "source_fingerprint_sha256": marker["source_fingerprint_sha256"],
            "episode_split_sha256": split["sha256"],
            "counts": {
                name: len(split["splits"][name]["episode_indices"])
                for name in ("train", "validation", "test")
            },
        },
        "contract": contract,
        "training_scope": {
            "config_flags": actual_tuning_scope,
            "frozen_modules": [
                "backbone.language_model",
                "backbone.visual",
            ],
            "trainable_modules": [
                "action_head.state_encoder",
                "action_head.action_encoder",
                "action_head.action_decoder",
                "action_head.position_embedding",
                "action_head.flow_matching_model",
                "action_head.vlln",
                "action_head.vl_self_attention",
                "progress_head (auxiliary candidate only)",
            ],
            "source": "GR00TN17.set_trainable_parameters and GrootPolicy configuration",
        },
        "dex1_adapter": {
            "path": "dex1_g1_synergy.json",
            "sha256": sha256(SYNERGY_ASSET),
            "official_q01_q99_audit": hand_audit,
        },
        "augmentation": {
            "dataset_cpu_transforms": False,
            "consistent_gpu_multiview": bool(
                config.get("consistent_gpu_augmentation", False)
            ),
            "coherence_scope": "all 2 timestamps x 3 policy cameras per sample",
            "inference_enabled": False,
        },
        "sidecars": {
            "progress_manifest_sha256": sha256(args.progress_manifest),
            "visual_rotation_manifest_sha256": sha256(args.visual_rotation_manifest),
            "progress_manifest": progress_manifest,
            "visual_rotation_manifest": visual_manifest,
            "artifacts": {
                name: sha256(path)
                for name, path in sidecar_artifacts.items()
            },
            "contact_sheet_review": {
                "required": True,
                "approved_sha256": (
                    sidecar_artifacts[
                        "orientation_contact_sheet.approved"
                    ]
                    .read_text(encoding="utf-8")
                    .strip()
                ),
            },
        },
        "eef_fk_audit": {
            "path": "eef_fk_audit.json",
            "sha256": sha256(eef_fk_audit_path),
            "source_repo_id": eef_fk_audit["source_repo_id"],
            "source_revision": eef_fk_audit["source_revision"],
            "episode_count": eef_fk_audit["coverage"]["episode_count"],
            "action_fk_residual_pass": True,
            "frame_assignment_pass": True,
            "selected_offset_frames": 0,
            "teacher_pair_status": eef_fk_audit["training_contract"][
                "teacher_pair_status"
            ],
            "fixed_release_gate": eef_fk_validation,
        },
        "lossless_video_cache": (
            {
                "manifest_sha256": sha256(args.video_cache_manifest),
                "schema_version": video_cache_manifest["schema_version"],
                "pixel_exact": True,
                "summary": video_cache_manifest["summary"],
            }
            if video_cache_manifest is not None
            else None
        ),
        "candidate_selection": selection,
        "simulation_evaluation": {
            "candidate_comparison_sha256": sha256(sim_comparison_path),
            "release_evaluation_sha256": sha256(sim_release_path),
            "temporal_validation_sha256": sha256(
                temporal_selection_path
            ),
            "bundle_manifest_sha256": sha256(sim_bundle_manifest_path),
            "bundle_file_count": sim_bundle_manifest["file_count"],
            "bundle_total_bytes": sim_bundle_manifest["total_bytes"],
            "selected_candidate": selection["selected"],
            "fixed_scene_success": "3/3",
            "fixed_scene_dr_profile": SIM_FIXED_DR_PROFILE,
            "unseen_dr_success": (
                f"{sim_release['unseen_dr']['success_count']}/50"
            ),
            "unseen_dr_profile": SIM_UNSEEN_DR_PROFILE,
            "claim_scope": sim_release["claim_scope"],
        },
        "checkpoint": {
            "model_safetensors_sha256": sha256(checkpoint),
            "config_sha256": sha256(model_dir / "config.json"),
        },
        "wandb_url": wandb_url,
    }
    (model_dir / "training_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": "groot_n17_training_manifest_v1",
        "dataset": record["dataset"],
        "contract": {
            "logical_state_dim": 49,
            "logical_action_dim": 53,
            "packed_state_dim": 132,
            "packed_action_dim": 132,
            "valid_action_dim": 46,
            "action_horizon": 40,
            "physical_command": "14 arm joint targets + 2 absolute Dex1-1 commands",
            "policy_cameras": ["head_left", "left_wrist", "right_wrist"],
            "head_right_used": False,
            "progress_in_action": False,
            "task_instruction": task_contract[0]["task"],
            "progress_head_shape": (
                [40, 1] if bool(config.get("progress_enabled", False)) else None
            ),
        },
        "candidate_selection": selection,
        "simulation_evaluation": record["simulation_evaluation"],
        "training_scope": record["training_scope"],
        "augmentation": record["augmentation"],
        "eef_fk_audit": record["eef_fk_audit"],
        "checkpoint": record["checkpoint"],
        "evaluation": evaluation,
        "training_output": str(training_output),
        "wandb_url": wandb_url,
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    shutil.copyfile(args.evaluation_report, model_dir / "evaluation_report.json")
    shutil.copyfile(args.selection_report, model_dir / "candidate_selection.json")
    shutil.copyfile(
        sim_comparison_path,
        model_dir / "sim_candidate_selection.json",
    )
    shutil.copyfile(
        sim_release_path,
        model_dir / "sim_release_evaluation.json",
    )
    shutil.copyfile(args.contract_report, model_dir / "groot_contract.json")
    shutil.copyfile(args.progress_manifest, model_dir / "progress_manifest.json")
    shutil.copyfile(
        args.visual_rotation_manifest,
        model_dir / "visual_rotation_manifest.json",
    )
    for name, path in sidecar_artifacts.items():
        shutil.copyfile(path, model_dir / name)
    shutil.copyfile(SYNERGY_ASSET, model_dir / "dex1_g1_synergy.json")
    shutil.copyfile(eef_fk_audit_path, model_dir / "eef_fk_audit.json")

    card = f"""---
tags:
- lerobot
- groot-n1.7
- flip-table
private: true
---

# Flip-table GR00T N1.7

This private checkpoint preserves the pinned GR00T N1.7 REAL_G1 contract:
49-D logical state, 53-D logical action, 132-D packed action, H40, and a
46-D valid action mask. It consumes head-left and both D405 wrist RGB streams.
Head-right is not used.

The executable output is restricted to 14 arm joint targets and two absolute
Dex1-1 commands. EEF is auxiliary supervision. Waist, legs, root, base-height,
and navigation are never sent to the robot.

FurnitureVLA-style progress is a separate diagnostic head. It does not add an
action slot and cannot switch phases or generate commands. Candidate selection
used the 17 offline validation episodes plus same-seed simulator validation;
the immutable test split was untouched until selection. The final report covers
held-out test episodes `{", ".join(str(value) for value in test_episodes)}`.

The included offline chunk-reset metrics use recorded observations and are not
closed-loop simulator or real-robot success rates. Sim-to-Real success is not
claimed until the staged real G1 + Dex1-1 evaluation is completed.
Offline and simulator flow-matching seeds are recorded per chunk or episode and
verified during clean-download validation.

- Dataset revision: `{record["dataset"]["revision"]}`
- Split SHA-256: `{record["dataset"]["episode_split_sha256"]}`
- Checkpoint SHA-256: `{record["checkpoint"]["model_safetensors_sha256"]}`
- Progress adopted: `{selection["auxiliary_adopted"]}`
- Progress sidecar SHA-256: `{record["sidecars"]["artifacts"]["progress.jsonl"]}`
- Contact-sheet approval SHA-256: `{record["sidecars"]["contact_sheet_review"]["approved_sha256"]}`
- EEF/FK pooled p95 (left position/rotation): `{eef_fk_validation["pooled_action_metrics"]["left"]["position_p95_m"]:.6f} m / {eef_fk_validation["pooled_action_metrics"]["left"]["rotation_p95_rad"]:.6f} rad`
- EEF/FK pooled p95 (right position/rotation): `{eef_fk_validation["pooled_action_metrics"]["right"]["position_p95_m"]:.6f} m / {eef_fk_validation["pooled_action_metrics"]["right"]["rotation_p95_rad"]:.6f} rad`
- EEF/FK episode diagnostic threshold exceedances: `{eef_fk_validation["episode_threshold_exceedance_count"]}/174`
- EEF/FK selected time offset: `{eef_fk_validation["selected_offset_frames"]} frames`
- Temporal decay: `{selected_temporal_lambda}`
- Physical steps between replans: `{selected_execution_steps}`
- Temporal validation SHA-256: `{record["simulation_evaluation"]["temporal_validation_sha256"]}`
- Simulator fixed scene: `3/3`
- Simulator fixed-scene profile: `{SIM_FIXED_DR_PROFILE}`
- Simulator unseen DR: `{sim_release["unseen_dr"]["success_count"]}/50`
- Simulator unseen-DR profile: `{SIM_UNSEEN_DR_PROFILE}` (categorical appearance holdout plus low-friction/high-restitution edge band)
- Vision/LLM frozen: `true`
- W&B: `{wandb_url}`
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
