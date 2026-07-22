#!/usr/bin/env python3
"""Run the resumable RGB-D, mask, pose-track, and phase-annotation batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


FEATURE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FEATURE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.flip_table_data_augmentation.runtime_contract import stable_tree_sha256
from data.flip_table_data_augmentation.io_utils import (
    atomic_write_json,
    read_json_object,
    sha256_file,
)


DEFAULT_OUTPUTS = FEATURE_ROOT / "outputs"
DEFAULT_ASSEMBLED_SCENE = Path(
    "/workspace/IROS_IKEA_V13_20260702/Scene02_flip_table_assembled.usd"
)
BATCH_SCHEMA_VERSION = "team_ramen_source_annotation_batch/v4"
GENERATED_SOURCE_DIRECTORY_NAMES = frozenset(
    {"outputs", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


def _episodes(value: str) -> tuple[int, ...]:
    selected: set[int] = set()
    try:
        for token in value.split(","):
            token = token.strip()
            if not token:
                raise ValueError
            if "-" in token:
                start_text, stop_text = token.split("-", 1)
                start, stop = int(start_text), int(stop_text)
                if stop < start:
                    raise ValueError
                selected.update(range(start, stop + 1))
            else:
                selected.add(int(token))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "episodes must use comma-separated indices or inclusive ranges"
        ) from exc
    if not selected or min(selected) < 0 or max(selected) >= 531:
        raise argparse.ArgumentTypeError("episodes must lie in [0,530]")
    return tuple(sorted(selected))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _execution_path(path: Path, outputs: Path, *, runtime_mode: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(outputs.resolve())
    except ValueError as exc:
        raise ValueError(f"batch path must remain under outputs root: {resolved}") from exc
    if runtime_mode == "direct":
        return str(resolved)
    if runtime_mode != "docker":
        raise ValueError("resolved runtime mode must be direct or docker")
    return f"/outputs/{relative.as_posix()}"


def _run(command: list[str], env: dict[str, str]) -> tuple[int, float]:
    print("+", " ".join(command), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, env=env, check=False)
    return result.returncode, time.monotonic() - started


def _batch_identity(
    config: dict[str, Any],
    requested: tuple[int, ...],
    *,
    augmentation_source_sha256: str,
    mesh_sha256: str,
) -> dict[str, Any]:
    canonical = json.dumps(
        config,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "augmentation_source_sha256": augmentation_source_sha256,
        "assembled_table_mesh_sha256": mesh_sha256,
        "source_repo_id": config["source"]["repo_id"],
        "source_revision": config["source"]["revision"],
        "episodes": list(requested),
        "container_digest": config["runtime"]["container_digest"],
    }


def _resume_command(command: list[str], resume: bool) -> list[str]:
    return [*command, "--resume"] if resume else list(command)


def _resolve_runtime_mode(runtime_mode: str) -> str:
    if runtime_mode not in {"auto", "docker", "direct"}:
        raise ValueError("FLIP_TABLE_AUG_RUNTIME_MODE must be auto, docker, or direct")
    if runtime_mode != "auto":
        return runtime_mode
    if shutil.which("docker") is None:
        return "direct"
    probe = subprocess.run(
        ["docker", "info"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return "docker" if probe.returncode == 0 else "direct"


def _v1_entrypoint(feature_root: Path, runtime_mode: str) -> Path:
    if runtime_mode not in {"docker", "direct"}:
        raise ValueError("resolved runtime mode must be direct or docker")
    name = "run_v1_direct.sh" if runtime_mode == "direct" else "run_v1_container.sh"
    return feature_root / "scripts" / name


def _failure_details(stage: str, stage_root: Path, outputs: Path) -> dict[str, Any]:
    details: dict[str, Any] = {"stage": stage}
    manifest_path = stage_root / "manifest.json"
    if not manifest_path.is_file():
        return details
    manifest = read_json_object(manifest_path)
    details.update(
        {
            "manifest": str(manifest_path.relative_to(outputs)),
            "manifest_sha256": sha256_file(manifest_path),
        }
    )
    for key in ("accepted", "rejection_reasons", "gate"):
        if key in manifest:
            details[key] = manifest[key]
    return details


def _failure_status(returncode: int, details: dict[str, Any]) -> str:
    reasons = details.get("rejection_reasons")
    gate = details.get("gate")
    quality_rejection = (
        returncode == 2
        and details.get("accepted") is False
        and isinstance(reasons, list)
        and bool(reasons)
        and isinstance(gate, dict)
        and gate.get("pass") is False
    )
    return "rejected" if quality_rejection else "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=_episodes, default=_episodes("0-530"))
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument(
        "--stop-after",
        choices=("prepare", "masks", "track", "annotate"),
        default="annotate",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-rejection", action="store_true")
    args = parser.parse_args()

    outputs = args.outputs_root.expanduser().resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    config_path = FEATURE_ROOT / "configs/pipeline_v1.json"
    config = read_json_object(config_path)
    runtime = FEATURE_ROOT / "scripts/run_object_pose_runtime.sh"
    runtime_mode = _resolve_runtime_mode(
        os.environ.get("FLIP_TABLE_AUG_RUNTIME_MODE", "auto")
    )
    v1 = _v1_entrypoint(FEATURE_ROOT, runtime_mode)
    env = dict(os.environ)
    env["FLIP_TABLE_AUG_OUTPUTS"] = str(outputs)
    mesh = outputs / "source/v1-table-mesh/Table001_assembled_body_frame.obj"
    mesh_manifest = mesh.with_name("manifest.json")
    if not mesh.is_file() or not mesh_manifest.is_file():
        command = [
            str(v1),
            "export-table-mesh",
            "--assembled-scene",
            str(DEFAULT_ASSEMBLED_SCENE),
            "--output-obj",
            _execution_path(mesh, outputs, runtime_mode=runtime_mode),
            "--manifest",
            _execution_path(mesh_manifest, outputs, runtime_mode=runtime_mode),
        ]
        returncode, _elapsed = _run(command, env)
        if returncode != 0:
            raise RuntimeError(f"V1 table mesh export failed with exit code {returncode}")
    augmentation_source_sha256, _augmentation_source_files = stable_tree_sha256(
        FEATURE_ROOT,
        excluded_directory_names=GENERATED_SOURCE_DIRECTORY_NAMES,
    )
    batch_root = outputs / "source/annotation-batch"
    ledger_root = batch_root / "episodes"
    requested = tuple(args.episodes)
    batch_identity = _batch_identity(
        config,
        requested,
        augmentation_source_sha256=augmentation_source_sha256,
        mesh_sha256=sha256_file(mesh),
    )
    config_sha = str(batch_identity["config_sha256"])
    identity_path = batch_root / "identity.json"
    if identity_path.exists():
        if not args.resume or read_json_object(identity_path) != batch_identity:
            raise ValueError("existing annotation batch has a different identity")
    else:
        atomic_write_json(identity_path, batch_identity)

    stage_order = ("prepare", "masks", "track", "annotate")
    final_stage = stage_order.index(args.stop_after)
    batch_started = time.monotonic()
    exit_code = 0
    for ordinal, episode in enumerate(requested, start=1):
        label = f"episode-{episode:06d}"
        ledger_path = ledger_root / f"{label}.json"
        if ledger_path.exists():
            ledger = read_json_object(ledger_path)
            if ledger.get("config_sha256") != config_sha:
                raise ValueError(f"{label} ledger uses a different config")
            if args.resume and ledger.get("status") in {"accepted", "rejected"}:
                print(f"[{ordinal}/{len(requested)}] {label}: resume {ledger['status']}", flush=True)
                continue
        else:
            ledger = {
                "episode_index": episode,
                "config_sha256": config_sha,
                "created_utc": _utc_now(),
                "history": [],
            }
        print(f"[{ordinal}/{len(requested)}] {label}", flush=True)
        input_dir = outputs / "source/foundationpose-input" / label
        mask_dir = outputs / "source/foundationpose-masks" / label
        track_dir = outputs / "source/foundationpose-tracks" / label
        annotation_dir = outputs / "source/annotations" / label
        commands = (
            (
                "prepare",
                [
                    str(v1),
                    "prepare-pose-input",
                    "--episode-index",
                    str(episode),
                    "--output-dir",
                    _execution_path(input_dir, outputs, runtime_mode=runtime_mode),
                ],
            ),
            (
                "masks",
                [
                    str(runtime),
                    "masks",
                    "--input-dir",
                    _execution_path(input_dir, outputs, runtime_mode=runtime_mode),
                    "--output-dir",
                    _execution_path(mask_dir, outputs, runtime_mode=runtime_mode),
                ],
            ),
            (
                "track",
                [
                    str(runtime),
                    "track",
                    "--input-dir",
                    _execution_path(input_dir, outputs, runtime_mode=runtime_mode),
                    "--mask-dir",
                    _execution_path(mask_dir, outputs, runtime_mode=runtime_mode),
                    "--mesh",
                    _execution_path(mesh, outputs, runtime_mode=runtime_mode),
                    "--output-dir",
                    _execution_path(track_dir, outputs, runtime_mode=runtime_mode),
                ],
            ),
            (
                "annotate",
                [
                    str(v1),
                    "annotate-source",
                    "--track-dir",
                    _execution_path(track_dir, outputs, runtime_mode=runtime_mode),
                    "--output-dir",
                    _execution_path(annotation_dir, outputs, runtime_mode=runtime_mode),
                ],
            ),
        )
        status = "accepted"
        for stage, command in commands[: final_stage + 1]:
            returncode, elapsed = _run(_resume_command(command, args.resume), env)
            ledger["history"].append(
                {
                    "stage": stage,
                    "returncode": returncode,
                    "elapsed_s": elapsed,
                    "finished_utc": _utc_now(),
                }
            )
            ledger["last_stage"] = stage
            ledger["updated_utc"] = _utc_now()
            if returncode != 0:
                stage_root = {
                    "prepare": input_dir,
                    "masks": mask_dir,
                    "track": track_dir,
                    "annotate": annotation_dir,
                }[stage]
                details = _failure_details(stage, stage_root, outputs)
                status = _failure_status(returncode, details)
                ledger["status"] = status
                ledger["failure_details"] = details
                atomic_write_json(ledger_path, ledger)
                if not args.continue_on_rejection or status == "failed":
                    exit_code = returncode
                break
            ledger["status"] = "running"
            atomic_write_json(ledger_path, ledger)
        else:
            status = "accepted" if args.stop_after == "annotate" else f"completed_{args.stop_after}"
        ledger["status"] = status
        ledger["updated_utc"] = _utc_now()
        atomic_write_json(ledger_path, ledger)
        if exit_code:
            break

    ledger_paths = [ledger_root / f"episode-{episode:06d}.json" for episode in requested]
    ledgers = [read_json_object(path) for path in ledger_paths if path.is_file()]
    counts: dict[str, int] = {}
    elapsed_by_stage: dict[str, float] = {}
    for ledger in ledgers:
        status = str(ledger.get("status"))
        counts[status] = counts.get(status, 0) + 1
        for record in ledger.get("history", []):
            stage = str(record["stage"])
            elapsed_by_stage[stage] = elapsed_by_stage.get(stage, 0.0) + float(
                record["elapsed_s"]
            )
    merged = None
    if args.stop_after == "annotate" and counts.get("accepted", 0):
        merged_path = outputs / "source/annotations.json"
        command = [
            str(v1),
            "merge-source-annotations",
            "--output",
            _execution_path(merged_path, outputs, runtime_mode=runtime_mode),
        ]
        for ledger in ledgers:
            if ledger.get("status") != "accepted":
                continue
            annotation = (
                outputs
                / "source/annotations"
                / f"episode-{int(ledger['episode_index']):06d}"
                / "annotation.json"
            )
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            command.extend(
                (
                    "--annotation",
                    _execution_path(annotation, outputs, runtime_mode=runtime_mode),
                )
            )
        returncode, elapsed = _run(command, env)
        if returncode != 0:
            exit_code = returncode
        else:
            merged = {
                "path": str(merged_path),
                "sha256": sha256_file(merged_path),
                "elapsed_s": elapsed,
            }
    completed_status = "accepted" if args.stop_after == "annotate" else f"completed_{args.stop_after}"
    finished = sum(
        ledger.get("status") in {completed_status, "rejected"} for ledger in ledgers
    )
    total_stage_elapsed = sum(elapsed_by_stage.values())
    mean_elapsed = total_stage_elapsed / finished if finished else None
    remaining = len(requested) - finished
    rejection_reason_counts: dict[str, int] = {}
    for ledger in ledgers:
        details = ledger.get("failure_details", {})
        reasons = details.get("rejection_reasons", []) if isinstance(details, dict) else []
        if isinstance(reasons, list):
            for reason in reasons:
                label = str(reason)
                rejection_reason_counts[label] = rejection_reason_counts.get(label, 0) + 1
    summary = {
        "schema_version": BATCH_SCHEMA_VERSION,
        **batch_identity,
        "requested_stop_after": args.stop_after,
        "updated_utc": _utc_now(),
        "counts": counts,
        "requested_episode_count": len(requested),
        "finished_episode_count": finished,
        "remaining_episode_count": remaining,
        "elapsed_by_stage_s": elapsed_by_stage,
        "wall_elapsed_s": time.monotonic() - batch_started,
        "mean_stage_elapsed_per_finished_episode_s": mean_elapsed,
        "estimated_remaining_stage_elapsed_s": (
            None if mean_elapsed is None else mean_elapsed * remaining
        ),
        "rejection_reason_counts": rejection_reason_counts,
        "merged_annotations": merged,
        "exit_code": exit_code,
    }
    atomic_write_json(batch_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
