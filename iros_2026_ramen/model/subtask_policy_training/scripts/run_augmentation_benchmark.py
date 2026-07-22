#!/usr/bin/env python3
"""Prepare or execute the fixed augmented-data comparison matrix.

The three training conditions differ only in the lineage-balanced source
mixture.  Each policy therefore uses one seed and one update budget across
conditions.  This is intentionally a launcher rather than another trainer:
the repository's existing ACT and Flow Matching entrypoints retain ownership
of materialization, checkpointing, and W&B logging.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURE_ROOT = REPO_ROOT / "model" / "subtask_policy_training"
DEFAULT_DATASET_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_augmented_2"
DEFAULT_WANDB_PROJECT = "iros2026-ramen-flip-table"
CONDITIONS = ("real_only", "real_sim_teleop", "real_sim_teleop_mimic")
POLICIES = ("act", "flow_matching")


@dataclass(frozen=True)
class BenchmarkJob:
    policy: str
    condition: str
    command: tuple[str, ...]
    output_dir: Path
    training_view_root: Path
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local clean LeRobot v3 checkout of the augmented dataset.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--act-steps", type=int, default=300_000)
    parser.add_argument("--flow-steps", type=int, default=300_000)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the six training jobs. Without this flag only write the immutable plan.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.act_steps <= 0 or args.flow_steps <= 0:
        raise ValueError("training step counts must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"--dataset-root is not a LeRobot v3 checkout: {dataset_root}")
    if not (dataset_root / "meta" / "augmentation" / "episodes.jsonl").is_file():
        raise FileNotFoundError(
            "--dataset-root lacks lineage metadata required for the comparison: "
            f"{dataset_root / 'meta' / 'augmentation' / 'episodes.jsonl'}"
        )


def build_jobs(
    *,
    output_root: Path,
    seed: int,
    act_steps: int,
    flow_steps: int,
) -> tuple[BenchmarkJob, ...]:
    jobs: list[BenchmarkJob] = []
    for policy in POLICIES:
        steps = act_steps if policy == "act" else flow_steps
        entrypoint = (
            "scripts/train_lerobot.sh"
            if policy == "act"
            else "scripts/train_flow_matching.sh"
        )
        for condition in CONDITIONS:
            base = output_root / policy / condition
            command = ("bash", entrypoint, "--seed", str(seed))
            jobs.append(
                BenchmarkJob(
                    policy=policy,
                    condition=condition,
                    command=command,
                    output_dir=base / "checkpoints",
                    training_view_root=base / "training_view",
                    steps=steps,
                )
            )
    return tuple(jobs)


def _job_environment(
    base: dict[str, str],
    *,
    job: BenchmarkJob,
    args: argparse.Namespace,
    dataset_root: Path,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "SUBTASK": "flip_table",
            "POLICY_TYPE": job.policy,
            "DATASET_REPO_ID": args.dataset_repo_id,
            "SOURCE_DATASET_ROOT": str(dataset_root),
            "DATASET_REVISION": "",
            "TRAINING_CONDITION": job.condition,
            "TRAINING_VIEW_ROOT": str(job.training_view_root),
            "FLOW_TRAINING_VIEW_ROOT": str(job.training_view_root),
            "OUTPUT_DIR": str(job.output_dir),
            "FLOW_OUTPUT_DIR": str(job.output_dir),
            "TRAIN_STEPS": str(job.steps),
            "JOB_NAME": f"flip-table-{job.policy}-{job.condition}-seed{args.seed}",
            "PUSH_TO_HUB": "false",
            "UPLOAD_AFTER_TRAIN": "false",
            "PRIVATE": "true",
            "MATERIALIZE_TRAINING_VIEW": "true",
            "TRAINING_VIEW_FORCE": "false",
            "WANDB_ENABLE": "true",
            "WANDB_PROJECT": args.wandb_project,
        }
    )
    return environment


def _manifest(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    jobs: Iterable[BenchmarkJob],
) -> dict[str, object]:
    return {
        "schema_version": "team_ramen_flip_table_augmented_benchmark/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "conditions": list(CONDITIONS),
        "policies": list(POLICIES),
        "execution_requested": bool(args.execute),
        "jobs": [
            {
                **asdict(job),
                "command": list(job.command),
                "output_dir": str(job.output_dir),
                "training_view_root": str(job.training_view_root),
            }
            for job in jobs
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = build_jobs(
        output_root=output_root,
        seed=args.seed,
        act_steps=args.act_steps,
        flow_steps=args.flow_steps,
    )
    manifest_path = output_root / "benchmark_plan.json"
    manifest = _manifest(args=args, dataset_root=dataset_root, jobs=jobs)
    _write_json(manifest_path, manifest)

    if not args.execute:
        print(f"Wrote benchmark plan without starting training: {manifest_path}")
        return 0

    statuses = []
    for job in jobs:
        environment = _job_environment(
            os.environ,
            job=job,
            args=args,
            dataset_root=dataset_root,
        )
        log_path = job.output_dir.parent / "launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                job.command,
                cwd=FEATURE_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        statuses.append(
            {
                "policy": job.policy,
                "condition": job.condition,
                "returncode": result.returncode,
                "log": str(log_path),
            }
        )
        manifest["execution_status"] = statuses
        _write_json(manifest_path, manifest)
        if result.returncode != 0:
            print(f"benchmark job failed; inspect {log_path}", file=sys.stderr)
            return result.returncode
    print(f"Completed benchmark matrix: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
