"""Capture reproducibility metadata for a completed native-policy run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def run(command: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--action-contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.training_output.resolve()
    repo_root = args.repository_root.resolve()
    checkpoint = model_dir / "model.safetensors"
    summary = read_json(output_dir / "summary.json")
    action_contract = read_json(args.action_contract.resolve())
    metrics_path = output_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    if not metrics:
        raise ValueError(f"empty metrics file: {metrics_path}")
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        cwd=repo_root,
    )
    record = {
        "source": {
            "git_head": run(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "git_status_porcelain": run(["git", "status", "--porcelain"], cwd=repo_root),
        },
        "hardware": {"nvidia_smi_at_completion": gpu},
        "action_contract": action_contract,
        "training": {
            "summary": summary,
            "metrics_sha256": sha256(metrics_path),
            "first_logged_metric": metrics[0],
            "last_logged_metric": metrics[-1],
        },
        "checkpoint": {"model_safetensors_sha256": sha256(checkpoint)},
    }
    (model_dir / "training_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
