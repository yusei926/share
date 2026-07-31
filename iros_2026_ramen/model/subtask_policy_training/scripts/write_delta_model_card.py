"""Write the provenance and offline evaluation record shipped with a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--training-run-record", type=Path)
    parser.add_argument("--wandb-url", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    marker = read_json(args.training_view / "meta" / "team_ramen_training_view.json")
    split = read_json(args.training_view / "meta" / "team_ramen_episode_split.json")
    evaluation = read_json(args.evaluation_report)
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = {
        "dataset": {
            "repo_id": marker["source_repo_id"],
            "revision": marker.get("source_revision"),
            "source_fingerprint_sha256": marker["source_fingerprint_sha256"],
            "action_representation": marker["action_representation"],
            "action_semantics": marker["action_semantics"],
            "episode_split_sha256": split["sha256"],
            "heldout_episode_indices": split["splits"]["test"]["episode_indices"],
            "training_episode_count": len(split["splits"]["train"]["episode_indices"]),
        },
        "checkpoint": {"model_safetensors_sha256": sha256(checkpoint)},
        "evaluation": evaluation,
        "wandb_url": args.wandb_url or None,
    }
    if args.training_run_record is not None:
        run_record = args.training_run_record.resolve()
        if not run_record.is_file():
            raise FileNotFoundError(run_record)
        destination = model_dir / "training_run_record.json"
        if run_record != destination.resolve():
            shutil.copyfile(run_record, destination)
        manifest["training_run_record"] = "training_run_record.json"
    (model_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Keep the machine-readable report as a first-class Hub artifact rather
    # than requiring consumers to extract it from the provenance manifest.
    shutil.copyfile(args.evaluation_report, model_dir / "evaluation_report.json")
    model_type = read_json(model_dir / "config.json").get("type", "policy")
    heldout = ", ".join(str(value) for value in manifest["dataset"]["heldout_episode_indices"])
    card = f"""---
tags:
- lerobot
- flip-table
- {model_type}
private: true
---

# Flip-table {model_type}

This policy was trained only on the immutable real-data view of
`{manifest['dataset']['repo_id']}`. The source view stores executable absolute
targets. This model predicts arm targets relative to the measured arm state at
each chunk start, while Dex1 gripper commands remain absolute.

## Contract

- Inputs: head-left RGB, left/right D405 RGB, 19D upper-body state.
- Outputs: 14D chunk-start-relative arm target and 2D absolute gripper command.
- Deployment: add the measured arm state at the chunk start exactly once; do
  not integrate from an earlier model action.
- Held-out episodes: `{heldout}`. They were excluded from training, augmentation,
  normalization statistics, and checkpoint selection.
- Evaluation: offline chunk-reset only. Every 16-frame chunk starts from the recorded
  state, therefore the metrics are not a closed-loop, simulator, or real-robot success rate.

## Provenance

- Source fingerprint: `{manifest['dataset']['source_fingerprint_sha256']}`
- Source revision: `{manifest['dataset']['revision']}`
- Split SHA-256: `{manifest['dataset']['episode_split_sha256']}`
- Checkpoint SHA-256: `{manifest['checkpoint']['model_safetensors_sha256']}`
- W&B: `{args.wandb_url or 'recorded in training artifacts'}`

`training_manifest.json` contains the exact training-view contract and offline evaluation report.
`training_run_record.json` records source revision, hardware, metric digest, and checkpoint digest.
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
