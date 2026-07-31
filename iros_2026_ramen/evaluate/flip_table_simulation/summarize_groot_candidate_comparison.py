"""Select baseline or auxiliary progress from same-seed simulator validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluate.flip_table_simulation.summarize_groot_release_evaluation import (
    summarize_candidate,
)
from model.subtask_policy_training.scripts.select_groot_n17_candidate import (
    SIM_VALIDATION_EPISODE_IDS,
    SIM_VALIDATION_SEED,
    VALIDATION_EPISODES,
    validation_score,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    SIM_VALIDATION_DR_PROFILE,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-dir", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--auxiliary-audit", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--auxiliary-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    trace = candidate["trace"]
    tracking = trace["tracking_rmse"]
    return (
        -float(candidate["success_rate"]),
        float(trace["target_jerk_rms"]),
        float(trace["target_acceleration_rms"]),
        math.inf if tracking is None else float(tracking),
        float(candidate["offline_validation_score"]),
        0.0 if candidate["candidate_name"] == "baseline" else 1.0,
    )


def main() -> None:
    args = parse_args()
    definitions = {
        "baseline": (
            args.baseline_dir,
            args.baseline_audit,
            args.baseline_validation,
        ),
        "auxiliary_progress": (
            args.auxiliary_dir,
            args.auxiliary_audit,
            args.auxiliary_validation,
        ),
    }
    candidates: dict[str, dict[str, Any]] = {}
    shared_seed: int | None = None
    shared_policy_seed: int | None = None
    shared_episode_ids: list[str] | None = None
    shared_validation_episodes: list[int] | None = None
    for name, (directory, audit_path, validation_path) in definitions.items():
        candidate = summarize_candidate(directory)
        manifest = read_json(directory / "candidate_manifest.json")
        audit = read_json(audit_path)
        validation = read_json(validation_path)
        if validation.get("declared_split") != "validation":
            raise ValueError(f"{name} offline report is not validation-only")
        seed = int(manifest["seed"])
        policy_seed = int(manifest["policy_inference_seed"])
        episode_ids = [str(value) for value in manifest["episode_ids"]]
        if (
            seed != SIM_VALIDATION_SEED
            or policy_seed != SIM_VALIDATION_SEED
            or episode_ids != SIM_VALIDATION_EPISODE_IDS
        ):
            raise ValueError(
                f"{name} did not use the reserved same-seed simulator validation episodes"
            )
        validation_episodes = [
            int(value) for value in validation.get("episodes") or ()
        ]
        if validation_episodes != VALIDATION_EPISODES:
            raise ValueError(
                f"{name} offline report changed the immutable validation split"
            )
        if shared_seed is None:
            shared_seed = seed
            shared_policy_seed = policy_seed
            shared_episode_ids = episode_ids
            shared_validation_episodes = validation_episodes
        elif (
            seed != shared_seed
            or policy_seed != shared_policy_seed
            or episode_ids != shared_episode_ids
            or validation_episodes != shared_validation_episodes
        ):
            raise ValueError(
                "baseline and auxiliary candidates did not use identical validation data"
            )
        candidate.update(
            {
                "candidate_name": name,
                "seed": seed,
                "policy_inference_seed": policy_seed,
                "episode_ids": episode_ids,
                "model_safetensors_sha256": audit[
                    "model_safetensors_sha256"
                ],
                "progress_enabled": bool(audit["progress_enabled"]),
                "offline_validation_score": validation_score(validation),
                "offline_validation_episodes": validation_episodes,
            }
        )
        if candidate["test_count"] != len(episode_ids):
            raise ValueError(f"{name} simulator episode count changed")
        if candidate["domain_randomization_profile"] != SIM_VALIDATION_DR_PROFILE:
            raise ValueError(f"{name} used a non-validation DR profile")
        if candidate["progress_enabled"] is not (
            name == "auxiliary_progress"
        ):
            raise ValueError(f"{name} progress-head mode changed")
        candidates[name] = candidate

    selected = min(candidates.values(), key=_selection_key)
    payload = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "selection_data": "same_seed_randomized_sim_validation",
        "seed": shared_seed,
        "policy_inference_seed": shared_policy_seed,
        "episode_ids": shared_episode_ids,
        "offline_validation_episodes": shared_validation_episodes,
        "domain_randomization_profile": SIM_VALIDATION_DR_PROFILE,
        "selection_basis": [
            "sim_success_rate_descending",
            "post_safety_target_jerk_ascending",
            "post_safety_target_acceleration_ascending",
            "tracking_rmse_ascending",
            "offline_validation_score_ascending",
            "baseline_on_exact_tie",
        ],
        "candidates": candidates,
        "selected": selected["candidate_name"],
        "test_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": payload["selected"]}, indent=2))


if __name__ == "__main__":
    main()
