"""Select baseline or auxiliary progress using validation data only.

The immutable test split is never used for selection. Offline validation first
checks imitation quality; same-seed simulator validation then decides which
candidate may proceed to the one-time test and release evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from model.subtask_policy_training.gr00t.n17_contract import (
    SIM_FIXED_DR_PROFILE,
    SIM_UNSEEN_DR_PROFILE,
    SIM_VALIDATION_DR_PROFILE,
    expected_sim_candidate_selection,
)


SIM_COMPARISON_SCHEMA = "groot_n17_sim_candidate_comparison_v1"
SIM_RELEASE_SCHEMA = "team_ramen_groot_n17_release_evaluation/v1"
OFFLINE_EVALUATION_SCHEMA = "groot_n17_offline_chunk_reset_v2"
SELECTION_DATA = "offline_validation_plus_same_seed_sim_validation"
SIM_VALIDATION_DATA = "same_seed_randomized_sim_validation"
VALIDATION_EPISODES = list(range(139, 156))
SIM_VALIDATION_SEED = 95001
SIM_VALIDATION_EPISODE_IDS = [
    f"{SIM_VALIDATION_SEED}:{index}" for index in range(5)
]


def read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != OFFLINE_EVALUATION_SCHEMA:
        raise ValueError(f"unexpected offline evaluation schema: {path}")
    if report.get("evaluation_type") != "offline_chunk_reset_not_closed_loop":
        raise ValueError(f"unexpected evaluation report type: {path}")
    return report


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_score(report: dict[str, Any]) -> float:
    metrics = report["aggregate"]
    arm_scale = max(float(metrics["target_arm_range_rad"]), 0.05)
    dex_scale = max(float(metrics["target_dex1_range"]), 0.25)
    arm = float(metrics["physical_arm_rmse_rad"]) / arm_scale
    dex = float(metrics["dex1_mae"]) / dex_scale
    stationary = float(metrics["stationary_frame_fraction"])
    return arm + 0.5 * dex + 0.25 * stationary


def _validate_offline_pair(
    baseline: dict[str, Any],
    auxiliary: dict[str, Any],
) -> tuple[float, float]:
    if baseline["episodes"] != auxiliary["episodes"]:
        raise ValueError(
            "candidate reports must cover exactly the same validation episodes"
        )
    if baseline.get("declared_split") != "validation" or auxiliary.get(
        "declared_split"
    ) != "validation":
        raise ValueError(
            "candidate selection may only use the declared validation split"
        )
    if baseline["contract"] != auxiliary["contract"]:
        raise ValueError("candidate reports use different GR00T contracts")
    baseline_randomness = baseline.get("randomness")
    auxiliary_randomness = auxiliary.get("randomness")
    if (
        not isinstance(baseline_randomness, dict)
        or baseline_randomness != auxiliary_randomness
        or int(baseline_randomness.get("base_seed", -1)) < 0
        or int(baseline_randomness.get("episode_stride", 0)) <= 0
        or int(baseline_randomness.get("uint32_modulus", 0)) != 2**32
    ):
        raise ValueError(
            "candidate reports must use the same deterministic offline inference seeds"
        )
    return validation_score(baseline), validation_score(auxiliary)


def _validate_sim_comparison(
    comparison: dict[str, Any],
    *,
    candidate_hashes: dict[str, str],
) -> str:
    if comparison.get("schema_version") != SIM_COMPARISON_SCHEMA:
        raise ValueError("unexpected simulator candidate-comparison schema")
    candidates = comparison.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(candidate_hashes):
        raise ValueError("simulator comparison must contain baseline and auxiliary")
    shared_seed = int(comparison.get("seed", -1))
    shared_policy_seed = int(comparison.get("policy_inference_seed", -1))
    shared_episode_ids = tuple(comparison.get("episode_ids") or ())
    if (
        shared_seed != SIM_VALIDATION_SEED
        or shared_policy_seed != SIM_VALIDATION_SEED
    ):
        raise ValueError("simulator comparison changed the policy inference seed")
    if list(shared_episode_ids) != SIM_VALIDATION_EPISODE_IDS:
        raise ValueError(
            "simulator comparison must declare the exact reserved five episode IDs"
        )
    if comparison.get("test_split_used") is not False:
        raise ValueError("simulator candidate comparison must not use the test split")
    if comparison.get("selection_data") != SIM_VALIDATION_DATA:
        raise ValueError("simulator comparison must be validation-only")
    if (
        comparison.get("domain_randomization_profile")
        != SIM_VALIDATION_DR_PROFILE
    ):
        raise ValueError("simulator comparison changed the validation DR profile")
    if comparison.get("offline_validation_episodes") != VALIDATION_EPISODES:
        raise ValueError("simulator comparison changed the validation split")
    seeds = set()
    episode_sets = set()
    for name, expected_hash in candidate_hashes.items():
        candidate = candidates[name]
        if candidate.get("model_safetensors_sha256") != expected_hash:
            raise ValueError(f"simulator comparison model hash changed for {name}")
        if int(candidate.get("test_count", -1)) != 5:
            raise ValueError(f"simulator comparison must use five episodes for {name}")
        success_count = int(candidate.get("success_count", -1))
        if not 0 <= success_count <= 5:
            raise ValueError(f"invalid simulator success count for {name}")
        candidate_seed = int(candidate.get("seed", -1))
        candidate_policy_seed = int(candidate.get("policy_inference_seed", -1))
        candidate_episode_ids = tuple(candidate.get("episode_ids") or ())
        candidate_episode_seeds = [
            int(value) for value in candidate.get("episode_inference_seeds") or ()
        ]
        if (
            candidate_seed != shared_seed
            or candidate_policy_seed != shared_policy_seed
            or candidate_episode_seeds
            != [shared_policy_seed + index for index in range(5)]
        ):
            raise ValueError(
                f"simulator comparison used inconsistent inference seeds for {name}"
            )
        if candidate.get("mode") != "randomized_validation":
            raise ValueError(f"simulator comparison used the wrong mode for {name}")
        if (
            candidate.get("domain_randomization_profile")
            != SIM_VALIDATION_DR_PROFILE
            or candidate.get("runtime_evaluation_mode") != "randomized"
        ):
            raise ValueError(
                f"simulator comparison used the wrong DR profile for {name}"
            )
        if candidate.get("offline_validation_episodes") != VALIDATION_EPISODES:
            raise ValueError(
                f"simulator comparison changed the validation split for {name}"
            )
        if candidate.get("progress_enabled") is not (
            name == "auxiliary_progress"
        ):
            raise ValueError(
                f"simulator comparison mislabeled progress mode for {name}"
            )
        if candidate_episode_ids != shared_episode_ids:
            raise ValueError(
                f"simulator candidates were not evaluated on the same episodes: {name}"
            )
        seeds.add(candidate_seed)
        episode_sets.add(candidate_episode_ids)
    if len(seeds) != 1 or next(iter(seeds)) < 0:
        raise ValueError("simulator candidates were not evaluated with the same seed")
    if len(episode_sets) != 1 or not next(iter(episode_sets)):
        raise ValueError(
            "simulator candidates were not evaluated on the same episodes"
        )
    selected = str(comparison.get("selected"))
    if selected not in candidate_hashes:
        raise ValueError("simulator comparison selected an unknown candidate")
    if selected != expected_sim_candidate_selection(comparison):
        raise ValueError(
            "simulator comparison selected a candidate inconsistent with its metrics"
        )
    return selected


def _validate_release_stage(
    stage: dict[str, Any],
    *,
    name: str,
    expected_count: int,
    expected_seed: int,
    expected_mode: str,
    expected_temporal_lambda: str,
    expected_execution_steps: int,
) -> None:
    if int(stage.get("test_count", -1)) != expected_count:
        raise ValueError(f"{name} evaluation used the wrong episode count")
    if int(stage.get("seed", -1)) != expected_seed or int(
        stage.get("policy_inference_seed", -1)
    ) != expected_seed:
        raise ValueError(f"{name} evaluation changed its inference seed")
    actual_episode_seeds = [
        int(value) for value in stage.get("episode_inference_seeds") or ()
    ]
    expected_episode_seeds = [
        expected_seed + index for index in range(expected_count)
    ]
    expected_episode_ids = [
        f"{expected_seed}:{index}" for index in range(expected_count)
    ]
    if actual_episode_seeds != expected_episode_seeds:
        raise ValueError(f"{name} evaluation has incomplete inference-seed evidence")
    if stage.get("episode_ids") != expected_episode_ids:
        raise ValueError(f"{name} evaluation has incomplete episode-ID evidence")
    if stage.get("mode") != expected_mode:
        raise ValueError(f"{name} evaluation used the wrong mode")
    if (
        str(stage.get("temporal_lambda")) != expected_temporal_lambda
        or int(stage.get("execution_steps", -1)) != expected_execution_steps
    ):
        raise ValueError(f"{name} evaluation changed the selected temporal setting")


def _validate_release(
    release: dict[str, Any],
    *,
    selected: str,
    selected_hash: str,
) -> None:
    if release.get("schema_version") != SIM_RELEASE_SCHEMA:
        raise ValueError("unexpected simulator release-evaluation schema")
    if release.get("model_safetensors_sha256") != selected_hash:
        raise ValueError("simulator release model differs from the selected candidate")
    if release.get("candidate_name") not in (None, selected):
        raise ValueError("simulator release candidate label changed")
    fixed = release.get("fixed_scene") or {}
    unseen = release.get("unseen_dr") or {}
    selected_temporal = release.get("selected_temporal_setting") or {}
    temporal_lambda = str(selected_temporal.get("temporal_lambda"))
    execution_steps = int(selected_temporal.get("execution_steps", -1))
    if temporal_lambda not in {"none", "-0.25", "-0.1", "0"} or execution_steps not in {
        5,
        10,
        20,
    }:
        raise ValueError("release report selected an invalid temporal setting")
    _validate_release_stage(
        fixed,
        name="fixed-scene",
        expected_count=3,
        expected_seed=93001,
        expected_mode="nominal",
        expected_temporal_lambda=temporal_lambda,
        expected_execution_steps=execution_steps,
    )
    if (
        fixed.get("domain_randomization_profile") != SIM_FIXED_DR_PROFILE
        or fixed.get("runtime_evaluation_mode") != "nominal"
    ):
        raise ValueError("fixed-scene evaluation used the wrong DR profile")
    _validate_release_stage(
        unseen,
        name="unseen-DR",
        expected_count=50,
        expected_seed=94001,
        expected_mode="unseen_dr",
        expected_temporal_lambda=temporal_lambda,
        expected_execution_steps=execution_steps,
    )
    if (
        unseen.get("domain_randomization_profile") != SIM_UNSEEN_DR_PROFILE
        or unseen.get("runtime_evaluation_mode") != "unseen_dr"
    ):
        raise ValueError("unseen-DR evaluation did not use the held-out profile")
    if int(fixed.get("success_count", -1)) != 3:
        raise ValueError("selected candidate did not pass fixed scene 3/3")
    if int(unseen.get("success_count", -1)) < 40:
        raise ValueError("selected candidate did not pass unseen DR 40/50")
    if (release.get("release_goal") or {}).get("unseen_dr_passed") is not True:
        raise ValueError("simulator release report does not declare its DR gate passed")


def select_candidate(
    baseline: dict[str, Any],
    auxiliary: dict[str, Any],
    *,
    sim_comparison: dict[str, Any] | None = None,
    sim_release: dict[str, Any] | None = None,
    candidate_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    baseline_score, auxiliary_score = _validate_offline_pair(
        baseline,
        auxiliary,
    )
    if sim_comparison is None:
        selected = (
            "auxiliary_progress"
            if auxiliary_score < baseline_score
            else "baseline"
        )
        return {
            "schema_version": "groot_n17_candidate_selection_v1",
            "selection_data": "validation_only",
            "episodes": baseline["episodes"],
            "baseline_score": baseline_score,
            "auxiliary_progress_score": auxiliary_score,
            "selected": selected,
            "auxiliary_adopted": selected == "auxiliary_progress",
        }
    if sim_release is None or candidate_hashes is None:
        raise ValueError(
            "simulator comparison, release report, and candidate hashes are required together"
        )
    if set(candidate_hashes) != {"baseline", "auxiliary_progress"}:
        raise ValueError("candidate hashes must identify baseline and auxiliary_progress")
    selected = _validate_sim_comparison(
        sim_comparison,
        candidate_hashes=candidate_hashes,
    )
    _validate_release(
        sim_release,
        selected=selected,
        selected_hash=candidate_hashes[selected],
    )
    return {
        "schema_version": "groot_n17_candidate_selection_v2",
        "selection_data": SELECTION_DATA,
        "episodes": baseline["episodes"],
        "score_definition": (
            "arm_rmse/target_arm_range + 0.5*dex1_mae/target_dex1_range "
            "+ 0.25*stationary_frame_fraction"
        ),
        "baseline_score": baseline_score,
        "auxiliary_progress_score": auxiliary_score,
        "sim_validation": sim_comparison,
        "sim_release": {
            "schema_version": sim_release["schema_version"],
            "model_safetensors_sha256": sim_release[
                "model_safetensors_sha256"
            ],
            "selected_temporal_setting": sim_release[
                "selected_temporal_setting"
            ],
            "fixed_scene": {
                "test_count": sim_release["fixed_scene"]["test_count"],
                "success_count": sim_release["fixed_scene"]["success_count"],
            },
            "unseen_dr": {
                "test_count": sim_release["unseen_dr"]["test_count"],
                "success_count": sim_release["unseen_dr"]["success_count"],
            },
        },
        "candidate_hashes": candidate_hashes,
        "selected": selected,
        "auxiliary_adopted": selected == "auxiliary_progress",
        "reason": (
            "selected by same-seed simulator validation after both candidates "
            "passed the offline validation contract"
        ),
    }


def _candidate_hash(report: dict[str, Any]) -> tuple[Path, str]:
    model_dir = Path(str(report.get("model_dir", ""))).expanduser().resolve()
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"offline validation report does not identify a candidate: {checkpoint}"
        )
    actual_hash = sha256(checkpoint)
    if report.get("model_safetensors_sha256") != actual_hash:
        raise ValueError(
            "offline validation report checkpoint hash differs from its candidate"
        )
    return model_dir, actual_hash


def _write_handoff(
    *,
    path: Path,
    baseline_report: Path,
    auxiliary_report: Path,
    baseline: dict[str, Any],
    auxiliary: dict[str, Any],
) -> None:
    baseline_dir, baseline_hash = _candidate_hash(baseline)
    auxiliary_dir, auxiliary_hash = _candidate_hash(auxiliary)
    payload = {
        "schema_version": "groot_n17_sim_handoff_v1",
        "status": "awaiting_same_seed_sim_candidate_comparison",
        "baseline": {
            "model_dir": str(baseline_dir),
            "model_safetensors_sha256": baseline_hash,
            "validation_report": str(baseline_report.resolve()),
            "validation_score": validation_score(baseline),
        },
        "auxiliary_progress": {
            "model_dir": str(auxiliary_dir),
            "model_safetensors_sha256": auxiliary_hash,
            "validation_report": str(auxiliary_report.resolve()),
            "validation_score": validation_score(auxiliary),
        },
        "required_outputs": [
            "sim_candidate_selection.json",
            "sim_release_evaluation.json",
            "sim_evaluation_bundle/",
        ],
        "test_split_used": False,
        "hf_upload_started": False,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--auxiliary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-report", type=Path)
    parser.add_argument("--sim-release-report", type=Path)
    args = parser.parse_args()

    baseline = read_report(args.baseline_report)
    auxiliary = read_report(args.auxiliary_report)
    _validate_offline_pair(baseline, auxiliary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim_report_path = args.sim_report or (
        args.output.parent / "sim_candidate_selection.json"
    )
    sim_release_path = args.sim_release_report or (
        args.output.parent / "sim_release_evaluation.json"
    )
    if not sim_report_path.is_file() or not sim_release_path.is_file():
        handoff = args.output.parent / "candidate_handoff.json"
        _write_handoff(
            path=handoff,
            baseline_report=args.baseline_report,
            auxiliary_report=args.auxiliary_report,
            baseline=baseline,
            auxiliary=auxiliary,
        )
        print(
            "Both H100 candidates are ready. Same-seed RTX5090 simulator "
            f"comparison is required before test/finalize/upload: {handoff}",
            file=sys.stderr,
        )
        raise SystemExit(75)

    baseline_dir, baseline_hash = _candidate_hash(baseline)
    auxiliary_dir, auxiliary_hash = _candidate_hash(auxiliary)
    del baseline_dir, auxiliary_dir
    result = select_candidate(
        baseline,
        auxiliary,
        sim_comparison=read_json(sim_report_path),
        sim_release=read_json(sim_release_path),
        candidate_hashes={
            "baseline": baseline_hash,
            "auxiliary_progress": auxiliary_hash,
        },
    )
    result["evidence"] = {
        "sim_candidate_comparison_path": str(sim_report_path.resolve()),
        "sim_candidate_comparison_sha256": sha256(sim_report_path),
        "sim_release_evaluation_path": str(sim_release_path.resolve()),
        "sim_release_evaluation_sha256": sha256(sim_release_path),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
