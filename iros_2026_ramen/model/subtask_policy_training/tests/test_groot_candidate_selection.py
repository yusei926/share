from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from model.subtask_policy_training.gr00t.n17_contract import (
    valid_sim_candidate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def load_selector():
    path = ROOT / "scripts" / "select_groot_n17_candidate.py"
    spec = importlib.util.spec_from_file_location("select_groot_n17_candidate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(*, arm: float, dex: float, stationary: float) -> dict:
    return {
        "schema_version": "groot_n17_offline_chunk_reset_v2",
        "evaluation_type": "offline_chunk_reset_not_closed_loop",
        "declared_split": "validation",
        "episodes": [139, 140],
        "contract": {"logical_action_dim": 53},
        "randomness": {
            "base_seed": 42,
            "episode_stride": 1_000_003,
            "uint32_modulus": 2**32,
        },
        "aggregate": {
            "physical_arm_rmse_rad": arm,
            "target_arm_range_rad": 1.0,
            "dex1_mae": dex,
            "target_dex1_range": 4.5,
            "stationary_frame_fraction": stationary,
        },
    }


def test_auxiliary_is_adopted_only_when_validation_score_improves() -> None:
    selector = load_selector()
    result = selector.select_candidate(
        report(arm=0.2, dex=0.4, stationary=0.2),
        report(arm=0.1, dex=0.2, stationary=0.1),
    )
    assert result["selected"] == "auxiliary_progress"
    assert result["auxiliary_adopted"] is True

    tie = selector.select_candidate(
        report(arm=0.1, dex=0.2, stationary=0.1),
        report(arm=0.1, dex=0.2, stationary=0.1),
    )
    assert tie["selected"] == "baseline"
    assert tie["auxiliary_adopted"] is False


def test_candidate_report_is_bound_to_checkpoint_hash(tmp_path: Path) -> None:
    selector = load_selector()
    model_dir = tmp_path / "candidate"
    model_dir.mkdir()
    checkpoint = model_dir / "model.safetensors"
    checkpoint.write_bytes(b"candidate")
    report_value = {
        "model_dir": str(model_dir),
        "model_safetensors_sha256": hashlib.sha256(b"candidate").hexdigest(),
    }

    assert selector._candidate_hash(report_value) == (
        model_dir.resolve(),
        report_value["model_safetensors_sha256"],
    )
    report_value["model_safetensors_sha256"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="checkpoint hash"):
        selector._candidate_hash(report_value)


def test_same_seed_sim_validation_controls_release_selection() -> None:
    selector = load_selector()
    hashes = {"baseline": "base-hash", "auxiliary_progress": "aux-hash"}
    comparison = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_ids": [f"95001:{index}" for index in range(5)],
        "test_split_used": False,
        "selection_data": "same_seed_randomized_sim_validation",
        "domain_randomization_profile": "validation_v1",
        "offline_validation_episodes": list(range(139, 156)),
        "selected": "baseline",
        "candidates": {
            "baseline": {
                "model_safetensors_sha256": "base-hash",
                "test_count": 5,
                "success_count": 5,
                "success_rate": 1.0,
                "seed": 95001,
                "policy_inference_seed": 95001,
                "episode_inference_seeds": [
                    95001 + index for index in range(5)
                ],
                "mode": "randomized_validation",
                "runtime_evaluation_mode": "randomized",
                "domain_randomization_profile": "validation_v1",
                "episode_ids": [f"95001:{index}" for index in range(5)],
                "offline_validation_episodes": list(range(139, 156)),
                "offline_validation_score": 0.3,
                "trace": {
                    "target_jerk_rms": 0.4,
                    "target_acceleration_rms": 0.3,
                    "tracking_rmse": 0.02,
                },
                "progress_enabled": False,
            },
            "auxiliary_progress": {
                "model_safetensors_sha256": "aux-hash",
                "test_count": 5,
                "success_count": 4,
                "success_rate": 0.8,
                "seed": 95001,
                "policy_inference_seed": 95001,
                "episode_inference_seeds": [
                    95001 + index for index in range(5)
                ],
                "mode": "randomized_validation",
                "runtime_evaluation_mode": "randomized",
                "domain_randomization_profile": "validation_v1",
                "episode_ids": [f"95001:{index}" for index in range(5)],
                "offline_validation_episodes": list(range(139, 156)),
                "offline_validation_score": 0.1,
                "trace": {
                    "target_jerk_rms": 0.2,
                    "target_acceleration_rms": 0.2,
                    "tracking_rmse": 0.01,
                },
                "progress_enabled": True,
            },
        },
    }
    release = {
        "schema_version": "team_ramen_groot_n17_release_evaluation/v1",
        "candidate_name": "baseline",
        "model_safetensors_sha256": "base-hash",
        "selected_temporal_setting": {
            "temporal_lambda": "-0.1",
            "execution_steps": 10,
        },
        "fixed_scene": {
            "test_count": 3,
            "success_count": 3,
            "seed": 93001,
            "policy_inference_seed": 93001,
            "episode_inference_seeds": [93001, 93002, 93003],
            "episode_ids": [f"93001:{index}" for index in range(3)],
            "mode": "nominal",
            "runtime_evaluation_mode": "nominal",
            "domain_randomization_profile": "nominal_v1",
            "temporal_lambda": "-0.1",
            "execution_steps": 10,
        },
        "unseen_dr": {
            "test_count": 50,
            "success_count": 42,
            "seed": 94001,
            "policy_inference_seed": 94001,
            "episode_inference_seeds": [
                94001 + index for index in range(50)
            ],
            "episode_ids": [f"94001:{index}" for index in range(50)],
            "mode": "unseen_dr",
            "runtime_evaluation_mode": "unseen_dr",
            "domain_randomization_profile": "held_out_v1",
            "temporal_lambda": "-0.1",
            "execution_steps": 10,
        },
        "release_goal": {"unseen_dr_passed": True},
    }
    result = selector.select_candidate(
        report(arm=0.2, dex=0.4, stationary=0.2),
        report(arm=0.1, dex=0.2, stationary=0.1),
        sim_comparison=comparison,
        sim_release=release,
        candidate_hashes=hashes,
    )
    assert result["selection_data"] == (
        "offline_validation_plus_same_seed_sim_validation"
    )
    assert result["selected"] == "baseline"
    assert result["auxiliary_adopted"] is False
    assert valid_sim_candidate_evidence(comparison, candidate_hashes=hashes)
    assert not valid_sim_candidate_evidence(
        comparison,
        candidate_hashes={"baseline": "base-hash"},
    )
    comparison["candidates"]["auxiliary_progress"][
        "domain_randomization_profile"
    ] = "held_out_v1"
    assert not valid_sim_candidate_evidence(
        comparison,
        candidate_hashes=hashes,
    )
    comparison["candidates"]["auxiliary_progress"][
        "domain_randomization_profile"
    ] = "validation_v1"

    release["unseen_dr"]["domain_randomization_profile"] = "validation_v1"
    with pytest.raises(ValueError, match="held-out profile"):
        selector._validate_release(
            release,
            selected="baseline",
            selected_hash="base-hash",
        )


def test_sim_selection_rejects_selected_name_tampering() -> None:
    selector = load_selector()
    episode_ids = [f"95001:{index}" for index in range(5)]
    common = {
        "test_count": 5,
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_inference_seeds": [95001 + index for index in range(5)],
        "mode": "randomized_validation",
        "runtime_evaluation_mode": "randomized",
        "domain_randomization_profile": "validation_v1",
        "episode_ids": episode_ids,
        "offline_validation_episodes": list(range(139, 156)),
        "trace": {
            "target_jerk_rms": 0.2,
            "target_acceleration_rms": 0.1,
            "tracking_rmse": 0.02,
        },
        "offline_validation_score": 0.2,
    }
    comparison = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_ids": episode_ids,
        "test_split_used": False,
        "selection_data": "same_seed_randomized_sim_validation",
        "domain_randomization_profile": "validation_v1",
        "offline_validation_episodes": list(range(139, 156)),
        "selected": "auxiliary_progress",
        "candidates": {
            "baseline": {
                **common,
                "model_safetensors_sha256": "base",
                "success_count": 5,
                "success_rate": 1.0,
                "progress_enabled": False,
            },
            "auxiliary_progress": {
                **common,
                "model_safetensors_sha256": "aux",
                "success_count": 4,
                "success_rate": 0.8,
                "progress_enabled": True,
            },
        },
    }
    hashes = {"baseline": "base", "auxiliary_progress": "aux"}
    with pytest.raises(ValueError, match="inconsistent with its metrics"):
        selector._validate_sim_comparison(
            comparison,
            candidate_hashes=hashes,
        )
    assert not valid_sim_candidate_evidence(
        comparison,
        candidate_hashes=hashes,
    )


def test_sim_selection_rejects_different_episode_sets() -> None:
    selector = load_selector()
    episode_ids = [f"95001:{index}" for index in range(5)]
    comparison = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_ids": episode_ids,
        "test_split_used": False,
        "selection_data": "same_seed_randomized_sim_validation",
        "domain_randomization_profile": "validation_v1",
        "offline_validation_episodes": list(range(139, 156)),
        "selected": "baseline",
        "candidates": {
            "baseline": {
                "model_safetensors_sha256": "base",
                "test_count": 5,
                "success_count": 5,
                "seed": 95001,
                "policy_inference_seed": 95001,
                "episode_inference_seeds": [95001, 95002, 95003, 95004, 95005],
                "mode": "randomized_validation",
                "runtime_evaluation_mode": "randomized",
                "domain_randomization_profile": "validation_v1",
                "episode_ids": episode_ids,
                "offline_validation_episodes": list(range(139, 156)),
                "progress_enabled": False,
            },
            "auxiliary_progress": {
                "model_safetensors_sha256": "aux",
                "test_count": 5,
                "success_count": 5,
                "seed": 95001,
                "policy_inference_seed": 95001,
                "episode_inference_seeds": [95001, 95002, 95003, 95004, 95005],
                "mode": "randomized_validation",
                "runtime_evaluation_mode": "randomized",
                "domain_randomization_profile": "validation_v1",
                "episode_ids": list(reversed(episode_ids)),
                "offline_validation_episodes": list(range(139, 156)),
                "progress_enabled": True,
            },
        },
    }
    with pytest.raises(ValueError, match="same episodes"):
        selector.select_candidate(
            report(arm=0.1, dex=0.1, stationary=0.1),
            report(arm=0.1, dex=0.1, stationary=0.1),
            sim_comparison=comparison,
            sim_release={},
            candidate_hashes={"baseline": "base", "auxiliary_progress": "aux"},
        )


def test_sim_selection_rejects_changed_policy_inference_seed() -> None:
    selector = load_selector()
    episode_ids = [f"95001:{index}" for index in range(5)]
    candidate = {
        "test_count": 5,
        "success_count": 5,
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_inference_seeds": [
            95001 + index for index in range(5)
        ],
        "mode": "randomized_validation",
        "runtime_evaluation_mode": "randomized",
        "domain_randomization_profile": "validation_v1",
        "episode_ids": episode_ids,
        "offline_validation_episodes": list(range(139, 156)),
    }
    comparison = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_ids": episode_ids,
        "test_split_used": False,
        "selection_data": "same_seed_randomized_sim_validation",
        "domain_randomization_profile": "validation_v1",
        "offline_validation_episodes": list(range(139, 156)),
        "selected": "baseline",
        "candidates": {
            "baseline": {
                **candidate,
                "model_safetensors_sha256": "base",
                "progress_enabled": False,
            },
            "auxiliary_progress": {
                **candidate,
                "model_safetensors_sha256": "aux",
                "episode_inference_seeds": [95002, 95003, 95004, 95005, 95006],
                "progress_enabled": True,
            },
        },
    }
    with pytest.raises(ValueError, match="inconsistent inference seeds"):
        selector.select_candidate(
            report(arm=0.1, dex=0.1, stationary=0.1),
            report(arm=0.1, dex=0.1, stationary=0.1),
            sim_comparison=comparison,
            sim_release={},
            candidate_hashes={"baseline": "base", "auxiliary_progress": "aux"},
        )


def test_sim_selection_rejects_split_or_progress_label_drift() -> None:
    selector = load_selector()
    episode_ids = [f"95001:{index}" for index in range(5)]
    candidate = {
        "test_count": 5,
        "success_count": 5,
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_inference_seeds": [95001 + index for index in range(5)],
        "mode": "randomized_validation",
        "runtime_evaluation_mode": "randomized",
        "domain_randomization_profile": "validation_v1",
        "episode_ids": episode_ids,
        "offline_validation_episodes": list(range(139, 156)),
    }
    comparison = {
        "schema_version": "groot_n17_sim_candidate_comparison_v1",
        "seed": 95001,
        "policy_inference_seed": 95001,
        "episode_ids": episode_ids,
        "test_split_used": False,
        "selection_data": "same_seed_randomized_sim_validation",
        "domain_randomization_profile": "validation_v1",
        "offline_validation_episodes": list(range(139, 156)),
        "selected": "baseline",
        "candidates": {
            "baseline": {
                **candidate,
                "model_safetensors_sha256": "base",
                "progress_enabled": False,
            },
            "auxiliary_progress": {
                **candidate,
                "model_safetensors_sha256": "aux",
                "progress_enabled": True,
            },
        },
    }

    comparison["offline_validation_episodes"] = list(range(140, 157))
    with pytest.raises(ValueError, match="validation split"):
        selector._validate_sim_comparison(
            comparison,
            candidate_hashes={"baseline": "base", "auxiliary_progress": "aux"},
        )

    comparison["offline_validation_episodes"] = list(range(139, 156))
    comparison["candidates"]["baseline"]["progress_enabled"] = True
    with pytest.raises(ValueError, match="progress mode"):
        selector._validate_sim_comparison(
            comparison,
            candidate_hashes={"baseline": "base", "auxiliary_progress": "aux"},
        )
