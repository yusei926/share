from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from evaluate.flip_table_simulation.summarize_groot_candidate_comparison import (
    main as summarize_candidate_comparison,
)
from evaluate.flip_table_simulation.summarize_groot_release_evaluation import (
    policy_trace_metrics,
    scripted_tracking_metrics,
    summarize_candidate,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    validate_temporal_selection_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_trace(
    directory: Path,
    rows: list[dict],
    *,
    test_index: int = 0,
) -> None:
    trace = directory / f"test_{test_index}" / "action_state_trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scripted_tracking_gate_checks_motion_and_error(tmp_path: Path) -> None:
    rows = []
    for index in range(30):
        actual = [0.0] * 14
        actual[0] = index / 29 * 0.1
        rows.append(
            {
                "actual_joint_position": [actual],
                "tracking_error": [[0.01] * 14],
            }
        )
    _write_trace(tmp_path, rows)
    result = scripted_tracking_metrics(tmp_path)
    assert result["passed"] is True
    assert result["actual_arm_range_rad"] == 0.1


def test_policy_metrics_use_post_safety_targets(tmp_path: Path) -> None:
    rows = []
    for index in range(5):
        target = [index * 0.01] * 14 + [2.0, 3.0]
        state = [0.0, 0.0, 0.0] + target
        rows.append(
            {
                "action_advanced": True,
                "safe_joint_target_16d": [target],
                "joint_state_after_19d": [state],
            }
        )
    _write_trace(tmp_path, rows)
    result = policy_trace_metrics(tmp_path)
    assert result["action_advances"] == 5
    assert result["tracking_rmse"] == 0.0
    assert result["target_step_rmse"] > 0.0


def test_candidate_summary_requires_declared_manifest(tmp_path: Path) -> None:
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "temporal_lambda": "-0.1",
                "execution_steps": 10,
                "seed": 42,
                "policy_inference_seed": 42,
                "episodes": 5,
                "episode_ids": [f"42:{index}" for index in range(5)],
                "mode": "randomized_validation",
                "domain_randomization_profile": "validation_v1",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval_results.json").write_text(
        json.dumps({"test_count": 5, "success_count": 4}),
        encoding="utf-8",
    )
    for episode_index in range(5):
        _write_trace(
            tmp_path,
            [
                {
                    "policy_inference_seed": 42 + episode_index,
                    "evaluation_mode": "randomized",
                    "domain_randomization_profile": "validation_v1",
                    "action_advanced": True,
                    "safe_joint_target_16d": [[0.0] * 16],
                    "joint_state_after_19d": [[0.0] * 19],
                }
            ],
            test_index=episode_index,
        )
    result = summarize_candidate(tmp_path)
    assert result["success_rate"] == 0.8
    assert result["execution_steps"] == 10


def test_candidate_summary_rejects_incomplete_seed_trace(tmp_path: Path) -> None:
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps(
            {
                "temporal_lambda": "-0.1",
                "execution_steps": 10,
                "seed": 42,
                "policy_inference_seed": 42,
                "episodes": 1,
                "episode_ids": ["42:0"],
                "mode": "nominal",
                "domain_randomization_profile": "nominal_v1",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval_results.json").write_text(
        json.dumps({"test_count": 1, "success_count": 0}),
        encoding="utf-8",
    )
    _write_trace(
        tmp_path,
        [
            {
                "evaluation_mode": "nominal",
                "domain_randomization_profile": "nominal_v1",
                "action_advanced": True,
                "safe_joint_target_16d": [[0.0] * 16],
                "joint_state_after_19d": [[0.0] * 19],
            }
        ],
    )
    with pytest.raises(ValueError, match="inference-seed evidence"):
        summarize_candidate(tmp_path)


def test_release_runner_separates_candidate_and_finalized_validation() -> None:
    runner = (ROOT / "run_groot_release_evaluation.sh").read_text(
        encoding="utf-8"
    )
    assert "validate_checkpoint_metadata" in runner
    assert "validate_furniture_training_candidate" in runner
    assert '"--candidate"' in runner
    assert "--candidate-dir \"$FIXED_DIR\"" in runner
    assert "--candidate-dir \"$DR_DIR\"" in runner
    assert '"temporal_validation": sweep' in runner
    assert '"temporal_validation_sha256"' in runner


def test_candidate_comparison_runs_both_models_with_identical_seed() -> None:
    runner = (ROOT / "run_groot_candidate_comparison.sh").read_text(
        encoding="utf-8"
    )
    assert "for candidate_name in baseline auxiliary_progress" in runner
    assert "SEED=95001" in runner
    assert "EPISODES=5" in runner
    assert 'FLIP_TABLE_GROOT_INFERENCE_SEED="$SEED"' in runner
    assert "FLIP_TABLE_GROOT_TEMPORAL_LAMBDA=-0.1" in runner
    assert "sim_candidate_selection.json" in runner
    assert "sim_release_evaluation.json" in runner
    assert "groot_apply_dr_profile validation_v1" in runner


def test_candidate_comparison_report_selects_sim_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validation_paths = {}
    audit_paths = {}
    for name, successes, model_hash, progress in (
        ("baseline", 5, "base-hash", False),
        ("auxiliary_progress", 4, "aux-hash", True),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "temporal_lambda": "-0.1",
                    "execution_steps": 10,
                    "seed": 95001,
                    "policy_inference_seed": 95001,
                    "episodes": 5,
                    "mode": "randomized_validation",
                    "domain_randomization_profile": "validation_v1",
                    "episode_ids": [
                        f"95001:{index}" for index in range(5)
                    ],
                }
            )
        )
        (directory / "eval_results.json").write_text(
            json.dumps({"test_count": 5, "success_count": successes})
        )
        for episode_index in range(5):
            _write_trace(
                directory,
                [
                    {
                        "policy_inference_seed": 95001 + episode_index,
                        "evaluation_mode": "randomized",
                        "domain_randomization_profile": "validation_v1",
                        "action_advanced": True,
                        "safe_joint_target_16d": [[episode_index * 0.01] * 16],
                        "joint_state_after_19d": [
                            [0.0, 0.0, 0.0] + [episode_index * 0.01] * 16
                        ],
                    }
                ],
                test_index=episode_index,
            )
        audit = tmp_path / f"{name}_audit.json"
        audit.write_text(
            json.dumps(
                {
                    "model_safetensors_sha256": model_hash,
                    "progress_enabled": progress,
                }
            )
        )
        audit_paths[name] = audit
        validation = tmp_path / f"{name}_validation.json"
        validation.write_text(
            json.dumps(
                {
                    "declared_split": "validation",
                    "episodes": list(range(139, 156)),
                    "aggregate": {
                        "physical_arm_rmse_rad": 0.1,
                        "target_arm_range_rad": 1.0,
                        "dex1_mae": 0.2,
                        "target_dex1_range": 4.5,
                        "stationary_frame_fraction": 0.1,
                    },
                }
            )
        )
        validation_paths[name] = validation

    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_groot_candidate_comparison.py",
            "--baseline-dir",
            str(tmp_path / "baseline"),
            "--auxiliary-dir",
            str(tmp_path / "auxiliary_progress"),
            "--baseline-audit",
            str(audit_paths["baseline"]),
            "--auxiliary-audit",
            str(audit_paths["auxiliary_progress"]),
            "--baseline-validation",
            str(validation_paths["baseline"]),
            "--auxiliary-validation",
            str(validation_paths["auxiliary_progress"]),
            "--output",
            str(output),
        ],
    )
    summarize_candidate_comparison()
    result = json.loads(output.read_text())
    assert result["selected"] == "baseline"
    assert result["test_split_used"] is False
    assert result["candidates"]["baseline"]["model_safetensors_sha256"] == (
        "base-hash"
    )


def test_temporal_selection_is_recomputed_from_all_twelve_candidates() -> None:
    candidates = []
    for temporal_lambda in ("none", "-0.25", "-0.1", "0"):
        for execution_steps in (5, 10, 20):
            selected = (temporal_lambda, execution_steps) == ("-0.1", 10)
            success_count = 5 if selected else 4
            candidates.append(
                {
                    "temporal_lambda": temporal_lambda,
                    "execution_steps": execution_steps,
                    "mode": "randomized_validation",
                    "runtime_evaluation_mode": "randomized",
                    "domain_randomization_profile": "validation_v1",
                    "seed": 92001,
                    "policy_inference_seed": 92001,
                    "episode_inference_seeds": [
                        92001 + index for index in range(5)
                    ],
                    "episode_ids": [
                        f"92001:{index}" for index in range(5)
                    ],
                    "test_count": 5,
                    "success_count": success_count,
                    "success_rate": success_count / 5,
                    "trace": {
                        "target_jerk_rms": 0.2,
                        "target_acceleration_rms": 0.1,
                        "tracking_rmse": 0.02,
                    },
                }
            )
    report = {
        "schema_version": "team_ramen_groot_n17_temporal_sweep/v1",
        "scripted_controller_tracking": {
            "arm_rmse_rad": 0.02,
            "arm_p95_abs_error_rad": 0.04,
            "actual_arm_range_rad": 0.2,
            "passed": True,
        },
        "candidates": candidates,
        "selected": {
            "temporal_lambda": "-0.1",
            "execution_steps": 10,
            "success_rate": 1.0,
        },
    }
    assert validate_temporal_selection_report(report) == report["selected"]

    report["selected"]["temporal_lambda"] = "none"
    with pytest.raises(ValueError, match="inconsistent with sweep metrics"):
        validate_temporal_selection_report(report)
