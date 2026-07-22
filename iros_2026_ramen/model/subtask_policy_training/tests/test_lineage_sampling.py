from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from model.subtask_policy_training.lineage_sampling import (
    PLAN_SCHEMA_VERSION,
    LineageBalancedSampler,
    build_sampling_plan,
)


def _record(
    episode_index: int,
    *,
    lineage: str,
    kind: str,
    trajectory_kind: str | None = None,
    appearance_variant: int = 0,
) -> dict:
    value = {
        "episode_index": episode_index,
        "kind": kind,
        "split": "train",
        "source_trajectory_lineage": lineage,
        "appearance_variant": appearance_variant,
    }
    if trajectory_kind is not None:
        value.update(
            {
                "trajectory_kind": trajectory_kind,
                "source_kind": "sim_teleop",
            }
        )
    return value


def test_full_sampling_plan_has_release_ratio() -> None:
    records = {
        0: _record(0, lineage="real:0", kind="real"),
        1: _record(
            1,
            lineage="sim:0",
            kind="synthetic",
            trajectory_kind="direct_sim_teleop",
        ),
        2: _record(2, lineage="mimic:0", kind="synthetic", trajectory_kind="mimic"),
    }
    plan = build_sampling_plan(
        records=records,
        train_episode_indices=[0, 1, 2],
        condition="full",
        split_sha256="a" * 64,
    )
    assert plan["category_weights"] == {
        "real": 0.5,
        "direct_sim_teleop": 0.1,
        "mimic": 0.4,
    }
    assert plan["eligible_episode_count"] == 3


def test_sampler_balances_category_lineage_and_variants(tmp_path: Path) -> None:
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "condition": "real_sim_teleop_mimic",
        "category_weights": {
            "real": 0.5,
            "direct_sim_teleop": 0.1,
            "mimic": 0.4,
        },
        "episodes": [
            {"episode_index": 0, "category": "real", "lineage": "real:0"},
            {"episode_index": 1, "category": "direct_sim_teleop", "lineage": "sim:0"},
            {
                "episode_index": 2,
                "category": "mimic",
                "lineage": "mimic:a",
                "appearance_variant": 0,
            },
            {
                "episode_index": 3,
                "category": "mimic",
                "lineage": "mimic:a",
                "appearance_variant": 1,
            },
            {
                "episode_index": 4,
                "category": "mimic",
                "lineage": "mimic:b",
                "appearance_variant": 0,
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    sampler = LineageBalancedSampler(
        [0, 100, 200, 300, 400],
        [100, 200, 300, 400, 500],
        episode_indices_to_use=[0, 1, 2, 3, 4],
        shuffle=True,
        seed=7,
        plan_path=path,
    )

    first = list(sampler._iter_epoch(0, 0))
    second = list(sampler._iter_epoch(0, 0))
    assert first == second
    episode_counts = Counter(value // 100 for value in first)
    assert episode_counts[0] == 250
    assert episode_counts[1] == 50
    assert episode_counts[2] == 50
    assert episode_counts[3] == 50
    assert episode_counts[4] == 100


def test_sampler_resume_offset_is_sample_exact(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "condition": "real_only",
                "category_weights": {"real": 1.0},
                "episodes": [
                    {"episode_index": 0, "category": "real", "lineage": "real:0"}
                ],
            }
        ),
        encoding="utf-8",
    )
    sampler = LineageBalancedSampler([0], [20], shuffle=True, seed=11, plan_path=path)
    complete = list(sampler._iter_epoch(3, 0))
    assert list(sampler._iter_epoch(3, 7)) == complete[7:]
    with pytest.raises(ValueError, match="resume offset"):
        list(sampler._iter_epoch(0, 21))
