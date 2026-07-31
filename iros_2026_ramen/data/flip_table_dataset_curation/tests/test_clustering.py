from __future__ import annotations

import numpy as np

from flip_table_curation.clustering import (
    cluster_descriptors,
    cluster_orientation_descriptors,
)


def test_dominant_pattern_is_order_invariant() -> None:
    rng = np.random.default_rng(4)
    dominant = rng.normal(0.0, 0.05, size=(60, 12))
    secondary = rng.normal(3.0, 0.05, size=(30, 12))
    values = np.r_[dominant, secondary]
    first = cluster_descriptors(
        values, min_cluster_size=15, min_samples=5, seed=42
    )
    permutation = rng.permutation(len(values))
    second = cluster_descriptors(
        values[permutation], min_cluster_size=15, min_samples=5, seed=42
    )
    first_members = set(np.flatnonzero(first.labels == first.largest_cluster))
    second_members = set(
        permutation[np.flatnonzero(second.labels == second.largest_cluster)]
    )
    assert first_members == second_members == set(range(60))


def test_bootstrap_stability_marks_dominant_core() -> None:
    rng = np.random.default_rng(8)
    values = np.r_[
        rng.normal(0.0, 0.03, size=(70, 8)),
        rng.normal(2.0, 0.03, size=(30, 8)),
    ]
    result = cluster_descriptors(
        values,
        min_cluster_size=15,
        min_samples=5,
        seed=42,
        stability_runs=10,
        stability_fraction=0.9,
    )
    core = result.labels == result.largest_cluster
    assert np.mean(result.stability[core]) > 0.8


def test_continuous_single_procedure_does_not_collapse_to_minimum_size() -> None:
    rng = np.random.default_rng(18)
    phase = np.linspace(0.0, 1.0, 160)
    values = np.stack(
        [
            phase,
            phase**2,
            np.sin(np.pi * phase),
            np.cos(np.pi * phase),
        ],
        axis=1,
    )
    values += rng.normal(0.0, 0.015, size=values.shape)
    result = cluster_descriptors(
        values,
        min_cluster_size=20,
        min_samples=10,
        seed=42,
    )
    assert result.largest_cluster is not None
    assert result.cluster_sizes[result.largest_cluster] >= 100


def test_orientation_can_select_one_continuous_mode_without_noise() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 0.1, size=(80, 10))
    result = cluster_orientation_descriptors(values, seed=42)
    assert np.all(result.labels >= 0)
    assert sum(result.cluster_sizes.values()) == 80
