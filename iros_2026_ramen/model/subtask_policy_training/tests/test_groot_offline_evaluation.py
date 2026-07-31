from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_evaluator():
    path = ROOT / "scripts" / "evaluate_groot_n17_offline.py"
    spec = importlib.util.spec_from_file_location("evaluate_groot_n17_offline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logical_action_decodes_to_only_arms_and_dex1() -> None:
    evaluator = load_evaluator()
    from model.subtask_policy_training.gr00t.dex1_hand_synergy import dex1_to_hand
    from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
        REAL_G1_RELATIVE_EEF_ACTION_SLICES,
    )

    action = np.zeros(53, dtype=np.float32)
    action[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_arm"])] = np.arange(7)
    action[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_arm"])] = np.arange(7, 14)
    action[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_hand"])] = dex1_to_hand(
        1.25, side="left", kind="action"
    )
    action[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_hand"])] = dex1_to_hand(
        3.75, side="right", kind="action"
    )

    physical = evaluator.action_to_physical(action)

    np.testing.assert_allclose(physical[:14], np.arange(14), atol=1e-6)
    np.testing.assert_allclose(physical[14:], [1.25, 3.75], atol=1e-6)


def test_exact_offline_prediction_has_zero_group_error() -> None:
    evaluator = load_evaluator()
    target = np.zeros((4, 53), dtype=np.float32)
    target[:, 32:46] = np.linspace(0.0, 0.3, 4, dtype=np.float32)[:, None]
    states = np.zeros((4, 49), dtype=np.float32)

    metrics = evaluator.compute_metrics(target.copy(), target, states)

    assert metrics["physical_arm_rmse_rad"] == 0.0
    assert metrics["dex1_mae"] == 0.0
    assert metrics["dex1_open_closed_accuracy"] == 1.0
    assert metrics["left_wrist_eef_9d_mae"] == 0.0
    assert metrics["right_arm_mae"] == 0.0


def test_orientation_group_metrics_do_not_mix_episode_groups() -> None:
    evaluator = load_evaluator()

    def trace(error: float) -> dict[str, np.ndarray]:
        target = np.zeros((3, 53), dtype=np.float32)
        predicted = target.copy()
        predicted[:, 32:46] = error
        return {
            "predicted_action": predicted,
            "target_action": target,
            "state": np.zeros((3, 49), dtype=np.float32),
        }

    grouped = evaluator.compute_orientation_group_metrics(
        {2: trace(0.2), 3: trace(0.0), 4: trace(0.1)},
        {2: "1", 3: "0", 4: "1"},
    )

    assert grouped["0"]["episodes"] == [3]
    assert grouped["0"]["aggregate"]["physical_arm_rmse_rad"] == 0.0
    assert grouped["1"]["episodes"] == [2, 4]
    assert grouped["1"]["aggregate"]["physical_arm_rmse_rad"] > 0.0


def test_offline_chunk_seed_is_independent_of_evaluation_order() -> None:
    evaluator = load_evaluator()
    expected = evaluator.offline_chunk_inference_seed(
        base_seed=42,
        episode_index=150,
        chunk_ordinal=7,
    )
    assert expected == evaluator.offline_chunk_inference_seed(
        base_seed=42,
        episode_index=150,
        chunk_ordinal=7,
    )
    assert expected != evaluator.offline_chunk_inference_seed(
        base_seed=42,
        episode_index=150,
        chunk_ordinal=8,
    )
    assert expected != evaluator.offline_chunk_inference_seed(
        base_seed=42,
        episode_index=151,
        chunk_ordinal=7,
    )


def test_offline_inference_seed_resets_python_numpy_and_torch() -> None:
    evaluator = load_evaluator()
    evaluator.seed_inference(1234)
    first = (random.random(), np.random.random(), evaluator.torch.rand(1).item())
    evaluator.seed_inference(1234)
    second = (random.random(), np.random.random(), evaluator.torch.rand(1).item())
    assert first == second
