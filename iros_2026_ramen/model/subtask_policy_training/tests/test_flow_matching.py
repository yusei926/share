from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model.subtask_policy_training.flow_matching import FlowMatchingConfig, FlowMatchingPolicy
from model.subtask_policy_training.flow_matching.data import (
    load_episode_split,
    valid_training_sample_indices,
)


def stats(size: int) -> dict[str, list[float]]:
    return {
        "mean": [0.1] * size,
        "std": [0.5] * size,
        "min": [-2.0] * size,
        "max": [2.0] * size,
    }


def small_config() -> FlowMatchingConfig:
    return FlowMatchingConfig(
        action_horizon=4,
        n_action_steps=2,
        image_height=48,
        image_width=64,
        model_dim=64,
        transformer_layers=1,
        transformer_heads=4,
        feedforward_multiplier=2,
        dropout=0.0,
        flow_inference_steps=2,
        image_encoder_weights="none",
    )


def test_config_enforces_upper_body_contract():
    with pytest.raises(ValueError, match="19-D observed state and 16-D"):
        FlowMatchingConfig(state_dim=18)
    with pytest.raises(ValueError, match="19-D observed state and 16-D"):
        FlowMatchingConfig(action_dim=19)
    with pytest.raises(ValueError, match="within"):
        FlowMatchingConfig(action_horizon=4, n_action_steps=5)


def test_invalid_camera_rows_remove_history_and_action_windows():
    valid = [True, True, False, True, True, True, True, True]
    episodes = [0, 0, 0, 0, 1, 1, 1, 1]
    assert valid_training_sample_indices(
        valid,
        episodes,
        action_horizon=2,
        history_frames=2,
    ) == [0, 4, 5, 6, 7]


def test_flow_loss_sampling_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    config = small_config()
    model = FlowMatchingPolicy(
        config,
        state_stats=stats(19),
        action_stats=stats(16),
        load_pretrained_encoder=False,
    ).eval()
    images = torch.rand(2, 3, 3, 48, 64)
    state = torch.rand(2, 19)
    actions = torch.rand(2, 4, 16)
    padding = torch.tensor([[False, False, False, False], [False, False, True, True]])

    loss = model.flow_loss(images, state, actions, padding)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    sampled = model.sample_actions(images, state)
    assert sampled.shape == (2, 4, 16)
    assert torch.isfinite(sampled).all()
    assert sampled.min() >= -2.0
    assert sampled.max() <= 2.0

    model.save_pretrained(tmp_path)
    restored = FlowMatchingPolicy.from_pretrained(tmp_path)
    restored_sample = restored.sample_actions(images, state)
    torch.testing.assert_close(sampled, restored_sample)
    metadata = json.loads((tmp_path / "flow_matching_policy.json").read_text())
    assert metadata["privileged_inputs"] == []
    assert metadata["policy_output"].startswith("16D arm/hand")


def test_episode_split_rejects_overlap(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"episode_indices": [0, 1]},
                    "validation": {"episode_indices": [2]},
                    "test": {"episode_indices": [3]},
                }
            }
        )
    )
    assert load_episode_split(path)["validation"] == [2]
    value = json.loads(path.read_text())
    value["splits"]["test"]["episode_indices"] = [1]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlap"):
        load_episode_split(path)


def test_held_out_evaluator_records_real_compatible_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "evaluate_flow_matching.py").read_text(
        encoding="utf-8"
    )

    assert 'choices=("validation", "test")' in source
    assert "FlowMatchingPolicy.from_pretrained" in source
    assert "augment=False" in source
    assert '"model_sha256"' in source
    assert '"privileged_inputs": []' in source
