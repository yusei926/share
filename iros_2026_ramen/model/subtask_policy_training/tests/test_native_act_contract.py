from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.subtask_policy_training.native_delta_policy import NativeACTConfig, kl_divergence


def test_native_act_serializes_the_requested_architecture() -> None:
    config = NativeACTConfig()

    assert config.type == "flip_table_native_act_chunk_relative"
    assert config.observation_horizon == 2
    assert config.action_horizon == 16
    assert config.action_execution_steps == 8
    assert config.dim_model == 512
    assert config.n_encoder_layers == 4
    assert config.n_decoder_layers == 7
    assert config.n_heads == 8
    assert config.latent_dim == 32
    assert config.separate_camera_encoders is True


def test_h100_runner_selects_native_act_and_contractual_freeze_schedule() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "run_h100_flip_table_delta.sh"
    ).read_text(encoding="utf-8")

    assert "scripts/train_native_act_delta.py" in script
    assert "--freeze-backbone-steps 10000" in script
    assert "--backbone-lr 1e-5" in script
    assert "--steps \"$TRAIN_STEPS\"" in script


def test_h100_runner_selects_ema_diffusion_contract() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "run_h100_flip_table_delta.sh"
    ).read_text(encoding="utf-8")

    assert "scripts/train_native_diffusion_delta.py" in script
    assert "--ema-decay 0.9999" in script
    assert "--batch-size \"$TRAIN_BATCH_SIZE\"" in script


def test_h100_runner_builds_a_training_view_only_short_gop_rgb_cache() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "run_h100_flip_table_delta.sh"
    ).read_text(encoding="utf-8")

    assert "build_resized_video_cache.py" in script
    assert "outputs/h100_flip_table_delta/video_cache_320x240_gop8" in script
    assert "LEROBOT_VIDEO_DECODER_CACHE_SIZE=10" in script
    assert "TRAIN_NUM_WORKERS=4" in script
    assert "--worker-restart-steps 400" in script


def test_chunk_relative_arms_use_one_measured_chunk_start_and_absolute_grippers() -> None:
    from model.subtask_policy_training.action_representation import (
        CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
        decode_action_chunk,
        encode_action_chunk,
    )

    state = torch.zeros((1, 2, 19), dtype=torch.float32)
    state[:, -1, 3:17] = torch.arange(14, dtype=torch.float32)
    actions = torch.zeros((1, 3, 16), dtype=torch.float32)
    actions[0, :, :14] = torch.arange(14, dtype=torch.float32).unsqueeze(0) + torch.tensor(
        [[1.0], [2.0], [3.0]]
    )
    actions[0, :, 14:] = torch.tensor([[1.0, 4.5], [2.0, 3.5], [3.0, 2.5]])

    encoded = encode_action_chunk(actions, state, CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER)

    assert encoded[0, :, :14].tolist() == [[1.0] * 14, [2.0] * 14, [3.0] * 14]
    assert torch.equal(encoded[:, :, 14:], actions[:, :, 14:])
    assert torch.equal(
        decode_action_chunk(encoded, state, CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER), actions
    )


def test_chunk_relative_action_does_not_integrate_from_prior_prediction() -> None:
    from model.subtask_policy_training.action_representation import (
        CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
        decode_action_chunk,
    )

    state = torch.zeros((1, 1, 19), dtype=torch.float32)
    state[:, :, 3:17] = 0.4
    model_actions = torch.zeros((1, 2, 16), dtype=torch.float32)
    model_actions[:, 0, :14] = 0.1
    model_actions[:, 1, :14] = 0.2

    decoded = decode_action_chunk(model_actions, state, CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER)

    assert torch.allclose(decoded[:, 0, :14], torch.full((1, 14), 0.5))
    assert torch.allclose(decoded[:, 1, :14], torch.full((1, 14), 0.6))


def test_chunk_relative_statistics_anchor_every_target_at_the_chunk_start(tmp_path: Path) -> None:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    from model.subtask_policy_training.action_representation import (
        compute_chunk_relative_action_stats,
    )

    root = tmp_path / "dataset"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "team_ramen_episode_split.json").write_text(
        json.dumps({"splits": {"train": {"episode_indices": [0]}}}),
        encoding="utf-8",
    )
    state = [[0.0, 0.0, 0.0] + [float(index) for index in range(14)] + [0.0, 0.0]] * 3
    action = [
        [float(index + offset) for index in range(14)] + [0.1, 0.2]
        for offset in (1, 2, 3)
    ]
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "observation.state": state,
                "action": action,
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )

    stats = compute_chunk_relative_action_stats(root)

    # Valid chunks contribute [1,2,3], [2,3], and [3] relative targets.
    assert stats["count"] == [6]
    assert stats["mean"][:14] == [14.0 / 6.0] * 14


def test_native_diffusion_uses_the_requested_official_model_shape() -> None:
    from model.subtask_policy_training.scripts.train_native_diffusion_delta import build_policy

    _, config = build_policy("cpu")
    assert config.n_obs_steps == 2
    assert config.horizon == 16
    assert config.n_action_steps == 8
    assert config.down_dims == (256, 512, 1024)
    assert config.use_separate_rgb_encoder_per_camera is True
    assert config.num_train_timesteps == 100
    assert config.num_inference_steps == 10
    assert config.clip_sample is False
    assert config.do_mask_loss_for_padding is True


def test_serial_video_decode_override_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounded decoder override is process-local and safe to call twice."""
    import types
    import sys

    class DatasetReader:
        pass

    dataset_reader = types.ModuleType("lerobot.datasets.dataset_reader")
    dataset_reader.DatasetReader = DatasetReader
    dataset_reader.dequantize_depth = lambda frames, **_: frames
    video_utils = types.ModuleType("lerobot.datasets.video_utils")
    video_utils.decode_video_frames = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "lerobot.datasets.dataset_reader", dataset_reader)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", video_utils)

    from model.subtask_policy_training.native_delta_policy import configure_serial_video_decode

    configure_serial_video_decode()
    first = DatasetReader._query_videos
    configure_serial_video_decode()
    assert DatasetReader._query_videos is first


def test_training_arguments_are_json_safe(tmp_path: Path) -> None:
    import argparse

    from model.subtask_policy_training.scripts.train_native_act_delta import serialize_arguments

    assert serialize_arguments(argparse.Namespace(output_dir=tmp_path, steps=1000)) == {
        "output_dir": str(tmp_path),
        "steps": 1000,
    }


def test_backbone_scheduler_uses_a_fixed_global_step_offset() -> None:
    """The unfreeze scheduler must not capture the mutable train-loop step."""
    import argparse

    from model.subtask_policy_training.scripts.train_native_act_delta import make_scheduler

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    args = argparse.Namespace(warmup_steps=5_000, steps=200_000)
    scheduler = make_scheduler(optimizer, args, step_offset=10_001)

    schedule = scheduler.lr_lambdas[0]
    assert schedule(0) > 0.99
    assert schedule(100_000) > 0.0
    assert schedule(189_999) == 0.0


def test_kl_divergence_is_non_negative_under_low_precision_inputs() -> None:
    """The VAE regularizer must not go negative through bf16 cancellation."""
    mean = torch.zeros((2, 32), dtype=torch.bfloat16)
    log_variance = torch.full((2, 32), 0.01, dtype=torch.bfloat16)

    value = kl_divergence(mean, log_variance)

    assert value.dtype == torch.float32
    assert value >= 0
