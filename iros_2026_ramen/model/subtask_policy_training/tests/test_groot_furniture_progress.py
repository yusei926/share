from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "lerobot_policy_furniture_groot"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def test_event_progress_is_ordered_and_optional() -> None:
    from gr00t.progress_annotations import MILESTONE_NAMES, annotate_episode

    length = 140
    hand_cmd = np.full((length, 2), 4.5)
    hand_cmd[10:45, 0] = 2.0
    hand_cmd[25:95, 1] = 2.0
    hand_cmd[70:, 0] = 2.0
    hand_cmd[95:, 1] = 4.5
    hand_state = hand_cmd.copy()
    ee_action = np.zeros((length, 12))
    ee_action[:105, 0] = np.linspace(0.0, 0.4, 105)
    robot_q_desired = np.zeros((length, 36))
    robot_q_desired[:105, 22] = np.linspace(0.0, 1.0, 105)

    annotation = annotate_episode(
        episode_index=7,
        hand_cmd=hand_cmd,
        hand_state=hand_state,
        ee_action=ee_action,
        robot_q_desired=robot_q_desired,
    )

    frames = [
        annotation.milestones[name].frame
        for name in MILESTONE_NAMES
        if annotation.milestones[name].valid
    ]
    assert frames == sorted(frames)
    assert annotation.primary_hand == "left"
    assert annotation.milestones["M1"].frame == 10
    assert annotation.milestones["M2"].frame == 25
    assert annotation.milestones["M3"].frame == 45
    assert annotation.milestones["M4"].frame == 70
    assert annotation.milestones["M5"].frame == 95
    valid_progress = np.asarray(annotation.progress)[annotation.progress_mask]
    assert np.all(np.diff(valid_progress) >= -1e-7)


def test_progress_horizon_masks_episode_tail() -> None:
    from gr00t.progress_annotations import progress_horizons

    values, masks = progress_horizons([0.0, 0.5, 1.0], [True, True, True], horizon=4)
    assert values[1] == [0.5, 1.0, 0.0, 0.0]
    assert masks[1] == [True, True, False, False]


def test_orientation_group_counts_come_from_episode_metadata(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "meta"
        / "episodes"
        / "chunk-000"
        / "file-000.parquet"
    )
    path.parent.mkdir(parents=True)
    clusters = [index % 4 for index in range(174)]
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(174)),
                "curation_orientation_cluster": clusters,
            }
        ),
        path,
    )
    script_path = ROOT / "scripts" / "build_flip_table_progress_sidecar.py"
    spec = importlib.util.spec_from_file_location(
        "build_flip_table_progress_sidecar_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._orientation_group_counts(tmp_path) == {
        "0": 44,
        "1": 44,
        "2": 43,
        "3": 43,
    }


def test_temporal_processor_keeps_rgb_history_but_selects_current_vectors() -> None:
    importlib.import_module("lerobot_policy_furniture_groot")
    from lerobot.types import TransitionKey
    from lerobot_policy_furniture_groot.processor_furniture_groot import (
        FurnitureGrootTemporalProgressStep,
        PROGRESS_TARGET_KEY,
        PROGRESS_VALID_KEY,
    )

    images = torch.zeros(2, 2, 3, 8, 8)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.arange(2 * 2 * 49).reshape(2, 2, 49),
            "observation.images.head_left": images,
            "observation.images.head_left_is_pad": torch.tensor(
                [[True, False], [False, False]]
            ),
            "observation.progress_horizon": torch.zeros(2, 2, 40),
            "observation.progress_mask": torch.ones(2, 2, 40, dtype=torch.bool),
        },
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    result = FurnitureGrootTemporalProgressStep()(transition)
    observation = result[TransitionKey.OBSERVATION]
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]

    assert observation["observation.state"].shape == (2, 49)
    assert observation["observation.images.head_left"].shape == (2, 2, 3, 8, 8)
    assert observation["observation.images.head_left_is_pad"].shape == (2, 2)
    assert "observation.progress_horizon" not in observation
    assert complementary[PROGRESS_TARGET_KEY].shape == (2, 40, 1)
    assert complementary[PROGRESS_VALID_KEY].shape == (2, 40, 1)


def test_temporal_processor_batches_unbatched_two_frame_rgb() -> None:
    from lerobot.types import TransitionKey
    from lerobot_policy_furniture_groot.processor_furniture_groot import (
        FurnitureGrootTemporalProgressStep,
    )

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.zeros(1, 49),
            "observation.images.head_left": torch.zeros(2, 3, 8, 8),
            "observation.images.head_left_is_pad": torch.tensor([True, False]),
        },
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    observation = FurnitureGrootTemporalProgressStep()(transition)[
        TransitionKey.OBSERVATION
    ]
    assert observation["observation.images.head_left"].shape == (1, 2, 3, 8, 8)
    assert observation["observation.images.head_left_is_pad"].shape == (1, 2)

    transition[TransitionKey.OBSERVATION]["observation.images.head_left"] = torch.zeros(
        3, 3, 8, 8
    )
    with pytest.raises(ValueError, match=r"\[2,C,H,W\]"):
        FurnitureGrootTemporalProgressStep()(transition)


def test_temporal_processor_rejects_malformed_image_padding_mask() -> None:
    from lerobot.types import TransitionKey
    from lerobot_policy_furniture_groot.processor_furniture_groot import (
        FurnitureGrootTemporalProgressStep,
    )

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.zeros(1, 49),
            "observation.images.head_left": torch.zeros(1, 2, 3, 8, 8),
            "observation.images.head_left_is_pad": torch.zeros(
                1, 3, dtype=torch.bool
            ),
        },
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    with pytest.raises(ValueError, match=r"\[B,2\]"):
        FurnitureGrootTemporalProgressStep()(transition)


def test_furniture_groot_config_preserves_official_dimensions() -> None:
    importlib.import_module("lerobot_policy_furniture_groot")
    from lerobot_policy_furniture_groot.configuration_furniture_groot import (
        FurnitureGrootConfig,
    )

    config = FurnitureGrootConfig(device="cpu")
    assert config.observation_delta_indices == [-20, 0]
    assert config.chunk_size == 40
    assert config.n_action_steps == 10
    assert config.max_state_dim == 132
    assert config.max_action_dim == 132
    assert config.valid_action_dim == 46
    assert config.consistent_gpu_augmentation is True
    assert (
        config.base_model_revision
        == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    )

    with pytest.raises(ValueError, match="H40"):
        FurnitureGrootConfig(device="cpu", chunk_size=54)
    with pytest.raises(ValueError, match="pinned GR00T N1.7"):
        FurnitureGrootConfig(device="cpu", base_model_revision="main")


def test_consistent_gpu_augmentation_is_coherent_and_training_only() -> None:
    from lerobot.types import TransitionKey
    from lerobot_policy_furniture_groot.processor_furniture_groot import (
        FurnitureGrootConsistentGpuAugmentationStep,
        VIDEO_KEY,
    )

    base_frame = torch.arange(8 * 8 * 3, dtype=torch.uint8).reshape(8, 8, 3)
    video = base_frame.reshape(1, 1, 1, 8, 8, 3).repeat(2, 2, 3, 1, 1, 1)
    transition = {
        TransitionKey.OBSERVATION: {VIDEO_KEY: video.numpy()},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    step = FurnitureGrootConsistentGpuAugmentationStep(
        training=True,
        device="cpu",
    )

    torch.manual_seed(7)
    result = step(transition)
    augmented = result[TransitionKey.OBSERVATION][VIDEO_KEY]

    assert augmented.shape == video.shape
    assert augmented.dtype == np.uint8
    assert not np.array_equal(augmented, video.numpy())
    for batch_index in range(augmented.shape[0]):
        reference = augmented[batch_index, 0, 0]
        for time_index in range(augmented.shape[1]):
            for view_index in range(augmented.shape[2]):
                assert np.array_equal(
                    augmented[batch_index, time_index, view_index], reference
                )
    assert "training" not in step.get_config()

    inference_video = video.numpy()
    inference_transition = {
        TransitionKey.OBSERVATION: {VIDEO_KEY: inference_video},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    inference_step = FurnitureGrootConsistentGpuAugmentationStep(
        training=False,
        device="cpu",
    )
    inference_result = inference_step(inference_transition)
    assert inference_result[TransitionKey.OBSERVATION][VIDEO_KEY] is inference_video


def test_furniture_processor_factory_detects_runtime_training_metadata(
    monkeypatch,
) -> None:
    from lerobot_policy_furniture_groot.configuration_furniture_groot import (
        FurnitureGrootConfig,
    )
    from lerobot_policy_furniture_groot import processor_furniture_groot as processor

    DummyVlmStep = type("GrootN17VLMEncodeStep", (), {})

    class DummyPipeline:
        def __init__(self) -> None:
            self.steps = [object(), object(), DummyVlmStep()]

    captured: dict[str, object] = {}

    def fake_make_processors(*, config, dataset_stats=None, dataset_meta=None):
        captured["dataset_meta"] = dataset_meta
        return DummyPipeline(), DummyPipeline()

    monkeypatch.setattr(
        processor,
        "make_groot_pre_post_processors",
        fake_make_processors,
    )
    config = FurnitureGrootConfig(device="cpu")
    runtime_meta = object()
    config._runtime_dataset_meta = runtime_meta

    preprocessor, _ = processor.make_furniture_groot_pre_post_processors(config)
    augmentation = next(
        step
        for step in preprocessor.steps
        if isinstance(step, processor.FurnitureGrootConsistentGpuAugmentationStep)
    )

    assert captured["dataset_meta"] is runtime_meta
    assert augmentation.training is True


def test_progress_loss_is_separate_and_monotonic() -> None:
    importlib.import_module("lerobot_policy_furniture_groot")
    from lerobot_policy_furniture_groot.configuration_furniture_groot import (
        FurnitureGrootConfig,
    )
    from lerobot_policy_furniture_groot.modeling_furniture_groot import (
        FurnitureGrootPolicy,
    )

    policy = object.__new__(FurnitureGrootPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = FurnitureGrootConfig(device="cpu")
    target = torch.linspace(0, 1, 40).repeat(2, 1).unsqueeze(-1)
    valid = torch.ones_like(target, dtype=torch.bool)

    progress_loss, monotonic_loss = policy._progress_losses(target, target, valid)
    assert progress_loss == pytest.approx(torch.tensor(0.0))
    assert monotonic_loss == pytest.approx(torch.tensor(0.0))

    reversed_prediction = target.flip(1)
    _, monotonic_loss = policy._progress_losses(reversed_prediction, target, valid)
    assert float(monotonic_loss) > 0


@pytest.mark.parametrize("progress_enabled", [False, True])
def test_progress_head_optimizer_membership_matches_configuration(
    progress_enabled: bool,
) -> None:
    from lerobot_policy_furniture_groot.modeling_furniture_groot import (
        FurnitureGrootPolicy,
    )

    policy = object.__new__(FurnitureGrootPolicy)
    torch.nn.Module.__init__(policy)
    policy.progress_head = torch.nn.Sequential(
        torch.nn.LayerNorm(4),
        torch.nn.Linear(4, 2),
    )
    policy.progress_head.requires_grad_(progress_enabled)

    optimized_parameters = {
        id(parameter)
        for group in policy.get_optim_params()
        for parameter in group["params"]
    }
    progress_parameters = {
        id(parameter) for parameter in policy.progress_head.parameters()
    }

    if progress_enabled:
        assert progress_parameters <= optimized_parameters
    else:
        assert progress_parameters.isdisjoint(optimized_parameters)
