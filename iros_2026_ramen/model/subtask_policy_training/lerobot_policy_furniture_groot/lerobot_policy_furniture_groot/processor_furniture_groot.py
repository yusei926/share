"""Temporal input and progress-label processing for Furniture-GR00T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .configuration_furniture_groot import FurnitureGrootConfig

PROGRESS_OBSERVATION_KEY = "observation.progress_horizon"
PROGRESS_MASK_OBSERVATION_KEY = "observation.progress_mask"
PROGRESS_TARGET_KEY = "progress_target"
PROGRESS_VALID_KEY = "progress_valid"
VIDEO_KEY = "video"


@ProcessorStepRegistry.register(name="furniture_groot_temporal_progress_v1")
@dataclass
class FurnitureGrootTemporalProgressStep(ProcessorStep):
    """Keep temporal RGB, select current vectors, and isolate auxiliary labels."""

    progress_horizon: int = 40

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        state = observation.get(OBS_STATE)
        if isinstance(state, torch.Tensor) and state.ndim == 3:
            observation[OBS_STATE] = state[:, -1, :]

        for key, value in observation.items():
            if not key.startswith(f"{OBS_IMAGES}.") or not isinstance(value, torch.Tensor):
                continue
            if key.endswith("_is_pad"):
                if value.ndim == 1:
                    if value.shape[0] != 2:
                        raise ValueError(
                            f"{key} unbatched temporal mask must be [2], got {tuple(value.shape)}"
                        )
                    observation[key] = value.unsqueeze(0)
                elif value.ndim != 2 or value.shape[1] != 2:
                    raise ValueError(
                        f"{key} must be a temporal mask [2] or [B,2], got {tuple(value.shape)}"
                    )
                continue
            if value.ndim == 4:
                if value.shape[0] != 2:
                    raise ValueError(
                        f"{key} unbatched temporal input must be [2,C,H,W], got {tuple(value.shape)}"
                    )
                observation[key] = value.unsqueeze(0)
            elif value.ndim != 5:
                raise ValueError(
                    f"{key} must be temporal [2,C,H,W] or [B,2,C,H,W], got {tuple(value.shape)}"
                )
            if observation[key].shape[1] != 2:
                raise ValueError(
                    f"{key} must preserve the official [-20,0] two-frame history"
                )

        progress = observation.pop(PROGRESS_OBSERVATION_KEY, None)
        progress_mask = observation.pop(PROGRESS_MASK_OBSERVATION_KEY, None)
        if progress is not None or progress_mask is not None:
            if not isinstance(progress, torch.Tensor) or not isinstance(progress_mask, torch.Tensor):
                raise TypeError("progress target and mask must both be tensors")
            progress = self._latest_vector(progress, name=PROGRESS_OBSERVATION_KEY)
            progress_mask = self._latest_vector(progress_mask, name=PROGRESS_MASK_OBSERVATION_KEY)
            if progress.shape[-1] != self.progress_horizon:
                raise ValueError(
                    f"progress horizon must be {self.progress_horizon}, got {progress.shape[-1]}"
                )
            complementary[PROGRESS_TARGET_KEY] = progress.to(torch.float32).unsqueeze(-1)
            complementary[PROGRESS_VALID_KEY] = progress_mask.to(torch.bool).unsqueeze(-1)

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    @staticmethod
    def _latest_vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim == 3:
            return value[:, -1, :]
        if value.ndim == 2:
            return value
        raise ValueError(f"{name} must have shape (B,H) or (B,T,H), got {tuple(value.shape)}")

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"progress_horizon": self.progress_horizon}


@ProcessorStepRegistry.register(name="furniture_groot_consistent_gpu_augmentation_v1")
@dataclass
class FurnitureGrootConsistentGpuAugmentationStep(ProcessorStep):
    """Apply one coherent augmentation draw across every view and time in a sample."""

    enabled: bool = True
    training: bool = False
    device: str = "cuda"
    max_num_transforms: int = 3
    affine_degrees: float = 5.0
    affine_translate: float = 0.05
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.5, 1.5)
    hue_range: tuple[float, float] = (-0.05, 0.05)
    sharpness_range: tuple[float, float] = (0.5, 1.5)

    def __post_init__(self) -> None:
        if not 1 <= self.max_num_transforms <= 6:
            raise ValueError("max_num_transforms must be in [1, 6]")
        if self.affine_degrees < 0 or not 0 <= self.affine_translate <= 1:
            raise ValueError("invalid affine augmentation range")

    @staticmethod
    def _uniform(values: tuple[float, float], count: int) -> torch.Tensor:
        low, high = values
        return torch.empty(count).uniform_(low, high)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled or not self.training or not torch.is_grad_enabled():
            return transition
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        video = observation.get(VIDEO_KEY)
        if video is None:
            return transition

        input_is_tensor = isinstance(video, torch.Tensor)
        input_device = video.device if input_is_tensor else torch.device("cpu")
        video_tensor = (
            video
            if input_is_tensor
            else torch.from_numpy(np.ascontiguousarray(video))
        )
        if video_tensor.ndim != 6 or video_tensor.shape[-1] not in (1, 3):
            raise ValueError(
                "packed video must be [B,T,V,H,W,C], "
                f"got {tuple(video_tensor.shape)}"
            )
        target_device = torch.device(self.device)
        video_tensor = video_tensor.to(
            target_device,
            non_blocking=target_device.type == "cuda",
        )
        batch_size, time_steps, views, height, width, channels = video_tensor.shape
        selections = torch.stack(
            [
                torch.randperm(6)[: self.max_num_transforms].sort().values
                for _ in range(batch_size)
            ]
        )
        brightness = self._uniform(self.brightness_range, batch_size)
        contrast = self._uniform(self.contrast_range, batch_size)
        saturation = self._uniform(self.saturation_range, batch_size)
        hue = self._uniform(self.hue_range, batch_size)
        sharpness = self._uniform(self.sharpness_range, batch_size)
        angle = self._uniform(
            (-self.affine_degrees, self.affine_degrees), batch_size
        )
        translate_x = self._uniform(
            (-self.affine_translate, self.affine_translate), batch_size
        )
        translate_y = self._uniform(
            (-self.affine_translate, self.affine_translate), batch_size
        )

        augmented_samples: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            frames = (
                video_tensor[batch_index]
                .permute(0, 1, 4, 2, 3)
                .reshape(time_steps * views, channels, height, width)
            )
            for transform_index in selections[batch_index].tolist():
                if transform_index == 0:
                    frames = tvf.adjust_brightness(
                        frames, float(brightness[batch_index])
                    )
                elif transform_index == 1:
                    frames = tvf.adjust_contrast(
                        frames, float(contrast[batch_index])
                    )
                elif transform_index == 2:
                    frames = tvf.adjust_saturation(
                        frames, float(saturation[batch_index])
                    )
                elif transform_index == 3:
                    frames = tvf.adjust_hue(frames, float(hue[batch_index]))
                elif transform_index == 4:
                    frames = tvf.adjust_sharpness(
                        frames, float(sharpness[batch_index])
                    )
                else:
                    frames = tvf.affine(
                        frames,
                        angle=float(angle[batch_index]),
                        translate=[
                            int(round(float(translate_x[batch_index]) * width)),
                            int(round(float(translate_y[batch_index]) * height)),
                        ],
                        scale=1.0,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR,
                    )
            augmented_samples.append(
                frames.reshape(time_steps, views, channels, height, width)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

        augmented_video = torch.stack(augmented_samples)
        if not input_is_tensor:
            observation[VIDEO_KEY] = augmented_video.cpu().numpy()
        elif input_device.type == "cpu":
            observation[VIDEO_KEY] = augmented_video.cpu()
        else:
            observation[VIDEO_KEY] = augmented_video.to(input_device)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "device": self.device,
            "max_num_transforms": self.max_num_transforms,
            "affine_degrees": self.affine_degrees,
            "affine_translate": self.affine_translate,
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "saturation_range": list(self.saturation_range),
            "hue_range": list(self.hue_range),
            "sharpness_range": list(self.sharpness_range),
        }


def make_furniture_groot_pre_post_processors(
    config: FurnitureGrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
):
    runtime_dataset_meta = dataset_meta or getattr(
        config, "_runtime_dataset_meta", None
    )
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=config,
        dataset_stats=dataset_stats,
        dataset_meta=runtime_dataset_meta,
    )
    preprocessor.steps.insert(
        2,
        FurnitureGrootTemporalProgressStep(progress_horizon=config.chunk_size),
    )
    vlm_step_index = next(
        index
        for index, step in enumerate(preprocessor.steps)
        if step.__class__.__name__ == "GrootN17VLMEncodeStep"
    )
    preprocessor.steps.insert(
        vlm_step_index,
        FurnitureGrootConsistentGpuAugmentationStep(
            enabled=config.consistent_gpu_augmentation,
            training=runtime_dataset_meta is not None,
            device=config.device,
        ),
    )
    return preprocessor, postprocessor
