"""Configuration for the contract-preserving Furniture-GR00T policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig

PINNED_BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"


@PreTrainedConfig.register_subclass("furniture_groot")
@dataclass
class FurnitureGrootConfig(GrootConfig):
    """GR00T N1.7 with a separate progress head and two-frame image history."""

    base_model_revision: str = PINNED_BASE_MODEL_REVISION
    progress_enabled: bool = True
    progress_loss_weight: float = 0.05
    progress_monotonicity_weight: float = 0.01
    progress_hidden_dim: int = 512
    consistent_gpu_augmentation: bool = True
    valid_action_dim: int = 46
    chunk_size: int = 40
    n_action_steps: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base_model_revision != PINNED_BASE_MODEL_REVISION:
            raise ValueError(
                "Furniture-GR00T must use the pinned GR00T N1.7 base revision "
                f"{PINNED_BASE_MODEL_REVISION}"
            )
        if self.chunk_size != 40:
            raise ValueError("Furniture-GR00T must retain the N1.7 action horizon H40")
        if self.max_state_dim != 132 or self.max_action_dim != 132:
            raise ValueError("Furniture-GR00T packed state/action dimensions must stay 132")
        if self.valid_action_dim != 46:
            raise ValueError("Furniture-GR00T valid action mask must cover exactly slots 0:46")
        if self.progress_loss_weight < 0 or self.progress_monotonicity_weight < 0:
            raise ValueError("progress loss weights must be non-negative")
        if self.progress_hidden_dim <= 0:
            raise ValueError("progress_hidden_dim must be positive")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [-20, 0]
