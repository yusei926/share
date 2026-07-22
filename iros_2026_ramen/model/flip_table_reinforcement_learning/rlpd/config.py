"""Configuration contract for the residual RLPD agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "team_ramen_residual_rlpd_v1"
ALL_ACTIONS_MASK = (1.0,) * 19


@dataclass(frozen=True)
class RLPDConfig:
    observation_dim: int
    action_dim: int = 19
    hidden_dim: int = 512
    hidden_layers: int = 3
    critic_ensemble_size: int = 10
    target_critic_sample_size: int = 2
    discount: float = 0.99
    target_update_rate: float = 0.005
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    temperature_learning_rate: float = 3.0e-4
    target_entropy: float = -30.0
    initial_temperature: float = 1.0e-3
    automatic_entropy_tuning: bool = False
    reward_scale: float = 0.1
    actor_q_normalization: float = 1.0
    prior_bc_weight: float = 10.0
    reference_bc_weight: float = 20.0
    min_residual_std: float = 0.02
    initial_residual_std: float = 0.15
    max_residual_std: float = 0.25
    stochastic_action_mask: tuple[float, ...] = ALL_ACTIONS_MASK

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim != 19:
            raise ValueError("RLPD requires a positive observation dimension and 19-D residual action")
        if self.hidden_dim <= 0 or self.hidden_layers < 1:
            raise ValueError("RLPD hidden dimensions must be positive")
        if not 2 <= self.target_critic_sample_size <= self.critic_ensemble_size:
            raise ValueError("target critic sample size must be within the critic ensemble")
        if not 0.0 < self.discount <= 1.0 or not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("discount and target update rate must be in (0, 1]")
        if min(
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.temperature_learning_rate,
            self.initial_temperature,
            self.reward_scale,
            self.actor_q_normalization,
            self.min_residual_std,
            self.initial_residual_std,
            self.max_residual_std,
        ) <= 0:
            raise ValueError("learning rates, temperature, reward and residual stds must be positive")
        if not (
            self.min_residual_std
            <= self.initial_residual_std
            <= self.max_residual_std
            <= 1.0
            and self.min_residual_std < self.max_residual_std
        ):
            raise ValueError(
                "residual stds must satisfy 0 < min <= initial <= max <= 1 and min < max"
            )
        if self.prior_bc_weight < 0:
            raise ValueError("prior BC weight must be non-negative")
        if self.reference_bc_weight < 0:
            raise ValueError("reference BC weight must be non-negative")
        if len(self.stochastic_action_mask) != self.action_dim:
            raise ValueError("stochastic action mask must contain exactly 19 values")
        if any(value not in {0.0, 1.0} for value in self.stochastic_action_mask):
            raise ValueError("stochastic action mask values must be 0 or 1")
        if not any(self.stochastic_action_mask):
            raise ValueError("stochastic action mask must enable at least one action")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RLPDConfig":
        payload = dict(value)
        schema = payload.pop("schema_version", None)
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported RLPD schema: {schema!r}")
        legacy_scale = payload.pop("residual_scale_rad", 1.0)
        if float(legacy_scale) != 1.0:
            raise ValueError("legacy residual_scale_rad must be 1.0")
        if "stochastic_action_mask" in payload:
            payload["stochastic_action_mask"] = tuple(payload["stochastic_action_mask"])
        return cls(**payload)
