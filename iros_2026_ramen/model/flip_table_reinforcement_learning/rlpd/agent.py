"""PyTorch RLPD agent using an ensemble critic and SAC residual actor."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .config import RLPDConfig
from .replay import ReplayBatch


def mlp(input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self, config: RLPDConfig) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.log_std_min = math.log(config.min_residual_std)
        self.log_std_max = math.log(config.max_residual_std)
        self.register_buffer(
            "stochastic_action_mask",
            torch.tensor(config.stochastic_action_mask, dtype=torch.float32),
            persistent=False,
        )
        self.network = mlp(
            config.observation_dim,
            2 * config.action_dim,
            config.hidden_dim,
            config.hidden_layers,
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        initial_log_std = math.log(config.initial_residual_std)
        normalized_initial = (
            2.0
            * (initial_log_std - self.log_std_min)
            / (self.log_std_max - self.log_std_min)
            - 1.0
        )
        raw_initial = math.atanh(max(-0.999999, min(0.999999, normalized_initial)))
        with torch.no_grad():
            final.bias[self.action_dim :].fill_(raw_initial)

    def distribution_parameters(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.network(observation).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            torch.tanh(raw_log_std) + 1.0
        )
        return mean, log_std

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        normal = torch.distributions.Normal(mean, log_std.exp())
        if deterministic:
            raw = mean
        else:
            sampled_raw = normal.rsample()
            mask = self.stochastic_action_mask.to(dtype=mean.dtype)
            raw = mean + mask * (sampled_raw - mean)
        mask = self.stochastic_action_mask.to(dtype=mean.dtype)
        action = torch.tanh(raw) * mask
        if deterministic:
            log_probability = torch.zeros((observation.shape[0], 1), device=observation.device)
        else:
            correction = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
            log_probability = (
                (normal.log_prob(raw) - correction)
                * self.stochastic_action_mask.to(dtype=mean.dtype)
            ).sum(dim=-1, keepdim=True)
        return action, log_probability


class CriticEnsemble(nn.Module):
    def __init__(self, config: RLPDConfig) -> None:
        super().__init__()
        input_dim = config.observation_dim + config.action_dim
        self.critics = nn.ModuleList(
            [
                mlp(input_dim, 1, config.hidden_dim, config.hidden_layers)
                for _ in range(config.critic_ensemble_size)
            ]
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((observation, action), dim=-1)
        return torch.cat([critic(value) for critic in self.critics], dim=1)


class RLPDAgent(nn.Module):
    def __init__(self, config: RLPDConfig, *, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.actor = ResidualActor(config).to(self.device)
        self.reference_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = CriticEnsemble(config).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_temperature), device=self.device)
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=config.temperature_learning_rate
        )
        self.update_steps = 0

    @torch.no_grad()
    def set_reference_actor_from_current(self) -> None:
        """Freeze the current actor as the trust-region center for fine-tuning."""

        if self.update_steps != 0:
            raise RuntimeError("reference actor can only be set before training")
        self.reference_actor.load_state_dict(self.actor.state_dict(), strict=True)
        self.reference_actor.requires_grad_(False)
        self.reference_actor.eval()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, observation: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        observation = observation.to(self.device)
        return self.actor.sample(observation, deterministic=deterministic)[0]

    @torch.no_grad()
    def initialize_actor_residual(self, residual: torch.Tensor) -> None:
        """Initialize the observation-independent actor mean from a successful prior."""

        if self.update_steps != 0:
            raise RuntimeError("actor residual initialization is only valid before training")
        value = torch.as_tensor(residual, dtype=torch.float32, device=self.device).reshape(-1)
        if value.shape != (self.config.action_dim,):
            raise ValueError(
                f"initial residual must have shape [{self.config.action_dim}], "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all() or bool((value.abs() >= 1.0).any()):
            raise ValueError("initial residual must be finite and strictly inside (-1, 1)")
        final = self.actor.network[-1]
        assert isinstance(final, nn.Linear)
        final.weight[: self.config.action_dim].zero_()
        final.bias[: self.config.action_dim].copy_(torch.atanh(value))
        self.set_reference_actor_from_current()

    def update(
        self,
        batch: ReplayBatch,
        *,
        prior_count: int = 0,
        update_actor: bool = True,
    ) -> dict[str, float]:
        data = batch.to(self.device)
        config = self.config
        batch_size = data.observation.shape[0]
        if not 0 <= prior_count <= batch_size:
            raise ValueError("prior_count must be within the replay batch")
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(data.next_observation)
            target_values = self.target_critic(data.next_observation, next_action)
            critic_indices = torch.randperm(
                config.critic_ensemble_size, device=self.device
            )[: config.target_critic_sample_size]
            target_value = target_values[:, critic_indices].min(dim=1, keepdim=True).values
            target_value -= self.temperature.detach() * next_log_probability
            target = config.reward_scale * data.reward + (
                1.0 - data.done
            ) * config.discount * target_value

        predictions = self.critic(data.observation, data.action)
        critic_loss = (predictions - target.expand_as(predictions)).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        sampled_action, log_probability = self.actor.sample(data.observation)
        # RLPD/REDQ uses the ensemble mean for the actor and a random-subset
        # minimum only for the Bellman target. Taking the minimum of the full
        # ensemble here is unnecessarily pessimistic and can freeze the actor.
        q_value = self.critic(data.observation, sampled_action).mean(dim=1, keepdim=True)
        q_abs_mean = q_value.detach().abs().mean().clamp(min=1.0)
        q_multiplier = config.actor_q_normalization / q_abs_mean
        actor_rl_loss = (
            self.temperature.detach() * log_probability - q_multiplier * q_value
        ).mean()
        actor_mean, _actor_log_std = self.actor.distribution_parameters(data.observation)
        deterministic_action = torch.tanh(actor_mean) * self.actor.stochastic_action_mask.to(
            dtype=actor_mean.dtype
        )
        with torch.no_grad():
            reference_action = self.reference_actor.sample(
                data.observation,
                deterministic=True,
            )[0]
        reference_bc_loss = F.mse_loss(deterministic_action, reference_action)
        prior_bc_loss = torch.zeros((), device=self.device)
        if prior_count and config.prior_bc_weight:
            prior_bc_loss = F.mse_loss(
                deterministic_action[:prior_count],
                data.action[:prior_count],
            )
        actor_loss = (
            actor_rl_loss
            + config.prior_bc_weight * prior_bc_loss
            + config.reference_bc_weight * reference_bc_loss
        )
        actor_gradient_norm = torch.zeros((), device=self.device)
        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 10.0
            )
            self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        temperature_loss = torch.zeros((), device=self.device)
        if config.automatic_entropy_tuning:
            temperature_loss = -(
                self.log_temperature
                * (log_probability.detach() + config.target_entropy)
            ).mean()
        if update_actor and config.automatic_entropy_tuning:
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, config.target_update_rate)
            mean, log_std = self.actor.distribution_parameters(data.observation)
            deterministic_residual = torch.tanh(mean)
        self.update_steps += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "actor_rl_loss": float(actor_rl_loss.detach()),
            "prior_bc_loss": float(prior_bc_loss.detach()),
            "reference_bc_loss": float(reference_bc_loss.detach()),
            "actor_q_multiplier": float(q_multiplier.detach()),
            "actor_updated": float(update_actor),
            "temperature_loss": float(temperature_loss.detach()),
            "temperature": float(self.temperature.detach()),
            "mean_reward": float(data.reward.mean()),
            "mean_q": float(q_value.detach().mean()),
            "critic_gradient_norm": float(critic_gradient_norm),
            "actor_gradient_norm": float(actor_gradient_norm),
            "residual_std_mean": float(log_std.exp().mean()),
            "residual_std_max": float(log_std.exp().max()),
            "deterministic_residual_abs_mean": float(
                deterministic_residual.abs().mean()
            ),
            "deterministic_residual_abs_max": float(
                deterministic_residual.abs().max()
            ),
        }

    def save_pretrained(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": self.config.to_dict(),
            "policy_inputs": [
                "frozen Flow Matching observation context",
                "normalized 19D upper-body state",
                "normalized 16D Flow Matching arm/hand base target",
                "previous 16D residual",
            ],
            "policy_output": "16D bounded residual over the Flow Matching arm/hand target",
            "privileged_inputs": [],
            "update_steps": self.update_steps,
        }
        (output / "rlpd_policy.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        state = {
            **{f"actor.{key}": value.detach().cpu().contiguous() for key, value in self.actor.state_dict().items()},
            "log_temperature": self.log_temperature.detach().cpu().contiguous(),
        }
        save_file(state, str(output / "model.safetensors"))

    def save_training_state(self, path: str | Path) -> None:
        torch.save(
            {
                "schema_version": "team_ramen_rlpd_training_v2",
                "model": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "temperature_optimizer": self.temperature_optimizer.state_dict(),
                "update_steps": self.update_steps,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if self.device.type == "cuda"
                else None,
            },
            Path(path),
        )

    def load_training_state(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        schema = payload.get("schema_version")
        if schema not in {
            "team_ramen_rlpd_training_v1",
            "team_ramen_rlpd_training_v2",
        }:
            raise ValueError("unsupported RLPD training checkpoint schema")
        if schema == "team_ramen_rlpd_training_v1":
            incompatible = self.load_state_dict(payload["model"], strict=False)
            expected_missing = {
                f"reference_actor.{name}" for name in self.reference_actor.state_dict()
            }
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError("legacy RLPD checkpoint has incompatible model keys")
            self.set_reference_actor_from_current()
        else:
            self.load_state_dict(payload["model"], strict=True)
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(payload["temperature_optimizer"])
        self.update_steps = int(payload["update_steps"])
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_states")
        if self.device.type == "cuda" and cuda_states is not None:
            # map_location also moves serialized RNG tensors, but this API
            # specifically requires CPU ByteTensors.
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, device: str | torch.device = "cpu"
    ) -> "RLPDAgent":
        root = Path(directory)
        metadata = json.loads((root / "rlpd_policy.json").read_text(encoding="utf-8"))
        agent = cls(RLPDConfig.from_dict(metadata["config"]), device=device)
        state = load_file(str(root / "model.safetensors"), device=str(device))
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        agent.actor.load_state_dict(actor_state)
        agent.log_temperature.data.copy_(state["log_temperature"])
        agent.update_steps = int(metadata.get("update_steps", 0))
        return agent
